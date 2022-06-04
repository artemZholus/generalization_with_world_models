import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl
import world_models


class Agent(common.Module):

  def __init__(self, config, logger, actspce, step, dataset):
    self.config = config
    self._logger = logger
    self._action_space = actspce
    self._num_act = actspce.n if hasattr(actspce, 'n') else actspce.shape[0]
    self.dtype = prec.global_policy().compute_dtype
    self._should_expl = elements.Until(int(
        config.expl_until / config.action_repeat))
    self._counter = step
    with tf.device('cpu:0'):
      self.step = tf.Variable(int(self._counter), tf.int64)
    self._dataset = dataset
    self.wm = dict(
      cema=lambda: world_models.CEMA(self.step, config),
      cema_ib=lambda: world_models.CEMA_IB(self.step, config),
      dual_no_cond=lambda: world_models.DualNoCond(self.step, config),
      dreamer=lambda: world_models.Dreamer(self.step, config),
    )[config.world_model]()
    if config.zero_shot:
      self._zero_shot_ac = ActorCritic(config, self.step, self._num_act) 
    self._task_behavior = ActorCritic(config, self.step, self._num_act)
    self.reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(actspce),
        plan2explore=lambda: expl.Plan2Explore(
            config, self.wm, self._num_act, self.step, self.reward),
        model_loss=lambda: expl.ModelLoss(
            config, self.wm, self._num_act, self.step, self.reward),
    )[config.expl_behavior]()
    # Train step to initialize variables including optimizer statistics.
    data = next(self._dataset)
    self.train(data)
    pass

  @tf.function
  def policy(self, obs, state=None, mode='train', second_agent=False):
    print('calling policy')
    tf.py_function(lambda: self.step.assign(
        int(self._counter), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._num_act))
      state = latent, action
    elif obs['reset'].any():
      state = tf.nest.map_structure(lambda x: x * common.pad_dims(
          1.0 - tf.cast(obs['reset'], x.dtype), len(x.shape)), state)
    latent, action = state
    data = self.wm.preprocess(obs)
    embed = self.wm.encoder(data)
    sample = (mode == 'train') or not self.config.eval_state_mean
    if 'task_vector' in data:
      task_vec = tf.cast(data['task_vector'], dtype=self.dtype)
    else:
      task_vec = None
    if 'obj_gt' in data:
      obj_gt = tf.cast(data['obj_gt'], dtype=self.dtype)
    else:
      obj_gt = None
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, task_vec=task_vec, sample=sample)
    feat = self.wm.rssm.get_feat(latent, key='policy', task_vec=task_vec, obj_gt=obj_gt)
    behaviour = self._task_behavior
    expl_behavour = self._expl_behavior
    if second_agent:
      behaviour = self._zero_shot_ac
      expl_behavour = self._zero_shot_ac
    if mode == 'eval':
      actor = behaviour.actor(feat)
      action = actor.mode()
    elif self._should_expl(self.step):
      actor = expl_behavour.actor(feat)
      action = actor.sample()
    else:
      actor = behaviour.actor(feat)
      action = actor.sample()
    noise = {'train': self.config.expl_noise, 'eval': self.config.eval_noise}
    action = common.action_noise(action, noise[mode], self._action_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None, do_wm_step=True, do_ac_step=True):
    print('calling train agent')
    metrics = {}
    if do_wm_step:
      outputs, mets = self.wm.train(data, state)
    else:
      outputs, mets = self.wm.wm_loss(data, state)
    metrics.update(mets)
    start = outputs['post']
    if self.config.pred_discount:  # Last step could be terminal.
      start = tf.nest.map_structure(lambda x: x[:, :-1], start)
    reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
    if self.config.zero_shot:
      zs_metrics = self._zero_shot_ac.train(self.wm, start, reward)
      metrics.update({f'zero-shot/{k}': v for k, v in zs_metrics.items()})
    if do_ac_step:
      if 'obj_gt' in data:
        obj_gt = tf.cast(data['obj_gt'], dtype=self.dtype)
      else:
        obj_gt = None
      metrics.update(self._task_behavior.train(self.wm, start, reward, 
                                               task_vec=data.get('task_vector', None), obj_gt=obj_gt))
    if self.config.expl_behavior != 'greedy':
      if self.config.pred_discount:
        data = tf.nest.map_structure(lambda x: x[:, :-1], data)
        outputs = tf.nest.map_structure(lambda x: x[:, :-1], outputs)
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return metrics

  @tf.function
  def report(self, data):
    return self.wm.video_pred(data)

class ActorCritic(common.Module):

  def __init__(self, config, step, num_actions):
    self.config = config
    self.step = step
    self.num_actions = num_actions
    self.actor = common.MLP(num_actions, **config.actor)
    self.critic = common.MLP([], **config.critic)
    if config.slow_target:
      self._target_critic = common.MLP([], **config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **config.critic_opt)

  def train(self, world_model, start, reward_fn, task_vec=None, obj_gt=None):
    print('calling ac train')
    metrics = {}
    hor = self.config.imag_horizon
    with tf.GradientTape() as actor_tape:
      pfeat, vfeat, rfeat, state, action, disc = world_model.imagine(self.actor, start, hor, task_vec=task_vec, obj_gt=obj_gt)
      reward = reward_fn(rfeat, state, action)
      target, weight, mets1 = self.target(vfeat, action, reward, disc)
      actor_loss, mets2 = self.actor_loss(pfeat, vfeat, action, target, weight)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets3 = self.critic_loss(vfeat, action, target, weight)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3)
    metrics['average_feat'] = rfeat.mean()
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, feat, vfeat, action, target, weight):
    print('calling a loss')
    metrics = {}
    policy = self.actor(tf.stop_gradient(feat))
    if self.config.actor_grad == 'dynamics':
      objective = target
    elif self.config.actor_grad == 'reinforce':
      baseline = self.critic(vfeat[:-1]).mode()
      advantage = tf.stop_gradient(target - baseline)
      objective = policy.log_prob(action)[:-1] * advantage
    elif self.config.actor_grad == 'both':
      baseline = self.critic(vfeat[:-1]).mode()
      advantage = tf.stop_gradient(target - baseline)
      objective = policy.log_prob(action)[:-1] * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.step)
      objective = mix * target + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.step)
    objective += ent_scale * ent[:-1]
    actor_loss = -(weight[:-1] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, feat, action, target, weight):
    print('calling c loss')
    dist = self.critic(feat)[:-1]
    target = tf.stop_gradient(target)
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, feat, action, reward, disc):
    reward = tf.cast(reward, tf.float32)
    disc = tf.cast(disc, tf.float32)
    value = self._target_critic(feat).mode()
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1], lambda_=self.config.discount_lambda, axis=0)
    weight = tf.stop_gradient(tf.math.cumprod(tf.concat(
        [tf.ones_like(disc[:1]), disc[:-1]], 0), 0))
    metrics = {}
    metrics['reward_mean'] = reward.mean()
    metrics['reward_std'] = reward.std()
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, weight, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)
