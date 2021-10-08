import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl
import proposal


class Agent(common.Module):

  def __init__(self, config, logger, actspce, step, dataset):
    self.config = config
    self._logger = logger
    self._action_space = actspce
    self._num_act = actspce.n if hasattr(actspce, 'n') else actspce.shape[0]
    self._should_expl = elements.Until(int(
        config.expl_until / config.action_repeat))
    self._counter = step
    with tf.device('cpu:0'):
      self.step = tf.Variable(int(self._counter), tf.int64)
    self._dataset = dataset
    self.wm = WorldModel(self.step, config)
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
      latent = self.wm.subj_rssm.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._num_act))
      state = latent, action
    elif obs['reset'].any():
      state = tf.nest.map_structure(lambda x: x * common.pad_dims(
          1.0 - tf.cast(obs['reset'], x.dtype), len(x.shape)), state)
    latent0, action = state
    data = self.wm.preprocess(obs)
    embed = self.wm.subj_encoder(data)
    sample = (mode == 'train') or not self.config.eval_state_mean
    if not (isinstance(latent0, dict) and 'subj_layer' in latent0):
      latent0 = {'subj_layer': latent0, 'obj_layer': latent0}
    subj_latent, _ = self.wm.subj_rssm.obs_step(latent0['subj_layer'], action, embed, sample)
    obj_embed = self.wm.obj_encoder(data)
    subj_action = self.wm.objective_input(subj_latent)
    obj_latent, _ = self.wm.obj_rssm.obs_step(latent0['obj_layer'], subj_action, obj_embed, sample)
    subj_feat = self.wm.subj_rssm.get_feat(subj_latent)
    obj_feat = self.wm.obj_rssm.get_feat(obj_latent)
    feat = tf.concat([subj_feat, obj_feat], -1)
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
    latent = {'subj_layer': subj_latent, 'obj_layer': obj_latent}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None, do_wm_step=True, do_ac_step=True):
    print('calling train agent')
    metrics = {}
    if do_wm_step:
      state, outputs, mets = self.wm.train(data, state)
    else:
      state, outputs, mets = self.wm.wm_loss(data, state)
    if do_wm_step:
      metrics.update(mets)
    start = {'subj_layer': outputs['subj_post'], 'obj_layer': outputs['obj_post']}
    if self.config.pred_discount:  # Last step could be terminal.
      start = tf.nest.map_structure(lambda x: x[:, :-1], start)
    reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
    if self.config.zero_shot:
      zs_metrics = self._zero_shot_ac.train(self.wm, start, reward)
      metrics.update({f'zero-shot/{k}': v for k, v in zs_metrics.items()})
    if do_ac_step:
      metrics.update(self._task_behavior.train(self.wm, start, reward))
    if self.config.expl_behavior != 'greedy':
      if self.config.pred_discount:
        data = tf.nest.map_structure(lambda x: x[:, :-1], data)
        outputs = tf.nest.map_structure(lambda x: x[:, :-1], outputs)
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics

  @tf.function
  def report(self, data):
    return {'openl': self.wm.video_pred(data)}


class WorldModel(common.Module):

  def __init__(self, step, config):
    self.step = step
    self.config = config
    self.subj_rssm = common.RSSM(**config.rssm)
    self.obj_rssm = common.RSSM(**config.rssm)
    self.heads = {}
    shape = config.image_size + (config.img_channels,)
    del config.encoder['keys']
    self.subj_encoder = common.ConvEncoder(**config.encoder.update(keys=['subj_image']))
    self.obj_encoder = common.ConvEncoder(**config.encoder.update(keys=['obj_image']))
    self.heads['image'] = common.ConvDecoder(shape, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    print('calling train wm')
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [
      self.subj_encoder, self.obj_encoder,
      self.subj_rssm, self.obj_rssm,
      *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def wm_loss(self, data, state=None):
    print('calling wm_loss')
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    return state, outputs, metrics

  def objective_input(self, subjective_post):
    feat = self.subj_rssm.get_feat(subjective_post)
    return tf.stop_gradient(feat)

  def loss(self, data, state=None):
    print('calling wm loss')
    data = self.preprocess(data)
    subj_embed = self.subj_encoder(data)
    obj_embed = self.obj_encoder(data)
    state = {}
    subj_post, subj_prior = self.subj_rssm.observe(subj_embed, data['action'], state.get('subj_layer'))
    subj_kl_loss, subj_kl_value = self.subj_rssm.kl_loss(subj_post, subj_prior, **self.config.kl)
    # objective layer "action"
    subj_actions = self.objective_input(subj_post)
    # note that here subj actions are shifted by 1 action
    obj_post, obj_prior = self.obj_rssm.observe(obj_embed, subj_actions, state.get('obj_layer'))
    obj_kl_loss, obj_kl_value = self.obj_rssm.kl_loss(obj_post, obj_prior, **self.config.kl)
    # stoch deter (mean std)/(logit)
    
    # assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'obj_kl': obj_kl_loss, 'subj_kl': subj_kl_loss}
    subj_feat = self.subj_rssm.get_feat(subj_post)
    obj_feat = self.obj_rssm.get_feat(obj_post)
    feat = tf.concat([subj_feat, obj_feat], -1)
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      like = tf.cast(head(inp).log_prob(data[name]), tf.float32)
      likes[name] = like
      if name == 'reward':
        like = (like * data['reward_mask']).sum() / data['reward_mask'].sum()
      losses[name] = -like.mean()
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        subj_embed=subj_embed, obj_embed=obj_embed, feat=feat, subj_post=subj_post,
        subj_prior=subj_prior, obj_post=obj_post, 
        obj_prior=obj_prior,
        likes=likes, obj_kl=obj_kl_value, subj_kl=subj_kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_subj_kl'] = subj_kl_value.mean()
    metrics['model_obj_kl'] = obj_kl_value.mean()
    metrics['prior_subj_ent'] = self.subj_rssm.get_dist(subj_prior).entropy().mean()
    metrics['post_subj_ent'] = self.subj_rssm.get_dist(subj_post).entropy().mean()
    metrics['prior_obj_ent'] = self.obj_rssm.get_dist(obj_prior).entropy().mean()
    metrics['post_obj_ent'] = self.obj_rssm.get_dist(obj_post).entropy().mean()
    state = {'obj_layer': obj_post, 'subj_layer': subj_post}
    return model_loss, state, outs, metrics

  def imagine(self, policy, start, horizon):
    print('calling wm imagine')
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    # start = {k: flatten(v) for k, v in start.items()}
    start = tf.nest.map_structure(flatten, start)
    def step(prev, _):
      state, _, _ = prev
      subj_feat = self.subj_rssm.get_feat(state['subj_layer'])
      obj_feat = self.obj_rssm.get_feat(state['obj_layer'])
      feat = tf.concat([subj_feat, obj_feat], 1)
      action = policy(feat).sample()
      subj_succ = self.subj_rssm.img_step(state['subj_layer'], action)
      subj_action = self.objective_input(subj_succ)
      obj_succ = self.obj_rssm.img_step(state['obj_layer'], subj_action)
      succ = {'subj_layer': subj_succ, 'obj_layer': obj_succ}
      return succ, feat, action
    subj_feat = self.subj_rssm.get_feat(start['subj_layer'])
    obj_feat = self.obj_rssm.get_feat(start['obj_layer'])
    feat = 0 * tf.concat([subj_feat, obj_feat], 1)
    action = policy(feat).mode()
    succs, feats, actions = common.static_scan(
        step, tf.range(horizon), (start, feat, action))
    states = succs
    # states = {k: tf.concat([
    #     start[k][None], v[:-1]], 0) for k, v in succs.items()}
    if 'discount' in self.heads:
      discount = self.heads['discount'](feats).mean()
    else:
      discount = self.config.discount * tf.ones_like(feats[..., 0])
    return feats, states, actions, discount

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    # second preprocessing for multitask ???
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    # obs['segmentation'] = obs['image'] * 0. + 1.
    if 'segmentation' in obs:
      # doesn't work when grayscale!
      img_depth = 1 if self.config.grayscale else 3
      n_cams = obs['image'].shape[-1] // img_depth
      repeats = [img_depth] * n_cams

      subject = tf.cast(obs['segmentation'] == 1, dtype)
      obj = tf.cast(obs['segmentation'] == 2, dtype)
      obs['subj_image'] = tf.repeat(subject, repeats=repeats, axis=-1) * obs['image']
      obs['obj_image'] = tf.repeat(obj, repeats=repeats, axis=-1) * obs['image']
    obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
    if 'discount' in obs:
      obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    subj_embed = self.subj_encoder(data)
    obj_embed = self.obj_encoder(data)
    subj_states, _ = self.subj_rssm.observe(subj_embed[:6, :5], data['action'][:6, :5])
    subj_actions = self.objective_input(subj_states)
    obj_states, _ = self.obj_rssm.observe(obj_embed[:6, :5], subj_actions)
    subj_feat = self.subj_rssm.get_feat(subj_states)
    obj_feat = self.obj_rssm.get_feat(obj_states)
    feat = tf.concat([subj_feat, obj_feat], -1)
    recon = self.heads['image'](feat).mode()[:6]
    subj_init = {k: v[:, -1] for k, v in subj_states.items()}
    obj_init = {k: v[:, -1] for k, v in obj_states.items()}
    init = {'subj_layer': subj_init, 'obj_init': obj_init}
    subj_prior = self.subj_rssm.imagine(data['action'][:6, 5:], subj_init)
    subj_action = self.objective_input(subj_prior)
    obj_prior = self.obj_rssm.imagine(subj_action, obj_init)
    subj_feat = self.subj_rssm.get_feat(subj_prior)
    obj_feat = self.obj_rssm.get_feat(obj_prior)
    feat = tf.concat([subj_feat, obj_feat], -1)
    openl = self.heads['image'](feat).mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    _, _, h, w, _ = model.shape
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    # vid = tf.transpose(video, (1, 4, 2, 0, 3))
    # vid = tf.reshape(vid, (50, 3, 3*w, 6*h))
    # vid = tf.cast(vid, tf.float32)
    # def log_(x):
    #   x = (x.astype(np.float32) * 255).astype(np.uint8)
    #   wandb.log({'agent/openl': wandb.Video(x, fps=30, format="gif")})
    # tf.print('steps elapsed:', self._log_step)
    # if tf.equal(tf.math.mod(self._log_step, 50), 0) and self._c['wdb']:
    #   tools.graph_summary(
    #       self._writer, log_, vid
    #   )
    # self._log_step.assign_add(1)
    video = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    return tf.concat(tf.split(video, C // 3, 3), 1)

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

  def train(self, world_model, start, reward_fn):
    print('calling ac train')
    metrics = {}
    hor = self.config.imag_horizon
    with tf.GradientTape() as actor_tape:
      feat, state, action, disc = world_model.imagine(self.actor, start, hor)
      reward = reward_fn(feat, state, action)
      target, weight, mets1 = self.target(feat, action, reward, disc)
      actor_loss, mets2 = self.actor_loss(feat, action, target, weight)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets3 = self.critic_loss(feat, action, target, weight)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, feat, action, target, weight):
    print('calling a loss')
    metrics = {}
    policy = self.actor(tf.stop_gradient(feat))
    if self.config.actor_grad == 'dynamics':
      objective = target
    elif self.config.actor_grad == 'reinforce':
      baseline = self.critic(feat[:-1]).mode()
      advantage = tf.stop_gradient(target - baseline)
      objective = policy.log_prob(action)[:-1] * advantage
    elif self.config.actor_grad == 'both':
      baseline = self.critic(feat[:-1]).mode()
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
