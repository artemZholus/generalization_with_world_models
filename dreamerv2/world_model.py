import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl
import proposal


class WorldModel(common.Module):

  def __init__(self, step, config):
    self.step = step
    self.config = config
    self.rssm = None
    self.encoder = None
    self.heads = {}
    self.encoder = None
    self.modules = []
    self.model_opt = common.Optimizer('model', **config.model_opt)
    self.dtype = prec.global_policy().compute_dtype
  
  def train(self, data, state=None):
    print('calling train wm')
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    metrics.update(self.model_opt(model_tape, model_loss, self.modules))
    return state, outputs, metrics

  def wm_loss(self, data, state=None):
    print('calling wm_loss')
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    return state, outputs, metrics

  def loss(self, data, state=None):
    print('calling wm loss')
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(embed, data['action'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    feat = self.rssm.get_feat(post)
    # stoch deter (mean std)/(logit)
    likes = {}
    if isinstance(kl_loss, dict):
      losses = {f'{layer}_kl': loss for layer, loss in kl_loss.items()}
    else:
      assert len(kl_loss.shape) == 0
      losses = {'kl': kl_loss}
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
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}

    return model_loss, post, outs, metrics

  def imagine(self, policy, start, horizon):
    print('calling wm imagine')
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    # start = {k: flatten(v) for k, v in start.items()}
    start = tf.nest.map_structure(flatten, start)
    def step(prev, _):
      state, _, _ = prev
      feat = self.rssm.get_feat(state)
      action = policy(tf.stop_gradient(feat)).sample()
      succ = self.rssm.img_step(state, action)
      return succ, feat, action
    feat = 0 * self.rssm.get_feat(start)
    action = policy(feat).mode()
    succs, feats, actions = common.static_scan(
        step, tf.range(horizon), (start, feat, action))
    states = succs
    if 'discount' in self.heads:
      discount = self.heads['discount'](feats).mean()
    else:
      discount = self.config.discount * tf.ones_like(feats[..., 0])
    return feats, states, actions, discount
  
  @tf.function
  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = tf.cast(obs['image'], self.dtype) / 255.0 - 0.5
    obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
    if 'discount' in obs:
      obs['discount'] *= self.config.discount
    return obs
  
  def video_pred(self, data):
    pass



class DualWorldModel(WorldModel):

  def __init__(self, step, config):
    super().__init__(step, config)
    self.rssm = common.DualRSSM(config.subj_rssm, config.obj_rssm)
    shape = config.image_size + (config.img_channels,)
    self.encoder = common.DualConvEncoder(config.subj_encoder, config.obj_encoder)
    self.heads['image'] = common.ConvDecoder(shape, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.modules = [
      self.encoder, self.rssm,
      *self.heads.values()]

  def preprocess(self, obs):
    obs = super().preprocess(obs)
    if 'segmentation' in obs and self.config.segmentation:
      # doesn't work when grayscale!
      img_depth = 1 if self.config.grayscale else 3
      n_cams = obs['image'].shape[-1] // img_depth
      repeats = [img_depth] * n_cams
      subject = tf.cast(obs['segmentation'] == 1, self.dtype)
      obj = tf.cast(obs['segmentation'] == 2, self.dtype)
      obs['subj_image'] = tf.repeat(subject, repeats=repeats, axis=-1) * obs['image']
      obs['obj_image'] = tf.repeat(obj, repeats=repeats, axis=-1) * obs['image']
    return obs
  
  def loss(self, data, state=None):
    model_loss, post, outs, metrics = super().loss(data, state)
    metrics['model_subj_kl'] = outs['kl_value']['subj'].mean()
    metrics['model_obj_kl'] = outs['kl_value']['obj'].mean()
    prior_dist = self.rssm.get_dist(outs['prior'])
    post_dist = self.rssm.get_dist(outs['post'])
    metrics['prior_subj_ent'] = prior_dist['subj'].entropy().mean()
    metrics['post_subj_ent'] = post_dist['subj'].entropy().mean()
    metrics['prior_obj_ent'] = prior_dist['obj'].entropy().mean()
    metrics['post_obj_ent'] = post_dist['obj'].entropy().mean()
    return model_loss, post, outs, metrics

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


class DreamerWorldModel(WorldModel):

  def __init__(self, step, config):
    super().__init__(step, config)
    self.rssm = common.RSSM(**config.rssm)
    shape = config.image_size + (config.img_channels,)
    self.encoder = common.ConvEncoder(**config.encoder)
    self.heads['image'] = common.ConvDecoder(shape, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.modules = [
      self.encoder, self.rssm,
      *self.heads.values()]

  def loss(self, data, state=None):
    model_loss, post, outs, metrics = super().loss(data, state)
    metrics['model_kl'] = outs['kl_value'].mean()
    metrics['prior_ent'] = self.rssm.get_dist(outs['prior']).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(outs['post']).entropy().mean()
    return model_loss, post, outs, metrics

  def imagine(self, policy, start, horizon):
    feats, states, actions, discount = super().imagine(policy, start, horizon)
    states = {k: tf.concat([
        start[k][None], v[:-1]], 0) for k, v in states.items()}
    return feats, states, actions, discount

  @tf.function
  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5])
    recon = self.heads['image'](
        self.rssm.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = self.heads['image'](self.rssm.get_feat(prior)).mode()
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