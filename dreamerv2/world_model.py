import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl
import math
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
      _, state, outputs, metrics = self.loss(data, state)
    return state, outputs, metrics

  def observe(self, data, state=None):
    print('calling wm observe')
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(embed, data['action'], state, task_vector=data.get('task_vector', None))
    feat = self.rssm.get_feat(post, key='reward')
    # stoch deter (mean std)/(logit)
    outs = dict(
      embed=embed, feat=feat, post=post,
      prior=prior
    )
    return post, outs
  
  @tf.function
  def observe_full(self, data, state=None):
    print('calling wm observe')
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(embed, data['action'], state, task_vector=data.get('task_vector', None))
    return post, prior

  def loss(self, data, state=None):
    print('calling wm loss')
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(embed, data['action'], state, task_vector=data.get('task_vector', None))
    rssm_loss, rssm_value = self.rssm.loss(post, prior, data=data, **self.config.kl)
    # stoch deter (mean std)/(logit)
    likes = {}
    losses = rssm_loss
    for name, head in self.heads.items():
      feat = self.rssm.get_feat(post, key=name, task_vec=data.get('task_vector', None))
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
        prior=prior, likes=likes, rssm=rssm_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics.update(self.mut_inf(post, prior))
    return model_loss, post, outs, metrics

  def mut_inf(self, post, prior):
    metrics = {
      'mi_q_subj': self.rssm.mut_inf(post, kind='subj'),
      # 'mi_q_util': self.rssm.mut_inf(post, kind='util'),
      'mi_q_obj': self.rssm.mut_inf(post, kind='obj'),
      'mi_p_subj': self.rssm.mut_inf(prior, kind='subj'),
      # 'mi_p_util': self.rssm.mut_inf(prior, kind='util'),
      'mi_p_obj': self.rssm.mut_inf(prior, kind='obj'),
    }
    return metrics

  def imagine(self, policy, start, horizon, task_vec=None):
    print('calling wm imagine')
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    # start = {k: flatten(v) for k, v in start.items()}
    start = tf.nest.map_structure(flatten, start)
    task_vec = flatten(task_vec)
    def step(prev, _):
      state, _, _, _, _ = prev
      pfeat = self.rssm.get_feat(state, key='policy', task_vec=task_vec)
      vfeat = self.rssm.get_feat(state, key='value', task_vec=task_vec)
      rfeat = self.rssm.get_feat(state, key='reward', task_vec=task_vec)
      action = policy(tf.stop_gradient(pfeat)).sample()
      succ = self.rssm.img_step(state, action, task_vec=task_vec)
      return succ, pfeat, vfeat, rfeat, action
    pfeat = 0 * self.rssm.get_feat(start, key='policy', task_vec=task_vec)
    vfeat = 0 * self.rssm.get_feat(start, key='value', task_vec=task_vec)
    rfeat = 0 * self.rssm.get_feat(start, key='reward', task_vec=task_vec)
    action = policy(pfeat).mode()
    succs, pfeats, vfeats, rfeats, actions = common.static_scan(
        step, tf.range(horizon), (start, pfeat, vfeat, rfeat, action))
    states = succs
    if 'discount' in self.heads:
      discount = self.heads['discount'](rfeats).mean()
    else:
      discount = self.config.discount * tf.ones_like(pfeats[..., 0])
    return pfeats, vfeats, rfeats, states, actions, discount
  
  @tf.function
  def preprocess(self, obs):
    obs = obs.copy()
    if self.config.transparent:
      obs['image'] = tf.cast(obs['image'], self.dtype) / 255.0 - 0.5
      img_depth = 1 if self.config.grayscale else 3
      n_cams = obs['image'].shape[-1] // img_depth
      repeats = [img_depth] * n_cams
      subject = tf.cast(obs['segmentation'][..., :1] == 1, self.dtype)
      obj = tf.cast(obs['segmentation'][..., 1:] == 2, self.dtype)
      subj_image = subject * obs['image'][..., :3]
      obj_image = obj * obs['image'][..., 3:]
      input_image = tf.concat([subj_image, obj_image], axis=-1)
      obs['input_image'] = input_image
      obs['image'] = obs['image'][..., :3]
    else:
      obs['image'] = tf.cast(obs['image'][..., :3], self.dtype) / 255.0 - 0.5
    obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
    if 'discount' in obs:
      obs['discount'] *= self.config.discount
    return obs
  
  @tf.function
  def video_pred(self, data, img_key='image'):
    data = self.preprocess(data)
    truth = data[img_key][:6] + 0.5
    embed = self.encoder(data)
    inp = lambda x: x[:6, :5]
    embed = tf.nest.map_structure(inp, embed)
    action = tf.nest.map_structure(inp, data['action'])
    if 'task_vector' in data:
      task_vector = data['task_vector']
      task_vector = task_vector[:6]
    else:
      task_vector = None
    states, _ = self.rssm.observe(embed, action, task_vector=task_vector)
    feat = self.rssm.get_feat(states, key=img_key, task_vec=task_vector)
    recon = self.heads[img_key](feat).mode()[:6]
    
    last = lambda x: x[:, -1]
    init = tf.nest.map_structure(last, states)
    if task_vector is not None:
      task_vector = task_vector[:, 5:]
    prior = self.rssm.imagine(data['action'][:6, 5:], init, task_vec=task_vector)
    feat = self.rssm.get_feat(prior, key=img_key, task_vec=task_vector)
    openl = self.heads[img_key](feat).mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    _, _, h, w, _ = model.shape
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    video = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    return tf.concat(tf.split(video, C // 3, 3), 1)



class DualWorldModel(WorldModel):

  def __init__(self, step, config):
    super().__init__(step, config)
    self.rssm = common.DualRSSM(config.subj_rssm, config.obj_rssm, config.subj_strategy)
    shape = config.image_size + (config.img_channels,)
    self.encoder = common.DualConvEncoder(config.subj_encoder, config.obj_encoder)
    if config.split_decoder:
      self.heads['obj_image'] = common.ConvDecoder(shape, **config.decoder)
      self.heads['subj_image'] = common.ConvDecoder(shape, **config.decoder)
    self.heads['image'] = common.ConvDecoder(shape, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    # for name in config.grad_heads:
    #   assert name in self.heads, name
    self.modules = [
      self.encoder, self.rssm,
      *self.heads.values()]

  def preprocess(self, obs):
    img = obs['image']
    img = tf.cast(img, self.dtype) / 255.0 - 0.5
    obs = super().preprocess(obs)
    img_depth = 1 if self.config.grayscale else 3
    n_cams = obs['image'].shape[-1] // img_depth
    repeats = [img_depth] * n_cams
    if self.config.transparent:
      subject = tf.cast(obs['segmentation'][..., :1] == 1, self.dtype)
      obj = tf.cast(obs['segmentation'][..., 1:] == 2, self.dtype)
      obs['subj_image'] = tf.repeat(subject, repeats=repeats, axis=-1) * img[..., :3]
      obs['obj_image'] = tf.repeat(obj, repeats=repeats, axis=-1) * img[..., 3:]
    else:
      subject = tf.cast(obs['segmentation'] == 1, self.dtype)
      obj = tf.cast(obs['segmentation'] == 2, self.dtype)
      obs['subj_image'] = tf.repeat(subject, repeats=repeats, axis=-1) * obs['image']
      obs['obj_image'] = tf.repeat(obj, repeats=repeats, axis=-1) * obs['image']
    return obs
  
  def loss(self, data, state=None):
    model_loss, post, outs, metrics = super().loss(data, state)
    metrics['model_subj_kl'] = outs['kl']['subj'].mean()
    metrics['model_obj_kl'] = outs['kl']['obj'].mean()
    prior_dist = self.rssm.get_dist(outs['prior'])
    post_dist = self.rssm.get_dist(outs['post'])
    metrics['prior_subj_ent'] = prior_dist['subj'].entropy().mean()
    metrics['post_subj_ent'] = post_dist['subj'].entropy().mean()
    metrics['prior_obj_ent'] = prior_dist['obj'].entropy().mean()
    metrics['post_obj_ent'] = post_dist['obj'].entropy().mean()
    return model_loss, post, outs, metrics

  @tf.function
  def partial_imagination(self, data, infer_start=False):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data)
    inp = lambda x: x[:6]
    embed = tf.nest.map_structure(inp, embed)
    action = tf.nest.map_structure(inp, data['action'])
    if not infer_start:
      state = self.rssm.initial(tf.shape(action)[0])
    else:
      start = lambda x: x[:, :5]
      cont = lambda x: x[:, 5:]
      embed_start = tf.nest.map_structure(start, embed)
      action_start = tf.nest.map_structure(start, action)
      states, _ = self.rssm.observe(embed_start, action_start)
      feat = self.rssm.get_feat(states)
      recon = self.heads['image'](feat).mode()[:6] + 0.5
      
      embed = tf.nest.map_structure(cont, embed)
      action = tf.nest.map_structure(cont, action)
      last = lambda x: x[:, -1]
      state = tf.nest.map_structure(last, states)

    subj_post, _ = self.rssm.subj_rssm.observe(embed['subj'], action, state['subj'])
    subj_actions = self.rssm.subj_action(state['subj'], subj_post)
    obj_prior = self.rssm.obj_rssm.imagine(subj_actions, state['obj'])
    states = {'subj': subj_post, 'obj': obj_prior}
    feat = self.rssm.get_feat(states)
    openl = self.heads['image'](feat).mode() + 0.5
    if not infer_start:
      model = openl
    else:
      model = tf.concat([recon[:, :5], openl], 1)
    
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
    return video #tf.concat(tf.split(video, C // 3, 3), 1)


class MutualWorldModel(WorldModel):

  def __init__(self, step, config):
    super().__init__(step, config)
    self.rssm = common.MutualRSSM(config.subj_rssm, config.obj_rssm, config.subj_strategy)
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
    metrics['model_subj_kl'] = outs['kl']['subj'].mean()
    metrics['model_obj_kl'] = outs['kl']['obj'].mean()
    prior_dist = self.rssm.get_dist(outs['prior'])
    post_dist = self.rssm.get_dist(outs['post'])
    metrics['prior_subj_ent'] = prior_dist['subj'].entropy().mean()
    metrics['post_subj_ent'] = post_dist['subj'].entropy().mean()
    metrics['prior_obj_ent'] = prior_dist['obj'].entropy().mean()
    metrics['post_obj_ent'] = post_dist['obj'].entropy().mean()
    return model_loss, post, outs, metrics


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
    metrics['model_kl'] = outs['kl'].mean()
    metrics['prior_ent'] = self.rssm.get_dist(outs['prior']).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(outs['post']).entropy().mean()
    return model_loss, post, outs, metrics

  def imagine(self, policy, start, horizon):
    feats, states, actions, discount = super().imagine(policy, start, horizon)
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = tf.nest.map_structure(flatten, start)
    states = {k: tf.concat([
        start[k][None], v[:-1]], 0) for k, v in states.items()}
    return feats, states, actions, discount


class CausalWorldModel(WorldModel):
  def __init__(self, step, config):
    super().__init__(step, config)
    shape = config.image_size + (config.img_channels,)
    self.rssm = common.DualReasoner(**config.rssm, 
      subj_kws=config.subj_rssm, cond_kws=config.cond_kws, obj_kws=config.obj_rssm, 
      feature_sets=config.feature_sets,
    )
    self.encoder = common.DualConvEncoder(config.subj_encoder, config.obj_encoder, config.obj_features)
    self.heads['subj_image'] = common.ConvDecoder(shape, **config.decoder)

    if config.obj_features == 'gt':
      self.heads['obj_gt'] = common.MLP((8,), **config.obj_gt_head)
    if config.obj_features == 'img':
      self.heads['obj_image'] = common.ConvDecoder(shape, **config.decoder)
    if config.obj_features == 'mixed':
      self.heads['obj_image'] = common.ConvDecoder(shape, **config.decoder)
      self.heads['obj_gt'] = common.MLP((8,), **config.obj_gt_head)

    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    self.modules = [
      self.rssm, self.encoder,
      *self.heads.values()
    ]

  def preprocess(self, obs):
    img = obs['image']
    img = tf.cast(img, self.dtype) / 255.0 - 0.5
    obs = super().preprocess(obs)
    img_depth = 1 if self.config.grayscale else 3
    n_cams = obs['image'].shape[-1] // img_depth
    repeats = [img_depth] * n_cams
    if 'task_vector' in obs:
      angle = obs['task_vector'][..., -1:]
      angle = angle / 180. * math.pi
      angle -= math.pi
      obs['task_vector'] = tf.concat([obs['task_vector'][..., :-1], angle], -1)
    if self.config.transparent:
      subject = tf.cast(obs['segmentation'][..., :1] == 1, self.dtype)
      obj = tf.cast(obs['segmentation'][..., 1:] == 2, self.dtype)
      obs['subj_image'] = tf.repeat(subject, repeats=repeats, axis=-1) * img[..., :3]
      obs['obj_image'] = tf.repeat(obj, repeats=repeats, axis=-1) * img[..., 3:]
    else:
      subject = tf.cast(obs['segmentation'] == 1, self.dtype)
      obj = tf.cast(obs['segmentation'] == 2, self.dtype)
      obs['subj_image'] = tf.repeat(subject, repeats=repeats, axis=-1) * obs['image']
      obs['obj_image'] = tf.repeat(obj, repeats=repeats, axis=-1) * obs['image']
    return obs

  def loss(self, data, state):
    model_loss, post, outs, metrics = super().loss(data, state)
    for loss_name, loss_value in outs['rssm'].items():
      metrics[f'model_{loss_name}'] = loss_value.mean()
    prior_dist = self.rssm.get_dist(outs['prior'])
    post_dist = self.rssm.get_dist(outs['post'])
    metrics['prior_subj_ent'] = prior_dist['subj'].entropy().mean()
    metrics['post_subj_ent'] = post_dist['subj'].entropy().mean()
    metrics['prior_obj_ent'] = prior_dist['obj'].entropy().mean()
    metrics['post_obj_ent'] = post_dist['obj'].entropy().mean()
    # metrics['post_util_ent'] = post_dist['util'].entropy().mean()
    # metrics['prior_util_ent'] = prior_dist['util'].entropy().mean()
    return model_loss, post, outs, metrics

  @tf.function
  def video_pred(self, data):
    pred =  {
      'subj': super().video_pred(data, img_key='subj_image')
    }
    if not self.config.obj_features == 'gt':
      pred['obj'] = super().video_pred(data, img_key='obj_image')
    return pred
