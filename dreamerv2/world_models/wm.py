import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

from common.tfutils import Module, Optimizer

class WM(Module):

  def __init__(self, step, config):
    self.step = step
    self.config = config
    self.rssm = None
    self.encoder = None
    self.heads = {}
    self.modules = []
    self.model_opt = Optimizer('model', **config.model_opt)
    self.dtype = prec.global_policy().compute_dtype

  def train(self, data, state=None, **kwargs):
    print('calling WM train')
    with tf.GradientTape() as model_tape:
      loss_inputs = self.loss_inputs(data, state)
      model_loss, losses, values = self.loss(loss_inputs)
      print()
    metrics = self.gather_metrics(losses, values, **loss_inputs)
    metrics.update(self.model_opt(model_tape, model_loss, self.modules))
    return loss_inputs, metrics

  def gather_metrics(self, losses, values, **kwargs):
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    return metrics

  def loss_inputs(self, data, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(embed, data['action'], state, task_vector=data['task_vector'])
    outs = dict(
      data=data,
      embed=embed, 
      post=post, 
      prior=prior)
    return outs

  def wm_loss(self, data, state=None):
    print('calling WM loss')
    with tf.GradientTape() as _:
      loss_inputs = self.loss_inputs(data, state)
      _, state, outputs, metrics = self.loss(loss_inputs)
    return state, outputs, metrics

  @tf.function
  def observe(self, data, state=None):
    print('calling WM observe')
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(embed, data['action'], state, task_vector=data['task_vector'])
    return post, prior

  def kl_loss(self, post, prior, **kwargs):
    print("calling KL loss")
    losses, values = self.rssm.kl_loss(post, prior, **self.config.kl)
    return values, losses

  def likelihood_loss(self, data, post, **kwargs):
    print("calling Likelihood loss")
    likes = {}
    losses = {}
    for name, head in self.heads.items():
      feat = self.rssm.get_feat(post, key=name, task_vec=data['task_vector'])
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      like = tf.cast(head(inp).log_prob(data[name]), tf.float32)
      likes[name] = like
      if name == 'reward':
          like = (like * data['reward_mask']).sum() / data['reward_mask'].sum()
      losses[name] = -like.mean()
    return likes, losses

  def loss(self, **inputs):
    print('computing loss')
    losses = {}
    values = {}
    kl_values, kl_losses = self.kl_loss(**inputs)
    like_values, like_losses = self.likelihood_loss(**inputs)
    losses.update(kl_losses)
    losses.update(like_losses)
    values.update(kl_values)
    values.update(like_values)
    model_loss = sum(
      self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    return model_loss, losses, values

  def imagine(self, policy, start, horizon, task_vec=None, obj_gt=None):
    print('calling wm imagine')
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = tf.nest.map_structure(flatten, start)
    if task_vec is not None:
      task_vec = flatten(task_vec)
    if obj_gt is not None:
      obj_gt = flatten(obj_gt)
    def step(prev, _):
      state, _, _, _, _ = prev
      pfeat = self.rssm.get_feat(state, key='policy', task_vec=task_vec, obj_gt=obj_gt)
      vfeat = self.rssm.get_feat(state, key='value', task_vec=task_vec, obj_gt=obj_gt)
      rfeat = self.rssm.get_feat(state, key='reward', task_vec=task_vec, obj_gt=obj_gt)
      action = policy(tf.stop_gradient(pfeat)).sample()
      succ = self.rssm.img_step(state, action, task_vec=task_vec)
      return succ, pfeat, vfeat, rfeat, action
    pfeat = 0 * self.rssm.get_feat(start, key='policy', task_vec=task_vec, obj_gt=obj_gt)
    vfeat = 0 * self.rssm.get_feat(start, key='value', task_vec=task_vec, obj_gt=obj_gt)
    rfeat = 0 * self.rssm.get_feat(start, key='reward', task_vec=task_vec, obj_gt=obj_gt)
    action = policy(pfeat).mode()
    succs, pfeats, vfeats, rfeats, actions = static_scan(
        step, tf.range(horizon), (start, pfeat, vfeat, rfeat, action))
    states = {k: tf.concat([
        start[k][None], v[:-1]], 0) for k, v in succs.items()}
    if 'discount' in self.heads:
      discount = self.heads['discount'](rfeats).mean()
    else:
      discount = self.config.discount * tf.ones_like(pfeats[..., 0])
    return pfeats, vfeats, rfeats, states, actions, discount

  @tf.function
  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = tf.cast(obs['image'], self.dtype) / 255.0 - 0.5
    img_dim = 1 if self.config.grayscale else 3
    if self.config.segmentation and self.config.transparent:
      # assuming we have two identical sets of cameras for obj and subj
      n_cams = obs['image'].shape[-1] // img_dim // 2
      obs_total_dim = n_cams * img_dim
      repeats = [img_dim] * n_cams
      assert obs['image'].shape[-1] == img_dim * 2 * n_cams

      # segm[..., :n_cams] is subj_mask, segm[..., n_cams:] is obj_mask
      # image[..., :obs_total_dim] is subj_img, image[..., obs_total_dim:] is obj_img
      subj_img_subj_mask = tf.cast(obs['segmentation'][..., :n_cams] == 1, self.dtype)
      obj_img_obj_mask = tf.cast(obs['segmentation'][..., n_cams:] == 2, self.dtype)
      subj_image = tf.repeat(subj_img_subj_mask, repeats=repeats, axis=-1) * obs['image'][..., :obs_total_dim]
      obj_image = tf.repeat(obj_img_obj_mask, repeats=repeats, axis=-1) * obs['image'][..., obs_total_dim:]
      input_image = tf.concat([subj_image, obj_image], axis=-1)
      obs['input_image'] = input_image
      obs['image'] = input_image
    if self.config.segmentation and not self.config.transparent:
      # one set of cameras for both obj and subj
      n_cams = obs['image'].shape[-1] // img_dim
      repeats = [img_dim] * n_cams
      assert obs['image'].shape[-1] == img_dim * n_cams

      # segm[..., :n_cams] is subj_mask, segm[..., n_cams:] is obj_mask
      subj_mask = tf.cast(obs['segmentation'] == 1, self.dtype)
      obj_mask = tf.cast(obs['segmentation'] == 2, self.dtype)
      subj_image = tf.repeat(subj_mask, repeats=repeats, axis=-1) * obs['image']
      obj_image = tf.repeat(obj_mask, repeats=repeats, axis=-1) * obs['image']
      input_image = tf.concat([subj_image, obj_image], axis=-1)
      obs['input_image'] = input_image
      obs['image'] = input_image

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
    task_vector = data['task_vector']
    task_vector = task_vector[:6]
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
    return {'openl': tf.concat(tf.split(video, C // 3, 3), 1)}