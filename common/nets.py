import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class RSSM(common.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self._cell = GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed, action = swap(embed), swap(action)
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action, sample)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden, self._act)(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('img_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value

class DualRSSM(common.Module):

  def __init__(self, subj_config, obj_config, subj_strategy, foresight):
    super().__init__()
    self.subj_rssm = RSSM(**subj_config)
    self.obj_rssm = RSSM(**obj_config)
    self.strategy = subj_strategy
    self.foresight = foresight
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)
    self._dtype = prec.global_policy().compute_dtype

  def initial(self, batch_size):
    subj_state = self.subj_rssm.initial(batch_size)
    obj_state = self.obj_rssm.initial(batch_size)
    state = {'subj': subj_state, 'obj': obj_state}
    return state

  def step_subj_action(self, prev_subj_states, curr_subj_states):
    if self.strategy == 'instant':
      feat = self.subj_rssm.get_feat(curr_subj_states)
    elif self.strategy == 'reactive':
      feat = self.subj_rssm.get_feat(prev_subj_states)
    elif self.strategy == 'delta':
      curr_feats = self.subj_rssm.get_feat(curr_subj_states)
      prev_feats = self.subj_rssm.get_feat(prev_subj_states)
      diff = lambda x, y: x - y 
      feat = tf.nest.map_structure(diff, curr_feats, prev_feats)
    return tf.stop_gradient(feat)

  def subj_action(self, start_state, subj_states, expand=True):
    if self.strategy == 'instant':
      feat = self.subj_rssm.get_feat(subj_states)
    elif expand:
      complete_shapes = lambda x: tf.expand_dims(x, 1)
      start_state = tf.nest.map_structure(complete_shapes, start_state)
    if self.strategy == 'reactive':
      states = {k: tf.concat([tf.cast(start_state[k], v.dtype), 
                              subj_states[k][:, :-1]], 1) for k, v in subj_states.items()}
      feat = self.subj_rssm.get_feat(states)
    elif self.strategy == 'delta':
      states = {k: tf.concat([tf.cast(start_state[k], v.dtype), 
                              subj_states[k]], 1) for k, v in subj_states.items()}
      feat = self.subj_rssm.get_feat(states)
      diff = lambda x: x[:, 1:] - x[:, :-1]
      feat = tf.nest.map_structure(diff, feat)
    return tf.stop_gradient(feat)

  def objective_input(self, subj_states):
    feat = self.subj_rssm.get_feat(subj_states)
    return tf.stop_gradient(feat)

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    subj_post, subj_prior = self.subj_rssm.observe(embed['subj'], action, state['subj'])
    subj_actions = self.subj_action(state['subj'], subj_post)
    obj_post, obj_prior = self.obj_rssm.observe(embed['obj'], subj_actions, state['obj'])
    post = {'subj': subj_post, 'obj': obj_post}
    prior = {'subj': subj_prior, 'obj': obj_prior}
    if self.foresight and self.strategy == 'reactive':
      goal_states = self.add_last_imag(subj_post, obj_post)
      post['goal'] = goal_states
    return post, prior

  def add_last_imag(self, subj, obj):
    last_subj = {k: subj[k][:, -1] for k in subj.keys()} 
    subj_action = self.subj_rssm.get_feat(last_subj)
    last_obj = {k: obj[k][:, -1] for k in obj.keys()}
    imag_obj = self.obj_rssm.img_step(last_obj, subj_action)
    goal_states = {k: tf.concat(
        [obj[k][:, 1:], tf.expand_dims(imag_obj[k], 1)], 1) for k in obj.keys()}
    return goal_states

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    subj_prior = self.subj_rssm.imagine(action, state['subj'])
    subj_action = self.subj_action(state['subj'], subj_prior)
    obj_prior = self.obj_rssm.imagine(subj_action, state['obj'])
    prior = {'subj': subj_prior, 'obj': obj_prior}
    if self.foresight and self.strategy == 'reactive':
      goal_states = self.add_last_imag(subj_prior, obj_prior)
      prior['goal'] = goal_states
    return prior

  def get_feat(self, state):
    subj_feat = self.subj_rssm.get_feat(state['subj'])
    obj_feat = self.obj_rssm.get_feat(state['obj'])
    if self.foresight and self.strategy == 'reactive' and 'goal' in state:
      goal_feat = self.obj_rssm.get_feat(state['goal'])
      # TODO: two options [subj_feat, obj_feat, goal_feat]
      # and: [subj_feat, goal_feat]
      return tf.concat([subj_feat, obj_feat, goal_feat], -1)  
      # return tf.concat([subj_feat, goal_feat], -1)  
    return tf.concat([subj_feat, obj_feat], -1)

  def get_dist(self, state):
    subj_dist = self.subj_rssm.get_dist(state['subj'])
    obj_dist = self.obj_rssm.get_dist(state['obj'])
    return {'subj': subj_dist, 'obj': obj_dist}

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    subj_post, subj_prior = self.subj_rssm.obs_step(prev_state['subj'], prev_action, embed['subj'], sample)
    subj_action = self.step_subj_action(prev_state['subj'], subj_post)
    obj_post, obj_prior = self.obj_rssm.obs_step(prev_state['obj'], subj_action, embed['obj'], sample)
    # if goal_agent and reactive:
    post = {'subj': subj_post, 'obj': obj_post}
    prior = {'subj': subj_prior, 'obj': obj_prior}
    if self.foresight and self.strategy == 'reactive':
      # double checked
      next_subj_action = self.subj_rssm.get_feat(subj_post)
      obj_goal = self.obj_rssm.img_step(obj_post, next_subj_action, sample=True)
      post['goal'] = obj_goal
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    subj_prior = self.subj_rssm.img_step(prev_state['subj'], prev_action, sample)
    subj_action = self.step_subj_action(prev_state['subj'], subj_prior)
    if self.strategy == 'reactive' and 'goal' in prev_state:
      obj_prior = prev_state['goal']
    else:
      obj_prior = self.obj_rssm.img_step(prev_state['obj'], subj_action, sample)
    prior = {'subj': subj_prior, 'obj': obj_prior}
    if self.foresight and self.strategy == 'reactive':
      # todo: by this we will sample from the obj prior twice. 
      # therefore we won't condition the agent by the traj. we'll go over.
      goal_state = self.obj_rssm.img_step(obj_prior, self.subj_rssm.get_feat(subj_prior), sample)
      prior['goal'] = goal_state
    return prior

  def kl_loss(self, post, prior, **kwargs):
    subj_loss, subj_value = self.subj_rssm.kl_loss(post['subj'], prior['subj'], **kwargs)
    obj_loss, obj_value = self.obj_rssm.kl_loss(post['obj'], prior['obj'], **kwargs)
    loss = {'subj': subj_loss, 'obj': obj_loss}
    value = {'subj': subj_value, 'obj': obj_value}
    return loss, value


class MutualRSSM(common.Module):

  def __init__(self, subj_config, obj_config, subj_strategy):
    super().__init__()
    self.subj_rssm = RSSM(**subj_config)
    self.obj_rssm = RSSM(**obj_config)
    self.strategy = subj_strategy
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)
    self._dtype = prec.global_policy().compute_dtype

  def initial(self, batch_size):
    subj_state = self.subj_rssm.initial(batch_size)
    obj_state = self.obj_rssm.initial(batch_size)
    state = {'subj': subj_state, 'obj': obj_state}
    return state

  def step_obj_action(self, prev_subj_states, curr_subj_states):
    if self.strategy == 'instant':
      feat = self.subj_rssm.get_feat(curr_subj_states)
    elif self.strategy == 'reactive':
      feat = self.subj_rssm.get_feat(prev_subj_states)
    elif self.strategy == 'delta':
      curr_feats = self.subj_rssm.get_feat(curr_subj_states)
      prev_feats = self.subj_rssm.get_feat(prev_subj_states)
      diff = lambda x, y: x - y 
      feat = tf.nest.map_structure(diff, curr_feats, prev_feats)
    return tf.stop_gradient(feat)

  def obj_action(self, start_state, subj_states):
    if self.strategy == 'instant':
      feat = self.subj_rssm.get_feat(subj_states)
    elif self.strategy == 'reactive':
      complete_shapes = lambda x: tf.expand_dims(x, 1) if len(x.shape) < 3 else x
      start_state = tf.nest.map_structure(complete_shapes, start_state)
      states = {k: tf.concat([tf.cast(start_state[k], v.dtype), 
                              subj_states[k][:, :-1]], 1) for k, v in subj_states.items()}
      feat = self.subj_rssm.get_feat(states)
    elif self.strategy == 'delta':
      complete_shapes = lambda x: tf.expand_dims(x, 1) if len(x.shape) < 3 else x
      start_state = tf.nest.map_structure(complete_shapes, start_state)
      states = {k: tf.concat([tf.cast(start_state[k], v.dtype), 
                              subj_states[k]], 1) for k, v in subj_states.items()}
      feat = self.subj_rssm.get_feat(states)
      diff = lambda x: x[:, 1:] - x[:, :-1]
      feat = tf.nest.map_structure(diff, feat)
    return tf.stop_gradient(feat)

  def subj_action(self, obj_state, action):
    action = self._cast(action)
    obj_feat = self.obj_rssm.get_feat(obj_state)
    subj_action = tf.concat([tf.stop_gradient(obj_feat), action], -1)
    return subj_action

  @tf.function
  def observe(self, embed, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed = tf.nest.map_structure(swap, embed)
    action = swap(action)
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = tf.nest.map_structure(swap, post)
    prior = tf.nest.map_structure(swap, prior)
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = tf.nest.map_structure(swap, prior)
    return prior

  def get_feat(self, state):
    subj_feat = self.subj_rssm.get_feat(state['subj'])
    obj_feat = self.obj_rssm.get_feat(state['obj'])
    return tf.concat([subj_feat, obj_feat], -1)

  def get_dist(self, state):
    subj_dist = self.subj_rssm.get_dist(state['subj'])
    obj_dist = self.obj_rssm.get_dist(state['obj'])
    return {'subj': subj_dist, 'obj': obj_dist}

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    subj_action = self.subj_action(prev_state['obj'], prev_action)
    subj_post, subj_prior = self.subj_rssm.obs_step(prev_state['subj'], subj_action, embed['subj'], sample)
    obj_action = self.step_obj_action(prev_state['subj'], subj_post)

    obj_post, obj_prior = self.obj_rssm.obs_step(prev_state['obj'], obj_action, embed['obj'], sample)
    post = {'subj': subj_post, 'obj': obj_post}
    prior = {'subj': subj_prior, 'obj': obj_prior}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    subj_action = self.subj_action(prev_state['obj'], prev_action)
    subj_prior = self.subj_rssm.img_step(prev_state['subj'], subj_action, sample)
    obj_action = self.step_obj_action(prev_state['subj'], subj_prior)

    obj_prior = self.obj_rssm.img_step(prev_state['obj'], obj_action, sample)
    prior = {'subj': subj_prior, 'obj': obj_prior}
    return prior

  def kl_loss(self, post, prior, **kwargs):
    subj_loss, subj_value = self.subj_rssm.kl_loss(post['subj'], prior['subj'], **kwargs)
    obj_loss, obj_value = self.obj_rssm.kl_loss(post['obj'], prior['obj'], **kwargs)
    loss = {'subj': subj_loss, 'obj': obj_loss}
    value = {'subj': subj_value, 'obj': obj_value}
    return loss, value

class ConvEncoder(common.Module):

  def __init__(
      self, depth=32, act=tf.nn.elu, kernels=(4, 4, 4, 4), depths=None, strides=None, keys=['image'], rect=False):
    #TODO add rect key
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._depth = depth
    self._depths = depths
    self._kernels = kernels
    self._rect = rect
    self._strides = strides
    self._keys = keys

  @tf.function
  def __call__(self, obs):
    if tuple(self._keys) == ('image',):
      x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
      if self._rect:
        x = self._act(self.get(f'h0', tfkl.Conv2D, 1 * self._depth, 4, 2)(x))
        x = self._act(self.get(f'h1', tfkl.Conv2D, 2 * self._depth, 4, 2)(x))
        x = self._act(self.get(f'h2', tfkl.Conv2D, 4 * self._depth, 4, 2)(x))
        x = self._act(self.get(f'h3', tfkl.Conv2D, 4 * self._depth, 3, (1, 2))(x))
        x = self._act(self.get(f'h4', tfkl.Conv2D, 8 * self._depth, 3, 1)(x))
      else:
        for i, kernel in enumerate(self._kernels):
          depth = 2 ** i * self._depth
          x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
      x_shape = tf.shape(x)
      x = tf.reshape(x, [x_shape[0], tf.math.reduce_prod(x_shape[1:])])
      shape = tf.concat([tf.shape(obs['image'])[:-3], tf.shape(x)[-1:]], 0)
      return tf.reshape(x, shape)
    else:
      dtype = prec.global_policy().compute_dtype
      features = []
      for key in self._keys:
        value = tf.convert_to_tensor(obs[key])
        if value.dtype.is_integer:
          value = tf.cast(value, dtype)
          semilog = tf.sign(value) * tf.math.log(1 + tf.abs(value))
          features.append(semilog[..., None])
        elif len(obs[key].shape) >= 4:
          x = tf.reshape(obs[key], (-1,) + tuple(obs[key].shape[-3:]))
          if self._rect:
            x = self._act(self.get(f'h0', tfkl.Conv2D, 1 * self._depth, 4, 2)(x))
            x = self._act(self.get(f'h1', tfkl.Conv2D, 2 * self._depth, 4, 2)(x))
            x = self._act(self.get(f'h2', tfkl.Conv2D, 4 * self._depth, 4, 2)(x))
            x = self._act(self.get(f'h3', tfkl.Conv2D, 4 * self._depth, 3, (1, 2))(x))
            x = self._act(self.get(f'h4', tfkl.Conv2D, 8 * self._depth, 3, 1)(x))
          else:
            for i, kernel in enumerate(self._kernels):
              depth = 2 ** i * self._depth
              x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
          x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
          shape = tf.concat([tf.shape(obs[key])[:-3], [x.shape[-1]]], 0)
          features.append(tf.reshape(x, shape))
        else:
          raise NotImplementedError((key, value.dtype, value.shape))
      return tf.concat(features, -1)


class DualConvEncoder(common.Module):

  def __init__(self, subj_config, obj_config):
    super().__init__()
    self.subj_encoder = ConvEncoder(**subj_config)
    self.obj_encoder = ConvEncoder(**obj_config)

  @tf.function
  def __call__(self, obs):
    subj_embed = self.subj_encoder(obs)
    obj_embed = self.obj_encoder(obs)
    embed = {'subj': subj_embed, 'obj': obj_embed}
    return embed


class ConvDecoder(common.Module):

  def __init__(
      self, shape=(64, 64, 3), depth=32, act=tf.nn.elu, kernels=(5, 5, 6, 6)):
    self._shape = shape
    self._depth = depth
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._kernels = kernels

  def __call__(self, features):
    ConvT = tfkl.Conv2DTranspose
    x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        depth = self._shape[-1]
        act = None
      x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class MLP(common.Module):

  def __init__(self, shape, layers, units, act=tf.nn.elu, dist_layer=True, **out):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._dist_layer = dist_layer
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._out = out

  def __call__(self, features):
    x = tf.cast(features, prec.global_policy().compute_dtype)
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    if self._dist_layer:
      return self.get('out', DistLayer, self._shape, **self._out)(x)
    else:
      if x.shape[-1] == self._shape[0]:
        return x
      else:
        return self.get(f'hout', tfkl.Dense, self._shape[0], tf.identity)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class AddressNet(common.Module):
  def __init__(self, hidden=200, act=tf.nn.elu):
    super().__init__()
    self.hidden = hidden
    self.act = act
    self._cell = tfkl.GRUCell(self.hidden)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return self._cell.get_initial_state(None, batch_size, dtype)

  def embed(self, obs, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    obs = tf.transpose(obs, [1, 0, 2])
    action = tf.transpose(action, [1, 0, 2])
    states = common.static_scan(
        lambda prev, inputs: self.step(prev, *inputs),
        (action, obs), state)
    return states
    
  @tf.function
  def step(self, prev_state, prev_action, state):
    x = tf.concat([prev_state, prev_action, state], -1)
    x = self.get('l1', tfkl.Dense, self.hidden, self.act)(x)
    x, new_state = self._cell(x, [prev_state])
    return new_state[0]

class DyneAddressNet(common.Module):
  def __init__(self, dyne, hidden=200, act=tf.nn.elu, obs_strategy='skip'):
    super().__init__()
    self.dyne = dyne
    self.action_size = dyne.action_size
    self.traj_len = dyne.traj_len
    self.hidden = hidden
    self.act = act
    self.obs_strategy = obs_strategy
    self._cell = tfkl.GRUCell(self.hidden)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return self._cell.get_initial_state(None, batch_size, dtype)

  def embed(self, obs, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    obs = self.prepare_obs(obs)
    action = self.prepare_action(action)
    obs = swap(obs)
    action = swap(action)
    states = common.static_scan(
        lambda prev, inputs: self.step(prev, *inputs),
        (action, obs), state)
    return states

  @tf.function
  def step(self, prev_state, prev_action, state):
    _, prev_action, _ = self.dyne.embed_action(prev_action)
    prev_action = tf.repeat(prev_action, self.dyne.action_repeat, 1)
    _, state, _ = self.dyne.embed_obs(state)
    prev_action = tf.stop_gradient(self._cast(prev_action))
    state = tf.stop_gradient(self._cast(state))
    x = tf.concat([prev_state, prev_action, state], -1)
    x = self.get('l1', tfkl.Dense, self.hidden, self.act)(x)
    x, new_state = self._cell(x, [prev_state])
    return new_state[0]

  def prepare_obs(self, obs):
    if self.obs_strategy == 'skip':
      obs = obs[:, ::self.traj_len]
    return obs

  def prepare_action(self, action):
    action = tf.reshape(action, [action.shape[0], -1, self.traj_len, self.action_size])
    return action

class DistLayer(common.Module):

  def __init__(self, shape, dist='mse', min_std=0.1, init_std=0.0):
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std

  def __call__(self, inputs):
    out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal', 'logvar_normal'):
      std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
      std = tf.cast(std, tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'logvar_normal':
      std = tf.math.exp(0.5 * std)
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * tf.tanh(out / 5)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, common.TanhBijector())
      dist = tfd.Independent(dist, len(self._shape))
      return common.SampleDist(dist)
    if self._dist == 'trunc_normal':
      std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return common.OneHotDist(out)
    NotImplementedError(self._dist)
