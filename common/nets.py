import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class PostPriorNet(common.Module):
  def get_dist(self, state):
    raise NotImplemented

  def _suff_stats_layer(self, name, x):
    raise NotImplemented

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


class ConditionModel(PostPriorNet):
  def __init__(self, size=32, hidden=200, layers=2, act=tf.nn.elu, discrete=False):
    super().__init__()
    self._size = size
    self._stoch = size
    self.forward_cond = MLP(shape=[size], units=hidden, layers=layers-1, dist_layer=False, act=act)
    self.backward_cond = MLP(shape=[size], units=hidden, layers=layers-1, dist_layer=False, act=act)
    self._discrete = discrete
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def imagine(self, state, sample=True):
    emb = self.forward_cond(state)
    stats = self._suff_stats_layer('img', emb)
    dist = self.get_dist(stats)
    condition = dist.sample() if sample else dist.mode()
    return {'stoch': condition, **stats}
  
  def observe(self, state, sample=True):
    emb = self.backward_cond(state)
    stats = self._suff_stats_layer('obs', emb)
    dist = self.get_dist(stats)
    condition = dist.sample() if sample else dist.mode()
    return {'stoch': condition, **stats}

  def get_feat(self, state):
    return self._cast(state['stoch'])

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
        logit=tf.zeros([batch_size, self._size, self._discrete], dtype),
        stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype))
    else:
      state = dict(
        mean=tf.zeros([batch_size, self._size], dtype),
        std=tf.zeros([batch_size, self._size], dtype),
        stoch=tf.zeros([batch_size, self._size], dtype))
    return state

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
  
  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._size * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._size, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._size, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = 2 * tf.nn.sigmoid(std / 2)
      return {'mean': mean, 'std': std}


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
