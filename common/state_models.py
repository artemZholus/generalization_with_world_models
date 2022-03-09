import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class RSSM(common.StochPostPriorNet, common.DeterPostPriorNet):

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
    self._cell = common.GRUCell(self._deter, norm=True)
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
  def observe(self, embed, action, state=None, **kws):
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

  def get_stoch(self, state):
    return self._cast(state['stoch'])

  def get_deter(self, state):
    return state['deter']

  def get_feat(self, state, key=None):
    stoch = self.get_stoch(state)
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    deter = self.get_deter(state)
    return tf.concat([stoch, deter], -1)

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

  def loss(self, post, prior, **kwargs):
    kl_loss, kl_value = self.kl_loss(post, prior, **kwargs)
    return {'kl': kl_loss}, {'kl': kl_value}


class Reasoner(RSSM):
  def __init__(
      self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1
    ):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self._cell = common.GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          curr_state=dict(
            deter=self._cell.get_initial_state(None, batch_size, dtype),
            x=tf.zeros([batch_size, self._hidden], dtype)
          )
      )
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
           curr_state=dict(
            deter=self._cell.get_initial_state(None, batch_size, dtype),
            x=tf.zeros([batch_size, self._hidden], dtype)
          )
      )
    return state

  @tf.function
  def observe(self, post_upd, prior_upd, state=None, **kws):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(prior_upd)[0])
    post_upd, prior_upd = swap(post_upd), swap(prior_upd)
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (prior_upd, post_upd), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, prior_upd, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(prior_upd)[0])
    assert isinstance(state, dict), state
    prior_upd = swap(prior_upd)
    prior = common.static_scan(self.img_step, prior_upd, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  @tf.function
  def obs_step(self, prev_state, post_update, curr_state=None, task_vec=None, sample=True):
    if curr_state is None:
      curr_state = self.trans_step(prev_state)
    post = self.update_step(curr_state, self._cast(post_update), name='obs', sample=sample)
    return post

  @tf.function
  def img_step(self, prev_state, prior_update, curr_state=None, task_vec=None, sample=True):
    if curr_state is None:
      curr_state = self.trans_step(prev_state)
    if task_vec is not None:
      prior_update = self._cast(tf.concat([prior_update, task_vec], -1))
    else:
      prior_update = self._cast(prior_update)
    prior = self.update_step(curr_state, prior_update, name='img', sample=sample)
    return prior

  def update_step(self, curr_state, update, name='', sample=True):
    state_emb = curr_state['out']
    x = tf.concat([state_emb, update], -1)
    x = self.get(f'{name}_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer(f'{name}_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    latent = {'stoch': stoch, 'curr_state': curr_state, **stats}
    return latent

  @tf.function
  def trans_step(self, prev_state):
    prev_stoch = self._cast(prev_state['stoch'])
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = self.get('trans_in', tfkl.Dense, self._hidden, self._act)(prev_stoch)
    deter = prev_state['curr_state']['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('trans_out', tfkl.Dense, self._hidden, self._act)(x)
    state = {'out': x, 'deter': deter}
    return state

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['curr_state']['deter']], -1)

class ReasonerMLP(RSSM):
  def __init__(
      self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1
    ):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self._cell = common.GRUCell(self._deter, norm=True)
    self._stoch_features = True
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype),
          out=tf.zeros([batch_size, self._hidden], dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype),
          out=tf.zeros([batch_size, self._hidden], dtype))
    return state

  @tf.function
  def observe(self, post_upd, prior_upd, state=None, **kws):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(prior_upd)[0])
    post_upd, prior_upd = swap(post_upd), swap(prior_upd)
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (prior_upd, post_upd), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, prior_upd, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(prior_upd)[0])
    assert isinstance(state, dict), state
    prior_upd = swap(prior_upd)
    prior = common.static_scan(self.img_step, prior_upd, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  @tf.function
  def obs_step(self, prev_state, current_state, post_update, task_vec=None, sample=True):
    post_update = self._cast(post_update)

    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(post_update)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    latent = {'stoch': stoch, 'deter': current_state['deter'], 'out': x, **stats}
    return latent

  @tf.function
  def img_step(self, prev_state, prior_update, task_vec=None, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    if task_vec is not None:
      prior_update = self._cast(tf.concat([prior_update, task_vec], -1))
    else:
      prior_update = self._cast(prior_update)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prior_update], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden, self._act)(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('img_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    latent = {'stoch': stoch, 'deter': deter, 'out': x, **stats}
    return latent

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    if self._stoch_features:
      return stoch
    else:
      return tf.concat([stoch, state['deter']], -1)

class Reasoner2Rnn(RSSM):
  def __init__(
      self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1
    ):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self.post_cell = common.GRUCell(self._deter, norm=True)
    self.prio_cell = common.GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size, prior=True):
    dtype = prec.global_policy().compute_dtype
    key = 'prio' if prior else 'post'
    if self._discrete:
      state = {
          'logit': tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          'stoch': tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          f'curr_state_{key}': dict(
            deter=self._cell.get_initial_state(None, batch_size, dtype),
            x=tf.zeros([batch_size, self._hidden], dtype)
          )
      }
    else:
      state = {
          'mean': tf.zeros([batch_size, self._stoch], dtype),
          'std': tf.zeros([batch_size, self._stoch], dtype),
          'stoch': tf.zeros([batch_size, self._stoch], dtype),
          f'curr_state_{key}': dict(
            deter=self._cell.get_initial_state(None, batch_size, dtype),
            x=tf.zeros([batch_size, self._hidden], dtype)
          ),
      }
    return state

  @tf.function
  def observe(self, post_upd, prior_upd, state=None, **kws):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      post_state = self.initial(tf.shape(prior_upd)[0], prior=False)
      prior_state = self.initial(tf.shape(prior_upd)[0], prior=True)
    post_upd, prior_upd = swap(post_upd), swap(prior_upd)
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (prior_upd, post_upd), (post_state, prior_state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, prior_upd, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(prior_upd)[0])
    assert isinstance(state, dict), state
    prior_upd = swap(prior_upd)
    prior = common.static_scan(self.img_step, prior_upd, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  # @tf.function
  # def obs_step(self, prev_state, post_update, curr_state=None, task_vec=None, sample=True):
  #   if curr_state is None:
  #     curr_state = self.trans_step(prev_state)
  #   post = self.update_step(curr_state, self._cast(post_update), name='obs', sample=sample)
  #   return post

  # @tf.function
  # def img_step(self, prev_state, prior_update, curr_state=None, task_vec=None, sample=True):
  #   if curr_state is None:
  #     curr_state = self.trans_step(prev_state)
  #   if task_vec is not None:
  #     prior_update = self._cast(tf.concat([prior_update, task_vec], -1))
  #   else:
  #     prior_update = self._cast(prior_update)
  #   prior = self.update_step(curr_state, prior_update, name='img', sample=sample)
  #   return prior

  # def update_step(self, curr_state, update, name='', sample=True):
  #   state_emb = curr_state['out']
  #   x = tf.concat([state_emb, update], -1)
  #   x = self.get(f'{name}_out', tfkl.Dense, self._hidden, self._act)(x)
  #   stats = self._suff_stats_layer(f'{name}_dist', x)
  #   dist = self.get_dist(stats)
  #   stoch = dist.sample() if sample else dist.mode()
  #   latent = {'stoch': stoch, 'curr_state': curr_state, **stats}
  #   return latent

  @tf.function
  def obs_step(self, prev_state, post_update, task_vec=None, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    post_update = self._cast(post_update)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, post_update], -1)
    x = self.get('obs_in', tfkl.Dense, self._hidden, self._act)(x)
    deter = prev_state['curr_state_post']['deter']
    x, deter = self.post_cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    curr_state = {'out': x, 'deter': deter}
    latent = {'stoch': stoch, 'curr_state_post': curr_state, **stats}
    return latent

  @tf.function
  def img_step(self, prev_state, prior_update, task_vec=None, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    if task_vec is not None:
      prior_update = self._cast(tf.concat([prior_update, task_vec], -1))
    else:
      prior_update = self._cast(prior_update)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prior_update], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden, self._act)(x)
    if 'curr_state_post' in prev_state:
      # use posterior rnn state for inference
      deter = prev_state['curr_state_post']['deter']
    elif 'curr_state_prio' in prev_state:
      # use prior states for imagination
      deter = prev_state['curr_state_prio']['deter']
    x, deter = self.prio_cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('img_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    curr_state = {'out': x, 'deter': deter}
    latent = {'stoch': stoch, 'curr_state_prio': curr_state, **stats}
    return latent

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    if 'curr_state_post' in state:
      return tf.concat([stoch, state['curr_state_post']['deter']], -1)
    elif 'curr_state_prio' in state:
      return tf.concat([stoch, state['curr_state_prio']['deter']], -1)


class DualReasoner(RSSM):
  def __init__(
    self,
    # these are base kwargs
    stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu, std_act='softplus', min_std=0.1,
    # per layer specific kwargs
    cond_kws=None, subj_kws=None, obj_kws=None,
    feature_sets=None
  ):
    # cond_stoch=50,
    if cond_kws is None:
      cond_kws = {}
    else:
      cond_kws = dict(**cond_kws)
    cond_kws = dict(cond_kws)
    cond_kws['hidden'] = cond_kws.get('hidden', hidden)
    cond_kws['act'] = cond_kws.get('act', act)
    cond_kws['discrete'] = cond_kws.get('discrete', discrete)
    cond_kws['layers'] = cond_kws.get('layers', 2)
    cond_kws['size'] = cond_kws.get('size', 50) # 50 is default cond model stoch size

    if subj_kws is None:
      subj_kws = {}
    else:
      subj_kws = dict(**subj_kws)
    subj_kws['stoch'] = subj_kws.get('stoch', stoch)
    subj_kws['deter'] = subj_kws.get('deter', deter)
    subj_kws['hidden'] = subj_kws.get('hidden', hidden)
    subj_kws['discrete'] = subj_kws.get('discrete', discrete)
    subj_kws['act'] = subj_kws.get('act', act)
    subj_kws['std_act'] = subj_kws.get('std_act', std_act)
    subj_kws['min_std'] = subj_kws.get('mid_std', min_std)

    if obj_kws is not None:
      obj_kws = {}
    else:
      obj_kws = dict(**obj_kws)
    obj_kws['stoch'] = obj_kws.get('stoch', stoch)
    obj_kws['deter'] = obj_kws.get('deter', deter)
    obj_kws['hidden'] = obj_kws.get('hidden', hidden)
    obj_kws['discrete'] = obj_kws.get('discrete', discrete)
    obj_kws['act'] = obj_kws.get('act', act)
    obj_kws['std_act'] = obj_kws.get('std_act', std_act)
    obj_kws['min_std'] = obj_kws.get('mid_std', min_std)

    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self.feature_sets = [] if feature_sets is None else feature_sets
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)
    self.subj_reasoner = RSSM(**subj_kws)
    self.condition_model = common.MLPRNNConditionModel(**cond_kws)
    self.obj_reasoner = ReasonerMLP(**obj_kws)

  @tf.function
  def top_down_step(self, prev_state, emb_obj=None, emb_subj=None, action=None, task_vec=None, current_state=None, sample=True):
    prev_subj, prev_obj = prev_state['subj'], prev_state['obj']
    if current_state is not None:
      current_subj, current_obj = current_state['subj'], current_state['obj']
    else:
      current_subj, current_obj = None, None
    # obj inference
    post_update_obj = emb_obj
    post_obj = self.obj_reasoner.obs_step(prev_state=prev_obj,
                                          current_state=current_obj,
                                          post_update=post_update_obj,
                                          sample=sample)
    # util inference
    post_update_util = self.obj_reasoner.get_feat(post_obj)
    post_util = self.condition_model.obs_step(state=None,
                                              post_update=post_update_util,
                                              sample=sample)
    # subj inference
    post_util_feat = self.condition_model.get_feat(post_util)
    post_update_subj = tf.concat([post_util_feat, emb_subj], -1)
    post_subj, _ = self.subj_reasoner.obs_step(prev_state=prev_subj, 
                                               embed=post_update_subj, 
                                               prev_action=action, 
                                               sample=sample)
    return {'subj': post_subj, 'obj': post_obj, 'util': post_util}

  def mut_inf(self, sample, kind='obj'):
    NUM_SAMPLES = 5
    dist = self.get_dist(sample)
    stoch = dist[kind].sample(NUM_SAMPLES)
    curr_prob = dist[kind].log_prob(stoch)
    mu, sigma = sample[kind]['mean'], sample[kind]['std']
    mu = tf.expand_dims(mu, 2)
    sigma = tf.expand_dims(sigma, 2)
    if kind == 'obj':
      expand_dist = self.obj_reasoner.get_dist({'mean': mu, 'std': sigma})
    elif kind == 'subj':
      expand_dist = self.subj_reasoner.get_dist({'mean': mu, 'std': sigma})
    elif kind == 'util':
      expand_dist = self.condition_model.get_dist({'mean': mu, 'std': sigma})
    stoch = tf.expand_dims(stoch, 2)
    prob = expand_dist.log_prob(stoch)
    marginal_prob = prob.logsumexp(2) - math.log(prob.shape[2])
    mi = (curr_prob - marginal_prob).mean()
    return mi

  @tf.function
  def bottom_up_step(self, prev_state, action, task_vec=None, current_state=None, sample=True):
    prev_subj, prev_obj = prev_state['subj'], prev_state['obj']
    if current_state is not None:
      current_subj, current_obj = current_state['subj'], current_state['obj']
    else:
      current_subj, current_obj = None, None
    # subj imagination
    prior_subj = self.subj_reasoner.img_step(prev_state=prev_subj,
                                             prev_action=action,
                                             #curr_state=subj_curr_state,
                                             sample=sample)
    # util imagination
    prior_subj_feat = self.subj_reasoner.get_feat(prior_subj)
    # prior_update_util = tf.stop_gradient(prior_update_util)
    if task_vec is not None:
      task_vec = self._cast(task_vec)
      prior_update_util = tf.concat([prior_subj_feat, task_vec], -1)
    prior_util = self.condition_model.img_step(state=None,
                                               prior_update=prior_update_util, 
                                               sample=sample)
    # obj imagination
    prior_update_subj = self.condition_model.get_feat(prior_util)
    prior_obj = self.obj_reasoner.img_step(prev_state=prev_obj,
                                           prior_update=prior_update_subj,
                                           sample=sample)
    return {'subj': prior_subj, 'obj': prior_obj, 'util': prior_util}

  @tf.function
  def img_step(self, state, action, task_vec=None, sample=True):
    return self.bottom_up_step(state, action, task_vec=task_vec, sample=sample)

  def get_feat(self, state, key=None, task_vec=None):
    features = []
    for name, feat_set in self.feature_sets.items():
      if key is not None and name in key:
        for feat in feat_set:
          if feat == 'subj':
            feat_vec = self.subj_reasoner.get_feat(state['subj'])
          elif feat == 'obj':
            feat_vec = self.obj_reasoner.get_feat(state['obj'])
          elif feat == 'util':
            feat_vec = self.condition_model.get_feat(state['util'])
          elif feat == 'task':
            feat_vec = self._cast(task_vec)
          features.append(feat_vec)
        return tf.concat(features, -1)
    subj_feat = self.subj_reasoner.get_feat(state['subj'])
    obj_feat = self.obj_reasoner.get_feat(state['obj'])
    util_feat = self.condition_model.get_feat(state['util'])
    return tf.concat([subj_feat, util_feat, obj_feat], -1)

  def get_dist(self, state):
    subj_dist = self.subj_reasoner.get_dist(state['subj'])
    obj_dist = self.obj_reasoner.get_dist(state['obj'])
    util_dist = self.condition_model.get_dist(state['util'])
    return {'subj': subj_dist, 'obj': obj_dist, 
            'util': util_dist
            }

  @tf.function
  def obs_step(self, state, action, emb, task_vec=None, sample=True):
    # TODO: do not infer rnn twice
    prior = self.bottom_up_step(state, action,
                                task_vec=task_vec,
                                sample=sample)
    post = self.top_down_step(state, emb['obj'], emb['subj'],
                              action=action, task_vec=task_vec,
                              current_state=prior, sample=sample)
    return post, prior

  def initial(self, batch_size):
    return {
      'obj': self.obj_reasoner.initial(batch_size),
      'subj': self.subj_reasoner.initial(batch_size),
      'util': self.condition_model.initial(batch_size)
    }

  @tf.function
  def observe(self, emb, actions, state=None, task_vector=None):
    subj_emb, obj_emb = emb['subj'], emb['obj']
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(actions)[0])
    subj_emb, obj_emb, actions = swap(subj_emb), swap(obj_emb), swap(actions)
    if task_vector is not None:
      task_vector = swap(task_vector)
    emb = {'obj': obj_emb, 'subj': subj_emb}
    tpl = (actions, emb)
    if task_vector is not None:
      tpl = (actions, emb, task_vector)
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        tpl, (state, state))
    post = tf.nest.map_structure(swap, post)
    prior = tf.nest.map_structure(swap, prior)
    return post, prior

  @tf.function
  def imagine(self, action, state=None, task_vec=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    if task_vec is not None:
      task_vec = swap(task_vec)
      tpl = (action, task_vec)
    else:
      tpl = (action,)
    prior = common.static_scan(lambda prev, inp: self.bottom_up_step(prev, *inp), tpl, state)
    prior = tf.nest.map_structure(swap, prior)
    return prior

  def loss(self, post, prior, **kwargs):
    # subj loss
    subj_loss, subj_value = self.subj_reasoner.loss(post['subj'], prior['subj'], **kwargs.get('subj', {}))
    subj_loss = {f'subj_{name}': loss for name, loss in subj_loss.items()}
    subj_value = {f'subj_{name}': loss for name, loss in subj_value.items()}
    # obj loss
    obj_loss, obj_value = self.obj_reasoner.loss(post['obj'], prior['obj'], **kwargs.get('obj', {}))
    obj_loss = {f'obj_{name}': loss for name, loss in obj_loss.items()}
    obj_value = {f'obj_{name}': loss for name, loss in obj_value.items()}
    # util loss
    util_loss, util_value = self.condition_model.loss(post['util'], prior['util'], **kwargs.get('util', {}))
    util_loss = {f'util_{name}': loss for name, loss in util_loss.items()}
    util_value = {f'util_{name}': loss for name, loss in util_value.items()}
    # losses and values
    loss = {}
    loss.update(**subj_loss, **obj_loss, **util_loss)
    value = {}
    value.update(**subj_value, **obj_value, **util_value)
    return loss, value


class DualRSSM(common.Module):

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
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    subj_prior = self.subj_rssm.imagine(action, state['subj'])
    subj_action = self.subj_action(state['subj'], subj_prior)
    obj_prior = self.obj_rssm.imagine(subj_action, state['obj'])
    return {'subj': subj_prior, 'obj': obj_prior}

  def get_feat(self, state, key=None):
    if 'subj' in key:
      subj_feat = self.subj_rssm.get_feat(state['subj'])
      return subj_feat
    elif 'obj' in key:
      obj_feat = self.obj_rssm.get_feat(state['obj'])
      return obj_feat
    else:
      subj_feat = self.subj_rssm.get_feat(state['subj'])
      obj_feat = self.obj_rssm.get_feat(state['obj'])
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
    post = {'subj': subj_post, 'obj': obj_post}
    prior = {'subj': subj_prior, 'obj': obj_prior}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    subj_prior = self.subj_rssm.img_step(prev_state['subj'], prev_action, sample)
    subj_action = self.step_subj_action(prev_state['subj'], subj_prior)
    obj_prior = self.obj_rssm.img_step(prev_state['obj'], subj_action, sample)
    prior = {'subj': subj_prior, 'obj': obj_prior}
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
