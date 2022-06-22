import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras.mixed_precision import experimental as prec

from common.nets import GRUCell
from common.state_models import RSSM
from common.other import static_scan


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
    self._cell = GRUCell(self._deter, norm=True)
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
    post, prior = static_scan(
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
    prior = static_scan(self.img_step, prior_upd, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  @tf.function
  def obs_step(self, prev_state, current_state_obj, post_update, current_state_subj, task_vec=None, sample=True):
    post_update = self._cast(post_update)

    print('ReasonerMLP post_update', post_update)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(post_update)
    print('ReasonerMLP x', x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    print('ReasonerMLP stoch', stoch)
    latent = {'stoch': stoch, 'deter': current_state_obj['deter'], 'out': x, **stats}
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