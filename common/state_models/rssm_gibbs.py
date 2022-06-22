import math
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common
from .rssm import RSSM

class RSSM_GIBBS(RSSM):

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

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, current_state_subj, current_state_obj, task_vec=None, sample=True):
    if current_state_subj is None:
      prior = self.img_step(prev_state, prev_action, sample)
    else:
      prior = current_state_subj
    # x = tf.concat([prior['deter'], embed, current_state_obj['deter']], -1)
    x = tf.concat([prior['deter'], embed], -1)
    print('RSSM_GIBBS x', x)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior