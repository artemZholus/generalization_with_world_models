import tensorflow as tf
from tensorflow.keras import layers as tfkl

from common.state_models import ReasonerMLP


class ReasonerMLPGibbs(ReasonerMLP):
  @tf.function
  def obs_step(self, prev_state, current_state_obj, post_update, current_state_subj, task_vec=None, sample=True):
    post_update = self._cast(post_update)

    x = tf.concat([post_update, current_state_subj['deter'], tf.cast(current_state_subj['stoch'], tf.float16)],
                  -1)
    print('ReasonerMLP x', x)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
    print('ReasonerMLP x 2', x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    print('ReasonerMLP stoch', stoch)
    latent = {'stoch': stoch, 'deter': current_state_obj['deter'], 'out': x, **stats}
    return latent