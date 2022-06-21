import math

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from common.state_models import RSSM, ReasonerMLP, DualNoCond
from common.other import static_scan


class DualGibbs(DualNoCond):
  def __init__(
    self,
    # these are base kwargs
    stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu, std_act='softplus', min_std=0.1,
    # per layer specific kwargs
    subj_kws=None, obj_kws=None,
    feature_sets=None,
    use_task_vector=True
  ):
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
    self.obj_reasoner = ReasonerMLP(**obj_kws)
    self._use_task_vector = use_task_vector


  @tf.function
  def obs_step(self, state, action, emb, task_vec=None, sample=True):
    prior = self.bottom_up_step(state, action,
                                task_vec=task_vec,
                                sample=sample)
    print('DualGibbs prior', prior)
    post = self.top_down_step(state, emb['obj'], emb['subj'],
                              action=action, prior_state=prior,
                              sample=sample)
    print('DualGibbs post', post)
    # post = self.top_down_step(state, emb['obj'], emb['subj'],
    #                           action=action, prior_state=post,
    #                           sample=sample)
    return post, prior

  @tf.function
  def top_down_step(self, prev_state, emb_obj=None, emb_subj=None, action=None, prior_state=None, sample=True):
    prev_subj, prev_obj = prev_state['subj'], prev_state['obj']
    if prior_state is not None:
      prior_subj, prior_obj = prior_state['subj'], prior_state['obj']
    else:
      prior_subj, prior_obj = None, None
    # obj inference
    post_update_obj = emb_obj
    post_obj = self.obj_reasoner.obs_step(prev_state=prev_obj,
                                          current_state=prior_obj,
                                          post_update=post_update_obj,
                                          sample=sample)
    # subj inference
    post_update_subj = emb_subj
    post_subj, _ = self.subj_reasoner.obs_step(prev_state=prev_subj,
                                               embed=post_update_subj,
                                               prev_action=action,
                                               sample=sample
    )
    return {'subj': post_subj, 'obj': post_obj}