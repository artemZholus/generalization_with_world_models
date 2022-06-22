import tensorflow as tf

from common.state_models import DualGibbs

class DualGibbsSeq(DualGibbs):
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
                                          current_state_obj=prior_obj,
                                          post_update=post_update_obj,
                                          current_state_subj=prior_subj,
                                          sample=sample)
    # subj inference
    post_update_subj = emb_subj
    post_subj, _ = self.subj_reasoner.obs_step(prev_state=prev_subj,
                                               embed=post_update_subj,
                                               prev_action=action,
                                               current_state_subj=prior_subj,
                                               current_state_obj=post_obj,
                                               sample=sample
    )
    return {'subj': post_subj, 'obj': post_obj}