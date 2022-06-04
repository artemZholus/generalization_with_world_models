import math

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from common.nets import ConditionModel
from common.state_models import RSSM, ReasonerMLP
from common.other import static_scan


class Influencer(RSSM):
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
    self.condition_model = ConditionModel(**cond_kws)
    self.obj_reasoner = ReasonerMLP(**obj_kws)

  @tf.function
  def uo_mut_inf(self, scorer, embs, u_samples, loo=False):
    NUM_SAMPLES = 1
    u_dists = scorer(embs)
    curr_prob = u_dists.log_prob(u_samples)
    mu, sigma = u_dists.mean(), u_dists.stddev()
    mu = tf.expand_dims(mu, 2)
    sigma = tf.expand_dims(sigma, 2)
    expand_dist = self.condition_model.get_dist({'mean': mu, 'std': sigma})
    u_samples = tf.expand_dims(u_samples, 1)
    prob = expand_dist.log_prob(u_samples)
    marginal_prob = prob.logsumexp(1) - math.log(prob.shape[2])
    if loo:
      marginal_prob = tf.stop_gradient(marginal_prob)
    mi = (curr_prob - marginal_prob).mean()
    return mi

  @tf.function
  def top_down_step(self, prev_state, emb_obj=None, emb_subj=None, action=None, task_vec=None, current_state=None, subj=False, sample=True):
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
    # subj inference
    post_update_subj = emb_subj
    post_subj, _ = self.subj_reasoner.obs_step(prev_state=prev_subj,
                                               embed=post_update_subj,
                                               prev_action=action,
                                               sample=sample
    )
    # util inference
    post_update_util = tf.concat(
      [self.obj_reasoner.get_feat(post_obj),
       self.subj_reasoner.get_feat(post_subj)], -1)
    post_util = self.condition_model.obs_step(post_update_util, sample=sample)

    return {'subj': post_subj, 'obj': post_obj, 'util': post_util}

  def mut_inf(self, sample, kind='obj'):
    NUM_SAMPLES = 1
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
    else:
      prior_update_util = prior_subj_feat
    prior_util = self.condition_model.img_step(prior_update_util, sample=sample)
    prior_update_obj = self.condition_model.get_feat(prior_util)
    prior_obj = self.obj_reasoner.img_step(prev_state=prev_obj,
                                           prior_update=prior_update_obj,
                                           sample=sample)
    return {'subj': prior_subj, 'obj': prior_obj, 'util': prior_util}

  @tf.function
  def img_step(self, state, action, task_vec=None, sample=True):
    return self.bottom_up_step(state, action, task_vec=task_vec, sample=sample)

  def get_feat(self, state, key=None, task_vec=None, obj_gt=None):
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
          elif feat == 'obj_gt':
            feat_vec = self._cast(obj_gt)
          features.append(feat_vec)
        return tf.concat(features, -1)
    subj_feat = self.subj_reasoner.get_feat(state['subj'])
    obj_feat = self.obj_reasoner.get_feat(state['obj'])
    util_feat = self.condition_model.get_feat(state['util'])
    return tf.concat([subj_feat, util_feat, obj_feat], -1)

  def get_dist(self, state):
    subj_dist = self.subj_reasoner.get_dist(state['subj'])
    obj_dist = self.obj_reasoner.get_dist(state['obj'])
    util = self.condition_model.get_dist(state['util'])
    return {'subj': subj_dist, 'obj': obj_dist, 'util': util}

  @tf.function
  def obs_step(self, state, action, emb, task_vec=None, full=True, sample=True):
    prior = self.bottom_up_step(state, action,
                                task_vec=task_vec,
                                sample=sample)
    post = self.top_down_step(state, emb['obj'], emb['subj'],
                              action=action, task_vec=task_vec, subj=full,
                              current_state=prior, sample=sample)
    return post, prior

  def initial(self, batch_size):
    return {
      'obj': self.obj_reasoner.initial(batch_size),
      'subj': self.subj_reasoner.initial(batch_size),
      'util': self.condition_model.initial(batch_size)
    }

  @tf.function
  def observe(self, emb, actions, state=None, full=True, task_vector=None):
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
    post, prior = static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs, full=full),
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
    prior = static_scan(lambda prev, inp: self.bottom_up_step(prev, *inp), tpl, state)
    prior = tf.nest.map_structure(swap, prior)
    return prior

  def kl_loss(self, post, prior, full=True, **kwargs):
    #if full:
    subj_loss, subj_value = self.subj_reasoner.kl_loss(post['subj'], prior['subj'], **kwargs.get('subj', {}))
    # obj KL
    obj_loss, obj_value = self.obj_reasoner.kl_loss(post['obj'], prior['obj'], **kwargs.get('obj', {}))
    # util KL
    util_loss, util_value = self.condition_model.kl_loss(post['util'], prior['util'], **kwargs.get('util', {}))
    # losses and values
    loss = {'subj': subj_loss, 'obj': obj_loss, 'util': util_loss}
    value = {'subj': subj_value, 'obj': obj_value, 'util': util_value}
    return loss, value

  def deter_kl_loss(self, post, prior, **kwargs):
    # subj KL
    subj_loss, subj_value = self.subj_reasoner.kl_loss(post['subj'], prior['subj'], **kwargs.get('subj', {}))
    # obj KL
    obj_loss, obj_value = self.obj_reasoner.kl_loss(post['obj'], prior['obj'], **kwargs.get('obj', {}))
    # obj deter KL
    deter_kl = ((
      post['obj']['curr_state_post']['deter'] -
      prior['obj']['curr_state_prio']['deter']
    ) ** 2).sum(-1).mean()
    deter_kl = tf.cast(deter_kl, tf.float32)
    # util KL
    util_loss, util_value = self.condition_model.kl_loss(post['util'], prior['util'], **kwargs.get('util', {}))
    # losses and values
    loss = {'subj': subj_loss, 'obj': obj_loss, 'obj_deter': deter_kl, 'util': util_loss}
    value = {'subj': subj_value, 'obj': obj_value, 'obj_deter': deter_kl, 'util': util_value}
    return loss, value