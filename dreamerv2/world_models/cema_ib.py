import tensorflow as tf

from common.state_models import Influencer
from common.nets import DualConvEncoder, ConvDecoder, MLP, MyMLP
from common.tfutils import Optimizer

from dreamerv2.world_models.wm import WM

class CEMA_IB(WM):

  def __init__(self, step, config):
    super().__init__(step, config)
    shape = config.image_size + (config.img_channels,)
    self.rssm = Influencer(
      **config.rssm,
      subj_kws=config.subj_rssm, 
      cond_kws=config.cond_kws, 
      obj_kws=config.obj_rssm,
      feature_sets=config.feature_sets,
    )
    self.encoder = DualConvEncoder(config.subj_encoder, config.obj_encoder, config.obj_features)
    self.sm_opt = Optimizer('scorer', lr=3e-4, wd=1e-5, eps=1e-5, clip=100)
    self.u_state_model = MyMLP(shape=[config.cond_kws['size']], layers=1, units=200)
    
    self.heads['subj_image'] = ConvDecoder(shape, **config.decoder)
    if config.obj_features == 'gt':
      self.heads['obj_gt'] = MLP((8,), **config.obj_gt_head)
    if config.obj_features == 'img':
      self.heads['obj_image'] = ConvDecoder(shape, **config.decoder)
    if config.obj_features == 'mixed':
      self.heads['obj_image'] = ConvDecoder(shape, **config.decoder)
      self.heads['obj_gt'] = MLP((8,), **config.obj_gt_head)
    self.heads['reward'] = MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = MLP([], **config.discount_head)
    
    self.modules = [
      self.rssm, self.encoder,
      *self.heads.values()
    ]

  def preprocess(self, obs):
    obs = super().preprocess(obs)
    if self.config.segmentation:
      obs_dim = obs['image'].shape[-1] // 2
      # obs['image'] is already preprocessed and contains segmentation in case of transparent env,
      # so we only need to take slices
      obs['subj_image'] = obs['image'][..., :obs_dim]
      obs['obj_image'] = obs['image'][..., obs_dim:]
    return obs

  def train(self, data, state=None, **kwargs):
    print('calling WM train')
    with tf.GradientTape() as model_tape:
      loss_inputs = self.loss_inputs(data, state)
      model_loss, losses, values = self.loss(**loss_inputs)
      model_loss += self.sm_loss(
        tf.stop_gradient(loss_inputs['embed']['obj']), 
        loss_inputs['post']['util']['stoch']
      )[1]
    with tf.GradientTape() as sm_tape:
      sm_loss, sm_value = self.sm_loss(
        tf.stop_gradient(loss_inputs['embed']['obj']), 
        tf.stop_gradient(loss_inputs['post']['util']['stoch'])
      )
    print()
    metrics = self.gather_metrics(losses, values, sm_value, **loss_inputs)
    metrics.update(self.model_opt(model_tape, model_loss, self.modules))
    metrics.update(self.sm_opt(sm_tape, sm_loss, [self.u_state_model]))
    return loss_inputs, metrics

  def sm_loss(self, obj_embed, u_samples):
    print('calling train score model')
    mi = self.rssm.uo_mut_inf(self.u_state_model, obj_embed, u_samples)
    loss = -mi
    return loss, mi

  def gather_metrics(self, losses, values, sm_value, prior, post, **kwargs):
    metrics = super().gather_metrics(losses, values)
    metrics['uo_mi'] = sm_value
    metrics['model_subj_kl'] = values['kl']['subj'].mean()
    metrics['model_obj_kl'] = values['kl']['obj'].mean()
    metrics['model_util_kl'] = values['kl']['util'].mean()
    prior_dist = self.rssm.get_dist(prior)
    post_dist = self.rssm.get_dist(post)
    metrics['prior_subj_ent'] = prior_dist['subj'].entropy().mean()
    metrics['post_subj_ent'] = post_dist['subj'].entropy().mean()
    metrics['prior_obj_ent'] = prior_dist['obj'].entropy().mean()
    metrics['post_obj_ent'] = post_dist['obj'].entropy().mean()
    metrics['post_util_ent'] = post_dist['util'].entropy().mean()
    metrics['prior_util_ent'] = prior_dist['util'].entropy().mean()
    metrics.update(self.mut_inf(post, prior))
    return metrics

  def kl_loss(self, post, prior, **kwargs):
    value, loss = super().kl_loss(post, prior)
    losses = {f'{layer}_kl': loss for layer, loss in loss.items()}
    values = {'kl': value}
    return values, losses

  def mut_inf(self, post, prior):
    metrics = {
      'mi_q_subj': self.rssm.mut_inf(post, kind='subj'),
      'mi_q_util': self.rssm.mut_inf(post, kind='util'),
      'mi_q_obj': self.rssm.mut_inf(post, kind='obj'),
      'mi_p_subj': self.rssm.mut_inf(prior, kind='subj'),
      'mi_p_util': self.rssm.mut_inf(prior, kind='util'),
      'mi_p_obj': self.rssm.mut_inf(prior, kind='obj'),
    }
    return metrics

  @tf.function
  def video_pred(self, data):
    pred =  {
      'subj': super().video_pred(data, img_key='subj_image')['openl']
    }
    if not self.config.obj_features == 'gt':
      pred['obj'] = super().video_pred(data, img_key='obj_image')['openl']
    return pred
