import tensorflow as tf

from common.state_models import Influencer
from common.nets import DualConvEncoder, ConvDecoder, MLP

from dreamerv2.world_models.wm import WM

class CEMA(WM):

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

  @tf.function
  def preprocess(self, obs):
    obs = super().preprocess(obs)
    if self.config.segmentation:
      obs_dim = obs['image'].shape[-1] // 2
      # obs['image'] is already preprocessed and contains segmentation in case of transparent env,
      # so we only need to take slices
      obs['subj_image'] = obs['image'][..., :obs_dim]
      obs['obj_image'] = obs['image'][..., obs_dim:]
    return obs

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

  def gather_metrics(self, losses, values, prior, post, **kwargs):
    metrics = super().gather_metrics(losses, values)
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

  @tf.function
  def video_pred(self, data):
    pred =  {
      'subj': super().video_pred(data, img_key='subj_image')['openl']
    }
    if not self.config.obj_features == 'gt':
      pred['obj'] = super().video_pred(data, img_key='obj_image')['openl']
    return pred
