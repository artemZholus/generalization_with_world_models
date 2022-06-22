import common.state_models as models
from common.nets import DualConvEncoder, ConvDecoder, MLP

from dreamerv2.world_models import DualWMNoCond

class DualWMGibbs(DualWMNoCond):

  def __init__(self, step, config):
    super().__init__(step, config)
    shape = config.image_size + (config.img_channels,)
    RSSM = getattr(models, 'DualGibbsSeq') #config.rssm_name)
    print('RSSM', RSSM)
    self.rssm = RSSM(
      **config.rssm,
      subj_kws=config.subj_rssm, 
      obj_kws=config.obj_rssm,
      feature_sets=config.feature_sets,
      rssm_obj=config.rssm_obj,
      rssm_subj=config.rssm_subj,
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
