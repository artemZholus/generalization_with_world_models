from common.state_models import RSSM_GIBBS
from common.nets import ConvEncoder, ConvDecoder, MLP

from dreamerv2.world_models import Dreamer

class DreamerGibbs(Dreamer):

  def __init__(self, step, config):
    super().__init__(step, config)
    shape = config.image_size + (config.img_channels,)
    self.rssm = RSSM_GIBBS(**config.rssm)
    self.encoder = ConvEncoder(**config.encoder)
    self.heads['image'] = ConvDecoder(shape, **config.decoder)
    self.heads['reward'] = MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.modules = [
      self.encoder, self.rssm,
      *self.heads.values()]