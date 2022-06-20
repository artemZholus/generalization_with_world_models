from common.state_models import RSSM_GIBBS
from common.nets import ConvEncoder, ConvDecoder, MLP

from dreamerv2.world_models.wm import WM

class DreamerGibbs(WM):

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

  def gather_metrics(self, losses, values, prior, post, **kwargs):
    metrics = super().gather_metrics(losses, values)
    metrics['model_kl'] = values['kl'].mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    metrics.update(self.mut_inf(post, prior))
    return metrics

  def kl_loss(self, post, prior, **kwargs):
    print('calling KL loss DreamerGibbs')
    value, loss = super().kl_loss(post, prior)
    value = {'kl': value}
    loss = {'kl': loss}
    print('DreamerGibbs', value)
    print('DreamerGibbs', loss)
    return value, loss

  def mut_inf(self, post, prior):
    metrics = {
      'mi_q': self.rssm.mut_inf(post),
      'mi_p': self.rssm.mut_inf(prior)
    }
    return metrics
