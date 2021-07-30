import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import pathlib
import agent
import common
import time
import math


class TrainProposal:
    def __init__(self, config, agnt, step, dataset):
      self.wm = agnt.wm
      self.ac = agnt._task_behavior
      self.reward = agnt.reward
      self.config = config
      self.dataset = dataset
      self.timed = common.Timed()
      self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def train(self, agnt):
      metrics = {}
      self.before_train()
      batch = self.propose_batch(agnt, metrics=metrics)
      with self.timed.action('train_agent'):
        _, mets = agnt.train(batch)
      mets.update(metrics)
      return _, mets
    
    def propose_batch(self, agnt, metrics):
      return next(self.dataset)

    def before_train(self):
      pass


class RawMultitask(TrainProposal):
  def __init__(self, config, agent, step, dataset):
    super().__init__(config, agent, step, dataset)
    path = pathlib.Path(config.multitask.data_path).expanduser()
    self.multitask_dataset = iter(common.Replay(path).dataset(**config.multitask.dataset))
  
  @tf.function
  def merge_batches(self, multitask_batch, task_batch, pct):
    # copy batches
    multitask_batch = tf.nest.map_structure(tf.identity, multitask_batch)
    task_batch = tf.nest.map_structure(tf.identity, task_batch)
    keys = ['image', 'action', 'reward', 'discount']
    # for k in keys:
    #   task_batch[k] = self._cast(task_batch[k])
    # calculate lengths of task and multitask parts of batch,
    # implicitly asserting that multitask_batch and task_batch are of the same length
    batch_len = len(task_batch['image'])
    task_part_len = int(math.floor(batch_len * (1-pct)))
    multitask_part_len = batch_len - task_part_len
    for k in ['action', 'reward', 'discount']:
      multitask_batch[k] = tf.cast(multitask_batch[k], tf.float32)
    multitask_batch = {
      k: tf.concat([
        task_batch[k], tf.stop_gradient(multitask_batch[k])[:int(len(multitask_batch[k]) * pct)]], 
        0) 
      for k in keys
    }
    discount = task_batch['discount'][:int(len(task_batch['discount']) * (1 - pct))]
    multitask_batch['discount'] = tf.concat(
      [discount, tf.ones_like(discount)], 0
    )
    # mask_fun = tf.zeros if self.mask_other_task_rewards else tf.ones
    mask_fun = tf.zeros
    length = multitask_batch['reward'].shape[1]
    multitask_batch['reward_mask'] = tf.concat([
      tf.ones((int(math.floor(len(multitask_batch['reward']) * (1 - pct))), length), dtype=multitask_batch['reward'].dtype),
      mask_fun((int(math.ceil(len(multitask_batch['reward']) * pct)), length), dtype=multitask_batch['reward'].dtype)
    ], 0)
    multitask_batch['reward_mask'] = tf.cast(multitask_batch['reward_mask'], tf.float32)
    return multitask_batch

  def propose_batch(self, agnt, metrics=None):
    task_batch = next(self.dataset)
    if np.random.rand() < self.config.multitask.multitask_probability:
      multitask_batch = next(self.multitask_dataset)
      pct = self.config.multitask.multitask_batch_fraction
      return self.merge_batches(multitask_batch, task_batch, pct)
    else:
      return task_batch
  

class RetrospectiveAddressing(RawMultitask):
  def __init__(self, config, agent, step, dataset):
    super().__init__(config, agent, step, dataset)
    self.addressing = common.AddressNet()
    self.encoder = self.wm.encoder
    
    if config.addressing.separate_enc_for_addr:
        self._encoder = common.ConvEncoder(config.encoder.depth, config.encoder.act, rect=config.encoder.rect)
    self.opt = common.Optimizer('addr', **config.addressing.optimizer)
    self.modules = [self.addressing]
    if not self.config.addressing.detach_cnn:
      self.modules.append(self.encoder)

  def propose_batch(self, agnt, metrics=None):
    with self.timed.action('batch'):
      batch = next(self.dataset)
      batch = self.wm.preprocess(batch)
      multitask_batches = []
      for _ in range(self.config.addressing.num_train_multitask_batches):
        multitask_batches.append(self.wm.preprocess(next(self.multitask_dataset)))
    with self.timed.action('train_addressing'):
      mets = self.train_addressing(batch, multitask_batches)
    agent_only = False #tf.constant(False)
    # addressing_probability == expert_batch_prop
    with self.timed.action('batch'):
      batch = next(self.dataset)
      addr_batch = tf.nest.map_structure(tf.identity, batch)
      addr_batch = self.wm.preprocess(addr_batch)
      addr_batch['action'] = self._cast(addr_batch['action'])
    randn = np.random.rand()
    if randn < self.config.multitask.multitask_probability:
      with self.timed.action('query'):
        batch = self.query_memory(addr_batch, batch)
      # agent_only = self.addr_agent_only #tf.constant(self.addr_agent_only)
    if metrics is not None:
      metrics.update(mets)
    return batch

  def task_reward(self, observations, actions, reduce=True):
    post, _ = self.wm.rssm.observe(observations, actions)
    feat = self.wm.rssm.get_feat(post)
    rewards = self._cast(self.reward(feat, post, actions))
    #transpose?
    return tf.reduce_sum(rewards, 1) if reduce else (rewards, feat)

  @tf.function
  def infer_address(self, task_batch, multitask_batch):
    task_embed = tf.stop_gradient(self.encoder(task_batch))
    multitask_embed = tf.stop_gradient(self.encoder(multitask_batch))
    state = self.addressing.embed(task_embed, task_batch['action'])[-1]
    memory = self.addressing.embed(multitask_embed, multitask_batch['action'])[-1]
    # 1st dim - obj; 2nd dim - dist
    logits = state @ tf.transpose(memory)
    return logits

  def query_memory(self, addr_batch, data):
    cache = []
    logits_all = []
    pct = self.config.multitask.multitask_batch_fraction
    for i in range(self.config.addressing.num_query_multitask_batches):
      multitask_batch = next(self.multitask_dataset)
      addr_mt_batch = tf.nest.map_structure(tf.identity, multitask_batch)
      addr_mt_batch = self.wm.preprocess(addr_mt_batch)
      addr_mt_batch['action'] = self._cast(addr_mt_batch['action'])
      cache.append(multitask_batch)
      logits = self.infer_address(addr_batch, addr_mt_batch)
      logits_all.append(tf.stop_gradient(logits))
    multitask_batch = self.calc_query(cache, logits_all)
    return self.merge_batches(multitask_batch, data, pct)

  @tf.function
  def calc_query(self, expert, logits, replacement=True):
    keys = ['image', 'action', 'reward', 'discount']
    logits = tf.concat(logits, -1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    if replacement:
      # log_probs should be 2d
      dist = common.OneHotDist(logits=log_probs)
      selection = tf.math.argmax(dist.sample(), -1)
    else:
      # TODO: select per-row argmax in 2d tensor without replacement
      gumbel = tfd.Gumbel(loc=tf.zeros_like(log_probs[0]), scale=1.)
      selection = tf.argsort(-log_probs[0] - gumbel.sample(), -1)
      selection = selection[:log_probs.shape[0]]
    multitask_batch = {
      k: tf.concat([c[k] for c in expert], 0)
      for k in keys
    }
    multitask_batch = {k: tf.gather(multitask_batch[k], selection)
                       for k in keys}
    return multitask_batch

  def select(self, logits, multitask_embedding, multitask_batch, soft):
    if soft:
      dist = common.OneHotDist(logits=logits)
      selection = dist.sample()
      embedding = tf.einsum('ij,jab->iab', selection, multitask_embedding)
      actions = tf.einsum('ij,jab->iab', selection, multitask_batch['action'])
      rewards = tf.einsum('ij,ja->ia', selection, multitask_batch['reward'])
    else:
      dist = tfd.Categorical(logits=logits)
      selection = dist.sample() # todo: consider multi-sample objectives
      embedding = tf.gather(multitask_embedding, selection)
      rewards = tf.gather(multitask_batch['reward'], selection)
      actions = tf.gather(multitask_batch['action'], selection)
    return dist, selection, embedding, actions, rewards

  @tf.function
  def train_addressing(self, task_batch, multitask_batches):
    task_batch = tf.nest.map_structure(tf.identity, task_batch)
    multitask_batches = tf.nest.map_structure(tf.identity, multitask_batches)
    metrics = {}
    with tf.GradientTape() as address_tape:
      task_obs = self.encoder(task_batch)
      task_batch['action'] = self._cast(task_batch['action'])
      task_actions = task_batch['action']
      if self.config.addressing.detach_cnn:
          task_obs = tf.stop_gradient(task_obs)
      multitask_batch = {
          k: tf.concat(
          [multitask_batches[i][k] for i in range(len(multitask_batches))]
          , 0) for k in multitask_batches[0].keys()
      }
      multitask_batch['action'] = self._cast(multitask_batch['action'])
      multitask_actions = multitask_batch['action']
      multitask_rewards = multitask_batch['reward']
      multitask_obs = self.encoder(multitask_batch)
      if self.config.addressing.detach_cnn:
          multitask_obs = tf.stop_gradient(multitask_obs)
      state = self.addressing.embed(task_obs, task_actions)[-1]
      memory = self.addressing.embed(multitask_obs, multitask_actions)[-1]
      if self.config.addressing.detach_task_embedding:
        state = tf.stop_gradient(state)
      elif self.config.addressing.detach_multitask_embedding:
        memory = tf.stop_gradient(memory)
      logits = state @ tf.transpose(memory)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      kind = self.config.addressing.kind
      if 'reinforce' in kind:
        dist, selection, selected_obs, selected_actions, selected_rewards = self.select(
          logits, multitask_obs, multitask_batch, soft=False
        )
        if 'baseline' in kind:
          if kind == 'reinforce_baseline_pred':
            baseline = self.task_reward(task_obs, task_actions)
          elif kind == 'reinforce_baseline_true':
            baseline = tf.reduce_sum(task_batch['reward'], 1)
        else:
          baseline = 0
        # this line fails with a very strange error :(
        # lp = dist.log_prob(selection)
        if 'pred' in kind:
          target = self.task_reward(selected_obs, selected_actions)
        else:
          target = tf.reduce_sum(selected_rewards, 1)
        rewards = target
        log_policy = tf.squeeze(tf.gather(log_probs, tf.expand_dims(selection, 1), batch_dims=1))
        advantage = target - baseline
        advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-6)
        advantage = tf.stop_gradient(advantage)
        loss = tf.cast(tf.reduce_mean(-log_policy * advantage), tf.float32)
      elif kind == 'value':
        dist, selection, selected_obs, selected_actions, selected_rewards = self.select_expert(
          logits, multitask_obs, multitask_batch, soft=True
        )
        #new_emb = tf.tile(tf.expand_dims(exp_emb, 1), [1, selection.shape[1], 1, 1])
        rewards, feat = self.task_reward(selected_obs, selected_actions, reduce=False)
        rewards = tf.stop_gradient(rewards)
        pcont = self.config.discount * tf.ones_like(rewards)
        discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
            [tf.ones_like(pcont[:, :1]), pcont[:, :-2]], 1), 1))
        values = self._value(feat).mode()
        returns = common.lambda_return(
            rewards[:, :-1], values[:, :-1], pcont[:, :-1],
            bootstrap=values[:, -1], lambda_=self.config.discount_lambda, axis=1)
        loss = -tf.reduce_mean(discount * returns)
        rewards = rewards.sum(1)

    addr_norm = self.opt(address_tape, loss, self.modules)
    diff = tf.expand_dims(log_probs, 0) - tf.expand_dims(log_probs, 1)
    bs = self.config.dataset.batch
    pair_jsd = tf.reduce_sum(diff * tf.exp(tf.expand_dims(log_probs, 0))) / (bs * (bs - 1))
    metrics['pairwise_jsd'] = pair_jsd
    metrics['address_entropy'] = dist.entropy().mean()
    metrics['address_loss'] = loss
    if kind == 'value' or 'pred' in kind:
      metrics['expert_batch_pred_reward'] = tf.reduce_mean(rewards)
    metrics['expert_batch_true_reward'] = tf.reduce_mean(tf.reduce_sum(selected_rewards, 1))
    metrics['expert_batch_expected_reward'] = tf.reduce_mean(
      tf.reduce_sum(tf.exp(log_probs) @ self._cast(multitask_rewards), 1)
    )
    metrics['address_net_norm'] = addr_norm['addr_grad_norm']
    return metrics


class DyneRetrospectiveAddressing(RetrospectiveAddressing):
  def __init__(self, config, agent, step, dataset, dyne):
    super().__init__(config, agent, step, dataset)
    self.encoder = dyne.obs_net_encoder
    self.addressing = common.DyneAddressNet(dyne)
    self.modules = [self.addressing]
