import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras import layers as tfkl
from tensorflow.python.keras.models import Sequential
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import pathlib
import agent
import common
import time
import math


class TrainProposal(common.Module):
    def __init__(self, config, agnt, step, dataset):
      self.wm = agnt.wm
      self.ac = agnt._task_behavior
      self.train_wm_only = config.train_wm_only
      self.train_ac_only = config.train_ac_only
      assert not (self.train_ac_only and self.train_wm_only)
      self.reward = agnt.reward
      self.config = config
      self.dataset = dataset
      self.timed = common.Timed()
      self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def train(self, agnt):
      metrics = {}
      self.before_train()
      batch, do_wm_step, do_ac_step = self.propose_batch(agnt, metrics=metrics)
      with self.timed.action('train_agent'):
        _, mets = agnt.train(batch, do_wm_step=do_wm_step, do_ac_step=do_ac_step)
      mets.update(metrics)
      return _, mets
    
    def propose_batch(self, agnt, metrics):
      return next(self.dataset), not self.train_ac_only, not self.train_wm_only

    def before_train(self):
      pass


class RawMultitask(TrainProposal):
  def __init__(self, config, agent, step, dataset, replay):
    super().__init__(config, agent, step, dataset)
    # path = pathlib.Path(config.multitask.data_path).expanduser()
    self.replay = replay
    self.multitask_dataset = iter(replay.dataset(**config.multitask.dataset))
  
  def select(self, logits, multitask_embedding, multitask_batch, soft, n=1):
    if soft:
      dist = common.OneHotDist(logits=logits)
      selection32 = dist.sample()
      selection = self._cast(selection32)
      embedding = tf.einsum('ij,jab->iab', selection, multitask_embedding)
      actions = tf.einsum('ij,jab->iab', selection, multitask_batch['action'])
      if multitask_batch['reward'].dtype == tf.float32:
        rewards = tf.einsum('ij,ja->ia', selection32, multitask_batch['reward'])
      else:
        rewards = tf.einsum('ij,ja->ia', selection, multitask_batch['reward'])
    else:
      dist = tfd.Categorical(logits=logits)
      if n == 1:
        selection = dist.sample() # todo: consider multi-sample objectives
      else:
        selection = dist.sample(n)
      embedding = tf.gather(multitask_embedding, selection)
      rewards = tf.gather(multitask_batch['reward'], selection)
      actions = tf.gather(multitask_batch['action'], selection)
    return dist, selection, embedding, actions, rewards

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
    task_part = int(math.floor(batch_len * (1-pct)))
    multitask_part = batch_len - task_part
    for k in ['action', 'reward', 'discount']:
      multitask_batch[k] = tf.cast(multitask_batch[k], tf.float32)
    multitask_batch = {
      k: tf.concat([
        task_batch[k][:task_part], tf.stop_gradient(multitask_batch[k])[:multitask_part]], 
        0) 
      for k in keys
    }
    multitask_batch['discount'] = task_batch['discount']
    # mask_fun = tf.zeros if self.mask_other_task_rewards else tf.ones
    mask_fun = tf.zeros
    length = multitask_batch['reward'].shape[1]
    multitask_batch['reward_mask'] = tf.concat([
      tf.ones((task_part, length), dtype=multitask_batch['reward'].dtype),
      mask_fun((multitask_part, length), dtype=multitask_batch['reward'].dtype)
    ], 0)
    multitask_batch['reward_mask'] = tf.cast(multitask_batch['reward_mask'], tf.float32)
    return multitask_batch

  def propose_batch(self, agnt, metrics=None):
    task_batch = next(self.dataset)
    if np.random.rand() < self.config.multitask.multitask_probability:
      multitask_batch = next(self.multitask_dataset)
      pct = self.config.multitask.multitask_batch_fraction
      return self.merge_batches(multitask_batch, task_batch, pct), True
    else:
      return task_batch, True
  
class ReturnBasedProposal(RawMultitask):
  class Temp(common.Module):
    def __init__(self):
      self.val = tf.Variable(1., dtype=tf.float32, trainable=True)
    def __call__(self, logits):
      return logits / tf.math.softplus(self.val)

  def __init__(self, config, agent, step, dataset, replay):
      super().__init__(config, agent, step, dataset, replay)
      self.temp = ReturnBasedProposal.Temp()
      self.opt = common.Optimizer('addr', **config.addressing.optimizer)
      self.modules = [self.temp]

  def propose_batch(self, agnt, metrics=None):
    with self.timed.action('batch'):
      # batch = next(self.dataset)
      # batch = self.wm.preprocess(batch)
      multitask_batches = []
      for _ in range(self.config.addressing.num_train_multitask_batches):
        multitask_batches.append(self.wm.preprocess(next(self.multitask_dataset)))
    with self.timed.action('train_addressing'):
      mets = self.train_proposal(multitask_batches)
    agent_only = False #tf.constant(False)
    # addressing_probability == expert_batch_prop
    with self.timed.action('batch'):
      batch = next(self.dataset)
    randn = np.random.rand()
    if randn < self.config.multitask.multitask_probability:
      with self.timed.action('query'):
        batch = self.query_memory(batch)
      # agent_only = self.addr_agent_only #tf.constant(self.addr_agent_only)
    if metrics is not None:
      metrics.update(mets)
    return batch, True

  def query_memory(self, data):
    cache = []
    logits_all = []
    pct = self.config.multitask.multitask_batch_fraction
    for i in range(self.config.addressing.num_query_multitask_batches):
      multitask_batch = next(self.multitask_dataset)
      addr_mt_batch = tf.nest.map_structure(tf.identity, multitask_batch)
      addr_mt_batch = self.wm.preprocess(addr_mt_batch)
      addr_mt_batch['action'] = self._cast(addr_mt_batch['action'])
      cache.append(multitask_batch)
      logits = self.infer_address(addr_mt_batch)
      logits_all.append(tf.stop_gradient(logits))
    multitask_batch = self.calc_query(cache, logits_all, n=len(data['reward']))
    return self.merge_batches(multitask_batch, data, pct)
  
  @tf.function
  def infer_address(self, multitask_batch):
    multitask_embed = self.wm.encoder(multitask_batch)
    post, _ = self.wm.rssm.observe(multitask_embed, multitask_batch['action'])
    feat = self.wm.rssm.get_feat(post)
    rewards = self.reward(feat, post, multitask_batch['action'])
    # mean for stability
    rewards = tf.stop_gradient(tf.reduce_mean(rewards, 1))
    logits = self.temp(rewards)
    return logits

  @tf.function
  def calc_query(self, expert, logits, replacement=True, n=1):
    keys = ['image', 'action', 'reward', 'discount']
    logits = tf.concat(logits, -1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    if replacement:
      # log_probs should be 2d
      dist = common.OneHotDist(logits=log_probs)
      if n == 1:
        selection = tf.math.argmax(dist.sample(), -1)
      else:
        selection = tf.math.argmax(dist.sample(n), -1)
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

  @tf.function
  def task_reward(self, observations, actions, reduce=True):
    post, _ = self.wm.rssm.observe(observations, actions)
    feat = self.wm.rssm.get_feat(post)
    rewards = self._cast(self.reward(feat, post, actions))
    #transpose?
    return tf.reduce_sum(rewards, 1) if reduce else (rewards, feat)
  
  @tf.function
  def train_proposal(self, multitask_batches):
    multitask_batches = tf.nest.map_structure(tf.identity, multitask_batches)
    metrics = {}
    with tf.GradientTape() as address_tape:
      multitask_batch = {
          k: tf.concat(
          [multitask_batches[i][k] for i in range(len(multitask_batches))]
          , 0) for k in multitask_batches[0].keys()
      }
      multitask_batch['action'] = self._cast(multitask_batch['action'])
      multitask_actions = multitask_batch['action']
      multitask_rewards = multitask_batch['reward']
      multitask_wm_obs = self.wm.encoder(multitask_batch)
      logits = self.infer_address(multitask_batch)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      kind = self.config.addressing.kind
      bs = self.config.dataset.batch
      dist, selection, selected_wm_obs, selected_actions, selected_rewards = self.select(
        logits, multitask_wm_obs, multitask_batch, soft=False, n=bs
      )
      target = self.task_reward(selected_wm_obs, selected_actions)
      log_policy = tf.gather(log_probs, selection)
      advantage = target
      advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-6)
      advantage = tf.stop_gradient(advantage)
      advantage = tf.cast(advantage, tf.float32)
      loss = tf.cast(tf.reduce_mean(-log_policy * advantage), tf.float32)
    addr_norm = self.opt(address_tape, loss, self.modules)
    diff = tf.expand_dims(log_probs, 0) - tf.expand_dims(log_probs, 1)
    
    pair_jsd = tf.reduce_sum(diff * tf.exp(tf.expand_dims(log_probs, 0))) / (bs * (bs - 1))
    metrics['pairwise_jsd'] = pair_jsd
    metrics['address_entropy'] = dist.entropy().mean()
    metrics['address_loss'] = loss
    if kind == 'value' or 'pred' in kind:
      metrics['expert_batch_pred_reward'] = tf.reduce_mean(target)
    metrics['expert_batch_true_reward'] = tf.reduce_mean(tf.reduce_sum(selected_rewards, 1))
    # metrics['expert_batch_expected_reward'] = \
    #   (tf.exp(log_probs) * multitask_rewards.sum(1)).sum()
    metrics['address_net_norm'] = addr_norm['addr_grad_norm']
    metrics['temperature'] = self.temp.val
    return metrics

class RetrospectiveAddressing(RawMultitask):
  def __init__(self, config, agent, step, dataset, replay):
    super().__init__(config, agent, step, dataset, replay)
    self.hidden = config.addressing.hidden
    self.addressing = common.AddressNet(hidden=self.hidden)
    self.encoder = self.wm.encoder
    self.latent_length = None
    
    if config.addressing.separate_enc_for_addr:
        self.encoder = common.ConvEncoder(config.encoder.depth, config.encoder.act, rect=config.encoder.rect)
    self.opt = common.Optimizer('addr', **config.addressing.optimizer)
    self.modules = [self.addressing]
    if not self.config.addressing.detach_cnn:
      self.modules.append(self.encoder)
    self.query = common.MLP([self.hidden], **self.config.addressing.query, dist_layer=False)
    self.key = common.MLP([self.hidden], **self.config.addressing.key,  dist_layer=False)
    self.modules.append(self.query)
    self.modules.append(self.key)
    #storage
    if self.config.addressing.query_full_memory:
      self.query_dataset = iter(self.replay.query_dataset(self.config.dataset.batch, self.config.dataset.length))
      self.linear_dataset = iter(self.replay.dataset(**self.config.dataset, sequential=True))
    else:
      self.linear_dataset = None
      self.query_dataset = None 
    self._updates = tf.Variable(0, tf.int64)
    self.latents = None

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
        if self.config.addressing.query_full_memory:
          if self._updates % self.config.addressing.recalc_latent_freq == 0:
            self.latents = self.get_latents()
          episodes, idxes, latents, ent, log_probs, logits = self.query_large_memory(addr_batch, batch)
          self.put_queue(episodes, idxes)
          selected_batch = next(self.query_dataset)
          for k in ['action', 'reward', 'discount']:
            selected_batch[k] = tf.cast(selected_batch[k], tf.float32)
          pct = self.config.multitask.multitask_batch_fraction
          batch = self.merge_batches(selected_batch, batch, pct)
          selection_metrics = self.query_metrics(ent, log_probs, selected_batch, latents)
          unique_pairs = len(set(zip(episodes.numpy().squeeze(), idxes.numpy().squeeze())))
          selection_metrics['unique_pairs'] = unique_pairs
          # batch, selection_metrics
        else:
          batch, selection_metrics = self.query_memory(addr_batch, batch)
      agent_only = self.config.addressing.agent_only #tf.constant(self.config.addressing.agent_only)
      if selection_metrics is not None:
        metrics.update(selection_metrics)
    if metrics is not None:
      metrics.update(mets)
    return batch, not agent_only, True

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
    query = self.query(state)
    keys = self.key(memory)
    logits = query @ tf.transpose(keys)
    return logits

  def put_queue(self, episodes, idx):
    self.replay.put(zip(episodes.numpy().squeeze(), idx.numpy().squeeze()))

  @tf.function
  def query_metrics(self, ent, log_probs, selected_batch, latents):
    metrics = {}
    self._updates.assign_add(1)
    metrics['big_select_entropy'] = ent
    cumsum = tf.math.cumsum(tf.sort(tf.exp(log_probs), axis=1, direction='DESCENDING'), axis=1, reverse=True)
    metrics['big_select_n09'] = tf.cast(cumsum < 0.9, tf.float32).sum(1).mean()
    diff = tf.expand_dims(log_probs, 0) - tf.expand_dims(log_probs, 1)
    bs = self.config.dataset.batch
    pair_jsd = tf.reduce_sum(diff * tf.exp(tf.expand_dims(log_probs, 0))) / (bs * (bs - 1))
    metrics['big_select_pair_jsd'] = pair_jsd
    metrics['big_select_reward'] = selected_batch['reward'].sum(1).mean()
    metrics['big_select_ess'] = ((log_probs.logsumexp(1) * 2) - (log_probs * 2).logsumexp(1)).exp().mean()
    return metrics

  @tf.function
  def query_large_memory(self, task_batch, data):
    metrics = {}
    task_embed = self.encoder(task_batch)
    state = self.addressing.embed(task_embed, task_batch['action'])[-1]
    query = self.query(state)
    logits = query @ tf.transpose(self.latents['latent'])
    t = self.config.addressing.temp
    log_probs = tf.nn.log_softmax(logits * t, axis=-1) # temperature is 10
    dist = common.OneHotDist(logits=tf.cast(log_probs, tf.float32))
    selection = tf.math.argmax(dist.sample(), -1)
    episodes = tf.gather(self.latents['ep_name'], selection)
    idxes = tf.gather(self.latents['idx'], selection)
    latents = tf.gather(self.latents['latent'], selection)
    return episodes, idxes, latents, dist.entropy().mean(), log_probs, logits
    # tf.py_function(self.put_queue, [episodes, idxes], [])
    # selected_batch = next(self.query_dataset)
    # for k in ['action', 'reward', 'discount']:
    #   selected_batch[k] = tf.cast(selected_batch[k], tf.float32)
    # self._updates.assign_add(1)
    # metrics['big_select_entropy'] = dist.entropy().mean()
    # cumsum = tf.math.cumsum(tf.sort(tf.exp(log_probs), axis=1, direction='DESCENDING'), axis=1, reverse=True)
    # metrics['big_select_n09'] = tf.cast(cumsum < 0.9, tf.float32).sum(1).mean()
    # diff = tf.expand_dims(log_probs, 0) - tf.expand_dims(log_probs, 1)
    # bs = self.config.dataset.batch
    # pair_jsd = tf.reduce_sum(diff * tf.exp(tf.expand_dims(log_probs, 0))) / (bs * (bs - 1))
    # metrics['big_select_pair_jsd'] = pair_jsd
    # pct = self.config.multitask.multitask_batch_fraction
    # return self.merge_batches(selected_batch, data, pct), metrics

  def get_latents(self):
    latents = []
    if self.latent_length is None:
      print('calculating multitask dataset size...')
      self.latent_length = self.replay.calculate_length()
      # self.latent_length = 10000
      print(f'multitask dataset size: {self.latent_length}')
    for i, batch in tqdm(enumerate(self.replay.dataset(**self.config.dataset, sequential=True))):
      latents.append(self.infer_latent(batch))
      if ((i + 1) * latents[-1]['latent'].shape[0] * self.config.dataset.length) >= self.latent_length:
        break
    # print('222\n\n',list(map(lambda x: x.decode('utf-8'), sum((latents[i]['ep_name'].numpy().squeeze().tolist()[(0 if i % 2== 0 else 10)::20] for i in range(len(latents))), []))))
    # latents = [self.infer_latent(batch) for batch in tqdm(self.replay.dataset(**self.config.dataset, sequential=True))]
    return {k: tf.concat([l[k] for l in latents], 0) for k in ['latent', 'ep_name', 'idx']}

  @tf.function
  def infer_latent(self, batch, is_query=False):
    print('this should print once')
    batch = self.wm.preprocess(batch)
    embed = self.encoder(batch)
    actions = self._cast(batch['action'])
    state = self.addressing.embed(embed, actions)[-1]
    if is_query:
      state = self.query(state)
    else:
      state = self.key(state)
    return {'latent': state, 'ep_name': batch['ep_name'], 'idx': batch['idx']}

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
    multitask_batch, unique_pairs = self.calc_query(cache, logits_all)
    metrics = {'unique_pairs': unique_pairs}
    return self.merge_batches(multitask_batch, data, pct), metrics

  @tf.function
  def calc_query(self, expert, logits, replacement=True):
    keys = ['image', 'action', 'reward', 'discount']
    logits = tf.concat(logits, -1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    if replacement:
      # log_probs should be 2d
      dist = common.OneHotDist(logits=log_probs)
      selection = tf.math.argmax(dist.sample(), -1)
      uni, _ = tf.unique(selection)
      uni = tf.shape(uni)[0]
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
    return multitask_batch, uni

  @tf.function
  def train_addressing(self, task_batch, multitask_batches):
    task_batch = tf.nest.map_structure(tf.identity, task_batch)
    multitask_batches = tf.nest.map_structure(tf.identity, multitask_batches)
    metrics = {}
    with tf.GradientTape() as address_tape:
      task_obs = self.encoder(task_batch)
      if self.config.addressing.separate_enc_for_addr:
        task_wm_obs = self.wm.encoder(task_batch)
      else:
        task_wm_obs = task_obs
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
      if self.config.addressing.separate_enc_for_addr:
        multitask_wm_obs = self.wm.encoder(multitask_batch)
      else:
        multitask_wm_obs = multitask_obs
      if self.config.addressing.detach_cnn:
          multitask_obs = tf.stop_gradient(multitask_obs)

      state = self.addressing.embed(task_obs, task_actions)[-1]
      memory = self.addressing.embed(multitask_obs, multitask_actions)[-1]
      if self.config.addressing.detach_task_embedding:
        state = tf.stop_gradient(state)
      elif self.config.addressing.detach_multitask_embedding:
        memory = tf.stop_gradient(memory)

      query = self.query(state)
      keys = self.key(memory)
      logits = query @ tf.transpose(keys)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      kind = self.config.addressing.kind
      if 'reinforce' in kind:
        dist, selection, selected_wm_obs, selected_actions, selected_rewards = self.select(
          logits, multitask_wm_obs, multitask_batch, soft=False
        )
        if 'baseline' in kind:
          if kind == 'reinforce_baseline_pred':
            baseline = self.task_reward(task_wm_obs, task_actions)
          elif kind == 'reinforce_baseline_true':
            baseline = tf.reduce_sum(task_batch['reward'], 1)
        else:
          baseline = 0
        # this line fails with a very strange error :(
        # lp = dist.log_prob(selection)
        if 'pred' in kind:
          target = self.task_reward(selected_wm_obs, selected_actions)
        else:
          target = tf.reduce_sum(selected_rewards, 1)
        rewards = target
        log_policy = tf.squeeze(tf.gather(log_probs, tf.expand_dims(selection, 1), batch_dims=1))
        advantage = target - baseline
        advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-6)
        advantage = tf.stop_gradient(advantage)
        loss = tf.cast(tf.reduce_mean(-log_policy * advantage), tf.float32)
      elif kind == 'value':
        dist, selection, selected_wm_obs, selected_actions, selected_rewards = self.select(
          logits, multitask_wm_obs, multitask_batch, soft=True
        )
        #new_emb = tf.tile(tf.expand_dims(exp_emb, 1), [1, selection.shape[1], 1, 1])
        rewards, feat = self.task_reward(selected_wm_obs, selected_actions, reduce=False)
        rewards = tf.stop_gradient(rewards)
        rewards = tf.cast(rewards, tf.float32)
        pcont = self.config.discount * tf.ones_like(rewards)
        discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
            [tf.ones_like(pcont[:, :1]), pcont[:, :-2]], 1), 1))
        values = self.ac._target_critic(feat).mode()
        returns = common.lambda_return(
            rewards[:, :-1], values[:, :-1], pcont[:, :-1],
            bootstrap=values[:, -1], lambda_=self.config.discount_lambda, axis=1)
        loss = -tf.reduce_mean(discount * returns)
        rewards = rewards.sum(1)

    addr_norm = self.opt(address_tape, loss, self.modules)
    diff = tf.expand_dims(log_probs, 0) - tf.expand_dims(log_probs, 1)
    bs = self.config.dataset.batch
    pair_jsd = tf.reduce_sum(diff * tf.exp(tf.expand_dims(log_probs, 0))) / (bs * (bs - 1))
    qs = self.query(state)
    ks = self.key(state)
    qm = self.query(memory)
    km = self.key(memory)
    metrics['query_diff'] = ((qs - ks) ** 2).sum(1).mean()
    metrics['key_diff'] = ((qm - km) ** 2).sum(1).mean()
    metrics['memory_var'] = ((tf.expand_dims(km.mean(0), 0) - km) ** 2).sum(1).mean()
    metrics['state_var'] = ((tf.expand_dims(qs.mean(0), 0) - qs) ** 2).sum(1).mean()
    metrics['selected_advantage'] = (selected_rewards - task_batch['reward']).sum(1).mean()
    metrics['pairwise_jsd'] = pair_jsd
    metrics['ess'] = ((log_probs.logsumexp(1) * 2) - (log_probs * 2).logsumexp(1)).exp().mean()
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
  def __init__(self, config, agent, step, dataset, replay, dyne):
    super().__init__(config, agent, step, dataset, replay)
    self.encoder = dyne.obs_net_encoder
    self.addressing = common.DyneAddressNet(dyne)
    self.modules = [self.addressing]
    if not self.config.addressing.detach_cnn:
      self.modules.append(self.encoder)
