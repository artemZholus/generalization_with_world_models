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


class TrainProposal:
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

def merge(d1,d2):
  if not isinstance(d1, dict):
    return tf.concat([d1, d2], 0)
  else:
    return {k: merge(d1[k], d2[k]) for k in d1.keys()}

class EvalTrainer(TrainProposal):
  def __init__(self, config, agnt, step, dataset, eval_ds):
    super().__init__(config, agnt, step, dataset)
    self.train_ac_eval = config.train_ac_eval
    self.train_wm_eval = config.train_wm_eval
    self.batch = None
    self.eval_dataset = eval_ds
    self.step = tf.Variable(0, dtype=tf.int64)
    self.eval_rate = 0.2

  def propose_batch(self, agnt, metrics):
    if np.random.uniform() > self.eval_rate:
      batch = next(self.dataset)
      do_wm_step, do_ac_step = not self.train_ac_only, not self.train_wm_only
      return batch, do_wm_step, do_ac_step, True
    else:
      batch = next(self.eval_dataset)
      do_wm_step = not self.train_ac_only and self.train_wm_eval
      do_ac_step = not self.train_wm_only and self.train_ac_eval
      return batch, do_wm_step, do_ac_step, True


class BatchPrioritizer(TrainProposal):
  def __init__(self, config, agnt, step, dataset):
    super().__init__(config, agnt, step, dataset)
    self.batch = None
    self.step = tf.Variable(0, dtype=tf.int64)
    self.target_rate = 0.1

  # @tf.function
  def update(self, batch):
    if self.batch is None:
      mask = batch['obj1_pos_quat'][:, :, :2].std(1).mean(1) > 0.01
      self.batch = tf.nest.map_structure(lambda x: x[mask], batch)
    elif self.batch['reward'].shape[0] < 50:
      mask = batch['obj1_pos_quat'][:, :, :2].std(1).mean(1) > 0.01
      new_batch = tf.nest.map_structure(lambda x: x[mask], batch)
      self.batch = merge(self.batch, new_batch)
    else:
      self.batch = tf.nest.map_structure(lambda x: x[:50], self.batch)
    # if tf.random.uniform((1,)) < self.target_rate:
    if np.random.uniform() < self.target_rate:
      tf.print(f'size on enter: {self.batch["reward"].shape[0]} and {self.step.numpy()} steps')
      obj = True
      c = tf.Variable(0, dtype=tf.int64)
      while self.batch['reward'].shape[0] < 50:
        new_batch = next(self.dataset)
        mask = new_batch['obj1_pos_quat'][:, :, :2].std(1).mean(1) > 0.01
        new_batch = tf.nest.map_structure(lambda x: x[mask], new_batch)
        self.batch = merge(self.batch, new_batch)
        c.assign_add(1)
      tf.print(f'took {c.numpy()} steps')
      self.batch = tf.nest.map_structure(lambda x: x[:50], self.batch)
      batch = self.batch
      self.batch = None
      self.step.assign(0)
    else:
      obj = True # temporarily train whole model
      self.step.assign_add(1)
    return batch

  def propose_batch(self, agnt, metrics):
    batch = next(self.dataset)
    do_wm_step, do_ac_step = not self.train_ac_only, not self.train_wm_only

    # if self.step < 5000: # start after 3k steps
    #   return batch, do_wm_step, do_ac_step, True
    batch = self.update(batch)
    return batch, do_wm_step, do_ac_step, True

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
