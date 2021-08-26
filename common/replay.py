import datetime
import io
import pathlib
import os
import math
import time
from collections import deque
import uuid

import numpy as np
import tensorflow as tf


class Replay:

  def __init__(self, directory, limit=None, rescan=1, cache=None, load=False):
    directory.mkdir(parents=True, exist_ok=True)
    self._directory = directory
    self._limit = limit
    self._rescan = rescan
    self._cache = cache
    self._step = sum(int(
        str(n).split('-')[-1][:-4]) - 1 for n in directory.glob('*.npz'))
    if load:
      print('Loading episodes...')
      self._episodes, self._total = load_episodes(directory, limit, return_total=True)
      print('episodes loaded!')
    else:
      self._episodes = {}
    self.queries = deque()

  @property
  def total_steps(self):
    return self._step

  @property
  def num_episodes(self):
    return len(self._episodes)

  @property
  def num_transitions(self):
    return sum(int(
        str(n).split('-')[-1][:-4]) - 1 for n in self._directory.glob('*.npz'))

  def add(self, episode):
    length = self._length(episode)
    self._step += length
    if self._limit:
      total = 0
      for key, ep in reversed(sorted(
          self._episodes.items(), key=lambda x: x[0])):
        if total <= self._limit - length:
          total += self._length(ep)
        else:
          del self._episodes[key]
    filename = save_episodes(self._directory, [episode])[0]
    # self._episodes[str(filename)] = episode
    return filename

  def put(self, queries):
    for (ep_name, idx) in queries:
      self.queries.append((ep_name, idx))

  def dataset(self, batch, length, oversample_ends, sequential=False):
    if len(self._episodes) == 0:
      self._episodes = load_episodes(self._directory, limit=10)
    example = self._episodes[next(iter(self._episodes.keys()))]
    if sequential:
      example['ep_name'] = tf.constant(['111'])
      example['idx'] = tf.convert_to_tensor([111])
    types = {k: v.dtype for k, v in example.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
    if not sequential:
      generator = lambda: sample_episodes(
          self._directory, length, 
          oversample_ends, rescan=self._rescan, cache=self._cache
      )
    else:
      generator = lambda: iterate_episodes(
        episodes=self._episodes, directory=self._directory, length=length
      )
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(10)
    return dataset

  def query_dataset(self, batch, length):
    example = self._episodes[next(iter(self._episodes.keys()))]
    example['ep_name'] = tf.constant(['111'])
    example['idx'] = tf.convert_to_tensor([111])
    types = {k: v.dtype for k, v in example.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
    generator = lambda: query_episodes(self._episodes, self._directory, self.queries, length)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(batch, drop_remainder=True)
    # dataset = dataset.prefetch(10)
    return dataset

  def _length(self, episode):
    return len(episode['reward']) - 1


def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  filenames = []
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward']) - 1
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    filenames.append(filename)
  return filenames


def query_episodes(episodes, directory, queue, length):
  while True:
    obj = None
    c = 0
    while obj is None:
      try:
        obj = queue.popleft()
      except:
        print(f'Tried {c} times')
        c += 1
        # time.sleep(0.01)
        pass
    ep_name, idx = obj
    ep_name = ep_name.decode("utf-8") 
    try:
      episode = episodes[ep_name]
    except KeyError:
      with (directory / ep_name).open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        episodes[ep_name] = episode
    chunk = {k: v[idx: idx + length] for k, v in episode.items()}
    chunk['ep_name'] = tf.constant([ep_name])
    chunk['idx'] = tf.convert_to_tensor([idx])
    yield chunk


def sample_episodes(directory=None, length=None, balance=False, rescan=100, cache=None, seed=0):
  random = np.random.RandomState(seed)
  episodes = {}
  while True:
    if cache:
      _, episode = next(iter(load_episodes_lazy(directory, limit=10)))
      ep_len = len(episode['reward']) - 1
      episodes = {}
      for key, val in load_episodes(directory, limit=int(cache or 0) * ep_len, random=True).items():
        episodes[key] = val
    else:
      prev = len(episodes)
      episodes = load_episodes(directory, current_episodes=episodes)
      print(f'loaded {len(episodes) - prev} novel episodes')
    for _ in range(rescan):
      episode = random.choice(list(episodes.values()))
      if length:
        total = len(next(iter(episode.values())))
        available = total - length
        if available < 1:
          print(f'Skipped short episode of length {total}.')
          continue
        if balance:
          index = min(random.randint(0, total), available)
        else:
          index = int(random.randint(0, available + 1))
        episode = {k: v[index: index + length] for k, v in episode.items()}
      yield episode

def iterate_episodes(episodes=None, directory=None, length=None):
  while True:
    if episodes is None:
      iterator = load_episodes_lazy(directory)
    else:
      prev = len(episodes)
      episodes = load_episodes(directory, current_episodes=episodes)
      print(f'loaded {len(episodes) - prev} novel episodes')
      iterator = episodes.items()
    for name, episode in iterator:
      total = len(next(iter(episode.values())))
      if length:
        for step in range(int(math.floor(total / length))):
          index = step * length
          chunk = {k: v[index: index + length] for k, v in episode.items()}
          chunk['ep_name'] = tf.constant([name])
          chunk['idx'] = tf.convert_to_tensor([step * length])
          yield chunk
      else:
        yield episode

def load_episodes_lazy(directory, limit=None, random=False):
  directory = pathlib.Path(directory).expanduser()
  total = 0
  if random:
    paths = list(directory.glob('*.npz'))
    paths = [paths[i] for i in np.random.permutation(len(paths))]
  else:
    paths = reversed(sorted(directory.glob('*.npz')))
  for filename in paths:
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    yield str(filename), episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break

def load_episodes(directory, limit=None, random=False, current_episodes=None, return_total=False):
  current_episodes = current_episodes or {}
  directory = pathlib.Path(directory).expanduser()
  episodes = {}
  total = 0
  ret_total = 0
  if random:
    paths = list(directory.glob('*.npz'))
    paths = [paths[i] for i in np.random.permutation(len(paths))]
  else:
    paths = reversed(sorted(directory.glob('*.npz')))
  for filename in paths:
    filename_inner = filename.parts[-1]
    if str(filename_inner) in current_episodes: 
      episodes[str(filename_inner)] = current_episodes[str(filename_inner)]
      total += len(episodes[str(filename_inner)]['reward']) - 1
      continue
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    episodes[str(filename)] = episode
    total += len(episode['reward']) - 1
    ret_total += len(episode['reward'])
    if limit and total >= limit:
      break
  if return_total:
    return episodes, total
  else:
    return episodes
