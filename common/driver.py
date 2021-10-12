import numpy as np
import os
import atexit
import sys
import traceback
from functools import partial
from tempfile import NamedTemporaryFile
from filelock import FileLock

class Async(object):

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, constructor, strategy='process'):
    if strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = mp.Pipe()
    self._process = mp.Process(target=self._worker, args=(constructor, conn))
    atexit.register(self.close)
    self._process.start()
    self._observ_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._observ_space:
      self._observ_space = self.__getattr__('observation_space')
    return self._observ_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=True):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, constructor, conn):
    try:
      env = constructor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      try:
        conn.send((self._EXCEPTION, stacktrace))
      except Exception:
        print('Failed to send exception back to main process.')
    try:
      conn.close()
    except Exception:
      print('Failed to properly close connection.')


class Driver:

  def __init__(self, env_ctor, num_envs, mode='local', lock=False, 
               lockfile=None, **kwargs):
    """Driver that runs agent on a set onf envs

    Args:
        env_ctor (callable): env creation function
        num_envs (int): number of envs to create
        mode (str): 'local', 'process'. Defaults to 'local'.
    """
    self.mode = mode
    if lock:
      os.makedirs('/tmp/driver_lock', exist_ok=True)
      if lockfile is None:  
        lockfile_ = NamedTemporaryFile(dir='/tmp/driver_lock', delete=False)
        lockfile = lockfile_.name
        lockfile_.close()
    if mode == 'local':
      self._envs = [env_ctor(syncfile=lockfile, worker_id=i) for i in range(num_envs)]
    elif mode == 'process':
      self._envs = [Async(partial(env_ctor, syncfile=lockfile, worker_id=i)) 
                    for i in range(num_envs)]
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._call_kws = {}
    if self.mode == 'process':
      self._call_kws = {'blocking': False}
    self._actspaces = [env.action_space.spaces for env in self._envs]
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._dones = [True] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def reset_envs_if_done(self):
    resets = []
    for i, done in enumerate(self._dones):
      if done:
        reset_result = self._envs[i].reset(**self._call_kws)
        resets.append(reset_result)
    if self.mode == 'process':
      return [r() for r in resets]
    else:
      return resets

  def step_envs(self, actions):
    obss = []
    for i in range(len(self._envs)):
      obss.append(self._envs[i].step(actions[i], **self._call_kws))
    if self.mode == 'process':
      return [o() for o in obss]
    else:
      return obss

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      reset_obs = self.reset_envs_if_done()
      for i, done in enumerate(self._dones):
        if done:
          self._obs[i] = ob = reset_obs[i]
          act = {k: np.zeros(v.shape) for k, v in self._actspaces[i].items()}
          tran = {**ob, **act, 'reward': 0.0, 'discount': 1.0, 'done': False, 'reward_mask': 1.0}
          [callback(tran, **self._kwargs) for callback in self._on_resets]
          self._eps[i] = [tran]
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      results = self.step_envs(actions)
      for i, (act, (ob, rew, done, info)) in enumerate(zip(actions, results)):
        obs = {k: self._convert(v) for k, v in obs.items()}
        disc = info.get('discount', np.array(1 - float(done)))
        tran = {**ob, **act, 'reward': rew, 'discount': disc, 'done': done, 'reward_mask': 1.0}
        [callback(tran, **self._kwargs) for callback in self._on_steps]
        self._eps[i].append(tran)
        if done:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          [callback(ep, **self._kwargs) for callback in self._on_episodes]
      obs, _, dones = zip(*[p[:3] for p in results])
      self._obs = list(obs)
      self._dones = list(dones)
      episode += sum(dones)
      step += len(dones)

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
