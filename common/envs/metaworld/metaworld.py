import gym
import os
import numpy as np


class MetaWorld:

  def __init__(self, name, action_repeat=1, size=(64, 64), 
      randomize_tasks=False, offscreen=True, cameras=None
    ):
    """
    Args: 
      cameras: "corner, corner2, corner3, topview, gripperPOV, behindGripper"
    """
    if offscreen:
      os.environ['MUJOCO_GL'] = 'egl'
    else:
      os.environ['MUJOCO_GL'] = 'glfw'
    self.offscreen = offscreen
    self.randomize_tasks = randomize_tasks
    import metaworld
    domain, task = name.split('_', 1)
    if domain == 'ml10':
      dom = metaworld.ML10()
      self._env = dom.train_classes[f'{task}-v2']
    if domain == 'ml1':
      dom = metaworld.ML1(f'{task}-v2')
    self._env = dom.train_classes[f'{task}-v2']()
    if domain == 'ml1':
      self._tasks = dom.train_tasks
      self._env.set_task(self._tasks[0])
    else:
      self._tasks = dom.train_classes

    self._action_repeat = action_repeat
    self._size = size
    if cameras is None:
      cameras = ['corner']
    elif isinstance(cameras, (str, int)):
      cameras = [cameras]
    self._cameras = []
    intmap = ['corner', 'corner2', 'corner3']
    for cam in cameras:
      if cam is None:
        cam = 1
      if isinstance(cam, int):
        cam = intmap[cam]
      assert isinstance(cam, str)
      self._cameras.append(cam)

  def load_tasks(self, path):
    import pickle
    with open(path, 'rb') as f:
      tasks = pickle.load(f)
    self._tasks = tasks
    self._env.set_task(self._tasks[0])

  def dump_tasks(self, path):
    import pickle
    with open(path, 'wb') as f:
      pickle.dump(self._tasks, f)

  @property
  def observation_space(self):
    spaces = {}
    spaces['flat_obs'] = gym.spaces.Box(
          -np.inf, np.inf, self._env.observation_space.shape, dtype=np.float32)
    for name, size in zip([
          'pos_hand', 'gripper_distance_apart', 'obj1_pos_quat', 'obj2_pos_quat', 
          'prev_step_pos_hand', 'prev_step_gripper_distance_apart', 'prev_step_obj1_pos_quat', 'prev_step_obj2_pos_quat',
          'goal_position'
        ], [3, 1, 7, 7, 3, 1, 7, 7, 3 ]):
      spaces[name] = gym.spaces.Box(
          -np.inf, np.inf, (size,), dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_space
    action = gym.spaces.Box(spec.low, spec.high, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def parse_obs(self, obs_vec):
    obs = {'flat_obs': obs_vec}
    obs['pos_hand'] = obs_vec[:3]
    obs['gripper_distance_apart'] = obs_vec[3]
    obs['obj1_pos_quat'] = obs_vec[4:11]
    obs['obj2_pos_quat'] = obs_vec[11:18]
    obs['prev_step_pos_hand'] = obs_vec[18:21]
    obs['prev_step_gripper_distance_apart'] = obs_vec[21]
    obs['prev_step_obj1_pos_quat'] = obs_vec[22:29]
    obs['prev_step_obj2_pos_quat'] = obs_vec[29:36]
    obs['goal_position'] = obs_vec[36:39]
    return obs

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    acc_reward = 0
    for _ in range(self._action_repeat):
      obs_vec, reward, done, info = self._env.step(action)
      acc_reward += reward or 0
      if done:
        break
    obs = self.parse_obs(obs_vec)

    obs['image'] = self.render()
    # info = {'discount': np.array(time_step.discount, np.float32)}
    info['discount'] = np.array(1. if not done else 0., np.float32)
    return obs, reward, done, info

  def reset(self):
    if self.randomize_tasks:
      task = self._tasks[np.random.choice(len(self._tasks))]
      self._env.set_task(task)
    position = self._env.reset()
    obs = self.parse_obs(position)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    images = []
    for cam in self._cameras:
      img = self._env.render(self.offscreen, cam, resolution=self._size)
      images.append(img)
    return np.concatenate(images, axis=2)