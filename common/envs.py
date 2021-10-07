import os
import threading

import gym
import numpy as np
from tensorflow.python.types.core import Value


MTW_GEOMS_MAP = {
  'button-press': [
    'g_buttonbox_base_stopbot', 'g_buttonbox_base_stopbuttonrim', 'g_buttonbox_base_stoptop',
    'g_buttonbox_base_col1', 'g_buttonbox_base_col2', 'g_buttonbox_base_col3', 'g_buttonbox_base_col4',
    'g_button_stopbutton', 'g_button_stopbuttonrod', 'g_button_cylinder1',
    'g_button_cylinder2', 'g_button_cylinder2', 'g_button_cylinder3'
  ],
  'door-open': [
    'g_doorlockB_safe', 'g_doorlockB_box1', 'g_doorlockB_box2',
    'g_doorlockB_box3', 'g_doorlockB_box4', 'g_doorlockB_box5',
    'g_door_link1', 'handle', 'g_door_link2', 'g_door_link3',
    'g_door_link4', 'g_dl_col1', 'g_dl_col2', 'g_dl_col3', 'g_dl_col4' 
  ],
  'drawer-close': [
    'g_drawercase_base', 'g_drawercase_col1', 'g_drawercase_col2', 
    'g_drawercase_col3', 'g_drawercase_col4', 'g_drawercase_col5',
    'g_drawer_beige', 'objGeom', 'g_drawer_col1', 'g_drawer_col2', 
    'g_drawer_col3', 'g_drawer_col4', 'g_drawer_col5', 
    'g_drawer_col6', 'g_drawer_col7', 'g_drawer_col8',  
  ],
  'drawer-open': [
    'g_drawercase_base', 'g_drawercase_col1', 'g_drawercase_col2', 
    'g_drawercase_col3', 'g_drawercase_col4', 'g_drawercase_col5',
    'g_drawer_beige', 'objGeom', 'g_drawer_col1', 'g_drawer_col2', 
    'g_drawer_col3', 'g_drawer_col4', 'g_drawer_col5', 
    'g_drawer_col6', 'g_drawer_col7', 'g_drawer_col8',  
  ],
  'peg-insert-side': [
    'g_peg_block_red', 'g_peg_block_wood', 'g_peg_block_col1',
    'g_peg_block_col2', 'g_peg_block_col3', 'g_peg_block_col4', 
    'g_peg_block_col5', 'g_peg_block_col6', 'g_peg_block_col7',
    'peg',  
  ],
  'window-open': [
    'g_window_base_base', 'g_window_base_frame', 'g_window_base_col1', 
    'g_window_base_col2', 'g_window_base_col3', 'g_window_base_col4', 'g_window_base_col5',
    'g_window_a_white1', 'g_window_a_white2', 'g_window_a_white3', 'g_window_a_white4', 
    'g_window_a_white5', 'g_window_a_frame', 'g_window_a_glass', 'g_window_a_col1', 
    'g_window_a_col2', 'g_window_a_col3', 'g_window_a_col4', 'g_window_a_col5', 
    'g_window_a_col6', 'g_window_a_col7', 'g_window_a_col8', 'g_window_a_col9', 
    'g_window_a_col10', 'g_window_a_col11', 'g_window_b_red', 'g_window_b_glass',
    'g_window_b_col1', 'g_window_b_col2', 'g_window_b_col3', 'g_window_b_col4', 
    'g_window_b_col5', 'g_window_b_col6', 
  ],
  'window-close': [
    'g_window_base_base', 'g_window_base_frame', 'g_window_base_col1', 
    'g_window_base_col2', 'g_window_base_col3', 'g_window_base_col4', 'g_window_base_col5',
    'g_window_a_white1', 'g_window_a_white2', 'g_window_a_white3', 'g_window_a_white4', 
    'g_window_a_white5', 'g_window_a_frame', 'g_window_a_glass', 'g_window_a_col1', 
    'g_window_a_col2', 'g_window_a_col3', 'g_window_a_col4', 'g_window_a_col5', 
    'g_window_a_col6', 'g_window_a_col7', 'g_window_a_col8', 'g_window_a_col9', 
    'g_window_a_col10', 'g_window_a_col11', 'g_window_b_red', 'g_window_b_glass',
    'g_window_b_col1', 'g_window_b_col2', 'g_window_b_col3', 'g_window_b_col4', 
    'g_window_b_col5', 'g_window_b_col6', 
  ],
  'reach': ['objGeom'],
  'pick-place': ['objGeom'],
  'push': ['objGeom'],
  'robot_body': [
    'g_torso', 'g_pedestal_mesh', 'g_right_arm_base_link_mesh', 
    'g_right_l0', 'g_head', 'g_right_l1', 'g_right_l2', 'g_right_l3', 'g_right_l4', 
    'g_right_l5', 'g_right_l6', 'g_right_hand_mesh', 'g_right_hand_cylinder', 'rail', 
    'rightpad_geom', 'leftpad_geom'
  ]
}

MTW_SITE_MAP = {
  'button-press': ['buttonStart'],
  'door-open': ['goal'],
  'drawer-open': ['goal'],
  'drawer-close': ['goal'],
  'peg-insert-side': ['goal'],
  'window-open': ['handleOpenStart', 'handleCloseStart', 'goal'],
  'window-close': ['handleOpenStart', 'handleCloseStart', 'goal'],
  'reach': ['goal'],
  'push': ['goal'],
  'pick-place': ['goal'],
  'robot_body': ['leftEndEffector', 'rightEndEffector']
}
class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)

class MetaWorld:

  def __init__(self, name, action_repeat=1, size=(64, 64), 
      randomize_tasks=False, offscreen=True, cameras=None, segmentation=True
    ):
    """
    Args: 
      cameras: "corner, corner2, corner3, topview, gripperPOV, behindGripper"
    """
    if segmentation and not offscreen:
      raise ValueError('Segmentation is supported only for offscreen.')
    if offscreen:
      os.environ['MUJOCO_GL'] = 'egl'
    else:
      os.environ['MUJOCO_GL'] = 'glfw'
    self.offscreen = offscreen
    self.segmentation = segmentation
    self.randomize_tasks = randomize_tasks
    import metaworld
    domain, task = name.split('_', 1)
    self.task = task
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
    if self.segmentation:
      obs['segmentation'] = self.render_segm()
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
    if self.segmentation:
      obs['segmentation'] = self.render_segm()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    images = []
    for cam in self._cameras:
      img = self._env.render(self.offscreen, cam, resolution=self._size, **kwargs)
      images.append(img)
    return np.concatenate(images, axis=2)

  def render_segm(self):
    masks = []
    for cam in self._cameras:
      raw_mask = self._env.render(self.offscreen, cam, resolution=self._size, segmentation=True)
      mask = self.segm2mask(raw_mask)
      masks.append(mask)
    return np.stack(masks, axis=2)


  def segm2mask(self, segm):
    res = np.zeros_like(segm[...,0])
    subj_mask = np.zeros_like(res, dtype=np.bool)
    obj_mask = np.zeros_like(res, dtype=np.bool)
    geom_mask = segm[..., 0] == 5
    site_mask = segm[..., 0] == 6

    for geom in MTW_GEOMS_MAP[self.task]:
      geom_id = self._env.model.geom_name2id(geom)
      obj_mask += ( (segm[..., 1] == geom_id) * geom_mask )
    for geom in MTW_GEOMS_MAP['robot_body']:
      geom_id = self._env.model.geom_name2id(geom)
      subj_mask += ( (segm[..., 1] == geom_id) * geom_mask )
    
    for site in MTW_SITE_MAP[self.task]:
      site_id = self._env.model.site_name2id(site)
      obj_mask += ( (segm[..., 1] == site_id) * site_mask )
    for site in MTW_SITE_MAP['robot_body']:
      site_id = self._env.model.site_name2id(site)
      subj_mask += ( (segm[..., 1] == site_id) * site_mask )
    
    res[subj_mask] = 1
    res[obj_mask] = 2

    return res
    

class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky_actions else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
    return gym.spaces.Dict({
        'image': self._env.observation_space,
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    })

  @property
  def action_space(self):
    return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs

  def step(self, action):
    action = action['action']
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)


class Dummy:

  def __init__(self):
    pass

  @property
  def observation_space(self):
    image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
    return gym.spaces.Dict({'image': image})

  @property
  def action_space(self):
    action = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    obs = {'image': np.zeros((64, 64, 3))}
    reward = 0.0
    done = False
    info = {}
    return obs, reward, done, info

  def reset(self):
    obs = {'image': np.zeros((64, 64, 3))}
    return obs


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.action_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert isinstance(env.action_space[key], gym.spaces.Discrete)
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env, key='reward'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class ResetObs:

  def __init__(self, env, key='reset'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(0, 1, (), dtype=np.bool)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reset'] = np.array(False, np.bool)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reset'] = np.array(True, np.bool)
    return obs
