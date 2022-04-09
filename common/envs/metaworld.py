from collections import defaultdict
from functools import partial
import os
import math
import threading
import pickle
from copy import deepcopy as copy

import gym
import metaworld
import numpy as np
import random
from scipy.signal import convolve2d
from filelock import FileLock
from tensorflow.python.framework.op_def_registry import sync
from tensorflow.python.types.core import Value
from .async_env import Async


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
  'drawer-close-rotated': [
    'g_drawercase_base', 'g_drawercase_col1', 'g_drawercase_col2', 
    'g_drawercase_col3', 'g_drawercase_col4', 'g_drawercase_col5',
    'g_drawer_beige', 'objGeom', 'g_drawer_col1', 'g_drawer_col2', 
    'g_drawer_col3', 'g_drawer_col4', 'g_drawer_col5', 
    'g_drawer_col6', 'g_drawer_col7', 'g_drawer_col8',  
  ],
  'drawer-open-rotated': [
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
  'drawer-open-rotated': ['goal'],
  'drawer-close-rotated': ['goal'],
  'peg-insert-side': ['goal'],
  'window-open': ['handleOpenStart', 'handleCloseStart', 'goal'],
  'window-close': ['handleOpenStart', 'handleCloseStart', 'goal'],
  'reach': ['goal'],
  'push': ['goal'],
  'pick-place': ['goal'],
  'robot_body': ['leftEndEffector', 'rightEndEffector']
}

class NullContext:
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        pass

class MetaWorld:

  def __init__(self, name, action_repeat=1, size=(64, 64), 
      randomize_env=True, randomize_tasks=False, offscreen=True, 
      cameras=None, segmentation=True, syncfile=None,
      worker_id=None, transparent=False,
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
    self.segm_size = (int(os.environ.get('SEGM_W', 128)),
                      int(os.environ.get('SEGM_H', 128)))
    self.offscreen = offscreen
    self.segmentation = segmentation
    self.randomize_env = randomize_env
    self.randomize_tasks = randomize_tasks
    self.ker = np.ones((3,3))
    self.ker /= 9.
    import metaworld
    domain, task = name.split('_', 1)
    self.task = task
    self.task_id = {}
    self.env_tasks = {}
    self.envs_cls = {}
    self.tr_envs_cls = {}
    self.tasks_generator = None
    self.transparent = transparent
    if domain == 'mt10':
      dom = metaworld.MT10()
    elif domain == 'ml1':
      task_name = f'{task}-v2'
      dom = metaworld.ML1(task_name)
      if transparent:
        dom_transparent = metaworld.ML1(task_name, transparent_sawyer=True)
    for name, env_cls in dom.train_classes.items():
      with NullContext() if syncfile is None else FileLock(syncfile):
        env = env_cls()
      all_tasks = [task for task in dom.train_tasks
                    if task.env_name == name]
      task_id = random.choice(range(len(all_tasks)))
      self.task_id[name] = task_id
      task = all_tasks[task_id]
      env.set_task(task)
      self.env_tasks[name] = (all_tasks, task_id)
      self.envs_cls[name] = env
      if transparent:
        with NullContext() if syncfile is None else FileLock(syncfile):
          tr_env_cls = dom_transparent.train_classes[name]
          tr_env = Async(partial(tr_env_cls, transparent_sawyer=True))
          self.tr_envs_cls[name] = tr_env
          tr_env.call('set_task', task)()
    np.random.seed(worker_id or 1)
    self.worker_id = worker_id
    if worker_id is None:
      self._curr_env = random.choice(list(self.envs_cls.keys()))
    else:
      env_keys = list(self.envs_cls.keys())
      self._curr_env = env_keys[worker_id % len(env_keys)]
    self._env = self.envs_cls[self._curr_env]
    self._tr_env = None
    if self.transparent:
      self._tr_env = self.tr_envs_cls[self._curr_env]
    self._tasks = self.env_tasks[self._curr_env][0]
    self.syncfile = syncfile
    print(f'sync file: {syncfile}')
    with NullContext() if syncfile is None else FileLock(syncfile):
      if syncfile is not None:
          path = f'{syncfile}.data'
          if not os.path.exists(path):
            with open(path, 'wb') as f:
              pickle.dump(self.env_tasks, f)
          else:
            with open(path, 'rb') as f:
              self.env_tasks = pickle.load(f)
            for name in self.envs_cls.keys():
              env = self.envs_cls[name]
              tasks, task_id = self.env_tasks[name]
              self.task_id[name] = task_id
              env.set_task(tasks[task_id])
              if transparent:
                self.tr_envs_cls[name].call('set_task', tasks[task_id])()
            self._tasks = self.env_tasks[self._curr_env][0]
    self.domain = domain

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
  
  def set_tasks_generator(self, gen):
    self.tasks_generator = gen

  def load_tasks(self, path):
    with open(path, 'rb') as f:
      tasks = pickle.load(f)
    self.env_tasks = tasks
    for name, env in self.envs_cls.items():
      task = self.env_tasks[name][0][self.env_tasks[name][1]]
      env.set_task(task)
      if self.transparent:
        self.tr_envs_cls[name].call('set_task', task)()

  def create_tasks(self, params):
    new_taskset = defaultdict(list)
    for name, env in self.envs_cls.items():
      for vec in params[name]:
        task = self.env_tasks[name][0][self.env_tasks[name][1]]
        new_task = pickle.loads(task.data)
        new_task['rand_vec'] = vec
        new_task = metaworld.Task(env_name=task.env_name, data=pickle.dumps(new_task))
        new_taskset[name].append(new_task)
    for name, env in self.envs_cls.items():
      self.env_tasks[name] = (new_taskset[name], 0)
    self.dump_tasks(f'{self.syncfile}_iid_eval.data')
    self.load_tasks(f'{self.syncfile}_iid_eval.data')

  def dump_tasks(self, path):
    with open(path, 'wb') as f:
      pickle.dump(self.env_tasks, f)
    
  @property
  def unwrapped(self):
    return self._env
  
  # def __getattr__(self, name):
  #   raise AttributeError(f'{type(self)} has no attr {name}!')

  def set_task_set(self, env_name, task_set):
    self.env_tasks[env_name] = (task_set, 0)
    self.task_id[env_name] = 0
    self.envs_cls[env_name].set_task(task_set[0])
    if self.transparent:
      self.tr_envs_cls[env_name].set_task(task_set[0])
    if env_name == self._curr_env:
      self._tasks = self.env_tasks[self._curr_env][0]

  def get_task_set(self, env_name):
    return self.env_tasks[env_name]

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
    obs['task_vector'] = self.unwrapped._last_rand_vec.copy()
    return obs

  def get_gt_objective(self, obs):
    handle_target_xy = obs["obj1_pos_quat"][:2] - self.unwrapped._target_pos[:2]
    handle_target_dist = np.linalg.norm(obs["obj1_pos_quat"][:2] - self.unwrapped._target_pos[:2])
    h_t_sin, h_t_cos = handle_target_xy[0]/handle_target_dist, handle_target_xy[1]/handle_target_dist
    drawer_xy = copy(self._env.sim.model.body_pos[self._env.model.body_name2id('drawer')][:2])
    handle_xy = copy(obs["obj1_pos_quat"][:2])
    task_angle = obs['task_vector'][-1] * ( math.pi/180. ) - math.pi
    task_sin = np.sin(task_angle)
    task_cos = np.cos(task_angle)
    
    return np.hstack((handle_target_dist, h_t_sin, h_t_cos,
                      drawer_xy, handle_xy,
                      task_sin, task_cos))

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    acc_reward = 0
    for _ in range(self._action_repeat):
      if self.transparent:
        tr_promise = self._tr_env.call('step', action)
      obs_vec, reward, done, info = self._env.step(action)
      if self.transparent:
        tr_obs_vec, tr_reward, tr_done, tr_info = tr_promise()
      acc_reward += reward or 0
      if done:
        break
    obs = self.parse_obs(obs_vec)
    obs['obj_gt'] = self.get_gt_objective(obs)

    # TODO: transparent here
    obs['image'] = self.render()
    if self.segmentation:
      obs['segmentation'] = self.render_segm()
    # info = {'discount': np.array(time_step.discount, np.float32)}
    info['discount'] = np.array(1. if not done else 0., np.float32)
    obs['task_name'] = self._curr_env
    return obs, reward, done, info

  def set_task_vector(self, vec):
    task_id = self.task_id[self._curr_env]
    task = copy(self._tasks[task_id])
    task_data = pickle.loads(task.data)
    task_data['rand_vec'] = vec
    task_data = pickle.dumps(task_data)
    task = metaworld.Task(env_name=task.env_name, data=task_data)
    self._env.set_task(task)
    if self.transparent:
      self._tr_env.call('set_task', task)()
    return task

  def get_task_vector(self):
    task_id = self.task_id[self._curr_env]
    return pickle.loads(self._tasks[task_id].data)['rand_vec']

  def reset(self):
    if self.randomize_env:
      self._curr_env = random.choice(list(self.envs_cls.keys()))
    elif self.worker_id is not None:
      env_keys = list(self.envs_cls.keys())
      self._curr_env = env_keys[self.worker_id % len(env_keys)]
    self._env = self.envs_cls[self._curr_env]
    if self.transparent:
      self._tr_env = self.tr_envs_cls[self._curr_env]
    self._tasks = self.env_tasks[self._curr_env][0]
    if self.randomize_tasks:
      # TODO: maybe add per-event random task sync. 
      # for now this is fine.
      if self.tasks_generator is not None:
        task_vec = next(self.tasks_generator)
        self.set_task_vector(task_vec)
      else:
        self.task_id[self._curr_env] = np.random.choice(len(self._tasks))
        task = self._tasks[self.task_id[self._curr_env]]
        self._env.set_task(task)
        if self.transparent:
          self._tr_env.call('set_task', task)()
    position = self._env.reset()
    if self.transparent:
      tr_position = self._tr_env.call('reset')()
    obs = self.parse_obs(position)
    obs['obj_gt'] = self.get_gt_objective(obs)
    # TODO: transparent here
    obs['image'] = self.render()
    if self.segmentation:
      obs['segmentation'] = self.render_segm()
    obs['task_name'] = self._curr_env
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    images = []
    tr_images = []
    for cam in self._cameras:
      if self.transparent:
        tr_promise = self._tr_env.call('render', self.offscreen, cam, resolution=self._size, **kwargs)
      img = self._env.render(self.offscreen, cam, resolution=self._size, **kwargs)
      if self.transparent:
        tr_img = tr_promise()
        tr_images.append(tr_img)
      images.append(img)
    images += tr_images
    return np.concatenate(images, axis=2)

  def render_segm(self):
    masks = []
    tr_masks = []
    for cam in self._cameras:
      if self.transparent:
        tr_promise = self._tr_env.call('render', self.offscreen, cam, resolution=self.segm_size, segmentation=True)
      raw_mask = self._env.render(self.offscreen, cam, resolution=self.segm_size, segmentation=True)
      mask = self.segm2mask(raw_mask)
      if self.transparent:
        tr_mask = tr_promise()
        tr_mask = self.segm2mask(tr_mask)
        tr_masks.append(tr_mask)
      masks.append(mask)
    masks += tr_masks
    return np.stack(masks, axis=2)

  def segm2mask(self, segm):
    res = np.zeros(self._size)
    subj_mask = np.zeros(self.segm_size, dtype=np.bool)
    obj_mask = np.zeros(self.segm_size, dtype=np.bool)
    geom_mask = segm[..., 0] == 5
    site_mask = segm[..., 0] == 6
    
    task = self.task if self.domain != 'mt10' else self._curr_env[:-len('-v2')]

    for geom in MTW_GEOMS_MAP[task]:
      geom_id = self._env.model.geom_name2id(geom)
      obj_mask |= ( (segm[..., 1] == geom_id) * geom_mask )
    for geom in MTW_GEOMS_MAP['robot_body']:
      geom_id = self._env.model.geom_name2id(geom)
      subj_mask |= ( (segm[..., 1] == geom_id) * geom_mask )
    
    for site in MTW_SITE_MAP[task]:
      site_id = self._env.model.site_name2id(site)
      obj_mask |= ( (segm[..., 1] == site_id) * site_mask )
    for site in MTW_SITE_MAP['robot_body']:
      site_id = self._env.model.site_name2id(site)
      subj_mask |= ( (segm[..., 1] == site_id) * site_mask )

    # roughly: we want to marginalize lone edges that emerged due to GPU-specific 
    # segmentation artifacts. This is done by convolving image w/ 3x3 avg. filter
    # and dropping elements <= 0.33 (those which occupy <= 1/3 of the window)
    conv_subj = convolve2d(subj_mask.astype(np.float32), self.ker, mode='same')
    subj_mask = conv_subj > 0.36
    conv_obj = convolve2d(obj_mask.astype(np.float32), self.ker, mode='same')
    obj_mask = conv_obj > 0.36
    
    subj_mask = subj_mask.reshape(self._size[0], self.segm_size[0] // self._size[0], 
                                  self._size[1], self.segm_size[1] // self._size[1]
    ).mean(axis=(1, 3)) > 0.5

    obj_mask = obj_mask.reshape(self._size[0], self.segm_size[0] // self._size[0], 
                                self._size[1], self.segm_size[1] // self._size[1]
    ).mean(axis=(1, 3)) > 0.5

    res[subj_mask] = 1
    res[obj_mask] = 2

    return res