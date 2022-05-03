import os

import gym
import numpy as np
from scipy.signal import convolve2d

from causal_world.envs import CausalWorld as CausalWorldEnv
from causal_world.task_generators import generate_task
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper


OBJ_IMG_OBJ_IDS = {
  'reaching': [],
  'pushing': [0, 1],
  'picking': [0, 1],
  'pick_and_place': [5],
  'stacking2': [4, 5],
  # 'stacked_blocks': [],
  'towers': [4, 5, 6, 7, 8],
  # 'creative_stacked_blocks': []
}
# GOAL_IDS = {
#   'reaching': [4, 5, 6],
#   'pushing': [5],
#   'picking': [5],
#   'pick_and_place': [6],
#   'stacking2': [6, 7],
#   'stacked_blocks': [],
#   'towers': [9, 10, 11, 12, 13],
#   'creative_stacked_blocks': []
# }
FULL_IMG_OBJ_IDS = {
  'reaching': [],
  'pushing': [4, 5],
  'picking': [4, 5],
  'pick_and_place': [5],
  'stacking2': [4, 5],
  # 'stacked_blocks': [],
  'towers': [4, 5, 6, 7, 8],
  # 'creative_stacked_blocks': []
}
FULL_IMG_ROBOT_IDS = {
  'reaching': [1],
  'pushing': [1],
  'picking': [1],
  'pick_and_place': [1],
  'stacking2': [1],
  # 'stacked_blocks': [],
  'towers': [1],
  # 'creative_stacked_blocks': []
}

class NullContext:
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        pass

class CausalWorld:

  def __init__(self, task_family, variables_space='space_a_b', action_repeat=1, size=(64, 64), 
               skip_frame=10, randomize_env=False, randomize_tasks=False, randomize_envs=True,
               offscreen=True,  worker_id=None, syncfile=None, observation_mode='structured'):
    if offscreen:
      os.environ['MUJOCO_GL'] = 'egl'
    else:
      os.environ['MUJOCO_GL'] = 'glfw'
    self.task_family = task_family
    self._action_repeat = action_repeat
    self.randomize_env = randomize_env
    self._worker_id = worker_id or 42
    self.observation_mode = observation_mode
    task = generate_task(task_generator_id=task_family, 
                         variables_space=variables_space)
    self._env = CausalWorldEnv(task, seed=self._worker_id, enable_visualization=False, 
                                  normalize_observations=False,
                                  normalize_actions=True,
                                  initialize_all_clients=False,
                                  skip_frame=skip_frame,
                                  camera_indicies=[0, 1],
                                  action_mode='end_effector_positions',
                                  observation_mode=observation_mode)
    if randomize_envs:
      goal_intervention = self._env.sample_new_goal()
      self._env.set_starting_state(goal_intervention)
    if randomize_tasks:
      self._env = CurriculumWrapper(self._env,
                            intervention_actors=[GoalInterventionActorPolicy()],
                            actives=[(0, 1000000000, 1, 0)])
    self._size=size
    self._yaws=[0, 120, 240]
    self._pitches=[-60, -60, -60]
    self._distances=[0.6, 0.6, 0.6]
    self._base_positions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    self._image_content = 'full'
    self._env.set_render_params(size, 
                                self._yaws,
                                self._pitches,
                                self._distances,
                                self._base_positions,
                                self._image_content)
    self.ker = np.ones((3,3))
    self.ker /= 9.
    
  @property
  def unwrapped(self):
    return self._env

  @property
  def observation_space(self):
    spaces = {}
    if self.observation_mode == 'pixel':
      spec = self._env.observation_space
      spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (9,), dtype=np.uint8)
      spaces['goal'] = gym.spaces.Box(spec.low[3:, ...], spec.high[3:, ...], dtype=np.uint8)
    else:
      spaces['flat_obs'] = gym.spaces.Box(
            -np.inf, np.inf, self._env.observation_space.shape, dtype=np.float32)
      # TODO:
      # do something about decoding structured observations
      spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (6,), dtype=np.uint8)
      spaces['segmentation'] = gym.spaces.Box(0, 255, self._size + (2,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  # checked
  @property
  def action_space(self):
    spec = self._env.action_space
    action = gym.spaces.Box(spec.low, spec.high, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  # TODO: modify this function for CausalWorld
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

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    acc_reward = 0
    for _ in range(self._action_repeat):
      obs_vec, reward, done, info = self._env.step(action)
      acc_reward += reward or 0
      if done:
        break
    if self.observation_mode == 'pixel':
      obs = {'image': obs_vec,
             'robot': obs_vec[:3, ...], 
             'goal': obs_vec[3:, ...]}
    else:
      obs, segm = self._env.render_with_masks()
      full_mask = self.convert_segm(segm, 'subj')
      obs = {'flat_obs': obs_vec,
             'image': np.concatenate(obs, axis=2),
             'segmentation': full_mask}

    info['discount'] = np.array(1. if not done else 0., np.float32)
    return obs, reward, done, info

  def reset(self):
    obs_vec = self._env.reset()
    
    # TODO: transparent here
    if self.observation_mode == 'pixel':
      obs = {'flat_obs': obs_vec, 'obs': obs_vec[:3, ...], 'goal': obs_vec[3:, ...]}
    else:
      obs, segm = self._env.render_with_masks()
      full_mask = self.convert_segm(segm, 'subj')
      obs = {'flat_obs': obs_vec,
             'image': np.concatenate(obs, axis=2),
             'segmentation': full_mask}
    return obs

  def render(self):
    full_obs, obj_obs, full_segm = self._env.render_with_masks()
    full_mask = self.convert_segm(full_segm, 'subj')
    obj_obs = np.stack(obj_obs, axis=0)
    obj_obs[(obj_obs == 255).all(axis=3), :] = 0
    return np.concatenate(full_obs, axis=2), np.concatenate(obj_obs, axis=2), full_mask

  def convert_segm(self, segm, kind='subj'):
    obj_img_obj_mask_ids = OBJ_IMG_OBJ_IDS[self.task_family]
    full_img_obj_mask_ids = FULL_IMG_OBJ_IDS[self.task_family]
    full_img_robot_mask_ids = FULL_IMG_ROBOT_IDS[self.task_family]
    if kind == 'obj':
      obj_ids = obj_img_obj_mask_ids
      robot_ids = None
    elif kind == 'subj':
      obj_ids = full_img_obj_mask_ids
      robot_ids = full_img_robot_mask_ids
    masks = [self.segm2mask(segm, obj_ids, robot_ids) for segm in segm]
    return np.stack(masks, axis=2)

  def segm2mask(self, segm, obj_ids, robot_ids):
    res = np.zeros(self._size)
    subj_mask = np.zeros(self._size, dtype=np.bool)
    obj_mask = np.zeros(self._size, dtype=np.bool)

    obj_ids = obj_ids or []
    robot_ids = robot_ids or []
    for obj_id in obj_ids:
      obj_mask |= (segm == obj_id)
    for subj_id in robot_ids:
      subj_mask |= (segm == subj_id)

    # roughly: we want to marginalize lone edges that emerged due to GPU-specific 
    # segmentation artifacts. This is done by convolving image w/ 3x3 avg. filter
    # and dropping elements <= 0.33 (those which occupy <= 1/3 of the window)
    conv_subj = convolve2d(subj_mask.astype(np.float32), self.ker, mode='same')
    subj_mask = conv_subj > 0.36
    conv_obj = convolve2d(obj_mask.astype(np.float32), self.ker, mode='same')
    obj_mask = conv_obj > 0.36
    
    subj_mask = subj_mask.reshape(self._size[0], self._size[0] // self._size[0], 
                                  self._size[1], self._size[1] // self._size[1]
    ).mean(axis=(1, 3)) > 0.5

    obj_mask = obj_mask.reshape(self._size[0], self._size[0] // self._size[0], 
                                self._size[1], self._size[1] // self._size[1]
    ).mean(axis=(1, 3)) > 0.5

    res[subj_mask] = 1
    res[obj_mask] = 2

    return res
