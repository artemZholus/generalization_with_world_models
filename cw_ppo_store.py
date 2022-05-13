"""
This tutorial shows you how to train a policy using stable baselines with PPO
"""
import argparse
import json
import math
import pathlib
import time
import os

import common
import elements
import gym
import numpy as np
import wandb

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.intervention_actors import PushingBlockInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from stable_baselines3 import PPO
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv


OBJ_IMG_OBJ_IDS = {
  'reaching': [],
  'pushing': [0, 1],
  'picking': [0, 1],
  'pick_and_place': [5],
  'stacking2': [4, 5],
  'towers': [4, 5, 6, 7, 8],
}
FULL_IMG_OBJ_IDS = {
  'reaching': [],
  'pushing': [4, 5],
  'picking': [4, 5],
  'pick_and_place': [5],
  'stacking2': [4, 5],
  'towers': [4, 5, 6, 7, 8],
}
FULL_IMG_ROBOT_IDS = {
  'reaching': [1],
  'pushing': [1],
  'picking': [1],
  'pick_and_place': [1],
  'stacking2': [1],
  'towers': [1],
}

logdir = pathlib.Path('/home/yivchenkov/data/ppo')
eval_every = 5e5

should_wdb_video = elements.Every(eval_every * 5)

os.environ['MUJOCO_GL'] = 'egl'
os.environ['WANDB_START_METHOD'] = 'thread'


class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        env: gym.Env,
        task_family: str='pushing'
    ):
        super().__init__(env=env)
        self.t_start = time.time()
        self.task_family = task_family
        self._size=(64, 64)
        self._yaws=[0, 120, 240]
        self._pitches=[-60, -60, -60]
        self._distances=[0.6, 0.6, 0.6]
        self._base_positions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self._image_content = 'full'
        self.env.set_render_params(self._size, 
                                   self._yaws,
                                   self._pitches,
                                   self._distances,
                                   self._base_positions,
                                   self._image_content)
        self.ker = np.ones((3,3))
        self.ker /= 9.
        self._ep = [None]
        self._task_info=dict()

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True
        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        obs_vec = self.env.reset(**kwargs)
        self._task_info = self.env.env._stage.get_object_full_state('tool_block')
        # obs, segm = self.env.render_with_masks()
        # full_mask = self.convert_segm(segm, 'subj')
        # ob = {'flat_obs': obs_vec,
        #       'image': np.concatenate(obs, axis=2),
        #       'segmentation': full_mask}
        tran = self.parse_obs(obs_vec)
        tran['action'] = np.zeros(self.env.action_space.shape)
        tran['reward'] = 0.0
        tran['discount'] = 1.0
        tran['done'] = False
        tran['task_vector'] = self.get_task_vector()
        self._ep[0] = [tran]
        return obs_vec

    def step(self, action):
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        obs_vec, rew, done, info = self.env.step(action)

        info['discount'] = np.array(1. if not done else 0., np.float32)
        disc = info.get('discount', np.array(1 - float(done)))

        tran = self.parse_obs(obs_vec)
        tran['action'] = np.array(action)
        tran['reward'] = rew
        tran['discount'] = disc
        tran['done'] = done
        tran['task_vector'] = self.get_task_vector()
        # obs, segm = self.env.render_with_masks()
        # full_mask = self.convert_segm(segm, 'subj')
        # ob = {'flat_obs': obs_vec,
        #       'image': np.concatenate(obs, axis=2),
        #       'segmentation': full_mask}
        
        # obs = {k: self._convert(v) for k, v in ob.items()}
        self._ep[0].append(tran)
        if done:
            ep = self._ep[0]
            ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
            per_episode(ep)
        return obs_vec, rew, done, info

    def parse_obs(self, obs_vec):
        obs = {'flat_obs': obs_vec}
        # following is valid only for `pushing` task
        obs['time_left'] = obs_vec[:1]
        obs['joint_positions'] = obs_vec[1:10]
        obs['joint_velocities'] = obs_vec[10:19]
        obs['end_effector_positions'] = obs_vec[19:28]
        obs['tool_type'] = obs_vec[28:29]
        obs['tool_size'] = obs_vec[29:32]
        obs['tool_position'] = obs_vec[32:35]
        obs['tool_orientation'] = obs_vec[35:39]
        obs['tool_linear_velocity'] = obs_vec[39:42]
        obs['tool_angular_velocity'] = obs_vec[42:45]
        obs['goal_type'] = obs_vec[45:46]
        obs['goal_size'] = obs_vec[46:49]
        obs['goal_position'] = obs_vec[49:52]
        obs['goal_orientation'] = obs_vec[52:56]
        obs['obj_gt'] = obs_vec[29:45]
        return obs

    def get_task_vector(self):
        mass = (1.0 - (-1.0)) * (self._task_info['mass'] - 0.015)/(0.1 - 0.015) + (-1.0)
        size = (1.0 - (-1.0)) * (self._task_info['size'][:2] - 0.055)/(0.095 / 0.055) + (-1.0)
        radius = (1.0 - (-1.0)) * (self._task_info['cylindrical_position'][:1] - 0.0)/(0.15 - 0.0) + (-1.0)
        angle = self._task_info['cylindrical_position'][1:2] / math.pi
        quat = self._task_info['orientation']
        return np.hstack((mass, size, radius, angle, quat))

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value


def per_episode(ep, log_wdb=True, env_name='pushing', mode='train'):
    length = len(ep['reward']) - 1
    task_name = None
    if 'task_name' in ep:
        task_name = ep['task_name'][0]
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'Train episode has {length} steps and return {score:.1f}.')
    # replay_ = dict(train=train_replay, eval=eval_replay)[mode if 'eval' not in mode else 'eval']
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    common.save_episodes(logdir, [ep])
    steps_recorded = len(list(logdir.expanduser().glob('*.npz'))) * length
    prefix = mode if 'eval' in mode else ''
    if task_name is not None and 'eval' in mode:
        task_name = task_name[:-len('-v2')]
        prefix = f'{prefix}_{task_name}'
    summ = {
        f'{env_name}/{prefix}_return': score,
        f'train_env_step': steps_recorded,
        f'{env_name}/{prefix}_length': length
    }
    if log_wdb:
        wandb.log(summ)
    # if should_wdb_video(steps_recorded) and log_wdb:
    #     video = np.transpose(ep['image'], (0, 3, 1, 2))
    #     videos = []
    #     rows = video.shape[1] // 3
    #     for row in range(rows):
    #         videos.append(video[:, row * 3: (row + 1) * 3])
    #     video = np.concatenate(videos, 3)
    #     if log_wdb:
    #         wandb.log({f"{mode}_policy": wandb.Video(video, fps=30, format="gif")})
    #     video = np.transpose(ep['segmentation'], (0, 3, 1, 2)).astype(np.float)
    #     video *= 100
    #     videos = []
    #     rows = video.shape[1]
    #     for row in range(rows):
    #         videos.append(video[:, row: row + 1])
    #     video = np.concatenate(videos, 3)
    #     if log_wdb:
    #         wandb.log({f"{mode}_segm_policy": wandb.Video(video, fps=30, format="gif")})


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, ppo_config, total_time_steps,
                 validate_every_timesteps, task_name):

    def _make_env(rank):

        def _init():
            wandb.init(entity='cds-mipt', project='oc_mbrl', group='causal',
                   name='causal_ppo_store')
            task = generate_task(task_generator_id=task_name)
            env = CausalWorld(task=task,
                              normalize_actions=True,
                              normalize_observations=True,
                              initialize_all_clients=False,
                              skip_frame=skip_frame,
                              enable_visualization=False,
                              seed=seed_num + rank,
                              action_mode='end_effector_positions',
                              max_episode_length=maximum_episode_length)
            inter_actor = PushingBlockInterventionActorPolicy(
                positions=True,
                orientations=True,
                masses=True,
                sizes=True,
                goals=False
            )
            env = CurriculumWrapper(env,
                                    intervention_actors=[inter_actor],
                                    actives=[(0, 1000000000, 1, 0)])
            env = Monitor(env, task_name)
            return env

        set_random_seed(seed_num)
        return _init

    os.makedirs(log_relative_path)
    policy_kwargs = dict(net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    model = PPO('MlpPolicy',
                 env,
                 _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1,
                 **ppo_config)
    save_config_file(ppo_config,
                     _make_env(0)(),
                     os.path.join(log_relative_path, 'config.json'))
    for i in range(int(total_time_steps / validate_every_timesteps)):
        model.learn(total_timesteps=validate_every_timesteps,
                    tb_log_name="ppo",
                    reset_num_timesteps=False)
        model.save(os.path.join(log_relative_path, 'saved_model'))
    return

def save_config_file(ppo_config, env, file_path):
    task_config = env.env.env._task.get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = env.env.env.get_world_params()
    env.close()
    configs_to_save = [task_config, env_config, ppo_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    #TODO: pass reward weights here!!
    ap.add_argument("--seed_num", required=False, default=0, help="seed number")
    ap.add_argument("--skip_frame",
                    required=False,
                    default=10,
                    help="skip frame")
    ap.add_argument("--max_episode_length",
                    required=False,
                    default=250,
                    help="maximum episode length")
    ap.add_argument("--total_time_steps_per_update",
                    required=False,
                    default=100000,
                    help="total time steps per update")
    ap.add_argument("--num_of_envs",
                    required=False,
                    default=20,
                    help="number of parallel environments")
    ap.add_argument("--task_name",
                    required=False,
                    default="pushing",
                    help="the task nam for training")
    ap.add_argument("--fixed_position",
                    required=False,
                    default=True,
                    help="define the reset intervention wrapper")
    ap.add_argument("--log_relative_path", required=True, help="log folder")
    args = vars(ap.parse_args())
    total_time_steps_per_update = int(args['total_time_steps_per_update'])
    num_of_envs = int(args['num_of_envs'])
    log_relative_path = str(args['log_relative_path'])
    maximum_episode_length = int(args['max_episode_length'])
    skip_frame = int(args['skip_frame'])
    seed_num = int(args['seed_num'])
    task_name = str(args['task_name'])
    fixed_position = bool(args['fixed_position'])
    assert (((float(total_time_steps_per_update) / num_of_envs) /
             5).is_integer())
    ppo_config = {
        "gamma": 0.99,
        "n_steps": 5000,
        "ent_coef": 0,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 10,
        "batch_size": 50,
        "n_epochs": 4,
        "tensorboard_log": log_relative_path
    }
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 ppo_config=ppo_config,
                 total_time_steps=60000000,
                 validate_every_timesteps=1000000,
                 task_name=task_name)
