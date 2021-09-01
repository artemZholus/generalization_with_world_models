from .atari import Atari
from .dmc import DMC
from .metaworld import MetaWorld
from .wrappers import NormalizeAction, OneHotAction, TimeLimit, RewardObs, ResetObs
import yaml


def make_env(mode, config, logdir):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = DMC(task, config.action_repeat, config.image_size)
    env = NormalizeAction(env)
  elif suite == 'atari':
    env = Atari(
        task, config.action_repeat, config.image_size, config.grayscale,
        life_done=False, sticky_actions=True, all_actions=True)
    env = OneHotAction(env)
  elif suite == 'metaworld':
    params = yaml.safe_load(config.env_params)
    env = MetaWorld(task, config.action_repeat, config.image_size, **params)
    env.dump_tasks(str(logdir / 'tasks.pkl'))
    env = NormalizeAction(env)
  elif suite == 'isaac':
    from common.envs import isaac
    params = yaml.safe_load(config.env_params)
    w = params['width']
    h = params['height']
    kwargs = {}
    env_config_file = params['config']
    env = isaac.Isaac(
      task, config.action_repeat, (h, w), grayscale=False,
      onehot=False,
      episode_steps=config.time_limit,
      env_config_file=env_config_file,
      **kwargs
    )
  else:
    raise NotImplementedError(suite)
  env = TimeLimit(env, config.time_limit)
  env = RewardObs(env)
  env = ResetObs(env)
  return env
