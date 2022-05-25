import collections
import functools
import logging
import os
import pathlib
import pickle
import sys
import warnings
from functools import partial
import wandb
import uuid
import atexit

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf
from tqdm import tqdm

import agent
import proposal
import embeddings
import elements
import common
from common.envs.async_env import Async

configs = pathlib.Path(sys.argv[0]).parent / 'configs_addressing.yaml'
configs = yaml.safe_load(configs.read_text())
config = elements.Config(configs['defaults'])
parsed, remaining = elements.FlagParser(configs=['defaults']).parse_known(
    exit_on_help=False)
for name in parsed.configs:
  config = config.update(configs[name])
config = elements.FlagParser(config).parse(remaining)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu) #str(args.gpu)
if '$' in config.logdir:
  config = config.update({
    'logdir': os.path.expandvars(config.logdir)
  })
if config.multitask.bootstrap:
  config = config.update({
    'multitask.data_path': os.path.join(config.logdir, "train_replay")
  })
if config.logging.wdb and '{run_id}' in config.logdir:
    config = config.update({'logdir': config.logdir.format(run_id=f'{wandb.run.name}_{wandb.run.id}')})
    if '{run_id}' in config.multitask.data_path:
      config = config.update(
        {'multitask.data_path': config.multitask.data_path.format(run_id=f'{wandb.run.name}_{wandb.run.id}')}
      )
logdir = pathlib.Path(config.logdir).expanduser()
config = config.update(
    steps=config.steps // config.action_repeat,
    eval_every=config.eval_every // config.action_repeat,
    log_every=config.log_every // config.action_repeat,
    time_limit=config.time_limit // config.action_repeat,
    prefill=config.prefill // config.action_repeat)
if config.segmentation and config.world_model == 'dreamer':
  config = config.update(img_channels=config.img_channels * 2)

tf.config.experimental_run_functions_eagerly(not config.jit)
message = 'No GPU found. To actually train on CPU remove this assert.'
assert tf.config.experimental.list_physical_devices('GPU'), message
for gpu in tf.config.experimental.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)
assert config.precision in (16, 32), config.precision
if config.precision == 16:
  from tensorflow.keras.mixed_precision import experimental as prec
  prec.set_policy(prec.Policy('mixed_float16'))

print('Logdir', logdir)
train_replay = common.Replay(logdir / 'train_replay', config.replay_size, **config.replay)
eval_replay = common.Replay(logdir / 'eval_replay', config.time_limit or 1, **config.replay)
Async.UID = str(uuid.uuid4().hex)
atexit.register(Async.close_all)

replays = dict()
replays["train"] = common.Replay(logdir / 'train_replay', config.replay_size, **config.replay)
replays["eval"] = common.Replay(logdir / 'eval_replay', config.time_limit or 1, **config.replay)
tronev = config.train_wm_eval or config.train_ac_eval
if config.multitask.mode == 'tronev':
  replays["eval_rand"] = common.Replay(logdir / 'eval_rand_replay', config.time_limit or 1, **config.replay)
if (config.multitask.mode != 'none') and (config.multitask.mode != 'tronev'):
  replays["mt"] = common.Replay(mt_path, load=config.keep_ram, **config.replay)
step = elements.Counter(replays["train"].total_steps)
# if config.multitask.mode != 'none':
#   mt_path = pathlib.Path(config.multitask.data_path).expanduser()
#   mt_replay = common.Replay(mt_path, load=config.keep_ram, **config.replay)
# else:
#   mt_replay = None
# step = elements.Counter(train_replay.total_steps)
step = elements.Counter(0)
outputs = [
    # elements.TerminalOutput(),
    elements.JSONLOutput(logdir),
    # elements.TensorBoardOutput(logdir),
]
logger = elements.Logger(step, outputs, multiplier=config.action_repeat)
metrics = collections.defaultdict(list)
should_train = elements.Every(config.train_every)
should_train_zero_shot = elements.Every(config.zero_shot_agent.train_every)
should_log = elements.Every(config.log_every)
should_video_train = elements.Every(config.eval_every)
should_video_eval = elements.Every(config.eval_every)
should_wdb_video = elements.Every(config.eval_every * 5)
should_openl_video = elements.Every(config.eval_every * 5)

def make_env(config, mode, **kws):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = common.DMC(task, config.action_repeat, config.image_size)
    env = common.NormalizeAction(env)
  elif suite == 'atari':
    env = common.Atari(
        task, config.action_repeat, config.image_size, config.grayscale,
        life_done=False, sticky_actions=True, all_actions=True)
    env = common.OneHotAction(env)
  elif suite == 'metaworld':
    if mode == 'eval':
      # in eval we freeze each worker to have a fixed env type
      if 'worker_id' in kws:
        del kws['worker_id']
    params = yaml.safe_load(config.env_params)
    params.update(kws)
    env = common.MetaWorld(
      task, config.action_repeat, config.image_size, transparent=config.transparent, **params
    )
    env.dump_tasks(str(logdir / 'tasks.pkl'))
    env = common.NormalizeAction(env)
  elif suite == 'causal':
    variables_space = dict(train='space_a', eval='space_b')[mode]
    # params = yaml.safe_load(config.env_params)
    params = config.cw_params._flat.copy()
    params.update(kws)
    env = common.CausalWorld(
      task, variables_space, config.action_repeat,
      config.image_size, **params
    )
  else:
    raise NotImplementedError(suite)
  env = common.TimeLimit(env, config.time_limit)
  env = common.RewardObs(env)
  env = common.ResetObs(env)
  return env

freezed_replay = False

def per_episode(ep, mode):
  global freezed_replay
  length = len(ep['reward']) - 1
  task_name = None
  if 'task_name' in ep:
    task_name = ep['task_name'][0]
  score = float(ep['reward'].astype(np.float64).sum())
  raw_score = float(ep['raw_reward'].astype(np.float64).sum())
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  # replay_ = dict(train=train_replay, eval=eval_replay)[mode if 'eval' not in mode else 'eval']
  replay_ = replays[mode]
  if not freezed_replay:
    ep_file = replay_.add(ep)
    # replays["mt"]._episodes[str(ep_file)] = ep
  logger.scalar(f'{mode}_transitions', replay_.num_transitions)
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_raw_return', raw_score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_eps', replay_.num_episodes)
  prefix = 'eval' if mode == 'eval' else ''
  if task_name is not None and mode == 'eval':
    task_name = task_name[:-len('-v2')]
    prefix = f'{prefix}_{task_name}'
  summ = {
    f'{config.logging.env_name}/{prefix}_return': score,
    f'{config.logging.env_name}/{prefix}_raw_return': raw_score,
    f'train_env_step': replays["train"].num_transitions,
    f'eval_env_step': replays["eval"].num_transitions,
    f'{config.logging.env_name}/{prefix}_length': length
  }
  # if config.logging.wdb:
  #   wandb.log(summ)
  should = {'train': should_video_train, 'eval': should_video_eval}[mode]
  if should(step):
    logger.video(f'{mode}_policy', ep['image'])
    # if should_wdb_video(step) and config.logging.wdb:
    #   video = np.transpose(ep['image'], (0, 3, 1, 2))
    #   videos = []
    #   rows = video.shape[1] // 3
    #   for row in range(rows):
    #     videos.append(video[:, row * 3: (row + 1) * 3])
    #   video = np.concatenate(videos, 3)
    #   # if config.logging.wdb:
    #   #   wandb.log({f"{mode}_policy": wandb.Video(video, fps=30, format="gif")})
    #   # TODO: save video to gif
    #   video = np.transpose(ep['segmentation'], (0, 3, 1, 2)).astype(np.float)
    #   video *= 100
    #   videos = []
    #   rows = video.shape[1]
    #   for row in range(rows):
    #     videos.append(video[:, row: row + 1])
    #   video = np.concatenate(videos, 3)
    #   # if config.logging.wdb:
    #   #   wandb.log({f"{mode}_segm_policy": wandb.Video(video, fps=30, format="gif")})
    #   # TODO: save video to gif
  logger.write()

print('Create envs.')
Async.UID = str(uuid.uuid4().hex)
atexit.register(Async.close_all)
# train_envs = [make_env(config, 'train') for _ in range(config.num_envs)]
# eval_envs = [make_env(config, 'eval') for _ in range(config.num_envs)]
dummy_env = make_env(config, 'train')
action_space = dummy_env.action_space['action']
parallel = 'process' if config.parallel else 'local'
train_driver = common.Driver(
  partial(make_env, config, 'train'), num_envs=2,
  mode=parallel, lock=config.num_envs > 1, lockfile=config.train_tasks_file,
)
# task_vec = None
# for env in train_driver._envs:
#   env.randomize_tasks = False
#   if task_vec is None:
#     if config.parallel:
#       task_vec = env.call('get_task_vector')()
#     else:
#       task_vec = env.get_task_vector()
train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
train_driver.on_step(lambda _: step.increment())
syncfile = None
if 'metaworld' in config.task:
  syncfile = train_driver._envs[0].syncfile
eval_driver = common.Driver(
  partial(make_env, config, 'eval'), num_envs=config.num_envs,
  mode=parallel, lock=config.num_envs > 1,
  lockfile=syncfile if config.test_tasks_file is None else config.test_tasks_file,
)
for env in eval_driver._envs:
  env.randomize_tasks = False
eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
kind = 'policy'
def ev_per_ep(ep, mode='eval'):
  global kind
  if kind == 'policy':
    per_episode(ep, mode='eval')
  elif kind == 'random':
    per_episode(ep, mode='rand_eval')
eval_driver.on_episode(ev_per_ep)
prefill = max(0, config.prefill - replays["train"].total_steps)
random_agent = common.RandomAgent(action_space)
if prefill:
  print(f'Prefill dataset ({prefill} steps).')
  random_agent = common.RandomAgent(action_space)
  train_driver(random_agent, episodes=1)
  eval_driver(random_agent, episodes=1)
  if config.multitask.mode == 'tronev':
    kind = 'random'
    eval_driver(random_agent, episodes=1)
    kind = 'policy'
  train_driver.reset()
  eval_driver.reset()
freezed_replay = True
print('Create agent.')

train_dataset = iter(replays["train"].dataset(**config.dataset))
eval_dataset = iter(replays["eval"].dataset(**config.dataset))
if tronev:
  train_rand_dataset = iter(replays["eval_rand"].dataset(**config.dataset))
# eval_dataset = iter(eval_replay.dataset(**config.dataset))

agnt = agent.Agent(config, logger, action_space, step, train_dataset)
print('Agent created')
if (logdir / 'variables.pkl').exists():
  print('Restoring trained agent')
  agnt.load(logdir / 'variables.pkl')
else:
  assert False, 'saved agent not found'

from collections import defaultdict
eval_policy = functools.partial(agnt.policy, mode='eval')
class MyStatsSaver:
  def __init__(self):
    self.task = None
    self.stats = defaultdict(list)

  def on_episode(self, ep, mode='train'):
    ep_return = sum(ep['raw_reward'])
    self.stats[self.task].append(ep_return)

  def dump(self, path):
    stats = {}
    stats['full'] = {k: v for k, v in self.stats.items()}
    stats['avg'] = {k: np.mean(v) for k, v in self.stats.items()}
    with open(path, 'wb') as f:
      pickle.dump(stats, f)
my_saver = MyStatsSaver()
eval_driver.on_episode(my_saver.on_episode)
# env_name = config.task.split('_', 2)[-1]
# env_name = env_name + '-v2'
# if config.parallel:
#   task_set, task_id = eval_driver._envs[0].call('get_task_set', env_name)()
# else:
#   task_set, task_id = eval_driver._envs[0].get_task_set(env_name)

def tasks_generator():
  x_mean = 0.95
  y_mean = 0.95
  mass_mean = 0.045

  for y_size in np.linspace(0.075, 0.115, 10): # 10 vals
    for mass in np.linspace(0.015, 0.100, 10): # 10 vals
      yield {'tool_block': {'mass': mass, 'size': np.array([x_mean, y_size, 0.085])}}, \
        (mass, x_mean, y_size)

  for x_size in np.linspace(0.075, 0.115, 10): # 10 vals
    for mass in np.linspace(0.015, 0.100, 10): # 10 vals
      yield {'tool_block': {'mass': mass, 'size': np.array([x_size, y_mean, 0.085])}}, \
        (mass, x_size, y_mean)

  for x_size in np.linspace(0.075, 0.115, 10): # 13 vals
    for y_size in np.linspace(0.075, 0.115, 10): # 13 vals
      yield {'tool_block': {'mass': mass_mean, 'size': np.array([x_size, y_size, 0.085])}}, \
        (mass_mean, x_size, y_size)

for task_id, task in tqdm(tasks_generator(), desc=logdir.stem):
  # curr_task_vec = task_vec.copy()
  my_saver.task = task
  for env in eval_driver._envs:
    if config.parallel:
      curr_task = env.call('set_starting_state', task_id, check_bounds=False)()
      # env.call('set_task_set', env_name, [curr_task])()
    else:
      curr_task = env.set_starting_state(task_id, check_bounds=False)
      # env.set_task_set(env_name, [curr_task])

  eval_driver(eval_policy, episodes=config.eval_episodes_per_env)
  my_saver.dump(logdir / 'stats.pkl')

# while step < config.steps:
#   logger.write()
#   print('Start evaluation.')
#   video = agnt.report(next(eval_dataset))
#   logger.add(video, prefix='eval')
#   if should_openl_video(step) and config.logging.wdb:
#     video = (np.transpose(video['openl'], (0, 3, 1, 2)) * 255).astype(np.uint8)
#     wandb.log({f"eval_openl": wandb.Video(video, fps=30, format="gif")})
#   eval_policy = functools.partial(agnt.policy, mode='eval')
#   eval_driver(eval_policy, episodes=config.eval_eps)
#   print('Start training.')
#   train_driver(agnt.policy, steps=config.eval_every)
#   agnt.save(logdir / 'variables.pkl')
# for env in train_envs + eval_envs:
#   try:
#     env.close()
#   except Exception:
#     pass
