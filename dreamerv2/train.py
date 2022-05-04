import collections
import functools
import logging
import os
import pathlib
import sys
import warnings
from functools import partial
from copy import deepcopy as copy
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
import uuid
from tqdm import tqdm

import agent
import proposal
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
if config.logging.wdb:
  wandb.init(entity='cds-mipt', project='oc_mbrl', config=common.flatten_conf(config), group=config.logging.exp_name,
             name=config.logging.run_name, settings=wandb.Settings(start_method='thread'))
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
Async.UID = str(uuid.uuid4().hex)
atexit.register(Async.close_all)

replays = dict()
replays["train"] = common.Replay(logdir / 'train_replay', config.replay_size, **config.replay)
replays["eval"] = common.Replay(logdir / 'eval_replay', config.time_limit or 1, **config.replay)
tronev = config.train_wm_eval or config.train_ac_eval
if config.multitask.mode == 'tronev':
  replays["eval_rand"] = common.Replay(logdir / 'eval_rand_replay', config.time_limit or 1, **config.replay)
if (config.multitask.mode != 'none') and (config.multitask.mode != 'tronev'):
  mt_path = pathlib.Path(config.multitask.data_path).expanduser()
  replays["mt"] = common.Replay(mt_path, load=config.keep_ram, **config.replay)

step = elements.Counter(replays["train"].total_steps)
outputs = [
    elements.TerminalOutput(),
    elements.JSONLOutput(logdir),
    elements.TensorBoardOutput(logdir),
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
    params = yaml.safe_load(config.env_params)
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

def per_episode(ep, mode):
  length = len(ep['reward']) - 1
  task_name = None
  if 'task_name' in ep:
    task_name = ep['task_name'][0]
  score = float(ep['reward'].astype(np.float64).sum())
  raw_score = float(ep['raw_reward'].astype(np.float64).sum())
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  replay_ = replays[mode]
  # replay_ = dict(train=train_replay, eval=eval_replay)[mode if 'eval' not in mode else 'eval']
  ep_file = replay_.add(ep)
  if mode == 'train' and config.multitask.bootstrap:
    # mt_replay.add(ep)
    replays["mt"]._episodes[str(ep_file)] = ep
  logger.scalar(f'{mode}_transitions', replay_.num_transitions)
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_raw_return', raw_score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_eps', replay_.num_episodes)
  prefix = mode if 'eval' in mode else ''
  if task_name is not None and 'eval' in mode:
    task_name = task_name[:-len('-v2')]
    prefix = f'{prefix}_{task_name}'
  summ = {
    f'{config.logging.env_name}/{prefix}_return': score,
    f'{config.logging.env_name}/{prefix}_raw_return': raw_score,
    f'train_env_step': replays["train"].num_transitions,
    f'eval_env_step': replays["eval"].num_transitions,
    f'{config.logging.env_name}/{prefix}_length': length
  }
  if config.logging.wdb:
    wandb.log(summ)
  should = {'train': should_video_train, 'eval': should_video_eval}[mode if 'eval' not in mode else 'eval']
  if should(step):
    logger.video(f'{mode}_policy', ep['image'])
    if should_wdb_video(step) and config.logging.wdb:
      video = np.transpose(ep['image'], (0, 3, 1, 2))
      videos = []
      rows = video.shape[1] // 3
      for row in range(rows):
        videos.append(video[:, row * 3: (row + 1) * 3])
      video = np.concatenate(videos, 3)
      if config.logging.wdb:
        wandb.log({f"{mode}_policy": wandb.Video(video, fps=30, format="gif")})
      video = np.transpose(ep['segmentation'], (0, 3, 1, 2)).astype(np.float)
      video *= 100
      videos = []
      rows = video.shape[1]
      for row in range(rows):
        videos.append(video[:, row: row + 1])
      video = np.concatenate(videos, 3)
      if config.logging.wdb:
        wandb.log({f"{mode}_segm_policy": wandb.Video(video, fps=30, format="gif")})
  logger.write()

print('Create envs.')
Async.UID = str(uuid.uuid4().hex)
atexit.register(Async.close_all)
# train_envs = [make_env(config, 'train') for _ in range(config.num_envs)]
# eval_envs = [make_env(config, 'eval') for _ in range(config.num_envs)]
dummy_env = make_env(config, 'train')
action_space = dummy_env.action_space['action']
def iter_tasks(kind):
  base = np.array([0.02, 0.9 , 0.  , 0.  ])
  while True:
    if kind == 'umbrella':
      if np.random.rand() > 0.5:
        angle = np.random.randint(135, 221)
      else:
        angle = np.random.randint(315, 401) % 360
    elif kind == 'monotonic':
      angle = np.random.randint(0, 236)
    elif kind == 'full':
      angle = np.random.randint(0, 361)
    else:
      raise ValueError(f'Unsupported kind: {kind}')
    print(f'next angle: {angle}')
    vec = copy(base)
    vec[-1] = angle
    yield vec

def procedural_env_ctor(mode, **kws):
  env = make_env(config, mode, **kws)
  env.set_tasks_generator(
    iter(iter_tasks(kind='monotonic' if 'monotonic' in config.train_tasks_file else 'umbrella'))
  )
  return env
parallel = 'process' if config.parallel else 'local'
train_driver = common.Driver(
  partial(make_env, config, 'train'), num_envs=config.num_envs,
  mode=parallel, lock=config.num_envs > 1, lockfile=config.train_tasks_file,
)
train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
train_driver.on_step(lambda _: step.increment())
syncfile = None
if 'metaworld' in config.task:
  syncfile = train_driver._envs[0].syncfile
def generate_tasks(name, kind):
  """
  generate random tasks for iid generalization
  rotated open/close env

  Args:
    name: drawer-open | drawer-close
    kind: monotonic | umbrella
  """
  base = np.array([0.02, 0.9 , 0.  , 0.  ])
  if kind == 'umbrella':
    high = np.random.randint(135, 221, 24)
    low = np.random.randint(315, 401, 24) % 360
    rng = np.concatenate([high, low], 0)
  elif kind == 'monotonic':
    rng = np.random.randint(0, 236, 48)
  elif kind == 'full':
    rng = np.random.randint(0, 361, 72)
  else:
    raise ValueError(f'Unsupported kind: {kind}')
  tasks = []
  for val in rng:
    vec = copy(base)
    vec[-1] = val
    tasks.append(vec)
  return {f'{name}-v2': tasks}

if config.iid_eval:
  lockfile = syncfile if config.test_tasks_file is None else config.test_tasks_file
  def env_ctor(**kws):
    env = make_env(config, 'eval', **kws)
    params = generate_tasks(config.task.split('_')[-1],
      kind='monotonic' if 'monotonic' in config.train_tasks_file else 'umbrella')
    env.create_tasks(params)
    return env

if config.iid_eval:
  lockfile = config.train_tasks_file if config.test_tasks_file is None else config.test_tasks_file
  iid_eval_driver = common.Driver(
    env_ctor, num_envs=config.num_envs,
    mode=parallel, lock=config.num_envs > 1,
    lockfile=f'{lockfile}_iid',
  )
  iid_eval_driver.on_episode(lambda ep: per_episode(ep, mode='iid_eval'))
else:
  iid_eval_driver = None

eval_driver = common.Driver(
  partial(make_env, config, 'eval'), num_envs=config.num_envs,
  mode=parallel, lock=config.num_envs > 1,
  lockfile=syncfile if config.test_tasks_file is None else config.test_tasks_file,
)
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
  train_driver(random_agent, steps=prefill, episodes=1)
  eval_driver(random_agent, episodes=1)
  if config.multitask.mode == 'tronev':
    kind = 'random'
    eval_driver(random_agent, episodes=1)
    kind = 'policy'
  train_driver.reset()
  eval_driver.reset()
  if config.iid_eval:
    iid_eval_driver(random_agent, episodes=1)
    iid_eval_driver.reset()

print('Create agent.')
train_dataset = iter(replays["train"].dataset(**config.dataset))
eval_dataset = iter(replays["eval"].dataset(**config.dataset))
if tronev:
  train_rand_dataset = iter(replays["eval_rand"].dataset(**config.dataset))

agnt = agent.Agent(config, logger, action_space, step, train_dataset)
if 'multitask' not in config or config.multitask.mode == 'none':
  batch_proposal = proposal.TrainProposal(config, agnt, step, train_dataset)
elif config.multitask.mode == 'tronev':
  batch_proposal = proposal.EvalTrainer(config, agnt, step, train_dataset, train_rand_dataset)
elif config.multitask.mode == 'raw':
  batch_proposal = proposal.RawMultitask(config, agnt, step, train_dataset, replays["mt"])
print('Agent created')
if (logdir / 'variables.pkl').exists() or config.agent_path != 'none' or config.wm_path != 'none' or config.ac_path != 'none':
  if config.wm_path != 'none':
    agnt.wm.load(config.wm_path)
  if config.ac_path != 'none':
    agnt._task_behavior.load(config.ac_path)
  if config.ac_path == 'none' and config.wm_path == 'none' and config.agent_path == 'none':
    agnt.load(logdir / 'variables.pkl')
  elif config.ac_path == 'none' and config.wm_path == 'none':
    agnt.load(config.agent_path)
    common.tfutils.reset_model(agnt._task_behavior)
else:
  config.pretrain and print('Pretrain agent.')
  for _ in tqdm(range(config.pretrain)):
    agnt.train(next(train_dataset))

def train_step(tran):
  if should_train(step):
    for _ in range(config.train_steps):
      _, mets = batch_proposal.train(agnt)
      # _, mets = agnt.train(next(train_dataset))
      [metrics[key].append(value) for key, value in mets.items()]
  if should_log(step):
    average = {f'agent/{k}': np.array(v, np.float64).mean() for k, v in metrics.items()}
    average['env_step'] = replays["train"].num_transitions
    if config.logging.wdb:
      wandb.log(average)
    print('\n'.join(batch_proposal.timed.summarize()))
    for name, values in metrics.items():
      logger.scalar(name, np.array(values, np.float64).mean())
      metrics[name].clear()
    logger.add(agnt.report(next(train_dataset)), prefix='train')
    logger.write(fps=True)
train_driver.on_step(train_step)
eval_driver._kwargs = {}
while step < config.steps:
  logger.write()
  print('Start evaluation.')
  video = agnt.report(next(eval_dataset))
  if isinstance(video, dict):
    for k, vid in video.items():
      logger.add({'openl': vid}, prefix=f'eval_{k}')
  else:
    logger.add({'openl': video}, prefix='eval')
  if should_openl_video(step) and config.logging.wdb:
    if isinstance(video, dict):
      for k, vid in video.items():
        vi = (np.transpose(vid, (0, 3, 1, 2)) * 255).astype(np.uint8)
        wandb.log({f"eval_openl_{k}": wandb.Video(vi, fps=30, format="gif")})
    else:
      video = (np.transpose(video['openl'], (0, 3, 1, 2)) * 255).astype(np.uint8)
      wandb.log({f"eval_openl": wandb.Video(video, fps=30, format="gif")})
  eval_policy = functools.partial(agnt.policy, mode='eval')
  eval_driver(eval_policy, episodes=config.eval_eps)
  if config.multitask.mode == 'tronev':
    kind = 'random'
    eval_driver(random_agent, episodes=config.eval_eps)
    kind = 'policy'
  if config.iid_eval:
    iid_eval_driver(eval_policy, episodes=config.eval_eps)
  print('Start training.')
  train_driver(agnt.policy, steps=config.eval_every)
  agnt.save(logdir / 'variables.pkl')
exit(0)
#for env in train_envs + eval_envs:
#  try:
#    env.close()
#  except Exception:
#    pass
