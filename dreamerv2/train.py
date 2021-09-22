import collections
import functools
import logging
import os
import pathlib
import sys
import warnings
import wandb

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
  wandb.init(project='python-tf_dreamer', config=common.flatten_conf(config), group=config.logging.exp_name, 
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
train_replay = common.Replay(logdir / 'train_replay', config.replay_size, **config.replay)
eval_replay = common.Replay(logdir / 'eval_replay', config.time_limit or 1, **config.replay)
if config.multitask.mode != 'none':
  mt_path = pathlib.Path(config.multitask.data_path).expanduser()
  mt_replay = common.Replay(mt_path, load=config.keep_ram, **config.replay)
else:
  mt_replay = None
if config.zero_shot:
  zs_eval_replay = common.Replay(logdir / 'zs_eval_replay', config.time_limit or 1, **config.replay)
step = elements.Counter(train_replay.total_steps)
outputs = [
    elements.TerminalOutput(),
    elements.JSONLOutput(logdir),
    elements.TensorBoardOutput(logdir),
]
logger = elements.Logger(step, outputs, multiplier=config.action_repeat)
metrics = collections.defaultdict(list)
should_train = elements.Every(config.train_every)
# should_train_zero_shot = elements.Every(config.zero_shot_agent.train_every)
should_log = elements.Every(config.log_every)
should_video_train = elements.Every(config.eval_every)
should_video_eval = elements.Every(config.eval_every)
should_video_zs_eval = elements.Every(config.zero_shot_agent.eval_every)
should_wdb_video = elements.Every(config.eval_every * 5)
should_openl_video = elements.Every(config.eval_every * 5)

def make_env(mode):
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
    params = yaml.safe_load(config.env_params)
    env = common.MetaWorld(task, config.action_repeat, config.image_size, **params)
    env.dump_tasks(str(logdir / 'tasks.pkl'))
    env = common.NormalizeAction(env)
  else:
    raise NotImplementedError(suite)
  env = common.TimeLimit(env, config.time_limit)
  env = common.RewardObs(env)
  env = common.ResetObs(env)
  return env

def per_episode(ep, mode):
  length = len(ep['reward']) - 1
  score = float(ep['reward'].astype(np.float64).sum())
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  replay_ = dict(train=train_replay, 
                 eval=eval_replay,
                 zs_eval=zs_eval_replay)[mode]
  ep_file = replay_.add(ep)
  if mode == 'train' and config.multitask.bootstrap:
    # mt_replay.add(ep)
    mt_replay._episodes[str(ep_file)] = ep
  logger.scalar(f'{mode}_transitions', replay_.num_transitions)
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_eps', replay_.num_episodes)
  prefix = dict(train='',
                eval='eval/',
                zs_eval='zs_eval/')[mode]
  summ = {
    f'{prefix}{config.logging.env_name}/return': score,
    f'{prefix}env_step'.replace('/', '_'): replay_.num_transitions,
    f'{prefix}{config.logging.env_name}/length': length
  }
  if config.logging.wdb:
    wandb.log(summ)
  should = {'train': should_video_train, 
            'eval': should_video_eval,
            'zs_eval': should_video_zs_eval}[mode]
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
  logger.write()

print('Create envs.')
train_envs = [make_env('train') for _ in range(config.num_envs)]
eval_envs = [make_env('eval') for _ in range(config.num_envs)]
zs_eval_envs = [make_env('zs_eval') for _ in range(config.num_envs)]
action_space = train_envs[0].action_space['action']
train_driver = common.Driver(train_envs)
train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
train_driver.on_step(lambda _: step.increment())
eval_driver = common.Driver(eval_envs)
eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
zs_eval_driver = common.Driver(zs_eval_envs)
zs_eval_driver.on_episode(lambda ep: per_episode(ep, mode='zs_eval'))

prefill = max(0, config.prefill - train_replay.total_steps)
if prefill:
  print(f'Prefill dataset ({prefill} steps).')
  random_agent = common.RandomAgent(action_space)
  train_driver(random_agent, steps=prefill, episodes=1)
  eval_driver(random_agent, episodes=1)
  zs_eval_driver(random_agent, episodes=1)
  train_driver.reset()
  eval_driver.reset()
  zs_eval_driver.reset()

print('Create agent.')
train_dataset = iter(train_replay.dataset(**config.dataset))
eval_dataset = iter(eval_replay.dataset(**config.dataset))
zs_eval_dataset = iter(zs_eval_replay.dataset(**config.dataset))
if config.embeddings.trainer.mode == 'sa_dyne':
  path = pathlib.Path(config.embeddings.data_path).expanduser()
  dyne_dataset = iter(common.Replay(path).dataset(length=config.embeddings.traj_len,
                                                  sequential=True,
                                                  **config.embeddings.dataset
                                                  ))
  dyne_encoder = embeddings.SADyneEncoder(config.embeddings.dyne, 
                                          config.embeddings.traj_len, 
                                          action_space)
  trainer = embeddings.DyneTrainer(config.embeddings.trainer, dyne_encoder)
  for i in range(config.embeddings.trainer.training_steps):
    mets = trainer.train(next(dyne_dataset))
    if i % config.embeddings.trainer.log_every == 0:
      mets['dyne_step'] = i
      if config.logging.wdb:
        wandb.log(mets)
  del dyne_dataset

agnt = agent.Agent(config, logger, action_space, step, train_dataset)
if 'multitask' not in config or config.multitask.mode == 'none':
  batch_proposal = proposal.TrainProposal(config, agnt, step, train_dataset)
elif config.multitask.mode == 'raw':
  batch_proposal = proposal.RawMultitask(config, agnt, step, train_dataset, mt_replay)
elif config.multitask.mode == 'return':
  batch_proposal = proposal.ReturnBasedProposal(config, agnt, step, train_dataset, mt_replay)
elif config.multitask.mode == 'addressing':
  batch_proposal = proposal.RetrospectiveAddressing(config, agnt, step, train_dataset, mt_replay)
elif config.multitask.mode == 'addressing_dyne':
  batch_proposal = proposal.DyneRetrospectiveAddressing(config, agnt, step, train_dataset, mt_replay, trainer.dyne_encoder)
print('Agent created')
if (logdir / 'variables.pkl').exists():
  agnt.load(logdir / 'variables.pkl')
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
    average['env_step'] = train_replay.num_transitions
    if config.logging.wdb:
      wandb.log(average)
    print('\n'.join(batch_proposal.timed.summarize()))
    for name, values in metrics.items():
      logger.scalar(name, np.array(values, np.float64).mean())
      metrics[name].clear()
    logger.add(agnt.report(next(train_dataset)), prefix='train')
    logger.write(fps=True)
train_driver.on_step(train_step)

while step < config.steps:
  logger.write()
  print('Start evaluation.')
  video = agnt.report(next(eval_dataset))
  logger.add(video, prefix='eval')
  if should_openl_video(step) and config.logging.wdb:
    video = (np.transpose(video['openl'], (0, 3, 1, 2)) * 255).astype(np.uint8)
    wandb.log({f"eval_openl": wandb.Video(video, fps=30, format="gif")})
  eval_policy = functools.partial(agnt.policy, mode='eval')
  eval_driver(eval_policy, episodes=config.eval_eps)
  if config.zero_shot:
    zs_eval_policy = functools.partial(agnt.policy, mode='eval', second_agent=True)
    zs_eval_driver(zs_eval_policy, episodes=config.zero_shot_agent.eval_eps)
  print('Start training.')
  train_driver(agnt.policy, steps=config.eval_every)
  agnt.save(logdir / 'variables.pkl')
for env in train_envs + eval_envs:
  try:
    env.close()
  except Exception:
    pass
