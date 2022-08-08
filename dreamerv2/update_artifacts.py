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
if config.logging.wdb:
  wandb.init(entity='cds-mipt', project='cema_iclr', config=common.flatten_conf(config), group=config.logging.exp_name,
             name=config.logging.run_name, settings=wandb.Settings(start_method='thread'), resume="must")
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
model = wandb.Artifact(f'{wandb.run.name}_{wandb.run.id}', type='model')
model.add_file(str(logdir / 'variables.pkl'))
wandb.log_artifact(model)
exit(0)
#for env in train_envs + eval_envs:
#  try:
#    env.close()
#  except Exception:
#    pass
