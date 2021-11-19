import pandas as pd
import atexit
import time
from argparse import ArgumentParser
from subprocess import Popen
parser = ArgumentParser()

parser.add_argument('--gpus', type=int, default=3)
parser.add_argument('--frm', type=int, default=0)
parser.add_argument('--to', type=int, default=0)
parser.add_argument('--per_gpu', type=int, default=3)
parser.add_argument('--n_workers', type=int, default=20)
args = parser.parse_args()
def cycle(gen):
    while True:
        for it in gen():
            yield it
gpu_load = {i: [None for _ in range(args.per_gpu)] for i in range(args.gpus)}
def resources():
    for gpu in range(len(gpu_load)):
        for worker_id in range(len(gpu_load[0])):
            yield gpu, worker_id, gpu_load[gpu][worker_id]
def cleanup():
    for _, _, p in resources():
        if p.poll() is None:
            p.kill()
    print('cleaned up!')
atexit.register(cleanup)
df = pd.read_csv('models.csv')
for i, (gpu, worker_id, proc) in zip(range(args.frm, args.to + 1), cycle(resources)):
    path, kws = df.iloc[i].path, df.iloc[i].args
    cmd = [
        "python", "dreamerv2/eval.py", "--configs", "defaults", "mtw", "ml1", "rotated_drawer_close", "open_umbrella",
    ]
    kws = kws.strip().split(' ')
    if kws[0] != '':
        cmd += kws
    cmd +=  [
        "--logdir", path, "--gpu", str(gpu), "--num_envs", str(args.n_workers),
    ]
    if proc is None or proc.wait() is not None:
        print(f'Calling command: {" ".join(cmd)}')
        new_proc = Popen(cmd)
        gpu_load[gpu][worker_id] = new_proc
for _, _, proc in resources():
    if proc is not None:
        proc.wait()
print('done!')
