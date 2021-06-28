# Instructions
#
# Test your setup:
#
# docker run -it --rm --gpus all tensorflow/tensorflow:2.4.2-gpu nvidia-smi
#
# Atari:
#
# docker build -t dreamerv2 .
# docker run -it --rm --gpus all -v ~/logdir:/logdir dreamerv2 \
#   python3 dreamerv2/train.py --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs defaults atari --task atari_pong
#
# DMC:
#
# docker build -t dreamerv2 . --build-arg MUJOCO_KEY="$(cat ~/.mujoco/mjkey.txt)"
# docker run -it --rm --gpus all -v ~/logdir:/logdir dreamerv2 \
#   python3 dreamerv2/train.py --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs defaults dmc --task dmc_walker_walk
# 
# Retrospective (vscode container):
# 
# docker run -it --rm --gpus all -v $(pwd):/dreamerv2 -w /dreamerv2 -d -v /mnt/data/users/<yourname>:/data/ dreamerv2 tail -F file

FROM tensorflow/tensorflow:2.4.2-gpu

# System packages.
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libgl1-mesa-dev \
  python3-pip \
  unrar \
  wget \
  git \
  && apt-get clean

# MuJoCo.
ENV MUJOCO_GL egl
RUN mkdir -p /root/.mujoco && \
  wget -nv https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d /root/.mujoco && \
  rm mujoco.zip

# Python packages.
RUN pip3 install --no-cache-dir \
  'gym[atari]' \
  elements \
  dm_control \
  ruamel.yaml \
  tensorflow_probability==0.12.2 \
  wandb

# Atari ROMS.
RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
  unrar x Roms.rar && \
  unzip ROMS.zip && \
  python3 -m atari_py.import_roms ROMS && \
  rm -rf Roms.rar ROMS.zip ROMS

# MuJoCo key.
ARG MUJOCO_KEY=""
RUN echo "$MUJOCO_KEY" > /root/.mujoco/mjkey.txt
RUN cat /root/.mujoco/mjkey.txt

# DreamerV2.
ENV TF_XLA_FLAGS --tf_xla_auto_jit=2
COPY . /app
WORKDIR /app
CMD [ \
  "python3", "dreamerv2/train.py", \
  "--logdir", "/logdir/$(date +%Y%m%d-%H%M%S)", \
  "--configs", "defaults", "atari", \
  "--task", "atari_pong" \
]
