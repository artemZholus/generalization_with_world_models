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
  libosmesa6-dev \
  libgl1-mesa-glx \
  libglfw3 \
  patchelf \
  python3-pip \
  unrar \
  wget \
  git \
  vim \
  && apt-get clean

# MuJoCo.
ENV MUJOCO_GL egl
RUN mkdir -p /root/.mujoco && \
  wget -nv https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d /root/.mujoco && \
  cp -r /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 && \
  rm mujoco.zip

# MuJoCo key.
ARG MUJOCO_KEY=""
RUN echo "$MUJOCO_KEY" > /root/.mujoco/mjkey.txt
RUN cat /root/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin"

# Python packages.
RUN pip3 install --no-cache-dir \
  'gym[atari]' \
  atari_py \
  mujoco_py \
  elements \
  dm_control \
  jupyterlab \
  ruamel.yaml \
  tensorflow_probability==0.12.2 \
  wandb

# Atari ROMS.
RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
  unrar x Roms.rar && \
  unzip ROMS.zip && \
  python3 -m atari_py.import_roms ROMS && \
  rm -rf Roms.rar ROMS.zip ROMS


# DreamerV2.
ENV TF_XLA_FLAGS --tf_xla_auto_jit=2
COPY . /app
WORKDIR /app

# wandb credentials
ARG NETRC_KEY=""
RUN touch $HOME/.netrc \
  && echo "machine api.wandb.ai" >> $HOME/.netrc \
  && echo "  login user" >> $HOME/.netrc \
  && echo "  password $NETRC_KEY" >> $HOME/.netrc 

CMD [ \
  "python3", "dreamerv2/train.py", \
  "--logdir", "/logdir/$(date +%Y%m%d-%H%M%S)", \
  "--configs", "defaults", "atari", \
  "--task", "atari_pong" \
]
