from nvcr.io/nvidia/isaac-sim:2020.2_ea

RUN apt-get update && apt-get install -y \
  make \
  cmake \
  gcc \
  g++ \
  vim \
  git \
  libapr1 \
  curl \
  wget \
  x11-xserver-utils \
  openjdk-8-jdk \
  xvfb \
  lsb-release \
  sudo \
  ffmpeg \
  software-properties-common


#RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-toolkit-10-1 
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y install cuda-toolkit-11-0

ENV PATH=$PATH:/usr/local/cuda-11.0/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

#RUN sudo apt-get install -y --allow-downgrades --reinstall libcublas10=10.2.1.243-1 libcublas-dev=10.2.1.243-1

#ENTRYPOINT /bin/bash
COPY ./cudnn-11.0-linux-x64-v8.0.5.39.tgz /usr/local
RUN tar -xvzf /usr/local/cudnn-11.0-linux-x64-v8.0.5.39.tgz -C /usr/local/
RUN rm -rf /usr/local/cudnn-11.0-linux-x64-v8.0.5.39.tgz
#ADD ./libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb /libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
#RUN dpkg -i /libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb \
# && rm /libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
  && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update && apt install -y \
  ros-melodic-desktop \
  ros-melodic-ros-base \
  python-rosdep \
  python-rosinstall \
  python-rosinstall-generator \
  python-wstool \
  build-essential

# for building psutil
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/_build/target-deps/kit_sdk_release/_build/target-deps/python/lib 
ENV LIBRARY_PATH=$LIBRARY_PATH:/isaac-sim/_build/target-deps/kit_sdk_release/_build/target-deps/python/lib 

ARG USER_ID=local

RUN adduser --disabled-password -u ${USER_ID} --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
ENV HOME=/home/user
RUN chmod 777 /home/user
RUN cp -r /isaac-sim /home/user/isaac-sim/ && chown -R user /home/user/isaac-sim
ENV CONDALINK "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
USER user
RUN curl -so ~/miniconda.sh $CONDALINK \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH

ADD ./isaac_env.yml /home/user/environment.yml

RUN conda install conda-build \
 && conda env create -f /home/user/environment.yml \
 && conda clean -ya && conda init
ENV CONDA_DEFAULT_ENV=env
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

WORKDIR /home/user/isaac-sim/python
RUN echo "conda activate isaac-sim" >> /home/user/.bashrc 
RUN echo "source /home/user/isaac-sim/_build/linux-x86_64/release/setup_python_env.sh" >> /home/user/.bashrc
RUN echo "source /home/user/isaac-sim/python_samples/setenv.sh" >> /home/user/.bashrc

ADD . /home/user/isaac-sim/python
RUN sudo chown -R user /home/user/isaac-sim/python
ADD ./.netrc /home/user/.netrc

RUN sudo apt-get install -y language-pack-en
ENV LANGUAGE=en_US.utf-8
ENV LC_ALL=en_US.utf-8

RUN pip install wandb moviepy elements imageio ruamel.yaml

#ENTRYPOINT /bin/bash
