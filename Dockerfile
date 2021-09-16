# syntax = docker/dockerfile:1.3.0-labs
# The top line is used by Buildkit. DO NOT ERASE IT.
# See the link below for documentation on Buildkit syntax.
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md

# All `ARG` variables must be redefined for every stage.
# `ENV` and `LABEL` variables are inherited only by child stages.

# `ccache` can only be used for a single project with exactly the same settings.
# Do not use `ccache` for any build other than PyTorch.

# See https://hub.docker.com/r/nvidia/cuda for all CUDA images.
# Default image is nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04.
# Note that only Ubuntu images will work because of `apt`.
ARG CUDA_VERSION=11.2.2
ARG CUDNN_VERSION=8
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG DEPLOY_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-${LINUX_DISTRO}${DISTRO_VERSION}


FROM ${BUILD_IMAGE} AS build-base
LABEL maintainer="joonhyung.lee@vuno.co"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# tzdata is a hidden dependecy here.
ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH=/opt/conda/bin:$PATH

ARG PYTHON_VERSION=3.8
# Conda is always the latest version but uses the specified version of Python.
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya


# Install everything required for build.
FROM build-base as build-install

# Magma version must match CUDA version of build image.
ARG MAGMA_VERSION=112

# TODO: Fix versions for these libraries.
RUN conda install -y \
    astunparse \
    numpy \
    ninja \
    pyyaml \
    mkl \
    mkl-include \
    setuptools \
    cmake \
    cffi \
    typing_extensions \
    future \
    six \
    requests \
    pillow \
    libpng

# `magma-cuda` appears to have only one version per CUDA version.
RUN conda install -y -c pytorch magma-cuda${MAGMA_VERSION} && \
    conda install -y -c conda-forge libjpeg-turbo && \
    conda clean -ya

WORKDIR /opt
# Using --jobs 0 gives a reasonable default value for parallel recursion.
RUN git clone --recursive --jobs 0 https://github.com/pytorch/pytorch
RUN git clone https://github.com/pytorch/vision.git
RUN git clone --recursive --jobs 0 https://github.com/pytorch/text


FROM build-install as build-torch

ARG PYTORCH_VERSION_TAG

# See https://developer.nvidia.com/cuda-gpus for the official list of
# CUDA Compute Capability (CC) versions for architectures.
# See https://pytorch.org/docs/stable/cpp_extension.html for an
# explanation of how to specify the `TORCH_CUDA_ARCH_LIST` variable.
# The `+PTX` means that PTX should be built for that CC.
# PyTorch will find the best CC for the host hardware even if
# `TORCH_CUDA_ARCH_LIST` is not given explicitly
# (default is set because of TorchVision).
ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Build PyTorch. `USE_CUDA` and `USE_CUDNN` are made explicit just in case.
WORKDIR /opt/pytorch
RUN if [ -n ${PYTORCH_VERSION_TAG} ]; then git checkout ${PYTORCH_VERSION_TAG}; fi
RUN --mount=type=cache,target=/opt/ccache \
    USE_CUDA=1 USE_CUDNN=1 \
    TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py bdist_wheel -d /tmp/dist

# Install PyTorch because TorchVision requires installed PyTorch.
RUN --mount=type=cache,target=/opt/ccache \
    USE_CUDA=1 USE_CUDNN=1 \
    TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install


FROM build-torch AS build-vision

ARG TORCHVISION_VERSION_TAG
ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
# Build TorchVision from source to satisfy PyTorch versioning requirements.
# Setting `FORCE_CUDA=1` creates bizarre errors unless CCs are specified explicitly.
# Fix this issue later if necessary by getting output from `torch.cuda.get_arch_list()`.
# Also not using `/opt/ccache` to preserve PyTorch cache, which takes far longer.
WORKDIR /opt/vision
RUN if [ -n ${TORCHVISION_VERSION_TAG} ]; then git checkout ${TORCHVISION_VERSION_TAG}; fi
RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FORCE_CUDA=1 \
    python setup.py bdist_wheel -d /tmp/dist

FROM build-torch AS build-text

ARG TORCHTEXT_VERSION_TAG

WORKDIR /opt/text
RUN if [ -n ${TORCHTEXT_VERSION_TAG} ]; then git checkout ${TORCHTEXT_VERSION_TAG}; fi
RUN python setup.py bdist_wheel -d /tmp/dist


# This stage exists to gather the outputs of all builds into one place.
FROM build-install AS build-last

COPY --from=build-vision /tmp/dist /tmp/dist
COPY --from=build-text /tmp/dist /tmp/dist

FROM ${TRAIN_IMAGE} as train
LABEL maintainer="joonhyung.lee@vuno.co"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ARG PROJECT_ROOT=/opt/project

ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,id=apt-train,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
    nano \
    tmux \
    openssh-server \
    tzdata && \
    rm -rf /var/lib/apt/lists/*

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu

RUN groupadd -g $GID $GRP && \
    useradd --shell /bin/bash --create-home -u $UID -g $GRP -p $(openssl passwd -1 $PASSWD) $USR && \
    echo "$GRP ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -aG sudo $USR

USER $USR

COPY --from=build-base --chown=$GRP:$USR /opt/conda /opt/conda
COPY --from=build-last --chown=$GRP:$USR /tmp/dist /tmp/dist

# Path order conveys precedence.
ENV PATH=$PROJECT_ROOT:/opt/conda/bin:/usr/local/cuda/bin:$PATH
ENV PYTHONPATH=$PROJECT_ROOT

RUN conda config --set pip_interop_enabled True

# Numpy from conda to use MKL. Specify version later.
RUN conda install -y \
    numpy && \
    conda clean -ya

RUN python -m pip install --no-cache-dir /tmp/dist/*.whl \
    pytorch-lightning==1.4.5 \
    pytorch-pfn-extras==0.4.2 \
    h5py==3.4.0 \
    captum==0.4.0 \
    mlflow==1.20.2 \
    tensorboard==2.5.0 \
    tensorboard-plugin-wit==1.8.0 \
    torch_tb_profiler==0.2.1 \
    hydra-core==1.1.0 \
    hydra_colorlog==1.1.0 \
    fire==0.4.0 \
    openpyxl==3.0.7 \
    cupy-cuda112==9.2.0 \
    SimpleITK==2.1.0 \
    seaborn==0.11.1 \
    albumentations==1.0.3 \
    nibabel==3.2.1 && \
    rm -rf /tmp/dist

# Edit .bashrc file for environment settings.
RUN echo "cd $PROJECT_ROOT" >> ~/.bashrc

# The `$PROJECT_ROOT` directory belongs to `$USR`
# as it is created after `USER` has been set.
WORKDIR $PROJECT_ROOT

CMD ["/bin/bash"]

# NOTE: Add code for the deployment image at some future date.
# If build must occur at the deployment environment on-site but 
# internet access is unavailable there, install everything in
# build-install and use it as a base image to build on-site.

# FROM ${DEPLOY_IMAGE} as deploy
