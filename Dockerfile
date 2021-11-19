# syntax = docker/dockerfile:1.3.0-labs
# The top line is used by BuildKit. DO NOT ERASE IT.
# See the link below for documentation on BuildKit syntax.
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md
# Perhaps the BuildKit dependency is not a good idea since not everyone can use it.
# However, the Dockerfile in the official PyTorch repository also uses BuildKit.

# Do not make changes to the `build` layers unless absolutely necessary.
# If another library needs to be built, add another build layer.
# Users are free to customize the `train` and `deploy` layers as they please.

# All `ARG` variables must be redefined for every stage,
# `ARG`s defined before `FROM` transfer their values to layers that redefine them.
# `ENV` and `LABEL` variables are inherited only by child stages.
# See https://docs.docker.com/engine/reference/builder on how to write Dockerfiles and
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices
# for best practices.

# Style guide: variables specified in the Dockerfile are written as ${ARGUMENT}
# while variables not specified by ARG/ENV are written as $ARGUMENT.

# See https://hub.docker.com/r/nvidia/cuda for all CUDA images.
# Default image is nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04.
# Currently, only Ubuntu images are implemented.
ARG USE_CUDA=1
ARG CUDA_VERSION=11.3.1
ARG CUDNN_VERSION=8
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG DEPLOY_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-${LINUX_DISTRO}${DISTRO_VERSION}

# Build stages exist to build PyTorch and subsidiary libraries.
# They can be easily extended to include builds for other libraries.
# They are not used in the final image, which only copies
# the build outputs from the build stages.
FROM ${BUILD_IMAGE} AS build-base-ubuntu

# Change default settings to allow `apt` cache in Docker image.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
    > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,id=apt-cache-build,target=/var/cache/apt \
    --mount=type=cache,id=apt-lib-build,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# FROM ${BUILD_IMAGE} AS build-base-centos
# FROM ${BUILD_IMAGE} AS build-base-ubi
# To build images based on CentOS or UBI,
# simply implement the install for the
# libraries installed by `apt` in the Ubuntu layer.


FROM build-base-${LINUX_DISTRO} AS build-base

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Conda packages have higher priority than system packages during build.
ENV PATH=/opt/conda/bin:$PATH

# Python won’t try to write .pyc or .pyo files on the import of source modules.
ENV PYTHONDONTWRITEBYTECODE=1
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging.
ENV PYTHONUNBUFFERED=1
# Allows UTF-8 characters as outputs in Docker.
ENV PYTHONIOENCODING=UTF-8

ARG PYTHON_VERSION=3.8
# Conda always uses the specified version of Python, regardless of Miniconda version.
# Use a different conda URL for different architectures. Default is x86_64.
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN curl -fsSL -v -o ~/miniconda.sh -O  ${CONDA_URL} && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

# Include `conda` in dynamic linking. Setting $LD_LIBRARY_PATH directly is bad practice.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache && ccache --max-size 0


# Install everything required for build.
FROM build-base AS build-install

# Magma version must match CUDA version of build image.
ARG MAGMA_VERSION=113

# Maybe fix versions for these libraries. Also maybe sort packages alphabetically.
# Perhaps multiple conda installs are not the best solution but
# Using multiple channels in one install would use older packages.
RUN --mount=type=cache,id=conda-build,target=/opt/conda/pkgs \
    conda install -y \
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
        pkgconfig && \
    conda install -y -c pytorch \
        magma-cuda${MAGMA_VERSION} && \
    conda install -y -c conda-forge \
        libpng \
        libjpeg-turbo

WORKDIR /opt
# Using --jobs 0 gives a reasonable default value for parallel recursion.
RUN git clone --recursive --jobs 0 https://github.com/pytorch/pytorch.git
RUN git clone --recursive --jobs 0 https://github.com/pytorch/vision.git
RUN git clone --recursive --jobs 0 https://github.com/pytorch/text.git
RUN git clone --recursive --jobs 0 https://github.com/pytorch/audio.git


FROM build-install AS build-torch

ARG USE_CUDA
ARG PYTORCH_VERSION_TAG

# See https://developer.nvidia.com/cuda-gpus for the official list of
# CUDA Compute Capability (CC) versions for architectures.
# See https://pytorch.org/docs/stable/cpp_extension.html for an
# explanation of how to specify the `TORCH_CUDA_ARCH_LIST` variable.
# The `+PTX` means that PTX should be built for that CC.
# PyTorch will find the best CC for the host hardware even if
# `TORCH_CUDA_ARCH_LIST` is not given explicitly
# but TorchVision and other subsidiary libraries cannot.
ARG TORCH_CUDA_ARCH_LIST
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Checkout to specific version and update submodules.
WORKDIR /opt/pytorch
RUN if [ -n ${PYTORCH_VERSION_TAG} ]; then \
    git checkout ${PYTORCH_VERSION_TAG} && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0; \
    fi

# Build PyTorch. `USE_CUDA`, `USE_CUDNN`, and `USE_ROCM` are made explicit just in case.
RUN --mount=type=cache,target=/opt/ccache \
    USE_CUDA=${USE_CUDA} USE_CUDNN=${USE_CUDA} USE_ROCM=0 \
    TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py bdist_wheel -d /tmp/dist

# Install PyTorch for subsidiary libraries.
RUN --mount=type=cache,target=/opt/ccache \
    USE_CUDA=${USE_CUDA} USE_CUDNN=${USE_CUDA} USE_ROCM=0 \
    TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install


FROM build-torch AS build-vision

ARG USE_CUDA
ARG TORCHVISION_VERSION_TAG
ARG TORCH_CUDA_ARCH_LIST
# Build TorchVision from source to satisfy PyTorch versioning requirements.
# Setting `FORCE_CUDA=1` creates bizarre errors unless CCs are specified explicitly.
# Fix this issue later if necessary by getting output from `torch.cuda.get_arch_list()`.
# Note that the `FORCE_CUDA` flag may be changed to `USE_CUDA` in later versions.
WORKDIR /opt/vision
RUN if [ -n ${TORCHVISION_VERSION_TAG} ]; then \
    git checkout ${TORCHVISION_VERSION_TAG} && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0; \
    fi

RUN --mount=type=cache,target=/opt/ccache \
    FORCE_CUDA=${USE_CUDA} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    python setup.py bdist_wheel -d /tmp/dist


FROM build-torch AS build-text

ARG TORCHTEXT_VERSION_TAG

WORKDIR /opt/text
RUN if [ -n ${TORCHTEXT_VERSION_TAG} ]; then \
    git checkout ${TORCHTEXT_VERSION_TAG} && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0; \
    fi

# TorchText does not use CUDA.
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist


FROM build-torch AS build-audio

ARG USE_CUDA
ARG TORCHAUDIO_VERSION_TAG
ARG TORCH_CUDA_ARCH_LIST

WORKDIR /opt/audio
RUN if [ -n ${TORCHAUDIO_VERSION_TAG} ]; then \
    git checkout ${TORCHAUDIO_VERSION_TAG} && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0; \
    fi

RUN --mount=type=cache,target=/opt/ccache \
    BUILD_SOX=1 USE_CUDA=${USE_CUDA} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    python setup.py bdist_wheel -d /tmp/dist


FROM ${BUILD_IMAGE} AS train-builds
# A convenience layer to gather PyTorch and subsidiary builds required for training.
# If other source builds are included later on, gather them here as well.
# The train layer should not have any dependencies other than this layer.

COPY --from=build-install /opt/conda /opt/conda
COPY --from=build-vision /tmp/dist /tmp/dist
COPY --from=build-text /tmp/dist /tmp/dist
COPY --from=build-audio /tmp/dist /tmp/dist


FROM ${TRAIN_IMAGE} AS train
######### *Customize for your use case by editing from here* #########
# The `train` image is the one actually used for training.
# It is designed to be separate from the `build` image,
# with only the build artifacts (e.g., pip wheels) copied over.
LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Set as `ARG` values to reduce the image footprint but not affect resulting images.
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Change default settings to allow `apt` cache in Docker image.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
    > /etc/apt/apt.conf.d/keep-cache

# `tzdata` requires a timezone and noninteractive mode.
ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

# Speedups in `apt` and `pip` installs for Korean users. Change URLs for other locations.
# http://archive.ubuntu.com/ubuntu is specific to nvidia/cuda CUDA Ubuntu images.
# Check `/etc/apt/sources.list` of your base image to find your Ubuntu URL.
# Download optimization is located here but not in the install image for 2 reasons.
# 1. Installation images should be modular and should not be affected by the timezone.
# 2. Installation is very short compared to build but a speedup is desirable if a build is already cached.
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
# Remove any pre-existing global `pip` configurations.
RUN if [ $TZ = Asia/Seoul ]; then \
    sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" \
    > /etc/pip.conf; \
    fi

RUN --mount=type=cache,id=apt-cache-train,target=/var/cache/apt \
    --mount=type=cache,id=apt-lib-train,target=/var/lib/apt \
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
# Create user with home directory and password-free sudo permissions.
# This may cause security issues. Use at your own risk.
RUN groupadd -g ${GID} ${GRP} && \
    useradd --shell /bin/bash --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${GRP} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -aG sudo ${USR}

USER ${USR}

# Enable colors on the bash terminal. This is a personal preference.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' $HOME/.bashrc
COPY --from=train-builds --chown=${UID}:${GID} /opt/conda /opt/conda

# Paths created by `--mount` are owned by root unless created beforehand.
# Expects home directory to be in the default location.
ENV PIP_DOWNLOAD_CACHE=/home/${USR}/.cache/pip
WORKDIR ${PIP_DOWNLOAD_CACHE}

# `PROJECT_ROOT` is where the project code will reside.
ARG PROJECT_ROOT=/opt/project

# `PROJECT_ROOT` belongs to `USR` if created after `USER` has been set.
# Not so for pre-existing directories, which will still belong to root.
WORKDIR ${PROJECT_ROOT}

# Path order conveys precedence.
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}

RUN conda config --set pip_interop_enabled True

# Install numpy from conda to use MKL.
RUN conda install -y \
        numpy==1.20.3 && \
    conda clean -ya

# Not using a `requirements.txt` file by design as this would create an external dependency.
# Also, the file would not be a true requirements file because of the source builds and conda installs.
# Preserving pip cache by omitting `--no-cache-dir`.
RUN --mount=type=cache,id=pip-train,target=${PIP_DOWNLOAD_CACHE} \
    --mount=type=bind,from=train-builds,source=/tmp/dist,target=/tmp/dist \
    python -m pip install \
        /tmp/dist/*.whl \
        torch_tb_profiler==0.2.1 \
        jupyterlab==3.2.0 \
        hydra-core==1.1.0 \
        hydra_colorlog==1.1.0 \
        accelerate==0.5.1 \
        pytorch-lightning==1.5.2 \
        seaborn==0.11.1 \
        pandas==1.3.1 \
        openpyxl==3.0.9 \
        scikit-learn==1.0 \
        wandb==0.12.4

CMD ["/bin/bash"]


# Create a deployment image as necessary.
# FROM ${DEPLOY_IMAGE} AS deploy
