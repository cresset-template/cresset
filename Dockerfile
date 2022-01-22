# syntax = docker/dockerfile:1.3
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.
# See the link below for documentation on BuildKit syntax.
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md
# Perhaps the BuildKit dependency is not a good idea since not everyone can use it.
# However, the Dockerfile in the official PyTorch repository also uses BuildKit.

# This image uses multi-stage builds. See the link below for a detailed description.
# https://docs.docker.com/develop/develop-images/multistage-build

# All `ARG` variables must be redefined for every stage,
# `ARG`s defined before `FROM` transfer their values to layers that redefine them.
# `ENV` and `LABEL` variables are inherited only by child stages.
# See https://docs.docker.com/engine/reference/builder on how to write Dockerfiles and
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices
# for best practices.

# Style guide: variables specified in the Dockerfile are written as ${ARGUMENT}
# while variables not specified by ARG/ENV are written as $ARGUMENT.

# See https://pytorch.org/docs/stable/cpp_extension.html for an
# explanation of how to specify the `TORCH_CUDA_ARCH_LIST` variable.

# See https://hub.docker.com/r/nvidia/cuda for all CUDA images.
# Default image is nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04.
# Magma version must match the CUDA version of the build image.
ARG USE_CUDA=1
ARG CONDA_NO_DEFAULTS=0
ARG MKL_MODE=include

ARG CUDA_VERSION=11.3.1
ARG MAGMA_VERSION=113
ARG CUDNN_VERSION=8
ARG PYTHON_VERSION=3.8
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
ARG TORCH_CUDA_ARCH_LIST
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG DEPLOY_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-${LINUX_DISTRO}${DISTRO_VERSION}

# Build related packages are pre-installed on `devel` images.
# Only the `cURL` package is downloaded from the package manager.
# Only the Ubuntu image has been tested.
FROM ${BUILD_IMAGE} AS build-install-ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

FROM ${BUILD_IMAGE} AS build-install-centos
RUN yum -y install curl && yum -y clean all  && rm -rf /var/cache

FROM ${BUILD_IMAGE} AS build-install-ubi
RUN yum -y install curl && yum -y clean all  && rm -rf /var/cache

FROM build-install-${LINUX_DISTRO} AS build-install

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Python won’t try to write .pyc or .pyo files on the import of source modules.
ENV PYTHONDONTWRITEBYTECODE=1
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging.
ENV PYTHONUNBUFFERED=1
# Allows UTF-8 characters as outputs in Docker.
ENV PYTHONIOENCODING=UTF-8

# Conda always uses the specified version of Python, regardless of Miniconda version.
# Use a different conda URL for a different CPU architecture or specific version.
# The default CPU architecture is x86_64.
# The Anaconda `defaults` channel is no longer free for commercial use.
# Anaconda (including Miniconda) itself is still open-source.
# Setting `defaults` to lowest priority as a result. Viva la Open Source!
# https://conda.io/en/latest/license.html
# https://www.anaconda.com/terms-of-service
# https://www.anaconda.com/end-user-license-agreement-miniconda

ARG PYTHON_VERSION
# Conda packages have higher priority than system packages during build.
ENV PATH=/opt/conda/bin:$PATH
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Channel priority: `conda-forge`, `pytorch`, `defaults`.
# Cannot set strict priority because of installation conflicts.
RUN curl -fsSL -v -o ~/miniconda.sh -O ${CONDA_URL} && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda config --add channels conda-forge && \
    conda config --append channels pytorch && \
    conda config --append channels defaults && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

COPY reqs/conda-build.requirements.txt /tmp/reqs/conda-build-requirements.txt


FROM build-install AS build-install-include-mkl

# Conda packages are preferable to system packages because they
# are much more likely to be the latest (and the greatest!) packages.
# Intel channel set to highest priority for optimal Intel builds.
ARG MAGMA_VERSION
RUN conda install -y -c intel \
        mkl \
        mkl-include \
        magma-cuda${MAGMA_VERSION}  \
        --file /tmp/reqs/conda-build-requirements.txt && \
    conda clean -ya


FROM build-install AS build-install-exclude-mkl

# The Intel(R) Math Kernel Library (MKL) places some restrictions on its use, though there are no
# restrictions on commercial use. See the Intel(R) Simplified Software License (ISSL) for details.
# Other Intel software such as the Intel OpenMP^* Runtime Library (iomp) are licensed under the
# Intel End User License Agreement for Developer Tools. See URL below for Intel licenses & EULAs.
# https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html
# Also, non-Intel CPUs may face slowdowns if MKL or other Intel tools are used as the backend.
ARG MAGMA_VERSION
RUN conda install -y \
        nomkl \
        magma-cuda${MAGMA_VERSION}  \
        --file /tmp/reqs/conda-build-requirements.txt && \
    conda clean -ya


FROM build-install-${MKL_MODE}-mkl AS build-base
# `build-base` is the basis for all builds in the Dockerfile.

# Set Jemalloc as the system memory allocator for faster and more efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD
# Anaconda build of Jemalloc does not have profiling enabled.
#ENV MALLOC_CONF="prof:true,lg_prof_sample:1,prof_accum:false,prof_prefix:jeprof.out"

WORKDIR /opt/ccache
ENV PATH=/opt/conda/bin/ccache:$PATH
# Enable `ccache` with unlimited memory size for faster builds.
RUN ccache --set-config=cache_dir=/opt/ccache && ccache --max-size 0

# Use LLD as the default linker for faster linking.
RUN ln -sf /opt/conda/bin/ld.lld /usr/bin/ld

# Include `conda` in dynamic linking.
# Use `ldconfig` to update link directories.
# Setting $LD_LIBRARY_PATH directly is bad practice.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig


FROM build-base AS build-torch

# Checkout to specific version and update submodules.
WORKDIR /opt/pytorch
ARG PYTORCH_VERSION_TAG
RUN git clone --recursive --jobs 0 https://github.com/pytorch/pytorch.git /opt/pytorch && \
    if [ -n ${PYTORCH_VERSION_TAG} ]; then \
        git checkout ${PYTORCH_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

# PyTorch itself can find the host GPU architecture
# on its own but its subsidiary libraries cannot,
# hence the need to specify the architecture list explicitly.
# Building PyTorch with several optimizations and bugfixes.
# `USE_CUDA`, `USE_CUDNN`, and `USE_ROCM` are made explicit just in case.
# Test builds are disabled by default to speed up the build time.
# Disabling Caffe2 is dangerous but most users do not need it.
# Read `setup.py` and `CMakeLists.txt` to find build flags appropriate for your needs.
ARG USE_CUDA
ARG USE_CUDNN=${USE_CUDA}
ARG TORCH_CUDA_ARCH_LIST
ARG BUILD_TEST=0
ARG USE_PRECOMPILED_HEADERS=1
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ARG BUILD_CAFFE2=0

RUN --mount=type=cache,target=/opt/ccache \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py bdist_wheel -d /tmp/dist

# Install PyTorch for subsidiary libraries.
RUN --mount=type=cache,target=/opt/ccache \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install


FROM build-torch AS build-vision

WORKDIR /opt/vision
ARG TORCHVISION_VERSION_TAG
RUN git clone --recursive --jobs 0 https://github.com/pytorch/vision.git /opt/vision && \
    if [ -n ${TORCHVISION_VERSION_TAG} ]; then \
        git checkout ${TORCHVISION_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

ARG USE_CUDA
ARG FORCE_CUDA=${USE_CUDA}
ARG TORCH_CUDA_ARCH_LIST
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist


FROM build-torch AS build-text

WORKDIR /opt/text
ARG TORCHTEXT_VERSION_TAG
RUN git clone --recursive --jobs 0 https://github.com/pytorch/text.git /opt/text && \
    if [ -n ${TORCHTEXT_VERSION_TAG} ]; then \
        git checkout ${TORCHTEXT_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

# TorchText does not use CUDA.
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist


FROM build-torch AS build-audio

WORKDIR /opt/audio
ARG TORCHAUDIO_VERSION_TAG
RUN git clone --recursive --jobs 0 https://github.com/pytorch/audio.git /opt/audio && \
    if [ -n ${TORCHAUDIO_VERSION_TAG} ]; then \
        git checkout ${TORCHAUDIO_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

ARG USE_CUDA
ARG BUILD_SOX=1
ARG TORCH_CUDA_ARCH_LIST
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist


FROM build-base AS build-pure

# Z-Shell related libraries.
RUN git clone https://github.com/sindresorhus/pure.git /opt/pure
RUN git clone https://github.com/zsh-users/zsh-autosuggestions /opt/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /opt/zsh-syntax-highlighting


FROM ${BUILD_IMAGE} AS train-builds
# A convenience stage to gather build artifacts (wheels, etc.) for the train stage.
# If other source builds are included later on, gather them here as well.
# The train stage should not have any dependencies other than this stage.
# This stage does not have anything installed. No variables are specified either.
# This stage is simply the `BUILD_IMAGE` with additional files and directories.
# All pip wheels are located in `/tmp/dist`.

# The `train` image is the one actually used for training.
# It is designed to be separate from the `build` image,
# with only the build artifacts (e.g., pip wheels) copied over.

# The order of `COPY` instructions is chosen to minimize cache misses.
COPY --from=build-install /opt/conda /opt/conda

COPY --from=build-vision  /tmp/dist  /tmp/dist
COPY --from=build-audio   /tmp/dist  /tmp/dist
COPY --from=build-text    /tmp/dist  /tmp/dist

COPY --from=build-pure /opt/pure /opt/pure
COPY --from=build-pure /opt/zsh-autosuggestions /opt/zsh-autosuggestions
COPY --from=build-pure /opt/zsh-syntax-highlighting /opt/zsh-syntax-highlighting

# Copying requirements files from context so that the `train` image
# can be built from this stage with no dependency on the Docker context.
COPY reqs/apt-train.requirements.txt /tmp/reqs/apt-train.requirements.txt
COPY reqs/pip-train.requirements.txt /tmp/reqs/pip-train.requirements.txt

FROM ${TRAIN_IMAGE} AS train
######### *Customize for your use case by editing from here* #########

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Set as `ARG`s to reduce image footprint but not affect the resulting images.
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Speedups in `apt` and `pip` installs for Korean users. Change URLs for other locations.
# http://archive.ubuntu.com/ubuntu is specific to NVIDIA's CUDA Ubuntu images.
# Check `/etc/apt/sources.list` of your base image to find your Ubuntu URL.
# Use `apt` and `pip` mirror links optimized for your location.
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
# `printf` is preferred over `echo` when escape characters are used
# because the behavior of `echo` is inconsistent across shells.
RUN sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" \
        > /etc/pip.conf

# `tzdata` requires a timezone and noninteractive mode.
ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
# The `readwrite` option is necessary because `apt` writes to `/tmp`.
# Requirements for `apt` should be in `reqs/apt-train.requirements.txt`.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
# Essential packages are installed explicitly.
RUN --mount=type=bind,from=train-builds,readwrite,source=/tmp,target=/tmp \
    apt-get update && sed 's/#.*//' /tmp/reqs/apt-train.requirements.txt  \
        | tr [:cntrl:] ' '  \
        | xargs -r apt-get install -y --no-install-recommends && \
    apt-get install -y --no-install-recommends --fix-broken \
        openssh-server \
        sudo \
        tzdata \
        zsh && \
    rm -rf /var/lib/apt/lists/*

# Include `conda` in dynamic linking.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
# The `zsh` shell will be used due to its convenience and popularity.
# Create user with home directory and password-free sudo permissions.
# This may cause security issues. Use at your own risk.
RUN groupadd -g ${GID} ${GRP} && \
    useradd --shell /bin/zsh --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USR}
# Docker must use absolute paths in `COPY` but cannot find `$HOME`.
# Setting $HOME to its default value explicitly as a fix.
ARG HOME=/home/${USR}

# Get conda with the directory ownership given to the user.
COPY --from=train-builds --chown=${UID}:${GID} /opt/conda /opt/conda

# `PROJECT_ROOT` is where the project code will reside.
ARG PROJECT_ROOT=/opt/project
# Path order conveys precedence.
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}

WORKDIR $HOME/.zsh

# Setting the prompt to `pure`, which is available on all terminals without additional settings.
# This is a personal preference and users may use any prompt that they wish (e.g., oh-my-zsh).
COPY --from=train-builds --chown=${UID}:${GID} /opt/pure $HOME/.zsh/pure
RUN printf "fpath+=$HOME/.zsh/pure\nautoload -Uz promptinit; promptinit\nprompt pure\n" >> $HOME/.zshrc

## Add autosuggestions from terminal history. May be somewhat distracting.
#COPY --from=train-builds --chown=${UID}:${GID} /opt/zsh-autosuggestions $HOME/.zsh/zsh-autosuggestions
#RUN echo "source $HOME/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" >> $HOME/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
COPY --from=train-builds --chown=${UID}:${GID} /opt/zsh-syntax-highlighting $HOME/.zsh/zsh-syntax-highlighting
RUN echo "source $HOME/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> $HOME/.zshrc

# Using the Intel channel to get Intel-optimized Python and Numpy.
# Conda configurations must be restated for the user.
RUN conda config --set pip_interop_enabled True && \
    conda config --add channels conda-forge && \
    conda install -y -c intel \
        numpy \
        jemalloc \
        libjpeg-turbo \
        libpng && \
    conda clean -ya

# Use Intel OpenMP.
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD
# Use Jemalloc for faster and more efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD

# The `/tmp/dist/*.whl` files are the wheels built in previous stages.
# `--find-links` gives higher priority to the wheels in `/tmp/dist`, just in case.
RUN --mount=type=bind,from=train-builds,source=/tmp,target=/tmp \
    python -m pip install --no-cache-dir --find-links /tmp/dist/ \
        -r /tmp/reqs/pip-train.requirements.txt \
        /tmp/dist/*.whl && \
    sudo ldconfig

# `PROJECT_ROOT` belongs to `USR` if created after `USER` has been set.
# Not so for pre-existing directories, which will still belong to root.
WORKDIR ${PROJECT_ROOT}

CMD ["/bin/zsh"]


# Minimalist deployment preparation layer.
FROM ${BUILD_IMAGE} AS deploy-builds

# The licenses for the Anaconda default channel and Intel MKL are not fully open-source.
# Enterprise users may therefore wish to remove them from their final product.
# The deployment therefore uses system Python. Conda is copied here just in case.
# Intel packages such as MKL can be removed by using MKL_MODE=exclude during the build.

COPY --from=build-install /opt/conda /opt/conda
COPY --from=build-vision  /tmp/dist  /tmp/dist

COPY reqs/apt-deploy.requirements.txt /tmp/reqs/apt-deploy.requirements.txt
COPY reqs/pip-deploy.requirements.txt /tmp/reqs/pip-deploy.requirements.txt

# Minimalist deployment Ubuntu image.
FROM ${DEPLOY_IMAGE} AS deploy

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Use mirror links optimized for your location and security level.
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
RUN sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" \
        > /etc/pip.conf

# Replace the `--mount=...` instructions with `COPY` if BuildKit is unavailable.
# The `readwrite` option is necessary because `apt` needs write permissions on `\tmp`.
ARG PYTHON_VERSION
RUN --mount=type=bind,from=deploy-builds,readwrite,source=/tmp,target=/tmp \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && sed 's/#.*//' /tmp/reqs/apt-deploy.requirements.txt \
        | tr [:cntrl:] ' '  \
        | xargs -r apt-get install -y --no-install-recommends && \
    apt-get install -y --no-install-recommends --fix-broken \
        python${PYTHON_VERSION} \
        python3-pip \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 3 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python2 2

# The `mkl` package must be installed for PyTorch to use MKL outside `conda`.
# The MKL major version used at runtime must match the version used to build PyTorch.
# The `ldconfig` command is necessary for PyTorch to find MKL and other libraries.
RUN --mount=type=bind,from=deploy-builds,source=/tmp,target=/tmp \
    python -m pip install --no-cache-dir --find-links /tmp/dist/ \
        -r /tmp/reqs/pip-deploy.requirements.txt \
        /tmp/dist/*.whl && \
    ldconfig

WORKDIR /
