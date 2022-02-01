# syntax = docker/dockerfile:1
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

# See the link below for documentation on BuildKit syntax.
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md
# Perhaps the BuildKit dependency is not a good idea since not everyone can use it.
# However, the Dockerfile in the official PyTorch repository also uses BuildKit.

# This image uses multi-stage builds. See the link below for a detailed description.
# https://docs.docker.com/develop/develop-images/multistage-build
# The build process is similar to a neural network with complicated control flow.
# The `stages` are like blocks in the network, with control flow implemented
# by the names of arguments and stages. This allows muh greater modularity in the build.
# It also makes the final image much more compact and efficient.

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

# cuDNN minor version is tied to CUDA minor version.
# See documentation below for available versions.
# https://developer.nvidia.com/rdp/cudnn-archive

ARG DEBUG=0
ARG USE_CUDA=1
ARG USE_ROCM=0
ARG CONDA_NO_DEFAULTS=0
ARG USE_PRECOMPILED_HEADERS=1
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


# Build related packages are pre-installed on CUDA `devel` images.
# Only the `cURL` package is downloaded from the package manager.
# The only use of cURL is to download Miniconda.
# Only the Ubuntu image has been tested.
# The `train` and `deploy` stages are designed for Ubuntu.
FROM ${BUILD_IMAGE} AS build-install-ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*


FROM ${BUILD_IMAGE} AS build-install-centos
RUN yum -y install curl && yum -y clean all  && rm -rf /var/cache


FROM ${BUILD_IMAGE} AS build-install-ubi
RUN yum -y install curl && yum -y clean all  && rm -rf /var/cache


FROM build-install-${LINUX_DISTRO} AS build-install

LABEL maintainer="veritas9872@gmail.com"
LABEL com.nvidia.volumes.needed="nvidia_driver"
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
ARG CONDA_NO_DEFAULTS
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
    if [ ${CONDA_NO_DEFAULTS} != 0 ]; then \
        conda config --remove channels defaults; \
    fi && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya


FROM build-install AS build-install-conda

# Get build requirements. Set package versions manually if compatibility issues arise.
COPY reqs/conda-build.requirements.txt /tmp/reqs/conda-build.requirements.txt

# Comment out the lines below if Mamba causes any issues.
RUN conda install -y -c conda-forge mamba && conda clean -ya
# Using Mamba instead of Conda as the package manager for faster installation.
#ENV conda=/opt/conda/bin/conda
ENV conda=/opt/conda/bin/mamba


FROM build-install-conda AS build-install-include-mkl

# Roundabout method to enable MKLDNN in PyTorch build when MKL is included.
ENV USE_MKLDNN=1

# Conda packages are preferable to system packages because they
# are much more likely to be the latest (and the greatest!) packages.
# Use fixed version numbers if versioning issues cause build failures.
# Intel channel set to highest priority for optimal Intel builds.
ARG MAGMA_VERSION
RUN $conda install -y -c intel \
        --file /tmp/reqs/conda-build.requirements.txt \
        magma-cuda${MAGMA_VERSION} \
        mkl \
        mkl-include && \
    conda clean -ya

# Use Intel OpenMP with optimizations enabled.
# Some compilers can use OpenMP for faster builds.
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD
ENV KMP_AFFINITY="granularity=fine,compact,1,0"
ENV KMP_BLOCKTIME=0


FROM build-install-conda AS build-install-exclude-mkl

# Roundabout method to disable MKLDNN in PyTorch build when MKL is excluded.
ENV USE_MKLDNN=0

# The Intel(R) Math Kernel Library (MKL) places some restrictions on its use, though there are no
# restrictions on commercial use. See the Intel(R) Simplified Software License (ISSL) for details.
# Other Intel software such as the Intel OpenMP^* Runtime Library (iomp) are licensed under the
# Intel End User License Agreement for Developer Tools. See URL below for Intel licenses & EULAs.
# https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html
# Also, non-Intel CPUs may face slowdowns if MKL or other Intel tools are used in the backend.
ARG MAGMA_VERSION
RUN $conda install -y \
        --file /tmp/reqs/conda-build.requirements.txt \
        magma-cuda${MAGMA_VERSION} \
        nomkl && \
    conda clean -ya


FROM build-install-${MKL_MODE}-mkl AS build-base
# `build-base` is the base stage for all builds in the Dockerfile.

# Use Jemalloc as the system memory allocator for faster and more efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD
# Anaconda build of Jemalloc does not have profiling enabled.
#ENV MALLOC_CONF="prof:true,lg_prof_sample:1,prof_accum:false,prof_prefix:jeprof.out"

# The Docker Daemon cache memory may be insufficient to hold the entire cache.
# A small Garbage Collection (GC) `defaultKeepStorage` value may slow builds
# by removing the caches of previous builds, forcing the compiler to recompile.
# The default GC size is often smaller than the compiler cache size of PyTorch.
# To configure GC settings, edit the Docker Daemon configuration JSON file.
# This is available in Settings -> Docker Engine on Docker Desktop for Windows.
# https://docs.docker.com/engine/reference/commandline/dockerd/#daemon-configuration-file
# https://github.com/docker/cli/issues/2325
WORKDIR /opt/ccache
ENV PATH=/opt/conda/bin/ccache:$PATH
# Enable `ccache` with unlimited memory size for faster builds.
RUN ccache --set-config=cache_dir=/opt/ccache && ccache --max-size 0

# Use LLD as the default linker for faster linking.
RUN ln -sf /opt/conda/bin/ld.lld /usr/bin/ld

# Include `conda` in dynamic linking.
# Use `ldconfig` to update link directories.
# Setting `LD_LIBRARY_PATH` directly is bad practice.
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
# Test builds are disabled by default to speed up the build time.
# Disabling Caffe2 is dangerous but most users do not need it.
# NNPack and QNNPack are also unnecessary for most users.
# `USE_MKLDNN` is restated to remind users that it has been set.
# Read `setup.py` and `CMakeLists.txt` to find build flags.
# Different flags are available for different versions of PyTorch.
# Variables without default values here recieve defaults from the top of the Dockerfile.
ARG DEBUG
ARG USE_CUDA
ARG USE_CUDNN=${USE_CUDA}
ARG USE_MKLDNN=${USE_MKLDNN}
ARG USE_ROCM
ARG USE_NNPACK=0
ARG USE_QNNPACK=0
ARG BUILD_TEST=0
ARG BUILD_CAFFE2=0
ARG USE_PRECOMPILED_HEADERS
ARG TORCH_CUDA_ARCH_LIST
ARG CMAKE_PREFIX_PATH=/opt/conda
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# Build wheel for installation in later stages.
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist
# Install PyTorch for subsidiary libraries.
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py install

###### Additional information for custom builds. ######

# Manually edit conda package versions if older PyTorch versions will not build.

# Use this to build with custom CMake settings.
#RUN --mount=type=cache,target=/opt/ccache \
#    python setup.py build --cmake-only && \
#    ccmake build  # or cmake-gui build

# Run the command below before building to enable ROCM builds.
# RUN python tools/amd_build/build_amd.py
# PyTorch builds with ROCM has not been tested.
# Note that PyTorch for ROCM is still in beta and
# the API for enabling ROCM builds may change.

# To build for Jetson Nano devices, see the link below for the necessary modifications.
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048


FROM build-torch AS build-vision

WORKDIR /opt/vision
ARG TORCHVISION_VERSION_TAG
RUN git clone --recursive --jobs 0 https://github.com/pytorch/vision.git /opt/vision && \
    if [ -n ${TORCHVISION_VERSION_TAG} ]; then \
        git checkout ${TORCHVISION_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

# Install Pillow SIMD before TorchVision build and add it to `/tmp/dist`.
# This may not work on older CPUs as it requires SSE4 and AVX2.
# Pillow will be uninstalled if it is present.
RUN python -m pip uninstall -y pillow && \
    CC="cc -mavx2" python -m pip install -U --force-reinstall --no-deps Pillow-SIMD && \
    CC="cc -mavx2" python -m pip wheel Pillow-SIMD --wheel-dir /tmp/dist

ARG DEBUG
ARG USE_CUDA
ARG USE_FFMPEG=1
ARG USE_PRECOMPILED_HEADERS
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
ARG USE_PRECOMPILED_HEADERS
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
ARG USE_ROCM
ARG USE_PRECOMPILED_HEADERS
ARG BUILD_TORCHAUDIO_PYTHON_EXTENSION=1
ARG BUILD_FFMPEG=1
ARG TORCH_CUDA_ARCH_LIST
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist


FROM build-base AS build-pure

# Z-Shell related libraries.
RUN git clone https://github.com/sindresorhus/pure.git /opt/pure
RUN git clone https://github.com/zsh-users/zsh-autosuggestions /opt/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /opt/zsh-syntax-highlighting


FROM build-base AS train-builds
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
# `conda` in `build-install-conda` has `mamba` installed, unlike `conda` in `build-install`.
COPY --from=build-install-conda /opt/conda /opt/conda

COPY --from=build-vision /tmp/dist /tmp/dist
COPY --from=build-audio  /tmp/dist /tmp/dist
COPY --from=build-text   /tmp/dist /tmp/dist

# `COPY` new builds here to minimize the likelihood of cache misses.

COPY --from=build-pure /opt/pure /opt/pure
COPY --from=build-pure /opt/zsh-autosuggestions /opt/zsh-autosuggestions
COPY --from=build-pure /opt/zsh-syntax-highlighting /opt/zsh-syntax-highlighting

# Copying requirements files from context so that the `train` image
# can be built from this stage with no dependency on the Docker context.
# The files are placed in different directories to allow changing one file
# without affecting the bind mount directory of the other files.
# If all files were placed in the same directory, changing just one file
# would cause a cache miss, forcing all requirements to reinstall.
COPY reqs/apt-train.requirements.txt   /tmp/reqs/apt/requirements.txt
COPY reqs/conda-train.requirements.txt /tmp/reqs/conda/requirements.txt
COPY reqs/pip-train.requirements.txt   /tmp/reqs/pip/requirements.txt


FROM ${TRAIN_IMAGE} AS train

LABEL maintainer="veritas9872@gmail.com"
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Set as `ARG`s to reduce image footprint but not affect the resulting images.
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Speedups in `apt` and `pip` installs for Korean users. Change URLs for other locations.
# http://archive.ubuntu.com/ubuntu is specific to NVIDIA's CUDA Ubuntu images.
# Check `/etc/apt/sources.list` of the base image to find the Ubuntu URL.
# Use `apt` and `pip` mirror links optimized for user location.
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
# `printf` is preferred over `echo` when escape characters are used
# because the behavior of `echo` is inconsistent across shells.
RUN sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" > /etc/pip.conf

# `tzdata` requires a timezone and noninteractive mode.
ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
# The `readwrite` option is necessary because `apt` writes to `/tmp`.
# Requirements for `apt` should be in `reqs/apt-train.requirements.txt`.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
# Using `sed` and `xargs` to imitate the behavior of a requirements file.
RUN --mount=type=bind,from=train-builds,source=/tmp/reqs/apt,target=/tmp/reqs/apt \
    apt-get update &&  \
    sed 's/#.*//g; s/\r//g' /tmp/reqs/apt/requirements.txt | \
    xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
# The `zsh` shell is used due to its convenience and popularity.
# Creating user with home directory and password-free sudo permissions.
# This may cause security issues.
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

# Using the Intel channel to get Intel-optimized numpy.
# Using `mamba` instead of `conda` for faster installation.
# Source wheels are installed to get dependencies from `conda` instead of `pip`.
# Mixing `conda` and `pip` is not recommended but `conda` should come first if this is unavoidable.
# https://www.anaconda.com/blog/using-pip-in-a-conda-environment
ARG conda=/opt/conda/bin/mamba
RUN --mount=type=bind,from=train-builds,source=/tmp/dist,target=/tmp/dist \
    --mount=type=bind,from=train-builds,source=/tmp/reqs/conda,target=/tmp/reqs/conda \
    conda config --set pip_interop_enabled True && \
    conda config --add channels conda-forge && \
    python -m pip install --no-cache-dir --no-deps /tmp/dist/*.whl && \
    $conda install -y -c intel --file /tmp/reqs/conda/requirements.txt && \
    conda config --set pip_interop_enabled False && \
    conda clean -ya

# The `/tmp/dist/*.whl` files are the wheels built in previous stages.
# `--find-links` gives higher priority to the wheels in `/tmp/dist`.
RUN --mount=type=bind,from=train-builds,source=/tmp/dist,target=/tmp/dist \
    --mount=type=bind,from=train-builds,source=/tmp/reqs/pip,target=/tmp/reqs/pip \
    python -m pip install --no-cache-dir --find-links /tmp/dist \
        -r /tmp/reqs/pip/requirements.txt \
        /tmp/dist/*.whl

# Use Intel OpenMP with optimizations. See documentation for details.
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# https://intel.github.io/intel-extension-for-pytorch/tutorials/performance_tuning/tuning_guide.html
# https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/optimization-and-programming-guide/openmp-support/openmp-library-support/thread-affinity-interface-linux-and-windows.html
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD
ENV KMP_AFFINITY="granularity=fine,compact,1,0"
ENV KMP_BLOCKTIME=0
# Use Jemalloc for faster and more efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD

# Temporarily switch to `root` for permissions.
USER root
# Include `conda` in dynamic linking.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig
USER ${USR}

# `PROJECT_ROOT` belongs to `USR` if created after `USER` has been set.
# Not so for pre-existing directories, which will still belong to root.
WORKDIR ${PROJECT_ROOT}

CMD ["/bin/zsh"]


# Minimalist deployment preparation layer.
FROM build-base AS deploy-builds

# If any `pip` packages must be compiled on installation, create a wheel in the
# `build` stages and move it to `/tmp/dist`. Otherwise, the installtion may fail.
# The `deploy` image is a CUDA `runtime` image without compiler tools.

# The licenses for the Anaconda defaults channel and Intel MKL are not fully open-source.
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
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Use mirror links optimized for user location and security level.
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
RUN sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" > /etc/pip.conf

# Replace the `--mount=...` instructions with `COPY` if BuildKit is unavailable.
# The `readwrite` option is necessary because `apt` needs write permissions on `\tmp`.
# Both `python` and `python3` are set to point to the installed version of Python.
# The pre-installed system Python3 may be overridden if the installed and pre-installed
# versions of Python3 are the same (e.g., Python 3.8 on Ubuntu 20.04 LTS).
# Using `sed` and `xargs` to imitate the behavior of a requirements file.
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=deploy-builds,readwrite,source=/tmp,target=/tmp \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    printf "\n python${PYTHON_VERSION} \n" >> /tmp/reqs/apt-deploy.requirements.txt && \
    sed 's/#.*//g; s/\r//g' /tmp/reqs/apt-deploy.requirements.txt |  \
    xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1

# The `mkl` package must be installed for PyTorch to use MKL outside `conda`.
# The MKL major version used at runtime must match the version used to build PyTorch.
# The `ldconfig` command is necessary for PyTorch to find MKL and other libraries.
RUN --mount=type=bind,from=deploy-builds,source=/tmp,target=/tmp \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --find-links /tmp/dist/ \
        -r /tmp/reqs/pip-deploy.requirements.txt \
        /tmp/dist/*.whl && \
    ldconfig
