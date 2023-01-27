# syntax = docker/dockerfile:1.4
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

# Use `export BUILDKIT_PROGRESS=plain` in the host terminal to see full build logs.

# Visit the following URL for available BuildKit syntax versions.
# https://hub.docker.com/r/docker/dockerfile

# See the link below for documentation on BuildKit syntax.
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md

# This image uses multi-stage builds. See the link below for a detailed description.
# https://docs.docker.com/develop/develop-images/multistage-build
# The build process is similar to a neural network with complicated control flow.
# The `stages` are like blocks in the network, with control flow implemented
# by the names of arguments and stages.

# Use `export BUILDKIT_PROGRESS=plain` to see full outputs during debugging.

# All `ARG` variables must be redefined for every stage,
# `ARG`s defined before `FROM` transfer their values to layers that redefine them.
# `ENV` and `LABEL` variables are inherited only by child stages.
# See https://docs.docker.com/engine/reference/builder on how to write Dockerfiles and
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices for best practices.
# See https://hub.docker.com/r/nvidia/cuda for all available CUDA images.

# Note that the CUDA 11+ now uses semantic versioning.
# Until CUDA 10.2, there are only the major and minor version numbers.
# From CUDA 11.0.0 onwards, the major, minor, and patch versions are included.
# A CUDA version such as `11.2` is therefore invalid.
# Users must specify the full version, e.g., `11.2.2`.

ARG MKL_MODE
ARG BUILD_MODE
ARG USE_CUDA=1
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG LINUX_DISTRO
ARG DISTRO_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG USE_PRECOMPILED_HEADERS=1
# Build-related packages are pre-installed on CUDA `devel` images.
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG DEPLOY_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-${LINUX_DISTRO}${DISTRO_VERSION}

########################################################################
FROM curlimages/curl:latest AS curl
# An image used solely to download files from the internet.

# Use a different CONDA_URL for a different CPU architecture or specific version.
# The Anaconda `defaults` channel is no longer free for commercial use.
# Using Miniforge or Mambaforge is strongly recommended. Viva la Open Source!
# Use Miniconda only if absolutely necessary.
# The defaults channel will be removed and the conda-forge channel will be used.
# https://conda.io/en/latest/license.html
# https://www.anaconda.com/terms-of-service
# https://www.anaconda.com/end-user-license-agreement-miniconda

ARG CONDA_URL
RUN mkdir /tmp/conda && curl -fsSL -v -o /tmp/conda/miniconda.sh -O ${CONDA_URL}

########################################################################
FROM ${BUILD_IMAGE} AS install-conda

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Python won’t try to write .pyc or .pyo files on the import of source modules.
ENV PYTHONDONTWRITEBYTECODE=1
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging.
ENV PYTHONUNBUFFERED=1
# Allows UTF-8 characters as outputs in Docker.
ENV PYTHONIOENCODING=UTF-8

# Conda packages have higher priority than system packages during the build.
ENV PATH=/opt/conda/bin:${PATH}

# `CONDA_MANAGER` may be either `mamba` or `conda`.
ARG CONDA_MANAGER
# Shortcut to simplify downstream installation.
ENV conda=/opt/conda/bin/${CONDA_MANAGER}

ARG PYTHON_VERSION
# The `.condarc` file in the installation directory portably configures the
# `conda-forge` channel and removes the `defaults` channel if Miniconda is used.
# No effect if Miniforge or Mambaforge is used as this is the default anyway.
RUN --mount=type=bind,from=curl,source=/tmp/conda,target=/tmp/conda \
    /bin/bash /tmp/conda/miniconda.sh -b -p /opt/conda && \
    printf "channels:\n  - conda-forge\n" > /opt/conda.condarc && \
    $conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

########################################################################
FROM install-conda AS install-mkl-base

# Get build requirements. Set package versions manually if compatibility issues arise.
COPY --link reqs/conda-build.requirements.txt /tmp/conda/build-requirements.txt

########################################################################
FROM install-mkl-base AS install-include-mkl

# Conda packages are preferable to system packages because they
# are much more likely to be the latest (and the greatest!) packages.
# Use fixed version numbers if versioning issues cause build failures.
# `sed 's/\.//; s/\..*//'` extracts `magma` versions from CUDA versions.
# For example, 11.5.1 becomes 115 and 10.2 becomes 102.
# Using the MatchSpec syntax for the magma-cuda package,
# which is only available from the PyTorch channel.
# All other packages should come from the conda-forge channel.
ARG CUDA_VERSION
RUN $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        pytorch::magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
        mkl-include \
        mkl && \
    conda clean -ya

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ENV MKL_DEBUG_CPU_TYPE=5
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c
ENV LD_PRELOAD=/opt/conda/libfakeintel.so:${LD_PRELOAD}

# Use Intel OpenMP with optimizations enabled.
# Some compilers can use OpenMP for faster builds.
ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:${LD_PRELOAD}

########################################################################
FROM install-mkl-base AS install-exclude-mkl

# The Intel(R) Math Kernel Library (MKL) places some restrictions on its use,
# though there are no restrictions on commercial use.
# See the Intel(R) Simplified Software License (ISSL) for details.
# Other Intel software such as the Intel OpenMP^* Runtime Library (iomp)
# are licensed under the Intel End User License Agreement for Developer Tools.
# See URL below for Intel licenses & EULAs.
# https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html
# Also, non-Intel CPUs may face slowdowns if MKL is used in the backend.
ARG CUDA_VERSION
RUN $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        pytorch::magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
        nomkl && \
    conda clean -ya

########################################################################
FROM install-${MKL_MODE}-mkl AS build-base
# `build-base` is the base stage for all heavy builds in the Dockerfile.

# Use Jemalloc as the system memory allocator for efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD
# See the documentation for an explanation of the following configuration.
# https://android.googlesource.com/platform/external/jemalloc_new/+/6e6a93170475c05ebddbaf3f0df6add65ba19f01/TUNING.md
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

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

# Use `ldconfig` to update link directories and include `conda` in dynamic linking.
# Setting `LD_LIBRARY_PATH` directly is bad practice.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

########################################################################
FROM build-base AS build-torch

# Updating git submodules is not fail-safe.
# If the build fails during `git clone`, just try again.
# The reason for failure is likely due to networking issues during installation.
# See https://stackoverflow.com/a/8573310/9289275
WORKDIR /opt/pytorch
ARG PYTORCH_VERSION_TAG
ARG TORCH_URL=https://github.com/pytorch/pytorch.git
# Minimize downloads by only cloning shallow branches and not the full `git` history.
RUN git clone --jobs 0 --depth 1 --single-branch --shallow-submodules \
        --recurse-submodules --branch ${PYTORCH_VERSION_TAG} \
        ${TORCH_URL} /opt/pytorch

# Read `setup.py` and `CMakeLists.txt` to find build flags.
# Different flags are available for different versions of PyTorch.
# Variables without default values here recieve defaults from the top of the Dockerfile.
# Disabling Caffe2, NNPack, and QNNPack as they are legacy and most users do not need them.
ARG USE_CUDA
ARG USE_CUDNN=${USE_CUDA}
ARG USE_NNPACK=0
ARG USE_QNNPACK=0
ARG BUILD_TEST=0
ARG BUILD_CAFFE2=0
ARG BUILD_CAFFE2_OPS=0
ARG USE_PRECOMPILED_HEADERS
ARG TORCH_CUDA_ARCH_LIST
ARG CMAKE_PREFIX_PATH=/opt/conda
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# Build wheel for installation in later stages.
# Install PyTorch for subsidiary libraries (e.g., TorchVision).
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist && \
    python setup.py install

###### Additional information for custom builds. ######

# Use the following to build with custom CMake settings.
#RUN --mount=type=cache,target=/opt/ccache \
#    python setup.py build --cmake-only && \
#    ccmake build  # or cmake-gui build

# See the SetupTools documentation for more setup.py options.
# https://setuptools.pypa.io/en/latest

# C++ developers using Libtoch can find the library in `torch/lib/tmp_install/lib/libtorch.so`.

# The default configuration removes all files except requirements files from the Docker context.
# To `COPY` your source files during the build, please edit the `.dockerignore` file.

# A detailed (if out of date) explanation of the buildsystem can be found below.
# https://pytorch.org/blog/a-tour-of-pytorch-internals-2
# The following repository may also be helpful for available options and possible issues.
# https://github.com/mratsim/Arch-Data-Science/blob/master/frameworks/python-pytorch-magma-mkldnn-cudnn-git/PKGBUILD

# Manually specify conda package versions if older PyTorch versions will not build.
# PyYAML, MKL-DNN, and SetupTools are known culprits.

# Run the command below before building to enable ROCM builds.
# RUN python tools/amd_build/build_amd.py
# PyTorch builds with ROCM has not been tested.
# Note that PyTorch for ROCM is still in beta and the ROCM build API may change.

# To build for Jetson Nano devices, see the link below for the necessary modifications.
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048

########################################################################
FROM install-conda AS build-pillow
# This stage is derived from `install-conda` instead of `build-base`
# as it is very lightweight and does not require many dependencies.
RUN $conda install -y libjpeg-turbo zlib && conda clean -ya

# Specify `Pillow-SIMD` version if necessary. Variable is not used yet.
ARG PILLOW_SIMD_VERSION
# Condition ensures that AVX2 instructions are built only if available.
RUN if [ -n "$(lscpu | grep avx2)" ]; then CC="cc -mavx2"; fi && \
    python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        Pillow-SIMD  # ==${PILLOW_SIMD_VERSION}

########################################################################
FROM build-torch AS build-vision

WORKDIR /opt/vision
ARG TORCHVISION_VERSION_TAG
ARG VISION_URL=https://github.com/pytorch/vision.git
RUN git clone --jobs 0 --depth 1 --single-branch --shallow-submodules \
        --recurse-submodules --branch ${TORCHVISION_VERSION_TAG} \
        ${VISION_URL} /opt/vision

# Install Pillow-SIMD before TorchVision build and add it to `/tmp/dist`.
# Pillow will be uninstalled if it is present.
RUN --mount=type=bind,from=build-pillow,source=/tmp/dist,target=/tmp/dist \
    python -m pip uninstall -y pillow && \
    python -m pip install --force-reinstall --no-deps /tmp/dist/*

ARG USE_CUDA
# Disable FFMPEG and remove it as a build dependency if TorchVision
# fails to compile with unhelpful error messages.
ARG USE_FFMPEG=1
ARG USE_PRECOMPILED_HEADERS
ARG FORCE_CUDA=${USE_CUDA}
ARG TORCH_CUDA_ARCH_LIST
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist

########################################################################
FROM install-conda AS fetch-pure

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

# Get `git` from `conda` to prevent version mismatch issues.
RUN $conda install -y git && conda clean -ya

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

########################################################################
FROM install-conda AS fetch-torch

# For users who wish to download wheels instead of building them.
ARG PYTORCH_INDEX_URL
ARG PYTORCH_VERSION
RUN python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        --index-url ${PYTORCH_INDEX_URL} \
        torch==${PYTORCH_VERSION}

########################################################################
FROM install-conda AS fetch-vision

ARG PYTORCH_INDEX_URL
ARG TORCHVISION_VERSION
RUN python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        --index-url ${PYTORCH_INDEX_URL} \
        torchvision==${TORCHVISION_VERSION}

########################################################################
FROM ${BUILD_IMAGE} AS train-builds-include
# A convenience stage to gather build artifacts (wheels, etc.) for the train stage.
# If other source builds are included later on, gather them here as well.
# All pip wheels are located in `/tmp/dist`.
# Using an image other than `BUILD_IMAGE` may contaminate
# `/opt/conda` and other key directories.

# The `train` image is the one actually used for training.
# It is designed to be separate from the `build` image,
# with only the build artifacts (e.g., pip wheels) copied over.
COPY --link --from=install-conda /opt/conda /opt/conda
COPY --link --from=build-pillow  /tmp/dist  /tmp/dist
COPY --link --from=build-vision  /tmp/dist  /tmp/dist
COPY --link --from=fetch-pure    /opt/zsh   /opt/zsh

########################################################################
FROM ${BUILD_IMAGE} AS train-builds-exclude
# No compiled libraries copied over in exclude mode except Pillow-SIMD.
# Note that `fetch` stages are derived from the `install-conda` stage
# with no dependency on the `build-base` stage. This skips installation
# of any build-time dependencies, saving both time and space.

COPY --link --from=install-conda /opt/conda /opt/conda
COPY --link --from=build-pillow  /tmp/dist  /tmp/dist
COPY --link --from=fetch-torch   /tmp/dist  /tmp/dist
COPY --link --from=fetch-vision  /tmp/dist  /tmp/dist
COPY --link --from=fetch-pure    /opt/zsh   /opt/zsh

########################################################################
FROM train-builds-${BUILD_MODE} AS train-builds
# Gather Python packages built in previous stages and
# install into a conda virtual environment using pip.
# Using a separate stage allows for build modularity
# and and parallel installation with system packages.

# Add a mirror `INDEX_URL` for PyPI via `PIP_CONFIG_FILE` if specified.
ARG INDEX_URL
ARG TRUSTED_HOST
ARG PIP_CONFIG_FILE=/opt/conda/pip.conf
RUN if [ ${INDEX_URL} ]; then \
    {   echo "[global]"; \
        echo "index-url=${INDEX_URL}"; \
        echo "trusted-host=${TRUSTED_HOST}"; \
    } > ${PIP_CONFIG_FILE}; \
    fi

ARG PATH=/opt/conda/bin:${PATH}

# `CONDA_MANAGER` should be either `mamba` or `conda`.
# See the `install-conda` stage above for details.
ARG CONDA_MANAGER
ARG conda=/opt/conda/bin/${CONDA_MANAGER}
COPY --link reqs/train-environment.yaml /tmp/train/environment.yaml
# Using `PIP_CACHE_DIR` and `CONDA_CACHE_DIR` to cache installations.
ARG PIP_CACHE_DIR=/tmp/.cache/pip
ARG CONDA_CACHE_DIR=/opt/conda/pkgs
ARG CONDA_ENV_FILE=/tmp/train/environment.yaml
COPY --link reqs/train-environment.yaml ${CONDA_ENV_FILE}
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    --mount=type=cache,target=${CONDA_CACHE_DIR} \
    find /tmp/dist -name '*.whl' | sed 's/^/      - /' >> ${CONDA_ENV_FILE} && \
    conda install -y python=${PYTHON_VERSION} ${CONDA_MANAGER} && \
    $conda env update --file ${CONDA_ENV_FILE}

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c

########################################################################
FROM ${TRAIN_IMAGE} AS train
# Example training image for Ubuntu 20.04+ on Intel x86_64 CPUs.
# Edit this section if necessary but use `docker-compose.yaml` if possible.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

ARG DEB_OLD
ARG DEB_NEW
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
# `tzdata` requires noninteractive mode.
ARG DEBIAN_FRONTEND=noninteractive
# Enable caching for `apt` packages.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > \
    /etc/apt/apt.conf.d/keep-cache

# Using `sed` and `xargs` to imitate the behavior of a requirements file.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
# See the `deploy` stage below to see how to add other apt reporitories.
# `apt` requirements are copied from the outside instead of from
# `train-builds` to allow parallel installation with pip.
COPY --link reqs/apt-train.requirements.txt /tmp/apt/requirements.txt
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ ${DEB_NEW} ]; then sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list; fi && \
    apt-get update && sed 's/#.*//g; s/\r//g' /tmp/apt/requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
# The `zsh` shell is used due to its convenience and popularity.
# Creating user with password-free sudo permissions.
# This may cause security issues. Use at your own risk.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell /bin/zsh --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Get conda with the directory ownership given to the user.
# Using conda for the virtual environment but not package installation.
COPY --link --from=train-builds --chown=${UID}:${GID} /opt/conda /opt/conda
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ENV MKL_DEBUG_CPU_TYPE=5
ENV LD_PRELOAD=/opt/conda/libfakeintel.so:${LD_PRELOAD}

# Use Intel OpenMP with optimizations. See documentation for details.
# https://intel.github.io/intel-extension-for-pytorch/tutorials/performance_tuning/tuning_guide.html
ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD

# Use Jemalloc for efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

USER ${USR}
# Docker must use absolute paths in `COPY` and cannot find `${HOME}`.
# Setting ${HOME} to its default value explicitly as a fix.
ARG HOME=/home/${USR}

# `PROJECT_ROOT` is where the project code will reside.
# The conda root path must be placed at the end of the
# PATH variable to prevent system program search errors.
# This is the opposite of the build stage.
ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:${PATH}
ENV PYTHONPATH=${PROJECT_ROOT}

# Setting the prompt to `pure`, which is available on all terminals without additional settings.
# This is a personal preference and users may use any prompt that they wish (e.g., oh-my-zsh).
ARG PURE_PATH=${HOME}/.zsh/pure
COPY --link --from=train-builds --chown=${UID}:${GID} /opt/zsh/pure ${PURE_PATH}
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
    } >> ${HOME}/.zshrc

## Add autosuggestions from terminal history. May be somewhat distracting.
#ARG ZSHA_PATH=${HOME}/.zsh/zsh-autosuggestions
#COPY --link --from=train-builds --chown=${UID}:${GID} /opt/zsh/zsh-autosuggestions ${ZSHA_PATH}
#RUN echo "source ${ZSHA_PATH}/zsh-autosuggestions.zsh" >> ${HOME}/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
ARG ZSHS_PATH=${HOME}/.zsh/zsh-syntax-highlighting
COPY --link --from=train-builds --chown=${UID}:${GID} \
    /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}
RUN echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${HOME}/.zshrc

# Add `ll` alias for convenience. The Mac version of `ll` is used
# instead of the Ubuntu version due to better configurability.
RUN echo "alias ll='ls -lh'" >> ${HOME}/.zshrc

# Enable mouse scrolling for tmux.
# iTerm2 users should change settings to use scrolling properly.
# RUN echo 'set -g mouse on' >> ${HOME}/.tmux.conf

# `PROJECT_ROOT` belongs to `USR` if created after `USER` has been set.
# Not so for pre-existing directories, which will still belong to root.
WORKDIR ${PROJECT_ROOT}

CMD ["/bin/zsh"]

########################################################################
FROM ${BUILD_IMAGE} AS deploy-builds-exclude

COPY --link --from=install-conda /opt/conda /opt/conda
COPY --link --from=build-pillow  /tmp/dist  /tmp/dist
COPY --link --from=fetch-torch   /tmp/dist  /tmp/dist
COPY --link --from=fetch-vision  /tmp/dist  /tmp/dist

########################################################################
FROM ${BUILD_IMAGE} AS deploy-builds-include

COPY --link --from=install-conda /opt/conda /opt/conda
COPY --link --from=build-pillow  /tmp/dist  /tmp/dist
COPY --link --from=build-vision  /tmp/dist  /tmp/dist

########################################################################
FROM deploy-builds-${BUILD_MODE} AS deploy-builds

# Minimalist deployment preparation layer.

# If any `pip` packages must be compiled on installation, create a wheel in the
# `build` stages and move it to `/tmp/dist`. Otherwise, the installtion may fail.
# See `Pillow-SIMD` in the TorchVision build process for an example.
# The `deploy` image is a CUDA `runtime` image without compiler tools.

# The Anaconda defaults channel and Intel MKL are not fully open-source.
# Enterprise users may therefore wish to remove them from their final product.
# The deployment therefore uses system Python. Conda is copied here just in case.
# Intel packages such as MKL can be removed by using MKL_MODE=exclude during the build.
# This may also be useful for non-Intel CPUs.

COPY --link reqs/apt-deploy.requirements.txt /tmp/apt/requirements.txt
COPY --link reqs/pip-deploy.requirements.txt /tmp/pip/requirements.txt

########################################################################
# Minimalist deployment Ubuntu image.
FROM ${DEPLOY_IMAGE} AS deploy

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Use mirror links optimized for user location and security level.
ARG DEB_OLD
ARG DEB_NEW

# Replace the `--mount=...` instructions with `COPY` if BuildKit is unavailable.
# The `readwrite` option is necessary because `apt` needs write permissions on `\tmp`.
# Both `python` and `python3` are set to point to the installed version of Python.
# The pre-installed system Python3 may be overridden if the installed and pre-installed
# versions of Python3 are the same (e.g., Python 3.8 on Ubuntu 20.04 LTS).
# `printf` is preferred over `echo` when escape characters are used due to
# the inconsistent behavior of `echo` across different shells.
# `software-properties-common` is required for the `add-apt-repository` command.
# Using `sed` and `xargs` to imitate the behavior of a requirements file.
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=deploy-builds,readwrite,source=/tmp/apt,target=/tmp/apt \
    if [ ${DEB_NEW} ]; then sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list; fi && \
    apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    printf "\n python${PYTHON_VERSION} \n" >> /tmp/apt/requirements.txt && \
    sed 's/#.*//g; s/\r//g' /tmp/apt/requirements.txt |  \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1

# The `mkl` package must be installed for PyTorch to use MKL outside `conda`.
# The MKL major version used at runtime must match the version used to build PyTorch.
# The `ldconfig` command is necessary for PyTorch to find MKL and other libraries.
# Installing all packages in one command allows `pip` to resolve dependencies correctly.
# Using multiple `pip` installs may break the dependencies of all but the last installation.
RUN --mount=type=bind,from=deploy-builds,source=/tmp/pip,target=/tmp/pip \
    --mount=type=bind,from=deploy-builds,source=/tmp/dist,target=/tmp/dist \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --find-links /tmp/dist \
        -r /tmp/pip/requirements.txt \
        /tmp/dist/*.whl && \
    ldconfig
