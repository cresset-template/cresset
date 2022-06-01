# syntax = docker/dockerfile:1.4
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

# Use `export BUILDKIT_PROGRESS=plain` in the host to see full build logs.
# See the link below for documentation on BuildKit syntax.
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md

# This image uses multi-stage builds. See the link below for a detailed description.
# https://docs.docker.com/develop/develop-images/multistage-build
# The build process is similar to a neural network with complicated control flow.
# The `stages` are like blocks in the network, with control flow implemented
# by the names of arguments and stages.

# All `ARG` variables must be redefined for every stage,
# `ARG`s defined before `FROM` transfer their values to layers that redefine them.
# `ENV` and `LABEL` variables are inherited only by child stages.
# See https://docs.docker.com/engine/reference/builder on how to write Dockerfiles and
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices for best practices.
# See https://hub.docker.com/r/nvidia/cuda for all available CUDA images.
ARG BUILD_MODE=exclude
ARG USE_CUDA=1
ARG USE_PRECOMPILED_HEADERS=1
ARG MKL_MODE=include
ARG CUDA_VERSION=11.5.2
ARG CUDNN_VERSION=8
ARG PYTHON_VERSION=3.9
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
ARG TORCH_CUDA_ARCH_LIST
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG DEPLOY_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-${LINUX_DISTRO}${DISTRO_VERSION}


# Build-related packages are pre-installed on CUDA `devel` images.
# Only the `cURL` package is downloaded from the package manager.
# The only use of cURL is to download Miniconda.
########################################################################
FROM ${BUILD_IMAGE} AS install-ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

########################################################################
FROM ${BUILD_IMAGE} AS install-centos
RUN yum -y install curl && yum -y clean all && rm -rf /var/cache

########################################################################
FROM ${BUILD_IMAGE} AS install-ubi
RUN yum -y install curl && yum -y clean all && rm -rf /var/cache

########################################################################
FROM ${BUILD_IMAGE} AS install-rockylinux
RUN yum -y install curl && yum -y clean all && rm -rf /var/cache

########################################################################
FROM install-${LINUX_DISTRO} AS install-base

LABEL maintainer=veritas9872@gmail.com
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
# The default CPU architecture is Intel x86_64.
# The Anaconda `defaults` channel is no longer free for commercial use.
# Anaconda (including Miniconda) itself is still open-source.
# Removing `defaults` channel as a result. Viva la Open Source!
# https://conda.io/en/latest/license.html
# https://www.anaconda.com/terms-of-service
# https://www.anaconda.com/end-user-license-agreement-miniconda
ARG MKL_MODE
ARG PYTHON_VERSION
# Conda packages have higher priority than system packages during build.
ENV PATH=/opt/conda/bin:${PATH}
# Available Miniconda installations: https://docs.conda.io/en/latest/miniconda.html
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Channel priority: `intel`, `conda-forge`, `pytorch`.
# Cannot set strict priority because of installation conflicts.
RUN curl -fsSL -v -o /tmp/miniconda.sh -O ${CONDA_URL} && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda config --append channels intel && \
    conda config --append channels conda-forge && \
    conda config --append channels pytorch && \
    conda config --remove channels defaults && \
    if [ ${MKL_MODE} != include ]; then \
        conda config --remove channels intel; \
    fi && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

########################################################################
FROM install-base AS install-conda

# Get build requirements. Set package versions manually if compatibility issues arise.
COPY --link reqs/conda-build.requirements.txt /tmp/conda/build-requirements.txt

# Comment out the lines below if Mamba causes any issues.
RUN conda install -y mamba && conda clean -ya
# Using Mamba instead of Conda as the package manager for faster installation.
ENV conda=/opt/conda/bin/mamba

########################################################################
FROM install-conda AS install-include-mkl

# Conda packages are preferable to system packages because they
# are much more likely to be the latest (and the greatest!) packages.
# Use fixed version numbers if versioning issues cause build failures.
# `sed 's/\.//; s/\..*//'` extracts `magma` versions from CUDA versions.
# For example, 11.5.1 becomes 115 and 10.2 becomes 102.
ARG CUDA_VERSION
RUN $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
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
FROM install-conda AS install-exclude-mkl

# The Intel(R) Math Kernel Library (MKL) places some restrictions on its use, though there are no
# restrictions on commercial use. See the Intel(R) Simplified Software License (ISSL) for details.
# Other Intel software such as the Intel OpenMP^* Runtime Library (iomp) are licensed under the
# Intel End User License Agreement for Developer Tools. See URL below for Intel licenses & EULAs.
# https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html
# Also, non-Intel CPUs may face slowdowns if MKL or other Intel tools are used in the backend.
ARG CUDA_VERSION
RUN $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
        nomkl && \
    conda clean -ya

########################################################################
FROM install-${MKL_MODE}-mkl AS build-base
# `build-base` is the base stage for all builds in the Dockerfile.

# Use Jemalloc as the system memory allocator for faster and more efficient memory management.
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

# Minimize downloads by only cloning shallow branches and not the full `git` history.
# If the build fails during `git clone`, just try again. Updating git submodules is not fail-safe.
# The reason for failure is probably networking issues during installation.
# See https://stackoverflow.com/a/8573310/9289275
WORKDIR /opt/pytorch
ARG PYTORCH_VERSION_TAG
ARG TORCH_URL=https://github.com/pytorch/pytorch.git
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

# A detailed (but out of date) explanation of the buildsystem can be found below.
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
FROM build-base AS build-pillow
# Specify the version of `Pillow-SIMD` if necessary.
# Condition ensures that AVX2 instructions are built only if available.
RUN if [ -n "$(lscpu | grep avx2)" ]; then \
        CC="cc -mavx2" \
        python -m pip wheel --no-deps --wheel-dir /tmp/dist Pillow-SIMD; \
    else \
        python -m pip wheel --no-deps --wheel-dir /tmp/dist Pillow-SIMD; \
    fi

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
ARG USE_FFMPEG=1
ARG USE_PRECOMPILED_HEADERS
ARG FORCE_CUDA=${USE_CUDA}
ARG TORCH_CUDA_ARCH_LIST
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist

########################################################################
FROM build-torch AS build-text

WORKDIR /opt/text
ARG TORCHTEXT_VERSION_TAG
ARG TEXT_URL=https://github.com/pytorch/text.git
RUN git clone --jobs 0 --depth 1 --single-branch --shallow-submodules \
        --recurse-submodules --branch ${TORCHTEXT_VERSION_TAG} \
        ${TEXT_URL} /opt/text

# TorchText does not use CUDA.
ARG USE_PRECOMPILED_HEADERS
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist

########################################################################
FROM build-base AS build-pure

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

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
COPY --link --from=install-base /opt/conda /opt/conda
COPY --link --from=build-pillow /tmp/dist  /tmp/dist
COPY --link --from=build-vision /tmp/dist  /tmp/dist
COPY --link --from=build-text   /tmp/dist  /tmp/dist
COPY --link --from=build-pure   /opt/zsh   /opt

########################################################################
FROM ${BUILD_IMAGE} AS train-builds-exclude
# Only build lightweight libraries.

COPY --link --from=install-base /opt/conda /opt/conda
COPY --link --from=build-pillow /tmp/dist  /tmp/dist
COPY --link --from=build-pure   /opt/zsh   /opt

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
        printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" > ${PIP_CONFIG_FILE}; \
    fi

ARG PATH=/opt/conda/bin:${PATH}
# Using `PIP_CACHE_DIR` to cache previous installations.
ARG PIP_CACHE_DIR=/tmp/.cache/pip
COPY --link reqs/pip-train.requirements.txt /tmp/pip/requirements.txt
# The `/tmp/dist/*.whl` files are the wheels built in previous stages.
# `--find-links` gives higher priority to the wheels in `/tmp/dist`.
# Installing all Python packages in a single command allows `pip` to resolve dependencies correctly.
# Using multiple `pip` installs may break the dependencies of all but the last installation.
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    python -m pip install --find-links /tmp/dist \
        -r /tmp/pip/requirements.txt \
        /tmp/dist/*.whl

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c

########################################################################
FROM ${TRAIN_IMAGE} AS train
# Example training image for Ubuntu 20.04 on an Intel x86_64 CPU. Edit if necessary.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1
ARG DEB_OLD
ARG DEB_NEW
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
# `tzdata` requires noninteractive mode.
ARG DEBIAN_FRONTEND=noninteractive
# Install `software-properties-common`, a requirement for the `add-apt-repository` command.
# Install `.deb` packages placed in `reqs/deb` on the project root directory.
RUN --mount=type=bind,source=reqs/deb,target=/tmp/deb \
    if [ ${DEB_NEW} ]; then sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list; fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        /tmp/deb/*.deb && \
    rm -rf /var/lib/apt/lists/*

# Using `sed` and `xargs` to imitate the behavior of a requirements file.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
# See the `deploy` stage below to see how to add other apt reporitories.
COPY --link reqs/apt-train.requirements.txt /tmp/apt/requirements.txt
RUN apt-get update && sed 's/#.*//g; s/\r//g' /tmp/apt/requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
# The `zsh` shell is used due to its convenience and popularity.
# Creating user with password-free sudo permissions. This may cause security issues.
RUN groupadd -g ${GID} ${GRP} && \
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
# This configuration only works for Ubuntu 20.04 or greater.
# Fix by building Jemalloc from source later.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

USER ${USR}
# Docker must use absolute paths in `COPY` and cannot find `${HOME}`.
# Setting ${HOME} to its default value explicitly as a fix.
ARG HOME=/home/${USR}

# `PROJECT_ROOT` is where the project code will reside.
ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:${PATH}
ENV PYTHONPATH=${PROJECT_ROOT}

# Setting the prompt to `pure`, which is available on all terminals without additional settings.
# This is a personal preference and users may use any prompt that they wish (e.g., oh-my-zsh).
# `printf` is preferred over `echo` when escape characters are used due to
# the inconsistent behavior of `echo` across different shells.
ARG PURE_PATH=${HOME}/.zsh/pure
COPY --link --from=train-builds --chown=${UID}:${GID} /opt/pure ${PURE_PATH}
RUN printf "fpath+=${PURE_PATH}\nautoload -Uz promptinit; promptinit\nprompt pure\n" >> ${HOME}/.zshrc

## Add autosuggestions from terminal history. May be somewhat distracting.
#ARG ZSHA_PATH=${HOME}/.zsh/zsh-autosuggestions
#COPY --link --from=train-builds --chown=${UID}:${GID} /opt/zsh-autosuggestions ${ZSHA_PATH}
#RUN echo "source ${ZSHA_PATH}/zsh-autosuggestions.zsh" >> ${HOME}/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
ARG ZSHS_PATH=${HOME}/.zsh/zsh-syntax-highlighting
COPY --link --from=train-builds --chown=${UID}:${GID} /opt/zsh-syntax-highlighting ${ZSHS_PATH}
RUN echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${HOME}/.zshrc

# Enable mouse scrolling for tmux. This also disables copying text from the terminal.
# RUN echo 'set -g mouse on' >> ${HOME}/.tmux.conf

# `PROJECT_ROOT` belongs to `USR` if created after `USER` has been set.
# Not so for pre-existing directories, which will still belong to root.
WORKDIR ${PROJECT_ROOT}

CMD ["/bin/zsh"]

########################################################################
# Minimalist deployment preparation layer.
FROM ${BUILD_IMAGE} AS deploy-builds

# If any `pip` packages must be compiled on installation, create a wheel in the
# `build` stages and move it to `/tmp/dist`. Otherwise, the installtion may fail.
# See `Pillow-SIMD` in the TorchVision build process for an example.
# The `deploy` image is a CUDA `runtime` image without compiler tools.

# The licenses for the Anaconda defaults channel and Intel MKL are not fully open-source.
# Enterprise users may therefore wish to remove them from their final product.
# The deployment therefore uses system Python. Conda is copied here just in case.
# Intel packages such as MKL can be removed by using MKL_MODE=exclude during the build.
# This may also be useful for non-Intel CPUs.

COPY --link --from=install-base /opt/conda /opt/conda
COPY --link --from=build-pillow /tmp/dist  /tmp/dist
COPY --link --from=build-vision /tmp/dist  /tmp/dist

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
RUN --mount=type=bind,from=deploy-builds,source=/tmp/pip,target=/tmp/pip \
    --mount=type=bind,from=deploy-builds,source=/tmp/dist,target=/tmp/dist \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --find-links /tmp/dist \
        -r /tmp/pip/requirements.txt \
        /tmp/dist/*.whl && \
    ldconfig
