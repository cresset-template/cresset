# syntax = docker/dockerfile:1
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

# Use `export BUILDKIT_PROGRESS=plain` in the host to see full build logs.
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
ARG USE_PRECOMPILED_HEADERS=1
ARG CLEAN_CACHE_BEFORE_BUILD=0
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
########################################################################
FROM ${BUILD_IMAGE} AS install-ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

########################################################################
FROM ${BUILD_IMAGE} AS install-centos
RUN yum -y install curl && yum -y clean all  && rm -rf /var/cache

########################################################################
FROM ${BUILD_IMAGE} AS install-ubi
RUN yum -y install curl && yum -y clean all  && rm -rf /var/cache

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
# The default CPU architecture is x86_64.
# The Anaconda `defaults` channel is no longer free for commercial use.
# Anaconda (including Miniconda) itself is still open-source.
# Removing `defaults` as a result. Viva la Open Source!
# https://conda.io/en/latest/license.html
# https://www.anaconda.com/terms-of-service
# https://www.anaconda.com/end-user-license-agreement-miniconda
ARG MKL_MODE
ARG PYTHON_VERSION
# Conda packages have higher priority than system packages during build.
ENV PATH=/opt/conda/bin:$PATH
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
COPY reqs/conda-build.requirements.txt /tmp/reqs/conda-build.requirements.txt

# Comment out the lines below if Mamba causes any issues.
RUN conda install -y mamba && conda clean -ya
# Using Mamba instead of Conda as the package manager for faster installation.
ENV conda=/opt/conda/bin/mamba

########################################################################
FROM install-conda AS install-include-mkl

# Roundabout method to enable MKLDNN in PyTorch build when MKL is included.
ENV USE_MKLDNN=1

# Conda packages are preferable to system packages because they
# are much more likely to be the latest (and the greatest!) packages.
# Use fixed version numbers if versioning issues cause build failures.
ARG MAGMA_VERSION
RUN $conda install -y \
        --file /tmp/reqs/conda-build.requirements.txt \
        magma-cuda${MAGMA_VERSION} \
        mkl-include \
        mkl && \
    conda clean -ya

# Use Intel OpenMP with optimizations enabled.
# Some compilers can use OpenMP for faster builds.
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD
ENV KMP_WARNINGS=0
ENV KMP_AFFINITY="granularity=fine,nonverbose,compact,1,0"
ENV KMP_BLOCKTIME=0

########################################################################
FROM install-conda AS install-exclude-mkl

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

########################################################################
FROM install-${MKL_MODE}-mkl AS build-base
# `build-base` is the base stage for all builds in the Dockerfile.

# Use Jemalloc as the system memory allocator for faster and more efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD
# See the documentation for an explanation of the following configuration.
# https://android.googlesource.com/platform/external/jemalloc_new/+/6e6a93170475c05ebddbaf3f0df6add65ba19f01/TUNING.md
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

# Settings common to both gomp and iomp.
ENV OMP_PROC_BIND=CLOSE
ENV OMP_SCHEDULE=STATIC

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

########################################################################
FROM build-base AS build-torch

# Checkout to specific version and update submodules.
WORKDIR /opt/pytorch
ARG PYTORCH_VERSION_TAG
ARG TORCH_URL=https://github.com/pytorch/pytorch.git
RUN git clone --recursive --jobs 0 ${TORCH_URL} /opt/pytorch && \
    if [ -n ${PYTORCH_VERSION_TAG} ]; then \
        git checkout ${PYTORCH_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

# PyTorch itself can find the host GPU architecture
# on its own but its subsidiary libraries cannot,
# hence the need to specify the architecture list explicitly.
# Building PyTorch with several optimizations and bugfixes.
# Disabling Caffe2, NNPack, and QNNPack as they are legacy and most users do not need them.
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
ARG BUILD_CAFFE2_OPS=0
ARG USE_PRECOMPILED_HEADERS
ARG CLEAN_CACHE_BEFORE_BUILD
ARG TORCH_CUDA_ARCH_LIST
ARG CMAKE_PREFIX_PATH=/opt/conda
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# Build wheel for installation in later stages.
# Install PyTorch for subsidiary libraries (e.g., TorchVision).

# The standard method for building PyTorch is shown below.
# Build artifacts such as `*.so` files are lost between builds.
# CCache will speed up builds but identical PyTorch configurations will still require rebuilds.
# Cleaning the `/opt/_pytorch` cache is equivalent to the example below.
#RUN --mount=type=cache,target=/opt/ccache \
#    python setup.py bdist_wheel -d /tmp/dist && \
#    python setup.py install

WORKDIR /opt/_pytorch
RUN --mount=type=cache,target=/opt/ccache \
    --mount=type=cache,target=/opt/_pytorch \
    if [ ${CLEAN_CACHE_BEFORE_BUILD} != 0 ]; then \
        find /opt/_pytorch -mindepth 1 -delete; \
    fi && \
    rsync -a /opt/pytorch/ /opt/_pytorch/ && \
    python setup.py bdist_wheel -d /tmp/dist && \
    python setup.py install && \
    rm -rf .git

# The following mechanism combines the reproducibility of Docker with the speed of local compilation.
# The entire directory is used as a BuildKit cache to speed up installation between separate builds.
# Compilation outputs between different builds must be saved.
# Otherwise, PyTorch must be rebuilt whenever any previous stage is modified,
# even if the version of PyTorch being compiled is identical because
# without whole directory caching, many build artifacts (e.g., `*.so` files) must be rebuilt.
# Even with CCache accceleration, this is a very slow process.

# Please note that ~20GB of memory for the entire process, which may be larger than the
# default cache size allowed by the Docker BuildKit garbage collection settings.
# Disable GC or increase `defaultKeepStorage`, the allowed cache size, to prevent recompiles.
# See the issue for help. https://github.com/docker/cli/issues/2325
# If a build failure occurs, try clearing the /opt/_*/ directory cache.
# Note that new build configuration may not be correctly processed if the cache is not cleared.
# Use `docker builder prune` to clear all build caches in BuildKit.

# On balance, the additional complexity is worth the increased ease of modifying the build layers.
# Enable `CLEAN_CACHE_BEFORE_BUILD` if multiple slightly different versions of
# PyTorch in different environments must be built continuously, such as in CI tests.

# C++ developers using Libtoch can find the library in `torch/lib/tmp_install/lib/libtorch.so`.

# The `.git` directory is deleted due to its large size (759MB of 846MB).

# The default configuration removes all files except requirements files from the Docker context.
# To `COPY` your source files during the build, please edit the `.dockerignore` file.

###### Additional information for custom builds. ######
# A detailed (but out of date) explanation of the buildsystem can be found below.
# https://pytorch.org/blog/a-tour-of-pytorch-internals-2
# The following repository may also be helpful for available options and possible issues.
# https://github.com/mratsim/Arch-Data-Science/blob/master/frameworks/python-pytorch-magma-mkldnn-cudnn-git/PKGBUILD

# Manually specify conda package versions if older PyTorch versions will not build.
# PyYAML, MKL-DNN, and SetupTools are known culprits.

# Use this to build with custom CMake settings.
#RUN --mount=type=cache,target=/opt/ccache \
#    python setup.py build --cmake-only && \
#    ccmake build  # or cmake-gui build

# See the SetupTools documentation for more setup.py options.
# https://setuptools.pypa.io/en/latest

# Run the command below before building to enable ROCM builds.
# RUN python tools/amd_build/build_amd.py
# PyTorch builds with ROCM has not been tested.
# Note that PyTorch for ROCM is still in beta and
# the API for enabling ROCM builds may change.

# To build for Jetson Nano devices, see the link below for the necessary modifications.
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048


########################################################################
FROM build-base AS build-pillow
# Specify the version if necessary.
# This may not work on older CPUs as it requires SSE4 and AVX2.
RUN CC="cc -mavx2" python -m pip wheel --no-deps --wheel-dir /tmp/dist Pillow-SIMD

########################################################################
FROM build-torch AS build-vision

WORKDIR /opt/vision
ARG TORCHVISION_VERSION_TAG
ARG VISION_URL=https://github.com/pytorch/vision.git
RUN git clone --recursive --jobs 0 ${VISION_URL} /opt/vision && \
    if [ -n ${TORCHVISION_VERSION_TAG} ]; then \
        git checkout ${TORCHVISION_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

# Install Pillow SIMD before TorchVision build and add it to `/tmp/dist`.
# Pillow will be uninstalled if it is present.
RUN --mount=type=bind,from=build-pillow,source=/tmp/dist,target=/tmp/dist \
    python -m pip uninstall -y pillow && \
    python -m pip install --force-reinstall --no-deps /tmp/dist/*


ARG DEBUG
ARG USE_CUDA
ARG USE_FFMPEG=1
ARG USE_PRECOMPILED_HEADERS
ARG CLEAN_CACHE_BEFORE_BUILD
ARG FORCE_CUDA=${USE_CUDA}
ARG TORCH_CUDA_ARCH_LIST
WORKDIR /opt/_vision
RUN --mount=type=cache,target=/opt/ccache \
    --mount=type=cache,target=/opt/_vision \
    if [ ${CLEAN_CACHE_BEFORE_BUILD} != 0 ]; then \
        find /opt/_vision -mindepth 1 -delete; \
    fi && \
    rsync -a /opt/vision/ /opt/_vision/ && \
    python setup.py bdist_wheel -d /tmp/dist && \
    rm -rf .git

########################################################################
FROM build-torch AS build-text

WORKDIR /opt/text
ARG TORCHTEXT_VERSION_TAG
ARG TEXT_URL=https://github.com/pytorch/text.git
RUN git clone --recursive --jobs 0 ${TEXT_URL} /opt/text && \
    if [ -n ${TORCHTEXT_VERSION_TAG} ]; then \
        git checkout ${TORCHTEXT_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

# TorchText does not use CUDA.
ARG USE_PRECOMPILED_HEADERS
ARG CLEAN_CACHE_BEFORE_BUILD
WORKDIR /opt/_text
RUN --mount=type=cache,target=/opt/ccache \
    --mount=type=cache,target=/opt/_text \
    if [ ${CLEAN_CACHE_BEFORE_BUILD} != 0 ]; then \
        find /opt/_text -mindepth 1 -delete; \
    fi && \
    rsync -a /opt/text/ /opt/_text/ && \
    python setup.py bdist_wheel -d /tmp/dist && \
    rm -rf .git

########################################################################
FROM build-torch AS build-audio

WORKDIR /opt/audio
ARG TORCHAUDIO_VERSION_TAG
ARG AUDIO_URL=https://github.com/pytorch/audio.git
RUN git clone --recursive --jobs 0 ${AUDIO_URL} /opt/audio && \
    if [ -n ${TORCHAUDIO_VERSION_TAG} ]; then \
        git checkout ${TORCHAUDIO_VERSION_TAG} && \
        git submodule sync && \
        git submodule update --init --recursive --jobs 0; \
    fi

ARG USE_CUDA
ARG USE_ROCM
ARG USE_PRECOMPILED_HEADERS
# The cache should be cleaned by default due to a bug in TorchAudio.
ARG CLEAN_CACHE_BEFORE_BUILD=1
ARG BUILD_TORCHAUDIO_PYTHON_EXTENSION=1
ARG BUILD_FFMPEG=1
ARG TORCH_CUDA_ARCH_LIST
WORKDIR /opt/_audio
RUN --mount=type=cache,target=/opt/ccache \
    --mount=type=cache,target=/opt/_audio \
    if [ ${CLEAN_CACHE_BEFORE_BUILD} != 0 ]; then \
        find /opt/_audio -mindepth 1 -delete; \
    fi && \
    rsync -a /opt/audio/ /opt/_audio/ && \
    python setup.py bdist_wheel -d /tmp/dist && \
    rm -rf .git

########################################################################
FROM build-base AS build-pure

# Z-Shell related libraries.
RUN git clone https://github.com/sindresorhus/pure.git /opt/zsh/pure
RUN git clone https://github.com/zsh-users/zsh-autosuggestions /opt/zsh/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /opt/zsh/zsh-syntax-highlighting

########################################################################
FROM ${BUILD_IMAGE} AS train-builds
# A convenience stage to gather build artifacts (wheels, etc.) for the train stage.
# If other source builds are included later on, gather them here as well.
# The train stage should not have any dependencies other than this stage.
# This stage does not have anything installed. No variables are specified either.
# This stage is simply the `BUILD_IMAGE` with additional files and directories.
# All pip wheels are located in `/tmp/dist`.
# Using an image other than `BUILD_IMAGE` may contaminate the `/opt/conda` and other key directories.

# The `train` image is the one actually used for training.
# It is designed to be separate from the `build` image,
# with only the build artifacts (e.g., pip wheels) copied over.

# The order of `COPY` instructions is chosen to minimize cache misses.
COPY --from=install-base /opt/conda /opt/conda
COPY --from=build-pillow /tmp/dist  /tmp/dist
COPY --from=build-vision /tmp/dist  /tmp/dist
COPY --from=build-audio  /tmp/dist  /tmp/dist
COPY --from=build-text   /tmp/dist  /tmp/dist

# `COPY` new builds here to minimize the likelihood of cache misses.
COPY --from=build-pure  /opt/zsh /opt

# Copying requirements files from context so that the `train` image
# can be built from this stage with no dependency on the Docker context.
# The files are placed in different directories to allow changing one file
# without affecting the bind mount directory of the other files.
# If all files were placed in the same directory, changing just one file
# would cause a cache miss, forcing all requirements to reinstall.
COPY reqs/apt-train.requirements.txt /tmp/reqs/apt/requirements.txt
COPY reqs/pip-train.requirements.txt /tmp/reqs/pip/requirements.txt

########################################################################
FROM ${TRAIN_IMAGE} AS train
# Example training image for Ubuntu on an Intel x86_64 CPU.
# Edit this image if necessary.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Set as `ARG`s to reduce image footprint but not affect the resulting images.
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Speedups in `apt` installs for Korean users. Change URLs for other locations.
# http://archive.ubuntu.com/ubuntu is specific to NVIDIA's CUDA Ubuntu images.
# Check `/etc/apt/sources.list` of the base image to find the Ubuntu URL.
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com

# `tzdata` requires a timezone and noninteractive mode.
ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
# Using `sed` and `xargs` to imitate the behavior of a requirements file.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
RUN --mount=type=bind,from=train-builds,source=/tmp/reqs/apt,target=/tmp/reqs/apt \
    sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
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
# Docker must use absolute paths in `COPY` and cannot find `$HOME`.
# Setting $HOME to its default value explicitly as a fix.
ARG HOME=/home/${USR}

# Get conda with the directory ownership given to the user.
# Using conda for the virtual environment but not package installation.
COPY --from=train-builds --chown=${UID}:${GID} /opt/conda /opt/conda

# `PROJECT_ROOT` is where the project code will reside.
ARG PROJECT_ROOT=/opt/project
# Path order conveys precedence.
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}

# Configure channels in case anyone uses `conda`.
# The configurations are not copied with `/opt/conda`.
ARG MKL_MODE
RUN conda config --append channels intel && \
    conda config --append channels conda-forge && \
    conda config --append channels pytorch && \
    conda config --remove channels defaults && \
    if [ ${MKL_MODE} != include ]; then \
        conda config --remove channels intel; \
    fi

# Setting the prompt to `pure`, which is available on all terminals without additional settings.
# This is a personal preference and users may use any prompt that they wish (e.g., oh-my-zsh).
# `printf` is preferred over `echo` when escape characters are used due to
# the inconsistent behavior of `echo` across different shells.
COPY --from=train-builds --chown=${UID}:${GID} /opt/pure $HOME/.zsh/pure
RUN printf "fpath+=$HOME/.zsh/pure\nautoload -Uz promptinit; promptinit\nprompt pure\n" >> $HOME/.zshrc

## Add autosuggestions from terminal history. May be somewhat distracting.
#COPY --from=train-builds --chown=${UID}:${GID} /opt/zsh-autosuggestions $HOME/.zsh/zsh-autosuggestions
#RUN echo "source $HOME/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" >> $HOME/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
COPY --from=train-builds --chown=${UID}:${GID} /opt/zsh-syntax-highlighting $HOME/.zsh/zsh-syntax-highlighting
RUN echo "source $HOME/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> $HOME/.zshrc

# The `/tmp/dist/*.whl` files are the wheels built in previous stages.
# `--find-links` gives higher priority to the wheels in `/tmp/dist`.
# Speedups in `pip` for Korean users. Change URLs for other locations.
# Installing all Python packages in a single command allows `pip` to resolve dependencies correctly.
# Using multiple `pip` installs will break the dependencies of all but the last installation.
# The numpy, scipy, and numba libraries are not MKL optimized when installed from PyPI.
# Install them from the Intel channel of Anaconda if desired.
# Including versioning and dependencies is too complicated.
ARG INDEX_URL=https://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
RUN --mount=type=bind,from=train-builds,source=/tmp/dist,target=/tmp/dist \
    --mount=type=bind,from=train-builds,source=/tmp/reqs/pip,target=/tmp/reqs/pip \
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" > /opt/conda/pip.conf && \
    python -m pip install --no-cache-dir --find-links /tmp/dist  \
        -r /tmp/reqs/pip/requirements.txt \
        /tmp/dist/*.whl

# Settings common to both gomp and iomp.
ENV OMP_PROC_BIND=CLOSE
ENV OMP_SCHEDULE=STATIC
# Use Intel OpenMP with optimizations. See documentation for details.
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# https://intel.github.io/intel-extension-for-pytorch/tutorials/performance_tuning/tuning_guide.html
ENV KMP_WARNINGS=0
ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD
ENV KMP_AFFINITY="granularity=fine,nonverbose,compact,1,0"

# Use Jemalloc for faster and more efficient memory management.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

# Temporarily switch to `root` for super-user permissions.
USER root
# Include `conda` in dynamic linking.
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig
USER ${USR}

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

COPY --from=install-base /opt/conda /opt/conda
COPY --from=build-pillow /tmp/dist  /tmp/dist
COPY --from=build-vision /tmp/dist  /tmp/dist

COPY reqs/apt-deploy.requirements.txt /tmp/reqs/apt-deploy.requirements.txt
COPY reqs/pip-deploy.requirements.txt /tmp/reqs/pip-deploy.requirements.txt

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
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
ARG INDEX_URL=https://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com

# Replace the `--mount=...` instructions with `COPY` if BuildKit is unavailable.
# The `readwrite` option is necessary because `apt` needs write permissions on `\tmp`.
# Both `python` and `python3` are set to point to the installed version of Python.
# The pre-installed system Python3 may be overridden if the installed and pre-installed
# versions of Python3 are the same (e.g., Python 3.8 on Ubuntu 20.04 LTS).
# Using `sed` and `xargs` to imitate the behavior of a requirements file.
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=deploy-builds,readwrite,source=/tmp,target=/tmp \
    sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
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
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" > /etc/pip.conf && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --find-links /tmp/dist/ \
        -r /tmp/reqs/pip-deploy.requirements.txt \
        /tmp/dist/*.whl && \
    ldconfig
