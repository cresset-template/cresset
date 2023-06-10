# syntax = docker/dockerfile:1
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
ARG INTERACTIVE_MODE
ARG USE_CUDA=1
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG IMAGE_FLAVOR
ARG LINUX_DISTRO
ARG DISTRO_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG USE_PRECOMPILED_HEADERS

# Fixing `git` to 2.38.1 as it is the last version to support `jobs=0`.
ARG GIT_IMAGE=alpine/git:edge-2.38.1
ARG CURL_IMAGE=curlimages/curl:latest

# Build-related packages are pre-installed on CUDA `devel` images.
# The `TRAIN_IMAGE` will use the `devel` flavor by default for convenience.
ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-${IMAGE_FLAVOR}-${LINUX_DISTRO}${DISTRO_VERSION}

########################################################################
FROM ${CURL_IMAGE} AS curl-conda
# An image used solely to download `conda` from the internet.

# Use a different CONDA_URL for a different CPU architecture or specific version.
# The Anaconda `defaults` channel is no longer free for commercial use.
# Using Miniforge or Mambaforge is strongly recommended. Viva la Open Source!
# Use Miniconda only if absolutely necessary.
# The defaults channel will be removed and the conda-forge channel will be used.
# https://conda.io/en/latest/license.html
# https://www.anaconda.com/terms-of-service
# https://www.anaconda.com/end-user-license-agreement-miniconda

ARG CONDA_URL
WORKDIR /tmp/conda
RUN curl -fvL -o /tmp/conda/miniconda.sh ${CONDA_URL}

########################################################################
FROM ${BUILD_IMAGE} AS install-conda

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Python won’t try to write `.pyc` or `.pyo` files on the import of source modules.
ENV PYTHONDONTWRITEBYTECODE=1
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging.
ENV PYTHONUNBUFFERED=1
# Allows UTF-8 characters as outputs in Docker.
ENV PYTHONIOENCODING=UTF-8

# Conda packages have higher run priority than system packages during the build.
ENV PATH=/opt/conda/bin:${PATH}

# `CONDA_MANAGER` may be either `mamba` or `conda`.
ARG CONDA_MANAGER
# Shortcut to simplify downstream installation.
ENV conda=/opt/conda/bin/${CONDA_MANAGER}

ARG PYTHON_VERSION
# The `.condarc` file in the installation directory portably configures the
# `conda-forge` channel and removes the `defaults` channel if Miniconda is used.
# No effect if Miniforge or Mambaforge is used as this is the default anyway.
# Clean out package and `__pycache__` directories to save space.
RUN --mount=type=bind,from=curl-conda,source=/tmp/conda,target=/tmp/conda \
    /bin/bash /tmp/conda/miniconda.sh -b -p /opt/conda && \
    printf "channels:\n  - conda-forge\n  - nodefaults\n" > /opt/conda/.condarc && \
    $conda install -y python=${PYTHON_VERSION} && \
    conda clean -fya && \
    find /opt/conda -type d -name '__pycache__' | xargs rm -rf

########################################################################
FROM install-conda AS install-mkl-base

# Get build requirements. Set package versions manually if compatibility issues arise.
COPY --link reqs/train-conda-build.requirements.txt /tmp/conda/build-requirements.txt

########################################################################
FROM install-mkl-base AS install-include-mkl

# Conda packages are preferable to system packages because they
# are much more likely to be the latest (and the greatest!) packages.
# Use fixed version numbers if versioning issues cause build failures.
# `sed 's/\.//; s/\..*//'` extracts `magma` versions from CUDA versions.
# For example, 11.5.1 becomes 115 and 10.2 becomes 102.
# Using the MatchSpec syntax for the magma-cuda package,
# which is only available from the PyTorch channel.
# All other packages should come from the `conda-forge` channel.
ARG CUDA_VERSION
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
RUN --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        pytorch::magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
        mkl-include \
        mkl

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
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
RUN --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    $conda install -y \
        --file /tmp/conda/build-requirements.txt \
        pytorch::magma-cuda$(echo ${CUDA_VERSION} | sed 's/\.//; s/\..*//') \
        nomkl

########################################################################
FROM install-${MKL_MODE}-mkl AS build-base
# `build-base` is the base stage for all heavy builds in the Dockerfile.

# Use Jemalloc as the system memory allocator for efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:${LD_PRELOAD}
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

# Ensure that `ccache` is used by `cmake`.
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Use LLD as the default linker for faster linking. Also update dynamic links.
RUN ln -sf /opt/conda/bin/ld.lld /usr/bin/ld && ldconfig

########################################################################
FROM ${GIT_IMAGE} AS clone-torch

# Updating git submodules is not fail-safe.
# If the build fails during `git clone`, just try again.
# The failure is likely due to networking issues.
# See https://stackoverflow.com/a/8573310/9289275
ARG PYTORCH_VERSION_TAG
ARG TORCH_URL=https://github.com/pytorch/pytorch.git
# Minimize downloads by only cloning shallow branches and not the full `git` history.
RUN git clone --jobs 0 --depth 1 --single-branch --shallow-submodules \
        --recurse-submodules --branch ${PYTORCH_VERSION_TAG} \
        ${TORCH_URL} /opt/pytorch

########################################################################
FROM build-base AS build-torch

WORKDIR /opt/pytorch
COPY --link --from=clone-torch /opt/pytorch /opt/pytorch

# Workaround for the header dependency bug in `nvcc`.
# Making this an `ENV` to allow downstream stages to use it as well.
ENV CMAKE_CUDA_COMPILER_LAUNCHER="python;/opt/pytorch/tools/nvcc_fix_deps.py;ccache"

# Read `setup.py` and `CMakeLists.txt` to find build flags.
# Different flags are available for different versions of PyTorch.
# Variables without defaults here recieve defaults from the top of the file.
# Variables must be defined both in the Dockerfile and `docker-compose.yaml`
# to be configurable via `.env`. Check `docker-compose.yaml` if a variable
# cannot be specified via `.env`. Note that variable definitions in
# `docker-compose.yaml` override default values specified in the Dockerfile.
# Variables specified in the `.env` file set values in the `docker-compose.yaml`
# file while default values in `docker-compose.yaml` have higher priority than
# default values specified for variables in the Dockerfile.
ARG USE_CUDA
ARG USE_CUDNN=${USE_CUDA}
ARG USE_NNPACK
ARG USE_QNNPACK
ARG BUILD_CAFFE2
ARG BUILD_CAFFE2_OPS
ARG BUILD_TEST
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

# Visit the Setuptools documentation for more `setup.py` options.
# https://setuptools.pypa.io/en/latest

# C++ developers using Libtorch can find the library in
# `torch/lib/tmp_install/lib/libtorch.so`.

# The default configuration removes all files except requirements files from the Docker context.
# To `COPY` your source files during the build, please edit the `.dockerignore` file.

# A detailed (if out of date) explanation of the buildsystem can be found below.
# https://pytorch.org/blog/a-tour-of-pytorch-internals-2
# The following repository may also be helpful for available options and possible issues.
# https://github.com/mratsim/Arch-Data-Science/blob/master/frameworks/python-pytorch-magma-mkldnn-cudnn-git/PKGBUILD

# Manually specify conda package versions if older PyTorch versions will not build.
# PyYAML, MKL-DNN, and Setuptools are known culprits.

# Run the command below before building to enable ROCM builds.
# RUN python tools/amd_build/build_amd.py
# PyTorch builds with ROCM have not been tested.
# Note that PyTorch for ROCM is still in beta and the ROCM build API may change.

# To build for Jetson Nano devices, see the link below for the necessary modifications.
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048

########################################################################
FROM install-conda AS build-pillow
# This stage is derived from `install-conda` instead of `build-base`
# as it is very lightweight and does not require many dependencies.
RUN $conda install -y libjpeg-turbo zlib && conda clean -fya

# Specify the `Pillow-SIMD` version if necessary. The variable is not used yet.
ARG PILLOW_SIMD_VERSION
# The condition ensures that AVX2 instructions are built only if available.
# May cause issues if the image is used on a machine with a different SIMD ISA.
RUN if [ ! "$(lscpu | grep -q avx2)" ]; then CC="cc -mavx2"; fi && \
    python -m pip wheel --no-deps --wheel-dir /tmp/dist \
        Pillow-SIMD  # ==${PILLOW_SIMD_VERSION}

########################################################################
FROM ${GIT_IMAGE} AS clone-vision

ARG TORCHVISION_VERSION_TAG
ARG VISION_URL=https://github.com/pytorch/vision.git
RUN git clone --jobs 0 --depth 1 --single-branch --shallow-submodules \
        --recurse-submodules --branch ${TORCHVISION_VERSION_TAG} \
        ${VISION_URL} /opt/vision

########################################################################
FROM build-torch AS build-vision

WORKDIR /opt/vision
COPY --link --from=clone-vision /opt/vision /opt/vision

# Install Pillow-SIMD before TorchVision build and add it to `/tmp/dist`.
# Pillow will be uninstalled if it is present.
RUN --mount=type=bind,from=build-pillow,source=/tmp/dist,target=/tmp/dist \
    python -m pip uninstall -y pillow && \
    python -m pip install --force-reinstall --no-deps /tmp/dist/*

ARG USE_CUDA
ARG USE_PRECOMPILED_HEADERS
ARG FORCE_CUDA=${USE_CUDA}
ARG TORCH_CUDA_ARCH_LIST
RUN --mount=type=cache,target=/opt/ccache \
    python setup.py bdist_wheel -d /tmp/dist

########################################################################
FROM ${GIT_IMAGE} AS fetch-pure

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

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
FROM ${BUILD_IMAGE} AS train-stash

# This stage prevents direct contact between the `train` stage and external files.
# Other files such as `.deb` package files may also be stashed here.
COPY --link reqs/train-apt.requirements.txt /tmp/apt/requirements.txt

########################################################################
FROM ${BUILD_IMAGE} AS train-builds-include
# A convenience stage to gather build artifacts (wheels, etc.) for the train stage.
# If other source builds are included later on, gather them here as well.
# All pip wheels are located in `/tmp/dist`.
# Using an image other than `BUILD_IMAGE` may contaminate
# `/opt/conda` and other key directories.

# The `train` stage is the one actually used for training.
# It is designed to be separate from the `build` stage,
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
# install using both conda and pip with a single file.
# Using a separate stage allows for build modularity
# and parallel installation with system packages.

# Adds a mirror `INDEX_URL` for PyPI via `PIP_CONFIG_FILE` if specified.
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
# Using `PIP_CACHE_DIR` and `CONDA_PKGS_DIRS`, both of which are
# native cache directory variables, to cache installations.
# Unclear which path `pip` inside a `conda` install uses for caching, however.
# https://pip.pypa.io/en/stable/topics/caching
# https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-package-directories-pkgs-dirs
# Remove `__pycache__` directories to save a bit of space.
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
ARG CONDA_ENV_FILE=/tmp/train/environment.yaml
COPY --link reqs/train-environment.yaml ${CONDA_ENV_FILE}
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    find /tmp/dist -name '*.whl' | sed 's/^/      - /' >> ${CONDA_ENV_FILE} && \
    $conda env update --file ${CONDA_ENV_FILE} && \
    find /opt/conda -type d -name '__pycache__' | xargs rm -rf

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c

########################################################################
FROM ${TRAIN_IMAGE} AS train-base
# Example Ubuntu training image on Intel x86_64 CPUs.
# Edit this section if necessary but use `docker-compose.yaml` if possible.
# Common configurations performed before creating a user should be placed here.

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

ARG TZ
ARG DEB_OLD
ARG DEB_NEW
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

# The `ZDOTDIR` variable specifies where to look for `zsh` configuration files.
# See the `zsh` manual for details. https://zsh-manual.netlify.app/files
ENV ZDOTDIR=/root

# Setting the prompt to `pure`, which is available on all terminals without additional settings.
# This is a personal preference and users may use any prompt that they wish (e.g., `oh-my-zsh`).
ARG PURE_PATH=${ZDOTDIR}/.zsh/pure
COPY --link --from=train-builds /opt/zsh/pure ${PURE_PATH}
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
    } >> ${ZDOTDIR}/.zshrc

## Add autosuggestions from terminal history. May be somewhat distracting.
#ARG ZSHA_PATH=${ZDOTDIR}/.zsh/zsh-autosuggestions
#COPY --link --from=train-builds /opt/zsh/zsh-autosuggestions ${ZSHA_PATH}
#RUN echo "source ${ZSHA_PATH}/zsh-autosuggestions.zsh" >> ${ZDOTDIR}/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
ARG ZSHS_PATH=${ZDOTDIR}/.zsh/zsh-syntax-highlighting
COPY --link --from=train-builds /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}
RUN echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR}/.zshrc

# `tzdata` requires noninteractive mode.
ARG DEBIAN_FRONTEND=noninteractive
# Enable caching for `apt` packages in Docker.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > \
    /etc/apt/apt.conf.d/keep-cache

# Using `sed` and `xargs` to imitate the behavior of a requirements file.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
# `apt` requirements are copied from the `train-stash` stage instead of from
# `train-builds` to allow parallel installation with `conda`.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,from=train-stash,source=/tmp/apt,target=/tmp/apt \
    if [ ${DEB_NEW} ]; then sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list; fi && \
    apt-get update && sed -e 's/#.*//g' -e 's/\r//g' /tmp/apt/requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

########################################################################
FROM train-base AS train-interactive-include
# This stage exists to create an interactive development environment with ease
# of experimentation and debugging in mind. A new `sudo` user is created to help
# prevent file ownership issues and accidents while not limiting freedom.
# All user-related and interactive configurations should be placed here.
# This is the default training stage that most users will use most of the time.

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
# The `zsh` shell is used due to its convenience and popularity.
# Creating user with password-free sudo permissions.
# This may cause security issues. Use at your own risk.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell $(which zsh) --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Get conda with the directory ownership given to the user.
COPY --link --from=train-builds --chown=${UID}:${GID} /opt/conda /opt/conda

# Add custom `zsh` aliases and settings.
# Add `ll` alias for convenience. The Mac version of `ll` is used
# instead of the Ubuntu version due to better configurability.
# Add `wns` as an alias for `watch nvidia-smi`, which is used often.
# Add `hist` as a shortcut to see the full history in `zsh`.
RUN {   echo "alias ll='ls -lh'"; \
        echo "alias wns='watch nvidia-smi'"; \
        echo "alias hist='history 1'"; \
    } >> ${ZDOTDIR}/.zshrc

########################################################################
FROM train-base AS train-interactive-exclude
# This stage exists to create images for use in Kubernetes clusters or for
# uploading images to a container registry, where interactive configurations
# are unnecessary and having the user set to `root` is most convenient.
# Singularity users may also find this stage useful.
# It is designed to be as close to the interactive development environment as
# possible, with the same `apt`, `conda`, and `pip` packages installed.
# Most users may safely ignore this stage except when publishing an image
# to a container repository for reproducibility.
# Note that this image does not require `zsh` but has `zsh` configs available.
# This allows users who download these images to use them interactively.

COPY --link --from=train-builds /opt/conda /opt/conda

########################################################################
FROM train-interactive-${INTERACTIVE_MODE} AS train
# Common configurations performed after `/opt/conda` installation
# should be placed here. Do not include any user-related options.

# Use Intel OpenMP with optimizations. See the documentation for details.
# https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html
# Intel OpenMP thread blocking time in ms.
ENV KMP_BLOCKTIME=0
# Configure CPU thread affinity.
ENV KMP_AFFINITY="granularity=fine,compact,1,0"
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:${LD_PRELOAD}

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ENV MKL_DEBUG_CPU_TYPE=5
ENV LD_PRELOAD=/opt/conda/libfakeintel.so:${LD_PRELOAD}

# Use Jemalloc for efficient memory management.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:${LD_PRELOAD}
# Jemalloc memory allocation configuration.
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# Change `/root` directory permissions to allow configuration sharing.
# Only the `/root` directory itself needs permission modification.
# Subdirectory permissions are intentionally left unmodified.
RUN chmod 711 /root

# Update dynamic link cache.
RUN ldconfig

# `PROJECT_ROOT` is where the project code will reside.
ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:${PATH}
ENV PYTHONPATH=${PROJECT_ROOT}
WORKDIR ${PROJECT_ROOT}
CMD ["/bin/zsh"]
