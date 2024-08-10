# syntax = docker/dockerfile:1
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

# This Dockerfile exists to provide a method of installing all packages from
# `conda` using only Official and Verified Docker images.
# The base image for training is an Official Docker Linux image, e.g., Ubuntu.
# The `git` image is from a verified publisher and only used to download files.
# The `gcc` image is an Official Docker image used for compilation.
# The training image does not include it and it remains in the build cache.

ARG ADD_USER
ARG LOCK_MODE
ARG BASE_IMAGE
# Fix `gcc` to a specific version if necessary.
ARG GCC_IMAGE=gcc:latest
# The Bitnami Docker verified git image has `curl` installed in `/usr/bin/curl`
# and recent versions have both AMD64 and ARM64 architecture support.
ARG GIT_IMAGE=bitnami/git:latest

########################################################################
FROM ${GIT_IMAGE} AS stash

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

ARG CONDA_URL
WORKDIR /tmp/conda
RUN curl -fksSL -o /tmp/conda/miniconda.sh ${CONDA_URL}

COPY --link ../reqs/simple-apt.requirements.txt /tmp/apt/requirements.txt

WORKDIR /

########################################################################
FROM ${BASE_IMAGE} AS conda-lock-exclude
# Use this stage for rough requirements specified during development.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# `CONDA_MANAGER` may be either `mamba` or `conda`.
ARG CONDA_MANAGER
ARG conda=/opt/conda/bin/${CONDA_MANAGER}

# Using package caching to speed up installs.
# A `CondaVerificationError` may occur if the cache is corrupted.
# Use `docker builder prune` to clear out the build cache if it does.
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
RUN --mount=type=bind,from=stash,source=/tmp/conda,target=/tmp/conda \
    /bin/bash /tmp/conda/miniconda.sh -b -p /opt/conda && \
    printf "channels:\n  - conda-forge\n  - nodefaults\nssl_verify: false\n" > /opt/conda/.condarc && \
    $conda clean -fya && find /opt/conda -type d -name '__pycache__' | xargs rm -rf

COPY --link ../reqs/simple-environment.yaml /tmp/req/environment.yaml

ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG TRUSTED_HOST
ARG PIP_CONFIG_FILE=/opt/conda/pip.conf
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    {   echo "[global]"; \
        echo "index-url=${INDEX_URL}"; \
        echo "extra-index-url=${EXTRA_INDEX_URL}"; \
        echo "trusted-host=${TRUSTED_HOST}"; \
    } > ${PIP_CONFIG_FILE} && \
    $conda env update --file /tmp/req/environment.yaml

# Cleaning must be in a separate `RUN` command to preserve the Docker cache.
RUN $conda clean -fya

########################################################################
FROM stash AS lock-stash
# Extra stash for using `conda-lock` files. This stage is derived from `stash`
# to reduce code repitition at the cost of unnecessary extra build cache.

ARG CONDA_MANAGER
# Weird paths necessary because `CONDA_PREFIX` is immutable post-installation.
ARG conda=/opt/_conda/bin/${CONDA_MANAGER}
RUN /bin/bash /tmp/conda/miniconda.sh -b -p /opt/_conda && \
    printf "channels:\n  - conda-forge\n  - nodefaults\nssl_verify: false\n" > /opt/_conda/.condarc && \
    $conda install conda-lock && $conda clean -fya && \
    find /opt/conda -type d -name '__pycache__' | xargs rm -rf

########################################################################
FROM ${BASE_IMAGE} AS conda-lock-include
# Use this stage only if installing from `conda-lock`.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

COPY --link --from=lock-stash /opt/_conda /opt/_conda
COPY --link ../reqs/simple-conda-lock.yaml /tmp/conda/lock.yaml
# Saves to `conda-linux-64.lock`, which can be installed via `conda create`.
RUN /opt/_conda/bin/conda-lock render -p linux-64 /tmp/conda/lock.yaml

ARG CONDA_MANAGER
ARG conda=/opt/_conda/bin/${CONDA_MANAGER}
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/_conda/pkgs
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    $conda create --no-deps --no-default-packages --copy -p /opt/conda \
        --file conda-linux-64.lock && \
    printf "channels:\n  - conda-forge\n  - nodefaults\nssl_verify: false\n" > /opt/conda/.condarc

########################################################################
FROM ${GCC_IMAGE} AS train-builds

WORKDIR /opt/conda

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c

########################################################################
FROM conda-lock-${LOCK_MODE} AS install-conda
# Cleanup before copying to reduce image size.
RUN find /opt/conda -type d -name '__pycache__' | xargs rm -rf

# Heuristic fix to find NVRTC for CUDA 11.2+.
# Change this for older CUDA versions and for CUDA 12.x.
RUN if [ -f "/opt/conda/lib/libnvrtc.so.11.2" ]; then \
        ln -s "/opt/conda/lib/libnvrtc.so.11.2" "/opt/conda/lib/libnvrtc.so"; \
    fi

# Copy the binary file to enable acceleration of AMD CPUs by the latest MKL versions.
COPY --link --from=train-builds /opt/conda/libfakeintel.so /opt/conda/libfakeintel.so

########################################################################
FROM ${BASE_IMAGE} AS train-base

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Using `sed` and `xargs` to imitate the behavior of a requirements file.
# The `--mount=type=bind` temporarily mounts a directory from another stage.
# `tzdata` requires noninteractive mode.
ARG TZ
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=stash,source=/tmp/apt,target=/tmp/apt \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    apt-get update && sed -e 's/#.*//g' -e 's/\r//g' /tmp/apt/requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

########################################################################
FROM train-base AS train-adduser-exclude
# Stage used to create images for Kubernetes clusters or for uploading to
# container registries such as Docker Hub. No users or interactive settings.
# Note that `zsh` configs are available but these images do not require `zsh`.
COPY --link --from=install-conda /opt/conda /opt/conda

########################################################################
FROM train-base AS train-adduser-include

ARG GID
ARG UID
ARG GRP
ARG USR
ARG PASSWD=ubuntu
# The `zsh` shell is used due to its convenience and popularity.
# Creating user with password-free sudo permissions.
# This may cause security issues. Use at your own risk.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell $(which zsh) --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Get conda with the directory ownership given to the user.
COPY --link --from=install-conda --chown=${UID}:${GID} /opt/conda /opt/conda

########################################################################
FROM train-adduser-${ADD_USER} AS train

# Necessary to find the NVIDIA Driver.
# Setting this in the image for alternative container runtime configurations.
ENV NVIDIA_VISIBLE_DEVICES=all

# Use Intel OpenMP with optimizations. See the documentation for details.
# https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html
ENV KMP_BLOCKTIME=0
# ENV KMP_AFFINITY="granularity=fine,compact,1,0"
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so${LD_PRELOAD:+:${LD_PRELOAD}}

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ENV MKL_DEBUG_CPU_TYPE=5
ENV LD_PRELOAD=/opt/conda/libfakeintel.so${LD_PRELOAD:+:${LD_PRELOAD}}
# Configure Jemalloc as the default memory allocator.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

ENV ZDOTDIR=/root
# Setting the prompt to `pure`.
ARG PURE_PATH=${ZDOTDIR}/.zsh/pure
ARG ZSHS_PATH=${ZDOTDIR}/.zsh/zsh-syntax-highlighting
COPY --link --from=stash /opt/zsh/pure ${PURE_PATH}
COPY --link --from=stash /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        # Change the `tmux` path color to cyan since
        # the default blue is unreadable on a dark terminal.
        echo "zmodload zsh/nearcolor"; \
        echo "zstyle :prompt:pure:path color cyan"; \
        echo "prompt pure"; \
    } >> ${ZDOTDIR}/.zshrc && \
    # Add autosuggestions from terminal history. May be somewhat distracting.
    # echo "source ${ZSHA_PATH}/zsh-autosuggestions.zsh" >> ${ZDOTDIR}/.zshrc && \
    # Add custom `zsh` aliases and settings.
    {   echo "alias ll='ls -lh'"; \
        echo "alias wns='watch nvidia-smi'"; \
        echo "alias hist='history 1'"; \
    } >> ${ZDOTDIR}/.zshrc && \
    # Syntax highlighting must be activated at the end of the `.zshrc` file.
    echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR}/.zshrc && \
    # Configure `tmux` to use `zsh` as a non-login shell on startup.
    echo "set -g default-command $(which zsh)" >> /etc/tmux.conf && \
    # For some reason, `tmux` does not read `/etc/tmux.conf`.
    echo 'cp /etc/tmux.conf ${HOME}/.tmux.conf' >> ${ZDOTDIR}/.zprofile && \
    # Change `ZDOTDIR` directory permissions to allow configuration sharing.
    chmod 755 ${ZDOTDIR} && \
    # Clear out `/tmp` and restore its default permissions.
    rm -rf /tmp && mkdir /tmp && chmod 1777 /tmp && \
    ldconfig  # Update dynamic link cache.

ENV PATH=/opt/conda/bin:${PATH}
ARG PROJECT_ROOT=/opt/project
ENV PYTHONPATH=${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
WORKDIR ${PROJECT_ROOT}
CMD ["/bin/zsh"]
