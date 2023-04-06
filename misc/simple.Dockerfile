# syntax = docker/dockerfile:1.4
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

# This Dockerfile exists to provide a method of installing all packages from
# `conda` using only Official and Verified Docker images.
# The base image for training is an Official Docker Linux image, e.g., Ubuntu.
# The `git` and `conda` images are both from verified publishers.
# Also, they are used only to download files and packages.
# The training image does not include them and they remain in the build cache.

ARG LOCK_MODE
ARG BASE_IMAGE
ARG INTERACTIVE_MODE
# The Bitnami Docker verified git image has `curl` installed in `/usr/bin/curl`
# and recent versions have both AMD64 and ARM64 architecture support.
ARG GIT_IMAGE=bitnami/git:latest
# The Miniconda3 verified image is only used
# if strict requirements are installed via `conda-lock` files.
ARG CONDA_IMAGE=continuumio/miniconda3:latest
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
RUN mkdir /tmp/conda && curl -fsSL -v -o /tmp/conda/miniconda.sh -O ${CONDA_URL}

COPY --link ../reqs/apt-simple.requirements.txt /tmp/apt/requirements.txt
COPY --link ../reqs/simple-environment.yaml /tmp/req/environment.yaml

########################################################################
FROM ${BASE_IMAGE} AS conda-lock-exclude
# Use this stage for loose requirements during development.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

RUN --mount=type=bind,from=stash,source=/tmp/conda,target=/tmp/conda \
    /bin/bash /tmp/conda/miniconda.sh -b -p /opt/conda && \
    printf "channels:\n  - conda-forge\n  - nodefaults\n" > /opt/conda/.condarc

# `CONDA_MANAGER` may be either `mamba` or `conda`.
ARG CONDA_MANAGER
ENV conda=/opt/conda/bin/${CONDA_MANAGER}

# Use package caching to speed up installs. Borrow `curl` from the git image.
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
RUN --mount=type=bind,from=stash,source=/tmp/req,target=/tmp/req \
    --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    $conda env update --file /tmp/req/environment.yaml

########################################################################
FROM ${CONDA_IMAGE} AS conda-lock
# Use this stage only if installing from `conda-lock`.
# Users must create their own `simple-conda-lock.yaml` file to use this stage.
COPY --link ../reqs/simple-conda-lock.yaml /tmp/conda/lock.yaml

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
# Set default channel to `conda-forge` for both the installing and installed
# `conda` envirnments to prevent any ambiguities during and after installation.
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    printf "channels:\n  - conda-forge\n  - nodefaults\n" > /opt/conda/.condarc && \
    conda create -p /opt/env --copy --file /tmp/conda/lock.yaml && \
    printf "channels:\n  - conda-forge\n  - nodefaults\n" > /opt/env/.condarc

########################################################################
FROM ${BASE_IMAGE} AS conda-lock-include

COPY --link --from=conda-lock /opt/env /opt/conda

########################################################################
FROM conda-lock-${LOCK_MODE} AS install-conda
# Cleanup before copying to reduce image size.
RUN /opt/conda/bin/conda clean -fya && \
    find /opt/conda -name '__pycache__' | xargs rm -rf

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
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=stash,source=/tmp/apt,target=/tmp/apt \
    apt-get update && sed -e 's/#.*//g' -e 's/\r//g' /tmp/apt/requirements.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ARG TZ
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

########################################################################
FROM train-base AS train-interactive-exclude
# Stage used to create images for Kubernetes clusters or for uploading to
# container registries such as Docker Hub. No users or interactive settings.
COPY --link --from=install-conda /opt/conda /opt/conda
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

########################################################################
FROM train-base AS train-interactive-include

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
COPY --link --from=install-conda --chown=${UID}:${GID} /opt/conda /opt/conda
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

USER ${USR}
ARG HOME=/home/${USR}

# Setting the prompt to `pure`.
ARG PURE_PATH=${HOME}/.zsh/pure
COPY --link --from=stash --chown=${UID}:${GID} /opt/zsh/pure ${PURE_PATH}
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
    } >> ${HOME}/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
ARG ZSHS_PATH=${HOME}/.zsh/zsh-syntax-highlighting
COPY --link --from=stash --chown=${UID}:${GID} \
    /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}
RUN echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${HOME}/.zshrc

# Add custom aliases and settings.
RUN {   echo "alias ll='ls -lh'"; \
        echo "alias wns='watch nvidia-smi'"; \
        echo "alias hist='history 1'"; \
    } >> ${HOME}/.zshrc

########################################################################
FROM train-interactive-${INTERACTIVE_MODE} AS train

# Use Intel OpenMP with optimizations. See the documentation for details.
# https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html
ENV KMP_BLOCKTIME=0
ENV KMP_AFFINITY="granularity=fine,compact,1,0"
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ENV MKL_DEBUG_CPU_TYPE=5
# Enable if compilation is included later.
#ENV LD_PRELOAD=/opt/conda/libfakeintel.so:${LD_PRELOAD}

ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:$LD_PRELOAD
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:${PATH}
ENV PYTHONPATH=${PROJECT_ROOT}

WORKDIR ${PROJECT_ROOT}

CMD ["/bin/zsh"]
