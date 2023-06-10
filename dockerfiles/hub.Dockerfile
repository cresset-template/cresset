# syntax = docker/dockerfile:1
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.
ARG PYTORCH_VERSION
ARG CUDA_SHORT_VERSION
ARG CUDNN_VERSION
ARG IMAGE_FLAVOR
ARG IMAGE_TAG=${PYTORCH_VERSION}-cuda${CUDA_SHORT_VERSION}-cudnn${CUDNN_VERSION}-${IMAGE_FLAVOR}
ARG BASE_IMAGE=pytorch/pytorch:${IMAGE_TAG}
ARG INTERACTIVE_MODE
ARG GIT_IMAGE=bitnami/git:latest

########################################################################
FROM ${GIT_IMAGE} AS stash

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

# Copy and install `apt` requirements for hub images.
COPY --link ../reqs/hub-apt.requirements.txt /tmp/apt/requirements.txt
COPY --link ../reqs/hub-conda.requirements.txt /tmp/req/requirements.txt

########################################################################
FROM ${BASE_IMAGE} AS train-base

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Install `apt` requirements.
# `tzdata` requires noninteractive mode.
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=stash,source=/tmp/apt,target=/tmp/apt \
    apt-get update && \
    sed -e 's/#.*//g' -e 's/\r//g' /tmp/apt/requirements.txt | \
    xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# `/opt/conda/bin` is expected to be on the $PATH already.
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
# `conda` is intentionally left as a `root` directory.
# Use `sudo` to install new `conda` packages during development if necessary.
# Configure conda use `conda-forge` and not `defaults`.
# Previous installations are preserved via the `--freeze-installed` flag.
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    --mount=type=bind,from=stash,source=/tmp/req,target=/tmp/req \
    conda install --freeze-installed -n base -c conda-forge \
        --file /tmp/req/requirements.txt && \
    ldconfig  # Run `ldconfig` to update dynamic linking paths.

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
# This part requires `gcc`, which may not be present in `runtime` imags.
ENV MKL_DEBUG_CPU_TYPE=5
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/conda/fakeintel.c && \
    gcc -shared -fPIC -o /opt/conda/libfakeintel.so /opt/conda/fakeintel.c
ENV LD_PRELOAD=/opt/conda/libfakeintel.so:${LD_PRELOAD}

ENV KMP_BLOCKTIME=0
ENV KMP_AFFINITY="granularity=fine,compact,1,0"
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:${LD_PRELOAD}

# Jemalloc configurations.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:${LD_PRELOAD}
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# Set timezone. This is placed in `train-base` for timezone consistency,
# though it may be more appropriate to have it only in interactive mode.
ARG TZ
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

ENV ZDOTDIR=/root
# Setting the prompt to `pure`.
ARG PURE_PATH=${ZDOTDIR}/.zsh/pure
COPY --link --from=stash /opt/zsh/pure ${PURE_PATH}
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
    } >> ${ZDOTDIR}/.zshrc

# Add syntax highlighting. This must be activated after auto-suggestions.
ARG ZSHS_PATH=${ZDOTDIR}/.zsh/zsh-syntax-highlighting
COPY --link --from=stash /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}
RUN echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR}/.zshrc

########################################################################
FROM train-base AS train-interactive-include

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
# Create user with password-free `sudo` permissions.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell $(which zsh) --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add custom aliases and settings.
RUN {   echo "alias ll='ls -lh'"; \
        echo "alias wns='watch nvidia-smi'"; \
        echo "alias hist='history 1'"; \
    } >> ${ZDOTDIR}/.zshrc

########################################################################
FROM train-base AS train-interactive-exclude
# This stage exists to create images for use in Kubernetes clusters or for
# uploading images to a container registry, where interactive configurations
# are unnecessary and having the user set to `root` is most convenient.
# Most users may safely ignore this stage except when publishing an image
# to a container repository for reproducibility.
# Note that `zsh` configs are available but these images do not require `zsh`.

########################################################################
FROM train-interactive-${INTERACTIVE_MODE} AS train

# Change `/root` directory permissions to allow configuration sharing.
RUN chmod 711 /root

ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:${PATH}
ENV PYTHONPATH=${PROJECT_ROOT}
WORKDIR ${PROJECT_ROOT}
CMD ["/bin/zsh"]
