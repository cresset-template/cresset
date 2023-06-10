# syntax = docker/dockerfile:1
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

ARG NGC_YEAR
ARG NGC_MONTH
ARG INTERACTIVE_MODE
ARG GIT_IMAGE=bitnami/git:latest
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:${NGC_YEAR}.${NGC_MONTH}-py3

########################################################################
FROM ${GIT_IMAGE} AS stash

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

# Copy and install `apt` requirements for ngc images.
COPY --link ../reqs/ngc-apt.requirements.txt /tmp/apt/requirements.txt
COPY --link ../reqs/ngc-conda.requirements.txt /tmp/req/requirements.txt

########################################################################
FROM ${BASE_IMAGE} AS install-conda
# Starting with the 22.11 PyTorch NGC container, miniforge is removed
# and all Python packages are installed in the default Python environment.
# A separate conda installation is provided to allow conda installation.
# Also, this will prevent user-installed Python packages from overwriting
# the default packages in the NGC image, which have been carefully configured.

LABEL maintainer=veritas9872@gmail.com
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

ARG CONDA_URL
ARG CONDA_MANAGER
WORKDIR /tmp/conda

# Weird paths necessary because `CONDA_PREFIX` is immutable post-installation.
ARG conda=/opt/_conda/bin/${CONDA_MANAGER}
RUN curl -fvL -o /tmp/conda/miniconda.sh ${CONDA_URL} && \
    /bin/bash /tmp/conda/miniconda.sh -b -p /opt/_conda && \
    printf "channels:\n  - conda-forge\n  - nodefaults\n" > /opt/_conda/.condarc && \
    /opt/_conda/bin/conda clean -fya && rm -rf /tmp/conda/miniconda.sh && \
    find /opt/_conda -type d -name '__pycache__' | xargs rm -rf

# Install the same version of Python as the system Python in the NGC image.
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/_conda/pkgs
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    --mount=type=bind,from=stash,source=/tmp/req,target=/tmp/req \
    $conda create --copy -p /opt/conda python=$(python -V | cut -d ' ' -f2) && \
    $conda install --copy -p /opt/conda --file /tmp/req/requirements.txt && \
    printf "channels:\n  - conda-forge\n  - nodefaults\n" > /opt/conda/.condarc

# Create a symbolic link to add `site-packages` to the PYTHONPATH later.
RUN ln -s \
    /opt/conda/lib/$(python -V | awk -F '[ \.]' '{print "python" $2 "." $3}') \
    /opt/conda/lib/python3

########################################################################
FROM ${BASE_IMAGE} AS train-base

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# The base NGC image sets `SHELL=bash`. Docker cannot unset an `ENV` variable,
# ergo, `SHELL=''` is used for best compatibility with the other services.
ENV SHELL=''

# Install `apt` requirements.
# `tzdata` requires noninteractive mode.
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=stash,source=/tmp/apt,target=/tmp/apt \
    apt-get update && \
    sed -e 's/#.*//g' -e 's/\r//g' /tmp/apt/requirements.txt | \
    xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

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

# Get conda with the directory ownership given to the user.
COPY --link --from=install-conda --chown=${UID}:${GID} /opt/conda /opt/conda

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
COPY --link --from=install-conda /opt/conda /opt/conda

########################################################################
FROM train-interactive-${INTERACTIVE_MODE} AS train

ENV KMP_BLOCKTIME=0
ENV KMP_AFFINITY="granularity=fine,compact,1,0"
# Use `/opt/conda/lib/libiomp5.so` for older NGC images using `conda`.
ENV LD_PRELOAD=/usr/local/lib/libiomp5.so:${LD_PRELOAD}

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
# This part requires `gcc`, which may not be present in `runtime` imags.
ENV MKL_DEBUG_CPU_TYPE=5
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /tmp/fakeintel.c && \
    gcc -shared -fPIC -o /usr/local/bin/libfakeintel.so /tmp/fakeintel.c
ENV LD_PRELOAD=/usr/local/bin/libfakeintel.so:${LD_PRELOAD}

# The `jemalloc` binary location and name may change between NGC versions.
# The `x86_64` architecture is also hard-coded by necessity.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so:${LD_PRELOAD}
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# Change `/root` directory permissions to allow configuration sharing.
RUN chmod 711 /root

ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:${PATH}
# Search for additional Python packages installed via `conda`.
# This requires `/opt/conda/lib/python3` to be created as a symlink.
ENV PYTHONPATH=${PROJECT_ROOT}:/opt/conda/lib/python3/site-packages
WORKDIR ${PROJECT_ROOT}
CMD ["/bin/zsh"]
