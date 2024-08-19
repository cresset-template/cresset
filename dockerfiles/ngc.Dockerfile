# syntax = docker/dockerfile:1
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

ARG NGC_YEAR
ARG NGC_MONTH
ARG ADD_USER
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

# Copy `apt` and `conda` requirements for ngc images.
COPY --link ../reqs/ngc-apt.requirements.txt /tmp/apt/requirements.txt
COPY --link ../reqs/ngc-environment.yaml /tmp/env/environment.yaml
COPY --link ../reqs/ngc-pip.uninstalls.txt /tmp/pip/uninstalls.txt

########################################################################
FROM ${BASE_IMAGE} AS install-conda
# Starting with the 22.11 PyTorch NGC container, miniforge is removed
# and all Python packages are installed in the default Python environment.
# A separate conda installation is provided to allow conda installation,
# which will also prevent user-installed Python packages from overwriting
# those in the NGC image, which have been carefully configured.
# NGC images prior to 22.11 are **incompatible** with the current Dockerfile.

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

ARG CONDA_URL
ARG CONDA_MANAGER
WORKDIR /tmp/conda

ARG conda=/opt/conda/bin/${CONDA_MANAGER}
RUN curl -fksSL -o /tmp/conda/miniconda.sh ${CONDA_URL} && \
    /bin/bash /tmp/conda/miniconda.sh -b -p /opt/conda && \
    printf "channels:\n  - conda-forge\n  - nodefaults\nssl_verify: false\n" > /opt/conda/.condarc && \
    python=$(python -V | cut -d ' ' -f2) && \
    $conda clean -fya && rm -rf /tmp/conda/miniconda.sh && \
    find /opt/conda -type d -name '__pycache__' | xargs rm -rf

# Install the same version of Python as the system Python in the NGC image.
# The `readwrite` option is necessary for `pip` installation via `conda`.
ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG TRUSTED_HOST
ARG PIP_CONFIG_FILE=/opt/conda/pip.conf
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
ARG CONDA_ENV_FILE=/tmp/env/environment.yaml
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    --mount=type=bind,readwrite,from=stash,source=/tmp/env,target=/tmp/env \
    {   echo "[global]"; \
        echo "index-url=${INDEX_URL}"; \
        echo "extra-index-url=${EXTRA_INDEX_URL}"; \
        echo "trusted-host=${TRUSTED_HOST}"; \
    } > ${PIP_CONFIG_FILE} && \
    $conda env update -p /opt/conda --file ${CONDA_ENV_FILE}

RUN $conda clean -fya && find /opt/conda -type d -name '__pycache__' | xargs rm -rf

########################################################################
FROM ${BASE_IMAGE} AS train-base

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# The base NGC image sets `SHELL=bash`. Docker cannot unset an `ENV` variable,
# therefore, `SHELL=''` is used for best compatibility with the other services.
ENV SHELL=''

# Install `apt` requirements.
# `tzdata` requires noninteractive mode.
ARG TZ
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=stash,source=/tmp/apt,target=/tmp/apt \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    apt-get update && \
    sed -e 's/#.*//g' -e 's/\r//g' /tmp/apt/requirements.txt | \
    xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Remove pre-installed `pip` packages that should use the versions installed via `conda` instead.
RUN --mount=type=bind,from=stash,source=/tmp/pip,target=/tmp/pip \
    python -m pip uninstall -y -r /tmp/pip/uninstalls.txt

########################################################################
FROM train-base AS train-adduser-include

ARG GID
ARG UID
ARG GRP
ARG USR
ARG PASSWD=ubuntu
# Create user with password-free `sudo` permissions.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell $(which zsh) --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Get conda with the directory ownership given to the user.
COPY --link --from=install-conda --chown=${UID}:${GID} /opt/conda /opt/conda

########################################################################
FROM train-base AS train-adduser-exclude
# This stage exists to create images for use in Kubernetes clusters or for
# uploading images to a container registry, where interactive configurations
# are unnecessary and having the user set to `root` is most convenient.
# Most users may safely ignore this stage except when publishing an image
# to a container repository for reproducibility.
# Note that `zsh` configs are available but these images do not require `zsh`.
COPY --link --from=install-conda /opt/conda /opt/conda

########################################################################
FROM train-adduser-${ADD_USER} AS train

ENV KMP_BLOCKTIME=0
# ENV KMP_AFFINITY="granularity=fine,compact,1,0"
# Use `/opt/conda/lib/libiomp5.so` for older NGC images using `conda`.
# Using the older system MKL to prevent version clashes with NGC packages.
ENV LD_PRELOAD=/usr/local/lib/libiomp5.so${LD_PRELOAD:+:${LD_PRELOAD}}

# Enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
# This part requires `gcc`, which may not be present in `runtime` imags.
ENV MKL_DEBUG_CPU_TYPE=5
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /tmp/fakeintel.c && \
    gcc -shared -fPIC -o /usr/local/bin/libfakeintel.so /tmp/fakeintel.c
ENV LD_PRELOAD=/usr/local/bin/libfakeintel.so${LD_PRELOAD:+:${LD_PRELOAD}}
# Configure Jemalloc as the default memory allocator.
ENV LD_PRELOAD=/opt/conda/lib/libjemalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}
ENV MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

ENV ZDOTDIR=/root
ARG PURE_PATH=${ZDOTDIR}/.zsh/pure
ARG ZSHS_PATH=${ZDOTDIR}/.zsh/zsh-syntax-highlighting
COPY --link --from=stash /opt/zsh/pure ${PURE_PATH}
COPY --link --from=stash /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}

ARG TMUX_HIST_LIMIT
# Search for additional Python packages installed via `conda`.
RUN ln -s /opt/conda/lib/$(python -V | awk -F '[ \.]' '{print "python" $2 "." $3}') \
    /opt/conda/lib/python3 && \
    # Create a symbolic link to add Python `site-packages` to `PYTHONPATH`.
    ln -s /usr/local/lib/$(python -V | awk -F '[ \.]' '{print "python" $2 "." $3}') \
    /usr/local/lib/python3 && \
    # Setting the prompt to `pure`.
    {   echo "fpath+=${PURE_PATH}"; \
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
    {   echo "set -g default-command $(which zsh)"; \
        echo "set -g history-limit ${TMUX_HIST_LIMIT}"; \
    } >> /etc/tmux.conf && \
    # For some reason, `tmux` does not read `/etc/tmux.conf`.
    echo 'cp /etc/tmux.conf ${HOME}/.tmux.conf' >> ${ZDOTDIR}/.zprofile && \
    # Change `ZDOTDIR` directory permissions to allow configuration sharing.
    chmod 755 ${ZDOTDIR} && \
    # Clear out `/tmp` and restore its default permissions.
    rm -rf /tmp && mkdir /tmp && chmod 1777 /tmp && \
    ldconfig  # Update dynamic link cache.

# No alternative to adding the `/opt/conda/bin` directory to `PATH`.
# The `conda` binaries are placed at the end of the `PATH` to ensure that
# system Python is used instead of `conda` python unlike in the other services.
# If a `conda` package must have higher priority than a system package,
# explicitly delete the system package as a workaraound.
ENV PATH=${PATH}:/opt/conda/bin

# Configure `PYTHONPATH` to prioritize system packages over `conda` packages to
# prevent conflict when `conda` installs different versions of the same package.
ARG PROJECT_ROOT=/opt/project
ENV PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_ROOT}
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python3/dist-packages
ENV PYTHONPATH=${PYTHONPATH}:/opt/conda/lib/python3/site-packages

WORKDIR ${PROJECT_ROOT}
CMD ["/bin/zsh"]
