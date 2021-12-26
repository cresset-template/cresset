# syntax = docker/dockerfile:1.3.0-labs
# Visit https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# for an up-to-date list of all NVIDIA NGC PyTorch images.
# Default values for `YEAR` and `MONTH` left empty on the Dockerfile by design
# to ~~force~~encourage users to try out the `docker-compose.yaml` file.
ARG YEAR
ARG MONTH
ARG NGC_IMAGE=nvcr.io/nvidia/pytorch:${YEAR}.${MONTH}-py3

FROM ${NGC_IMAGE} AS base
FROM ${NGC_IMAGE} AS ngc

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Set as `ARG` values to reduce the image footprint but not affect resulting images.
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    printf 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
    > /etc/apt/apt.conf.d/keep-cache

ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
RUN if [ $TZ = Asia/Seoul ]; then \
    sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
    printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" \
    > /etc/pip.conf; \
    fi

RUN --mount=type=cache,id=apt-cache-train,target=/var/cache/apt \
    --mount=type=cache,id=apt-lib-train,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        nano \
        openssh-server \
        sudo \
        tmux \
        tzdata \
        zsh && \
    rm -rf /var/lib/apt/lists/*

# Remove pre-existing `conda` owned by root.
# Do not download anything to `/opt/conda` before `conda` is restored.
RUN rm -rf /opt/conda

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu

# This may cause security issues. Use at your own risk.
RUN groupadd -g ${GID} ${GRP} && \
    useradd --shell /bin/zsh --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    printf "%s ALL=(ALL) NOPASSWD:ALL" ${GRP} >> /etc/sudoers && \
    usermod -aG sudo ${USR}

USER ${USR}

# This wierd copy exists to change ownership of the `/opt/conda` directory from root to user.
# The `base` layer exists solely because `--from` can only take literal values.
COPY --from=base --chown=${UID}:${GID} /opt/conda /opt/conda

ENV PIP_DOWNLOAD_CACHE=$HOME/.cache/pip

WORKDIR $HOME/.zsh
RUN git clone https://github.com/sindresorhus/pure.git $HOME/.zsh/pure
RUN printf "fpath+=%s/.zsh/pure\nautoload -Uz promptinit; promptinit\nprompt pure" $HOME >> $HOME/.zshrc

ARG PROJECT_ROOT=/opt/project
WORKDIR ${PROJECT_ROOT}
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}
RUN conda config --set pip_interop_enabled True

COPY --chown=${UID}:${GID} ngc.requirements.txt /tmp/ngc.requirements.txt

# Preserving pip cache by omitting `--no-cache-dir`.
RUN --mount=type=cache,id=pip-ngc,target=${PIP_DOWNLOAD_CACHE},uid=${UID},gid=${GID} \
    python -m pip install \
      -r /tmp/ngc.requirements.txt && \
    rm /tmp/ngc.requirements.txt

CMD ["/bin/zsh"]
