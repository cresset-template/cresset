# syntax = docker/dockerfile:1.3.0-labs
# Visit https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# for an up-to-date list of all NVIDIA NGC PyTorch images.
# Default values for `YEAR` and `MONTH` left empty on the Dockerfile by design
# to ~~force~~encourage users to try out the `docker-compose.yaml` file.
ARG YEAR
ARG MONTH
ARG NGC_IMAGE=nvcr.io/nvidia/pytorch:${YEAR}.${MONTH}-py3

########################################################################
FROM ${NGC_IMAGE} AS base
FROM ${NGC_IMAGE} AS ngc

LABEL maintainer="veritas9872@gmail.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Set as `ARG` values to reduce the image footprint but not affect resulting images.
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com

# Copy `apt` requirements.
COPY reqs/apt-ngc.requirements.txt /tmp/apt-ngc.requirements.txt
RUN sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list && \
    apt-get update && \
    sed 's/#.*//g; s/\r//g' /tmp/apt-ngc.requirements.txt | \
    xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* /tmp/apt-ngc.requirements.txt

# Delete conda so that it can be replaced by a version owned by USER later.
RUN rm -rf /opt/conda

ARG GID
ARG UID
ARG GRP=user
ARG USR=user
ARG PASSWD=ubuntu
RUN groupadd -g ${GID} ${GRP} && \
    useradd --shell /bin/zsh --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USR}

# This wierd copy changes the ownership of the `/opt/conda` directory contents.
# The `base` layer exists solely because `--from` can only take literal values.
COPY --from=base --chown=${UID}:${GID} /opt/conda /opt/conda

ARG HOME=/home/${USR}
WORKDIR $HOME/.zsh
RUN git clone https://github.com/sindresorhus/pure.git $HOME/.zsh/pure && \
    printf "fpath+=$HOME/.zsh/pure\nautoload -Uz promptinit; promptinit\nprompt pure\n" >> $HOME/.zshrc

RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git \
        $HOME/.zsh/zsh-syntax-highlighting && \
    echo "source $HOME/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> $HOME/.zshrc

# Separate configuration update necessary because NVIDIA's conda
# may have its own `pip.conf` file with it.
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}
# `printf` is preferred over `echo` when escape characters are used due to
# the inconsistent behavior of `echo` across different shells.
COPY --chown=${UID}:${GID} reqs/pip-ngc.requirements.txt /tmp/pip-ngc.requirements.txt
RUN printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" \
        > /opt/conda/pip.conf && \
    python -m pip install --no-cache-dir \
        -r /tmp/pip-ngc.requirements.txt && \
    rm /tmp/pip-ngc.requirements.txt

ENV OMP_PROC_BIND=CLOSE
ENV OMP_SCHEDULE=STATIC
ENV KMP_WARNINGS=0
ENV KMP_BLOCKTIME=0
ENV LD_PRELOAD=/opt/conda/lib/libiomp5.so:$LD_PRELOAD
ENV KMP_AFFINITY="granularity=fine,nonverbose,compact,1,0"
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
ENV MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000

USER root
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig
USER ${USR}

WORKDIR ${PROJECT_ROOT}

CMD ["/bin/zsh"]
