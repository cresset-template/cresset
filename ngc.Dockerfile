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

ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

ARG DEB_OLD=http://archive.ubuntu.com
ARG DEB_NEW=http://mirror.kakao.com
RUN sed -i "s%${DEB_OLD}%${DEB_NEW}%g" /etc/apt/sources.list

# Copy `apt` requirements.
COPY reqs/apt-ngc.requirements.txt /tmp/reqs/apt-ngc.requirements.txt
RUN apt-get update && sed 's/#.*//' /tmp/reqs/apt-ngc.requirements.txt \
        | tr [:cntrl:] ' ' | xargs -r apt-get install -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        autoconf \
        git \
        openssh-server \
        sudo \
        tzdata \
        zsh && \
    rm -rf /var/lib/apt/lists/* /tmp/reqs/apt-ngc.requirements.txt

ARG JEMALLOC_VERSION_TAG
WORKDIR /opt/jemalloc
# `autogen.sh` requires the `autoconf` package.
RUN git clone https://github.com/jemalloc/jemalloc.git /opt/jemalloc && \
    if [ -n ${JEMALLOC_VERSION_TAG} ]; then \
        git checkout ${JEMALLOC_VERSION_TAG}; \
    fi && \
    ./autogen.sh && make && make install
ENV LD_PRELOAD=/opt/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

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
# USER does not own `/opt/conda` itself but owns all directories within.
# The `base` layer exists solely because `--from` can only take literal values.
COPY --from=base --chown=${UID}:${GID} /opt/conda /opt/conda

# Separate configuration update necessary because NVIDIA's conda
# may have its own `pip.conf` file with it.
ARG INDEX_URL=http://mirror.kakao.com/pypi/simple
ARG TRUSTED_HOST=mirror.kakao.com
# `printf` is preferred over `echo` when escape characters are used due to
# the inconsistent behavior of `echo` across different shells.
RUN printf "[global]\nindex-url=${INDEX_URL}\ntrusted-host=${TRUSTED_HOST}\n" > /opt/conda/pip.conf

WORKDIR $HOME/.zsh
RUN git clone https://github.com/sindresorhus/pure.git $HOME/.zsh/pure && \
    printf "fpath+=$HOME/.zsh/pure\nautoload -Uz promptinit; promptinit\nprompt pure\n" >> $HOME/.zshrc

#RUN git clone https://github.com/zsh-users/zsh-autosuggestions \
#        $HOME/.zsh/zsh-autosuggestions &&  \
#    echo "source $HOME/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" >> $HOME/.zshrc

RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git \
        $HOME/.zsh/zsh-syntax-highlighting && \
    echo "source $HOME/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> $HOME/.zshrc

ARG PROJECT_ROOT=/opt/project
WORKDIR ${PROJECT_ROOT}
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}
COPY --chown=${UID}:${GID} reqs/pip-ngc.requirements.txt /tmp/ngc.requirements.txt
RUN conda config --set pip_interop_enabled True && \
    python -m pip install --no-cache-dir \
        -r /tmp/pip-ngc.requirements.txt && \
    rm /tmp/pip-ngc.requirements.txt && sudo ldconfig

CMD ["/bin/zsh"]
