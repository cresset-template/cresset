.PHONY: env di up exec rebuild start down

# Creates a `.env` file in PWD if it does not exist already or is empty.
# This will help prevent UID/GID bugs in `docker-compose.yaml`,
# which unfortunately cannot use shell outputs in the file.
# Image names have the user name appended to them for user separation.
ENV_FILE = .env
IMAGE_NAME_FULL = full
env:
	test -s ${ENV_FILE} || printf "GID=$(shell id -g)\nUID=$(shell id -u)\n\
		IMAGE_NAME_FULL=${IMAGE_NAME_FULL}-$(shell id -un)" >> ${ENV_FILE}

# Create a `.dockerignore` file in PWD if it does not exist already or is empty.
DI_FILE = .dockerignore
di:
	test -s ${DI_FILE} || printf "*\n!reqs/*requirements*.txt\n!*requirements*.txt\n" >> ${DI_FILE}

# Convenience commands for Docker Compose. See URL below for documentation.
# https://docs.docker.com/engine/reference/commandline/compose
# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes.
SERVICE = full
COMMAND = /bin/zsh
PROJECT = "${SERVICE}-$(shell id -un)"
up:  # Start service. Creates a new container from the image.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up -d ${SERVICE}
rebuild:  # Start service. Rebuilds the image before creating a new container.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up --build -d ${SERVICE}
exec:  # Execute service. Enter interactive shell.
	DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} exec ${SERVICE} ${COMMAND}
start:  # Start a stopped service without recreating a new container. Not recommended.
	docker compose -p ${PROJECT} start ${SERVICE}
down:  # Shut down service and delete containers, volumes, networks, etc.
	docker compose -p ${PROJECT} down


########################################################################
########## Building images with the `Makefile` is depricated. ##########
########################################################################

.PHONY: cca all all-full build-install-full build-install build-torch build-train
.PHONY: build-train-clean build-train-full-clean build-torch-full build-train-full

# Prevent builds if `CCA` (Compute Capability) is undefined.
cca:
	test -n "${CCA}" || error "CCA variable (Compute Capability) not defined."

# The following are the default builds for the make commands.
# Compute Capability is specified by the `CCA` variable and
# the build should fail if `CCA` is not specified.
CCA                     =
IMAGE_NAME              = train
TZ                      = Asia/Seoul
PYTORCH_VERSION_TAG     = v1.11.0
TORCHVISION_VERSION_TAG = v0.12.0
TORCHTEXT_VERSION_TAG   = v0.12.0
TORCHAUDIO_VERSION_TAG  = v0.11.0
TORCH_NAME              = build_torch-${PYTORCH_VERSION_TAG}
INSTALL_NAME            = build_install

all: env cca build-install build-torch build-train

build-install:
	DOCKER_BUILDKIT=1 docker build \
		--target build-install \
		--tag pytorch_source:${INSTALL_NAME} \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		-f Dockerfile .

build-torch: cca
	DOCKER_BUILDKIT=1 docker build \
		--target train-builds \
		--cache-from=pytorch_source:${INSTALL_NAME} \
		--tag pytorch_source:${TORCH_NAME} \
		--build-arg TORCH_CUDA_ARCH_LIST="${CCA}" \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		-f Dockerfile .

# Docker build arguments from all previous stages
# must be specified again or otherwise the default values of
# those arguments will be used as the inputs for the Dockerfile.
# This will cause a cache miss, leading to recompilation with the default arguments.
# This both wastes time and, more importantly, causes environment mismatch.
# Both the install and build images should be specified as caches.
# Otherwise, the installation process will cause a cache miss.
build-train: cca
	DOCKER_BUILDKIT=1 docker build \
		--target train \
		--cache-from=pytorch_source:${INSTALL_NAME} \
		--cache-from=pytorch_source:${TORCH_NAME} \
		--tag pytorch_source:${IMAGE_NAME} \
		--build-arg TORCH_CUDA_ARCH_LIST="${CCA}" \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		--build-arg TZ=${TZ} \
		-f Dockerfile .


# The following builds are `full` builds, i.e., builds specifying all available options.
# Settings for CUDA 10 by default as an example.
LINUX_DISTRO      = ubuntu
DISTRO_VERSION    = 18.04
CUDA_VERSION      = 10.2
CUDNN_VERSION     = 8
PYTHON_VERSION    = 3.9
MAGMA_VERSION     = 102  # Magma version must match CUDA version.
TORCH_NAME_FULL   = build_torch-${PYTORCH_VERSION_TAG}-${LINUX_DISTRO}${DISTRO_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-py${PYTHON_VERSION}
INSTALL_NAME_FULL = build_install-${LINUX_DISTRO}${DISTRO_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-py${PYTHON_VERSION}
IMAGE_NAME_FULL   = full

all-full: env cca build-install-full build-torch-full build-train-full

build-install-full:
	DOCKER_BUILDKIT=1 docker build \
		--target build-install \
		--tag pytorch_source:${INSTALL_NAME_FULL} \
		--build-arg LINUX_DISTRO=${LINUX_DISTRO} \
		--build-arg DISTRO_VERSION=${DISTRO_VERSION} \
		--build-arg CUDA_VERSION=${CUDA_VERSION} \
		--build-arg CUDNN_VERSION=${CUDNN_VERSION} \
		--build-arg MAGMA_VERSION=${MAGMA_VERSION} \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		-f Dockerfile .

build-torch-full: cca
	DOCKER_BUILDKIT=1 docker build \
		--target train-builds \
		--cache-from=pytorch_source:${INSTALL_NAME_FULL} \
		--tag pytorch_source:${TORCH_NAME_FULL} \
		--build-arg TORCH_CUDA_ARCH_LIST="${CCA}" \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg LINUX_DISTRO=${LINUX_DISTRO} \
		--build-arg DISTRO_VERSION=${DISTRO_VERSION} \
		--build-arg CUDA_VERSION=${CUDA_VERSION} \
		--build-arg CUDNN_VERSION=${CUDNN_VERSION} \
		--build-arg MAGMA_VERSION=${MAGMA_VERSION} \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		-f Dockerfile .

build-train-full: cca
	DOCKER_BUILDKIT=1 docker build \
		--target train \
		--tag pytorch_source:${IMAGE_NAME_FULL} \
		--cache-from=pytorch_source:${INSTALL_NAME_FULL} \
		--cache-from=pytorch_source:${TORCH_NAME_FULL} \
		--build-arg TORCH_CUDA_ARCH_LIST="${CCA}" \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg LINUX_DISTRO=${LINUX_DISTRO} \
		--build-arg DISTRO_VERSION=${DISTRO_VERSION} \
		--build-arg CUDA_VERSION=${CUDA_VERSION} \
		--build-arg CUDNN_VERSION=${CUDNN_VERSION} \
		--build-arg MAGMA_VERSION=${MAGMA_VERSION} \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		--build-arg TZ=${TZ} \
		-f Dockerfile .

# The following builds are `clean` builds, i.e., builds that do not use caches from previous builds.
# Their main purpose is to test whether the commands work properly without cached runs.
build-train-clean: cca
	DOCKER_BUILDKIT=1 docker build \
		--target train \
		--no-cache \
		--tag pytorch_source:${IMAGE_NAME} \
		--build-arg TORCH_CUDA_ARCH_LIST="${CCA}" \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		--build-arg TZ=${TZ} \
		-f Dockerfile .

build-train-full-clean: cca
	DOCKER_BUILDKIT=1 docker build \
		--target train \
		--no-cache \
		--tag pytorch_source:${IMAGE_NAME_FULL} \
		--build-arg TORCH_CUDA_ARCH_LIST="${CCA}" \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg LINUX_DISTRO=${LINUX_DISTRO} \
		--build-arg DISTRO_VERSION=${DISTRO_VERSION} \
		--build-arg CUDA_VERSION=${CUDA_VERSION} \
		--build-arg CUDNN_VERSION=${CUDNN_VERSION} \
		--build-arg MAGMA_VERSION=${MAGMA_VERSION} \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		--build-arg TZ=${TZ} \
		-f Dockerfile .
