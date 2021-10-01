# Basic Makefile for starting projects.
# For more sophisticated settings, 
# please use the Dockerfile directly.
# See https://developer.nvidia.com/cuda-gpus for CCs.
# Also assumes Unix shell for UID, GID.
GPU_CC                  =
PYTORCH_VERSION_TAG     = v1.9.1
TORCHVISION_VERSION_TAG = v0.10.1
TORCHTEXT_VERSION_TAG   = v0.10.1
TORCHAUDIO_VERSION_TAG  = v0.9.1

.PHONY: all build-install build-torch build-train

all: build-install build-torch build-train

build-install:
	DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--target build-install \
		--tag pytorch_source:build_install \
		- < Dockerfile

build-torch:
	DOCKER_BUILDKIT=1 docker build \
		--cache-from=pytorch_source:build_install \
		--target build-torch \
		--tag pytorch_source:build_torch \
		--build-arg TORCH_CUDA_ARCH_LIST=${GPU_CC} \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		- < Dockerfile

build-train:
	DOCKER_BUILDKIT=1 docker build \
		--cache-from=pytorch_source:build_torch \
		--target train \
		--tag pytorch_source:train \
		--build-arg TORCH_CUDA_ARCH_LIST=${GPU_CC} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		- < Dockerfile
