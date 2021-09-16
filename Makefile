# Basic Makefile for starting projects.
# For more sophisticated settings, 
# please use the Dockerfile directly.
# See https://developer.nvidia.com/cuda-gpus for CCs.
# Also assumes Unix shell for UID, GID.
TARGET_GPU_CC=

.PHONY: all build-install build-train

all: build-install build-train

build-install:
	DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--target build-install \
		--tag pytorch_source:build_install \
		- < Dockerfile

build-train:
	DOCKER_BUILDKIT=1 docker build \
		--cache-from=pytorch_source:build_install \
		--target train \
		--tag pytorch_source:train \
		--build-arg TORCH_CUDA_ARCH_LIST=${TARGET_GPU_CC} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		- < Dockerfile
