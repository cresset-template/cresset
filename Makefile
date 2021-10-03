# Basic Makefile for starting projects.
# For more sophisticated settings, please use the Dockerfile directly.
# See https://developer.nvidia.com/cuda-gpus to find GPU CCs.
# Also assumes Unix shell for UID, GID.
# See https://pytorch.org/docs/stable/cpp_extension.html
# for an in-depth guide on how to set the `TORCH_CUDA_ARCH_LIST` variable,
# which is specified by `GPU_CC` in the `Makefile`.
GPU_CC                  = "5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
TRAIN_IMAGE_NAME        = train
TZ                      = Asia/Seoul
PYTORCH_VERSION_TAG     = v1.9.1
TORCHVISION_VERSION_TAG = v0.10.1
TORCHTEXT_VERSION_TAG   = v0.10.1
TORCHAUDIO_VERSION_TAG  = v0.9.1

.PHONY: all build-install build-train

all: build-install build-train

build-install:
	DOCKER_BUILDKIT=1 docker build \
		--target build-install \
		--tag pytorch_source:build_install \
		- < Dockerfile


# Hopefully, different PyTorch version build images will be automatically cached as intermediate layers.
build-train:
	DOCKER_BUILDKIT=1 docker build \
		--target train \
		--cache-from=pytorch_source:build_install \
		--tag pytorch_source:${TRAIN_IMAGE_NAME} \
		--build-arg TORCH_CUDA_ARCH_LIST=${GPU_CC} \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		--build-arg TZ=${TZ} \
		- < Dockerfile
