# Basic Makefile for starting projects.
# For more sophisticated settings, please use the Dockerfile directly.
# See https://developer.nvidia.com/cuda-gpus to find GPU CCs.
# Also assumes Unix shell for UID, GID.
# See https://pytorch.org/docs/stable/cpp_extension.html
# for an in-depth guide on how to set the `TORCH_CUDA_ARCH_LIST` variable,
# which is specified by `GPU_CC` in the `Makefile`.
GPU_CC                  = "5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
TRAIN_NAME              = train
TZ                      = Asia/Seoul
PYTORCH_VERSION_TAG     = v1.9.1
TORCHVISION_VERSION_TAG = v0.10.1
TORCHTEXT_VERSION_TAG   = v0.10.1
TORCHAUDIO_VERSION_TAG  = v0.9.1
TORCH_NAME              = build_torch-${PYTORCH_VERSION_TAG}

.PHONY: all build-install build-torch build-train

all: build-install build-torch build-train

build-install:
	DOCKER_BUILDKIT=1 docker build \
		--target build-install \
		--tag pytorch_source:build_install \
		- < Dockerfile


build-torch:
	DOCKER_BUILDKIT=1 docker build \
		--target train-builds \
		--cache-from=pytorch_source:build_install \
		--tag pytorch_source:${TORCH_NAME} \
		--build-arg TORCH_CUDA_ARCH_LIST=${GPU_CC} \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		- < Dockerfile

# ARG variables given in `build-torch` (and other previous layers) must be given again here.
# Otherwise, Docker will rebuild `build-torch` but with the default values for ARG variables.
# This will both waste time and cause inconsistency in the training image.
build-train:
	DOCKER_BUILDKIT=1 docker build \
		--target train \
		--cache-from=pytorch_source:${TORCH_NAME} \
		--tag pytorch_source:${TRAIN_NAME} \
		--build-arg TORCH_CUDA_ARCH_LIST=${GPU_CC} \
		--build-arg PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG} \
		--build-arg TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG} \
		--build-arg TORCHTEXT_VERSION_TAG=${TORCHTEXT_VERSION_TAG} \
		--build-arg TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG} \
		--build-arg GID="$(shell id -g)" \
		--build-arg UID="$(shell id -u)" \
		--build-arg TZ=${TZ} \
		- < Dockerfile
