# PyTorch-Template
Template repository for building PyTorch from source.

PyTorch built from source is much faster (as much as x4 times on some benchmarks) 
than PyTorch installed from `pip`/`conda` but building from source is a 
difficult and bug-prone process.

This repository is a highly modular template to build 
any version of PyTorch from source on any version of CUDA.
It provides an easy-to-use Dockerfile which can be integrated 
into any Ubuntu-based image or project.

For researchers unfamiliar with Docker, 
the generated wheel files can be extracted 
to install PyTorch on their local environments.

A Makefile is provided both as an interface for easy use and as 
a tutorial for building custom images.    

## Quick Start

To build an image, first edit the Dockerfile `train` stage to include 
desired packages from `apt`/`conda`/`pip`.

Then, visit https://developer.nvidia.com/cuda-gpus to find the
Compute Capability (CC) of the target GPU device.

Finally, run `make all GPU_CC=TARGET_GPU_CC(s)`.

Examples: (1) `make all GPU_CC="8.6"` for RTX 3090, 
(2) `make all GPU_CC="7.5; 8.6"` for both RTX 2080Ti and RTX 3090 
(building for many GPU CCs will increase build time).

This will result in an image, `pytorch_source:train`, which can be used for training.

Note that CCs for devices not available during the build can be used to build the image.

For example, if the image must be used on an RTX 2080Ti machine but the user only has an RTX 3090, 
the user can set `GPU_CC="7.5"` to enable the image to operate on the RTX 2080Ti GPU.

See https://pytorch.org/docs/stable/cpp_extension.html 
for an in-depth guide on how to set the `TORCH_CUDA_ARCH_LIST` variable, 
which is specified by `GPU_CC` in the `Makefile`.


### Timezone Settings

International users may find this section helpful.

The `train` image has its timezone set by the `TZ` variable using the `tzdata` package.

The default timezone is `Asia/Seoul` but this can be changed by specifying the `TZ` variable when calling `make`.

Use [IANA](https://www.iana.org/time-zones) time zone names to specify the desired timezone.

Example: `make all GPU_CC="8.6" TZ=America/Los_Angeles` to use LA time on the training image.

Note: Only the training image has timezone settings. 
The installation and build images do not use timezone information.

## Multiple Training Images

To use multiple training images on the same host, 
give a different name to `TRAIN_IMAGE_NAME`, 
which has a default value of `train`.

Assuming that the PyTorch build has already been completed, use 
`make build-train 
TORCH_IMAGE_NAME=EXISTING_PYTORCH_IMAGE_NAME 
TRAIN_IMAGE_NAME=YOUR_TRAINING_IMAGE_NAME`
to create new training images without having to rebuild PyTorch.

This is useful for the following use cases.
1. Allowing different users, who have different UID/GIDs, 
to use separate training images.
2. Using different versions of the final training image with 
different library installations and configurations.

Example: Assume that `pytorch_source:build_torch-v1.9.1` has already been created.
Alice would use `make build-train TORCH_IMAGE_NAME=build_torch-v1.9.1 TRAIN_IMAGE_NAME=train_alice` and 
Bob would use `make build-train TORCH_IMAGE_NAME=build_torch-v1.9.1 TRAIN_IMAGE_NAME=train_bob` 
to create a separate image. 

This way, Alice's image would have her UID and GID while Bob's image would have his UID and GID.

This procedure is necessary because training images have their users set during build.

Also, different users may install different libraries in their training images.

Their environment variables and other settings may also be different.


## Specific PyTorch Version

To change the version of PyTorch,
set the `PYTORCH_VERSION_TAG`, `TORCHVISION_VERSION_TAG`, 
`TORCHTEXT_VERSION_TAG`, and `TORCHAUDIO_VERSION_TAG` variables
to matching versions.

The `*_TAG` variables must be GitHub tags or branch names of those repositories.

Visit the GitHub repositories of each library to find the appropriate tags.

__*PyTorch subsidiary libraries only work with matching versions of PyTorch.*__

Example: To build on an RTX 3090 GPU with PyTorch 1.9.1, use the following command:

`make all GPU_CC="8.6" 
TRAIN_IMAGE_NAME=train_torch191
PYTORCH_VERSION_TAG=v1.9.1 
TORCHVISION_VERSION_TAG=v0.10.1 
TORCHTEXT_VERSION_TAG=v0.10.1
TORCHAUDIO_VERSION_TAG=v0.9.1`.

The resulting image is `pytorch_source:train_torch191`, 
which can be used for training with PyTorch 1.9.1 on GPUs with Compute Capability 8.6.
