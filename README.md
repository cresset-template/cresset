# PyTorch-Template
Template repository for PyTorch projects.

## Quick Start

To build an image, first edit the Dockerfile `train` stage to include 
desired packages from `apt`/`conda`/`pip`.

Then, visit https://developer.nvidia.com/cuda-gpus to find the
Compute Capability (CC) of the target GPU device.

Finally, run `make all GPU_CC=TARGET_GPU_CC(s)`.

Examples: (1) `make all GPU_CC="7.5"` for RTX 2080Ti, 
(2) `make all GPU_CC="7.5; 8.6"` for both RTX 2080Ti and RTX 3090.

This will result in an image, `pytorch_source:train`, which can be used for training.

Note that building for many GPU CCs increases build times.

See https://pytorch.org/docs/stable/cpp_extension.html 
for an in-depth guide on how to set the `TORCH_CUDA_ARCH_LIST` variable, 
which is specified by `GPU_CC` in the `Makefile`.


## Multiple Training Images

To use multiple training images on the same host, 
give a different name to `TRAIN_IMAGE_NAME`, 
which has a default value of `train`.

This is useful for the following use cases.
1. Allowing different users, who have different UID/GIDs, 
to use separate training images.
2. Using different versions of the final training image with 
different library installations and configurations.

Example: Alice would use `make all GPU_CC="7.5" TRAIN_IMAGE_NAME=train_alice` and 
Bob would use `make all GPU_CC="7.5" TRAIN_IMAGE_NAME=train_bob` to create a separate image.

This procedure is necessary because training images have their users set during build.


## Specific PyTorch Version

To change the version of PyTorch,
set the `PYTORCH_VERSION_TAG`, `TORCHVISION_VERSION_TAG`, 
`TORCHTEXT_VERSION_TAG`, and `TORCHAUDIO_VERSION_TAG` variables
to matching versions.

The `*_TAG` variables must be GitHub tags or branch names of those repositories.

Visit the GitHub repositories of each library to find the appropriate tags.

__*PyTorch subsidiary libraries only work with matching versions of PyTorch.*__

Example: To build on an RTX 2080Ti GPU with PyTorch 1.9.1, use the following command:

`make all GPU_CC="7.5" 
TRAIN_IMAGE_NAME=train_torch191
PYTORCH_VERSION_TAG=v1.9.1 
TORCHVISION_VERSION_TAG=v0.10.1 
TORCHTEXT_VERSION_TAG=v0.10.1
TORCHAUDIO_VERSION_TAG=v0.9.1`.

The resulting image is `pytorch_source:train_torch191`, 
which can be used for training with PyTorch 1.9.1 on GPUs with Compute Capability 7.5.
