# PyTorch-Template
Template repository for PyTorch projects.

## QuickStart

To build an image, first edit the Dockerfile `train` stage to include 
desired packages from `apt`/`conda`/`pip`.

Then, visit https://developer.nvidia.com/cuda-gpus to find the
Compute Capability (CC) of the target GPU device.

Finally, run `make all GPU_CC=TARGET_GPU_CC(s)`.

Examples: (1) `make all GPU_CC="7.5"` for RTX 2080Ti, 
(2) `make all GPU_CC="7.5; 8.6"` for both RTX 2080Ti and RTX 3090.

Note that building for many GPU CCs increases build times.

See https://pytorch.org/docs/stable/cpp_extension.html 
for an in-depth guide on how to set the `TORCH_CUDA_ARCH_LIST` variable, 
which is specified by `GPU_CC` in the `Makefile`.

To change the version of PyTorch to be built,
set the `PYTORCH_VERSION_TAG`, `TORCHVISION_VERSION_TAG`, 
`TORCHTEXT_VERSION_TAG`, and `TORCHAUDIO_VERSION_TAG` variables
to matching versions.

The `*_TAG` variables must be GitHub tags of those repositories.

Visit the GitHub repositories of each library to find the appropriate tags.
PyTorch subsidiary libraries are designed to work only with their matching version of PyTorch.

Example: To build on an RTX 2080Ti with PyTorch 1.9.1, use the following command.

`make all GPU_CC="7.5" 
PYTORCH_VERSION_TAG=v1.9.1 
TORCHVISION_VERSION_TAG=v0.10.1 
TORCHTEXT_VERSION_TAG=v0.10.1
TORCHAUDIO_VERSION_TAG=v0.9.1`
