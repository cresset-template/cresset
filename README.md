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

See https://pytorch.org/docs/stable/cpp_extension.html 
for an in-depth guide on how to set the `TORCH_CUDA_ARCH_LIST` variable, 
which is specified by `GPU_CC` in the `Makefile`.
