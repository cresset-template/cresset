"""Tests to check if PyTorch can execute in the given environment.

The test logs may also be used as compute speed performance benchmarks.
Compatible with PyTorch 1.7.0+ and TorchVision 0.8.1+.
Add your own benchmark models as necessary.
See link below for an explanation of timing in CUDA.
https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc

To check for GPU utilization, Linux users can use `watch nvidia-smi`.
Windows users using WSL or native Windows can use
`while ($True) {nvidia-smi; sleep 2; clear}` on Powershell.

Model input sizes should be adjusted to saturate
volatile GPU utilization while fitting on GPU memory.
GPU utilization must be 100% for a meaningful comparison.

Windows users should disable Windows Security real-time protection
and other antivirus programs for best performance.
The hit to performance from antivirus programs is nontrivial.

Please note that a clean installation of PyTorch on the same image
as provided in the `Dockerfile` will probably not give any speedup.
Use your environment as you were using it for a fair comparison.
"""
import logging
import os
import platform
import subprocess
from typing import Callable, NamedTuple, Sequence

import pytest
import torch
from torch import Tensor, nn
from torchvision.models import resnet50, vgg19
from torchvision.models.video import r3d_18
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture(scope="session", autouse=True)
def _enable_cudnn_benchmarking():
    torch.backends.cudnn.benchmark = True


@pytest.fixture(scope="session", autouse=True)
def _allow_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@pytest.fixture(scope="session")
def device(pytestconfig) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{pytestconfig.getoption('gpu')}")
    else:
        device = torch.device("cpu")
        msg = "No GPUs found for this container. Please check run configurations."
        logger.critical(msg)
    return device


class Config(NamedTuple):
    # Configuration specifications.
    name: str
    # The network is set to be a function instead of the actual model
    # to allow lazy initialization and removal after each test.
    network_func: Callable[[], nn.Module]
    input_shapes: tuple


# Specify model configurations manually.
_configs = [
    Config(
        name="Transformer",
        network_func=nn.Transformer,
        input_shapes=((1, 512, 512), (1, 512, 512)),
    ),
    Config(name="r3d_18", network_func=r3d_18, input_shapes=((1, 3, 64, 64, 64),)),
    Config(name="resnet50", network_func=resnet50, input_shapes=((2, 3, 512, 512),)),
    Config(name="vgg19", network_func=vgg19, input_shapes=((2, 3, 256, 256),)),
]


@pytest.fixture(scope="session")
def num_steps(pytestconfig):
    return pytestconfig.getoption("num_steps")


@pytest.mark.parametrize(("name", "network_func", "input_shapes"), _configs)
def test_inference_run(
    name: str,
    network_func: Callable[[], nn.Module],
    input_shapes: Sequence[Sequence[int]],
    device: torch.device,
    num_steps,
    enable_amp: bool = False,
    enable_scripting: bool = False,
):
    if enable_amp and enable_scripting:
        msg = "AMP is incompatible with TorchScript."
        raise RuntimeError(msg)
    logger.info(f"Model: {name}.")
    logger.info(f"Input shapes: {input_shapes}.")
    logger.info(f"Automatic Mixed Precision Enabled: {enable_amp}.")
    logger.info(f"TorchScript Enabled: {enable_scripting}.")
    logger.info(f"Benchmarking Enabled: {torch.backends.cudnn.benchmark}.")

    network = network_func()
    network.eval()
    network = network.to(device)
    inputs = tuple(torch.rand(*s, device=device) for s in input_shapes)
    if enable_scripting:
        network = torch.jit.trace(network, inputs)
        network = torch.jit.freeze(network)

    if enable_amp:
        from torch.cuda.amp import autocast

        with autocast():
            elapsed_time = _infer(network=network, inputs=inputs, num_steps=num_steps)
    else:
        elapsed_time = _infer(network=network, inputs=inputs, num_steps=num_steps)

    logger.info(f"Average time: {elapsed_time / num_steps:7.3f} milliseconds.")
    logger.info(f"Total time: {round(elapsed_time / 1000):3d} seconds.")


# Backwards compatibility with legacy Pytorch 1.x versions.
no_grad = getattr(torch, "inference_mode", torch.no_grad)


@no_grad()
def _infer(network: nn.Module, inputs: Sequence[Tensor], num_steps: int) -> float:
    # Initialization
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    # GPU Warmup
    warmup_steps = 16
    for _ in range(warmup_steps):
        network(*inputs)

    # Start measurement.
    tic.record()
    for _ in tqdm(range(num_steps), leave=False):
        network(*inputs)
    toc.record()
    toc.synchronize()
    return tic.elapsed_time(toc)  # Time in milliseconds.


@pytest.fixture(scope="session", autouse=True)
def _get_cuda_info(device):  # Using as a fixture to get device info.
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    if not torch.cuda.is_available():
        return

    dp = torch.cuda.get_device_properties(device)
    logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
    cd = torch.backends.cudnn.version()
    logger.info(f"PyTorch cuDNN Version: {cd}")
    al = tuple(torch.cuda.get_arch_list())
    logger.info(f"PyTorch Architecture List: {al}")
    logger.info(f"GPU Device Name: {dp.name}")
    logger.info(f"GPU Compute Capability: {dp.major}.{dp.minor}")
    # No way to check if the GPU has TF32 hardware, only whether it is allowed.
    mm_tf32 = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") != "0"
    mm_tf32 |= torch.backends.cuda.matmul.allow_tf32
    logger.info(f"MatMul TF32 Allowed: {mm_tf32}")
    logger.info(f"cuDNN TF32 Allowed: {torch.backends.cudnn.allow_tf32}")

    # Python3.7+ required for `subprocess` to work as intended.
    if int(platform.python_version_tuple()[1]) > 6:
        dv = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device.index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        logger.info(f"NVIDIA Driver Version: {dv}")
