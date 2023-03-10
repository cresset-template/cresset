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
def enable_cudnn_benchmarking():
    torch.backends.cudnn.benchmark = True


@pytest.fixture(scope="session")
def device(gpu: int = 0) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(gpu)}")
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


@pytest.mark.parametrize(["name", "network_func", "input_shapes"], _configs)
def test_inference_run(
    name: str,
    network_func: Callable[[], nn.Module],
    input_shapes: Sequence[Sequence[int]],
    device: torch.device,
    num_steps: int = 64,
    enable_amp: bool = False,
    enable_scripting: bool = False,
):
    if enable_amp and enable_scripting:
        raise RuntimeError("AMP is incompatible with TorchScript.")
    logger.info("Model: {mn}.", extra={"mn": name})
    logger.info("Input shapes: {shape}.", extra={"shape": input_shapes})
    logger.info("Automatic Mixed Precision Enabled: {amp}.", extra={"amp": enable_amp})
    logger.info("TorchScript Enabled: {script}.", extra={"script": enable_scripting})
    logger.info(
        "Benchmarking Enabled: {benchmark}.", extra={"benchmark": torch.backends.cudnn.benchmark}
    )

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

    logger.info("Average time: {ms:7.3f} milliseconds.", extra={"ms": elapsed_time / num_steps})
    logger.info("Total time: {secs} seconds.", extra={"secs": round(elapsed_time / 1000)})


# Backwards compatibility with legacy Pytorch 1.x versions.
@getattr(torch, "inference_mode", torch.no_grad)()
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
def get_cuda_info(device):  # Using as a fixture to get device info.
    logger.info("Python Version: {ver}", extra={"ver": platform.python_version()})
    logger.info("PyTorch Version: {ver}", extra={"ver": torch.__version__})
    if not torch.cuda.is_available():
        return

    dp = torch.cuda.get_device_properties(device)
    logger.info("PyTorch CUDA Version: {ver}", extra={"ver": torch.version.cuda})
    logger.info("PyTorch cuDNN Version: {ver}", extra={"ver": torch.backends.cudnn.version()})
    logger.info("PyTorch Architecture List: {al}", extra={"al": tuple(torch.cuda.get_arch_list())})
    logger.info("GPU Device Name: {dn}", extra={"dn": dp.name})
    logger.info(
        "GPU Compute Capability: {major}.{minor}", extra={"major": dp.major, "minor": dp.minor}
    )

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
        logger.info("NVIDIA Driver Version: {dv}", extra={"dv": dv})
