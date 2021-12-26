"""
Benchmarking code for accurate measurement of compute speeds.
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
The hit to code performance from antivirus programs is nontrivial.
"""
from collections import namedtuple
import subprocess
import warnings
from typing import Sequence, Union

import torch
from torch import nn, Tensor
from torchvision.models import (
    vgg19,
    resnet50,
)
from torchvision.models.segmentation import (
    fcn_resnet50,
    deeplabv3_resnet50,
)
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
)
from torchvision.models.video import r3d_18
# Too useful to do without, even if it is an external library.
from tqdm import tqdm

eval_mode = getattr(torch, 'inference_mode', 'no_grad')


@eval_mode()
def _infer(
        network: nn.Module,
        inputs: Sequence[Tensor],
        num_steps: int
) -> float:
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


@eval_mode()
def infer(
        network: nn.Module,
        input_shapes: Sequence[Sequence[int]],
        num_steps: int,
        device: torch.device,
        enable_amp: bool = False,
        enable_scripting: bool = False,
) -> float:
    if enable_amp and enable_scripting:
        raise RuntimeError('AMP is incompatible with TorchScript.')

    network.eval()
    network = network.to(device)
    inputs = tuple(torch.rand(*s, device=device) for s in input_shapes)
    if enable_scripting:
        network = torch.jit.trace(network, inputs)
    if enable_amp:
        with torch.cuda.amp.autocast():
            elapsed_time = _infer(
                network=network,
                inputs=inputs,
                num_steps=num_steps
            )
    else:
        elapsed_time = _infer(
            network=network,
            inputs=inputs,
            num_steps=num_steps
        )
    return elapsed_time


def get_cuda_info(device: Union[torch.device, str, int]) -> dict:
    tv = torch.__version__
    dp = torch.cuda.get_device_properties(device)
    cc = f'{dp.major}.{dp.minor}'  # GPU Compute Capability
    dn = dp.name  # GPU device name (e.g., RTX 3090)
    # The list of architectures that PyTorch was built for.
    al = tuple(torch.cuda.get_arch_list())
    tc = torch.version.cuda  # CUDA version used by PyTorch.
    dv = subprocess.run([
        'nvidia-smi',
        '--query-gpu=driver_version',
        '--format=csv,noheader'
    ], capture_output=True, text=True)
    dv = dv.stdout.strip()  # NVIDIA Driver version.
    info = {
        'PyTorch Version': tv,
        'PyTorch CUDA Version': tc,
        'PyTorch Architecture List': al,
        'GPU Device Name': dn,
        'GPU Compute Capability': cc,
        'NVIDIA Driver Version': dv,
    }
    return info


def get_device(gpu: int) -> torch.device:
    if torch.cuda.is_available() and isinstance(gpu, int):
        device = torch.device(f'cuda:{gpu}')
    else:
        print('No valid GPU was found. Using CPU.')
        device = torch.device('cpu')
    return device


Config = namedtuple('Config', ('name', 'network', 'input_shapes'))

def measure(
        cfgs: Sequence[Config],
        num_steps: int,
        enable_amp: bool = False,
        enable_scripting: bool = False,
        gpu: int = 0,
):
    device = get_device(gpu)
    info = get_cuda_info(device)
    for k, v in info.items():
        print(f'{k}: {v}')
    print(f'Automatic Mixed Precision Enabled: {enable_amp}')
    print(f'TorchScript Enabled: {enable_scripting}')

    for cfg in cfgs:
        ms = infer(
            network=cfg.network,
            input_shapes=cfg.input_shapes,
            num_steps=num_steps,
            device=device,
            enable_amp=enable_amp,
            enable_scripting=enable_scripting
        )
        print(f'\nModel: {cfg.name}')
        print(f'Input shapes: {cfg.input_shapes}')
        print(f'Average time: {ms / num_steps:7.3f} milliseconds.')
        print(f'Total time: {round(ms / 1000):3d} seconds.')


if __name__ == '__main__':
    # Specify model configurations manually.
    configs = [
        Config(
            name='vgg19',
            network=vgg19(),
            input_shapes=((8, 3, 224, 224),)
        ),
        Config(
            name='resnet50',
            network=resnet50(),
            input_shapes=((8, 3, 224, 224),)
        ),
        Config(
            name='fcn_resnet50',
            network=fcn_resnet50(pretrained_backbone=False),
            input_shapes=((1, 3, 512, 512),)
        ),
        Config(
            name='deeplabv3_resnet50',
            network=deeplabv3_resnet50(pretrained_backbone=False),
            input_shapes=((1, 3, 512, 512),)
        ),
        Config(
            name='retinanet_resnet50_fpn',
            network=retinanet_resnet50_fpn(pretrained_backbone=False),
            input_shapes=((1, 3, 512, 512),)
        ),
        Config(
            name='r3d_18',
            network=r3d_18(),
            input_shapes=((1, 3, 64, 128, 128),)
        ),
        Config(
            name='Transformer',
            network=nn.Transformer(),
            input_shapes=((1, 512, 512), (1, 512, 512))
        ),
    ]

    # Start of benchmarking.
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Comment this out to see warnings.
        measure(
            cfgs=configs,
            num_steps=1024,
            enable_amp=False,
            enable_scripting=False,
            gpu=0,
        )
