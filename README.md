# Cresset: The One Template to Train Them All

[![GitHub stars](https://img.shields.io/github/stars/cresset-template/cresset?style=for-the-badge)](https://github.com/cresset-template/cresset/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/cresset-template/cresset?style=for-the-badge)](https://github.com/cresset-template/cresset/issues)
[![GitHub forks](https://img.shields.io/github/forks/cresset-template/cresset?style=for-the-badge)](https://github.com/cresset-template/cresset/network)
[![GitHub license](https://img.shields.io/github/license/cresset-template/cresset?style=for-the-badge)](https://github.com/cresset-template/cresset/blob/main/LICENSE)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fveritas9872%2FPyTorch-Universal-Docker-Template?style=for-the-badge)](https://twitter.com/intent/tweet?text=Awesome_Project!!!:&url=https%3A%2F%2Fgithub.com%2Fveritas9872%2FPyTorch-Universal-Docker-Template)

**Translations:
[한국어](https://github.com/veritas9872/PyTorch-Universal-Docker-Template/blob/main/KOREAN.README.md)**

#### Notice
The project has been renamed `Cresset`.
Please be aware that the project URL has also also changed.
There are also breaking changes in the API. Compute Capability `CC` has been renamed `CCA`.
My apologies for the inconvenience.
Documentation updates are also planned within a month or two.
Until then, please read the source code and comments carefully.
Ubuntu 16.04 LTS support has been implemented but not tested rigorously.

## TL;DR
__*PyTorch built from source can be x4 faster than a naïve PyTorch install.
This repository provides a template for building PyTorch pip wheel binaries from source 
for any PyTorch version on any CUDA version on any environment. 
These can be used in any project environment, including on local `conda` environments, on any CUDA GPU.*__

__*In addition, a new MLOps paradigm for deep learning development using Docker Compose is also proposed
[here](https://github.com/veritas9872/PyTorch-Universal-Docker-Template#interactive-development--mlops-with-docker-compose).
Hopefully, this method will become best practice in both academia and industry.*__


## Preamble
Recent years have seen tremendous academic effort go into the design and implementation of 
efficient neural networks to cope with the ever-increasing amount 
of data on ever-smaller and more efficient devices.
Yet, as of the time of writing, most deep learning practitioners 
are unaware of even the most basic GPU acceleration techniques.
Especially in academia, many do not even use Automatic Mixed Precision (AMP), 
which can reduce memory requirements to 1/4 and increase speeds by x4~5.
This is the case even though AMP can be enabled without much hassle using the 
[HuggingFace Accelerate](https://github.com/huggingface/accelerate) or 
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) libraries.
The Accelerate library in particular can be integrated into any pre-existing 
PyTorch project with only a few lines of code.

Even the novice who has only just dipped their toes into the mysteries 
of deep learning knows that more compute is a key ingredient for success.
No matter how brilliant the scientist, 
outperforming a rival with x10 more compute is no mean feat.
This template was created with the aim of enabling researchers and engineers without much knowledge of 
GPUs, CUDA, Docker, etc. to squeeze every last drop of performance from their GPUs 
__*using the same hardware and neural networks*.__ 

Although Docker images with PyTorch source builds are already available in 
the official [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch) repository and 
the [NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) repository,
these images have a multitude of other packages installed with them, 
making it difficult to integrate them into pre-existing projects.
Moreover, many practitioners prefer local environments over Docker images.

This project is different from any other. 
It has no additional libraries to work with except for those installed by the user. 
Even better, the generated wheels can be extracted for use in any environment
with no need to use Docker, though the second part of this project provides a 
`docker-compose.yaml` file to make using Docker much easier.

If you are among those who could but only yearn for a quicker end to 
the long hours endured staring at Tensorboard as your models inched past the epochs, 
this project may be the answer to your woes.
When using a source build of PyTorch with the latest version of CUDA, combined with AMP, 
one may achieve compute times __*x10 faster than a naïve PyTorch environment*__. 

I sincerely hope that my project will be of service to practitioners in both academia and industry.
Users who find my work beneficial are more than welcome to show their appreciation by starring this repository.


## Warning
__*Before using this template, first check whether you are actually using your GPU!*__

In most scenarios, slow training is caused by an inefficient Extract, Transform, Load (ETL) pipeline.
Training is slow because the data is not getting to the GPU fast enough, not because the GPU is running slowly.
Run `watch nvidia-smi` to check whether GPU utilization is high enough to justify compute optimizations.
If GPU utilization is low or peaks sporadically, first design an efficient ETL pipeline.
Otherwise, faster compute will not help very much as it will not be the bottleneck.

See https://www.tensorflow.org/guide/data_performance for a guide on designing an efficient ETL pipeline.

The [NVIDIA DALI](https://github.com/NVIDIA/DALI) library may also be helpful. 
The [DALI PyTorch plugin](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_tutorials.html)
provides an API for efficient ETL pipelines in PyTorch.


## Introduction
To use this template for a new project, press the green `Use this template` button on the top.
This is more convenient than forking or cloning this repository.
Delete any unnecessary files and start making your project.

The first part of the `README` will explain the purpose of the 
`Dockerfile` and the advantages of using a custom source build of PyTorch.
The second part proposes a new paradigm for deep learning development using Docker Compose.

PyTorch built from source can be much faster than PyTorch installed from 
`pip`/`conda` but building from source is an arduous and bug-prone process.

This repository is a highly modular template to build 
_**any version of PyTorch**_ from source on _**any version of CUDA**_.
It provides an easy-to-use Dockerfile that can be integrated 
into any Linux-based image or project.

For researchers unfamiliar with Docker, the generated wheel files,
located in `/tmp/dist/`, can be extracted to install 
PyTorch on their local environments.
_**Windows users may also use this project via WSL**_.

A `docker-compose.yaml` file is also provided as a simple MLOps system.
It provides a convenient interactive development experience using Docker.
See [here](https://github.com/veritas9872/PyTorch-Universal-Docker-Template#initial-setup)
to get started with Docker Compose on your system.

The speed gains from this template come from the following factors:
1. Using the latest version of CUDA and associated libraries (cuDNN, cuBLAS, etc.).
2. Using a source build made specifically for the target machine with the latest software customizations
instead of a build that must be compatible with different hardware and software environments.
3. Using the latest version of PyTorch and subsidiary libraries. 
Many users do not update their PyTorch
version because of compatibility issues with their pre-existing environment.
4. Informing users on where to look for solutions to their speed problems 
(this may be the most important factor).

See the [documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation) for details.

Combined with techniques such as AMP and cuDNN benchmarking, 
computational throughput can be increased dramatically 
(possibly x10) __*on the same hardware*__.

Even if you do not wish to use Docker in your project,
you may still find this template useful.

**_The wheel files generated by the build can be used in any Python environment with no dependency on Docker._**

This project can thus be used to generate custom wheel files, 
providing dramatic compute speedups for any environment (conda, pip, etc.).


## Quickstart
__*This project is a template, and users are expected to customize it to fit their needs.
Users are free to customize the `train` stage of the `Dockerfile` as they please. 
However, do not change the `build` stages unless absolutely necessary as this will cause a build cache miss.
If a new package must be built, add a new `build` layer.*__

The code is assumed to be running on a Linux host with 
the necessary NVIDIA Drivers and a recent version of Docker & Docker Compose V2 pre-installed.
If this is not the case, install these first. Older versions may not be compatible with this project.
The NVIDIA drivers are especially prone to error. Please check the 
[compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
to verify that your driver version is compatible with your GPU hardware and the CUDA version of the image.


### Timezone Settings
International users may find this section helpful.

The `train` image has its timezone set by the 
`TZ` variable using the `tzdata` package.
The default timezone is `Asia/Seoul` but this can be changed by 
specifying the `TZ` variable.
Use [IANA](https://www.iana.org/time-zones) 
timezone names to specify the desired timezone.


_N.B._ Only the training image has timezone settings. 
The installation and build images do not use timezone information.

In addition, the training image has `apt` and `pip` 
installation URLs updated for Korean users.
If you wish to speed up your installs, 
please find URLs optimized for your location, 
though the installation caches may make this unnecessary.


## Specific PyTorch Version
__*PyTorch subsidiary libraries only work with matching versions of PyTorch.*__

To change the version of PyTorch,
set the [`PYTORCH_VERSION_TAG`](https://github.com/pytorch/pytorch), 
[`TORCHVISION_VERSION_TAG`](https://github.com/pytorch/vision), 
[`TORCHTEXT_VERSION_TAG`](https://github.com/pytorch/text), and 
[`TORCHAUDIO_VERSION_TAG`](https://github.com/pytorch/audio) 
variables to matching versions.

The `*_TAG` variables must be GitHub tags or branch names of those repositories.
Visit the GitHub repositories of each library to find the appropriate tags.

Example: To build on an RTX 3090 GPU with PyTorch 1.9.1, use the following settings:

`CCA="8.6" 
PYTORCH_VERSION_TAG=v1.9.1 
TORCHVISION_VERSION_TAG=v0.10.1 
TORCHTEXT_VERSION_TAG=v0.10.1
TORCHAUDIO_VERSION_TAG=v0.9.1`.

The resulting image can be used for training with 
PyTorch 1.9.1 on GPUs with Compute Capability 8.6.

### Windows
Windows users may use this template by updating to Windows 11 and installing 
Windows Subsystem for Linux (WSL).
WSL on Windows 11 gives a similar experience to using native Linux.

Cresset has been tested on Windows 11 WSL
with the Windows CUDA driver and Docker Desktop for Windows.
There is no need to install a separate WSL CUDA driver or Docker for Linux inside WSL.

_N.B._ Windows Security real-time protection causes 
significant slowdown if enabled.
Disable any active antivirus programs on Windows for best performance.
However, this will obviously create security risks.


# Interactive Development & MLOps with Docker Compose

## _Raison d'Être_
The purpose of this section is to introduce 
a new paradigm for deep learning development. 
I hope that, eventually, using Docker Compose for 
deep learning development will become best practice.

Developing in local environments with `conda` or `pip` 
is commonplace in the deep learning community.
However, this risks rendering the development environment, 
and the code meant to run on it, unreproducible.
This is a serious detriment to scientific progress
that many readers of this article 
will have experienced at first-hand.

Docker containers are the standard method for
providing reproducible programs 
across different computing environments. 
They create isolated environments where programs 
can run without interference from the host or from one another.
See https://www.docker.com/resources/what-container for details.

But in practice, Docker containers are often misused. 
Containers are meant to be transient.
Best practice dictates that a new container be created for each run.
This, however, is very inconvenient for development, 
especially for deep learning applications, 
where new libraries must constantly be installed and 
bugs are often only evident at runtime.
This leads many researchers to develop inside interactive containers.
Docker users often have `run.sh` files with commands such as
`docker run -v my_data:/mnt/data -p 8080:22 -t my_container my_image:latest /bin/bash`
(look familiar, anyone?) and use SSH to connect to running containers.
VSCode also provides a remote development mode to code inside containers.

The problem with this approach is that these interactive containers 
become just as unreproducible as local development environments.
A running container cannot connect to a new port or attach a new
[volume](https://docs.docker.com/storage/volumes/).
But if the computing environment within the container 
was created over several months of installs and builds, 
the only way to keep it is to save the container as an 
image and create a new container from the saved image.
After a few iterations of this process, 
the resulting images become bloated and 
no less scrambled than the local environments
that they were meant to replace.

Problems become even more evident when preparing for deployment.
MLOps, defined as a set of practices that aims to deploy and maintain 
machine learning models reliably and efficiently,
has gained enormous popularity of late as many practitioners have come to realize the 
importance of continuously maintaining ML systems long after the initial development phase ends.

However, bad practices such as those mentioned above mean that
much coffee has been spilled turning research code into anything resembling a production-ready product.
Often, even the original developers cannot retrain the same model after a few months.
Many firms thus have entire teams dedicated to model translation, a huge expenditure.

To alleviate these problems, 
I propose the use of Docker Compose as a simple MLOps solution.
Using Docker and Docker Compose, the entire training environment can be reproduced.
Compose has not yet caught on in the deep learning community,
possibly because it is usually advertised as a multi-container solution.
This is a misunderstanding as it can be used for single-container development just as well.

A `docker-compose.yaml` file is provided for easy management of containers.
**Using the provided `docker-compose.yaml` file will create an interactive environment,
providing a programming experience very similar to using a terminal on a remote server.
Integrations with popular IDEs (PyCharm, VSCode) are also available.**
Moreover, it also allows the user to specify settings for both build and run,
removing the need to manage the environment with custom shell scripts.
Connecting a new volume is as simple as removing the current container,
adding a line in the `docker-compose.yaml`/`Dockerfile` file, 
then creating a new container from the same image. 
Build caches allow new images to be built very quickly,
removing another barrier to Docker adoption, the long initial build time.
For more information on Compose, visit the [documentation](https://docs.docker.com/compose).

Docker Compose can also be used directly for deployment with swarm mode, 
which is useful for small-scale deployments.
See https://docs.docker.com/engine/swarm for documentation.
If and when large-scale deployments using Kubernetes becomes necessary,
using Docker from the very beginning will accelerate 
the development process and smooth the path to MLOps adoption.
Accelerating time-to-market by streamlining the development process
is a competitive edge for any firm, whether lean startup or tech titan.

With luck, the techniques I propose here will enable 
the deep learning community to "_write once, train anywhere_".
But even if I fail in persuading the majority of users 
of the merits of my method,
I may still spare many a hapless grad student from the 
sisyphean labor of setting up their `conda` environment,
only to have it crash and burn right before their paper submission is due.


### Initial Setup
If this is your first time using this project, follow these steps:

1. Install Docker Compose V2 for Linux as described in https://docs.docker.com/compose/cli-command/#install-on-linux.
Visit the website for the latest installation information.
Installation does **not** require `root` permissions. 
Please check the version and architecture tags in the URL before installing. 
The following commands will install Docker Compose V2 (v2.1.0, Linux x86_64) for a single user.

```
mkdir -p ~/.docker/cli-plugins/
curl -SL https://github.com/docker/compose/releases/download/v2.1.0/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
```

The instructions above are for Linux hosts.
WSL users should instead enable "Use Docker Compose V2" on Docker Desktop for Windows.

2. Run `make env` on the terminal to create a basic `.env` file. 
Environment variables can be saved in a `.env` file placed on the project root,
allowing different projects and different users to set their own variables as required.
To create a basic `.env` file with the UID and GID, run `make env`.
Then read the `docker-compose.yaml` file to fill in extra variables.
Also edit `docker-compose.yaml` as necessary for your project.
Feel free to change session names, hostnames, etc. for different projects and configurations.



### General Usage
Using Docker Compose V2 (see https://docs.docker.com/compose/cli-command),
run the following two commands, where `train` is the default service name 
in the provided `docker-compose.yaml` file.


This will open an interactive shell with settings specified 
by the `train` service in the `docker-compose.yaml` file. 

Example `.env` file for RTX 3090 GPUs:
```
UID=1000
GID=1000
CCA=8.6
```

This is extremely convenient for managing reproducible development environments.
For example, if a new `pip` or `apt` package must be installed for the project,
users can simply edit the `train` layer of the 
`Dockerfile` by adding the package to the 
`apt-get install` or `pip install` commands, 
then run the following command:

`docker compose up -d --build full`.

This will remove the current `full` session, rebuild the image, 
and start a new `full` session.
It will not, however, rebuild PyTorch (assuming no cache miss occurs).
Users thus need only wait a few minutes for the additional downloads, 
which are accelerated by caching and fast mirror URLs.

To stop and restart a service after editing the 
`Dockerfile` or `docker-compose.yaml` file,
simply run `docker compose up -d --build full` again.

To stop services and remove containers, use the following command:

`docker compose down`.

Users with remote servers may use Docker contexts
(see https://docs.docker.com/engine/context/working-with-contexts)
to access their containers from their local environments.
For more information on Docker Compose, see
https://github.com/compose-spec/compose-spec/blob/master/spec.md.
For more information on Docker Compose CLI commands, see
https://docs.docker.com/compose/reference.



## Compose as Best Practice

Docker Compose is a far superior option to using custom shell scripts for each environment.
Not only does it gather all variables and commands for both build and run into a single file,
but its native integration with Docker means that it makes complicated Docker 
build/run setups simple to implement.

I wish to emphasize that using Docker Compose this way is a general-purpose technique 
that does not depend on anything about this project.
As an example, an image from the NVIDIA NGC PyTorch repository 
has been used as the base image in `ngc.Dockerfile`.
The NVIDIA NGC PyTorch images contain many optimizations 
for the latest GPU architectures and provide
a multitude of pre-installed machine learning libraries. 
For those starting new projects, and thus with no dependencies,
using the latest NGC image is recommended.

To use the NGC images, use the following commands:

1. `docker compose up -d ngc`
2. `docker compose exec ngc zsh`

The only difference with the previous example is the session name.


### Using Compose with PyCharm and VSCode

The Docker Compose container environment can be used with popular Python IDEs, not just in the terminal.
PyCharm and Visual Studio Code, both very popular in the deep learning community,
are compatible with Docker Compose.

0. If you are using a remote server, first create a Docker 
[context](https://docs.docker.com/engine/context/working-with-contexts)
to connect your local Docker with the remote Docker.

1. **PyCharm** (Professional only): Both Docker and Docker Compose are natively available as Python interpreters. 
See tutorials for [Docker](https://www.jetbrains.com/help/pycharm/docker.html) and 
[Compose](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#summary) for details.
JetBrains [Gateway](https://www.jetbrains.com/remote-development/gateway)
can also be used to connect to running containers.
JetBrains Fleet IDE, with much more advanced features, will become available in early 2022.
_N.B._ PyCharm Professional and other JetBrains IDEs are available 
free of charge to anyone with a valid university e-mail address.

2. **VSCode**: Install the Remote Development extension pack. 
See [tutorial](https://code.visualstudio.com/docs/remote/containers-tutorial) for details.


# Known Issues

1. Connecting to a running container by `ssh` will remove all variables set by `ENV`.
This is because `sshd` starts a new environment, wiping out all previous variables.
Using `docker`/`docker compose` to enter containers is strongly recommended.

2. WSL users using Compose must disable `ipc: host`. WSL cannot use this option.

3. `torch.cuda.is_available()` will return a `... UserWarning: CUDA initialization:...` error 
or the image will simply not start if the CUDA driver on the host 
is incompatible with the CUDA version on the Docker image.
Either upgrade the host CUDA driver or downgrade the CUDA version of the image.
Check the [compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
to see if the host CUDA driver is compatible with the desired version of CUDA.
Also check if the CUDA driver has been configured correctly on the host.
The CUDA driver version can be found using the `nvidia-smi` command.


# Desiderata

0. **MORE STARS**. If you are reading this, please star this repository immediately.
_**No Contribution Without Appreciation!**_

1. Only the latest versions has been tested rigorously.
Please go to the discussions or raise an issue if there are any versions that do not build properly. 
However, please check that your host Docker, Docker Compose, 
and especially NVIDIA Driver are up-to-date before doing so.
Note that some combinations of PyTorch version and CUDA environment 
may simply be impossible to build because of issues in the underlying source code.

2. Translations into other languages and updates to existing translations are welcome. 
Please make a separate `LANG.README.md` file and create a PR.

3. A method to build `magma` from source would be greatly appreciated.
Although the code for building the `magma` package is available at
https://github.com/pytorch/builder/tree/main/magma,
it is updated several months after a new CUDA version is released.
The NVIDIA NGC images use NVIDIA's in-house build of `magma`.

4. Please feel free to share this project! I wish you good luck and happy coding!
