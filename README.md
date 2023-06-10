# Cresset: The One Template to Train Them All

[![GitHub stars](https://img.shields.io/github/stars/cresset-template/cresset?style=flat)](https://github.com/cresset-template/cresset/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/cresset-template/cresset?style=flat)](https://github.com/cresset-template/cresset/issues)
[![GitHub forks](https://img.shields.io/github/forks/cresset-template/cresset?style=flat)](https://github.com/cresset-template/cresset/network)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![GitHub license](https://img.shields.io/github/license/cresset-template/cresset?style=flat)](https://github.com/cresset-template/cresset/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7939089.svg)](https://doi.org/10.5281/zenodo.7939089)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2Fcresset-template%2Fcresset)](https://twitter.com/intent/tweet?text=Awesome_Project!!!:&url=https%3A%2F%2Fgithub.com%2Fcresset-template%2Fcresset)

![Cresset Logo](https://github.com/cresset-template/cresset/blob/main/assets/logo.png "Logo")

---

## TL;DR

**_A new MLOps system for deep learning development using Docker Compose
with the aim of providing reproducible and easy-to-use interactive
development environments for deep learning practitioners.
Hopefully, the methods presented here will become
best practice in both academia and industry._**

## Introductory Video (In English)

## [![Weights and Biases Presentation](https://res.cloudinary.com/marcomontalbano/image/upload/v1649474431/video_to_markdown/images/youtube--sW3VxlJl46o-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/sW3VxlJl46o?t=6865 "Weights and Biases Presentation")

## Installation on a New Host

If this is your first time using this project, follow these steps:

1. Install the NVIDIA CUDA [Driver](https://www.nvidia.com/download/index.aspx)
   appropriate for the target host and NVIDIA GPU.
   If the driver has already been installed,
   check that the installed version is compatible with the target CUDA version.
   CUDA driver version mismatch is the single most common issue for new users.
   See the
   [compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
   for compatible versions of the CUDA driver and CUDA Toolkit.

2. Install [Docker](https://docs.docker.com/get-docker) (v23.0+ is recommended)
   or update to a recent version compatible with Docker Compose V2.
   Docker incompatibility with Docker Compose V2 is another common issue for new users.
   Note that Windows users may use WSL (Windows Subsystem for Linux).
   Cresset has been tested on Windows 11 WSL2 with the Windows CUDA driver
   using Docker Desktop for Windows. There is no need to install a separate
   WSL CUDA driver or Docker for Linux inside WSL.
   _N.B._ Windows Security real-time protection causes significant slowdown if enabled.
   Disable any active antivirus programs on Windows for best performance.
   _N.B._ Linux hosts may also install via this
   [repo](https://github.com/docker/docker-install).

3. Run `make install-compose` to install Docker Compose V2 for Linux hosts.
   Installation does _**not**_ require `root` permissions. Visit the
   [documentation](https://docs.docker.com/compose/cli-command/#install-on-linux)
   for the latest installation information. Note that Docker Compose V2
   is available for WSL users with Docker Desktop by default.

4. Run `make env SERVICE=(train|devel|ngc|hub|simple)` on the terminal
   at project root to create a basic `.env` file.
   The `.env` file provides environment variables for `docker-compose.yaml`,
   allowing different users and machines to set their own variables as required.
   The Makefile has also been configured to read values from the `.env` file
   if it exists, allowing non-default values to be specified only once.
   Each host should have a separate `.env` file for host-specific configurations.

5. Run `make over` to create a `docker-compose.override.yaml` file.
   Add configurations that should not be shared via source control there.
   For example, volume-mount pairs specific to each host machine.

6. If Cresset is being placed within a pre-existing project's subdirectory,
   change the `volume` pairing from `.:${PROJECT_ROOT}` to `..:${PROJECT_ROOT}`.
   All commands in Cresset assume that they are being run at project root
   but this can be changed easily.

### Explanation of services

Different Docker Compose services are organized to serve different needs.

- `train`, the default service, should be used when compiled dependencies are
  necessary or when PyTorch needs to be compiled from source due to
  Compute Capability issues, etc.
- `devel` is designed for PyTorch CUDA/C++ developers who need to recompile
  frequently and have many complex dependencies.
- `ngc` is derived from the official NVIDIA PyTorch NGC images with the option
  to install additional packages. It is recommended for users who wish to base
  their projects on the NGC images provided by NVIDIA. Note that the NGC images
  change between different releases and that configurations for one
  release may not work for another one.
- `hub` is derived from the official PyTorch Docker Hub image and serves a
  similar function as the `ngc` service described above. However,
  the PyTorch Docker images have a more stable interface than the NGC images.
- `simple` is derived from the Official Ubuntu Linux image by default as some
  corporations restrict the use of Docker images not officially verified by
  Docker. It installs all packages via `conda` by default and can optionally
  install highly reproducible environments via `conda-lock`. Note that
  `pip` packages can also be installed via `conda`. Also, the base image can
  be configured to use images other than the Official Linux Docker images
  by specifying the `BASE_IMAGE` argument directly in the `.env` file.
  PyTorch runtime performance may be superior in official NVIDIA CUDA images
  under certain circumstances. Use the tests to benchmark runtime speeds.
  **The `simple` service is recommended for users without compiled dependencies.**

The `Makefile` has been configured to take values specified in the `.env` file
if the `.env` file exists. Therefore, all `make` commands will automatically
use the `${SERVICE}` specified by `make env SERVICE=${SERVICE}` after the
`.env` file is created.

### Notes for Rootless Users

Many institutions forbid the use of Docker because it requires `root` permissions, compromising security.
For users without Docker access, using `Podman Compose` is recommended.
`Podman` is an alternative containerization tool developed by RedHat with high compatibility with Docker.
`Podman Compose` is a Python library designed to mimic the functionality of Docker Compose.

**`Podman` and `Podman Compose` do not require `root` permissions by default.**

Run `conda install -c conda-forge podman podman-compose` on a local Conda environment to install the latest versions.
A desktop version of Podman is also available on the [website](https://podman-desktop.io/docs/Installation).

To use Podman Compose, only two changes are needed.

1. Convert all `COPY --link` instructions to `COPY` in the `Dockerfile`s.
   The build system behind Podman, `buildah`, does not support the `--link` flag as of the time of writing.
2. Convert all `docker compose` commands in the `Makefile` recipes to `podman-compose`
   and remove the now redundant Docker-related variables.
   For best results, add the `--podman-build-args='--format docker --jobs 2'` flag to build-related commands.

Note that Podman Compose in rootless mode is much slower than using Docker, even with the additional options.
Also, rootless Podman will not work out-of-the-box. Please refer to the
[documentation](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md) for details.

## Project Configuration

1. To build PyTorch from source, set `BUILD_MODE=include` and the
   CUDA Compute Capability (CCA) of the target NVIDIA GPU in the `.env` file.
   Visit the NVIDIA [website](https://developer.nvidia.com/cuda-gpus#compute)
   to find compute capabilities of NVIDIA GPUs. Visit the
   [documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
   for an explanation of compute capability and its relevance.
   Note that the Docker cache will save previously built binaries
   if the given configurations are identical.

2. Read the `docker-compose.yaml` file to fill in extra variables in `.env`.
   Also, feel free to edit `docker-compose.yaml` as necessary by changing
   session names, hostnames, etc. for different projects and configurations.
   The `docker-compose.yaml` file provides reasonable default values but these
   can be overridden by values specified in the `.env` file.
   An important configuration is `ipc: host`, which allows the container to
   access the shared memory of the host. This is required for multiprocessing,
   e.g., to use `num_workers` in the PyTorch `DataLoader` class.
   Disable this configuration on WSL and specify `shm_size:` instead as WSL
   cannot use host IPC as of the time of writing.

3. Edit requirements in `reqs/apt-train.requirements.txt`
   and `reqs/train-environment.yaml`.
   These contain project package dependencies.
   The `apt` requirements are designed to resemble an
   ordinary Python `requirements.txt` file.

4. Edit the `volumes` section of a service
   to include external directories in the container environment.
   Run `make over` to create a `docker-compose.override.yaml` file
   to add custom volumes and configurations.
   The `docker-compose.override.yaml` file is excluded from version control
   to allow per-user and per-server settings.

5. (Advanced) If an external file must be included in the Docker image build process,
   edit the `.dockerignore` file to allow the Docker context to find the external file.
   By default, all files except requirements
   files are excluded from the Docker build context.

Example `.env` file for user with username `USERNAME`,
group name `GROUPNAME`, user id `1000`, group id `1000` on service `train`.
Use the `simple` service if no dependencies need to be compiled and requirements
can either be downloaded or installed via `apt`, `conda`, or `pip`.

```text
# Generated automatically by `make env`.
GID=1000
UID=1000
GRP=GROUPNAME
USR=USERNAME
PROJECT=train-username             # `PROJECT` must be in lowercase.
SERVICE=train
COMMAND=/bin/zsh                   # Command to execute on starting the container.
IMAGE_NAME=cresset:train-username  # `IMAGE_NAME` is also converted to lowercase.
PROJECT_ROOT=/opt/project

# [[Optional]]: Fill in these configurations manually if the defaults do not suffice.

# NVIDIA GPU Compute Capability (CCA) values may be found at https://developer.nvidia.com/cuda-gpus
CCA=8.6              # Compute capability. CCA=8.6 for RTX3090.
# CCA='8.6+PTX'      # The '+PTX' enables forward compatibility. Multiple CCAs can also be specified.
# CCA='7.5 8.6+PTX'  # Visit https://pytorch.org/docs/stable/cpp_extension.html for details.

# Used only if building PyTorch from source (`BUILD_MODE=include`).
# The `*_TAG` variables are used only if `BUILD_MODE=include`. No effect otherwise.
BUILD_MODE=exclude               # Whether to build PyTorch from source.
PYTORCH_VERSION_TAG=v2.0.0       # Any `git` branch or tag name can be used.
TORCHVISION_VERSION_TAG=v0.15.1

# General environment configurations.
LINUX_DISTRO=ubuntu   # Visit the NVIDIA Docker Hub repo for available base images.
DISTRO_VERSION=22.04  # https://hub.docker.com/r/nvidia/cuda/tags
CUDA_VERSION=11.8.0   # Must be compatible with hardware and CUDA driver.
CUDNN_VERSION=8       # Only major version specifications are available.
PYTHON_VERSION=3.10   # Specify the Python version.
MKL_MODE=include      # Enable MKL for Intel CPUs.
TZ=Asia/Seoul         # Set the container timezone.

# Advanced Usage.
TARGET_STAGE=train        # Target Dockerfile stage. The `*.whl` files are available in `train-builds`.
INTERACTIVE_MODE=include  # Whether to create an interactive image or a static one for publishing.
```

## General Usage After Initial Installation and Configuration

1. Run `make build` to build the image from the Dockerfile and start the service.
   The `make` commands are defined in the
   `Makefile` and target the `train` service by default.
   Run `make up` if the image has already been built and
   rebuilding the image from the Dockerfile is not necessary.
2. Run `make exec` to enter the interactive container environment.
   Using `tmux` inside the container is recommended.
3. There is no step 3. Just start coding.
   Check out the documentation or create an issue if anything goes wrong.

## Makefile Instructions

The Makefile contains shortcuts for common docker compose commands.
Please read the Makefile to see the exact commands.

1. `make build` builds the Docker image from the Dockerfile
   regardless of whether the image already exists.
   This will reinstall packages to the updated requirements files,
   and then recreate the container.
2. `make up` creates a fresh container from the image,
   undoing any changes to the container made by the user.
   Allows changing container settings as network ports,
   mounted volumes, shared memory configurations, etc.
   Recommended method for using this project.
3. `make exec` enters the interactive terminal of the container
   created by `make build` or `make up`.
4. `make down` stops Compose containers and deletes networks.
   Necessary for service teardown.
5. `make start` restarts a stopped container without recreating it.
   Similar to `make up` but does not delete the current container.
   Not recommended unless data saved in container are absolutely necessary.
6. `make ls` shows all Docker Compose services, both active and inactive.
7. `make run` is used for debugging. Containers are removed on exit.
   If a service fails to start, use this to find the error.
8. `make build-only` builds the Docker image from the Dockerfile
   without starting the service.
   It exists to help publish images to container registries.

### Tips

- The `PROJECT`, `SERVICE`, and `COMMAND` variables in the Makefile
  use variables specified in the user's terminal if available.
- If something does not work, first try `make down` to remove the current container and
  then `make up` to create a new container from the image.
  Explicitly tearing the container down is often necessary when something happens to the host.
- If the service startup stalls during `make up`,
  check `docker system df` to see if there is space left on the host machine.
- `make up` is akin to rebooting a computer.
  The current container is removed and a new container is created from the current image.
- `make build` is akin to resetting/formatting a computer.
  The current image, if present, is removed and a new image is built from the Dockerfile,
  after which a container is created from the resulting image.
  In contrast, `make up`
  only creates an image from source if the specified image is not present.
- `make exec` is akin to logging into a computer.
  It is the most important command
  and allows the user to access the container's terminal interactively.
- Configurations such as connected volumes and network ports cannot
  be changed in a running container, requiring a new container to be created.
- Docker automatically caches all builds up to `defaultKeepStorage`.
  Builds use caches from previous builds by default,
  greatly speeding up later builds by only building modified layers.
- If the build fails during `git clone`,
  try `make build` again with a stable internet connection.
- If the build fails during `pip install`,
  check the PyPI mirror URLs and package requirements.
- If any networking issues arise, check `docker network ls` and check for conflicts.
  Most networking and SSH problems can be solved by running `docker network prune`.

## Project Overview

The main components of the project are as follows. The other files are utilities.

1. Dockerfile
2. docker-compose.yaml
3. docker-compose.override.yaml
4. reqs/(`*requirements.txt`|`*environment.yaml`)
5. .env

When the user inputs `make up` or another `make` command,
commands specified in the `Makefile` are executed.
The `Makefile` is used to specify shorthand commands and variables.

When a command related to Docker Compose (e.g., `make build`) is executed,
The `docker-compose.yaml` file and the `.env` file are read by Docker Compose.
The `docker-compose.yaml` file specifies reasonable default values
but users may wish to change them as per their needs.
The values specified in the `.env` file take precedence over
the defaults specified in the `docker-compose.yaml` file.
Environment variables specified in the shell
take precedence over those in the `.env` file.
The `.env` file is deliberately excluded from source control
to allow different users and machines to use different configurations.

The `docker-compose.yaml` file manages configurations,
builds, runs, etc. using the `Dockerfile`.
Visit the Docker Compose [Specification](https://github.com/compose-spec/compose-spec/blob/master/spec.md)
and [Reference](https://docs.docker.com/compose/compose-file/compose-file-v3/) for details.

The `docker-compose.override.yaml` is read by the `docker-compose.yaml` file
during the setup phase. Add configurations specific to each host that should not be
shared via source control such as volume mounts for host-specific paths.

The `Dockerfile` is configured to read only requirements files in the `reqs` directory.
Edit `reqs/pip-train.requirements.txt` to specify Python package requirements.
Edit `reqs/apt-train.requirements.txt` to specify Ubuntu package requirements.
Users must edit the `.dockerignore` file to `COPY` other files into the Docker build,
for example, when building from private code during the Docker build.

The `Dockerfile` uses Docker BuildKit and a multi-stage build where
control flow is specified via stage names and build-time environment variables
given via `docker-compose.yaml`. See the Docker BuildKit
[Syntax](https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md)
for more information on Docker BuildKit.
The `train` service specified in the `docker-compose.yaml` file uses
the `train` stage specified in the `Dockerfile`, which assumes an Ubuntu image.

## _Raison d'Être_

The purpose of this section is to introduce a new paradigm for deep learning development.
The hope is that Cresset, or at least the ideas behind it, will eventually become
best practice for small to medium-scale deep learning research and development.

Developing in local environments with `conda` or `pip`
is commonplace in the deep learning community.
However, this risks rendering the development environment,
and the code meant to run on it, unreproducible.
This state of affairs is a serious detriment to scientific progress
that many readers of this article will have experienced at first-hand.

Docker containers are the standard method for providing reproducible programs
across different computing environments.
They create isolated environments where programs
can run without interference from the host or from one another.
For details, see the
[documentation](https://www.docker.com/resources/what-container).

But in practice, Docker containers are often misused.
Containers are meant to be transient and best practice dictates
that a new container be created for each run.
However, this is very inconvenient for development,
especially for deep learning applications,
where new libraries must constantly be installed and
bugs are often only evident at runtime.
This leads many researchers to develop inside interactive containers.
Docker users often have `run.sh` files with commands such as
`docker run -v my_data:/mnt/data -p 8080:22 -t my_container my_image:latest /bin/bash`
(look familiar, anyone?) and use SSH to connect to running containers.
VSCode even provides a remote development mode to code inside containers.

The problem with this approach is that these interactive containers
become just as unreproducible as local development environments.
A running container cannot connect to a new port or attach a new
[volume](https://docs.docker.com/storage/volumes).
But if the computing environment within the container was created over
several months of installs and builds, the only way to keep it is to
save the container as an image and create a new container from the saved image.
After a few iterations of this process, the resulting images become bloated and
no less scrambled than the local environments that they were meant to replace.

Problems become even more evident when preparing for deployment.
MLOps, defined as a set of practices that aims to deploy and maintain
machine learning models reliably and efficiently, has gained enormous popularity
of late as many practitioners have come to realize the importance of
continuously maintaining ML systems long after the initial development phase ends.

However, bad practices such as those mentioned above mean that much coffee has
been spilled turning research code into anything resembling a production-ready product.
Often, even the original developers cannot recreate the same model after a few months.
Many firms thus have entire teams dedicated to model translation, a huge expenditure.

To alleviate these problems, Docker Compose is proposed as a simple MLOps solution.
Using Docker and Docker Compose, the entire training environment can be reproduced.
Compose has not yet caught on in the deep learning community,
possibly because it is usually advertised as a multi-container solution.
This is a misunderstanding
as it can be used for single-container development just as well.

A `docker-compose.yaml` file is provided for easy management of containers.
**Using the provided `docker-compose.yaml` file will create an interactive environment,
providing a programming experience very similar to using a terminal on a remote server.
Integrations with popular IDEs (PyCharm, VSCode) are also available.**

Moreover, it also allows the user to specify settings for both build and run,
removing the need to manage the environment with custom shell scripts.
Connecting a new volume or port is as simple as removing the current container,
adding a line in the `docker-compose.yaml` file, then running `make up`
to create a new container from the same image.

Build caches allow new images to be built very quickly,
removing another barrier to Docker adoption, the long initial build time.
For more information on Compose, visit the
[documentation](https://docs.docker.com/compose).

Docker [Compose](https://www.compose-spec.io) can also be used for deployment,
which is useful for small to medium-sized deployments.
If and when large-scale deployments using container orchestration such as
Kubernetes becomes necessary, using reproducible Docker environments from
the very beginning will accelerate the development process
and smooth the path to MLOps adoption.
Accelerating time-to-market by streamlining the development process
is a competitive edge for any firm, whether lean startup or tech titan.

With luck, the techniques proposed here will enable
the deep learning community to "_write once, train anywhere_".
But even if most users are not persuaded of the merits of this method,
Many a hapless grad student may be spared from the
sisyphean labor of setting up their `conda` environment,
only to have it crash and burn right before their paper submission is due.

## Compose as Best Practice

Docker Compose is superior to using custom shell scripts for each environment.
Not only does it gather all variables and commands
for both build and run into a single file,
but its native integration with Docker means that it makes complicated
Docker build/run setups simple to implement and use.

Using Docker Compose this way is a general-purpose technique
that does not depend on anything about this project.
The other services available in the project emphasize this point.

### Using Compose with PyCharm and VSCode

The Docker Compose container environment can be used with popular Python IDEs,
not just in the terminal.
PyCharm and Visual Studio Code, both very popular in the deep learning community,
are compatible with Docker Compose.

#### PyCharm (Professional only)

Both Docker and Docker Compose are natively available as Python interpreters.
See tutorials for [Docker](https://www.jetbrains.com/help/pycharm/docker.html) and
[Compose](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#summary)
for details. JetBrains [Gateway](https://www.jetbrains.com/remote-development/gateway)
can also be used to connect to running containers.

_N.B._ PyCharm Professional and other JetBrains IDEs are available
free of charge to anyone with a valid university e-mail address.

#### VSCode

Install the Remote Development extension pack. See
[tutorial](https://code.visualstudio.com/docs/remote/containers-tutorial)
for details.

##### VSCode Tips

VSCode may fail to start up when accessing remote containers created by
Cresset because of the `${HOME}/.vscode-server` volume mounted in the
`docker-compose.yaml` file, which is used to preserve the `.vscode-server`
directory between separate containers.

The reason for VSCode connection failure is that if any host directory
specified as a volume does not exist, Docker will automatically create
the specified host directory with the directory owner set to `root`.
Directories that already exist retain their directory ownership.
When the `.vscode-server` directory is created by Docker this way,
VSCode is unable to install any files in the `.vscode-server` directory.

This has been fixed in the Makefile but problems related to
the `.vscode-server` directory occur frequently.
To solve this problem, simply change the directory ownership to the
user with `sudo chown -R $(id -u):$(id -g) ${HOME}/.vscode-server`.
This command can be run either on the host or inside the container,
which is useful if `sudo` permissions are unavailable on the host.

Also, when one user switches between multiple Cresset-based containers
on a single machine, VSCode may not be able to find the container workspace.
This is because the `docker-compose.yaml` file mounts the host's
`~/.vscode-server` directory to the `/home/${USR}/.vscode-server` directory
of all containers to preserve VSCode extensions between containers.
To fix this issue, create a new directory on the host
to mount the containers' `.vscode-server` directories.
For example, one can set volume pairs as
`${HOME}/.vscode-project1:/home/${USR}/.vscode-server` for project1 and
`${HOME}/.vscode-project2:/home/${USR}/.vscode-server` for project2.
Do not forget to create `${HOME}/.vscode-project1` and
`${HOME}/.vscode-project2` on the host first.
Otherwise, the directory will be owned by `root`,
which will cause VSCode to stall indefinitely due to permission issues.

For other VSCode problems, try deleting `~/.vscode-server` on the host.

# Known Issues

1. Connecting to a running container by `ssh` will remove all variables
   set by `ENV`. This is because `sshd` starts a new environment,
   deleting all previous variables. Using `docker`/`docker compose`
   to enter containers is strongly recommended.

2. `pip install package[option]` will fail on the terminal because of
   Z-shell globbing. Characters such as `[`,`]`,`*`, etc. will be
   interpreted by Z-shell as special commands. Use string literals,
   e.g., `pip install 'package[option]'`, for cross-shell consistency.

3. If the build fails during `git clone`, simply try `make build` again.
   Most of the build will be cached. Failure is probably due to
   networking issues during installation. Updating git submodules is
   [not fail-safe](https://stackoverflow.com/a/8573310/9289275).

4. `torch.cuda.is_available()` will return a
   `... UserWarning: CUDA initialization:...`
   error or the image will simply not start if the host CUDA driver is
   incompatible with the CUDA version on the Docker image.
   Either upgrade the host CUDA driver or downgrade the CUDA version of the image.
   Check the
   [compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
   to see if the host CUDA driver is compatible with the desired version of CUDA.
   Also, check if the CUDA driver has been configured correctly on the host.
   The CUDA driver version can be found using the `nvidia-smi` command.

5. Docker Compose V2 will silently fail if the installed Docker engine
   version is too low on Linux hosts. Update Docker to the latest
   version (23.0+) to use Docker Compose V2.

# Desiderata

1. **MORE STARS**. _**No Contribution Without Appreciation!**_

2. A method of building `Magma` from source would be appreciated.
   Currently, Cresset depends on the `magma-cudaXXX` package
   provided in the PyTorch channel of Anaconda.

3. Bug reports are welcome. Only the latest versions have been tested rigorously.
   Please raise an issue if there are any versions that do not build properly.
   However, please check that your host Docker, Docker Compose,
   and especially NVIDIA Driver are up-to-date before doing so.

4. Translations into other languages and updates to existing translations are welcome.
   Please create a separate `LANG.README.md` file and make a pull request.
