# Requirements

This directory contains `conda`, `apt`, and `pip` 
requirements files for the Dockerfiles.

Using requirements files should minimize 
the need to manually edit the Dockerfiles.

Note that the current project structure only allows the Dockerfile to find
requirements files in the `reqs` directory and project root directory because of the `.dockerignore` file.

To use files in other directories, 
please modify the `.dockerignore` file.

# Build Dependency Versions

Edit the package versions in `*-build.requirements.txt` if the latest versions cannot be used for older versions of PyTorch and other libraries.

`Setuptools` must be set to `<=59.5.0` for PyTorch `v1.10.x` and below.

`PyYAML` may cause issues for early versions of PyTorch.

More versioning issues will arise with the passing of time but the latest versions of libraries will use the latest versions of their dependencies.
