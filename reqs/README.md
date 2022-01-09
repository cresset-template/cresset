# Requirements

This directory contains `apt` and `pip` 
requirements files for the Dockerfiles.

Using requirements files should minimize 
the need to manually edit the Dockerfiles.

Note that the current project structure only allows the Dockerfile to find
requirements files in the `reqs` directory because of the `.dockerignore` file.

To use requirements files in other directories, modify the `.dockerignore` file.
