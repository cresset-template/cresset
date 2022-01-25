# Requirements

This directory contains `conda`, `apt`, and `pip` 
requirements files for the Dockerfiles.

Using requirements files should minimize 
the need to manually edit the Dockerfiles.

Note that the current project structure only allows the Dockerfile to find
requirements files in the `reqs` directory and project root directory because of the `.dockerignore` file.

To use files in other directories, 
please modify the `.dockerignore` file.
