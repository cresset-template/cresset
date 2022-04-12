.PHONY: env di up exec rebuild start down ls

# Create a `.dockerignore` file in PWD if it does not exist already or is empty.
# Set to ignore all files except requirements files at project root or `reqs`.
DI_FILE = .dockerignore
di:
	test -s ${DI_FILE} || printf "*\n!reqs/*requirements*.txt\n!*requirements*.txt\n" >> ${DI_FILE}

# Convenience `make` recipes for Docker Compose.
# See URL below for documentation on Docker Compose.
# https://docs.docker.com/engine/reference/commandline/compose
# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes.
# Change `SERVICE` to specify other services and projects.
SERVICE = full
COMMAND = /bin/zsh
PROJECT = "${SERVICE}-$(shell id -un)"
up:  # Start service. Creates a new container from the image. Recommended method of starting services.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up -d ${SERVICE}
rebuild:  # Start service. Rebuilds the image from the Dockerfile before creating a new container.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up --build -d ${SERVICE}
exec:  # Execute service. Enter interactive shell.
	DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} exec ${SERVICE} ${COMMAND}
start:  # Start a stopped service without recreating the container. Useful if the previous container must not deleted.
	docker compose -p ${PROJECT} start ${SERVICE}
down:  # Shut down service and delete containers, volumes, networks, etc.
	docker compose -p ${PROJECT} down
ls:  # List all services.
	docker compose ls -a

# Creates a `.env` file in PWD if it does not exist already or is empty.
# This will help prevent UID/GID bugs in `docker-compose.yaml`,
# which unfortunately cannot use shell outputs in the file.
# Image names have the user name appended to them for user separation.
ENV_FILE = .env
GID = $(shell id -g)
UID = $(shell id -u)
GRP = $(shell id -gn)
USR = $(shell id -un)
IMAGE_NAME = "${SERVICE}-${USR}"
env:
	test -s ${ENV_FILE} || printf "GID=${GID}\nUID=${UID}\nGRP=${GRP}\nUSR=${USR}\nIMAGE_NAME=${IMAGE_NAME}\n" >> ${ENV_FILE}
