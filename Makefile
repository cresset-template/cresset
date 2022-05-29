.PHONY: env di up exec build rebuild start down run ls guard overrides init

# Convenience `make` recipes for Docker Compose.
# See URL below for documentation on Docker Compose.
# https://docs.docker.com/engine/reference/commandline/compose
# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes.
# Change `SERVICE` to specify other services and projects.
SERVICE = full
COMMAND = /bin/zsh
PROJECT = "${SERVICE}-${USR}"

# Creates a `.env` file in PWD if it does not exist already or is empty.
# This will help prevent UID/GID bugs in `docker-compose.yaml`,
# which unfortunately cannot use shell outputs in the file.
# Image names have the usernames appended to them to prevent
# name collisions between different users.
ENV_FILE = .env
GID = $(shell id -g)
UID = $(shell id -u)
GRP = $(shell id -gn)
USR = $(shell id -un)
IMAGE_NAME = "${SERVICE}-${USR}"
env:
	test -s ${ENV_FILE} || printf "GID=${GID}\nUID=${UID}\nGRP=${GRP}\nUSR=${USR}\nIMAGE_NAME=${IMAGE_NAME}\n" >> ${ENV_FILE}

guard:  # Checks if the `.env` file exists.
	@test -s ${ENV_FILE} || echo "File \`${ENV_FILE}\` does not exist. Run \`make env\` to create \`${ENV_FILE}\`" && test -s ${ENV_FILE}

OVERRIDE_FILE = docker-compose.override.yaml
OVERRIDE_BASE = "services:\n  ${SERVICE}:\n    volumes:"
# Create override file for Docker Compose configurations for each user.
# For example, different users may use different host volume directories.
overrides:
	test -s ${OVERRIDE_FILE} || printf ${OVERRIDE_BASE} >> ${OVERRIDE_FILE}

build: guard  # Start service. Rebuilds the image from the Dockerfile before creating a new container.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up --build -d ${SERVICE}
rebuild: build  # Deprecated alias for `build`.
up: guard  # Start service. Creates a new container from the image.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up -d ${SERVICE}
exec:  # Execute service. Enter interactive shell.
	DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} exec ${SERVICE} ${COMMAND}
start:  # Start a stopped service without recreating the container. Useful if the previous container must not deleted.
	docker compose -p ${PROJECT} start ${SERVICE}
down:  # Shut down service and delete containers, volumes, networks, etc.
	docker compose -p ${PROJECT} down
run: guard  # Used for debugging cases where service will not start.
	docker compose -p ${PROJECT} run ${SERVICE}
ls:  # List all services.
	docker compose ls -a


# Create a `.dockerignore` file in PWD if it does not exist already or is empty.
# Set to ignore all files except requirements files at project root or `reqs`.
DI_FILE = .dockerignore
di:
	test -s ${DI_FILE} || printf "*\n!reqs/*requirements*.txt\n!*requirements*.txt\n" >> ${DI_FILE}