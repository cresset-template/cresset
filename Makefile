.PHONY: up exec build rebuild start down run ls check init

# Convenience `make` recipes for Docker Compose.
# See URL below for documentation on Docker Compose.
# https://docs.docker.com/engine/reference/commandline/compose

# **Change `SERVICE` to specify other services and projects.**
SERVICE = train
COMMAND = /bin/zsh

# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes.
PROJECT = "${SERVICE}-${USR}"
PROJECT_ROOT = /opt/project

# Creates a `.env` file in PWD if it does not exist.
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

# Makefiles require `$\` at the end of a line for multi-line string values.
# https://www.gnu.org/software/make/manual/html_node/Splitting-Lines.html
ENV_TEXT = "$\
GID=${GID}\n$\
UID=${UID}\n$\
GRP=${GRP}\n$\
USR=${USR}\n$\
IMAGE_NAME=${IMAGE_NAME}\n$\
PROJECT_ROOT=${PROJECT_ROOT}\n$\
"
${ENV_FILE}:  # Creates the `.env` file if it does not exist.
	printf ${ENV_TEXT} >> ${ENV_FILE}

env: ${ENV_FILE}

check:  # Checks if the `.env` file exists.
	@if [ ! -f "${ENV_FILE}" ]; then \
		printf "File \`${ENV_FILE}\` does not exist. " && \
		printf "Run \`make env\` to create \`${ENV_FILE}\`.\n" && \
		exit 1; \
	fi

OVERRIDE_FILE = docker-compose.override.yaml
# Indentation for the next line is included at the end of
# the previous line because Makefiles do not read the initial spaces.
OVERRIDE_BASE = "$\
services:\n  $\
  ${SERVICE}:\n    $\
    volumes:\n      $\
      - \n$\
"
# Create override file for Docker Compose configurations for each user.
# For example, different users may use different host volume directories.
${OVERRIDE_FILE}:
	printf ${OVERRIDE_BASE} >> ${OVERRIDE_FILE}
# Cannot use `override` as a recipe name as it is a `make` keyword.
over: ${OVERRIDE_FILE}

build: check  # Rebuilds the image from the Dockerfile before creating a new container.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
	docker compose -p ${PROJECT} up	--build -d ${SERVICE}
up: check  # Start service. Creates a new container from the image.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
	docker compose -p ${PROJECT} up -d ${SERVICE}
exec:  # Execute service. Enter interactive shell.
	DOCKER_BUILDKIT=1 \
	docker compose -p ${PROJECT} exec ${SERVICE} ${COMMAND}
# Useful if the previous container must not be deleted.
start:  # Start a stopped service without recreating the container.
	docker compose -p ${PROJECT} start ${SERVICE}
down:  # Shut down the service and delete containers, volumes, networks, etc.
	docker compose -p ${PROJECT} down
run: check  # Used for debugging cases where the service will not start.
	docker compose -p ${PROJECT} run ${SERVICE}
ls:  # List all services.
	docker compose ls -a

# Create a `.dockerignore` file in PWD if it does not exist already or is empty.
# Set to ignore all files except requirements files at project root or `reqs`.
DI_FILE = .dockerignore
DI_TEXT = "$\
**\n$\
!*requirements*.txt\n$\
!**/*requirements*.txt\n$\
"
${DI_FILE}:
	printf ${DI_TEXT} >> ${DI_FILE}

di: ${DI_FILE}
