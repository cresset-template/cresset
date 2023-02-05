.PHONY: up exec build start down run ls check vs

# Convenience `make` recipes for Docker Compose.
# See URL below for documentation on Docker Compose.
# https://docs.docker.com/engine/reference/commandline/compose

# **Change `SERVICE` to specify other services and projects.**
# `SERVICE`, `COMMAND`, and `PROJECT` take environment variables from
# the user's shell if specified, making it easier to configure commands.
SERVICE ?= train
COMMAND ?= /bin/zsh

# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes,
# which may cause issues if multiple users are using the same account.
# Specify `PROJECT` for the `make` command if this is the case.
_PROJECT = "${SERVICE}-${USR}"
# The `COMPOSE_PROJECT_NAME` variable must be lowercase.
PROJECT ?= $(shell echo ${_PROJECT} | tr "[:upper:]" "[:lower:]")
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

REPOSITORY = cresset
TAG = "${SERVICE}-${USR}"
_IMAGE_NAME = "${REPOSITORY}:${TAG}"
# Image names are made lowercase even though Docker can
# recognize uppercase for cross-platform compatibility.
IMAGE_NAME = $(shell echo ${_IMAGE_NAME} | tr "[:upper:]" "[:lower:]")

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


# Creates VSCode server directory to prevent Docker Compose
# from creating the directory with `root` ownership.
VSCODE_SERVER_PATH = ${HOME}/.vscode-server
vs:
	@mkdir -p ${VSCODE_SERVER_PATH}

OVERRIDE_FILE = docker-compose.override.yaml
# The newline symbol is placed at the start of the line because
# Makefiles do not read the initial spaces otherwise.
# The user's $HOME directory on the host should not be mounted on the
# container's $HOME directory as this would override the configurations
# inside the container with those from the host.
# The home directory is therefore mounted in a separate directory,
# which also serves as an example of how to make volume pairings.
OVERRIDE_BASE = "$\
services:$\
\n  ${SERVICE}:$\
\n    volumes:$\
\n      - $$"{HOME}":/mnt/home$\
\n"
# Create override file for Docker Compose configurations for each user.
# For example, different users may use different host volume directories.
${OVERRIDE_FILE}:
	printf ${OVERRIDE_BASE} >> ${OVERRIDE_FILE}
# Cannot use `override` as a recipe name as it is a `make` keyword.
over: ${OVERRIDE_FILE}

# Rebuilds the image from the Dockerfile before creating a new container.
build: check vs
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
	docker compose -p ${PROJECT} up	--build -d ${SERVICE}
up: check vs  # Start service. Creates a new container from the image.
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
run: check vs  # Used for debugging cases where the service will not start.
	docker compose -p ${PROJECT} run --rm ${SERVICE} ${COMMAND}
ls:  # List all services.
	docker compose ls -a

# Utility for installing Docker Compose on Linux (but not WSL) systems.
# Visit https://docs.docker.com/compose/install for the full documentation.
COMPOSE_VERSION = v2.15.1
COMPOSE_OS_ARCH = linux-x86_64
COMPOSE_URL = https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-${COMPOSE_OS_ARCH}
COMPOSE_PATH = ${HOME}/.docker/cli-plugins
COMPOSE_FILE = ${COMPOSE_PATH}/docker-compose

${COMPOSE_FILE}:
	mkdir -p "${COMPOSE_PATH}"
	curl -SL "${COMPOSE_URL}" -o "${COMPOSE_FILE}"
	chmod +x "${COMPOSE_FILE}"

install-compose: ${COMPOSE_FILE}
