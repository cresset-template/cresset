.PHONY: env over build up exec down run ls start
.PHONY: check vs install-compose pre-commit pyre-apply

# Convenience `make` recipes for Docker Compose.
# See URL below for documentation on Docker Compose.
# https://docs.docker.com/engine/reference/commandline/compose

# **Change `SERVICE` to specify other services and projects.**
# Note that variables defined in the host shell are ignored if the
# `.env` file also defines those variables due to the current logic.
SERVICE = train
COMMAND = /bin/zsh

# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes,
# which may cause issues if multiple users are using the same account.
# Specify `PROJECT` for the `make` command if this is the case.
_PROJECT = "${SERVICE}-${USR}"
# The `COMPOSE_PROJECT_NAME` variable must be lowercase.
PROJECT = $(shell echo ${_PROJECT} | tr "[:upper:]" "[:lower:]")
PROJECT_ROOT = /opt/project

# Creates a `.env` file in ${PWD} if it does not exist.
# This will help prevent UID/GID bugs in `docker-compose.yaml`,
# which unfortunately cannot use shell outputs in the file.
# Image names have the usernames appended to them to prevent
# name collisions between different users.
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
PROJECT=${PROJECT}\n$\
SERVICE=${SERVICE}\n$\
COMMAND=${COMMAND}\n$\
IMAGE_NAME=${IMAGE_NAME}\n$\
PROJECT_ROOT=${PROJECT_ROOT}\n$\
"

# The `.env` file must be checked via shell as is cannot be a Makefile target.
# Doing so would make it impossible to reference `.env` in the `-include` command.
env:  # Creates the `.env` file if it does not exist.
	@if [ -f ${ENV_FILE} ]; then echo "\`${ENV_FILE}\` already exists!"; \
  	else printf ${ENV_TEXT} >> ${ENV_FILE}; fi

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

# Optionally read variables from the environment file if it exists.
# This line must be placed after all other variable definitions to allow
# variables in the `${ENV_FILE}` to be overridden by user-defined values.
ENV_FILE = .env
-include ${ENV_FILE}

build: check vs # Rebuild the image before creating a new container.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
	docker compose -p ${PROJECT} up	--build -d ${SERVICE}
build-only: check # Build the image without creating a new container.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
	docker compose -p ${PROJECT} build ${SERVICE}
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
COMPOSE_VERSION = v2.18.1
COMPOSE_OS_ARCH = linux-x86_64
COMPOSE_URL = https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-${COMPOSE_OS_ARCH}
COMPOSE_PATH = ${HOME}/.docker/cli-plugins
COMPOSE_FILE = ${COMPOSE_PATH}/docker-compose

${COMPOSE_FILE}:
	mkdir -p "${COMPOSE_PATH}"
	curl -fvL "${COMPOSE_URL}" -o "${COMPOSE_FILE}"
	chmod +x "${COMPOSE_FILE}"

install-compose: ${COMPOSE_FILE}

pre-commit:
	pre-commit run --all-files

PYRE_CONFIGURATION = .pyre_configuration
${PYRE_CONFIGURATION}:
	pyre init

# Perform static analysis on the codebase and
# apply the annotations to the code in-place.
pyre-apply: ${PYRE_CONFIGURATION}
	pyre infer -i
