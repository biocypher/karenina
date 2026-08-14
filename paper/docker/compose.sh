#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE_FILE="$SCRIPT_DIR/compose.yaml"
AGENTIC_FILE="$SCRIPT_DIR/compose.agentic.yaml"
CLAUDE_FILE="$SCRIPT_DIR/compose.claude.yaml"

usage() {
    printf '%s\n' \
        "Usage: paper/docker/compose.sh COMMAND [ARGUMENTS]" \
        "" \
        "Commands:" \
        "  build                 Build the Karenina paper controller image" \
        "  build-amd64           Build the controller image for linux/amd64" \
        "  build-agent-images    Build both BixBench sandbox images" \
        "  build-agent-images-amd64" \
        "                        Build both sandbox images for linux/amd64" \
        "  run MODULE [ARGS]     Run a paper Python module" \
        "  run-agentic MODULE    Run BixBench with access to the Docker daemon" \
        "  run-claude MODULE     Run standalone Claude Code with mounted config" \
        "  shell                 Open a shell in the controller image" \
        "  config                Render the resolved base Compose configuration"
}

resolve_data_root() {
    if [ -n "${KARENINA_PAPER_DATA:-}" ]; then
        candidate=$KARENINA_PAPER_DATA
    elif [ -d "$REPOSITORY_ROOT/../karenina-paper-experiments-data" ]; then
        candidate=$REPOSITORY_ROOT/../karenina-paper-experiments-data
    elif [ -d "$REPOSITORY_ROOT/karenina-paper-experiments-data" ]; then
        candidate=$REPOSITORY_ROOT/karenina-paper-experiments-data
    else
        printf '%s\n' \
            "Paper data deposit not found." \
            "Extract karenina-paper-experiments-data beside or inside the repository," \
            "or set KARENINA_PAPER_DATA." >&2
        exit 2
    fi
    if [ ! -d "$candidate" ]; then
        printf 'KARENINA_PAPER_DATA is not a directory: %s\n' "$candidate" >&2
        exit 2
    fi
    KARENINA_PAPER_DATA=$(CDPATH= cd -- "$candidate" && pwd)
    export KARENINA_PAPER_DATA
}

resolve_identity() {
    KARENINA_REPOSITORY_ROOT=$REPOSITORY_ROOT
    KARENINA_HOST_UID=$(id -u)
    KARENINA_HOST_GID=$(id -g)
    export KARENINA_REPOSITORY_ROOT KARENINA_HOST_UID KARENINA_HOST_GID
}

resolve_docker_socket() {
    if ! docker info >/dev/null 2>&1; then
        printf '%s\n' "Docker daemon is not reachable from the current context." >&2
        exit 2
    fi

    case "${DOCKER_HOST:-}" in
        unix://*) client_socket=${DOCKER_HOST#unix://} ;;
        *)
            context_host=$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)
            case "$context_host" in
                unix://*) client_socket=${context_host#unix://} ;;
                *) client_socket=/var/run/docker.sock ;;
            esac
            ;;
    esac

    if [ "$(uname -s)" = Darwin ]; then
        # Docker Desktop and Colima expose a macOS client socket, but bind
        # sources are resolved by the Linux VM daemon. Its socket is at the
        # conventional daemon-side path.
        KARENINA_DOCKER_SOCKET=/var/run/docker.sock
        KARENINA_DOCKER_GID=$(docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            --entrypoint /usr/bin/stat karenina-paper:latest \
            -c %g /var/run/docker.sock)
    else
        KARENINA_DOCKER_SOCKET=$client_socket
        if [ ! -S "$KARENINA_DOCKER_SOCKET" ]; then
            printf 'Docker socket not found: %s\n' "$KARENINA_DOCKER_SOCKET" >&2
            exit 2
        fi
        KARENINA_DOCKER_GID=$(stat -c '%g' "$KARENINA_DOCKER_SOCKET")
    fi
    export KARENINA_DOCKER_SOCKET KARENINA_DOCKER_GID
}

resolve_identity

command=${1:-}
case "$command" in
    build)
        shift
        KARENINA_PAPER_DATA=$REPOSITORY_ROOT
        export KARENINA_PAPER_DATA
        exec docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" build paper "$@"
        ;;
    build-amd64)
        shift
        exec docker buildx build --platform linux/amd64 --load \
            -f "$SCRIPT_DIR/Dockerfile" \
            -t karenina-paper:latest \
            "$@" "$REPOSITORY_ROOT"
        ;;
    build-agent-images)
        shift
        KARENINA_PAPER_DATA=$REPOSITORY_ROOT
        KARENINA_DOCKER_SOCKET=/var/run/docker.sock
        KARENINA_DOCKER_GID=0
        export KARENINA_PAPER_DATA
        export KARENINA_DOCKER_SOCKET KARENINA_DOCKER_GID
        docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" -f "$AGENTIC_FILE" build bixbench-base "$@"
        exec docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" -f "$AGENTIC_FILE" build bixbench-claude "$@"
        ;;
    build-agent-images-amd64)
        shift
        docker buildx build --platform linux/amd64 --load \
            -f "$SCRIPT_DIR/bixbench/Dockerfile" \
            -t karenina-bixbench:latest \
            "$@" "$REPOSITORY_ROOT"
        exec docker buildx build --platform linux/amd64 --load \
            -f "$SCRIPT_DIR/bixbench/Dockerfile.claude" \
            -t karenina-bixbench-claude:latest \
            "$@" "$REPOSITORY_ROOT"
        ;;
    run)
        shift
        resolve_data_root
        exec docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" run --rm paper "$@"
        ;;
    run-agentic)
        shift
        resolve_data_root
        resolve_docker_socket
        exec docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" -f "$AGENTIC_FILE" run --rm paper "$@"
        ;;
    run-claude)
        shift
        resolve_data_root
        if [ -z "${KARENINA_CLAUDE_CONFIG_DIR:-}" ] || [ ! -d "$KARENINA_CLAUDE_CONFIG_DIR" ]; then
            printf '%s\n' \
                "Set KARENINA_CLAUDE_CONFIG_DIR to a Claude Code configuration directory" \
                "that contains authentication and the open-targets MCP server." >&2
            exit 2
        fi
        KARENINA_CLAUDE_CONFIG_DIR=$(CDPATH= cd -- "$KARENINA_CLAUDE_CONFIG_DIR" && pwd)
        export KARENINA_CLAUDE_CONFIG_DIR
        exec docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" -f "$CLAUDE_FILE" run --rm paper "$@"
        ;;
    shell)
        shift
        resolve_data_root
        exec docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" run --rm --entrypoint /bin/bash paper "$@"
        ;;
    config)
        shift
        resolve_data_root
        exec docker compose --project-directory "$REPOSITORY_ROOT" \
            -f "$COMPOSE_FILE" config "$@"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
