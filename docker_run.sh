#!/bin/bash
# Run GBox CUA Agent in Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
IMAGE_NAME="gbox-cua"
IMAGE_TAG="latest"
CONTAINER_NAME="gbox-cua-container"

# Load environment variables from .env file
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}Loading environment variables from $ENV_FILE${NC}"
    # Use a safer method to load .env file that handles values with spaces
    set -a
    source "$ENV_FILE"
    set +a
else
    echo -e "${YELLOW}Warning: .env file not found at $ENV_FILE${NC}"
    echo -e "${YELLOW}Please copy env.example to .env and configure it:${NC}"
    echo -e "${YELLOW}  cp $SCRIPT_DIR/env.example $SCRIPT_DIR/.env${NC}"
    echo ""
    echo -e "${YELLOW}Continuing with environment variables from current shell...${NC}"
fi

# Check required environment variables
REQUIRED_VARS=("GBOX_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}Error: Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo -e "${RED}  - $var${NC}"
    done
    echo ""
    echo -e "${YELLOW}Please set these variables in .env file or export them:${NC}"
    echo -e "${YELLOW}  export GBOX_API_KEY=your_key${NC}"
    exit 1
fi

# Check VLM provider configuration
if [ "$VLM_PROVIDER" = "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY is required when VLM_PROVIDER=openai${NC}"
    exit 1
elif [ "$VLM_PROVIDER" = "openrouter" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY is required when VLM_PROVIDER=openrouter${NC}"
    exit 1
elif [ "$VLM_PROVIDER" = "vllm" ] && [ -z "$VLLM_API_BASE" ]; then
    echo -e "${RED}Error: VLLM_API_BASE is required when VLM_PROVIDER=vllm${NC}"
    exit 1
fi

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
    echo -e "${YELLOW}Docker image ${IMAGE_NAME}:${IMAGE_TAG} not found.${NC}"
    echo -e "${YELLOW}Building image...${NC}"
    "$SCRIPT_DIR/build_docker.sh"
fi

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping and removing existing container...${NC}"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Prepare Docker command arguments
DOCKER_ARGS=(
    run
    --name "$CONTAINER_NAME"
    --rm
    -it
    --network host
)

# Add environment variables
DOCKER_ARGS+=(-e GBOX_API_KEY="${GBOX_API_KEY}")
DOCKER_ARGS+=(-e VLM_PROVIDER="${VLM_PROVIDER:-openai}")

if [ -n "$OPENAI_API_KEY" ]; then
    DOCKER_ARGS+=(-e OPENAI_API_KEY="${OPENAI_API_KEY}")
fi
if [ -n "$OPENAI_API_BASE" ]; then
    DOCKER_ARGS+=(-e OPENAI_API_BASE="${OPENAI_API_BASE}")
fi
if [ -n "$OPENROUTER_API_KEY" ]; then
    DOCKER_ARGS+=(-e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}")
fi
if [ -n "$OPENROUTER_API_BASE" ]; then
    DOCKER_ARGS+=(-e OPENROUTER_API_BASE="${OPENROUTER_API_BASE}")
fi
if [ -n "$VLLM_API_BASE" ]; then
    DOCKER_ARGS+=(-e VLLM_API_BASE="${VLLM_API_BASE}")
fi
if [ -n "$VLLM_API_KEY" ]; then
    DOCKER_ARGS+=(-e VLLM_API_KEY="${VLLM_API_KEY}")
fi

# Add optional environment variables
[ -n "$MODEL_NAME" ] && DOCKER_ARGS+=(-e MODEL_NAME="${MODEL_NAME}")
[ -n "$MAX_TOKENS" ] && DOCKER_ARGS+=(-e MAX_TOKENS="${MAX_TOKENS}")
[ -n "$TEMPERATURE" ] && DOCKER_ARGS+=(-e TEMPERATURE="${TEMPERATURE}")
[ -n "$TOP_P" ] && DOCKER_ARGS+=(-e TOP_P="${TOP_P}")
[ -n "$MAX_TURNS" ] && DOCKER_ARGS+=(-e MAX_TURNS="${MAX_TURNS}")
[ -n "$BOX_TYPE" ] && DOCKER_ARGS+=(-e BOX_TYPE="${BOX_TYPE}")
[ -n "$LOG_LEVEL" ] && DOCKER_ARGS+=(-e LOG_LEVEL="${LOG_LEVEL}")

# Mount volumes
DOCKER_ARGS+=(-v "$SCRIPT_DIR/logs:/app/logs")

# Image name
DOCKER_ARGS+=("${IMAGE_NAME}:${IMAGE_TAG}")

# Command arguments (can be overridden by command line arguments)
CMD_ARGS=()

# Get task from environment or command line
TASK_ARG="${TASK:-Open the Settings app}"

# If command line arguments provided, use them; otherwise use environment variables
if [ $# -gt 0 ]; then
    # If first argument is a single word and looks like a task, combine all args as task
    # Otherwise, use all args as-is
    if [ $# -eq 1 ] || [[ "$1" =~ ^[a-zA-Z]+$ ]]; then
        # Single word or starts with a word - might be a task, combine all args
        CMD_ARGS=("$@")
    else
        # Multiple args, use as-is
        CMD_ARGS=("$@")
    fi
else
    # Use TASK from env, ensuring it's properly quoted if it contains spaces
    CMD_ARGS=("$TASK_ARG")
    [ "$VERBOSE" = "true" ] && CMD_ARGS+=(--verbose)
    [ -n "$BOX_TYPE" ] && CMD_ARGS+=(--box-type "$BOX_TYPE")
    [ -n "$MODEL_NAME" ] && CMD_ARGS+=(--model "$MODEL_NAME")
    [ -n "$VLM_PROVIDER" ] && CMD_ARGS+=(--vlm-provider "$VLM_PROVIDER")
    [ -n "$VLLM_API_BASE" ] && CMD_ARGS+=(--vllm-api-base "$VLLM_API_BASE")
    [ -n "$OPENAI_API_KEY" ] && CMD_ARGS+=(--openai-api-key "$OPENAI_API_KEY")
    [ -n "$OPENROUTER_API_KEY" ] && CMD_ARGS+=(--openrouter-api-key "$OPENROUTER_API_KEY")
fi

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GBox CUA Agent - Docker Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "${GREEN}Container: ${CONTAINER_NAME}${NC}"
echo -e "${GREEN}VLM Provider: ${VLM_PROVIDER:-openai}${NC}"
echo -e "${GREEN}Model: ${MODEL_NAME:-gpt-4o}${NC}"
echo -e "${GREEN}Task: ${CMD_ARGS[0]:-${TASK_ARG}}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run the container
echo -e "${GREEN}Starting container...${NC}"
docker "${DOCKER_ARGS[@]}" gbox-cua "${CMD_ARGS[@]}"

