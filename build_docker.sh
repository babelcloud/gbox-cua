#!/bin/bash
# Build Docker image for GBox CUA Agent

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
IMAGE_NAME="gbox-cua"
IMAGE_TAG="latest"
BUILD_CONTEXT="$SCRIPT_DIR"

echo -e "${GREEN}Building GBox CUA Agent Docker image...${NC}"
echo -e "${YELLOW}Image name: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "${YELLOW}Build context: ${BUILD_CONTEXT}${NC}"
echo ""

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found in $SCRIPT_DIR${NC}"
    echo -e "${YELLOW}Please copy env.example to .env and configure it:${NC}"
    echo -e "${YELLOW}  cp $SCRIPT_DIR/env.example $SCRIPT_DIR/.env${NC}"
    echo ""
fi

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$BUILD_CONTEXT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    echo -e "${GREEN}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo -e "${YELLOW}To run the container, use:${NC}"
    echo -e "  $SCRIPT_DIR/docker_run.sh"
else
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
fi

