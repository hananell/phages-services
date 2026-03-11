#!/usr/bin/env bash
# Build and run the Evo2 embedding service.
#
# The service loads the evo2_7b model (~14GB) on first request.
# The HuggingFace cache is mounted so the model persists between runs.
#
# Docker run flags:
#   --gpus all          Expose all host NVIDIA GPUs to the container.
#   --rm                Remove the container automatically after it exits.
#   -p 8001:8001        Expose the service on host port 8001.
#   -v ~/.cache/huggingface:/root/.cache/huggingface
#                       Mount the host HuggingFace cache so the downloaded
#                       model (~14GB) persists between runs.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE_NAME="evo2-service"

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "Starting evo2-service on port 8001..."
docker run \
    --gpus all \
    --rm \
    -p 8001:8001 \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMAGE_NAME"
