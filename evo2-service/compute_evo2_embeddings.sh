#!/usr/bin/env bash
# compute_evo2_embeddings.sh
#
# Builds the evo2-service Docker image, starts the service, waits until the
# model is loaded, runs Evo2 embedding computation for all phages in the
# unified dataset, then stops the container.
#
# Usage:
#   ./compute_evo2_embeddings.sh
#
# Optional environment overrides:
#   EVO2_SERVICE_DIR    Path to evo2-service repo  (default: script's directory)
#   PHAGES_DATASET_DIR  Path to phages_dataset repo (default: ../../phages_dataset relative to script)
#   CACHE_DIR           Feature cache directory     (default: $PHAGES_DATASET_DIR/cache)
#   HF_HOME             HuggingFace cache directory (default: ~/.cache/huggingface)
#   SERVICE_PORT        Port to expose              (default: 8001)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO2_SERVICE_DIR="${EVO2_SERVICE_DIR:-$SCRIPT_DIR}"
PHAGES_DATASET_DIR="${PHAGES_DATASET_DIR:-$SCRIPT_DIR/../../phages_dataset}"
PHAGES_DATASET_DIR="$(cd "$PHAGES_DATASET_DIR" && pwd)"
CACHE_DIR="${CACHE_DIR:-$PHAGES_DATASET_DIR/cache}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SERVICE_PORT="${SERVICE_PORT:-8001}"
SERVICE_URL="http://localhost:${SERVICE_PORT}"
CONTAINER_NAME="evo2-service"
IMAGE_NAME="evo2-service"

echo "=== evo2 embedding pipeline ==="
echo "  evo2-service dir : $EVO2_SERVICE_DIR"
echo "  phages_dataset   : $PHAGES_DATASET_DIR"
echo "  cache dir        : $CACHE_DIR"
echo "  HuggingFace cache: $HF_HOME"
echo "  service port     : $SERVICE_PORT"
echo ""

# ── Cleanup on exit / interrupt ────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "==> Stopping evo2-service container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    echo "==> Done."
}
trap cleanup EXIT INT TERM

# ── Step 1: Build Docker image ─────────────────────────────────────────────────
echo "==> [1/4] Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" "$EVO2_SERVICE_DIR"

# ── Step 2: Stop any pre-existing container with the same name ─────────────────
if docker ps -q --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    echo "==> Stopping existing '$CONTAINER_NAME' container..."
    docker stop "$CONTAINER_NAME"
fi

# ── Step 3: Start the service ──────────────────────────────────────────────────
echo "==> [2/4] Starting evo2-service on port ${SERVICE_PORT}..."
mkdir -p "$HF_HOME"
docker run \
    --detach \
    --rm \
    --gpus all \
    --name "$CONTAINER_NAME" \
    -p "${SERVICE_PORT}:${SERVICE_PORT}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMAGE_NAME"

# ── Step 4: Wait for model to load ────────────────────────────────────────────
echo "==> [3/4] Waiting for evo2-service to be healthy (model load ~2-5 min)..."
MAX_WAIT=600
ELAPSED=0
until curl -sf "${SERVICE_URL}/health" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('model_loaded') else 1)" \
        2>/dev/null; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Service did not become healthy within ${MAX_WAIT}s."
        echo "Last container logs:"
        docker logs --tail 60 "$CONTAINER_NAME"
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ... still loading (${ELAPSED}s elapsed)"
done
echo "==> evo2-service is healthy and model is loaded."

# ── Step 5: Run embedding computation ─────────────────────────────────────────
echo "==> [4/4] Running Evo2 feature computation (resumable)..."
cd "$PHAGES_DATASET_DIR"
EVO2_SERVICE_URL="$SERVICE_URL" \
    uv run phages features compute --group evo2 --cache "$CACHE_DIR"
