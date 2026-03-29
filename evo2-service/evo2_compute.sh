#!/bin/bash
#SBATCH --job-name=evo2-compute
#SBATCH --output=%j_evo2_compute.out
#SBATCH --error=%j_evo2_compute.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

echo "=== evo2-compute job started ==="
echo "Node:    $HOSTNAME"
echo "Date:    $(date)"
echo "GPU:     $SLURM_JOB_GPUS"

# Make sure uv is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── 1. Start evo2-service in the background ──────────────────────────────────
echo "Starting evo2-service..."
cd ~/phages_services/evo2-service
# Remove old venv (may have been built with wrong Python version)
rm -rf .venv
uv sync --quiet

# Reinstall transformer-engine with CUDA build (the PyPI stub is empty)
echo "Reinstalling transformer-engine with PyTorch extensions..."
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
uv pip install --no-build-isolation --no-cache "transformer-engine[pytorch]"
uv pip install psutil
uv pip install --no-build-isolation --no-cache flash-attn

uv run uvicorn service:app --host 0.0.0.0 --port 8001 &
SERVICE_PID=$!
echo "evo2-service PID: $SERVICE_PID"

# ── 2. Wait until the service is healthy (model load takes a few minutes) ────
echo "Waiting for evo2-service to become healthy..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8001/health | grep -q '"model_loaded":true'; then
        echo "Service is healthy (attempt $i)"
        break
    fi
    if ! kill -0 $SERVICE_PID 2>/dev/null; then
        echo "ERROR: evo2-service process died!"
        exit 1
    fi
    echo "  attempt $i/120 — waiting 10s..."
    sleep 10
done

# Final check
if ! curl -sf http://localhost:8001/health | grep -q '"model_loaded":true'; then
    echo "ERROR: service did not become healthy after 20 minutes"
    kill $SERVICE_PID 2>/dev/null
    exit 1
fi

# ── 3. Run evo2 feature computation ──────────────────────────────────────────
echo "Starting evo2 embedding computation..."
cd ~/phages_dataset
uv sync --quiet
EVO2_SERVICE_URL=http://localhost:8001 uv run phages features compute --group evo2

echo "=== Computation complete at $(date) ==="

# ── 4. Shut down the service ──────────────────────────────────────────────────
kill $SERVICE_PID 2>/dev/null
wait $SERVICE_PID 2>/dev/null
echo "Service stopped."
