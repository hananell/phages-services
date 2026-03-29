#!/bin/bash
#SBATCH --job-name=evo2-service
#SBATCH --output=%j_evo2_service.out
#SBATCH --error=%j_evo2_service.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

echo "=== evo2-service job started ==="
echo "Node:    $HOSTNAME"
echo "Date:    $(date)"
echo "GPU:     $SLURM_JOB_GPUS"

# Write node hostname so we can build the SSH tunnel from the client machine
echo "$HOSTNAME" > ~/evo2_node.txt
echo "Node hostname written to ~/evo2_node.txt"

# Install uv if not already available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Make sure uv is on PATH for this session
export PATH="$HOME/.local/bin:$PATH"

# Go to the service directory
cd ~/phages_services/evo2-service

# Sync dependencies (creates .venv, installs all packages)
echo "Syncing dependencies..."
uv sync

echo "Starting evo2-service on port 8001..."
uv run uvicorn service:app --host 0.0.0.0 --port 8001
