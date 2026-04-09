#!/usr/bin/env bash
# Start phage services for local development.
#
# Usage:
#   ./start_services.sh                  # start all CPU services
#   ./start_services.sh --gpu            # start all services (CPU + GPU)
#   ./start_services.sh hmm bacphlip     # start specific services
#
# Services are started as background processes. PIDs are written to .pids
# for easy cleanup. Stop all with:
#   ./start_services.sh --stop

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.service_pids"

# ── Stop mode ───────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        while read -r pid name; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping $name (PID $pid)"
                kill "$pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    else
        echo "No .service_pids file found."
    fi
    exit 0
fi

# ── Parse arguments ─────────────────────────────────────────────────
GPU=false
SERVICES=()
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU=true ;;
        *)     SERVICES+=("$arg") ;;
    esac
done

# Default service list
if [[ ${#SERVICES[@]} -eq 0 ]]; then
    SERVICES=(hmm bacphlip)
    if $GPU; then
        SERVICES+=(megadna deeppl)
    fi
fi

> "$PID_FILE"

start_service() {
    local name="$1" dir="$2" port="$3"
    echo "Starting $name on :$port ..."
    cd "$dir"
    uv sync 2>/dev/null
    uv run uvicorn service:app --host 0.0.0.0 --port "$port" &
    local pid=$!
    echo "$pid $name" >> "$PID_FILE"
    echo "  $name started (PID $pid)"
    cd "$SCRIPT_DIR"
}

start_hmm() {
    echo "Starting hmm on :8002 ..."
    cd "$SCRIPT_DIR/hmm-service"
    uv sync 2>/dev/null
    uv run hmm-service &
    local pid=$!
    echo "$pid hmm" >> "$PID_FILE"
    echo "  hmm started (PID $pid)"
    cd "$SCRIPT_DIR"
}

for svc in "${SERVICES[@]}"; do
    case "$svc" in
        megadna)  start_service megadna  "$SCRIPT_DIR/megadna-service"  8000 ;;
        hmm)      start_hmm ;;
        bacphlip) start_service bacphlip "$SCRIPT_DIR/bacphlip-service" 8003 ;;
        deeppl)   start_service deeppl   "$SCRIPT_DIR/deeppl-service"   8004 ;;
        phabox)   start_service phabox   "$SCRIPT_DIR/phabox-service"   8005 ;;
        *)        echo "Unknown service: $svc" ;;
    esac
done

echo ""
echo "All services started. PIDs in $PID_FILE"
echo "Stop with: $0 --stop"
