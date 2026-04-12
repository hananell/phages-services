# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# All services (CPU only)
./start_services.sh

# All services including GPU
./start_services.sh --gpu

# Specific services
./start_services.sh hmm megadna

# Stop all
./start_services.sh --stop

# Docker Compose
docker compose up                   # CPU services
docker compose --profile gpu up     # include GPU services

# Individual service
cd <service-name> && uv sync && uv run uvicorn service:app --host 0.0.0.0 --port <port>
```

## Tests

Each service has its own test suite with mocked dependencies (no GPU/model/DB needed):
```bash
cd <service-name> && uv run pytest tests/ -v
```

## Service Map

| Service | Port | Python | GPU | External deps |
|---------|------|--------|-----|---------------|
| megadna-service | 8000 | 3.12 | ~2GB VRAM | megaDNA model weights (`external/megaDNA`) |
| hmm-service | 8002 | 3.12 | no | PHROGs HMM database |
| bacphlip-service | 8003 | 3.9 | no | HMMER3 binary (`apt-get install hmmer`) |
| deeppl-service | 8004 | 3.10+ | ~1GB VRAM | DNABERT model weights |
| phabox-service | 8005 | 3.10+ | no | PhaBOX2 conda env + database (`download_db.sh`) |

## Architecture

- Each service is a standalone FastAPI app in `service.py` with `settings.py` (pydantic-settings, env var prefix `<SERVICE>_*`). **Exception:** hmm-service uses `src/hmm_service/main.py` + `config.py` instead of the root-level `service.py`/`settings.py` pattern.
- **Shared contracts** (`contracts/`): Pydantic request/response models used by both services and `phages_dataset` feature calculators. Install via `uv add phages-contracts --path ../contracts`.
- **Config loading priority**: env vars > `.env` file > `config.yaml` > defaults in `settings.py`.
- All services expose `GET /health` for health checks.
- `start_services.sh` writes PIDs to `.service_pids` for process management.

## Key Constraints

- **bacphlip-service requires Python 3.9** due to pinned `scikit-learn==0.24.2` dependency from the upstream bacphlip package.
- **megadna-service** depends on a local package at `external/megaDNA` (referenced via `[tool.uv.sources]`).
- **phabox-service** additionally requires a conda environment (`environment.yaml`) for binary tools (DIAMOND, BLAST, MCL, prodigal-gv).
