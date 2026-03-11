# HMM Profile Matching Service

Local REST API service for searching protein sequences against the PHROGs HMM database using PyHMMER.

## Overview

This service provides fast HMM profile matching for phage protein annotation. It uses lazy initialization - the HMM database (~38K profiles) is loaded only on the first search request, not at startup.

## Usage

### Start the service

```bash
uv run hmm-service
```

Or with custom settings:

```bash
HMM_PORT=8080 HMM_CPUS=8 uv run hmm-service
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and database status |
| `/database` | GET | Database information |
| `/database/load` | POST | Explicitly load database (warm-up) |
| `/search` | POST | Search proteins against HMM database |

### Search Example

```bash
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{
    "proteins": [
      {"protein_id": "prot1", "sequence": "MKTLLLTGFGG...", "genome_id": "NC_001416"}
    ]
  }'
```

Response:
```json
{
  "genome_results": [
    {
      "genome_id": "NC_001416",
      "protein_count": 1,
      "hmm_hit_count": 5,
      "hmm_hit_count_normalized": 5.0,
      "unique_phrogs": ["phrog_1", "phrog_42", ...]
    }
  ],
  "total_proteins_searched": 1,
  "total_hits": 5
}
```

## Configuration

Environment variables (prefix `HMM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HMM_HOST` | 0.0.0.0 | Server bind address |
| `HMM_PORT` | 8002 | Server port |
| `HMM_PHROGS_DB_PATH` | ~/.phrogs/all_phrogs.h3m | Path to pressed HMM database |
| `HMM_E_VALUE_THRESHOLD` | 1e-5 | E-value cutoff for hits |
| `HMM_CPUS` | 4 | CPU threads for search |

Copy `.env.example` to `.env` and modify as needed.

## PHROGs Database Setup

Download and press the PHROGs database:

```bash
mkdir -p ~/.phrogs
wget -O ~/.phrogs/all_phrogs.hmm https://phrogs.lmge.uca.fr/downloads/all_phrogs.hmm
# Press for faster loading (run once)
python -c "import pyhmmer; pyhmmer.hmmer.hmmpress(pyhmmer.plan7.HMMFile('~/.phrogs/all_phrogs.hmm'))"
```
