# Phage Bioinformatics Services

A collection of self-hosted REST API microservices for phage genome analysis, including sequence embedding, lifestyle prediction, taxonomy, host prediction, and protein functional annotation.

These services are the computational backend for the [phages_dataset](../phages_dataset/) pipeline, which calls them via HTTP during feature computation. Each service wraps a published bioinformatics tool or ML model behind a FastAPI REST API.

## Services

| Service | Port | Purpose | Model/Tool |
|---------|------|---------|------------|
| [megadna-service](#megadna-service) | 8000 | DNA sequence embeddings | megaDNA 277M |
| [hmm-service](#hmm-service) | 8002 | Protein functional annotation | PHROGs HMM database (~38K profiles) |
| [bacphlip-service](#bacphlip-service) | 8003 | Phage lifestyle prediction | BACPHLIP (HMMER3 + Random Forest) |
| [deeppl-service](#deeppl-service) | 8004 | Phage lifestyle prediction | DeepPL (fine-tuned DNABERT) |
| [phabox-service](#phabox-service) | 8005 | Taxonomy, lifestyle, host prediction | PhaBOX2 (PhaTYP + PhaGCN + CHERRY) |

## Quick Start

### Option 1: Startup script (recommended for local dev)

```bash
./start_services.sh              # start CPU services (hmm, bacphlip)
./start_services.sh --gpu        # start all services (CPU + GPU)
./start_services.sh hmm megadna  # start specific services
./start_services.sh --stop       # stop all running services
```

### Option 2: Docker Compose

```bash
docker compose up                   # start CPU services
docker compose --profile gpu up     # start all services (CPU + GPU)
docker compose up hmm bacphlip      # start specific services
```

### Option 3: Manual

Each service is a standalone Python package managed by `uv`:

```bash
cd <service-name>
uv sync
uv run uvicorn <module>:app --host 0.0.0.0 --port <port>
```

For example:
```bash
cd megadna-service && uv run uvicorn service:app --host 0.0.0.0 --port 8000
cd bacphlip-service && uv run uvicorn service:app --host 0.0.0.0 --port 8003
```

### GPU requirements

| Service | GPU Required | VRAM |
|---------|-------------|------|
| megadna-service | Yes | ~2 GB |
| deeppl-service | Yes (recommended) | ~1 GB |
| hmm-service | No | CPU only |
| bacphlip-service | No | CPU only (HMMER3) |
| phabox-service | No | CPU only (DIAMOND/BLAST) |

## Integration with phages_dataset

The companion [phages_dataset](../phages_dataset/) repository calls these services during feature computation. Each feature calculator reads its service URL from an environment variable:

| Variable | Default | Service |
|----------|---------|---------|
| `MEGADNA_SERVICE_URL` | `http://localhost:8000` | megadna-service |
| `HMM_SERVICE_URL` | `http://localhost:8002` | hmm-service |
| `BACPHLIP_SERVICE_URL` | `http://localhost:8003` | bacphlip-service |
| `DEEPPL_SERVICE_URL` | `http://localhost:8004` | deeppl-service |
| `PHABOX_SERVICE_URL` | `http://localhost:8005` | phabox-service |

Feature computation commands:
```bash
cd ../phages_dataset
uv run phages features compute --group megadna     # requires megadna-service on :8000
uv run phages features compute --group bacphlip    # requires bacphlip-service on :8003
uv run phages features compute --group phrog       # requires hmm-service on :8002
uv run phages features compute --group tnf gene comp trna  # no service required
```

---

## megadna-service

**Port**: 8000 | **GPU**: Yes (~2 GB VRAM)

Generates dense embeddings for phage DNA sequences using the pre-trained [megaDNA](https://github.com/lingxusb/megaDNA) hierarchical transformer (277M parameters). The model encodes variable-length sequences up to 96,000 bp into 512-dimensional vectors via masked mean-pooling over transformer hidden states.

### Endpoints

- `POST /embed` — embed a single sequence
- `POST /batch_embed` — embed multiple sequences (recommended for throughput)

### Output format

**Single sequence** (`POST /embed`):
```json
{
  "embedding": [0.123, -0.456, ...],
  "sequence_length": 17,
  "embedding_dimension": 512,
  "layer_index": 0
}
```

**Batch** (`POST /batch_embed`):
```json
{
  "embeddings": [[0.123, -0.456, ...], [0.234, -0.567, ...]],
  "layer_index": 0
}
```

The `layer_index` selects which hierarchical layer to extract: `0` = coarsest, `1` = intermediate, `2` = finest. Sequences longer than 96,000 bp are automatically split into overlapping tiles and mean-pooled.

---

## hmm-service

**Port**: 8002 | **GPU**: No

Searches amino acid sequences against the [PHROGs](https://phrogs.lmge.uca.fr/) HMM database (~38,000 profiles) using [PyHMMER](https://pyhmmer.readthedocs.io/). Results are aggregated per genome. The database is loaded lazily on the first request.

The phages_dataset `phrog` feature calculator uses this service to annotate predicted genes (from pyrodigal) with PHROG functional categories: DNA/RNA metabolism, head and packaging, integration and excision, lysis, moron/AMG/host takeover, tail, transcription regulation, other, and unknown function.

### Endpoints

- `GET /health` — health check and database status
- `GET /database` — database info
- `POST /database/load` — pre-warm the database
- `POST /search` — search proteins against PHROGs HMMs

### Input format

```json
{
  "proteins": [
    {
      "protein_id": "prot1",
      "sequence": "MKTLLLTGFGG...",
      "genome_id": "NC_001416"
    }
  ]
}
```

### Output format

```json
{
  "genome_results": [
    {
      "genome_id": "NC_001416",
      "protein_count": 1,
      "hmm_hit_count": 5,
      "hmm_hit_count_normalized": 5.0,
      "unique_phrogs": ["phrog_1", "phrog_42"]
    }
  ],
  "total_proteins_searched": 1,
  "total_hits": 5
}
```

`hmm_hit_count_normalized` is the number of unique PHROG family hits divided by the number of proteins searched for that genome. Pass `include_details=true` to receive per-protein hit details.

---

## bacphlip-service

**Port**: 8003 | **GPU**: No

Predicts phage lifestyle (virulent vs temperate) from a genome DNA sequence using [BACPHLIP](https://github.com/adamhockenberry/bacphlip). HMMER3 scans the predicted proteins against 206 curated HMM profiles; the resulting binary feature vector is fed to a Random Forest classifier.

The 206 HMM domain features are also used by the `TieredLifecycleStrategy` in phages_dataset — 22 lysogeny-associated domains (integrases, recombinases, excisionases) serve as Tier 2 evidence for temperate lifestyle assignment.

### Endpoints

- `POST /predict` — predict lifestyle for a single sequence

### Input format

```json
{
  "sequence": "ATGCGATCGATCG...",
  "sequence_id": "phage_001"
}
```

### Output format

```json
{
  "genome_id": "phage_001",
  "predicted_lifestyle": "Virulent",
  "virulent_probability": 0.85,
  "temperate_probability": 0.15,
  "hmm_hits": {
    "domain_name_1": 1,
    "domain_name_2": 0
  }
}
```

`hmm_hits` contains 206 binary features (1 = profile hit present, 0 = absent) that were used as input to the classifier.

---

## deeppl-service

**Port**: 8004 | **GPU**: Yes (recommended, ~1 GB VRAM)

Predicts phage lifestyle (virulent vs temperate) using a fine-tuned DNABERT model following the [DeepPL](https://github.com/zhenchengfang/DeepPL) approach. The genome is split into overlapping 105 bp windows; windows with P(Lysogenic) > 0.9 vote temperate; a genome is classified as temperate if >= 1.6% of windows vote temperate.

Service v0.2.0 features a fully vectorized numpy tokenizer and stride=10 (10x fewer windows than DeepPL's default stride=1), achieving ~50x speedup. With fp16 autocast and batch_size=4096 on RTX 5090: ~7,200 windows/sec, ~1.1 phages/sec.

### Endpoints

- `POST /predict` — predict lifestyle for one or more sequences

### Input format

```json
{
  "sequences": ["ATGCGATCGATCG...", "GCTAGCTA..."],
  "sequence_ids": ["phage_1", "phage_2"]
}
```

### Output format

```json
{
  "results": [
    {
      "sequence_id": "phage_1",
      "predicted_lifestyle": "Virulent",
      "virulent_probability": 0.75,
      "temperate_probability": 0.25,
      "windows_evaluated": 1234
    }
  ]
}
```

`windows_evaluated` is the number of 105 bp sliding windows scored. The default stride is 10 bp.

---

## phabox-service

**Port**: 8005 | **GPU**: No

A FastAPI wrapper around [PhaBOX2](https://github.com/KennthShang/PhaBOX) that runs three analysis tools in sequence:

- **PhaTYP** — lifestyle prediction (virulent / temperate)
- **PhaGCN** — taxonomic classification to genus level
- **CHERRY** — bacterial host prediction

Sequences shorter than 3,000 bp are skipped. Only one PhaBOX2 process runs at a time (DIAMOND/BLAST are already multi-threaded internally). Estimated throughput: ~0.3-0.5 phages/sec on 24 CPU threads.

### Prerequisites

```bash
# One-time setup:
conda env create -f environment.yaml
bash download_db.sh  # downloads ~1 GB phabox_db_v2

# Running:
conda activate phabox-tools
uv run uvicorn service:app --host 0.0.0.0 --port 8005
```

### Endpoints

- `POST /predict` — run full PhaBOX2 analysis on one or more sequences

### Input format

```json
{
  "sequences": ["ATGC...", "GCTA..."],
  "sequence_ids": ["phage_1", "phage_2"]
}
```

### Output format

```json
{
  "results": [
    {
      "sequence_id": "phage_1",
      "phatyp_lifestyle": "Virulent",
      "phatyp_score": 0.95,
      "lineage": "Tequatrovirus T4",
      "phagcn_score": "95.6",
      "genus": "Tequatrovirus",
      "genus_cluster": "GC123",
      "host": "Escherichia coli",
      "cherry_score": 0.85,
      "cherry_method": "Alignment",
      "host_ncbi_lineage": "...",
      "host_gtdb_lineage": "...",
      "skipped": false
    }
  ]
}
```

`skipped: true` is set for sequences below the 3,000 bp minimum length threshold.

---

## Shared Contracts

The `contracts/` directory contains a `phages-contracts` Python package with shared Pydantic request/response models for all service APIs. This prevents contract drift between services (server side) and phages_dataset feature calculators (client side).

```
contracts/src/phages_contracts/
├── __init__.py       # re-exports all models
├── health.py         # HealthResponse (standard for all services)
├── megadna.py        # EmbedRequest, EmbedBatchRequest, EmbedResponse, EmbedBatchResponse
├── bacphlip.py       # BacphlipPredictRequest, BacphlipPredictResponse
├── deeppl.py         # DeepPLBatchRequest, DeepPLBatchResponse, DeepPLResult
├── hmm.py            # HMMSearchRequest, HMMSearchResponse, ProteinInput, GenomeHMMResult
└── phabox.py         # PhaboxBatchRequest, PhaboxBatchResponse, PhaboxResult
```

Install as a dependency: `uv add phages-contracts --path ../contracts`

---

## Configuration

All services support configuration via **pydantic-settings** with environment variable overrides. Each service has a `settings.py` module with a `BaseSettings` subclass and an env var prefix:

| Service | Env Prefix | Example |
|---------|-----------|---------|
| hmm-service | `HMM_` | `HMM_PORT=9002 HMM_CPUS=8` |
| megadna-service | `MEGADNA_` | `MEGADNA_PORT=9000 MEGADNA_DEVICE=cpu` |
| deeppl-service | `DEEPPL_` | `DEEPPL_PORT=9004 DEEPPL_STRIDE=1` |
| bacphlip-service | `BACPHLIP_` | `BACPHLIP_PORT=9003 BACPHLIP_HMMER_THREADS=8` |
| phabox-service | `PHABOX_` | `PHABOX_PORT=9005 PHABOX_THREADS=32` |

Configuration precedence: environment variables > `.env` file > `config.yaml` (backward compatible) > defaults.

---

## Testing

Each service has integration tests in its `tests/` directory:

```bash
# Run tests for a specific service:
cd hmm-service && uv run pytest tests/ -v
cd megadna-service && uv run pytest tests/ -v
cd deeppl-service && uv run pytest tests/ -v
cd bacphlip-service && uv run pytest tests/ -v
cd phabox-service && uv run pytest tests/ -v
```

Tests use mock models/tools so they don't require GPU, HMMER3, or PhaBOX2 databases. They verify endpoint routing, request validation, and response shapes.

---

## Retry & Resilience

The phages_dataset feature calculators use **automatic retry with exponential backoff** when calling these services. Transient failures (connection errors, timeouts, HTTP 429/500/502/503/504) are retried up to 3 times with 1s/2s/4s delays. This is handled by `phages.io.retry.RetryTransport` in the phages_dataset repo — no changes needed on the service side.
