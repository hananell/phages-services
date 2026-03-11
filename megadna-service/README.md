# MegaDNA Embedding Service

Self-hosted FastAPI service that generates embeddings for phage DNA sequences using the pre-trained megaDNA transformer model (145M parameters).

## Components

- **service.py**: FastAPI application with a single `/embed` endpoint
- **config.yaml**: YAML configuration for model path, device, and server settings
- **megaDNA model**: Hierarchical transformer that encodes DNA sequences into 512-dimensional embeddings

The service loads the model once at startup and handles requests via mean-pooled transformer hidden states.

## Requirements

- Pre-trained megaDNA model checkpoint (`.pt` file) — see below
- megaDNA Python package (local path dependency)

## Downloading the Model

The model checkpoint (`megaDNA_phage_277M.pt`) is **not tracked in git** due to its size (1.1 GB).
Download it from the official megaDNA HuggingFace repository:

**<https://huggingface.co/lingxusb/megaDNA_variants>**

Direct download with `wget` or `curl` (run from the `megadna-service/` directory):

```bash
wget -O megaDNA_phage_277M.pt \
  "https://huggingface.co/lingxusb/megaDNA_variants/resolve/main/megaDNA_phage_277M.pt"
```

Other available checkpoints (145M, 78M, fine-tuned on *E. coli* phage) are listed at the
[megaDNA GitHub repository](https://github.com/lingxusb/megaDNA). Update `config.yaml →
model.path` to point to whichever file you download.

## Usage

Start the server:

```bash
uv run python service.py
```

Send a request:

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATGCGATCGATCGATCG"}'
```

Response:

```json
{
  "embedding": [0.123, -0.456, ...],
  "sequence_length": 17,
  "embedding_dimension": 512
}
```

## Configuration

Edit `config.yaml`:

```yaml
model:
  path: "../external/megaDNA_phage_145M.pt"
  device: null  # auto-detect cuda/cpu
  max_sequence_length: 96000

server:
  host: "0.0.0.0"
  port: 8000

embedding:
  layer_index: 0  # 0=coarsest, 1=intermediate, 2=finest
```
