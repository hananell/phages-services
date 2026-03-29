# Phage Bioinformatics Services

A collection of self-hosted REST API microservices for phage genome analysis, including sequence embedding, lifestyle prediction, taxonomy, host prediction, and protein annotation.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| [megadna-service](#megadna-service) | 8000 | DNA sequence embeddings (megaDNA 277M) |
| [evo2-service](#evo2-service) | 8001 | DNA sequence embeddings (Evo2 7B) |
| [hmm-service](#hmm-service) | 8002 | Protein HMM annotation (PHROGs) |
| [bacphlip-service](#bacphlip-service) | 8003 | Phage lifestyle prediction (BACPHLIP) |
| [deeppl-service](#deeppl-service) | 8004 | Phage lifestyle prediction (DeepPL/DNABERT) |
| [phabox-service](#phabox-service) | 8005 | Taxonomy, lifestyle & host prediction (PhaBOX2) |

---

## megadna-service

**Port**: 8000

Generates dense embeddings for phage DNA sequences using the pre-trained [megaDNA](https://github.com/lingxusb/megaDNA) hierarchical transformer (277M parameters). The model encodes variable-length sequences up to 96,000 bp into 512-dimensional vectors via masked mean-pooling over transformer hidden states.

### Endpoints

- `POST /embed` — embed a single sequence
- `POST /batch_embed` — embed multiple sequences

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

The `layer_index` selects which hierarchical layer to extract: `0` = coarsest, `1` = intermediate, `2` = finest.

---

## evo2-service

**Port**: 8001

Generates per-layer embeddings from phage DNA sequences using the [Evo2](https://github.com/evo-design/evo) 7B transformer model. Supports full-sequence or automatic chunked inference for long sequences, with embeddings base64-encoded in the response.

### Endpoints

- `POST /embed` — embed one or more sequences

### Output format

```json
{
  "results": [
    {
      "embeddings": {
        "blocks.28.mlp.l3": "<base64-encoded float32 array>",
        "blocks.31": "<base64-encoded float32 array>"
      },
      "sequence_length": 50000
    }
  ],
  "embedding_dimensions": {
    "blocks.28.mlp.l3": 4096,
    "blocks.31": 4096
  }
}
```

Each embedding value is a base64-encoded NumPy `float32` array of shape `(embedding_dim,)`, produced by mean-pooling over the sequence length. The layers extracted are configurable in `config.yaml`.

---

## hmm-service

**Port**: 8002

Searches amino acid sequences against the [PHROGs](https://phrogs.lmge.uca.fr/) HMM database (~38K profiles) using [PyHMMER](https://pyhmmer.readthedocs.io/). Results are aggregated per genome. The database is loaded lazily on the first request.

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

**Port**: 8003

Predicts phage lifestyle (virulent vs temperate) from a genome DNA sequence using [BACPHLIP](https://github.com/adamhockenberry/bacphlip). HMMER3 scans the predicted proteins against 206 curated HMM profiles; the resulting binary feature vector is fed to a Random Forest classifier.

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

**Port**: 8004

Predicts phage lifestyle (virulent vs temperate) using a fine-tuned DNABERT model following the [DeepPL](https://github.com/zhenchengfang/DeepPL) approach. The genome is split into overlapping 105 bp windows; windows with P(Lysogenic) > 0.9 vote temperate; a genome is classified as temperate if ≥ 1.6% of windows vote temperate.

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

`windows_evaluated` is the number of 105 bp sliding windows scored. The default stride is 10 bp (10× faster than DeepPL's stride=1).

---

## phabox-service

**Port**: 8005

A FastAPI wrapper around [PhaBOX2](https://github.com/KennthShang/PhaBOX) that runs three analysis tools in sequence:

- **PhaTYP** — lifestyle prediction (virulent / temperate)
- **PhaGCN** — taxonomic classification to genus level
- **CHERRY** — bacterial host prediction

Sequences shorter than 3,000 bp are skipped. Only one PhaBOX2 process runs at a time (DIAMOND/BLAST are already multi-threaded internally).

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

`skipped: true` is set for sequences below the 3,000 bp minimum length threshold. All PhaBOX2 fields are `null` for skipped sequences.
