"""Evo2 Embedding Service.

A self-hosted FastAPI service that generates per-layer mean-pooled embeddings
for phage DNA sequences using the pre-trained Evo2 7B model.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH: Path = Path(__file__).parent / "config.yaml"

DeviceType = Literal["cpu", "cuda"]


class ModelConfig(BaseModel):
    name: str = "evo2_7b"
    device: DeviceType | None = None
    max_sequence_length: int = 1_000_000


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001


class EmbeddingConfig(BaseModel):
    layer_names: list[str] = Field(default_factory=lambda: ["blocks.28.mlp.l3", "blocks.31"])
    tile_size: int = 1_000_000
    batch_size: int = 8


class Config(BaseModel):
    model: ModelConfig
    server: ServerConfig
    embedding: EmbeddingConfig


def load_config(config_path: Path = CONFIG_PATH) -> Config:
    with open(config_path) as f:
        raw: dict = yaml.safe_load(f)
    return Config(**raw)


# =============================================================================
# Model wrapper
# =============================================================================

VALID_NUCLEOTIDES: frozenset[str] = frozenset("ATCGN")
PAD_TOKEN_ID: int = 0  # pad with token 0; positions are masked during mean-pool


def validate_sequence(sequence: str) -> str | None:
    if not sequence:
        return "Sequence cannot be empty"
    invalid = set(sequence.upper()) - VALID_NUCLEOTIDES
    if invalid:
        return f"Invalid characters: {invalid}. Only A, T, C, G, N allowed."
    return None


class Evo2EmbeddingModel:
    """Wraps the Evo2 model for batch embedding extraction."""

    def __init__(self, model_name: str, device: DeviceType | None = None) -> None:
        self.model_name = model_name
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load(self) -> None:
        from evo2 import Evo2
        logger.info(f"Loading Evo2 model '{self.model_name}' on {self.device}")
        self._model = Evo2(self.model_name)
        logger.info("Evo2 model loaded successfully")

    def get_embeddings_batch(
        self,
        sequences: list[str],
        layer_names: list[str],
    ) -> list[dict[str, np.ndarray]]:
        """Return mean-pooled embeddings for a batch of sequences.

        Args:
            sequences: List of DNA sequences (uppercase, validated).
            layer_names: Which intermediate layers to extract.

        Returns:
            List of dicts mapping layer_name -> float32 ndarray of shape (embed_dim,).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize each sequence
        token_seqs: list[list[int]] = [
            self._model.tokenizer.tokenize(seq) for seq in sequences
        ]
        lengths: list[int] = [len(t) for t in token_seqs]
        max_len: int = max(lengths)

        # Pad to max_len with PAD_TOKEN_ID
        padded = [t + [PAD_TOKEN_ID] * (max_len - len(t)) for t in token_seqs]
        input_ids = torch.tensor(padded, dtype=torch.int).to(self.device)

        # torch.inference_mode() is a stricter superset of no_grad():
        # disables autograd tracking AND version tracking -> less memory overhead.
        with torch.inference_mode():
            _, raw_embeddings = self._model(
                input_ids,
                return_embeddings=True,
                layer_names=layer_names,
            )

        # input_ids no longer needed — free GPU memory before touching embeddings.
        del input_ids

        # Build a (B, max_len) float mask once — shared across all layers.
        # mask[i, j] = 1.0 if position j is a real token for sequence i, else 0.0.
        B = len(sequences)
        mask: torch.Tensor = torch.zeros(B, max_len, dtype=torch.float32, device=self.device)
        for i, L in enumerate(lengths):
            mask[i, :L] = 1.0
        # (B, 1) for broadcasting against (B, embed_dim)
        lengths_t: torch.Tensor = torch.tensor(
            lengths, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        # For each layer: one vectorized masked mean-pool + one GPU→CPU transfer.
        # This replaces the previous B×L per-sequence loop with per-layer tensor ops.
        results: list[dict[str, np.ndarray]] = [{} for _ in sequences]
        for layer in layer_names:
            layer_emb: torch.Tensor = raw_embeddings[layer].float()  # (B, max_len, embed_dim)
            # Broadcast mask to (B, max_len, 1) and multiply, then sum over seq dim.
            means: torch.Tensor = (layer_emb * mask.unsqueeze(-1)).sum(dim=1) / lengths_t  # (B, embed_dim)
            means_np: np.ndarray = means.cpu().numpy()  # single GPU→CPU transfer per layer
            for i, row in enumerate(means_np):
                results[i][layer] = row  # float32 ndarray, shape (embed_dim,)

        # Release the large embedding tensors from GPU memory before returning.
        del raw_embeddings
        torch.cuda.empty_cache()

        return results


# =============================================================================
# API models
# =============================================================================

class BatchEmbedRequest(BaseModel):
    sequences: list[str] = Field(..., min_length=1, description="List of DNA sequences")
    layer_names: list[str] | None = Field(
        default=None,
        description="Layers to extract. Uses config default if omitted.",
    )


class SequenceResult(BaseModel):
    embeddings: dict[str, str]  # base64-encoded little-endian float32 arrays
    sequence_length: int


class BatchEmbedResponse(BaseModel):
    results: list[SequenceResult]
    embedding_dimensions: dict[str, int]


# =============================================================================
# Application
# =============================================================================

config: Config = load_config()
embedding_model = Evo2EmbeddingModel(
    model_name=config.model.name,
    device=config.model.device,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if embedding_model._model is None:
        embedding_model.load()
    yield
    logger.info("Shutting down evo2-service")


app = FastAPI(
    title="Evo2 Embedding Service",
    description="Generate per-layer embeddings for phage DNA sequences using Evo2 7B",
    version="1.0.0",
    lifespan=lifespan,
)


def _embed_sequences_batched(
    sequences: list[str],
    layer_names: list[str],
    tile_size: int,
    batch_size: int,
) -> list[dict[str, list[float]]]:
    # Phase 1: tile every sequence, track ownership
    flat_tiles: list[str] = []
    tile_owners: list[int] = []
    for seq_idx, seq in enumerate(sequences):
        for i in range(0, len(seq), tile_size):
            flat_tiles.append(seq[i : i + tile_size])
            tile_owners.append(seq_idx)

    # Phase 2: process tiles in chunks of batch_size
    tile_embeddings: list[dict[str, list[float]]] = []
    for start in range(0, len(flat_tiles), batch_size):
        chunk = flat_tiles[start : start + batch_size]
        tile_embeddings.extend(embedding_model.get_embeddings_batch(chunk, layer_names))

    # Phase 3: reassemble per-sequence by averaging tile embeddings
    seq_tile_embs: list[list[dict]] = [[] for _ in sequences]
    for k, seq_idx in enumerate(tile_owners):
        seq_tile_embs[seq_idx].append(tile_embeddings[k])

    results: list[dict[str, list[float]]] = []
    for bucket in seq_tile_embs:
        result: dict[str, list[float]] = {}
        for layer in layer_names:
            vecs = np.array([e[layer] for e in bucket], dtype=np.float32)
            result[layer] = vecs.mean(axis=0).tolist()
        results.append(result)
    return results


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest) -> BatchEmbedResponse:
    """Generate embeddings for a batch of DNA sequences."""
    layer_names = request.layer_names or config.embedding.layer_names

    sequences: list[str] = []
    for raw_seq in request.sequences:
        seq = raw_seq.upper().strip()
        if err := validate_sequence(seq):
            raise HTTPException(status_code=400, detail=err)
        if len(seq) > config.model.max_sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Sequence too long: {len(seq)} > {config.model.max_sequence_length}",
            )
        sequences.append(seq)

    logger.info(
        f"Processing batch of {len(sequences)} sequences, layers={layer_names}, "
        f"tile_size={config.embedding.tile_size}, batch_size={config.embedding.batch_size}"
    )

    all_embeddings = _embed_sequences_batched(
        sequences,
        layer_names,
        tile_size=config.embedding.tile_size,
        batch_size=config.embedding.batch_size,
    )

    results = [
        SequenceResult(
            embeddings={
                layer: base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")
                for layer, arr in emb.items()
            },
            sequence_length=len(seq),
        )
        for seq, emb in zip(sequences, all_embeddings)
    ]

    # Infer embedding dimensions from first result (ndarray .size = number of elements)
    emb_dims = (
        {layer: all_embeddings[0][layer].size for layer in layer_names}
        if all_embeddings else {}
    )

    return BatchEmbedResponse(results=results, embedding_dimensions=emb_dims)


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {"status": "healthy", "model_loaded": embedding_model._model is not None}


def main() -> None:
    import uvicorn
    logger.info(f"Starting evo2-service on {config.server.host}:{config.server.port}")
    uvicorn.run("service:app", host=config.server.host, port=config.server.port, reload=False)


if __name__ == "__main__":
    main()
