"""MegaDNA Embedding Service.

A self-hosted FastAPI service that generates embeddings for phage DNA sequences
using the pre-trained megaDNA transformer model.
"""

from __future__ import annotations

import math
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

NT_VOCAB: list[str] = ["**", "A", "T", "C", "G", "#"]
VALID_NUCLEOTIDES: set[str] = {"A", "T", "C", "G"}


class ModelConfig(BaseModel):
    """Model configuration."""

    path: str
    device: DeviceType | None = None
    max_sequence_length: int = 96000


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    layer_index: int = 0


class Config(BaseModel):
    """Application configuration."""

    model: ModelConfig
    server: ServerConfig
    embedding: EmbeddingConfig


def load_config(config_path: Path = CONFIG_PATH) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration object.
    """
    with open(config_path) as f:
        raw_config: dict = yaml.safe_load(f)
    return Config(**raw_config)


def resolve_model_path(model_path: str, config_dir: Path) -> str:
    """Resolve model path relative to config directory.

    Args:
        model_path: Model path from config (can be relative or absolute).
        config_dir: Directory containing the config file.

    Returns:
        Resolved absolute path to the model.
    """
    path = Path(model_path)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


# =============================================================================
# DNA Encoding
# =============================================================================


def encode_sequence(sequence: str) -> list[int]:
    """Encode a DNA sequence to numerical tokens for the megaDNA model.

    Token mapping:
        0 -> '**' (start token)
        1 -> 'A'
        2 -> 'T'
        3 -> 'C'
        4 -> 'G'
        5 -> '#' (end token)

    Args:
        sequence: Raw DNA sequence string (e.g., "ATCGATCG").

    Returns:
        List of integers with start/end tokens.
    """
    encoded: list[int] = [0]  # Start token
    for nucleotide in sequence.upper():
        if nucleotide in NT_VOCAB:
            encoded.append(NT_VOCAB.index(nucleotide))
        else:
            # Unknown characters default to 'A'
            encoded.append(1)
    encoded.append(5)  # End token
    return encoded


def validate_sequence(sequence: str) -> str | None:
    """Validate a DNA sequence.

    Args:
        sequence: DNA sequence to validate.

    Returns:
        Error message if invalid, None if valid.
    """
    if not sequence:
        return "Sequence cannot be empty"

    sequence_upper = sequence.upper()
    invalid_chars = set(sequence_upper) - VALID_NUCLEOTIDES
    if invalid_chars:
        return f"Invalid characters in sequence: {invalid_chars}. Only A, T, C, G are allowed."

    return None


# =============================================================================
# Model Inference
# =============================================================================


class EmbeddingModel:
    """Wrapper for the megaDNA model for embedding extraction."""

    def __init__(self, model_path: str, device: DeviceType | None = None) -> None:
        """Initialize the embedding model.

        Args:
            model_path: Path to the pre-trained model file.
            device: Compute device ('cpu' or 'cuda'). Auto-detects if None.
        """
        self.device: DeviceType = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path: str = model_path
        self.model: torch.nn.Module | None = None

    def load(self) -> None:
        """Load the model from disk."""
        logger.info(f"Loading megaDNA model from {self.model_path}")
        logger.info(f"Using device: {self.device}")

        self.model = torch.load(
            self.model_path, map_location=torch.device(self.device), weights_only=False
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def get_embedding(self, sequence: str, layer_index: int = 0) -> np.ndarray:
        """Extract embedding for a DNA sequence.

        Args:
            sequence: DNA sequence string.
            layer_index: Which transformer layer to use (0=coarsest).

        Returns:
            1D numpy array of shape (embed_dim,).
        """
        return self.get_embeddings_batch([sequence], layer_index)[0]

    def get_embeddings_batch(self, sequences: list[str], layer_index: int = 0) -> np.ndarray:
        """Extract embeddings for a batch of DNA sequences in one forward pass.

        Sequences are padded to the same length at the input level. The model's
        hierarchical architecture compresses the input 1024× before producing
        hidden states (local stride 16 × middle stride 64), so the mask is built
        at the compressed output dimension, not at the raw token length.

        Args:
            sequences: List of DNA sequence strings (already validated, uppercase).
            layer_index: Which transformer layer to use (0=coarsest).

        Returns:
            2D numpy array of shape (len(sequences), embed_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        encoded_list: list[list[int]] = [encode_sequence(seq) for seq in sequences]
        token_lengths: list[int] = [len(enc) for enc in encoded_list]
        max_len: int = max(token_lengths)

        # Pad shorter sequences with 0 (start token); real positions tracked via mask.
        padded: list[list[int]] = [
            enc + [0] * (max_len - len(enc)) for enc in encoded_list
        ]

        input_tensor: torch.Tensor = torch.tensor(padded, dtype=torch.long).to(self.device)

        with torch.no_grad():
            hidden_states: list[torch.Tensor] = self.model(input_tensor, return_value="embedding")
            layer_embeddings: torch.Tensor = hidden_states[layer_index]  # (B, compressed_len, embed_dim)

        # The hierarchical model compresses input tokens before producing hidden states.
        # Local layer: stride 16 → ceil(L/16) local tokens.
        # Middle/global layers: stride 64 on local tokens → ceil(ceil(L/16)/64) tokens.
        # layer_index=2 is the finest (local) layer; 0 and 1 are at the middle/global level.
        compressed_lengths: list[int] = [
            math.ceil(math.ceil(L / 16) / 64) if layer_index < 2 else math.ceil(L / 16)
            for L in token_lengths
        ]

        compressed_dim: int = layer_embeddings.shape[1]

        # Build (B, compressed_dim, 1) float mask — 1.0 for real tokens, 0.0 for padding.
        mask: torch.Tensor = torch.zeros(
            len(sequences), compressed_dim, 1, dtype=layer_embeddings.dtype, device=self.device
        )
        for i, comp_len in enumerate(compressed_lengths):
            mask[i, :comp_len, 0] = 1.0

        comp_lengths_tensor: torch.Tensor = torch.tensor(
            compressed_lengths, dtype=layer_embeddings.dtype, device=self.device
        ).unsqueeze(-1)  # (B, 1)

        # Masked mean pool: sum over real compressed tokens / count of real tokens.
        embeddings: torch.Tensor = (layer_embeddings * mask).sum(dim=1) / comp_lengths_tensor

        return embeddings.cpu().numpy()  # (B, embed_dim)


# =============================================================================
# API Models
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request body for embedding endpoint."""

    sequence: str = Field(
        ...,
        description="DNA sequence (A, T, C, G only)",
        min_length=1,
        examples=["ATGCGATCGATCG"],
    )
    layer_index: int | None = Field(
        default=None,
        description="Transformer layer for embedding (0=coarsest, 1=intermediate, 2=finest). Uses config default if not specified.",
        ge=0,
        le=2,
    )


class EmbeddingResponse(BaseModel):
    """Response body for embedding endpoint."""

    embedding: list[float] = Field(..., description="Embedding vector")
    sequence_length: int = Field(..., description="Length of input sequence")
    embedding_dimension: int = Field(..., description="Dimension of embedding vector")
    layer_index: int = Field(..., description="Transformer layer used for embedding")


class BatchEmbeddingRequest(BaseModel):
    """Request body for batch embedding endpoint."""

    sequences: list[str] = Field(
        ...,
        description="List of DNA sequences (A, T, C, G only). All must fit within max_sequence_length.",
        min_length=1,
    )
    layer_index: int | None = Field(
        default=None,
        description="Transformer layer for embedding (0=coarsest, 1=intermediate, 2=finest). Uses config default if not specified.",
        ge=0,
        le=2,
    )


class BatchEmbeddingResponse(BaseModel):
    """Response body for batch embedding endpoint."""

    embeddings: list[list[float]] = Field(..., description="Embedding vectors, one per input sequence")
    layer_index: int = Field(..., description="Transformer layer used for embedding")


# =============================================================================
# Application
# =============================================================================

config: Config = load_config()
resolved_model_path: str = resolve_model_path(config.model.path, CONFIG_PATH.parent)
embedding_model: EmbeddingModel = EmbeddingModel(
    model_path=resolved_model_path, device=config.model.device
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for model loading."""
    embedding_model.load()
    yield
    logger.info("Shutting down service")


app = FastAPI(
    title="MegaDNA Embedding Service",
    description="Generate embeddings for phage DNA sequences using megaDNA model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_sequence(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embedding for a DNA sequence.

    Args:
        request: Request containing the DNA sequence.

    Returns:
        Embedding response with the vector and metadata.

    Raises:
        HTTPException: If sequence is invalid or too long.
    """
    sequence = request.sequence.upper().strip()

    # Validate sequence
    error = validate_sequence(sequence)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Check length
    if len(sequence) > config.model.max_sequence_length:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence too long: {len(sequence)} > {config.model.max_sequence_length}",
        )

    layer_index: int = (
        request.layer_index if request.layer_index is not None else config.embedding.layer_index
    )
    logger.info(f"Processing sequence of length {len(sequence)}, layer {layer_index}")

    embedding: np.ndarray = embedding_model.get_embedding(sequence, layer_index=layer_index)

    return EmbeddingResponse(
        embedding=embedding.tolist(),
        sequence_length=len(sequence),
        embedding_dimension=len(embedding),
        layer_index=layer_index,
    )


@app.post("/embed_batch", response_model=BatchEmbeddingResponse)
async def embed_batch(request: BatchEmbeddingRequest) -> BatchEmbeddingResponse:
    """Generate embeddings for a batch of DNA sequences in one GPU forward pass.

    Sequences are padded to the same length internally; padding positions are
    masked out before mean-pooling so they do not influence the embeddings.

    Args:
        request: Request containing a list of DNA sequences.

    Returns:
        Batch embedding response with one vector per input sequence.

    Raises:
        HTTPException: If any sequence is invalid or too long.
    """
    layer_index: int = (
        request.layer_index if request.layer_index is not None else config.embedding.layer_index
    )

    validated: list[str] = []
    for i, seq in enumerate(request.sequences):
        seq = seq.upper().strip()
        error = validate_sequence(seq)
        if error:
            raise HTTPException(status_code=400, detail=f"Sequence {i}: {error}")
        if len(seq) > config.model.max_sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Sequence {i} too long: {len(seq)} > {config.model.max_sequence_length}",
            )
        validated.append(seq)

    logger.info(f"Processing batch of {len(validated)} sequences, layer {layer_index}")

    embeddings_array = embedding_model.get_embeddings_batch(validated, layer_index=layer_index)

    return BatchEmbeddingResponse(
        embeddings=embeddings_array.tolist(),
        layer_index=layer_index,
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status dictionary.
    """
    return {"status": "healthy"}


def main() -> None:
    """Run the service."""
    import uvicorn

    logger.info(f"Starting MegaDNA Embedding Service on {config.server.host}:{config.server.port}")
    uvicorn.run(
        "service:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
