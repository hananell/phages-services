"""MegaDNA service request/response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Single sequence embedding request."""

    sequence: str = Field(..., min_length=1, description="DNA sequence (ATCG)")
    layer_index: int = Field(0, ge=0, le=2, description="Embedding layer (0=coarsest)")


class EmbedResponse(BaseModel):
    """Single sequence embedding response."""

    embedding: list[float]
    sequence_length: int
    embedding_dimension: int
    layer_index: int


class EmbedBatchRequest(BaseModel):
    """Batch embedding request."""

    sequences: list[str] = Field(..., min_length=1)
    layer_index: int = Field(0, ge=0, le=2)


class EmbedBatchResponse(BaseModel):
    """Batch embedding response."""

    embeddings: list[list[float]]
    layer_index: int
