"""DeepPL service request/response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DeepPLBatchRequest(BaseModel):
    """Batch prediction request."""

    sequences: list[str] = Field(..., min_length=1)
    sequence_ids: list[str] | None = None


class DeepPLResult(BaseModel):
    """Single sequence prediction result."""

    sequence_id: str
    predicted_lifestyle: str = Field(..., description="'Virulent' or 'Temperate'")
    virulent_probability: float = Field(..., ge=0, le=1)
    temperate_probability: float = Field(..., ge=0, le=1)
    windows_evaluated: int = Field(..., ge=0)


class DeepPLBatchResponse(BaseModel):
    """Batch prediction response."""

    results: list[DeepPLResult]
