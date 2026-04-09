"""BACPHLIP service request/response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BacphlipPredictRequest(BaseModel):
    """Single sequence prediction request."""

    sequence: str = Field(..., min_length=1, description="DNA sequence")
    sequence_id: str = Field(..., description="Phage identifier")


class BacphlipPredictResponse(BaseModel):
    """Single sequence prediction response."""

    genome_id: str
    predicted_lifestyle: str = Field(..., description="'Virulent' or 'Temperate'")
    virulent_probability: float = Field(..., ge=0, le=1)
    temperate_probability: float = Field(..., ge=0, le=1)
    hmm_hits: dict[str, int] = Field(
        ..., description="Domain name -> binary presence (0/1)"
    )
