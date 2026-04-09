"""HMM service request/response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProteinInput(BaseModel):
    """Single protein input for HMM search."""

    protein_id: str = Field(..., description="Unique protein identifier")
    sequence: str = Field(..., min_length=10, description="Amino acid sequence")
    genome_id: str = Field(..., description="Parent genome accession")


class HMMSearchRequest(BaseModel):
    """Request for HMM profile search."""

    proteins: list[ProteinInput] = Field(..., min_length=1)


class GenomeHMMResult(BaseModel):
    """Aggregated HMM results for a single genome."""

    genome_id: str
    protein_count: int
    hmm_hit_count: int
    hmm_hit_count_normalized: float
    unique_phrogs: list[str] = Field(default_factory=list)


class HMMSearchResponse(BaseModel):
    """Response from HMM profile search."""

    genome_results: list[GenomeHMMResult]
    total_proteins_searched: int
    total_hits: int
    detailed_hits: list | None = None
