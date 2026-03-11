"""Pydantic models for HMM Service API."""

from typing import Literal

from pydantic import BaseModel, Field


class ProteinInput(BaseModel):
    """Single protein input for HMM search."""

    protein_id: str = Field(..., description="Unique protein identifier")
    sequence: str = Field(..., min_length=10, description="Amino acid sequence")
    genome_id: str = Field(..., description="Parent genome accession")


class SearchRequest(BaseModel):
    """Request for HMM profile search."""

    proteins: list[ProteinInput] = Field(
        ..., min_length=1, description="List of proteins to search"
    )


class HMMHit(BaseModel):
    """Single HMM hit result."""

    protein_id: str
    hmm_name: str = Field(..., description="PHROG profile name")
    e_value: float
    bit_score: float


class GenomeHMMResult(BaseModel):
    """Aggregated HMM results for a single genome."""

    genome_id: str
    protein_count: int = Field(..., description="Number of proteins from this genome")
    hmm_hit_count: int = Field(
        ..., description="Count of unique PHROG families with hits"
    )
    hmm_hit_count_normalized: float = Field(
        ..., description="hmm_hit_count / protein_count"
    )
    unique_phrogs: list[str] = Field(
        default_factory=list, description="List of unique PHROG IDs hit"
    )


class SearchResponse(BaseModel):
    """Response from HMM profile search."""

    genome_results: list[GenomeHMMResult] = Field(
        ..., description="Per-genome aggregated results"
    )
    total_proteins_searched: int
    total_hits: int
    detailed_hits: list[HMMHit] | None = Field(
        default=None, description="Detailed per-protein hits (if requested)"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    database_loaded: bool
    database_path: str | None = None
    hmm_profile_count: int | None = None
    version: str


class DatabaseInfoResponse(BaseModel):
    """Information about loaded HMM database."""

    loaded: bool
    path: str | None = None
    profile_count: int | None = None
    alphabet: str | None = None
