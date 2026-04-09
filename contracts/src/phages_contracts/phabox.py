"""PhaBOX service request/response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PhaboxBatchRequest(BaseModel):
    """Batch prediction request."""

    sequences: list[str] = Field(..., min_length=1)
    sequence_ids: list[str]


class PhaboxResult(BaseModel):
    """Single phage prediction result from PhaBOX2."""

    sequence_id: str
    # PhaTYP
    phatyp_lifestyle: str | None = None
    phatyp_score: float | None = None
    # PhaGCN
    lineage: str | None = None
    phagcn_score: str | None = None
    genus: str | None = None
    genus_cluster: str | None = None
    # CHERRY
    host: str | None = None
    cherry_score: float | None = None
    cherry_method: str | None = None
    host_ncbi_lineage: str | None = None
    host_gtdb_lineage: str | None = None
    skipped: bool = False


class PhaboxBatchResponse(BaseModel):
    """Batch prediction response."""

    results: list[PhaboxResult]
