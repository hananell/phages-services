"""Shared Pydantic models for phage service APIs.

These contracts define the request/response shapes used by both:
  - phages_services (server side — FastAPI endpoints)
  - phages_dataset (client side — feature calculators)

Keeping them in one package prevents client/server contract drift.
"""

from phages_contracts.megadna import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
)
from phages_contracts.bacphlip import (
    BacphlipPredictRequest,
    BacphlipPredictResponse,
)
from phages_contracts.deeppl import (
    DeepPLBatchRequest,
    DeepPLBatchResponse,
    DeepPLResult,
)
from phages_contracts.hmm import (
    HMMSearchRequest,
    HMMSearchResponse,
    ProteinInput,
    GenomeHMMResult,
)
from phages_contracts.phabox import (
    PhaboxBatchRequest,
    PhaboxBatchResponse,
    PhaboxResult,
)
from phages_contracts.health import HealthResponse

__all__ = [
    # megadna
    "EmbedRequest",
    "EmbedResponse",
    "EmbedBatchRequest",
    "EmbedBatchResponse",
    # bacphlip
    "BacphlipPredictRequest",
    "BacphlipPredictResponse",
    # deeppl
    "DeepPLBatchRequest",
    "DeepPLBatchResponse",
    "DeepPLResult",
    # hmm
    "HMMSearchRequest",
    "HMMSearchResponse",
    "ProteinInput",
    "GenomeHMMResult",
    # phabox
    "PhaboxBatchRequest",
    "PhaboxBatchResponse",
    "PhaboxResult",
    # common
    "HealthResponse",
]
