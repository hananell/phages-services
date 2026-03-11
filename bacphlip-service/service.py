"""BACPHLIP Phage Lifecycle Prediction Service.

A self-hosted FastAPI service that predicts bacteriophage lifecycle types
(virulent vs temperate) using the BACPHLIP machine learning classifier.

Requirements:
    - HMMER3 must be installed: sudo apt-get install hmmer
"""

import shutil
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

# Suppress deprecation warnings from bacphlip's use of pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="bacphlip")

import bacphlip

from config_model import Settings, load_settings

settings: Settings = load_settings()


def check_hmmer_installed() -> bool:
    """Check if HMMER3 is installed and available."""
    return shutil.which("hmmsearch") is not None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    if not check_hmmer_installed():
        logger.error("HMMER3 is not installed! Install with: sudo apt-get install hmmer")
        raise RuntimeError("HMMER3 not installed")
    logger.info("HMMER3 is available")
    yield
    logger.info("Shutting down BACPHLIP service")


app = FastAPI(
    title="BACPHLIP Phage Lifecycle Prediction Service",
    description=(
        "Predict bacteriophage lifecycle types (virulent vs temperate) "
        "using the BACPHLIP machine learning classifier based on HMMER3."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# API Models
# =============================================================================


class SequenceRequest(BaseModel):
    """Request body for single sequence prediction."""

    sequence: str = Field(
        ...,
        description="Complete phage genome DNA sequence",
        min_length=100,
        examples=["ATGCGATCGATCG..."],
    )
    sequence_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for the sequence",
        examples=["phage_001"],
    )


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""

    genome_id: str = Field(..., description="Genome identifier")
    predicted_lifestyle: Literal["Virulent", "Temperate"] = Field(
        ..., description="Predicted lifestyle type"
    )
    virulent_probability: float = Field(
        ...,
        description="Probability of virulent lifestyle",
        ge=0.0,
        le=1.0,
    )
    temperate_probability: float = Field(
        ...,
        description="Probability of temperate lifestyle",
        ge=0.0,
        le=1.0,
    )


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str = Field(..., description="Service status")
    hmmer_available: bool = Field(..., description="Whether HMMER3 is available")


# =============================================================================
# API Endpoints
# =============================================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict_sequence(request: SequenceRequest) -> PredictionResponse:
    """Predict phage lifecycle type from a DNA sequence.

    Args:
        request: Request containing the DNA sequence.

    Returns:
        Prediction response with lifestyle type and probabilities.

    Raises:
        HTTPException: If sequence is invalid or prediction fails.
    """
    sequence_id = request.sequence_id or "unknown"

    try:
        # Create temp directory for FASTA file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fasta_path = temp_path / f"{sequence_id}.fasta"

            # Write sequence to temp FASTA
            with open(fasta_path, "w") as f:
                f.write(f">{sequence_id}\n{request.sequence}\n")

            # Run bacphlip pipeline
            bacphlip.run_pipeline(str(fasta_path), force_overwrite=True)

            # Read results
            results_file = fasta_path.with_suffix(".fasta.bacphlip")
            if not results_file.exists():
                raise RuntimeError("BACPHLIP results file not found")

            results_df = pd.read_csv(results_file, sep="\t")
            virulent_prob: float = float(results_df["Virulent"].iloc[0])
            temperate_prob: float = float(results_df["Temperate"].iloc[0])

            predicted: Literal["Virulent", "Temperate"] = (
                "Temperate" if temperate_prob > virulent_prob else "Virulent"
            )

            return PredictionResponse(
                genome_id=sequence_id,
                predicted_lifestyle=predicted,
                virulent_probability=round(virulent_prob, 6),
                temperate_probability=round(temperate_prob, 6),
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status of the service.
    """
    hmmer_ok = check_hmmer_installed()
    return HealthResponse(
        status="healthy" if hmmer_ok else "unhealthy",
        hmmer_available=hmmer_ok,
    )


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with service information."""
    return {
        "service": "BACPHLIP Phage Lifecycle Prediction Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


def main() -> None:
    """Run the service."""
    import uvicorn

    logger.info(
        f"Starting BACPHLIP service on {settings.server.host}:{settings.server.port}"
    )
    uvicorn.run(
        "service:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
