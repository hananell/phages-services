"""FastAPI application for HMM Profile Matching Service."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from hmm_service import __version__, logger
from hmm_service.config import settings
from hmm_service.hmm_matcher import hmm_matcher
from hmm_service.models import (
    DatabaseInfoResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info(f"HMM Service v{__version__} starting on {settings.host}:{settings.port}")
    logger.info(f"Database path configured: {settings.phrogs_db_path}")
    logger.info("Database will be loaded lazily on first request")
    yield
    logger.info("HMM Service shutting down")


app = FastAPI(
    title="HMM Profile Matching Service",
    description="Local service for searching protein sequences against PHROGs HMM database using PyHMMER",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check service health status.

    Returns health status and database loading state.
    """
    db_info = hmm_matcher.database_info

    if hmm_matcher.is_initialized:
        status = "healthy"
    else:
        # Not initialized yet is OK - it's lazy loading
        status = "healthy"

    return HealthResponse(
        status=status,
        database_loaded=hmm_matcher.is_initialized,
        database_path=db_info.get("path"),
        hmm_profile_count=db_info.get("profile_count"),
        version=__version__,
    )


@app.get("/database", response_model=DatabaseInfoResponse)
async def database_info() -> DatabaseInfoResponse:
    """
    Get information about the HMM database.

    If database is not yet loaded, returns minimal info.
    """
    info = hmm_matcher.database_info
    return DatabaseInfoResponse(
        loaded=info["loaded"],
        path=info.get("path"),
        profile_count=info.get("profile_count"),
        alphabet=info.get("alphabet"),
    )


@app.post("/database/load")
async def load_database() -> dict:
    """
    Explicitly load the HMM database.

    This triggers database loading without performing a search.
    Useful for warming up the service.
    """
    if hmm_matcher.is_initialized:
        return {"status": "already_loaded", "profile_count": hmm_matcher.database_info["profile_count"]}

    try:
        hmm_matcher._ensure_initialized()
        return {
            "status": "loaded",
            "profile_count": hmm_matcher.database_info["profile_count"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to load HMM database")
        raise HTTPException(status_code=500, detail=f"Failed to load database: {e}")


@app.post("/search", response_model=SearchResponse)
async def search_proteins(
    request: SearchRequest,
    include_details: bool = Query(
        default=False, description="Include per-protein hit details in response"
    ),
) -> SearchResponse:
    """
    Search protein sequences against PHROGs HMM database.

    On first request, this will trigger lazy loading of the HMM database.

    Args:
        request: Search request containing proteins to search
        include_details: Whether to include detailed per-protein hits

    Returns:
        Aggregated per-genome HMM hit results
    """
    try:
        genome_results, detailed_hits = hmm_matcher.search(
            proteins=request.proteins,
            return_detailed_hits=include_details,
        )

        total_hits = sum(r.hmm_hit_count for r in genome_results)

        return SearchResponse(
            genome_results=genome_results,
            total_proteins_searched=len(request.proteins),
            total_hits=total_hits,
            detailed_hits=detailed_hits,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"HMM database not available: {e}",
        )
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


def run() -> None:
    """Run the HMM service."""
    uvicorn.run(
        "hmm_service.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
