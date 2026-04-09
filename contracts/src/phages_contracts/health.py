"""Shared health check response model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Standard health check response for all services."""

    status: Literal["healthy", "degraded", "unhealthy"]
    service: str
    version: str = "0.1.0"
    device: str | None = None
    model_loaded: bool | None = None
