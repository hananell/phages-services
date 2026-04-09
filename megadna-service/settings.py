"""Standardized configuration for megadna-service using pydantic-settings.

Supports configuration via environment variables (MEGADNA_ prefix),
.env file, or YAML config file (for backward compatibility).

Environment variables take precedence over YAML values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MegaDNASettings(BaseSettings):
    """MegaDNA service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MEGADNA_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Model
    model_path: str = "megaDNA_phage_277M.pt"
    device: Literal["cpu", "cuda"] | None = None  # auto-detect if None
    max_sequence_length: int = 96000

    # Embedding
    layer_index: int = 0

    @field_validator("model_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> str:
        return str(Path(v).expanduser())


def load_settings() -> MegaDNASettings:
    """Load settings, optionally merging from config.yaml for backward compat."""
    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    overrides: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        model_cfg = raw.get("model", {})
        server_cfg = raw.get("server", {})
        embed_cfg = raw.get("embedding", {})
        overrides = {
            "model_path": model_cfg.get("path", "megaDNA_phage_277M.pt"),
            "device": model_cfg.get("device"),
            "max_sequence_length": model_cfg.get("max_sequence_length", 96000),
            "host": server_cfg.get("host", "0.0.0.0"),
            "port": server_cfg.get("port", 8000),
            "layer_index": embed_cfg.get("layer_index", 0),
        }
        # Remove None values so env vars aren't overridden
        overrides = {k: v for k, v in overrides.items() if v is not None}

    # pydantic-settings env vars take precedence over these defaults
    return MegaDNASettings(**overrides)
