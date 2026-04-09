"""Standardized configuration for deeppl-service using pydantic-settings.

Supports configuration via environment variables (DEEPPL_ prefix),
.env file, or YAML config file (for backward compatibility).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepPLSettings(BaseSettings):
    """DeepPL service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPPL_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8004

    # Model
    model_path: str = ""
    kmer: int = 6
    window_bp: int = 105
    max_seq_length: int = 100
    stride: int = 10
    max_batch_size: int = 4096

    # Voting thresholds
    confidence_threshold: float = 0.9
    lysogenic_window_fraction: float = 0.016

    @field_validator("model_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> str:
        return str(Path(v).expanduser())


def load_settings() -> DeepPLSettings:
    """Load settings, merging from config.yaml for backward compat."""
    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    overrides: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        model_cfg = raw.get("model", {})
        server_cfg = raw.get("server", {})
        overrides = {
            "model_path": model_cfg.get("path", ""),
            "kmer": model_cfg.get("kmer", 6),
            "window_bp": model_cfg.get("window_bp", 105),
            "max_seq_length": model_cfg.get("max_seq_length", 100),
            "stride": model_cfg.get("stride", 10),
            "max_batch_size": model_cfg.get("max_batch_size", 4096),
            "confidence_threshold": model_cfg.get("confidence_threshold", 0.9),
            "lysogenic_window_fraction": model_cfg.get("lysogenic_window_fraction", 0.016),
            "host": server_cfg.get("host", "0.0.0.0"),
            "port": server_cfg.get("port", 8004),
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}

    return DeepPLSettings(**overrides)
