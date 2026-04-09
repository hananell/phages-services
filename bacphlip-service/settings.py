"""Standardized configuration for bacphlip-service using pydantic-settings.

Supports configuration via environment variables (BACPHLIP_ prefix),
.env file, or YAML config file (for backward compatibility).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class BacphlipSettings(BaseSettings):
    """BACPHLIP service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BACPHLIP_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8003

    # BACPHLIP
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    hmmer_threads: int = 4
    cleanup_intermediate: bool = True


def load_settings() -> BacphlipSettings:
    """Load settings, merging from config.yaml for backward compat."""
    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    overrides: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        server_cfg = raw.get("server", {})
        overrides = {
            "data_dir": raw.get("data_dir", "data"),
            "output_dir": raw.get("output_dir", "output"),
            "hmmer_threads": raw.get("hmmer_threads", 4),
            "cleanup_intermediate": raw.get("cleanup_intermediate", True),
            "host": server_cfg.get("host", "0.0.0.0"),
            "port": server_cfg.get("port", 8003),
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}

    return BacphlipSettings(**overrides)
