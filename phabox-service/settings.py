"""Standardized configuration for phabox-service using pydantic-settings.

Supports configuration via environment variables (PHABOX_ prefix),
.env file, or YAML config file (for backward compatibility).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PhaboxSettings(BaseSettings):
    """PhaBOX service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PHABOX_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8005

    # PhaBOX
    dbdir: str = "./phabox_db_v2"
    min_len: int = 3000
    threads: int = 24
    timeout: int = 7200


def load_settings() -> PhaboxSettings:
    """Load settings, merging from config.yaml for backward compat."""
    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    overrides: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        phabox_cfg = raw.get("phabox", {})
        server_cfg = raw.get("server", {})
        overrides = {
            "dbdir": phabox_cfg.get("dbdir", "./phabox_db_v2"),
            "min_len": phabox_cfg.get("min_len", 3000),
            "threads": phabox_cfg.get("threads", 24),
            "timeout": phabox_cfg.get("timeout", 7200),
            "host": server_cfg.get("host", "0.0.0.0"),
            "port": server_cfg.get("port", 8005),
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}

    return PhaboxSettings(**overrides)
