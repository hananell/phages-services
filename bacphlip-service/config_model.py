"""Configuration model for bacphlip service."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8003


class Settings(BaseModel):
    """Configuration settings for bacphlip service."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    hmmer_threads: int = 4
    cleanup_intermediate: bool = True
    server: ServerConfig = ServerConfig()


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load settings from config.yaml.

    Args:
        config_path: Optional path to config file. Defaults to config.yaml
            in the current directory.

    Returns:
        Settings object with configuration values.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            config_data: dict = yaml.safe_load(f) or {}
        return Settings(**config_data)

    return Settings()
