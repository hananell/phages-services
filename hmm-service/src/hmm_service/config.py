"""Configuration for HMM Service."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration loaded from environment or config file."""

    model_config = SettingsConfigDict(
        env_prefix="HMM_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8002
    workers: int = 1
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # HMM database settings
    phrogs_db_path: Path = Path("~/.phrogs/all_phrogs.h3m").expanduser()
    e_value_threshold: float = 1e-5
    bit_score_threshold: float | None = None

    # Processing settings
    batch_size: int = 1000
    cpus: int = 4

    @field_validator("phrogs_db_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand user home directory in path."""
        return Path(v).expanduser()


settings = Settings()
