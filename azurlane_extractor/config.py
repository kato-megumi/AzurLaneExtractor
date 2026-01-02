"""Configuration and logging setup for Azur Lane Extractor."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname).1s] %(message)s' if debug else '%(message)s',
        handlers=[logging.StreamHandler()]
    )


@dataclass
class Config:
    """Global configuration."""
    debug: bool = False
    external_scaler: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("."))
    extract_no_bg: bool = False  # Extract _n (no background) variants
    save_layers: bool = False  # Save intermediate layers (debug)
    name_map: dict[str, str] = field(default_factory=dict)
    reverse_map: dict[str, list[str]] = field(default_factory=dict)
    base_map: dict[str, str] = field(default_factory=dict)


# Global config instance
_config = Config()


def get_config() -> Config:
    """Get global config instance."""
    return _config
