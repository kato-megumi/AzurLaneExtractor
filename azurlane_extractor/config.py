"""Configuration and logging setup for Azur Lane Extractor."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from tqdm import tqdm
from .constants import LAYER_POSITION_OVERRIDES

if TYPE_CHECKING:
    from .upscaler import ImageUpscaler

log = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that uses tqdm.write() to avoid breaking progress bars."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(debug: bool = False):
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname).1s] %(message)s' if debug else '%(message)s',
        handlers=[TqdmLoggingHandler()]
    )


@dataclass
class Config:
    """Global configuration."""
    debug: bool = False
    asset_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("."))
    save_textures: bool = False  # Save temporary texture layers
    ship_collection: Optional[object] = None
    upscaler: Optional["ImageUpscaler"] = None  # AI upscaler instance
    layer_position_overrides: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(LAYER_POSITION_OVERRIDES)
    )  # Manual position overrides (starts with hardcoded defaults, can be extended via CLI)


# Global config instance
_config = Config()


def get_config() -> Config:
    """Get global config instance."""
    return _config
