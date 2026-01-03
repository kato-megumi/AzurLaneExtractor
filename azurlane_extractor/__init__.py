"""
Azur Lane Painting Extractor Package

Reconstructs character paintings from Unity assets with support for:
- Batch scaling with external tools
- Face overlay compositing
"""
from .config import Config, setup_logging, get_config
from .constants import (
    CACHE_NAME,
)
from .asset import AzurlaneAsset
from .layer import GameObjectLayer, RectTransform
from .name_map import fetch_name_map, Skin, Ship
from .extractor import process_painting_group, finalize_and_save, reset_state

__version__ = "1.0.0"
__all__ = [
    "Config",
    "setup_logging",
    "get_config",
    "CACHE_NAME",
    "AzurlaneAsset",
    "GameObjectLayer",
    "RectTransform",
    "fetch_name_map",
    "process_painting_group",
    "finalize_and_save",
    "reset_state",
]
