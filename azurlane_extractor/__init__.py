"""
Azur Lane Painting Extractor Package

Reconstructs character paintings from Unity assets with support for:
- Batch scaling with external tools
- Face overlay compositing
- Variant extraction (_n, _hx, _n_hx)
"""
from .config import Config, setup_logging, get_config
from .constants import (
    strip_variant_suffix,
    find_painting_variants,
    NAME_MAP_CACHE,
    VARIANT_SUFFIXES,
    VARIANT_LABELS,
)
from .asset import AzurlaneAsset
from .layer import GameObjectLayer, RectTransform
from .name_map import fetch_name_map, find_paintings_by_char_name, get_display_name, get_char_and_skin_name
from .extractor import process_painting_group, finalize_and_save, reset_state

__version__ = "1.0.0"
__all__ = [
    "Config",
    "setup_logging",
    "get_config",
    "strip_variant_suffix",
    "find_painting_variants",
    "NAME_MAP_CACHE",
    "VARIANT_SUFFIXES",
    "VARIANT_LABELS",
    "AzurlaneAsset",
    "GameObjectLayer",
    "RectTransform",
    "fetch_name_map",
    "find_paintings_by_char_name",
    "get_display_name",
    "get_char_and_skin_name",
    "process_painting_group",
    "finalize_and_save",
    "reset_state",
]
