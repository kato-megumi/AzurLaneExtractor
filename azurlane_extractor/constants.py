"""Constant definitions and helper utilities."""
import re
from pathlib import Path

# Variant suffixes - order matters: longer first for proper stripping
NO_BG_SUFFIXES = ("_n_hx", "_n")  # No background variants (optional)
OTHER_SUFFIXES = ("_hx",)  # Other variants (always extracted)
VARIANT_SUFFIXES = NO_BG_SUFFIXES + OTHER_SUFFIXES  # Combined for stripping
VARIANT_LABELS = {"_n": "(No BG)", "_hx": "(Censored)", "_n_hx": "(No BG Censored)"}

# Name map caching
NAME_MAP_CACHE = Path(__file__).parent.parent / "name_map_cache.json"
CACHE_MAX_AGE = 86400  # 24 hours

# Mesh reconstruction regex patterns (compiled once)
MESH_VR = re.compile(r'v ')
MESH_TR = re.compile(r'vt ')
MESH_SR = re.compile(r' ')


def strip_variant_suffix(name: str) -> tuple[str, str]:
    """Strip variant suffix from painting name. Returns (base_name, suffix)."""
    for suffix in VARIANT_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    return name, ""


def find_painting_variants(painting_name: str, asset_dir: Path, include_no_bg: bool = False) -> list[str]:
    """Find variants of a painting that exist in the asset directory.
    
    Args:
        painting_name: Base painting name
        asset_dir: Asset directory path
        include_no_bg: Whether to include _n (no background) variants
    
    Returns:
        List of variant names (always includes _hx, optionally _n/_n_hx)
    """
    painting_dir = asset_dir / "painting"
    variants = [painting_name]
    
    # Always include other variants like _hx (censored)
    for suffix in OTHER_SUFFIXES:
        variant_name = painting_name + suffix
        if (painting_dir / variant_name).exists():
            variants.append(variant_name)
    
    # Only include no-bg variants if requested
    if include_no_bg:
        for suffix in NO_BG_SUFFIXES:
            variant_name = painting_name + suffix
            if (painting_dir / variant_name).exists():
                variants.append(variant_name)
    
    return variants
