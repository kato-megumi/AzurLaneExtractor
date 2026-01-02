"""Name mapping functionality for character/skin names."""
import json
import logging
import re
import time
import urllib.request
import urllib.error

from .constants import NAME_MAP_URL, NAME_MAP_CACHE, CACHE_MAX_AGE, VARIANT_LABELS, strip_variant_suffix
from .config import get_config

log = logging.getLogger(__name__)


def fetch_name_map() -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Fetch/load painting -> english name mapping. Returns (name_map, reverse_map, base_map)."""
    data = None
    cached_etag = None
    cached_last_modified = None
    cache_valid = False
    
    # Try loading from cache first
    if NAME_MAP_CACHE.exists():
        try:
            with open(NAME_MAP_CACHE, 'r', encoding='utf-8') as f:
                cache_obj = json.load(f)
            
            if isinstance(cache_obj, dict) and '_cache_meta' in cache_obj:
                cached_etag = cache_obj['_cache_meta'].get('etag')
                cached_last_modified = cache_obj['_cache_meta'].get('last_modified')
                cache_time = cache_obj['_cache_meta'].get('cached_at', 0)
                data = cache_obj.get('data')
                
                if time.time() - cache_time < CACHE_MAX_AGE:
                    cache_valid = True
                    log.debug(f"NAME_MAP cache is fresh (age: {int(time.time() - cache_time)}s)")
            else:
                data = cache_obj
            
            log.debug(f"NAME_MAP loaded from cache: {NAME_MAP_CACHE}")
        except Exception as e:
            log.debug(f"NAME_MAP cache read failed: {e}")
    
    # Fetch from network if no cache or cache is stale
    if data is None or not cache_valid:
        try:
            log.debug(f"NAME_MAP fetching from {NAME_MAP_URL}")
            
            req = urllib.request.Request(NAME_MAP_URL)
            if cached_etag:
                req.add_header('If-None-Match', cached_etag)
            if cached_last_modified:
                req.add_header('If-Modified-Since', cached_last_modified)
            
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    new_data = json.loads(response.read().decode('utf-8'))
                    new_etag = response.headers.get('ETag')
                    new_last_modified = response.headers.get('Last-Modified')
                    
                    cache_obj = {
                        '_cache_meta': {
                            'etag': new_etag,
                            'last_modified': new_last_modified,
                            'cached_at': time.time()
                        },
                        'data': new_data
                    }
                    try:
                        with open(NAME_MAP_CACHE, 'w', encoding='utf-8') as f:
                            json.dump(cache_obj, f)
                        log.debug(f"NAME_MAP cached to: {NAME_MAP_CACHE}")
                    except Exception as e:
                        log.debug(f"NAME_MAP cache write failed: {e}")
                    
                    data = new_data
                    log.debug("NAME_MAP fetched new data")
            except urllib.error.HTTPError as e:
                if e.code == 304:
                    log.debug("NAME_MAP not modified (304), using cache")
                    if NAME_MAP_CACHE.exists():
                        try:
                            with open(NAME_MAP_CACHE, 'r', encoding='utf-8') as f:
                                cache_obj = json.load(f)
                            cache_obj['_cache_meta']['cached_at'] = time.time()
                            with open(NAME_MAP_CACHE, 'w', encoding='utf-8') as f:
                                json.dump(cache_obj, f)
                        except Exception:
                            pass
                else:
                    raise
        except Exception as e:
            log.debug(f"NAME_MAP failed to fetch: {e}")
            if data is None:
                return {}, {}, {}
    
    # Build mappings
    name_map = {}
    reverse_map = {}
    base_map = {}
    
    def has_cjk(text: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in text)
    
    for entry in data.values():
        if isinstance(entry, dict) and 'painting' in entry and 'name' in entry:
            painting = entry['painting']
            name = entry['name']
            if painting and name:
                clean_name = re.sub(r'[<>:"/\\|?*]', '', name).strip()
                if clean_name:
                    if painting in name_map:
                        if not has_cjk(name_map[painting]) or has_cjk(clean_name):
                            continue
                    name_map[painting] = clean_name
                    key = clean_name.lower()
                    if key not in reverse_map:
                        reverse_map[key] = []
                    if painting not in reverse_map[key]:
                        reverse_map[key].append(painting)
                    # Track base painting
                    base = re.sub(r'_(\d+|g|h)$', '', painting)
                    base = re.sub(r'_(\d+)_([a-z]+)$', r'_\2', base)
                    base_map[painting] = base
    
    log.debug(f"NAME_MAP loaded {len(name_map)} mappings")
    return name_map, reverse_map, base_map


def find_paintings_by_char_name(char_name: str) -> list[str]:
    """Find all painting names for a character by English name (case-insensitive, partial match)."""
    config = get_config()
    
    if not config.reverse_map:
        return [char_name]
    
    char_lower = char_name.lower().strip()
    matches = set()
    
    for key, paintings in config.reverse_map.items():
        if char_lower == key or char_lower in key:
            matches.update(paintings)
    
    for painting, eng_name in config.name_map.items():
        if char_lower in eng_name.lower():
            matches.add(painting)
    
    if not matches:
        return [char_name]
    
    # Find all variants
    base_paintings = set(matches)
    for base in list(base_paintings):
        for painting in config.name_map.keys():
            if painting.startswith(base + "_") and painting not in base_paintings:
                matches.add(painting)
    
    matched_bases = {config.base_map.get(p) for p in matches if config.base_map.get(p)}
    for painting, base in config.base_map.items():
        if base in matched_bases and painting not in matches:
            matches.add(painting)
    
    return sorted(matches)


def get_display_name(painting_name: str) -> str:
    """Get display name for a painting. Format: 'CharName - SkinName' or just 'CharName'."""
    config = get_config()
    base_name, variant_suffix = strip_variant_suffix(painting_name)
    
    if not config.name_map:
        return painting_name
    
    skin_name = config.name_map.get(base_name)
    if not skin_name:
        return painting_name
    
    base_painting = config.base_map.get(base_name, base_name)
    
    if base_painting == base_name:
        display = skin_name
    else:
        char_name = config.name_map.get(base_painting, base_painting)
        if skin_name.lower().startswith(char_name.lower().split()[0].lower()):
            display = skin_name
        else:
            display = f"{char_name} - {skin_name}"
    
    if variant_suffix:
        display = f"{display} {VARIANT_LABELS.get(variant_suffix, variant_suffix)}"
    
    return display


def get_char_and_skin_name(painting_name: str) -> tuple[str, str]:
    """Get (char_name, skin_folder_name) for folder structure."""
    config = get_config()
    base_name, _ = strip_variant_suffix(painting_name)
    
    if not config.name_map:
        return (painting_name, painting_name)
    
    skin_name = config.name_map.get(base_name)
    if not skin_name:
        return (painting_name, painting_name)
    
    base_painting = config.base_map.get(base_name, base_name)
    char_name = skin_name if base_painting == base_name else config.name_map.get(base_painting, base_painting)
    
    return (char_name, get_display_name(painting_name))
