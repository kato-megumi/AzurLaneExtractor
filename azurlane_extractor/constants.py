"""Constant definitions and helper utilities."""
import re
from pathlib import Path

# Name map caching
SHIP_SKIN_URL = "https://raw.githubusercontent.com/Fernando2603/AzurLane/main/ship_skin_list.json"
SKIN_PAINTING_URL = "https://raw.githubusercontent.com/AzurLaneTools/AzurLaneData/main/EN/ShareCfg/ship_skin_template.json"
PAINTING_MAP_URL = "https://raw.githubusercontent.com/AzurLaneTools/AzurLaneData/main/EN/ShareCfg/painting_filte_map.json"
CACHE_NAME = "AzurlaneCache"

# Mesh reconstruction regex patterns (compiled once)
MESH_VR = re.compile(r'v ')
MESH_TR = re.compile(r'vt ')
MESH_SR = re.compile(r' ')

# Hardcoded manual layer position overrides
# Format: "layer_name": (x, y)
# These are applied automatically and can be overridden by --layer-pos command line argument
LAYER_POSITION_OVERRIDES: dict[str, tuple[int, int]] = {
    "aotuo_3_rw": (862,370),
    "xiefeierde_4_front": (2437, 394),  # Found via coherence matching
}
