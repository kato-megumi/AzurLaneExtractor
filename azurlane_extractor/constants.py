"""Constant definitions and helper utilities."""
import re

# Name map caching
SHIP_SKIN_URL = "https://raw.githubusercontent.com/Fernando2603/AzurLane/main/ship_skin_list.json"
SHIP_SKIN_TEMPLATE_URL = "https://raw.githubusercontent.com/AzurLaneTools/AzurLaneData/main/EN/ShareCfg/ship_skin_template.json"
PAINTING_MAP_URL = "https://raw.githubusercontent.com/AzurLaneTools/AzurLaneData/main/EN/ShareCfg/painting_filte_map.json"
SECRETARY_SHIP_URL = "https://raw.githubusercontent.com/AzurLaneTools/AzurLaneData/refs/heads/main/EN/ShareCfg/secretary_special_ship.json"
CACHE_NAME = "AzurlaneCache"

# Resource-suffix tokens used to group painting sub-resource bundles back to
# their base painting prefab. A local file belongs to base painting P when its
# name equals P or P followed by a sequence of these underscore-delimited tokens
# (e.g. "aotuo_3_n_rw_hx_tex" -> base "aotuo_3"). Tokens that denote a separate
# painting/skin (numbers, "g", "h", "alter", "wjz", ...) are intentionally
# excluded so distinct skins are never merged.
RES_SUFFIX_TOKENS: frozenset[str] = frozenset({
    "tex", "n", "hx", "rw", "shophx", "shadow",
    "bj", "bj1", "bj2", "bg", "bg1", "bd",
    "front", "mid", "middle", "back",
    "pt", "ex", "jz", "jz1", "jz2",
    "rank", "renwu", "f1", "f2", "tx2", "tx3",
})

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
