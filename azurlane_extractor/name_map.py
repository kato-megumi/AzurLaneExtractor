"""Name mapping functionality for character/skin names."""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from requests_cache import CachedSession

from .constants import NAME_MAP_CACHE, VARIANT_LABELS, strip_variant_suffix
from .config import get_config

log = logging.getLogger(__name__)

SHIP_SKIN_URL = "https://raw.githubusercontent.com/Fernando2603/AzurLane/main/ship_skin_list.json"
SKIN_PAINTING_URL = "https://raw.githubusercontent.com/AzurLaneTools/AzurLaneData/main/EN/ShareCfg/ship_skin_template.json"
PAINTING_MAP_URL = "https://raw.githubusercontent.com/AzurLaneTools/AzurLaneData/main/EN/ShareCfg/painting_filte_map.json"

@dataclass
class Skin:
    skin_id: int
    painting: str # main painting asset
    res_list: List[str] = field(default_factory=list)
    name: str = ""
    type: str = ""
    have_censor: bool = False
    ship: Optional["Ship"] = None
    tag: List[str] = field(default_factory=list)
    
    def display_name(self):
        if self.type == "Default":
            return self.name
        else:
            return f"{self.ship.name} - {self.name} ({self.type})"
    
@dataclass
class Ship:
    id: int
    name: str
    skins: List["Skin"] = field(default_factory=list)
    
@dataclass
class ShipCollection:
    ships: List[Ship] = field(default_factory=list)
    skins: List[Skin] = field(default_factory=list)
    _painting_map: Dict[str, Skin] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        self._painting_map = {s.painting: s for s in self.skins}
    
    def get_skin(self, painting_name: str) -> Optional[Skin]:
        """Get Skin object by painting asset name (strips variants)."""
        base_name, _ = strip_variant_suffix(painting_name)
        return self._painting_map.get(base_name.lower())

    def get_display_name(self, painting_name: str) -> str:
        """Get formatted display name: 'CharName - SkinName (Type)'."""
        base_name, variant_suffix = strip_variant_suffix(painting_name)
        skin = self.get_skin(base_name)
        if not skin:
            return painting_name
        display = skin.display_name()
        if variant_suffix:
            display = f"{display} {VARIANT_LABELS.get(variant_suffix, variant_suffix)}"
        return display

    def get_char_and_skin_name(self, painting_name: str) -> Tuple[str, str]:
        """Get (char_name, skin_folder_name) for folder structure."""
        skin = self.get_skin(painting_name)
        if not skin:
            return (painting_name, painting_name)
        return (skin.ship.name, self.get_display_name(painting_name))

    def find_paintings(self, query: str) -> List[str]:
        """Find all painting names matching a character or skin name."""
        query = query.lower().strip()
        matches = set()
        for ship in self.ships:
            if query in ship.name.lower():
                for skin in ship.skins:
                    matches.add(skin.painting)
        for skin in self.skins:
            if query in skin.display_name().lower():
                matches.add(skin.painting)
        return sorted(list(matches))

    def ships_by_name(self, name: str) -> List[Ship]:
        return [ship for ship in self.ships if name.lower() in ship.name.lower()]
    
    def skins_by_name(self, name: str) -> List[Skin]:
        return [skin for skin in self.skins if name.lower() in skin.display_name().lower()]
    
    def ship_by_id(self, ship_id: int) -> Optional[Ship]:
        for ship in self.ships:
            if ship.id == ship_id:
                return ship
        return None
    
    def skin_by_id(self, skin_id: int) -> Optional[Skin]:
        for skin in self.skins:
            if skin.skin_id == skin_id:
                return skin
        return None

def fetch_name_map() -> ShipCollection:
    """Fetch/load ship and skin data. Returns a ShipCollection."""
    config = get_config()
    
    # Use requests_cache for efficient fetching
    session = CachedSession(str(NAME_MAP_CACHE.with_suffix('')), cache_control=True)
    
    try:
        log.debug("Fetching ship skin data...")
        resp = session.get(SKIN_PAINTING_URL, timeout=15)
        skin_painting = resp.json()
        resp = session.get(PAINTING_MAP_URL, timeout=15)
        painting_map = resp.json()
        resp = session.get(SHIP_SKIN_URL, timeout=15)
        ship_skin_list = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch name map data: {e}")
        return ShipCollection()

    ship_collection = ShipCollection()

    for entry in ship_skin_list:
        ship_id = entry.get("gid")
        ship = Ship(id=ship_id, name=entry.get("name"))
        ship_collection.ships.append(ship)

        for s in entry.get("skins"):
            skin_id = s.get("id")
            skin_data = skin_painting.get(str(skin_id))
            if not skin_data:
                continue
                
            painting = skin_data.get("painting").lower()
            
            p_map_entry = painting_map.get(painting)
            if p_map_entry:
                res_list = p_map_entry.get("res_list", []).copy()
            else:
                res_list = [painting]
                
            res_list = [res for res in res_list if "shophx" not in res.lower()]
            
            skin_obj = Skin(
                skin_id=skin_id,
                painting=painting,
                res_list=res_list,
                have_censor=any("_hx" in res.lower() for res in res_list),
                name=s.get("name"),
                type=s.get("type"),
                ship=ship,
                tag=(s.get("tag") or []).copy()
            )
            ship.skins.append(skin_obj)
            ship_collection.skins.append(skin_obj)

    # Re-initialize to build internal maps
    ship_collection.__post_init__()
    config.ship_collection = ship_collection
    log.debug(f"Loaded {len(ship_collection.ships)} ships and {len(ship_collection.skins)} skins")
    return ship_collection
