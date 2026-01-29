"""Name mapping functionality for character/skin names."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

from requests_cache import CachedSession

from .constants import (
    CACHE_NAME,
    SHIP_SKIN_URL,
    SKIN_PAINTING_URL,
    PAINTING_MAP_URL,
)
from .config import get_config

log = logging.getLogger(__name__)


@dataclass
class Skin:
    skin_id: int
    painting: str  # main painting asset
    res_list: List[str] = field(default_factory=list)
    name: str = ""
    type: str = ""
    have_censor: bool = False
    texture_only_censor: bool = False
    remap: Dict[str, str] = field(default_factory=dict)
    ship: Optional["Ship"] = None
    tag: List[str] = field(default_factory=list)
    
    def display_name(self):
        if not self.ship or self.type == "Default":
            return self.name
        if self.type == "Retrofit":
            return self.name
        if not self.type:
            return f"{self.ship.name} - {self.name}"
        
        raw = f"{self.ship.name} - {self.name} ({self.type})"
        # Remove characters not allowed in filenames on Windows/most filesystems
        forbidden = '<>:"/\\|?*'
        sanitized = ''.join(c for c in raw if c not in forbidden and ord(c) >= 32)
        # Trim trailing spaces and dots (not allowed on Windows)
        sanitized = sanitized.rstrip(' .')
        return sanitized
    
@dataclass
class Ship:
    id: int
    name: str
    skins: List["Skin"] = field(default_factory=list)
    
    def default_skin(self) -> Optional[Skin]:
        for skin in self.skins:
            if skin.type == "Default":
                return skin
        return None
    
@dataclass
class ShipCollection:
    ships: List[Ship] = field(default_factory=list)
    skins: List[Skin] = field(default_factory=list)
    _painting_map: Dict[str, Skin] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        self._painting_map = {s.painting: s for s in self.skins}
    
    def skins_by_painting(self, painting_name: str) -> Optional[Skin]:
        return self._painting_map.get(painting_name.lower())

    def ships_by_name(self, name: str) -> List[Ship]:
        return [ship for ship in self.ships if name.lower() in ship.name.lower()]
    
    def skins_by_name(self, name: str) -> List[Skin]:
        return [skin for skin in self.skins if name.lower() in skin.name.lower()]
    
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
    
    def have_skin(self, painting_name: str) -> bool:
        """Check if a skin with the given painting name exists."""
        return painting_name.lower() in self._painting_map
        

def fetch_name_map() -> ShipCollection:
    """Fetch/load ship and skin data. Returns a ShipCollection."""
    config = get_config()
    
    # Use requests_cache for efficient fetching
    # Store cache in project folder (parent of this module's directory)
    project_dir = Path(__file__).parent.parent
    cache_path = project_dir / CACHE_NAME
    session = CachedSession(str(cache_path), cache_control=True)
    
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
            if (not skin_data) or (ship_id != skin_data.get("ship_group")):
                continue
                
            painting = skin_data.get("painting").lower()
            
            p_map_entry = painting_map.get(painting)
            if p_map_entry:
                res_list = p_map_entry.get("res_list", []).copy()
            else:
                res_list = [painting]
                
            res_list = [r[len("painting/"):] if r.startswith("painting/") else r for r in res_list]
            res_list = [res for res in res_list if "shophx" not in res.lower() and "shadow" not in res.lower()]
            
            name = s.get("name")
            if any(s for s in ship.skins if s.name == name):
                name += " Alt"

            have_censor = any("_hx" in res.lower() and "_n_hx" not in res.lower() for res in res_list)
            texture_only_censor = have_censor and (painting + "_hx") not in res_list
            # Build sprite name remapping for texture-only censoring
            # Maps uncensored sprite names to their censored _hx equivalents
            remap = {}
            for res in res_list:
                if "_hx" in res.lower() and "_tex" in res.lower():
                    remap[res.replace("_hx", "").replace("_tex", "")] = res.replace("_tex", "")
            
            skin_obj = Skin(
                skin_id=skin_id,
                painting=painting,
                res_list=res_list,
                have_censor=have_censor,
                texture_only_censor=texture_only_censor,
                remap=remap,
                name=name,
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
