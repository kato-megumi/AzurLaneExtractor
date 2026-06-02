"""Name mapping functionality for character/skin names."""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

from requests_cache import CachedSession

from .constants import (
    CACHE_NAME,
    SECRETARY_SHIP_URL,
    SHIP_SKIN_URL,
    SHIP_SKIN_TEMPLATE_URL,
    PAINTING_MAP_URL,
    RES_SUFFIX_TOKENS,
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
        

def _strip_res_suffixes(name: str) -> str:
    """Strip trailing resource-suffix tokens to recover the base painting name.

    e.g. "aotuo_3_n_rw_hx_tex" -> "aotuo_3". Tokens denoting a separate skin
    (numbers, "g", "h", "alter", "wjz", ...) are not in RES_SUFFIX_TOKENS, so
    distinct paintings are never collapsed together.
    """
    cur = name
    while True:
        idx = cur.rfind("_")
        if idx <= 0:
            return cur
        if cur[idx + 1:] in RES_SUFFIX_TOKENS:
            cur = cur[:idx]
        else:
            return cur


def _collect_local_paintings(config) -> Dict[str, List[str]]:
    """Enumerate the local painting bundle directory and group sub-resource
    bundles under their base painting prefab.

    Returns a mapping of base painting name -> list of related bundle filenames
    (reconstructing what painting_filte_map's res_list would contain). This stays
    as fresh as the downloaded assets, independent of stale upstream metadata.
    """
    painting_dir = config.asset_dir / "painting"
    if not painting_dir.is_dir():
        return {}

    files = [p.name for p in painting_dir.iterdir() if p.is_file()]
    fileset = set(files)
    groups: Dict[str, List[str]] = {}
    for f in files:
        base = _strip_res_suffixes(f)
        if base not in fileset:
            # No bare base bundle exists; treat the file as its own base.
            base = f
        groups.setdefault(base, []).append(f)
    return groups


def _make_local_skin(painting: str, files: List[str],
                     meta: Optional[Dict] = None) -> Skin:
    """Build a synthetic Skin from locally enumerated bundle files.

    If ``meta`` (resolved English ship/skin name info) is provided, it is used
    to populate the skin's name/type/tag; otherwise the painting prefab is used
    as a fallback display name.
    """
    res_list = [f for f in files
                if "shophx" not in f.lower() and "shadow" not in f.lower()]
    # Mirror the official ordering convention: most-suffixed first, base last.
    res_list.sort(key=lambda r: (-len(r), r))

    have_censor = any("_hx" in res.lower() and "_n_hx" not in res.lower()
                      for res in res_list)
    texture_only_censor = have_censor and (painting + "_hx") not in res_list
    remap: Dict[str, str] = {}
    for res in res_list:
        if "_hx" in res.lower() and "_tex" in res.lower():
            remap[res.replace("_hx", "").replace("_tex", "")] = res.replace("_tex", "")

    skin_id = meta.get("skin_id", -1) if meta else -1
    name = meta.get("skin_name") if meta else None
    type_ = meta.get("type", "") if meta else ""
    tag = list(meta.get("tag", [])) if meta else []

    return Skin(
        skin_id=skin_id,
        painting=painting,
        res_list=res_list,
        have_censor=have_censor,
        texture_only_censor=texture_only_censor,
        remap=remap,
        name=name or painting,
        type=type_ or "",
        tag=tag,
    )


def _build_name_index(ship_skin_list, *templates) -> Dict:
    """Build the lookup structures used to resolve English names for a painting
    prefab, including support for skins too new to exist in any template.

    Returns a dict with:
      - ``direct``: ``{painting_lower: {skin_id, ship_name, skin_name, type, tag}}``
        for paintings present in a template (joined to Fernando by skin_id).
      - ``name_by_id``: ``{skin_id: {ship_name, skin_name, type, tag}}`` (Fernando).
      - ``ids_by_group``: ``{ship_group: set(skin_id)}`` (Fernando).
      - ``painting_to_group``: ``{painting_lower: ship_group}`` (templates).
      - ``numbered_by_group``: ``{ship_group: [(suffix_int, skin_id), ...]}`` for
        paintings whose prefab ends in ``_<number>`` (templates).

    The last three enable extrapolating a brand-new painting's skin_id from its
    numbered siblings (skin_id increases by 1 with each painting number within a
    ship), which is then resolved against Fernando's current name data.
    When multiple templates are passed, earlier ones win on conflicts.
    """
    name_by_id: Dict[int, Dict] = {}
    ids_by_group: Dict[int, set] = {}
    for entry in ship_skin_list or []:
        gid = entry.get("gid")
        ship_name = entry.get("name", "")
        for s in entry.get("skins", []) or []:
            sid = s.get("id")
            if sid is None:
                continue
            name_by_id[int(sid)] = {
                "ship_name": ship_name,
                "skin_name": s.get("name", ""),
                "type": s.get("type", ""),
                "tag": (s.get("tag") or []),
            }
            if gid is not None:
                ids_by_group.setdefault(int(gid), set()).add(int(sid))

    direct: Dict[str, Dict] = {}
    painting_to_group: Dict[str, int] = {}
    numbered_by_group: Dict[int, List] = {}
    for template in templates:
        if not template:
            continue
        for v in template.values():
            if not isinstance(v, dict):
                continue
            painting = v.get("painting") or v.get("prefab")
            sid = v.get("id")
            if not painting or sid is None:
                continue
            key = str(painting).lower()
            grp = v.get("ship_group")
            if grp is not None:
                painting_to_group.setdefault(key, int(grp))
                m = re.match(r"^(.*)_(\d+)$", key)
                if m:
                    numbered_by_group.setdefault(int(grp), []).append((int(m.group(2)), int(sid)))
            if key not in direct:
                info = name_by_id.get(int(sid))
                if info:
                    direct[key] = {"skin_id": int(sid), **info}

    return {
        "direct": direct,
        "name_by_id": name_by_id,
        "ids_by_group": ids_by_group,
        "painting_to_group": painting_to_group,
        "numbered_by_group": numbered_by_group,
    }


def _resolve_painting_name(painting: str, index: Dict) -> Optional[Dict]:
    """Resolve a painting prefab to ``{skin_id, ship_name, skin_name, type, tag}``.

    First tries a direct template join. If the painting is too new to appear in
    any template, extrapolates its skin_id from numbered siblings of the same
    ship (skin_id is contiguous with the painting number within a ship) and
    looks the name up in Fernando's current data.
    """
    key = painting.lower()
    direct = index.get("direct", {})
    if key in direct:
        return direct[key]

    m = re.match(r"^(.*)_(\d+)$", key)
    if not m:
        return None
    base, number = m.group(1), int(m.group(2))

    # Find the ship_group via the base (default) painting or any sibling.
    painting_to_group = index.get("painting_to_group", {})
    grp = painting_to_group.get(base)
    if grp is None:
        return None

    numbered = index.get("numbered_by_group", {}).get(grp, [])
    if not numbered:
        return None

    # Nearest known numbered sibling; skin_id moves in lockstep with the number.
    suffix_known, id_known = min(numbered, key=lambda t: abs(t[0] - number))
    candidate_id = id_known + (number - suffix_known)

    # Only trust the extrapolation if Fernando actually has that skin id for
    # this ship (guards against gaps / non-contiguous numbering).
    ids_by_group = index.get("ids_by_group", {})
    if candidate_id not in ids_by_group.get(grp, set()):
        return None

    info = index.get("name_by_id", {}).get(candidate_id)
    if not info:
        return None
    return {"skin_id": candidate_id, **info}


def _fetch_name_index(session) -> Dict:
    """Best-effort fetch of the data needed to resolve English painting names.

    Used by the offline fallback path; each source is optional so a partial
    network outage still yields whatever names are reachable.
    """
    def _get(url):
        try:
            r = session.get(url, timeout=15)
            data = r.json()
            r.close()
            return data
        except Exception:
            return None

    fernando = _get(SHIP_SKIN_URL) or []
    en_template = _get(SHIP_SKIN_TEMPLATE_URL) or {}
    return _build_name_index(fernando, en_template)


def _augment_with_local_paintings(ship_collection: "ShipCollection", config,
                                  name_index: Optional[Dict] = None) -> int:
    """Add synthetic skins for paintings present locally but missing from the
    (potentially stale) upstream metadata. Returns the number added.

    ``name_index`` optionally supplies the name lookup structures (from
    :func:`_build_name_index`) so synthetic skins are resolved to proper English
    ship/skin names, including new skins absent from every template.
    """
    try:
        local_paintings = _collect_local_paintings(config)
    except Exception as e:
        log.warning(f"Could not enumerate local paintings: {e}")
        return 0

    existing = {s.painting.lower() for s in ship_collection.skins}
    added = 0
    for base, files in local_paintings.items():
        if base.lower() in existing:
            continue
        meta = _resolve_painting_name(base, name_index) if name_index else None
        skin = _make_local_skin(base, files, meta)
        # Wrap in a Ship so asset loading can gather res_list. Use the resolved
        # ship name when available so display_name() reads naturally.
        ship_name = meta.get("ship_name") if meta else None
        ship = Ship(id=-1, name=ship_name or base, skins=[skin])
        skin.ship = ship
        ship_collection.ships.append(ship)
        ship_collection.skins.append(skin)
        existing.add(base.lower())
        added += 1
    return added


def fetch_name_map() -> ShipCollection:
    """Fetch/load ship and skin data. Returns a ShipCollection."""
    config = get_config()
    
    # Use requests_cache for efficient fetching
    # Store cache in project folder (parent of this module's directory)
    project_dir = Path(__file__).parent.parent
    cache_path = project_dir / CACHE_NAME
    session = CachedSession(str(cache_path), 
                            # cache_control=True,
                            expire_after=60*60*24,  # 1 day
                            )
    
    try:
        log.debug("Fetching ship skin data...")
        resp = session.get(SHIP_SKIN_TEMPLATE_URL, timeout=15)
        ship_skin_template = resp.json()
        resp.close()
        resp = session.get(PAINTING_MAP_URL, timeout=15)
        painting_map = resp.json()
        resp.close()
        resp = session.get(SHIP_SKIN_URL, timeout=15)
        ship_skin_list = resp.json()
        resp.close()
        resp = session.get(SECRETARY_SHIP_URL, timeout=15)
        secretary_ship_json = resp.json()
        resp.close()
    except Exception as e:
        log.error(f"Failed to fetch name map data: {e}")
        # Fall back to whatever paintings are available locally so extraction
        # can still proceed offline / when upstream data is stale. Still try to
        # resolve names from any data we can reach.
        offline = ShipCollection()
        name_index = _fetch_name_index(session)
        added = _augment_with_local_paintings(offline, config, name_index)
        offline.__post_init__()
        config.ship_collection = offline
        log.warning(f"Using local-only painting data ({added} paintings found).")
        return offline

    ship_collection = ShipCollection()

    for ship_id, skin_list in secretary_ship_json["get_id_list_by_character_id"].items():
        ship = Ship(id=int(ship_id), name=secretary_ship_json[str(ship_id)]["name"])
        ship_collection.ships.append(ship)
        for skin_id in skin_list:
            painting = secretary_ship_json[str(skin_id)]["prefab"]
            if painting in painting_map:
                res_list = painting_map[painting].get("res_list", [])
            else:
                res_list = [painting]
            res_list = [r[len("painting/"):] if r.startswith("painting/") else r for r in res_list]
            
            name = str(skin_id)
            for ship_id, ship_data in ship_skin_template.items():
                if ship_data["painting"] == painting:
                    name = ship_data["name"]
                    break
            skin = Skin(
                    skin_id=skin_id,
                    painting=painting,
                    res_list=res_list,
                    have_censor=False,
                    texture_only_censor=False,
                    remap={},
                    name=name,
                    type="Others",
                    ship=ship,
                    tag=[]
                )
            ship_collection.skins.append(skin)
            ship.skins.append(skin)

    for entry in ship_skin_list:
        ship_id = entry.get("gid")
        ship = Ship(id=ship_id, name=entry.get("name"))
        ship_collection.ships.append(ship)

        for s in entry.get("skins"):
            skin_id = s.get("id")
            skin_data = ship_skin_template.get(str(skin_id))
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

    # Augment with paintings available locally but missing from upstream data
    # (handles stale upstream by reading the actual downloaded asset filenames).
    # Resolve English names for the new paintings by joining Fernando's
    # ship_skin_list (names by skin_id) to the EN template (painting prefab by
    # skin_id), extrapolating skin_ids for skins too new for the template.
    name_index = _build_name_index(ship_skin_list, ship_skin_template)

    added = _augment_with_local_paintings(ship_collection, config, name_index)
    if added:
        log.debug(f"Added {added} local paintings missing from upstream metadata.")

    # Re-initialize to build internal maps
    ship_collection.__post_init__()
    config.ship_collection = ship_collection
    log.debug(f"Loaded {len(ship_collection.ships)} ships and {len(ship_collection.skins)} skins")
    return ship_collection
