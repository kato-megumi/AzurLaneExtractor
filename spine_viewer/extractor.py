"""Extractor and cache manager for Azur Lane Spine 2D painting assets."""
import json
import logging
import re
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import UnityPy

# Ensure UnityPy uses compatible version for Azur Lane 5.x.x headers
UnityPy.config.FALLBACK_UNITY_VERSION = "2020.3.48f1"

log = logging.getLogger("spine_viewer.extractor")


def extract_text_asset_content(obj) -> tuple[str, bytes]:
    """Extract TextAsset name and raw bytes reliably from a UnityPy object."""
    raw = obj.get_raw_data()
    nlen = struct.unpack("<I", raw[:4])[0]
    pos = 4 + nlen
    if pos % 4 != 0:
        pos += 4 - (pos % 4)
    name = raw[4:4 + nlen].decode("utf-8", errors="replace")

    slen = struct.unpack("<I", raw[pos:pos + 4])[0]
    if pos + 4 + slen > len(raw):
        # Scan forward for length that matches total byte length
        for candidate_pos in range(4 + nlen, min(len(raw) - 4, 128), 4):
            candidate_len = struct.unpack("<I", raw[candidate_pos:candidate_pos + 4])[0]
            if candidate_pos + 4 + candidate_len == len(raw):
                pos = candidate_pos
                slen = candidate_len
                break

    content = raw[pos + 4:pos + 4 + slen]
    return name, content


def pair_skels_and_atlases(skels: List[str], atlases: List[str]) -> List[Dict[str, Any]]:
    """Pair skeletons with their matching atlas and assign rendering order.

    Azur Lane models often use multi-part skeletons:
      - B / _bg: Background layer (order 0)
      - M: Midground layer (order 5)
      - T: Top / Foreground character layer (order 10)
      - Numbered (e.g. 1..7): Sequential layers (order 1..7)
    """
    def get_stem(filename: str) -> str:
        return re.sub(r"\.(skel|skel\.bytes|atlas|atlas\.txt|json)$", "", filename)
    atlas_map = {get_stem(a): a for a in atlases}

    layers = []
    for s in skels:
        stem = get_stem(s)
        # 1. Exact match
        matched_atlas = atlas_map.get(stem)
        if not matched_atlas:
            # 2. Try stripping _hx or variant
            clean_stem = stem.replace("_hx", "")
            matched_atlas = atlas_map.get(clean_stem)
        if not matched_atlas and atlases:
            matched_atlas = atlases[0]

        is_hx = "_hx" in stem
        clean_stem = stem.replace("_hx", "")

        order = 10
        role = "Character"
        if clean_stem.endswith("B") or clean_stem.endswith("_bg"):
            order = 0
            role = "Background"
        elif clean_stem.endswith("M"):
            order = 5
            role = "Midground"
        elif clean_stem.endswith("T"):
            order = 10
            role = "Foreground"
        elif re.search(r"\d+$", clean_stem):
            num = int(re.search(r"\d+$", clean_stem).group())
            order = num
            role = f"Layer {num}"

        layers.append({
            "id": stem,
            "skel": s,
            "atlas": matched_atlas,
            "is_hx": is_hx,
            "is_json": s.endswith(".json"),
            "order": order,
            "role": role,
        })
    layers.sort(key=lambda x: (x["is_hx"], x["order"]))
    return layers


def list_spine_models(spine_dir: Path, ship_collection=None) -> List[Dict[str, Any]]:
    """Scan spinepainting directory and match with ship metadata.

    Returns a list of dicts:
      {
        "id": prefab_name,
        "bundle_file": filename,
        "ship_name": "Shoukaku",
        "skin_name": "The Crane's Gratitude",
        "skin_id": 307053,
        "tags": ["dynamic", "bg"],
        "size_mb": 8.5,
        "has_hx": False,
      }
    """
    if not spine_dir.is_dir():
        log.warning(f"Spine directory does not exist: {spine_dir}")
        return []

    files = [f for f in spine_dir.iterdir() if f.is_file()]
    model_dict: Dict[str, Dict[str, Any]] = {}

    for f in files:
        name = f.name
        if name.endswith("_res"):
            prefab = name[:-4]  # Strip _res
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            model_dict[prefab] = {
                "id": prefab,
                "bundle_file": name,
                "size_mb": size_mb,
            }
        elif f.stat().st_size > 500_000:
            # Standalone bundles like 2b_2, a2_2, yingrui_4
            prefab = name
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            model_dict.setdefault(prefab, {
                "id": prefab,
                "bundle_file": name,
                "size_mb": size_mb,
            })

    # Load fallback name_index once if ship_collection is absent
    name_index = None
    if not ship_collection:
        try:
            from azurlane_extractor.name_map import _fetch_name_index
            import requests_cache
            session = requests_cache.CachedSession("AzurlaneCache")
            name_index = _fetch_name_index(session)
        except Exception:
            pass

    results = []
    for prefab, item in model_dict.items():
        ship_name = ""
        skin_name = ""
        skin_type = ""
        skin_id = -1
        tags = []

        # Remove _hx for metadata matching if needed
        base_prefab = prefab[:-3] if prefab.endswith("_hx") else prefab

        # 1. Match from ShipCollection
        if ship_collection:
            skin = ship_collection.skins_by_painting(base_prefab)
            if not skin and prefab != base_prefab:
                skin = ship_collection.skins_by_painting(prefab)

            if skin:
                skin_id = skin.skin_id
                skin_name = skin.name
                skin_type = getattr(skin, "type", "")
                tags = list(skin.tag)
                if skin.ship:
                    ship_name = skin.ship.name

        # 2. Fallback to name_index resolver if unresolved
        if (not ship_name or skin_id == -1) and name_index:
            try:
                from azurlane_extractor.name_map import _resolve_painting_name
                meta = _resolve_painting_name(base_prefab, name_index)
                if not meta and prefab != base_prefab:
                    meta = _resolve_painting_name(prefab, name_index)
                if meta:
                    ship_name = meta.get("ship_name", "")
                    skin_name = meta.get("skin_name", "")
                    skin_id = meta.get("skin_id", -1)
                    skin_type = meta.get("type", "")
                    tags = meta.get("tag", [])
            except Exception:
                pass
        # Normalize skin_name
        if skin_name and ship_name and skin_name.lower() == ship_name.lower():
            skin_name = "Default"
        elif not skin_name and not ship_name:
            parts = base_prefab.split("_")
            ship_name = parts[0].capitalize()
            skin_name = " ".join(parts[1:]).capitalize() if len(parts) > 1 else "Default"
        elif not skin_name:
            skin_name = "Default"

        has_hx = (f"{prefab}_hx_res" in [f.name for f in files]) or ("_hx" in prefab)

        results.append({
            "id": prefab,
            "bundle_file": item["bundle_file"],
            "ship_name": ship_name,
            "skin_name": skin_name,
            "skin_type": skin_type,
            "skin_id": skin_id,
            "tags": tags,
            "size_mb": item["size_mb"],
            "has_hx": has_hx,
        })

    # Sort by ship name, then skin name
    results.sort(key=lambda x: (x["ship_name"].lower(), x["skin_name"].lower()))
    return results


def extract_spine_model(bundle_path: Path, cache_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Extract .skel, .atlas, and .png textures from a spine bundle.

    Caches results in cache_dir / prefab.
    Returns manifest dictionary with paths of extracted components.
    """
    prefab = bundle_path.name
    if prefab.endswith("_res"):
        prefab = prefab[:-4]

    out_dir = cache_dir / prefab
    manifest_path = out_dir / "manifest.json"

    # Fast path: already cached
    if not force and manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "layers" not in data and "skels" in data and "atlases" in data:
                    data["layers"] = pair_skels_and_atlases(data["skels"], data["atlases"])
                    with open(manifest_path, "w", encoding="utf-8") as fw:
                        json.dump(data, fw, indent=2)
                if data.get("layers") and data.get("skels"):
                    return data
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    env = UnityPy.load(str(bundle_path))

    skels = []
    atlases = []
    textures = []

    for obj in env.objects:
        if obj.type.name == "TextAsset":
            name, content = extract_text_asset_content(obj)
            target = out_dir / name
            target.write_bytes(content)
            if name.endswith(".skel") or name.endswith(".skel.bytes"):
                skels.append(name)
            elif name.endswith(".atlas") or name.endswith(".atlas.txt"):
                atlases.append(name)
            elif name.endswith(".json"):
                skels.append(name)
            elif content.lstrip().startswith(b"{") and (b'"skeleton"' in content[:500] or b'"bones"' in content[:1000]):
                name = f"{name}.json"
                skels.append(name)
        elif obj.type.name == "Texture2D":
            data = obj.read()
            name = getattr(data, "m_Name", getattr(data, "name", "texture"))
            img = data.image
            filename = f"{name}.png"
            img.save(out_dir / filename)
            textures.append(filename)

    manifest = {
        "id": prefab,
        "skels": skels,
        "atlases": atlases,
        "textures": textures,
        "layers": pair_skels_and_atlases(skels, atlases),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


