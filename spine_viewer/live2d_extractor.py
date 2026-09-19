"""Extractor and cache manager for Azur Lane Live2D (Cubism 3/4) models."""
import binascii
import json
import logging
import re
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import UnityPy

# Ensure UnityPy uses compatible version for Azur Lane 5.x.x headers
UnityPy.config.FALLBACK_UNITY_VERSION = "2020.3.48f1"

log = logging.getLogger("spine_viewer.live2d_extractor")


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
        for candidate_pos in range(4 + nlen, min(len(raw) - 4, 128), 4):
            candidate_len = struct.unpack("<I", raw[candidate_pos:candidate_pos + 4])[0]
            if candidate_pos + 4 + candidate_len == len(raw):
                pos = candidate_pos
                slen = candidate_len
                break

    content = raw[pos + 4:pos + 4 + slen]
    return name, content


def list_live2d_models(live2d_dir: Path, ship_collection=None) -> List[Dict[str, Any]]:
    """Scan live2d directory and match with ship metadata."""
    if not live2d_dir.exists():
        return []

    model_dict: Dict[str, Dict[str, Any]] = {}

    for f in live2d_dir.iterdir():
        if f.is_dir() or f.name.startswith(".") or f.name.endswith(".meta") or f.name.endswith(".sqlite"):
            continue

        name = f.name
        is_hx = name.endswith("_hx")
        base = name[:-3] if is_hx else name

        if base not in model_dict:
            model_dict[base] = {
                "base": base,
                "has_hx": False,
                "hx_file": None,
                "main_file": None,
                "size_mb": 0.0,
            }

        if is_hx:
            model_dict[base]["has_hx"] = True
            model_dict[base]["hx_file"] = f.name
        else:
            model_dict[base]["main_file"] = f.name
            try:
                model_dict[base]["size_mb"] = round(f.stat().st_size / (1024 * 1024), 2)
            except Exception:
                pass

    # Build skin lookup map from ShipCollection if available
    skin_map = {}
    if ship_collection:
        for skin in getattr(ship_collection, "skins", []):
            p = getattr(skin, "painting", "")
            if p:
                skin_map[p.lower()] = skin

    results = []
    for prefab, item in model_dict.items():
        ship_name = ""
        skin_name = ""
        skin_type = "Default"
        skin_id = None
        tags = []

        skin_obj = skin_map.get(prefab.lower())
        if not skin_obj:
            clean = re.sub(r"_\d+$", "", prefab.lower())
            skin_obj = skin_map.get(clean)

        if skin_obj:
            ship_obj = getattr(skin_obj, "ship", None)
            ship_name = getattr(ship_obj, "name", "") if ship_obj else ""
            skin_name = getattr(skin_obj, "name", "")
            skin_type = getattr(skin_obj, "type", "Default")
            skin_id = getattr(skin_obj, "skin_id", None)
            tags = getattr(skin_obj, "tag", [])
        else:
            ship_name = prefab.replace("_", " ").title()
            skin_name = "Live2D"

        results.append({
            "id": prefab,
            "type": "live2d",
            "ship_name": ship_name,
            "skin_name": skin_name,
            "skin_type": skin_type,
            "skin_id": skin_id,
            "tags": tags,
            "size_mb": item["size_mb"],
            "has_hx": item["has_hx"],
            "bundle_file": item["main_file"] or item["hx_file"] or prefab,
        })

    results.sort(key=lambda x: (x["ship_name"].lower(), x["skin_name"].lower()))
    return results


def extract_live2d_model(bundle_path: Path, cache_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Extract .moc3, textures, and .physics3.json from a Live2D bundle.

    Caches results in cache_dir / prefab and returns the model3 manifest dictionary.
    """
    prefab = bundle_path.name
    if prefab.endswith("_hx"):
        prefab_clean = prefab[:-3]
    else:
        prefab_clean = prefab

    out_dir = cache_dir / f"{prefab}_l2d"
    manifest_path = out_dir / "model3.json"

    # Fast path: already cached (must include Motions to avoid stale cache)
    if not force and manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if "Motions" in cached.get("FileReferences", {}) and cached.get("Meta", {}).get("extractorVersion") == 2:
                    return cached
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    textures_dir = out_dir / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    motions_dir = out_dir / "motions"
    motions_dir.mkdir(parents=True, exist_ok=True)

    env = UnityPy.load(str(bundle_path))

    moc_filename = None
    physics_filename = None
    display_filename = None
    pose_filename = None
    textures = []
    param_ids = set()

    for obj in env.objects:
        if obj.type.name == "MonoBehaviour":
            data = obj.read()
            script_ref = getattr(data, "m_Script", None)
            if script_ref:
                script_obj = script_ref.read()
                s_name = getattr(script_obj, "m_Name", getattr(script_obj, "name", "unknown"))
                if s_name == "CubismMoc":
                    raw = obj.get_raw_data()
                    idx = raw.find(b"MOC3")
                    if idx >= 4:
                        length = struct.unpack("<I", raw[idx-4:idx])[0]
                        moc_bytes = raw[idx:idx + length]
                        moc_file = out_dir / f"{prefab_clean}.moc3"
                        moc_file.write_bytes(moc_bytes)
                        moc_filename = moc_file.name
                        log.info(f"Extracted MOC3: {moc_filename} ({len(moc_bytes)} bytes)")
        elif obj.type.name == "TextAsset":
            t_name, content = extract_text_asset_content(obj)
            lower_name = t_name.lower()
            if "physics" in lower_name or lower_name.endswith(".physics3"):
                p_file = out_dir / f"{prefab_clean}.physics3.json"
                p_file.write_bytes(content)
                physics_filename = p_file.name
            elif "cdi" in lower_name or lower_name.endswith(".cdi3"):
                c_file = out_dir / f"{prefab_clean}.cdi3.json"
                c_file.write_bytes(content)
                display_filename = c_file.name
            elif "pose" in lower_name or lower_name.endswith(".pose3"):
                pose_file = out_dir / f"{prefab_clean}.pose3.json"
                pose_file.write_bytes(content)
                pose_filename = pose_file.name
        elif obj.type.name == "Texture2D":
            data = obj.read()
            tex_name = getattr(data, "m_Name", getattr(data, "name", "texture"))
            img = data.image
            filename = f"{tex_name}.png"
            img.save(textures_dir / filename)
            textures.append(f"textures/{filename}")
        elif obj.type.name == "GameObject":
            data = obj.read()
            go_name = getattr(data, "m_Name", getattr(data, "name", ""))
            if go_name.startswith("Param"):
                param_ids.add(go_name)

    # Sort texture paths naturally (texture_00.png, texture_01.png, ...)
    textures.sort()
    # Extract motions from AnimationClips
    crc_map = _build_hierarchy_crc_map(env)
    motion_refs = _extract_animation_clips(env, crc_map, motions_dir)

    file_refs: Dict[str, Any] = {
        "Moc": moc_filename or f"{prefab_clean}.moc3",
        "Textures": textures,
    }
    if physics_filename:
        file_refs["Physics"] = physics_filename
    if display_filename:
        file_refs["DisplayInfo"] = display_filename
    if pose_filename:
        file_refs["Pose"] = pose_filename
    if motion_refs:
        file_refs["Motions"] = motion_refs
    groups = []
    # Eye blink parameters
    eye_blink_params = [p for p in ["ParamEyeLOpen", "ParamEyeROpen"] if p in param_ids] or ["ParamEyeLOpen", "ParamEyeROpen"]
    groups.append({
        "Target": "Parameter",
        "Name": "EyeBlink",
        "Ids": eye_blink_params,
    })

    # Lip sync parameters
    lip_sync_params = [p for p in ["ParamMouthOpenY", "ParamMouthY"] if p in param_ids] or ["ParamMouthOpenY"]
    groups.append({
        "Target": "Parameter",
        "Name": "LipSync",
        "Ids": lip_sync_params,
    })

    manifest: Dict[str, Any] = {
        "Version": 3,
        "FileReferences": file_refs,
        "Groups": groups,
        "Meta": {
            "id": prefab,
            "type": "live2d",
            "paramCount": len(param_ids),
            "motions": list(motion_refs.keys()),
            "extractorVersion": 2,
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also save with prefab prefix for standard model loader compatibility
    pref_manifest_path = out_dir / f"{prefab_clean}.model3.json"
    with open(pref_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _build_hierarchy_crc_map(env) -> Dict[int, str]:
    """Build a mapping of CRC32(relative_hierarchy_path) -> relative_hierarchy_path."""
    go_dict = {}
    transforms = {}
    for obj in env.objects:
        if obj.type.name == "GameObject":
            data_go = obj.read()
            go_dict[obj.path_id] = (getattr(data_go, "m_Name", ""), data_go)
        elif obj.type.name == "Transform":
            transforms[obj.path_id] = obj.read()

    transform_to_go = {}
    for pid, (name, go) in go_dict.items():
        for comp in getattr(go, "m_Components", []):
            if comp.type.name == "Transform":
                transform_to_go[comp.path_id] = pid

    def get_rel_path(t_id):
        if t_id not in transforms:
            return ""
        t = transforms[t_id]
        go_id = transform_to_go.get(t_id)
        go_name = go_dict[go_id][0] if go_id in go_dict else ""
        father = getattr(t, "m_Father", None)
        if father and father.path_id in transforms and father.path_id != 0:
            parent_path = get_rel_path(father.path_id)
            return f"{parent_path}/{go_name}" if parent_path else go_name
        return ""

    crc_to_path = {}
    for tid in transforms:
        p = get_rel_path(tid)
        if p:
            crc = binascii.crc32(p.encode("utf-8"))
            crc_to_path[crc] = p
    return crc_to_path


def _read_streamed_clip(streamed_clip) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """Unpack StreamedClip uint32 array into frame list."""
    raw_bytes = struct.pack(f"<{len(streamed_clip.data)}I", *streamed_clip.data)
    pos = 0
    frames = []
    while pos < len(raw_bytes):
        time, num_keys = struct.unpack_from("<fi", raw_bytes, pos)
        pos += 8
        keys = []
        for _ in range(num_keys):
            idx, c0, c1, c2, c3 = struct.unpack_from("<iffff", raw_bytes, pos)
            pos += 20
            keys.append({"index": idx, "value": c3})
        frames.append((time, keys))
    return frames


def _extract_animation_clips(env, crc_to_path: Dict[int, str], motions_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """Convert Unity AnimationClips to Live2D Cubism .motion3.json files."""
    motion_refs = {}

    for obj in env.objects:
        if obj.type.name != "AnimationClip":
            continue
        clip = obj.read()
        clip_name = getattr(clip, "m_Name", "")
        if not clip_name:
            continue

        mc = getattr(clip, "m_MuscleClip", None)
        if not mc:
            continue
        m_clip = getattr(mc, "m_Clip", None)
        if not m_clip:
            continue
        clip_data = m_clip.data
        cbc = getattr(clip, "m_ClipBindingConstant", None)
        bindings = getattr(cbc, "genericBindings", None) if cbc else None
        if not bindings:
            continue

        streamed = getattr(clip_data, "m_StreamedClip", None)
        dense = getattr(clip_data, "m_DenseClip", None)
        constant = getattr(clip_data, "m_ConstantClip", None)

        streamed_count = streamed.curveCount if streamed else 0
        dense_count = getattr(dense, "m_CurveCount", 0) if dense else 0
        constant_values = constant.data if (constant and hasattr(constant, "data")) else []
        constant_count = len(constant_values)

        if streamed_count == 0 and constant_count == 0:
            continue

        duration = round(getattr(mc, "m_StopTime", 1.0), 3)
        if duration <= 0:
            duration = 1.0
        fps = round(getattr(clip, "m_SampleRate", 30.0), 1)
        if fps <= 0:
            fps = 30.0

        curves = []
        total_segments = 0
        total_points = 0
        seen_targets = set()

        # 1. Streamed (animated) curves
        if streamed_count > 0 and streamed:
            frames = _read_streamed_clip(streamed)
            if len(frames) >= 3:
                curve_bindings = []
                for i in range(streamed_count):
                    if i >= len(bindings):
                        break
                    b = bindings[i]
                    p_name = crc_to_path.get(b.path, f"Param_{b.path}")
                    target_id = p_name.split("/")[-1]
                    target_type = "PartOpacity" if b.attribute == 2353026298 else "Parameter"
                    curve_bindings.append((target_type, target_id))

                curve_keyframes = {i: [] for i in range(len(curve_bindings))}
                for t, k_list in frames[1:-1]:
                    t_sec = round(t, 4)
                    for k in k_list:
                        idx = k["index"]
                        if idx in curve_keyframes:
                            curve_keyframes[idx].append((t_sec, round(k["value"], 4)))

                for i, (target_type, target_id) in enumerate(curve_bindings):
                    kfs = curve_keyframes[i]
                    if not kfs:
                        continue
                    compact = [kfs[0]]
                    for k in kfs[1:]:
                        if k[0] > compact[-1][0]:
                            compact.append(k)
                    if len(compact) < 2:
                        segments = [0.0, compact[0][1], 0, duration, compact[0][1]]
                    else:
                        segments = [compact[0][0], compact[0][1]]
                        for k in compact[1:]:
                            segments.extend([0, k[0], k[1]])
                    num_segs = (len(segments) - 2) // 3
                    total_segments += num_segs
                    total_points += (1 + num_segs)
                    curves.append({
                        "Target": target_type,
                        "Id": target_id,
                        "Segments": segments
                    })
                    seen_targets.add((target_type, target_id))

        # 2. Constant curves (fixed pose parameters and part opacities)
        if constant_count > 0:
            constant_offset = streamed_count + dense_count
            for j, val in enumerate(constant_values):
                b_idx = constant_offset + j
                if b_idx >= len(bindings):
                    break
                b = bindings[b_idx]
                p_name = crc_to_path.get(b.path, f"Param_{b.path}")
                target_id = p_name.split("/")[-1]
                target_type = "PartOpacity" if b.attribute == 2353026298 else "Parameter"

                if (target_type, target_id) in seen_targets:
                    continue
                seen_targets.add((target_type, target_id))

                val_rounded = round(float(val), 4)
                segments = [0.0, val_rounded, 0, duration, val_rounded]
                total_segments += 1
                total_points += 2
                curves.append({
                    "Target": target_type,
                    "Id": target_id,
                    "Segments": segments
                })
        if not curves:
            continue

        motion_file = motions_dir / f"{clip_name}.motion3.json"
        motion_data = {
            "Version": 3,
            "Meta": {
                "Duration": duration,
                "Fps": fps,
                "Loop": "idle" in clip_name.lower(),
                "CurveCount": len(curves),
                "TotalSegmentCount": total_segments,
                "TotalPointCount": total_points,
                "UserDataCount": 0
            },
            "Curves": curves
        }
        with open(motion_file, "w", encoding="utf-8") as f:
            json.dump(motion_data, f, indent=2)

        motion_refs[clip_name] = [{"File": f"motions/{clip_name}.motion3.json"}]

    return motion_refs
