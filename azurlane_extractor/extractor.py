"""Main extraction logic for Azur Lane paintings."""
import logging
import UnityPy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from PIL import Image
from UnityPy.enums import ClassIDType
from tqdm import tqdm
from .asset import AzurlaneAsset
from .layer import GameObjectLayer
from .config import get_config
from .name_map import Skin
from .upscaler import ImageUpscaler

log = logging.getLogger(__name__)


def init_upscaler(model_path: str, tile_type: str = "exact", tile: int = 384,
                  dtype: str = "BF16", cpu_scale: bool = False):
    """Initialize the upscaler in config."""
    config = get_config()
    config.upscaler = ImageUpscaler(
        model_path=model_path,
        tile_type=tile_type,
        tile=tile,
        dtype=dtype,
        cpu_scale=cpu_scale,
        max_concurrent=1
    )
    log.debug(f"Upscaler initialized with model: {model_path}")


def get_upscaler() -> Optional[ImageUpscaler]:
    """Get the upscaler instance from config."""
    return get_config().upscaler


def process_paintings_concurrent(skins: list[Skin], max_workers: int = 4):
    """Process multiple paintings concurrently with progress bar."""
    if not skins:
        return
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_painting_group, skin): skin for skin in skins}
        
        with tqdm(total=len(futures), desc="Processing paintings", unit="painting") as pbar:
            for future in as_completed(futures):
                skin = futures[future]
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    log.error(f"Error processing {skin.painting}: {e}")
                    pbar.update(1)


def get_face_images(skin: Skin, is_censored: bool = False) -> dict[str, Image.Image]:
    """Get face images for a painting. Caches by base name."""
    base_name = skin.painting + ("_hx" if is_censored else "")
    config = get_config()
    
    facefile = config.asset_dir / "paintingface" / base_name
    
    try:
        fasset = UnityPy.load(str(facefile))
        faces: dict[str, Image.Image] = {}
        for unity_object in fasset.objects:
            if unity_object.type == ClassIDType.Texture2D:
                tex = unity_object.read()
                faces[tex.m_Name] = tex.image
        return faces
    except Exception as e:
        log.debug(f"Failed to load face asset '{base_name}': {e}")
        return {}


def _compute_face_mse(base_rgb: np.ndarray, face_rgb: np.ndarray, 
                      face_alpha: np.ndarray, x: int, y: int) -> float:
    """Compute MSE between face and base region at given position."""
    fh, fw = face_rgb.shape[:2]
    region = base_rgb[y:y+fh, x:x+fw]
    diff = (region - face_rgb) * face_alpha[:, :, np.newaxis]
    return float(np.mean(diff ** 2))


def _find_edge_mask(face_alpha: np.ndarray) -> np.ndarray:
    """Find edge pixels: non-transparent pixels at the edge of the rectangular image."""
    fh, fw = face_alpha.shape
    
    edge_mask = np.zeros_like(face_alpha, dtype=bool)
    
    # Top and bottom rows
    edge_mask[0, :] = face_alpha[0, :] > 0.01
    edge_mask[fh-1, :] = face_alpha[fh-1, :] > 0.01
    
    # Left and right columns
    edge_mask[:, 0] = face_alpha[:, 0] > 0.01
    edge_mask[:, fw-1] = face_alpha[:, fw-1] > 0.01
    
    return edge_mask


def _compute_edge_coherence(base_rgb: np.ndarray, face_rgb: np.ndarray,
                            face_alpha: np.ndarray, edge_mask: np.ndarray,
                            x: int, y: int) -> float:
    """Compute edge coherence score - how well face edges blend with base.
    
    Compares the face edge pixels with what's underneath in the base image.
    For good placement, the colors should match smoothly. Lower = better.
    """
    fh, fw = face_rgb.shape[:2]
    region = base_rgb[y:y+fh, x:x+fw]
    
    if not np.any(edge_mask):
        return float('inf')
    
    # Compare face edge colors with base colors at those positions
    diff = np.abs(region[edge_mask] - face_rgb[edge_mask])
    
    return float(np.mean(diff ** 2))


def _search_region_coherence(base_rgb: np.ndarray, face_rgb: np.ndarray, face_alpha: np.ndarray,
                             edge_mask: np.ndarray,
                             x_range: range, y_range: range, step: int = 1) -> tuple[int, int, float]:
    """Search for best coherence (edge blending). Returns (best_x, best_y, best_score)."""
    fh, fw = face_rgb.shape[:2]
    bh, bw = base_rgb.shape[:2]
    
    best_x, best_y = x_range.start, y_range.start
    best_score = float('inf')
    
    for y in range(y_range.start, y_range.stop, step):
        if y < 0 or y + fh > bh:
            continue
        for x in range(x_range.start, x_range.stop, step):
            if x < 0 or x + fw > bw:
                continue
            score = _compute_edge_coherence(base_rgb, face_rgb, face_alpha, edge_mask, x, y)
            if score < best_score:
                best_score = score
                best_x, best_y = x, y
    
    return best_x, best_y, best_score


def find_coherent_face_offset(base_img: Image.Image, face_img: Image.Image,
                              calc_x: int, calc_y: int,
                              search_radius: int = 10) -> tuple[tuple[int, int], float]:
    """Find best face position based on edge coherence (for no-base-face cases).
    
    Args:
        base_img: The base painting image
        face_img: The face image to place
        calc_x, calc_y: Calculated position to search around
        search_radius: Pixels to search in each direction
        
    Returns:
        ((offset_x, offset_y), coherence_score)
    """
    base_rgb = np.array(base_img.convert('RGB')).astype(np.float32)
    face_rgb = np.array(face_img.convert('RGB')).astype(np.float32)
    face_alpha = np.array(face_img)[:, :, 3].astype(np.float32) / 255.0
    
    fh, fw = face_rgb.shape[:2]
    bh, bw = base_rgb.shape[:2]
    
    # Find edge mask once (pixels at the boundary of the face)
    edge_mask = _find_edge_mask(face_alpha)
    
    # Search around calculated position
    x_range = range(max(0, calc_x - search_radius), min(bw - fw + 1, calc_x + search_radius + 1))
    y_range = range(max(0, calc_y - search_radius), min(bh - fh + 1, calc_y + search_radius + 1))
    
    best_x, best_y, best_score = _search_region_coherence(base_rgb, face_rgb, face_alpha, edge_mask, x_range, y_range)
    
    if best_score == float('inf'):
        return ((0, 0), best_score)
    else:
        return ((best_x - calc_x, best_y - calc_y), best_score)


def _search_region(base_rgb: np.ndarray, face_rgb: np.ndarray, face_alpha: np.ndarray,
                   x_range: range, y_range: range, step: int = 1) -> tuple[int, int, float]:
    """Search a region for best face match. Returns (best_x, best_y, best_mse)."""
    fh, fw = face_rgb.shape[:2]
    bh, bw = base_rgb.shape[:2]
    
    best_x, best_y = x_range.start, y_range.start
    best_mse = float('inf')
    
    for y in range(y_range.start, y_range.stop, step):
        if y < 0 or y + fh > bh:
            continue
        for x in range(x_range.start, x_range.stop, step):
            if x < 0 or x + fw > bw:
                continue
            mse = _compute_face_mse(base_rgb, face_rgb, face_alpha, x, y)
            if mse < best_mse:
                best_mse = mse
                best_x, best_y = x, y
    
    return best_x, best_y, best_mse


def find_best_face_offset(base_img: Image.Image, face_img: Image.Image | list[Image.Image], 
                          calc_x: int, calc_y: int,
                          search_radius: int = 3,
                          mse_threshold: float = 2000.0) -> tuple[tuple[int, int], float, bool]:
    """Find the best face position using template matching.
    
    Args:
        base_img: The base painting image (without face overlay)
        face_img: The face image(s) to match - single image or list of up to 3 images
        calc_x, calc_y: Calculated position to search around
        search_radius: Pixels to search in each direction
        mse_threshold: If MSE exceeds this, expand search to whole image
        
    Returns:
        ((offset_x, offset_y), final_mse, expanded_search) - offset adjustment, MSE value, whether expanded search was used
    """
    
    # Convert single image to list
    face_images = [face_img] if not isinstance(face_img, list) else face_img[:3]  # Limit to 3 faces
    
    base_rgb = np.array(base_img.convert('RGB')).astype(np.float32)
    bh, bw = base_rgb.shape[:2]
    
    # Phase 1: Local search around calculated position using up to 3 faces
    best_x, best_y, best_mse = calc_x, calc_y, float('inf')
    
    # Use the first face for expanded search if needed
    first_face_rgb = np.array(face_images[0].convert('RGB')).astype(np.float32)
    first_face_alpha = np.array(face_images[0])[:, :, 3].astype(np.float32) / 255.0
    fh, fw = first_face_rgb.shape[:2]
    
    for face_image in face_images:
        face_rgb = np.array(face_image.convert('RGB')).astype(np.float32)
        face_alpha = np.array(face_image)[:, :, 3].astype(np.float32) / 255.0
        
        face_h, face_w = face_rgb.shape[:2]
        
        x_range = range(max(0, calc_x - search_radius), min(bw - face_w + 1, calc_x + search_radius + 1))
        y_range = range(max(0, calc_y - search_radius), min(bh - face_h + 1, calc_y + search_radius + 1))
        
        x, y, mse = _search_region(base_rgb, face_rgb, face_alpha, x_range, y_range)
        if mse < best_mse:
            best_x, best_y, best_mse = x, y, mse
            # Early out if match is good enough
            if best_mse <= mse_threshold:
                break
    
    expanded = False
    
    # Check if match is good enough
    if best_mse > mse_threshold:
        expanded = True
        
        # For small faces (<120px), skip the coarse step (16px) as it's too large
        skip_coarse = (fh < 120 or fw < 120)
        
        if skip_coarse:
            # Skip Phase 2, go directly to medium search over full image
            log.debug(f"Face size {fw}x{fh} < 120px, skipping 16px coarse step")
            coarse_x, coarse_y = 0, 0
            medium_radius = max(bw - fw, bh - fh)  # Search full image with medium step
        else:
            # Phase 2: Coarse search over entire image (step=16)
            coarse_step = 16
            x_range_full = range(0, bw - fw + 1)
            y_range_full = range(0, bh - fh + 1)
            
            coarse_x, coarse_y, coarse_mse = _search_region(
                base_rgb, first_face_rgb, first_face_alpha, x_range_full, y_range_full, step=coarse_step
            )
            medium_radius = coarse_step
        
        # Phase 3: Medium search around coarse result (step=4)
        x_range_med = range(max(0, coarse_x - medium_radius), min(bw - fw + 1, coarse_x + medium_radius + 1))
        y_range_med = range(max(0, coarse_y - medium_radius), min(bh - fh + 1, coarse_y + medium_radius + 1))
        
        med_x, med_y, med_mse = _search_region(
            base_rgb, first_face_rgb, first_face_alpha, x_range_med, y_range_med, step=4
        )
        
        # Phase 4: Fine search around medium result (step=1)
        fine_radius = 4
        x_range_fine = range(max(0, med_x - fine_radius), min(bw - fw + 1, med_x + fine_radius + 1))
        y_range_fine = range(max(0, med_y - fine_radius), min(bh - fh + 1, med_y + fine_radius + 1))
        
        fine_x, fine_y, fine_mse = _search_region(
            base_rgb, first_face_rgb, first_face_alpha, x_range_fine, y_range_fine, step=1
        )
        
        if fine_mse < best_mse:
            best_x, best_y, best_mse = fine_x, fine_y, fine_mse
    
    offset_x = best_x - calc_x
    offset_y = best_y - calc_y
    
    if abs(offset_x) <= search_radius and abs(offset_y) <= search_radius:
        expanded = False
    
    return ((offset_x, offset_y), best_mse, expanded)


def setup_layers(asset: AzurlaneAsset) -> tuple[GameObjectLayer, Optional[GameObjectLayer], tuple[int, int], tuple[int, int]]:
    """Setup and calculate layer hierarchy. Returns (parent_goi, facelayer, canvas_size, offset_adjustment)."""
    parent_object = asset.getObjectByPathID(asset.container.path_id)
    parent_goi = GameObjectLayer(asset, parent_object)
    parent_goi.retrieveChildren()
    parent_goi.calculateLocalOffset()
    parent_goi.calculateGlobalOffset()

    log.debug("Layer hierarchy:")
    parent_goi.printHierarchy()

    # Get bounding box to account for negative offsets
    min_x, min_y, max_x, max_y = parent_goi.getBounds()
    canvas_width = max(1, int(max_x - min_x))
    canvas_height = max(1, int(max_y - min_y))
    
    # Offset adjustment needed to shift all layers into positive space
    offset_adjustment = (int(-min_x), int(-min_y))
    
    if offset_adjustment != (0, 0):
        log.debug(f"Canvas has negative offsets: min=({min_x}, {min_y}), adjustment={offset_adjustment}")

    facelayer = parent_goi.findChildLayer('face')
    return parent_goi, facelayer, (canvas_width, canvas_height), offset_adjustment


def render_image(parent_goi: GameObjectLayer, facelayer: Optional[GameObjectLayer], 
                 canvas_size: tuple[int, int], offset_adjustment: tuple[int, int],
                 faceimg: Optional[Image.Image] = None,
                 face_alignment_offset: tuple[int, int] = (0, 0)) -> Image.Image:
    """Render the image with optional face overlay.
    
    Args:
        offset_adjustment: (x, y) to add to all layer offsets to shift into positive space.
            This compensates for layers with negative global offsets.
        face_alignment_offset: (x, y) template-matched adjustment for face position.
    """
    
    if faceimg and facelayer:
        facelayer.loadImageSimple(faceimg)

    layers_list = list(parent_goi.yieldLayers())
    
    # Detailed logging for composite debug
    log.debug(f"=== Compositing {len(layers_list)} layers, canvas={canvas_size}, adj={offset_adjustment} ===")

    canvas = Image.new('RGBA', canvas_size)
    for i, layer in enumerate(layers_list):
        # Check for manual position override
        from .config import get_config
        config = get_config()
        layer_name = layer.gameobject.m_Name
        
        if layer_name in config.layer_position_overrides:
            # Use manual override (absolute position, no adjustments)
            offx, offy = config.layer_position_overrides[layer_name]
            log.debug(f"  Using MANUAL position override for '{layer_name}': ({offx}, {offy})")
        else:
            # Apply offset adjustment to shift layers with negative coords into positive space
            offx = layer.global_offset[0] + offset_adjustment[0]
            offy = layer.global_offset[1] + offset_adjustment[1]
        
        # Apply face-specific offset adjustments: template-matched alignment
        if layer is facelayer:
            if face_alignment_offset != (0, 0):
                offx += face_alignment_offset[0]
                offy += face_alignment_offset[1]
        
        offx, offy = int(offx), int(offy)
        
        log.debug(f"  [{i}] {layer.gameobject.m_Name}: img={layer.image.size if layer.image else 'None'}, "
                 f"local={layer.local_offset}, global={layer.global_offset}, pos=({offx},{offy})")
        
        if layer.image and (offx < canvas_size[0] and offy < canvas_size[1] and 
            offx + layer.image.width > 0 and offy + layer.image.height > 0):
            canvas.alpha_composite(layer.image, (offx, offy))
    return canvas


def process_painting_group(skin: Skin):
    """Process a painting and all its variants, reusing the base asset."""
    base_name = skin.painting
    config = get_config()

    painting_subpath = Path("painting", base_name)
    painting_file = config.asset_dir / painting_subpath
    
    if not painting_file.exists():
        log.debug(f"SKIP {base_name}: asset file not found")
        return
    
    base_asset = AzurlaneAsset(skin)
    face_images = get_face_images(skin)
    
    _process_single_painting(skin, False, base_asset, face_images)
    if skin.have_censor:
        base_asset_censored = AzurlaneAsset(skin, True)
        censor_face_path = config.asset_dir / "paintingface" / (base_name + "_hx")
        if censor_face_path.exists():
            face_images_censor = get_face_images(skin, True)
        else:
            face_images_censor = face_images
        _process_single_painting(skin, True, base_asset_censored, face_images_censor)


def _process_single_painting(skin: Skin, is_censored: bool, asset: AzurlaneAsset, 
                             face_images: dict[str, Image.Image]):
    """Process a single painting variant and save immediately."""
    painting_name = skin.painting + ("_hx" if is_censored else "")
    config = get_config()
    
    try:
        display_name = skin.display_name()
        if is_censored:
            display_name = f"{display_name} (Censored)"
        
        if display_name != painting_name:
            log.debug(f"NAME_MAP {painting_name} -> {display_name}")

        outdir = config.output_dir
        outdir.mkdir(parents=True, exist_ok=True)

        # print(f" >>>>>>>>>> {display_name}")
        parent_goi, facelayer, canvas_size, offset_adjustment = setup_layers(asset)
        
        if face_images:
            frames: list[Image.Image] = []
            has_face_zero = '0' in face_images
            # if has_face_zero:
            #     log.warning(f"{skin.painting}: Face '0' detected")
            # return
            
            # Calculate face alignment offset using template matching
            face_alignment_offset = (0, 0)
            if facelayer:
                # Always render base image without face for template matching
                base_img = render_image(parent_goi, facelayer, canvas_size, 
                                       offset_adjustment, None)
                
                # Prepare up to 3 face images for template matching
                match_face_imgs = []
                if '0' in face_images:
                    match_face_imgs.append(face_images['0'])
                # Add up to 2 more faces from the remaining faces
                for face_key, face_img in face_images.items():
                    if face_key != '0' and len(match_face_imgs) < 3:
                        match_face_imgs.append(face_img)
                
                if not match_face_imgs:
                    match_face_imgs.append(next(iter(face_images.values())))
                
                # Calculate base position for face using first face for sizing
                first_face = match_face_imgs[0]
                facelayer.loadImageSimple(first_face)
                # Reload all face images with correct size - use facelayer.image for first, keep originals for rest
                if facelayer.image is not None:
                    match_face_imgs = [facelayer.image] + [img for img in match_face_imgs[1:] if img is not None]
                else:
                    match_face_imgs = [img for img in match_face_imgs if img is not None]
                
                if not match_face_imgs:
                    log.warning(f"{display_name}: No valid face images found for matching")
                    match_face_imgs = [first_face]  # Fallback to original first face
                
                calc_x = int(facelayer.global_offset[0] + offset_adjustment[0])
                calc_y = int(facelayer.global_offset[1] + offset_adjustment[1])
                
                if has_face_zero:
                    # mse_threshold=float('inf') means don't expand for face '0' case
                    face_alignment_offset, mse, expanded = find_best_face_offset(
                        base_img, match_face_imgs, calc_x, calc_y)
                    
                    # Check if base has a face by matching face '0' against base
                    # Bad match (high MSE) → base has no face → use coherence search
                    # Complete match (very low MSE) → face '0' == base → skip base  
                    # Good but non-complete match → base has different face → include base
                    if mse > 2000:  # Bad match - no base face, use coherence search
                        coherence_offset, coherence_score = find_coherent_face_offset(
                            base_img, first_face, calc_x, calc_y)
                        if coherence_score != float('inf'):
                            face_alignment_offset = coherence_offset
                            log.warning(f"{display_name}: No base face detected, using coherence offset={coherence_offset} (score={coherence_score:.0f})")
                        else:
                            face_alignment_offset = (0, 0)
                            log.warning(f"{display_name}: No base face detected (MSE={mse:.0f}), but coherence search failed, using no offset")
                    elif mse < 10:  # Complete match - face '0' is same as base
                        log.debug(f"{display_name}: Face '0' matches base (MSE={mse:.0f}), skipping base frame")
                    else:  # Good match but not identical - base has a different face
                        log.debug(f"{display_name}: Base has different face than '0' (MSE={mse:.0f}), including base frame")
                        frames.append(base_img.convert('RGBA'))
                else:
                    # No face '0' - base definitely has a face baked in
                    frames.append(base_img.convert('RGBA'))
                    face_alignment_offset, mse, expanded = find_best_face_offset(base_img, match_face_imgs, calc_x, calc_y)
                    if expanded:
                        log.warning(f"{display_name}: Face position misaligned, used expanded search (MSE={mse:.0f}), offset={face_alignment_offset}")
                    elif face_alignment_offset != (0, 0):
                        log.debug(f"{display_name}: Face alignment adjusted by {face_alignment_offset} (MSE={mse:.0f})")
            else:
                # No face layer - render base without template matching
                base_img = render_image(parent_goi, facelayer, canvas_size, 
                                       offset_adjustment, None)
                frames.append(base_img.convert('RGBA'))

            for face_type, faceimg in face_images.items():
                frame = render_image(parent_goi, facelayer, canvas_size, 
                                    offset_adjustment, faceimg, 
                                    face_alignment_offset)
                frames.append(frame.convert('RGBA'))

            # Verify all frames share the same size as the canvas
            mismatch = False
            for i, f in enumerate(frames):
                if f.size != canvas_size:
                    log.error(f"Frame {i} size {f.size} does not match canvas size {canvas_size} for {display_name}")
                    mismatch = True

            outpath = outdir / f"{display_name}.webp"
            if mismatch:
                log.error(f"Size mismatch detected - skipping animated WebP for {display_name}.")
            else:
                # Save as animated WebP: first frame + appended frames
                frames[0].save(
                    outpath,
                    format='WEBP',
                    save_all=True,
                    append_images=frames[1:],
                    duration=500,
                    loop=0,
                    lossless=True
                )
                log.debug(f"Saved animated WebP: {outpath}")
        else:
            result = render_image(parent_goi, facelayer, canvas_size, offset_adjustment, None)
            outpath = outdir / f"{display_name}.png"
            result.save(outpath)
            log.debug(f"Saved: {outpath}")

        log.debug(f"DONE {painting_name}")

    except Exception as e:
        log.error(f"{painting_name}: {e}")
