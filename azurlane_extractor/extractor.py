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
    log.info(f"Upscaler initialized with model: {model_path}")


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


def find_best_face_offset(base_img: Image.Image, face_img: Image.Image, 
                          calc_x: int, calc_y: int, search_radius: int = 3) -> tuple[int, int]:
    """Find the best face position using template matching.
    
    Args:
        base_img: The base painting image (without face overlay)
        face_img: The face image to match
        calc_x, calc_y: Calculated position to search around
        search_radius: Pixels to search in each direction
        
    Returns:
        (offset_x, offset_y) adjustment to apply to calculated position
    """
    
    base_rgb = np.array(base_img.convert('RGB')).astype(np.float32)
    face_rgb = np.array(face_img.convert('RGB')).astype(np.float32)
    face_alpha = np.array(face_img)[:, :, 3].astype(np.float32) / 255.0
    
    fh, fw = face_rgb.shape[:2]
    bh, bw = base_rgb.shape[:2]
    
    best_x, best_y = calc_x, calc_y
    best_mse = float('inf')
    
    for y in range(max(0, calc_y - search_radius), min(bh - fh, calc_y + search_radius + 1)):
        for x in range(max(0, calc_x - search_radius), min(bw - fw, calc_x + search_radius + 1)):
            region = base_rgb[y:y+fh, x:x+fw]
            diff = (region - face_rgb) * face_alpha[:, :, np.newaxis]
            mse = np.mean(diff ** 2)
            if mse < best_mse:
                best_mse = mse
                best_x, best_y = x, y
    
    offset_x = best_x - calc_x
    offset_y = best_y - calc_y
    
    if offset_x != 0 or offset_y != 0:
        log.debug(f"Face alignment adjusted by ({offset_x}, {offset_y}) via template matching (MSE: {best_mse:.0f})")
    
    return (offset_x, offset_y)


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
                 faceimg: Image.Image = None,
                 face_alignment_offset: tuple[int, int] = (0, 0)) -> Image.Image:
    """Render the image with optional face overlay.
    
    Args:
        offset_adjustment: (x, y) to add to all layer offsets to shift into positive space.
            This compensates for layers with negative global offsets.
        face_alignment_offset: (x, y) template-matched adjustment for face position.
    """
    
    face_offset_adj = (0, 0)
    if faceimg and facelayer:
        face_offset_adj = facelayer.loadImageSimple(faceimg)

    layers_list = list(parent_goi.yieldLayers())
    
    # Detailed logging for composite debug
    log.debug(f"=== Compositing {len(layers_list)} layers, canvas={canvas_size}, adj={offset_adjustment} ===")

    canvas = Image.new('RGBA', canvas_size)
    for i, layer in enumerate(layers_list):
        # Apply offset adjustment to shift layers with negative coords into positive space
        offx = layer.global_offset[0] + offset_adjustment[0]
        offy = layer.global_offset[1] + offset_adjustment[1]
        
        # Apply face-specific offset adjustments: size mismatch + template-matched alignment
        if layer is facelayer:
            if face_offset_adj != (0, 0):
                offx += face_offset_adj[0]
                offy += face_offset_adj[1]
            if face_alignment_offset != (0, 0):
                offx += face_alignment_offset[0]
                offy += face_alignment_offset[1]
        
        offx, offy = int(offx), int(offy)
        
        log.debug(f"  [{i}] {layer.gameobject.m_Name}: img={layer.image.size}, "
                 f"local={layer.local_offset}, global={layer.global_offset}, pos=({offx},{offy})")
        
        if (offx < canvas_size[0] and offy < canvas_size[1] and 
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
            # Create frames: skip base if face '0' exists (it replaces the base)
            frames: list[Image.Image] = []
            has_face_zero = '0' in face_images
            
            # Calculate face alignment offset using template matching on first face
            face_alignment_offset = (0, 0)
            if has_face_zero:
                # No base image to match against - warn user and use default position
                log.warning(f"{display_name}: Has no base face, face position may be misaligned.")
            elif facelayer:
                # Render base image first to use for template matching
                base_img = render_image(parent_goi, facelayer, canvas_size, 
                                       offset_adjustment, None)
                frames.append(base_img.convert('RGBA'))
                
                # Use first face for template matching
                first_face_id, first_face_img = next(iter(face_images.items()))
                
                # Calculate base position for face
                size_adj = facelayer.loadImageSimple(first_face_img)
                calc_x = int(facelayer.global_offset[0] + offset_adjustment[0] + size_adj[0])
                calc_y = int(facelayer.global_offset[1] + offset_adjustment[1] + size_adj[1])
                
                # Find best match position
                face_alignment_offset = find_best_face_offset(base_img, first_face_img, calc_x, calc_y)
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
