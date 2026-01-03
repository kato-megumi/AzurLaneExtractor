"""Main extraction logic for Azur Lane paintings."""
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from UnityPy.enums import ClassIDType

from .asset import AzurlaneAsset
from .layer import GameObjectLayer
from .config import get_config
from .constants import strip_variant_suffix
from .scaler import (
    reset_batch_scaler as _reset_batch_scaler,
    flush_batch_scaler,
    queue_face_images_for_scaling,
    get_batch_state,
)

log = logging.getLogger(__name__)


@dataclass 
class PendingRender:
    """A painting queued for rendering after batch scaling."""
    parent_goi: GameObjectLayer
    facelayer: Optional[GameObjectLayer]
    canvas_size: tuple[int, int]
    offset_adjustment: tuple[int, int]
    face_images: dict[str, Image.Image]
    outdir: Path
    display_name: str
    face_scale_ids: dict[str, str]


# Global state
_pending_renders: list[PendingRender] = []
_face_cache: dict[str, dict[str, Image.Image]] = {}


def reset_state():
    """Reset all global state."""
    global _pending_renders, _face_cache
    _reset_batch_scaler()
    _pending_renders = []
    _face_cache = {}


def get_face_images(painting_name: str, asset_dir: Path) -> dict[str, Image.Image]:
    """Get face images for a painting. Caches by base name."""
    base_name, _ = strip_variant_suffix(painting_name)
    
    if base_name in _face_cache:
        return _face_cache[base_name]
    
    facebpath = Path("paintingface", base_name)
    facefile = asset_dir / facebpath
    
    if not facefile.exists():
        _face_cache[base_name] = {}
        return {}
    
    try:
        fasset = AzurlaneAsset(asset_dir, facebpath)
        faces = {}
        for unity_object in fasset.bundle.objects:
            if unity_object.type == ClassIDType.Texture2D:
                tex = unity_object.read()
                faces[tex.m_Name] = tex.image
        _face_cache[base_name] = faces
        return faces
    except Exception as e:
        log.debug(f"Failed to load face asset '{base_name}': {e}")
        _face_cache[base_name] = {}
        return {}


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
    sizex = max(1, int(max_x - min_x))
    sizey = max(1, int(max_y - min_y))
    
    # Offset adjustment needed to shift all layers into positive space
    offset_adjustment = (int(-min_x), int(-min_y))
    
    if offset_adjustment != (0, 0):
        log.debug(f"Canvas has negative offsets: min=({min_x}, {min_y}), adjustment={offset_adjustment}")

    facelayer = parent_goi.findChildLayer('face')
    return parent_goi, facelayer, (sizex, sizey), offset_adjustment


def render_image(parent_goi: GameObjectLayer, facelayer: Optional[GameObjectLayer], 
                 canvas_size: tuple[int, int], offset_adjustment: tuple[int, int],
                 faceimg: Image.Image = None, layer_output_dir: Optional[Path] = None) -> Image.Image:
    """Render the image with optional face overlay.
    
    Args:
        offset_adjustment: (x, y) to add to all layer offsets to shift into positive space.
            This compensates for layers with negative global offsets.
    """
    config = get_config()
    
    if faceimg and facelayer:
        facelayer.loadImageSimple(faceimg)

    layers_list = list(parent_goi.yieldLayers())
    log.debug(f"Compositing {len(layers_list)} layers: {[layer.gameobject.m_Name for layer in layers_list]}")

    canvas = Image.new('RGBA', canvas_size)
    for i, layer in enumerate(layers_list):
        # Apply offset adjustment to shift layers with negative coords into positive space
        offx = int(layer.global_offset[0] + offset_adjustment[0])
        offy = int(layer.global_offset[1] + offset_adjustment[1])
        log.debug(f"  Layer {i}: {layer.gameobject.m_Name} at ({offx}, {offy}), size={layer.image.size}")
        if (offx < canvas_size[0] and offy < canvas_size[1] and 
            offx + layer.image.width > 0 and offy + layer.image.height > 0):
            canvas.alpha_composite(layer.image, (offx, offy))
            if layer_output_dir and config.save_layers:
                layer_path = layer_output_dir / f"_layer_{i:02d}_{layer.gameobject.m_Name}.png"
                layer.image.save(layer_path)
                log.debug(f"  Saved layer: {layer_path.name}")
    return canvas


def process_painting_group(base_name: str, variants: list[str], asset_dir: Path):
    """Process a painting and all its variants, reusing the base asset."""
    painting_subpath = Path("painting", base_name)
    painting_file = asset_dir / painting_subpath
    
    if not painting_file.exists():
        log.debug(f"SKIP {base_name}: asset file not found")
        return
    
    base_asset = AzurlaneAsset(asset_dir, painting_subpath)
    base_asset.loadDependencies()
    
    face_images = get_face_images(base_name, asset_dir)
    
    _process_single_painting(base_name, base_asset, face_images, asset_dir)
    
    for variant_name in variants:
        if variant_name == base_name:
            continue
        
        variant_subpath = Path("painting", variant_name)
        variant_file = asset_dir / variant_subpath
        if not variant_file.exists():
            log.debug(f"SKIP {variant_name}: variant file not found")
            continue
        
        variant_asset = AzurlaneAsset(asset_dir, variant_subpath)
        variant_asset.bundle.load([str(asset_dir / painting_subpath)])
        variant_asset.loadDependencies()
        
        _process_single_painting(variant_name, variant_asset, face_images, asset_dir)


def _process_single_painting(painting_name: str, asset: AzurlaneAsset, 
                             face_images: dict[str, Image.Image], asset_dir: Path):
    """Process a single painting variant."""
    config = get_config()
    
    try:
        if config.ship_collection:
            display_name = config.ship_collection.get_display_name(painting_name)
            char_name, skin_name = config.ship_collection.get_char_and_skin_name(painting_name)
        else:
            display_name = painting_name
            char_name, skin_name = painting_name, painting_name
        
        if display_name != painting_name:
            log.debug(f"NAME_MAP {painting_name} -> {display_name}")

        outdir = config.output_dir / char_name / skin_name
        outdir.mkdir(parents=True, exist_ok=True)

        parent_goi, facelayer, canvas_size, offset_adjustment = setup_layers(asset)

        face_scale_ids = {}
        if face_images and config.external_scaler and facelayer:
            target_size = (int(facelayer.rect_transform.size_delta[0]),
                           int(facelayer.rect_transform.size_delta[1]))
            face_scale_ids = queue_face_images_for_scaling(face_images, target_size)

        _pending_renders.append(PendingRender(
            parent_goi=parent_goi,
            facelayer=facelayer,
            canvas_size=canvas_size,
            offset_adjustment=offset_adjustment,
            face_images=face_images,
            outdir=outdir,
            display_name=display_name,
            face_scale_ids=face_scale_ids
        ))

        log.debug(f"QUEUED {painting_name}")

    except Exception as e:
        log.error(f"{painting_name}: {e}")


def finalize_and_save():
    """Flush batch scaler, apply scaled images, and save all renders."""
    global _pending_renders
    config = get_config()
    
    if not _pending_renders:
        return
    
    scaled_images = {}
    batch_state = get_batch_state()
    if config.external_scaler and batch_state.pending:
        scaled_images = flush_batch_scaler()
    
    for render in _pending_renders:
        try:
            if scaled_images:
                render.parent_goi.applyScaledImages(scaled_images)
                
                # Recalculate canvas size after applying scaled images
                # because external scaler may have changed image dimensions
                min_x, min_y, max_x, max_y = render.parent_goi.getBounds()
                canvas_size = (max(1, int(max_x - min_x)), max(1, int(max_y - min_y)))
                offset_adjustment = (int(-min_x), int(-min_y))
                
                if offset_adjustment != (0, 0):
                    log.debug(f"Canvas recalculated after scaling: size={canvas_size}, adjustment={offset_adjustment}")
                
                render.canvas_size = canvas_size
                render.offset_adjustment = offset_adjustment
            
            scaled_faces = {}
            if render.face_images:
                target_size = None
                if render.facelayer:
                    target_size = (int(render.facelayer.rect_transform.size_delta[0]),
                                   int(render.facelayer.rect_transform.size_delta[1]))
                for face_type, faceimg in render.face_images.items():
                    if face_type in render.face_scale_ids and render.face_scale_ids[face_type] in scaled_images:
                        scaled_faces[face_type] = scaled_images[render.face_scale_ids[face_type]]
                    elif target_size and faceimg.size != target_size:
                        scaled_faces[face_type] = faceimg.resize(target_size, Image.LANCZOS)
                    else:
                        scaled_faces[face_type] = faceimg
            
            layer_dir = render.outdir if config.save_layers else None
            
            if render.face_images:
                rendered = []
                
                result = render_image(render.parent_goi, render.facelayer, render.canvas_size, render.offset_adjustment, None, layer_dir)
                rendered.append((result, render.outdir / f"{render.display_name}_0.png"))
                
                for face_type, faceimg in scaled_faces.items():
                    result = render_image(render.parent_goi, render.facelayer, render.canvas_size, render.offset_adjustment, faceimg, None)
                    rendered.append((result, render.outdir / f"{render.display_name}_{face_type}.png"))

                def save_image(args):
                    img, path = args
                    img.save(path)
                    return path

                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    results = list(executor.map(save_image, rendered))
                    for outpath in results:
                        log.debug(f"Saved: {outpath}")
            else:
                result = render_image(render.parent_goi, render.facelayer, render.canvas_size, render.offset_adjustment, None, layer_dir)
                outpath = render.outdir / f"{render.display_name}.png"
                result.save(outpath)
                log.debug(f"Saved: {outpath}")
                    
        except Exception as e:
            log.error(f"Saving {render.display_name}: {e}")
    
    _pending_renders = []
