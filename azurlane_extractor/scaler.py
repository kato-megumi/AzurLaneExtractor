"""Batch image scaling functionality."""
import logging
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import get_config

log = logging.getLogger(__name__)


@dataclass
class BatchScalerState:
    """State for batch image scaling."""
    input_dir: Optional[Path] = None
    pending: dict[str, tuple[Image.Image, tuple[int, int], str]] = field(default_factory=dict)
    counter: int = 0


# Global state
_batch_state = BatchScalerState()


def reset_batch_scaler():
    """Reset batch scaler state and clean up temp directories."""
    global _batch_state
    if _batch_state.input_dir and _batch_state.input_dir.exists():
        shutil.rmtree(_batch_state.input_dir, ignore_errors=True)
    _batch_state = BatchScalerState()


def add_to_batch_scaler(image: Image.Image, target_size: tuple[int, int], layer_name: str = "") -> str:
    """Add image to batch for scaling. Returns image ID."""
    if _batch_state.input_dir is None:
        _batch_state.input_dir = Path(tempfile.mkdtemp())
    
    img_id = f"{_batch_state.counter:04d}"
    _batch_state.counter += 1
    
    input_path = _batch_state.input_dir / f"{img_id}.png"
    image.save(input_path)
    _batch_state.pending[img_id] = (image, target_size, layer_name)
    
    log.debug(f"Batch queued '{layer_name}': {image.size} -> {target_size}")
    
    return img_id


def flush_batch_scaler() -> dict[str, Image.Image]:
    """Run external scaler on all pending images and return scaled results."""
    config = get_config()
    
    if not config.external_scaler or not _batch_state.pending or not _batch_state.input_dir:
        return {}
    
    output_dir = Path(tempfile.mkdtemp())
    cmd = config.external_scaler.format(input=str(_batch_state.input_dir), output=str(output_dir))
    
    log.info(f"Running batch scaler on {len(_batch_state.pending)} images...")
    log.debug(f"Batch scaler command: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        log.warning(f"Batch scaler exited with code {result.returncode}")
    
    # Load scaled images
    scaled = {}
    for img_id, (original, target_size, layer_name) in _batch_state.pending.items():
        output_path = output_dir / f"{img_id}.png"
        if output_path.exists():
            img = Image.open(output_path).convert('RGBA')
            if img.size != target_size:
                log.debug(f"Batch adjusting '{layer_name}': {img.size} -> {target_size}")
                img = img.resize(target_size, Image.LANCZOS)
            scaled[img_id] = img
        else:
            log.debug(f"Batch fallback for '{layer_name}': scaler output not found")
            scaled[img_id] = original.resize(target_size, Image.LANCZOS)
    
    # Cleanup
    shutil.rmtree(_batch_state.input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    _batch_state.input_dir = None
    _batch_state.pending = {}
    
    log.info("Batch scaling complete")
    return scaled


def queue_face_images_for_scaling(face_images: dict[str, Image.Image], target_size: tuple[int, int]) -> dict[str, str]:
    """Queue face images for batch scaling. Returns dict mapping face_type to img_id."""
    config = get_config()
    
    if not config.external_scaler or not face_images:
        return {}
    
    face_scale_ids = {}
    for face_type, faceimg in face_images.items():
        if faceimg.size != target_size:
            img_id = add_to_batch_scaler(faceimg, target_size, f"face_{face_type}")
            face_scale_ids[face_type] = img_id
    
    return face_scale_ids


def get_batch_state() -> BatchScalerState:
    """Get current batch scaler state."""
    return _batch_state
