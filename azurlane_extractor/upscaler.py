from resselt import load_from_file
import torch
import numpy as np
from PIL import Image
from reutils.tiling import ExactTileSize, MaxTileSize, NoTiling, process_tiles
import gc
from threading import Semaphore


class ImageUpscaler:
    """AI-based image upscaler that works entirely in memory."""
    
    def __init__(self, model_path: str, tile_type: str = "exact", tile: int = 384, 
                 dtype: str = "BF16", cpu_scale: bool = False, max_concurrent: int = 1):
        """
        Initialize the upscaler with model and settings.
        
        Args:
            model_path: Path to the upscale model file
            tile_type: Tiling mode - "exact", "max", or "no_tiling"
            tile: Tile size for "exact" mode
            dtype: Data type - "F16", "BF16", or "F32"
            cpu_scale: Use CPU instead of CUDA
            max_concurrent: Maximum concurrent upscale operations (default: 1)
        """
        self.model_path = model_path
        self.tile_type = tile_type
        self.tile_size = tile
        self.dtype_str = dtype
        self.cpu_scale = cpu_scale
        
        # Semaphore for controlling concurrent access
        self._semaphore = Semaphore(max_concurrent)
        
        # Load model
        self.model = load_from_file(model_path).eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu_scale else 'cpu')
        self.model.to(self.device)
        
        # Set dtype
        if dtype == "F16":
            self.dtype = torch.half
        elif dtype == "BF16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        # Tiler setup
        if tile_type == "exact":
            self.tiler = ExactTileSize(tile)
        elif tile_type == "max":
            self.tiler = MaxTileSize()
        else:
            self.tiler = NoTiling()
        
        self.scale = self.model.parameters_info.upscale
    
    def upscale(self, img: Image.Image) -> Image.Image:
        """
        Upscale a PIL Image in memory.
        
        Args:
            img: Input PIL Image
            
        Returns:
            Upscaled PIL Image with same mode (RGB/RGBA preserved)
        """
        # Handle transparency
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        if has_alpha:
            rgba = img.convert("RGBA")
            r, g, b, alpha = rgba.split()
            rgb = Image.merge("RGB", (r, g, b))
        else:
            rgb = img.convert("RGB")
        
        # Convert PIL to numpy (0-1 range float32)
        img_array = np.array(rgb).astype(np.float32) / 255.0
        
        # Upscale - semaphore only around GPU operation
        with self._semaphore:
            upscaled_array = process_tiles(
                img_array, tiler=self.tiler, model=self.model, device=self.device, 
                dtype=self.dtype, scale=self.scale
            )
        
        # Convert back to PIL
        upscaled_array = (upscaled_array * 255).clip(0, 255).astype(np.uint8)
        upscaled_rgb = Image.fromarray(upscaled_array)
        
        # Reattach alpha if needed
        if has_alpha:
            alpha_up = alpha.resize(upscaled_rgb.size, resample=Image.LANCZOS)
            r, g, b = upscaled_rgb.split()
            return Image.merge("RGBA", (r, g, b, alpha_up))
        
        return upscaled_rgb


if __name__ == "__main__":
    # Test parameters
    input_path = "R:/test.png"
    model_path = r"D:\UpscaleModel\janai\IllustrationJaNai_V3detail\2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.safetensors"
    output_path = "R:/test_upscaled.png"
    
    print(f"Initializing upscaler with model: {model_path}")
    upscaler = ImageUpscaler(
        model_path=model_path,
        tile_type="exact",
        tile=384,
        dtype="BF16",
        cpu_scale=False,
        max_concurrent=1
    )
    
    print(f"\nLoading image: {input_path}")
    img = Image.open(input_path)
    print(f"Original size: {img.size}, mode: {img.mode}")
    
    print(f"\nUpscaling...")
    upscaled_img = upscaler.upscale(img)
    
    print(f"Upscaled size: {upscaled_img.size}, mode: {upscaled_img.mode}")
    
    print(f"\nSaving to: {output_path}")
    upscaled_img.save(output_path)
    
    print("Done!")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
