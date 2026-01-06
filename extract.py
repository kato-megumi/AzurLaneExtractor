#!/usr/bin/env python3
"""
Azur Lane Painting Extractor - Main Entry Point

Reconstructs character paintings from Unity assets with support for:
- AI upscaling with external models
- Face overlay compositing  
"""
import os
from argparse import ArgumentParser
from pathlib import Path

from azurlane_extractor import (
    setup_logging,
    fetch_name_map,
    process_paintings_concurrent,
    get_config,
    init_upscaler,
)


def main(argv=None):
    """Main entry point for extraction.
    
    Args:
        argv: Optional list of command-line arguments. If None, uses sys.argv.
              Example: ['-c', 'aotuo_3', '-d', 'D:\\Azurlane', '-o', 'output']
    """
    parser = ArgumentParser(description="Extract Azur Lane character paintings from Unity assets.")
    parser.add_argument("-c", "--char_name", type=str, default="",
        help="Comma-separated list of character names (English).")
    parser.add_argument("-p", "--painting_name", type=str, default="",
        help="Comma-separated list of specific painting names to extract.")
    parser.add_argument("-d", "--asset_directory", type=Path, default=Path(r"D:\Azurlane"),
        help="Directory containing all client assets")
    parser.add_argument("-o", "--output", type=Path, default=Path(r"R:\AzurlaneSkinExtract"),
        help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and save textures")
    parser.add_argument("--save-textures", action="store_true", help="Save temporary texture")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4,
        help="Number of concurrent jobs (default: CPU count)")
    
    # Upscaler options
    parser.add_argument("-m", "--upscaler-model", type=str, default=None,
        help="Path to AI upscaler model file")
    parser.add_argument("--upscaler-tile", type=int, default=384,
        help="Tile size for upscaler (default: 384)")
    parser.add_argument("--upscaler-tile-type", type=str, default="exact",
        choices=["exact", "max", "no_tiling"],
        help="Tiling mode for upscaler (default: exact)")
    parser.add_argument("--upscaler-dtype", type=str, default="BF16",
        choices=["F16", "BF16", "F32"],
        help="Data type for upscaler (default: BF16)")
    parser.add_argument("--upscaler-cpu", action="store_true",
        help="Use CPU instead of CUDA for upscaling")

    # Ensure at least one of -c/--char_name or -p/--painting_name is provided.
    _orig_parse_args = parser.parse_args
    def _parse_args_and_validate(argv=None):
        args = _orig_parse_args(argv)
        if not args.char_name and not args.painting_name:
            parser.error("Either -c/--char_name or -p/--painting_name is required.")
        return args
    parser.parse_args = _parse_args_and_validate
    
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(args.debug)

    # Configure global state
    config = get_config()
    config.debug = args.debug
    config.save_textures = args.save_textures or args.debug
    config.asset_dir = args.asset_directory
    config.output_dir = args.output
    config.ship_collection = fetch_name_map()
    
    # Initialize upscaler if model path provided
    if args.upscaler_model:
        init_upscaler(
            model_path=args.upscaler_model,
            tile_type=args.upscaler_tile_type,
            tile=args.upscaler_tile,
            dtype=args.upscaler_dtype,
            cpu_scale=args.upscaler_cpu
        )
    
    # Resolve inputs to Skin objects
    skins = []
    
    if args.char_name:
        char_names = [n.strip() for n in args.char_name.split(",") if n.strip()]
        for char_name in char_names:
            if config.ship_collection:
                for ship in config.ship_collection.ships_by_name(char_name):
                    skins.extend(ship.skins)
    
    if args.painting_name:
        p_names = [n.strip() for n in args.painting_name.split(",") if n.strip()]
        for p_name in p_names:
            if config.ship_collection:
                skin = config.ship_collection.skins_by_painting(p_name)
                if skin:
                    skins.append(skin)

    if not skins:
        import logging
        logging.getLogger(__name__).warning("No paintings found for the given inputs.")
        return

    # Deduplicate skins by base painting name
    unique_skins = {s.painting: s for s in skins}
    sorted_skins = sorted(unique_skins.values(), key=lambda s: s.painting)
    
    import logging
    logging.getLogger(__name__).debug(f"Found {len(sorted_skins)} base paintings to process.")
    
    # Process paintings concurrently
    process_paintings_concurrent(sorted_skins, max_workers=args.jobs)


if __name__ == "__main__":
    main()
