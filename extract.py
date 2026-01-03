#!/usr/bin/env python3
"""
Azur Lane Painting Extractor - Main Entry Point

Reconstructs character paintings from Unity assets with support for:
- Batch scaling with external tools
- Face overlay compositing  
- Variant extraction (_hx always, _n with --no-bg flag)
"""
from argparse import ArgumentParser
from pathlib import Path

from azurlane_extractor import (
    setup_logging,
    fetch_name_map,
    find_painting_variants,
    process_painting_group,
    finalize_and_save,
    reset_state,
    get_config,
    strip_variant_suffix,
    NAME_MAP_CACHE,
)


def main(argv=None):
    """Main entry point for extraction.
    
    Args:
        argv: Optional list of command-line arguments. If None, uses sys.argv.
              Example: ['-c', 'aotuo_3', '-d', 'D:\\Azurlane', '-o', 'output']
    """
    parser = ArgumentParser(description="Extract Azur Lane character paintings from Unity assets.")
    parser.add_argument("-c", "--char_name", type=str, required=True,
        help="Comma-separated list of character names (English).")
    parser.add_argument("-d", "--asset_directory", type=Path, default=Path(r"D:\Azurlane"),
        help="Directory containing all client assets")
    parser.add_argument("-o", "--output", type=Path, default=Path(r"R:\AzurlaneSkinExtract"),
        help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and save intermediate layers")
    parser.add_argument("--no-bg", action="store_true", dest="extract_no_bg",
        help="Also extract no-background variants (_n, _n_hx)")
    parser.add_argument("--scaler", type=str, default=None,
        help="External scaler command with {input} and {output} placeholders")
    parser.add_argument("--refresh-cache", action="store_true", dest="refresh_cache",
        help="Force refresh the name map cache from GitHub")
    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(args.debug)

    # Configure global state
    config = get_config()
    config.debug = args.debug
    config.save_layers = args.debug  # Always save layers when debug is enabled
    config.extract_no_bg = args.extract_no_bg
    config.external_scaler = args.scaler
    config.output_dir = args.output
    
    if args.refresh_cache:
        if NAME_MAP_CACHE.exists():
            NAME_MAP_CACHE.unlink()
        sqlite_cache = NAME_MAP_CACHE.with_suffix('.sqlite')
        if sqlite_cache.exists():
            sqlite_cache.unlink()
        import logging
        logging.getLogger(__name__).debug("NAME_MAP cache deleted")
    
    config.ship_collection = fetch_name_map()
    
    # Resolve char names to painting names
    char_names = [n.strip() for n in args.char_name.split(",") if n.strip()]
    painting_names = []
    for char_name in char_names:
        found = config.ship_collection.find_paintings(char_name) if config.ship_collection else [char_name]
        import logging
        logging.getLogger(__name__).debug(f"CHAR {char_name} -> {found}")
        painting_names.extend(found)
    
    if not painting_names:
        import logging
        logging.getLogger(__name__).warning("No paintings found for the given character names.")
        return

    # Group paintings by base name
    painting_groups: dict[str, list[str]] = {}
    for name in painting_names:
        base_name, _ = strip_variant_suffix(name)
        if base_name not in painting_groups:
            painting_groups[base_name] = [base_name]
        variants = find_painting_variants(base_name, args.asset_directory, include_no_bg=config.extract_no_bg)
        for v in variants:
            if v not in painting_groups[base_name]:
                painting_groups[base_name].append(v)
    
    total_paintings = sum(len(v) for v in painting_groups.values())
    import logging
    logging.getLogger(__name__).debug(f"VARIANTS: {len(painting_groups)} base paintings, {total_paintings} total")

    # Reset state
    reset_state()
    
    # Process each painting group
    for base_name, variants in painting_groups.items():
        process_painting_group(base_name, variants, args.asset_directory)
    
    # Finalize and save
    finalize_and_save()
    
    import logging
    logging.getLogger(__name__).info(f"Done. Processed {total_paintings} paintings.")


if __name__ == "__main__":
    main()
