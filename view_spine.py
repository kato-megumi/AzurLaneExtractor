"""Entry point CLI to launch the Azur Lane Spine 2D Viewer."""
import argparse
import logging
import sys
import tempfile
import webbrowser
from pathlib import Path

from spine_viewer.server import SpineViewerServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spine_viewer")


def main():
    parser = argparse.ArgumentParser(description="Azur Lane Spine 2D Painting Viewer")
    parser.add_argument(
        "-d", "--spine-dir",
        type=Path,
        default=Path(r"D:\Azurlane\spinepainting"),
        help="Path to directory containing spinepainting asset bundles (default: D:\\Azurlane\\spinepainting)",
    )
    parser.add_argument(
        "-l", "--live2d-dir",
        type=Path,
        default=Path(r"D:\Azurlane\live2d"),
        help="Path to directory containing live2d asset bundles (default: D:\\Azurlane\\live2d)",
    )
    parser.add_argument(
        "-c", "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "spine_cache",
        help="Path to cache directory for extracted assets (default: %%TEMP%%\\spine_cache)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Automatically open web browser on startup",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip preloading ship name metadata on startup",
    )

    args = parser.parse_args()

    if not args.spine_dir.is_dir():
        log.error(f"Spine directory not found: {args.spine_dir}")
        sys.exit(1)

    server_manager = SpineViewerServer(
        spine_dir=args.spine_dir,
        live2d_dir=args.live2d_dir if args.live2d_dir.is_dir() else None,
        cache_dir=args.cache_dir,
        host=args.host,
        port=args.port,
    )

    server = server_manager.start(preload_names=not args.no_preload)
    url = f"http://{server_manager.host}:{server_manager.port}/"

    print("\n" + "=" * 60)
    print(f"  ⚓ Azur Lane Spine 2D Viewer Ready!")
    print(f"  URL: {url}")
    print(f"  Spine Directory:  {args.spine_dir}")
    if args.live2d_dir.is_dir():
        print(f"  Live2D Directory: {args.live2d_dir}")
    print("=" * 60 + "\n")

    if args.browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Spine Viewer server...")
        server.shutdown()
        server.server_close()
        print("Server closed.")


if __name__ == "__main__":
    main()
