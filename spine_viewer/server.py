"""HTTP server and API endpoints for the Spine 2D Viewer."""
import json
import logging
import mimetypes
import os
import tempfile
import urllib.parse
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional

from azurlane_extractor.name_map import fetch_name_map
from .extractor import list_spine_models, extract_spine_model
from .live2d_extractor import list_live2d_models, extract_live2d_model

log = logging.getLogger("spine_viewer.server")

# Ensure proper MIME type mappings
mimetypes.add_type("application/octet-stream", ".skel")
mimetypes.add_type("text/plain", ".atlas")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/octet-stream", ".moc3")
mimetypes.add_type("application/json", ".json")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server for smooth concurrent asset loading."""
    daemon_threads = True


class SpineViewerHandler(SimpleHTTPRequestHandler):
    """HTTP Request handler for Spine Viewer API and static assets."""

    # Server state injected by SpineViewerServer
    spine_dir: Path
    live2d_dir: Optional[Path] = None
    cache_dir: Path
    static_dir: Path
    ship_collection = None

    def log_message(self, format, *args):
        # Clean logging
        log.debug(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path in ("", "/", "/index.html"):
                self.serve_index()
            elif path.startswith("/static/"):
                self.serve_static(path[len("/static/"):])
            elif path == "/api/characters":
                self.api_characters()
            elif path.startswith("/api/model/"):
                self.handle_model_api(path[len("/api/model/"):])
            elif path.startswith("/api/live2d/"):
                self.handle_live2d_api(path[len("/api/live2d/"):])
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as e:
            log.exception(f"Error handling {path}: {e}")
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def serve_index(self):
        index_file = self.static_dir / "index.html"
        if not index_file.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "index.html not found")
            return
        content = index_file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(content)

    def serve_static(self, rel_path: str):
        safe_path = (self.static_dir / rel_path).resolve()
        # Security: protect against path traversal
        if not str(safe_path).startswith(str(self.static_dir.resolve())) or not safe_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return

        content_type, _ = mimetypes.guess_type(str(safe_path))
        content = safe_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def api_characters(self):
        if not SpineViewerHandler.ship_collection:
            try:
                log.info("Loading character and skin name metadata...")
                SpineViewerHandler.ship_collection = fetch_name_map()
            except Exception as e:
                log.warning(f"Could not load ship names on demand: {e}")

        models = []
        if self.spine_dir and self.spine_dir.exists():
            try:
                spine_models = list_spine_models(self.spine_dir, SpineViewerHandler.ship_collection)
                for m in spine_models:
                    m["engine"] = "spine"
                models.extend(spine_models)
            except Exception as e:
                log.error(f"Error listing Spine models: {e}")

        if self.live2d_dir and self.live2d_dir.exists():
            try:
                live2d_models = list_live2d_models(self.live2d_dir, SpineViewerHandler.ship_collection)
                for m in live2d_models:
                    m["engine"] = "live2d"
                models.extend(live2d_models)
            except Exception as e:
                log.error(f"Error listing Live2D models: {e}")

        models.sort(key=lambda x: (x["ship_name"].lower(), x["skin_name"].lower()))
        payload = json.dumps(models, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def handle_model_api(self, sub_path: str):
        # Route: <prefab>/info or <prefab>/files/<filename>
        parts = sub_path.split("/")
        if len(parts) < 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid model path")
            return

        prefab = parts[0]
        action = parts[1]

        # Locate bundle file
        bundle_file = self.spine_dir / f"{prefab}_res"
        if not bundle_file.is_file():
            bundle_file = self.spine_dir / prefab
        if not bundle_file.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"Spine bundle for {prefab} not found")
            return

        if action == "info":
            manifest = extract_spine_model(bundle_file, self.cache_dir)
            payload = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif action == "files" and len(parts) >= 3:
            filename = parts[2]
            target_file = (self.cache_dir / prefab / filename).resolve()
            expected_parent = (self.cache_dir / prefab).resolve()
            if not target_file.is_file():
                extract_spine_model(bundle_file, self.cache_dir)

            if not str(target_file).startswith(str(expected_parent)) or not target_file.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, f"File {filename} not found")
                return

            content_type, _ = mimetypes.guess_type(str(target_file))
            if target_file.suffix == ".skel":
                content_type = "application/octet-stream"
            elif target_file.suffix == ".atlas":
                content_type = "text/plain; charset=utf-8"

            content = target_file.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown action")
    def handle_live2d_api(self, sub_path: str):
        # Route: <prefab>/info or <prefab>/files/<path>
        parts = sub_path.split("/")
        if len(parts) < 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Live2D model path")
            return

        prefab = parts[0]
        action = parts[1]

        if not self.live2d_dir or not self.live2d_dir.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Live2D directory not configured or missing")
            return

        # Locate bundle file
        bundle_file = self.live2d_dir / prefab
        if not bundle_file.is_file():
            bundle_file = self.live2d_dir / f"{prefab}_hx"
        if not bundle_file.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"Live2D bundle for {prefab} not found")
            return

        if action == "info":
            manifest = extract_live2d_model(bundle_file, self.cache_dir)
            payload = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(payload)
        elif action == "files" and len(parts) >= 3:
            rel_path = "/".join(parts[2:])
            target_file = (self.cache_dir / f"{prefab}_l2d" / rel_path).resolve()
            expected_parent = (self.cache_dir / f"{prefab}_l2d").resolve()
            if not target_file.is_file():
                extract_live2d_model(bundle_file, self.cache_dir)
            elif rel_path.endswith("model3.json"):
                try:
                    with open(target_file, "r", encoding="utf-8") as f:
                        d = json.load(f)
                        if "Motions" not in d.get("FileReferences", {}) or d.get("Meta", {}).get("extractorVersion") != 2:
                            extract_live2d_model(bundle_file, self.cache_dir, force=True)
                except Exception:
                    pass
            if not str(target_file).startswith(str(expected_parent)) or not target_file.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, f"File {rel_path} not found")
                return

            content_type, _ = mimetypes.guess_type(str(target_file))
            if target_file.suffix == ".moc3":
                content_type = "application/octet-stream"
            elif target_file.suffix == ".json":
                content_type = "application/json; charset=utf-8"

            content = target_file.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown action")



class SpineViewerServer:
    """Manager for the Spine 2D Web Viewer server."""

    def __init__(
        self,
        spine_dir: Path,
        live2d_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        static_dir: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        self.spine_dir = Path(spine_dir)
        self.live2d_dir = Path(live2d_dir) if live2d_dir else None
        self.cache_dir = Path(cache_dir or Path(tempfile.gettempdir()) / "spine_cache")
        self.static_dir = Path(static_dir or Path(__file__).parent / "static")
        self.host = host
        self.port = port
        self.server: Optional[ThreadedHTTPServer] = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)

    def start(self, preload_names: bool = True):
        ship_collection = None
        if preload_names:
            try:
                log.info("Loading character and skin name metadata...")
                ship_collection = fetch_name_map()
            except Exception as e:
                log.warning(f"Could not load ship names: {e}")

        # Inject configuration into request handler class
        SpineViewerHandler.spine_dir = self.spine_dir
        SpineViewerHandler.live2d_dir = self.live2d_dir
        SpineViewerHandler.cache_dir = self.cache_dir
        SpineViewerHandler.static_dir = self.static_dir
        SpineViewerHandler.ship_collection = ship_collection

        # Auto-find available port if requested port is taken
        for p in range(self.port, self.port + 50):
            try:
                self.server = ThreadedHTTPServer((self.host, p), SpineViewerHandler)
                self.port = p
                break
            except OSError:
                continue

        if not self.server:
            raise RuntimeError(f"Could not bind server on {self.host}:{self.port}-{self.port + 49}")

        log.info(f"Spine 2D Viewer server running at http://{self.host}:{self.port}/")
        return self.server
