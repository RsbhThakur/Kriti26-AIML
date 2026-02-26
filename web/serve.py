"""
No-cache HTTP server for bdh-visualizer/web.
Usage:
    python web/serve.py
Then open:
    http://127.0.0.1:8000/web/setup.html
"""
from __future__ import annotations

import http.server
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_hebbian_data import compute_hebbian
from scripts.export_web_data import export_data


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _ok(self, payload: dict):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _err(self, msg: str):
        self.send_response(500)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "error", "message": msg}).encode("utf-8"))

    def do_POST(self):  # noqa: N802
        if self.path == "/api/run":
            self._handle_run()
            return
        if self.path == "/api/hebbian-run":
            self._handle_hebbian()
            return
        self.send_error(404, "Endpoint not found")

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _handle_run(self):
        try:
            data = self._read_json()
            start_r, start_c = data.get("start", [0, 0])
            end_r, end_c = data.get("end", [0, 0])
            neurons = int(data.get("neurons", 1000))
            if neurons not in (300, 500, 1000, 2000, 5000):
                neurons = 1000

            out_path = ROOT / "web" / "data" / "viz_data.json"
            print(
                f"\n[API] /api/run start=({start_r},{start_c}) end=({end_r},{end_c}) "
                f"neurons={neurons}"
            )
            export_data(
                output_path=str(out_path),
                m_neurons=neurons,
                threshold=0.035,
                start_pos=(int(start_r), int(start_c)),
                end_pos=(int(end_r), int(end_c)),
                max_attempts=60,
            )
            print("[API] viz_data.json generated\n")
            self._ok({"status": "success", "neurons": neurons})
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self._err(str(e))

    def _handle_hebbian(self):
        try:
            data = self._read_json()
            n_boards = int(data.get("n_boards", 30))
            neurons = int(data.get("neurons", 1000))
            if neurons not in (300, 500, 1000, 2000, 5000):
                neurons = 1000

            out_path = ROOT / "web" / "data" / "hebbian_data.json"
            print(f"\n[API] /api/hebbian-run boards={n_boards} neurons={neurons}")
            compute_hebbian(
                output_path=str(out_path),
                n_boards=n_boards,
                m_neurons=neurons,
                threshold=0.035,
                max_edges=2000,
            )
            print("[API] hebbian_data.json generated\n")
            self._ok({"status": "success", "neurons": neurons, "n_boards": n_boards})
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self._err(str(e))


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print(f"Serving on http://{host}:{port}/web/setup.html")
    server = http.server.HTTPServer((host, port), NoCacheHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
