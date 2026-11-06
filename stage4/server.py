"""Stage 4 window: a local web server that renders the operator UI and exposes the loop as JSON.

Why a local web page rather than a desktop toolkit: the operator machine only needs Python and a browser, the
layout is easy to keep legible (large image next to the model output), and every button maps to one HTTP call
so the UI has no hidden state of its own -- `core.py` remains the single source of truth.

Run it with `python stage4/run.py`, which opens http://127.0.0.1:8765/ automatically.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from core import (
    A_S_LEVELS, DEFAULT_MODELS, DEFAULT_TIMEOUT_S, Stage4Session, now_iso, read_jsonl,
)

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "ui.html")

SESSION = Stage4Session()
LOCK = threading.Lock()


def json_response(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200,
                  content_type: str = "text/plain; charset=utf-8") -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    server_version = "Stage4/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003 -- keep the console readable
        return

    # -- helpers --------------------------------------------------------------------------------
    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise RuntimeError("request body is not valid JSON")

    def _fail(self, message: str, status: int = 400) -> None:
        json_response(self, {"ok": False, "error": message, "state": SESSION.state()}, status)

    # -- routes ---------------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                with open(UI_PATH, "rb") as handle:
                    body = handle.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/state":
                json_response(self, {"ok": True, "state": SESSION.state()})
                return

            if path == "/api/sessions":
                root = SESSION.sessions_root
                names = sorted(
                    (n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))),
                    reverse=True,
                ) if os.path.isdir(root) else []
                json_response(self, {"ok": True, "sessions": names})
                return

            if path.startswith("/api/sessions/") and path.endswith("/episodes"):
                session_id = path[len("/api/sessions/"):-len("/episodes")].strip("/")
                base = os.path.join(SESSION.sessions_root, session_id)
                rows = read_jsonl(os.path.join(base, "episodes.jsonl"))
                brief = [{
                    "episode_id": r.get("episode_id"), "model": r.get("model"), "marker": r.get("marker"),
                    "a_s": r.get("a_s"), "mode": r.get("mode"), "passed": r.get("passed"),
                    "total_steps": r.get("total_steps"), "end_reason": r.get("end_reason"),
                } for r in rows]
                json_response(self, {"ok": True, "episodes": brief})
                return

            if path.startswith("/api/image/"):
                image_id = path[len("/api/image/"):]
                if not SESSION.dir:
                    self._fail("no session open")
                    return
                images_dir = os.path.join(SESSION.dir, "images")
                for name in os.listdir(images_dir):
                    if name.startswith(image_id):
                        data = open(os.path.join(images_dir, name), "rb").read()
                        content_type = "image/png" if name.lower().endswith(".png") else "image/jpeg"
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                self._fail("image not found", 404)
                return

            self._fail("unknown path", 404)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._fail("%s: %s" % (type(error).__name__, error), 500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._body()
        except Exception as error:  # noqa: BLE001
            self._fail(str(error))
            return

        try:
            with LOCK:
                if path == "/api/session":
                    state = SESSION.open_session(
                        body.get("session_id") or "session",
                        {
                            "scene": body.get("scene"),
                            "marker": body.get("marker"),
                            "marker_descriptor": body.get("marker_descriptor"),
                            "mode": body.get("mode"),
                            "model": body.get("model"),
                            "a_s": body.get("a_s"),
                            "max_steps": body.get("max_steps"),
                            "memory_id": body.get("memory_id"),
                            "api_key": body.get("api_key"),
                            "endpoint": body.get("endpoint"),
                            "timeout_s": body.get("timeout_s") or DEFAULT_TIMEOUT_S,
                            "disable_thinking": body.get("disable_thinking", True),
                        },
                    )
                    SESSION.config["timeout_s"] = float(body.get("timeout_s") or DEFAULT_TIMEOUT_S)
                    json_response(self, {"ok": True, "state": state})
                    return

                if path == "/api/memory":
                    source = body.get("source")
                    if source == "episode":
                        session_id = body.get("session_id")
                        base = os.path.join(SESSION.sessions_root, session_id)
                        rows = read_jsonl(os.path.join(base, "episodes.jsonl"))
                        target = next((r for r in rows if r.get("episode_id") == body.get("episode_id")), None)
                        if not target:
                            self._fail("episode not found in that session")
                            return
                        # the steps of that episode come from its own steps.jsonl
                        saved_steps = SESSION.steps
                        SESSION.steps = read_jsonl(os.path.join(base, "steps.jsonl"))
                        SESSION.memory = SESSION.build_memory_from_episode(target)
                        SESSION.steps = saved_steps
                    elif source == "file":
                        path_in = body.get("path") or ""
                        if not os.path.exists(path_in):
                            self._fail("memory file not found: %s" % path_in)
                            return
                        SESSION.memory = SESSION.load_memory_file(path_in)
                    else:
                        self._fail("source must be 'episode' or 'file'")
                        return
                    with open(os.path.join(SESSION.dir, "memory", "%s.json" % SESSION.memory["memory_id"]),
                              "w", encoding="utf-8") as handle:
                        json.dump(SESSION.memory, handle, indent=2, ensure_ascii=False)
                    json_response(self, {"ok": True, "state": SESSION.state()})
                    return

                if path == "/api/start":
                    json_response(self, {"ok": True, "state": SESSION.start_episode()})
                    return

                if path == "/api/capture":
                    data = body.get("data_base64") or ""
                    if not data:
                        self._fail("no image data")
                        return
                    raw = base64.b64decode(data)
                    state = SESSION.capture(body.get("filename") or "frame.png", raw,
                                            body.get("source") or "upload")
                    json_response(self, {"ok": True, "state": state})
                    return

                if path == "/api/digest":
                    json_response(self, {"ok": True, "state": SESSION.run_digest()})
                    return

                if path == "/api/call":
                    json_response(self, {"ok": True, "state": SESSION.call_model()})
                    return

                if path == "/api/action":
                    state = SESSION.action_executed(
                        feedback=body.get("feedback") or "executed",
                        collision=bool(body.get("collision")),
                        position_x=body.get("position_x"),
                        position_z=body.get("position_z"),
                        torso_rotation=body.get("torso_rotation"),
                        passed=body.get("passed"),
                        note=body.get("note") or "",
                    )
                    json_response(self, {"ok": True, "state": state})
                    return

                if path == "/api/finish":
                    if not SESSION.episode:
                        self._fail("no open episode")
                        return
                    state = SESSION.finish(bool(body.get("passed")), body.get("end_reason") or "")
                    json_response(self, {"ok": True, "state": state})
                    return

                if path == "/api/stop":
                    json_response(self, {"ok": True, "state": SESSION.stop()})
                    return

                if path == "/api/export":
                    result = SESSION.export()
                    json_response(self, {"ok": True, **result})
                    return

            self._fail("unknown path", 404)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._fail("%s: %s" % (type(error).__name__, error), 500)


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = serve(port=port)
    print("Stage 4 window on http://127.0.0.1:%d/  (Ctrl+C to stop)" % port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
