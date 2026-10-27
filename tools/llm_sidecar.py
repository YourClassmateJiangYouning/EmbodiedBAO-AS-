"""Local OpenAI-compatible HTTP sidecar.

Why this exists: the Isaac Python interpreter can issue small gateway requests,
but in this environment its large image POSTs stall (urllib and http.client
both time out at 120 s).  A normal Python interpreter on the same host sends
the same 1.6 MB image in ~7 s.  This sidecar runs under the normal interpreter,
accepts the Isaac process's requests on localhost, and forwards them upstream
unchanged.

It is deliberately a transparent proxy: no prompt rewriting, no retry policy,
no model-specific logic.  The experiment's request body, headers and response
are preserved.

    BOYUE_API_KEY=... python3 tools/llm_sidecar.py \
        --port 3899 --upstream http://35.220.164.252:3888 --timeout 300

Then point the experiment at:

    BOYUE_BASE_URL=http://127.0.0.1:3899/v1

The upstream must be the gateway ORIGIN (no /v1 suffix); incoming paths such
as /v1/chat/completions and /v1/models are forwarded verbatim.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socket
import sys
import urllib.error
import urllib.request

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "llm-sidecar/1.0"

    def _proxy(self, method: str) -> None:
        upstream = self.server.upstream  # type: ignore[attr-defined]
        timeout = self.server.timeout_s  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        url = upstream.rstrip("/") + self.path
        request = urllib.request.Request(url, data=body, method=method)
        for name, value in self.headers.items():
            if name.lower() in HOP_BY_HOP:
                continue
            request.add_header(name, value)
        if not request.has_header("Authorization"):
            key = os.environ.get("BOYUE_API_KEY") or os.environ.get("TAOTOKEN_API_KEY")
            if key:
                request.add_header("Authorization", "Bearer " + key)

        started = __import__("time").time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            content_type = exc.headers.get("Content-Type", "application/json")
        except Exception as exc:  # noqa: BLE001
            payload = (
                '{"error":{"message":"sidecar upstream failed: '
                + type(exc).__name__
                + ': '
                + str(exc).replace('"', "'")[:300]
                + '"}}'
            ).encode()
            status = 502
            content_type = "application/json"

        elapsed = __import__("time").time() - started
        sys.stderr.write(
            f"sidecar {method} {self.path} -> {status} "
            f"req={length}B resp={len(payload)}B {elapsed:.1f}s\n"
        )
        sys.stderr.flush()

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._proxy("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._proxy("POST")

    def log_message(self, fmt: str, *args) -> None:
        # The default logger writes request paths; keep it, but avoid noisy
        # per-connection messages at INFO level.
        sys.stderr.write("sidecar http: " + (fmt % args) + "\n")


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3899)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    server = Server((args.host, args.port), Handler)
    server.upstream = args.upstream  # type: ignore[attr-defined]
    server.timeout_s = args.timeout  # type: ignore[attr-defined]
    print(
        f"llm sidecar listening on http://{args.host}:{args.port} "
        f"-> {args.upstream} (timeout {args.timeout:g}s)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
