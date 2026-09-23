"""Find which image payload formats this gateway actually accepts.

A batch verify run reported every model failing with HTTP 400, yet a text-only
request to the same model succeeded -- so the models were fine and the image
payload was not.  The 1x1 PNG used there is the prime suspect: several vision
frontends reject images below a minimum edge length.

This tries a ladder of payloads against one known-good model and prints the
error body for each, so the accepted format is identified by measurement rather
than guessed at.

    python tools/image_format_probe.py --model qwen-vl-max
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"


def make_png(width: int, height: int, rgb=(200, 40, 40)) -> str:
    """Build a solid-colour PNG with zlib only -- no PIL dependency."""
    raw = b""
    row = bytes(rgb * width)
    for _ in range(height):
        raw += b"\x00" + row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def jpeg_placeholder() -> str:
    """A tiny but valid baseline JPEG, built from a known-good constant."""
    # 8x8 grey JPEG produced offline; kept as a literal so no encoder is needed.
    data = base64.b64decode(
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
        "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
        "AAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
    )
    return base64.b64encode(data).decode("ascii")


def request(base_url: str, api_key: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return "OK", time.time() - started, (text or "").strip()[:70]
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        return f"HTTP {exc.code}", time.time() - started, detail or str(exc)
    except Exception as exc:  # noqa: BLE001
        return "FAIL", time.time() - started, str(exc)[:200]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe accepted image payloads.")
    parser.add_argument("--model", default="qwen-vl-max")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit("No API key. Set BOYUE_API_KEY or pass --api-key.")
    base_url = args.base_url or os.environ.get("BOYUE_BASE_URL") or DEFAULT_BASE_URL

    text_only = "What colour is this image? One word."

    cases = [
        (
            "text only (control)",
            [{"type": "text", "text": "Reply with exactly: OK"}],
        ),
        (
            "text + 1x1 PNG (what failed)",
            [
                {"type": "text", "text": text_only},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + make_png(1, 1)},
                },
            ],
        ),
        (
            "text + 8x8 PNG",
            [
                {"type": "text", "text": text_only},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + make_png(8, 8)},
                },
            ],
        ),
        (
            "text + 64x64 PNG",
            [
                {"type": "text", "text": text_only},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + make_png(64, 64)},
                },
            ],
        ),
        (
            "text + tiny JPEG",
            [
                {"type": "text", "text": text_only},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + jpeg_placeholder()},
                },
            ],
        ),
        (
            "64x64 PNG, image_url as plain string",
            [
                {"type": "text", "text": text_only},
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64," + make_png(64, 64),
                },
            ],
        ),
    ]

    print(f"endpoint {base_url}   model {args.model}")
    print()
    for label, content in cases:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 30,
        }
        state, seconds, detail = request(base_url, api_key, payload, args.timeout)
        print(f"{label:<38} {state:<9} {seconds:>6.1f}s  {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
