"""Verify which gateway models can actually serve this experiment.

The benchmark sends a camera frame every step, so a model is only usable if it
accepts image input.  Text-only models in the list (deepseek-v4-pro, glm-5.3,
qwen3.5-397b-a17b, ...) would fail mid-episode, and a model that 500s would be
discovered only after a long run.

Each candidate gets one request carrying BOTH text and a 1x1 PNG, which proves
image input and reachability together.  The candidates are grouped by role:

    MIRRORBENCH  present in MirrorBench Table II, so scores are comparable
    CN-NEW       domestic vision models newer than the MirrorBench ones
    CN-?         domestic models whose image support is unknown

    python tools/verify_models.py --out analysis/model_verify.json
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
from typing import Dict, List, Tuple

DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"

CANDIDATES: List[Tuple[str, str]] = [
    # --- MirrorBench Table II: needed for a comparable threshold ---
    ("MIRRORBENCH", "gemini-2.5-pro"),
    ("MIRRORBENCH", "gemini-2.5-flash"),
    ("MIRRORBENCH", "qwen-vl-max"),
    ("MIRRORBENCH", "qwen2.5-vl-72b-instruct"),
    ("MIRRORBENCH", "glm-4.5v"),
    ("MIRRORBENCH", "gpt-4o"),
    ("MIRRORBENCH", "gpt-4.1"),
    ("MIRRORBENCH", "claude-sonnet-4-5-20250929"),
    ("MIRRORBENCH", "gpt-4o-mini"),
    # --- domestic vision models newer than the MirrorBench generation ---
    ("CN-NEW", "qwen3-vl-235b-a22b-instruct"),
    ("CN-NEW", "qwen3-vl-235b-a22b-thinking"),
    ("CN-NEW", "qwen3-vl-32b-instruct"),
    ("CN-NEW", "qwen3-vl-30b-a3b-instruct"),
    ("CN-NEW", "qwen3-vl-8b-instruct"),
    ("CN-NEW", "qwen3-vl-plus"),
    ("CN-NEW", "glm-4.6v"),
    ("CN-NEW", "glm-5v-turbo"),
    ("CN-NEW", "z-ai/glm-4.5v"),
    ("CN-NEW", "opengvlab/internvl3-14b"),
    ("CN-NEW", "deepseek-v4-flash-vision-exp"),
    ("CN-NEW", "qvq-72b-preview"),
    # --- image support unknown ---
    ("CN-?", "MiniMax-M3"),
    ("CN-?", "doubao-seed-2-0-pro-260215"),
    ("CN-?", "mimo-v2-omni"),
    ("CN-?", "stepfun-ai/Step-3.5-Flash"),
    ("CN-?", "llama-3.2-11b-vision-instruct"),
    ("CN-?", "mistralai/pixtral-large-2411"),
]

def make_png(width: int, height: int, rgb=(200, 40, 40)) -> str:
    """Build a solid-colour PNG using zlib only -- no image library needed.

    The payload must be at least 11 px on each edge.  Measured on this gateway:

        height:1 or width:1 must be larger than 10
        height:8 or width:8 must be larger than 10

    so a 1x1 probe image (the obvious minimal choice) makes EVERY model look
    broken with HTTP 400, which is exactly the false negative this avoids.
    """
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


PROBE_IMAGE = "data:image/png;base64," + make_png(64, 64)


def probe(base_url: str, api_key: str, model: str, timeout: float) -> Dict[str, object]:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What colour is this image? One word.",
                        },
                        {"type": "image_url", "image_url": {"url": PROBE_IMAGE}},
                    ],
                }
            ],
            "temperature": 0.0,
            # 300, not 20: reasoning models spend the budget thinking and return
            # empty content with finish_reason=length at 20 tokens, which made
            # gemini-2.5-pro and gemini-2.5-flash look broken when both answer
            # correctly at 300.
            "max_tokens": 300,
        }
    ).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body)
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        seconds = time.time() - started
        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        finish = str(choice.get("finish_reason") or "")
        if not text:
            # Distinguish "the budget ran out mid-thought" from "the model said
            # nothing", because only the first is a probe artefact.
            hint = " (truncated: raise max_tokens)" if finish == "length" else ""
            return {
                "state": "EMPTY",
                "seconds": seconds,
                "detail": f"no content, finish_reason={finish or '?'}{hint}",
            }
        # The probe image is solid red.  A model that accepts the request but
        # cannot actually see would name something else, so requiring the colour
        # makes this a test of image understanding, not just of status codes.
        if "red" not in text.lower():
            return {
                "state": "NO-VISION",
                "seconds": seconds,
                "detail": f"did not name the colour: {text[:60]}",
            }
        return {"state": "OK", "seconds": seconds, "detail": text[:60].replace("\n", " ")}
    except Exception as exc:  # noqa: BLE001 - report every failure kind
        return {
            "state": "FAIL",
            "seconds": time.time() - started,
            "detail": str(exc)[:100],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify gateway model usability.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit("No API key. Set BOYUE_API_KEY or pass --api-key.")
    base_url = (
        args.base_url
        or os.environ.get("BOYUE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )

    print(f"endpoint {base_url}")
    print("each candidate gets one text+image request; the image is solid red, so")
    print("OK means the model both accepted the image and could see it")
    print()
    print(f"{'group':<12} {'model':<36} {'state':<10} {'secs':>7}  detail")
    print("-" * 112)

    results: List[Dict[str, object]] = []
    usable: Dict[str, List[str]] = {}
    for group, model in CANDIDATES:
        result = probe(base_url, api_key, model, args.timeout)
        results.append({"group": group, "model": model, **result})
        if result["state"] == "OK":
            usable.setdefault(group, []).append(model)
        print(
            f"{group:<12} {model:<36} {result['state']:<10} "
            f"{result['seconds']:>7.1f}  {result['detail']}"
        )

    print("-" * 112)
    for group in ("MIRRORBENCH", "CN-NEW", "CN-?"):
        names = usable.get(group, [])
        print(f"{group} usable ({len(names)}): {', '.join(names) if names else '-'}")

    report = {
        "base_url": base_url,
        "results": results,
        "usable": usable,
    }
    out_path = args.out
    if out_path:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
