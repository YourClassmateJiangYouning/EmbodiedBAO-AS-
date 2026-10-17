"""Connectivity and suitability check for one model, before a long sweep.

A model is only usable for this benchmark if it passes four separate gates, and
they fail independently, so checking only the first is not enough:

  0. the NAME is one the gateway actually serves, checked verbatim against
     `/v1/models`.  This gate exists because the other three cannot see the
     problem it catches: the gateway does NOT reject an unknown model name.  It
     either returns 503, or silently routes the request to a fallback that
     answers plausibly and cannot see the image -- so a typo, or a name copied
     from a stale list inside this repository, presents as "the model is blind".
     That is exactly what happened on 2026-10-15: `qwen3-vl-8b-instruct` (from
     tools/verify_models.py's candidate list) answered "Orange" to a solid red
     frame while the same model under its real spelling,
     `Qwen/Qwen3-VL-8B-Instruct`, passed every gate.  The wrong name was mine;
     the model was never broken.  This gate costs one GET and no model call.
  1. the route answers at all;
  2. it can SEE -- the experiment sends a camera frame every step, and
     deepseek-v4-flash and deepseek-v4-pro both accept images while answering
     "Orange" or "Unknown." about a solid red one;
  3. it returns the JSON object the harness parses, which is a different failure
     from being unable to see and shows up as a flood of invalid steps.

Gate 3 is tested with the real response instruction and the real action list, not
an invented prompt, because a model can be perfectly capable and still wrap its
answer in prose or a code fence.

    python tools/check_model.py --model deepseek-v4.1-flash
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from typing import Optional
import time
import urllib.error
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# NO HARD-CODED GATEWAY ADDRESS; must come from --base_url or an environment variable.  A plaintext
# http:// default on a public IP put the API key on the open internet, and a tool whose default is a
# specific lab gateway is not portable anyway.
DEFAULT_BASE_URL: Optional[str] = None


def solid_png(size: int, rgb) -> str:
    """A legal solid-colour PNG; the gateway rejects edges under 11 px."""
    raw = b"".join(b"\x00" + bytes(rgb * size) for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def listed_models(base_url: str, key: str, timeout: float):
    """Every model id the gateway advertises, or None when the list is unreadable.

    None is deliberately distinct from an empty set: "could not ask" must not be
    reported as "the name is not served".
    """
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return {m.get("id") for m in (data.get("data") or []) if m.get("id")}
    except Exception:  # noqa: BLE001
        return None


def post(base_url: str, key: str, body: dict, timeout: float) -> tuple:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=payload)
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "state": "OK",
            "seconds": time.time() - started,
            "text": (message.get("content") or "").strip(),
            "finish": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:220]
        except Exception:
            detail = str(exc)
        return {
            "state": f"HTTP {exc.code}",
            "seconds": time.time() - started,
            "text": detail.replace("\n", " "),
            "finish": None,
            "usage": {},
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "state": "FAIL",
            "seconds": time.time() - started,
            "text": str(exc)[:160],
            "finish": None,
            "usage": {},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check one model before a sweep.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    key = (
        os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        raise SystemExit("No API key. Set BOYUE_API_KEY.")
    base_url = (args.base_url or os.environ.get("BOYUE_BASE_URL")
                or os.environ.get("TAOTOKEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or DEFAULT_BASE_URL)
    if not base_url:
        print("check_model: no base URL. Pass --base_url or set BOYUE_BASE_URL.", file=sys.stderr)
        return 2
    timeout = args.timeout or float(os.environ.get("BAO_LLM_TIMEOUT", "90"))
    model = args.model

    print(f"endpoint {base_url}")
    print(f"model    {model}")
    # NEVER print any part of the key.  This line used to print the first 8 and last 4 characters
    # (`key {key[:8]}...{key[-4:]}`), and callers redirect this output into logs/
    # (logs/queue_mimo_gemini.log, logs/STATUS.txt); the operator's key has leaked from this
    # repository once already.  Length plus presence is enough to confirm the right variable was
    # picked up, and cannot be reversed into the key.
    print(f"key      <redacted, {len(key)} chars>")
    print()

    failures = []

    # ---- gate 0: is this a name the gateway actually serves? --------------
    # Checked FIRST, because it is free and because the other three gates cannot
    # distinguish "blind model" from "name the gateway does not serve".
    print("gate 0 - advertised name (verbatim, against /v1/models)")
    ids = listed_models(base_url, key, timeout)
    if ids is None:
        print("  could not read /v1/models; this gate is skipped, not passed")
    elif model in ids:
        print(f"  listed verbatim, among {len(ids)} advertised models")
    else:
        stem = model.split("/")[-1].lower()
        near = sorted(
            i for i in ids
            if i.lower() == model.lower() or stem in i.lower() or model.lower() in i.lower()
        )[:8]
        print(f"  NOT ADVERTISED: {model!r} is not among the {len(ids)} models /v1/models lists")
        if near:
            print("  the gateway does list names close to it:")
            for name in near:
                print(f"    {name}")
        print()
        print("Refusing to spend a model call on an unadvertised name.")
        print("The gateway does not reject an unknown model: it either returns 503 or silently")
        print("routes to a fallback that cannot see the image, which then looks like a blind")
        print("model rather than a wrong name.  Use a spelling from the list above.")
        return 2
    print()

    # ---- gate 1: does it answer at all ------------------------------------
    print("gate 1 - reachable")
    result = post(
        base_url,
        key,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            # Reasoning models spend the budget thinking; a small cap returns
            # empty content with finish_reason=length and looks like a failure.
            "max_tokens": 512,
        },
        timeout,
    )
    text_ok = result["state"] == "OK" and bool(result["text"])
    print(
        f"  {result['state']:<10} {result['seconds']:>6.1f}s  "
        f"finish={result['finish']}  {result['text'][:50]!r}"
    )
    if not text_ok:
        failures.append("gate 1: no usable text reply")
    print()

    # ---- gate 2: can it actually see -------------------------------------
    print("gate 2 - vision (a solid red frame must be named red)")
    red = post(
        base_url,
        key,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this image? One word."},
                        {"type": "image_url", "image_url": {"url": solid_png(64, (200, 40, 40))}},
                    ],
                }
            ],
            "max_tokens": 512,
        },
        timeout,
    )
    sees_red = red["state"] == "OK" and "red" in (red["text"] or "").lower()
    print(
        f"  red  {red['state']:<10} {red['seconds']:>6.1f}s  "
        f"finish={red['finish']}  {red['text'][:50]!r}"
    )

    # A second colour guards against a model that always answers "red".
    blue = post(
        base_url,
        key,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this image? One word."},
                        {"type": "image_url", "image_url": {"url": solid_png(64, (30, 60, 200))}},
                    ],
                }
            ],
            "max_tokens": 512,
        },
        timeout,
    )
    sees_blue = blue["state"] == "OK" and "blue" in (blue["text"] or "").lower()
    print(
        f"  blue {blue['state']:<10} {blue['seconds']:>6.1f}s  "
        f"finish={blue['finish']}  {blue['text'][:50]!r}"
    )
    if not (sees_red and sees_blue):
        failures.append("gate 2: does not reliably name image colours")
    print()

    # ---- gate 3: does it emit the JSON the harness parses ----------------
    print("gate 3 - response format (must be one JSON object the harness parses)")
    import protocol

    prompt = protocol.build_prompt(
        state={"position": [0.5, 0.0, 0.0], "torso_rotation": 0.0, "camera_yaw": 0.0},
        history=[],
        max_steps=30,
    )
    formatted = post(
        base_url,
        key,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": solid_png(256, (60, 60, 60))}},
                    ],
                }
            ],
            # Generous on purpose.  A reasoning model spends most of the budget
            # before it emits the object, and deepseek-v4.1-flash returned empty
            # content with finish_reason=length at 1024 -- 1819 tokens used and
            # nothing to parse.  Reporting that as "cannot produce JSON" would
            # wrongly disqualify a model that is merely verbose.
            "max_tokens": 8192,
        },
        timeout,
    )
    print(
        f"  {formatted['state']:<10} {formatted['seconds']:>6.1f}s  "
        f"finish={formatted['finish']}  tokens={(formatted['usage'] or {}).get('total_tokens')}"
    )
    print(f"  reply: {(formatted['text'] or '')[:200]!r}")

    parsed = None
    if formatted["state"] == "OK":
        from ai_agent import parse_action_json

        parsed = parse_action_json(formatted["text"])
    if parsed is None:
        if formatted["finish"] == "length":
            failures.append(
                "gate 3: budget exhausted before any output "
                "(finish_reason=length) -- raise max_tokens, not a model defect"
            )
            print("  -> truncated before emitting; inconclusive, not a failure")
        else:
            failures.append("gate 3: reply is not a parseable action object")
            print("  -> could NOT parse an action object")
    else:
        action = parsed.get("action")
        valid = action in protocol.ACTIONS
        print(f"  -> parsed action={action!r} valid={valid}")
        if not valid:
            failures.append(f"gate 3: action {action!r} is not one of the actions")
    print()

    # ---- verdict ---------------------------------------------------------
    print("=" * 74)
    if failures:
        print(f"NOT READY - {len(failures)} problem(s):")
        for item in failures:
            print(f"  - {item}")
        if any("gate 2" in item for item in failures):
            print()
            print("A gate 2 failure means the model cannot see, which makes it")
            print("unusable for this benchmark however well it reasons:")
            print("deepseek-v4-pro answered 'Unknown.' and deepseek-v4-flash")
            print("answered 'Orange' to a solid red frame.")
        return 1
    print(f"READY - {model} answered, sees colour, and returns a valid action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
