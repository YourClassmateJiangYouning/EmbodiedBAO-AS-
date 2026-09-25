"""Which reasoning-effort knobs does the gateway accept, and do they help?

deepseek-v4.1-flash is the slowest model on the roster (30-86 s per real call; the
probe in models.json says 3.5 s because that probe was a 64x64 one-shot).  Its
replies run to ~1800 tokens, i.e. it spends most of the time thinking, so turning
the thinking down is worth measuring -- but there is no standard spelling for that
across providers, and this is an aggregator gateway, so guessing is useless.

Two stages, because a rejected parameter and a slow one need different evidence:

  stage 1  a one-word text request per variant: is the key ACCEPTED at all?
           Cheap (a few seconds each), so every candidate spelling is tried.
  stage 2  only for the variants stage 1 accepted: the REAL prompt with a 512 px
           frame, measured against the untouched baseline.  This is the number
           that matters, and it is the expensive one.

Nothing is modified here: this only reports.

    python tools/probe_reasoning.py --model deepseek-v4.1-flash
    python tools/probe_reasoning.py --model deepseek-v4.1-flash --stage 2 --repeat 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.check_model import solid_png  # noqa: E402  (same PNG writer, same gateway)

DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"

# Candidate spellings for "think less", from the providers this gateway fronts.
VARIANTS: dict = {
    "baseline (what the sweep sends)": {},
    'reasoning_effort="low"': {"reasoning_effort": "low"},
    'reasoning_effort="minimal"': {"reasoning_effort": "minimal"},
    'reasoning_effort="none"': {"reasoning_effort": "none"},
    'reasoning_effort="medium"': {"reasoning_effort": "medium"},
    "enable_thinking=False": {"enable_thinking": False},
    "chat_template_kwargs.thinking=False": {
        "chat_template_kwargs": {"thinking": False}
    },
    'thinking={"type":"disabled"}': {"thinking": {"type": "disabled"}},
    "max_tokens=512": {"max_tokens": 512},
    "max_tokens=1024": {"max_tokens": 1024},
    "max_tokens=2048": {"max_tokens": 2048},
    "max_tokens=4096": {"max_tokens": 4096},
}


def post(base_url: str, key: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
    )
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
            "tokens": (data.get("usage") or {}).get("total_tokens"),
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = str(exc)
        return {
            "state": f"HTTP {exc.code}",
            "seconds": time.time() - started,
            "text": " ".join(detail.split()),
            "finish": None,
            "tokens": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "state": "FAIL",
            "seconds": time.time() - started,
            "text": str(exc)[:200],
            "finish": None,
            "tokens": None,
        }


def stage1(model: str, base_url: str, key: str, timeout: float) -> list:
    print("stage 1 - is the key accepted?  (one-word text request)")
    print(f"  {'variant':<40} {'state':<10} {'s':>7}  reply")
    accepted = []
    for label, extra in VARIANTS.items():
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        }
        body.update(extra)
        result = post(base_url, key, body, timeout)
        print(
            f"  {label:<40} {result['state']:<10} {result['seconds']:>7.1f}  "
            f"{(result['text'] or '')[:60]!r}"
        )
        if result["state"] == "OK":
            accepted.append(label)
    print()
    print(f"  accepted: {len(accepted)}/{len(VARIANTS)}")
    for label in accepted:
        print(f"    - {label}")
    return accepted


def stage2(
    model: str,
    base_url: str,
    key: str,
    timeout: float,
    labels: list,
    repeat: int,
) -> dict:
    import protocol

    prompt = protocol.build_prompt(
        state={
            "position": [7.25, 0.0, 0.0],
            "torso_rotation": 0.0,
            "camera_yaw": 0.0,
            "camera_pitch": 0.0,
        },
        history=[],
        max_steps=30,
    )
    image = solid_png(512, (70, 74, 80))
    print(f"stage 2 - the REAL workload, {repeat} run(s) per variant")
    print(f"  prompt {len(prompt)} chars, 512 px frame, no max_tokens (as the sweep sends it)")
    print()
    print(
        f"  {'variant':<40} {'n':>2} {'median s':>9} {'min s':>7} {'tokens':>7}  "
        f"JSON+action"
    )
    results = {}
    for label in labels:
        extra = VARIANTS[label]
        times = []
        tokens = []
        valid = 0
        for _ in range(repeat):
            body = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image}},
                        ],
                    }
                ],
                "temperature": 0.0,
            }
            body.update(extra)
            result = post(base_url, key, body, timeout)
            if result["state"] != "OK":
                print(f"  {label:<40} {result['state']} {result['text'][:80]!r}")
                break
            times.append(result["seconds"])
            if result["tokens"]:
                tokens.append(result["tokens"])
            try:
                parsed = json.loads(result["text"])
                if isinstance(parsed, dict) and parsed.get("action"):
                    valid += 1
            except Exception:
                pass
        if times:
            results[label] = {
                "median": statistics.median(times),
                "min": min(times),
                "tokens": int(statistics.median(tokens)) if tokens else None,
                "valid": valid,
                "n": len(times),
            }
            print(
                f"  {label:<40} {len(times):>2} {statistics.median(times):>9.1f} "
                f"{min(times):>7.1f} "
                f"{(int(statistics.median(tokens)) if tokens else 0):>7}  "
                f"{valid}/{len(times)}"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe reasoning-effort knobs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--stage", type=int, default=0, help="1, 2 or 0 for both")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated substrings; only matching variant labels are tested",
    )
    args = parser.parse_args()

    key = (
        os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        raise SystemExit("No API key. Set BOYUE_API_KEY.")
    base_url = args.base_url or os.environ.get("BOYUE_BASE_URL") or DEFAULT_BASE_URL
    print(f"endpoint {base_url}")
    print(f"model    {args.model}")
    print(f"timeout  {args.timeout:.0f}s")
    print()

    accepted = list(VARIANTS)
    if args.only:
        wanted = [part.strip() for part in args.only.split(",") if part.strip()]
        accepted = [
            label
            for label in VARIANTS
            if any(part.lower() in label.lower() for part in wanted)
        ]
        print(f"filtered to {len(accepted)} variant(s): {accepted}")
        print()
    if args.stage in (0, 1):
        accepted = stage1(args.model, base_url, key, args.timeout)
        print()
    if args.stage in (0, 2):
        if args.stage == 2 and not args.only:
            # On this gateway every spelling returns HTTP 200, so stage 1 cannot
            # tell an accepted key from an ignored one; stage 2 is the only test.
            accepted = list(VARIANTS)
        stage2(args.model, base_url, key, args.timeout, accepted, args.repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
