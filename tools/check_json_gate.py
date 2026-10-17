"""Check the JSON gate the way the harness actually calls the model.

ai_agent.py sends no max_tokens at all -- the request is model, messages and
temperature -- so the gateway's own default applies.  A probe that sets a cap
therefore tests a different request from the one the sweep will make, and an
earlier run mis-called deepseek-v4.1-flash unusable because a 1024-token cap was
consumed entirely by its reasoning (finish_reason=length, empty content).

This repeats the JSON gate with no cap, then with a large one, so the harness
behaviour and the worst case are both covered.

    python tools/check_json_gate.py --model deepseek-v4.1-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# No hard-coded gateway: a plaintext http:// default on a public IP puts the API key on the open
# internet, and a lab-specific address is not portable.  Supply it via an environment variable.
DEFAULT_BASE_URL: Optional[str] = None


def post(base_url: str, key: str, body: dict, timeout: float) -> dict:
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
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            detail = str(exc)
        return {"state": f"HTTP {exc.code}", "seconds": time.time() - started,
                "text": detail.replace("\n", " "), "finish": None, "usage": {}}
    except Exception as exc:  # noqa: BLE001
        return {"state": "FAIL", "seconds": time.time() - started,
                "text": str(exc)[:150], "finish": None, "usage": {}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    key = (
        os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        raise SystemExit("No API key. Set BOYUE_API_KEY.")
    base_url = args.base_url or os.environ.get("BOYUE_BASE_URL") or DEFAULT_BASE_URL
    timeout = float(os.environ.get("BAO_LLM_TIMEOUT", "180"))

    import protocol
    from ai_agent import parse_action_json

    prompt = protocol.build_prompt(
        state={"position": [0.5, 0.0, 0.0], "torso_rotation": 0.0, "camera_yaw": 0.0},
        history=[],
        max_steps=30,
    )

    # (label, extra request fields) -- the first matches ai_agent exactly.
    trials = [
        ("as the harness calls it (no max_tokens)", {}),
        ("max_tokens=8192", {"max_tokens": 8192}),
    ]

    print(f"endpoint {base_url}")
    print(f"model    {args.model}")
    print()
    # THE EXIT CODE IS THE POINT OF A GATE.  This tool printed "if the harness-shaped call parses, the
    # model is usable" and then returned 0 whatever it had seen -- so verify_professor_machine.sh, which
    # runs it as a check, passed on a model that answered nothing parseable at all.
    any_valid = False
    for label, extra in trials:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        body.update(extra)
        result = post(base_url, key, body, timeout)
        usage = result["usage"] or {}
        parsed = parse_action_json(result["text"]) if result["state"] == "OK" else None
        action = (parsed or {}).get("action")
        ok = parsed is not None and action in protocol.ACTIONS
        any_valid = any_valid or ok
        print(f"{label}")
        print(f"  state={result['state']:<8} {result['seconds']:>6.1f}s  "
              f"finish={result['finish']}  tokens={usage.get('total_tokens')}")
        print(f"  reply: {result['text'][:220]!r}")
        print(f"  -> parsed={parsed is not None} action={action!r} valid={ok}")
        print()

    print("if the harness-shaped call parses, the model is usable regardless of")
    print("what a capped probe says.")
    return 0 if any_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
