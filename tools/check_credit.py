"""Tell apart "no credit" from "bad key" and "model unavailable" on the gateway.

There is no balance endpoint on this gateway, so the three failure modes have to
be separated by what the API actually says and by whether the failure follows the
key or the model:

  * a key problem fails EVERY model, including cheap ones;
  * a balance problem also fails every model, but the error body names quota or
    balance rather than token;
  * a model problem fails only some models while others succeed.

The last distinction matters most in practice: a single 401 while one model is
being used looks like a dead account, but a sweep across several providers shows
immediately whether the account is fine and only that model is unavailable.

    python tools/check_credit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"

# One cheap model per family, ordered so the cheapest reachable one answers first.
PROBES = [
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "qwen-vl-max",
    "qwen3-vl-235b-a22b-instruct",
    "deepseek-v4.1-flash",
    "glm-4.6v",
    "claude-sonnet-4-5-20250929",
]

# Keywords that indicate the account rather than the request.
CREDIT_HINTS = ("quota", "balance", "insufficient", "credit", "exceeded", "欠费", "余额")
KEY_HINTS = ("invalid token", "unauthorized", "authentication", "api key", "no permission")


def describe(body: str) -> str:
    low = body.lower()
    if any(h in low for h in CREDIT_HINTS):
        return "ACCOUNT (quota/balance mentioned)"
    if any(h in low for h in KEY_HINTS):
        return "KEY (token rejected)"
    return "unknown"


def probe(base_url: str, key: str, model: str, timeout: float) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 5,
        }
    ).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=payload)
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
        body = json.loads(raw)
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return {
            "state": "OK",
            "seconds": time.time() - started,
            "detail": text.strip()[:40] or "(empty)",
            "headers": dict(response.headers),
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = str(exc)
        return {
            "state": f"HTTP {exc.code}",
            "seconds": time.time() - started,
            "detail": detail.replace("\n", " "),
            "headers": dict(exc.headers or {}),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "state": "FAIL",
            "seconds": time.time() - started,
            "detail": str(exc)[:200],
            "headers": {},
        }


def main() -> int:
    key = (
        os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        raise SystemExit("No API key. Set BOYUE_API_KEY.")
    base_url = os.environ.get("BOYUE_BASE_URL") or DEFAULT_BASE_URL
    timeout = float(os.environ.get("BAO_LLM_TIMEOUT", "60"))

    print(f"endpoint : {base_url}")
    print(f"key      : {key[:8]}...{key[-4:]}  ({len(key)} chars)")
    print()
    print(f"{'model':<36} {'state':<10} {'secs':>6}  detail")
    print("-" * 100)

    results = []
    for model in PROBES:
        result = probe(base_url, key, model, timeout)
        results.append((model, result))
        print(
            f"{model:<36} {result['state']:<10} {result['seconds']:>6.1f}  {result['detail'][:56]}"
        )

    ok = [m for m, r in results if r["state"] == "OK"]
    print("-" * 100)
    print(f"reachable: {len(ok)}/{len(results)}  {ok if ok else ''}")

    if ok:
        print()
        print("VERDICT: the account works. Any failure above is specific to that")
        print("         model or provider, not to the key or the balance.")
    else:
        verdicts = {describe(r["detail"]) for _, r in results}
        print()
        print("VERDICT: every model failed. Causes, by error text:")
        for v in sorted(verdicts):
            print(f"  - {v}")
        print("  ACCOUNT means top up; KEY means replace the key; unknown means")
        print("  read the detail column above.")

    # Rate-limit and quota headers, when the gateway sends them.
    interesting = ("ratelimit", "quota", "balance", "credit", "x-")
    for model, result in results:
        for name, value in (result.get("headers") or {}).items():
            if any(t in name.lower() for t in interesting):
                print(f"  header[{model}] {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
