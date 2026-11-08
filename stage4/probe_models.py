"""Probe each model once with the restored Stage-1 envelope, using a real bank frame.

The envelope changed (system message added, response_format added, max_tokens removed), so before committing
an hour of GPU-free but wall-clock-expensive episodes, check that every model still answers with a legal
action and that the request really carried the system prompt.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import core  # noqa: E402
import protocol  # noqa: E402

BANK = r"D:\stage4_photos"
MODELS = [
    "deepseek-v4-flash-vision-exp",
    "gemini-3-flash-preview",
    "mimo-v2.6-pro",
    "claude-haiku-4-5-20251001",
    "gpt-5",
]
FRAME = os.path.join(BANK, "1.2红色", "1.PNG")

print("frame:", FRAME, "exists:", os.path.isfile(FRAME))
print("system prompt:", protocol.SYSTEM_PROMPT[:60], "...")
print()

results = []
for model in MODELS:
    session = core.Stage4Session(sessions_root=os.path.join(HERE, "sessions_probe"))
    session.open_session("probe", {
        "scene": "R1", "marker": "M2", "marker_descriptor": "red circle", "mode": "A",
        "model": model, "a_s": 1.2, "max_steps": 30,
    })
    session.start_episode()
    session.capture_from_bank(BANK)
    started = __import__("time").time()
    try:
        outcome = session.call_model()
    except Exception as error:  # noqa: BLE001
        print("%-32s EXCEPTION %s: %s" % (model, type(error).__name__, error))
        results.append((model, None, 0, str(error)))
        continue
    elapsed = __import__("time").time() - started
    record = session.history[-1]
    attempts = (session.last_call or {}).get("attempts") or []
    first = attempts[0] if attempts else {}
    print("%-32s action=%-10s %4.1fs  json_mode=%s finish=%s  att=%d  err=%s" % (
        model, record.get("action"), elapsed, first.get("json_mode"),
        first.get("finish_reason"), len(attempts), first.get("error")))
    results.append((model, record.get("action"), elapsed, first.get("error")))

print()
bad = [r for r in results if not r[1]]
print("models with a legal action:", len(results) - len(bad), "/", len(results))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROBE PASSED")
