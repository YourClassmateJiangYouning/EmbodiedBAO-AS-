"""Run the three-gate check over a list of models and summarise the verdicts.

Calling tools/check_model.py once per model is fine for one model; before a
sweep it is worth checking the whole batch, so this loops and collects a single
table with the per-gate verdict and the reason for any failure.

Gate 2 (vision) matters most here: several ids on this gateway answer text but
cannot see, and a model that cannot see produces a whole Level of failures that
looks like a capability result.  deepseek-v4-pro answered "Unknown." and
deepseek-v4-flash answered "Orange" to a solid red frame.

    python tools/check_batch.py qwen3-vl-32b-instruct glm-4.6v gemini-2.5-flash
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODELS = [
    "qwen3-vl-32b-instruct",
    "glm-4.6v",
    "gemini-2.5-flash",
    "gpt-4o-mini",
    "gpt-4o",
]


def gates_from_report(out: str) -> dict:
    """Parse check_model.py's own failure list, which names the failing gate."""
    failed = set(
        int(m.group(1))
        for m in re.finditer(r"-\s*gate\s*(\d)\s*:", out)
    )
    # A gate that is not listed as failing passed.  The report always covers
    # gates 1-3, so anything absent from the failure list is OK.
    return {n: ("FAIL" if n in failed else "OK") for n in (1, 2, 3)}


def main() -> int:
    models = sys.argv[1:] or DEFAULT_MODELS
    script = os.path.join(ROOT, "tools", "check_model.py")

    print(f"checking {len(models)} model(s); each runs three gates")
    print("  gate 1 reachable   gate 2 vision   gate 3 JSON action")
    print()
    print(f"{'model':<34} {'gate1':<7} {'gate2':<7} {'gate3':<7} verdict")
    print("-" * 88)

    not_ready = []
    for model in models:
        proc = subprocess.run(
            [sys.executable, script, "--model", model],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        out = proc.stdout
        gates = gates_from_report(out)
        ready = proc.returncode == 0 and "READY -" in out
        if not ready:
            not_ready.append(model)
        print(
            f"{model:<34} {gates[1]:<7} {gates[2]:<7} {gates[3]:<7} "
            f"{'READY' if ready else 'NOT READY'}"
        )
        # Surface why, when the model is not ready.
        if not ready:
            for line in out.splitlines():
                if line.strip().startswith("- gate"):
                    print(f"    {line.strip()}")

    print()
    if not_ready:
        print("not ready: " + ", ".join(not_ready))
        print(f"  python tools/check_model.py --model {not_ready[0]}")
    else:
        print("all models passed all three gates")
    return 1 if not_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
