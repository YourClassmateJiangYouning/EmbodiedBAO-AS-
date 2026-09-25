"""Verify run_all_models.sh tags each model exactly as main.py will.

The sweep script must produce the same tag that `main.py --tag X` would, because
the tag is the directory key for both the results tree and the resume
checkpoint.  The shell used to rewrite the model name itself (`tr '/.' '--'`),
which turned dots into dashes, so `gemini-2.5-pro` became `gemini-2-5-pro` in a
sweep but stayed `gemini-2.5-pro` in a manual run -- two directories for one
model, and --resume unable to find the other's checkpoint.

The tag also carries the protocol version, so a run under a changed prompt or
action semantics cannot resume onto the previous protocol's episodes.  This
script therefore compares the shell's tag against main.effective_tag, i.e. the
value the runner actually uses, rather than against sanitize_tag alone.

    python tools/check_sweep_tags.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    import main as entry
    from protocol import PROTOCOL_TAG

    roster = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))
    models = [entry_["runner"] for entry_ in roster["models"]]
    # Names that exercise the characters the two implementations disagreed on.
    models += ["gemini-2.5-pro", "deepseek-v4.1-flash", "a/b", "x:y", "", ".."]

    script = (
        "import sys\n"
        'sys.path.insert(0, r"%s")\n'
        "from protocol import PROTOCOL_TAG\n"
        "from persistence import sanitize_tag\n"
        'print(f"{sanitize_tag(sys.argv[1])}-{PROTOCOL_TAG}")\n' % ROOT
    )

    print(f"protocol tag: {PROTOCOL_TAG}")
    print("%-36s %-46s %s" % ("model", "sweep tag", "main.effective_tag"))
    print("-" * 112)
    mismatches = 0
    seen: dict = {}
    collisions = []
    for model in models:
        got = subprocess.run(
            [sys.executable, "-c", script, model],
            capture_output=True,
            text=True,
        ).stdout.strip()
        want = entry.effective_tag(model)
        flag = ""
        if got != want:
            mismatches += 1
            flag = "   <-- MISMATCH"
        if got in seen and seen[got] != model:
            collisions.append((got, seen[got], model))
        seen[got] = model
        print("%-36s %-46s %s%s" % (model, got, want, flag))

    print()
    print(f"mismatches: {mismatches}")
    if collisions:
        print("tag collisions (two models would share a results directory):")
        for tag, first, second in collisions:
            print(f"  {tag!r} used by both {first!r} and {second!r}")
    else:
        print("tag collisions: none")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
