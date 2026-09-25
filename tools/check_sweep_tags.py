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


def shell_composes_the_tag_itself(source: str) -> bool:
    """True if run_all_models.sh still builds a tag instead of asking main.

    That re-implementation is the bug this file exists for, and it has happened
    twice: once rewriting the model name in shell (`tr '/.' '--'`, which turned
    dots into dashes) and once appending the protocol tag here while main appended
    it again.  Both times the printed tag stopped being the directory actually
    written.  So the guard is now on the SOURCE: the shell must delegate.
    """
    marker = "model_tag() {"
    start = source.find(marker)
    if start < 0:
        return False
    body = source[start : source.find("}", start)]
    delegates = "from main import effective_tag" in body and "effective_tag(" in body
    builds_its_own = "sanitize_tag" in body or "PROTOCOL_TAG" in body
    return not delegates or builds_its_own


def main() -> int:
    import main as entry
    from protocol import PROTOCOL_TAG

    roster = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))
    models = [entry_["runner"] for entry_ in roster["models"]]
    # Names that exercise the characters the two implementations disagreed on.
    models += ["gemini-2.5-pro", "deepseek-v4.1-flash", "a/b", "x:y", "", ".."]

    # 1. The shell must not compose a tag of its own: ask main for the one it will
    #    use, so the printed tag IS the directory.  Emulating the shell in Python
    #    (as this file used to) cannot catch a divergence in the shell, because the
    #    emulation is written by hand and drifts.
    with open(os.path.join(ROOT, "run_all_models.sh"), encoding="utf-8") as handle:
        shell_source = handle.read()
    problems = 0
    if shell_composes_the_tag_itself(shell_source):
        problems += 1
        print(
            "run_all_models.sh builds its own tag instead of calling "
            "main.effective_tag; the printed tag can then disagree with the "
            "results directory"
        )
    else:
        print("run_all_models.sh delegates its tag to main.effective_tag")

    # 2. Whatever the shell prints must survive being fed back through --tag, which
    #    is exactly what the sweep does, and must be unique per model.
    print()
    print(f"protocol tag: {PROTOCOL_TAG}")
    print("%-36s %-46s %s" % ("model", "tag from effective_tag", "fed back in"))
    print("-" * 112)
    mismatches = 0
    seen: dict = {}
    collisions = []
    for model in models:
        got = entry.effective_tag(model)
        want = entry.effective_tag(model, got)
        flag = ""
        if got != want:
            mismatches += 1
            flag = "   <-- MISMATCH"
        if got in seen and seen[got] != model:
            collisions.append((got, seen[got], model))
        seen[got] = model
        print("%-36s %-46s %s%s" % (model, got, want, flag))

    print()
    print(f"mismatches: {mismatches}, shell problems: {problems}")
    if collisions:
        print("tag collisions (two models would share a results directory):")
        for tag, first, second in collisions:
            print(f"  {tag!r} used by both {first!r} and {second!r}")
    else:
        print("tag collisions: none")
    return 1 if (mismatches or problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
