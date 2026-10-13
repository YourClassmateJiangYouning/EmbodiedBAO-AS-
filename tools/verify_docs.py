"""Verify every checkable claim in REMOTE_WORKORDER.md and REMOTE_KICKOFF.md.

Written because these two documents are instructions to another machine, and an instruction that
contains a wrong number is worse than no instruction: the remote session will trust it.  One claim in
the kickoff was in fact wrong when first written ("assets/isaac is gitignored and never committed"),
and this script exists so the rest are checked rather than asserted.

Run from the repository root:  python tools/verify_docs.py
"""
from __future__ import annotations

import glob
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ("IMPLEMENTATION_PROTOCOL.md", "REMOTE_WORKORDER.md", "REMOTE_KICKOFF.md")


def tracked(pattern: str) -> int:
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True)
    return len([line for line in out.stdout.splitlines() if line.strip()])


def main() -> int:
    problems = []
    for name in DOCS:
        if not os.path.exists(os.path.join(ROOT, name)):
            problems.append(f"{name} does not exist")
    if problems:
        print("\n".join(problems))
        return 1

    text = "".join(open(os.path.join(ROOT, name), encoding="utf-8").read() for name in DOCS)

    isaac, bucket = tracked("assets/isaac"), tracked("assets/bucket")
    total = tracked("assets")
    facts = [
        (total == 1192, f"tracked assets is {total}, docs say updated",
         f"tracked assets = {total}"),
        (isaac == 1090, f"assets/isaac is {isaac}, docs say updated",
         f"assets/isaac = {isaac}"),
        (bucket == 102, f"assets/bucket is {bucket}, docs say updated",
         f"assets/bucket = {bucket}"),
        (len(glob.glob(os.path.join(ROOT, "test_bao_*.py"))) == 9,
         f"test file count is {len(glob.glob(os.path.join(ROOT, 'test_bao_*.py')))}, docs say updated",
         "9 test files"),
        ("173" in text, "docs no longer state the check total 173", "docs state the total 173"),
        (os.path.exists(os.path.join(ROOT, "tools", "audit_dressing.py")),
         "tools/audit_dressing.py is referenced but missing", "audit_dressing.py present"),
        (os.path.exists(os.path.join(ROOT, "tools", "prove_audit_fails.py")),
         "tools/prove_audit_fails.py is referenced but missing", "prove_audit_fails.py present"),
        (os.path.exists(os.path.join(ROOT, "tools", "measure_named_props.py")),
         "tools/measure_named_props.py is referenced but missing", "measure_named_props.py present"),
        (os.path.exists(os.path.join(ROOT, "tools", "fetch_bucket_set.py")),
         "tools/fetch_bucket_set.py is referenced but missing", "fetch_bucket_set.py present"),
        # The protocol is the document the human asked for: every instruction and caution they gave,
        # in one place.  If it loses the directives they actually stated, the next session cannot
        # follow them, so the presence of each is checked rather than assumed.  These are their words'
        # substance, not a style preference.
        ("主人" in text, "the protocol no longer tells the session how to address the human",
         "addresses the human as instructed"),
        ("xyz轴" in text or "xyz" in text,
         "the protocol no longer warns that the XYZ axes must not be confused",
         "carries the axis warning"),
        ("穿模" in text, "the protocol no longer forbids clipping",
         "forbids clipping"),
        ("不要指使我做事" in text or "先把能做的做完" in text,
         "the protocol no longer records that the human must not be given work the session can do",
         "records the do-it-yourself-first rule"),
        ("全部步骤" in text, "the protocol no longer requires every step to be listed each time",
         "requires all steps to be listed"),
        # Checking for the CLAIM, not for the words.  An earlier version searched for a Chinese phrase
        # that also appears in the sentence CORRECTING the claim, so the correction tripped the check
        # on its own prose -- the seventh time a text-scanning check in this project has done that.
        ("assets/isaac is gitignored" not in text and "assets/isaac/` 被" not in text,
         "a document still claims assets/isaac is excluded from the repository, which is false",
         "no document claims assets/isaac is excluded"),
    ]
    # Each entry carries its own success wording, because printing the FAILURE text next to "OK " reads
    # as though the check failed -- which is exactly what the first version of this script did.
    for ok, failure, success in facts:
        print(f"  {'OK ' if ok else 'BAD'} {success if ok else failure}")
        if not ok:
            problems.append(failure)

    # Every backticked path with a slash must exist.
    referenced = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_/]*\.(?:py|md|txt|usd))`", text))
    for path in sorted(referenced):
        if "/" in path and not os.path.exists(os.path.join(ROOT, path)):
            problems.append(f"documented path does not exist: {path}")
    print(f"  checked {len(referenced)} backticked path(s)")

    if problems:
        print(f"\n{len(problems)} problem(s) in the handoff documents:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nthe handoff documents' checkable claims all hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
