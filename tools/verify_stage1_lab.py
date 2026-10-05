"""Decide whether a stage 1 laboratory run is sound enough to scale up, without a human.

Written because the sweep is a day long and the decision to launch it has to be made by
something that is awake.  It reads the run's own log and its result directories and prints one
verdict per check, each with the evidence it used, so that a failure at three in the morning is
readable at nine.

Deliberately tolerant about the result files: an unrecognised structure is reported as SKIP
rather than crashing, because a verifier that dies is worse than one that admits ignorance.
The log checks are the load-bearing ones -- they are what say whether the prompts named the
right marker and whether the gateway answered at all.

Run:  python3 tools/verify_stage1_lab.py --log logs/smoke.log --model gpt-4o-mini
Exit code 0 means every check passed or was skipped; 1 means at least one failed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

QUOTA_PATTERNS = ("quota", "insufficient", "exceeded your current quota", "429", "402",
                  "unauthorized", "401", "invalid api key", "credit")
EXPECTED_LEVELS = 17
MARKERS_PER_LEVEL = 5


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str, skipped: bool = False) -> None:
        if skipped:
            print(f"[SKIP] {name}: {detail}")
        elif ok:
            print(f"[PASS] {name}: {detail}")
        else:
            self.failures += 1
            print(f"[FAIL] {name}: {detail}")

    def verdict(self) -> int:
        if self.failures:
            print(f"\nVERDICT: {self.failures} check(s) failed -- do NOT start the full sweep.")
            return 1
        print("\nVERDICT: sound.  The full sweep may start.")
        return 0


def read_log(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def check_log(report: Report, text: str, model: str) -> None:
    if not text:
        report.check("log present", False, "no log file to read")
        return
    print(f"--- log checks ({len(text.splitlines())} lines) ---")

    slots = [int(m) for m in re.findall(r"marker slot (\d+):", text)]
    report.check("marker slots cycle", len(set(slots)) == MARKERS_PER_LEVEL and slots,
                 f"distinct slots seen: {sorted(set(slots))} "
                 f"(expected 1..{MARKERS_PER_LEVEL}); {len(slots)} swap(s)")

    named = re.findall(r'prompt now says "([^"]+)"', text)
    colours = {name.split()[0] for name in named}
    report.check("prompts name the marker", len(named) > 0 and len(colours) > 1,
                 f"{len(named)} naming(s), {len(colours)} distinct colour word(s): "
                 f"{sorted(colours)}")
    report.check("no stale red marker in prompts", "red marker" not in text,
                 "the frozen phrase 'red marker' does not survive into a scene's prompts"
                 if "red marker" not in text else "found the old phrase, so naming did not take")

    hits = [p for p in QUOTA_PATTERNS if p in text.lower()]
    report.check("gateway answered", not hits, f"quota/auth markers found: {hits}" if hits
                 else "no quota, 401/402/429 or credit markers in the log")

    bad = len(re.findall(r'"?invalid_response_count"?\s*[:=]\s*(?!0\b)\d+', text))
    report.check("no invalid responses", bad == 0,
                 f"{bad} episode(s) reported a non-zero invalid_response_count" if bad
                 else "every episode reported invalid_response_count 0")

    crashes = len(re.findall(r"Traceback \(most recent call last\)", text))
    report.check("no tracebacks", crashes == 0, f"{crashes} traceback(s) in the log")

    passes = len(re.findall(r'"passed"\s*:\s*true', text)) + text.count("passed in ")
    fails = len(re.findall(r'"passed"\s*:\s*false', text)) + text.count("failed after ")
    report.check("outcomes recorded", passes + fails > 0,
                 f"{passes} passed, {fails} failed lines; a run with no outcomes recorded "
                 f"has not started", skipped=(passes + fails == 0))


def check_results(report: Report, results: str, model: str) -> None:
    print("--- result checks ---")
    pattern = os.path.join(results, "level*", model, f"*stage1.1*")
    dirs = sorted(glob.glob(pattern))
    report.check("episode directories exist", bool(dirs),
                 f"{len(dirs)} director(y/ies) matching {pattern}")
    if not dirs:
        return
    files = [f for d in dirs for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)]
    files = [f for f in files if os.path.isfile(f)]
    levels = {re.search(r"level(\d+)", d).group(1) for d in dirs if re.search(r"level(\d+)", d)}
    report.check("all levels present", len(levels) == EXPECTED_LEVELS,
                 f"{len(levels)} level(s) written, expected {EXPECTED_LEVELS}")
    report.check("episodes written", len(files) > 0, f"{len(files)} file(s) under those dirs")

    # If any episode file is JSON and carries a pass/fail flag, use it; otherwise say so.
    seen_pass = seen_fail = 0
    for path in files:
        if not path.endswith(".json"):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                blob = json.load(handle)
        except Exception:  # noqa: BLE001
            continue
        text = json.dumps(blob).lower()
        if '"passed": true' in text or '"success": true' in text:
            seen_pass += 1
        elif '"passed": false' in text or '"success": false' in text:
            seen_fail += 1
    if seen_pass + seen_fail == 0:
        report.check("pass rate sane", True, "no per-episode pass flag found in the JSON; "
                     "not checked", skipped=True)
    else:
        rate = seen_pass / float(seen_pass + seen_fail)
        # The widest aperture in the ladder is twice the shoulder width; a model that cannot
        # pass that is not judging an aperture, it is failing for some other reason.
        report.check("pass rate sane", rate > 0.2,
                     f"{seen_pass} passed / {seen_fail} failed = {rate:.1%}; a smoke run this "
                     f"low means the run is broken, not that the model is poor")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a stage 1 lab run.")
    parser.add_argument("--log", default="")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    report = Report()
    check_log(report, read_log(args.log), args.model)
    check_results(report, args.results, args.model)
    return report.verdict()


if __name__ == "__main__":
    sys.exit(main())
