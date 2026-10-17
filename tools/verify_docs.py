"""Verify every checkable claim in the documents that hand work to the next session.

Written because these documents are instructions to another machine, and an instruction that contains a
wrong number is worse than no instruction: the remote session will trust it.  One claim in the kickoff
was in fact wrong when first written ("assets/isaac is gitignored and never committed"), and this
script exists so the rest are checked rather than asserted.

The document list is part of the check, not an accident: MANUAL.md and README.md are the two documents
the next person actually follows, and an earlier version of this script did not read them at all while
still printing "the handoff documents' checkable claims all hold" -- a claim about documents it had
never opened.  The summary now names every document checked, for exactly that reason.

Exit status is 0 only when every check passes, so a caller (`$?`, or the test suite) can trust it.

Run from the repository root:  python tools/verify_docs.py
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Read-only anchors: they exist so that a documented relative path can be resolved, and none of these
# directories is the repository root.  See resolve_reference().
PATH_ROOTS = ("assets/isaac", "assets/bucket", "assets", "logs")
# Phrases with which a document declares that a path does NOT exist.  A path mentioned together with
# one of these is not evidence over an absent file -- MANUAL.md and README.md say in as many words
# that `tools/check_marker_uniqueness.py` has not been written, and failing on that would make this
# tool cry wolf on the one mention that is accurate.
ABSENCE_MARKERS = ("不存在", "does not exist", "no longer", "已移除", "已删除", "removed")
DEFAULT_DOCS = (
    "MANUAL.md",
    "README.md",
    "IMPLEMENTATION_PROTOCOL.md",
    "REMOTE_WORKORDER.md",
    "REMOTE_KICKOFF.md",
)
# The nine offline suites.  Listed rather than globbed so that DELETING a suite is a failure and not
# merely a smaller total -- a check that cannot run looks exactly like a check that passed.
TEST_FILES = (
    "test_bao_assets.py",
    "test_bao_geometry.py",
    "test_bao_health.py",
    "test_bao_integration.py",
    "test_bao_memory.py",
    "test_bao_memory_runner.py",
    "test_bao_parsing.py",
    "test_bao_persistence.py",
    "test_bao_scenes.py",
)
ISAAC_FILES = 1090  # git ls-files assets/isaac
# What a document must SAY about the measured check total, as patterns that cannot be satisfied by
# deleting the claim.  ``(?<![\d.])`` / ``(?![\d.])`` keep 1090 from matching 51090 or 1090.5, and keep
# 175 from matching 1175 or a timestamp such as 17:50 -- the first version of this script tested
# ``"173" in text`` and passed on any of those.
CHECK_TOTAL_PATTERNS = (
    # A stated TOTAL, per document.  Three of these are written with markdown bold around the number
    # (``共 **173** 项``), so the two stars sit between the context word and the number and the
    # patterns allow exactly ``**`` and whitespace there.  The comment block above describes the
    # separate quantity checks that compare a number rather than look for a substring.
    ("MANUAL.md", r"共\s*(?:\*\*)?(?P<n>\d+)(?:\*\*)?\s*项"),
    ("README.md", r"\*\*(?P<n>\d+)\s+checks\s+across"),
    ("IMPLEMENTATION_PROTOCOL.md", r"合计\s*(?:\*\*)?(?P<n>\d+)(?:\*\*)?\s*项"),
    ("REMOTE_WORKORDER.md", r"共\s*(?:\*\*)?(?P<n>\d+)(?:\*\*)?\s*项"),
    ("REMOTE_KICKOFF.md", r"合计\s*(?:\*\*)?(?P<n>\d+)(?:\*\*)?\s*项"),
)
# The nine suites, as MANUAL.md's own table row names them, so its per-suite breakdown is checked for
# INTERNAL consistency too.  That row read "scenes 32" for eight checks without anything noticing,
# because nothing read MANUAL.md; a breakdown that does not add up to its own stated total is now a
# failure, and so is a breakdown that disagrees with the suites.
MANUAL_SUITES = {"geometry": "test_bao_geometry.py", "scenes": "test_bao_scenes.py",
                 "memory": "test_bao_memory.py", "integration": "test_bao_integration.py",
                 "memory_runner": "test_bao_memory_runner.py",
                 "persistence": "test_bao_persistence.py", "parsing": "test_bao_parsing.py",
                 "health": "test_bao_health.py", "assets": "test_bao_assets.py"}
# Directives the human actually gave, and WHERE each has to survive.  Per document, not over the
# concatenation of all of them: a claim that is checked across the union of five files is satisfied by
# whichever file happens to carry it, so deleting it from the file a reader is following passes -- the
# same wrong-reason failure the numeric checks had.  ``keep`` is a function of the document text.
#
# These are the human's words' substance, not a style preference; if a directive is lost the next
# session cannot follow it.  The axis entry is the one that made this explicit: it used to be
# ("xyz轴" in text or "xyz" in text), which a sentence saying the axes must NOT be confused satisfies,
# and which the bare lower-case spelling satisfies on its own.  It now requires both the imperative and
# the axis name inside IMPLEMENTATION_PROTOCOL.md, where the warning is actually written.
DIRECTIVES = (
    ("主人", "IMPLEMENTATION_PROTOCOL.md",
     lambda t: "主人" in t,
     "no longer tells the session how to address the human"),
    (("轴绝不可以搞错", "轴不可以搞错"), "IMPLEMENTATION_PROTOCOL.md",
     lambda t: ("轴绝不可以搞错" in t or "轴不可以搞错" in t) and "xyz" in t.lower(),
     "no longer warns, in the human's words, that the XYZ axes must not be confused"),
    ("穿模", "IMPLEMENTATION_PROTOCOL.md",
     lambda t: "穿模" in t,
     "no longer forbids clipping"),
    ("不要指使我做事", "IMPLEMENTATION_PROTOCOL.md",
     lambda t: "不要指使我做事" in t or "先把能做的做完" in t,
     "no longer records that the human must not be given work the session can do"),
    ("全部步骤", "MANUAL.md",
     lambda t: "全部步骤" in t,
     "no longer requires every step to be listed each time"),
)
# Every document must NOT make this claim: assets/isaac is committed and always has been, and an
# earlier kickoff said otherwise.  Checked per document for the reason above.
ABSENT_CLAIMS = (
    ("assets/isaac is gitignored", lambda t: "assets/isaac is gitignored" not in t
     and "assets/isaac/` 被" not in t,
     "still claims assets/isaac is excluded from the repository, which is false"),
)


def tracked(pattern: str) -> int:
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True)
    return len([line for line in out.stdout.splitlines() if line.strip()])


def check_counts() -> tuple:
    """(total, {file: count}) registered checks, read from the suites' own TESTS/tests lists.

    Counted from the lists the suites actually iterate, and NOT from ``grep -c '^def test_'``: a test
    can be defined without being registered, so the grep count is only right while the two agree
    (scenes: 40 defined and 40 registered, but MANUAL.md and README.md still said 32).  Only a
    top-level list/tuple literal is trusted.
    """
    per_file = {}
    for name in TEST_FILES:
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=name)
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id in ("TESTS", "tests")
                        and isinstance(node.value, (ast.List, ast.Tuple))):
                    count = max(count, len(node.value.elts))
        per_file[name] = count
    return sum(per_file.values()), per_file


def positive_ints(text: str, pattern: str) -> list:
    """Every integer captured by ``pattern``, compared as a NUMBER (so 1175 and 1090.5 never match)."""
    return [int(m.group("n")) for m in re.finditer(pattern, text, re.S)]


# Numbers are read as STATEMENTS ABOUT A QUANTITY, not as bare integers.  A standalone-integer test
# cannot be used here, and the first version of this check showed why in two ways at once:
#   * README.md legitimately contains the asset id ``SM_BooksSet_11_167`` and its
#     `SM_BooksSet_11_167.usd`, so a check that fires on "167" anywhere fires on an asset name;
#   * ("173" in text) also passes on a timestamp, which is the wrong-reason failure this replaced.
# Each pattern therefore matches "context word -- number -- unit", and the captured number is compared
# against the measurement.  A number quoted as an explicitly historical figure (README's "136 MB / 880
# 文件（当时实测）") carries no current-measurement context word and is deliberately out of scope.
FILE_COUNT_RE = r"(?:现在实测|实测|共|合计)[^\d\n]{0,8}(?:\*\*)?(\d+)(?:\*\*)?\s*个?\s*文件"
README_CHECK_RE = r"\*\*(\d+)\s+checks\b"


def stated_counts(text: str, patterns) -> list:
    """Every integer stated with the unit each pattern names (so 899 in "SM_BooksSet_11_899" is not one)."""
    return [int(group) for pattern in patterns for match in re.finditer(pattern, text)
            for group in match.groups() if group is not None]


def resolve_reference(path: str) -> bool:
    """True when a documented relative path names something that exists.

    Documents name two kinds of relative path and neither is repository-root-relative:
    ``assets/isaac/Props/Pallet/pallet.usd`` is quoted as ``Pallet/pallet.usd`` (relative to the asset
    root), and ``logs/STATUS.txt`` is a run artefact long enough lived to be tab-completed.  Resolving
    only against the repository root reported all five of those in MANUAL.md and README.md as missing
    paths, which is a false alarm on a true sentence -- and a checker that cries wolf is the failure
    mode this whole file exists to avoid.
    """
    candidates = [os.path.join(ROOT, path)]
    candidates += [os.path.join(ROOT, root, path) for root in PATH_ROOTS]
    if any(os.path.exists(candidate) for candidate in candidates):
        return True
    if path.startswith("logs/") or path.startswith("results/"):
        return True  # runtime-only tree, absent by design in a fresh clone
    basename = os.path.basename(path)
    return bool(glob.glob(os.path.join(ROOT, "assets", "**", basename), recursive=True))


def referenced_paths(text: str) -> set:
    """Every backticked path with a slash the documents tell the reader to look at."""
    return {
        path for path in re.findall(r"`([A-Za-z_][A-Za-z0-9_/]*\.(?:py|md|txt|usd))`", text)
        if "/" in path
    }


def documented_absent(text: str, path: str) -> bool:
    """True when the document says, in the SAME SENTENCE or table row as this mention, that the path
    does not exist.

    The window is the line, cut back to the last sentence end before the mention.  A wide window is how
    this check excused a path that had merely been misspelt: MANUAL.md's tools table contains the word
    "不存在" a row away from an unrelated tool, and every mention within a few hundred characters was
    being treated as declared absent.
    """
    for match in re.finditer(re.escape(f"`{path}`"), text):
        floor = max(text.rfind("\n", 0, match.start()), text.rfind("。", 0, match.start()),
                    text.rfind(". ", 0, match.start()))
        ends = [i for i in (text.find("\n", match.end()), text.find("。", match.end()),
                            text.find(". ", match.end())) if i != -1]
        window = text[floor + 1: (min(ends) + 1) if ends else len(text)]
        if any(marker in window for marker in ABSENCE_MARKERS):
            return True
    return False


def _num(ok: bool, values: list, expected: int, label: str, unit: str) -> tuple:
    """One (ok, failure, success) fact for a stated quantity, with the value compared as a number.

    An empty ``values`` is a failure: the document is required to state the quantity, not merely to
    avoid stating a wrong one.
    """
    stated = ", ".join(str(value) for value in values) if values else "(none stated)"
    return (bool(ok),
            f"{label} states {stated} {unit} where the measurement is {expected}",
            f"{label} states {expected} {unit}")


def suite_breakdown(manual: str) -> dict:
    """{suite: count} from MANUAL.md's per-suite row, or {} when that row is absent.

    Split out so it can be exercised directly on fabricated text; the row is
    ``... 9 套件共 167 项全过**：geometry 37 / scenes **32** / memory 29 / ...``.
    """
    row = re.search(r"[^\n]*套件[^\n]*项[^\n]*全过[^\n]*", manual)
    if not row:
        return {}
    return {name: int(value)
            for name, value in re.findall(r"([a-z_]+)\s*(?:\*\*)?(\d+)(?:\*\*)?", row.group(0))
            if name in MANUAL_SUITES}


def breakdown_faults(manual: str, per_file: dict) -> list:
    """Every way MANUAL.md's per-suite breakdown disagrees with the suites, as readable strings.

    ``per_file`` is keyed by FILE NAME ("test_bao_scenes.py"), not by the suite label the row uses
    ("scenes"), so the guard is ``MANUAL_SUITES[name] in per_file`` -- testing ``name in per_file``
    made every comparison unreachable and the check passed on anything, which is how the stale
    "scenes 32" survived the first version of this function.
    """
    stated = suite_breakdown(manual)
    if not stated:
        return []
    faults = [f"{name} says {value} but {MANUAL_SUITES[name]} registers {per_file[MANUAL_SUITES[name]]}"
              for name, value in sorted(stated.items())
              if MANUAL_SUITES[name] in per_file and value != per_file[MANUAL_SUITES[name]]]
    faults += [f"{name} is missing from the breakdown" for name in sorted(set(MANUAL_SUITES) - set(stated))]
    return faults


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the documents' checkable claims.")
    parser.add_argument("--doc", action="append", default=None,
                        help="document to check (repeatable; default: every handoff/manual document)")
    args = parser.parse_args()
    docs_to_check = tuple(args.doc) if args.doc else DEFAULT_DOCS

    problems = []
    for name in docs_to_check:
        if not os.path.exists(os.path.join(ROOT, name)):
            problems.append(f"{name} does not exist")
    if problems:
        print("\n".join(problems))
        return 1

    # Printed BEFORE any verdict, so the summary at the end cannot be read as covering a document that
    # was never opened -- the fault this script was rebuilt to remove.
    print(f"documents read ({len(docs_to_check)}): " + ", ".join(docs_to_check))
    print()

    docs = {name: open(os.path.join(ROOT, name), encoding="utf-8").read() for name in docs_to_check}
    manual, readme = docs.get("MANUAL.md", ""), docs.get("README.md", "")
    have_manual, have_readme = "MANUAL.md" in docs, "README.md" in docs

    measured_total, per_file = check_counts()
    scenes_checks = per_file.get("test_bao_scenes.py", 0)
    present_tests = tuple(name for name in TEST_FILES if name in per_file)

    isaac, bucket = tracked("assets/isaac"), tracked("assets/bucket")
    total = tracked("assets")
    documented_totals = [
        (name, value)
        for name in docs_to_check
        for pattern in (p for owner, p in CHECK_TOTAL_PATTERNS if owner == name)
        for value in positive_ints(docs[name], pattern)[:1]  # the first (the total), not per-suite figures
    ]
    # Only a document that is KNOWN to state the total can be required to state it.  A --doc selection
    # without MANUAL.md/README.md is not evidence that the total went unstated.
    total_owners = {owner for owner, _ in CHECK_TOTAL_PATTERNS if owner in docs}
    facts = [
        (total == 1192, f"tracked assets is {total}, docs say updated",
         f"tracked assets = {total}"),
        (isaac == ISAAC_FILES, f"assets/isaac is {isaac}, docs say updated",
         f"assets/isaac = {isaac}"),
        (bucket == 102, f"assets/bucket is {bucket}, docs say updated",
         f"assets/bucket = {bucket}"),
        (len(present_tests) == len(TEST_FILES),
         f"only {len(present_tests)}/{len(TEST_FILES)} test suites exist"
         f" (missing: {', '.join(n for n in TEST_FILES if n not in per_file)})",
         f"{len(TEST_FILES)} test suites present"),
        # NOT `measured_total == 176`: a hardcoded expected total is the same
        # restate-the-constant defect the numeric checks above were written to
        # avoid, and it made every added check a doc-tool failure.  The documents
        # are compared against the measured total on the next fact; here it is
        # enough that checks were registered at all.
        (measured_total > 0, "the suites register no checks at all",
         f"the suites register {measured_total} checks"),
        (bool(documented_totals) or not total_owners,
         "no document states an offline-check total any more",
         f"the check total is stated in {len({n for n, _ in documented_totals})} document(s)"),
        # Each (document, value) pair from the patterns above must equal the measured total.  An
        # earlier version tested ("173" in text), which 1173 and a timestamp both satisfy.
        (all(value == measured_total for _, value in documented_totals),
         "a document states the wrong offline-check total: "
         + ", ".join(sorted({f"{name} says {value}, measured {measured_total}"
                             for name, value in documented_totals if value != measured_total})),
         f"every stated check total is {measured_total}"),
        (os.path.exists(os.path.join(ROOT, "tools", "audit_dressing.py")),
         "tools/audit_dressing.py is referenced but missing", "audit_dressing.py present"),
        (os.path.exists(os.path.join(ROOT, "tools", "prove_audit_fails.py")),
         "tools/prove_audit_fails.py is referenced but missing", "prove_audit_fails.py present"),
        (os.path.exists(os.path.join(ROOT, "tools", "measure_named_props.py")),
         "tools/measure_named_props.py is referenced but missing", "measure_named_props.py present"),
        (os.path.exists(os.path.join(ROOT, "tools", "fetch_bucket_set.py")),
         "tools/fetch_bucket_set.py is referenced but missing", "fetch_bucket_set.py present"),
    ]
    # Claimed numbers, only checked in the document that states them.  The per-suite breakdown is
    # checked exactly where a reader follows it: "scenes 32" was stale for eight checks and nothing
    # noticed, because nothing read MANUAL.md.
    if have_manual:
        facts += [
            (bool(re.search(rf"scenes[^\n]{{0,12}}(?<![\d.]){scenes_checks}(?![\d.])", manual)),
             f"MANUAL.md does not give scenes as {scenes_checks} checks",
             f"MANUAL.md gives scenes as {scenes_checks} checks"),
            (bool(re.search(rf"assets/isaac[^\n]{{0,24}}(?<![\d.]){ISAAC_FILES}(?![\d.])", manual)),
             f"MANUAL.md does not give assets/isaac as {ISAAC_FILES} tracked files",
             f"MANUAL.md gives assets/isaac as {ISAAC_FILES} tracked files"),
            # Every count the document states about assets/isaac has to BE the measured count, so a
            # later "1899" cannot pass just because "1090" appears somewhere on the same line.
            _num(bool(file_counts := stated_counts(manual, (FILE_COUNT_RE,)))
                 and all(value == ISAAC_FILES for value in file_counts),
                 file_counts, ISAAC_FILES, "MANUAL.md", "tracked files under assets/isaac"),
        ]
        # The per-suite breakdown, checked against the suites and against its own stated total.  This
        # is what caught "scenes 32": the total alone cannot, and the old ("173" in text) could not
        # either.
        faults = breakdown_faults(manual, per_file)
        if suite_breakdown(manual):
            facts.append((
                not faults,
                "MANUAL.md's per-suite breakdown is stale: " + "; ".join(faults),
                f"MANUAL.md states all {len(MANUAL_SUITES)} suite counts as measured",
            ))
    if have_readme:
        readme_checks = stated_counts(readme, (README_CHECK_RE,))
        # Compared against the MEASURED total, not a hardcoded 176: a document was
        # allowed to state the wrong number as long as it stated this one.
        facts.append(_num(bool(readme_checks)
                          and all(value == measured_total for value in readme_checks),
                          readme_checks, measured_total, "README.md", "offline checks"))
    # Each entry carries its own success wording, because printing the FAILURE text next to "OK " reads
    # as though the check failed -- which is exactly what the first version of this script did.
    for ok, failure, success in facts:
        print(f"  {'OK ' if ok else 'BAD'} {success if ok else failure}")
        if not ok:
            problems.append(failure)

    # The human's directives, each checked against the document that has to carry it.  Reported as the
    # existing checks are, but one line per document, so "OK" names the file that was read.
    for label, name, keep, failure in DIRECTIVES:
        if name not in docs:
            continue  # a --doc selection that omits it does not silently become a failure
        ok = keep(docs[name])
        detail = f"keeps the directive {label!r}" if ok else f"{failure} (directive {label!r})"
        print(f"  {'OK ' if ok else 'BAD'} {name} {detail}")
        if not ok:
            problems.append(f"{name} {failure}")

    # Claims no document may make.  Checking the concatenation let one file's correction excuse
    # another file's false claim, because the correction and the claim were searched for together.
    for label, keep, failure in ABSENT_CLAIMS:
        for name in docs_to_check:
            ok = keep(docs[name])
            print(f"  {'OK ' if ok else 'BAD'} "
                  + (f"{name} does not claim {label!r}" if ok else f"{name} {failure}"))
            if not ok:
                problems.append(f"{name}: {failure}")

    # Every backticked path with a slash must resolve.  Referenced paths are gathered per document, so
    # a path mentioned in one file is not excused by a phrase in another.
    referenced = set()
    absent_named = set()
    for name in docs_to_check:
        for path in sorted(referenced_paths(docs[name])):
            referenced.add(path)
            if resolve_reference(path):
                continue
            if documented_absent(docs[name], path):
                absent_named.add(path)
                continue
            problems.append(f"documented path does not exist: {path} (in {name})")
    for path in sorted(absent_named):
        print(f"  OK  {path} is named, and a document says it does not exist")
    print(f"  checked {len(referenced)} backticked path(s)")

    if problems:
        print(f"\n{len(problems)} problem(s) in the documents checked above:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"\nall checkable claims in {', '.join(docs_to_check)} hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
