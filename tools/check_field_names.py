"""Check that every record field the analysis reads actually exists.

The failure mode this exists for: read a field name that was never written, with
``.get()`` or a default, and get a plausible constant instead of an error.

Two of those were live in ``lab_logs/stage1_results.py``:

* ``first_turn_x`` -- the x of a step lives in the per-step sidecar, not in the episode
  record, so "where the first turn happened" reported ``none`` for all 660 episodes;
* ``"strafe_left"``/``"strafe_right"`` -- the protocol calls those actions ``left`` and
  ``right``, so that count was always zero.

Neither raised anything, and neither would have been caught by a test that checks the
numbers the code *does* compute.  A name that does not exist is only visible if
something compares the names against the schema, which is what this does.

Two checks, because the two bugs had different shapes:

* a field name read off a variable that holds a record, against the schema;
* an action name used in a count or a membership test, against ``protocol.ACTIONS``.

The schema is read from the sources of truth rather than copied: the Stage 1 record and
sidecar keys come out of the committed archive, the Stage 2 round-record and per-run
summary keys are parsed out of their functions' return values, and the memory-block
attempt shape is the one documented in ``memory_block``.

Scoped deliberately to variables that hold records.  An earlier version scanned every
``.get()`` in the repository and reported 170 names, nearly all of them HTTP response
fields (``choices``, ``message``) or roster JSON -- a check nobody would keep running.

Run from the repository root:  python tools/check_field_names.py
"""

import ast
import json
import os
import re
import sys
import tarfile

sys.path.insert(0, ".")
import protocol  # noqa: E402

ARCHIVE = "lab_logs/bao_v7_all.tgz"
SCANNED = ["lab_logs", "tools", "."]
SKIP_DIRS = ("_test_tmp", ".git", "__pycache__", "extracted")

# Variables that hold a record, a row, a sidecar step or a saved attempt.  Anything
# else -- task_dict, argparse, an HTTP body, the roster file -- is out of scope.
RECORD_VARS = (
    "record", "records", "episode", "episodes", "sample", "samples", "row", "rows",
    "attempt", "attempts", "run_one", "sidecar", "step", "item", "items", "entry",
    "arg", "summary", "checkpoint",
)
FIELD = re.compile(
    r"\b(" + "|".join(RECORD_VARS) + r")\w*\s*\.\s*get\(\s*[\"']([a-z_][a-z0-9_]*)[\"']"
)
# The second bug's shape: a name that was clearly *meant* to be an action.  A verb stem
# plus a direction word is what keeps this to the strafe_left case: metric names that
# merely start with a stem (turn_pct, look_n, pass_if_look) are not actions and are not
# flagged.
ACTION_SHAPED = re.compile(
    r"[\"']((?:turn|look|strafe|walk|step|move|sidle|go)_"
    r"(?:left|right|down|up|forward|backward|ahead))[\"']"
)

# Fields of other objects those same variables sometimes point at: the task dictionary,
# argparse results, the roster file, a history item, the Stage 1 runner's own
# per-episode summary (a different schema from Stage 2's per-run one), and the local
# annotations the analysis scripts add to a record (all underscore-prefixed).
OTHER = {
    "action_steps", "move_step", "start_x", "max_steps", "headless", "runs",
    "output_root", "save_obs", "resume", "image_size", "reasoning", "episode_id",
    "orientation", "position", "camera_yaw", "camera_pitch", "yaw",
    "curve", "label", "valued_rounds", "rho", "improving", "runs", "learning",
    "probe", "runs", "gap_curve", "rounds", "tag", "timestamp", "completed",
    "round", "passed", "steps", "note", "model", "model_name", "level",
    # history items
    "step", "action", "feedback", "success",
    # models.json
    "runner", "label", "group", "origin", "probe_latency_s", "verified", "models",
    # the Stage 1 runner's per-episode summary
    "episodes", "turned_rate", "passed_sideways_count", "first_turn_step_mean",
    "avg_success_steps", "avg_passage_rotation_deg", "resolved_tag",
    # a Stage 3 dressing item (scenes.py) and a scene material entry (scene_builder.py)
    "asset", "colour", "mount", "at", "size", "collides", "used_asset", "how", "url",
    "parts", "slot", "scene", "label", "kind",
}


def stage1_schema():
    """The record and sidecar field names, read from the committed archive."""
    record, sidecar = set(), set()
    with tarfile.open(ARCHIVE) as archive:
        for member in archive.getmembers():
            name = member.name
            if not name.endswith(".json"):
                continue
            if "_steps" in name and not sidecar:
                sidecar = set(json.load(archive.extractfile(member))[0])
            elif "/episode_" in name and not record:
                record = set(json.load(archive.extractfile(member)))
    return record, sidecar


def returned_keys(function_name: str, module_path: str) -> set:
    """Keys of the dict a function returns, read from its source."""
    with open(module_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    return {k.value for k in sub.value.keys if isinstance(k, ast.Constant)}
    return set()


def sources():
    seen = set()
    for folder in SCANNED:
        for root, _, files in os.walk(folder):
            if any(part in root for part in SKIP_DIRS):
                continue
            for filename in sorted(files):
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(root, filename)
                real = os.path.realpath(path)
                # SCANNED overlaps (both "." and "tools"), and this file's own docstring
                # quotes the strafe_left bug by way of explanation.
                if real in seen or os.path.basename(path) == os.path.basename(__file__):
                    continue
                seen.add(real)
                yield path


def docstring_lines(tree: ast.AST) -> set:
    """Line numbers occupied by a module/class/function docstring.

    A docstring that explains the strafe_left bug must not be reported as the bug, and
    neither must a comment.  Which is how this check first failed: on its own docstring.
    """
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                end = getattr(body[0], "end_lineno", body[0].lineno)
                lines.update(range(body[0].lineno, end + 1))
    return lines


def main() -> int:
    record, sidecar = stage1_schema()
    stage2 = returned_keys("round_record", "memory_protocol.py")
    stage2 |= returned_keys("summarise_run", "memory_protocol.py")
    stage2 |= returned_keys("curve_stats", "memory_metrics.py")
    stage1 = set(record) | set(sidecar)
    known = stage1 | stage2 | OTHER
    # A name that exists in one study's records but not the other's is the bug that got
    # through: first_turn_x IS a Stage 2 field, so a union schema calls it valid, and the
    # Stage 1 script that read it got None for all 660 episodes.  Files that read the
    # Stage 1 archive are therefore checked against the Stage 1 schema, not the union.
    stage2_only = stage2 - stage1
    actions = set(protocol.ACTIONS)
    print(f"schema: {len(record)} Stage 1 record fields, {len(sidecar)} sidecar fields, "
          f"{len(stage2)} Stage 2 record/summary fields, {len(known)} names known, "
          f"{len(actions)} valid actions")

    bad_fields, bad_actions, crossovers = [], [], []
    for path in sources():
        text = open(path, encoding="utf-8").read()
        reads_stage1 = "bao_v7_all.tgz" in text or "embodiedbao_v7_episodes.csv" in text
        skip = docstring_lines(ast.parse(text))
        for number, line in enumerate(text.splitlines(), 1):
            if number in skip or line.lstrip().startswith("#"):
                continue
            for variable, name in FIELD.findall(line):
                # Underscore-prefixed names are the local annotations the analysis
                # scripts add to a record themselves.
                if name.startswith("_"):
                    continue
                if name in stage2_only and reads_stage1:
                    crossovers.append((path, number, f"{variable}.get({name!r})", line))
                elif name not in known:
                    bad_fields.append((path, number, f"{variable}.get({name!r})", line))
            for name in ACTION_SHAPED.findall(line):
                if name not in actions:
                    bad_actions.append((path, number, f"action-shaped {name!r}", line))

    failed = False
    for title, findings in (
        ("field names not in any schema", bad_fields),
        ("Stage 2 fields read by a Stage 1 script", crossovers),
        ("action names not in protocol.ACTIONS", bad_actions),
    ):
        if findings:
            failed = True
            print(f"\n{len(findings)} {title}:")
            for path, number, what, line in findings:
                print(f"  {path}:{number}  {what}")
                print(f"      {line.strip()[:100]}")
        else:
            print(f"ok: no {title}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
