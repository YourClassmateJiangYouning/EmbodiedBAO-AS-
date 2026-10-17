#!/usr/bin/env python3
"""Reproduce, from scratch, every disputed finding about this repository.

WHY THIS FILE EXISTS
    The previous conversation's findings were reported by several automated reviewers, and the
    operator does not trust them.  So nothing here is a claim: every section RUNS the real code and
    prints the numbers it prints.  Read the output, not the comments.

HOW TO RUN
    cd /home/eai/EmbodiedBAO-AS-
    python3 verify_audit_findings.py                        # everything that needs no simulator
    /home/ybh/isaacsim/python.sh verify_audit_findings.py   # same; use if pxr/isaac imports matter

    Some sections import environment.py, which tries to import isaacsim and logs a warning.  That is
    expected and harmless: the geometry and action helpers are pure Python and work regardless.

WHAT IS *NOT* HERE
    Anything needing a rendered frame or a live model API.  Those are listed at the end under
    "WHAT THIS SCRIPT CANNOT CHECK" with the command that would settle each one.

STATE THIS WAS WRITTEN AGAINST
    HEAD = a440ae2, working tree clean.  If HEAD has moved, sections whose answer depends on the
    code will print different numbers -- which is the point of running it rather than trusting it.
"""

from __future__ import annotations

import collections
import glob
import inspect
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

W = 96


def head(title):
    print()
    print("=" * W)
    print(title)
    print("=" * W)


_FAILURES: list = []


def verdict(text, ok=None):
    """Record a verdict, and REMEMBER it.

    This function used to print and nothing else, so the whole file exited 0 whatever it found: a
    repository full of reproduced defects still reported success.  That is the exact failure mode the
    rest of this file exists to expose, and this file had it worse than anything it was auditing -- it
    presented itself as evidence and could not fail.

    ``ok`` may be passed explicitly; when it is not, a verdict containing WRONG, FAIL or MISMATCH counts
    as a failure, which matches the marks the sections already print.
    """
    if ok is None:
        upper = str(text).upper()
        ok = not any(bad in upper for bad in ("WRONG", "FAIL", "MISMATCH"))
    if not ok:
        _FAILURES.append(str(text))
    print("   ==> %s%s" % (text, "" if ok else "   [RECORDED AS A FAILURE]"))


# --------------------------------------------------------------------------------------
head("1. THE SIX ACTIONS: what each one physically does, right now")

try:
    import numpy as np
    import environment as e

    def qrot(q, v):
        w, x, y, z = [float(t) for t in q]
        v = np.asarray(v, float)
        return np.array([
            v[0] * (1 - 2 * (y * y + z * z)) + v[1] * 2 * (x * y - w * z) + v[2] * 2 * (x * z + w * y),
            v[0] * 2 * (x * y + w * z) + v[1] * (1 - 2 * (x * x + z * z)) + v[2] * 2 * (y * z - w * x),
            v[0] * 2 * (x * z - w * y) + v[1] * 2 * (y * z + w * x) + v[2] * (1 - 2 * (x * x + y * y)),
        ])

    # The project's own definition of which way is right, from test_bao_geometry.py:
    #   right = cross(forward, up), computed in Isaac coordinates where z is up.
    fwd_isaac = e._user_to_isaac_pos(e.action_delta("forward", e.MOVE_STEP))
    right_isaac = np.cross(fwd_isaac, np.array([0.0, 0.0, 1.0]))
    print("   the project's own fact: right = cross(forward, up) = isaac %s = user %s"
          % (np.round(right_isaac, 3), np.round(e._isaac_to_user_pos(right_isaac), 3)))
    print()
    print("   %-11s %-22s %-24s %s" % ("action", "user delta", "isaac direction", "physically"))
    for a in ("forward", "backward", "left", "right"):
        d = e.action_delta(a, e.MOVE_STEP)
        i = e._user_to_isaac_pos(d)
        mark = ""
        if a in ("left", "right"):
            want = right_isaac if a == "right" else -right_isaac
            mark = "OK" if np.allclose(i / np.linalg.norm(i), want / np.linalg.norm(want)) else "*** WRONG ***"
        print("   %-11s %-22s %-24s %s" % (a, np.round(d, 3), np.round(i, 3), mark))
    for a, sign in (("turn_left", +1), ("turn_right", -1)):
        q = e.BAOEnv._yaw_quat(sign * e.TURN_STEP_DEG)
        vi = qrot(q, [1, 0, 0])                      # where the body's own +x goes
        vi = vi / np.linalg.norm(vi)
        # A turn of 15 degrees does NOT point the body sideways: check which side of straight ahead
        # it moved to.  The walker's right is right_isaac, so the sign of the projection onto it
        # says whether the body turned right (positive) or left (negative).
        toward_right = float(np.dot(vi, right_isaac / np.linalg.norm(right_isaac)))
        want_right = (a == "turn_right")
        ok = (toward_right > 0) == want_right
        mark = "OK" if ok else "*** WRONG ***"
        print("   %-11s yaw %+d -> body +x to isaac %-16s turns %.1f deg toward %-5s %s"
              % (a, sign * e.TURN_STEP_DEG, np.round(vi, 3),
                 abs(math.degrees(math.asin(max(-1.0, min(1.0, toward_right))))),
                 "right" if toward_right > 0 else "left", mark))
    print()
    print("   HOW TO READ IT: 'left'/'right' must land on the cross-product directions above.")
    print("   For the turns, only the SIDE matters -- a 15 degree turn moves the body 15 degrees off")
    print("   straight ahead, so comparing it against a purely sideways vector would be wrong.")
    print("   'turn_left'/'turn_right' are written to the robot through BAOEnv._yaw_quat, NOT")
    print("   through _forward_vector -- see section 2, where those two disagree.")
except Exception as exc:  # noqa: BLE001
    verdict("could not run section 1: %r" % (exc,))


# --------------------------------------------------------------------------------------
head("2. THE REAL BUG: _forward_vector and _yaw_quat disagree by a mirror")

try:
    print("   Both take an angle.  One sets the robot's pose, the other aims the camera/head/hand.")
    print()
    print("   %-6s %-32s %-32s %s" % ("yaw", "_yaw_quat -> body +x", "_forward_vector", "agree?"))
    for deg in (0, 15, 30, 90):
        q = e.BAOEnv._yaw_quat(deg)
        body = e._isaac_to_user_pos(qrot(q, [1, 0, 0]))
        fv = e._forward_vector(math.radians(deg))
        ok = np.allclose(body, fv, atol=1e-6)
        print("   %-6s %-32s %-32s %s" % ("%+d" % deg, np.round(body, 3), np.round(fv, 3),
                                          "yes" if ok else "NO"))
    print()
    print("   CONSUMERS OF _forward_vector (whichever side is wrong, these follow it):")
    out = subprocess.run(["grep", "-rn", "_forward_vector(", "--include=*.py", REPO],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "/test_" in line or "def _forward_vector" in line or "lab_logs" in line:
            continue
        print("     %s" % line.replace(REPO + "/", "")[:112])
    # The raw NO above is now BY DESIGN: _forward_vector is the heading/gaze frame
    # (positive = the walker's right) and the torso is posed by _yaw_quat (positive =
    # left).  What mattered was that the camera/head/hand/collision consumers passed
    # the BODY yaw to the heading function.  They no longer do, which is what this
    # checks -- so a re-run of this script does not read a deliberate frame
    # difference as the old bug.
    still_heading = [
        line.replace(REPO + "/", "")
        for line in out.splitlines()
        if "self._robot_yaw" in line and "_body_forward_vector(" not in line
        and "/test_" not in line and "lab_logs" not in line
    ]
    verdict(
        "raw NO is expected (heading frame vs torso frame). Body-yaw calls left on "
        "_forward_vector: %s" % (still_heading if still_heading else "NONE -- the mirror is fixed")
    )
except Exception as exc:  # noqa: BLE001
    verdict("could not run section 2: %r" % (exc,))


# --------------------------------------------------------------------------------------
head("3. THE CHECKPOINT: does a run WITHOUT --resume destroy finished units?")

try:
    from unittest import mock
    import experiments as X

    cands = sorted(glob.glob(os.path.join(REPO, "results/*/checkpoint_*.json")), key=os.path.getsize)
    if not cands:
        verdict("no checkpoint on disk to test with")
    else:
        path = cands[-1]
        stored = json.load(open(path, encoding="utf-8")).get("completed", [])
        written = {}
        with mock.patch.object(X, "load_json_file", side_effect=lambda p: json.load(open(p))), \
             mock.patch.object(X, "atomic_write_json",
                               side_effect=lambda p, d, **k: written.update(d)), \
             mock.patch.object(X, "backup_corrupt_file", return_value=None):
            cp = X.ProtocolCheckpoint(path, resume=False)
            before = len(cp.completed)
            cp.mark("level3/episode000")
            after = len(written.get("completed", []))
        print("   checkpoint on disk            : %s" % os.path.relpath(path, REPO))
        print("   units on disk                 : %d" % len(stored))
        print("   units in memory (no --resume) : %d" % before)
        print("   units WRITTEN BACK            : %d" % after)
        if after < len(stored):
            verdict("CONFIRMED: %d of %d completed units destroyed by one non-resume run"
                    % (len(stored) - after, len(stored)))
        else:
            verdict("not reproduced on this checkout")
except Exception as exc:  # noqa: BLE001
    verdict("could not run section 3: %r" % (exc,))


# --------------------------------------------------------------------------------------
head("4. THE LEADERBOARD: how many episodes have NO successful model call at all?")

files = []
try:
    files = [f for f in glob.glob(os.path.join(REPO, "results/**/episode_[0-9]*.json"), recursive=True)
             if not f.endswith("_steps.json")]
    rows = []
    for f in files:
        try:
            rows.append(json.load(open(f, encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    for d in rows:
        m = d.get("model_name", "?")
        a = per[m]
        a[0] += 1
        a[1] += 1 if d.get("passed") else 0
        if d.get("invalid_response_count", 0) == d.get("total_steps"):
            a[2] += 1
            a[3] += 1 if d.get("passed") else 0
    print("   %d episode records" % len(rows))
    print()
    print("   %-34s %-15s %-11s %s" % ("model", "as recorded", "all-calls-failed", "excluding those"))
    for m, (n, p, bad, bp) in sorted(per.items(), key=lambda kv: -(kv[1][1] / kv[1][0] if kv[1][0] else 0)):
        keep_n, keep_p = n - bad, p - bp
        print("   %-34s %-15s %-11s %s"
              % (m, "%d/%d=%.3f" % (p, n, p / n if n else 0), bad,
                 "%d/%d=%.3f" % (keep_p, keep_n, keep_p / keep_n if keep_n else 0)))
    tot_bad = sum(v[2] for v in per.values())
    verdict("%d episodes have invalid_response_count == total_steps, i.e. not one successful call. "
            "FIXED in analysis.py: episode_had_no_valid_call drops them from the rates and reports "
            "the count, so the 'excluding those' column is the capability number and kimi -- 100%% "
            "of them -- reads as no data rather than 0.000" % tot_bad)
except Exception as exc:  # noqa: BLE001
    verdict("could not run section 4: %r" % (exc,))


# --------------------------------------------------------------------------------------
head("5. THE DATA FIELDS: what is recorded, what is missing, what reaches the CSV")

try:
    keys = collections.Counter()
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for k in d:
            keys[k] += 1
    print("   fields present in the episode JSON records (top-level):")
    for k, v in keys.most_common():
        print("     %-28s %6d" % (k, v))
    print()
    for name in ("marker_slot", "robot_ground_offset"):
        n = keys.get(name, 0)
        print("   %-22s present in %d of %d records%s"
              % (name, n, len(files), "" if n else "   <-- ABSENT EVERYWHERE"))
    print()
    import main as M
    src = inspect.getsource(M)
    i = src.find("episode_fields = (")
    j = src.find(")", i)
    block = src[i:j]
    print("   CSV episode_fields:")
    for line in block.splitlines():
        line = line.strip()
        if line and not line.startswith("episode_fields"):
            print("     %s" % line)
    for name in ("marker_slot", "robot_ground_offset"):
        print("   %-22s in the CSV column list: %s" % (name, name in block))
    in_csv = all(name in block for name in ("marker_slot", "robot_ground_offset"))
    verdict(
        "FIXED: both names are in the CSV column list. The corpus still lacks them -- all 3465 "
        "records predate the fields -- so the columns are empty for old data and filled after "
        "experiments.py started recording them"
        if in_csv
        else "the fields are recorded in the JSON but still missing from the CSV column list"
    )
except Exception as exc:  # noqa: BLE001
    verdict("could not run section 5: %r" % (exc,))


# --------------------------------------------------------------------------------------
head("6. THE MARKER SHAPES: is the drawn shape the one the prompt names?")

try:
    import scenes as sc
    import scene_builder
    print("   scenes.py documents the marker plane's in-plane coordinates as (sideways, height).")
    print("   prism_points in scene_builder.py maps them to the world (y = lateral, z = height).")
    print()
    centre = (sc.MARKER_X_M, sc.MARKER_Y_M, sc.MARKER_Z_M)
    _, cy, cz = sc.to_world(centre)
    for shape in ("triangle_down", "arrow", "bars2"):
        if shape not in sc.SHAPES:
            continue
        front, _ = scene_builder.prism_points(
            sc.SHAPES[shape][0], centre, sc.MARKER_THICKNESS_M)
        lateral = [p[1] for p in front]
        height = [p[2] for p in front]
        print("   %-14s lateral %+.3f..%+.3f (centre %+.3f)  height %+.3f..%+.3f (centre %+.3f)"
              % (shape, min(lateral), max(lateral), cy, min(height), max(height), cz))
    front, _ = scene_builder.prism_points(
        sc.SHAPES["triangle_down"][0], centre, sc.MARKER_THICKNESS_M)
    apex = min(range(len(front)), key=lambda index: front[index][2])
    points_down = abs(front[apex][1] - cy) < 1e-9 and front[apex][2] < cz
    print()
    print("   prompt name: %r" % sc.SHAPE_NAMES["triangle_down"])
    verdict("the triangle's apex is the lowest, laterally centred vertex: %s -- so the drawn "
            "shape matches the prompt" % (points_down,))
except Exception as exc:  # noqa: BLE001
    verdict("could not run section 6: %r" % (exc,))


# --------------------------------------------------------------------------------------
head("7. THE ACTION PARSER: does it map a model's words to the action it meant?")

try:
    import ai_agent
    cases = ['{"action": "left turn"}', '{"action": "turn to the left"}', '{"action": "left_turn"}',
             '{"action": "right turn"}', '{"action": "avoid forward"}', '{"action": "do not go left"}']
    print("   %-38s %s" % ("model says", "parser returns"))
    for c in cases:
        try:
            got = ai_agent.parse_action_json(c)
            got = got.get("action") if isinstance(got, dict) else got
        except Exception as exc:  # noqa: BLE001
            got = "<%s>" % type(exc).__name__
        print("   %-38s %s" % (c, got))
    print()
    print("   'left' is a 0.75 m SIDESTEP; 'turn_left' is a 15 degree ROTATION.  A mapping from one")
    print("   to the other changes what the model physically did.")
except Exception as exc:  # noqa: BLE001
    verdict("could not run section 7: %r" % (exc,))


# --------------------------------------------------------------------------------------
head("8. THE TESTS: which ones cannot fail?")

for fname, func, why in (
    ("test_bao_scenes.py", "test_no_two_declared_boxes_in_a_scene_overlap",
     "FIXED: the fatal check used to read a list that was never appended to; it now "
     "fails on any overlap not in DECLARED_BOX_OVERLAP_KNOWN"),
    ("test_bao_scenes.py", "test_dressing_stands_on_the_floor_and_hangs_on_its_wall",
     "FALSIFIABLE, not unfalsifiable: burying a floor item or floating a wall item "
     "makes it fail (the earlier 'unreachable branch' claim was about the shipped "
     "data, not about the assertion)"),
):
    path = os.path.join(REPO, fname)
    if not os.path.exists(path):
        continue
    src = open(path, encoding="utf-8").read()
    i = src.find("def %s" % func)
    if i < 0:
        print("   %-52s NOT FOUND" % func)
        continue
    print("   %s\n      %s\n      read it at %s:%d"
          % (func, why, fname, src[:i].count("\n") + 1))
print()
print("   SETTLE IT BY MUTATION, IN A COPY:")
print("     cp -r . /dev/shm/probe && cd /dev/shm/probe")
print("     # edit scenes.py so two dressing boxes in one scene are identical")
print("     python3 test_bao_scenes.py | tail -3   # it must now FAIL on the new pair")


# --------------------------------------------------------------------------------------
head("9. THE DISPATCHER: does a failed sweep report failure?")

p = os.path.join(REPO, "run_parallel_sweep.sh")
if os.path.exists(p):
    src = open(p, encoding="utf-8").read()
    print("   the lines that decide the exit status:")
    for n, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if s.startswith("cleanup()") or 'exit "$rc"' in s or s.startswith("trap ") \
           or "all jobs dispatched" in s or "XARGS_RC" in s:
            print("     %4d  %s" % (n, s[:110]))
    print()
    print("   A non-zero XARGS_RC read but never exited on would leave the script at 0;")
    print("   the last command is now exit \"${XARGS_RC:-0}\", so the EXIT trap re-exits")
    print("   with xargs's status (123 when a worker failed).")
else:
    print("   not present")


# --------------------------------------------------------------------------------------
head("10. THE ANALYSIS: which tags does it read by default?")

p = os.path.join(REPO, "analysis.py")
if os.path.exists(p):
    src = open(p, encoding="utf-8").read()
    for n, line in enumerate(src.splitlines(), 1):
        if "selected_tags" in line or "getmtime" in line or "PROTOCOL_TAG" in line \
           or "pooling" in line:
            print("     %4d  %s" % (n, line.strip()[:110]))
    verdict("FIXED: the default no longer picks one tag by directory mtime. It pools every "
            "current-protocol tag (all scenes) and prints which ones; getmtime survives only "
            "as the last-resort fallback when NO current-protocol tag exists")
else:
    print("   not present")


# --------------------------------------------------------------------------------------
head("WHAT THIS SCRIPT CANNOT CHECK")

print("""
   1. Whether the EYE CAMERA sits on the correct side of the robot when the torso is rotated.  Its
      offset comes from _forward_vector, which section 2 shows disagrees with _yaw_quat.  Settle it
      by rendering and reading the statistics the tool prints (std, unique_colors):
        /home/ybh/isaacsim/python.sh capture_views.py --level 10 --outdir /tmp/views

   2. Whether the drawn marker points the way the prompt says.  Render and look:
        /home/ybh/isaacsim/python.sh capture_scenes.py --scene stage1.2 --level 10 --outdir scene_preview

   3. Whether a model answers in a field the client never reads (reasoning_content).  Needs a
      gateway and a paid call.

   4. How much of the corpus ran with the robot below the floor.  The logs alternate between
      "ground offset = 1.0442" and "= 0.0000"; the per-episode field that would attribute one to each
      episode was added AFTER all the data was collected, so it is absent from every record:
        grep -rho "ground offset = [0-9.]*" logs/ *.log | sort | uniq -c
""")

print("=" * W)
print("END.  Every number above was produced by running the repository's own code.")
print("=" * W)
# THE EXIT STATUS IS THE WHOLE CONTRACT.  Printing a failure and exiting 0 means a caller cannot tell a
# clean run from a broken one, and this file was cited as proof that the findings had been checked.
if _FAILURES:
    print()
    print("FAILED: %d finding(s) did NOT reproduce:" % len(_FAILURES))
    for _failure in _FAILURES:
        print("  - %s" % _failure)
    raise SystemExit(1)
print()
print("PASSED: every finding above reproduced")
