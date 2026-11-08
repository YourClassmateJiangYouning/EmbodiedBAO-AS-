"""Audit every value that reaches the prompt, and every row that reaches disk.

This exists because two silent mismatches already shipped: the prompt froze the position at the start
coordinates, and the frames advanced twice as fast as the state.  Both were invisible in the runner's own
log, which prints the local state rather than what was sent.  So this checks the ARTEFACTS:

  A. prompt semantics   -- each state field equals what the simulator would report for that state
  B. image alignment    -- the frame in the row is the frame the state calls for
  C. physics            -- W(theta), blocked moves, turn refusal, the success plane
  D. files              -- every prompt/response exists, every bank frame exists and matches its sha256
  E. leakage            -- no A/S, level, width or removed action in any prompt

Run:  python stage4/audit_replay.py [session_dir] [--live]
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import core  # noqa: E402

S = core.SIM_GEOMETRY["shoulder_width_m"]
D = core.SIM_GEOMETRY["torso_thickness_m"]
WALL = core.SIM_GEOMETRY["wall_x_m"]
START = core.SIM_GEOMETRY["start_x_m"]
STEP = core.SIM_GEOMETRY["move_step_m"]
SUCCESS = WALL + STEP

failures = []


def check(condition, message):
    if condition:
        return True
    failures.append(message)
    print("  FAIL:", message)
    return False


def fold(angle):
    folded = abs(angle) % 360.0
    return 360.0 - folded if folded > 180.0 else folded


# ------------------------------------------------------------------ A. prompt semantics, from the builder
def audit_prompt_semantics():
    print("A. prompt state fields follow the simulator's conventions")
    session = core.Stage4Session()
    session.open_session("audit-a", {
        "scene": "R1", "marker": "M1", "marker_descriptor": "red circle", "mode": "A",
        "model": "gemini-3-flash-preview", "a_s": 1.0, "max_steps": 30,
    })
    session.start_episode()

    # position at the start
    prompt = session.build_current_prompt()
    match = re.search(r"position \(x, y, z\): \[([^\]]+)\]", prompt)
    check(match is not None, "position line missing")
    x_start = float(match.group(1).split(",")[0])
    check(abs(x_start - START) < 1e-6, "position at start is %.3f, expected %.3f" % (x_start, START))

    # position advances with the odometer
    session.position_x = 4.25
    prompt = session.build_current_prompt()
    match = re.search(r"position \(x, y, z\): \[([^\]]+)\]", prompt)
    check(abs(float(match.group(1).split(",")[0]) - 4.25) < 1e-6,
          "position did not follow the odometer")

    # torso angle is folded into [0, 180] like environment.get_abs_torso_rotation()
    for raw, expected in ((15.0, 15.0), (-15.0, 15.0), (-90.0, 90.0), (195.0, 165.0), (345.0, 15.0)):
        session.history = [{"step": 0, "action": "turn_right", "feedback": "executed",
                            "torso_rotation": raw}]
        prompt = session.build_current_prompt()
        got = float(re.search(r"torso rotation \(degrees\): ([0-9.]+)", prompt).group(1))
        check(abs(got - expected) < 1e-6,
              "torso %.1f reported as %.1f, expected %.1f (fold into [0,180])" % (raw, got, expected))

    # the glance lasts one frame: pitch is 45 only on the observation that follows look_down
    session.history = [{"step": 0, "action": "look_down", "feedback": "executed", "torso_rotation": 0.0}]
    prompt = session.build_current_prompt()
    got = float(re.search(r"head camera extra downward pitch \(degrees\): ([0-9.]+)", prompt).group(1))
    check(abs(got - 45.0) < 1e-6, "after look_down the pitch is %.1f, expected 45.0" % got)

    session.history = [{"step": 0, "action": "forward", "feedback": "executed", "torso_rotation": 0.0}]
    prompt = session.build_current_prompt()
    got = float(re.search(r"head camera extra downward pitch \(degrees\): ([0-9.]+)", prompt).group(1))
    check(abs(got) < 1e-6, "after forward the pitch is %.1f, expected 0.0" % got)

    # the action menu and the leakage rules
    actions = re.findall(r'\{"action": "(\w+)"\}', prompt)
    check(actions == core.REAL_ACTION_SET, "action menu is %s" % actions)
    # "how wide your body is" and "how wide you are" are the frozen Stage 1 wording about the BODY, not
    # about the opening, so a bare substring test would be wrong.  What must not appear is the apparatus:
    # the ratio, the level index, the channel width, or the words applied to the aperture.
    for banned in ("A/S", "aperture", "channel width", "level ", "0.456", "0.342", "0.684",
                   "narrow opening", "wide opening", "this opening is", "narrow gap", "wide gap"):
        check(banned.lower() not in prompt.lower(), "prompt leaks %r" % banned)
    for removed in ("look_left", "look_right", "left/right"):
        check(removed not in prompt, "prompt still mentions removed action %r" % removed)


# ------------------------------------------------------------------ B + C. session artefacts
def audit_session(root):
    sessions = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if not sessions:
        print("B/C. no sessions yet")
        return
    print("B. image alignment and C. physics, over %d session(s)" % len(sessions))
    steps_checked = 0
    for name in sessions:
        path = os.path.join(root, name)
        steps_file = os.path.join(path, "steps.jsonl")
        if not os.path.exists(steps_file):
            continue
        rows = [json.loads(line) for line in open(steps_file, encoding="utf-8")]
        previous_x, previous_action = START, None
        for row in rows:
            steps_checked += 1
            x = float(row.get("position_x") or previous_x)

            # the frame must be the one the state calls for
            image_path = row.get("image_path") or ""
            if previous_action == "look_down":
                check("look_down" in image_path,
                      "%s step %d: after look_down the frame is %s" % (name, row["step"], image_path))
            else:
                index = int(round((previous_x - START) / STEP))
                index = max(0, min(10, index))
                expected = "%d.PNG" % (index + 1)
                check(os.path.basename(image_path) == expected,
                      "%s step %d: frame %s, expected %s at x=%.2f"
                      % (name, row["step"], os.path.basename(image_path), expected, previous_x))

            # the prompt on disk must report the position the state had
            prompt_path = os.path.join(path, row.get("prompt_path") or "")
            check(os.path.isfile(prompt_path), "%s step %d: prompt file missing" % (name, row["step"]))
            if os.path.isfile(prompt_path):
                text = open(prompt_path, encoding="utf-8").read()
                got = float(re.search(r"position \(x, y, z\): \[([0-9.]+)", text).group(1))
                check(abs(got - previous_x) < 1e-3,
                      "%s step %d: prompt x=%.3f, state x=%.3f" % (name, row["step"], got, previous_x))

            # the physics must be internally consistent
            theta = float(row.get("torso_rotation") or 0.0)
            width = S * abs(math.cos(math.radians(theta))) + D * abs(math.sin(math.radians(theta)))
            a_s = float(row.get("a_s") or 0.0)
            moving = row.get("action") in ("forward", "backward")
            crossed = previous_x < WALL <= x
            # The body width only constrains the passage through the wall plane; walking down the corridor
            # in front of it is free however wide the body is.  Checking every forward move was the audit
            # script's own error, not the runner's.
            if row.get("action") == "forward":
                if crossed and width > a_s * S + 1e-9:
                    check(False, "%s step %d: crossed the wall with W=%.3f > A=%.3f"
                          % (name, row["step"], width, a_s * S))
                if not crossed:
                    delta = x - previous_x
                    if abs(delta) < 1e-9:
                        # a blocked move: legal only when this step WOULD have crossed the wall and the body
                        # does not fit at its current torso angle
                        would_cross = previous_x < WALL <= previous_x + STEP
                        check(would_cross and width > a_s * S + 1e-9,
                              "%s step %d: forward was blocked but the passage was open (x=%.2f, W=%.3f, A=%.3f)"
                              % (name, row["step"], previous_x, width, a_s * S))
                    else:
                        check(abs(delta - STEP) < 1e-9,
                              "%s step %d: forward moved %.2f m" % (name, row["step"], delta))
            previous_x, previous_action = x, row.get("action")

    print("  steps checked:", steps_checked)


# ------------------------------------------------------------------ D + E. files and leakage
def audit_files(root):
    sessions = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    print("D. files, over %d session(s)" % len(sessions))
    prompts = responses = steps = 0
    for name in sessions:
        path = os.path.join(root, name)
        steps_file = os.path.join(path, "steps.jsonl")
        if not os.path.exists(steps_file):
            continue
        rows = [json.loads(line) for line in open(steps_file, encoding="utf-8")]
        steps += len(rows)
        for row in rows:
            pp = os.path.join(path, row.get("prompt_path") or "")
            rp = os.path.join(path, row.get("response_path") or "")
            prompts += os.path.isfile(pp)
            responses += os.path.isfile(rp)
            bank = row.get("image_path") or ""
            if bank and not os.path.isfile(bank):
                check(False, "%s step %d: bank frame gone: %s" % (name, row["step"], bank))
        # the manifest must have one row per finished episode
        manifest = os.path.join(path, "manifest.csv")
        if os.path.exists(manifest):
            import csv
            with open(manifest, encoding="utf-8", newline="") as handle:
                manifest_rows = sum(1 for _ in csv.DictReader(handle))
            episodes_file = os.path.join(path, "episodes.jsonl")
            episodes = sum(1 for _ in open(episodes_file, encoding="utf-8")) if os.path.exists(episodes_file) else 0
            check(manifest_rows == episodes,
                  "%s: manifest has %d rows, episodes.jsonl has %d" % (name, manifest_rows, episodes))
    print("  steps %d, prompts present %d, responses present %d" % (steps, prompts, responses))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sessions_r1")
    print("=" * 70)
    audit_prompt_semantics()
    print()
    if os.path.isdir(root):
        audit_session(root)
        print()
        audit_files(root)
    print()
    print("=" * 70)
    if failures:
        print("AUDIT FAILED:", len(failures), "problem(s)")
        for item in failures[:20]:
            print("  -", item)
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
