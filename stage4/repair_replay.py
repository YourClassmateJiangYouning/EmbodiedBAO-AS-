"""Repair mode: re-run only the episodes whose prompt contained the wrong blocked-move feedback.

An episode that never received a blocked message had a prompt history that never contained the divergent
string, so its decisions are exactly what the corrected code would have produced -- it can be kept.  An
episode that did receive one has a contaminated history from that step onward, so the whole episode is re-run.

    python stage4/replay_runner.py --repair

For every planned (model, A/S, marker) cell:
  * session absent                      -> run it
  * session present, no blocked step    -> keep, skip
  * session present, has a blocked step -> move the directory to sessions_r1_discarded/ and re-run
"""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import core  # noqa: E402
import replay_runner as runner  # noqa: E402

BANK = runner.BANK
ROOT = os.path.join(HERE, "sessions_r1")
DISCARD = os.path.join(HERE, "sessions_r1_discarded")


def session_id_for(spec):
    return runner.session_id_for(spec)


def episode_state(session_dir):
    """Return ("absent"|"clean"|"contaminated", blocked_count) for one cell."""
    steps_file = os.path.join(session_dir, "steps.jsonl")
    manifest = os.path.join(session_dir, "manifest.csv")
    if not os.path.isfile(steps_file):
        return "absent", 0
    blocked = 0
    with open(steps_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "blocked" in (row.get("feedback") or ""):
                blocked += 1
    # a session directory with steps but no manifest row was interrupted; treat it as absent so it re-runs
    if not os.path.isfile(manifest):
        return "absent", blocked
    return ("contaminated" if blocked else "clean"), blocked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(ROOT, exist_ok=True)
    plan = runner.plan_matrix()

    todo, keep, discard = [], 0, 0
    for spec in plan:
        directory = os.path.join(ROOT, session_id_for(spec))
        state, blocked = episode_state(directory)
        if state == "clean":
            keep += 1
        elif state == "contaminated":
            todo.append((spec, directory, blocked))
            discard += 1
        else:
            todo.append((spec, directory, 0))

    print("cells        :", len(plan))
    print("keep (clean) :", keep)
    print("re-run       :", len(todo), " (of which contaminated:", discard, ")")
    if args.dry_run:
        print()
        for spec, directory, blocked in todo:
            print("  re-run %-32s A/S %.1f  %-24s blocked=%d"
                  % (spec["model"], spec["a_s"], spec["marker"], blocked))
        return 0

    if not todo:
        print("nothing to do")
        return 0

    os.makedirs(DISCARD, exist_ok=True)
    results = []
    for index, (spec, directory, blocked) in enumerate(todo, 1):
        if os.path.isdir(directory):
            target = os.path.join(DISCARD, os.path.basename(directory) + "_" + core.new_id("old"))
            shutil.move(directory, target)
            print("[%3d/%3d] discarded %s (blocked=%d) -> %s"
                  % (index, len(todo), os.path.basename(directory), blocked, os.path.basename(target)))
        print("[%3d/%3d] %-32s A/S %.1f  %s"
              % (index, len(todo), spec["model"], spec["a_s"], spec["marker"]))
        session = core.Stage4Session(sessions_root=ROOT)
        session.open_session(session_id_for(spec), {
            "scene": "R1", "marker": spec["marker_slot"], "marker_descriptor": spec["marker"],
            "mode": "A", "model": spec["model"], "a_s": spec["a_s"],
            "max_steps": runner.MAX_STEPS, "disable_thinking": True,
            "shoulder_m": runner.S, "move_step": runner.STEP,
        })
        result = runner.run_episode(session, spec, verbose=False)
        results.append({"spec": spec, **result})
        print("      -> passed=%s steps=%s end=%s"
              % (result.get("passed"), result.get("steps"), result.get("end_reason")))
    print()
    print("re-ran", len(results), "episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
