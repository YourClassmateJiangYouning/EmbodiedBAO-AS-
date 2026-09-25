"""Did the 2 mm body inflation ever decide a recorded step?

Removing BODY_CLEARANCE makes the gate strictly more permissive, so the question
for already-collected data is one-directional: a step that the OLD model BLOCKED
but the NEW model allows would have gone differently, and the episode is stale.
(A step the old model allowed is still allowed by the new one, so accepted steps
can never disagree.)

Only blocked steps can differ, and for those the recorded pose is the pre-step
pose, so the attempted move can be rebuilt from the log: position, torso yaw and
the action are all recorded.  Each attempted move is then re-judged twice, once
with the old 2 mm inflation and once with the current 0 mm.

    python3 tools/check_skin_sensitivity.py                       # reads results/
    python3 tools/check_skin_sensitivity.py --root /path/whatever
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment as env  # noqa: E402
from environment import (  # noqa: E402
    LEVEL_CHANNEL_WIDTHS,
    MOVE_STEP,
    SUCCESS_X,
    TURN_STEP_DEG,
    _forward_vector,
    _right_vector,
    _translation_path_is_clear,
    _turn_path_is_clear,
)

OLD_INFLATION = 0.002
TRANSLATIONS = ("forward", "backward", "left", "right")
TURNS = ("turn_left", "turn_right")


def attempted_move(row: dict) -> tuple[str, str, np.ndarray, float, float] | None:
    """Rebuild a blocked step's attempt: (kind, action, pos, yaw_before, yaw_after)."""
    try:
        pos = np.array([float(row["position_x"]), 0.0, float(row["position_z"])])
        yaw = float(row["torso_rotation"])
        action = str(row["action"]).strip().lower()
    except (KeyError, TypeError, ValueError):
        return None
    if action in TURNS:
        delta = TURN_STEP_DEG if action == "turn_left" else -TURN_STEP_DEG
        return ("turn", action, pos, yaw, yaw + delta)
    if action in TRANSLATIONS:
        return ("move", action, pos, yaw, yaw)
    return None


def gate_blocks(
    kind: str,
    action: str,
    pos: np.ndarray,
    yaw: float,
    before: float,
    after: float,
    width: float,
) -> bool:
    """Would the gate reject this attempted move, at the current inflation?"""
    if kind == "turn":
        return _turn_path_is_clear(pos, before, after, width) is not None
    yaw_rad = np.radians(yaw)
    if action == "forward":
        delta = _forward_vector(yaw_rad) * MOVE_STEP
    elif action == "backward":
        delta = _forward_vector(yaw_rad) * -MOVE_STEP
    elif action == "right":
        delta = _right_vector(yaw_rad) * MOVE_STEP
    else:
        delta = _right_vector(yaw_rad) * -MOVE_STEP
    return _translation_path_is_clear(pos, pos + delta, yaw_rad, width) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"no such results directory: {args.root}")
        print("run this where the episodes live (the lab machine)")
        return 1

    per_level: dict[int, dict[str, int]] = defaultdict(
        lambda: {"steps": 0, "blocked": 0, "flips": 0, "passes": 0}
    )
    flips: dict[str, int] = defaultdict(int)
    files = 0

    for dirpath, _dirnames, filenames in os.walk(args.root):
        for name in sorted(filenames):
            if not name.endswith("_steps.json"):
                continue
            level = None
            for part in dirpath.replace("\\", "/").split("/"):
                if part.startswith("level") and part[5:].isdigit():
                    level = int(part[5:])
            if level is None or level not in LEVEL_CHANNEL_WIDTHS:
                continue
            width = LEVEL_CHANNEL_WIDTHS[level]
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    rows = json.load(handle)
            except (OSError, ValueError):
                continue
            if not isinstance(rows, list):
                continue
            files += 1
            stats = per_level[level]
            for row in rows:
                stats["steps"] += 1
                if float(row.get("position_x", 0.0)) >= SUCCESS_X:
                    stats["passes"] += 1
                if not row.get("collision"):
                    continue
                attempted = attempted_move(row)
                if attempted is None:
                    continue
                kind, action, pos, before, after = attempted
                stats["blocked"] += 1
                env.BODY_CLEARANCE = OLD_INFLATION
                old = gate_blocks(kind, action, pos, before, before, after, width)
                env.BODY_CLEARANCE = 0.0
                new = gate_blocks(kind, action, pos, before, before, after, width)
                if old and not new:
                    stats["flips"] += 1
                    flips[f"level{level}/{os.path.basename(dirpath)}/{name}"] += 1

    env.BODY_CLEARANCE = 0.0

    if files == 0:
        print(f"no episode step files found under {args.root}")
        return 1

    print(f"scanned {files} step files under {args.root}")
    print(f"old inflation {OLD_INFLATION * 1000:.1f} mm per side, current 0.0 mm")
    print()
    header = (
        f"{'Level':>5} {'steps':>8} {'blocked':>8} {'would now be legal':>19}"
    )
    print(header)
    print("-" * len(header))
    total = 0
    for level in sorted(per_level):
        stats = per_level[level]
        total += stats["flips"]
        print(
            f"{level:>5} {stats['steps']:>8} {stats['blocked']:>8} "
            f"{stats['flips']:>19}"
        )
    print()
    print(f"blocked moves the current model would allow instead: {total}")
    if total == 0:
        print(
            "Zero.  Every step in the collected episodes is judged the same way by "
            "both collision models, so no trajectory changes and only Level 4 has "
            "to be re-run."
        )
    else:
        print("Those steps would have taken a different path.  Files affected:")
        for key, count in sorted(flips.items()):
            print(f"   {key}: {count}")
        print(
            "Either re-run those episodes, or keep the inflation and widen Level 4 "
            "by the same 2 mm instead (tools/compare_l4_options.py option B)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
