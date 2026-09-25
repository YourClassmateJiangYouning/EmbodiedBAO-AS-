"""Explain why a straight frontal walk stops short, if it ever does.

An early version of tools/check_clearance.py reported that walking straight ahead
reaches only x=10.25 at Levels 0-3 and x=7.25 at Levels 4-5, against the then
success plane of x>11.0.  That would have meant NO Level was passable by walking
forward, contradicting the recorded runs where Levels 0-3 passed 60-100% of the
time.  The report was the bug (the walk reaches 12.50 unbounded), and this tool
exists to re-check that from the grid rather than from a formula.

The step grid is 0.75 m jumps from x=0.5: 0.5, 1.25, ..., 8.0, 8.75, 9.5.  The
success plane is at 8.75, one stride past the wall, so the walk-in poses that
matter are 7.25 (last one before the wall) and 8.0, 8.75 (through it).

    python tools/check_frontal.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    import numpy as np

    from environment import (
        LEVEL_CHANNEL_WIDTHS,
        MOVE_STEP,
        ROBOT_START_POS,
        SUCCESS_X,
        WALL_X,
        _check_scene_collision,
        _check_wall_collision,
        _translation_path_is_clear,
    )

    start = float(ROBOT_START_POS[0])
    print(f"start x={start}  step={MOVE_STEP}  success x>{SUCCESS_X}  wall x={WALL_X}")
    print()

    for level, channel in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        print(f"=== level {level}  channel {channel:.2f} m ===")
        # The reachable grid positions along a straight walk.
        positions = []
        x = start
        for _ in range(40):
            positions.append(round(x, 6))
            x += MOVE_STEP
        interesting = [p for p in positions if p >= SUCCESS_X - 3 * MOVE_STEP]
        print("  grid positions near the goal:", interesting[:8])
        for x in interesting[:8]:
            pose = np.array([x, 0.0, 0.0])
            wall = _check_wall_collision(pose, 0.0, channel)
            scene = _check_scene_collision(pose, 0.0, channel)
            where = "fine"
            if scene is not None:
                where = str(scene.get("boundary") or scene.get("part") or scene)
            print(
                f"    x={x:6.2f}  wall={'legal' if wall is None else 'BLOCKED'}"
                f"  scene={'legal' if scene is None else 'BLOCKED (' + where + ')'}"
            )

        # Walk it, reporting the first rejected step.
        x = start
        while x < SUCCESS_X + 2 * MOVE_STEP:
            target = np.array([x + MOVE_STEP, 0.0, 0.0])
            bad = _translation_path_is_clear(
                np.array([x, 0.0, 0.0]), target, 0.0, channel
            )
            if bad is not None:
                where = str(bad.get("boundary") or bad.get("part") or bad)
                print(
                    f"  first rejected step: {x:.2f} -> {x + MOVE_STEP:.2f} "
                    f"({where})"
                )
                break
            x += MOVE_STEP
        else:
            print(f"  walked to {x:.2f} without rejection")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
