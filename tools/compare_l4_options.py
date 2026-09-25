"""Compare the two ways of making Level 4 passable, against the real gate.

Option A (current tree): drop the 2 mm body inflation, keep the channel at 0.570
so the printed A/S stays exactly 1.000.

Option B (widening): keep the 2 mm inflation and widen the Level 4 channel to
0.574, so the width the collision model asks for is exactly the channel and the
nominal A/S becomes 1.007.

Both are measured here on the same gate, per Level: the nominal and effective
A/S, the free space per side, and whether a straight frontal walk from the start
pose actually reaches the success plane.

    python tools/compare_l4_options.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment as env  # noqa: E402
from environment import (  # noqa: E402
    MOVE_STEP,
    ROBOT_SHOULDER_WIDTH,
    ROBOT_START_POS,
    SUCCESS_X,
    _check_wall_collision,
    _translation_path_is_clear,
)

OPTIONS = {
    "A current (skin 0.000, L4 0.570)": (0.0, 0.570),
    "B widened (skin 0.002, L4 0.574)": (0.002, 0.574),
}


def frontal_walk(channel_width: float) -> tuple[bool, float, int, float]:
    """Straight walk from the start pose; returns (reached, max x, steps, max |z|)."""
    pos = np.asarray(ROBOT_START_POS, dtype=float).copy()
    yaw = 0.0
    delta = np.array([MOVE_STEP, 0.0, 0.0])
    steps = 0
    max_z = 0.0
    while steps < 40:
        target = pos + delta
        if (
            _translation_path_is_clear(pos, target, yaw, channel_width)
            is not None
        ):
            break
        pos = target
        steps += 1
        max_z = max(max_z, abs(float(pos[2])))
        if pos[0] >= SUCCESS_X:
            break
    return bool(pos[0] >= SUCCESS_X), float(pos[0]), steps, max_z


def main() -> int:
    original_skin = env.BODY_CLEARANCE
    original_l4 = env.LEVEL_CHANNEL_WIDTHS[4]
    print(f"nominal shoulder width {ROBOT_SHOULDER_WIDTH:.4f} m")
    print(f"success plane x >= {SUCCESS_X}   start x = {ROBOT_START_POS[0]}")
    print()

    for name, (skin, l4_width) in OPTIONS.items():
        env.BODY_CLEARANCE = float(skin)
        env.LEVEL_CHANNEL_WIDTHS[4] = float(l4_width)
        required = ROBOT_SHOULDER_WIDTH + 2.0 * skin
        print(f"{name}")
        print(f"  body the gate tests: {required:.4f} m "
              f"(shoulder + {skin * 1000:.0f} mm per side)")
        head = (
            f"  {'L':>2} {'channel':>8} {'A/S shown':>10} {'A/S eff':>8} "
            f"{'margin':>9} {'fits':>5} {'frontal walk':>13} {'max |z|':>8}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for level, channel in sorted(env.LEVEL_CHANNEL_WIDTHS.items()):
            fits = (
                _check_wall_collision(
                    np.array([env.WALL_X, 0.0, 0.0]), 0.0, channel
                )
                is None
            )
            reached, max_x, steps, max_z = frontal_walk(channel)
            print(
                f"  {level:>2} {channel:>8.3f} {channel / ROBOT_SHOULDER_WIDTH:>10.3f} "
                f"{channel / required:>8.3f} {(channel - required) / 2.0:>+9.4f} "
                f"{'yes' if fits else 'NO':>5} "
                f"{('PASS x=%.2f (%d steps)' % (max_x, steps)) if reached else 'stops@%.2f' % max_x:>13} "
                f"{max_z:>8.4f}"
            )
        print()

    env.BODY_CLEARANCE = original_skin
    env.LEVEL_CHANNEL_WIDTHS[4] = original_l4
    print("Both options make Level 4 passable by a straight walk.  They differ in")
    print("the printed A/S (1.000 vs 1.007) and in where the collision boundary")
    print("sits for the OTHER Levels: option A moves it 2 mm for every Level,")
    print("option B leaves every other Level untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
