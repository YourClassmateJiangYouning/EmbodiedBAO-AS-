"""Report the width the collision model actually demands at each Level.

A/S is defined as channel width divided by the NOMINAL shoulder width (0.57 m),
and that is the number recorded in every episode record and printed in every
table.  The collision model, however, tests a body inflated by BODY_CLEARANCE on
each side, so the width it requires is 1.4 mm per side larger than the nominal
figure.  At most Levels the difference is irrelevant; at A/S = 1.00 it decides
the outcome, because the gap and the body are then nominally equal and the
inflation makes the body strictly wider.

This prints both numbers per Level so the discrepancy is visible rather than
assumed, and reports which Levels a straight frontal walk is geometrically
capable of clearing.

    python tools/check_clearance.py
"""

from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    import numpy as np

    from environment import (
        BODY_CLEARANCE,
        LEVEL_CHANNEL_WIDTHS,
        ROBOT_SHOULDER_WIDTH,
        ROBOT_START_POS,
        ROBOT_TORSO_THICKNESS,
        SUCCESS_X,
        WALL_THICKNESS,
        WALL_X,
        _check_wall_collision,
    )

    required = ROBOT_SHOULDER_WIDTH + 2.0 * BODY_CLEARANCE
    print(f"nominal shoulder width        {ROBOT_SHOULDER_WIDTH:.4f} m")
    print(f"BODY_CLEARANCE per side       {BODY_CLEARANCE:.4f} m")
    print(f"width the collision model asks {required:.4f} m  "
          f"({required - ROBOT_SHOULDER_WIDTH:+.4f} m vs nominal)")
    print()
    print(
        f"{'L':<3} {'channel':>8} {'A/S shown':>10} {'A/S effective':>14} "
        f"{'margin':>9} {'frontal walk':>13}"
    )
    print("-" * 76)

    for level, channel in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        shown = channel / ROBOT_SHOULDER_WIDTH
        effective = channel / required
        margin = (channel - required) / 2.0
        # Can a body facing +x walk through?  Tested with the real gate.
        can_walk = (
            _check_wall_collision(
                np.array([WALL_X, 0.0, 0.0]), 0.0, channel
            )
            is None
        )
        print(
            f"{level:<3} {channel:>8.2f} {shown:>10.2f} {effective:>14.3f} "
            f"{margin:>+9.4f} {'yes' if can_walk else 'NO':>13}"
        )

    print()
    print("A/S shown     = channel / 0.570, the value recorded in every episode")
    print("A/S effective = channel / (0.570 + 2 * clearance), what the geometry allows")
    print("margin        = free space per side; negative means it cannot fit at all")
    print()
    print(
        "The gap between the two columns is 4 mm of body inflation.  It is "
        "invisible above A/S 1.0 and decisive at Level 4, where the channel and "
        "the nominal shoulder are equal: the inflated body is then 2 mm too wide "
        "per side, so a frontal passage is geometrically impossible and the "
        "recorded 1.00 is really 0.993."
    )
    print()
    print("For frontal REACHABILITY see tools/check_frontal.py, which walks the")
    print("action grid with the real translation gate.  An earlier version of")
    print("this file reported that a straight walk stops at x=10.25 even at")
    print("Level 0, which is wrong: it is 12.50, well past the plane at 11.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
