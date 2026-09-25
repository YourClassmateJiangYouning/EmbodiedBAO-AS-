"""What each Level actually demands of the body: orientation and aim.

Three measurements, all with the shipped translation gate:

  1. crossing table -- at a fixed torso yaw, walk one repeated action from 2 m in
     front of the wall.  A pose that fits the opening is not yet a passage: every
     translation step at a non-zero yaw also moves the body sideways, and the
     panel is only 0.02 m thick, so a rotated body can slip a corner *past* the
     plate and still be stopped one step later.  Only this test settles it.

  2. frontal aim tolerance -- with the body facing the gap, how far off centre can
     it start and still cross?  This is the real "margin" of a Level, and the
     number that says whether A/S = 1.00 is a knife edge.

  3. turns needed -- read off the crossing table, in the agent's own 15 degree
     turn lattice.

    python tools/check_passage.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment as env  # noqa: E402
from environment import (  # noqa: E402
    LEVEL_CHANNEL_WIDTHS,
    MOVE_STEP,
    ROBOT_SHOULDER_WIDTH,
    ROBOT_TORSO_THICKNESS,
    SUCCESS_X,
    TURN_STEP_DEG,
    _translation_path_is_clear,
)

BODY_DIRECTIONS = {
    "forward": lambda t: (np.cos(t), -np.sin(t)),
    "backward": lambda t: (-np.cos(t), np.sin(t)),
    "right": lambda t: (np.sin(t), np.cos(t)),
    "left": lambda t: (-np.sin(t), -np.cos(t)),
}

START_X = 6.0
MAX_STEPS = 15
TOL = 1e-12
FINE = 0.001


def cross(channel_width: float, yaw_deg: float, direction: str, z0: float = 0.0) -> dict:
    """Walk one repeated action from (START_X, z0) and report how far it got."""
    pos = np.array([START_X, 0.0, float(z0)])
    yaw_rad = np.radians(yaw_deg)
    dx, dz = BODY_DIRECTIONS[direction](yaw_rad)
    delta = np.array([dx, 0.0, dz]) * MOVE_STEP
    blocked_by = None
    for _ in range(MAX_STEPS):
        target = pos + delta
        hit = _translation_path_is_clear(pos, target, yaw_rad, channel_width)
        if hit is not None:
            blocked_by = hit["part"]
            break
        pos = target
        if pos[0] >= SUCCESS_X:
            break
    return {
        "reached": bool(pos[0] >= SUCCESS_X - TOL),
        "x": float(pos[0]),
        "z": float(pos[2]),
        "blocked_by": blocked_by,
    }


def max_offset(channel_width: float, direction: str = "forward", yaw_deg: float = 0.0) -> float:
    """Largest |z0| from which this walk still crosses, on a 1 mm grid."""
    best = 0.0
    steps = int(round(0.300 / FINE))
    for index in range(steps + 1):
        offset = index * FINE
        for sign in (1.0, -1.0):
            if cross(channel_width, yaw_deg, direction, sign * offset)["reached"]:
                best = max(best, offset)
    return best


def main() -> int:
    yaws = list(range(0, 91, int(TURN_STEP_DEG)))
    print(f"shoulder {ROBOT_SHOULDER_WIDTH} m   thickness {ROBOT_TORSO_THICKNESS} m")
    print(f"step {MOVE_STEP} m   turn step {TURN_STEP_DEG} deg   success x >= {SUCCESS_X}")
    print(f"crossing attempts start at x = {START_X}, up to {MAX_STEPS} repeated steps")
    print(f"body inflation in the gate: {env.BODY_CLEARANCE * 1000:.1f} mm per side")
    print()

    print("1) crossing: one repeated action at a fixed yaw, starting on the centre line")
    print()
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        print(
            f"   Level {level}  width {width:.3f}  A/S {width / ROBOT_SHOULDER_WIDTH:.3f}"
        )
        print(f"   {'yaw':>5} " + " ".join(f"{d:>11}" for d in BODY_DIRECTIONS))
        for yaw in yaws:
            cells = []
            for direction in BODY_DIRECTIONS:
                result = cross(width, yaw, direction)
                if result["reached"]:
                    cell = "PASS"
                else:
                    part = "wall" if result["blocked_by"] == "body" else "room"
                    cell = f"{part}@{result['x']:.1f}"
                cells.append(f"{cell:>11}")
            print(f"   {yaw:>5} " + " ".join(cells))
        print()
    print("   PASS = that repeated action carried the body past the success plane")
    print("   wall@ = the body itself was too wide where it crossed")
    print("   room@ = wrong direction: it hit the room boundary first")
    print()

    print("2) frontal aim tolerance: how far off the centre line may the body start")
    print("   and still cross, walking forward with the torso unturned")
    print()
    head = (
        f"   {'L':>2} {'width':>7} {'A/S':>6} {'frontal?':>9} {'max |z|':>9} "
        f"{'walk stops at |z|':>18}"
    )
    print(head)
    print("   " + "-" * (len(head) - 3))
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        frontal = cross(width, 0.0, "forward")["reached"]
        if not frontal:
            print(
                f"   {level:>2} {width:>7.3f} {width / ROBOT_SHOULDER_WIDTH:>6.3f} "
                f"{'NO':>9} {'-':>9} {'cannot cross unturned':>18}"
            )
            continue
        tolerance = max_offset(width)
        stop = tolerance + FINE
        if not cross(width, 0.0, "forward", stop)["reached"]:
            stops = f"{stop:.3f} m"
        else:
            stops = "not within 0.300 m"
        print(
            f"   {level:>2} {width:>7.3f} {width / ROBOT_SHOULDER_WIDTH:>6.3f} "
            f"{'yes':>9} {tolerance:>9.3f} {stops:>18}"
        )
    print()

    print("3) turns needed, in the 15 degree lattice, read off section 1")
    print()
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        straight = cross(width, 0.0, "forward")["reached"]
        if straight:
            print(
                f"   Level {level}: 0 turns -- walk straight through "
                f"(a strafe at 90 deg also works, but is not needed)"
            )
        else:
            best = None
            for k in range(1, 7):
                yaw = k * TURN_STEP_DEG
                for direction in ("forward", "right", "left", "backward"):
                    if cross(width, yaw, direction)["reached"]:
                        if best is None or k < best[0]:
                            best = (k, yaw, direction)
            if best is None:
                print(f"   Level {level}: no crossing found within 90 deg of turning")
            else:
                k, yaw, direction = best
                print(
                    f"   Level {level}: {k} turns ({yaw:.0f} deg) then repeated "
                    f"'{direction}' -- walking forward after the turn would move "
                    f"the body along the wall, not through the gap"
                )
    print()

    print("4) what a 2 mm body inflation would buy at Level 4, if the channel were")
    print("   widened by that same 2 mm instead of the inflation being removed")
    print()
    saved_skin = env.BODY_CLEARANCE
    saved_width = env.LEVEL_CHANNEL_WIDTHS[4]
    print(f"   {'option':<38} {'A/S shown':>10} {'A/S eff':>8} {'tolerance':>10}")
    for name, skin, width in (
        ("A now: skin 0.000 mm, L4 0.570 m", 0.0, 0.570),
        ("B widen: skin 2.000 mm, L4 0.574 m", 0.002, 0.574),
    ):
        env.BODY_CLEARANCE = skin
        env.LEVEL_CHANNEL_WIDTHS[4] = width
        required = ROBOT_SHOULDER_WIDTH + 2.0 * skin
        print(
            f"   {name:<38} {width / ROBOT_SHOULDER_WIDTH:>10.3f} "
            f"{width / required:>8.3f} {max_offset(width):>10.3f}"
        )
    env.BODY_CLEARANCE = saved_skin
    env.LEVEL_CHANNEL_WIDTHS[4] = saved_width
    print()
    print("   Both options leave the aim tolerance at exactly zero: widening the")
    print("   channel by precisely the inflation just relabels the same geometry,")
    print("   it does not reserve an edge.  They differ only in the printed A/S")
    print("   (1.000 vs 1.007) and in whether the other five Levels keep the")
    print("   collision boundary they had when the earlier data was recorded.")
    print()
    print("   Neither makes a partial turn survivable either: at 15 deg the body")
    print("   is 0.607 m wide, and no channel below that admits it at any offset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
