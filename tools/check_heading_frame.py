"""Compare the two candidate movement frames, Level by Level.

FRAME "body" (what the code did until this change): forward/backward/left/right
translate along the body's own axes, so a turn redirects the walk.  Passing a
narrow channel therefore means turning 90 degrees and then using right/left -- in
body coordinates that is a sidestep, which is what a human does, but the action
word is not "forward".

FRAME "heading" (the shipped frame, and the trunk-rotation paradigm): translation
is along the heading (here +x, toward the far wall) whatever the torso is doing,
and turn_left/right rotates the torso in place.  Passing a narrow channel means
rotating the torso and continuing to walk forward -- exactly the shoulder
rotation the human A/S literature measures (Warren & Whang 1987).

For each frame this reports, per Level, which torso yaws can actually traverse,
in the agent's own 15 degree turn lattice, and how much lateral aim tolerance
each of those yaws buys.  It finishes by checking that the shipped
``action_delta`` really is the heading column, so this file cannot quietly become
a description of a frame the code no longer uses.

    python tools/check_heading_frame.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import (  # noqa: E402
    LEVEL_CHANNEL_WIDTHS,
    MOVE_STEP,
    ROBOT_SHOULDER_WIDTH,
    ROBOT_TORSO_THICKNESS,
    SUCCESS_X,
    TURN_STEP_DEG,
    _translation_path_is_clear,
    action_delta,
)

START_X = 6.0
MAX_STEPS = 15
SIDE_FINE = 0.005


def traverse(channel_width: float, yaw_deg: float, z0: float, frame: str) -> dict:
    """Walk repeatedly toward the wall from (START_X, z0) at this torso yaw."""
    pos = np.array([START_X, 0.0, float(z0)])
    yaw_rad = np.radians(yaw_deg)
    if frame == "heading":
        delta = np.array([MOVE_STEP, 0.0, 0.0])
    else:
        delta = np.array(
            [np.cos(yaw_rad) * MOVE_STEP, 0.0, -np.sin(yaw_rad) * MOVE_STEP]
        )
    blocked_by = None
    steps = 0
    for _ in range(MAX_STEPS):
        target = pos + delta
        hit = _translation_path_is_clear(pos, target, yaw_rad, channel_width)
        if hit is not None:
            blocked_by = hit["part"]
            break
        pos = target
        steps += 1
        if pos[0] >= SUCCESS_X:
            break
    return {
        "reached": bool(pos[0] >= SUCCESS_X),
        "x": float(pos[0]),
        "z": float(pos[2]),
        "blocked_by": blocked_by,
        "steps": steps,
    }


def max_offset(channel_width: float, yaw_deg: float, frame: str) -> float:
    best = 0.0
    for index in range(int(round(0.35 / SIDE_FINE)) + 1):
        offset = index * SIDE_FINE
        for sign in (1.0, -1.0):
            if traverse(channel_width, yaw_deg, sign * offset, frame)["reached"]:
                best = max(best, offset)
    return best


def first_viable(width: float, frame: str, yaws: list[int]) -> int | None:
    for yaw in yaws:
        if yaw and traverse(width, yaw, 0.0, frame)["reached"]:
            return yaw
    return None


def main() -> int:
    yaws = list(range(0, 91, int(TURN_STEP_DEG)))
    print(f"shoulder {ROBOT_SHOULDER_WIDTH} m   thickness {ROBOT_TORSO_THICKNESS} m")
    print(f"step {MOVE_STEP} m   turn step {TURN_STEP_DEG} deg   success x >= {SUCCESS_X}")
    print(f"each walk starts at x = {START_X}, z = 0, up to {MAX_STEPS} steps")
    print()

    for frame, title in (
        ("body", "FRAME body  (superseded: forward followed the torso)"),
        ("heading", "FRAME heading  (shipped: forward is toward the far wall)"),
    ):
        print(title)
        head = (
            f"   {'L':>2} {'width':>7} {'A/S':>6} "
            + " ".join(f"{y:>4}" for y in yaws)
            + "   viable yaws"
        )
        print(head)
        print("   " + "-" * (len(head) - 3))
        for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
            cells = []
            viable = []
            for yaw in yaws:
                ok = traverse(width, yaw, 0.0, frame)["reached"]
                cells.append(f"{'PASS' if ok else '.':>4}")
                if ok:
                    viable.append(yaw)
            band = ",".join(str(v) for v in viable) if viable else "none"
            print(
                f"   {level:>2} {width:>7.3f} {width / ROBOT_SHOULDER_WIDTH:>6.3f} "
                + " ".join(cells)
                + f"   {band}"
            )
        print()

    print(f"aim tolerance, {int(SIDE_FINE * 1000)} mm grid, at three reference yaws")
    print()
    head = (
        f"   {'frame':>8} {'L':>2} {'width':>7} "
        f"{'yaw 0':>9} {'yaw 90':>9} {'smallest viable yaw':>22}"
    )
    print(head)
    print("   " + "-" * (len(head) - 3))
    for frame in ("body", "heading"):
        for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
            smallest = first_viable(width, frame, yaws)
            if smallest is None:
                tiny = "none above 0 deg"
            else:
                tiny = f"{smallest} deg: {max_offset(width, smallest, frame):.3f} m"
            zero = (
                f"{max_offset(width, 0.0, frame):.3f}"
                if traverse(width, 0.0, 0.0, frame)["reached"]
                else "no pass"
            )
            ninety = (
                f"{max_offset(width, 90.0, frame):.3f}"
                if traverse(width, 90.0, 0.0, frame)["reached"]
                else "no pass"
            )
            print(
                f"   {frame:>8} {level:>2} {width:>7.3f} {zero:>9} {ninety:>9} "
                f"{tiny:>22}"
            )
        print()

    print("what the two frames imply for the experiment")
    print()
    print("  body frame:    a turn redirects the walk, so only yaw 0 and yaw 90")
    print("                 traverse at ANY Level -- at every intermediate yaw a")
    print("                 0.75 m step slides the body sideways by more than the")
    print("                 channel can absorb.  Rotating is therefore never a")
    print("                 graded adjustment: it is 0 or 90, all-or-nothing, and")
    print("                 the model must then discover that the gap is beside it.")
    print()
    print("  heading frame: the walk never drifts, so the viable set is exactly")
    print("                 the geometric one -- every yaw where the body is no")
    print("                 wider than the opening.  Rotating becomes a graded")
    print("                 shoulder rotation measured in 15 degree steps, which is")
    print("                 the quantity the human A/S literature reports.")
    print()
    print("minimum rotation that traverses each Level, in the 15 degree lattice")
    print()
    shipped = {
        name: action_delta(name, MOVE_STEP)
        for name in ("forward", "backward", "left", "right")
    }
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        best = None
        for yaw in yaws:
            if traverse(width, float(yaw), 0.0, "heading")["reached"]:
                best = yaw
                break
        if best is None:
            print(f"  Level {level}: not traversable at any yaw up to 90 deg")
        elif best == 0:
            print(f"  Level {level}: 0 deg -- walking straight is enough")
        else:
            print(
                f"  Level {level}: {best} deg ({int(best / TURN_STEP_DEG)} turns) "
                f"then forward"
            )

    # The frame must be the one the code actually implements.
    check_deltas = {
        "forward": np.array([MOVE_STEP, 0.0, 0.0]),
        "backward": np.array([-MOVE_STEP, 0.0, 0.0]),
        "right": np.array([0.0, 0.0, MOVE_STEP]),
        "left": np.array([0.0, 0.0, -MOVE_STEP]),
    }
    for name, expected in check_deltas.items():
        got = shipped[name]
        if got is not None and not np.allclose(got, expected):
            print()
            print(f"  MISMATCH: shipped action_delta({name!r}) = {got}")
            print(f"            the heading column above assumes {expected}")
            return 1
    print()
    print("cross-check: the shipped action_delta matches the heading column above")

    # Are the two sidestep actions ever useful?  Translation is axis-aligned and
    # the channel is centred on z = 0, so the reachable lateral positions are
    # exactly the multiples of MOVE_STEP.  If none of those is passable, then a
    # sidestep can only ever move the agent off the line and the model has to
    # undo it exactly.
    print()
    print("is a sidestep ever useful?  pass/fail by lateral offset (metres)")
    head = f"  {'L':>2} " + " ".join(f"{k * MOVE_STEP:>+8.2f}" for k in (0, 1, 2))
    print(head)
    print("  " + "-" * (len(head) - 2))
    useful_anywhere = False
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        cells = []
        for k in (0, 1, 2):
            offset = k * MOVE_STEP
            # Best case over the yaws that traverse at all: a sidestep that only
            # works when the torso is already turned still counts as useful.
            ok = any(
                traverse(width, float(yaw), offset, "heading")["reached"]
                for yaw in yaws
            )
            cells.append(f"{'pass' if ok else '.':>8}")
            if ok and k > 0:
                useful_anywhere = True
        print(f"  {level:>2} " + " ".join(cells))
    print()
    if useful_anywhere:
        print("  A sidestep is passable somewhere, so left/right can be part of a route.")
    else:
        print("  No nonzero multiple of the sidestep is passable at any Level, in any")
        print("  orientation: with the channel centred on the start line, left/right")
        print("  can only move the agent off it, and the displacement has to be undone")
        print("  exactly.  They are distractors, not tools -- worth knowing before")
        print("  reading a trace that uses them as a mistake, which it is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
