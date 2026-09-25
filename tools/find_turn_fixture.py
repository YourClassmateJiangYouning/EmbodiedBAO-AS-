"""Search for a pose pair that fits at both ends of a turn but not midway.

test_turn_sweep_checks_intermediate_orientations guards a real property: the turn
gate must sample the swept arc, not only the endpoints, or a robot could rotate a
corner through the room boundary.  Its fixture was chosen when the collision body
carried a 2 mm skin; removing that skin shrank the body by 2 mm and the fixture
stopped crossing anything, so the guard would pass vacuously.

The subtlety that made a first search fail: the gate samples the arc in whole
TURN_STEP_DEG steps, so -60 -> -75 is checked at -60, -65, -70, -75 and never at
the arithmetic midpoint -67.5.  A fixture therefore has to be blocked AT ONE OF
THE SAMPLED ORIENTATIONS, not merely somewhere between the endpoints.

    python tools/find_turn_fixture.py
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
        LEVEL_CHANNEL_WIDTHS,
        TURN_STEP_DEG,
        _check_room_boundary,
        _check_scene_collision,
        _turn_path_is_clear,
    )

    width = LEVEL_CHANNEL_WIDTHS[1]
    print(f"channel {width:.2f} m; turn step {TURN_STEP_DEG} deg")
    print("searching for a pose whose turn is clear at both endpoints but blocked")
    print("at one of the gate's sampled intermediate orientations")
    print()

    candidates = []
    for root_x in (0.2, 0.3, 0.4, 0.5, 0.6):
        for root_z in [round(1.95 + 0.002 * i, 4) for i in range(0, 300)]:
            # Wider arcs, because in a narrow band the footprint's z-extent is
            # monotone in the yaw and a blocked interior orientation is then
            # impossible: at -60 -> -75 the half-extent falls from 0.2378 to
            # 0.1800, so if both endpoints fit the interior must too.  A longer
            # turn passes through orientations where the extent is larger than
            # at either end, which is exactly the case the gate must catch.
            for start_yaw in (0.0, -15.0, -30.0, 15.0, 30.0, 45.0):
                for delta in (90.0, 75.0, 60.0, 45.0, -45.0, -60.0, -75.0, -90.0):
                    end_yaw = start_yaw + delta
                    pose = np.array([root_x, 0.0, root_z])

                    if _check_scene_collision(pose, math.radians(start_yaw), width):
                        continue
                    if _check_scene_collision(pose, math.radians(end_yaw), width):
                        continue

                    # Which sampled orientation is blocked?
                    step = TURN_STEP_DEG if end_yaw >= start_yaw else -TURN_STEP_DEG
                    count = int(round(abs(end_yaw - start_yaw) / TURN_STEP_DEG))
                    blocked_at = [
                        start_yaw + step * i
                        for i in range(0, count + 1)
                        if _check_room_boundary(
                            pose, math.radians(start_yaw + step * i)
                        )
                        is not None
                    ]
                    if not blocked_at:
                        continue

                    collision = _turn_path_is_clear(pose, start_yaw, end_yaw, width)
                    if collision is None or collision.get("boundary") != "near":
                        continue

                    candidates.append(
                        (root_x, root_z, start_yaw, end_yaw, blocked_at)
                    )

    if not candidates:
        print("no fixture found")
        return 1

    # Prefer the fixture furthest from the boundary at its endpoints, so the
    # fixture cannot be invalidated by a rounding change.
    env = sys.modules["environment"]

    def half_z(yaw_deg: float) -> float:
        t = math.radians(yaw_deg)
        c, s = abs(math.cos(t)), abs(math.sin(t))
        return (
            env.ROBOT_TORSO_THICKNESS / 2.0 * s
            + env.ROBOT_SHOULDER_WIDTH / 2.0 * c
        )

    room_half = env.ROOM_WIDTH_Z / 2.0

    def endpoint_margin(c) -> float:
        root_x, root_z, start_yaw, end_yaw, _ = c
        return min(
            room_half - (root_z + half_z(start_yaw)),
            room_half - (root_z + half_z(end_yaw)),
        )

    candidates.sort(key=endpoint_margin, reverse=True)
    print(f"{len(candidates)} candidate(s); best by endpoint margin:")
    for c in candidates[:5]:
        root_x, root_z, start_yaw, end_yaw, blocked_at = c
        print(
            f"  root=({root_x}, 0.0, {root_z})  yaw {start_yaw} -> {end_yaw}  "
            f"blocked at {blocked_at}  endpoint margin {endpoint_margin(c):+.6f}"
        )
    print()
    root_x, root_z, start_yaw, end_yaw, blocked_at = candidates[0]
    print("use this fixture:")
    print(f"    root = np.array([{root_x}, 0.0, {root_z}])")
    print(f"    start_yaw = {start_yaw}")
    print(f"    end_yaw = {end_yaw}")
    print(f"    # blocked at {blocked_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
