"""Is each Level actually passable, using the real collision model?

A six-Level sweep produced exactly one pass, at Level 3, which is suspicious: the
widest channels should be the easiest.  Before blaming the model, this checks
whether the geometry even permits a passage at each Level, so a capability
conclusion is not drawn from an impossible task.

It reuses environment._check_wall_collision rather than re-deriving the geometry,
and answers two questions per Level:

  * what rotation is needed for the body to fit through the opening at all;
  * the furthest x the body centre can reach at that rotation, versus the
    success plane.

Note the pull in opposite directions: rotating lets the body's long axis carry
the centre further along x, but it also widens the body's x footprint while the
shoulders are still passing the panel.  Which effect wins is not obvious by
inspection, which is why this is measured.

    python tools/passability_probe.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import numpy as np

    from environment import (
        BODY_CLEARANCE,
        LEVEL_CHANNEL_WIDTHS,
        MOVE_STEP,
        ROBOT_SHOULDER_WIDTH,
        ROBOT_TORSO_THICKNESS,
        ROBOT_START_POS,
        SUCCESS_X,
        WALL_THICKNESS,
        WALL_X,
        _check_wall_collision,
        action_delta,
    )

    bt = ROBOT_TORSO_THICKNESS / 2.0 + BODY_CLEARANCE
    bw = ROBOT_SHOULDER_WIDTH / 2.0 + BODY_CLEARANCE
    start_x = float(ROBOT_START_POS[0])

    def legal(x: float, yaw_deg: float, width: float) -> bool:
        return (
            _check_wall_collision(
                np.array([x, 0.0, 0.0]), math.radians(yaw_deg), width
            )
            is None
        )

    def max_x_at(yaw_deg: float, width: float) -> float:
        """Furthest root x on a fixed-yaw, collision-free action-grid path."""
        position = np.array([start_x, 0.0, 0.0])
        yaw = math.radians(yaw_deg)
        # The shipped walking frame: forward is toward the far wall whatever the
        # torso is doing, so the advancing action is simply "forward".
        direction = action_delta("forward", MOVE_STEP)
        for _ in range(30):
            target = position + direction
            from environment import _translation_path_is_clear
            if _translation_path_is_clear(position, target, yaw, width) is not None:
                break
            position = target
            if position[0] >= SUCCESS_X:
                break
        return float(position[0])

    print(
        "body half-extents: along facing %.3f, lateral %.3f (incl %.3f clearance)"
        % (bt, bw, BODY_CLEARANCE)
    )
    print(
        "panel near face x=%.3f   success plane x=%.2f   step %.2f   start %.2f"
        % (WALL_X - WALL_THICKNESS / 2.0, SUCCESS_X, MOVE_STEP, start_x)
    )
    print()
    print(
        "%-6s %-8s %-6s %-10s %-9s %-9s %-11s %s"
        % ("level", "channel", "A/S", "fit angle", "max x@90", "best x", "best yaw", "passable")
    )
    print("-" * 96)

    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        # Smallest rotation at which the body fits laterally through the channel.
        fit = None
        for deg in range(0, 91):
            t = math.radians(deg)
            lateral = bw * abs(math.cos(t)) + bt * abs(math.sin(t))
            if lateral <= width / 2.0:
                fit = deg
                break

        # Best reachable x over all yaws, and the yaw that achieves it.
        best_x, best_yaw = -1.0, 0
        for deg in range(0, 181, 5):
            mx = max_x_at(deg, width)
            if mx == mx and mx > best_x:  # NaN-safe
                best_x, best_yaw = mx, deg

        x90 = max_x_at(90.0, width)
        passable = "YES" if best_x >= SUCCESS_X - 1e-9 else "NO"
        print(
            "%-6d %-8.2f %-6.2f %-10s %-9.2f %-9.2f %-11s %s"
            % (
                level,
                width,
                width / ROBOT_SHOULDER_WIDTH,
                ("%d deg" % fit) if fit is not None else "impossible",
                x90,
                best_x,
                "%d deg" % best_yaw,
                passable,
            )
        )

    print()
    print("max x@90  = furthest centre x when fully sideways")
    print("best x    = furthest centre x over every yaw, on the action grid")
    print("passable  = best x reaches the success plane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
