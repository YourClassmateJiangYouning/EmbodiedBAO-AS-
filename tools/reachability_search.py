"""Search the real action space to find how far each Level can actually get.

Earlier probes asked whether a single pose is legal, which is the wrong question:
a pose on the far side of the wall is trivially legal for a robot that could
teleport there, so an isolated far-side pose can appear to have "passed" while
the wall was clearly in the way.  What
matters is whether a collision-free PATH exists, which only the same checks the
simulator uses can answer.

This runs a breadth-first search over the eight actions, using
_apply_move/_apply_turn semantics via environment's own path gates, and reports
for every Level:

  * the largest x reachable at all within the step budget;
  * whether the success plane is reachable;
  * the shortest action count to reach it, so the budget can be judged.

Reported as measurement, not inference, because the pull in opposite directions
-- rotating narrows the lateral footprint through the opening but widens the x
footprint while the shoulders pass the panel -- is not decidable by inspection.

    python tools/reachability_search.py
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import numpy as np

    import environment as env

    start_x = float(env.ROBOT_START_POS[0])
    step = env.MOVE_STEP
    turn = env.TURN_STEP_DEG
    budget = 30
    # Snap states so the search is finite: 0.01 m and 5 degrees.
    qx = 0.01
    qyaw = 5.0

    def key(x: float, z: float, yaw: float):
        return (
            int(round(x / qx)),
            int(round(z / qx)),
            int(round((yaw % 360.0) / qyaw)),
        )

    print(
        "start x=%.2f  step=%.2f  turn=%.0f deg  budget=%d  success x>=%.2f"
        % (start_x, step, turn, budget, env.SUCCESS_X)
    )
    print()
    print(
        "%-6s %-8s %-6s %-11s %-9s %-10s %s"
        % ("level", "channel", "A/S", "max x", "passable", "min steps", "note")
    )
    print("-" * 80)

    for level, width in sorted(env.LEVEL_CHANNEL_WIDTHS.items()):
        start = (start_x, 0.0, 0.0)
        seen = {key(*start)}
        queue = deque([(start, 0)])
        best_x = start_x
        best_steps = None

        while queue:
            (x, z, yaw), depth = queue.popleft()
            if x > best_x:
                best_x = x
            if x >= env.SUCCESS_X - 1e-9 and best_steps is None:
                best_steps = depth
                break
            if depth >= budget:
                continue

            yaw_rad = math.radians(yaw)
            fwd = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)])
            right = np.array(
                [math.sin(yaw_rad), 0.0, math.cos(yaw_rad)]
            )
            moves = {
                "forward": fwd * step,
                "backward": -fwd * step,
                "left": -right * step,
                "right": right * step,
            }
            pos = np.array([x, 0.0, z])

            for _name, delta in moves.items():
                target = pos + delta
                if (
                    env._translation_path_is_clear(pos, target, yaw_rad, width)
                    is None
                ):
                    nxt = (float(target[0]), float(target[2]), yaw)
                    k = key(*nxt)
                    if k not in seen:
                        seen.add(k)
                        queue.append((nxt, depth + 1))

            for d in (turn, -turn):
                new_yaw = yaw + d
                if (
                    env._turn_path_is_clear(pos, yaw, new_yaw, width) is None
                ):
                    nxt = (x, z, new_yaw)
                    k = key(*nxt)
                    if k not in seen:
                        seen.add(k)
                        queue.append((nxt, depth + 1))

        passable = "YES" if best_steps is not None else "NO"
        note = ""
        if best_steps is not None and best_steps > budget:
            note = "exceeds budget"
        print(
            "%-6d %-8.2f %-6.2f %-11.2f %-9s %-10s %s"
            % (
                level,
                width,
                width / env.ROBOT_SHOULDER_WIDTH,
                best_x,
                passable,
                best_steps if best_steps is not None else "-",
                note,
            )
        )

    print()
    print("max x    = furthest body-centre x reached by any collision-free route")
    print("passable = a route reaches x >= SUCCESS_X within the budget")
    print("min steps= shortest such route, from a breadth-first search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
