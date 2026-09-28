"""Breadth-first search over the real actions at the two Stage 2 widths.

`tools/reachability_search.py` does this for the 12-Level ladder.  Stage 2 uses
two widths that are not on that ladder, so this repeats the same search at
A/S 0.80 (0.456 m) and A/S 1.10 (0.627 m) and reports the shortest collision-free
action sequence together with the rotation it ends up using.

The arithmetic prediction is 11 forward steps (8.25 m / 0.75 m) plus the fewest
15-degree turns whose body projection fits the opening.  A search is worth running
anyway because a rotation widens the footprint in x while narrowing it in z, and
whether that helps or hurts at the panels is not decidable by inspection.

    python lab_logs/verify_optimal_steps.py
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHOULDER = 0.570
TORSO = 0.220


def needed(theta_deg: float) -> float:
    t = math.radians(theta_deg)
    return SHOULDER * abs(math.cos(t)) + TORSO * abs(math.sin(t))


def analytic(width: float) -> int:
    forward = math.ceil((8.75 - 0.5) / 0.75)
    for turns in range(0, 9):
        if needed(turns * 15.0) <= width:
            return forward + turns
    return -1


def search(width: float, budget: int = 30):
    import numpy as np

    import environment as env

    start = (float(env.ROBOT_START_POS[0]), 0.0, 0.0)
    qx, qyaw = 0.01, 5.0

    def key(x, z, yaw):
        return (int(round(x / qx)), int(round(z / qx)), int(round((yaw % 360.0) / qyaw)))

    seen = {key(*start): None}
    queue = deque([(start, 0, [])])
    while queue:
        (x, z, yaw), depth, path = queue.popleft()
        if x >= env.SUCCESS_X - 1e-9:
            return depth, path
        if depth >= budget:
            continue
        moves = {
            name: env.action_delta(name, env.MOVE_STEP)
            for name in ("forward", "backward", "left", "right")
        }
        pos = np.array([x, 0.0, z])
        for name, delta in moves.items():
            target = pos + delta
            if env._translation_path_is_clear(pos, target, math.radians(yaw), width) is None:
                nxt = (float(target[0]), float(target[2]), yaw)
                k = key(*nxt)
                if k not in seen:
                    seen[k] = None
                    queue.append((nxt, depth + 1, path + [name]))
        for name, d in (("turn_left", env.TURN_STEP_DEG), ("turn_right", -env.TURN_STEP_DEG)):
            new_yaw = yaw + d
            if env._turn_path_is_clear(pos, yaw, new_yaw, width) is None:
                nxt = (x, z, new_yaw)
                k = key(*nxt)
                if k not in seen:
                    seen[k] = None
                    queue.append((nxt, depth + 1, path + [name]))
    return None, None


def main() -> int:
    print("Stage 2 widths, searched over the real action space")
    print(f"{'A/S':>5} {'width':>7} {'needed(0)':>10} {'analytic':>9} {'BFS':>5} {'frontal?':>9}  sequence")
    for ratio in (0.80, 1.10, 0.90):
        width = SHOULDER * ratio
        steps, path = search(width)
        frontal = "yes" if needed(0.0) <= width else "no"
        turns = sum(1 for a in (path or []) if a.startswith("turn"))
        forward = sum(1 for a in (path or []) if a == "forward")
        summary = f"{forward} forward + {turns} turn"
        print(f"{ratio:>5.2f} {width:>7.3f} {needed(0.0):>10.4f} {analytic(width):>9} "
              f"{steps!s:>5} {frontal:>9}  {summary}")
        if path:
            print(f"      one shortest sequence: {','.join(path)}")
    print()
    print("needed(theta) = 0.570*|cos| + 0.220*|sin|; a route exists iff needed(theta) <= width "
          "for whatever theta it holds through the panels")
    print("frontal? = whether a straight walk fits, which is what the A/S 1.10 probe relies on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
