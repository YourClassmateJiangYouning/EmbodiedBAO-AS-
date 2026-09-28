"""How often does an episode never reach the wall (x >= 7.0)?

Answers one question for STAGE23_DESIGN.md 6.2: if `gap` is the minimum of
``needed(theta) - W`` over the steps with ``x >= 7.0``, how often is that set
empty, and what does a model actually do in those episodes?

Reads the per-step sidecars under ``results/level*/<model>/<tag>/``.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import statistics
import sys

BODY_LENGTH = 0.570
BODY_WIDTH = 0.220
DOOR_X = 7.0
STEP = 0.75
START_X = 0.5

APERTURE = [
    1.140, 1.083, 1.026, 0.969, 0.912, 0.855,
    0.798, 0.741, 0.684, 0.627, 0.570, 0.513,
]


def needed(theta_deg: float) -> float:
    t = math.radians(theta_deg)
    return BODY_LENGTH * abs(math.cos(t)) + BODY_WIDTH * abs(math.sin(t))


def load_episodes(root: str):
    """Yield (level, model, tag, name, steps, passed) from a directory or a .tgz."""
    if root.endswith((".tgz", ".tar.gz")):
        yield from _load_from_tar(root)
        return
    for path in sorted(glob.glob(os.path.join(root, "level*", "*", "*", "episode_*_steps.json"))):
        parts = path.replace("\\", "/").split("/")
        level = int(parts[-4].replace("level", ""))
        model = parts[-3]
        tag = parts[-2]
        record = path.replace("_steps.json", ".json")
        passed = None
        if os.path.exists(record):
            with open(record, encoding="utf-8") as handle:
                passed = json.load(handle).get("passed")
        with open(path, encoding="utf-8") as handle:
            steps = json.load(handle)
        yield level, model, tag, os.path.basename(path), steps, passed


def _load_from_tar(archive: str):
    import tarfile

    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        for name in sorted(names):
            if not name.endswith("_steps.json"):
                continue
            parts = name.split("/")
            if len(parts) != 5 or not parts[1].startswith("level"):
                continue
            level = int(parts[1].replace("level", ""))
            model, tag = parts[2], parts[3]
            record = name.replace("_steps.json", ".json")
            passed = None
            if record in names:
                member = tar.extractfile(record)
                if member is not None:
                    passed = json.loads(member.read().decode("utf-8")).get("passed")
            member = tar.extractfile(name)
            if member is None:
                continue
            steps = json.loads(member.read().decode("utf-8"))
            yield level, model, tag, os.path.basename(name), steps, passed


def min_turns(width: float) -> int:
    for turns in range(0, 7):
        if needed(turns * 15.0) <= width:
            return turns
    return 7


def optimal_steps(width: float) -> int:
    forward = math.ceil((8.75 - START_X) / STEP)
    return forward + min_turns(width)


def main(root: str) -> None:
    per_level = collections.defaultdict(lambda: collections.Counter())
    pass_steps = collections.defaultdict(list)
    fail_steps = collections.defaultdict(list)
    l11_fail_actions = collections.Counter()
    l11_models = collections.defaultdict(lambda: collections.Counter())
    no_door_examples = []

    for level, model, tag, name, steps, passed in load_episodes(root):
        width = APERTURE[level] if level < len(APERTURE) else float("nan")
        max_x = max(step["position_x"] for step in steps)
        reached = max_x >= DOOR_X
        if passed:
            pass_steps[level].append(len(steps))
        elif passed is False:
            fail_steps[level].append(len(steps))
        c = per_level[level]
        c["episodes"] += 1
        c["door"] += int(reached)
        c["fail"] += int(passed is False)
        c["fail_no_door"] += int(passed is False and not reached)
        c["pass"] += int(passed is True)
        # Did it ever hold a posture that would fit, while at the door?
        best = None
        for step in steps:
            if step["position_x"] >= DOOR_X:
                value = needed(step["torso_rotation"]) - width
                best = value if best is None else min(best, value)
        c["fail_door_wrong_posture"] += int(passed is False and reached and (best is None or best > 0))
        c["fail_door_fitting_posture"] += int(passed is False and reached and best is not None and best <= 0)
        c["no_forward"] += int(not any(step["action"] == "forward" for step in steps))
        if level == 11:
            l11_models[model]["episodes"] += 1
            l11_models[model]["no_door"] += int(not reached)
            l11_models[model]["no_forward"] += int(not any(s["action"] == "forward" for s in steps))
            if passed is False:
                actions = collections.Counter(step["action"] for step in steps)
                for key in ("forward", "backward", "left", "right", "turn_left", "turn_right", "look_down"):
                    l11_fail_actions[key] += actions.get(key, 0)
                if not reached:
                    no_door_examples.append((model, name, max_x, dict(actions)))

    print(f"{'level':>5} {'A/S':>5} {'W':>6} {'eps':>4} {'door':>5} {'fail':>5} "
          f"{'f_no_door':>9} {'f_wrong_post':>12} {'f_fit_post':>10} {'no_forward':>10}")
    for level in sorted(per_level):
        c = per_level[level]
        width = APERTURE[level] if level < len(APERTURE) else float("nan")
        ratio = width / BODY_LENGTH
        print(f"{level:>5} {ratio:>5.2f} {width:>6.3f} {c['episodes']:>4} {c['door']:>5} {c['fail']:>5} "
              f"{c['fail_no_door']:>9} {c['fail_door_wrong_posture']:>12} "
              f"{c['fail_door_fitting_posture']:>10} {c['no_forward']:>10}")

    total = sum(c["episodes"] for c in per_level.values())
    door = sum(c["door"] for c in per_level.values())
    fail = sum(c["fail"] for c in per_level.values())
    fail_no_door = sum(c["fail_no_door"] for c in per_level.values())
    total_no_forward = sum(c["no_forward"] for c in per_level.values())
    print()
    print(f"all levels: {total} episodes, {door} reached x>={DOOR_X} "
          f"({100.0 * (total - door) / total:.1f}% never did), {fail} failures, "
          f"{fail_no_door} of them never reached the door "
          f"({100.0 * fail_no_door / max(fail, 1):.1f}% of failures), "
          f"{total_no_forward} episodes never issued a single forward "
          f"({100.0 * total_no_forward / total:.1f}%)")

    print()
    print("level 11 (A/S 0.90) per model: episodes / never reached door / never walked forward")
    for model in sorted(l11_models):
        c = l11_models[model]
        print(f"  {model:<32} {c['episodes']:>3} {c['no_door']:>6} {c['no_forward']:>6}")

    print()
    print("level 11 failed episodes, total actions by kind (sum over failures):")
    for key, value in l11_fail_actions.most_common():
        print(f"  {key:<12} {value:>6}")

    print()
    print(f"level 11 failed episodes that never reached the door: {len(no_door_examples)}")
    for model, name, max_x, actions in no_door_examples:
        print(f"  {model:<30} {name:<20} max_x={max_x:<5.2f} {actions}")

    print()
    print("optimal step count vs what was actually observed (passing rounds only)")
    print(f"{'level':>5} {'W':>6} {'turns':>5} {'optimal':>7} {'min_pass':>8} {'median_pass':>11} {'n_pass':>6}")
    for level in sorted(per_level):
        width = APERTURE[level] if level < len(APERTURE) else float("nan")
        observed = pass_steps[level]
        median = statistics.median(observed) if observed else None
        print(f"{level:>5} {width:>6.3f} {min_turns(width):>5} {optimal_steps(width):>7} "
              f"{(min(observed) if observed else None)!s:>8} {median!s:>11} {len(observed):>6}")

    print()
    print("total_steps of FAILED rounds, by level (a failed round runs to the 30-step cap)")
    for level in sorted(per_level):
        values = sorted(set(fail_steps[level]))
        print(f"  level {level:>2}: n={len(fail_steps[level]):>3} distinct total_steps={values}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
