"""Summarise every completed model into one A/S threshold table.

Reads the episode JSON directly rather than the progress log, so the numbers come
from the records rather than from a text file, and reports per Level: pass rate,
sideways rate, the largest torso rotation reached, and the first turn step.

max|yaw| matters because it separates two very different failures: a model that
never rotates is not trying, while a model that rotates past 90 degrees and still
fails is trying the right thing at the wrong time.

    python tools/summarise_models.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    from environment import LEVEL_CHANNEL_WIDTHS, ROBOT_SHOULDER_WIDTH, SUCCESS_X

    results_root = os.path.join(ROOT, "results")
    # model -> tag is discovered from the tree: results/level*/{model}/{tag}/
    found: dict = {}
    for level_dir in sorted(glob.glob(os.path.join(results_root, "level*"))):
        level = os.path.basename(level_dir)
        for model_dir in sorted(glob.glob(os.path.join(level_dir, "*"))):
            if not os.path.isdir(model_dir):
                continue
            model = os.path.basename(model_dir)
            for tag_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
                if not os.path.isdir(tag_dir):
                    continue
                tag = os.path.basename(tag_dir)
                episodes = [
                    p
                    for p in glob.glob(os.path.join(tag_dir, "episode_*.json"))
                    if not p.endswith("_steps.json")
                ]
                if episodes:
                    found.setdefault((model, tag), {}).setdefault(level, episodes)

    if not found:
        print("no episode records found under results/")
        return 1

    print(
        "threshold = widest A/S (smallest Level index) with any successful passage"
    )
    print(f"success plane x>{SUCCESS_X}, shoulder width {ROBOT_SHOULDER_WIDTH} m")
    print()

    summary_rows = []
    for (model, tag) in sorted(found):
        levels = found[(model, tag)]
        total_eps = sum(len(v) for v in levels.values())
        print("=" * 104)
        print(f"{model}    tag={tag}    {total_eps} episode(s) over {len(levels)} Level(s)")
        print("=" * 104)
        print(
            f"  {'L':<3} {'channel':>8} {'A/S':>6} {'pass':>8} {'sideways':>9} "
            f"{'max|yaw|':>9} {'1st turn':>9} {'avg steps':>10}"
        )
        best_threshold = None
        for level_name in sorted(levels, key=lambda s: int(s.replace("level", ""))):
            level = int(level_name.replace("level", ""))
            eps = [json.load(open(p, encoding="utf-8")) for p in levels[level_name]]
            n = len(eps)
            ok = sum(1 for e in eps if e.get("passed"))
            sw = sum(1 for e in eps if e.get("passed") and e.get("passed_sideways"))
            steps = [e["total_steps"] for e in eps if e.get("passed")]
            yaw_max = max(
                (max(abs(s["torso_rotation"]) for s in e["steps"]) for e in eps),
                default=0.0,
            )
            turns = [
                e["first_turn_step"]
                for e in eps
                if e.get("first_turn_step") is not None
            ]
            channel = LEVEL_CHANNEL_WIDTHS.get(level, float("nan"))
            ratio = channel / ROBOT_SHOULDER_WIDTH
            if ok and (best_threshold is None or ratio < best_threshold):
                best_threshold = ratio
            print(
                f"  {level:<3} {channel:>8.2f} {ratio:>6.2f} "
                f"{ok:>4}/{n:<3} {sw:>4}/{n:<4} {yaw_max:>9.0f} "
                f"{(sum(turns) / len(turns) if turns else float('nan')):>9.1f} "
                f"{(sum(steps) / len(steps) if steps else float('nan')):>10.1f}"
            )
        print(
            f"  -> threshold A/S: "
            f"{best_threshold:.2f}" if best_threshold else "  -> threshold A/S: none"
        )
        print()
        summary_rows.append((model, best_threshold))

    print("=" * 104)
    print("THRESHOLDS (human reference 1.30)")
    print("=" * 104)
    for model, threshold in sorted(summary_rows, key=lambda r: -(r[1] or 0)):
        value = f"{threshold:.2f}" if threshold else "no passage at any Level"
        gap = f"{threshold - 1.30:+.2f}" if threshold else "-"
        print(f"  {model:<36} {value:>10}   vs human {gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
