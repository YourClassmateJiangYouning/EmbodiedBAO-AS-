"""Summarise every completed run into one A/S threshold table, with integrity counts.

Reads the episode JSON directly rather than the progress log, so the numbers come
from the records rather than from a text file, and reports per Level: pass rate,
sideways rate, the largest torso rotation reached, the first turn step, invalid
responses and how the episodes ended.

max|yaw| matters because it separates two very different failures: a model that
never rotates is not trying, while a model that rotates past 90 degrees and still
fails is trying the right thing at the wrong time.

The channel width and A/S come from EACH RECORD, not from the current ladder.  They
used to be looked up by Level index, which silently relabelled every run recorded
under an older ladder with the current ladder's widths -- so 6-Level runs from an
earlier protocol appeared as the first six rows of the 12-width series, and their
pass rates were read as one continuous curve.

Integrity is reported per run because "is my data saved properly" is a question
worth answering from the tree itself: episodes found, files that will not parse,
files the runner quarantined as corrupt, and gaps in the episode id sequence.

    python tools/summarise_models.py
    python tools/summarise_models.py --tag qwen3-vl-235b-a22b-instruct-v5-12widths
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=None, help="only this run tag")
    parser.add_argument("--model", default=None, help="only this model")
    args = parser.parse_args()

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
                if args.tag and tag != args.tag:
                    continue
                if args.model and model != args.model:
                    continue
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

    print(f"success plane x>{SUCCESS_X}, shoulder width {ROBOT_SHOULDER_WIDTH} m")
    print(
        "channel/A-S are read from each RECORD, so runs from different ladders are "
        "not relabelled"
    )
    print()

    summary_rows = []
    grand_episodes = 0
    grand_steps = 0
    grand_corrupt = 0
    for (model, tag) in sorted(found):
        levels = found[(model, tag)]
        all_paths = [p for paths in levels.values() for p in paths]
        unreadable = 0
        corrupt = 0
        unreadable_paths = []
        episodes = {}
        for path in all_paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    episodes[path] = json.load(handle)
            except (OSError, ValueError):
                unreadable += 1
                unreadable_paths.append(os.path.basename(path))
        sidecars = 0
        for dirpath in {os.path.dirname(p) for p in all_paths}:
            corrupt += len(glob.glob(os.path.join(dirpath, "*.corrupt-*")))
            sidecars += len(glob.glob(os.path.join(dirpath, "episode_*_steps.json")))
        grand_episodes += len(all_paths)
        grand_steps += sidecars
        grand_corrupt += corrupt
        gaps = []
        for level_name, paths in sorted(levels.items()):
            ids = sorted(
                int(os.path.basename(p).replace("episode_", "").replace(".json", ""))
                for p in paths
            )
            expected = list(range(len(ids)))
            if ids != expected:
                gaps.append(f"{level_name}:{ids}")
        print("=" * 104)
        print(
            f"{model}    tag={tag}    {len(all_paths)} episode(s) over "
            f"{len(levels)} Level(s)"
        )
        print(
            f"  integrity: {len(all_paths)} found, {sidecars} step sidecar(s)"
            f"{'' if sidecars == len(all_paths) else '  <-- MISMATCH'}, "
            f"{unreadable} unreadable, {corrupt} quarantined as corrupt, "
            f"id gaps: {', '.join(gaps) if gaps else 'none'}"
        )
        if unreadable_paths:
            print(f"  UNREADABLE: {', '.join(unreadable_paths[:6])}")
        print(
            f"  {'L':<3} {'channel':>8} {'A/S':>6} {'pass':>8} {'sideways':>9} "
            f"{'max|yaw|':>9} {'1st turn':>9} {'avg steps':>10} {'invalid':>8}  end"
        )
        best_threshold = None
        for level_name in sorted(levels, key=lambda s: int(s.replace("level", ""))):
            level = int(level_name.replace("level", ""))
            eps = [
                episodes[p] for p in levels[level_name] if p in episodes
            ]
            if not eps:
                continue
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
            invalid = sum(int(e.get("invalid_response_count", 0) or 0) for e in eps)
            ends = {}
            for e in eps:
                key = str(e.get("end_reason"))
                ends[key] = ends.get(key, 0) + 1
            # The record's own width, not the current ladder's.
            channel = eps[0].get("channel_width")
            if channel is None:
                channel = LEVEL_CHANNEL_WIDTHS.get(level, float("nan"))
            ratio = eps[0].get("a_s_ratio")
            if ratio is None:
                ratio = float(channel) / ROBOT_SHOULDER_WIDTH
            if ok and (best_threshold is None or float(ratio) < best_threshold):
                best_threshold = float(ratio)
            print(
                f"  {level:<3} {float(channel):>8.2f} {float(ratio):>6.2f} "
                f"{ok:>4}/{n:<3} {sw:>4}/{n:<4} {yaw_max:>9.0f} "
                f"{(sum(turns) / len(turns) if turns else float('nan')):>9.1f} "
                f"{(sum(steps) / len(steps) if steps else float('nan')):>10.1f} "
                f"{invalid:>8}  "
                f"{', '.join(f'{k}:{v}' for k, v in sorted(ends.items()))}"
            )
        print(
            f"  -> threshold A/S: "
            f"{best_threshold:.2f}" if best_threshold else "  -> threshold A/S: none"
        )
        print()
        summary_rows.append((f"{model} [{tag}]", best_threshold))

    print("=" * 104)
    print(
        f"TOTAL: {grand_episodes} episode record(s), {grand_steps} step sidecar(s), "
        f"{grand_corrupt} quarantined file(s), {len(found)} run(s)"
    )
    print(
        "  the two counts must match each other, and the episode count must match "
        "`find results -name 'episode_*.json' ! -name '*_steps.json' | wc -l` taken "
        "at the same moment -- the tree grows while a sweep is running"
    )
    print("=" * 104)
    print("THRESHOLDS (human reference 1.30) -- one line per RUN, not per model:")
    print("  runs under different ladders are different experiments, which is why")
    print("  the tag is printed here and why the widths above come from the records")
    print("=" * 104)
    for name, threshold in sorted(summary_rows, key=lambda r: -(r[1] or 0)):
        value = f"{threshold:.2f}" if threshold else "no passage at any Level"
        gap = f"{threshold - 1.30:+.2f}" if threshold else "-"
        print(f"  {name:<58} {value:>10}   vs human {gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
