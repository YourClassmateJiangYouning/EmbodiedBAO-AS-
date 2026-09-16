"""A/S threshold analysis for EmbodiedBAO.

Consumes the per-episode JSON files written by ``experiments.py``
(``results/level{level}/{model}/episode_*.json``) and reports, for each model,
how its behaviour changes as the channel narrows.

Headline metrics (per Level, and per model overall):

* **Pass rate**       -- fraction of episodes that got the body past x > 2.5 m.
* **Sideways rate**   -- fraction of episodes that passed while the torso was
  rotated into the sideways band (45-135 degrees).
* **Sideways threshold** -- the widest A/S ratio (largest channel) at which the
  model still chose to pass sideways.  This is the number to compare against
  the human threshold of **1.30** (Warren & Whang, 1987).  A model that only
  goes sideways once the channel is already narrower than its shoulders has
  threshold < 1.0, i.e. it reacts to contact instead of anticipating it.
* **First turn step** -- mean step index of the first turn, i.e. how early the
  agent starts re-orienting.
* **Mean pass steps**  -- average steps needed for a successful passage.

The classification of a model mirrors the affordance-perception question:

    anticipatory   threshold >= 1.30   rotates before the channel is tight
    borderline     1.00 <= threshold < 1.30   rotates while still passable
    reactive       threshold <  1.00   only rotates after being blocked

Outputs: a per-model JSON report, a CSV table, a Markdown table, and (unless
``--no_plot``) a threshold plot with the human 1.30 reference marked.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HUMAN_THRESHOLD = 1.30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episodes(results_root: str, level: int, model: str) -> List[Dict[str, Any]]:
    """Load and sort per-episode JSON files for one model at one Level."""
    model_dir = os.path.join(results_root, f"level{level}", model)
    paths = sorted(glob.glob(os.path.join(model_dir, "episode_*.json")))
    # Exclude the per-step sidecars written next to the episode records.
    paths = [p for p in paths if not p.endswith("_steps.json")]
    if not paths:
        # Legacy layout: results/level{level}/{model}/round*/episode_*.json
        paths = sorted(
            glob.glob(os.path.join(model_dir, "round*", "episode_*.json"))
        )
        paths = [p for p in paths if not p.endswith("_steps.json")]

    episodes: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            episode = json.load(handle)
        if not episode.get("steps"):
            sidecar = path[: -len(".json")] + "_steps.json"
            if os.path.exists(sidecar):
                with open(sidecar, "r", encoding="utf-8") as handle:
                    episode["steps"] = json.load(handle)
        episodes.append(episode)
    episodes.sort(key=lambda ep: int(ep.get("episode_id", 0)))
    return episodes


def discover_models(results_root: str) -> List[str]:
    """List models that have episode results under any Level."""
    models: set[str] = set()
    for level_dir in glob.glob(os.path.join(results_root, "level*")):
        if not os.path.isdir(level_dir):
            continue
        for name in sorted(os.listdir(level_dir)):
            if not os.path.isdir(os.path.join(level_dir, name)):
                continue
            if glob.glob(os.path.join(level_dir, name, "episode_*.json")) or glob.glob(
                os.path.join(level_dir, name, "round*", "episode_*.json")
            ):
                models.add(name)
    return sorted(models)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _sideways_yaw(yaw_deg: float) -> bool:
    yaw = abs(float(yaw_deg)) % 360.0
    if yaw > 180.0:
        yaw = 360.0 - yaw
    return 45.0 <= yaw <= 135.0


def episode_passed_sideways(episode: Dict[str, Any]) -> bool:
    """Fall back to the step trace when the field is missing (legacy data)."""
    if "passed_sideways" in episode:
        return bool(episode["passed_sideways"])
    if not episode.get("passed"):
        return False
    steps = episode.get("steps", [])
    if not steps:
        return False
    return _sideways_yaw(steps[-1].get("torso_rotation", 0.0))


def episode_passed(episode: Dict[str, Any]) -> bool:
    if "passed" in episode:
        return bool(episode["passed"])
    return bool(episode.get("success"))


def episode_first_turn_step(episode: Dict[str, Any]) -> Optional[int]:
    if episode.get("first_turn_step") is not None:
        return int(episode["first_turn_step"])
    for step in episode.get("steps", []):
        if str(step.get("action")) in ("turn_left", "turn_right"):
            return int(step.get("step", 0))
    return None


def summarize_level(level: int, episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the A/S metrics for one Level."""
    if not episodes:
        return {"level": int(level), "episodes": 0}

    passed = [episode_passed(ep) for ep in episodes]
    sideways = [episode_passed_sideways(ep) for ep in episodes]
    first_turns = [
        step for step in (episode_first_turn_step(ep) for ep in episodes) if step is not None
    ]
    success_steps = [
        int(ep.get("total_steps", len(ep.get("steps", []))))
        for ep in episodes
        if episode_passed(ep)
    ]
    sideways_success_steps = [
        int(ep.get("total_steps", len(ep.get("steps", []))))
        for ep in episodes
        if episode_passed(ep) and episode_passed_sideways(ep)
    ]

    return {
        "level": int(level),
        "channel_width": float(episodes[0].get("channel_width", float("nan"))),
        "a_s_ratio": float(episodes[0].get("a_s_ratio", float("nan"))),
        "episodes": len(episodes),
        "pass_rate": float(np.mean(passed)),
        "passed_count": int(sum(passed)),
        "sideways_rate": float(np.mean(sideways)),
        "passed_sideways_count": int(sum(sideways)),
        "turned_rate": float(
            np.mean([episode_first_turn_step(ep) is not None for ep in episodes])
        ),
        "first_turn_step_mean": (float(np.mean(first_turns)) if first_turns else None),
        "first_turn_step_min": (int(min(first_turns)) if first_turns else None),
        "avg_success_steps": (
            float(np.mean(success_steps)) if success_steps else None
        ),
        "avg_sideways_success_steps": (
            float(np.mean(sideways_success_steps)) if sideways_success_steps else None
        ),
        "avg_total_rotation_deg": float(
            np.mean(
                [
                    float(
                        ep.get(
                            "total_rotation",
                            sum(
                                15.0
                                for s in ep.get("steps", [])
                                if str(s.get("action")) in ("turn_left", "turn_right")
                            ),
                        )
                    )
                    for ep in episodes
                ]
            )
        ),
    }


def sideways_threshold(level_summaries: Dict[int, Dict[str, Any]]) -> Optional[float]:
    """Widest A/S ratio at which the model still passed sideways."""
    candidates = [
        summary
        for summary in level_summaries.values()
        if int(summary.get("passed_sideways_count", 0)) > 0
    ]
    if not candidates:
        return None
    return float(max(candidates, key=lambda s: float(s["a_s_ratio"]))["a_s_ratio"])


def anticipation_class(threshold: Optional[float]) -> str:
    """Bucket a threshold against the human reference of 1.30."""
    if threshold is None:
        return "no_sideways"
    if threshold >= HUMAN_THRESHOLD:
        return "anticipatory"
    if threshold >= 1.0:
        return "borderline"
    return "reactive"


def analyze_model(
    results_root: str, model: str, levels: Sequence[int]
) -> Dict[str, Any]:
    """Build the full per-model report across every requested Level."""
    per_level: Dict[int, Dict[str, Any]] = {}
    for level in levels:
        episodes = load_episodes(results_root, level, model)
        if not episodes:
            continue
        per_level[int(level)] = summarize_level(int(level), episodes)

    threshold = sideways_threshold(per_level)
    all_episodes = [
        ep for level in levels for ep in load_episodes(results_root, level, model)
    ]
    total_passed = sum(1 for ep in all_episodes if episode_passed(ep))
    total_sideways = sum(1 for ep in all_episodes if episode_passed_sideways(ep))

    return {
        "model": model,
        "levels": per_level,
        "sideways_threshold": threshold,
        "human_reference_threshold": HUMAN_THRESHOLD,
        "threshold_gap_vs_human": (
            None if threshold is None else float(threshold - HUMAN_THRESHOLD)
        ),
        "threshold_class": anticipation_class(threshold),
        "overall": {
            "episodes": len(all_episodes),
            "passed_count": total_passed,
            "pass_rate": (total_passed / len(all_episodes)) if all_episodes else 0.0,
            "passed_sideways_count": total_sideways,
            "sideways_rate": (
                total_sideways / len(all_episodes) if all_episodes else 0.0
            ),
            "first_turn_step_mean": (
                float(
                    np.mean(
                        [
                            step
                            for step in (
                                episode_first_turn_step(ep) for ep in all_episodes
                            )
                            if step is not None
                        ]
                    )
                )
                if any(episode_first_turn_step(ep) is not None for ep in all_episodes)
                else None
            ),
            "avg_success_steps": (
                float(
                    np.mean(
                        [
                            int(ep.get("total_steps", 0))
                            for ep in all_episodes
                            if episode_passed(ep)
                        ]
                    )
                )
                if total_passed
                else None
            ),
            "total_wall_collisions": int(
                sum(int(ep.get("wall_collision_count", 0)) for ep in all_episodes)
            ),
            "total_invalid_responses": int(
                sum(int(ep.get("invalid_response_count", 0)) for ep in all_episodes)
            ),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_markdown_table(
    reports: Sequence[Dict[str, Any]], levels: Sequence[int]
) -> str:
    """Per-(model, level) table plus a threshold summary per model."""
    header = [
        "Model",
        "Level",
        "A/S",
        "Width (m)",
        "Pass %",
        "Sideways %",
        "1st Turn",
        "Avg Pass Steps",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for report in reports:
        model = report["model"]
        for level in levels:
            summary = report["levels"].get(int(level))
            if not summary:
                continue
            first_turn = summary.get("first_turn_step_mean")
            avg_steps = summary.get("avg_success_steps")
            values = [
                str(model),
                str(level),
                f"{summary['a_s_ratio']:.2f}",
                f"{summary['channel_width']:.2f}",
                f"{100.0 * summary['pass_rate']:.1f}",
                f"{100.0 * summary['sideways_rate']:.1f}",
                "-" if first_turn is None else f"{first_turn:.1f}",
                "-" if avg_steps is None else f"{avg_steps:.2f}",
            ]
            lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    threshold_header = [
        "Model",
        "Sideways Threshold (A/S)",
        "Human Reference",
        "Gap vs Human",
        "Class",
    ]
    lines.append("| " + " | ".join(threshold_header) + " |")
    lines.append("|" + "---|" * len(threshold_header))
    for report in reports:
        threshold = report["sideways_threshold"]
        gap = report["threshold_gap_vs_human"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(report["model"]),
                    "-" if threshold is None else f"{threshold:.2f}",
                    f"{HUMAN_THRESHOLD:.2f}",
                    "-" if gap is None else f"{gap:+.2f}",
                    str(report["threshold_class"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def table_rows(
    reports: Sequence[Dict[str, Any]], levels: Sequence[int]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in reports:
        for level in levels:
            summary = report["levels"].get(int(level))
            if not summary:
                continue
            rows.append(
                {
                    "model": report["model"],
                    "level": level,
                    "channel_width": summary["channel_width"],
                    "a_s_ratio": summary["a_s_ratio"],
                    "episodes": summary["episodes"],
                    "pass_rate": summary["pass_rate"],
                    "sideways_rate": summary["sideways_rate"],
                    "passed_sideways_count": summary["passed_sideways_count"],
                    "first_turn_step_mean": summary["first_turn_step_mean"],
                    "avg_success_steps": summary["avg_success_steps"],
                    "avg_total_rotation_deg": summary["avg_total_rotation_deg"],
                }
            )
        rows.append(
            {
                "model": report["model"],
                "level": "ALL",
                "sideways_threshold": report["sideways_threshold"],
                "threshold_class": report["threshold_class"],
                "overall_pass_rate": report["overall"]["pass_rate"],
                "overall_sideways_rate": report["overall"]["sideways_rate"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def plot_thresholds(
    reports: Sequence[Dict[str, Any]], levels: Sequence[int], out_path: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    for report in reports:
        xs = [
            report["levels"][int(level)]["a_s_ratio"]
            for level in levels
            if int(level) in report["levels"]
        ]
        ys = [
            100.0 * report["levels"][int(level)]["sideways_rate"]
            for level in levels
            if int(level) in report["levels"]
        ]
        if not xs:
            continue
        order = np.argsort(xs)
        plt.plot(
            np.asarray(xs)[order],
            np.asarray(ys)[order],
            marker="o",
            label=str(report["model"]),
        )
        threshold = report["sideways_threshold"]
        if threshold is not None:
            plt.axvline(
                threshold,
                linestyle=":",
                alpha=0.6,
                label=f"{report['model']} threshold {threshold:.2f}",
            )

    plt.axvline(
        HUMAN_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"human threshold {HUMAN_THRESHOLD:.2f}",
    )
    plt.axvline(1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    plt.xlabel("A/S ratio (channel width / shoulder width)")
    plt.ylabel("Sideways passage rate (%)")
    plt.title("Sideways passage vs channel-to-shoulder ratio")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the EmbodiedBAO A/S threshold results."
    )
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5]
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None, help="Model names; default: all"
    )
    parser.add_argument("--out_dir", type=str, default="analysis")
    parser.add_argument("--no_plot", action="store_true", help="Skip matplotlib plots")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    models = args.models or discover_models(args.results_root)
    if not models:
        print(f"No episode results found under {args.results_root}.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    reports = [analyze_model(args.results_root, model, args.levels) for model in models]
    reports = [r for r in reports if r["levels"]]
    if not reports:
        print("No analyzable episodes.")
        return 1

    for report in reports:
        path = os.path.join(
            args.out_dir,
            f"threshold_{report['model'].replace('/', '-')}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    markdown = format_markdown_table(reports, args.levels)
    for name in ("threshold_table.md", "threshold_table.txt"):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as handle:
            handle.write(markdown + "\n")

    rows = table_rows(reports, args.levels)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with open(
            os.path.join(args.out_dir, "threshold_table.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if not args.no_plot:
        try:
            plot_thresholds(
                reports, args.levels, os.path.join(args.out_dir, "thresholds.png")
            )
        except Exception as exc:
            print(f"[analysis] plotting failed: {exc}")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
