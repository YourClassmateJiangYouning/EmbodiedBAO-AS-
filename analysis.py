"""A/S threshold analysis for EmbodiedBAO.

Consumes the per-episode JSON files written by ``experiments.py``
(``results/level{level}/{model}/episode_*.json``) and reports, for each model,
how its behaviour changes as the channel narrows.

Headline metrics (per Level, and per model overall):

* **Pass rate**       -- fraction of episodes that reached the x >= 11 m goal plane.
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
import io
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from persistence import atomic_write_json, atomic_write_text

HUMAN_THRESHOLD = 1.30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episodes(
    results_root: str, level: int, model: str, tag: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load per-episode JSON files for one model/Level, optionally one run tag."""
    model_dir = os.path.join(results_root, f"level{level}", model)
    if tag:
        paths = sorted(glob.glob(os.path.join(model_dir, tag, "episode_*.json")))
    else:
        # Preserve legacy direct files.  For tagged runs, analyse only the most
        # recently modified run directory rather than mixing incompatible runs.
        paths = sorted(glob.glob(os.path.join(model_dir, "episode_*.json")))
        if not paths:
            run_dirs = [
                path
                for path in glob.glob(os.path.join(model_dir, "*"))
                if os.path.isdir(path)
            ]
            if run_dirs:
                latest = max(run_dirs, key=os.path.getmtime)
                paths = sorted(glob.glob(os.path.join(latest, "episode_*.json")))
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
        try:
            with open(path, "r", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            # One damaged record must not take the whole analysis down: warn,
            # name the file, and carry on with the records that do parse.
            print(f"[analysis] WARNING: skipping unreadable {path} ({exc})")
            continue
        if not episode.get("steps"):
            sidecar = path[: -len(".json")] + "_steps.json"
            if os.path.exists(sidecar):
                try:
                    with open(sidecar, "r", encoding="utf-8") as handle:
                        episode["steps"] = json.load(handle)
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[analysis] WARNING: skipping unreadable {sidecar} ({exc})")
        episodes.append(episode)
    episodes.sort(key=lambda ep: int(ep.get("episode_id", 0)))
    return episodes


def _holds_episode_records(directory: str) -> bool:
    """True when ``directory`` directly contains at least one episode record.

    ``episode_*_steps.json`` sidecars live next to the records, and
    ``episode_*.json`` matches them too, so they are filtered out: a directory
    holding only sidecars has nothing to analyse.
    """
    for path in glob.glob(os.path.join(directory, "episode_*.json")):
        if not path.endswith("_steps.json"):
            return True
    return False


def discover_models(results_root: str) -> List[str]:
    """List models that have episode results under any Level.

    Three layouts are recognised, because all three exist in the wild:

    * ``level{n}/{model}/{tag}/episode_*.json`` -- what ``main.py`` writes for
      every tagged run, and therefore every run ``run_all_models.sh`` makes;
    * ``level{n}/{model}/episode_*.json`` -- untagged runs from before the tag
      became a directory component;
    * ``level{n}/{model}/round*/episode_*.json`` -- older still.

    The tagged form was missing here, which made a finished sweep invisible:
    ``python analysis.py`` printed "No episode results found" and exited 1
    while every episode sat on disk under its run tag.
    """
    models: set[str] = set()
    for level_dir in glob.glob(os.path.join(results_root, "level*")):
        if not os.path.isdir(level_dir):
            continue
        for name in sorted(os.listdir(level_dir)):
            model_dir = os.path.join(level_dir, name)
            if not os.path.isdir(model_dir):
                continue
            if _holds_episode_records(model_dir) or glob.glob(
                os.path.join(model_dir, "round*", "episode_*.json")
            ):
                models.add(name)
                continue
            # Tagged layout: one directory per run tag, named by the tag.
            for tag_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
                if os.path.isdir(tag_dir) and _holds_episode_records(tag_dir):
                    models.add(name)
                    break
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


def turned_threshold(level_summaries: Dict[int, Dict[str, Any]]) -> Optional[float]:
    """Widest A/S ratio at which the model rotated its torso at all.

    Distinct from ``sideways_threshold``, which additionally requires the episode
    to have SUCCEEDED.  The human reference (Warren & Whang 1987, 1.30) measures
    whether the shoulders were rotated, not whether the person got through, so
    this is the closer analogue: it separates the decision from the execution and
    cannot be depressed by a model that rotates correctly and then fails to
    follow through.
    """
    candidates = [
        summary
        for summary in level_summaries.values()
        if float(summary.get("turned_rate", 0.0)) > 0.0
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
    results_root: str,
    model: str,
    levels: Sequence[int],
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the full per-model report across every requested Level."""
    selected_tag = tag
    if selected_tag is None:
        tagged_dirs = [
            path
            for level in levels
            for path in glob.glob(
                os.path.join(results_root, f"level{level}", model, "*")
            )
            if os.path.isdir(path) and not os.path.basename(path).startswith("round")
        ]
        if tagged_dirs:
            # Prefer a directory from the CURRENT protocol.  The tag carries the
            # protocol version (protocol.PROTOCOL_TAG, appended by
            # main.effective_tag), so a directory that ends with it is this
            # protocol's data and an older one is a different experiment with
            # different action semantics.  Choosing purely by mtime would let a
            # partially re-run model -- or a stray touch -- summarise the old
            # protocol under the new protocol's name, which is exactly the mix
            # the tag exists to prevent.
            from protocol import PROTOCOL_TAG

            current = [
                path
                for path in tagged_dirs
                if os.path.basename(path).endswith("-" + PROTOCOL_TAG)
                or os.path.basename(path) == PROTOCOL_TAG
            ]
            pool = current or tagged_dirs
            selected_tag = os.path.basename(max(pool, key=os.path.getmtime))
            if not current:
                print(
                    f"[analysis] WARNING: no {model} results tagged with the "
                    f"current protocol ({PROTOCOL_TAG}); using {selected_tag!r}. "
                    f"Run main.py with a fresh tag before comparing."
                )
            older = sorted(
                {
                    os.path.basename(path)
                    for path in tagged_dirs
                    if os.path.basename(path) != selected_tag
                }
            )
            if older:
                print(
                    f"[analysis] {model}: analysing tag {selected_tag!r}; "
                    f"ignoring {len(older)} other tag(s): {', '.join(older)}"
                )

    per_level: Dict[int, Dict[str, Any]] = {}
    for level in levels:
        episodes = load_episodes(results_root, level, model, tag=selected_tag)
        if not episodes:
            continue
        per_level[int(level)] = summarize_level(int(level), episodes)

    threshold = sideways_threshold(per_level)
    turned = turned_threshold(per_level)
    all_episodes = [
        ep
        for level in levels
        for ep in load_episodes(results_root, level, model, tag=selected_tag)
    ]
    total_passed = sum(1 for ep in all_episodes if episode_passed(ep))
    total_sideways = sum(1 for ep in all_episodes if episode_passed_sideways(ep))

    return {
        "model": model,
        "levels": per_level,
        "sideways_threshold": threshold,
        "turned_threshold": turned,
        "human_reference_threshold": HUMAN_THRESHOLD,
        "threshold_gap_vs_human": (
            None if threshold is None else float(threshold - HUMAN_THRESHOLD)
        ),
        "threshold_class": anticipation_class(threshold),
        "turned_threshold_class": anticipation_class(turned),
        "turned_threshold_gap_vs_human": (
            None if turned is None else float(turned - HUMAN_THRESHOLD)
        ),
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
        "Turned %",
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
                f"{100.0 * summary['turned_rate']:.1f}",
                f"{100.0 * summary['sideways_rate']:.1f}",
                "-" if first_turn is None else f"{first_turn:.1f}",
                "-" if avg_steps is None else f"{avg_steps:.2f}",
            ]
            lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    threshold_header = [
        "Model",
        "Turned Threshold (A/S)",
        "Sideways Threshold (A/S)",
        "Human Reference",
        "Gap vs Human",
        "Class",
    ]
    lines.append("| " + " | ".join(threshold_header) + " |")
    lines.append("|" + "---|" * len(threshold_header))
    for report in reports:
        threshold = report["sideways_threshold"]
        turned = report.get("turned_threshold")
        gap = report["threshold_gap_vs_human"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(report["model"]),
                    "-" if turned is None else f"{turned:.2f}",
                    "-" if threshold is None else f"{threshold:.2f}",
                    f"{HUMAN_THRESHOLD:.2f}",
                    "-" if gap is None else f"{gap:+.2f}",
                    str(report["threshold_class"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Turned Threshold is the widest A/S at which the torso was rotated at all;"
    )
    lines.append(
        "Sideways Threshold additionally requires the episode to have succeeded."
    )
    lines.append(
        "The human reference of 1.30 measures whether the shoulders were rotated,"
    )
    lines.append("so Turned Threshold is the closer analogue.")
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
                    "turned_rate": summary["turned_rate"],
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
                "turned_threshold": report.get("turned_threshold"),
                "threshold_class": report["threshold_class"],
                "turned_threshold_class": report.get("turned_threshold_class"),
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


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_results_root() -> str:
    return os.path.join(_project_root(), "results")


def _default_out_dir() -> str:
    return os.path.join(_project_root(), "analysis")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the EmbodiedBAO A/S threshold results."
    )
    # Defaults are anchored to the repository so `python analysis.py` reports
    # on the run that was just produced, wherever it is invoked from.
    parser.add_argument("--results_root", type=str, default=_default_results_root())
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5]
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None, help="Model names; default: all"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Analyze one run tag; default: latest tagged run (or legacy files)",
    )
    parser.add_argument("--out_dir", type=str, default=_default_out_dir())
    parser.add_argument("--no_plot", action="store_true", help="Skip matplotlib plots")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    models = args.models or discover_models(args.results_root)
    if not models:
        print(f"No episode results found under {args.results_root}.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    reports = [
        analyze_model(args.results_root, model, args.levels, tag=args.tag)
        for model in models
    ]
    reports = [r for r in reports if r["levels"]]
    if not reports:
        print("No analyzable episodes.")
        return 1

    for report in reports:
        path = os.path.join(
            args.out_dir,
            f"threshold_{report['model'].replace('/', '-')}.json",
        )
        # Atomic: a report that is half-written is worse than a stale one, and
        # these files are what the paper's tables are generated from.
        atomic_write_json(path, report)

    markdown = format_markdown_table(reports, args.levels)
    for name in ("threshold_table.md", "threshold_table.txt"):
        atomic_write_text(os.path.join(args.out_dir, name), markdown + "\n")

    rows = table_rows(reports, args.levels)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(
            os.path.join(args.out_dir, "threshold_table.csv"), buffer.getvalue()
        )

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
