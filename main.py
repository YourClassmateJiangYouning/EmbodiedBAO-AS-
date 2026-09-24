"""EmbodiedBAO program entry point.

Usage:
    python main.py --model gpt-4o --level 0                 # one Level
    python main.py --model gpt-4o --levels 0 1 2 3 4 5      # all six Levels
    python main.py --model gpt-4o --all-levels
    python main.py --model random --level 3 --episodes 2    # smoke test
    python main.py --model gemini-2.5-pro --all-levels --resume

Flow: initialize the Isaac Sim scene (``environment.setup_scene``), fail fast on
an invalid model configuration (``ai_agent.create_agent``), run the requested
Levels with ``experiments.BAOExperimentRunner``, save one JSON summary per Level
plus one flat CSV per (model, tag), print the A/S threshold table, and close the
environment.

Hierarchy of outputs:
    results/level{level}/{model}/{tag}/episode_{id:03d}.json       episode record
    results/level{level}/{model}/{tag}/episode_{id:03d}_steps.json per-step record
    results/level{level}/{model}/{tag}/summary_{tag}.json          Level metrics
    results/{model}/checkpoint_{tag}.json                      resume state
    results/{model}_level{level}_{tag}.csv                     flat per-step CSV
    logs/{tag}/level{level}_episode{id:03d}_agent.txt          raw model I/O
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Protocol constants -- intentionally duplicated here as plain literals.
#
# environment.py and experiments.py import Isaac Sim at module scope, and
# `isaacsim.core` only becomes importable AFTER SimulationApp has started.  A
# top-level `from environment import ...` in the entry point therefore runs
# before Isaac Sim is ready, raises ModuleNotFoundError, and permanently
# latches environment._HAS_ISAAC_SIM = False -- after which the scene can never
# be built.  Keeping this module free of those imports (they happen lazily
# inside run_experiment) is what avoids that trap.
#
# Keep in sync with environment.LEVEL_CHANNEL_WIDTHS and the experiments
# DEFAULT_* constants; test_bao_geometry.py asserts the values.
# ---------------------------------------------------------------------------
PROTOCOL_LEVELS: Sequence[int] = (0, 1, 2, 3, 4, 5)
DEFAULT_EPISODES_PER_LEVEL = 10
DEFAULT_MAX_STEPS = 30


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not (parsed > 0.0) or parsed == float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EmbodiedBAO experiment entry point.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument(
        "--level", type=int, choices=list(PROTOCOL_LEVELS), default=0, help="Level to run"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        choices=list(PROTOCOL_LEVELS),
        default=None,
        help="Run multiple Levels in one process",
    )
    parser.add_argument(
        "--all-levels", action="store_true", help="Run Levels 0, 1, 2, 3, 4, 5"
    )
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        default=DEFAULT_EPISODES_PER_LEVEL,
        help=f"Episodes per Level (default {DEFAULT_EPISODES_PER_LEVEL})",
    )
    parser.add_argument(
        "--max_steps",
        type=_positive_int,
        default=DEFAULT_MAX_STEPS,
        help=f"Step limit per episode (default {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes already recorded in this run's checkpoint",
    )
    parser.add_argument(
        "--save_obs", action="store_true", help="Save per-step camera PNGs"
    )
    parser.add_argument(
        "--eye_height",
        type=float,
        default=None,
        help=(
            "Head camera height in metres (default: the H1 head height, "
            "environment.ROBOT_HEAD_HEIGHT). Use this to place the eye "
            "elsewhere on the body."
        ),
    )
    parser.add_argument(
        "--eye_pitch",
        type=float,
        default=None,
        help=(
            "Head camera downward pitch in degrees (default: "
            "environment.EYE_PITCH_DEG). 0 looks perfectly level, which renders "
            "a flat wall as a featureless grey plane."
        ),
    )
    parser.add_argument(
        "--start_x",
        type=float,
        default=None,
        help=(
            "Robot start x in metres (default: 0.5). Standing further back "
            "makes the channel readable at the cost of travel."
        ),
    )
    parser.add_argument(
        "--move_step",
        type=float,
        default=None,
        help=(
            "Translation per action in metres (default: 0.75). The prompt uses "
            "the same configured value, so overrides remain self-consistent."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=_positive_int,
        default=None,
        help=(
            "Downscale the camera frame to this square size before sending it "
            "to the model (default: the camera resolution, e.g. 1024). Smaller "
            "images are much cheaper per call; measured latency was ~31 s/step "
            "at 1024."
        ),
    )
    parser.add_argument(
        "--llm_timeout",
        type=_positive_float,
        default=None,
        help=(
            "Per-request timeout in seconds, overriding BAO_LLM_TIMEOUT "
            "(default 90). A recorded run lost 9 of 30 steps to timeouts."
        ),
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default="{}",
        help='JSON dict passed to BAOEnv, e.g. \'{"rendermode":"RaytracedLighting","spp":4}\'',
    )
    return parser.parse_args(argv)


def resolve_levels(args: argparse.Namespace) -> List[int]:
    """Turn the CLI level flags into a validated, de-duplicated level list."""
    if args.all_levels and args.levels:
        raise ValueError("--all-levels and --levels cannot be used together")
    if args.all_levels:
        return list(PROTOCOL_LEVELS)
    if args.levels:
        requested = list(args.levels)
    else:
        requested = [args.level]
    seen: List[int] = []
    for level in requested:
        if level not in seen:
            seen.append(level)
    return seen


def save_episodes_csv(
    episodes: Sequence[Dict[str, Any]],
    model: str,
    level: int,
    results_root: str = "results",
    timestamp: str = "",
    tag: str = "",
) -> str:
    """Flatten episodes into one CSV row per step.

    The tag goes in the filename, not just the timestamp.  Episode JSON is
    isolated per run under ``results/level{n}/{model}/{tag}/``, but the CSV used
    to be named ``level{n}_{timestamp}.csv`` in the model directory, so every
    re-run of the same Level overwrote the previous file and there was no way to
    tell which CSV belonged to which tag.  Across an 11-model sweep that makes
    the flat tables unusable.
    """
    safe_model = model.replace("/", "-").replace("\\", "-")
    out_dir = os.path.join(results_root, safe_model)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    safe_tag = str(tag).replace("/", "-").replace("\\", "-") or "untagged"
    path = os.path.join(out_dir, f"level{level}_{safe_tag}_{timestamp}.csv")

    fields = [
        "episode_id",
        "level",
        "channel_width",
        "a_s_ratio",
        "passed",
        "passed_sideways",
        "total_rotation",
        "first_turn_step",
        "total_steps",
        "action_sequence",
        "step",
        "action",
        "torso_rotation",
        "position_x",
        "position_z",
        "collision",
        "step_success",
        "llm_response_time_ms",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            for step in episode.get("steps", []):
                writer.writerow(
                    {
                        "episode_id": episode.get("episode_id"),
                        "level": episode.get("level"),
                        "channel_width": episode.get("channel_width"),
                        "a_s_ratio": episode.get("a_s_ratio"),
                        "passed": episode.get("passed"),
                        "passed_sideways": episode.get("passed_sideways"),
                        "total_rotation": episode.get("total_rotation"),
                        "first_turn_step": episode.get("first_turn_step"),
                        "total_steps": episode.get("total_steps"),
                        "action_sequence": episode.get("action_sequence"),
                        "step": step.get("step"),
                        "action": step.get("action"),
                        "torso_rotation": step.get("torso_rotation"),
                        "position_x": step.get("position_x"),
                        "position_z": step.get("position_z"),
                        "collision": step.get("collision"),
                        "step_success": step.get("step_success"),
                        "llm_response_time_ms": step.get("llm_response_time_ms"),
                    }
                )
    return path


def _runner_class() -> Any:
    """Return BAOExperimentRunner, importing experiments lazily.

    `experiments` imports `environment`, which imports Isaac Sim at module
    scope, so it must not be imported until SimulationApp is running.  Every
    module-level helper that needs the runner goes through here instead of
    referencing a name that only exists inside run_experiment.
    """
    from experiments import BAOExperimentRunner

    return BAOExperimentRunner


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def print_threshold_table(summaries: Dict[int, Dict[str, Any]]) -> None:
    """Print the per-Level metric table and the derived A/S threshold."""
    header = (
        f"{'Level':>5}  {'A/S':>5}  {'Width':>7}  {'Pass%':>6}  {'Sideways%':>9}  "
        f"{'1stTurn':>8}  {'AvgSteps':>8}  {'Collisions':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for level in sorted(summaries):
        summary = summaries[level]
        if not summary.get("episodes"):
            continue
        print(
            f"{level:>5}  {summary['a_s_ratio']:>5.2f}  "
            f"{summary['channel_width']:>7.2f}  "
            f"{100.0 * summary['pass_rate']:>6.1f}  "
            f"{100.0 * summary['sideways_rate']:>9.1f}  "
            f"{_fmt(summary.get('first_turn_step_mean'), 1):>8}  "
            f"{_fmt(summary.get('avg_success_steps'), 2):>8}  "
            f"{summary['total_wall_collisions']:>10}"
        )
    threshold = _runner_class().sideways_threshold(summaries)
    print("-" * len(header))
    if threshold is None:
        print(
            "Sideways threshold: not observed (no Level produced a sideways passage)."
        )
    else:
        print(
            f"Sideways threshold: A/S = {threshold:.2f} "
            f"(channel {threshold * 0.57:.3f} m; human reference A/S = 1.30)"
        )


def _progress_callback(
    level: int,
    completed: int,
    total: int,
    episode: Dict[str, Any],
    episodes_done: Sequence[Dict[str, Any]],
) -> None:
    """Progress reporter matching BAOExperimentRunner's callback signature.

    It must accept all five arguments (level, completed, total, the episode
    just finished, and the running list) even though it only uses some of them.

    The interval is configurable because a fixed 10 meant a run with
    ``--episodes 10`` wrote exactly one progress line per Level, so a multi-hour
    sweep across models looked stalled from run_progress.txt alone.  The runner
    already prints every episode to stdout; this only controls the durable,
    greppable line.
    """
    try:
        every = max(1, int(os.environ.get("BAO_PROGRESS_EVERY", "5")))
    except ValueError:
        every = 5
    if completed % every == 0 or completed == total:
        success_count = sum(1 for ep in episodes_done if ep.get("passed"))
        rate = success_count / completed if completed else 0.0
        message = (
            f"level {level}: episode {completed}/{total}, "
            f"pass_rate={rate:.3f}"
        )
        print(f"[main] {message}")
        _write_progress(message)


def _write_progress(message: str) -> None:
    """Append a timestamped line to run_progress.txt (stdout is swallowed)."""
    path = os.path.join(os.getcwd(), "run_progress.txt")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def run_experiment(args: argparse.Namespace) -> Dict[int, Dict[str, Any]]:
    """Set up Isaac Sim, run the Levels, and return the per-Level summaries."""
    levels = resolve_levels(args)
    _write_progress(
        f"main start: model={args.model} levels={levels} "
        f"episodes={args.episodes} max_steps={args.max_steps}"
    )
    if args.image_size is not None:
        os.environ["BAO_IMAGE_SIZE"] = str(int(args.image_size))
    if args.llm_timeout is not None:
        os.environ["BAO_LLM_TIMEOUT"] = str(float(args.llm_timeout))
    env = None
    simulation_app = None
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": args.headless})
        _write_progress("SimulationApp started")

        import ai_agent
        import environment
        from experiments import BAOExperimentRunner, ProtocolCheckpoint

        # Fail fast if the API key/model configuration is invalid; this also
        # supports the "random" baseline via the create_agent factory.
        ai_agent.create_agent(model=args.model)
        _write_progress(f"agent ready: {args.model}")

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        # Convenience overrides so the camera can be tuned from the command line
        # without editing environment.py.
        if args.eye_height is not None:
            task_dict["eye_camera_height"] = float(args.eye_height)
        if args.eye_pitch is not None:
            task_dict["eye_pitch_deg"] = float(args.eye_pitch)
        if args.start_x is not None:
            task_dict["start_x"] = float(args.start_x)
        if args.move_step is not None:
            task_dict["move_step"] = float(args.move_step)
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        _write_progress("environment created")

        runner = BAOExperimentRunner(
            env=env,
            model=args.model,
            max_steps=args.max_steps,
            episodes_per_level=args.episodes,
            tag=args.tag,
            save_obs=args.save_obs,
        )
        runner.save_args(args)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_model = args.model.replace("/", "-").replace("\\", "-")
        checkpoint_path = os.path.join(
            "results", safe_model, f"checkpoint_{runner.tag}.json"
        )
        checkpoint = ProtocolCheckpoint(path=checkpoint_path, resume=args.resume)
        if args.resume and checkpoint.completed:
            print(f"[checkpoint] resumed with {len(checkpoint.completed)} completed units")
        _write_progress(f"checkpoint path: {checkpoint_path}")

        summaries: Dict[int, Dict[str, Any]] = {}
        for level in levels:
            _write_progress(f"level {level} start")
            print(
                f"\n[main] === Level {level}: channel "
                f"{environment.level_channel_width(level):.2f} m ==="
            )
            episodes = runner.run_level(
                level=level,
                episodes=args.episodes,
                progress_callback=_progress_callback,
                checkpoint=checkpoint,
            )
            summary = BAOExperimentRunner.summarize_level(level, episodes)
            summaries[level] = summary
            csv_path = save_episodes_csv(
                episodes,
                model=args.model,
                level=level,
                timestamp=timestamp,
                tag=getattr(args, "tag", "") or "",
            )
            print(f"[main] saved {csv_path}")
            _write_progress(f"csv saved: {csv_path}")
            _write_progress(
                f"level {level} done: pass_rate={summary['pass_rate']:.3f} "
                f"sideways_rate={summary['sideways_rate']:.3f}"
            )

        print_threshold_table(summaries)
        _write_progress("all levels done")
        return summaries
    except Exception as exc:
        _write_progress(f"ERROR: {type(exc).__name__}: {exc}")
        _write_progress(traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        elif simulation_app is not None:
            simulation_app.close()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
