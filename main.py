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

Hierarchy of outputs (all under ``output_root()``: the repository, or
``BAO_OUTPUT_ROOT`` when that is set):
    results/level{level}/{model}/{tag}/episode_{id:03d}.json       episode record
    results/level{level}/{model}/{tag}/episode_{id:03d}_steps.json per-step record
    results/level{level}/{model}/{tag}/summary_{tag}.json          Level metrics
    results/{model}/checkpoint_{tag}.json                      resume state
    results/{model}/level{level}_{tag}_{timestamp}.csv         flat per-step CSV
    logs/{tag}/level{level}_episode{id:03d}_agent.txt          raw model I/O
    logs/{tag}/args.json                                       effective settings
    run_progress.txt                                           greppable timeline

Every JSON/CSV artefact is written atomically (see ``persistence.py``), so an
interrupted run can never leave a truncated record behind, and the flat CSV is
refreshed after every episode so an interrupted Level is still exported.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import persistence

# ---------------------------------------------------------------------------
# The run tag carries the protocol version
#
# The tag is the directory key for both the results tree and the resume
# checkpoint, and the checkpoint decides which episodes `--resume` treats as
# already done.  A per-model tag alone is therefore not enough: when the walking
# frame changed, every prompt and every action meaning changed with it, and a
# resumed run would have found the previous protocol's episodes, counted them as
# complete, and produced a dataset mixing two protocols with no way to tell them
# apart afterwards.
#
# So the protocol version is appended to whatever the caller asks for, here and
# in run_all_models.sh, and tools/check_sweep_tags.py compares the two.
# ---------------------------------------------------------------------------


def effective_tag(model: str, tag: str = "") -> str:
    """Results-directory tag for a run, including the protocol version.

    Idempotent: a tag that already carries the protocol version is returned
    unchanged.  run_all_models.sh composes the tag itself (so the tag it prints
    is the directory it will use) and then passes it through ``--tag``, which
    reaches this function a second time; without the check the directory became
    ``<model>-v4-walkframe-v4-walkframe``, the sweep log disagreed with the path
    actually written, and a manual ``main.py --tag <model>`` could not find the
    sweep's resume checkpoint.
    """
    from protocol import PROTOCOL_TAG

    base = persistence.sanitize_tag(tag or model, "untagged")
    if base == PROTOCOL_TAG or base.endswith("-" + PROTOCOL_TAG):
        return base
    return f"{base}-{PROTOCOL_TAG}"

# ---------------------------------------------------------------------------
# Output layout
#
# Results, logs and the progress file are anchored to the *repository*, not to
# the process working directory.  `python /path/to/main.py` executed from
# somewhere else used to scatter results/ and run_progress.txt into whatever
# directory the operator happened to be in, which on a machine you do not
# control looks exactly like a run that saved nothing.  Set BAO_OUTPUT_ROOT to
# send every artefact somewhere else (e.g. a large scratch disk).
# ---------------------------------------------------------------------------
RESULTS_DIRNAME = "results"
LOGS_DIRNAME = "logs"


def project_root() -> str:
    """Directory that holds this file (the repository root)."""
    return os.path.dirname(os.path.abspath(__file__))


def output_root() -> str:
    """Root under which results/, logs/ and run_progress.txt are written."""
    override = os.environ.get("BAO_OUTPUT_ROOT", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return project_root()


def results_dir() -> str:
    return os.path.join(output_root(), RESULTS_DIRNAME)


def logs_dir() -> str:
    return os.path.join(output_root(), LOGS_DIRNAME)


def progress_path() -> str:
    return os.path.join(output_root(), "run_progress.txt")


# ---------------------------------------------------------------------------
# Pre-flight
#
# Deliberately duplicated from ai_agent.py (which cannot be imported here: it
# pulls in environment.py, and importing environment before SimulationApp has
# started latches _HAS_ISAAC_SIM = False for the whole process -- see the note
# above PROTOCOL_LEVELS).
#
# The point is to catch a model/credential problem in seconds rather than
# after the ~150 s Isaac Sim startup.  A recorded incident on this gateway is
# worth naming: an exhausted account balance is returned as HTTP 403 with
# ``pre_consume_token_quota_failed``, the tooling classifies that as an AUTH
# failure, and the UI reports it as an invalid API key.  The key was fine.
# ---------------------------------------------------------------------------
API_KEY_ENV_VARS = ("BOYUE_API_KEY", "TAOTOKEN_API_KEY", "OPENAI_API_KEY")
BASE_URL_ENV_VARS = ("BOYUE_BASE_URL", "TAOTOKEN_BASE_URL", "OPENAI_BASE_URL")
BOYUE_DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _configured_api_key() -> str:
    for name in API_KEY_ENV_VARS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _configured_base_url() -> str:
    for name in BASE_URL_ENV_VARS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value.rstrip("/")
    if (os.environ.get("BOYUE_API_KEY") or "").strip():
        return BOYUE_DEFAULT_BASE_URL
    return OPENAI_DEFAULT_BASE_URL


def preflight_model(model: str, timeout: float = 10.0) -> str:
    """Check the credential before Isaac Sim starts; return a status line.

    Raises ``ValueError`` when the run cannot possibly succeed (no key at all).
    Anything the gateway says about a key that *is* present is reported but
    never fatal: an aggregator's 401/403 may be a quota problem rather than a
    bad key, and second-guessing it here would block a run that would work.
    """
    if str(model).strip().lower() == "random":
        return "model=random baseline: no API key needed"

    key = _configured_api_key()
    if not key:
        raise ValueError(
            "no API key in the environment. Export one of "
            f"{' / '.join(API_KEY_ENV_VARS)} before starting the run "
            "(the checkpoint and results are unaffected; nothing has started yet)."
        )
    if not key.isascii():
        raise ValueError(
            "the configured API key contains non-ASCII characters; it was "
            "probably pasted from a document with smart quotes or a placeholder."
        )

    base_url = _configured_base_url()
    request = urllib.request.Request(
        base_url + "/models", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        if exc.code == 401:
            return (
                f"WARNING: the gateway rejected the key (HTTP 401 from {base_url}). "
                "This is a credential problem, not a quota problem. body=" + body
            )
        if exc.code == 403:
            return (
                f"WARNING: {base_url} answered HTTP 403. An aggregator reports an "
                "exhausted balance this way (code pre_consume_token_quota_failed), "
                "and some tools then display it as an invalid API key -- check the "
                "account balance before re-issuing the key. body=" + body
            )
        return f"WARNING: {base_url} answered HTTP {exc.code}; continuing. body={body}"
    except Exception as exc:  # network problems must not block a local run
        return (
            f"WARNING: could not reach {base_url} ({type(exc).__name__}: {exc}); "
            "continuing -- the first model call will report the real error"
        )
    return f"endpoint reachable (HTTP {status}), key accepted: {base_url}"

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
    safe_model = persistence.sanitize_tag(
        model.replace("/", "-").replace("\\", "-"), "model"
    )
    out_dir = os.path.join(results_root, safe_model)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    safe_tag = persistence.sanitize_tag(tag, "untagged")
    path = os.path.join(out_dir, f"level{level}_{safe_tag}_{timestamp}.csv")

    fields = [
        "episode_id",
        "level",
        "channel_width",
        "a_s_ratio",
        "passed",
        "passed_sideways",
        "passage_rotation_deg",
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

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
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
                    "passage_rotation_deg": episode.get("passage_rotation_deg"),
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
    # Atomic: this file is rewritten after every episode, and an interrupted
    # rewrite would otherwise destroy the accumulated table for the Level.
    persistence.atomic_write_text(path, buffer.getvalue())
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
    """Append a timestamped line to run_progress.txt (stdout is swallowed).

    The path is anchored to the output root rather than the working directory,
    and a failure to write it never aborts an experiment: the progress file is
    a convenience, the episode records are the data.
    """
    path = progress_path()
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
            handle.flush()
    except OSError as exc:
        print(f"[main] WARNING: could not append to {path}: {exc}", flush=True)


def run_experiment(args: argparse.Namespace) -> Dict[int, Dict[str, Any]]:
    """Set up Isaac Sim, run the Levels, and return the per-Level summaries."""
    levels = resolve_levels(args)
    run_results_root = results_dir()
    run_logs_root = logs_dir()
    print(f"[main] output root: {output_root()}")
    _write_progress(
        f"main start: model={args.model} levels={levels} "
        f"episodes={args.episodes} max_steps={args.max_steps} "
        f"output_root={output_root()}"
    )
    if args.image_size is not None:
        os.environ["BAO_IMAGE_SIZE"] = str(int(args.image_size))
    if args.llm_timeout is not None:
        os.environ["BAO_LLM_TIMEOUT"] = str(float(args.llm_timeout))
    env = None
    simulation_app = None
    try:
        # Credential check first: it costs a second, while the Isaac Sim startup
        # below costs minutes.
        preflight_message = preflight_model(args.model)
        print(f"[main] preflight: {preflight_message}")
        _write_progress(f"preflight: {preflight_message}")

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
            tag=effective_tag(args.model, args.tag),
            save_obs=args.save_obs,
            results_root=run_results_root,
            logs_root=run_logs_root,
        )
        runner.save_args(args)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_model = persistence.sanitize_tag(
            args.model.replace("/", "-").replace("\\", "-"), "model"
        )
        checkpoint_path = os.path.join(
            run_results_root, safe_model, f"checkpoint_{runner.tag}.json"
        )
        checkpoint = ProtocolCheckpoint(path=checkpoint_path, resume=args.resume)
        if args.resume and checkpoint.completed:
            print(f"[checkpoint] resumed with {len(checkpoint.completed)} completed units")
        _write_progress(f"checkpoint path: {os.path.abspath(checkpoint_path)}")

        def _export_csv(level: int, episodes_done: Sequence[Dict[str, Any]]) -> str:
            """Refresh the flat per-step CSV for one Level."""
            return save_episodes_csv(
                episodes_done,
                model=args.model,
                level=level,
                results_root=run_results_root,
                timestamp=timestamp,
                tag=runner.tag,
            )

        def _on_episode(
            level: int,
            completed: int,
            total: int,
            episode: Dict[str, Any],
            episodes_done: Sequence[Dict[str, Any]],
        ) -> None:
            """Progress reporting plus an always-current CSV for the Level.

            The CSV used to be written only after a whole Level finished, so an
            interruption in the middle of a Level left the JSON records on disk
            but no flat table.  Rewriting it per episode is cheap (tens of rows)
            and means the exported data is never behind the saved records.
            """
            _progress_callback(level, completed, total, episode, episodes_done)
            try:
                _export_csv(level, list(episodes_done))
            except OSError as exc:
                _write_progress(
                    f"WARNING: partial CSV export for level {level} failed: {exc}"
                )

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
                progress_callback=_on_episode,
                checkpoint=checkpoint,
            )
            summary = BAOExperimentRunner.summarize_level(level, episodes)
            summaries[level] = summary
            csv_path = _export_csv(level, episodes)
            print(f"[main] saved {csv_path}")
            _write_progress(f"csv saved: {csv_path}")
            _write_progress(
                f"level {level} done: pass_rate={summary['pass_rate']:.3f} "
                f"sideways_rate={summary['sideways_rate']:.3f}"
            )

        print_threshold_table(summaries)
        _write_progress("all levels done")
        return summaries
    except KeyboardInterrupt:
        # Ctrl-C is how a long sweep is normally stopped; make that visible in
        # the timeline instead of leaving it indistinguishable from a crash.
        _write_progress("INTERRUPTED by user (KeyboardInterrupt)")
        print("\n[main] interrupted; completed episodes are already on disk")
        raise
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
