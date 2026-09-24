"""Regression suite for the crash-safety of the saved results.

A scored sweep is 11 models x 6 Levels x 10 episodes x 30 steps of real API
calls.  Losing that data to a truncated file is not recoverable by re-running:
the model's answers are not deterministic and the calls cost money.  So the
persistence contract is tested here rather than trusted:

* every artefact is written atomically, and a failed write keeps the previous
  file intact;
* a damaged checkpoint is quarantined and *rebuilt* from the episode records,
  instead of silently restarting the run over good data;
* a damaged episode record makes that episode re-run, not crash the sweep;
* results, logs and run_progress.txt follow the repository (or
  ``BAO_OUTPUT_ROOT``), never the process working directory;
* a ``--tag`` containing a path separator or a Windows-reserved character
  still produces exactly one directory component.

    python test_bao_persistence.py

The source is deliberately ASCII-only: this file is edited from shells whose
default encoding is not UTF-8, and a mis-decoded em dash is a syntax error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import analysis  # noqa: E402
import main as main_module  # noqa: E402
import persistence  # noqa: E402
import test_bao_integration as harness  # noqa: E402
from experiments import ProtocolCheckpoint  # noqa: E402

BAOExperimentRunner = harness.BAOExperimentRunner
FakeBAOEnv = harness.FakeBAOEnv
DEFAULT_MAX_STEPS = harness.DEFAULT_MAX_STEPS
make_temp_dir = harness.make_temp_dir
check = harness.check


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_atomic_write_json_round_trips() -> None:
    tmp = make_temp_dir()
    try:
        path = os.path.join(tmp, "nested", "episode_000.json")
        # Non-ASCII content must survive: the records quote model reasoning.
        payload = {"level": 0, "note": "non-ascii \u4e2d\u6587", "steps": [1, 2, 3]}
        persistence.atomic_write_json(path, payload)
        check(os.path.exists(path), "atomic_write_json did not create the file")
        with open(path, "r", encoding="utf-8") as handle:
            check(json.load(handle) == payload, "round-tripped payload differs")
        leftovers = [n for n in os.listdir(os.path.dirname(path)) if ".tmp-" in n]
        check(not leftovers, f"temporary files left behind: {leftovers}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] atomic JSON writes round-trip and leave no temporary file")


def test_failed_write_keeps_the_previous_file() -> None:
    """A write that dies halfway must not destroy the record already on disk."""
    tmp = make_temp_dir()
    try:
        path = os.path.join(tmp, "summary_tag.json")
        persistence.atomic_write_json(path, {"episodes": 4, "pass_rate": 0.5})

        real_replace = os.replace

        def boom(src, dst):  # noqa: ANN001 - test double
            raise OSError("simulated crash during rename")

        os.replace = boom
        try:
            try:
                persistence.atomic_write_json(path, {"episodes": 9})
                check(False, "the simulated failure did not raise")
            except OSError:
                pass
        finally:
            os.replace = real_replace

        with open(path, "r", encoding="utf-8") as handle:
            surviving = json.load(handle)
        check(
            surviving == {"episodes": 4, "pass_rate": 0.5},
            f"a failed write damaged the previous file: {surviving}",
        )
        leftovers = [n for n in os.listdir(tmp) if ".tmp-" in n]
        check(not leftovers, f"failed write left temporary files: {leftovers}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] a failed write keeps the previous file and cleans up after itself")


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_sanitize_tag_yields_one_safe_component() -> None:
    cases = {
        "run-a": "run-a",
        "gpt-4o": "gpt-4o",
        "a/b": "a-b",
        "a\\b": "a-b",
        "run:1": "run-1",
        "  padded  ": "padded",
        "trailing.": "trailing",
        "": "untagged",
        "..": "untagged",
        ".": "untagged",
        "a?b*c": "a-b-c",
    }
    for raw, expected in cases.items():
        got = persistence.sanitize_tag(raw)
        check(
            got == expected, f"sanitize_tag({raw!r}) == {got!r}, expected {expected!r}"
        )
        check(
            os.sep not in got and "/" not in got and "\\" not in got,
            f"sanitize_tag({raw!r}) still contains a path separator: {got!r}",
        )
    check(
        persistence.sanitize_tag("", fallback="x") == "x",
        "the fallback argument was ignored",
    )
    print("[ok] run tags are reduced to one filesystem-safe path component")


def test_runner_tag_reaches_paths_intact() -> None:
    tmp = make_temp_dir()
    try:
        runner = BAOExperimentRunner(
            env=FakeBAOEnv(),
            model="vendor/model",
            tag="run:1/../evil",
            results_root=os.path.join(tmp, "results"),
            logs_root=os.path.join(tmp, "logs"),
        )
        check("/" not in runner.tag, f"tag still has a separator: {runner.tag!r}")
        check(
            os.sep not in runner.model,
            f"model still has a separator: {runner.model!r}",
        )
        level_dir = runner._result_dir(0)
        check(
            os.path.abspath(os.path.dirname(level_dir))
            == os.path.abspath(os.path.join(tmp, "results", "level0", runner.model)),
            f"tag escaped its directory: {level_dir}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] a hostile --tag cannot escape or split the results directory")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_recovers_from_episode_records() -> None:
    """A truncated checkpoint must not silently restart the whole tag."""
    tmp = make_temp_dir()
    try:
        results_root = os.path.join(tmp, "results")
        model = "recover-model"
        tag = "recover-tag"
        level_dir = os.path.join(results_root, "level2", model, tag)
        os.makedirs(level_dir, exist_ok=True)
        for episode_id in (0, 1):
            persistence.atomic_write_json(
                os.path.join(level_dir, f"episode_{episode_id:03d}.json"),
                {"episode_id": episode_id, "level": 2, "steps": []},
            )
        # A third record that is present but damaged must NOT count as done.
        damaged = os.path.join(level_dir, "episode_002.json")
        with open(damaged, "w", encoding="utf-8") as fh:
            fh.write('{"episode_id": 2, "level": 2, "steps": [')

        checkpoint_path = os.path.join(results_root, model, f"checkpoint_{tag}.json")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            fh.write('{"completed": ["level2/episode000"')  # truncated

        checkpoint = ProtocolCheckpoint(checkpoint_path, resume=True)
        check(
            checkpoint.is_done("level2/episode000")
            and checkpoint.is_done("level2/episode001"),
            f"valid episodes were not recovered: {sorted(checkpoint.completed)}",
        )
        check(
            not checkpoint.is_done("level2/episode002"),
            "a damaged episode record was counted as completed",
        )
        check(
            checkpoint.recovered_from and os.path.exists(checkpoint.recovered_from),
            "the damaged checkpoint was not quarantined",
        )
        check(
            not os.path.exists(checkpoint_path),
            "the damaged checkpoint was left in place",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] a damaged checkpoint is quarantined and rebuilt from episode records")


def test_checkpoint_without_recoverable_data_starts_empty() -> None:
    tmp = make_temp_dir()
    try:
        path = os.path.join(tmp, "results", "m", "checkpoint_t.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("not json at all")
        checkpoint = ProtocolCheckpoint(path, resume=True)
        check(not checkpoint.completed, "a corrupt checkpoint invented completions")
        checkpoint.mark("level0/episode000")
        with open(path, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
        check(
            stored["completed"] == ["level0/episode000"],
            f"mark() did not rewrite the checkpoint: {stored}",
        )
        leftovers = [n for n in os.listdir(os.path.dirname(path)) if ".tmp-" in n]
        check(not leftovers, f"mark() left a temporary file behind: {leftovers}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] an unrecoverable checkpoint starts empty and is rewritten atomically")


def test_resume_reruns_a_damaged_episode() -> None:
    """--resume must re-run a damaged episode rather than skip it silently."""
    harness._install_scripted_agent("sideways")
    try:
        tmp = make_temp_dir()
        try:
            results_root = os.path.join(tmp, "results")
            logs_root = os.path.join(tmp, "logs")
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="damaged",
                max_steps=DEFAULT_MAX_STEPS,
                tag="damaged-run",
                results_root=results_root,
                logs_root=logs_root,
            )
            checkpoint_path = os.path.join(
                results_root, "damaged", "checkpoint_damaged-run.json"
            )
            first = runner.run_level(
                level=0, episodes=2, checkpoint=ProtocolCheckpoint(checkpoint_path)
            )
            check(len(first) == 2, "the first pass did not run both episodes")

            # Truncate one episode record and leave the checkpoint claiming it
            # is complete, which is exactly what an interrupted write leaves.
            damaged = runner._episode_path(0, 1)
            with open(damaged, "w", encoding="utf-8") as handle:
                handle.write('{"episode_id": 1, "level": 0, "steps": [')

            env2 = FakeBAOEnv()
            runner2 = BAOExperimentRunner(
                env=env2,
                model="damaged",
                max_steps=DEFAULT_MAX_STEPS,
                tag="damaged-run",
                results_root=results_root,
                logs_root=logs_root,
            )
            resumed = runner2.run_level(
                level=0,
                episodes=2,
                checkpoint=ProtocolCheckpoint(checkpoint_path, resume=True),
            )
            check(
                len(resumed) == 2, f"resume returned {len(resumed)} episodes, expected 2"
            )
            check(
                all(ep.get("steps") for ep in resumed),
                "a resume returned an episode with no step trace",
            )
            # Episode 0 was intact and must not have been run again.
            check(
                env2.steps < 2 * DEFAULT_MAX_STEPS,
                "resume re-ran every episode instead of only the damaged one",
            )
            backups = [
                n for n in os.listdir(os.path.dirname(damaged)) if ".corrupt-" in n
            ]
            check(backups, "the damaged episode record was not quarantined")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        harness._restore_agent_adapter()
    print("[ok] --resume re-runs a damaged episode and keeps a copy of the bad bytes")


def test_analysis_skips_a_damaged_record() -> None:
    tmp = make_temp_dir()
    try:
        results_root = os.path.join(tmp, "results")
        level_dir = os.path.join(results_root, "level0", "mixed")
        os.makedirs(level_dir, exist_ok=True)
        with open(
            os.path.join(level_dir, "episode_000.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(
                {
                    "episode_id": 0,
                    "level": 0,
                    "passed": True,
                    "passed_sideways": False,
                    "channel_width": 0.90,
                    "a_s_ratio": 1.58,
                    "steps": [
                        {"step": 0, "action": "forward", "torso_rotation": 0.0}
                    ],
                },
                fh,
            )
        with open(
            os.path.join(level_dir, "episode_001.json"), "w", encoding="utf-8"
        ) as fh:
            fh.write('{"episode_id": 1, "level": 0, "steps": [')
        episodes = analysis.load_episodes(results_root, 0, "mixed")
        check(
            len(episodes) == 1 and episodes[0]["episode_id"] == 0,
            f"analysis did not skip the damaged record: {episodes}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] the analyser skips a damaged record instead of aborting")


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------


def test_csv_export_is_tag_qualified_and_atomic() -> None:
    tmp = make_temp_dir()
    try:
        results_root = os.path.join(tmp, "results")
        episodes = [
            {
                "episode_id": 0,
                "level": 0,
                "channel_width": 0.90,
                "a_s_ratio": 1.58,
                "passed": True,
                "passed_sideways": False,
                "total_rotation": 0.0,
                "first_turn_step": None,
                "total_steps": 1,
                "action_sequence": "forward",
                "steps": [
                    {
                        "step": 0,
                        "action": "forward",
                        "torso_rotation": 0.0,
                        "position_x": 1.0,
                        "position_z": 0.0,
                        "collision": False,
                        "step_success": False,
                        "llm_response_time_ms": 12.5,
                    }
                ],
            }
        ]
        path = main_module.save_episodes_csv(
            episodes,
            model="vendor/model",
            level=0,
            results_root=results_root,
            timestamp="T",
            tag="run:1/../x",
        )
        check(os.path.exists(path), f"CSV was not written: {path}")
        check(
            os.path.dirname(path) == os.path.join(results_root, "vendor-model"),
            f"the model directory was not sanitised: {path}",
        )
        name = os.path.basename(path)
        check(
            "run-1-..-x" in name,
            f"the tag did not reach the CSV filename: {name}",
        )
        with open(path, "r", encoding="utf-8", newline="") as handle:
            header = handle.readline().strip().split(",")
        check("step" in header and "action" in header, f"unexpected CSV header: {header}")
        leftovers = [n for n in os.listdir(os.path.dirname(path)) if ".tmp-" in n]
        check(not leftovers, f"CSV export left temporary files: {leftovers}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] the flat CSV stays tag-qualified, sanitised and atomic")


def test_output_root_is_independent_of_cwd() -> None:
    script = (
        "import main, os;"
        "print(main.project_root());"
        "print(main.results_dir());"
        "print(main.logs_dir());"
        "print(main.progress_path())"
    )
    cwd = make_temp_dir()
    try:
        env = dict(os.environ)
        env.pop("BAO_OUTPUT_ROOT", None)
        # `python -c` puts the *current* directory on sys.path, so point the
        # import at the repository explicitly: this test is about where main.py
        # writes, not about how the module is found.
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
        )
        check(
            result.returncode == 0,
            f"importing main from another cwd failed: {result.stderr.strip()[:400]}",
        )
        lines = [
            line.strip() for line in result.stdout.strip().splitlines() if line.strip()
        ]
        check(len(lines) == 4, f"unexpected probe output: {result.stdout!r}")
        check(
            lines[0] == ROOT,
            f"project_root() followed the cwd: {lines[0]!r} != {ROOT!r}",
        )
        for line, name in zip(lines[1:], ("results", "logs", "run_progress.txt")):
            check(
                os.path.abspath(line).startswith(os.path.abspath(ROOT)),
                f"{name} is not anchored to the repository: {line!r}",
            )
            check(
                not os.path.abspath(line).startswith(os.path.abspath(cwd)),
                f"{name} followed the working directory: {line!r}",
            )
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    print("[ok] results, logs and the progress file ignore the working directory")


def test_output_root_override_and_progress_file() -> None:
    tmp = make_temp_dir()
    previous = os.environ.get("BAO_OUTPUT_ROOT")
    os.environ["BAO_OUTPUT_ROOT"] = tmp
    try:
        check(
            os.path.abspath(main_module.results_dir())
            == os.path.abspath(os.path.join(tmp, "results")),
            "BAO_OUTPUT_ROOT was ignored for results/",
        )
        check(
            os.path.abspath(main_module.progress_path())
            == os.path.abspath(os.path.join(tmp, "run_progress.txt")),
            "BAO_OUTPUT_ROOT was ignored for run_progress.txt",
        )
        main_module._write_progress("persistence-test")
        with open(
            os.path.join(tmp, "run_progress.txt"), "r", encoding="utf-8"
        ) as handle:
            content = handle.read()
        check("persistence-test" in content, f"progress line missing: {content!r}")
    finally:
        if previous is None:
            os.environ.pop("BAO_OUTPUT_ROOT", None)
        else:
            os.environ["BAO_OUTPUT_ROOT"] = previous
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] BAO_OUTPUT_ROOT relocates every artefact, including the progress file")


def test_run_level_writes_every_artifact() -> None:
    """The full on-disk contract for one Level, including the partial summary."""
    harness._install_scripted_agent("frontal")
    try:
        tmp = make_temp_dir()
        try:
            results_root = os.path.join(tmp, "results")
            logs_root = os.path.join(tmp, "logs")
            runner = BAOExperimentRunner(
                env=FakeBAOEnv(),
                model="artifact-model",
                max_steps=DEFAULT_MAX_STEPS,
                tag="artifact-tag",
                results_root=results_root,
                logs_root=logs_root,
            )
            runner.save_args(argparse.Namespace(model="artifact-model"))
            episodes = runner.run_level(level=0, episodes=2)
            check(len(episodes) == 2, "expected two episodes")

            level_dir = os.path.join(
                results_root, "level0", "artifact-model", "artifact-tag"
            )
            for name in (
                "episode_000.json",
                "episode_000_steps.json",
                "episode_001.json",
                "episode_001_steps.json",
                "summary_artifact-tag.json",
            ):
                path = os.path.join(level_dir, name)
                check(os.path.exists(path), f"missing artefact: {path}")
                with open(path, "r", encoding="utf-8") as handle:
                    json.load(handle)

            args_path = os.path.join(logs_root, "artifact-tag", "args.json")
            with open(args_path, "r", encoding="utf-8") as handle:
                args_payload = json.load(handle)
            check(
                args_payload.get("resolved_tag") == "artifact-tag",
                f"args.json does not record the resolved tag: {args_payload}",
            )
            agent_log = os.path.join(
                logs_root, "artifact-tag", "level0_episode000_agent.txt"
            )
            check(os.path.exists(agent_log), f"missing agent log: {agent_log}")
            with open(agent_log, "r", encoding="utf-8") as handle:
                log_text = handle.read()
            check(
                "episode start" in log_text,
                "the agent log has no per-attempt header",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        harness._restore_agent_adapter()
    print("[ok] one Level writes episodes, sidecars, a summary, args and logs")


def test_save_obs_flag_is_honoured_and_survives_bad_frames() -> None:
    """--save_obs writes PNGs, and a frame the encoder rejects is not fatal."""
    harness._install_scripted_agent("frontal")
    try:
        tmp = make_temp_dir()
        try:
            logs_root = os.path.join(tmp, "logs")
            runner = BAOExperimentRunner(
                env=FakeBAOEnv(),
                model="obs-model",
                max_steps=DEFAULT_MAX_STEPS,
                tag="obs-tag",
                save_obs=True,
                results_root=os.path.join(tmp, "results"),
                logs_root=logs_root,
            )
            check(os.path.isdir(runner.obs_dir), "the observation directory is missing")
            # A frame the encoder cannot serialise must not end the episode.
            runner._save_observation(0, 0, 0, "forward", "not-an-image")
            episodes = runner.run_level(level=0, episodes=1)
            check(len(episodes) == 1, "the episode did not complete")
            pngs = [n for n in os.listdir(runner.obs_dir) if n.endswith(".png")]
            check(pngs, f"no observation PNG was written to {runner.obs_dir}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        harness._restore_agent_adapter()
    print("[ok] --save_obs writes PNGs and tolerates an unusable frame")


TESTS = [
    test_atomic_write_json_round_trips,
    test_failed_write_keeps_the_previous_file,
    test_sanitize_tag_yields_one_safe_component,
    test_runner_tag_reaches_paths_intact,
    test_checkpoint_recovers_from_episode_records,
    test_checkpoint_without_recoverable_data_starts_empty,
    test_resume_reruns_a_damaged_episode,
    test_analysis_skips_a_damaged_record,
    test_csv_export_is_tag_qualified_and_atomic,
    test_output_root_is_independent_of_cwd,
    test_output_root_override_and_progress_file,
    test_run_level_writes_every_artifact,
    test_save_obs_flag_is_honoured_and_survives_bad_frames,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"[FAIL] {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{failures}/{len(TESTS)} persistence checks FAILED")
        return 1
    print(f"all {len(TESTS)} persistence checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
