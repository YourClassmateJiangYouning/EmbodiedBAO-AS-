"""Offline end-to-end checks of the Stage 2 runner.

The runner is the part of Stage 2 that spends money, so its plumbing is exercised
here against a mock environment and a scripted agent: no Isaac Sim, no API key, no
network.  What is checked is everything that would silently corrupt the experiment
rather than crash it:

1. **The loop** -- 6 runs of 17 rounds shape, but here one run at a time: a record
   and a step sidecar per round, in the documented layout.
2. **The memory** -- round 1's prompt has no memory block, round 2's quotes round
   1's note, and the block grows; rolling mode carries only the last note and no
   outcome log at all.
3. **The note call** -- once per round except the last (16 per run), stored verbatim
   in the record and in its own log file, and its prompt carries the reasoning in
   full.
4. **The phase change** -- the width really changes at round 13, and it is applied
   before the scene reset, with the built width read back.
5. **Resume** -- a finished run re-runs nothing, and a deleted record is the only
   thing re-run.
6. **The call budget** -- `--max_calls` stops cleanly with the finished rounds
   intact instead of raising out of `run_all`.

Run with a plain Python interpreter:

    python test_bao_memory_runner.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

import numpy as np

import experiments
import memory_experiment
import memory_metrics
import memory_protocol

WORKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_tmp", "memory_runner")


class Failure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


class MockResult:
    def __init__(self, feedback: str, legal: bool, collision: Optional[Dict[str, Any]], state: Dict[str, Any]):
        self.feedback = feedback
        self.legal = legal
        self.collision = collision
        self.state = state


class MockEnvironment:
    """A kinematic stand-in for BAOEnv: no walls, but the same interface."""

    def __init__(self) -> None:
        self.channel_width = memory_protocol.WIDTH_LEARNING_M
        self.width_history: List[float] = []
        self.resets = 0
        self.x = 0.5
        self.z = 0.0
        self.yaw = 0.0

    # -- configuration -------------------------------------------------
    def set_channel_width(self, width: float) -> None:
        self.channel_width = float(width)
        self.width_history.append(float(width))

    def get_channel_width(self) -> float:
        return self.channel_width

    def reset_scene(self) -> None:
        self.resets += 1
        self.x = 0.5
        self.z = 0.0
        self.yaw = 0.0

    # -- observation ---------------------------------------------------
    def get_camera_image(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_robot_state(self) -> Dict[str, Any]:
        return {
            "position": [self.x, 0.0, self.z],
            "torso_rotation": self.yaw,
            "camera_yaw": 0.0,
            "camera_pitch": 0.0,
            "move_step": memory_metrics.MOVE_STEP_M,
        }

    def get_torso_rotation(self) -> float:
        return self.yaw

    # -- actuation -----------------------------------------------------
    def execute_action(self, action: str) -> MockResult:
        if action == "forward":
            self.x = round(self.x + memory_metrics.MOVE_STEP_M, 6)
        elif action == "backward":
            self.x = round(self.x - memory_metrics.MOVE_STEP_M, 6)
        elif action == "left":
            self.z = round(self.z - memory_metrics.MOVE_STEP_M, 6)
        elif action == "right":
            self.z = round(self.z + memory_metrics.MOVE_STEP_M, 6)
        elif action in ("turn_left", "turn_right"):
            sign = 1.0 if action == "turn_left" else -1.0
            self.yaw = round(self.yaw + sign * memory_metrics.TURN_STEP_DEG, 6)
        return MockResult("executed", True, None, self.get_robot_state())

    def check_success(self) -> bool:
        return self.x >= memory_metrics.SUCCESS_X_M - 1e-9

    def check_collision_with_wall(self) -> bool:
        return False

    def close(self) -> None:
        pass


class FakeAdapter:
    """Scripted stand-in for ``experiments.AgentAdapter``.

    The instance index is the round number for a single-run invocation, which is
    what lets one scripted plan fail the first three rounds and pass the rest.
    """

    created: List["FakeAdapter"] = []
    # Step indices (per attempt) at which this agent replies with nothing usable.
    invalid_at: set = set()

    def __init__(self, model: str = "fake", log_file: Optional[str] = None) -> None:
        self.model = model
        self.log_file = log_file
        self.history: List[Dict[str, Any]] = []
        self.prompts: List[str] = []
        self.note_prompts: List[str] = []
        self.index = len(FakeAdapter.created) + 1
        FakeAdapter.created.append(self)

    def _plan(self) -> List[str]:
        if self.index <= 3:
            return ["look_down"] * 30
        return ["turn_left"] * 5 + ["forward"] * 11

    def query(self, prompt: str, image: Any, state: Dict[str, Any]):
        self.prompts.append(prompt)
        if len(self.history) in FakeAdapter.invalid_at:
            return None, "", "", 1.0
        plan = self._plan()
        action = plan[len(self.history)] if len(self.history) < len(plan) else "look_down"
        return action, "{}", "reasoning " * 8, 1.0

    def record(self, action: str, feedback: str, reasoning: str = "", step: Optional[int] = None) -> None:
        entry: Dict[str, Any] = {"action": action, "feedback": feedback}
        if reasoning:
            entry["reasoning"] = reasoning
        if step is not None:
            entry["step"] = int(step)
        self.history.append(entry)

    def note(self, prompt: str) -> str:
        self.note_prompts.append(prompt)
        return f"note from round {self.index}"


def fresh_workspace() -> None:
    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR)


def make_runner(runs: List[int], max_calls: int = 0, env: Optional[MockEnvironment] = None):
    return memory_experiment.MemoryExperimentRunner(
        env=env or MockEnvironment(),
        model="fake-model",
        runs=runs,
        results_root=os.path.join(WORKDIR, "results"),
        logs_root=os.path.join(WORKDIR, "logs"),
        max_calls=max_calls,
    )


class patched_adapter:
    """Swap in the scripted adapter for the duration of a block."""

    def __enter__(self):
        self._saved = experiments.AgentAdapter
        experiments.AgentAdapter = FakeAdapter
        FakeAdapter.created = []
        FakeAdapter.invalid_at = set()
        return FakeAdapter

    def __exit__(self, *exc: Any) -> None:
        experiments.AgentAdapter = self._saved
        return False


# ---------------------------------------------------------------------------
# 1. The loop, the records and the memory
# ---------------------------------------------------------------------------


def test_one_run_writes_every_round_and_grows_its_memory() -> None:
    fresh_workspace()
    env = MockEnvironment()
    with patched_adapter() as adapter_class:
        runner = make_runner([1], env=env)
        records = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))

    check(len(records) == 17, f"a run should be 17 rounds, got {len(records)}")
    check(
        [record["round"] for record in records] == list(range(1, 18)),
        "the rounds are out of order or missing",
    )
    for record in records:
        tag = runner.tag_for_run(1)
        check(
            os.path.exists(runner.record_path(tag, 1, record["round"])),
            f"round {record['round']} has no record file",
        )
        check(
            os.path.exists(runner.step_path(tag, 1, record["round"])),
            f"round {record['round']} has no step sidecar",
        )

    # The scripted plan fails the first three rounds and passes the rest.
    check(
        [record["passed"] for record in records[:3]] == [False, False, False],
        "the first three rounds were meant to fail",
    )
    check(all(record["passed"] for record in records[3:]), "rounds 4-17 were meant to pass")
    check(
        records[0]["end_reason"] == "max_steps" and records[0]["total_steps"] == 30,
        "a failing round should run to the 30-step cap",
    )
    check(
        records[3]["end_reason"] == "success" and records[3]["total_steps"] == 16,
        f"the optimal route passes in 16 steps, got {records[3]['total_steps']}",
    )
    check(records[3]["excess"] == 0, "the optimal route wastes no actions")
    check(
        records[3]["strategy_label"] == memory_metrics.ROT_EARLY,
        f"turning first at x = 0.5 should be ROT_EARLY, got {records[3]['strategy_label']}",
    )
    check(
        isinstance(records[3]["gap"], float) and records[3]["gap"] < 0.0,
        "75 degrees fits the A/S 0.80 opening, so its gap is negative",
    )
    check(
        records[12]["gap"] is None and records[12]["phase"] == memory_protocol.PHASE_PROBE,
        "the probe phase must not carry a gap",
    )

    # Round 1 has no memory at all; round 2 quotes round 1's note; round 3 quotes
    # both and reports their outcomes.
    first, second, third = adapter_class.created[0], adapter_class.created[1], adapter_class.created[2]
    check("Previous attempts" not in first.prompts[0], "round 1 was given a memory block")
    check("memory from" not in first.prompts[0].lower(), "round 1 was given a note")
    check(
        'your note: "note from round 1"' in second.prompts[0],
        "round 2 does not quote round 1's note",
    )
    check(
        "- round 1: failed (did not reach the red marker within 30 steps)" in second.prompts[0],
        "round 2 does not report round 1's outcome",
    )
    check(
        "- round 1:" in third.prompts[0] and "- round 2:" in third.prompts[0],
        "round 3 should see both earlier attempts",
    )
    check(
        records[0]["note_text"] == "note from round 1"
        and records[1]["memory_injected_chars"] > 0
        and records[0]["memory_injected_chars"] == 0,
        "the note or the injected memory size is not recorded as it was used",
    )
    check(
        records[2]["memory_injected_chars"] > records[1]["memory_injected_chars"],
        "the cumulative memory block should grow with every attempt",
    )
    print(
        f"[ok] one run writes 17 rounds with records and sidecars, fails where scripted, "
        f"and the cumulative memory grows from {records[0]['memory_injected_chars']} to "
        f"{records[16]['memory_injected_chars']} characters"
    )


def test_rolling_runs_carry_only_the_last_note() -> None:
    fresh_workspace()
    with patched_adapter() as adapter_class:
        runner = make_runner([5])
        records = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))

    check(
        all(record["memory_mode"] == memory_protocol.MODE_ROLLING for record in records),
        "run 5 should be a rolling run",
    )
    check(
        runner.tag_for_run(5) == f"{memory_experiment.default_tag_for_model('fake-model')}-roll",
        f"run 5's tag is {runner.tag_for_run(5)}",
    )
    prompt = adapter_class.created[3].prompts[0]
    check("Previous attempts" not in prompt, "the rolling block logged outcomes")
    check("- round 1:" not in prompt, "the rolling block carries round numbers")
    check("failed" not in prompt and "passed" not in prompt, "the rolling block reports outcomes")
    check(
        'Your note from your previous attempt:\n"note from round 3"' in prompt,
        "the rolling block should carry exactly the note the model rewrote last",
    )
    print("[ok] rolling runs carry only the single rewritten note: no round numbers, no outcomes")


def test_the_note_is_written_once_per_round_except_the_last() -> None:
    fresh_workspace()
    with patched_adapter() as adapter_class:
        runner = make_runner([1])
        records = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))

    total_notes = sum(len(adapter.note_prompts) for adapter in adapter_class.created)
    check(total_notes == 16, f"17 rounds should cost 16 note calls, got {total_notes}")
    tag = runner.tag_for_run(1)
    check(
        not os.path.exists(runner.note_path(tag, 1, 17)),
        "the last round of a run has no next attempt and should have no note",
    )
    check(
        os.path.exists(runner.note_path(tag, 1, 1)) and os.path.exists(runner.note_path(tag, 1, 16)),
        "a note log is missing",
    )
    with open(runner.note_path(tag, 1, 1), encoding="utf-8") as handle:
        body = handle.read()
    check("NOTE PROMPT:" in body and "NOTE RESPONSE:" in body, "the note log is not self-describing")
    check(
        "NOTE SYSTEM PROMPT:" in body and "plain text" in body,
        "the note log should record the note system prompt, which is not the action one",
    )
    check("note from round 1" in body, "the note log does not contain what the model wrote")
    check(
        records[16]["note_text"] == "",
        "the last round should record an empty note rather than inventing one",
    )
    # The note prompt carries the reasoning in full, unlike the action prompt.
    note_prompt = adapter_class.created[0].note_prompts[0]
    action_prompt = adapter_class.created[0].prompts[0]
    full_reasoning = "reasoning " * 8
    check(full_reasoning.strip() in note_prompt, "the note prompt lost the model's reasoning")
    check(
        note_prompt.count("reasoning") > action_prompt.count("reasoning"),
        "the note prompt should repeat reasoning per step, in full",
    )
    print("[ok] 17 rounds cost 16 note calls, each logged with its response, and the last one is empty")


# ---------------------------------------------------------------------------
# 2. The phase change
# ---------------------------------------------------------------------------


def test_the_width_changes_at_round_thirteen_before_the_reset() -> None:
    fresh_workspace()
    env = MockEnvironment()
    with patched_adapter():
        runner = make_runner([1], env=env)
        records = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))

    widths = env.width_history
    check(len(widths) == 17, f"the width should be set once per round, got {len(widths)} times")
    check(
        all(abs(width - memory_metrics.WIDTH_LEARNING_M) < 1e-12 for width in widths[:12]),
        "rounds 1-12 are not all at the A/S 0.80 width",
    )
    check(
        all(abs(width - memory_metrics.WIDTH_PROBE_M) < 1e-12 for width in widths[12:]),
        "rounds 13-17 are not all at the A/S 1.10 width",
    )
    check(env.resets == 17, f"the scene should be reset once per round, got {env.resets}")
    check(
        abs(records[11]["a_s_ratio"] - 0.80) < 1e-9 and abs(records[12]["a_s_ratio"] - 1.10) < 1e-9,
        "the recorded A/S does not switch between rounds 12 and 13",
    )
    check(
        records[12]["optimal_steps"] == 11 and records[11]["optimal_steps"] == 16,
        "the recorded optimum does not switch with the width",
    )
    print(
        f"[ok] the width is set and read back once per round: A/S 0.80 for rounds 1-12, "
        f"1.10 for 13-17, scene reset {env.resets} times"
    )


def test_a_scene_that_ignores_the_width_is_caught() -> None:
    class Stubborn(MockEnvironment):
        def get_channel_width(self) -> float:
            return memory_metrics.WIDTH_LEARNING_M

    fresh_workspace()
    with patched_adapter():
        runner = make_runner([1], env=Stubborn())
        runner.env.set_channel_width(memory_metrics.WIDTH_PROBE_M)
        try:
            runner.run_round(1, 13, [])
        except RuntimeError as exc:
            check("requires" in str(exc), f"unhelpful error: {exc}")
        else:
            raise Failure(
                "a scene that ignored the probe width was accepted; the probe phase would "
                "have run at the learning width"
            )
    print("[ok] a scene that does not actually build the requested width is rejected before use")


# ---------------------------------------------------------------------------
# 3. Resume and the call budget
# ---------------------------------------------------------------------------


def test_resume_reruns_only_what_is_missing() -> None:
    fresh_workspace()
    with patched_adapter():
        runner = make_runner([1])
        first = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))
        calls_after_first = runner.calls
        check(calls_after_first > 17 * 16, "the first pass should have spent its model calls")

    with patched_adapter():
        again = make_runner([1])
        repeated = again.run_all(checkpoints=again.make_checkpoints(resume=True))
    check(len(repeated) == 17, f"the resumed pass returned {len(repeated)} rounds")
    check(
        again.calls == 0,
        f"a finished run re-ran {again.calls} model call(s) instead of skipping them",
    )

    # Delete one record and the sidecar's partner; only that round may re-run.
    tag = again.tag_for_run(1)
    os.remove(again.record_path(tag, 1, 5))
    with patched_adapter():
        third = make_runner([1])
        recovered = third.run_all(checkpoints=third.make_checkpoints(resume=True))
    check(len(recovered) == 17, "the recovered pass lost rounds")
    check(third.calls > 0, "the deleted round was not re-run")
    check(
        third.calls <= 30 * 2,
        f"only the deleted round should have run, but {third.calls} calls were spent",
    )
    check(os.path.exists(again.record_path(tag, 1, 5)), "the deleted record was not rewritten")
    print(
        f"[ok] a finished run resumes with 0 model calls, and deleting one record re-runs "
        f"exactly that round ({third.calls} calls)"
    )


def test_the_call_budget_stops_cleanly() -> None:
    fresh_workspace()
    with patched_adapter():
        runner = make_runner([1], max_calls=20)
        records = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))
    check(records == [], f"the budget stopped mid-round but {len(records)} records appeared")
    tag = runner.tag_for_run(1)
    check(
        not os.path.exists(runner.record_path(tag, 1, 1)),
        "an unfinished round must not leave a record behind",
    )
    # The checkpoint must not claim the round either, or resume would skip it.
    checkpoint = memory_experiment.RoundCheckpoint(
        os.path.join(runner.record_dir(tag), "checkpoint.json"), resume=True
    )
    check(
        not checkpoint.is_done(memory_experiment.round_key(1, 1)),
        "the checkpoint claims a round that never finished",
    )
    print("[ok] --max_calls stops cleanly, leaves no partial record, and does not claim the round")


def test_checkpoint_rebuilds_from_the_records_it_can_read() -> None:
    fresh_workspace()
    with patched_adapter():
        runner = make_runner([1])
        runner.run_all(checkpoints=runner.make_checkpoints(resume=False))
    tag = runner.tag_for_run(1)
    path = os.path.join(runner.record_dir(tag), "checkpoint.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{ this is not json")
    rebuilt = memory_experiment.RoundCheckpoint(path, resume=True)
    check(
        len(rebuilt.completed) == 17,
        f"a damaged checkpoint should be rebuilt from the 17 records, got {len(rebuilt.completed)}",
    )
    check(rebuilt.recovered_from is not None, "the damaged checkpoint was not kept aside")
    print("[ok] a damaged checkpoint is quarantined and rebuilt from the round records on disk")


# ---------------------------------------------------------------------------
# 4. Small pieces
# ---------------------------------------------------------------------------


def test_run_selection_and_paths() -> None:
    check(memory_experiment.parse_runs("1-3,5") == [1, 2, 3, 5], "range parsing is wrong")
    check(memory_experiment.parse_runs("5,5,4") == [4, 5], "duplicates are not collapsed")
    for bad in ("7", "0", "1-9", "x"):
        try:
            memory_experiment.parse_runs(bad)
        except ValueError:
            pass
        else:
            raise Failure(f"{bad!r} should be rejected as a run selection")

    runner = make_runner([1, 5])
    check(runner.tag_for_run(1).endswith("-cum") and runner.tag_for_run(4).endswith("-cum"), "runs 1-4 are cumulative")
    check(runner.tag_for_run(5).endswith("-roll") and runner.tag_for_run(6).endswith("-roll"), "runs 5-6 are rolling")
    check(
        runner.record_path("t", 3, 7).replace("\\", "/").endswith("memory/fake-model/t/run03_round07.json"),
        f"record path is {runner.record_path('t', 3, 7)}",
    )
    check(
        runner.note_path("t", 3, 7).replace("\\", "/").endswith("logs/t/run03_round07_note.txt"),
        f"note path is {runner.note_path('t', 3, 7)}",
    )
    try:
        runner.tag_for_run(7)
    except ValueError:
        pass
    else:
        raise Failure("run 7 should be rejected")
    print("[ok] run selection, tags and the two path shapes are derived and validated")


def test_an_invalid_reply_is_recorded_rather_than_crashing_the_round() -> None:
    """The invalid-reply path has to record a step, not raise.

    This is a regression check with a specific history: the first version of the
    runner bound the step's ``collision`` flag only on the legal-action path, so the
    very first unparseable model reply raised NameError and would have ended a
    2-3 night sweep.  The mock agent here replies with nothing usable at steps 0 and
    1 of round 1, which is what a real model does when it ignores the JSON
    instruction.
    """
    fresh_workspace()
    with patched_adapter() as adapter_class:
        adapter_class.invalid_at = {0, 1}
        runner = make_runner([1])
        records = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))

    first = records[0]
    check(first["invalid_response_count"] == 2, f"expected 2 invalid replies, got {first['invalid_response_count']}")
    check(first["total_steps"] == 30, "an invalid reply is still a step and the attempt continues")
    tag = runner.tag_for_run(1)
    with open(runner.step_path(tag, 1, 1), encoding="utf-8") as handle:
        steps = json.load(handle)
    check(steps[0]["action"] == "invalid", f"step 0 recorded {steps[0]['action']!r}")
    check(
        steps[0]["collision"] is False and steps[1]["collision"] is False,
        "an invalid reply moved nothing and cannot have collided",
    )
    check(
        first["action_sequence"].startswith("invalid,invalid,"),
        f"the action sequence is {first['action_sequence'][:40]!r}",
    )
    print("[ok] an unusable model reply is recorded as a step with no collision, and the attempt continues")


def test_the_frame_size_is_set_for_the_client() -> None:
    """512 px reaches the client through the variable it reads.

    Stage 2 fixes the frame size to compare against the threshold sweep, and the
    only thing that reads it is ``ai_agent.encode_image``, through
    ``BAO_IMAGE_SIZE``.  A runner that never set it would send the camera's native
    1024 and cost about five times as much per step.
    """
    fresh_workspace()
    saved = os.environ.pop("BAO_IMAGE_SIZE", None)
    try:
        runner = make_runner([1])
        check(
            os.environ.get("BAO_IMAGE_SIZE") == "512",
            f"BAO_IMAGE_SIZE is {os.environ.get('BAO_IMAGE_SIZE')!r} after construction",
        )
        check(runner.image_size == 512, f"the runner reports image_size {runner.image_size}")
        other = memory_experiment.MemoryExperimentRunner(
            env=MockEnvironment(),
            model="fake-model",
            runs=[1],
            results_root=os.path.join(WORKDIR, "results"),
            logs_root=os.path.join(WORKDIR, "logs"),
            image_size=0,
        )
        check(other.image_size == 0, "image_size=0 should mean the camera's own size")
    finally:
        os.environ.pop("BAO_IMAGE_SIZE", None)
        if saved is not None:
            os.environ["BAO_IMAGE_SIZE"] = saved
    print("[ok] the runner sets BAO_IMAGE_SIZE to 512, which is the only thing the client reads")


def test_save_obs_writes_a_frame_per_step() -> None:
    try:
        import PIL  # noqa: F401
    except Exception:
        # The runner is supposed to disable observations rather than fail when
        # Pillow is missing, which is the case on the Isaac Sim workstation, so
        # there is nothing to check here beyond that behaviour.
        runner = memory_experiment.MemoryExperimentRunner(
            env=MockEnvironment(),
            model="fake-model",
            runs=[1],
            results_root=os.path.join(WORKDIR, "results"),
            logs_root=os.path.join(WORKDIR, "logs"),
            save_obs=True,
        )
        check(
            runner.save_obs is False,
            "without Pillow the runner should switch observations off instead of "
            "warning once per step for the whole sweep",
        )
        print("[skip] --save_obs needs Pillow (absent here); the runner disables it with one warning")
        return

    fresh_workspace()
    with patched_adapter():
        runner = memory_experiment.MemoryExperimentRunner(
            env=MockEnvironment(),
            model="fake-model",
            runs=[1],
            results_root=os.path.join(WORKDIR, "results"),
            logs_root=os.path.join(WORKDIR, "logs"),
            save_obs=True,
        )
        runner.run_all(checkpoints=runner.make_checkpoints(resume=False))
    tag = runner.tag_for_run(1)
    directory = runner.obs_dir(tag)
    frames = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
    check(frames, "--save_obs wrote no frames")
    check(
        frames[0].startswith("run01_round01_step00_") and frames[0].endswith(".png"),
        f"the first frame is named {frames[0]}",
    )
    check(
        len(frames) > 100,
        f"only {len(frames)} frames for 17 rounds; --save_obs is meant to save every step",
    )
    print(f"[ok] --save_obs writes one frame per step ({len(frames)} files, named run/round/step)")


def test_the_default_tag_is_model_scoped() -> None:
    """Two models must never share a log directory.

    Notes and per-step agent logs live under ``logs/{tag}/``, which is not
    model-scoped.  A stem that did not carry the model name would make the nine
    models with no request parameters write to the same paths, so each model's notes
    would overwrite the previous model's -- and the notes ARE the manipulation.
    """
    import json

    project_root = os.path.dirname(os.path.abspath(__file__))
    roster = [
        entry["runner"] for entry in json.load(
            open(os.path.join(project_root, "models.json"), encoding="utf-8")
        )["models"]
    ]
    stems = {model: memory_experiment.default_tag_for_model(model) for model in roster}
    check(
        len(set(stems.values())) == len(roster),
        f"the roster does not get one tag each: {sorted(set(stems.values()))}",
    )
    for model, stem in stems.items():
        check(stem.startswith(model + "-"), f"{model} produced the stem {stem}")
        check(
            memory_protocol.PROTOCOL_TAG in stem,
            f"{stem} does not carry the Stage 2 protocol version",
        )
    check(
        stems["deepseek-v4.1-flash"].endswith("effortnone")
        and stems["glm-4.6v"].endswith("nothinking"),
        "the per-model request parameters are missing from the tags",
    )

    explicit = memory_experiment.MemoryExperimentRunner(
        env=MockEnvironment(),
        model="fake-model",
        runs=[1],
        tag="pilot",
        results_root=os.path.join(WORKDIR, "results"),
        logs_root=os.path.join(WORKDIR, "logs"),
    )
    check(explicit.tag == "pilot", f"an explicit tag was not honoured: {explicit.tag}")
    default = make_runner([1])
    check(
        default.tag == memory_experiment.default_tag_for_model("fake-model"),
        f"the runner's default tag is {default.tag}",
    )
    check(
        len({memory_experiment.default_tag_for_model(model) for model in roster}) == len(roster),
        "two models share a tag",
    )
    print(
        f"[ok] the {len(roster)} roster models get {len(set(stems.values()))} distinct tags, each "
        f"carrying the model name and the protocol (e.g. {stems['deepseek-v4.1-flash']})"
    )


def test_main_runs_end_to_end_with_a_stubbed_app() -> None:
    """main() is what actually runs on the workstation, so it gets exercised too.

    A regression check with a specific history: after ``run_all``'s parameter was
    renamed from ``checkpoint`` to ``checkpoints`` the call inside ``main`` still said
    ``checkpoint``, so the first lab run printed its banner, raised TypeError, and
    took the process down through ``env.close()`` before the traceback reached the
    terminal.  Nothing caught it because every other test drives the runner directly,
    and main() needs SimulationApp.  Everything Isaac-shaped is stubbed here.
    """
    import types

    fresh_workspace()
    saved_argv = list(sys.argv)
    saved_cwd = os.getcwd()
    saved_adapter = experiments.AgentAdapter
    import environment as environment_module

    saved_bao_env = environment_module.BAOEnv
    saved_isaacsim = sys.modules.get("isaacsim")
    calls: Dict[str, Any] = {}

    class FakeApp:
        def __init__(self, config: Any) -> None:
            calls["config"] = config

        def close(self) -> None:
            calls["app_closed"] = True

    class FakeEnv:
        def __init__(self, app: Any, task_dict: Any = None) -> None:
            self.inner = MockEnvironment()
            calls["task_dict"] = task_dict

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def close(self) -> None:
            calls["env_closed"] = True

    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = FakeApp
    sys.modules["isaacsim"] = fake_isaacsim
    environment_module.BAOEnv = FakeEnv
    experiments.AgentAdapter = FakeAdapter
    FakeAdapter.created = []
    FakeAdapter.invalid_at = set()
    sys.argv = [
        "memory_experiment.py", "--model", "fake-model", "--runs", "1", "--max_calls", "40",
    ]
    try:
        os.chdir(WORKDIR)
        memory_experiment.main()
    finally:
        os.chdir(saved_cwd)
        sys.argv = saved_argv
        environment_module.BAOEnv = saved_bao_env
        experiments.AgentAdapter = saved_adapter
        if saved_isaacsim is None:
            sys.modules.pop("isaacsim", None)
        else:
            sys.modules["isaacsim"] = saved_isaacsim

    # main() closes the ENVIRONMENT; whether that also closes the app is the
    # environment's business, and the threshold study's entry point makes the same
    # division.
    check(calls.get("env_closed") is True, "main() did not close the environment")
    check(
        calls.get("task_dict", {}).get("headless") is False,
        f"main() did not pass headless through to the environment: {calls.get('task_dict')}",
    )
    check(
        os.path.isdir(os.path.join(WORKDIR, "results", "memory", "fake-model")),
        "main() wrote no results directory",
    )
    check(
        os.path.exists(os.path.join(WORKDIR, "logs", "fake-model-v8-memory-a08-a11-cum", "args.json")),
        "main() did not record the effective settings",
    )
    # The timeline opens the way the threshold study's does, which is what makes
    # "which model is running" answerable from run_progress.txt alone.
    progress = os.path.join(WORKDIR, "run_progress.txt")
    check(os.path.exists(progress), "main() wrote no run_progress.txt")
    with open(progress, encoding="utf-8") as handle:
        timeline = handle.read()
    check(
        "SimulationApp started" in timeline,
        f"the timeline has no start line: {timeline[:200]!r}",
    )
    check(
        "agent ready: fake-model" in timeline,
        f"the timeline has no agent-ready line: {timeline[:200]!r}",
    )
    check(
        "pass_rate=" in timeline and "run1/round01" in timeline,
        "the timeline does not report each round with a pass rate",
    )
    print("[ok] main() parses, constructs, saves args, runs a round and closes cleanly under a stubbed app")


def test_a_round_carries_the_threshold_studys_fields_and_leaves_a_summary() -> None:
    """The Stage 1 standard: same comparability fields, and a summary on disk.

    The threshold study writes a per-Level summary after every episode so an
    interrupted sweep still has a summary matching what it finished, and it records
    whether the episode turned, how far it rotated in total, the angle it ended on and
    whether it passed sideways.  Stage 2 has to carry the same fields or the two
    datasets cannot be tabulated together without re-deriving them.
    """
    fresh_workspace()
    with patched_adapter():
        runner = make_runner([1, 5])
        records = runner.run_all(checkpoints=runner.make_checkpoints(resume=False))

    for record in records:
        for name in (
            "turned",
            "total_rotation",
            "final_torso_rotation",
            "first_sideways_step",
            "passed_sideways",
        ):
            check(
                name in record,
                f"{name} is missing from run {record['run']} round {record['round']}",
            )
    turning = [r for r in records if r["n_turn"] >= 5]
    check(turning, "the scripted plan turns in most rounds, so some record should")
    sample = turning[0]
    check(sample["turned"] is True, "a round with turns should report turned=True")
    check(
        abs(sample["total_rotation"] - sample["n_turn"] * memory_metrics.TURN_STEP_DEG) < 1e-9,
        f"total_rotation {sample['total_rotation']} does not match its turn count",
    )
    check(
        abs(sample["final_torso_rotation"] - 75.0) < 1e-9,
        f"final torso angle is {sample['final_torso_rotation']}, expected 75",
    )
    check(
        sample["first_sideways_step"] == 2,
        f"the first sideways step is {sample['first_sideways_step']}; three 15 degree "
        "turns reach the 45 degree band, and they are steps 0, 1 and 2",
    )
    check(
        sample["passed_sideways"] is True,
        "a round that passed at 75 degrees has passed sideways",
    )
    front = [r for r in records if r["n_turn"] == 0]
    check(front, "the scripted plan has rounds that never turn, so one should be here")
    check(
        front[0]["turned"] is False and front[0]["first_sideways_step"] is None,
        "a round that never turned cannot have entered the sideways band",
    )

    tag = runner.tag_for_run(1)
    summary_path = os.path.join(runner.record_dir(tag), "summary.json")
    check(os.path.exists(summary_path), "no per-tag summary was written")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    check(summary["rounds"] == 17, f"the summary covers {summary['rounds']} rounds")
    check(
        summary["memory_mode"] == memory_protocol.MODE_CUMULATIVE,
        f"the summary says the mode is {summary['memory_mode']}",
    )
    run_one = summary["runs"].get("1")
    check(run_one is not None, "the summary has no entry for run 1")
    for name in ("d", "first_pass_round", "acquired", "terminal_state",
                 "probe_rotation_deg", "probe_turn", "probe_look_down",
                 "excess_series", "curve"):
        check(name in run_one, f"the summary's run block has no {name}")
    check(
        run_one["curve"].get("label") in (
            memory_metrics.INSIGHT, memory_metrics.GRADUAL,
            memory_metrics.PERSEVERATION, memory_metrics.OSCILLATING,
            memory_metrics.NEVER_APPROACHED,
        ),
        f"the summary's curve verdict is {run_one['curve'].get('label')!r}",
    )
    check(
        len(run_one["probe_rotation_deg"]) == memory_protocol.ROUNDS_PROBE,
        "the probe series should have five entries",
    )

    progress = os.path.join(os.path.dirname(runner.results_root), "run_progress.txt")
    check(os.path.exists(progress), "no run_progress.txt timeline was written")
    with open(progress, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    check(
        len(lines) == len(records),
        f"{len(lines)} progress lines for {len(records)} rounds",
    )
    check(
        " passed=" in lines[0] and "round01" in lines[0],
        f"the first progress line is {lines[0]!r}",
    )

    sidecar = None
    with open(runner.step_path(tag, 1, 1), encoding="utf-8") as handle:
        sidecar = json.load(handle)
    check(
        "step_success" in sidecar[0],
        "the step sidecar is missing step_success, which the threshold study records",
    )
    print(
        "[ok] rounds carry turned/total_rotation/final_torso_rotation/first_sideways_step/"
        "passed_sideways, steps carry step_success, and each round leaves a summary and "
        "a progress line"
    )


def test_the_stage2_csv_lines_up_with_the_stage1_csv() -> None:
    """The two tables must share their column names, and this uses the real Stage 1 CSV.

    Not a copy of its column list: the file is in the repository, so renaming a column
    on either side fails here.  That is the mistake that would break a merged table or
    a figure redrawn across both experiments, and it is invisible otherwise.
    """
    import csv as csv_module

    fresh_workspace()
    with patched_adapter():
        runner = make_runner([1, 5])
        runner.run_all(checkpoints=runner.make_checkpoints(resume=False))

    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root, "lab_logs"))
    import export_memory_table as exporter
    import export_table as stage1_exporter

    # The contract is the Stage 1 exporter's column list, not a shipped file: the CSV is
    # generated from the archive, so requiring it to be present would make this suite
    # fail on a clone whose data file was never checked out -- which is exactly what
    # happened on the workstation.  When the file IS there, its header is compared too,
    # so drift between the shipped table and the exporter that writes it is caught.
    shared = set(stage1_exporter.COLUMNS) - {"level", "episode_id"}
    missing = shared - set(exporter.ROUND_COLUMNS)
    check(not missing, f"the Stage 2 table is missing {sorted(missing)} from Stage 1's")
    stage1_csv = os.path.join(root, "lab_logs", "embodiedbao_v7_episodes.csv")
    if os.path.exists(stage1_csv):
        with open(stage1_csv, newline="", encoding="utf-8") as handle:
            shipped = set(next(csv_module.reader(handle)))
        check(
            shipped == set(stage1_exporter.COLUMNS),
            "the committed Stage 1 CSV and its exporter disagree on: %s"
            % sorted(shipped ^ set(stage1_exporter.COLUMNS)),
        )

    records = exporter.load_records(os.path.join(runner.results_root, "memory"))
    rounds = [exporter.round_row(r) for r in records]
    runs = exporter.run_rows(records)
    check(len(rounds) == 34, f"the round table has {len(rounds)} rows for two runs")
    check(len(runs) == 2, f"the run table has {len(runs)} rows for two runs")
    check(
        all(name in rounds[0] for name in shared),
        "a row is missing a column the header advertises",
    )
    # The two passages must be the two the protocol names: the narrow one fits nobody
    # and the wide one needs no turn.  Checked per phase, because a table that put one
    # width on every row would still look plausible.
    learning = [row for row in rounds if row["phase"] == "A"]
    probe = [row for row in rounds if row["phase"] == "B"]
    check(learning and probe, f"the table has {len(learning)} A rows and {len(probe)} B")
    check(
        all(abs(float(row["channel_width"]) - 0.456) < 1e-6 for row in learning),
        "an A row is not at 0.456 m: %s"
        % sorted({row["channel_width"] for row in learning}),
    )
    check(
        all(abs(float(row["channel_width"]) - 0.627) < 1e-6 for row in probe),
        "a B row is not at 0.627 m: %s" % sorted({row["channel_width"] for row in probe}),
    )
    # An attempt that never got near the wall has no gap, and writing 0 or -1 there
    # would put a number into the mean that was never measured.
    unmeasured = [row for row in rounds if row["gap"] == ""]
    check(
        unmeasured,
        "no row has an empty gap, but the scripted plan has attempts that never move",
    )
    check(
        all(isinstance(row["gap"], float) for row in rounds if row["gap"] != ""),
        "a measured gap is not a number",
    )
    probe_keys = ["probe_rotation_%d" % i for i in range(1, memory_protocol.ROUNDS_PROBE + 1)]
    check(
        all(isinstance(runs[0][key], float) for key in probe_keys),
        "the run table's probe series is not five numbers: %s"
        % [runs[0].get(key) for key in probe_keys],
    )
    check(
        runs[0]["curve_label"] in (
            memory_metrics.INSIGHT, memory_metrics.GRADUAL,
            memory_metrics.PERSEVERATION, memory_metrics.OSCILLATING,
            memory_metrics.NEVER_APPROACHED,
        ),
        f"the run table's verdict is {runs[0]['curve_label']!r}",
    )
    check(
        "," in rounds[0]["action_sequence"],
        f"action_sequence is {rounds[0]['action_sequence']!r}",
    )
    print(
        "[ok] the Stage 2 round and run tables carry every column name the Stage 1 CSV "
        "shares, keep an unmeasured gap empty, and classify every run"
    )


def main() -> int:
    tests = [
        test_one_run_writes_every_round_and_grows_its_memory,
        test_rolling_runs_carry_only_the_last_note,
        test_the_note_is_written_once_per_round_except_the_last,
        test_the_width_changes_at_round_thirteen_before_the_reset,
        test_a_scene_that_ignores_the_width_is_caught,
        test_resume_reruns_only_what_is_missing,
        test_the_call_budget_stops_cleanly,
        test_an_invalid_reply_is_recorded_rather_than_crashing_the_round,
        test_the_frame_size_is_set_for_the_client,
        test_save_obs_writes_a_frame_per_step,
        test_checkpoint_rebuilds_from_the_records_it_can_read,
        test_run_selection_and_paths,
        test_the_default_tag_is_model_scoped,
        test_main_runs_end_to_end_with_a_stubbed_app,
        test_a_round_carries_the_threshold_studys_fields_and_leaves_a_summary,
        test_the_stage2_csv_lines_up_with_the_stage1_csv,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Failure as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    print()
    if failures:
        print(f"{failures}/{len(tests)} checks FAILED")
        return 1
    print(f"all {len(tests)} memory-runner checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
