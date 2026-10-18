"""End-to-end integration test for the EmbodiedBAO runner.

Isaac Sim is unavailable in most development environments, so this test drives
the *entire* experiment pipeline -- ``BAOExperimentRunner`` -> episode records
-> Level summaries -> CSV export -> ``analysis.py`` -- against a kinematic mock
environment built from the real pure-geometry helpers in ``environment.py``.

What it verifies:

* the runner's step loop, record fields and success detection,
* the exact field contract from the task brief (episode-level and step-level),
* collisions are recorded but never terminate an episode,
* a scripted "always frontal" policy passes Levels 0-4 and fails Level 5,
* a scripted "rotate then walk forward" policy passes every Level and is scored as
  a sideways passage, including when it straightens up again after getting
  through,
* the exported CSV carries the required columns,
* ``analysis.py`` recovers the right sideways threshold from saved results,
* the within-episode action history reaches every model call in full.

Run with a plain Python interpreter:

    python test_bao_integration.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environment import (
    CAMERA_PITCH_STEP_DEG,
    CAMERA_TURN_STEP_DEG,
    LEVEL_CHANNEL_WIDTHS,
    MOVE_STEP,
    ROBOT_START_POS,
    SUCCESS_X,
    TURN_STEP_DEG,
    WALL_X,
    _check_wall_collision,
    _translation_path_is_clear,
    _turn_path_is_clear,
    action_delta,
    a_s_ratio,
)
from experiments import (
    DEFAULT_MAX_STEPS,
    BAOExperimentRunner,
    ProtocolCheckpoint,
)
import analysis

ROBOT_START = ROBOT_START_POS.copy()


class Failure(AssertionError):
    pass


class Skipped(Exception):
    """A test that could not run on this interpreter (missing Pillow, ruff, ...).

    Raised instead of returning: a test that returns counts as passed by the
    runner, so a skip and a pass were indistinguishable in the "all N passed"
    line.  ``main`` catches this separately and reports it as skipped.
    """


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# Test artefacts stay inside the project directory: the harness file sandbox
# only guarantees write access to the workspace, not to the system temp dir.
_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_tmp")
_TMP_COUNTER = {"n": 0}


def make_temp_dir() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    _TMP_COUNTER["n"] += 1
    path = os.path.join(_TMP_ROOT, f"case{_TMP_COUNTER['n']:02d}")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


class FakeBAOEnv:
    """Kinematic stand-in for ``BAOEnv`` that reuses its geometry helpers."""

    def __init__(self, channel_width: float = 0.90) -> None:
        self._channel_width = float(channel_width)
        self._position = ROBOT_START.copy()
        self._yaw = 0.0
        self._camera_yaw = 0.0
        self._camera_pitch = 0.0
        self.steps = 0

    # -- scene ---------------------------------------------------------
    def set_channel_width(self, width: float) -> float:
        self._channel_width = float(width)
        return self._channel_width

    def get_channel_width(self) -> float:
        return float(self._channel_width)

    def get_a_s_ratio(self) -> float:
        return a_s_ratio(self._channel_width)

    def reset_scene(self, episode_id: Optional[int] = None) -> np.ndarray:
        # The real environment takes the episode index so the marker slot is a pure function of it
        # rather than a process-local reset counter (which drifted under --resume).  The mock mirrors
        # that interface, and records the slot, so the contract test can see it in the record.
        if episode_id is None:
            self._current_marker_slot = 1
        else:
            self._current_marker_slot = int(episode_id) % 5 + 1
        self._robot_ground_offset = 1.0442
        self._position = ROBOT_START.copy()
        self._yaw = 0.0
        self._camera_yaw = 0.0
        self._camera_pitch = 0.0
        self.steps = 0
        return self.get_camera_image()

    reset = reset_scene

    def get_camera_image(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    # -- state ---------------------------------------------------------
    def get_robot_state(self) -> Dict[str, Any]:
        return {
            "position": self._position.tolist(),
            "torso_rotation": float(self._yaw),
            "camera_yaw": float(self._camera_yaw),
            "camera_pitch": float(self._camera_pitch),
            "channel_width": self.get_channel_width(),
            "a_s_ratio": self.get_a_s_ratio(),
            "move_step": float(getattr(self, "_move_step", MOVE_STEP)),
        }

    def get_torso_rotation(self) -> float:
        return float(self._yaw)

    # -- task ----------------------------------------------------------
    def check_success(self) -> bool:
        return bool(self._position[0] >= SUCCESS_X)

    def check_collision_with_wall(self) -> bool:
        return (
            _check_wall_collision(
                self._position, np.radians(self._yaw), self._channel_width
            )
            is not None
        )

    def execute_action(self, action: str) -> Any:
        from environment import StepResult

        self.steps += 1
        action = str(action).strip().lower()
        legal = True
        feedback = "executed"
        collision = None
        yaw_rad = np.radians(self._yaw)
        # Mirrors the real _apply_action: every glance lasts one frame, so both
        # offsets are cleared for each action and set only by a look_*.  This fake
        # must track the environment it stands in for, or the offset test passes
        # or fails on the fake instead of on the protocol.
        self._camera_yaw = 0.0
        self._camera_pitch = 0.0

        if action in ("forward", "backward", "left", "right"):
            # The shipped walking frame, not a re-derived one: this fake must
            # mirror _apply_action or it silently validates the wrong protocol.
            direction = action_delta(action, MOVE_STEP)
            candidate = self._position + direction
            collision = _translation_path_is_clear(
                self._position, candidate, yaw_rad, self._channel_width
            )
            if collision is not None:
                legal = False
                feedback = f"blocked by obstacle or room boundary ({collision['part']})"
            else:
                self._position = candidate
        elif action in ("turn_left", "turn_right"):
            delta = TURN_STEP_DEG if action == "turn_left" else -TURN_STEP_DEG
            candidate_yaw = self._yaw + delta
            collision = _turn_path_is_clear(
                self._position, self._yaw, candidate_yaw, self._channel_width
            )
            if collision is not None:
                legal = False
                feedback = "cannot turn: body would collide with obstacle or room boundary"
            else:
                self._yaw = candidate_yaw
        elif action in ("look_left", "look_right", "look_down"):
            # Assignment, not accumulation: the real environment returns the gaze
            # to straight ahead instead of letting two glances add up to 60, and
            # look_down pitches the head further down on its own axis.
            #
            # The SIGNS come from the real environment (_apply_action): look_left is
            # -CAMERA_TURN_STEP_DEG, because the reported camera_yaw is the heading
            # frame (negative = the walker's left).  This double used to hardcode
            # the pre-2026-10-16 opposite, which let the prompt legend and the fake
            # agree with each other and disagree with the simulator.
            if action == "look_down":
                self._camera_pitch = CAMERA_PITCH_STEP_DEG
            else:
                self._camera_yaw = (
                    -CAMERA_TURN_STEP_DEG
                    if action == "look_left"
                    else CAMERA_TURN_STEP_DEG
                )
        else:
            legal = False
            feedback = f"unknown action: {action}"

        return StepResult(
            rgb=self.get_camera_image(),
            legal=legal,
            feedback=feedback,
            success=self.check_success(),
            collision=collision,
            state=self.get_robot_state(),
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Scripted policies (installed in place of the model client)
# ---------------------------------------------------------------------------


class ScriptedAgent:
    """Deterministic policy used instead of a real MLLM."""

    def __init__(self, policy: str) -> None:
        self.policy = policy
        self.history: List[Dict[str, str]] = []
        self.phase = 0

    def get_action(self, **kwargs: Any) -> Dict[str, Any]:
        state = kwargs.get("state") or {}
        yaw = float(state.get("torso_rotation", 0.0))
        position = state.get("position", ROBOT_START.tolist())
        x, z = float(position[0]), float(position[2])

        if self.policy == "frontal":
            action = "forward"
        elif self.policy == "sideways":
            # Route, using rotation to change the body's width, not to steer.
            # The phase counter is essential: a stateless rule like "turn until
            # yaw is 90" oscillates forever once it is one step past the target.
            #   0. turn_left x6  -> the torso is 90 degrees to the walking
            #      direction, so the body is only 0.22 m wide across the opening
            #   1. "forward"     -> walks at the far wall; the direction is fixed,
            #      so the rotated body goes through the channel sideways
            #
            # The agent never needs to rotate back: the channel is an opening
            # through the wall, so once the body is past it the walk continues to
            # the success plane (x = 8.75).  Rotating back mid-channel is what the
            # gate correctly forbids, because the shoulders would sweep the panel.
            if self.phase == 0:
                if yaw >= 90.0 - 1e-6:
                    self.phase = 1
                else:
                    return self._reply("turn_left")
            action = "forward"
        elif self.policy == "sideways-then-straight":
            # What a person actually does: rotate to get through, then straighten
            # up again once past the wall.  The rotation that matters happened at
            # the wall plane, so scoring the orientation at the END of the episode
            # would call this "never turned sideways" while the final torso
            # rotation is back at 0.
            if self.phase == 0:
                if yaw >= 90.0 - 1e-6:
                    self.phase = 1
                else:
                    return self._reply("turn_left")
            if self.phase == 1:
                if x >= WALL_X:
                    self.phase = 2
                else:
                    return self._reply("forward")
            if self.phase == 2:
                if yaw <= 1e-6:
                    self.phase = 3
                else:
                    return self._reply("turn_right")
            action = "forward"
        elif self.policy == "random_walk":
            action = "forward" if (x + z) % 2 < 1 else "turn_left"
        elif self.policy == "look-then-forward":
            # Glance left twice and down once, then walk.  Used to check that the
            # head-camera offsets the environment tracks actually become visible to
            # the agent: a client that never reports them reads a view 30 degrees
            # off its walking direction (or 45 degrees down) as if it were normal.
            # Two glances in the same direction must NOT add up, and look_down must
            # be reported on its own axis.
            if self.phase < 2:
                self.phase += 1
                return self._reply("look_left")
            if self.phase == 2:
                self.phase = 3
                return self._reply("look_down")
            action = "forward"
        else:
            raise ValueError(self.policy)
        return self._reply(action)

    def _reply(self, action: str) -> Dict[str, Any]:
        return {"action": action, "confidence": 1.0, "reasoning": self.policy}


def _install_scripted_agent(policy: str) -> None:
    """Patch ``experiments.AgentAdapter`` to return a scripted policy."""
    import experiments

    class _Adapter(experiments.AgentAdapter):  # type: ignore[misc]
        def __init__(self, model: str, log_file: Optional[str] = None) -> None:
            self.model = model
            self.log_file = None
            self.history = []
            self._callable = None
            self._agent = ScriptedAgent(policy)

    experiments.AgentAdapter = _Adapter


def _restore_agent_adapter() -> None:
    import importlib

    import experiments

    importlib.reload(experiments)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

EPISODE_FIELDS = {
    "episode_id",
    "level",
    "channel_width",
    "a_s_ratio",
    "passed",
    "passed_sideways",
    # Recorded so the marker slot and the robot ground offset are auditable from
    # results/.  They were written but asserted by no test, and the comment claiming
    # otherwise pointed at test_bao_geometry, which never mentioned them.
    #
    # scene belongs with them: a marker slot is only meaningful together with the scene it belongs to,
    # because slot 3 of stage1.1 and slot 3 of stage1.2 are different conditions. The record carried
    # only the slot, so the scene was recoverable from the tag directory and nowhere else.
    "scene",
    "marker_slot",
    "robot_ground_offset",
    "total_rotation",
    "first_turn_step",
    "total_steps",
    "action_sequence",
}

STEP_FIELDS = {
    "step",
    "action",
    "torso_rotation",
    "position_x",
    "position_z",
    "collision",
    "step_success",
}


def test_episode_record_contract() -> None:
    """Every required episode and step field must be present and typed."""
    _install_scripted_agent("frontal")
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-frontal",
                max_steps=DEFAULT_MAX_STEPS,
                tag="contract",
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            episodes = runner.run_level(level=0, episodes=2)
            check(len(episodes) == 2, f"expected 2 episodes, got {len(episodes)}")

            for episode in episodes:
                missing = EPISODE_FIELDS - set(episode)
                check(not missing, f"episode record missing fields: {sorted(missing)}")
                # Present is not enough: the two auditable fields must carry the
                # value the environment actually used, or a later reader cannot tell
                # which marker an episode saw or whether the robot was on the floor.
                check(
                    int(episode["marker_slot"]) in (1, 2, 3, 4, 5),
                    f"marker_slot {episode['marker_slot']!r} is not a recorded slot",
                )
                check(
                    float(episode["robot_ground_offset"]) > 0.0,
                    "robot_ground_offset was recorded as "
                    f"{episode['robot_ground_offset']!r}, not the lift in use",
                )
                check(
                    episode["channel_width"] == LEVEL_CHANNEL_WIDTHS[0],
                    f"level 0 channel width is {episode['channel_width']}",
                )
                check(
                    abs(episode["a_s_ratio"] - a_s_ratio(LEVEL_CHANNEL_WIDTHS[0])) < 1e-12,
                    "a_s_ratio does not match the channel width",
                )
                check(episode["total_steps"] == len(episode["steps"]), "step count drift")
                for step in episode["steps"]:
                    missing = STEP_FIELDS - set(step)
                    check(not missing, f"step record missing fields: {sorted(missing)}")

            # Level 0 is 0.90 m wide: a frontal walk must pass within 30 steps.
            check(
                all(ep["passed"] for ep in episodes),
                "frontal policy should pass at Level 0",
            )
            check(
                not any(ep["passed_sideways"] for ep in episodes),
                "frontal policy should not be scored as sideways",
            )
            check(
                all(ep["first_turn_step"] is None for ep in episodes),
                "frontal policy should never turn",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] episode and step records match the required field contract")


def test_frontal_policy_matches_design_table() -> None:
    """A straight-walking agent must pass exactly where A/S >= 1.0.

    A/S = 1.00 -- channel 0.570 m, shoulders 0.570 m -- is geometrically possible
    to walk straight through, and the design calls it a frontal passage.  It
    expected False while the collision body carried a 2 mm skin, which made the
    gate require 0.574 m and quietly turned that Level's own advertised A/S into a
    width it could not honour.  Only the narrowest Level (A/S = 0.9) is narrower
    than the shoulders.  Derived from the ratio, not tabulated.
    """
    _install_scripted_agent("frontal")
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-frontal",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            for level in sorted(LEVEL_CHANNEL_WIDTHS):
                expected = a_s_ratio(LEVEL_CHANNEL_WIDTHS[level]) >= 1.0 - 1e-12
                episodes = runner.run_level(level=level, episodes=1)
                summary = BAOExperimentRunner.summarize_level(level, episodes)
                passed = summary["passed_count"] == 1
                check(
                    passed == expected,
                    f"level {level}: frontal policy passed={passed}, "
                    f"expected {expected}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] the frontal policy passes wherever A/S >= 1.00, not below it")


def test_collisions_do_not_end_episode() -> None:
    """A blocked agent must keep stepping until the step limit is exhausted."""
    _install_scripted_agent("frontal")
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-frontal",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            narrow = max(LEVEL_CHANNEL_WIDTHS)
            episodes = runner.run_level(level=narrow, episodes=1)
            episode = episodes[0]
            check(
                not episode["passed"],
                f"frontal policy cannot pass the narrowest Level "
                f"({LEVEL_CHANNEL_WIDTHS[narrow]:.3f} m)",
            )
            check(
                episode["end_reason"] == "max_steps",
                f"blocked episode ended as {episode['end_reason']}",
            )
            check(
                episode["total_steps"] == DEFAULT_MAX_STEPS,
                f"blocked episode ran {episode['total_steps']} steps, "
                f"expected {DEFAULT_MAX_STEPS}",
            )
            check(
                episode["wall_collision_count"] > 0,
                "blocked episode should have recorded collisions",
            )
            check(
                any(step["collision"] for step in episode["steps"]),
                "at least one step should be flagged as a collision",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] collisions are recorded without terminating the episode")


def test_sideways_policy_is_scored_sideways() -> None:
    """The rotate-then-walk-forward policy must pass every Level and score sideways."""
    _install_scripted_agent("sideways")
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-sideways",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            for level in sorted(LEVEL_CHANNEL_WIDTHS):
                episodes = runner.run_level(level=level, episodes=1)
                episode = episodes[0]
                check(
                    episode["passed"],
                    f"level {level}: sideways policy failed after "
                    f"{episode['total_steps']} steps",
                )
                check(
                    episode["passed_sideways"],
                    f"level {level}: sideways policy not scored as sideways",
                )
                check(
                    episode["first_turn_step"] == 0,
                    f"level {level}: first turn recorded at step "
                    f"{episode['first_turn_step']}",
                )
                check(
                    episode["total_rotation"] > 0,
                    f"level {level}: total_rotation was not accumulated",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] the sideways policy passes every Level and is scored sideways")


def test_csv_export_columns() -> None:
    """The flat CSV must carry the documented columns and one row per step."""
    import main as main_module

    _install_scripted_agent("sideways")
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-sideways",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            episodes = runner.run_level(level=2, episodes=1)
            csv_path = main_module.save_episodes_csv(
                episodes,
                model="scripted-sideways",
                level=2,
                results_root=os.path.join(tmp, "results"),
                timestamp="test",
                tag="runA",
            )
            # The tag must be in the filename.  Episode JSON is isolated per run
            # under results/level{n}/{model}/{tag}/, but the CSV lives beside the
            # model directory; without the tag every re-run of the same Level
            # overwrites the previous file and there is no way to tell which run
            # a CSV came from.
            check(
                "runA" in os.path.basename(csv_path),
                f"CSV filename {os.path.basename(csv_path)!r} does not carry the "
                f"run tag, so reruns would be indistinguishable",
            )
            untagged = main_module.save_episodes_csv(
                episodes,
                model="scripted-sideways",
                level=2,
                results_root=os.path.join(tmp, "results"),
                timestamp="test2",
            )
            check(
                untagged != csv_path,
                "an untagged export reused the tagged path",
            )
            check(
                "untagged" in os.path.basename(untagged),
                f"an untagged export produced {os.path.basename(untagged)!r}, "
                f"which does not say it is untagged",
            )
            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            check(rows, "CSV has no data rows")
            check(
                len(rows) == episodes[0]["total_steps"],
                f"CSV rows {len(rows)} != total_steps {episodes[0]['total_steps']}",
            )
            required = {
                "episode_id", "level", "channel_width", "a_s_ratio", "passed",
                "passed_sideways", "marker_slot", "robot_ground_offset",
                "total_rotation", "first_turn_step",
                "total_steps", "action_sequence", "step", "action",
                "torso_rotation", "position_x", "position_z", "collision",
                "step_success",
            }
            check(
                required <= set(rows[0]),
                f"CSV missing columns: {sorted(required - set(rows[0]))}",
            )
            # And the two auditable fields must carry their values, not an empty
            # column produced by a name that matches nothing.
            check(
                rows[0]["marker_slot"] not in ("", "None")
                and rows[0]["robot_ground_offset"] not in ("", "None"),
                "marker_slot / robot_ground_offset exported as empty columns: "
                f"{rows[0]['marker_slot']!r}, {rows[0]['robot_ground_offset']!r}",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] CSV export carries the required columns")


def test_analysis_recovers_threshold() -> None:
    """analysis.py must find the widest A/S at which sideways passage occurs."""
    tmp = make_temp_dir()
    try:
        # Synthetic results: sideways once the aperture is at or below A/S 1.2, which is
        # inside the human band, so the verdict must be "borderline" -- it rotates while
        # the channel is still passable.  Chosen by A/S and not by ladder index on
        # purpose: taking the two narrowest Levels made this test change verdict when the
        # ladder grew from 12 to 17, which is a fact about the index, not about the
        # behaviour under test.
        sideways_levels = [
            level for level, width in LEVEL_CHANNEL_WIDTHS.items()
            if a_s_ratio(width) <= 1.2 + 1e-9
        ]
        expected_threshold = max(
            a_s_ratio(LEVEL_CHANNEL_WIDTHS[level]) for level in sideways_levels
        )
        results_root = os.path.join(tmp, "results")
        for level, width in LEVEL_CHANNEL_WIDTHS.items():
            out_dir = os.path.join(results_root, f"level{level}", "synthetic")
            os.makedirs(out_dir, exist_ok=True)
            sideways = level in sideways_levels
            # Three episodes per Level: the threshold rule now needs at least two
            # events at a Level to call it above the wider-Level background, so a
            # one-episode-per-Level fixture could never produce a threshold.
            for episode_id in range(3):
                episode = {
                    "episode_id": episode_id,
                    "level": level,
                    "model_name": "synthetic",
                    "channel_width": width,
                    "a_s_ratio": a_s_ratio(width),
                    "passed": True,
                    "passed_sideways": sideways,
                    "total_rotation": 90.0 if sideways else 0.0,
                    "first_turn_step": 0 if sideways else None,
                    "total_steps": 21 if sideways else 20,
                    "action_sequence": "turn_left" if sideways else "forward",
                    "end_reason": "success",
                    "wall_collision_count": 0,
                    "invalid_response_count": 0,
                    "steps": [
                        {
                            "step": 0,
                            "action": "turn_left" if sideways else "forward",
                            "torso_rotation": 90.0 if sideways else 0.0,
                            "position_x": 1.55,
                            "position_z": 0.0,
                            "collision": False,
                            "step_success": True,
                        }
                    ],
                }
                with open(
                    os.path.join(out_dir, f"episode_{episode_id:03d}.json"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    json.dump(episode, fh)

        report = analysis.analyze_model(results_root, "synthetic", list(LEVEL_CHANNEL_WIDTHS))
        threshold = report["sideways_threshold"]
        check(
            threshold is not None and abs(threshold - expected_threshold) < 1e-9,
            f"recovered threshold {threshold}, expected {expected_threshold} "
            f"(the widest A/S with a sideways passage)",
        )
        check(
            report["threshold_class"] == "borderline",
            f"threshold class {report['threshold_class']}, expected borderline",
        )
        check(
            abs(report["threshold_gap_vs_human"] - (expected_threshold - 1.30)) < 1e-9,
            "threshold gap vs human is wrong",
        )

        # A model that never goes sideways must report no threshold.
        report_empty = analysis.analyze_model(
            os.path.join(tmp, "missing"), "nobody", [0]
        )
        check(
            report_empty["sideways_threshold"] is None
            and report_empty["threshold_class"] == "no_threshold",
            "an absent model should report no_threshold",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] analysis recovers the A/S threshold and classifies it")


def test_analysis_excludes_episodes_with_no_valid_call() -> None:
    """An episode whose every call failed must not be scored as a model failure.

    It is recorded like a failure (``passed=False``, ``end_reason=max_steps``), so
    a pass-rate denominator that keeps it turns a gateway outage into a capability
    number.  215 of the 3465 committed episodes are like this and every one is an
    HTTP error in the agent log.  They must be counted and reported separately.
    """
    tmp = make_temp_dir()
    try:
        root = os.path.join(tmp, "results")
        out_dir = os.path.join(root, "level0", "outage-model", "run-a")
        os.makedirs(out_dir, exist_ok=True)

        def write(episode_id: int, passed: bool, invalid: int) -> None:
            with open(
                os.path.join(out_dir, f"episode_{episode_id:03d}.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "episode_id": episode_id,
                        "level": 0,
                        "channel_width": 0.90,
                        "a_s_ratio": 1.58,
                        "passed": passed,
                        "passed_sideways": False,
                        "total_steps": 5,
                        "invalid_response_count": invalid,
                        "end_reason": "success" if passed else "max_steps",
                        "steps": [{"step": 0, "action": "forward"}],
                    },
                    handle,
                )

        write(0, True, 0)      # a real measurement
        write(1, False, 5)     # every one of its calls failed
        write(2, False, 5)     # and another

        report = analysis.analyze_model(root, "outage-model", [0], tag="run-a")
        summary = report["levels"][0]
        check(
            summary["episodes"] == 1,
            f"the failed-call episodes were scored: episodes={summary['episodes']}",
        )
        check(
            summary["excluded_no_valid_call"] == 2,
            "the excluded count is wrong: "
            f"{summary.get('excluded_no_valid_call')}",
        )
        check(
            abs(summary["pass_rate"] - 1.0) < 1e-12,
            f"pass_rate {summary['pass_rate']} counted an infrastructure failure",
        )
        check(
            report["overall"]["excluded_no_valid_call"] == 2
            and report["overall"]["episodes"] == 1,
            f"overall did not exclude them: {report['overall']}",
        )

        # A tag whose EVERY episode failed leaves the Level with no metric fields
        # at all (episodes=0 plus the excluded count).  The table builders must skip
        # it rather than index it: before the guard, analysis.py crashed with
        # KeyError: 'a_s_ratio' on the real corpus (kimi, all 85 episodes failed).
        all_failed_dir = os.path.join(root, "level0", "outage-model", "run-b")
        os.makedirs(all_failed_dir, exist_ok=True)
        out_dir = all_failed_dir
        write(0, False, 5)
        write(1, False, 5)
        outage = analysis.analyze_model(root, "outage-model", [0], tag="run-b")
        check(
            outage["levels"][0]["episodes"] == 0,
            f"an all-failed tag reported episodes="
            f"{outage['levels'][0].get('episodes')}",
        )
        markdown = analysis.format_markdown_table([outage], [0])
        rows = analysis.table_rows([outage], [0])
        check(
            isinstance(markdown, str) and isinstance(rows, list),
            "the table builders did not survive an all-failed Level",
        )
        check(
            not any(row.get("level") == 0 for row in rows),
            "an all-failed Level produced a per-Level table row",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] episodes with no successful call are excluded from the rates")


def test_threshold_ignores_a_single_wide_level_event() -> None:
    """A threshold must not be set by one chance pass at a wide opening.

    The old rule was "the widest Level with ANY sideways pass", and at the measured
    ~5.7% per-episode wide-Level sideways rate one such pass appears somewhere in
    the wide Levels for ~96% of models -- it scored gemini-2.5-pro at A/S 2.0.  The
    rule now needs at least two events and a one-sided binomial p < 0.05 against the
    pooled rate at STRICTLY WIDER openings (the candidate excluded from its own null).
    """

    def summary(a_s: float, n: int, sideways: int, turned: int) -> Dict[str, Any]:
        return {
            "a_s_ratio": a_s,
            "episodes": n,
            "passed_sideways_count": sideways,
            "turned_count": turned,
            "turned_rate": (turned / n) if n else 0.0,
        }

    levels = {
        0: summary(2.0, 5, 1, 1),   # one chance sideways pass / turn, widest opening
        1: summary(1.9, 5, 0, 0),
        2: summary(1.3, 5, 0, 0),
        5: summary(0.9, 5, 3, 5),   # a real effect at a narrow width
    }
    check(
        analysis.sideways_threshold(levels) == 0.9,
        "a single wide-Level sideways pass still moved the threshold: "
        f"{analysis.sideways_threshold(levels)}",
    )
    check(
        analysis.turned_threshold(levels) == 0.9,
        "a single wide-Level turn still moved the threshold: "
        f"{analysis.turned_threshold(levels)}",
    )

    # Four sideways passes at A/S 1.3 against the wider background IS a threshold,
    # and the wide single-pass Level is still excluded.
    levels[2]["passed_sideways_count"] = 4
    check(
        analysis.sideways_threshold(levels) == 1.3,
        f"a real effect at A/S 1.3 was not detected: "
        f"{analysis.sideways_threshold(levels)}",
    )
    print("[ok] the threshold rule ignores single wide-Level chance events")


def test_checkpoint_resume() -> None:
    """--resume must skip completed episodes instead of rerunning them."""
    _install_scripted_agent("sideways")
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-sideways",
                max_steps=DEFAULT_MAX_STEPS,
                tag="resume-test",
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            checkpoint_path = os.path.join(tmp, "checkpoint.json")
            first = runner.run_level(
                level=0, episodes=3, checkpoint=ProtocolCheckpoint(checkpoint_path)
            )
            check(len(first) == 3, "first pass should run all 3 episodes")

            env2 = FakeBAOEnv()
            runner2 = BAOExperimentRunner(
                env=env2,
                model="scripted-sideways",
                max_steps=DEFAULT_MAX_STEPS,
                tag="resume-test",
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            resumed = runner2.run_level(
                level=0,
                episodes=3,
                checkpoint=ProtocolCheckpoint(checkpoint_path, resume=True),
            )
            check(
                len(resumed) == 3,
                f"resume should reload 3 episodes, got {len(resumed)}",
            )
            check(
                env2.steps == 0,
                f"resume re-ran {env2.steps} environment steps instead of skipping",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] checkpoint resume skips already-completed episodes")


def test_run_tags_isolate_episode_files() -> None:
    """Different run tags must never overwrite or resume each other's episodes."""
    _install_scripted_agent("frontal")
    try:
        tmp = make_temp_dir()
        try:
            root = os.path.join(tmp, "results")
            runner_a = BAOExperimentRunner(
                env=FakeBAOEnv(), model="tagged", tag="run-a",
                results_root=root, logs_root=os.path.join(tmp, "logs"),
            )
            runner_b = BAOExperimentRunner(
                env=FakeBAOEnv(), model="tagged", tag="run-b",
                results_root=root, logs_root=os.path.join(tmp, "logs"),
            )
            runner_a.run_level(0, episodes=1)
            runner_b.run_level(0, episodes=1)
            path_a = runner_a._episode_path(0, 0)
            path_b = runner_b._episode_path(0, 0)
            check(path_a != path_b, "run tags share one episode path")
            check(os.path.exists(path_a) and os.path.exists(path_b), "tagged episode missing")
            check(
                analysis.load_episodes(root, 0, "tagged", tag="run-a")[0]["run_tag"]
                == "run-a",
                "analysis loaded the wrong run tag",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] run tags isolate episode persistence and analysis")


def test_passage_rotation_is_scored_at_the_wall_plane() -> None:
    """Turning back after getting through must not erase the rotation.

    A person who sidesteps through a narrow opening straightens up again once
    past it.  Scoring "did it go through sideways" from the torso angle at the
    END of the episode would record that person as never having turned, which
    would depress the threshold exactly for the most human-like policy.  The
    rotation is therefore scored where the body reaches the wall plane, and the
    angle itself is kept, giving the rotation-against-A/S curve rather than only a
    boolean.
    """
    _install_scripted_agent("sideways-then-straight")
    try:
        tmp = make_temp_dir()
        try:
            root = os.path.join(tmp, "results")
            runner = BAOExperimentRunner(
                env=FakeBAOEnv(),
                model="scripted-rotate-back",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=root,
                logs_root=os.path.join(tmp, "logs"),
            )
            runner.run_level(level=0, episodes=1)
            episodes = analysis.load_episodes(
                root, 0, "scripted-rotate-back", tag=runner.tag
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()

    check(len(episodes) == 1, f"expected one episode on disk, got {len(episodes)}")
    episode = episodes[0]

    check(bool(episode.get("passed")), "the rotate-back route did not pass")
    check(
        bool(episode.get("passed_sideways")),
        "a passage made with the torso at 90 degrees was not scored as sideways",
    )
    check(
        abs(float(episode.get("passage_rotation_deg", 0.0)) - 90.0) < 1e-6,
        f"passage rotation was {episode.get('passage_rotation_deg')!r}, expected 90",
    )
    check(
        abs(float(episode.get("final_torso_rotation", 0.0))) < 1e-6,
        "the scripted policy did not straighten up, so this test proves nothing "
        "about scoring the final orientation",
    )
    check(
        not analysis._sideways_yaw(float(episode.get("final_torso_rotation", 0.0))),
        "the final orientation would have counted as sideways after all",
    )
    print("[ok] the passage orientation is scored at the wall plane, angle included")


def test_analysis_discovers_tagged_runs() -> None:
    """``python analysis.py`` must find the runs a tagged sweep leaves behind.

    ``discover_models`` only looked for the untagged and ``round*`` layouts, so
    a tree written by ``main.py`` (``level{n}/{model}/{tag}/episode_*.json``,
    which is what ``run_all_models.sh`` produces) was reported as empty even
    though every episode was on disk.
    """
    _install_scripted_agent("sideways")
    try:
        tmp = make_temp_dir()
        try:
            root = os.path.join(tmp, "results")
            runner = BAOExperimentRunner(
                env=FakeBAOEnv(),
                model="tagged-model",
                tag="sweep-1",
                results_root=root,
                logs_root=os.path.join(tmp, "logs"),
            )
            runner.run_all(levels=[0, 1], episodes_per_level=1)

            found = analysis.discover_models(root)
            check(
                found == ["tagged-model"],
                f"tagged run was not discovered (got {found})",
            )

            # A tag directory holding only per-step sidecars has no episode
            # record, so it must not make the model look analysed.
            sidecar_only = os.path.join(root, "level0", "sidecar-only", "sweep-1")
            os.makedirs(sidecar_only, exist_ok=True)
            with open(
                os.path.join(sidecar_only, "episode_000_steps.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("[]")
            check(
                "sidecar-only" not in analysis.discover_models(root),
                "a directory holding only _steps.json sidecars counted as results",
            )

            # And the CLI itself must produce a report from that tree.
            out_dir = os.path.join(tmp, "analysis")
            rc = analysis.main(
                [
                    "--results_root", root,
                    "--levels", "0", "1",
                    "--out_dir", out_dir,
                    "--no_plot",
                ]
            )
            check(rc == 0, f"analysis CLI returned {rc} for a tagged run")
            # The file name carries the tag now, because a sweep writes one report per SCENE and
            # two scenes of one model would otherwise overwrite each other.
            report_path = os.path.join(out_dir, "threshold_tagged-model_sweep-1.json")
            check(
                os.path.exists(report_path),
                "analysis CLI wrote no threshold report for a tagged run",
            )
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            check(
                bool(report.get("levels")),
                "the CLI report analysed no Levels",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] analysis discovers runs written under a tag directory")


def test_analysis_prefers_the_current_protocol_tag() -> None:
    """Analysis must not summarise the previous protocol by accident.

    ``analyze_model`` with no ``--tag`` used to pick the single most recently
    modified tag directory.  Now that the tag carries the protocol version, an old
    directory can still be the newest one -- a re-run that stopped early, or a
    stray touch -- and the report would then describe an experiment with different
    action semantics under the current protocol's name.  Every current-protocol tag
    must be pooled and an old-protocol one excluded altogether, whatever the mtimes.
    """
    from protocol import PROTOCOL_TAG

    tmp = make_temp_dir()
    try:
        root = os.path.join(tmp, "results")
        old_dir = os.path.join(root, "level0", "m1", "old-egocentric")
        new_dir = os.path.join(root, "level0", "m1", f"m1-{PROTOCOL_TAG}")
        for directory, passed in ((old_dir, False), (new_dir, True)):
            os.makedirs(directory, exist_ok=True)
            with open(
                os.path.join(directory, "episode_000.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "level": 0,
                        "episode_id": 0,
                        "channel_width": 0.90,
                        "a_s_ratio": 1.58,
                        "passed": passed,
                        "total_steps": 5,
                    },
                    handle,
                )
        # Make the OLD directory the most recently modified one.
        os.utime(old_dir, None)

        report = analysis.analyze_model(root, "m1", [0])
        summary = report["levels"].get(0)
        check(summary is not None, "analysis found no Level 0 episodes")
        check(
            summary["passed_count"] == 1,
            "analysis used the old-protocol tag directory: the current-protocol "
            "episode was not the one summarised",
        )
        check(
            report.get("turned_threshold") is None
            or isinstance(report["turned_threshold"], float),
            "turned_threshold is missing or not a ratio",
        )

        # And a tag that carries a request-parameter suffix AFTER the protocol tag
        # is still this protocol's data.  Matched by containment, not by suffix:
        # an endswith test classified every "...-effortnone" directory as a
        # different protocol, which is what the models whose thinking is switched
        # off are tagged with.
        for params_tag, passed in ((f"m1-{PROTOCOL_TAG}-effortnone", False),):
            suffix_dir = os.path.join(root, "level0", "m1", params_tag)
            os.makedirs(suffix_dir, exist_ok=True)
            with open(
                os.path.join(suffix_dir, "episode_000.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "level": 0,
                        "episode_id": 0,
                        "channel_width": 1.14,
                        "a_s_ratio": 2.0,
                        "passed": passed,
                        "total_steps": 5,
                    },
                    handle,
                )
        os.utime(old_dir, None)
        report = analysis.analyze_model(root, "m1", [0])
        check(
            report["tags"] == [f"m1-{PROTOCOL_TAG}"],
            "the default must report exactly ONE current-protocol tag, not pool them: the suffixed "
            "variant is a repeat run with a request parameter, and the five markers of a Level are "
            f"five CONDITIONS, so pooling doubles n. Got: {report['tags']}",
        )
        check(
            report["levels"][0]["episodes"] == 1,
            "the episode count must come from the single chosen tag, not from both: "
            f"{report['levels'][0]['episodes']} episode(s), expected 1",
        )
        check(
            report["levels"][0]["passed_count"] == 1,
            "the pooled summary lost an episode: "
            f"{report['levels'][0]['passed_count']}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] analysis prefers the current protocol's tag directory")


def test_turned_threshold_separates_decision_from_success() -> None:
    """The threshold must exist for a model that rotates and still fails.

    The human reference (1.30) counts rotated shoulders, not successful
    passages, so a model that rotates at a wide Level and then trips must still
    register a threshold there.  ``sideways_threshold`` cannot do that by
    construction: it requires the episode to have passed.
    """
    summaries = {
        0: {"a_s_ratio": 1.58, "episodes": 5, "turned_count": 0,
            "turned_rate": 0.0, "passed_sideways_count": 0},
        2: {"a_s_ratio": 1.30, "episodes": 5, "turned_count": 3,
            "turned_rate": 0.6, "passed_sideways_count": 0},
        5: {"a_s_ratio": 0.79, "episodes": 5, "turned_count": 5,
            "turned_rate": 1.0, "passed_sideways_count": 3},
    }
    check(
        analysis.sideways_threshold(summaries) == 0.79,
        "sideways_threshold changed: it is the widest A/S that PASSED sideways",
    )
    check(
        analysis.turned_threshold(summaries) == 1.30,
        "turned_threshold is not the widest A/S at which the torso was rotated",
    )
    check(
        analysis.turned_threshold({0: summaries[0]}) is None,
        "turned_threshold reported a value for a model that never rotated",
    )
    print("[ok] turned threshold records the decision, not the outcome")


def test_prompt_uses_configured_move_step() -> None:
    from protocol import ACTION_DESCRIPTIONS, action_options_string, build_prompt

    prompt = build_prompt(state={}, move_step=0.6)
    check("60cm" in prompt, "prompt ignored configured move step")
    check("75cm" not in prompt, "prompt leaked the default move step")
    # The configured step must not cost the prompt its semantics.  The move_step
    # path used to render a bare "move forward 60cm", so the models were never
    # told which frame the movement was in -- the runner always passes move_step.
    check(
        "walking direction" in prompt,
        "prompt lost the walking-frame semantics on the configured-step path",
    )
    check(
        ACTION_DESCRIPTIONS["forward"].replace("75cm", "60cm") in prompt
        or "walk 60cm straight ahead" in prompt,
        "prompt did not carry the forward description on the configured-step path",
    )
    check(
        action_options_string(0.6) in prompt,
        "the prompt's action menu is not the rendered one",
    )
    print("[ok] prompt uses the environment's configured move step, semantics intact")


def test_invalid_action_is_recorded() -> None:
    """An unparseable model response must be recorded, not crash the episode."""
    import experiments

    class _BadAdapter(experiments.AgentAdapter):  # type: ignore[misc]
        def __init__(self, model: str, log_file: Optional[str] = None) -> None:
            self.model = model
            self.log_file = None
            self.history = []

        def query(
            self, prompt: str, image: Any, state: Any
        ) -> Tuple[None, str, str, float]:
            return None, "not json at all", "", 1.0

    original = experiments.AgentAdapter
    experiments.AgentAdapter = _BadAdapter
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="broken",
                max_steps=5,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            episodes = runner.run_level(level=1, episodes=1)
            episode = episodes[0]
            check(not episode["passed"], "a broken agent should not pass")
            check(
                episode["invalid_response_count"] == 5,
                f"expected 5 invalid responses, got {episode['invalid_response_count']}",
            )
            check(
                all(step["action"] == "invalid" for step in episode["steps"]),
                "invalid responses should be recorded as 'invalid' actions",
            )
            check(
                episode["total_steps"] == 5,
                "invalid responses must still consume steps",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        experiments.AgentAdapter = original
    print("[ok] invalid model responses are recorded without crashing")


def test_episode_memory_reaches_the_model() -> None:
    """The agent must see its own past steps, in order, for the whole episode.

    Two channels can carry that memory to a model client -- the rendered prompt
    block and the ``history=`` argument -- and the runner must hand both the
    same, growing record: the action it took, the environment's feedback, and
    its own reasoning.  A six-step sliding window used to break this from step
    seven onward.
    """
    import experiments

    seen: List[Dict[str, Any]] = []

    class _ProbeAdapter(experiments.AgentAdapter):  # type: ignore[misc]
        def __init__(self, model: str, log_file: Optional[str] = None) -> None:
            self.model = model
            self.log_file = None
            self.history = []
            self._callable = None
            self._agent = ScriptedAgent("frontal")

        def query(self, prompt: str, image: Any, state: Any) -> Any:
            seen.append({"prompt": prompt, "history": list(self.history)})
            return super().query(prompt, image, state)

    original = experiments.AgentAdapter
    experiments.AgentAdapter = _ProbeAdapter
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-memory",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            episodes = runner.run_level(level=0, episodes=1)
            episode = episodes[0]
            actions = [step["action"] for step in episode["steps"]]

            check(len(seen) == len(actions), "the probe missed a model call")
            check(
                len(actions) > 6,
                f"the episode is only {len(actions)} steps: it cannot expose a "
                "six-step window",
            )

            # Step 0 is a fresh episode: no memory, no cross-episode leakage.
            check(
                seen[0]["history"] == [],
                f"the first call already saw history {seen[0]['history']}",
            )
            check(
                "Action history" not in seen[0]["prompt"],
                "the first prompt advertises an empty history block",
            )

            for index, observation in enumerate(seen):
                check(
                    len(observation["history"]) == index,
                    f"call {index} saw {len(observation['history'])} past steps",
                )
                check(
                    [entry["action"] for entry in observation["history"]]
                    == actions[:index],
                    f"call {index} saw the wrong action sequence",
                )
                check(
                    [entry["step"] for entry in observation["history"]]
                    == list(range(index)),
                    f"call {index} saw wrongly numbered steps",
                )
                # The model's own reasoning must travel with the action.
                check(
                    all(
                        entry["reasoning"] == "frontal"
                        for entry in observation["history"]
                    ),
                    f"call {index} lost the agent's own reasoning",
                )
                # ... and the rendered prompt must agree with the argument.
                for position, action in enumerate(actions[:index]):
                    check(
                        f"- step {position}: {action} ->" in observation["prompt"],
                        f"call {index} prompt omits step {position} ({action})",
                    )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        experiments.AgentAdapter = original
    print("[ok] the whole within-episode memory reaches the model")


def test_full_protocol_and_analysis() -> None:
    """Run all six Levels end to end, then analyse the saved results.

    This exercises ``run_all`` -> saved JSON -> ``analysis.analyze_model`` ->
    the Markdown/CSV writers, i.e. the whole path a real experiment takes.
    """
    import main as main_module

    _install_scripted_agent("sideways")
    try:
        tmp = make_temp_dir()
        try:
            results_root = os.path.join(tmp, "results")
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-sideways",
                max_steps=DEFAULT_MAX_STEPS,
                tag="full",
                results_root=results_root,
                logs_root=os.path.join(tmp, "logs"),
            )
            results = runner.run_all(
                levels=list(LEVEL_CHANNEL_WIDTHS), episodes_per_level=2
            )
            check(
                sorted(results) == sorted(LEVEL_CHANNEL_WIDTHS),
                f"run_all returned levels {sorted(results)}",
            )
            check(
                all(len(eps) == 2 for eps in results.values()),
                "each Level should have produced 2 episodes",
            )

            summaries = {
                level: BAOExperimentRunner.summarize_level(level, eps)
                for level, eps in results.items()
            }
            # The sideways policy is scored sideways at EVERY Level, including the
            # widest openings where no rotation is needed.  That is a flat policy,
            # not a threshold: rotation_gradedness classifies it flat_always and the
            # threshold is None.  The old "widest Level with any sideways pass" rule
            # reported the widest A/S here, which is exactly the false positive the
            # significance rule removes.
            threshold = BAOExperimentRunner.sideways_threshold(summaries)
            check(
                threshold is None,
                f"a flat-always policy reported a threshold of {threshold}",
            )

            # analysis.py must agree when it reads the files back off disk.
            report = analysis.analyze_model(
                results_root, "scripted-sideways", list(LEVEL_CHANNEL_WIDTHS)
            )
            check(
                report["sideways_threshold"] is None,
                f"analysis threshold {report['sideways_threshold']} disagrees with "
                "the runner",
            )
            check(
                report["rotation_gradedness"] == "flat_always",
                "a policy that rotates at every width should be flat_always, got "
                f"{report['rotation_gradedness']}",
            )
            check(
                report["threshold_class"] == "no_threshold",
                f"a flat-always policy should classify as no_threshold, got "
                f"{report['threshold_class']}",
            )
            check(
                report["overall"]["passed_count"] == 2 * len(LEVEL_CHANNEL_WIDTHS),
                "every episode should have been recorded as passed",
            )
            markdown = analysis.format_markdown_table(
                [report], list(LEVEL_CHANNEL_WIDTHS)
            )
            check(
                "Sideways Threshold" in markdown,
                "the Markdown report is missing the threshold summary",
            )
            rows = analysis.table_rows([report], list(LEVEL_CHANNEL_WIDTHS))
            check(len(rows) == len(LEVEL_CHANNEL_WIDTHS) + 1, "unexpected table rows")

            # The runner's own CSV export must work for every Level.
            for level, episodes in results.items():
                csv_path = main_module.save_episodes_csv(
                    episodes,
                    model="scripted-sideways",
                    level=level,
                    results_root=results_root,
                    timestamp="full",
                )
                check(os.path.exists(csv_path), f"missing CSV for level {level}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] the full protocol runs and analysis reads the results back")


def test_progress_callback_signature() -> None:
    """main.py's progress callback must accept every argument the runner passes.

    Regression guard: the runner calls
    ``progress_callback(level, completed, total, episode_result, all_episodes)``
    but an earlier version of main.py accepted only four of those, so a real run
    died with `TypeError: _progress_callback() takes 4 positional arguments but
    5 were given` on the very first episode -- after the expensive
    SimulationApp startup, and after the first episode had already been saved.
    """
    import inspect

    import main as main_module

    params = [
        p
        for p in inspect.signature(main_module._progress_callback).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    check(
        len(params) == 5,
        f"_progress_callback accepts {len(params)} positional args, "
        f"but the runner calls it with 5",
    )

    # Drive the real call site with a spy so future drift fails here instead of
    # on the lab machine.
    seen: List[Tuple[Any, ...]] = []

    def spy(*args: Any, **kwargs: Any) -> None:
        seen.append(args)

    _install_scripted_agent("sideways")
    original_callback = main_module._progress_callback
    main_module._progress_callback = spy
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-sideways",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            runner.run_all(
                levels=[0, 1],
                episodes_per_level=1,
                progress_callback=main_module._progress_callback,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        main_module._progress_callback = original_callback
        _restore_agent_adapter()

    check(seen, "the progress callback was never invoked")
    for args in seen:
        check(
            len(args) == 5,
            f"runner invoked the callback with {len(args)} args, expected 5",
        )
        check(isinstance(args[0], int), f"arg0 should be the level, got {args[0]!r}")
        check(isinstance(args[1], int), f"arg1 should be completed, got {args[1]!r}")
        check(isinstance(args[2], int), f"arg2 should be total, got {args[2]!r}")
        check(isinstance(args[3], dict), "arg3 should be the episode dict")
        check(isinstance(args[4], list), "arg4 should be the episode list")
    print("[ok] the progress callback accepts the runner's five arguments")


def test_cli_flags_reach_the_runner() -> None:
    """Every CLI flag that should affect a run must actually be forwarded.

    Regression guard: `--save_obs` was defined and parsed but never passed to
    BAOExperimentRunner, so it silently produced no PNGs at all -- which cost a
    real round trip on the lab machine when we needed to inspect what the head
    camera actually saw.
    """
    import inspect

    import main as main_module

    source = inspect.getsource(main_module.run_experiment)
    for forwarded in (
        "save_obs=args.save_obs",
        "max_steps=args.max_steps",
        "episodes_per_level=args.episodes",
        # The call now spans lines and carries the config suffix (P1-3), so the pieces are asserted
        # rather than one exact string: what matters is that the tag is derived through effective_tag
        # AND includes the run-shaping flags.
        "tag=effective_tag(",
        "config_tag_suffix(args)",
        "model=args.model",
    ):
        check(
            forwarded in source,
            f"run_experiment does not forward {forwarded!r} to the runner",
        )
    # The tag must go through effective_tag, which is what keeps a protocol
    # change from resuming onto the previous protocol's episodes.
    check(
        "tag=args.tag" not in source,
        "run_experiment passes the raw tag; the protocol version would be lost "
        "and --resume could mix two protocols",
    )

    # Every flag parse_args declares must be referenced somewhere in main.py;
    # an unreferenced flag is a flag that does nothing.  Extract the flag
    # strings from the source with AST so --some-flag maps to args.some_flag.
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(main_module.parse_args)))
    declared: set = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            flag = node.args[0].value.lstrip("-").replace("-", "_")
            if flag != "help":
                declared.add(flag)
    check(declared, "could not extract any CLI flags from parse_args")
    referenced = set()
    for obj in vars(main_module).values():
        if inspect.isfunction(obj) or inspect.isclass(obj):
            try:
                text = inspect.getsource(obj)
            except (OSError, TypeError):
                continue
            for dest in declared:
                if f"args.{dest}" in text:
                    referenced.add(dest)
    unused = sorted(declared - referenced)
    check(
        not unused,
        f"CLI flags parsed but never used in main.py: {unused}",
    )
    print("[ok] every CLI flag reaches the runner (no dead options)")


def test_head_camera_offset_reaches_the_model() -> None:
    """The agent must be told where its head camera is aimed.

    The gaze is pinned to the WALKING direction, so turn_left/turn_right do not
    change the view at all; look_left/look_right give a single 30 degree glance
    that clears on the next action, so the agent can never be looking more than 30
    degrees off its path.  The direction the agent is walking and the direction it
    is looking can still differ, and nothing in a single frame reveals it: without
    this line an agent that had just glanced would read the sideways view as
    straight ahead.

    The environment already produced the value in get_robot_state(); the prompt
    simply never rendered it.  This pins it end to end through the runner, and
    checks the exact 0.0 and -30.0 labels plus the absence of a -60.0 one -- an
    earlier version matched a string the prompt no longer contains, so it passed
    while asserting nothing.  The glance is NEGATIVE for look_left: the reported
    camera_yaw is the heading frame, in which negative is the walker's left (the
    pre-2026-10-16 code and the fake environment both had this sign backwards).
    """
    import experiments

    seen: List[str] = []

    class _ProbeAdapter(experiments.AgentAdapter):  # type: ignore[misc]
        def __init__(self, model: str, log_file: Optional[str] = None) -> None:
            self.model = model
            self.log_file = None
            self.history = []
            self._callable = None
            self._agent = ScriptedAgent("look-then-forward")

        def query(self, prompt: str, image: Any, state: Any) -> Any:
            seen.append(prompt)
            return super().query(prompt, image, state)

    original = experiments.AgentAdapter
    experiments.AgentAdapter = _ProbeAdapter
    try:
        tmp = make_temp_dir()
        try:
            env = FakeBAOEnv()
            runner = BAOExperimentRunner(
                env=env,
                model="scripted-headcamera",
                max_steps=DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
            )
            runner.run_level(level=0, episodes=1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        experiments.AgentAdapter = original

    check(bool(seen), "the probe never saw a prompt")
    check(
        "head camera" in seen[0].lower(),
        "the prompt never mentions the head camera, so the agent cannot tell "
        "which way it is looking",
    )
    # The offsets must actually track the actions: a prompt right after a look_*
    # has to report that glance, every other prompt has to report zero, and the
    # yaw glance must NOT accumulate -- the head returns to straight ahead instead
    # of leaving the agent staring 60 degrees off its path with nothing saying so.
    #
    # These labels are matched exactly on purpose.  An earlier version keyed off
    # "head camera offset from your torso (degrees): 0.0"; when the gaze was pinned
    # to the walking direction the label changed, the old string never appeared
    # again, and the check silently asserted nothing.
    zero_yaw = "head camera yaw offset from straight ahead (degrees): 0.0"
    # look_left reports -30, NOT +30: the sign is the heading frame (negative =
    # the walker's left), which is what the real environment emits.
    glance_yaw = "head camera yaw offset from straight ahead (degrees): -30.0"
    zero_pitch = "head camera extra downward pitch (degrees): 0.0"
    glance_pitch = "head camera extra downward pitch (degrees): 45.0"
    check(
        any(zero_yaw in p for p in seen),
        f"no prompt reported a zero sideways offset; expected {zero_yaw!r}",
    )
    check(
        any(glance_yaw in p for p in seen),
        "no prompt reported the -30 degree glance after a look_left, so a look_* "
        "action never became visible to the agent",
    )
    check(
        any(glance_pitch in p for p in seen),
        "no prompt reported the 45 degree downward glance after a look_down, so "
        "the agent cannot tell that it is looking at its own body",
    )
    check(
        not any(
            "head camera yaw offset from straight ahead (degrees): -60.0" in p
            for p in seen
        ),
        "two look_left actions added up to -60 degrees, so the glance persists",
    )
    check(
        zero_yaw in seen[-1] and zero_pitch in seen[-1],
        "the gaze did not return to its usual direction after the glances",
    )
    print(
        "[ok] the head camera glances are reported to the agent, and clear "
        f"({len(seen)} prompt(s) inspected)"
    )


def test_parameterised_models_survive_the_compat_client() -> None:
    """The stdlib client must swallow per-model parameters it does not name.

    Measured failure this pins down.  deepseek-v4.1-flash and glm-4.6v are the only
    two models on the roster that carry request parameters (reasoning_effort and
    thinking).  On the real run every one of their calls died locally with

        TypeError: _OpenAICompatCompletions.create() got an unexpected keyword
        argument 'reasoning_effort'

    because AgentAPI._request merges request_params_for() into the call kwargs while
    the compat client's explicit signature only named model/messages/temperature/
    response_format.  Nothing reached the network, so the runner recorded 30
    bodiless steps per episode at the start position and the run looked exactly
    like a gateway outage.  The nine parameterless models were unaffected -- which
    is why this needs its own test: the parameterised models are the ones being
    measured, and a smoke run of any other model cannot catch it.
    """
    import ai_agent

    sent: List[Dict[str, Any]] = []

    class _FakeResponse:
        def __init__(self) -> None:
            self._body = json.dumps(
                {"choices": [{"message": {"content": '{"action": "forward"}'}}]}
            ).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    def fake_urlopen(request: Any, timeout: Any = None) -> "_FakeResponse":
        sent.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse()

    original = ai_agent.urllib.request.urlopen
    ai_agent.urllib.request.urlopen = fake_urlopen  # type: ignore[assignment]
    try:
        client = ai_agent._OpenAICompatClient("k", "http://example.invalid/v1", 5.0)
        for model, params in (
            ("deepseek-v4.1-flash", {"reasoning_effort": "none"}),
            ("glm-4.6v", {"thinking": {"type": "disabled"}}),
        ):
            sent.clear()
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
                **params,
            )
            check(bool(sent), f"{model}: no request was sent")
            for key, value in params.items():
                check(
                    sent[0].get(key) == value,
                    f"{model}: {key} did not reach the payload "
                    f"(got {sent[0].get(key)!r}, want {value!r})",
                )
            check(
                completion.choices[0].message.content == '{"action": "forward"}',
                f"{model}: the reply did not survive the compat client",
            )
            print(f"[ok] {model}: {sorted(params)} accepted and sent")
    finally:
        ai_agent.urllib.request.urlopen = original  # type: ignore[assignment]


def test_a_fatal_status_reaches_the_process_exit_code() -> None:
    """`main()` must turn run_experiment's fatal status into a non-zero exit code.

    main() called run_experiment(args) and DISCARDED what it returned, and __main__ did not call
    sys.exit either, so `return 2` from the fatal credential path and from the --start_x validation
    both exited 0. run_all_models.sh then saw every model "succeed" and printed "all models finished",
    and every job in run_parallel_sweep.sh logged exit=0 -- the exact accident those fatal paths exist
    to prevent.

    This drives the real main() with parse_args stubbed and run_experiment replaced, so it tests the
    wiring rather than a copy of it.
    """
    import argparse as _argparse

    import main as _main

    original_parse = _main.parse_args
    original_run = _main.run_experiment
    try:
        _main.parse_args = lambda *a, **k: _argparse.Namespace()  # type: ignore[assignment]
        for status, expected in ((2, 2), (0, 0)):
            _main.run_experiment = lambda args, _s=status: _s  # type: ignore[assignment]
            try:
                _main.main()
            except SystemExit as exit_error:
                check(
                    exit_error.code == expected,
                    f"main() exited with {exit_error.code!r} for a run that returned {status}; "
                    "a fatal status has to reach the process exit code or a sweep reports success",
                )
            else:
                check(False, f"main() did not raise SystemExit for a run that returned {status}")
        # The normal path returns the per-Level summaries, which must still exit 0.
        _main.run_experiment = lambda args: {0: {"pass_rate": 1.0}}  # type: ignore[assignment]
        try:
            _main.main()
        except SystemExit as exit_error:
            check(
                exit_error.code == 0,
                f"a successful run exited {exit_error.code!r} instead of 0",
            )
        else:
            check(False, "main() did not raise SystemExit on the success path")
    finally:
        _main.parse_args = original_parse  # type: ignore[assignment]
        _main.run_experiment = original_run  # type: ignore[assignment]
    print("[ok] a fatal status reaches the exit code, and success still exits 0")


def test_the_threshold_family_wise_error_rate_is_controlled() -> None:
    """Under a pure null the threshold search must almost never report a threshold.

    Holm's guarantee depends on the family being the set that was SEARCHED. This used to skip every
    Level with fewer than THRESHOLD_MIN_COUNT events BEFORE forming the family, so the family was the
    two or three Levels that happened to have events -- not the seventeen searched -- and the first
    cut-off was alpha/1 instead of alpha/17. Measured by Monte Carlo at a nominal 0.05: FWER 0.220
    (q=0.05) and 0.264 (q=0.057). It is 0.029 and 0.039 after the fix.

    The check is deliberately loose (<= 0.10 against a nominal 0.05) because a Monte Carlo estimate
    from a fixed seed is not the same number on every numpy version; what it catches is the order of
    magnitude, which is the whole defect.
    """
    import numpy as np

    import analysis as _analysis

    rng = np.random.default_rng(7)
    trials = 400
    false_positives = 0
    for _ in range(trials):
        levels: Dict[int, Dict[str, Any]] = {}
        for level in range(17):
            events = int(rng.binomial(5, 0.05))
            levels[level] = {
                "level": level,
                "a_s_ratio": 2.0 - 0.1 * level,
                "episodes": 5,
                "passed_sideways_count": events,
                "turned_count": events,
                "turned_rate": events / 5.0,
            }
        if _analysis.sideways_threshold(levels) is not None:
            false_positives += 1
    rate = false_positives / trials
    check(
        rate <= 0.10,
        f"under a pure null the threshold search reported a threshold {rate:.3f} of the time "
        f"({false_positives}/{trials}); against a nominal 0.05 that is the Holm family-size defect "
        "back again",
    )
    print(f"[ok] the threshold FWER under a null is {rate:.3f}, within the 0.10 bound")


def test_choose_tag_counts_the_episodes_it_ranks_by() -> None:
    """`choose_tag` must rank tags by their real episode count.

    `_episode_count` globbed `{model}/*{tag}*.json` while the records live at
    `{model}/{tag}/episode_*.json`, so it always returned 0 -- and choose_tag ranks by it. With every
    count tied at 0 the choice fell through to "shortest name, then newest directory", so a short,
    incomplete tag could beat a complete one and the default report described the wrong run.
    """
    import analysis

    tmp = make_temp_dir()
    try:
        results_root = os.path.join(tmp, "results")
        tag_small = "m-current"
        tag_big = "m-current-longer-name"
        for tag, count in ((tag_small, 2), (tag_big, 5)):
            level_dir = os.path.join(results_root, "level0", "m", tag)
            os.makedirs(level_dir, exist_ok=True)
            for episode_id in range(count):
                with open(os.path.join(level_dir, f"episode_{episode_id:03d}.json"),
                          "w", encoding="utf-8") as handle:
                    json.dump({"level": 0, "episode_id": episode_id, "passed": True}, handle)

        check(
            analysis._episode_count(results_root, "m", tag_big, [0]) == 5,
            "the episode count does not see the records: "
            f"{analysis._episode_count(results_root, 'm', tag_big, [0])}",
        )
        check(
            analysis._episode_count(results_root, "m", tag_small, [0]) == 2,
            "the episode count is wrong for the small tag: "
            f"{analysis._episode_count(results_root, 'm', tag_small, [0])}",
        )

        # The ranking must pick by COUNT, not by name length: tag_big is the longer name, so the
        # old "shortest name wins on a tie" tiebreak would have chosen the wrong one.
        import protocol as _protocol
        named_big = f"m-{_protocol.PROTOCOL_TAG}-longer-name"
        named_small = f"m-{_protocol.PROTOCOL_TAG}"
        for tag, count in ((named_small, 2), (named_big, 5)):
            level_dir = os.path.join(results_root, "level0", "m", tag)
            os.makedirs(level_dir, exist_ok=True)
            for episode_id in range(count):
                with open(os.path.join(level_dir, f"episode_{episode_id:03d}.json"),
                          "w", encoding="utf-8") as handle:
                    json.dump({"level": 0, "episode_id": episode_id, "passed": True}, handle)
        chosen = analysis.choose_tag(results_root, "m", [0])
        check(
            chosen == named_big,
            "choose_tag picked by name length instead of by episode count: "
            f"{chosen!r}, expected {named_big!r}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] choose_tag ranks tags by the episodes it actually counts")


def test_run_shaping_flags_reach_the_tag() -> None:
    """Two runs with different settings must not share a results directory.

    The tag is the --resume key. It carried the model, protocol, scene, request parameters and
    --image_size but not --max_steps, --move_step, --start_x, --eye_height/--eye_pitch or
    --env_config, so a MAX_STEPS=5 smoke test and a default 30 run shared a directory: the smoke
    episodes were skipped as finished and args.json was overwritten.
    """
    import argparse as _argparse

    import main as _main

    def namespace(**overrides: Any) -> _argparse.Namespace:
        values: Dict[str, Any] = {
            "max_steps": 30, "move_step": None, "start_x": None,
            "eye_height": None, "eye_pitch": None, "env_config": None,
        }
        values.update(overrides)
        return _argparse.Namespace(**values)

    check(
        _main.config_tag_suffix(namespace()) == "",
        "a default run now carries a config suffix, which would move every committed directory",
    )
    default_tag = _main.effective_tag("m", "", "stage1.2", "")
    smoke_tag = _main.effective_tag("m", "", "stage1.2", _main.config_tag_suffix(namespace(max_steps=5)))
    check(
        default_tag != smoke_tag,
        f"a MAX_STEPS=5 run and a default run share the tag {default_tag!r}, so --resume will skip "
        "the smoke episodes",
    )
    check(
        smoke_tag.endswith("-ms5"),
        f"the max_steps suffix is missing from {smoke_tag!r}",
    )
    check(
        _main.effective_tag("m", "", "stage1.2", _main.config_tag_suffix(namespace(move_step=0.5)))
        .endswith("-mv0.5"),
        "--move_step does not reach the tag",
    )
    check(
        _main.effective_tag("m", "", "stage1.2", _main.config_tag_suffix(namespace(start_x=1.2)))
        .endswith("-sx1.2"),
        "--start_x does not reach the tag",
    )
    print("[ok] run-shaping flags change the tag, and defaults leave it alone")


def test_analysis_does_not_import_the_simulator() -> None:
    """The offline report must not pull in Isaac Sim.

    analysis.py imported LEVEL_CHANNEL_WIDTHS from environment.py, which re-exports it -- so
    `python3 analysis.py` printed a full Isaac Sim ImportError traceback before its first table, and
    the module's own promise that offline analysis needs no simulator was false. The ladder lives in
    protocol.py; the name has to be imported from there.
    """
    import subprocess as _subprocess
    import sys as _sys

    probe = (
        "import sys; sys.path.insert(0, '.'); import analysis; "
        "print('IMPORTED' if 'environment' in sys.modules else 'CLEAN')"
    )
    result = _subprocess.run(
        [_sys.executable, "-c", probe],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True, timeout=120,
    )
    check(
        "CLEAN" in result.stdout,
        "importing analysis also imports environment, so the offline report needs the simulator: "
        f"{result.stdout.strip()[-200:]}{result.stderr.strip()[-200:]}",
    )
    check(
        "isaacsim" not in result.stderr.lower(),
        f"importing analysis printed an Isaac Sim traceback: {result.stderr.strip()[:200]}",
    )
    print("[ok] analysis imports neither environment nor the simulator")


def test_a_real_baoenv_scores_the_way_the_fake_does() -> None:
    """Exercise the REAL BAOEnv, which no test did.

    Every other test in this repository drives FakeBAOEnv. A grep over test_bao_*.py finds 21
    `BAOEnv(` occurrences and all 21 are `FakeBAOEnv(` -- zero real ones -- so check_success,
    reset_scene's marker rule and execute_action were only ever tested against the stand-in. That is
    fine for the wiring, and it is not evidence that the environment behaves as the stand-in assumes.

    This needs Isaac Sim, and constructing BAOEnv needs SimulationApp, which costs about two minutes.
    Under an interpreter without the simulator it raises Skipped rather than passing, so a run that
    cannot do it says so instead of looking green.

    OPT-IN, AND WHY. Starting SimulationApp inside the suite takes about 160 s and Kit takes over stdout,
    so its prints are swallowed and the rest of the suite cannot be observed; a version that tried it
    unconditionally produced no output at all and looked like a hang. A test that can silently kill the
    run it belongs to is worse than no test, so this one runs only when BAO_RUN_ISAAC_TESTS=1 is set:

        BAO_RUN_ISAAC_TESTS=1 /home/ybh/isaacsim/python.sh test_bao_integration.py

    and skips otherwise. Its own behaviour under Isaac has NOT been confirmed by this repository yet --
    treat the first opt-in run as the experiment, not as a formality.
    """
    if os.environ.get("BAO_RUN_ISAAC_TESTS") != "1":
        raise Skipped(
            "opt-in: set BAO_RUN_ISAAC_TESTS=1 to build a real BAOEnv (about 160 s of SimulationApp, "
            "and Kit takes over stdout)"
        )
    try:
        from isaacsim import SimulationApp
    except Exception as exc:  # noqa: BLE001 - absent simulator, not a failure
        raise Skipped(
            f"a real BAOEnv needs Isaac Sim ({type(exc).__name__}: {exc})"
        ) from exc

    import environment

    # SimulationApp FIRST, then the availability check.  `isaacsim.core` only becomes importable once
    # the app has started, so checking before it always reports False -- which is exactly what
    # environment._HAS_ISAAC_SIM used to latch on, and why isaac_available() re-checks rather than
    # remembering. Testing it in the wrong order here made the whole test skip under the Isaac
    # interpreter, i.e. a green run that exercised nothing.
    simulation_app = SimulationApp({"headless": True})
    try:
        if not environment.isaac_available():
            raise Skipped("Isaac Sim did not become importable even after SimulationApp started")
        env = environment.BAOEnv(simulation_app, task_dict={"headless": True})
        try:
            # A real environment must build a stage the agent can act in, and the three methods the
            # fake stands in for must exist and behave: the channel width the ladder says, a success
            # predicate that is False at the start pose, and an action that moves the robot.
            check(
                abs(env.get_channel_width() - environment.LEVEL_CHANNEL_WIDTHS[0]) < 1e-9,
                f"the real environment started at {env.get_channel_width()} m, not Level 0's "
                f"{environment.LEVEL_CHANNEL_WIDTHS[0]} m",
            )
            check(
                env.check_success() is False,
                "the real environment reports success before the robot has moved",
            )
            start = float(env.get_robot_state()["position"][0])
            result = env.execute_action("forward")
            check(
                result.legal,
                f"a forward step from the start pose is not legal: {result.feedback}",
            )
            moved = float(env.get_robot_state()["position"][0]) - start
            check(
                moved > 0.0,
                f"a forward step moved the robot {moved:+.3f} m along x, so the frame disagrees with "
                "the action name",
            )
        finally:
            env.close()
    finally:
        # THE APP IS DELIBERATELY NOT CLOSED.  simulation_app.close() shuts Kit down and ends the
        # process, so the suite's own summary -- and every test after this one -- never ran: the first
        # version of this test produced no output at all, which looked like a hang. The process exits
        # when it exits; leaving the app up is what lets the rest of the suite finish.
        pass
    print("[ok] a real BAOEnv builds, starts, rejects an immediate success and moves forward")


def test_analysis_finds_a_model_whose_name_contains_a_slash() -> None:
    """half the roster has a slash in its name, and analysis could not read any of them.

    experiments.Runner writes episodes under model.replace("/", "-"), but analysis.load_episodes,
    choose_tag and the threshold lookups all joined the RAW model name -- so for Qwen/Qwen3-VL-8B-Instruct
    every path became "results/level0/Qwen/Qwen3-VL-8B-Instruct/...", which does not exist. Measured on a
    real run: load_episodes raised FileNotFoundError and tools/verify_stage1_lab.py reported "0
    episode directories" for a run that had produced five.

    A directory fixture, not a source assertion: what matters is that the file is FOUND.
    """
    import tempfile

    import analysis

    check(
        analysis.model_dir_name("Qwen/Qwen3-VL-8B-Instruct") == "Qwen-Qwen3-VL-8B-Instruct",
        "model_dir_name no longer maps a slash to a dash",
    )
    check(
        analysis.model_dir_name("Qwen-Qwen3-VL-8B-Instruct") == "Qwen-Qwen3-VL-8B-Instruct",
        "model_dir_name is not idempotent, so an already-slugged name would move",
    )

    with tempfile.TemporaryDirectory() as tmp:
        tag = "Qwen-Qwen3-VL-8B-Instruct-v8-actions-flipped-stage1.1"
        directory = os.path.join(
            tmp, "level0", "Qwen-Qwen3-VL-8B-Instruct", tag
        )
        os.makedirs(directory, exist_ok=True)
        record = {
            "episode_id": 0,
            "level": 0,
            "channel_width": 1.14,
            "a_s_ratio": 2.0,
            "passed": True,
            "total_steps": 11,
            "marker_slot": 1,
            "scene": "stage1.1",
            "steps": [],
        }
        with open(os.path.join(directory, "episode_000.json"), "w") as handle:
            json.dump(record, handle)

        found = analysis.load_episodes(tmp, 0, "Qwen/Qwen3-VL-8B-Instruct", tag=tag)
        check(
            len(found) == 1,
            f"load_episodes found {len(found)} episode(s) for a model with a slash; the directory is "
            "named with the slug and the lookup must use it",
        )
        # ... and the already-slugged spelling must find the same file, not a second one.
        check(
            len(analysis.load_episodes(tmp, 0, "Qwen-Qwen3-VL-8B-Instruct", tag=tag)) == 1,
            "the slugged spelling does not resolve to the same directory",
        )
    print("[ok] analysis reads a model whose name contains a slash")


def main() -> int:
    tests = [
        test_episode_record_contract,
        test_frontal_policy_matches_design_table,
        test_collisions_do_not_end_episode,
        test_sideways_policy_is_scored_sideways,
        test_csv_export_columns,
        test_a_fatal_status_reaches_the_process_exit_code,
        test_the_threshold_family_wise_error_rate_is_controlled,
        test_analysis_recovers_threshold,
        test_choose_tag_counts_the_episodes_it_ranks_by,
        test_analysis_finds_a_model_whose_name_contains_a_slash,
        test_a_real_baoenv_scores_the_way_the_fake_does,
        test_run_shaping_flags_reach_the_tag,
        test_analysis_does_not_import_the_simulator,
        test_analysis_excludes_episodes_with_no_valid_call,
        test_threshold_ignores_a_single_wide_level_event,
        test_checkpoint_resume,
        test_run_tags_isolate_episode_files,
        test_passage_rotation_is_scored_at_the_wall_plane,
        test_analysis_discovers_tagged_runs,
        test_analysis_prefers_the_current_protocol_tag,
        test_turned_threshold_separates_decision_from_success,
        test_prompt_uses_configured_move_step,
        test_invalid_action_is_recorded,
        test_episode_memory_reaches_the_model,
        test_head_camera_offset_reaches_the_model,
        test_parameterised_models_survive_the_compat_client,
        test_full_protocol_and_analysis,
        test_progress_callback_signature,
        test_cli_flags_reach_the_runner,
    ]
    failures = 0
    skipped = []
    for test in tests:
        try:
            test()
        except Skipped as exc:
            skipped.append(test.__name__)
            print(f"[skip] {test.__name__}: {exc}")
        except Failure as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report unexpected crashes too
            failures += 1
            print(f"[ERROR] {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    passed = len(tests) - failures - len(skipped)
    if failures:
        print(
            f"{failures}/{len(tests)} integration checks FAILED "
            f"({passed} passed, {len(skipped)} skipped)"
        )
        return 1
    if skipped:
        print(
            f"{passed}/{len(tests)} integration checks passed, "
            f"{len(skipped)} SKIPPED (not run): {', '.join(skipped)}"
        )
        return 0
    print(f"all {len(tests)} integration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
