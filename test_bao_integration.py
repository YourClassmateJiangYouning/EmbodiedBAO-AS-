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
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment import (
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
    _is_sideways_yaw,
)
import analysis

ROBOT_START = ROBOT_START_POS.copy()


class Failure(AssertionError):
    pass


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
        self.steps = 0

    # -- scene ---------------------------------------------------------
    def set_channel_width(self, width: float) -> float:
        self._channel_width = float(width)
        return self._channel_width

    def get_channel_width(self) -> float:
        return float(self._channel_width)

    def get_a_s_ratio(self) -> float:
        return a_s_ratio(self._channel_width)

    def reset_scene(self) -> np.ndarray:
        self._position = ROBOT_START.copy()
        self._yaw = 0.0
        self._camera_yaw = 0.0
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
        elif action in ("look_left", "look_right"):
            self._camera_yaw += 30.0 if action == "look_left" else -30.0
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
            # x >= 11.0.  Rotating back mid-channel is what the gate correctly
            # forbids, because the shoulders would sweep the panel.
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
                if x >= WALL_X + 0.5:
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
            # Turn the head twice, then walk.  Used to check that the head-camera
            # offset the environment tracks actually becomes visible to the agent:
            # a client that never reports it is aiming its view 60 degrees off its
            # walking direction with nothing saying so.
            if self.phase < 2:
                self.phase += 1
                return self._reply("look_left")
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
    """A straight-walking agent must pass exactly Levels 0-4.

    Level 4 is A/S = 1.00 -- channel 0.570 m, shoulders 0.570 m -- so walking
    straight through is geometrically possible and the design table calls it a
    frontal passage.  It expected False while the collision body carried a 2 mm
    skin, which made the gate require 0.574 m and quietly turned the level's own
    advertised A/S into a width it could not honour.  Only Level 5, at 0.45 m, is
    narrower than the shoulders.
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
            expected = {0: True, 1: True, 2: True, 3: True, 4: True, 5: False}
            for level in sorted(LEVEL_CHANNEL_WIDTHS):
                episodes = runner.run_level(level=level, episodes=1)
                summary = BAOExperimentRunner.summarize_level(level, episodes)
                passed = summary["passed_count"] == 1
                check(
                    passed == expected[level],
                    f"level {level}: frontal policy passed={passed}, "
                    f"expected {expected[level]}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        _restore_agent_adapter()
    print("[ok] frontal policy passes Levels 0-4 and is blocked at Level 5")


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
            episodes = runner.run_level(level=5, episodes=1)
            episode = episodes[0]
            check(not episode["passed"], "frontal policy cannot pass 0.45 m")
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
                "passed_sideways", "total_rotation", "first_turn_step",
                "total_steps", "action_sequence", "step", "action",
                "torso_rotation", "position_x", "position_z", "collision",
                "step_success",
            }
            check(
                required <= set(rows[0]),
                f"CSV missing columns: {sorted(required - set(rows[0]))}",
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
        # Synthetic results: sideways only at Levels 4 and 5 (A/S 1.00, 0.79).
        results_root = os.path.join(tmp, "results")
        for level, width in LEVEL_CHANNEL_WIDTHS.items():
            out_dir = os.path.join(results_root, f"level{level}", "synthetic")
            os.makedirs(out_dir, exist_ok=True)
            sideways = level in (4, 5)
            episode = {
                "episode_id": 0,
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
            with open(os.path.join(out_dir, "episode_000.json"), "w", encoding="utf-8") as fh:
                json.dump(episode, fh)

        report = analysis.analyze_model(results_root, "synthetic", list(LEVEL_CHANNEL_WIDTHS))
        threshold = report["sideways_threshold"]
        check(
            threshold is not None and abs(threshold - 1.00) < 1e-9,
            f"recovered threshold {threshold}, expected 1.00",
        )
        check(
            report["threshold_class"] == "borderline",
            f"threshold class {report['threshold_class']}, expected borderline",
        )
        check(
            abs(report["threshold_gap_vs_human"] - (1.00 - 1.30)) < 1e-9,
            "threshold gap vs human is wrong",
        )

        # A model that never goes sideways must report no threshold.
        report_empty = analysis.analyze_model(
            os.path.join(tmp, "missing"), "nobody", [0]
        )
        check(
            report_empty["sideways_threshold"] is None
            and report_empty["threshold_class"] == "no_sideways",
            "an absent model should report no_sideways",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ok] analysis recovers the A/S threshold and classifies it")


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
            report_path = os.path.join(out_dir, "threshold_tagged-model.json")
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

    ``analyze_model`` with no ``--tag`` used to pick the most recently modified
    tag directory.  Now that the tag carries the protocol version, an old
    directory can still be the newest one -- a re-run that stopped early, or a
    stray touch -- and the report would then describe an experiment with
    different action semantics under the current protocol's name.  A directory
    tagged with the current protocol must win regardless of mtime.
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
        0: {"a_s_ratio": 1.58, "turned_rate": 0.0, "passed_sideways_count": 0},
        2: {"a_s_ratio": 1.30, "turned_rate": 0.5, "passed_sideways_count": 0},
        5: {"a_s_ratio": 0.79, "turned_rate": 1.0, "passed_sideways_count": 3},
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
            # The sideways policy is scored sideways at every Level, so the
            # threshold is the widest channel on the ladder.
            threshold = BAOExperimentRunner.sideways_threshold(summaries)
            check(
                threshold is not None
                and abs(threshold - a_s_ratio(LEVEL_CHANNEL_WIDTHS[0])) < 1e-9,
                f"runner threshold {threshold} should be the widest A/S",
            )

            # analysis.py must agree when it reads the files back off disk.
            report = analysis.analyze_model(
                results_root, "scripted-sideways", list(LEVEL_CHANNEL_WIDTHS)
            )
            check(
                report["sideways_threshold"] is not None
                and abs(report["sideways_threshold"] - a_s_ratio(LEVEL_CHANNEL_WIDTHS[0]))
                < 1e-9,
                f"analysis threshold {report['sideways_threshold']} disagrees",
            )
            check(
                report["threshold_class"] == "anticipatory",
                f"sideways at every Level should be 'anticipatory', got "
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
        "tag=effective_tag(args.model, args.tag)",
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
    """The agent must be told how far its head camera is turned.

    The gaze is pinned to the WALKING direction, so turn_left/turn_right do not
    change the view at all; look_left/look_right offset the gaze from that
    direction, the offset PERSISTS, and a torso turn does not remove it.  The
    direction the agent is walking and the direction it is looking can therefore
    differ by up to 90 degrees, and nothing in a single frame reveals it: an agent
    that had turned its head three times would otherwise be judging the opening
    from a view rotated 90 degrees off its path with no way to know.

    The environment already produced the value in get_robot_state(); the prompt
    simply never rendered it.  This pins it end to end through the runner, and
    checks the exact 0.0 and 60.0 labels -- an earlier version matched a string
    the prompt no longer contains, so it passed while asserting nothing.
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
        "head camera offset" in seen[0].lower(),
        "the prompt does not state the head camera offset, so the agent cannot "
        "tell which way it is looking",
    )
    # The offset must actually track the actions: at least one later prompt has
    # to report a non-zero value once a look_* action has run, otherwise the line
    # is present but inert.
    #
    # This keyed off "head camera offset from your torso (degrees): 0.0" until the
    # gaze was pinned to the walking direction and the label changed to
    # "from straight ahead".  The old string then never appeared, every entry
    # scored 1.0, and `any(offsets)` was true no matter what the environment did:
    # the check passed while asserting nothing.  It now matches the current label
    # and requires the exact 0.0 and 60.0 values a two-look_left run produces.
    zero_label = "head camera offset from straight ahead (degrees): 0.0"
    check(
        any(zero_label in p for p in seen),
        f"no prompt reported a zero head offset; expected the line {zero_label!r}",
    )
    check(
        any(
            "head camera offset from straight ahead (degrees): 60.0" in p
            for p in seen
        ),
        "no prompt reported the 60 degree offset after two look_left actions, so "
        "a look_* action never became visible to the agent",
    )
    print(
        "[ok] the head camera offset is reported to the agent "
        f"({len(seen)} prompt(s) inspected)"
    )


def main() -> int:
    tests = [
        test_episode_record_contract,
        test_frontal_policy_matches_design_table,
        test_collisions_do_not_end_episode,
        test_sideways_policy_is_scored_sideways,
        test_csv_export_columns,
        test_analysis_recovers_threshold,
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
        test_full_protocol_and_analysis,
        test_progress_callback_signature,
        test_cli_flags_reach_the_runner,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Failure as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report unexpected crashes too
            failures += 1
            print(f"[ERROR] {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{failures}/{len(tests)} integration checks FAILED")
        return 1
    print(f"all {len(tests)} integration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
