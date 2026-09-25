"""EmbodiedBAO A/S threshold protocol.

The benchmark asks a single question: does an MLLM-driven humanoid rotate its
body *before* a channel becomes too narrow for a frontal passage, the way
humans do at a channel-to-shoulder ratio of about 1.30 (Warren & Whang, 1987)?

The Levels differ **only** in the physical channel width, and the ladder is the
aperture series of the human study this benchmark follows: 12 widths from A/S = 2.0
down to A/S = 0.9 in steps of 0.1 (Warren & Whang 1987; Keizer et al. 2013).  The
prompt is identical across Levels, and it never discloses the channel width, the
body dimensions, or whether a turn is needed.

    Level  0   1.14 m channel  A/S 2.0   trivially passable
    ...
    Level 10   0.57 m channel  A/S 1.0   flush: an aligned body just fits
    Level 11   0.51 m channel  A/S 0.9   cannot be walked through facing forward

Movement is in the walking frame: ``forward`` advances at the far wall whatever the
torso is doing, and ``turn_left``/``turn_right`` rotate the torso relative to that
direction, which is the shoulder rotation the human studies measure.  The
per-Level angles are measured, not derived -- see tools/check_heading_frame.py.

Every Level runs ``5`` independent episodes of at most ``30`` steps, so a model is
scored on 12 x 5 = 60 episodes: the 12-width series of the reference design with
enough repetitions per width for a per-model rotation-onset curve.

Every Level runs ``10`` independent episodes of at most ``30`` steps.  An
episode ends only on success (the body clears the wall: centre reaches
``x >= 8.75`` m, one stride past the wall plane) or step exhaustion; wall
collisions are recorded but never terminate the episode.

Each step follows the canonical loop:

    image  = env.get_camera_image()
    state  = env.get_robot_state()
    prompt = protocol.build_prompt(state, history, max_steps)
    action = ai_agent.get_action(prompt, image, state, history)
    result = env.execute_action(action)
    passed = env.check_success()

The module only imports Isaac Sim inside ``main()``, so prompt building, action
parsing and record shaping can be unit-tested with plain Python.

Expected ``ai_agent`` interface (one of):
    ai_agent.get_action(prompt, image, state, history, options) -> action
    ai_agent.create_agent(model, log_file) -> object with get_action(...)
    ai_agent.get_agent(model, log_file)   -> object with get_action(...)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from persistence import (
    atomic_write_json,
    backup_corrupt_file,
    load_json_file,
    sanitize_tag,
)
from environment import (
    ACTIONS,
    LEVEL_CHANNEL_WIDTHS,
    SIDEWAYS_YAW_MAX_DEG,
    SIDEWAYS_YAW_MIN_DEG,
    SUCCESS_X,
    WALL_X,
    TURN_STEP_DEG,
    a_s_ratio,
    level_channel_width,
)
from protocol import ACTION_OPTIONS_STRING, build_prompt

# Derived from the ladder so the two can never disagree: adding a width to
# environment.LEVEL_CHANNEL_WIDTHS adds a Level to the protocol.
DEFAULT_LEVELS: Tuple[int, ...] = tuple(sorted(LEVEL_CHANNEL_WIDTHS))
DEFAULT_EPISODES_PER_LEVEL = 5
DEFAULT_MAX_STEPS = 30


class ProtocolCheckpoint:
    """Persistent completion state for pause/resume across experiment runs.

    The checkpoint is what stops a resumed sweep from re-running (and thereby
    overwriting) episodes that a previous invocation already paid for, so it is
    both written atomically and rebuilt from the episode records whenever the
    stored file cannot be read.
    """

    def __init__(self, path: str, resume: bool = False) -> None:
        self.path = path
        self.completed: set[str] = set()
        self.recovered_from: Optional[str] = None
        if resume and os.path.exists(path):
            try:
                data = load_json_file(path)
                self.completed = set(str(item) for item in data.get("completed", []))
            except Exception as exc:
                # Never silently forget collected data.  Resetting to an empty
                # set here used to make the run restart and overwrite episodes
                # that were still intact on disk; recover them instead.
                self.completed = self._recover(exc)

    def _recover(self, exc: Exception) -> set:
        """Quarantine an unreadable checkpoint and rebuild its state."""
        backup = backup_corrupt_file(self.path)
        self.recovered_from = backup
        print(
            f"[checkpoint] WARNING: cannot read {self.path} "
            f"({type(exc).__name__}: {exc})"
        )
        if backup:
            print(f"[checkpoint] damaged checkpoint kept as {backup}")
        rebuilt = self._rebuild_from_episodes()
        if rebuilt:
            print(
                f"[checkpoint] rebuilt {len(rebuilt)} completed unit(s) from the "
                "episode records already on disk"
            )
        else:
            print(
                "[checkpoint] no completed episode could be recovered from disk; "
                "every episode of this tag will be re-run"
            )
        return rebuilt

    def _rebuild_from_episodes(self) -> set:
        """Reconstruct completion state from ``results/level*/{model}/{tag}/``.

        The checkpoint lives at ``<results>/{model}/checkpoint_{tag}.json``, so
        the model, the tag and the results root can all be derived from its own
        path.  Only episode files that parse as valid JSON count as complete:
        the point is to re-run what is damaged, not to trust it.
        """
        completed: set = set()
        try:
            checkpoint = os.path.abspath(self.path)
            model_dir = os.path.dirname(checkpoint)
            results_root = os.path.dirname(model_dir)
            base = os.path.basename(checkpoint)
            if not (base.startswith("checkpoint_") and base.endswith(".json")):
                return completed
            tag = base[len("checkpoint_") : -len(".json")]
            model = os.path.basename(model_dir)
            pattern = os.path.join(results_root, "level*", model, tag, "episode_*.json")
            for path in sorted(glob.glob(pattern)):
                name = os.path.basename(path)
                match = re.match(r"episode_(\d+)\.json$", name)
                if not match:
                    continue
                try:
                    load_json_file(path)
                except Exception:
                    continue
                level_dir = os.path.basename(
                    os.path.dirname(os.path.dirname(os.path.dirname(path)))
                )
                if not level_dir.startswith("level"):
                    continue
                completed.add(f"{level_dir}/episode{int(match.group(1)):03d}")
        except Exception:
            return completed
        return completed

    def is_done(self, key: str) -> bool:
        return key in self.completed

    def mark(self, key: str) -> None:
        self.completed.add(key)
        atomic_write_json(self.path, {"completed": sorted(self.completed)})



# ---------------------------------------------------------------------------
# Small helpers over the environment interface
# ---------------------------------------------------------------------------


def _env_channel_width(env: Any, level: int) -> float:
    """Read back the channel width the environment actually built."""
    getter = getattr(env, "get_channel_width", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:
            pass
    return level_channel_width(level)


def _env_a_s_ratio(env: Any, channel_width: float) -> float:
    """Read back A/S from the environment, falling back to pure arithmetic."""
    getter = getattr(env, "get_a_s_ratio", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:
            pass
    return a_s_ratio(channel_width)


def _env_passed(env: Any) -> bool:
    """Success query that also works against a lightweight mock environment."""
    checker = getattr(env, "check_success", None)
    if callable(checker):
        return bool(checker())
    position = getattr(env, "_root_position", None)
    if callable(position):
        return bool(float(position()[0]) >= SUCCESS_X)
    return False


def _env_collision(env: Any) -> bool:
    checker = getattr(env, "check_collision_with_wall", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _abs_yaw(yaw_deg: float) -> float:
    """Fold a torso yaw into [0, 180] so +/- rotations behave symmetrically."""
    yaw = abs(float(yaw_deg)) % 360.0
    return float(360.0 - yaw if yaw > 180.0 else yaw)


def _is_sideways_yaw(yaw_deg: float) -> bool:
    return bool(SIDEWAYS_YAW_MIN_DEG <= _abs_yaw(yaw_deg) <= SIDEWAYS_YAW_MAX_DEG)


def analytic_pass_check(channel_width: float, yaw_deg: float) -> bool:
    """Can the body box pass a channel of ``channel_width`` at ``yaw_deg``?

    The projected lateral width of the torso box is
    ``shoulder * |cos yaw| + thickness * |sin yaw|``, which is the quantity the
    analytic collision gate effectively compares against the channel width.
    """
    from environment import ROBOT_SHOULDER_WIDTH, ROBOT_TORSO_THICKNESS

    rad = float(np.radians(yaw_deg))
    needed = ROBOT_SHOULDER_WIDTH * abs(
        float(np.cos(rad))
    ) + ROBOT_TORSO_THICKNESS * abs(float(np.sin(rad)))
    return bool(needed <= float(channel_width) + 1e-9)


class AgentAdapter:
    """Thin adapter over the ``ai_agent`` module."""

    def __init__(self, model: str, log_file: Optional[str] = None) -> None:
        self.model = model
        self.log_file = log_file
        self.history: List[Dict[str, str]] = []
        self._callable = None
        self._agent = None

        import ai_agent

        if callable(getattr(ai_agent, "get_action", None)):
            self._callable = ai_agent.get_action
        elif callable(getattr(ai_agent, "create_agent", None)):
            self._agent = ai_agent.create_agent(model=model, log_file=log_file)
        elif callable(getattr(ai_agent, "get_agent", None)):
            self._agent = ai_agent.get_agent(model=model, log_file=log_file)
        else:
            raise RuntimeError(
                "ai_agent must expose get_action(prompt, image, state, history, options) "
                "or a create_agent/get_agent factory."
            )

    def query(
        self, prompt: str, image: np.ndarray, state: Dict[str, Any]
    ) -> Tuple[Optional[str], str, str, float]:
        """Call the model and return (action_name, raw_response, reasoning, latency_ms)."""
        t0 = time.perf_counter()
        if self._callable is not None:
            try:
                raw = self._callable(
                    prompt=prompt,
                    image=image,
                    state=state,
                    history=self.history,
                    options=ACTION_OPTIONS_STRING,
                    model_name=self.model,
                    log_file=self.log_file,
                )
            except TypeError:
                try:
                    raw = self._callable(
                        prompt=prompt,
                        image=image,
                        state=state,
                        history=self.history,
                        options=ACTION_OPTIONS_STRING,
                    )
                except TypeError:
                    raw = self._callable(prompt, image, self.history)
        else:
            raw = self._agent.get_action(
                prompt=prompt,
                image=image,
                state=state,
                history=self.history,
                options=ACTION_OPTIONS_STRING,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action_name, raw_text, reasoning = self._normalize(raw)
        self._append_log(prompt, raw_text)
        return action_name, raw_text, reasoning, latency_ms

    def record(
        self,
        action_name: str,
        feedback: str,
        reasoning: str = "",
        step: Optional[int] = None,
    ) -> None:
        """Append one completed step to the agent's within-episode memory.

        ``self.history`` is the single source of truth for that memory: the same
        list is rendered into the next prompt by ``protocol.build_prompt`` and
        handed to the model client as ``history=``, so the two can never drift.
        """
        entry: Dict[str, Any] = {"action": action_name, "feedback": feedback}
        if reasoning:
            entry["reasoning"] = reasoning
        if step is not None:
            entry["step"] = int(step)
        self.history.append(entry)

    def _normalize(self, raw: Any) -> Tuple[Optional[str], str, str]:
        """Return ``(action_name, raw_text, reasoning)`` for a model reply."""
        if isinstance(raw, str):
            return parse_action_text(raw), raw, ""
        if raw is None:
            return None, "", ""
        if isinstance(raw, dict):
            name = raw.get("action")
            if name not in ACTIONS:
                name = None
            return (
                name,
                json.dumps(raw, ensure_ascii=False),
                str(raw.get("reasoning") or ""),
            )
        raw_text = str(getattr(raw, "text", "") or raw)
        reasoning = str(getattr(raw, "reasoning", "") or "")
        if hasattr(raw, "action_choice"):
            index = int(getattr(raw, "action_choice"))
            name = ACTIONS[index - 1] if 1 <= index <= len(ACTIONS) else None
            return name, raw_text, reasoning
        name = getattr(raw, "action", None) or getattr(raw, "name", None)
        if name is None:
            name = str(raw).strip().lower()
        return (name if name in ACTIONS else None), raw_text, reasoning

    def _append_log(self, prompt: str, raw_text: str) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write("-" * 40 + "\n")
            handle.write(prompt + "\n")
            handle.write("MODEL RESPONSE:\n" + raw_text + "\n")
            handle.write("-" * 40 + "\n")


def parse_action_text(text: Any) -> Optional[str]:
    """Extract an action name from a model response like 'Choice: [3]'.

    Defensive fallback for endpoints that ignore the JSON instruction; the
    primary path is the JSON ``action`` field.
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    match = re.search(r"Choice:?[\n\s]*\[?(\d+)\]?", text, re.IGNORECASE)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(ACTIONS):
            return ACTIONS[index - 1]
    lowered = text.lower()
    for action in sorted(ACTIONS, key=len, reverse=True):
        if action in lowered:
            return action
    return None


class BAOExperimentRunner:
    """Runs the six-Level A/S threshold protocol against a BAOEnv instance."""

    def __init__(
        self,
        env: Any,
        model: str,
        max_steps: int = DEFAULT_MAX_STEPS,
        episodes_per_level: int = DEFAULT_EPISODES_PER_LEVEL,
        tag: Optional[str] = None,
        save_obs: bool = False,
        seed: int = 0,
        results_root: str = "results",
        logs_root: str = "logs",
    ) -> None:
        self.env = env
        self.model = model.replace("/", "-").replace("\\", "-")
        self.max_steps = int(max_steps)
        self.episodes_per_level = int(episodes_per_level)
        self.save_obs = bool(save_obs)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # The tag becomes a directory component in three places, so it is
        # sanitised once, here, and every consumer sees the same value.
        self.tag = (
            sanitize_tag(tag)
            if str(tag or "").strip()
            else time.strftime("%Y%m%d-%H%M%S")
        )
        self.results_root = results_root
        self.logs_root = logs_root
        self.log_dir = os.path.join(logs_root, self.tag)
        self.obs_dir = os.path.join(self.log_dir, "obs")
        os.makedirs(self.log_dir, exist_ok=True)
        if self.save_obs:
            os.makedirs(self.obs_dir, exist_ok=True)
        # Absolute paths in the log: the most common "where did my results go"
        # failure is running the entry point from a different working
        # directory than the one the operator is looking at.
        print(
            f"[runner] model={self.model} tag={self.tag}\n"
            f"[runner] results -> {os.path.abspath(self.results_root)}\n"
            f"[runner] logs    -> {os.path.abspath(self.log_dir)}"
        )
        if self.save_obs:
            # Derived from the ladder, not hard-coded: this said "6" and the free
            # disk-space warning was therefore a third of the truth once the ladder
            # became the reference 12 widths.
            levels = len(DEFAULT_LEVELS)
            expected = self.episodes_per_level * levels * self.max_steps
            print(
                f"[runner] --save_obs is on: up to {expected} PNGs can be written "
                f"for a full {levels}-Level run at {self.episodes_per_level} "
                f"episodes x {self.max_steps} steps"
            )


    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_all(
        self,
        levels: Sequence[int] = DEFAULT_LEVELS,
        episodes_per_level: Optional[int] = None,
        progress_callback: Optional[
            Callable[[int, int, int, Dict[str, Any], List[Dict[str, Any]]], None]
        ] = None,
        checkpoint: Optional[ProtocolCheckpoint] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Run every requested Level; Levels are fully independent."""
        episodes = (
            self.episodes_per_level
            if episodes_per_level is None
            else int(episodes_per_level)
        )
        results: Dict[int, List[Dict[str, Any]]] = {}
        for level in levels:
            results[int(level)] = self.run_level(
                level=int(level),
                episodes=episodes,
                progress_callback=progress_callback,
                checkpoint=checkpoint,
            )
        return results

    def run_level(
        self,
        level: int,
        episodes: Optional[int] = None,
        progress_callback: Optional[
            Callable[[int, int, int, Dict[str, Any], List[Dict[str, Any]]], None]
        ] = None,
        checkpoint: Optional[ProtocolCheckpoint] = None,
    ) -> List[Dict[str, Any]]:
        """Run ``episodes`` independent episodes at one channel width."""
        episodes = self.episodes_per_level if episodes is None else int(episodes)
        requested_width = level_channel_width(level)
        self.env.set_channel_width(requested_width)
        built_width = _env_channel_width(self.env, level)
        ratio = _env_a_s_ratio(self.env, built_width)
        if abs(built_width - requested_width) > 1e-9:
            raise RuntimeError(
                f"level {level}: environment built a {built_width:.4f} m channel "
                f"but the protocol requires {requested_width:.4f} m"
            )

        all_episodes: List[Dict[str, Any]] = []
        for episode_id in range(episodes):
            unit_key = f"level{level}/episode{episode_id:03d}"
            if checkpoint is not None and checkpoint.is_done(unit_key):
                saved = self._load_episode(level, episode_id)
                if saved is not None:
                    print(f"[checkpoint] skip {unit_key} (already completed)")
                    all_episodes.append(saved)
                    continue
                # Marked complete but the record is gone or damaged: re-running
                # costs one episode, skipping it would silently shrink the
                # sample that the paper's statistics are computed from.
                print(
                    f"[checkpoint] {unit_key} is marked complete but its episode "
                    "record is missing or unreadable; running it again"
                )

            episode = self._run_episode(
                level=level,
                episode_id=episode_id,
                channel_width=built_width,
                ratio=ratio,
            )
            all_episodes.append(episode)
            self._save_episode(episode)
            if checkpoint is not None:
                # Order matters: the episode record is on disk before the
                # checkpoint claims it is done, so a crash in between costs a
                # re-run rather than losing the episode.
                checkpoint.mark(unit_key)

            print(
                f"[level {level} A/S {ratio:.2f}] episode {episode_id + 1}/{episodes} "
                f"passed={episode['passed']} "
                f"sideways={episode['passed_sideways']} "
                f"steps={episode['total_steps']} "
                f"end={episode['end_reason']}"
            )
            if progress_callback is not None:
                progress_callback(
                    level, episode_id + 1, episodes, episode, all_episodes
                )
            # Refresh the Level summary after every episode as well, so an
            # interrupted Level still has a summary that matches its episodes.
            self._save_summary(level, all_episodes)

        if all_episodes:
            self._save_summary(level, all_episodes)
        return all_episodes

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        level: int,
        episode_id: int,
        channel_width: float,
        ratio: float,
    ) -> Dict[str, Any]:
        agent_log = os.path.join(
            self.log_dir, f"level{level}_episode{episode_id:03d}_agent.txt"
        )
        # Logs are appended to across a resume, so stamp where this attempt
        # starts: otherwise a re-run of the same episode looks like one
        # impossibly long model conversation.
        with open(agent_log, "a", encoding="utf-8") as handle:
            handle.write(
                f"{'=' * 60}\n"
                f"episode start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"model={self.model} tag={self.tag} level={level} "
                f"episode={episode_id:03d}\n"
            )
        agent = AgentAdapter(model=self.model, log_file=agent_log)

        # The adapter owns the within-episode memory.  Binding the local name to
        # its list (rather than to a second, parallel list) is what makes the
        # prompt block and the ``history=`` argument provably the same object.
        history: List[Dict[str, Any]] = agent.history

        self.env.reset_scene()
        steps: List[Dict[str, Any]] = []
        action_sequence: List[str] = []
        wall_collision_count = 0
        invalid_response_count = 0
        total_llm_time_ms = 0.0

        passed = False
        end_reason = "max_steps"
        first_turn_step: Optional[int] = None
        first_sideways_step: Optional[int] = None
        passed_sideways = False
        passage_sideways: Optional[bool] = None
        # The torso angle at the moment the body reaches the wall plane.  Stored
        # as an angle and not only as the sideways boolean because the human
        # comparison is a CURVE -- shoulder rotation against aperture width
        # (Warren & Whang 1987) -- and a boolean at an arbitrary 45 degree
        # boundary throws away exactly the graded part of the measurement that
        # the walking-frame action space now makes available.
        passage_rotation_deg: Optional[float] = None
        # The largest torso angle reached at any point in the episode.  The human
        # studies report the AMPLITUDE of shoulder rotation as a function of A/S,
        # and an agent can rotate and then turn back before entering; the passage
        # angle alone would score that as "never turned".
        max_rotation_deg = 0.0
        total_rotation = 0.0
        final_rotation = 0.0
        final_position_x = 0.0
        final_position_z = 0.0

        for step in range(self.max_steps):
            rgb = self.env.get_camera_image()
            state = self.env.get_robot_state()
            prompt = build_prompt(
                state=state,
                history=history,
                max_steps=self.max_steps,
                move_step=state.get("move_step"),
            )
            action_name, _raw, reasoning, latency_ms = agent.query(prompt, rgb, state)
            total_llm_time_ms += latency_ms

            action_legal = False
            collision_info: Optional[Dict[str, Any]] = None
            if action_name is None:
                action_taken = "invalid"
                feedback = "invalid action response"
                collision = False
                invalid_response_count += 1
                # Nothing moved, so re-read the pose to keep the record truthful.
                current = self.env.get_robot_state()
                torso_rotation = float(
                    current.get("torso_rotation", self.env.get_torso_rotation())
                )
                position = list(current.get("position", [0.0, 0.0, 0.0]))
            else:
                result = self.env.execute_action(action_name)
                action_taken = action_name
                feedback = result.feedback
                action_legal = bool(result.legal)
                collision_info = result.collision
                if collision_info is not None and not isinstance(collision_info, dict):
                    collision_info = {"part": "body"}
                # A collision is a rejected action carrying analytic collision
                # details (or a pose that somehow ends overlapping the wall).
                # Invalid responses and unrelated illegality are not collisions.
                collision = bool(collision_info is not None or _env_collision(self.env))
                new_state = result.state or self.env.get_robot_state()
                torso_rotation = float(
                    new_state.get("torso_rotation", self.env.get_torso_rotation())
                )
                position = list(new_state.get("position", [0.0, 0.0, 0.0]))

            obstacle_collision = bool(
                collision_info is not None and "panel" in collision_info
            )
            if obstacle_collision:
                wall_collision_count += 1
            if action_legal and action_taken in ("turn_left", "turn_right"):
                total_rotation += TURN_STEP_DEG
                if first_turn_step is None:
                    first_turn_step = step
            if first_sideways_step is None and _is_sideways_yaw(torso_rotation):
                first_sideways_step = step
            max_rotation_deg = max(max_rotation_deg, _abs_yaw(torso_rotation))

            # Score passage orientation when the body first reaches the wall
            # plane, not three metres later at the success plane: a model that
            # rotates back to face forward after getting through -- which is what
            # a person does -- would otherwise be scored as never having turned.
            if passage_sideways is None and float(position[0]) >= WALL_X:
                passage_sideways = bool(_is_sideways_yaw(torso_rotation))
                passage_rotation_deg = float(torso_rotation)

            passed = _env_passed(self.env)
            final_rotation = torso_rotation
            final_position_x = float(position[0])
            final_position_z = float(position[2])

            steps.append(
                {
                    "step": step,
                    "action": action_taken,
                    "torso_rotation": torso_rotation,
                    "position_x": final_position_x,
                    "position_z": final_position_z,
                    "collision": bool(collision),
                    "step_success": bool(passed),
                    "llm_response_time_ms": round(latency_ms, 3),
                }
            )
            action_sequence.append(action_taken)
            agent.record(action_taken, feedback, reasoning, step=step)

            if self.save_obs:
                self._save_observation(level, episode_id, step, action_taken, rgb)

            if passed:
                passed_sideways = bool(
                    passage_sideways
                    if passage_sideways is not None
                    else _is_sideways_yaw(torso_rotation)
                )
                end_reason = "success"
                break

        return {
            "episode_id": episode_id,
            "level": int(level),
            "model_name": self.model,
            "run_tag": self.tag,
            "channel_width": float(channel_width),
            "a_s_ratio": float(ratio),
            "passed": bool(passed),
            "passed_sideways": bool(passed_sideways),
            "passage_rotation_deg": (
                None if passage_rotation_deg is None else float(passage_rotation_deg)
            ),
            "max_rotation_deg": float(max_rotation_deg),
            "total_rotation": total_rotation,
            "first_turn_step": first_turn_step,
            "first_sideways_step": first_sideways_step,
            "total_steps": len(steps),
            "action_sequence": ",".join(action_sequence),
            "final_torso_rotation": final_rotation,
            "final_position_x": final_position_x,
            "final_position_z": final_position_z,
            "max_steps": self.max_steps,
            "end_reason": end_reason,
            "wall_collision_count": wall_collision_count,
            "invalid_response_count": invalid_response_count,
            "total_llm_time_ms": round(total_llm_time_ms, 3),
            "turned": bool(first_turn_step is not None),
            "steps": steps,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _result_dir(self, level: int) -> str:
        # Isolate every run tag so checkpoint resume can never load episodes
        # produced by another scene configuration or invocation.
        path = os.path.join(
            self.results_root, f"level{level}", self.model, self.tag
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _episode_path(self, level: int, episode_id: int) -> str:
        return os.path.join(self._result_dir(level), f"episode_{episode_id:03d}.json")

    def _load_episode(self, level: int, episode_id: int) -> Optional[Dict[str, Any]]:
        """Read one episode record, quarantining it instead of crashing on it.

        A truncated JSON file (an interrupted write, a full disk, an editor
        saving over it) used to raise straight out of ``--resume`` before the
        run could do anything about it.
        """
        path = self._episode_path(level, episode_id)
        if not os.path.exists(path):
            return None
        try:
            return load_json_file(path)
        except Exception as exc:
            backup = backup_corrupt_file(path)
            print(
                f"[results] WARNING: {path} is unreadable "
                f"({type(exc).__name__}: {exc}); this episode will be re-run"
            )
            if backup:
                print(f"[results] damaged record kept as {backup}")
            return None

    def _save_episode(self, episode: Dict[str, Any]) -> None:
        """Write the episode JSON plus a separate per-step sidecar.

        Both files are written atomically: a half-written episode record is
        unreadable by ``json.load``, so an interrupted write would invalidate
        the very data the sweep exists to collect.
        """
        path = self._episode_path(episode["level"], episode["episode_id"])
        atomic_write_json(path, episode)

        steps_path = os.path.join(
            self._result_dir(episode["level"]),
            f"episode_{episode['episode_id']:03d}_steps.json",
        )
        atomic_write_json(steps_path, episode.get("steps", []))

    def _save_summary(self, level: int, episodes: List[Dict[str, Any]]) -> None:
        path = os.path.join(self._result_dir(level), f"summary_{self.tag}.json")
        atomic_write_json(path, self.summarize_level(level, episodes))

    def _save_observation(
        self, level: int, episode_id: int, step: int, action: str, rgb: Any
    ) -> None:
        """Save one camera frame; never let a PNG failure end an episode.

        Observations are a diagnostic extra.  Losing a frame is acceptable,
        aborting a scored episode (or the whole sweep) over one is not.
        """
        from PIL import Image

        path = os.path.join(
            self.obs_dir, f"level{level}_ep{episode_id:03d}_step{step:02d}_{action}.png"
        )
        try:
            import numpy as _np

            array = _np.asarray(rgb)
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3]
            if array.dtype != _np.uint8:
                array = _np.clip(array, 0, 255).astype(_np.uint8)
            Image.fromarray(array).save(path)
        except Exception as exc:  # pragma: no cover - depends on the renderer
            print(
                f"[results] WARNING: could not save observation {path} "
                f"({type(exc).__name__}: {exc})"
            )

    def save_args(self, args: Any) -> None:
        """Record the effective configuration next to the run's logs."""
        path = os.path.join(self.log_dir, "args.json")
        payload: Dict[str, Any] = dict(vars(args))
        # The resolved values, not just the flags: --tag may have been empty
        # and --resume/--save_obs are meaningless without the resulting paths.
        payload["resolved_tag"] = self.tag
        payload["resolved_results_root"] = os.path.abspath(self.results_root)
        payload["resolved_logs_root"] = os.path.abspath(self.logs_root)
        payload["resolved_obs_dir"] = (
            os.path.abspath(self.obs_dir) if self.save_obs else None
        )
        atomic_write_json(path, payload)


    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_level(
        level: int, episodes: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate the headline metrics for one Level."""
        if not episodes:
            return {"level": int(level), "episodes": 0}

        passed = [bool(e.get("passed")) for e in episodes]
        sideways = [bool(e.get("passed_sideways")) for e in episodes]
        first_turns = [
            int(e["first_turn_step"])
            for e in episodes
            if e.get("first_turn_step") is not None
        ]
        success_steps = [int(e.get("total_steps", 0)) for e in episodes if e.get("passed")]
        sideways_success_steps = [
            int(e.get("total_steps", 0))
            for e in episodes
            if e.get("passed") and e.get("passed_sideways")
        ]
        action_counter: Counter = Counter()
        for episode in episodes:
            rows = episode.get("steps", [])
            if rows:
                for step in rows:
                    action_counter.update([str(step.get("action", ""))])
            else:
                action_counter.update(
                    a for a in str(episode.get("action_sequence", "")).split(",") if a
                )

        first_turn_mean = float(np.mean(first_turns)) if first_turns else None
        return {
            "level": int(level),
            "model_name": episodes[0].get("model_name"),
            "channel_width": float(episodes[0].get("channel_width", float("nan"))),
            "a_s_ratio": float(episodes[0].get("a_s_ratio", float("nan"))),
            "episodes": len(episodes),
            "pass_rate": float(np.mean(passed)),
            "passed_count": int(sum(passed)),
            "sideways_rate": float(np.mean(sideways)),
            "passed_sideways_count": int(sum(sideways)),
            "turned_rate": float(np.mean([bool(e.get("turned")) for e in episodes])),
            "first_turn_step_mean": first_turn_mean,
            "first_turn_step_min": (int(min(first_turns)) if first_turns else None),
            "first_turn_step_max": (int(max(first_turns)) if first_turns else None),
            "avg_success_steps": (
                float(np.mean(success_steps)) if success_steps else None
            ),
            "avg_sideways_success_steps": (
                float(np.mean(sideways_success_steps)) if sideways_success_steps else None
            ),
            "avg_total_rotation_deg": float(
                np.mean([float(e.get("total_rotation", 0.0)) for e in episodes])
            ),
            # Shoulder rotation at the wall plane, the graded measurement; None
            # rather than 0.0 when no episode recorded one, so a legacy or empty
            # set cannot read as "never rotated".
            "avg_passage_rotation_deg": (
                float(
                    np.mean(
                        [
                            float(e["passage_rotation_deg"])
                            for e in episodes
                            if e.get("passage_rotation_deg") is not None
                        ]
                    )
                )
                if any(
                    e.get("passage_rotation_deg") is not None for e in episodes
                )
                else None
            ),
            "end_reason_counts": dict(Counter(str(e.get("end_reason")) for e in episodes)),
            "total_wall_collisions": int(
                sum(int(e.get("wall_collision_count", 0)) for e in episodes)
            ),
            "total_invalid_responses": int(
                sum(int(e.get("invalid_response_count", 0)) for e in episodes)
            ),
            "avg_llm_time_ms": float(
                np.mean([float(e.get("total_llm_time_ms", 0.0)) for e in episodes])
            ),
            "action_counts": dict(action_counter),
        }

    @staticmethod
    def sideways_threshold(
        level_summaries: Dict[int, Dict[str, Any]]
    ) -> Optional[float]:
        """Widest A/S ratio at which the agent chose to pass sideways.

        Levels are inspected from the widest channel (largest A/S) downwards, so
        the result is the FIRST A/S at which sideways passage appears -- which is
        also the widest one, and the direct analogue of the human threshold of
        1.30, above which people stop rotating.
        """
        candidates = [
            summary
            for summary in level_summaries.values()
            if int(summary.get("passed_sideways_count", 0)) > 0
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda s: float(s["a_s_ratio"]))
        return float(best["a_s_ratio"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BAO A/S threshold protocol.")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name passed to ai_agent"
    )
    parser.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES_PER_LEVEL,
        help=f"Episodes per level (default {DEFAULT_EPISODES_PER_LEVEL})",
    )
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="Log/result tag")
    parser.add_argument("--save_obs", action="store_true", help="Save per-step camera PNGs")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Skip episodes already in the checkpoint"
    )
    parser.add_argument(
        "--env_config", type=str, default="{}", help="JSON dict passed to BAOEnv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    env = None
    try:
        from environment import BAOEnv

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        env = BAOEnv(simulation_app, task_dict=task_dict)
        runner = BAOExperimentRunner(
            env=env,
            model=args.model,
            max_steps=args.max_steps,
            episodes_per_level=args.episodes,
            tag=args.tag,
            save_obs=args.save_obs,
            seed=args.seed,
        )
        runner.save_args(args)
        checkpoint = ProtocolCheckpoint(
            os.path.join("results", runner.model, f"checkpoint_{runner.tag}.json"),
            resume=args.resume,
        )
        results = runner.run_all(levels=args.levels, checkpoint=checkpoint)
        for level, episodes in results.items():
            summary = BAOExperimentRunner.summarize_level(level, episodes)
            print(
                f"[level {level}] pass_rate={summary['pass_rate']:.3f} "
                f"sideways_rate={summary['sideways_rate']:.3f}"
            )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
