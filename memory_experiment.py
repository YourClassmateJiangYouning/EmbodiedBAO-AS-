"""The Stage 2 runner: 6 runs of 17 attempts, with a self-written note between them.

``STAGE23_DESIGN.md`` is the authority.  Each model runs the whole thing 6 times; a
run is 12 attempts at an opening it cannot walk through facing forward (A/S 0.80)
followed by 5 at one it can (A/S 1.10).  Between attempts the model is asked, in a
separate text-only call, to write itself a note; that note is the only thing it can
read about the attempt next time.

The step loop here is a deliberate sibling of ``experiments._run_episode`` rather
than a call into it, because three things differ: the prompt carries a memory block,
the adapter is rebuilt every attempt (a fresh history is the manipulation), and the
attempt ends with a note call.  Everything else is shared, so a step means the same
thing in both experiments: the same ``AgentAdapter``, the same environment probes,
the same per-step record shape, and the same ``round_record``-style shaping.

Two boundaries this file has to respect:

* **No module-level ``import environment`` or ``import experiments``.**  Both pull in
  ``environment``, which latches ``environment._HAS_ISAAC_SIM`` to False when
  ``isaacsim`` cannot be imported -- after which the scene can never be built and the
  process exits 0 having done nothing.  ``main()`` constructs ``SimulationApp`` first
  and imports the environment inside the function, exactly as ``main.py`` does; the
  helpers from ``experiments`` are imported inside the methods that use them.
* **Nothing here writes outside ``results/`` and ``logs/``.**  Every round is written
  atomically before its checkpoint claims it is done, so an interruption costs the
  attempt in flight and a resumed invocation continues rather than starting over.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import persistence
from memory_protocol import (
    MAX_STEPS,
    MODE_ROLLING,
    PROTOCOL_TAG,
    ROUNDS_PER_RUN,
    RUNS_PER_MODEL,
    a_s_ratio_of,
    build_action_prompt,
    build_note_prompt,
    memory_block,
    memory_mode,
    note_log_path,
    phase_of,
    round_record,
    run_plan,
    width_of,
    write_note_log,
)

DEFAULT_RESULTS_ROOT = "results"
DEFAULT_LOGS_ROOT = "logs"


class BudgetExhausted(RuntimeError):
    """Raised when ``--max_calls`` is reached; caught to stop cleanly."""


class RoundCheckpoint:
    """Completion state for one run tag, written atomically.

    Same contract as ``experiments.ProtocolCheckpoint`` -- a completed unit is only
    recorded after its record is on disk -- but this one rebuilds itself from the
    ``run{RR}_round{NN}.json`` files next to it, which is what a Stage 2 results
    directory holds.  It is reimplemented here rather than subclassed so that this
    module imports nothing that touches Isaac Sim.
    """

    def __init__(self, path: str, resume: bool = False) -> None:
        self.path = path
        self.completed: set = set()
        self.recovered_from: Optional[str] = None
        if resume and os.path.exists(path):
            try:
                data = persistence.load_json_file(path)
                self.completed = set(str(item) for item in data.get("completed", []))
            except Exception as exc:
                # Never silently forget paid-for rounds: rebuild instead of
                # restarting, which would overwrite intact records.
                self.recovered_from = persistence.backup_corrupt_file(path)
                print(
                    f"[checkpoint] WARNING: cannot read {path} "
                    f"({type(exc).__name__}: {exc})"
                )
                if self.recovered_from:
                    print(f"[checkpoint] damaged checkpoint kept as {self.recovered_from}")
                self.completed = self._rebuild_from_records()
                print(
                    f"[checkpoint] rebuilt {len(self.completed)} completed round(s) "
                    "from the records already on disk"
                )

    def _rebuild_from_records(self) -> set:
        completed: set = set()
        directory = os.path.dirname(os.path.abspath(self.path))
        for path in sorted(glob.glob(os.path.join(directory, "run*_round*.json"))):
            match = re.match(r"run(\d+)_round(\d+)\.json$", os.path.basename(path))
            if not match or path.endswith("_steps.json"):
                continue
            try:
                persistence.load_json_file(path)
            except Exception:
                continue
            completed.add(round_key(int(match.group(1)), int(match.group(2))))
        return completed

    def is_done(self, key: str) -> bool:
        return key in self.completed

    def mark(self, key: str) -> None:
        self.completed.add(key)
        persistence.atomic_write_json(self.path, {"completed": sorted(self.completed)})


def round_key(run: int, round_number: int) -> str:
    return f"run{int(run)}/round{int(round_number):02d}"


def parse_runs(text: str) -> List[int]:
    """Parse a ``--runs`` list such as ``1,2,5`` or ``1-3`` into run numbers."""
    runs: List[int] = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            low, _, high = chunk.partition("-")
            if not (low.isdigit() and high.isdigit()):
                raise ValueError(f"cannot read the run range {chunk!r}")
            runs.extend(range(int(low), int(high) + 1))
        else:
            if not chunk.isdigit():
                raise ValueError(f"cannot read the run number {chunk!r}")
            runs.append(int(chunk))
    ordered = sorted(set(runs))
    if not ordered:
        raise ValueError("no runs were selected")
    for run in ordered:
        if not 1 <= run <= RUNS_PER_MODEL:
            raise ValueError(f"run {run} is outside 1..{RUNS_PER_MODEL}")
    return ordered


class MemoryExperimentRunner:
    """Runs the repeated-attempt protocol against a BAOEnv instance."""

    def __init__(
        self,
        env: Any,
        model: str,
        runs: Optional[Sequence[int]] = None,
        max_steps: int = MAX_STEPS,
        tag: str = PROTOCOL_TAG,
        results_root: str = DEFAULT_RESULTS_ROOT,
        logs_root: str = DEFAULT_LOGS_ROOT,
        max_calls: int = 0,
        save_obs: bool = False,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.model = model.replace("/", "-").replace("\\", "-")
        self.runs = list(runs) if runs else list(range(1, RUNS_PER_MODEL + 1))
        for run in self.runs:
            memory_mode(run)  # validates the range early, not on round 1
        self.max_steps = int(max_steps)
        self.tag = persistence.sanitize_tag(tag) if str(tag or "").strip() else PROTOCOL_TAG
        self.results_root = results_root
        self.logs_root = logs_root
        self.max_calls = int(max_calls)
        self.save_obs = bool(save_obs)
        self.seed = int(seed)
        self.calls = 0
        print(
            f"[memory] model={self.model} tag={self.tag} runs={self.runs} "
            f"rounds/run={ROUNDS_PER_RUN} max_steps={self.max_steps}\n"
            f"[memory] results -> {os.path.abspath(self.results_root)}\n"
            f"[memory] logs    -> {os.path.abspath(self.logs_root)}"
        )

    # ------------------------------------------------------------------
    # Paths.  Every one of them is derived, so a record and its sidecar and its
    # note cannot end up under different names.
    # ------------------------------------------------------------------

    def tag_for_run(self, run: int) -> str:
        """``{tag}-cum`` for runs 1-4 and ``{tag}-roll`` for runs 5-6."""
        return f"{self.tag}-{'cum' if memory_mode(run) == 'cumulative' else 'roll'}"

    def record_dir(self, tag: str) -> str:
        return os.path.join(self.results_root, "memory", self.model, tag)

    def record_path(self, tag: str, run: int, round_number: int) -> str:
        return os.path.join(self.record_dir(tag), f"run{run:02d}_round{round_number:02d}.json")

    def step_path(self, tag: str, run: int, round_number: int) -> str:
        return os.path.join(
            self.record_dir(tag), f"run{run:02d}_round{round_number:02d}_steps.json"
        )

    def agent_log_path(self, tag: str, run: int, round_number: int) -> str:
        return os.path.join(
            self.logs_root, tag, f"run{run:02d}_round{round_number:02d}_agent.txt"
        )

    def note_path(self, tag: str, run: int, round_number: int) -> str:
        return note_log_path(self.logs_root, tag, run, round_number)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_record(self, tag: str, run: int, round_number: int) -> Optional[Dict[str, Any]]:
        path = self.record_path(tag, run, round_number)
        if not os.path.exists(path):
            return None
        try:
            return persistence.load_json_file(path)
        except Exception as exc:
            print(f"[memory] WARNING: unreadable {path} ({type(exc).__name__}: {exc})")
            return None

    def completed_attempts(self, run: int) -> List[Dict[str, Any]]:
        """Every finished attempt of this run, oldest first, with its note.

        Read from disk rather than kept in memory: a resumed invocation has to
        rebuild the block the model would have seen, and the notes ARE the
        manipulation, so they cannot be reconstructed from anything else.
        """
        tag = self.tag_for_run(run)
        attempts: List[Dict[str, Any]] = []
        for round_number in range(1, ROUNDS_PER_RUN + 1):
            record = self.load_record(tag, run, round_number)
            if record is None:
                break
            attempts.append(
                {
                    "round": round_number,
                    "passed": bool(record.get("passed")),
                    "steps": int(record.get("total_steps") or 0),
                    "note": str(record.get("note_text") or ""),
                }
            )
        return attempts

    def save_args(self, args: Any) -> None:
        """Record the effective settings once per run tag being written."""
        payload = {
            "model": self.model,
            "protocol_tag": self.tag,
            "runs": self.runs,
            "rounds_per_run": ROUNDS_PER_RUN,
            "max_steps": self.max_steps,
            "max_calls": self.max_calls,
            "save_obs": self.save_obs,
            "seed": self.seed,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cli": {key: str(value) for key, value in sorted(vars(args).items())}
            if args is not None
            else {},
            "plan": run_plan(),
        }
        for run in self.runs:
            tag = self.tag_for_run(run)
            directory = os.path.join(self.logs_root, tag)
            os.makedirs(directory, exist_ok=True)
            persistence.atomic_write_json(os.path.join(directory, "args.json"), payload)
            os.makedirs(self.record_dir(tag), exist_ok=True)

    # ------------------------------------------------------------------
    # The experiment
    # ------------------------------------------------------------------

    def make_checkpoints(self, resume: bool = False) -> Dict[str, RoundCheckpoint]:
        """One checkpoint per run tag, because a tag is what ``--resume`` keys on.

        A single checkpoint for the whole invocation would have to live under one
        of the two tag directories, and its rebuild-on-corruption scan would then
        miss the other tag's records.
        """
        checkpoints: Dict[str, RoundCheckpoint] = {}
        for run in self.runs:
            tag = self.tag_for_run(run)
            os.makedirs(self.record_dir(tag), exist_ok=True)
            checkpoints[tag] = RoundCheckpoint(
                os.path.join(self.record_dir(tag), "checkpoint.json"), resume=resume
            )
        return checkpoints

    def run_all(
        self, checkpoints: Optional[Dict[str, RoundCheckpoint]] = None
    ) -> List[Dict[str, Any]]:
        """Run every selected run, resuming past rounds already on disk."""
        records: List[Dict[str, Any]] = []
        try:
            for run in self.runs:
                tag = self.tag_for_run(run)
                os.makedirs(self.record_dir(tag), exist_ok=True)
                os.makedirs(os.path.join(self.logs_root, tag), exist_ok=True)
                checkpoint = (checkpoints or {}).get(tag)
                for round_number in range(1, ROUNDS_PER_RUN + 1):
                    key = round_key(run, round_number)
                    if checkpoint is not None and checkpoint.is_done(key):
                        saved = self.load_record(tag, run, round_number)
                        if saved is not None:
                            print(f"[checkpoint] skip {tag} {key} (already completed)")
                            records.append(saved)
                            continue
                        print(
                            f"[checkpoint] {tag} {key} is marked complete but its "
                            "record is missing or unreadable; running it again"
                        )
                    attempts = self.completed_attempts(run)
                    record = self.run_round(run, round_number, attempts)
                    records.append(record)
                    if checkpoint is not None:
                        # Order matters: the record is on disk before the
                        # checkpoint claims it, so a crash in between costs a
                        # re-run instead of losing the round.
                        checkpoint.mark(key)
        except BudgetExhausted as exc:
            print(f"[memory] stopping: {exc}")
        return records

    def run_round(
        self,
        run: int,
        round_number: int,
        attempts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """One attempt: fresh agent, memory block, up to 30 steps, then a note."""
        # Imported here, not at module level: experiments imports environment, and
        # importing environment before SimulationApp exists latches the scene
        # unbuildable for the life of the process.
        from environment import WALL_X
        from experiments import AgentAdapter, _env_collision, _env_passed

        mode = memory_mode(run)
        tag = self.tag_for_run(run)
        width = width_of(round_number)
        phase = phase_of(round_number)
        ratio = a_s_ratio_of(round_number)
        injected = memory_block(mode, attempts)

        # Width first, then the reset, then read it back: a reset that rebuilt the
        # scene at the default width would otherwise silently run the probe phase at
        # the learning width, which is the whole experiment.
        self.env.set_channel_width(width)
        self.env.reset_scene()
        built = float(width)
        getter = getattr(self.env, "get_channel_width", None)
        if callable(getter):
            try:
                built = float(getter())
            except Exception:
                built = float(width)
        if abs(built - width) > 1e-9:
            raise RuntimeError(
                f"run {run} round {round_number}: environment built a {built:.4f} m "
                f"channel but the protocol requires {width:.4f} m"
            )

        agent_log = self.agent_log_path(tag, run, round_number)
        os.makedirs(os.path.dirname(agent_log), exist_ok=True)
        if not os.path.exists(agent_log):
            with open(agent_log, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{'=' * 60}\nattempt start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"model={self.model} tag={tag} run={run} round={round_number:02d} "
                    f"phase={phase} width={width:.3f}\n"
                )
        agent = AgentAdapter(model=self.model, log_file=agent_log)
        history: List[Dict[str, Any]] = agent.history

        steps: List[Dict[str, Any]] = []
        wall_collision_count = 0
        invalid_response_count = 0
        passage_rotation_deg: Optional[float] = None
        max_rotation_deg = 0.0
        final_x = 0.0
        final_z = 0.0
        passed = False
        end_reason = "max_steps"

        for step in range(self.max_steps):
            self._charge_call()
            rgb = self.env.get_camera_image()
            state = self.env.get_robot_state()
            prompt = build_action_prompt(
                state=state,
                history=history,
                mode=mode,
                attempts=attempts,
                max_steps=self.max_steps,
                move_step=state.get("move_step"),
            )
            action_name, _raw, reasoning, _latency = agent.query(prompt, rgb, state)

            collision_info: Optional[Dict[str, Any]] = None
            if action_name is None:
                action_taken = "invalid"
                feedback = "invalid action response"
                invalid_response_count += 1
                current = self.env.get_robot_state()
                torso_rotation = float(
                    current.get("torso_rotation", self.env.get_torso_rotation())
                )
                position = list(current.get("position", [0.0, 0.0, 0.0]))
            else:
                result = self.env.execute_action(action_name)
                action_taken = action_name
                feedback = result.feedback
                collision_info = result.collision
                if collision_info is not None and not isinstance(collision_info, dict):
                    collision_info = {"part": "body"}
                # A collision is a rejected action carrying analytic collision
                # details, or a pose that somehow ends overlapping the wall.
                # Invalid responses and unrelated illegality are not collisions,
                # and only a panel collision counts towards the wall tally -- the
                # same split the threshold study makes.
                collision = bool(collision_info is not None or _env_collision(self.env))
                if collision_info is not None and "panel" in collision_info:
                    wall_collision_count += 1
                new_state = result.state or self.env.get_robot_state()
                torso_rotation = float(
                    new_state.get("torso_rotation", self.env.get_torso_rotation())
                )
                position = list(new_state.get("position", [0.0, 0.0, 0.0]))

            max_rotation_deg = max(max_rotation_deg, _abs_yaw(torso_rotation))
            if passage_rotation_deg is None and float(position[0]) >= WALL_X:
                passage_rotation_deg = float(torso_rotation)

            passed = _env_passed(self.env)
            final_x = float(position[0])
            final_z = float(position[2])
            steps.append(
                {
                    "step": step,
                    "action": action_taken,
                    "torso_rotation": torso_rotation,
                    "position_x": final_x,
                    "position_z": final_z,
                    "collision": bool(collision),
                }
            )
            agent.record(action_taken, feedback, reasoning, step=step)
            if passed:
                end_reason = "success"
                break

        # The note is written after every attempt except the last of the run: the
        # last one has no next attempt to remember for, which is why a run costs 16
        # note calls rather than 17.
        note_text = ""
        if round_number < ROUNDS_PER_RUN:
            previous_note = None
            if mode == MODE_ROLLING and attempts:
                previous_note = str(attempts[-1].get("note") or "")
            note_text = self.write_note(
                tag=tag,
                run=run,
                round_number=round_number,
                agent=agent,
                history=history,
                passed=passed,
                steps=len(steps),
                mode=mode,
                previous_note=previous_note,
            )

        record = round_record(
            model=self.model,
            run=run,
            round_number=round_number,
            passed=passed,
            total_steps=len(steps),
            end_reason=end_reason,
            steps=steps,
            final_x=final_x,
            final_z=final_z,
            max_rotation_deg=max_rotation_deg,
            passage_rotation_deg=passage_rotation_deg,
            wall_collision_count=wall_collision_count,
            invalid_response_count=invalid_response_count,
            note_text=note_text,
            memory_injected_chars=len(injected),
            max_steps=self.max_steps,
        )
        record["tag"] = tag
        record["max_steps"] = self.max_steps
        record["total_llm_calls"] = self.calls

        persistence.atomic_write_json(self.step_path(tag, run, round_number), steps)
        persistence.atomic_write_json(self.record_path(tag, run, round_number), record)
        print(
            f"[run {run} round {round_number:02d} phase {phase} A/S {ratio:.2f}] "
            f"passed={passed} steps={len(steps)} end={end_reason} "
            f"label={record['strategy_label']} gap="
            f"{'n/a' if record['gap'] is None else format(record['gap'], '+.4f')} "
            f"note={record['note_chars']}ch memory={record['memory_injected_chars']}ch"
        )
        return record

    def write_note(
        self,
        *,
        tag: str,
        run: int,
        round_number: int,
        agent: Any,
        history: Sequence[Dict[str, Any]],
        passed: bool,
        steps: int,
        mode: str,
        previous_note: Optional[str],
    ) -> str:
        """One text-only call, stored verbatim, plus its log."""
        prompt = build_note_prompt(
            history,
            passed=passed,
            steps=steps,
            mode=mode,
            previous_note=previous_note,
            max_steps=self.max_steps,
        )
        self._charge_call()
        response = agent.note(prompt)
        text = str(response or "").strip()
        path = self.note_path(tag, run, round_number)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_note_log(path, prompt, text)
        if not text:
            print(
                f"[memory] WARNING: run {run} round {round_number} produced no note; "
                "the next attempt gets an empty memory"
            )
        return text

    def _charge_call(self) -> None:
        if self.max_calls and self.calls >= self.max_calls:
            raise BudgetExhausted(
                f"--max_calls={self.max_calls} reached after {self.calls} model call(s); "
                "every finished round is on disk and the checkpoint is current, so "
                "re-running with --resume continues from here"
            )
        self.calls += 1


def _abs_yaw(yaw_deg: float) -> float:
    """Fold a torso yaw into [0, 180] so +/- rotations behave symmetrically."""
    yaw = abs(float(yaw_deg)) % 360.0
    return float(360.0 - yaw if yaw > 180.0 else yaw)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: repeated attempts at an opening with a self-written note "
            "between them (see STAGE23_DESIGN.md)"
        )
    )
    parser.add_argument("--model", required=True, help="model name from models.json")
    parser.add_argument(
        "--runs",
        default="1-6",
        help="which runs to execute, e.g. 1-4, 5,6 or 3 (default 1-6)",
    )
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--tag", default=PROTOCOL_TAG, help="protocol tag stem")
    parser.add_argument("--resume", action="store_true", help="skip finished rounds")
    parser.add_argument(
        "--max_calls",
        type=int,
        default=0,
        help="stop cleanly after this many model calls (0 = no limit)",
    )
    parser.add_argument("--save_obs", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env_config",
        default="{}",
        help="JSON overrides handed to BAOEnv, as main.py does",
    )
    return parser.parse_args(argv)


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
        runner = MemoryExperimentRunner(
            env=env,
            model=args.model,
            runs=parse_runs(args.runs),
            max_steps=args.max_steps,
            tag=args.tag,
            max_calls=args.max_calls,
            save_obs=args.save_obs,
            seed=args.seed,
        )
        runner.save_args(args)
        records = runner.run_all(checkpoint=runner.make_checkpoints(resume=args.resume))
        passed = sum(1 for record in records if record["passed"])
        print(
            f"[memory] finished {len(records)} round(s), {passed} passed, "
            f"{runner.calls} model call(s)"
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
