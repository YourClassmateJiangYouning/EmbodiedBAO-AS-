"""The repeated-attempt experiment: run plan, prompts, memory and round records.

``STAGE23_DESIGN.md`` is the authority for everything here.  Stage 2 gives a model
12 attempts at an opening it cannot walk through facing forward (A/S 0.80), letting
it write itself a note between attempts, and then moves it to an opening it *can*
walk through facing forward (A/S 1.10) for 5 attempts, to see whether the rotation
it has learned has become a habit.

This module is the Stage 2 counterpart of ``protocol.py``: the plan, the three
prompts, the memory block, and the shape of a round record.  The round loop that
drives the simulator belongs in ``memory_experiment.py``, the way ``experiments.py``
drives Stage 1, and the measures live in ``memory_metrics.py``.

Three things are deliberate and worth reading before changing anything:

* **The action prompt is not a rewrite.**  It is built by ``protocol.build_prompt``
  with a memory block and one wording change, so the task sentence, the action list,
  the walking-frame note, the state block and the response format are the same
  strings Stage 1 used.  A second, hand-written prompt would drift from the first,
  and the threshold study is this experiment's baseline.
* **The memory block is the only thing a model can read about its previous
  attempts.**  Within a round it sees its own steps; at the start of the next round
  it sees the note it wrote, and in cumulative mode a one-line outcome for each
  earlier attempt.  Nothing else survives, which is what makes "did it learn"
  a question about the model rather than about the log.
* **Nothing tells it what the obstacle is.**  The task sentence, the memory block,
  the outcome lines and the note instruction never mention an opening, a channel,
  width, shoulders, sideways movement or turning to fit.  A model that improves has
  to have worked it out.  ``FORBIDDEN_WORDS`` below is the vocabulary the tests
  scan our own text for; a model's own note is exempt, because it can only write
  what it has concluded.
"""

from __future__ import annotations

import collections
import os
from typing import Any, Dict, List, Optional, Sequence

import protocol
from memory_metrics import (
    SHOULDER_WIDTH_M,
    WIDTH_LEARNING_M,
    WIDTH_PROBE_M,
    excess,
    gap_for,
    optimal_steps,
    reached_door,
    strategy_label,
)

# ---------------------------------------------------------------------------
# The plan (STAGE23_DESIGN.md 2)
# ---------------------------------------------------------------------------
# Every model runs the whole thing 6 times.  Runs 1-4 accumulate notes and an
# outcome log; runs 5-6 keep only the single note the model rewrote last.  The
# difference is whether the environment supplies a factual history, which is what
# separates "practised 12 times" from "was told it had practised 12 times".
RUNS_PER_MODEL = 6
CUMULATIVE_RUNS = 4
ROLLING_RUNS = 2

# 12 attempts at a width that cannot be walked through facing forward, then 5 at a
# width that can.  The probe phase is the habit measurement, and its prompt is
# byte-identical to the learning phase's: the model is not told the obstacle moved.
ROUNDS_LEARNING = 12
ROUNDS_PROBE = 5
ROUNDS_PER_RUN = ROUNDS_LEARNING + ROUNDS_PROBE

MAX_STEPS = 30

PHASE_LEARNING = "A"
PHASE_PROBE = "B"

MODE_CUMULATIVE = "cumulative"
MODE_ROLLING = "rolling"
MEMORY_MODES = (MODE_CUMULATIVE, MODE_ROLLING)

PROTOCOL_TAG = "v8-memory-a08-a11"

# The two end reasons the runner can produce; a record whose pass flag disagrees
# with its end reason is a bug, not a result.
END_REASONS = ("success", "max_steps")

# STAGE23_DESIGN.md 4.4: words that would hand the model the answer, and the
# numbers that would hand it the geometry.  Scanned case-insensitively.
FORBIDDEN_WORDS = (
    "opening",
    "channel",
    "wide",
    "narrow",
    "shoulder",
    "sideways",
    "turn to fit",
    "you should",
    "a/s",
    "0.80",
    "0.90",
    "1.10",
    "0.57",
    "0.456",
    "0.627",
)

# Which action counts as what.  ``look_down`` is reported on its own because it is
# the self-inspection glance the threshold study found predicts a turn 2.9x more
# often than baseline; the two sideways glances are one number.
TURN_ACTIONS = ("turn_left", "turn_right")
LATERAL_ACTIONS = ("left", "right")


# ---------------------------------------------------------------------------
# The plan, as data
# ---------------------------------------------------------------------------


def memory_mode(run: int) -> str:
    """Which memory the run uses: runs 1-4 cumulative, runs 5-6 rolling."""
    if not 1 <= int(run) <= RUNS_PER_MODEL:
        raise ValueError(f"run {run} is outside 1..{RUNS_PER_MODEL}")
    return MODE_CUMULATIVE if int(run) <= CUMULATIVE_RUNS else MODE_ROLLING


def phase_of(round_number: int) -> str:
    """``A`` for the 12 learning rounds, ``B`` for the 5 probe rounds."""
    if not 1 <= int(round_number) <= ROUNDS_PER_RUN:
        raise ValueError(f"round {round_number} is outside 1..{ROUNDS_PER_RUN}")
    return PHASE_LEARNING if int(round_number) <= ROUNDS_LEARNING else PHASE_PROBE


def width_of(round_number: int) -> float:
    """The opening width this round is run at, in metres."""
    return WIDTH_LEARNING_M if phase_of(round_number) == PHASE_LEARNING else WIDTH_PROBE_M


def a_s_ratio_of(round_number: int) -> float:
    """The round's width divided by the body's shoulder width."""
    return width_of(round_number) / SHOULDER_WIDTH_M


def tag_for(mode: str) -> str:
    """The run tag for a memory mode; results are keyed on it."""
    if mode not in MEMORY_MODES:
        raise ValueError(f"unknown memory mode {mode!r}, expected one of {MEMORY_MODES}")
    return f"{PROTOCOL_TAG}-{'cum' if mode == MODE_CUMULATIVE else 'roll'}"


def run_plan() -> List[Dict[str, Any]]:
    """Every (run, round) pair the experiment consists of, in order.

    6 runs x 17 rounds = 102 rounds per model.  Built as data rather than looped in
    place so the plan can be checked (and printed) without running anything.
    """
    plan: List[Dict[str, Any]] = []
    for run in range(1, RUNS_PER_MODEL + 1):
        mode = memory_mode(run)
        for round_number in range(1, ROUNDS_PER_RUN + 1):
            width = width_of(round_number)
            plan.append(
                {
                    "run": run,
                    "round": round_number,
                    "memory_mode": mode,
                    "phase": phase_of(round_number),
                    "channel_width": width,
                    "a_s_ratio": width / SHOULDER_WIDTH_M,
                    "tag": tag_for(mode),
                    "optimal_steps": optimal_steps(width),
                }
            )
    return plan


# ---------------------------------------------------------------------------
# The memory the model is given at the start of an attempt
# ---------------------------------------------------------------------------


def outcome_phrase(passed: bool, steps: int, max_steps: int = MAX_STEPS) -> str:
    """The environment's own words for how an attempt ended.

    Deliberately about reaching the marker only.  Anything about *why* it did not
    (blocked, too narrow, should have turned) would be the answer.
    """
    if passed:
        return f"passed (reached the red marker in {int(steps)} steps)"
    return f"failed (did not reach the red marker within {int(max_steps)} steps)"


def memory_block(
    mode: str,
    attempts: Sequence[Dict[str, Any]],
) -> str:
    """The block that carries memory from one attempt to the next.

    ``attempts`` is one dict per *finished* attempt in this run, oldest first, with
    ``round``, ``passed``, ``steps`` and ``note``.  The first attempt of a run gets
    no block at all, in either mode: there is nothing to remember yet.

    Cumulative mode shows a one-line outcome and the note for every attempt, so the
    model knows how many times it practised and how often it succeeded.  Rolling
    mode shows only the single note it rewrote last time -- no round numbers, no
    outcomes -- so its only memory is a sentence it wrote itself and has now
    rewritten eleven times (Bartlett's serial reproduction, as a manipulation).
    """
    attempts = list(attempts)
    if not attempts:
        return ""
    if mode == MODE_ROLLING:
        note = str(attempts[-1].get("note") or "")
        return f'Your note from your previous attempt:\n"{note}"'
    if mode == MODE_CUMULATIVE:
        lines = ["Previous attempts (oldest first):"]
        for attempt in attempts:
            lines.append(
                f"- round {int(attempt['round'])}: "
                f"{outcome_phrase(bool(attempt.get('passed')), int(attempt.get('steps') or 0))}"
            )
            lines.append(f'  your note: "{str(attempt.get("note") or "")}"')
        return "\n".join(lines)
    raise ValueError(f"unknown memory mode {mode!r}, expected one of {MEMORY_MODES}")


# ---------------------------------------------------------------------------
# Prompt A / C: the action call
# ---------------------------------------------------------------------------


def build_action_prompt(
    state: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    mode: Optional[str] = None,
    attempts: Optional[Sequence[Dict[str, Any]]] = None,
    max_steps: int = MAX_STEPS,
    move_step: Optional[float] = None,
) -> str:
    """The prompt for a step of an attempt.

    The same call serves the learning phase and the probe phase: the phase changes
    the obstacle, not the text, which is how the probe measures a habit rather than
    a response to being told the situation changed.
    """
    block = memory_block(mode, attempts or []) if mode else ""
    return protocol.build_prompt(
        state=state,
        history=history,
        max_steps=max_steps,
        move_step=move_step,
        memory_block=block or None,
        attempt_wording=True,
    )


# ---------------------------------------------------------------------------
# Prompt B: the note call
# ---------------------------------------------------------------------------
NOTE_INSTRUCTION_CUMULATIVE = (
    "Write a short note to yourself for your next attempt. Your note is the only "
    "thing you will be able to read about this attempt next time; the record above "
    "will not be shown to you again."
)

NOTE_INSTRUCTION_ROLLING = (
    "Here is your previous note and the complete record of this attempt. Rewrite "
    "your note to yourself for your next attempt - it will replace the previous "
    "one. Your note is the only thing you will be able to read about this attempt "
    "next time; the record above will not be shown to you again."
)

NOTE_PROMPT_HEADER = (
    "You are a Unitree H1 humanoid robot. You have just finished one attempt at "
    "your task: reaching the red marker on the far wall."
)


def render_attempt_record(
    history: Sequence[Dict[str, Any]],
    passed: bool,
    steps: int,
    max_steps: int = MAX_STEPS,
) -> str:
    """The complete record of one attempt, with the reasoning in full.

    Unlike the action prompt, which clips reasoning to
    ``protocol.HISTORY_REASONING_CHARS`` to keep 30 steps readable, this renders
    every word the model wrote.  The note is the model's summary of its own
    thinking, so handing it a truncated version of that thinking would make the
    memory an artefact of the prompt budget.
    """
    lines: List[str] = []
    for index, item in enumerate(history):
        label = item.get("step")
        if label is None:
            label = index
        line = f"- step {label}: {item.get('action')} -> {item.get('feedback')}"
        reasoning = " ".join(str(item.get("reasoning") or "").split())
        if reasoning:
            line += f' | your reasoning: "{reasoning}"'
        lines.append(line)
    lines.append(f"- the attempt ended: {outcome_phrase(passed, steps, max_steps)}")
    return "\n".join(lines)


def build_note_prompt(
    history: Sequence[Dict[str, Any]],
    passed: bool,
    steps: int,
    mode: str,
    previous_note: Optional[str] = None,
    max_steps: int = MAX_STEPS,
) -> str:
    """The prompt for the end-of-attempt note: text only, no image.

    In rolling mode the model's previous note is shown above the record, because
    its instruction refers to "your previous note and the complete record of this
    attempt" and the note is no longer anywhere else: rolling mode replaced the
    outcome log with that one sentence.
    """
    parts = [NOTE_PROMPT_HEADER]
    if mode == MODE_ROLLING and previous_note is not None:
        parts.append(f'Your note from your previous attempt:\n"{previous_note}"')
    parts.append("Complete record of this attempt:\n" + render_attempt_record(history, passed, steps, max_steps))
    if mode == MODE_CUMULATIVE:
        parts.append(NOTE_INSTRUCTION_CUMULATIVE)
    elif mode == MODE_ROLLING:
        parts.append(NOTE_INSTRUCTION_ROLLING)
    else:
        raise ValueError(f"unknown memory mode {mode!r}, expected one of {MEMORY_MODES}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# The round record (STAGE23_DESIGN.md 5)
# ---------------------------------------------------------------------------


def round_record(
    *,
    model: str,
    run: int,
    round_number: int,
    passed: bool,
    total_steps: int,
    end_reason: str,
    steps: Sequence[Dict[str, Any]],
    final_x: float,
    final_z: float,
    max_rotation_deg: float,
    passage_rotation_deg: Optional[float],
    wall_collision_count: int,
    invalid_response_count: int,
    note_text: str = "",
    memory_injected_chars: int = 0,
    max_steps: int = MAX_STEPS,
) -> Dict[str, Any]:
    """Shape one finished round into the record the analysis reads.

    Every derived field is computed here rather than by the caller, so the record
    and the measures in ``memory_metrics`` cannot disagree.  ``steps`` is the
    per-step sidecar: one dict per action with at least ``action``,
    ``position_x`` and ``torso_rotation``.
    """
    if end_reason not in END_REASONS:
        raise ValueError(f"end_reason {end_reason!r} is not one of {END_REASONS}")
    if bool(passed) != (end_reason == "success"):
        raise ValueError(
            f"round {round_number} of run {run}: passed={passed} with "
            f"end_reason={end_reason!r}; the two have to agree"
        )

    phase = phase_of(round_number)
    width = width_of(round_number)
    actions = [str(step["action"]) for step in steps]
    counts = collections.Counter(actions)

    first_turn_step: Optional[int] = None
    for index, action in enumerate(actions):
        if action in TURN_ACTIONS:
            first_turn_step = index
            break
    first_turn_x: Optional[float] = None
    if first_turn_step is not None:
        first_turn_x = float(steps[first_turn_step]["position_x"])

    n_lateral = sum(counts[action] for action in LATERAL_ACTIONS)
    n_turn = sum(counts[action] for action in TURN_ACTIONS)

    # gap is a learning-phase measure only: at A/S 1.10 the widest possible body
    # (0.6110 m) fits a 0.627 m opening, so the probe phase's gap is <= 0 for every
    # posture and carries no information (STAGE23_DESIGN.md 6.2).
    gap = gap_for(steps, width) if phase == PHASE_LEARNING else None

    return {
        "model": model,
        "run": int(run),
        "memory_mode": memory_mode(run),
        "phase": phase,
        "round": int(round_number),
        "a_s_ratio": width / SHOULDER_WIDTH_M,
        "channel_width": width,
        "passed": bool(passed),
        "total_steps": int(total_steps),
        "end_reason": end_reason,
        "action_sequence": ",".join(actions),
        "n_forward": counts["forward"],
        "n_backward": counts["backward"],
        "n_lateral": n_lateral,
        "n_turn": n_turn,
        "n_look_down": counts["look_down"],
        "n_glance": counts["look_left"] + counts["look_right"],
        "max_rotation_deg": float(max_rotation_deg),
        "passage_rotation_deg": (
            None if passage_rotation_deg is None else float(passage_rotation_deg)
        ),
        "first_turn_step": first_turn_step,
        "first_turn_x": first_turn_x,
        "final_x": float(final_x),
        "final_z": float(final_z),
        "wall_collision_count": int(wall_collision_count),
        "invalid_response_count": int(invalid_response_count),
        "reached_door": reached_door(steps),
        "optimal_steps": optimal_steps(width),
        "gap": gap,
        "excess": excess(total_steps, width),
        "strategy_label": strategy_label(n_lateral, n_turn, first_turn_x),
        "note_text": str(note_text or ""),
        "note_chars": len(str(note_text or "")),
        "memory_injected_chars": int(memory_injected_chars),
    }


# ---------------------------------------------------------------------------
# Note logs
# ---------------------------------------------------------------------------


def note_log_path(log_root: str, tag: str, run: int, round_number: int) -> str:
    """Where one note call's input and output are kept.

    Both halves go in one file with the prompt first, because the question a reader
    asks of it is "what did it write, given what it saw", and splitting them into
    two files makes that a join.
    """
    return os.path.join(log_root, tag, f"run{int(run):02d}_round{int(round_number):02d}_note.txt")


def write_note_log(path: str, prompt: str, response: str) -> None:
    """Write the note call's prompt and reply, atomically."""
    import persistence

    body = (
        "NOTE PROMPT:\n" + prompt + "\n" + "=" * 40 + "\nNOTE RESPONSE:\n" + (response or "")
    )
    if not body.endswith("\n"):
        body += "\n"
    persistence.atomic_write_text(path, body)
