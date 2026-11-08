"""Run the real-photograph replay experiment.

The corridor is not re-established for every decision: the photographs were captured once, renamed 1..11 by
distance, and the states are replayed against them.  What the agent sees is a real photograph; what happens to
it is the simulator's own geometry, applied without a renderer:

    W(theta) = S|cos theta| + D|sin theta|          the width the body occupies
    the opening admits it iff W(theta) <= A
    a forward move that would cross the wall plane while W(theta) > A is refused (blocked, position unchanged)
    a turn is refused once the body is close enough to the wall for the arc to sweep into a panel
    success is reaching x >= 8.75 m (one step past the wall), 30 steps maximum

Everything else -- prompt, model call, parsing, retries, recording, export -- is Stage4Session, so the data
lands in exactly the same layout as the operator-driven runs.

    python stage4/replay_runner.py --smoke          # one episode, verbose
    python stage4/replay_runner.py                  # the full 5 x 4 x 5 matrix, resumable
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import core  # noqa: E402

BANK = r"D:\stage4_photos"

# The five models Stage 3 used (the Stage-1 top five), unchanged.
MODELS = [
    "deepseek-v4-flash-vision-exp",
    "gemini-3-flash-preview",
    "mimo-v2.6-pro",
    "claude-haiku-4-5-20251001",
    "gpt-5",
]
LEVELS = [1.2, 1.0, 0.8, 0.6]
# Slot numbers follow core.MARKER_PRESETS, so the manifest's marker column means the same thing it did in
# Stages 1-3 (M1 is the green rectangle, not the red circle).
MARKER_ORDER = [
    "green rectangle",
    "red circle",
    "yellow arrow",
    "blue triangle",
    "purple four-pointed star",
]
MARKERS = [(descriptor, "M%d" % (MARKER_ORDER.index(descriptor) + 1)) for descriptor in MARKER_ORDER]

S = core.SIM_GEOMETRY["shoulder_width_m"]
D = core.SIM_GEOMETRY["torso_thickness_m"]
WALL_X = core.SIM_GEOMETRY["wall_x_m"]
START_X = core.SIM_GEOMETRY["start_x_m"]
SUCCESS_X = WALL_X + core.SIM_GEOMETRY["move_step_m"]
STEP = core.SIM_GEOMETRY["move_step_m"]
TURN_DEG = core.SIM_GEOMETRY["turn_step_deg"]
MAX_STEPS = core.SIM_GEOMETRY["max_steps"]

# A turn is refused when the swept rectangle reaches a wall PANEL.  The panels begin at |z| = A/2, and a rotated
# S x D rectangle projects W(theta) on both axes, so the body is clear of the panels exactly while W(theta) <= A;
# once W(theta) > A it reaches the wall plane when x + W(theta)/2 >= WALL_X.  (The earlier rule refused any turn
# within hypot(S,D)/2 of the wall regardless of the opening, which wrongly forbade turning at a wide gap.)
def turn_is_refused(x: float, theta: float, a_s: float, delta: float) -> bool:
    opening = a_s * S
    worst = max(width_at(theta), width_at(theta + delta))
    if worst <= opening + 1e-9:
        return False          # the body still fits through the opening, so the panels are never touched
    return x + worst / 2.0 >= WALL_X - 1e-9


def width_at(theta_deg: float) -> float:
    theta = math.radians(abs(theta_deg))
    return S * abs(math.cos(theta)) + D * abs(math.sin(theta))


# The feedback strings are the SIMULATOR'S OWN WORDS, because the history is part of the prompt.  A more
# helpful wording is a hint: "the opening is narrower than your body" tells the agent that turning would help,
# which is precisely the judgement the benchmark is trying to measure, and it measurably inflated rotation.
FEEDBACK_EXECUTED = "executed"
FEEDBACK_BLOCKED = "blocked by obstacle or room boundary (wall_panel)"
FEEDBACK_TURN_BLOCKED = "cannot turn: body would collide with obstacle or room boundary"
FEEDBACK_INVALID = "invalid action response"


def apply_action(action: str, x: float, theta: float, a_s: float):
    """One step of the geometry. Returns (x, theta, feedback, blocked, collision)."""
    opening = a_s * S
    if action == "forward":
        next_x = x + STEP
        # crossing the wall plane requires the body to fit at its current torso angle
        if x < WALL_X <= next_x and width_at(theta) > opening + 1e-9:
            return x, theta, FEEDBACK_BLOCKED, True, True
        return next_x, theta, FEEDBACK_EXECUTED, False, False
    if action == "backward":
        return max(START_X, x - STEP), theta, FEEDBACK_EXECUTED, False, False
    if action in ("turn_left", "turn_right"):
        delta = TURN_DEG if action == "turn_left" else -TURN_DEG
        if turn_is_refused(x, theta, a_s, delta):
            return x, theta, FEEDBACK_TURN_BLOCKED, True, False
        return x, (theta + delta) % 360.0, FEEDBACK_EXECUTED, False, False
    if action == "look_down":
        return x, theta, FEEDBACK_EXECUTED, False, False
    return x, theta, FEEDBACK_INVALID, True, False


def session_id_for(spec):
    """Stable directory name for one (model, A/S, marker) cell, used by main() and repair_replay."""
    return "r1-%s-%s-%s" % (
        spec["model"].split("/")[0].replace("-", "")[:20],
        ("%.1f" % spec["a_s"]).replace(".", ""),
        spec["marker"].split()[0])


def plan_matrix():
    episodes = []
    for model in MODELS:
        for level in LEVELS:
            for marker, slot in MARKERS:
                episodes.append({"model": model, "a_s": level, "marker": marker, "marker_slot": slot})
    return episodes


def run_episode(session: core.Stage4Session, spec: dict, verbose: bool = True) -> dict:
    """One closed-loop episode against the photograph bank."""
    session.config["a_s"] = spec["a_s"]
    session.config["channel_width_m"] = core.channel_width_m(spec["a_s"], session.config["robot_shoulder_m"])
    session.config["marker_descriptor"] = spec["marker"]
    session.config["marker"] = spec["marker_slot"]
    session.config["model"] = spec["model"]

    session.start_episode()
    x, theta = START_X, 0.0
    passed = False
    end_reason = "max_steps"

    for step in range(MAX_STEPS):
        try:
            session.capture_from_bank(BANK)
        except RuntimeError as error:
            print("    FRAME ERROR:", error)
            end_reason = "missing_frame"
            break

        state = session.call_model()
        record = session.history[-1]
        action = record.get("action")
        if action is None:
            # invalid reply: the step is recorded as invalid and costs a step, exactly as in Stage 1
            x, theta, feedback, blocked, collision = x, theta, FEEDBACK_INVALID, True, False
        else:
            x, theta, feedback, blocked, collision = apply_action(action, x, theta, spec["a_s"])
            session.position_x, session.torso_rotation = x, theta

        reached = x >= SUCCESS_X - 1e-9
        session.action_executed(
            feedback=feedback,
            collision=collision,
            position_x=x,
            torso_rotation=theta,
            note="replay",
            advance_odometer=False,   # the replay owns x and theta; see apply_action()
            # report the arrival here: action_executed closes a max-steps episode as failed, and without
            # this an episode that arrived on the very last step would be recorded a failure.
            passed=True if reached else None,
        )
        if verbose:
            print("    step %2d  %-11s  x=%5.2f theta=%3.0f  W=%.3f/%.3f  %s"
                  % (step, action, x, theta, width_at(theta), spec["a_s"] * S,
                     "blocked" if blocked else "ok"))
        if reached:
            passed = True
            end_reason = "passed"
            break
        if session.episode is None:      # action_executed finished the episode at the step limit
            break

    if session.episode is not None:
        session.finish(passed=passed, end_reason=end_reason)
    return {
        "episode_id": session.episodes[-1]["episode_id"] if session.episodes else None,
        "passed": passed,
        "steps": len(session.history),
        "final_x": round(x, 3),
        "final_theta": round(theta, 1),
        "end_reason": end_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one episode only")
    parser.add_argument("--limit", type=int, default=0, help="stop after N episodes")
    parser.add_argument("--out", default=None, help="bank of sessions root (default stage4/sessions-r1)")
    args = parser.parse_args()

    root = args.out or os.path.join(HERE, "sessions_r1")
    plan = plan_matrix()
    if args.smoke:
        plan = plan[:1]
    elif args.limit:
        plan = plan[:args.limit]

    print("bank     :", BANK)
    print("sessions :", root)
    print("episodes :", len(plan))
    print()

    results = []
    started = time.time()
    for index, spec in enumerate(plan, 1):
        session = core.Stage4Session(sessions_root=root)
        session_id = session_id_for(spec)
        session.open_session(session_id, {
            "scene": "R1", "marker": spec["marker_slot"], "marker_descriptor": spec["marker"],
            "mode": "A", "model": spec["model"], "a_s": spec["a_s"],
            "max_steps": MAX_STEPS, "disable_thinking": True,
            "shoulder_m": S, "move_step": STEP,
        })
        print("[%3d/%3d] %-32s A/S %.1f  %s" % (index, len(plan), spec["model"], spec["a_s"], spec["marker"]))
        try:
            result = run_episode(session, spec, verbose=args.smoke)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            result = {"passed": None, "error": "%s: %s" % (type(error).__name__, error)}
        results.append({"spec": spec, **result})
        print("      -> passed=%s steps=%s end=%s"
              % (result.get("passed"), result.get("steps"), result.get("end_reason", result.get("error"))))

    passed = sum(1 for r in results if r.get("passed") is True)
    print()
    print("=" * 70)
    print("episodes: %d   passed: %d   elapsed: %.1f min" % (len(results), passed, (time.time() - started) / 60))
    summary_path = os.path.join(root, "replay_summary.json")
    os.makedirs(root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
