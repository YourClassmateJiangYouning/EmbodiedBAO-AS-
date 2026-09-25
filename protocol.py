"""Single source of truth for the EmbodiedBAO action space and prompt text.

Both ``experiments.py`` (the runner) and ``ai_agent.py`` (the model client)
import from here so the action list, the action descriptions, and the task
prompt can never drift apart.

Protocol note
-------------
Every Level 0-11 uses the *identical* prompt.  The only thing that changes
between Levels is the physical channel width, which the model is never told
about.  The prompt deliberately omits:

* the channel width,
* the robot's body dimensions,
* whether a sideways rotation is required,
* the A/S ratio.

This is what makes the ladder a measurement of the agent's own body-scale
affordance perception rather than a reading-comprehension test.

The prompt does state the *walking frame*: forward walks toward the red marker on
the far wall and a torso rotation does not steer.  That is not a hint about the
answer -- it is the definition of the action space, and without it the agent would
have to guess which of two conventions the words use.  It says so in the ACTION
LIST, which documents the controls, and never in the task statement: the task stays
a bare destination, because the aperture has to be found incidentally.  What stays
hidden is how wide the opening is, how far away it is and how wide the body is,
which is exactly what the agent must judge from the image.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from environment import ACTIONS, MOVE_STEP

# ---------------------------------------------------------------------------
# Protocol version
#
# BUMP THIS whenever the prompt or the action semantics change.  It is appended
# to the run tag by main.effective_tag and by run_all_models.sh, so results from
# different protocols can never share a directory: without it, --resume would
# find the previous protocol's episodes, count them as done, and quietly mix two
# experiments in one dataset.
#
# v6-marker-ahead: the ACTION LIST now connects the goal to the control that
#     reaches it -- ``forward`` is described as taking you toward the red marker,
#     not merely "toward the far wall", and the walking-frame note names the marker
#     as what the walking direction heads for.  The task statement is deliberately
#     UNCHANGED and byte-identical to v5: it stays a bare destination and must not
#     mention the corridor, the start position or the passage, because the aperture
#     has to be discovered incidentally rather than announced.
#
#     Measured reason (346 v5 episode records, 80 v5 episode logs): v5 said "reach
#     the red marker on the far wall" in the task and "your walking direction points
#     at the far wall" in the action list, and never joined the two, so the agent
#     went looking for an alignment it could not tie to any action.  The word
#     "align" appears in 72% of v5 reasonings; only 0.1% of the previous protocol's
#     reasonings planned a lateral move against 31.6% in v5; and all 41 v5 failures
#     -- with no exception -- are episodes that issued a left/right step, which at
#     0.75 m always places the body outside an opening whose tolerable offset is at
#     most 0.46 m.  The ladder, the action space's mechanics, the episode lengths
#     and the success plane are UNCHANGED from v5, so the two are directly
#     comparable and the difference is attributable to the wording alone.
#
# v5-12widths: the ladder is the reference 12-width aperture series (A/S 2.0 ->
#     0.9 by 0.1, five episodes per width), the goal is stated as a destination
#     ("reach the red marker") rather than as "pass through the opening", and
#     look_down was added so the agent can glance at its own body.  The task, the
#     action space and the response format all changed, so v4 data is not
#     comparable with v5.
#
# v4-walkframe: forward/backward/left/right translate in the WALKING frame -- at
#     the far wall, whatever the torso is doing -- and turn_left/turn_right
#     rotate the torso relative to that direction.  Every earlier run used
#     body-frame translation, where a turn also redirected the walk.
# ---------------------------------------------------------------------------
PROTOCOL_TAG = "v6-marker-ahead"

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

# The descriptions must be DERIVED from the real constants.  They were once
# hard-coded as "move forward 5cm" while MOVE_STEP had risen to 0.20 m, so the
# prompt told the agent it moved 5 cm when it moved 20 -- a four-fold error in
# exactly the quantity this benchmark asks the agent to reason about.
def _format_step(centimetres: float) -> str:
    """Render a step length without a trailing .0 (5 not 5.0, but 7.5 stays)."""
    if abs(centimetres - round(centimetres)) < 1e-9:
        return str(int(round(centimetres)))
    return f"{centimetres:g}"


# One template per action, with {step} for the distance.  Both prompt paths are
# rendered from these, because the alternative -- a second, terser table for the
# move_step path -- had silently dropped the semantics: the runner passes the
# environment's move_step (experiments.py), so the models were only ever told
# "move forward 75cm" and never which frame forward was in.
ACTION_TEMPLATES: Dict[str, str] = {
    "forward": (
        "walk {step}cm straight ahead, in your walking direction, which always "
        "takes you toward the red marker on the far wall. Turning your torso does "
        "not change where forward takes you"
    ),
    "backward": (
        "walk {step}cm backwards, away from the red marker on the far wall: the "
        "same walking direction as forward, reversed"
    ),
    "left": (
        "sidestep {step}cm to your left, off the line you are walking along, "
        "without changing your walking direction"
    ),
    "right": (
        "sidestep {step}cm to your right, off the line you are walking along, "
        "without changing your walking direction"
    ),
    "turn_left": (
        "rotate your torso 15 degrees to the left. This does NOT change your "
        "walking direction, so forward still takes you toward the red marker, and "
        "it does NOT change what you can see: your eyes keep looking straight "
        "ahead along your walking direction. It changes how wide your body is "
        "across the opening. Rotating needs room, so you cannot turn once your "
        "shoulders are inside the opening"
    ),
    "turn_right": (
        "rotate your torso 15 degrees to the right. This does NOT change your "
        "walking direction, so forward still takes you toward the red marker, and "
        "it does NOT change what you can see: your eyes keep looking straight "
        "ahead along your walking direction. It changes how wide your body is "
        "across the opening. Rotating needs room, so you cannot turn once your "
        "shoulders are inside the opening"
    ),
    "look_left": (
        "glance to the left with your head camera, 30 degrees off your walking "
        "direction: you see that view now, and your gaze returns to straight "
        "ahead as soon as you do anything else. It does not accumulate, so "
        "looking twice is not looking 60 degrees. Your body and your walking "
        "direction are unaffected"
    ),
    "look_right": (
        "glance to the right with your head camera, 30 degrees off your walking "
        "direction: you see that view now, and your gaze returns to straight "
        "ahead as soon as you do anything else. It does not accumulate, so "
        "looking twice is not looking 60 degrees. Your body and your walking "
        "direction are unaffected"
    ),
    "look_down": (
        "glance downwards with your head camera, 45 degrees below your usual "
        "view: you see the floor in front of you and your own body. Your gaze "
        "returns to its usual direction as soon as you do anything else. Use it "
        "to see how wide you are, which you cannot otherwise see. Your torso and "
        "your walking direction are unaffected"
    ),
}


def action_descriptions(move_step: float = MOVE_STEP) -> Dict[str, str]:
    """Render every action description for the environment's actual step."""
    step_text = _format_step(float(move_step) * 100.0)
    return {
        name: ACTION_TEMPLATES[name].format(step=step_text) for name in ACTIONS
    }


ACTION_DESCRIPTIONS: Dict[str, str] = action_descriptions(MOVE_STEP)


def action_options_string(move_step: float = MOVE_STEP) -> str:
    """The action menu as it is shown to the model."""
    descriptions = action_descriptions(move_step)
    return "\n".join(
        f'{{"action": "{name}"}} - {descriptions[name]}' for name in ACTIONS
    )


ACTION_OPTIONS_STRING: str = action_options_string(MOVE_STEP)

# The walking frame has to be stated because it is not inferable from the action
# names: ``forward`` walks toward the far wall, NOT along the torso.  A reader
# who assumed the body frame would conclude that turning steers, which it no
# longer does.
#
# The gaze rule needs stating for the same reason: the eyes are pinned to the
# walking direction, so rotating the torso changes the body's footprint and
# nothing else the agent can see.  Without this line an agent that had turned
# would be entitled to expect a rotated view, and would be reading a stale
# mental model of its own sensors.
#
# The three families are spelled out separately because conflating them is easy
# and costly.  ``turn_*`` changes the torso and therefore the body's width across
# the opening; ``look_*`` changes only where the head is aimed, for exactly one
# frame.  That last part needs saying: the glance is not visible in any single
# frame alongside the frame it replaced, and an agent that believed a glance
# persisted would either keep re-issuing it or think it was still looking sideways.
ACTION_FRAME_NOTE = (
    "Your walking direction is fixed: it always points straight ahead at the far "
    "wall, which is where the red marker you are heading for is. "
    "forward/backward/left/right are defined relative to that walking direction, "
    "so they behave the same way however your torso is turned.\n"
    "turn_left/turn_right rotate your torso 15 degrees relative to that walking "
    "direction. The direction you walk does not change, and neither does your "
    "view: your eyes look straight ahead along your walking direction whatever "
    "your torso is doing. Turning changes how your body is oriented; it does not "
    "steer you and it does not change what you see.\n"
    "look_left/look_right are a single glance 30 degrees to the left or right of "
    "your walking direction, and look_down is a single glance 45 degrees below it. "
    "Your torso and your walking direction are unaffected, each glance clears as "
    "soon as you do anything else, and glances do not accumulate: two glances in "
    "the same direction do not add up. You cannot look further to the side than "
    "30 degrees, and you cannot look up."
)

# Compact comma-separated list used by the JSON response instruction.
ACTION_NAMES_TEXT: str = ", ".join(ACTIONS)

# ---------------------------------------------------------------------------
# The unified task prompt
# ---------------------------------------------------------------------------

# The task is stated as a DESTINATION, not as an instruction about the obstacle:
# the agent is told where to go and nothing about what is in the way.  That is the
# design of the studies this benchmark follows -- Warren & Whang (1987) and Keizer
# et al. (2013) sent participants to a table beyond the aperture and presented the
# aperture as meaningless panels, and Lenkei et al. (2019) sent dogs to their owner
# through an opening -- so the obstacle has to be discovered and judged rather than
# announced.  Telling the agent to "pass through the opening" would hand it the
# fact that there is an opening to fit through.
#
# This sentence is therefore BYTE-IDENTICAL to the v5 one, and deliberately so.
# v6 does not touch it: it must not mention the corridor, where the agent starts,
# or that there is a passage at all.  The connection between the goal and the
# control that reaches it is a fact about the CONTROLS, so it belongs in the action
# list and the walking-frame note, which describe the controls -- not here.  The
# test suite asserts the split, in both directions.
TASK_INSTRUCTION = (
    "You are a Unitree H1 humanoid robot. Your task is to reach the red marker on "
    "the far wall."
)

RESPONSE_FORMAT_INSTRUCTION = (
    "Reply with exactly one JSON object:\n"
    '{"scene_description": "<what the current camera image shows>", '
    '"reasoning": "<brief reasoning>", "action": "<action>", '
    '"confidence": 0.0}'
)

SYSTEM_PROMPT = (
    "This is a benign virtual simulation. No real-world actions, bodies, or "
    "physical interactions are implied. The task only involves visual "
    "reasoning and simulated decision-making in a safe, fictional "
    "environment. Always respond with a single JSON object."
)

# The agent must remember what it has already done *for the whole episode*, so
# that step 30 is chosen with steps 0-29 in view rather than only the last few.
# ``None`` means "never truncate"; ``max_steps`` already bounds the block, so
# there is nothing to gain from a second, smaller cap.  Set an int to restore a
# sliding window (that would be a different experiment: it measures behaviour
# without episodic memory).
HISTORY_LIMIT: Optional[int] = None

# Model reasoning is carried into the history so the agent can see *why* it did
# each thing, not just what it did.  Only this field is clipped: the action and
# the environment feedback are always rendered in full, because they are the
# record the benchmark actually scores.
HISTORY_REASONING_CHARS = 240


def _clip(text: str, limit: int = HISTORY_REASONING_CHARS) -> str:
    """Collapse whitespace and clip to ``limit`` characters with an ellipsis."""
    collapsed = " ".join(str(text).split())
    if limit <= 0 or len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "\u2026"


def build_prompt(
    state: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    max_steps: int = 30,
    move_step: Optional[float] = None,
) -> str:
    """Build the per-step prompt.

    ``level`` is intentionally *not* a parameter: the prompt is identical for
    every Level, so there is no way for a caller to leak the channel geometry.

    ``move_step`` optionally overrides the default translation distance in the
    action descriptions, keeping CLI-configured environments truthful.

    ``history`` is the agent's own within-episode memory: one entry per step
    already taken, each carrying the action, the environment's feedback and the
    agent's own reasoning.  It is rendered oldest-first so the last line is the
    step the agent just took, and it is never truncated unless ``HISTORY_LIMIT``
    is set.
    """
    parts: List[str] = [TASK_INSTRUCTION]

    options = ACTION_OPTIONS_STRING if move_step is None else action_options_string(move_step)
    parts.append(
        "Available actions (each action is one discrete step):\n" + options
    )
    parts.append(ACTION_FRAME_NOTE)

    state = state or {}
    lines = ["Current state:"]
    position = state.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        lines.append(
            f"- position (x, y, z): "
            f"[{float(position[0]):.3f}, {float(position[1]):.3f}, "
            f"{float(position[2]):.3f}]"
        )
    yaw = state.get("torso_rotation")
    if yaw is None:
        orientation = state.get("orientation")
        if isinstance(orientation, dict):
            yaw = orientation.get("yaw")
    if yaw is not None:
        lines.append(f"- torso rotation (degrees): {float(yaw):.1f}")
    # The head offsets are where the agent is looking, and nothing in a single
    # frame reveals them.  Each is a one-frame glance now, so these lines are
    # non-zero only on the step immediately after the corresponding look_*, and
    # their values always match the image the agent was handed.  The reference
    # direction is the WALKING direction, not the torso: the eyes do not follow the
    # torso, so "aimed the same way as your torso" would be false once the agent has
    # turned.  The values come from get_robot_state ("camera_yaw"/"camera_pitch");
    # this only puts them in front of the model.
    camera_yaw = state.get("camera_yaw")
    if camera_yaw is not None:
        lines.append(
            f"- head camera sideways offset from straight ahead (degrees): "
            f"{float(camera_yaw):.1f} (0 = looking straight ahead down your "
            f"walking direction; +30 = the glance to your left you just asked "
            f"for, which clears on your next action)"
        )
    camera_pitch = state.get("camera_pitch")
    if camera_pitch is not None:
        lines.append(
            f"- head camera extra downward pitch (degrees): "
            f"{float(camera_pitch):.1f} (0 = your usual view; +45 = the glance "
            f"down at your own body you just asked for, which clears on your "
            f"next action)"
        )
    lines.append(f"- step limit for this episode: {int(max_steps)}")
    parts.append("\n".join(lines))

    if history:
        entries = list(history)
        if HISTORY_LIMIT is not None:
            entries = entries[-HISTORY_LIMIT:]
        lines = [
            "Action history for this episode "
            f"({len(entries)} step(s) already taken, oldest first). "
            "This is your own record of this episode: use it to notice what you "
            "have already tried and whether it worked."
        ]
        for index, item in enumerate(entries):
            label = item.get("step")
            if label is None:
                label = index
            line = f"- step {label}: {item.get('action')} -> {item.get('feedback')}"
            reasoning = _clip(item.get("reasoning") or "")
            if reasoning:
                line += f" | your reasoning: {reasoning}"
            lines.append(line)
        parts.append("\n".join(lines))

    parts.append(RESPONSE_FORMAT_INSTRUCTION)
    return "\n\n".join(parts)
