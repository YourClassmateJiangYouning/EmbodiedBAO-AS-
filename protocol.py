"""Single source of truth for the EmbodiedBAO action space and prompt text.

Both ``experiments.py`` (the runner) and ``ai_agent.py`` (the model client)
import from here so the action list, the action descriptions, and the task
prompt can never drift apart.

Protocol note
-------------
Every Level 0-5 uses the *identical* prompt.  The only thing that changes
between Levels is the physical channel width, which the model is never told
about.  The prompt deliberately omits:

* the channel width,
* the robot's body dimensions,
* whether a sideways rotation is required,
* the A/S ratio.

This is what makes the ladder a measurement of the agent's own body-scale
affordance perception rather than a reading-comprehension test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from environment import ACTIONS, MOVE_STEP

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

# The descriptions must be DERIVED from the real constants.  They were once
# hard-coded as "move forward 5cm" while MOVE_STEP had risen to 0.20 m, so the
# prompt told the agent it moved 5 cm when it moved 20 -- a four-fold error in
# exactly the quantity this benchmark asks the agent to reason about.
_STEP_CM = MOVE_STEP * 100.0


def _format_step(centimetres: float) -> str:
    """Render a step length without a trailing .0 (5 not 5.0, but 7.5 stays)."""
    if abs(centimetres - round(centimetres)) < 1e-9:
        return str(int(round(centimetres)))
    return f"{centimetres:g}"


_STEP_TEXT = _format_step(_STEP_CM)

ACTION_DESCRIPTIONS: Dict[str, str] = {
    "forward": f"move forward {_STEP_TEXT}cm in the direction your torso faces",
    "backward": f"move backward {_STEP_TEXT}cm, opposite to the way your torso faces",
    "left": (
        f"move {_STEP_TEXT}cm straight to your own left, without changing "
        f"which way you face"
    ),
    "right": (
        f"move {_STEP_TEXT}cm straight to your own right, without changing "
        f"which way you face"
    ),
    "turn_left": (
        "rotate your torso 15 degrees to the left. Your head camera rotates "
        "with it, so the view turns 15 degrees left as well. This changes which "
        "way you face, and therefore which way forward moves you."
    ),
    "turn_right": (
        "rotate your torso 15 degrees to the right. Your head camera rotates "
        "with it, so the view turns 15 degrees right as well. This changes which "
        "way you face, and therefore which way forward moves you."
    ),
    "look_left": (
        "rotate only your head camera 30 degrees to the left. Your body stays "
        "exactly where it is and keeps facing the same way, and forward still "
        "moves you in the same direction. Use this to inspect the scene, not to "
        "travel."
    ),
    "look_right": (
        "rotate only your head camera 30 degrees to the right. Your body stays "
        "exactly where it is and keeps facing the same way, and forward still "
        "moves you in the same direction. Use this to inspect the scene, not to "
        "travel."
    ),
}

# ``forward``/``backward``/``left``/``right`` are egocentric: they translate
# the robot along its own current facing direction, exactly like a human
# stepping.  This has to be stated because it is not inferable from the action
# names alone, and it is the same for every Level.
#
# The two families are spelled out separately because conflating them is easy
# and costly.  ``turn_*`` changes the body and therefore the walking direction;
# ``look_*`` changes only the view.  The head offset left by ``look_*`` also
# PERSISTS rather than snapping back, and it rides along when the body turns --
# stated explicitly because it is not visible in any single frame, and a run
# that made three look_left calls while believing it faced forward would be
# judging its alignment from a view rotated 90 degrees.
ACTION_FRAME_NOTE = (
    "Movement is egocentric: forward/backward move along the direction your "
    "torso currently faces, and left/right move along your own left/right.\n"
    "turn_left/turn_right rotate your whole body; your head camera turns with "
    "it, so both your facing and your view change together.\n"
    "look_left/look_right rotate only your head camera. Your body and your "
    "walking direction are unaffected.\n"
    "A head-camera offset left by look_left/look_right persists, and it stays "
    "offset by the same amount when you later turn your body. Use the opposite "
    "look action if you want to face the same way as your body again."
)

ACTION_OPTIONS_STRING: str = "\n".join(
    f'{{"action": "{name}"}} - {ACTION_DESCRIPTIONS[name]}' for name in ACTIONS
)


def action_options_string(move_step: float = MOVE_STEP) -> str:
    """Render action descriptions for the environment's actual move step."""
    step_text = _format_step(float(move_step) * 100.0)
    descriptions = dict(ACTION_DESCRIPTIONS)
    for name, direction in (
        ("forward", "forward"),
        ("backward", "backward"),
        ("left", "left"),
        ("right", "right"),
    ):
        descriptions[name] = f"move {direction} {step_text}cm"
    return "\n".join(
        f'{{"action": "{name}"}} - {descriptions[name]}' for name in ACTIONS
    )

# Compact comma-separated list used by the JSON response instruction.
ACTION_NAMES_TEXT: str = ", ".join(ACTIONS)

# ---------------------------------------------------------------------------
# The unified task prompt
# ---------------------------------------------------------------------------

TASK_INSTRUCTION = (
    "You are a Unitree H1 humanoid robot. Your task is to pass through the "
    "opening in the wall."
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
    # The head-camera offset is part of where the agent is looking, and nothing
    # in a single frame reveals it.  Without this line an agent that has called
    # look_left three times is judging its alignment from a view rotated 90
    # degrees while the prompt says nothing about it.  The value is generated by
    # the environment already (get_robot_state returns "camera_yaw"); this only
    # puts it in front of the model.
    camera_yaw = state.get("camera_yaw")
    if camera_yaw is not None:
        lines.append(
            f"- head camera offset from your torso (degrees): "
            f"{float(camera_yaw):.1f} (0 = camera aimed the same way as your "
            f"torso; positive = turned left)"
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
