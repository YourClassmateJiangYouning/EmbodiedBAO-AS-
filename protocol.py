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

from environment import ACTIONS

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

ACTION_DESCRIPTIONS: Dict[str, str] = {
    "forward": "move forward 5cm",
    "backward": "move backward 5cm",
    "left": "move left 5cm",
    "right": "move right 5cm",
    "turn_left": "rotate body 15 degrees counterclockwise",
    "turn_right": "rotate body 15 degrees clockwise",
    "look_left": "rotate head camera 30 degrees to the left",
    "look_right": "rotate head camera 30 degrees to the right",
}

# ``forward``/``backward``/``left``/``right`` are egocentric: they translate
# the robot along its own current facing direction, exactly like a human
# stepping.  This has to be stated because it is not inferable from the action
# names alone, and it is the same for every Level.
ACTION_FRAME_NOTE = (
    "Movement is egocentric: forward/backward move along the direction your "
    "torso currently faces, and left/right move along your own left/right. "
    "Turning rotates your torso in place and also rotates your head camera."
)

ACTION_OPTIONS_STRING: str = "\n".join(
    f'{{"action": "{name}"}} - {ACTION_DESCRIPTIONS[name]}' for name in ACTIONS
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

HISTORY_LIMIT = 6


def build_prompt(
    state: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, str]]] = None,
    max_steps: int = 30,
) -> str:
    """Build the per-step prompt.

    ``level`` is intentionally *not* a parameter: the prompt is identical for
    every Level, so there is no way for a caller to leak the channel geometry.
    """
    parts: List[str] = [TASK_INSTRUCTION]

    parts.append(
        "Available actions (each action is one discrete step):\n"
        + ACTION_OPTIONS_STRING
    )
    parts.append(ACTION_FRAME_NOTE)

    state = state or {}
    lines = ["Current state:"]
    position = state.get("position")
    if position is not None and len(position) >= 3:
        lines.append(
            f"- position (x, y, z): "
            f"[{float(position[0]):.3f}, {float(position[1]):.3f}, "
            f"{float(position[2]):.3f}]"
        )
    yaw = state.get("torso_rotation")
    if yaw is None:
        yaw = state.get("orientation", {}).get("yaw")
    if yaw is not None:
        lines.append(f"- torso rotation (degrees): {float(yaw):.1f}")
    lines.append(f"- step limit for this episode: {int(max_steps)}")
    parts.append("\n".join(lines))

    if history:
        lines = ["Action history (most recent first):"]
        for item in list(history)[-HISTORY_LIMIT:][::-1]:
            lines.append(f"- {item.get('action')} -> {item.get('feedback')}")
        parts.append("\n".join(lines))

    parts.append(RESPONSE_FORMAT_INSTRUCTION)
    return "\n\n".join(parts)
