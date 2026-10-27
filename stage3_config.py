"""Stage 3 supplementary design: two smaller Unitree bodies, A/S 1.1-0.9.

Stage 3 keeps the Stage 1 scene, actions, prompt and scoring, and changes only
the robot body.  The five models are the Stage 1 top five by Sfinal/NPS:

  1 gemini-3-flash-preview
  2 deepseek-v4-flash-vision-exp
  3 mimo-v2.6-pro
  4 claude-haiku-4-5-20251001
  5 gemini-2.5-flash

Design: 5 models x 5 A/S values x 2 robots = 50 episodes, one episode per cell,
in the Stage 1.1 scene.  A/S is channel_width / shoulder_width for that body.
"""
from __future__ import annotations

from typing import Tuple

STAGE3_MODELS: Tuple[str, ...] = (
    "gemini-3-flash-preview",
    "deepseek-v4-flash-vision-exp",
    "mimo-v2.6-pro",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
)

STAGE3_ROBOTS: Tuple[str, ...] = ("G1", "R1")
STAGE3_AS_VALUES: Tuple[float, ...] = (1.10, 1.05, 1.00, 0.95, 0.90)
STAGE3_SCENE = "stage1.1"
STAGE3_EPISODES_PER_CELL = 1
STAGE3_EXPECTED = len(STAGE3_MODELS) * len(STAGE3_ROBOTS) * len(STAGE3_AS_VALUES)
