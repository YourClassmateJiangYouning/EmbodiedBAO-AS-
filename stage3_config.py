"""Stage 3 supplementary design: two smaller Unitree bodies, A/S 1.1-0.9.

Stage 3 keeps the Stage 1 scene, actions, prompt, step length and scoring, and
changes only the robot body.  The five models are the Stage 1 top five by
overall pass rate:

  1 deepseek-v4-flash-vision-exp  99.765%
  2 gemini-3-flash-preview        98.824%
  3 mimo-v2.6-pro                 86.118%
  4 claude-haiku-4-5-20251001     68.471%
  5 gpt-5                         68.471%  (tied with claude; both included)

gemini-2.5-flash is 67.294% and is sixth, so it is not in this five.

Design: 10 source conditions per model = 2 per Stage 1 scene, 5 assigned to G1
and 5 to R1.  5 models x 10 = 50 episodes.  A/S 1.1/1.0/0.9; the channel width
is always recalculated from the target robot's shoulder width.
"""
from __future__ import annotations

from typing import Tuple

STAGE3_MODELS: Tuple[str, ...] = (
    "gemini-3-flash-preview",
    "deepseek-v4-flash-vision-exp",
    "mimo-v2.6-pro",
    "claude-haiku-4-5-20251001",
    "gpt-5",
)

STAGE3_ROBOTS: Tuple[str, ...] = ("G1", "R1")
STAGE3_AS_VALUES: Tuple[float, ...] = (1.10, 1.00, 0.90)
STAGE3_SCENE = "all five Stage 1 scenes"
STAGE3_EPISODES_PER_CELL = 1
STAGE3_EXPECTED = 50
