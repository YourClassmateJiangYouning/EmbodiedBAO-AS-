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

FINAL DESIGN (2026-10-27): 5 models x 2 bodies x 10 cells = 100 episodes.
Per model and body: A/S 1.3 x3, 1.1 x3, 1.0 x2, 0.9 x2; every Stage 1 scene
exactly twice.  G1 keeps its five existing cells and adds five; R1 mirrors the
final ten G1 conditions.  A/S always uses the target robot's official width.
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
STAGE3_AS_VALUES: Tuple[float, ...] = (1.30, 1.10, 1.00, 0.90)
STAGE3_SCENE = "all five Stage 1 scenes"
STAGE3_EPISODES_PER_CELL = 1
STAGE3_EXPECTED = 100
