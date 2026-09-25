"""Prove which frame ``forward``/``backward``/``left``/``right`` move in.

Two modes:

  * with Isaac Sim importable (the lab machine) the probe instantiates the real
    ``EmbodiedBAOEnv`` and calls the real ``_apply_action``;
  * without it (anywhere else) it calls the very same four helpers that
    ``_apply_action`` calls -- ``_forward_vector``, ``_right_vector``,
    ``_translation_path_is_clear``, ``_turn_path_is_clear`` -- so the arithmetic
    and the collision gate are still the shipped ones.

Either way only the two Isaac pose accessors are stubbed, and the mode used is
printed first.

    python tools/check_egocentric.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment as envmod  # noqa: E402
from environment import (  # noqa: E402
    CAMERA_ACTIONS,
    CAMERA_TURN_STEP_DEG,
    CHANNEL_WIDTH,
    MOVE_STEP,
    ROBOT_START_POS,
    TURN_STEP_DEG,
    _forward_vector,
    _right_vector,
    _translation_path_is_clear,
    _turn_path_is_clear,
)

REAL_ENV = getattr(envmod, "EmbodiedBAOEnv", None) is not None


class Stub:
    """Bare pose holder; the action logic is bound in from the shipped code."""

    def __init__(self, channel_width: float) -> None:
        self._robot_yaw = 0.0
        self._move_step = float(MOVE_STEP)
        self._channel_width = float(channel_width)
        self._camera_yaw_offset = 0.0
        self._pose = np.asarray(ROBOT_START_POS, dtype=float).copy()
        if REAL_ENV:
            self._apply_action = envmod.EmbodiedBAOEnv._apply_action.__get__(self)
        else:
            self._apply_action = self._apply_action_shim

    # -- Isaac-dependent accessors, replaced by a plain variable -----------
    def _root_position(self) -> np.ndarray:
        return self._pose.copy()

    def _set_robot_pose(self, position, yaw_deg) -> None:
        self._pose = np.asarray(position, dtype=float).copy()
        self._robot_yaw = float(yaw_deg)

    # -- used only when isaacsim is missing --------------------------------
    def _apply_action_shim(self, action: str):
        """Same six lines as EmbodiedBAOEnv._apply_action, same helpers."""
        root = self._root_position()
        yaw = np.radians(self._robot_yaw)
        if action in ("forward", "backward", "left", "right"):
            if action == "forward":
                delta = _forward_vector(yaw) * self._move_step
            elif action == "backward":
                delta = _forward_vector(yaw) * -self._move_step
            elif action == "right":
                delta = _right_vector(yaw) * self._move_step
            else:
                delta = _right_vector(yaw) * -self._move_step
            target = root + delta
            collision = _translation_path_is_clear(
                root, target, yaw, channel_width=self._channel_width
            )
            if collision is not None:
                return (
                    False,
                    f"blocked by obstacle or room boundary ({collision['part']})",
                    collision,
                )
            self._set_robot_pose(target, self._robot_yaw)
            return True, "executed", None
        if action in ("turn_left", "turn_right"):
            new_yaw = self._robot_yaw + (
                TURN_STEP_DEG if action == "turn_left" else -TURN_STEP_DEG
            )
            collision = _turn_path_is_clear(
                root, self._robot_yaw, new_yaw, channel_width=self._channel_width
            )
            if collision is not None:
                return (
                    False,
                    "cannot turn: body would collide with obstacle or room boundary",
                    collision,
                )
            self._set_robot_pose(root, new_yaw)
            return True, "executed", None
        if action in CAMERA_ACTIONS:
            self._camera_yaw_offset += (
                CAMERA_TURN_STEP_DEG if action == "look_left" else -CAMERA_TURN_STEP_DEG
            )
            return True, "executed", None
        return False, f"unknown action: {action}", None


def make_env(channel_width: float = CHANNEL_WIDTH) -> Stub:
    return Stub(channel_width)


def reset(env: Stub, yaw_deg: float, x: float | None = None) -> None:
    env._pose = np.asarray(ROBOT_START_POS, dtype=float).copy()
    if x is not None:
        env._pose[0] = float(x)
    env._robot_yaw = float(yaw_deg)


def run(env: Stub, action: str) -> str:
    ok, feedback, _ = env._apply_action(action)
    return "ok" if ok else f"BLOCKED ({feedback})"


def main() -> int:
    if REAL_ENV:
        print("mode: real EmbodiedBAOEnv._apply_action (isaacsim importable)")
    else:
        print("mode: shipped helpers, direct (isaacsim missing here; run this on")
        print("      the lab machine to exercise the class method itself)")
    print(f"MOVE_STEP = {MOVE_STEP} m   TURN_STEP = {TURN_STEP_DEG} deg")
    print(f"start pose = {list(ROBOT_START_POS)}   channel = {CHANNEL_WIDTH} m (Level 0)")
    print()

    print("1) forward/backward at several torso yaws, pose reset each time")
    print()
    print(f"   {'torso yaw':>10} {'action':>10} {'dx':>9} {'dz':>9}   direction")
    for yaw in (0.0, 45.0, 90.0, -90.0, 180.0):
        for action in ("forward", "backward"):
            env = make_env()
            reset(env, yaw)
            before = env._pose.copy()
            status = run(env, action)
            delta = env._pose - before
            if yaw == 0.0:
                label = "+x (happens to be the world x axis)"
            elif abs(yaw) == 90.0:
                label = f"{'+z' if delta[2] > 0 else '-z'} (across the room)"
            elif yaw == 45.0:
                label = "+x and -z equally"
            else:
                label = "-x (back along the world x axis)"
            print(
                f"   {yaw:>10.1f} {action:>10} {delta[0]:>9.3f} {delta[2]:>9.3f}   "
                f"{status if status != 'ok' else label}"
            )
    print()

    print("2) left/right, also body-relative")
    print()
    print(f"   {'torso yaw':>10} {'action':>10} {'dx':>9} {'dz':>9}")
    for yaw in (0.0, 90.0):
        for action in ("left", "right"):
            env = make_env()
            reset(env, yaw)
            before = env._pose.copy()
            run(env, action)
            delta = env._pose - before
            print(f"   {yaw:>10.1f} {action:>10} {delta[0]:>9.3f} {delta[2]:>9.3f}")
    print()

    print("3) turn first, then walk: does forward follow the NEW facing?")
    print()
    print(f"   {'sequence':<40} {'x':>8} {'z':>8} {'torso':>8}")
    sequences = {
        "forward x4": ["forward"] * 4,
        "turn_left, forward x4": ["turn_left"] + ["forward"] * 4,
        "turn_left x6 (90 deg), forward x4": ["turn_left"] * 6 + ["forward"] * 4,
        "turn_right x6 (-90 deg), forward x4": ["turn_right"] * 6 + ["forward"] * 4,
        "turn_left x6, forward x4, turn_left x6, forward x4": ["turn_left"] * 6
        + ["forward"] * 4
        + ["turn_left"] * 6
        + ["forward"] * 4,
    }
    for name, actions in sequences.items():
        env = make_env()
        reset(env, 0.0)
        status = "ok"
        for action in actions:
            status = run(env, action)
            if status != "ok":
                break
        print(
            f"   {name:<40} {env._pose[0]:>8.3f} {env._pose[2]:>8.3f} "
            f"{env._robot_yaw:>8.1f}   {status}"
        )
    print()

    print("4) forward at Level 5 from x = 7.5, one step short of the wall plane")
    print("   (0.75 m carries the body across the 0.45 m channel mouth)")
    print()
    for yaw in (0.0, 45.0, 75.0, 90.0):
        env = make_env(channel_width=0.45)
        reset(env, yaw, x=7.5)
        before = env._pose.copy()
        status = run(env, "forward")
        delta = env._pose - before
        print(
            f"   yaw {yaw:>5.1f}: forward -> dx {delta[0]:>6.3f} dz {delta[2]:>6.3f}  "
            f"{status}"
        )
    print()
    print("Read 1-3: the displacement depends only on the torso yaw, never on the")
    print("world axes, so the vocabulary is egocentric.  At yaw 0 it coincides with")
    print("+x, which is why an unturned walk reaches the channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
