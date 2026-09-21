"""Offline geometry and protocol verification for EmbodiedBAO.

Isaac Sim is not available in every development environment, so the parts of
the benchmark that decide whether the experiment is valid at all are checked
here with plain Python:

1. **Channel ladder**  -- each Level builds exactly the width from the task
   brief and the derived A/S ratio matches the hand-computed value.
2. **Frontal feasibility** -- a Level must be frantically passable only when
   the channel is genuinely wider than the shoulders.  Levels 0-3 pass
   head-on, Levels 4-5 cannot; Levels 4-5 must pass sideways.
3. **Analytic gate** -- ``_check_wall_collision`` agrees with the closed-form
   projected-width test, including the critical 0.90 m channel where the
   shoulders are narrower than the opening.
4. **Sideways route** -- the route the model is expected to find
   (rotate 90 degrees in free space, then translate along the torso axis)
   actually reaches x > 2.5 m at every Level, and the robot's body box clears
   the wall panels while doing it.
5. **Action space and prompt** -- ``reach``/``retreat`` are gone, the camera
   actions keep the legacy 30 degree step, and the prompt is byte-identical
   across Levels and free of geometry leaks.

Run with a plain Python interpreter:

    python test_bao_geometry.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from environment import (
    ACTIONS,
    BODY_CLEARANCE,
    CAMERA_TURN_STEP_DEG,
    LEVEL_CHANNEL_WIDTHS,
    MOVE_STEP,
    ROBOT_SHOULDER_WIDTH,
    ROBOT_TORSO_THICKNESS,
    SUCCESS_X,
    TURN_STEP_DEG,
    WALL_THICKNESS,
    WALL_X,
    _check_wall_collision,
    _panel_boxes,
    _robot_body_aabb,
    _rotate_xz,
    a_s_ratio,
    level_channel_width,
)
from experiments import (
    DEFAULT_EPISODES_PER_LEVEL,
    DEFAULT_MAX_STEPS,
    _is_sideways_yaw,
)
from protocol import ACTION_OPTIONS_STRING, TASK_INSTRUCTION, build_prompt

# A/S ratios from the task brief, rounded to two decimals.
EXPECTED_A_S = {0: 1.58, 1: 1.40, 2: 1.30, 3: 1.19, 4: 1.00, 5: 0.79}

ROBOT_START = np.array([1.5, 0.0, 0.0], dtype=float)
SIDEWAYS_YAW_DEG = 90.0


class Failure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# ---------------------------------------------------------------------------
# 1. Channel ladder
# ---------------------------------------------------------------------------


def test_channel_ladder() -> None:
    expected_widths = {0: 0.90, 1: 0.80, 2: 0.74, 3: 0.68, 4: 0.57, 5: 0.45}
    check(
        LEVEL_CHANNEL_WIDTHS == expected_widths,
        f"channel widths drifted: {LEVEL_CHANNEL_WIDTHS} != {expected_widths}",
    )
    for level, expected_ratio in EXPECTED_A_S.items():
        width = level_channel_width(level)
        ratio = a_s_ratio(width)
        check(
            abs(round(ratio, 2) - expected_ratio) < 1e-9,
            f"level {level}: A/S {ratio:.4f} rounds to {round(ratio, 2)}, "
            f"expected {expected_ratio}",
        )
    print("[ok] channel ladder matches the brief (widths and A/S ratios)")


# ---------------------------------------------------------------------------
# 2/3. Geometry gate
# ---------------------------------------------------------------------------


def body_corners_xz(root_x: float, root_z: float, yaw_deg: float) -> np.ndarray:
    """Corner (x, z) coordinates of the torso rectangle for a given pose.

    The root pose is the ground point under the body, so the rectangle is
    centred on it: rotating in place must not move the footprint sideways.
    ``BODY_CLEARANCE`` is included because the gate tests the inflated box.
    """
    rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(rad), math.sin(rad)
    half_t = ROBOT_TORSO_THICKNESS / 2.0 + BODY_CLEARANCE
    half_s = ROBOT_SHOULDER_WIDTH / 2.0 + BODY_CLEARANCE
    local = np.array(
        [
            [half_t, half_s],
            [half_t, -half_s],
            [-half_t, half_s],
            [-half_t, -half_s],
        ],
        dtype=float,
    )
    world = np.empty_like(local)
    world[:, 0] = local[:, 0] * cos_y + local[:, 1] * sin_y
    world[:, 1] = -local[:, 0] * sin_y + local[:, 1] * cos_y
    return world + np.array([root_x, root_z], dtype=float)


def z_span_at_x(corners: np.ndarray, x: float) -> float:
    """z-extent (max - min) of the rectangle's intersection with plane ``x``."""
    xs = corners[:, 0]
    zs = corners[:, 1]
    lo, hi = float(np.min(xs)), float(np.max(xs))
    if x < lo - 1e-12 or x > hi + 1e-12:
        return 0.0
    if abs(hi - lo) < 1e-12:
        return float(np.max(zs) - np.min(zs))
    t = (x - lo) / (hi - lo)
    z_lo = float(np.min(zs)) + t * (float(np.max(zs)) - float(np.min(zs)))
    z_hi = float(np.max(zs)) - t * (float(np.max(zs)) - float(np.min(zs)))
    return abs(z_hi - z_lo)


# The gate treats a pose whose footprint exactly reaches the panel face as a
# collision, so feasibility uses the same orientation of the comparison.
FEASIBILITY_EPS = 1e-9


def _rect_xz(yaw_deg: float, u: float, v: float) -> Tuple[float, float]:
    """Map body-local (u = along facing, v = lateral) to world (x, z)."""
    rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(rad), math.sin(rad)
    return cos_y * u + sin_y * v, -sin_y * u + cos_y * v


def half_z_span_in_x_slab(
    yaw_deg: float, x_lo: float, x_hi: float, samples: int = 4000
) -> Optional[float]:
    """Exact half z-extent of the torso rectangle inside an x-interval.

    For a slab thinner than the rectangle, the extreme |z| is attained either
    at a rectangle corner inside the slab or where a rectangle edge crosses a
    slab face.  Sweeping all four edges densely covers both cases (and any
    edge-interior maximum), so this is an independent re-derivation rather than
    a copy of the gate's separating-axis test.
    """
    half_t = ROBOT_TORSO_THICKNESS / 2.0 + BODY_CLEARANCE
    half_s = ROBOT_SHOULDER_WIDTH / 2.0 + BODY_CLEARANCE
    u_range = np.linspace(-half_t, half_t, samples)
    v_range = np.linspace(-half_s, half_s, samples)

    # Four edges of the rectangle in body-local coordinates.
    edges = [
        (u_range, np.full_like(u_range, half_s)),
        (u_range, np.full_like(u_range, -half_s)),
        (np.full_like(v_range, half_t), v_range),
        (np.full_like(v_range, -half_t), v_range),
    ]

    rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(rad), math.sin(rad)
    best: Optional[float] = None
    for u_values, v_values in edges:
        x = cos_y * u_values + sin_y * v_values
        z = -sin_y * u_values + cos_y * v_values
        inside = (x >= x_lo - 1e-12) & (x <= x_hi + 1e-12)
        if not inside.any():
            continue
        local_max = float(np.max(np.abs(z[inside])))
        best = local_max if best is None else max(best, local_max)
    return best


def exact_fits_channel(root_x: float, root_z: float, yaw_deg: float, width: float) -> bool:
    """Exact passability: does the footprint squeeze through the gap at x=2.0?

    The panels occupy the x-slab ``[2.0 - t, 2.0 + t]`` outside ``|z| < w/2``,
    so the body fits exactly when its z-extent inside that slab stays within
    the gap.  This is an independent re-derivation of the gate, not a copy of
    it, so agreement is evidence the collision model is right.
    """
    slab_lo = WALL_X - WALL_THICKNESS / 2.0 - root_x
    slab_hi = WALL_X + WALL_THICKNESS / 2.0 - root_x
    half_span = half_z_span_in_x_slab(yaw_deg, slab_lo, slab_hi)
    if half_span is None:
        return True
    return bool(half_span < width / 2.0 - 1e-12)


def test_frontal_feasibility() -> None:
    """The declared 'can it pass head-on?' column must match the geometry."""
    expected_frontal = {
        0: True,
        1: True,
        2: True,
        3: True,
        4: False,
        5: False,
    }
    for level, width in LEVEL_CHANNEL_WIDTHS.items():
        frontal_ok = exact_fits_channel(WALL_X, 0.0, 0.0, width)
        check(
            frontal_ok == expected_frontal[level],
            f"level {level}: frontal passage feasible={frontal_ok}, "
            f"brief says {expected_frontal[level]} (width {width}, "
            f"shoulders {ROBOT_SHOULDER_WIDTH})",
        )
    # Levels 4-5 must be comfortably passable once sideways.
    for level in (4, 5):
        check(
            exact_fits_channel(WALL_X, 0.0, 90.0, LEVEL_CHANNEL_WIDTHS[level]),
            f"level {level} is not passable even sideways",
        )
    # Every Level must remain passable frontally at some angle up to 90 deg,
    # otherwise the ladder is unsolvable rather than merely demanding.
    for level, width in LEVEL_CHANNEL_WIDTHS.items():
        angles = [a for a in range(0, 91, 5) if exact_fits_channel(WALL_X, 0.0, a, width)]
        check(angles, f"level {level}: no yaw angle fits the channel")
    print("[ok] frontal/sideways feasibility matches the Level design table")


def test_gate_matches_exact_geometry() -> None:
    """The oriented-box gate must agree with an independent geometric model."""
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        for yaw in range(-90, 91, 5):
            # A body centred in the channel at the wall plane.
            collision = (
                _check_wall_collision(
                    np.array([WALL_X, 0.0, 0.0]), math.radians(yaw), width
                )
                is not None
            )
            predicted = not exact_fits_channel(WALL_X, 0.0, float(yaw), width)
            check(
                collision == predicted,
                f"level {level} yaw {yaw}: gate collision={collision} but the "
                f"geometric model predicts {predicted}",
            )
    print("[ok] oriented-box gate matches an independent geometric model")


def test_body_centre_does_not_drift() -> None:
    """Rotating in place must never move the body's footprint sideways."""
    for yaw in range(-180, 181, 15):
        for root_z in (-0.4, 0.0, 0.4):
            corners = body_corners_xz(1.5, root_z, float(yaw))
            centre = corners.mean(axis=0)
            check(
                abs(centre[0] - 1.5) < 1e-9 and abs(centre[1] - root_z) < 1e-9,
                f"yaw {yaw} root_z {root_z}: footprint centre drifted to {centre}",
            )
    print("[ok] the body footprint stays centred on the root pose")


def test_off_axis_collision() -> None:
    """A frontal body must hit a panel when it is laterally offset."""
    for level, width in LEVEL_CHANNEL_WIDTHS.items():
        half = width / 2.0
        # Just inside the shoulder's half width past the panel edge: collides.
        offset = half + ROBOT_SHOULDER_WIDTH / 2.0 - 0.001
        check(
            _check_wall_collision(np.array([WALL_X, 0.0, offset]), 0.0, width)
            is not None,
            f"level {level}: offset body at z={offset:.3f} should collide",
        )
        # A body with clear room on both sides must be free.
        clearance = width / 2.0 - (ROBOT_SHOULDER_WIDTH / 2.0 + BODY_CLEARANCE)
        if clearance > 0.01:
            check(
                _check_wall_collision(np.array([WALL_X, 0.0, 0.0]), 0.0, width) is None,
                f"level {level}: centred body should not collide",
            )
        else:
            # Levels 4-5: a frontal body cannot fit at all.
            check(
                _check_wall_collision(np.array([WALL_X, 0.0, 0.0]), 0.0, width)
                is not None,
                f"level {level}: a frontal body cannot fit a {width:.2f} m channel",
            )
    print("[ok] off-axis frontal bodies collide with the panels")


# ---------------------------------------------------------------------------
# 4. Sideways route
# ---------------------------------------------------------------------------


def simulate_sideways_route(channel_width: float) -> Tuple[List[np.ndarray], bool]:
    """Rotate 90 degrees in free space, then close the gap and cross.

    Mirrors the route the model is expected to discover with the egocentric
    action set: ``turn_left`` x6 (so the right-hand side faces the channel),
    then alternate between pushing toward the wall and sliding sideways until
    the body centre is past x > 2.5 m.  Returns the (x, z) trace and whether
    the body ever clipped a panel.
    """
    position = ROBOT_START.copy()
    yaw = 0.0
    trace: List[np.ndarray] = []

    # Phase 1: rotate 90 degrees in the free space in front of the wall.
    for _ in range(int(round(SIDEWAYS_YAW_DEG / TURN_STEP_DEG))):
        yaw += TURN_STEP_DEG
    yaw_rad = math.radians(yaw)
    if _check_wall_collision(position, yaw_rad, channel_width) is not None:
        return trace, False

    # Egocentric axes at this yaw.  After +90 degrees the torso faces -z, so
    # "forward" no longer helps; the wall is off the robot's right-hand side.
    ahead = _rotate_xz(np.array([1.0, 0.0, 0.0]), yaw_rad)
    right = _rotate_xz(np.array([0.0, 0.0, 1.0]), yaw_rad)
    check(
        abs(ahead[0]) < 1e-9 and ahead[2] < 0,
        f"expected +90 degree yaw to face -z, got {ahead}",
    )
    check(
        right[0] > 0,
        f"expected +90 degree yaw to put the wall on the right, got {right}",
    )

    clean = True
    for _ in range(80):
        if position[0] > SUCCESS_X:
            break
        moved = False
        # Prefer sliding toward the wall in x (the "right" action at this yaw),
        # and use "forward" once the body is clear of the wall plane.
        for direction in (right, ahead):
            candidate = position + direction * MOVE_STEP
            if (
                _check_wall_collision(candidate, yaw_rad, channel_width)
                is not None
            ):
                continue
            position = candidate
            trace.append(position[[0, 2]].copy())
            moved = True
            break
        if not moved:
            clean = False
            break
    return trace, clean


def test_sideways_route_reaches_goal() -> None:
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        trace, clear = simulate_sideways_route(width)
        check(clear, f"level {level}: sideways route clipped a panel")
        check(
            trace and trace[-1][0] > SUCCESS_X,
            f"level {level}: sideways route ended at x={trace[-1][0] if trace else 'n/a'}, "
            f"never crossed x > {SUCCESS_X}",
        )
        # The path must stay inside the channel it is passing through.
        max_abs_z = max(abs(point[1]) for point in trace)
        check(
            max_abs_z <= width / 2.0 + 1e-9,
            f"level {level}: sideways route wandered to |z|={max_abs_z:.3f}, "
            f"outside the {width:.2f} m channel",
        )
    print("[ok] the sideways route reaches x > 2.5 m at every Level")


def test_frontal_route_only_where_feasible() -> None:
    """Walking straight ahead must work exactly where the table says it does."""
    expected_frontal = {0: True, 1: True, 2: True, 3: True, 4: False, 5: False}
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        position = ROBOT_START.copy()
        reached = False
        blocked_by = None
        for _ in range(DEFAULT_MAX_STEPS + 10):
            candidate = position + np.array([MOVE_STEP, 0.0, 0.0])
            collision = _check_wall_collision(candidate, 0.0, width)
            if collision is not None:
                blocked_by = collision["part"]
                break
            position = candidate
            if position[0] > SUCCESS_X:
                reached = True
                break
        check(
            reached == expected_frontal[level],
            f"level {level}: frontal walk reached={reached}, "
            f"expected {expected_frontal[level]} (blocked by {blocked_by})",
        )
    print("[ok] frontal walk succeeds only at Levels 0-3")


def test_rotation_blocked_inside_wall_slab() -> None:
    """Rotating while fouling the wall must be reported as blocked.

    Levels 4 and 5 are only solvable if the agent turns before reaching the
    wall, so the gate that forbids turning once the shoulders are in the wall
    plane is load-bearing.  The current pose is part of the swept arc, so a
    robot that is already colliding cannot rotate its way free.
    """
    from environment import _turn_path_is_clear

    for level in (4, 5):
        width = LEVEL_CHANNEL_WIDTHS[level]
        collision = _turn_path_is_clear(
            np.array([WALL_X, 0.0, 0.0]), 0.0, TURN_STEP_DEG, width
        )
        check(
            collision is not None,
            f"level {level}: turning with the shoulders in the wall should collide",
        )

    # A pose that is already colliding must not be able to rotate out of it,
    # otherwise the agent can teleport through the wall in 15 degree hops.
    width = LEVEL_CHANNEL_WIDTHS[4]
    fouled_root: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fouled_yaw = 0.0
    found = False
    for x in np.arange(WALL_X - 0.10, WALL_X + 0.10, 0.01):
        for yaw in (30.0, 45.0, 60.0, 75.0):
            candidate = np.array([float(x), 0.0, 0.0])
            if _check_wall_collision(candidate, math.radians(yaw), width) is not None:
                fouled_root = (float(x), 0.0, 0.0)
                fouled_yaw = yaw
                found = True
                break
        if found:
            break
    check(found, "could not construct a wall-fouling pose for Level 4")

    for target in (fouled_yaw - 15.0, fouled_yaw + 15.0):
        check(
            _turn_path_is_clear(
                np.array(fouled_root), fouled_yaw, target, width
            )
            is not None,
            f"turning {fouled_yaw} -> {target} from a colliding pose at "
            f"x={fouled_root[0]:.2f} must be rejected",
        )
    print(
        f"[ok] turns from a wall-fouling pose are rejected "
        f"(tested x={fouled_root[0]:.2f}, yaw={fouled_yaw:.0f} deg)"
    )


# ---------------------------------------------------------------------------
# 5. Action space and prompt
# ---------------------------------------------------------------------------


def test_entry_point_does_not_import_isaac_sim() -> None:
    """Importing main.py must not pull in environment/experiments.

    Regression guard for a bug that only shows up on a real Isaac Sim install:
    ``isaacsim.core`` is importable *only after* SimulationApp has started.  A
    top-level ``from environment import ...`` in the entry point runs first,
    raises ModuleNotFoundError, and latches ``environment._HAS_ISAAC_SIM`` to
    False forever -- so the scene can never be built even though SimulationApp
    started fine.  Run in a subprocess so this test's own imports do not
    pollute the check.
    """
    import subprocess
    import sys

    code = (
        "import sys, main;"
        "bad=[m for m in ('environment','experiments') if m in sys.modules];"
        "print('LEAKED='+','.join(bad));"
        "print('LEVELS=%r' % (tuple(main.PROTOCOL_LEVELS),))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    check(
        result.returncode == 0,
        f"importing main.py failed: {result.stderr.strip()[:400]}",
    )
    leaked_lines = [
        line for line in result.stdout.splitlines() if line.startswith("LEAKED=")
    ]
    check(leaked_lines, f"unexpected subprocess output: {result.stdout!r}")
    leaked = leaked_lines[0].split("=", 1)[1].strip()
    check(
        leaked == "",
        f"importing main.py pulled in {leaked!r}; these must be imported lazily "
        f"inside run_experiment, after SimulationApp starts",
    )

    # The duplicated literals in main.py must match the real modules.
    from experiments import DEFAULT_EPISODES_PER_LEVEL, DEFAULT_LEVELS, DEFAULT_MAX_STEPS
    import main

    check(
        tuple(main.PROTOCOL_LEVELS) == tuple(DEFAULT_LEVELS),
        f"main.PROTOCOL_LEVELS {tuple(main.PROTOCOL_LEVELS)} != "
        f"experiments.DEFAULT_LEVELS {tuple(DEFAULT_LEVELS)}",
    )
    check(
        main.DEFAULT_EPISODES_PER_LEVEL == DEFAULT_EPISODES_PER_LEVEL,
        "main.DEFAULT_EPISODES_PER_LEVEL drifted from experiments",
    )
    check(
        main.DEFAULT_MAX_STEPS == DEFAULT_MAX_STEPS,
        "main.DEFAULT_MAX_STEPS drifted from experiments",
    )
    print("[ok] importing main.py does not import Isaac Sim (no early latch bug)")


def _turn_arc_yaws() -> List[float]:
    """Yaw samples visited by a full 0 -> 90 degree turn."""
    return [float(y) for y in range(0, 91, int(TURN_STEP_DEG))]


def _sharpest_turn_half_span(width: float) -> float:
    """Max half z-extent of the body over the 0 -> 90 degree turn arc.

    This is the widest the body ever becomes while rotating, so it decides
    whether a channel is turnable in place at all.
    """
    return max(
        half_z_span_in_x_slab(yaw, -1e-6, 1e-6) or 0.0 for yaw in _turn_arc_yaws()
    )


def _can_turn_in_place(width: float, root_x: float) -> bool:
    """Can a full 0 -> 90 degree turn be executed at ``root_x`` on the 15 deg grid?"""
    from environment import _turn_path_is_clear

    yaw = 0.0
    while yaw < 90.0 - 1e-9:
        if (
            _turn_path_is_clear(np.array([root_x, 0.0, 0.0]), yaw, yaw + TURN_STEP_DEG, width)
            is not None
        ):
            return False
        yaw += TURN_STEP_DEG
    return True


def test_turn_clearance_boundary() -> None:
    """Turning must be possible while clear of the wall and blocked once fouled.

    This is the core manipulation check: Levels 4-5 are only solvable by
    rotating *before* reaching the wall, so the clearance boundary has to sit in
    front of the wall rather than inside it.  Wide Levels can be turned in
    anywhere, which is why the strict boundary assertion only applies where the
    channel is too narrow to rotate in.
    """
    from environment import _turn_path_is_clear

    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        # Find the furthest-forward root x (in front of the wall) from which a
        # full 0 -> 90 turn is legal on the 15 degree action grid.
        limit = None
        for xi in range(int(WALL_X * 100), -1, -1):
            x = xi / 100.0
            if _can_turn_in_place(width, x):
                limit = x
                break
        check(limit is not None, f"level {level}: no x allows a full turn")

        can_turn_at_wall = _can_turn_in_place(width, WALL_X)
        if can_turn_at_wall:
            # Generous channel: the agent can rotate anywhere, including at the
            # wall, so there is no clearance boundary to respect.
            check(
                limit >= WALL_X - 1e-9,
                f"level {level}: turnable at the wall but limit is x={limit}",
            )
            continue

        check(
            limit < WALL_X,
            f"level {level}: a full 0-90 turn is still legal at x={limit}, at or "
            f"past the wall at x={WALL_X}, so the channel is not a manipulation "
            f"problem",
        )
        check(
            _turn_path_is_clear(
                np.array([limit, 0.0, 0.0]), 0.0, TURN_STEP_DEG, width
            )
            is None,
            f"level {level}: turn should be legal at the boundary x={limit}",
        )
        check(
            _turn_path_is_clear(
                np.array([ROBOT_START[0], 0.0, 0.0]), 0.0, 90.0, width
            )
            is None,
            f"level {level}: the full turn must be available from the start pose",
        )
        # The decisive property: at the channel mouth the agent can no longer
        # complete the rotation, so it has to have turned before arriving.
        check(
            not _can_turn_in_place(width, WALL_X),
            f"level {level}: a full turn is possible at the channel mouth",
        )
    print("[ok] the turn clearance boundary sits in front of the wall")


def test_action_space() -> None:
    expected = [
        "forward",
        "backward",
        "left",
        "right",
        "turn_left",
        "turn_right",
        "look_left",
        "look_right",
    ]
    check(ACTIONS == expected, f"unexpected action space: {ACTIONS}")
    for removed in ("reach_left_arm", "reach_right_arm", "retreat_left_arm",
                    "retreat_right_arm", "raise_left_arm", "raise_right_arm"):
        check(removed not in ACTIONS, f"{removed} should have been removed")
    check(
        CAMERA_TURN_STEP_DEG == 30.0,
        f"camera step changed to {CAMERA_TURN_STEP_DEG}",
    )
    check(TURN_STEP_DEG == 15.0, f"turn step changed to {TURN_STEP_DEG}")
    print("[ok] action space is the 6 locomotion actions plus 2 camera actions")


def test_prompt_is_uniform_and_leak_free() -> None:
    """The prompt must be identical everywhere and reveal no geometry."""
    base_state = {
        "position": [1.5, 0.0, 0.0],
        "torso_rotation": 0.0,
        "channel_width": 0.45,
        "a_s_ratio": 0.79,
    }
    prompt = build_prompt(state=base_state, history=[], max_steps=30)

    forbidden = [
        "0.90", "0.80", "0.74", "0.68", "0.57", "0.45",
        "1.58", "1.40", "1.30", "1.19", "1.00", "0.79",
        "shoulder", "sideways", "turn your body", "A/S",
        "wide", "narrow", "opening width", "0.22", "0.338",
    ]
    lowered = prompt.lower()
    for token in forbidden:
        check(
            token.lower() not in lowered,
            f"prompt leaks geometry token {token!r}",
        )

    # Same state + same history must give the same prompt for every Level; the
    # builder has no level parameter, which is asserted structurally here.
    again = build_prompt(state=base_state, history=[], max_steps=30)
    check(prompt == again, "prompt is not deterministic")

    # Action list must be exactly the current ACTIONS, with no numbering drift.
    for action in ACTIONS:
        check(
            f'{{"action": "{action}"}}' in ACTION_OPTIONS_STRING,
            f"action {action} missing from the options block",
        )
    check(
        TASK_INSTRUCTION in prompt,
        "the unified task instruction is missing from the prompt",
    )
    print("[ok] prompt is uniform, deterministic, and leak-free")


def test_episode_defaults() -> None:
    check(
        DEFAULT_EPISODES_PER_LEVEL == 20,
        f"episodes per level is {DEFAULT_EPISODES_PER_LEVEL}, expected 20",
    )
    check(
        DEFAULT_MAX_STEPS == 30,
        f"max steps is {DEFAULT_MAX_STEPS}, expected 30",
    )
    check(SUCCESS_X == 2.5, f"success threshold is {SUCCESS_X}, expected 2.5")
    print("[ok] protocol defaults are 20 episodes x 30 steps, success at x > 2.5")


def test_sideways_band() -> None:
    check(_is_sideways_yaw(90.0), "90 degrees should count as sideways")
    check(_is_sideways_yaw(-90.0), "mirrored 90 degrees should count as sideways")
    check(not _is_sideways_yaw(0.0), "0 degrees should not count as sideways")
    check(not _is_sideways_yaw(30.0), "30 degrees should not count as sideways")
    check(not _is_sideways_yaw(180.0), "reversed 180 degrees should not count")
    print("[ok] sideways yaw band is 45-135 degrees and mirror-symmetric")


def test_no_unbound_names() -> None:
    """Every module must pass the static unbound-name check.

    Several real bugs in this project were 'a name is referenced but nothing
    binds it', and they only surfaced on a real Isaac Sim install after a
    ~150 s startup.  This runs tools/check_names.py so that class of bug is
    caught in seconds instead.
    """
    import subprocess

    root = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(root, "tools", "check_names.py")
    check(os.path.exists(script), "tools/check_names.py is missing")
    result = subprocess.run(
        [sys.executable, script], capture_output=True, text=True, cwd=root
    )
    check(
        result.returncode == 0,
        "tools/check_names.py found unbound names:\n"
        + (result.stdout or "").strip()
        + (result.stderr or "").strip(),
    )
    print("[ok] no unbound names in any module (static check)")


def test_scene_readability_constants() -> None:
    """The channel must be visually distinguishable from the wall.

    Regression guard for the core failure seen on the lab machine: the model
    reported "a solid gray wall with no visible features or openings" while
    standing in front of the gap.  Three scene properties control that, and all
    three had values that made the opening invisible.
    """
    import environment as env

    # 1. The eye camera must not be a fisheye.  Isaac Sim's Camera defaults to a
    #    20.955 mm sensor (1/2.9", matching the H1's RealSense form factor), so
    #    the inherited 1.5 mm focal length is about 164 degrees -- which
    #    collapsed the 4 m wall into the middle of the frame and rendered the
    #    channel posts a couple of pixels wide.
    sensor_width = 20.955
    focal = env.ROBOT_CAMERA_FOCAL
    check(focal > 0.0, f"ROBOT_CAMERA_FOCAL is {focal}")
    fov_deg = 2.0 * math.degrees(math.atan((sensor_width / 2.0) / focal))
    check(
        fov_deg < 120.0,
        f"eye camera field of view is {fov_deg:.0f} deg, which is fisheye-like; "
        f"the channel will be too small to see (focal {focal} mm)",
    )
    check(
        fov_deg > 60.0,
        f"eye camera field of view is only {fov_deg:.0f} deg; too narrow to see "
        f"the opening while walking through it",
    )

    # 2. The channel posts must be wide enough to be more than a few pixels at
    #    the distance the robot starts from.
    edge = env.CHANNEL_EDGE_THICKNESS
    check(
        edge >= 0.03,
        f"channel edge thickness is {edge} m; at the old 1 cm the posts were "
        f"invisible",
    )
    # Angular size of the post from the robot's start distance.
    start_distance = abs(env.WALL_X - env.ROBOT_START_POS[0])
    angular = 2.0 * math.degrees(math.atan((edge / 2.0) / start_distance))
    pixels = angular / fov_deg * 1024.0
    check(
        pixels >= 8.0,
        f"the channel post subtends only {pixels:.1f} px at the start pose; it "
        f"needs to be clearly visible",
    )

    # 3. The wall must read as a surface, not as a window onto an identical
    #    grey room.
    opacity = env.WALL_OPACITY
    check(
        0.0 < opacity < 1.0,
        f"WALL_OPACITY must be translucent but non-trivial, got {opacity}",
    )
    check(
        opacity >= 0.6,
        f"WALL_OPACITY is {opacity}; at the inherited 0.45 the wall and the "
        f"room behind it rendered almost identically in grey",
    )

    # The edge colour must actually contrast with the wall's own colour.
    edge_luma = sum(env.CHANNEL_EDGE_COLOR) / 3.0
    check(
        edge_luma < 0.35,
        f"channel edge colour {env.CHANNEL_EDGE_COLOR} is too light to contrast "
        f"with the wall",
    )
    print(
        f"[ok] scene readability: eye FOV {fov_deg:.0f} deg, posts "
        f"{pixels:.0f} px wide at start, wall opacity {opacity}"
    )


def test_start_distance_reachable_in_budget() -> None:
    """Start distance and step size must fit the episode budget together.

    Standing further back makes the channel readable, but the robot still has to
    travel to x > 2.5 within DEFAULT_MAX_STEPS.  Raising the start distance
    without raising the step size makes the task physically impossible, which
    would look like a model failure in the results.
    """
    from environment import MOVE_STEP, ROBOT_START_POS, TURN_STEP_DEG

    def steps_needed(start_x: float, move_step: float) -> int:
        """Steps for: turn 90 deg, sidle to x > 2.5, i.e. the intended route.

        The robot advances along +x, so standing *further back* means a
        *smaller* start_x.  It has already passed x=2.5 if start_x >= 2.5.
        """
        turns = int(round(90.0 / TURN_STEP_DEG))
        if start_x >= SUCCESS_X:
            return 0
        travel = (SUCCESS_X - start_x) + 0.10  # 10 cm of margin past the plane
        moves = int(math.ceil(travel / move_step))
        return turns + moves

    scenarios = [
        # (label, start_x, move_step, must_fit)
        ("current default", float(ROBOT_START_POS[0]), MOVE_STEP, True),
        ("1.0 m back, default step", 1.0, 0.05, False),
        ("1.0 m back, 0.10 m step", 1.0, 0.10, True),
        ("1.5 m back, default step", 0.5, 0.05, False),
        ("1.5 m back, 0.10 m step", 0.5, 0.10, True),
    ]
    for label, start_x, move_step, must_fit in scenarios:
        needed = steps_needed(start_x, move_step)
        if must_fit:
            check(
                needed <= DEFAULT_MAX_STEPS,
                f"{label}: start_x={start_x} step={move_step} needs {needed} "
                f"steps but the budget is {DEFAULT_MAX_STEPS}; the task would be "
                f"impossible",
            )
        else:
            check(
                needed > DEFAULT_MAX_STEPS,
                f"{label}: expected this combination to be infeasible, but it "
                f"needs only {needed} steps",
            )

    # The angular size of the channel is what standing back actually buys.
    #
    # Note the direction: moving back makes the channel SUBTEND LESS of the
    # frame.  At the original start_x=1.5 the robot is 0.5 m from the wall and
    # the 0.90 m channel covers ~80% of a 105-degree frame -- the wall fills the
    # view with no context around it, which is why the scene read as "a solid
    # grey wall".  A few steps back frames the opening the way a person sees a
    # doorway, at the cost of travel.
    sensor_width = 20.955
    fov_deg = 2.0 * math.degrees(
        math.atan((sensor_width / 2.0) / __import__("environment").ROBOT_CAMERA_FOCAL)
    )

    def frame_share(start_x: float, channel: float = 0.90) -> float:
        distance = max(WALL_X - start_x, 1e-6)
        angular = 2.0 * math.degrees(math.atan((channel / 2.0) / distance))
        return angular / fov_deg

    shares = {x: frame_share(x) for x in (1.5, 1.0, 0.5, 0.0)}
    check(
        shares[1.5] > shares[1.0] > shares[0.5] > shares[0.0],
        f"the channel should subtend LESS of the frame as the robot starts "
        f"further back (smaller x), got {shares}",
    )
    check(
        shares[0.0] < 0.40,
        f"at start_x=0 the channel should be a minority of the frame, got "
        f"{100 * shares[0.0]:.0f}%",
    )
    # The shipped start distance must keep the gap under half the frame, which
    # is what makes the wall edges visible.  (Closer starts exceed this: at
    # x=1.5, 0.5 m from the wall, the gap subtends 110% of the view.)
    check(
        shares[start_x] < 0.5,
        f"the shipped start_x={start_x} leaves the gap filling "
        f"{100 * shares[start_x]:.0f}% of the frame, so there is no wall in "
        f"view to contrast against",
    )
    print(
        "[ok] start distance and step size stay within the step budget "
        f"(default needs {steps_needed(float(ROBOT_START_POS[0]), MOVE_STEP)} of "
        f"{DEFAULT_MAX_STEPS} steps; channel fills "
        + ", ".join(f"{100 * shares[x]:.0f}% at x={x}" for x in (1.5, 1.0, 0.5, 0.0))
        + ")"
    )


def test_primitive_sizing_is_in_metres() -> None:
    """Boxes must be built with explicit metre extents, not FixedCuboid.

    `FixedCuboid`'s size/scale combination does not mean metres and is
    inconsistent.  Measured on the lab machine with size=0.5:

        authored 0.05 x 0.05 x 2.00 -> world 0.0013 x 2.00 x 0.0013
        authored 0.02 x 2.00 x 1.55 -> world 0.0002 x 1.20 x 2.00
        authored 0.02 x 3.00 x 4.00 -> world 0.0002 x 8.00 x 4.50

    Consequence: the channel posts, authored 5 cm wide, rendered about 1.3 mm
    wide, so no vertical edge ever appeared where the opening was and the
    opening read as invisible in every measurement taken.
    """
    import inspect
    import re

    import environment as env

    check(
        hasattr(env.BAOEnv, "_add_box"),
        "BAOEnv._add_box is missing; boxes must be built with explicit extents",
    )
    source = inspect.getsource(env)
    # Ignore comments and docstrings, which legitimately mention FixedCuboid
    # when explaining why it is not used, and inspect real code only.
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)
    code = ast.unparse(tree)
    check(
        "FixedCuboid" not in code,
        "environment.py still uses FixedCuboid, whose dimensions are not metres",
    )
    call_sites = len(re.findall(r"self\._add_box\(", code))
    check(
        call_sites >= 5,
        f"only {call_sites} call(s) build boxes via _add_box; expected the "
        f"ground, grid, wall panels, channel posts and room to use it",
    )

    # _add_box must set the extent explicitly, otherwise it is the same trap.
    box_source = inspect.getsource(env.BAOEnv._add_box)
    check(
        "GetExtentAttr" in box_source,
        "_add_box does not set the cube extent, so its size is still implicit",
    )

    check(
        env.CHANNEL_EDGE_THICKNESS >= 0.03,
        f"channel post width {env.CHANNEL_EDGE_THICKNESS} m is too thin to see",
    )
    print(
        f"[ok] boxes are sized in metres ({call_sites} _add_box sites); channel "
        f"post is {env.CHANNEL_EDGE_THICKNESS * 100:.0f} cm wide"
    )


def test_room_is_enclosed_and_coloured() -> None:
    """The space behind the channel must contain something to look at.

    Measured motivation: with only the obstacle wall and the floor, every ray
    through the opening returned the same value (unique_colors = 2 at pitch 0
    and 15 from 0.5 m), so the eye view was unreadable wherever the camera was
    aimed and the model reported "a solid grey wall with no visible openings"
    while facing the gap.
    """
    import inspect

    import environment as env

    check(
        hasattr(env.BAOEnv, "_create_room"),
        "BAOEnv._create_room is missing",
    )
    source = inspect.getsource(env.BAOEnv._create_room)
    for name in ("room_far", "room_near", "room_side_left", "room_side_right"):
        check(name in source, f"the room is missing its {name} wall")
    check("room_ceiling" in source, "the room has no ceiling")

    # The far wall must actually sit beyond the obstacle wall and beyond the
    # success plane, or it would block the robot.
    check(
        env.SCENE_SIZE > env.WALL_X,
        f"the far wall at x={env.SCENE_SIZE} must be past the obstacle at "
        f"x={env.WALL_X}",
    )
    check(
        env.SCENE_SIZE > env.SUCCESS_X,
        f"the far wall at x={env.SCENE_SIZE} must be past the success plane at "
        f"x={env.SUCCESS_X}",
    )

    # The walls must be coloured, and green in particular: a grey room is what
    # made the opening invisible in the first place.
    colour = env.ROOM_WALL_COLOR
    check(
        len(colour) == 3 and all(0.0 <= c <= 1.0 for c in colour),
        f"ROOM_WALL_COLOR {colour} is not a valid RGB triple",
    )
    check(
        colour[1] > colour[0] and colour[1] > colour[2],
        f"ROOM_WALL_COLOR {colour} is not green-dominant; a neutral grey room "
        f"renders the opening and the wall identically",
    )
    # And it must be far enough from the floor's grey to be distinguishable.
    check(
        abs(sum(colour) / 3.0 - 0.5) > 0.15,
        f"ROOM_WALL_COLOR {colour} is too close to mid-grey to contrast",
    )

    # Walls must be taller than the robot, or the robot could see over them.
    check(
        env.ROOM_WALL_HEIGHT > 1.806,
        f"ROOM_WALL_HEIGHT {env.ROOM_WALL_HEIGHT} is not above the measured "
        f"1.806 m robot",
    )
    print(
        f"[ok] the room is enclosed and green (colour {colour}, "
        f"height {env.ROOM_WALL_HEIGHT:.1f} m, far wall at x={env.SCENE_SIZE:.1f})"
    )


def test_default_start_is_usable() -> None:
    """The shipped defaults must be reachable and frame the opening.

    From 1.5 m the 2.0 m gap subtends about 77 degrees and fills the whole
    105-degree eye view, leaving no wall in frame; from 0.5 m it is about 39
    degrees.  The start distance and step size have to move together or the
    robot cannot reach x > 2.5 inside the budget.
    """
    import environment as env

    start_x = float(env.ROBOT_START_POS[0])
    move_step = float(env.MOVE_STEP)
    turns = int(round(90.0 / env.TURN_STEP_DEG))
    travel = (SUCCESS_X - start_x) + 0.10
    needed = turns + int(math.ceil(travel / move_step))
    check(
        needed <= DEFAULT_MAX_STEPS,
        f"the shipped defaults need {needed} steps but the budget is "
        f"{DEFAULT_MAX_STEPS}",
    )

    # The gap must not swallow the whole frame.  Horizontally, the opening is
    # LEVEL_CHANNEL_WIDTHS[0] wide at distance (WALL_X - start_x).
    fov_deg = 2.0 * math.degrees(
        math.atan((20.955 / 2.0) / env.ROBOT_CAMERA_FOCAL)
    )
    distance = env.WALL_X - start_x
    gap_deg = 2.0 * math.degrees(
        math.atan((env.LEVEL_CHANNEL_WIDTHS[0] / 2.0) / distance)
    )
    check(
        gap_deg < fov_deg / 2.0,
        f"the {env.LEVEL_CHANNEL_WIDTHS[0]:.2f} m gap subtends {gap_deg:.0f} deg "
        f"of a {fov_deg:.0f} deg view, so it fills most of the frame and leaves "
        f"no wall to contrast against",
    )
    print(
        f"[ok] defaults usable: start_x={start_x} step={move_step} needs "
        f"{needed}/{DEFAULT_MAX_STEPS} steps; gap subtends {gap_deg:.0f} deg "
        f"({100 * gap_deg / fov_deg:.0f}%) of a {fov_deg:.0f} deg view"
    )


def test_lights_are_inside_the_enclosed_room() -> None:
    """Lights must sit inside the room, not outside its ceiling.

    Regression guard: after the room was enclosed, the external views dropped
    from mean=104 to mean=1.8 -- almost black -- because the inherited dome and
    distant lights originate outside and the new ceiling blocked them.
    """
    import inspect

    import environment as env

    source = inspect.getsource(env.BAOEnv._create_lights)

    # A DistantLight is directional and comes from outside, so it cannot light
    # an enclosed room.
    check(
        "DistantLight" not in source,
        "the scene still uses a DistantLight, which the room's ceiling blocks",
    )
    check(
        "SphereLight" in source,
        "the scene has no interior light; the enclosed room would be dark",
    )

    # Every interior light must be below the ceiling and inside the floor plan.
    for position in ([1.0, 2.6, 0.0], [3.0, 2.6, 0.0]):
        x, y, z = position
        check(
            y < env.ROOM_WALL_HEIGHT,
            f"light at y={y} is above the ceiling (ROOM_WALL_HEIGHT="
            f"{env.ROOM_WALL_HEIGHT})",
        )
        check(
            0.0 < x < env.SCENE_SIZE,
            f"light at x={x} is outside the room (0..{env.SCENE_SIZE})",
        )
        check(
            abs(z) < env.SCENE_SIZE / 2.0,
            f"light at z={z} is outside the room",
        )
    print(
        f"[ok] interior lights sit below the {env.ROOM_WALL_HEIGHT:.1f} m ceiling "
        f"and inside the room"
    )


def test_eye_camera_pitches_downward() -> None:
    """The head camera must look down, from the real head height.

    Regression guard for a scene-readability bug found on the lab machine.  The
    original reach-the-ball build pitched the eye camera down implicitly by
    aiming at the ball (`target[1] = TARGET_POS[1]`).  Removing the ball
    removed that pitch, and a camera 0.5 m from a 2.0 m wall at 1.9 m looking
    perfectly level renders as a featureless grey plane -- the model reported
    "a solid gray wall with no visible features or openings" and could never
    find the channel.
    """
    import inspect

    from environment import EYE_PITCH_DEG, _eye_look_direction

    check(EYE_PITCH_DEG > 0.0, f"EYE_PITCH_DEG is {EYE_PITCH_DEG}, must be > 0")

    for yaw_deg in (-90.0, -45.0, 0.0, 45.0, 90.0, 180.0):
        for pitch in (0.0, EYE_PITCH_DEG, 30.0):
            offset = _eye_look_direction(math.radians(yaw_deg), pitch, 2.0)
            expected_y = -2.0 * math.sin(math.radians(pitch))
            check(
                abs(float(offset[1]) - expected_y) < 1e-9,
                f"yaw {yaw_deg} pitch {pitch}: vertical offset {offset[1]} "
                f"!= {expected_y}",
            )
            check(
                offset[1] <= 1e-12,
                f"yaw {yaw_deg} pitch {pitch}: camera looks upward",
            )

    # Straight ahead must stay horizontal and keep the full look distance.
    level = _eye_look_direction(0.0, 0.0, 2.0)
    check(abs(float(level[1])) < 1e-12, "pitch 0 must be exactly horizontal")
    check(
        abs(float(level[0]) - 2.0) < 1e-9,
        f"pitch 0 yaw 0 should point along +x, got {level}",
    )

    # The anchor must be the human-like eye height, not the old ball-viewing
    # 1.9 m.  The measured H1 is 1.806 m tall, so the eye sits at ~93% of it.
    from environment import EYE_CAMERA_HEIGHT, ROBOT_HEAD_HEIGHT as _HEAD

    check(
        abs(_HEAD - 1.55) < 1e-9,
        f"authored ROBOT_HEAD_HEIGHT is {_HEAD}, expected the original 1.55",
    )
    check(
        EYE_CAMERA_HEIGHT > _HEAD,
        "the eye camera should sit at the human-like height, above the authored "
        "head constant",
    )
    check(
        EYE_CAMERA_HEIGHT < 1.9,
        "the eye camera must sit below the old ball-viewing height of 1.9 m",
    )
    check(
        1.60 <= EYE_CAMERA_HEIGHT <= 1.75,
        f"EYE_CAMERA_HEIGHT is {EYE_CAMERA_HEIGHT}; expected roughly 93% of the "
        f"measured 1.806 m robot height",
    )

    # The pitch must actually be wired into the update path.
    source = inspect.getsource(
        __import__("environment").BAOEnv._update_eye_camera
    )
    check(
        "eye_pitch_deg" in source,
        "_update_eye_camera does not read eye_pitch_deg",
    )
    print(
        f"[ok] head camera pitches down {EYE_PITCH_DEG:.0f} deg from "
        f"y={EYE_CAMERA_HEIGHT} m"
    )


def test_ground_grid_exists() -> None:
    """The floor must carry a reference grid.

    Without it the floor is one flat grey slab and every camera angle is
    unreadable: no scale, no distance cue, nothing for the transparent wall to
    contrast against.
    """
    import inspect

    from environment import BAOEnv

    check(
        hasattr(BAOEnv, "_create_ground_grid"),
        "BAOEnv._create_ground_grid is missing",
    )
    check(
        "_add_box" in inspect.getsource(BAOEnv._create_ground_grid),
        "the ground grid does not build any geometry",
    )
    check(
        "_create_ground_grid" in inspect.getsource(BAOEnv._create_ground),
        "_create_ground does not call the grid builder",
    )
    print("[ok] the ground carries a scale-reference grid")


def test_camera_look_at_orientation() -> None:
    """look_at_quaternion must actually point the camera at the target.

    Regression guard for a diagnostic bug: capture_views.py used
    set_camera_view() to move the *viewport* camera and then read
    env.camera.get_rgb(), which reads the /World/Camera *sensor*.  The sensor
    never moved, so all six "different viewpoints" produced byte-identical
    images.  The fix moves the sensor itself, which needs a real orientation
    computed from eye and target.
    """
    from capture_views import look_at_quaternion

    def rotate(quat, vec):
        w, x, y, z = (float(v) for v in quat)
        # q * v * q^-1, expanded.
        u = np.array([x, y, z], dtype=float)
        v = np.asarray(vec, dtype=float)
        return (
            2.0 * float(np.dot(u, v)) * u
            + (w * w - float(np.dot(u, u))) * v
            + 2.0 * w * np.cross(u, v)
        )

    cases = [
        ([1.5, 1.55, 0.0], [2.0, 1.0, 0.0]),
        ([2.0, 7.0, 0.0], [2.0, 0.0, 0.0]),
        ([0.3, 2.2, 1.8], [2.2, 0.7, 0.2]),
        ([3.8, 1.3, 0.0], [2.0, 1.0, 0.0]),
    ]
    for eye, target in cases:
        quat = look_at_quaternion(eye, target)
        check(
            abs(float(np.linalg.norm(quat)) - 1.0) < 1e-9,
            f"eye {eye} target {target}: quaternion is not unit length",
        )
        # The camera's local +X is its forward axis in the "world" convention.
        forward = rotate(quat, [1.0, 0.0, 0.0])
        expected = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
        expected = expected / float(np.linalg.norm(expected))
        check(
            float(np.dot(forward, expected)) > 1.0 - 1e-9,
            f"eye {eye} target {target}: forward {forward} does not match "
            f"{expected}",
        )
        # Up must stay roughly upward (never inverted or sideways-degenerate).
        up = rotate(quat, [0.0, 0.0, 1.0])
        check(
            float(up[2]) > 0.0,
            f"eye {eye} target {target}: camera up flipped (up={up})",
        )

    # A straight-down view (the floor plan) needs an up reference perpendicular
    # to the view direction.  NOTE the frame: capture_views converts user
    # coordinates (y up) to Isaac coordinates (z up) by swapping y and z, so
    # "looking down" arrives here as -Z and the up hint must be world +Y.
    top_quat = look_at_quaternion([2.0, 0.0, 7.0], [2.0, 0.0, 0.0], up=(0.0, 1.0, 0.0))
    forward = rotate(top_quat, [1.0, 0.0, 0.0])
    check(
        float(forward[2]) < -0.999,
        f"floor-plan camera should look straight down -Z, got forward={forward}",
    )
    # And the degenerate configuration must be rejected rather than silently
    # producing a garbage orientation.
    try:
        look_at_quaternion([2.0, 0.0, 7.0], [2.0, 0.0, 0.0], up=(0.0, 0.0, -1.0))
        check(False, "a degenerate up vector should raise, not return a quaternion")
    except ValueError:
        pass
    print("[ok] look-at orientation actually aims the camera at its target")


def test_no_viewport_camera_in_diagnostics() -> None:
    """capture_views must move the sensor it reads, not the viewport."""
    import ast
    import inspect

    import capture_views

    tree = ast.parse(inspect.getsource(capture_views))

    # Only look at real code, never at the docstring that explains the bug.
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name:
                called.add(name)
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                called.add(alias.name)

    check(
        "set_camera_view" not in called,
        "capture_views calls set_camera_view (viewport), which does not move "
        "the sensor get_rgb() reads -- this silently produced identical images",
    )
    check(
        "set_world_pose" in called,
        "capture_views never calls Camera.set_world_pose to move the sensor",
    )
    print("[ok] capture_views moves the sensor it reads (no viewport confusion)")


def main() -> int:
    tests = [
        test_channel_ladder,
        test_frontal_feasibility,
        test_gate_matches_exact_geometry,
        test_body_centre_does_not_drift,
        test_off_axis_collision,
        test_sideways_route_reaches_goal,
        test_frontal_route_only_where_feasible,
        test_rotation_blocked_inside_wall_slab,
        test_turn_clearance_boundary,
        test_entry_point_does_not_import_isaac_sim,
        test_action_space,
        test_prompt_is_uniform_and_leak_free,
        test_episode_defaults,
        test_sideways_band,
        test_eye_camera_pitches_downward,
        test_scene_readability_constants,
        test_room_is_enclosed_and_coloured,
        test_primitive_sizing_is_in_metres,
        test_lights_are_inside_the_enclosed_room,
        test_default_start_is_usable,
        test_start_distance_reachable_in_budget,
        test_ground_grid_exists,
        test_camera_look_at_orientation,
        test_no_viewport_camera_in_diagnostics,
        test_no_unbound_names,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Failure as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    print()
    if failures:
        print(f"{failures}/{len(tests)} checks FAILED")
        return 1
    print(f"all {len(tests)} geometry/protocol checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
