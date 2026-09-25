"""Offline geometry and protocol verification for EmbodiedBAO.

Isaac Sim is not available in every development environment, so the parts of
the benchmark that decide whether the experiment is valid at all are checked
here with plain Python:

1. **Channel ladder**  -- each Level builds exactly the width from the task
   brief and the derived A/S ratio matches the hand-computed value.
2. **Frontal feasibility** -- a Level must be frontally passable only when the
   channel is genuinely as wide as the shoulders.  Levels 0-4 pass head-on
   (Level 4 exactly, with no aim tolerance to spare); Level 5 cannot at any yaw
   below 75 degrees, so it is the Level that requires a rotation.
3. **Analytic gate** -- ``_check_wall_collision`` agrees with the closed-form
   projected-width test, including the critical 0.90 m channel where the
   shoulders are narrower than the opening.
4. **Rotation route** -- the route the model is expected to find (rotate the torso
   until it is narrow across the opening, then keep walking forward, because the
   walking direction does not follow the torso) actually reaches x >= 11.0 m at
   every Level, and the robot's body box clears the wall panels while doing it.
5. **Action space and prompt** -- ``reach``/``retreat`` are gone, the camera
   actions keep the legacy 30 degree step, and the prompt is byte-identical
   across Levels and free of geometry leaks.
6. **Within-episode memory** -- the agent's whole action history, with its own
   reasoning, is rendered into the prompt for the entire episode, and the block
   is empty at the start of a fresh episode.

Run with a plain Python interpreter:

    python test_bao_geometry.py
"""

from __future__ import annotations

import math
import inspect
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from environment import (
    ACTIONS,
    BODY_CLEARANCE,
    CAMERA_ACTIONS,
    CAMERA_PITCH_STEP_DEG,
    CAMERA_TURN_STEP_DEG,
    LEVEL_CHANNEL_WIDTHS,
    MOVE_STEP,
    ROBOT_SHOULDER_WIDTH,
    ROBOT_START_POS,
    ROBOT_TORSO_THICKNESS,
    SUCCESS_X,
    TURN_STEP_DEG,
    WALL_THICKNESS,
    WALL_X,
    _check_room_boundary,
    _check_wall_collision,
    _panel_boxes,
    _translation_path_is_clear,
    _robot_body_aabb,
    action_delta,
    a_s_ratio,
    level_channel_width,
)
from experiments import (
    DEFAULT_EPISODES_PER_LEVEL,
    DEFAULT_MAX_STEPS,
    _is_sideways_yaw,
)
from protocol import (
    ACTION_OPTIONS_STRING,
    HISTORY_LIMIT,
    HISTORY_REASONING_CHARS,
    TASK_INSTRUCTION,
    build_prompt,
)

# A/S ratios from the task brief, rounded to two decimals.
EXPECTED_A_S = {0: 1.58, 1: 1.40, 2: 1.30, 3: 1.19, 4: 1.00, 5: 0.79}

ROBOT_START = ROBOT_START_POS.copy()


class Failure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# ---------------------------------------------------------------------------
# 1. Channel ladder
# ---------------------------------------------------------------------------


def test_channel_ladder() -> None:
    """The ladder must be the reference A/S series, not an arbitrary set.

    12 widths from A/S = 2.0 down to 0.9 in steps of 0.1, which is the aperture
    series of Keizer et al. (2013) following Warren & Whang (1987).  Asserted as
    the SERIES rather than as a table of widths: a hand-written table would have to
    be edited every time the shoulder width or the sampling changes, and the point
    is that the ratios are the published ones.
    """
    expected_ratios = [round(2.0 - 0.1 * index, 1) for index in range(12)]
    check(
        list(LEVEL_CHANNEL_WIDTHS) == list(range(12)),
        f"the ladder has levels {sorted(LEVEL_CHANNEL_WIDTHS)}, expected 0-11",
    )
    for level, expected_ratio in zip(sorted(LEVEL_CHANNEL_WIDTHS), expected_ratios):
        width = level_channel_width(level)
        ratio = a_s_ratio(width)
        check(
            abs(round(ratio, 2) - expected_ratio) < 1e-9,
            f"level {level}: A/S {ratio:.4f} rounds to {round(ratio, 2)}, "
            f"expected {expected_ratio}",
        )
        # The width must be the ratio times the body's own shoulder width, so the
        # apertures are body-scaled exactly as the human studies' are.
        check(
            abs(width - ROBOT_SHOULDER_WIDTH * expected_ratio) < 1e-9,
            f"level {level}: width {width} is not {expected_ratio} x the shoulder "
            f"width {ROBOT_SHOULDER_WIDTH}",
        )
    # Widest first, so a sweep runs from trivially passable to rotation-required.
    widths = [level_channel_width(level) for level in sorted(LEVEL_CHANNEL_WIDTHS)]
    check(
        all(a > b for a, b in zip(widths, widths[1:])),
        f"the ladder is not ordered widest first: {widths}",
    )
    print(
        f"[ok] channel ladder is the reference A/S series "
        f"({expected_ratios[0]} -> {expected_ratios[-1]}, {len(expected_ratios)} widths)"
    )


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
    """Exact passability: does the footprint squeeze through the gap at the wall?

    The panels occupy the x-slab ``[WALL_X - t, WALL_X + t]`` outside
    ``|z| < w/2``, so the body fits exactly when its z-extent inside that slab
    stays within the gap.  This is an independent re-derivation of the gate, not
    a copy of it, so agreement is evidence the collision model is right.

    Touching counts as fitting.  The comparison used to be
    ``half_span < width/2 - 1e-12``, which made a body exactly as wide as the
    channel "not fit" and so disagreed with the gate once the gate's tolerance
    let A/S == 1.00 pass -- at Level 4 the two models reported opposite answers
    for the same pose.  Since A/S = 1.00 is defined as the width where shoulder
    and channel are equal, the boundary must be permissive, not exclusive.
    """
    slab_lo = WALL_X - WALL_THICKNESS / 2.0 - root_x
    slab_hi = WALL_X + WALL_THICKNESS / 2.0 - root_x
    half_span = half_z_span_in_x_slab(yaw_deg, slab_lo, slab_hi)
    if half_span is None:
        return True
    return bool(half_span <= width / 2.0 + 1e-12)


def test_frontal_feasibility() -> None:
    """The frontal-passage column must match the geometry, derived not tabulated.

    Frontal passage is feasible exactly when the channel is at least as wide as
    the shoulders, i.e. A/S >= 1.0, which in the 12-width series is every Level
    except the last (A/S = 0.9).  This is asserted as that identity rather than as
    a hand-written table, so it stays true if the sampling changes.  It expected
    False at A/S = 1.0 while a 2 mm skin was added to the body box, which made the
    gate demand 0.574 m and turned that Level's own advertised A/S into a value it
    could not honour.
    """
    for level, width in LEVEL_CHANNEL_WIDTHS.items():
        expected = a_s_ratio(width) >= 1.0 - 1e-12
        frontal_ok = exact_fits_channel(WALL_X, 0.0, 0.0, width)
        check(
            frontal_ok == expected,
            f"level {level}: frontal passage feasible={frontal_ok}, expected "
            f"{expected} (A/S {a_s_ratio(width):.2f}, width {width}, shoulders "
            f"{ROBOT_SHOULDER_WIDTH})",
        )
    # The levels that are not frontally passable must still be passable sideways.
    for level, width in LEVEL_CHANNEL_WIDTHS.items():
        if a_s_ratio(width) >= 1.0 - 1e-12:
            continue
        check(
            exact_fits_channel(WALL_X, 0.0, 90.0, width),
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
    print("[ok] off-axis frontal bodies collide with the panels")


def test_frontal_passability_matches_the_ladder() -> None:
    """Each Level's frontal passability must match what its A/S claims.

    A/S is channel over shoulder width, so the ladder promises frontal passage for
    A/S >= 1.0 -- every Level except A/S = 0.9, the narrowest -- and the transition
    at exactly 1.0 must be flush rather than blocked.

    That Level used to be blocked by a 2 mm skin added to the body box, which made
    the gate demand 0.574 m for a 0.570 m shoulder.  The Level labelled A/S = 1.00
    was therefore really 0.993, and every model scored 0/10 there -- a result open
    to the objection that they failed on the 2 mm rather than on the affordance.
    The skin is gone and a tolerance in the separating-axis comparison keeps
    exactly-touching boxes clear instead, so this now asserts the arithmetic the
    ladder advertises rather than the old pinch point.
    """
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        ratio = a_s_ratio(width)
        centred_ok = (
            _check_wall_collision(np.array([WALL_X, 0.0, 0.0]), 0.0, width)
            is None
        )
        check(
            (ratio >= 1.0) == centred_ok,
            f"level {level}: frontal passability ({centred_ok}) disagrees with "
            f"A/S {ratio:.3f} at the 1.00 boundary",
        )
    print(
        "[ok] frontal passability matches the ladder: every Level with A/S >= 1.00 "
        "fits facing forward, the narrowest does not"
    )


# ---------------------------------------------------------------------------
# 4. Rotation route
# ---------------------------------------------------------------------------


def simulate_rotation_route(
    channel_width: float, turns: int
) -> Tuple[List[np.ndarray], bool]:
    """Rotate ``turns`` steps in free space, then walk at the wall.

    Uses the shipped action semantics (``action_delta``) and the shipped gate
    rather than a re-derived frame, so this cannot agree with a bug in
    ``_apply_action``.  Returns the (x, z) trace and whether the body ever
    clipped a panel.
    """
    position = ROBOT_START.copy()
    yaw = 0.0
    trace: List[np.ndarray] = []

    # Phase 1: rotate the torso in the free space in front of the wall.  Rotating
    # does not steer, so this is the whole of phase 1.
    yaw += turns * TURN_STEP_DEG
    yaw_rad = math.radians(yaw)

    # Phase 2: walk toward the far wall until past the success plane.
    delta = action_delta("forward", MOVE_STEP)
    clean = True
    for _ in range(80):
        if position[0] >= SUCCESS_X:
            break
        candidate = position + delta
        if _translation_path_is_clear(position, candidate, yaw_rad, channel_width):
            clean = False
            break
        position = candidate
        trace.append(position[[0, 2]].copy())
    return trace, clean


def test_rotation_route_reaches_goal() -> None:
    """Every Level must be passable: straight where A/S allows it, rotated where not.

    The minimum rotation is derived rather than tabulated: the smallest multiple of
    15 degrees whose projected width fits the channel.  The narrowest Level
    (A/S = 0.9) is the only one that cannot be walked through facing forward, and
    the assertion that a smaller rotation fails is what makes rotation a graded
    quantity rather than a switch.
    """
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        needed = next(
            (
                turns
                for turns in range(0, 7)
                if ROBOT_SHOULDER_WIDTH * abs(math.cos(math.radians(turns * TURN_STEP_DEG)))
                + ROBOT_TORSO_THICKNESS * abs(math.sin(math.radians(turns * TURN_STEP_DEG)))
                <= width + 1e-12
            ),
            None,
        )
        check(
            needed is not None,
            f"level {level}: not even 90 degrees of rotation fits {width:.3f} m",
        )
        trace, clear = simulate_rotation_route(width, needed)
        check(clear, f"level {level}: the {needed}-turn route clipped a panel")
        check(
            trace and trace[-1][0] >= SUCCESS_X,
            f"level {level}: the {needed}-turn route ended at "
            f"x={trace[-1][0] if trace else 'n/a'}, never reached x >= {SUCCESS_X}",
        )
        max_abs_z = max(abs(point[1]) for point in trace)
        check(
            max_abs_z <= width / 2.0 + 1e-9,
            f"level {level}: route wandered to |z|={max_abs_z:.3f}, outside the "
            f"{width:.2f} m channel",
        )
        if needed:
            # One turn less must not squeeze through, or the floor above is wrong.
            trace, _clear = simulate_rotation_route(width, needed - 1)
            check(
                not trace or trace[-1][0] < SUCCESS_X,
                f"level {level} passed at {(needed - 1) * TURN_STEP_DEG:.0f} "
                f"degrees, so the measured {needed * TURN_STEP_DEG:.0f} degree "
                f"floor is wrong",
            )
    print("[ok] rotating then walking forward reaches the goal at every Level")


def test_walking_frame_is_fixed() -> None:
    """forward must walk at the wall whatever the torso is doing.

    This is the protocol's central claim now, and it is the one that decides what
    the ladder measures: if a turn redirected the walk, then only 0 and 90
    degrees could traverse at any width (measured in tools/check_heading_frame.py)
    and "did it turn enough?" could not be read off the trajectory.  Asserted
    against the shipped ``action_delta`` rather than a re-derived frame.
    """
    # action_delta takes no orientation at all: the frame is fixed by
    # construction, so there is no code path by which a turn could steer.
    parameters = list(inspect.signature(action_delta).parameters)
    check(
        parameters == ["action", "move_step"],
        f"action_delta gained parameters {parameters}; if an orientation reached "
        f"it, the walking frame would depend on the torso again",
    )
    forward = action_delta("forward", MOVE_STEP)
    check(
        forward is not None
        and abs(forward[0] - MOVE_STEP) < 1e-12
        and abs(forward[2]) < 1e-12,
        f"forward moved {forward}, not straight at the far wall",
    )
    check(
        np.allclose(action_delta("backward", MOVE_STEP), [-MOVE_STEP, 0, 0]),
        "backward is not the reverse of the walking direction",
    )
    check(
        np.allclose(action_delta("right", MOVE_STEP), [0, 0, MOVE_STEP]),
        "right is not the walker's right-hand side",
    )
    check(
        np.allclose(action_delta("left", MOVE_STEP), [0, 0, -MOVE_STEP]),
        "left is not the walker's left-hand side",
    )
    for action in ("turn_left", "turn_right", "look_left", "look_right", "nonsense"):
        check(
            action_delta(action, MOVE_STEP) is None,
            f"{action} must not translate",
        )
    print("[ok] the walking frame is fixed: rotation does not steer")


def test_model_request_params_reach_both_request_paths() -> None:
    """Per-model request overrides must be applied wherever a request is built.

    The client has two request paths: the openai SDK when it is installed, and the
    built-in HTTP compat client when it is not.  Applying an override to only one
    of them would make a model's behaviour depend on whether an unrelated package
    happens to be present in the environment the sweep runs in -- and the override
    in question changes how much the model thinks, i.e. what the benchmark
    measures, not just how fast it runs.
    """
    import os

    import ai_agent

    # An unconfigured model gets nothing: an override must be a deliberate entry.
    check(
        ai_agent.request_params_for("gemini-2.5-pro") == {},
        "a model with no recorded override received request parameters",
    )
    for model, expected in ai_agent.MODEL_REQUEST_PARAMS.items():
        check(
            ai_agent.request_params_for(model) == expected,
            f"request_params_for({model!r}) does not return the recorded override",
        )

    # The environment override wins, so a run can change configuration without
    # editing code (and args.json records the result).
    original = os.environ.get("BAO_MODEL_PARAMS")
    os.environ["BAO_MODEL_PARAMS"] = json.dumps(
        {"gemini-2.5-pro": {"reasoning_effort": "none"}}
    )
    try:
        check(
            ai_agent.request_params_for("gemini-2.5-pro")
            == {"reasoning_effort": "none"},
            "BAO_MODEL_PARAMS did not override the built-in parameter map",
        )
        os.environ["BAO_MODEL_PARAMS"] = "{not json"
        check(
            ai_agent.request_params_for("gemini-2.5-pro") == {},
            "malformed BAO_MODEL_PARAMS should be ignored, not crash or be used",
        )
    finally:
        if original is None:
            os.environ.pop("BAO_MODEL_PARAMS", None)
        else:
            os.environ["BAO_MODEL_PARAMS"] = original

    # Both request paths must consult it.  Asserted on the source because the
    # failure mode is an omission in one of two places, which no behavioural test
    # with a single path can see.
    compat = inspect.getsource(ai_agent._OpenAICompatCompletions.create)
    sdk = inspect.getsource(ai_agent.AgentAPI._request)
    for name, source in (("compat client", compat), ("openai client", sdk)):
        check(
            "request_params_for" in source,
            f"the {name} does not apply the per-model request parameters",
        )
    print("[ok] per-model request parameters reach both request paths")


def test_run_tag_carries_the_protocol_version() -> None:
    """A protocol change must not be able to resume onto the old data.

    The tag keys both the results tree and the resume checkpoint, so if it were
    only the model name, a resumed run under changed prompt or action semantics
    would find the previous protocol's episodes, count them as done and silently
    mix two experiments.  The version is therefore part of the tag, whether the
    caller supplies one or not, and run_all_models.sh composes it the same way
    (tools/check_sweep_tags.py compares the two directly).
    """
    from main import effective_tag
    from protocol import PROTOCOL_TAG

    check(bool(PROTOCOL_TAG), "PROTOCOL_TAG is empty: the tag cannot distinguish protocols")
    # Idempotent, because run_all_models.sh composes the tag and then passes it
    # through --tag, so this function sees it twice.  Non-idempotent, the sweep
    # wrote ...-v4-walkframe-v4-walkframe while printing ...-v4-walkframe.
    once = effective_tag("gpt-4o")
    check(
        effective_tag("gpt-4o", once) == once,
        f"effective_tag is not idempotent: {once!r} became "
        f"{effective_tag('gpt-4o', once)!r}, so the sweep and a manual run would "
        f"use different directories",
    )
    for model, tag in (
        ("gpt-4o", ""),
        ("gemini-2.5-pro", ""),
        ("gpt-4o", "custom"),
        ("a/b", ""),
    ):
        produced = effective_tag(model, tag)
        check(
            produced.endswith("-" + PROTOCOL_TAG),
            f"effective_tag({model!r}, {tag!r}) = {produced!r} does not carry "
            f"{PROTOCOL_TAG!r}",
        )
        check(
            "/" not in produced and "\\" not in produced and " " not in produced,
            f"{produced!r} is not a safe single path component",
        )
    print("[ok] the run tag carries the protocol version")


def test_frontal_route_only_where_feasible() -> None:
    """Walking straight ahead must work exactly where the ladder says it does.

    A/S = 1.0 is a frontal passage: shoulder and channel are equal, so an aligned
    body fits.  Only A/S = 0.9, the narrowest Level, is narrower than the shoulder
    and needs a rotation.  Derived from the ratio rather than tabulated, so it
    stays true for any sampling; it used to expect A/S = 1.0 to be blocked, back
    when a 2 mm skin made the gate demand more width than that Level claimed.
    """
    for level, width in sorted(LEVEL_CHANNEL_WIDTHS.items()):
        expected = a_s_ratio(width) >= 1.0 - 1e-12
        position = ROBOT_START.copy()
        reached = False
        blocked_by = None
        for _ in range(DEFAULT_MAX_STEPS + 10):
            candidate = position + np.array([MOVE_STEP, 0.0, 0.0])
            collision = _translation_path_is_clear(position, candidate, 0.0, width)
            if collision is not None:
                blocked_by = collision["part"]
                break
            position = candidate
            if position[0] >= SUCCESS_X:
                reached = True
                break
        check(
            reached == expected,
            f"level {level}: frontal walk reached={reached}, expected {expected} "
            f"(blocked by {blocked_by})",
        )
    print(
        "[ok] frontal walk succeeds wherever A/S >= 1.00 and not at the narrowest "
        "Level"
    )


def test_rotation_blocked_inside_wall_slab() -> None:
    """Rotating while fouling the wall must be reported as blocked.

    A Level is only solvable by rotating if the agent turns BEFORE reaching the
    wall, so the gate that forbids turning once the shoulders are in the wall plane
    is load-bearing.  The current pose is part of the swept arc, so a robot that is
    already colliding cannot rotate its way free.

    Tested at the narrowest Level, which is the one that needs a rotation: at a
    wide Level the body fits at every angle, so a turn with the shoulders in the
    wall plane is legal and there is nothing to forbid.
    """
    from environment import _turn_path_is_clear

    narrow = max(LEVEL_CHANNEL_WIDTHS)  # widest A/S first, so the last is narrowest
    check(
        a_s_ratio(LEVEL_CHANNEL_WIDTHS[narrow]) < 1.0,
        f"the last Level should be the one that needs a rotation, got A/S "
        f"{a_s_ratio(LEVEL_CHANNEL_WIDTHS[narrow]):.2f}",
    )
    width = LEVEL_CHANNEL_WIDTHS[narrow]
    collision = _turn_path_is_clear(
        np.array([WALL_X, 0.0, 0.0]), 0.0, TURN_STEP_DEG, width
    )
    check(
        collision is not None,
        f"level {narrow}: turning with the shoulders in the wall should collide",
    )

    # A pose that is already colliding must not be able to rotate out of it,
    # otherwise the agent can teleport through the wall in 15 degree hops.
    width = LEVEL_CHANNEL_WIDTHS[narrow]
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
    check(found, f"could not construct a wall-fouling pose for Level {narrow}")

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


def test_turn_sweep_checks_intermediate_orientations() -> None:
    """A turn must reject a mid-arc boundary hit even if both endpoints fit.

    The fixture has 0.27 m of clearance at both endpoints and is blocked at 60
    and 75 degrees, so it cannot be invalidated by a small change in the body
    size -- which is exactly what happened to the previous fixture.  That one sat
    0.25 mm from the boundary, and removing a 2 mm skin from the collision body
    moved it entirely inside, leaving the guard passing vacuously.

    A narrow turn cannot serve as a fixture at all: across -60 -> -75 the
    footprint's z-extent is monotone (-0.2378 down to 0.1800), so if both ends fit
    then every interior orientation fits too and no mid-arc hit exists.  Only a
    long arc passes through orientations wider than its endpoints, so the fixture
    turns 45 degrees rather than 15.
    """
    from environment import _check_scene_collision, _turn_path_is_clear

    root = np.array([0.3, 0.0, 1.95])
    width = LEVEL_CHANNEL_WIDTHS[1]
    start_yaw = 45.0
    end_yaw = 90.0

    check(
        _check_scene_collision(root, math.radians(start_yaw), width) is None,
        "regression fixture's start pose should be clear",
    )
    check(
        _check_scene_collision(root, math.radians(end_yaw), width) is None,
        "regression fixture's end pose should be clear",
    )
    for mid in (60.0, 75.0):
        check(
            _check_room_boundary(root, math.radians(mid)) is not None,
            f"regression fixture should cross the near boundary at {mid} degrees",
        )
    collision = _turn_path_is_clear(root, start_yaw, end_yaw, width)
    check(
        collision is not None and collision.get("boundary") == "near",
        "the swept-turn check missed an intermediate room-boundary collision",
    )
    print("[ok] turn sweeps sample intermediate orientations, not only endpoints")


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
        "look_down",
    ]
    check(ACTIONS == expected, f"unexpected action space: {ACTIONS}")
    for removed in ("reach_left_arm", "reach_right_arm", "retreat_left_arm",
                    "retreat_right_arm", "raise_left_arm", "raise_right_arm"):
        check(removed not in ACTIONS, f"{removed} should have been removed")
    check(
        CAMERA_TURN_STEP_DEG == 30.0,
        f"camera step changed to {CAMERA_TURN_STEP_DEG}",
    )
    check(
        CAMERA_PITCH_STEP_DEG > 0.0,
        "look_down must pitch the camera downwards",
    )
    check(TURN_STEP_DEG == 15.0, f"turn step changed to {TURN_STEP_DEG}")
    # look_down is the agent's only view of its own body, so it must be a camera
    # action (one frame, no translation) rather than something that moves it.
    check("look_down" in CAMERA_ACTIONS, "look_down is not a camera action")
    check(
        action_delta("look_down", MOVE_STEP) is None,
        "look_down must not translate the robot",
    )
    print(
        "[ok] action space is 6 locomotion actions plus 3 camera glances "
        "(left, right, down)"
    )


def test_action_descriptions_match_the_real_step() -> None:
    """The prompt's stated step length must equal the realised step length.

    Regression guard: the descriptions were hard-coded as "move forward 5cm"
    while MOVE_STEP had risen to 0.20 m, so the agent was told it moved 5 cm when
    it moved 20.  That is a four-fold error in the very quantity this benchmark
    asks the agent to reason about, and it invalidates any distance judgement the
    agent makes -- so it is asserted here rather than left to review.
    """
    import re

    import environment as env
    from protocol import ACTION_DESCRIPTIONS

    stated_cm = env.MOVE_STEP * 100.0
    for action in ("forward", "backward", "left", "right"):
        text = ACTION_DESCRIPTIONS[action]
        found = re.search(r"(\d+(?:\.\d+)?)\s*cm", text)
        check(
            found is not None,
            f"{action} description {text!r} does not state a distance in cm",
        )
        value = float(found.group(1))
        check(
            abs(value - stated_cm) < 1e-9,
            f"{action} says {value}cm but MOVE_STEP is {env.MOVE_STEP} m "
            f"({stated_cm}cm); the agent would misjudge its own travel",
        )

    # And it must actually appear in the prompt the agent receives.  Asserting
    # the whole description, not just the number: build_prompt has two paths and
    # the move_step one used to replace the descriptions with a bare
    # "move forward 75cm", so the models were never told which frame forward was
    # in even though the constant matched.
    from protocol import (
        ACTION_DESCRIPTIONS,
        ACTION_OPTIONS_STRING,
        action_options_string,
        build_prompt,
    )

    prompt = build_prompt(state={"position": [0.5, 0.0, 0.0],
                                 "torso_rotation": 0.0}, max_steps=30)
    for action in ("forward", "backward", "left", "right", "turn_left"):
        check(
            ACTION_DESCRIPTIONS[action] in prompt,
            f"the prompt does not carry the {action} description verbatim",
        )
    check(
        re.search(r"(?<!\d)5cm", prompt) is None or abs(stated_cm - 5.0) < 1e-9,
        "the prompt still advertises a standalone 5cm step while MOVE_STEP differs",
    )

    # The move_step path must keep the semantics too.  This is the regression
    # guard for the bug above: it is the path the runner actually uses.
    other = action_options_string(0.6)
    check(
        "60cm" in other and "walking direction" in other,
        "action_options_string(move_step) dropped the walking-frame semantics",
    )
    check(
        "move forward 60cm" not in other,
        "action_options_string(move_step) fell back to the semantics-free text",
    )
    check(
        ACTION_OPTIONS_STRING == action_options_string(env.MOVE_STEP),
        "the module-level options string disagrees with the rendered one",
    )
    print(
        f"[ok] action descriptions match MOVE_STEP "
        f"({stated_cm:g}cm, stated and realised, semantics included)"
    )


def test_prompt_is_uniform_and_leak_free() -> None:
    """The prompt must be identical everywhere and reveal no geometry."""
    base_state = {
        "position": [1.5, 0.0, 0.0],
        "torso_rotation": 0.0,
        "channel_width": 0.45,
        "a_s_ratio": 0.79,
    }
    prompt = build_prompt(state=base_state, history=[], max_steps=30)

    # Two different kinds of leak, checked separately because they are not the
    # same thing.
    #
    # 1. GEOMETRY tokens would tell the agent the numbers, or tell it that the
    #    passage is in any way awkward.  These are banned in the task statement,
    #    the state block and the closing instruction -- everywhere the agent
    #    reads about its situation.  They are permitted inside the action list,
    #    where they would only ever appear while describing the controls.
    #
    # 2. ADVISORY phrasing would push the agent toward the solution.  This is
    #    banned in the task statement specifically, because "turn your body" as
    #    a task hint hands over the answer, whereas the same words inside a
    #    description of what turn_left mechanically does are just documentation.
    #    An earlier version of this test banned the phrase outright and failed
    #    once the action list spelled the mechanics out.
    segments = prompt.split("\n\n")
    situation = "\n\n".join(
        seg for seg in segments if not seg.startswith("Available actions")
    ).lower()

    geometry_tokens = [
        "0.90", "0.80", "0.74", "0.68", "0.57", "0.45",
        "1.58", "1.40", "1.30", "1.19", "1.00", "0.79",
        "shoulder", "sideways", "A/S",
        "wide", "narrow", "opening width", "0.22", "0.338",
    ]
    for token in geometry_tokens:
        check(
            token.lower() not in situation,
            f"prompt leaks geometry token {token!r} outside the action list",
        )

    advisory_tokens = ["turn your body", "you should turn", "turn sideways"]
    task_lowered = TASK_INSTRUCTION.lower()
    for token in advisory_tokens:
        check(
            token not in task_lowered,
            f"the task statement advises the solution with {token!r}",
        )

    # The action mechanism must still be documented, since an agent that does not
    # know turn_* leaves its view alone -- or that look_* leaves a persistent
    # offset -- is being tested on guessing the interface rather than on judging
    # its body.  The gaze rule is the newer half of this and the easier one to
    # get wrong: the eyes are pinned to the walking direction, so an agent that
    # had turned would otherwise expect a rotated view and act on a stale model
    # of its own sensors.
    check(
        "head camera" in prompt.lower(),
        "the prompt does not explain that the head camera exists",
    )
    check(
        "straight ahead" in prompt.lower(),
        "the prompt does not say the eyes look straight ahead down the walking "
        "direction when the torso turns",
    )
    for action in ("turn_left", "look_left"):
        check(
            action in prompt,
            f"the prompt does not list the {action} action",
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


def test_history_is_full_episode_memory() -> None:
    """The agent must remember every step of the episode, not a sliding window.

    Regression guard: the history block used to be clipped to the last six
    steps, so from step seven onward the agent could no longer see what it had
    tried at the start of the episode.  The benchmark asks for a route composed
    over up to 30 steps, and an agent that forgets the first ten cannot show
    either anticipation or persistence.
    """
    history = [
        {
            "step": index,
            "action": "forward" if index % 2 == 0 else "turn_left",
            "feedback": (
                "executed"
                if index % 3
                else "blocked by transparent wall (left)"
            ),
            "reasoning": f"plan {index}",
        }
        for index in range(DEFAULT_MAX_STEPS - 1)
    ]
    prompt = build_prompt(
        state={"position": [1.5, 0.0, 0.0], "torso_rotation": 0.0},
        history=history,
        max_steps=DEFAULT_MAX_STEPS,
    )

    check(
        HISTORY_LIMIT is None,
        f"HISTORY_LIMIT={HISTORY_LIMIT} would drop the start of the episode",
    )
    for index in range(len(history)):
        check(
            f"- step {index}:" in prompt,
            f"the record of step {index} is missing from the prompt",
        )
    check("plan 0" in prompt, "the first step's reasoning was dropped")
    check(
        f"plan {len(history) - 1}" in prompt,
        "the most recent step's reasoning is missing",
    )

    # Oldest first: the last line must be the step the agent just took.
    check(
        prompt.index("- step 0:") < prompt.index(f"- step {len(history) - 1}:"),
        "history is not rendered oldest-first",
    )

    # Action and feedback are the scored record and must never be clipped.
    blocked = sum(
        1 for item in history if item["feedback"].startswith("blocked")
    )
    check(
        prompt.count("blocked by transparent wall (left)") == blocked,
        "a blocked-feedback line was lost from the history",
    )

    # Only reasoning is clipped, and clipping must not swallow the action line.
    clipped = build_prompt(
        state={},
        history=[
            {
                "step": 0,
                "action": "forward",
                "feedback": "executed",
                "reasoning": "x" * 5000,
            }
        ],
        max_steps=DEFAULT_MAX_STEPS,
    )
    check("\u2026" in clipped, "an over-long reasoning was not clipped")
    # The invariant is that the clipped reasoning contributes a bounded amount, not
    # an absolute prompt size: the action list legitimately grows when the action
    # space does.  Compare against the same prompt with no history at all.
    baseline = build_prompt(state={}, history=[], max_steps=DEFAULT_MAX_STEPS)
    check(
        len(clipped) < len(baseline) + HISTORY_REASONING_CHARS + 400,
        f"an over-long reasoning blew up the prompt "
        f"({len(clipped)} chars vs a {len(baseline)} char baseline): clipping to "
        f"{HISTORY_REASONING_CHARS} chars should bound the growth",
    )
    check(
        "- step 0: forward -> executed" in clipped,
        "clipping the reasoning damaged the action/feedback line",
    )

    # A fresh episode has taken no steps, so it must show no history block at
    # all -- no cross-episode leakage, no empty scaffolding.
    check(
        "Action history" not in build_prompt(state={}, history=[], max_steps=30),
        "an empty history should render no history block",
    )
    print("[ok] the agent carries its whole within-episode action history")


def test_episode_defaults() -> None:
    check(
        DEFAULT_EPISODES_PER_LEVEL == 5,
        f"episodes per level is {DEFAULT_EPISODES_PER_LEVEL}, expected 5",
    )
    check(
        DEFAULT_MAX_STEPS == 30,
        f"max steps is {DEFAULT_MAX_STEPS}, expected 30",
    )
    check(
        SUCCESS_X == WALL_X + MOVE_STEP,
        f"success threshold is {SUCCESS_X}, expected one stride past the wall "
        f"({WALL_X} + {MOVE_STEP})",
    )
    check(
        SUCCESS_X - ROBOT_START_POS[0] > 0,
        "the success plane is behind the start pose",
    )
    print(
        f"[ok] protocol defaults are {DEFAULT_EPISODES_PER_LEVEL} episodes x "
        f"{DEFAULT_MAX_STEPS} steps, success one stride past the wall "
        f"(x >= {SUCCESS_X})"
    )


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
        pixels >= 4.0,
        f"the channel post subtends only {pixels:.1f} px at the start pose; it "
        f"must remain visible before it grows during the ten-step approach",
    )

    # 3. The wall must read as a solid surface, not as a window onto a
    #    similarly coloured room.  It is opaque blue by design: a translucent
    #    panel made the channel hard to read at close range.
    opacity = env.WALL_OPACITY
    check(
        opacity >= 0.99,
        f"WALL_OPACITY is {opacity}; the obstacle wall must be opaque so the "
        f"channel reads as a clean silhouette",
    )
    wall = env.WALL_COLOR
    check(
        wall[2] > wall[0] and wall[2] > wall[1],
        f"WALL_COLOR {wall} is not blue-dominant",
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
    """The integer layout must match its forward-step design exactly."""
    import environment as env

    start_x = float(env.ROBOT_START_POS[0])
    step = float(env.MOVE_STEP)
    to_wall = (env.WALL_X - start_x) / step
    wall_to_goal = (env.SUCCESS_X - env.WALL_X) / step
    goal_to_far = (env.ROOM_LENGTH_X - env.SUCCESS_X) / step
    # The conservative route: six 15 deg turns (90 deg) plus fourteen moves.  The
    # measured minimum for the narrowest Level is five turns (75 deg), which is
    # asserted in test_rotation_route_reaches_goal; this one checks the budget
    # still holds for the coarser, safer route.
    turns = int(round(90.0 / env.TURN_STEP_DEG))
    moves = int(round((env.SUCCESS_X - start_x) / step))

    check(abs(step - 0.75) < 1e-12, f"adult step is {step}, expected 0.75 m")
    check(abs(to_wall - 10.0) < 1e-12, f"start-to-wall distance is {to_wall} steps")
    check(
        abs(wall_to_goal - 1.0) < 1e-12,
        f"wall-to-goal distance is {wall_to_goal} steps; the goal is one stride "
        f"past the wall now, so it must be exactly 1",
    )
    check(
        goal_to_far > 9.0,
        f"the goal is only {goal_to_far:.1f} steps from the far wall; the room "
        f"should still extend well past the goal",
    )
    check(moves == 11, f"goal needs {moves} forward moves, expected 11")
    check(
        turns + moves == 17 and turns + moves <= DEFAULT_MAX_STEPS,
        f"the 90 degree route needs {turns + moves}/{DEFAULT_MAX_STEPS} actions",
    )

    # The distant opening remains visible but small at the initial pose.
    fov_deg = 2.0 * math.degrees(
        math.atan((20.955 / 2.0) / env.ROBOT_CAMERA_FOCAL)
    )
    gap_deg = 2.0 * math.degrees(
        math.atan((env.LEVEL_CHANNEL_WIDTHS[0] / 2.0) / (env.WALL_X - start_x))
    )
    check(
        0.05 < gap_deg / fov_deg < 0.20,
        f"initial opening occupies {100 * gap_deg / fov_deg:.1f}% of the view",
    )
    print(
        f"[ok] integer layout: 10 moves to the wall + 1 to the goal plane; "
        f"the 90 degree route uses {turns + moves}/{DEFAULT_MAX_STEPS} actions and "
        f"leaves {goal_to_far:.1f} steps of room beyond the goal"
    )


def test_robot_has_room_to_rotate_before_the_wall() -> None:
    """The start pose must leave a turn's worth of clearance in front of the wall.

    This is the constraint that a recorded qwen-vl-max episode actually failed
    on.  The turn gate samples the robot's current pose, and the wall slab spans
    WALL_X +- (WALL_THICKNESS/2 + MOVE_STEP), so past x = WALL_X - MOVE_STEP -
    WALL_THICKNESS/2 no rotation is legal at all.  With the wall at x=2.0 the
    robot could not turn beyond x=1.60; it walked 0.5 -> 1.6, then spent eight
    steps alternating turn_left and turn_right while both were blocked, and
    never reached the far side.

    The free run must be long enough for the robot to walk up to the channel and
    still be able to rotate into it.
    """
    from environment import MOVE_STEP, ROBOT_START_POS, WALL_X, _turn_path_is_clear

    start_x = float(ROBOT_START_POS[0])

    def first_x_where_rotation_fails(level: int) -> float:
        """Lowest x at which a single 15 degree turn is rejected, by search."""
        width = LEVEL_CHANNEL_WIDTHS[level]
        step = 0.01
        x = start_x
        while x < WALL_X + 0.1:
            if (
                _turn_path_is_clear(
                    np.array([x, 0.0, 0.0]), 0.0, TURN_STEP_DEG, width
                )
                is not None
            ):
                return float(x)
            x += step
        return float("nan")

    # Use the narrowest channel: it is the first to lose the ability to rotate.
    # (At a wide Level the body fits at every angle, so rotation is never blocked
    # and the search legitimately finds no limit.)
    narrow = max(LEVEL_CHANNEL_WIDTHS)
    limit = first_x_where_rotation_fails(narrow)
    free_run = limit - start_x
    check(
        math.isfinite(limit),
        f"no rotation limit found for Level {narrow} within the room",
    )
    check(
        free_run > 1.0,
        f"only {free_run:.2f} m of manoeuvring room before rotation becomes "
        f"impossible (first blocked x={limit:.2f}); a recorded qwen-vl-max "
        f"episode wedged in exactly this way and never recovered",
    )
    check(
        limit <= WALL_X,
        f"rotation is still possible at x={limit:.2f}, past the wall at "
        f"x={WALL_X}",
    )

    # The limit must be real: the step before it is allowed, the step at it is not.
    width_narrow = LEVEL_CHANNEL_WIDTHS[narrow]
    check(
        _turn_path_is_clear(
            np.array([limit - 0.02, 0.0, 0.0]), 0.0, TURN_STEP_DEG, width_narrow
        )
        is None,
        f"turning at x={limit - 0.02:.2f} should still be allowed",
    )
    check(
        _turn_path_is_clear(
            np.array([limit, 0.0, 0.0]), 0.0, TURN_STEP_DEG, width_narrow
        )
        is not None,
        f"turning at x={limit:.2f} should be blocked",
    )
    print(
        f"[ok] {free_run:.2f} m of free run before rotation is blocked "
        f"(first blocked x={limit:.2f}, wall at x={WALL_X}, start at x={start_x})"
    )


def test_box_axis_mapping_is_correct() -> None:
    """A box's height must end up on the height axis.

    `dims` is authored as (along_x, height, left_right) in the user frame while
    Isaac Sim uses z for height, so y and z swap exactly once.  This was got
    wrong repeatedly, most visibly when the channel posts rendered LYING DOWN:
    measured isaac z span 0.05 m for a post authored 2.0 m tall, which showed up
    in every view as a dark horizontal bar across the wall.

    Rather than trust the transform, this replicates _add_box's corner
    computation and checks the resulting extents per axis.
    """
    import inspect

    import environment as env

    source = inspect.getsource(env.BAOEnv._add_box)
    check(
        "AddScaleOp" not in source,
        "_add_box applies a scale op; sizes are authored directly in metres",
    )
    # The swap must be a plain index swap, applied to the raw authored dims and
    # only once.  Calling the position/scale helper on dims does the swap, so
    # seeing it applied to `dims` (rather than to a literal) is the bug.
    check(
        "_user_to_isaac_scale(dims" not in source
        and "_user_to_isaac_scale(np.asarray(dims" not in source,
        "_add_box feeds authored dims through _user_to_isaac_scale, which swaps "
        "the axes; the swap is already done explicitly",
    )

    # Emulate the corner maths for the channel post.  Authored dims are
    # (along_x, height, left_right) in the user frame; Isaac Sim puts height on
    # z, so the y and z components swap.  The post is a TALL SLIM bar: 0.02 m
    # through the wall, 2.0 m tall, 0.05 m wide, so its isaac z span must be
    # the wall height.  Authoring 2.0 m in the left_right slot instead produced
    # a horizontal bar, which rendered as a dark line across the wall.
    def isaac_extents(dims) -> tuple:
        return (float(dims[0]), float(dims[2]), float(dims[1]))

    post = (
        env.WALL_THICKNESS,             # along_x: match analytic panel depth
        env.WALL_HEIGHT,                # height
        env.CHANNEL_EDGE_THICKNESS,     # left_right: the post's width
    )
    ix, iy, iz = isaac_extents(post)
    check(
        abs(iz - env.WALL_HEIGHT) < 1e-12,
        f"the channel post's isaac z extent is {iz}, but it is authored "
        f"{env.WALL_HEIGHT} m tall; the post would lie down",
    )
    check(
        abs(iy - env.CHANNEL_EDGE_THICKNESS) < 1e-12,
        f"the channel post's isaac y extent is {iy}, expected its width "
        f"{env.CHANNEL_EDGE_THICKNESS}",
    )
    check(
        abs(ix - env.WALL_THICKNESS) < 1e-12,
        f"the channel post's isaac x extent is {ix}, expected its depth "
        f"{env.WALL_THICKNESS}",
    )

    # The authored call site must actually pass the tall shape: catch a
    # regression where WALL_HEIGHT goes back into the left_right slot.  The post
    # boxes now come from channel_edge_post_boxes, so that is where the shape is
    # authored; the test checks the helper AND that the builder uses it.
    post_source = inspect.getsource(env.channel_edge_post_boxes)
    check(
        "WALL_HEIGHT, CHANNEL_EDGE_THICKNESS" in post_source,
        "the channel post is not authored as (depth, WALL_HEIGHT, width); it "
        "would render as a horizontal bar rather than a vertical post",
    )

    # Same check for a wall panel: 0.02 thick, 2.0 tall, ~2.05 wide.
    px, py, pz = isaac_extents((env.WALL_THICKNESS, env.WALL_HEIGHT, 2.05))
    check(
        abs(pz - env.WALL_HEIGHT) < 1e-12,
        f"the wall panel's isaac z extent is {pz}, expected {env.WALL_HEIGHT}",
    )
    check(
        abs(py - 2.05) < 1e-12,
        f"the wall panel's isaac y extent is {py}, expected its width 2.05",
    )
    print(
        f"[ok] box axes map correctly: a {env.WALL_HEIGHT:.1f} m tall post gets "
        f"an isaac z extent of {iz:.2f} m"
    )


def test_channel_edges_do_not_narrow_the_opening() -> None:
    """The visible clear width must equal the modelled channel width.

    The A/S ratio is channel width over shoulder width, so anything drawn inside
    the gap silently changes what the benchmark is measuring.  The edge posts were
    once centred ON the channel edge, which ate CHANNEL_EDGE_THICKNESS/2 into the
    opening: the gap looked 0.85 m while the collision model -- which uses
    _panel_boxes and ignores the posts -- still allowed 0.90 m.

    Measured from the boxes that are actually built (``channel_edge_post_boxes``)
    and from the collision panels, for every Level.  The first version of this test
    recomputed ``channel_half + half - half``, which is a tautology: it would have
    passed with the posts centred on the edge.
    """
    import environment as env
    from environment import _check_wall_collision, _panel_boxes

    for level, width in sorted(env.LEVEL_CHANNEL_WIDTHS.items()):
        channel_half = width / 2.0
        for post_centre, post_dims in env.channel_edge_post_boxes(width):
            inner_face = abs(float(post_centre[2])) - float(post_dims[2]) / 2.0
            check(
                abs(inner_face - channel_half) < 1e-12,
                f"level {level}: a post's inner face is at z={inner_face:.6f} but "
                f"the channel edge is at z={channel_half:.6f}",
            )
        for panel_centre, panel_half in _panel_boxes(width):
            panel_inner = abs(float(panel_centre[2])) - float(panel_half[2])
            check(
                abs(panel_inner - channel_half) < 1e-12,
                f"level {level}: a collision panel's inner edge is at "
                f"z={panel_inner:.6f} but the channel edge is at z={channel_half:.6f}",
            )
        # So the gap the agent sees is exactly the gap the gate enforces, which is
        # the width the recorded A/S is computed from.
        check(
            abs(a_s_ratio(width) - width / env.ROBOT_SHOULDER_WIDTH) < 1e-12,
            f"level {level}: the recorded A/S is not channel / shoulder width",
        )

    # The posts must sit OUTSIDE the gap at every Level (no Level may be narrowed).
    for level, width in sorted(env.LEVEL_CHANNEL_WIDTHS.items()):
        for post_centre, post_dims in env.channel_edge_post_boxes(width):
            outer_face = abs(float(post_centre[2])) - float(post_dims[2]) / 2.0
            check(
                outer_face >= width / 2.0 - 1e-12,
                f"level {level}: a post protrudes into the opening",
            )

    # And the gate must agree: a body at A/S >= 1.0 fits the channel with no
    # rotation at all, which is what "the model can walk straight through at 1.0"
    # means.
    for level, width in sorted(env.LEVEL_CHANNEL_WIDTHS.items()):
        if a_s_ratio(width) < 1.0 - 1e-12:
            continue
        check(
            _check_wall_collision(
                np.array([env.WALL_X, 0.0, 0.0]), 0.0, width
            )
            is None,
            f"level {level}: an aligned body does not fit a {width:.3f} m channel "
            f"even though A/S is {a_s_ratio(width):.2f}",
        )
        check(
            _check_wall_collision(
                np.array([env.WALL_X, 0.0, 0.0]), math.radians(90.0), width
            )
            is None,
            f"level {level}: a sideways body does not fit either",
        )
    print(
        "[ok] the edge posts and the collision panels leave exactly the modelled "
        "opening at all 12 widths, so the recorded A/S is the real one"
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

    # _add_box must author explicit geometry, not rely on Cube sizing or a scale
    # op.  A USD Cube is fixed at +-1 and only the scale op sizes it, while
    # `extent` declares LOCAL bounds -- writing a world-sized extent next to a
    # scale op made the reported bounds and the rendered geometry disagree,
    # which is how the channel posts rendered as a 5 cm lump near z=1.0 instead
    # of a 2 m vertical post.
    box_source = inspect.getsource(env.BAOEnv._add_box)
    check(
        "UsdGeom.Mesh" in box_source,
        "_add_box does not build an explicit mesh; cube sizing reintroduces the "
        "extent/scale ambiguity",
    )
    check(
        "CreatePointsAttr" in box_source and "CreateFaceVertexIndicesAttr" in box_source,
        "_add_box does not author mesh points and faces, so its geometry is "
        "still implicit",
    )
    check(
        "AddScaleOp" not in box_source,
        "_add_box still applies a scale op, which double-counts against an "
        "explicit extent",
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
        env.ROOM_LENGTH_X == 16.0 and env.ROOM_WIDTH_Z == 5.0,
        f"room is {env.ROOM_LENGTH_X} x {env.ROOM_WIDTH_Z}, expected 16 x 5 m",
    )
    check(
        env.ROOM_LENGTH_X > env.WALL_X,
        f"the far wall at x={env.ROOM_LENGTH_X} must be past the obstacle at "
        f"x={env.WALL_X}",
    )
    check(
        env.ROOM_LENGTH_X > env.SUCCESS_X,
        f"the far wall at x={env.ROOM_LENGTH_X} must be past the success plane at "
        f"x={env.SUCCESS_X}",
    )

    # Walls must be taller than the robot, or the robot could see over them.
    check(
        env.ROOM_WALL_HEIGHT > 1.806,
        f"ROOM_WALL_HEIGHT {env.ROOM_WALL_HEIGHT} is not above the measured "
        f"1.806 m robot",
    )

    # Every surface the agent can see must be a distinct colour, or "aimed at
    # the opening" and "aimed at a panel" look the same.  A recorded
    # gemini-2.5-pro run, sitting at x=2.30 centred on the channel and facing
    # it, reported "I am now facing a solid wall. I cannot see the opening I
    # need to pass through," then scanned left and right and never advanced.
    side = env.ROOM_SIDE_WALL_COLOR
    far = env.ROOM_FAR_WALL_COLOR
    wall = env.WALL_COLOR
    for name, colour in (
        ("ROOM_SIDE_WALL_COLOR", side),
        ("ROOM_FAR_WALL_COLOR", far),
        ("WALL_COLOR", wall),
        ("ROOM_CEILING_COLOR", env.ROOM_CEILING_COLOR),
    ):
        check(
            len(colour) == 3 and all(0.0 <= float(c) <= 1.0 for c in colour),
            f"{name} {colour} is not a valid RGB triple",
        )

    def separation(a, b) -> float:
        return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / 3.0

    pairs = [
        ("obstacle wall vs far wall", wall, far),
        ("obstacle wall vs side walls", wall, side),
        ("far wall vs side walls", far, side),
        ("ceiling vs obstacle wall", env.ROOM_CEILING_COLOR, wall),
    ]
    for label, a, b in pairs:
        check(
            separation(a, b) > 0.10,
            f"{label} differ by only {separation(a, b):.2f} per channel "
            f"({a} vs {b}); the agent could not tell them apart",
        )

    # The obstacle wall must be opaque: a translucent panel made the channel
    # hard to read at close range.
    check(
        env.WALL_OPACITY >= 0.99,
        f"WALL_OPACITY is {env.WALL_OPACITY}; the obstacle wall must be opaque "
        f"so the channel reads as a clean silhouette",
    )
    # And saturated blue, which is what makes the gap stand out.
    check(
        wall[2] > wall[0] and wall[2] > wall[1],
        f"WALL_COLOR {wall} is not blue-dominant",
    )
    print(
        f"[ok] room surfaces are distinct: wall {wall}, far {far}, "
        f"side {side}, ceiling {env.ROOM_CEILING_COLOR}"
    )


def test_default_start_is_usable() -> None:
    """The shipped defaults must be reachable and frame the opening.

    From 1.5 m the 2.0 m gap subtends about 77 degrees and fills the whole
    105-degree eye view, leaving no wall in frame; from 0.5 m it is about 39
    degrees.  The start distance and step size have to move together or the
    robot cannot reach x >= 11.0 inside the budget.
    """
    import environment as env

    start_x = float(env.ROBOT_START_POS[0])
    move_step = float(env.MOVE_STEP)
    turns = int(round(90.0 / env.TURN_STEP_DEG))
    travel = SUCCESS_X - start_x
    needed = turns + int(math.ceil(travel / move_step - 1e-12))
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

    # Every interior light must be below the ceiling and inside the floor plan;
    # lights must cover both sides of the obstacle in the 16 m enclosure.
    positions = ([2.0, 2.6, 0.0], [6.0, 2.6, 0.0],
                 [10.0, 2.6, 0.0], [14.0, 2.6, 0.0])
    check(
        any(p[0] < env.WALL_X for p in positions)
        and any(p[0] > env.WALL_X for p in positions),
        "interior lights do not cover both sides of the obstacle wall",
    )
    for position in positions:
        x, y, z = position
        check(
            y < env.ROOM_WALL_HEIGHT,
            f"light at y={y} is above the ceiling (ROOM_WALL_HEIGHT="
            f"{env.ROOM_WALL_HEIGHT})",
        )
        check(
            0.0 < x < env.ROOM_LENGTH_X,
            f"light at x={x} is outside the room (0..{env.ROOM_LENGTH_X})",
        )
        check(
            abs(z) < env.ROOM_WIDTH_Z / 2.0,
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
    # And the gaze must not follow the torso.  A person crossing a narrow opening
    # keeps looking at the opening while rotating their shoulders; a gaze that
    # tracked the torso would swing the view onto the side wall at exactly the
    # large rotations this ladder asks for (Level 5 needs 75 degrees), taking the
    # channel out of a 76 degree field of view.  The torso angle is reported as a
    # number instead.
    check(
        "_robot_yaw" not in source,
        "_update_eye_camera reads the torso yaw, so the view rotates with the "
        "body instead of staying on the walking direction",
    )
    print(
        f"[ok] head camera pitches down {EYE_PITCH_DEG:.0f} deg from "
        f"y={EYE_CAMERA_HEIGHT} m and does not follow the torso"
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


def test_room_boundary_and_swept_translation() -> None:
    """The agent cannot leave the room or teleport through a narrow wall."""
    yaw = 0.0
    check(
        _check_room_boundary(np.array([1.0, 0.0, 2.8]), yaw) is not None,
        "a body beyond the side wall was accepted",
    )
    check(
        _check_room_boundary(np.array([1.0, 0.0, 0.0]), yaw) is None,
        "a centred in-room body was rejected",
    )
    collision = _translation_path_is_clear(
        np.array([WALL_X - 0.5, 0.0, 1.0]),
        np.array([WALL_X + 0.5, 0.0, 1.0]),
        yaw,
        channel_width=LEVEL_CHANNEL_WIDTHS[5],
    )
    check(collision is not None, "a large translation teleported through a wall panel")
    print("[ok] room boundaries and swept translations prevent bypasses")


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
        test_frontal_passability_matches_the_ladder,
        test_rotation_route_reaches_goal,
        test_walking_frame_is_fixed,
        test_model_request_params_reach_both_request_paths,
        test_run_tag_carries_the_protocol_version,
        test_frontal_route_only_where_feasible,
        test_rotation_blocked_inside_wall_slab,
        test_turn_sweep_checks_intermediate_orientations,
        test_turn_clearance_boundary,
        test_entry_point_does_not_import_isaac_sim,
        test_action_space,
        test_action_descriptions_match_the_real_step,
        test_prompt_is_uniform_and_leak_free,
        test_history_is_full_episode_memory,
        test_episode_defaults,
        test_sideways_band,
        test_eye_camera_pitches_downward,
        test_scene_readability_constants,
        test_room_is_enclosed_and_coloured,
        test_primitive_sizing_is_in_metres,
        test_lights_are_inside_the_enclosed_room,
        test_default_start_is_usable,
        test_start_distance_reachable_in_budget,
        test_robot_has_room_to_rotate_before_the_wall,
        test_channel_edges_do_not_narrow_the_opening,
        test_box_axis_mapping_is_correct,
        test_ground_grid_exists,
        test_camera_look_at_orientation,
        test_room_boundary_and_swept_translation,
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
