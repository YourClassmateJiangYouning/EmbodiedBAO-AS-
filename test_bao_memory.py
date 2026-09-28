"""Offline verification of the Stage 2 measures (memory and habit experiment).

Stage 2 scores a model on how close each attempt came to fitting through an
opening, how many actions it wasted, what kind of attempt it was, and whether the
run of attempts looks like insight, gradual improvement, perseveration or noise.
Those quantities decide the result of the experiment, so they are checked here
against an independent geometric model and against the simulator's own collision
gate -- before any Stage 2 episode is run, because a criterion written after seeing
the data is not a criterion.

Isaac Sim is not needed.  In fact the whole point of most of these checks is that
``memory_metrics`` must not need it:

1. **Constants** -- every scene constant the module restates equals the
   simulator's, so the duplicate definitions cannot drift apart.
2. **Geometry** -- ``needed_width`` agrees with an independent projection of the
   rotated body rectangle, with the closed form in the design document
   (0.6110 cos(theta - 21.1 degrees)), and with the simulator's analytic gate.
3. **Cost** -- ``optimal_steps`` is 16 at A/S 0.80, 11 at A/S 1.10 and 15 at
   A/S 0.90, which is what a breadth-first search over the real action space
   measured (``lab_logs/verify_optimal_steps.py``).
4. **Gap** -- the sign of ``gap`` agrees with the collision gate at the wall, and
   a round that never reached the wall gets no value at all rather than an
   invented one.
5. **Attempt labels** -- the priority order of the design document's table, and
   the two consequences that follow from it.
6. **Curve criteria** -- the pre-registered insight / gradual / perseveration /
   oscillation rules on hand-worked curves, and the threshold as a parameter so
   the sensitivity analysis cannot silently change them.
7. **Isolation** -- ``memory_metrics`` imports nothing but the standard library, so
   it can be imported before ``SimulationApp`` exists.

Run with a plain Python interpreter:

    python test_bao_memory.py
"""

from __future__ import annotations

import ast
import math
import os
import sys

import numpy as np

import memory_metrics
from environment import (
    MOVE_STEP,
    ROBOT_SHOULDER_WIDTH,
    ROBOT_START_POS,
    ROBOT_TORSO_THICKNESS,
    SUCCESS_X,
    TURN_STEP_DEG,
    WALL_X,
    _check_wall_collision,
    _translation_path_is_clear,
    level_channel_width,
)

MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_metrics.py")


class Failure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def step(x: float, theta: float) -> dict:
    """One per-step record, with the two fields the metrics read."""
    return {"position_x": float(x), "torso_rotation": float(theta)}


# ---------------------------------------------------------------------------
# 1. The restated constants
# ---------------------------------------------------------------------------


def test_constants_match_the_simulator() -> None:
    """The module restates the simulator's constants; they must not drift."""
    expected = {
        "SHOULDER_WIDTH_M": float(ROBOT_SHOULDER_WIDTH),
        "TORSO_THICKNESS_M": float(ROBOT_TORSO_THICKNESS),
        "MOVE_STEP_M": float(MOVE_STEP),
        "START_X_M": float(ROBOT_START_POS[0]),
        "SUCCESS_X_M": float(SUCCESS_X),
        "TURN_STEP_DEG": float(TURN_STEP_DEG),
    }
    for name, value in expected.items():
        actual = getattr(memory_metrics, name)
        check(
            abs(actual - value) < 1e-12,
            f"memory_metrics.{name} = {actual}, environment says {value}",
        )

    # The two Stage 2 widths are A/S x shoulder width, and the wide one is Level 9.
    check(
        abs(memory_metrics.WIDTH_LEARNING_M - ROBOT_SHOULDER_WIDTH * 0.80) < 1e-12,
        f"learning width {memory_metrics.WIDTH_LEARNING_M} is not 0.80 x the shoulder width",
    )
    check(
        abs(memory_metrics.WIDTH_PROBE_M - ROBOT_SHOULDER_WIDTH * 1.10) < 1e-12,
        f"probe width {memory_metrics.WIDTH_PROBE_M} is not 1.10 x the shoulder width",
    )
    check(
        abs(memory_metrics.WIDTH_PROBE_M - level_channel_width(9)) < 1e-9,
        "the probe width should be exactly the width of Level 9 (A/S 1.10)",
    )
    # Stage 2's learning width is deliberately harder than anything already
    # measured, so it cannot be one of the ladder's widths.
    nearest = min(abs(level_channel_width(level) - memory_metrics.WIDTH_LEARNING_M) for level in range(12))
    check(
        nearest > 0.05,
        f"the learning width is only {nearest:.3f} m from a ladder width; it was meant to be harder than all of them",
    )
    print(
        "[ok] every restated constant equals the simulator's, the probe width is Level 9, "
        f"and the learning width is {nearest:.2f} m away from the nearest Level"
    )


def test_door_window_is_always_reached_by_a_body_at_the_wall() -> None:
    """The x >= 7.0 window must be populated by any body that walks at the wall.

    Measured rather than derived: for every torso angle, sweep a body forward until
    the simulator's own gate stops it, and take the earliest stop over all angles.
    If that were below 7.0, ``gap`` could be None for a round that did walk at the
    wall, and the curve would be missing values for a reason that has nothing to do
    with the model.
    """
    width = memory_metrics.WIDTH_LEARNING_M
    earliest = None
    for theta_deg in range(0, 181):
        yaw = math.radians(float(theta_deg))
        x = memory_metrics.DOOR_X_M
        while x <= WALL_X:
            if _check_wall_collision(np.array([x, 0.0, 0.0]), yaw, width) is not None:
                earliest = x if earliest is None else min(earliest, x)
                break
            x += 0.01
    check(
        earliest is not None,
        "no torso angle is stopped by the wall at all, so the gate is not being exercised",
    )
    check(
        earliest > memory_metrics.DOOR_X_M,
        f"a body is stopped at x = {earliest:.3f}, inside the x >= "
        f"{memory_metrics.DOOR_X_M} window, so a round that walked at the wall could "
        "still have no gap value",
    )
    print(
        f"[ok] the door window opens at x = {memory_metrics.DOOR_X_M}, and the earliest the wall "
        f"stops a body is x = {earliest:.2f}, a {earliest - memory_metrics.DOOR_X_M:.2f} m margin"
    )


# ---------------------------------------------------------------------------
# 2. Geometry
# ---------------------------------------------------------------------------


def test_needed_width_matches_an_independent_rectangle_projection() -> None:
    """Re-derive the projection from the body's four rotated corners."""

    def corner_extent(theta_deg: float) -> float:
        half_shoulder = memory_metrics.SHOULDER_WIDTH_M / 2.0
        half_torso = memory_metrics.TORSO_THICKNESS_M / 2.0
        theta = math.radians(theta_deg)
        zs = []
        for along_shoulder in (-half_shoulder, half_shoulder):
            for along_torso in (-half_torso, half_torso):
                # Local axes: the shoulder spans z, the torso thickness spans x.
                zs.append(
                    -along_torso * math.sin(theta) + along_shoulder * math.cos(theta)
                )
        return max(zs) - min(zs)

    angles = [float(value) for value in range(-180, 181, 5)] + [21.1, 21.12, 7.0, 62.83]
    worst = 0.0
    for theta in angles:
        worst = max(worst, abs(corner_extent(theta) - memory_metrics.needed_width(theta)))
    check(worst < 1e-12, f"projection disagrees with the corner model by {worst:.3e} m")

    # Symmetry: a body turned -30 degrees is as wide as one turned +30.
    for theta in (0.0, 15.0, 30.0, 45.0, 75.0, 90.0):
        check(
            abs(memory_metrics.needed_width(theta) - memory_metrics.needed_width(-theta)) < 1e-12,
            f"needed_width is not symmetric about zero at {theta} degrees",
        )
    print(f"[ok] needed_width matches an independent corner projection over {len(angles)} angles")


def test_needed_width_matches_the_design_document_formula() -> None:
    """The document's closed form: 0.6110 cos(theta - 21.1 degrees).

    Valid on the first quadrant only, which is where the document's table lives
    (0, 21.1 and 75 degrees).  Beyond it the projection takes absolute values, so
    the cosine goes on to negative values while the projection cannot; the rest of
    the circle is covered by the two symmetries checked below rather than by the
    same formula.
    """
    amplitude = memory_metrics.max_needed_width()
    phase = memory_metrics.max_needed_angle_deg()
    check(abs(amplitude - 0.6110) < 5e-5, f"amplitude is {amplitude:.4f} m, the document says 0.6110")
    check(abs(phase - 21.1) < 0.05, f"peak angle is {phase:.2f} degrees, the document says 21.1")

    worst = 0.0
    for half_degrees in range(0, 181):
        theta = half_degrees / 2.0
        closed = amplitude * math.cos(math.radians(theta - phase))
        worst = max(worst, abs(closed - memory_metrics.needed_width(theta)))
    check(worst < 1e-12, f"closed form differs from the projection by {worst:.3e} m")
    check(
        amplitude * math.cos(math.radians(150.0 - phase)) < 0.0,
        "the cosine form goes negative past the first quadrant, which is why it is not used there",
    )

    # The projection cannot go negative, and it repeats by reflection: a body
    # turned backwards or mirrored is exactly as wide across the opening.
    for theta in (5.0, 30.0, 62.5, 88.0, 90.0):
        reference = memory_metrics.needed_width(theta)
        for mirrored in (-theta, 180.0 - theta, 180.0 + theta, 360.0 - theta):
            check(
                abs(memory_metrics.needed_width(mirrored) - reference) < 1e-12,
                f"needed_width({mirrored}) is not the mirror of needed_width({theta})",
            )
    check(
        memory_metrics.needed_width(180.0) > 0.0,
        "the projection can never be negative, whatever the torso angle",
    )

    # The three values the design document quotes by hand.
    for theta, expected in ((0.0, 0.570), (21.1, 0.6110), (75.0, 0.3600)):
        actual = memory_metrics.needed_width(theta)
        check(
            abs(actual - expected) < 1e-4,
            f"needed_width({theta}) = {actual:.4f}, the document says {expected:.4f}",
        )
    # Turning a little is worse than not turning: the frontal body is 0.570 m and
    # 15 degrees takes it to 0.608 m.
    check(
        memory_metrics.needed_width(15.0) > memory_metrics.needed_width(0.0),
        "a 15 degree turn should make the body wider across the opening, not narrower",
    )
    print(
        f"[ok] needed_width matches 0.6110 cos(theta - {phase:.1f} deg) exactly, "
        "and a small turn is wider than no turn"
    )


# ---------------------------------------------------------------------------
# 3. Cost
# ---------------------------------------------------------------------------


def test_optimal_steps_match_the_breadth_first_search() -> None:
    """16 / 11 / 15, as measured over the real action space, not as inferred."""
    check(memory_metrics.forward_steps_to_success() == 11, "the route needs 11 forward steps")
    cases = (
        (memory_metrics.WIDTH_LEARNING_M, 5, 16),
        (memory_metrics.WIDTH_PROBE_M, 0, 11),
        (level_channel_width(11), 4, 15),
        (level_channel_width(10), 0, 11),
    )
    for width, turns, steps in cases:
        actual_turns = memory_metrics.min_turns(width)
        actual_steps = memory_metrics.optimal_steps(width)
        check(
            actual_turns == turns,
            f"width {width:.3f} m needs {actual_turns} turns, expected {turns}",
        )
        check(
            actual_steps == steps,
            f"width {width:.3f} m needs {actual_steps} steps, expected {steps}",
        )
    # Every Level that can be walked through facing forward costs the same 11.
    for level in range(11):
        check(
            memory_metrics.optimal_steps(level_channel_width(level)) == 11,
            f"Level {level} should be passable frontally in 11 steps",
        )
    # A/S 1.00 is the exact geometric limit: the body fits with nothing to spare.
    check(
        memory_metrics.min_turns(level_channel_width(10)) == 0,
        "A/S 1.00 must still be passable without turning",
    )
    try:
        memory_metrics.min_turns(0.20)
    except ValueError:
        pass
    else:
        raise Failure("a 0.20 m opening is narrower than the body and must be rejected")
    print("[ok] optimal_steps is 16 at A/S 0.80, 11 at A/S 1.10 and 15 at A/S 0.90, as searched")


def test_excess_uses_each_phases_own_optimum() -> None:
    """One constant for both phases would score a perfect probe round as -5."""
    check(memory_metrics.excess(16, memory_metrics.WIDTH_LEARNING_M) == 0, "16 is optimal at A/S 0.80")
    check(memory_metrics.excess(22, memory_metrics.WIDTH_LEARNING_M) == 6, "22 steps waste 6 at A/S 0.80")
    check(memory_metrics.excess(11, memory_metrics.WIDTH_PROBE_M) == 0, "11 is optimal at A/S 1.10")
    check(memory_metrics.excess(30, memory_metrics.WIDTH_LEARNING_M) == 14, "a capped round wastes 14")
    check(memory_metrics.excess(30, memory_metrics.WIDTH_PROBE_M) == 19, "a capped round wastes 19")
    check(
        11 - memory_metrics.optimal_steps(memory_metrics.WIDTH_LEARNING_M) == -5,
        "the wrong single-constant formula should be recognisable as a negative excess",
    )
    print("[ok] excess subtracts the optimum of its own phase, so a capped round is 14 or 19, never negative")


# ---------------------------------------------------------------------------
# 4. Gap
# ---------------------------------------------------------------------------


def test_gap_sign_agrees_with_the_collision_gate() -> None:
    """gap <= 0 must mean exactly what the simulator means by "it can get through".

    The check is a translation through the opening, not a pose at the wall plane.
    A pose can be legal for a body that cannot get through: turned 45 degrees, the
    part of the body inside the wall slab is a sliver under 3 cm wide around one
    edge's midpoint, while the two corners that decide the passage are 0.28 m off
    axis on either side of the wall.  Fitting means the whole body crosses, which
    is what the path gate measures and what the metric is a model of.
    """
    widths = (
        memory_metrics.WIDTH_LEARNING_M,
        memory_metrics.WIDTH_PROBE_M,
        level_channel_width(10),
        level_channel_width(11),
    )
    start = np.array([memory_metrics.DOOR_X_M, 0.0, 0.0])
    target = np.array([SUCCESS_X + 0.05, 0.0, 0.0])
    checked = 0
    for width in widths:
        for theta in range(0, 91, 15):
            blocked = (
                _translation_path_is_clear(start, target, math.radians(theta), width)
                is not None
            )
            gap = memory_metrics.gap_for([step(SUCCESS_X, theta)], width)
            check(
                (gap <= 0.0) != blocked,
                f"width {width:.3f} m at {theta} degrees: gap {gap:+.4f} says "
                f"{'fits' if gap <= 0 else 'does not fit'} but the gate says "
                f"{'blocked' if blocked else 'clear'}",
            )
            checked += 1
    print(f"[ok] the sign of gap agrees with the passage gate in all {checked} width/angle cases")


def test_gap_is_none_when_the_round_never_reached_the_wall() -> None:
    """No value is returned, and nothing is invented in its place."""
    stayed = [step(0.5, 0.0)] * 30
    check(memory_metrics.gap_for(stayed, memory_metrics.WIDTH_LEARNING_M) is None, "no wall, no gap")
    check(not memory_metrics.reached_door(stayed), "a body at x = 0.5 is not at the door")

    stopped_short = [step(0.5 + 0.75 * index, 0.0) for index in range(9)]  # up to x = 6.5
    check(
        memory_metrics.gap_for(stopped_short, memory_metrics.WIDTH_LEARNING_M) is None,
        "stopping at x = 6.5 says nothing about fitting through the opening",
    )
    check(
        not memory_metrics.reached_door(stopped_short),
        "x = 6.5 is outside the window",
    )

    # The specific values that must not be substituted for a missing measurement.
    frontal = memory_metrics.needed_width(0.0) - memory_metrics.WIDTH_LEARNING_M
    worst = memory_metrics.max_needed_width() - memory_metrics.WIDTH_LEARNING_M
    check(
        memory_metrics.gap_for(stayed, memory_metrics.WIDTH_LEARNING_M) not in (frontal, worst),
        "a round that never approached the opening must not be given a posture's value",
    )

    at_door = [step(7.25, 0.0)]
    value = memory_metrics.gap_for(at_door, memory_metrics.WIDTH_LEARNING_M)
    check(value is not None and abs(value - frontal) < 1e-12, "x = 7.25 is inside the window")
    check(
        memory_metrics.gap_for([step(7.0, 0.0)], memory_metrics.WIDTH_LEARNING_M) is not None,
        "the window includes x = 7.0 itself",
    )
    print(
        f"[ok] gap is None rather than {frontal:+.3f} or {worst:+.3f} when a round never "
        "reached the wall, and x = 7.0 itself is inside the window"
    )


def test_gap_reports_the_best_posture_of_the_round() -> None:
    """The minimum over the steps at the wall, not the first or the last."""
    steps = [
        step(7.25, 0.0),
        step(7.25, 75.0),
        step(8.0, 30.0),
    ]
    width = memory_metrics.WIDTH_LEARNING_M
    expected = memory_metrics.needed_width(75.0) - width
    actual = memory_metrics.gap_for(steps, width)
    check(
        abs(actual - expected) < 1e-12,
        f"gap {actual:+.4f} is not the best posture of the round ({expected:+.4f})",
    )
    check(actual < 0.0, "turning to 75 degrees does fit an A/S 0.80 opening")
    print("[ok] gap takes the round's best posture at the wall, so a late improvement counts")


# ---------------------------------------------------------------------------
# 5. Attempt labels
# ---------------------------------------------------------------------------


def test_strategy_labels_follow_the_priority_order() -> None:
    """The design document's table, first match wins."""
    cases = (
        ((0, 0, None), memory_metrics.FRONTAL),
        ((2, 0, None), memory_metrics.FRONTAL),
        ((3, 0, None), memory_metrics.SIDEWAYS),
        ((5, 2, 5.0), memory_metrics.SIDEWAYS),  # sideways outranks the turn's position
        ((0, 1, 5.0), memory_metrics.ROT_MID),   # a turn outranks FRONTAL
        ((0, 1, 4.0), memory_metrics.ROT_MID),
        ((0, 1, 3.99), memory_metrics.ROT_EARLY),
        ((0, 1, 6.99), memory_metrics.ROT_MID),
        ((0, 1, 7.0), memory_metrics.ROT_LATE),
        ((0, 2, 7.25), memory_metrics.ROT_LATE),
        ((1, 3, 7.25), memory_metrics.ROT_LATE),  # three turns is not sideways any more
        ((9, 4, 3.0), memory_metrics.ROT_EARLY),
    )
    for (lateral, turns, first_x), expected in cases:
        actual = memory_metrics.strategy_label(lateral, turns, first_x)
        check(
            actual == expected,
            f"{lateral} lateral, {turns} turn(s), first at x = {first_x}: "
            f"got {actual}, expected {expected}",
        )

    for bad, why in (
        ((0, 1, None), "a turn with no position"),
        ((1, 0, 5.0), "a position with no turn"),
        ((-1, 0, None), "a negative count"),
        ((0, -2, None), "a negative count"),
    ):
        try:
            memory_metrics.strategy_label(*bad)
        except ValueError:
            pass
        else:
            raise Failure(f"{bad} should be rejected as inconsistent: {why}")
    print(f"[ok] the priority order of {len(cases)} attempt shapes is enforced, and inconsistent counts are rejected")


def test_frontal_means_never_rotated() -> None:
    """The consequence of the priority order, pinned because it is surprising.

    Every round with at least one turn matches a ROT_* rule, so FRONTAL can only be
    reached with no turn at all, and the six names in the table describe five
    reachable kinds of round.  That is worth knowing before SetIndex is read
    against a random baseline: the baseline is 1/5, not the 1/6 the document quotes,
    and at a 0.6 threshold the difference changes nothing.
    """
    seen = set()
    for lateral in range(0, 13):
        for turns in range(0, 9):
            positions = [None] if turns == 0 else [0.0, 3.99, 4.0, 6.99, 7.0, 12.0]
            for first_x in positions:
                label = memory_metrics.strategy_label(lateral, turns, first_x)
                seen.add(label)
                if label == memory_metrics.FRONTAL:
                    check(
                        turns == 0,
                        f"FRONTAL with {turns} turn(s): the priority order is not being applied",
                    )
    check(
        memory_metrics.MIXED not in seen,
        "MIXED became reachable; the design document's table and this module now disagree",
    )
    check(len(seen) == 5, f"expected five reachable labels, saw {sorted(seen)}")
    print(f"[ok] FRONTAL implies no turn and only five of the six names are reachable: {sorted(seen)}")


# ---------------------------------------------------------------------------
# 6. Curve criteria
# ---------------------------------------------------------------------------


def test_improvement_margin_is_below_the_finest_real_change() -> None:
    """The 1 mm margin cannot mistake a real improvement for noise."""
    achievable = sorted(
        {round(memory_metrics.needed_width(float(theta)), 9) for theta in range(0, 181, 15)}
    )
    finest = min(b - a for a, b in zip(achievable, achievable[1:]))
    check(
        finest > memory_metrics.GAP_IMPROVEMENT_MARGIN_M,
        f"the closest two achievable required widths differ by {finest:.4f} m, which is "
        f"within the {memory_metrics.GAP_IMPROVEMENT_MARGIN_M} m improvement margin",
    )
    check(
        memory_metrics.GAP_IMPROVEMENT_MARGIN_M > 1e-9,
        "the margin must be far above floating-point noise",
    )
    print(
        f"[ok] the finest change a 15 degree action can make is {finest * 1000:.1f} mm, "
        f"above the {memory_metrics.GAP_IMPROVEMENT_MARGIN_M * 1000:.0f} mm improvement margin"
    )


def test_curve_statistics_are_right_on_hand_worked_curves() -> None:
    """Every criterion, on a curve whose answer is known by inspection."""
    front = memory_metrics.FRONTAL
    early = memory_metrics.ROT_EARLY
    mid = memory_metrics.ROT_MID

    # One jump from "does not fit" to "fits", and stays there.
    insight_gaps = [0.114] * 4 + [-0.096] * 8
    insight_labels = [front] * 4 + [early] * 8
    insight_passed = [False] * 4 + [True] * 8
    stats = memory_metrics.classify_curve(insight_gaps, insight_labels, insight_passed)
    check(abs(float(stats["I"]) - 0.210) < 1e-9, f"I = {stats['I']}, expected 0.210")
    check(abs(float(stats["S"]) - 1.0) < 1e-9, f"S = {stats['S']}, expected 1.0")
    check(stats["jump_round"] == 4, f"the jump is at round index {stats['jump_round']}, expected 4")
    check(stats["label"] == memory_metrics.INSIGHT, f"classified {stats['label']}, expected insight")

    # Same total improvement, spread evenly over six rounds.
    gradual_gaps = [0.1516 - 0.04952 * index for index in range(6)]
    gradual = memory_metrics.classify_curve(gradual_gaps, [mid] * 6, [True] * 6)
    check(abs(float(gradual["I"]) - 0.2476) < 1e-3, f"I = {gradual['I']}, expected about 0.248")
    check(
        abs(float(gradual["S"]) - 0.2) < 0.01,
        f"S = {gradual['S']}, expected about 0.2",
    )
    check(gradual["improving_rounds"] == 5, f"{gradual['improving_rounds']} improvements, expected 5")
    check(abs(float(gradual["rho"]) + 1.0) < 1e-9, f"rho = {gradual['rho']}, expected -1")
    check(gradual["label"] == memory_metrics.GRADUAL, f"classified {gradual['label']}, expected gradual")

    # Never improves on its own first posture.
    flat = memory_metrics.classify_curve([0.114] * 12, [memory_metrics.SIDEWAYS] * 12, [False] * 12)
    check(float(flat["I"]) == 0.0, f"I = {flat['I']}, expected 0")
    check(float(flat["set_index"]) == 1.0, f"SetIndex = {flat['set_index']}, expected 1")
    check(
        flat["label"] == memory_metrics.PERSEVERATION,
        f"classified {flat['label']}, expected perseveration",
    )

    # A big first drop that does not hold, and no repeated strategy.
    wobble_gaps = [0.114, -0.05, 0.10, -0.05, 0.10, -0.05]
    wobble_labels = [early, mid, early, mid, early, mid]
    wobble = memory_metrics.classify_curve(wobble_gaps, wobble_labels, [False] * 6)
    check(abs(float(wobble["S"]) - 1.0) < 1e-9, f"S = {wobble['S']}, expected 1.0")
    check(
        wobble["label"] == memory_metrics.OSCILLATING,
        f"classified {wobble['label']}, expected oscillating: the drop did not hold",
    )

    # Never reached the wall at all.
    absent = memory_metrics.classify_curve([None] * 12, [front] * 12, [False] * 12)
    check(absent["never_approached"] is True, "an all-missing curve never approached the opening")
    check(
        absent["label"] == memory_metrics.NEVER_APPROACHED,
        f"classified {absent['label']}, expected never_approached",
    )
    check(absent["I"] is None, "I has no value when no round reached the wall")
    print("[ok] insight, gradual, perseveration, oscillation and never-approached curves are classified correctly")


def test_a_late_first_measurement_is_used_as_the_baseline() -> None:
    """I is measured from the first round with a value, and the wait is reported."""
    gaps = [None, None, 0.114, 0.05, -0.096, -0.096]
    labels = [memory_metrics.FRONTAL] * 4 + [memory_metrics.ROT_MID] * 2
    stats = memory_metrics.classify_curve(gaps, labels, [False, False, False, False, True, True])
    check(stats["approach_latency"] == 2, f"approach_latency = {stats['approach_latency']}, expected 2")
    check(stats["valued_rounds"] == 4, f"{stats['valued_rounds']} valued rounds, expected 4")
    check(
        abs(float(stats["I"]) - 0.210) < 1e-9,
        f"I = {stats['I']}, expected 0.210 measured from round 3, not round 1",
    )
    check(
        stats["label"] == memory_metrics.INSIGHT,
        f"classified {stats['label']}, expected insight",
    )
    print("[ok] a curve whose first rounds never reached the wall is baselined on the first that did")


def test_insight_threshold_is_a_parameter() -> None:
    """The 0.5 / 0.6 / 0.7 sensitivity analysis must be three calls, not a rerun."""
    gaps = [0.20, 0.0875, -0.05, -0.05, -0.05, -0.05]
    labels = [memory_metrics.FRONTAL] * 2 + [memory_metrics.ROT_MID] * 4
    passed = [False, False, True, True, True, True]
    middle = memory_metrics.classify_curve(gaps, labels, passed)
    check(abs(float(middle["S"]) - 0.55) < 1e-9, f"S = {middle['S']}, expected exactly 0.55")
    check(
        middle["label"] == memory_metrics.OSCILLATING,
        f"at S >= 0.6 a 0.55 drop is not insight, but it classified as {middle['label']}",
    )
    loose = memory_metrics.classify_curve(gaps, labels, passed, insight_s_min=0.5)
    strict = memory_metrics.classify_curve(gaps, labels, passed, insight_s_min=0.7)
    check(loose["label"] == memory_metrics.INSIGHT, f"at S >= 0.5 it should be insight, got {loose['label']}")
    check(strict["label"] == memory_metrics.OSCILLATING, f"at S >= 0.7 it should not be, got {strict['label']}")
    print("[ok] the insight threshold is a parameter: the same curve is insight at 0.5 and not at 0.6 or 0.7")


def test_set_index_and_rank_correlation() -> None:
    """The two numbers behind the perseveration and gradual rules."""
    labels = ["A", "A", "B", "A"]
    passed = [False, False, True, False]
    check(
        abs(float(memory_metrics.set_index(labels, passed)) - 0.5) < 1e-12,
        f"SetIndex = {memory_metrics.set_index(labels, passed)}, expected 1 of 2 failed rounds repeated",
    )
    check(
        memory_metrics.set_index(["A", "B", "A"], [True, True, True]) is None,
        "SetIndex has no value when no round failed with another round after it",
    )
    try:
        memory_metrics.set_index(labels, passed[:2])
    except ValueError:
        pass
    else:
        raise Failure("SetIndex accepted mismatched labels and outcomes")

    check(
        abs(float(memory_metrics.spearman_rho([0.3, 0.2, 0.1, 0.0])) + 1.0) < 1e-12,
        "a strictly falling curve has rho -1",
    )
    check(
        abs(float(memory_metrics.spearman_rho([0.0, 0.1, 0.2, 0.3])) - 1.0) < 1e-12,
        "a strictly rising curve has rho +1",
    )
    tied = memory_metrics.spearman_rho([0.3, 0.2, 0.1, 0.1, 0.0])
    check(
        abs(float(tied) + 0.974679) < 1e-5,
        f"rho with one tie is {tied}, expected about -0.9747 (average ranks)",
    )
    check(
        memory_metrics.spearman_rho([0.1, 0.1, 0.1, 0.1]) is None,
        "a flat curve has no rank correlation to report",
    )
    check(
        memory_metrics.spearman_rho([0.1, 0.0]) is None,
        "two points are not enough for a rank correlation",
    )
    check(
        memory_metrics.spearman_rho([0.1, 0.0, None, 0.05]) is not None,
        "missing rounds should be skipped, not turn the whole curve into a missing value",
    )
    print("[ok] SetIndex and the rank correlation agree with hand-worked cases, including ties and gaps")


# ---------------------------------------------------------------------------
# 7. Isolation
# ---------------------------------------------------------------------------


def test_module_does_not_import_the_environment() -> None:
    """It has to be importable before SimulationApp exists.

    Importing ``environment`` before ``SimulationApp`` latches
    ``environment._HAS_ISAAC_SIM`` to False for the life of the process, after which
    the scene can never be built and the run exits 0 having done nothing.
    """
    with open(MODULE_PATH, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=MODULE_PATH)
    allowed = {"math", "typing", "__future__"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    check(
        imported <= allowed,
        f"memory_metrics imports {sorted(imported - allowed)}; the metrics layer must "
        f"import only {sorted(allowed)}, at any level, so it can be loaded before "
        "SimulationApp exists",
    )
    check(
        not hasattr(memory_metrics, "np"),
        "memory_metrics pulled in numpy; the metrics must stay dependency-free",
    )
    print(f"[ok] memory_metrics imports only {sorted(imported)} and can be loaded before SimulationApp")


def main() -> int:
    tests = [
        test_constants_match_the_simulator,
        test_door_window_is_always_reached_by_a_body_at_the_wall,
        test_needed_width_matches_an_independent_rectangle_projection,
        test_needed_width_matches_the_design_document_formula,
        test_optimal_steps_match_the_breadth_first_search,
        test_excess_uses_each_phases_own_optimum,
        test_gap_sign_agrees_with_the_collision_gate,
        test_gap_is_none_when_the_round_never_reached_the_wall,
        test_gap_reports_the_best_posture_of_the_round,
        test_strategy_labels_follow_the_priority_order,
        test_frontal_means_never_rotated,
        test_improvement_margin_is_below_the_finest_real_change,
        test_curve_statistics_are_right_on_hand_worked_curves,
        test_a_late_first_measurement_is_used_as_the_baseline,
        test_insight_threshold_is_a_parameter,
        test_set_index_and_rank_correlation,
        test_module_does_not_import_the_environment,
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
    print(f"all {len(tests)} memory-metrics checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
