"""The measures the memory-and-habit experiment is scored on (Stage 2).

Stage 2 (see ``STAGE23_DESIGN.md``) gives a model 12 attempts at an opening it
cannot walk through facing forward (A/S 0.80), lets it write itself a note between
attempts, and then moves it to an opening it *can* walk through facing forward
(A/S 1.10) for 5 attempts, to see whether the rotation it has just learned has
become a habit.

This module is only the arithmetic over records that already exist: how wide the
body is across the opening at a given torso angle, how close a round came to
fitting, how many actions it wasted, what kind of attempt it was, and whether a
sequence of rounds looks like sudden insight, gradual improvement, perseveration
or noise.  It imports nothing but ``math`` and ``typing`` -- no Isaac Sim, no API,
no file access -- which is what lets the pre-registered quantities be written,
tested and frozen before a single Stage 2 episode is run, and lets a clone
re-derive them on any machine.

The constants below restate the simulator's.  That duplication is deliberate: a
module-level ``import environment`` latches ``environment._HAS_ISAAC_SIM`` to False
for the life of the process (see the README), and this module has to be importable
by the runner *before* ``SimulationApp`` exists.  ``test_bao_memory.py`` asserts
every one of them equals the simulator's, so they cannot drift apart.

Three places where the design document leaves a choice are marked
``OPERATIONALISED``, both here and at the code that implements them.  Each is the
tightest reading that adds no new constant:

* an **improving round** is one whose gap is lower than the previous valued
  round's by more than 1 mm.  The finest change the 15 degree action set can
  produce is 3.9 mm (0.6036 m to 0.6075 m of required width), so 1 mm cannot
  mistake a real change for noise and floating-point error cannot reach it;
* **no trend**, in the perseveration rule, is read as "not the downward trend
  that gradual learning requires", i.e. ``rho > -0.7``, which reuses the
  pre-registered threshold instead of inventing a second one;
* **the strategy changed qualitatively at the jump** is read as the jump round
  being the first rotating strategy to follow a non-rotating one, which is the
  behavioural change the criterion is there to capture.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Scene constants (must equal environment.py's; asserted in test_bao_memory.py)
# ---------------------------------------------------------------------------
# The body is an oriented rectangle.  Facing forward its shoulder width spans the
# opening; turned 90 degrees its torso thickness does.  The width it occupies
# across a vertical opening at torso angle theta is therefore
#
#     needed(theta) = SHOULDER * |cos theta| + TORSO * |sin theta|,
#
# which peaks at 21.1 degrees -- 0.6110 m, *wider* than the frontal 0.570 m,
# because a slightly turned body is broader across the opening than an aligned one.
# That is why "just turn a little" is not a solution, and why the ladder's
# narrowest Level needs 75 degrees rather than a nudge.
SHOULDER_WIDTH_M = 0.570
TORSO_THICKNESS_M = 0.220
MOVE_STEP_M = 0.75
START_X_M = 0.5
SUCCESS_X_M = 8.75
TURN_STEP_DEG = 15.0

# Only the last metre before the wall is scored.  The aperture is at x = 8.0 with a
# 0.02 m slab, so its near face is at 7.99, and sweeping every torso angle forward
# against the simulator's own gate puts the earliest a body can be stopped at
# x = 7.79 (measured in test_bao_memory.py).  That number is measured rather than
# derived from the half-extents because the corner that is widest across the
# opening is not the corner furthest along x, and the passage is decided by the
# former: at 62 degrees the body's 0.3033 m of x half-extent belongs to a corner
# only 0.0367 m off axis, while the corner 0.2309 m off axis sits at 0.2000 m.  A
# step before x = 7.0 therefore says nothing about whether the body would fit
# through the opening, which is the point of the window.
DOOR_X_M = 7.0

# The two widths Stage 2 uses, as A/S x shoulder width.  A/S 0.80 is not on the
# 12-Level ladder (which stops at 0.9) precisely because it is harder than the
# hardest Level already measured; A/S 1.10 is on it, as Level 9.
WIDTH_LEARNING_M = 0.456
WIDTH_PROBE_M = 0.627

# ---------------------------------------------------------------------------
# Attempt classification (STAGE23_DESIGN.md 6.1); the priority order is the rule
# ---------------------------------------------------------------------------
SIDEWAYS = "SIDEWAYS"
ROT_LATE = "ROT_LATE"
ROT_MID = "ROT_MID"
ROT_EARLY = "ROT_EARLY"
FRONTAL = "FRONTAL"
MIXED = "MIXED"
STRATEGY_LABELS = (SIDEWAYS, ROT_LATE, ROT_MID, ROT_EARLY, FRONTAL, MIXED)

# "It rotated at all", used to decide whether a change of label is a change of
# kind rather than a move between two ways of not rotating.
ROTATING_LABELS = frozenset({ROT_LATE, ROT_MID, ROT_EARLY})

ROT_MID_MIN_X_M = 4.0
ROT_LATE_MIN_X_M = 7.0
SIDEWAYS_MIN_LATERAL = 3
SIDEWAYS_MAX_TURNS = 2
FRONTAL_MAX_TURNS = 1
FRONTAL_MAX_LATERAL = 2

# ---------------------------------------------------------------------------
# Curve criteria (STAGE23_DESIGN.md 6.2), pre-registered
# ---------------------------------------------------------------------------
INSIGHT_S_MIN = 0.6
GRADUAL_S_MAX = 0.35
GRADUAL_RHO_MAX = -0.7
GRADUAL_IMPROVING_MIN = 3
SET_INDEX_MIN = 0.6
INSIGHT_STABLE_ROUNDS = 2

# OPERATIONALISED: what counts as an improvement, see the module docstring.
GAP_IMPROVEMENT_MARGIN_M = 0.001
# A rank correlation over two or three points is arithmetic, not evidence.
SPEARMAN_MIN_ROUNDS = 3

INSIGHT = "insight"
GRADUAL = "gradual"
PERSEVERATION = "perseveration"
OSCILLATING = "oscillating"
NEVER_APPROACHED = "never_approached"

# The floating-point tolerance used when asking whether a width fits.  The
# simulator's own gate is a separating-axis test with a 2e-7 m tolerance, so a body
# that touches the panels exactly counts as clear; 1e-12 here is only there so that
# a width derived as A/S x shoulder cannot fail its own boundary by one ulp.
FIT_TOLERANCE_M = 1e-12

# The yaw band the threshold study calls "sideways": the torso turned far enough that
# the narrow dimension faces the opening.  Restated here, like the scene constants, so
# a Stage 2 round can be compared with a Stage 1 episode without this module importing
# environment; test_bao_memory.py pins both against the simulator's own band and
# against experiments._is_sideways_yaw.
SIDEWAYS_YAW_MIN_DEG = 45.0
SIDEWAYS_YAW_MAX_DEG = 135.0


def fold_yaw(yaw_deg: float) -> float:
    """Fold a torso yaw into [0, 180] so +/- rotations behave symmetrically."""
    yaw = abs(float(yaw_deg)) % 360.0
    return float(360.0 - yaw if yaw > 180.0 else yaw)


def is_sideways_yaw(yaw_deg: float) -> bool:
    """Whether a torso angle sits in the band the threshold study scores as sideways."""
    return SIDEWAYS_YAW_MIN_DEG <= fold_yaw(yaw_deg) <= SIDEWAYS_YAW_MAX_DEG


# ---------------------------------------------------------------------------
# Geometry and cost
# ---------------------------------------------------------------------------


def needed_width(theta_deg: float) -> float:
    """Width the body occupies across the opening at this torso angle, in metres."""
    theta = math.radians(theta_deg)
    return SHOULDER_WIDTH_M * abs(math.cos(theta)) + TORSO_THICKNESS_M * abs(math.sin(theta))


def max_needed_width() -> float:
    """The widest the body can be across an opening: 0.6110 m, at 21.1 degrees.

    The rotated rectangle's projection is the norm of (SHOULDER, TORSO), so this is
    exact rather than a maximum over samples.
    """
    return math.hypot(SHOULDER_WIDTH_M, TORSO_THICKNESS_M)


def max_needed_angle_deg() -> float:
    """The torso angle at which the body is widest across the opening."""
    return math.degrees(math.atan2(TORSO_THICKNESS_M, SHOULDER_WIDTH_M))


def forward_steps_to_success() -> int:
    """Forward translations from the start pose to the success plane: 11."""
    return math.ceil((SUCCESS_X_M - START_X_M) / MOVE_STEP_M)


def min_turns(width_m: float) -> int:
    """Fewest 15 degree turns that make the body fit this opening.

    0 at A/S 1.10 and above, 4 at A/S 0.90, 5 at A/S 0.80.  Searched rather than
    inverted, because the projection is not monotone in the angle: it rises to a
    peak at 21.1 degrees before falling, so "the smallest angle that fits" is not a
    closed form over the 15 degree samples.
    """
    for turns in range(0, 7):
        if needed_width(turns * TURN_STEP_DEG) <= width_m + FIT_TOLERANCE_M:
            return turns
    raise ValueError(
        f"no torso angle between 0 and 90 degrees fits a {width_m:.3f} m opening; "
        f"the narrowest the body ever gets is {needed_width(90.0):.3f} m"
    )


def optimal_steps(width_m: float) -> int:
    """Fewest actions that reach the goal through this opening.

    16 at A/S 0.80 (11 forward + 5 turns), 11 at A/S 1.10 (11 forward, no turn),
    15 at A/S 0.90.  Verified against a breadth-first search over the real action
    space in ``lab_logs/verify_optimal_steps.py``.
    """
    return forward_steps_to_success() + min_turns(width_m)


def excess(total_steps: int, width_m: float) -> int:
    """Actions spent beyond the shortest route through this width.

    Defined for every round, including failed ones: a failed round runs to the
    30-step cap, so its excess is always ``30 - optimal_steps(W)`` -- 14 in the
    learning phase and 19 in the probe phase.  That constant carries no information
    about efficiency, which is why the "did it get more efficient" question is
    asked of passing rounds only (STAGE23_DESIGN.md 7, Q2).
    """
    return int(total_steps) - optimal_steps(width_m)


# ---------------------------------------------------------------------------
# Per-round measures
# ---------------------------------------------------------------------------


def reached_door(steps: Sequence[Dict[str, object]], door_x: float = DOOR_X_M) -> bool:
    """Whether any step of this round was at the wall (x >= door_x)."""
    return any(float(step["position_x"]) >= door_x for step in steps)


def gap_for(
    steps: Sequence[Dict[str, object]],
    width_m: float,
    door_x: float = DOOR_X_M,
) -> Optional[float]:
    """How far this round's best attempt at the opening was from fitting.

    The minimum of ``needed(theta) - width_m`` over the steps at the wall, so a
    negative value means the body fitted at some point and zero means it fitted
    exactly.  ``steps`` are per-step records with ``position_x`` and
    ``torso_rotation``.

    ``None`` is returned -- and that is a result, not a missing value to fill in --
    when the round never reached the wall at all (STAGE23_DESIGN.md 6.2).  A round
    that never approached the opening has not measured posture at the opening, and
    every imputation would invent one: Stage 1's 660 episodes show the case is rare
    (8 episodes, 1.2%, and 0 of 660 ever failed to issue a forward) but it is not
    impossible, and the analysis treats a valueless round as its own outcome rather
    than as a bad attempt.
    """
    at_door = [
        needed_width(float(step["torso_rotation"])) - width_m
        for step in steps
        if float(step["position_x"]) >= door_x
    ]
    return min(at_door) if at_door else None


def strategy_label(
    n_lateral: int,
    n_turn: int,
    first_turn_x: Optional[float],
) -> str:
    """What kind of attempt this was (STAGE23_DESIGN.md 6.1), first match wins.

    ``n_lateral`` counts ``left`` and ``right``, ``n_turn`` counts ``turn_left``
    and ``turn_right``, and glances are not counted anywhere.  ``first_turn_x`` is
    the x of the first turn in the round, or ``None`` if there was no turn.

    Consequence of the priority order, and the reason the design document states
    it: every round with at least one turn matches one of the three ``ROT_*``
    rules, so ``FRONTAL`` means "never rotated" rather than "rotated a little",
    and the six names describe five distinguishable kinds of round.
    """
    if n_lateral < 0 or n_turn < 0:
        raise ValueError(f"action counts cannot be negative: {n_lateral} lateral, {n_turn} turns")
    if n_turn == 0 and first_turn_x is not None:
        raise ValueError(f"first_turn_x={first_turn_x} given for a round with no turn")
    if n_turn > 0 and first_turn_x is None:
        raise ValueError(f"{n_turn} turn(s) but no first_turn_x to place them at")

    if n_lateral >= SIDEWAYS_MIN_LATERAL and n_turn <= SIDEWAYS_MAX_TURNS:
        return SIDEWAYS
    if n_turn > 0:
        if float(first_turn_x) >= ROT_LATE_MIN_X_M:
            return ROT_LATE
        if float(first_turn_x) >= ROT_MID_MIN_X_M:
            return ROT_MID
        return ROT_EARLY
    if n_turn <= FRONTAL_MAX_TURNS and n_lateral <= FRONTAL_MAX_LATERAL:
        return FRONTAL
    return MIXED


# ---------------------------------------------------------------------------
# Curve statistics (STAGE23_DESIGN.md 6.2), over the learning phase only
# ---------------------------------------------------------------------------


def _valued(gaps: Sequence[Optional[float]]) -> List[Tuple[int, float]]:
    return [(index, float(value)) for index, value in enumerate(gaps) if value is not None]


def insight_index(gaps: Sequence[Optional[float]]) -> Optional[float]:
    """``I``: how far the first measured round was from the best one reached.

    Measured from the first round that has a gap at all, not from round 1, because
    a round that never reached the wall has no value to start from (the skipped
    rounds are reported separately, as ``approach_latency``).  ``I <= 0`` means the
    run never improved on its own starting posture, which is the perseveration
    criterion; ``None`` means no round in this phase reached the wall.
    """
    valued = _valued(gaps)
    if not valued:
        return None
    return valued[0][1] - min(value for _, value in valued)


def largest_drop(gaps: Sequence[Optional[float]]) -> Optional[float]:
    """``D``: the biggest single-round fall in gap, over adjacent measured rounds.

    A rise is not a drop, so this floors at 0.  ``None`` when fewer than two rounds
    have a value, because then no drop exists to measure.
    """
    valued = _valued(gaps)
    if len(valued) < 2:
        return None
    return max(0.0, max(valued[i][1] - valued[i + 1][1] for i in range(len(valued) - 1)))


def jump_round(gaps: Sequence[Optional[float]]) -> Optional[int]:
    """Index of the round the largest drop landed on, or ``None``."""
    valued = _valued(gaps)
    if len(valued) < 2:
        return None
    best = 0.0
    landed: Optional[int] = None
    for i in range(len(valued) - 1):
        drop = valued[i][1] - valued[i + 1][1]
        if drop > best:
            best = drop
            landed = valued[i + 1][0]
    return landed


def abruptness(gaps: Sequence[Optional[float]]) -> Optional[float]:
    """``S = D / I``: the share of the whole improvement that arrived in one round.

    ``None`` unless ``I > 0``: with no room to improve, the ratio is undefined
    rather than infinite, and the document says such a run is classified before
    this is asked.
    """
    index = insight_index(gaps)
    drop = largest_drop(gaps)
    if index is None or drop is None or index <= 0.0:
        return None
    return drop / index


# OPERATIONALISED: 1 mm counts as an improvement, see the module docstring.
def improving_rounds(
    gaps: Sequence[Optional[float]],
    margin_m: float = GAP_IMPROVEMENT_MARGIN_M,
) -> int:
    """How many rounds improved on the previous measured round."""
    valued = _valued(gaps)
    return sum(
        1 for i in range(len(valued) - 1) if valued[i + 1][1] < valued[i][1] - margin_m
    )


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman_rho(gaps: Sequence[Optional[float]]) -> Optional[float]:
    """Rank correlation between round number and gap, over the measured rounds.

    Negative means the gap is falling as the rounds go on, which is what gradual
    learning looks like.  Ties are averaged, and ``None`` is returned for fewer
    than three measured rounds or for a flat curve: a correlation with no variation
    to correlate is not a number.
    """
    valued = _valued(gaps)
    if len(valued) < SPEARMAN_MIN_ROUNDS:
        return None
    rounds = [float(index) for index, _ in valued]
    values = [value for _, value in valued]
    ranked_rounds = _average_ranks(rounds)
    ranked_values = _average_ranks(values)
    n = len(values)
    mean_r = sum(ranked_rounds) / n
    mean_v = sum(ranked_values) / n
    cov = sum((ranked_rounds[i] - mean_r) * (ranked_values[i] - mean_v) for i in range(n))
    var_r = sum((ranked_rounds[i] - mean_r) ** 2 for i in range(n))
    var_v = sum((ranked_values[i] - mean_v) ** 2 for i in range(n))
    if var_r <= 0.0 or var_v <= 0.0:
        return None
    return cov / math.sqrt(var_r * var_v)


def set_index(labels: Sequence[str], passed: Sequence[bool]) -> Optional[float]:
    """``P(strategy(r+1) == strategy(r) | round r failed)``.

    The rate at which a failed round is followed by the same kind of attempt, which
    is what perseveration looks like from the outside.  The design document quotes a
    random baseline of 1/6; with the current six names describing five reachable
    kinds of round it is 1/5, which changes nothing at the 0.6 threshold.  ``None``
    when no round failed with another round after it.
    """
    if len(labels) != len(passed):
        raise ValueError(f"{len(labels)} labels against {len(passed)} outcomes")
    pairs = [
        (labels[i], labels[i + 1])
        for i in range(len(labels) - 1)
        if passed[i] is False or passed[i] == 0
    ]
    if not pairs:
        return None
    return sum(1 for before, after in pairs if before == after) / len(pairs)


def curve_stats(
    gaps: Sequence[Optional[float]],
    labels: Sequence[str],
    passed: Sequence[bool],
) -> Dict[str, object]:
    """Every number the pre-registered criteria are computed from.

    ``gaps``, ``labels`` and ``passed`` are the learning-phase rounds, oldest
    first.  The returned keys are ``I``, ``D``, ``S``, ``rho``,
    ``improving_rounds``, ``set_index``, ``jump_round``, ``valued_rounds``,
    ``approach_latency`` and ``never_approached``.
    """
    if len(labels) != len(gaps) or len(passed) != len(gaps):
        raise ValueError(
            f"{len(gaps)} gaps, {len(labels)} labels and {len(passed)} outcomes "
            "must be the same rounds"
        )
    valued = _valued(gaps)
    first = valued[0][0] if valued else None
    return {
        "I": insight_index(gaps),
        "D": largest_drop(gaps),
        "S": abruptness(gaps),
        "rho": spearman_rho(gaps),
        "improving_rounds": improving_rounds(gaps),
        "set_index": set_index(labels, passed),
        "jump_round": jump_round(gaps),
        "valued_rounds": len(valued),
        "approach_latency": first,
        "never_approached": first is None,
    }


def _stable_below_zero(gaps: Sequence[Optional[float]], start: int, count: int) -> bool:
    """Whether the first ``count`` measured rounds from ``start`` all fit."""
    seen: List[float] = []
    for index in range(start, len(gaps)):
        value = gaps[index]
        if value is None:
            continue
        seen.append(float(value))
        if len(seen) == count:
            break
    return len(seen) == count and all(value <= 0.0 for value in seen)


# OPERATIONALISED: "changed qualitatively" means it started rotating, see the
# module docstring.
def _strategy_changed_kind(labels: Sequence[str], index: int) -> bool:
    if index <= 0 or index >= len(labels):
        return False
    return labels[index] in ROTATING_LABELS and labels[index - 1] not in ROTATING_LABELS


# OPERATIONALISED: "no trend" means not the downward trend gradual learning
# requires, see the module docstring.
def _trendless(stats: Dict[str, object]) -> bool:
    rho = stats["rho"]
    return rho is None or float(rho) > GRADUAL_RHO_MAX


def classify_curve(
    gaps: Sequence[Optional[float]],
    labels: Sequence[str],
    passed: Sequence[bool],
    insight_s_min: float = INSIGHT_S_MIN,
) -> Dict[str, object]:
    """Classify one run's learning curve, in the pre-registered order.

    Returns ``curve_stats`` plus ``label`` and ``insight_s_min``.  The order is the
    document's: insight, then gradual, then perseveration, then everything else.
    ``insight_s_min`` is a parameter so the 0.5 / 0.6 / 0.7 sensitivity analysis is
    three calls over the same data rather than a rerun.
    """
    stats = curve_stats(gaps, labels, passed)
    stats["insight_s_min"] = insight_s_min
    label: str
    if stats["never_approached"]:
        label = NEVER_APPROACHED
    elif (
        stats["S"] is not None
        and float(stats["S"]) >= insight_s_min
        and stats["jump_round"] is not None
        and _stable_below_zero(gaps, int(stats["jump_round"]), INSIGHT_STABLE_ROUNDS)
        and _strategy_changed_kind(labels, int(stats["jump_round"]))
    ):
        label = INSIGHT
    elif (
        stats["S"] is not None
        and float(stats["S"]) <= GRADUAL_S_MAX
        and int(stats["improving_rounds"]) >= GRADUAL_IMPROVING_MIN
        and stats["rho"] is not None
        and float(stats["rho"]) <= GRADUAL_RHO_MAX
    ):
        label = GRADUAL
    elif (stats["I"] is not None and float(stats["I"]) <= 0.0) or (
        stats["set_index"] is not None
        and float(stats["set_index"]) >= SET_INDEX_MIN
        and _trendless(stats)
    ):
        label = PERSEVERATION
    else:
        label = OSCILLATING
    stats["label"] = label
    return stats
