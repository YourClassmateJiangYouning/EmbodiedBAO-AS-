"""EmbodiedBAO environment for NVIDIA Isaac Sim.

The benchmark recreates the psychological "Body-as-Obstacle" (BAO) task in a
16 m long by 5 m wide room.  World coordinates follow the project convention:

    x : forward axis (room spans x in [0, 16]; the robot starts at x = 0.5)
    y : up axis (the ground plane is y = 0)
    z : lateral axis (room spans z in [-2.5, 2.5])

The opaque obstacle wall sits on the plane x = 8.0 and spans the room width.  A vertical channel centred at z = 0 runs from the ground to the top of
the wall.  The channel width is the single independent variable of the
benchmark: it is set per Level so that the channel-to-shoulder ratio (A/S)
sweeps past the human threshold of 1.30 (Warren & Whang, 1987).

The task is a pure gap-traversal problem: the robot must move its body centre
to the goal plane on the far side of the wall (x >= 11.0 m).  There is no reachable target object; the
only question is whether the agent rotates its body before the channel becomes
too narrow for a frontal passage.

    A/S = channel_width / ROBOT_SHOULDER_WIDTH

The environment follows MirrorBench's interaction pattern: the robot is moved
kinematically (world-pose teleports), every action is gated by an analytic
collision check against the wall, and the head camera is a 1024x1024 RGB
camera that follows the robot root.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
    import carb

    _HAS_ISAAC_SIM = True
except Exception as _isaac_import_error:  # pragma: no cover - only outside Isaac Sim
    # Report the real failure.  `isaacsim.core` is importable ONLY after
    # SimulationApp has started, so a bare `except: pass` here hides the one
    # mistake that matters most: importing this module too early, which latches
    # _HAS_ISAAC_SIM to False and makes the scene permanently unbuildable.
    import sys as _sys
    import traceback as _tb

    print(
        "[BAOEnv] Isaac Sim import failed "
        f"({type(_isaac_import_error).__name__}: {_isaac_import_error}). "
        "If SimulationApp is already running this is a bug; if not, this "
        "module was imported too early.",
        file=_sys.stderr,
    )
    _tb.print_exc()
    _HAS_ISAAC_SIM = False


# ---------------------------------------------------------------------------
# Scene constants (metres / degrees)
# ---------------------------------------------------------------------------

# Longitudinal and lateral dimensions are deliberately independent: extending
# the approach/run-out must never widen the room or alter any channel width.
ROOM_LENGTH_X = 16.0
ROOM_WIDTH_Z = 5.0
GROUND_THICKNESS = 0.02

# `FixedCuboid`'s size/scale combination does NOT mean metres, and behaves
# inconsistently.  Measured on the lab machine with size=0.5:
#
#     authored 0.05 x 0.05 x 2.00 -> world 0.0013 x 2.00 x 0.0013
#     authored 0.02 x 2.00 x 1.55 -> world 0.0002 x 1.20 x 2.00
#     authored 0.02 x 3.00 x 4.00 -> world 0.0002 x 8.00 x 4.50
#
# The visible consequence was the channel posts: authored 5 cm wide they
# rendered about 1.3 mm wide, so the opening never had a visible edge in any
# capture.  Every box is now built by BAOEnv._add_box, which sets a USD Cube's
# extent explicitly in metres, so world size == authored size by construction.

# Room enclosure.  Without a far wall the space behind the channel is just the
# floor plus empty background, so looking through the opening renders as one
# uniform grey -- measured: unique_colors = 2 at pitch 0 and 15 from 0.5 m away,
# and 19 at 0.5 m with the robot hidden.  The model then reports "a solid grey
# wall with no visible openings" while standing in front of the gap.  Closing
# the room and colouring the walls gives the opening something to reveal.
ROOM_WALL_HEIGHT = 3.0
ROOM_WALL_THICKNESS = 0.02
# Side and near walls: light grey.
ROOM_SIDE_WALL_COLOR = [0.72, 0.73, 0.75]
# The far wall -- seen THROUGH the channel -- is green, so the destination
# reads as a distinct, reachable place.
ROOM_FAR_WALL_COLOR = [0.13, 0.42, 0.20]
# The obstacle wall is opaque blue.  It used to be translucent glass, which at
# close range made "aimed at the opening" and "aimed at a panel" render almost
# identically: a recorded gemini-2.5-pro run sitting at x=2.30, centred on the
# channel and facing it, reported "I am now facing a solid wall. I cannot see
# the opening I need to pass through," then scanned left and right and never
# advanced.  A saturated opaque surface makes the channel a plain blue-on-blue
# silhouette, which is a far stronger cue than a transparent panel.
WALL_COLOR = [0.13, 0.28, 0.72]
WALL_OPACITY = 1.0
ROOM_CEILING_COLOR = [0.92, 0.93, 0.95]

# Goal marker: a mirrored pair of red bands on the side walls at the goal line.
# Red because every other surface is blue, green, grey or white, so it is the
# only warm colour in the scene and stays identifiable from the far end of a
# 10 m corridor through a 0.45 m opening.
GOAL_MARKER_COLOR = [0.85, 0.15, 0.12]

WALL_X = 8.0
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.02
CHANNEL_WIDTH = 0.90  # Level 0 default; every Level overrides this
CHANNEL_HALF_WIDTH = CHANNEL_WIDTH / 2.0
PANEL_WIDTH = (ROOM_WIDTH_Z - CHANNEL_WIDTH) / 2.0

# With a 0.75 m adult step, the obstacle is exactly ten forward translations
# from the start: 0.5 + 10 * 0.75 = 8.0 m.
ROBOT_START_POS = np.array([0.5, 0.0, 0.0], dtype=float)
ROBOT_START_YAW_DEG = 0.0
# The goal is 3 m behind the obstacle: four further forward translations reach
# x=11 exactly.  The inclusive check makes that fourth step count as success.
SUCCESS_X = 11.0

# ---------------------------------------------------------------------------
# A/S threshold ladder (Warren & Whang 1987 human threshold is A/S = 1.30)
# ---------------------------------------------------------------------------
# The channel widths are the primary protocol constants from the task brief.
# The A/S ratios are derived from them so the geometry can never drift out of
# sync with the recorded `a_s_ratio` field.
LEVEL_CHANNEL_WIDTHS: Dict[int, float] = {
    0: 0.90,
    1: 0.80,
    2: 0.74,
    3: 0.68,
    4: 0.57,
    5: 0.45,
}

# Approximate step length for a 1.80 m adult man: 1.80 * 0.415 ~= 0.747 m.
MOVE_STEP = 0.75
TURN_STEP_DEG = 15.0
# Collision sampling within one discrete turn.  Sampling only at TURN_STEP_DEG
# checks the two endpoint poses but can miss a corner touching a wall or room
# boundary midway through the rotation.
TURN_COLLISION_SAMPLE_DEG = 1.0
CAMERA_TURN_STEP_DEG = 30.0
TURN_TOLERANCE_DEG = 1e-6

# Downward pitch of the head camera, in degrees.  The original build got this
# implicitly by aiming at the target ball; now it is explicit.  See
# BAOEnv._update_eye_camera.
EYE_PITCH_DEG = 15.0

# Eye camera height in metres.  The measured H1 is 1.806 m tall, and a person's
# eyes sit at roughly 93% of their height, so 1.68 m is the human-like anchor.
# The old 1.9 m existed only to see a ball floating at 1.2 m.
EYE_CAMERA_HEIGHT = 1.68

# Width (metres) and colour of the dark posts framing the channel.  These give
# the opening a hard visual boundary; see BAOEnv._create_wall.
CHANNEL_EDGE_THICKNESS = 0.05
CHANNEL_EDGE_COLOR = [0.10, 0.11, 0.13]

# Focal length of the robot eye camera, in mm, on a 20.955 mm wide sensor.
#
#   field of view = 2 * atan(20.955 / (2 * focal))
#   13.36 mm -> 76 deg, roughly a person's binocular-and-then-some view
#    8.00 mm -> 105 deg
#    1.50 mm -> 164 deg (the inherited value: a fisheye)
#
# Two separate mistakes lived here.  First the field of view was computed with
# a 36 mm sensor instead of the 20.955 mm one Isaac Sim actually uses, so 8.0 mm
# was believed to be 105 degrees when it is 105 only at 8.0 * (36/20.955).
# Second, and worse, Camera.set_focal_length() is read back correctly but does
# not reach the renderer: with it set to 8.0 the measured field of view was
# still 174 degrees over a 1.68 m drop onto a 0.5 m floor grid.  The focal
# length is now written to the USD attribute as well; see _apply_focal_length.
ROBOT_CAMERA_FOCAL = 13.36

# H1 kinematic constants (used for analytic collision checks).
ROBOT_SHOULDER_WIDTH = 0.57
ROBOT_TORSO_THICKNESS = 0.22
ROBOT_BODY_CENTER_Y = 0.9
ROBOT_BODY_HALF_HEIGHT = 0.9
ROBOT_HEAD_HEIGHT = 1.55
MODEL_YAW_OFFSET_DEG = 0.0

# Yaw band that counts as "sideways" (body rotated so the narrow torso
# dimension faces the channel).
SIDEWAYS_YAW_MIN_DEG = 45.0
SIDEWAYS_YAW_MAX_DEG = 135.0

# Safety skin (metres) added to the body box before the wall test.  It keeps
# A/S == 1.00 an impassable pinch point instead of a zero-clearance squeeze:
# at Level 4 the channel and the shoulders are both exactly 0.57 m, and without
# this margin a perfectly aligned agent could creep through untouched, which is
# not the intended affordance test.
BODY_CLEARANCE = 0.002

# H1 arm hang pose, applied once at reset so the robot looks natural.
ARM_HANG_SHOULDER_PITCH_RAD = 0.0
ARM_HANG_ELBOW_PITCH_RAD = 1.57

ACTIONS = [
    "forward",
    "backward",
    "left",
    "right",
    "turn_left",
    "turn_right",
    "look_left",
    "look_right",
]

# Camera-yaw controls keep the legacy 30-degree head-rotation mechanism.
CAMERA_ACTIONS = ("look_left", "look_right")


def level_channel_width(level: int) -> float:
    """Return the protocol channel width for a Level (raises on unknown)."""
    try:
        return float(LEVEL_CHANNEL_WIDTHS[int(level)])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"unknown level {level!r}; expected one of {sorted(LEVEL_CHANNEL_WIDTHS)}"
        ) from exc


def a_s_ratio(channel_width: float) -> float:
    """A/S = channel width / shoulder width."""
    return float(channel_width) / float(ROBOT_SHOULDER_WIDTH)


def channel_width_for_ratio(ratio: float) -> float:
    """Inverse of :func:`a_s_ratio`; handy for tests and plots."""
    return float(ratio) * float(ROBOT_SHOULDER_WIDTH)


# ---------------------------------------------------------------------------
# Pure geometry helpers (unit-testable without Isaac Sim)
# ---------------------------------------------------------------------------


def _rotate_xz(vec: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate a 3D vector around the up axis (+y) by yaw_rad."""
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([x * c + z * s, y, -x * s + z * c], dtype=float)


def _forward_vector(yaw_rad: float) -> np.ndarray:
    """Unit vector pointing in the robot's facing direction (x/z plane)."""
    return np.array([np.cos(yaw_rad), 0.0, -np.sin(yaw_rad)], dtype=float)


def _right_vector(yaw_rad: float) -> np.ndarray:
    """Unit vector pointing along the robot's right-hand side (x/z plane)."""
    return _rotate_xz(np.array([0.0, 0.0, 1.0], dtype=float), yaw_rad)


def _user_to_isaac_pos(pos: np.ndarray) -> np.ndarray:
    """Map a user-frame position (y up) to Isaac Sim (z up): swap y and z."""
    p = np.asarray(pos, dtype=float)
    return np.array([p[0], p[2], p[1]], dtype=float)


def _isaac_to_user_pos(pos: np.ndarray) -> np.ndarray:
    """Map an Isaac Sim position (z up) back to the user frame (y up)."""
    p = np.asarray(pos, dtype=float)
    return np.array([p[0], p[2], p[1]], dtype=float)


def _user_to_isaac_scale(scale: np.ndarray) -> np.ndarray:
    """Swap the y/z components of a scale vector for Isaac Sim."""
    s = np.asarray(scale, dtype=float)
    return np.array([s[0], s[2], s[1]], dtype=float)


def _user_dims_to_isaac(dims: np.ndarray) -> np.ndarray:
    """Swap the y/z components of a SIZE triple for Isaac Sim.

    This is the counterpart of :func:`_user_to_isaac_scale` and exists purely
    for documentation: the two are the same operation, and calling the wrong one
    at the wrong point is what made the channel posts render lying down.
    """
    return _user_to_isaac_scale(dims)


def _panel_boxes(
    channel_width: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return (center, half-extents) of the two wall panels."""
    if channel_width is None:
        channel_width = CHANNEL_WIDTH
    channel_width = float(channel_width)
    half_width = channel_width / 2.0
    panel_width = (ROOM_WIDTH_Z - channel_width) / 2.0
    z_center = panel_width / 2.0 + half_width
    half = np.array(
        [WALL_THICKNESS / 2.0, WALL_HEIGHT / 2.0, panel_width / 2.0], dtype=float
    )
    return [
        (np.array([WALL_X, WALL_HEIGHT / 2.0, -z_center], dtype=float), half.copy()),
        (np.array([WALL_X, WALL_HEIGHT / 2.0, z_center], dtype=float), half.copy()),
    ]


def _robot_body_aabb(
    root_pos: np.ndarray, yaw_rad: float
) -> Tuple[np.ndarray, np.ndarray]:
    """World-space AABB of the robot body box (torso + shoulders).

    The root pose is taken to be the ground-level point directly under the
    body, so the box centre is the root plus a purely vertical offset: rotating
    the body in place must never move its footprint sideways.

    The AABB is only used for telemetry and for the vertical extent of the
    collision test; :func:`_check_wall_collision` uses the exact oriented box.
    """
    c, s = abs(float(np.cos(yaw_rad))), abs(float(np.sin(yaw_rad)))
    hx = ROBOT_TORSO_THICKNESS / 2.0
    hz = ROBOT_SHOULDER_WIDTH / 2.0
    center = np.asarray(root_pos, dtype=float) + np.array(
        [0.0, ROBOT_BODY_CENTER_Y, 0.0], dtype=float
    )
    half = np.array([hx * c + hz * s, ROBOT_BODY_HALF_HEIGHT, hx * s + hz * c])
    return center, half


def _project_rect(
    center: np.ndarray, axes: np.ndarray, half: np.ndarray, axis: np.ndarray
) -> Tuple[float, float]:
    """Project an oriented rectangle (2D) onto ``axis``; return (lo, hi)."""
    radius = float(np.abs(axes[0] @ axis) * half[0] + np.abs(axes[1] @ axis) * half[1])
    position = float(center @ axis)
    return position - radius, position + radius


def _oriented_rects_overlap(
    center_a: np.ndarray,
    axes_a: np.ndarray,
    half_a: np.ndarray,
    center_b: np.ndarray,
    axes_b: np.ndarray,
    half_b: np.ndarray,
) -> bool:
    """Separating-axis test for two oriented rectangles in the x/z plane."""
    a_lo, a_hi = _project_rect(center_a, axes_a, half_a, axes_a[0])
    b_lo, b_hi = _project_rect(center_b, axes_b, half_b, axes_a[0])
    if b_hi < a_lo or a_hi < b_lo:
        return False
    a_lo, a_hi = _project_rect(center_a, axes_a, half_a, axes_a[1])
    b_lo, b_hi = _project_rect(center_b, axes_b, half_b, axes_a[1])
    if b_hi < a_lo or a_hi < b_lo:
        return False
    a_lo, a_hi = _project_rect(center_a, axes_a, half_a, axes_b[0])
    b_lo, b_hi = _project_rect(center_b, axes_b, half_b, axes_b[0])
    if b_hi < a_lo or a_hi < b_lo:
        return False
    a_lo, a_hi = _project_rect(center_a, axes_a, half_a, axes_b[1])
    b_lo, b_hi = _project_rect(center_b, axes_b, half_b, axes_b[1])
    if b_hi < a_lo or a_hi < b_lo:
        return False
    return True


def _check_wall_collision(
    root_pos: np.ndarray,
    yaw_rad: float,
    channel_width: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Check the robot body against the wall panels.

    The torso footprint is treated as an oriented rectangle with half-extents
    (torso thickness / 2) along the facing axis and (shoulder width / 2) along
    the lateral axis, and each panel as an axis-aligned rectangle.  The
    separating-axis test is exact, which matters because a rotated body reaches
    a panel with a *corner*: an axis-aligned bounding box would be both
    over- and under-permissive near the channel mouth.

    Doing this properly is what makes the benchmark the intended manipulation
    check: the agent has to rotate while there is still clearance, and rotating
    once the shoulders are in the wall plane is rejected.
    """
    root = np.asarray(root_pos, dtype=float)
    root_xz = root[[0, 2]]
    cos_y, sin_y = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    body_axes = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=float)
    body_half = np.array(
        [
            ROBOT_TORSO_THICKNESS / 2.0 + BODY_CLEARANCE,
            ROBOT_SHOULDER_WIDTH / 2.0 + BODY_CLEARANCE,
        ],
        dtype=float,
    )
    body_center_xz = root_xz
    panel_axes = np.eye(2)

    for index, (panel_center, panel_half) in enumerate(_panel_boxes(channel_width)):
        # Cheap y-rejection: the panels and the body both stand on the floor.
        if (
            min(ROBOT_BODY_CENTER_Y - ROBOT_BODY_HALF_HEIGHT, 0.0)
            > float(panel_center[1]) + float(panel_half[1])
            or float(panel_center[1]) - float(panel_half[1])
            > ROBOT_BODY_CENTER_Y + ROBOT_BODY_HALF_HEIGHT
        ):
            continue
        if not _oriented_rects_overlap(
            body_center_xz,
            body_axes,
            body_half,
            np.asarray(panel_center, dtype=float)[[0, 2]],
            panel_axes,
            np.asarray(panel_half, dtype=float)[[0, 2]],
        ):
            continue
        point = np.array(
            [
                float(panel_center[0]) - float(panel_half[0]),
                ROBOT_BODY_CENTER_Y,
                float(
                    np.clip(
                        root[2],
                        panel_center[2] - panel_half[2],
                        panel_center[2] + panel_half[2],
                    )
                ),
            ],
            dtype=float,
        )
        return {
            # The analytic collider is one combined torso-and-shoulder box; it
            # does not contain enough geometry to identify a narrower body part.
            "part": "body",
            "panel": index,
            "point": point.tolist(),
            "body_center": (root + np.array([0.0, ROBOT_BODY_CENTER_Y, 0.0])).tolist(),
            "yaw_deg": float(np.degrees(yaw_rad)),
        }
    return None


def _check_room_boundary(
    root_pos: np.ndarray, yaw_rad: float
) -> Optional[Dict[str, Any]]:
    """Reject poses whose body footprint leaves the enclosed room."""
    root = np.asarray(root_pos, dtype=float)
    if root.shape[0] < 3 or not np.all(np.isfinite(root[:3])):
        return {"part": "body", "boundary": "invalid_pose", "point": root.tolist()}
    c, s = abs(float(np.cos(yaw_rad))), abs(float(np.sin(yaw_rad)))
    half_x = (ROBOT_TORSO_THICKNESS / 2.0 + BODY_CLEARANCE) * c + (
        ROBOT_SHOULDER_WIDTH / 2.0 + BODY_CLEARANCE
    ) * s
    half_z = (ROBOT_TORSO_THICKNESS / 2.0 + BODY_CLEARANCE) * s + (
        ROBOT_SHOULDER_WIDTH / 2.0 + BODY_CLEARANCE
    ) * c
    limits = (
        (root[0] - half_x, 0.0, "near"),
        (ROOM_LENGTH_X - (root[0] + half_x), 0.0, "far"),
        (root[2] - half_z, -ROOM_WIDTH_Z / 2.0, "left"),
        (ROOM_WIDTH_Z / 2.0 - (root[2] + half_z), 0.0, "right"),
    )
    for value, minimum, name in limits:
        if value < minimum:
            return {"part": "body", "boundary": name, "point": root.tolist()}
    return None


def _check_scene_collision(
    root_pos: np.ndarray, yaw_rad: float, channel_width: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Check both the obstacle panels and the enclosing room boundary."""
    return _check_wall_collision(root_pos, yaw_rad, channel_width) or _check_room_boundary(
        root_pos, yaw_rad
    )


def _translation_path_is_clear(
    start: np.ndarray,
    target: np.ndarray,
    yaw_rad: float,
    channel_width: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Sample a translation densely enough that it cannot teleport through a wall."""
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    distance = float(np.linalg.norm(target - start))
    spacing = max(0.005, WALL_THICKNESS / 2.0)
    count = max(1, int(math.ceil(distance / spacing)))
    for index in range(1, count + 1):
        sample = start + (target - start) * (index / count)
        collision = _check_scene_collision(sample, yaw_rad, channel_width)
        if collision is not None:
            return collision
    return None


def _turn_path_is_clear(
    root_pos: np.ndarray,
    from_yaw_deg: float,
    to_yaw_deg: float,
    channel_width: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Check the complete swept yaw arc of a turn.

    Sampling at most ``TURN_COLLISION_SAMPLE_DEG`` apart catches a corner that
    sweeps through a wall panel or room boundary even when both endpoint poses
    are clear.  The current pose is sampled first, which also stops a robot that
    is already fouling the wall from rotating its way free: its only recovery is
    to translate away.  Because normal poses are collision-free, turnaround
    rotation remains available on the far side of the wall.
    """
    delta = float(to_yaw_deg) - float(from_yaw_deg)
    count = max(1, int(math.ceil(abs(delta) / TURN_COLLISION_SAMPLE_DEG)))
    for i in range(0, count + 1):
        sample = float(from_yaw_deg) + delta * (i / count)
        collision = _check_scene_collision(
            root_pos, np.radians(sample), channel_width=channel_width
        )
        if collision is not None:
            return collision
    return None


def _eye_look_direction(yaw_rad: float, pitch_deg: float, distance: float) -> np.ndarray:
    """Offset from the eye camera to its look-at point.

    Exposed as a pure function so the "the head camera must look downward"
    invariant can be checked without Isaac Sim.  The original build only got
    this pitch as a side effect of aiming at the target ball; making it
    explicit is what stops a level, featureless view from silently returning.
    """
    look = _forward_vector(yaw_rad)
    pitch = math.radians(float(pitch_deg))
    return np.array(
        [
            look[0] * distance * math.cos(pitch),
            -distance * math.sin(pitch),
            look[2] * distance * math.cos(pitch),
        ],
        dtype=float,
    )


@dataclass
class StepResult:
    """Result of one environment action."""

    rgb: Optional[np.ndarray]
    legal: bool
    feedback: str
    success: bool
    collision: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None


class BAOEnv:
    """Isaac Sim environment for the Body-as-Obstacle channel-passage task."""

    def __init__(self, sim_app: Any = None, task_dict: Optional[Dict[str, Any]] = None) -> None:
        if not _HAS_ISAAC_SIM:
            raise RuntimeError(
                "Isaac Sim could not be imported. Run this script with "
                "%ISAACSIM_ROOT%\\python.bat (Windows) or $ISAACSIM_ROOT/python.sh (Linux)."
            )
        self.sim_app = sim_app
        self.task_dict = task_dict or {}
        if os.environ.get("EMBODIEDBAO_HIDE_ROBOT", "0") == "1":
            self.task_dict["hide_robot"] = True

        settings = carb.settings.get_settings()
        settings.set(
            "/rtx/rendermode",
            str(self.task_dict.get("rendermode", "RaytracedLighting")),
        )
        settings.set("/rtx/pathtracing/spp", int(self.task_dict.get("spp", 32)))

        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage

        self.robot_prim_path = "/World/H1"
        self.robot_usd_path: str = ""
        self.robot_root: Optional[XFormPrim] = None
        self.hand_xform: Optional[XFormPrim] = None
        self.hand_prim_path: Optional[str] = None
        self.head_xform: Optional[XFormPrim] = None
        self.head_prim_path: Optional[str] = None
        self.head_visual_path: Optional[str] = None
        self.eye_camera: Optional[Camera] = None

        self._articulation: Any = None
        self._articulation_ok = False
        self._articulation_error = ""
        self._robot_yaw = ROBOT_START_YAW_DEG
        self._robot_ground_offset = 0.0
        # Filled in by _compute_robot_ground_offset from the loaded USD, so the
        # real robot height can be reported instead of assumed.
        self._robot_measured_height: Optional[float] = None
        # Start pose and translation step are configurable: standing further
        # back makes the channel readable, but costs travel, so the two have to
        # move together.  See ROBOT_START_POS / MOVE_STEP and the --start_x /
        # --move_step CLI flags.
        start_x = float(self.task_dict.get("start_x", ROBOT_START_POS[0]))
        self._start_pos = np.array(
            [start_x, ROBOT_START_POS[1], ROBOT_START_POS[2]], dtype=float
        )
        self._move_step = float(self.task_dict.get("move_step", MOVE_STEP))
        if not math.isfinite(self._move_step) or not (0.0 < self._move_step <= ROOM_LENGTH_X):
            raise ValueError(
                f"move_step must be finite and in (0, {ROOM_LENGTH_X}], got {self._move_step}"
            )
        self._channel_width = float(self.task_dict.get("channel_width", CHANNEL_WIDTH))
        self._camera_yaw_offset = 0.0

        self._create_ground()
        self._create_room()
        self._create_wall()
        if self.task_dict.get("hide_wall", False):
            self._remove_wall()
        self._create_goal_marker()
        self._create_lights()
        self._create_camera()
        self._load_robot()
        self._create_eye_camera()
        self._update_camera()
        self._update_eye_camera()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _add_box(
        self,
        name: str,
        center: np.ndarray,
        dims: np.ndarray,
        material: Optional[Tuple[str, List[float]]] = None,
    ) -> str:
        """Add an axis-aligned box of EXACT metre dimensions.

        ``FixedCuboid``'s ``size``/``scale`` combination proved not to mean
        metres, and inconsistently so.  Measured on the lab machine with
        ``size=0.5``:

            authored 0.05 x 0.05 x 2.00 -> world 0.0013 x 2.00 x 0.0013
            authored 0.02 x 2.00 x 1.55 -> world 0.0002 x 1.20 x 2.00
            authored 0.02 x 3.00 x 4.00 -> world 0.0002 x 8.00 x 4.50

        so the channel posts -- authored 5 cm wide -- rendered about 1.3 mm
        wide, which is why the opening had no visible edge in any capture.

        This builds a USD Cube whose extent is set explicitly in metres and
        whose transform is applied with Xform ops, leaving no scaling
        ambiguity: the world size is exactly ``dims``.
        """
        path = f"/World/{name}"
        # A USD Cube's geometry is fixed at +-1 and only the scale op sizes it,
        # while `extent` is a declaration about the LOCAL bounds, not a way to
        # set size.  Writing a world-sized extent alongside a scale op made the
        # two disagree: the reported bounds and the rendered geometry stopped
        # matching, which is how the channel posts ended up rendering as a 5 cm
        # lump near z=1.0 rather than a 2 m vertical post.
        #
        # This builds an explicit mesh instead: eight corners authored directly
        # in metres, so local geometry, world size and reported bounds are the
        # same number by construction, with no extent or scale semantics left to
        # misinterpret.
        centre_isaac = _user_to_isaac_pos(np.asarray(center, dtype=float))
        # `dims` is authored as (along_x, height, left_right) in the USER frame.
        # Isaac Sim uses z for height, so the y and z components swap -- once.
        # Previously _user_to_isaac_scale was applied to a triple that had
        # already been swapped, so the height and the lateral extent traded
        # places and the channel posts rendered lying down (measured z span
        # 0.05 m instead of 2.0 m).
        dims_isaac = np.array(
            [float(dims[0]), float(dims[2]), float(dims[1])], dtype=float
        )
        hx, hy, hz = (float(dims_isaac[0]) / 2.0,
                      float(dims_isaac[1]) / 2.0,
                      float(dims_isaac[2]) / 2.0)
        cx, cy, cz = (float(centre_isaac[0]),
                      float(centre_isaac[1]),
                      float(centre_isaac[2]))
        corners = [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ]
        faces = [
            (0, 1, 2, 3),  # -z
            (4, 7, 6, 5),  # +z
            (0, 4, 5, 1),  # -y
            (3, 2, 6, 7),  # +y
            (0, 3, 7, 4),  # -x
            (1, 5, 6, 2),  # +x
        ]
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
        mesh.CreateFaceVertexCountsAttr([4] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(cx - hx, cy - hy, cz - hz),
                Gf.Vec3f(cx + hx, cy + hy, cz + hz),
            ]
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)

        xform = UsdGeom.Xformable(mesh.GetPrim())
        xform.ClearXformOpOrder()

        if material is not None:
            mat_name, colour = material
            self._create_and_bind_material(
                path,
                f"/World/Looks/{mat_name}",
                color=list(colour),
                metallic=0.0,
                roughness=0.6,
            )
        return path

    def _create_ground(self) -> None:
        self._add_box(
            "Ground",
            np.array([ROOM_LENGTH_X / 2.0, -GROUND_THICKNESS / 2.0, 0.0]),
            np.array([ROOM_LENGTH_X, GROUND_THICKNESS, ROOM_WIDTH_Z]),
        )
        self._create_ground_grid()

    def _create_room(self) -> None:
        """Close the room with four walls and a full-length ceiling.

        Measured motivation: with only the obstacle wall and the floor, the
        space behind the channel was floor plus empty background, so every ray
        through the opening returned the same value -- unique_colors = 2 from
        0.5 m at pitch 0 and 15.  The eye view was therefore unreadable no
        matter where the camera was aimed.  A closed, coloured room means the
        opening reveals something distinct.

        The obstacle wall at ``WALL_X`` is built separately by :meth:`_create_wall`;
        this only adds the enclosure around the 16 m by 5 m floor.
        """
        half_x = ROOM_LENGTH_X / 2.0
        half_z = ROOM_WIDTH_Z / 2.0
        height = ROOM_WALL_HEIGHT
        thickness = ROOM_WALL_THICKNESS

        # (name, centre, dims, colour) in user coordinates.
        walls = [
            # Far wall: the surface visible THROUGH the channel.  Deliberately a
            # different colour from the others, so "aimed at the opening" and
            # "aimed at a panel" do not look the same up close.
            (
                "room_far",
                [ROOM_LENGTH_X + thickness / 2.0, height / 2.0, 0.0],
                [thickness, height, ROOM_WIDTH_Z + 2.0 * thickness],
                ROOM_FAR_WALL_COLOR,
            ),
            # Near wall, behind the robot.
            (
                "room_near",
                [-thickness / 2.0, height / 2.0, 0.0],
                [thickness, height, ROOM_WIDTH_Z + 2.0 * thickness],
                ROOM_SIDE_WALL_COLOR,
            ),
            # Side walls.
            (
                "room_side_left",
                [half_x, height / 2.0, -half_z - thickness / 2.0],
                [ROOM_LENGTH_X, height, thickness],
                ROOM_SIDE_WALL_COLOR,
            ),
            (
                "room_side_right",
                [half_x, height / 2.0, half_z + thickness / 2.0],
                [ROOM_LENGTH_X, height, thickness],
                ROOM_SIDE_WALL_COLOR,
            ),
        ]
        for name, centre, dims, colour in walls:
            self._add_box(
                name,
                np.array(centre, dtype=float),
                np.array(dims, dtype=float),
                material=(f"{name}Material", colour),
            )

        # Ceiling, so rays above the wall tops do not escape to the background.
        self._add_box(
            "room_ceiling",
            np.array([half_x, height + thickness / 2.0, 0.0]),
            np.array([ROOM_LENGTH_X, thickness, ROOM_WIDTH_Z]),
            material=("room_ceilingMaterial", ROOM_CEILING_COLOR),
        )

    def _create_goal_marker(self) -> None:
        """A red square on the green far wall, as a distance cue.

        Purely decorative and NOT part of :func:`_panel_boxes`, so no collision
        check sees it and it cannot block the robot.

        A recorded run shows why a cue is needed.  With the obstacle at x=8 and
        the unbroken green far wall at x=16, an agent that has walked to x=7.25 --
        correctly centred on the channel at z=0.00 and already entering it --
        reports:

            "After nine consecutive forward steps, the robot has not yet passed
             through an opening and instead appears to be facing a solid green
             wall."

        and then spends the rest of the episode turning and sidestepping, looking
        for an opening that was behind it.  An independent run with a different
        model produced the same description, so it is a property of the scene
        rather than of one agent.

        A fixed-size mark on the far wall is a CONTINUOUS distance cue, which a
        marker sitting at the goal is not: its apparent size is inversely
        proportional to the agent's distance, so "am I getting closer?" can be
        answered by comparing the current frame against the previous one, without
        needing any notion of world coordinates.  A ribbon at x=11 only became
        informative on arrival.

        Size and placement are measured rather than chosen by eye.  A 0.80 m
        square was tried first and rejected: at x=14, only 2 m from the wall, its
        apparent edge reached 420 px and the view became a red field with no
        visible edges, so it stopped reading as an object at the very moment its
        growth should have been most informative.  It is now 0.30 m, trading a
        small far-end mark for edges that stay visible throughout the approach.

        Apparent edge at 1024 px and a 52 degree vertical field of view:

            agent x=0.5  ->  20 px     (start: a small distant mark)
            agent x=4.0  ->  26 px
            agent x=7.0  ->  34 px     (approaching the obstacle)
            agent x=11.0 ->  63 px     (at the goal)
            agent x=14.0 -> 158 px     (close, edges still clear)

        Centred at 1.40 m it spans 1.25 m to 1.55 m.  The camera sits at 1.68 m,
        so from the start (15.5 m away) the mark is 1.0 degree below the optical
        axis while the frame's lower edge is 11 degrees below it at a 52 degree
        vertical field of view -- comfortably inside the frame, and inside the
        0.0 to 2.0 m opening, so it is visible through the channel from the very
        start, which is what makes it usable during the approach.
        """
        if not self.task_dict.get("goal_marker", True):
            return
        size = float(self.task_dict.get("goal_marker_height", 0.30))
        centre_y = float(self.task_dict.get("goal_marker_base", 1.40))
        thickness = float(self.task_dict.get("goal_marker_span", 0.02))
        self._add_box(
            "GoalMarker",
            np.array(
                [ROOM_LENGTH_X - thickness / 2.0 - 0.01, centre_y, 0.0]
            ),
            np.array([thickness, size, size]),
            material=("GoalMarkerMaterial", GOAL_MARKER_COLOR),
        )

    def _create_ground_grid(self, spacing: float = 0.5) -> None:
        """Mark the floor with a faint grid.

        The floor is otherwise a single flat grey slab, which makes every
        camera view unreadable: there is no scale reference, no way to judge
        distance, and nothing to contrast the obstacle wall against.  Thin
        dark strips every ``spacing`` metres give the scene a readable
        reference frame without changing any of the task geometry.
        """
        thickness = 0.012
        height = 0.002

        # Lines at fixed x run across the 5 m width.
        x_steps = int(round(ROOM_LENGTH_X / spacing))
        for i in range(x_steps + 1):
            x = i * spacing
            self._add_box(
                f"Grid_x_{i}",
                np.array([x, height / 2.0, 0.0]),
                np.array([thickness, height, ROOM_WIDTH_Z]),
                material=(f"GridMaterial_x_{i}", [0.30, 0.32, 0.35]),
            )

        # Lines at fixed z run along the full 16 m length.
        z_steps = int(round(ROOM_WIDTH_Z / spacing))
        for i in range(z_steps + 1):
            z = -ROOM_WIDTH_Z / 2.0 + i * spacing
            self._add_box(
                f"Grid_z_{i}",
                np.array([ROOM_LENGTH_X / 2.0, height / 2.0, z]),
                np.array([ROOM_LENGTH_X, height, thickness]),
                material=(f"GridMaterial_z_{i}", [0.30, 0.32, 0.35]),
            )

    def _create_wall(self) -> None:
        """The obstacle wall: opaque blue panels with a channel between them.

        The panels are solid, not glass.  A translucent wall left the opening
        hard to distinguish at close range (see WALL_COLOR), whereas a saturated
        opaque surface turns the channel into a clean silhouette.
        """
        channel_half = self._channel_width / 2.0
        panel_width = (ROOM_WIDTH_Z - self._channel_width) / 2.0
        z_center = panel_width / 2.0 + channel_half
        for i, sign in enumerate((-1.0, 1.0)):
            self._add_box(
                f"WallPanel_{i}",
                np.array([WALL_X, WALL_HEIGHT / 2.0, sign * z_center]),
                np.array([WALL_THICKNESS, WALL_HEIGHT, panel_width]),
                material=(f"WallPanelMaterial_{i}", WALL_COLOR),
            )

        # Dark posts mark the channel edges, so the opening's boundary is
        # unambiguous rather than a colour boundary alone.
        #
        # They sit OUTSIDE the channel, with their inner face exactly on the
        # channel edge.  Centring a 5 cm post on the edge would eat 5 cm into
        # the gap: the opening would look 0.85 m while the collision model (which
        # uses _panel_boxes and ignores the posts) allowed 0.90 m.  The whole
        # benchmark compares shoulder width against this gap, so the visible
        # clear width must equal the modelled one.
        post_centre_offset = channel_half + CHANNEL_EDGE_THICKNESS / 2.0
        for sign in (-1.0, 1.0):
            edge_id = 0 if sign < 0 else 1
            # dims are (along_x, height, left_right): the post must be a TALL
            # SLIM bar, so the height slot gets WALL_HEIGHT.  Authoring it with
            # WALL_HEIGHT in the left_right slot made it a horizontal bar 2 m
            # long and 5 cm tall, which is what rendered as the dark line across
            # the wall in every view.
            self._add_box(
                f"ChannelEdge_{edge_id}",
                np.array([WALL_X, WALL_HEIGHT / 2.0, sign * post_centre_offset]),
                # Match the panel depth so the visible frame never protrudes
                # beyond the analytic collision geometry along x.
                np.array([WALL_THICKNESS, WALL_HEIGHT, CHANNEL_EDGE_THICKNESS]),
                material=(
                    f"ChannelEdgeMaterial_{edge_id}",
                    CHANNEL_EDGE_COLOR,
                ),
            )

    def _remove_wall(self) -> None:
        for path in (
            "/World/WallPanel_0",
            "/World/WallPanel_1",
            "/World/ChannelEdge_0",
            "/World/ChannelEdge_1",
        ):
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                self.stage.RemovePrim(path)

    def set_channel_width(self, width: float) -> float:
        """Resize the wall channel and rebuild its visual/collision prims."""
        width = float(width)
        if not 0.1 <= width <= ROOM_WIDTH_Z - 0.1:
            raise ValueError(f"channel width must be in [0.1, {ROOM_WIDTH_Z - 0.1}]")
        if abs(width - self._channel_width) < 1e-9:
            return self._channel_width
        self._channel_width = width
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_playing():
                timeline.stop()
        except Exception:
            pass
        self._remove_wall()
        self._create_wall()
        return self._channel_width

    def get_channel_width(self) -> float:
        return float(self._channel_width)

    def get_a_s_ratio(self) -> float:
        return a_s_ratio(self._channel_width)

    def _apply_focal_length(self, camera: Any, path: str, focal_mm: float) -> None:
        """Set a camera's focal length through USD, not the sensor wrapper.

        Measured on the lab machine: after ``Camera.set_focal_length(8.0)`` the
        getter read back 8.0, yet the rendered field of view measured 174
        degrees -- still the inherited fisheye.  The wrapper therefore accepts
        the value without the renderer honouring it, which is why raising the
        focal length from 1.5 to 8.0 across four runs changed nothing.

        The authoritative attributes are ``focalLength`` and ``horizontalAperture``
        on the camera prim, in mm and tenths of a mm-of-sensor respectively.
        Setting ``focalLength`` alone (with the aperture left at its default of
        the whole sensor width) gives a predictable field of view:
        ``2 * atan(horizontalAperture / (2 * focalLength))``.
        """
        try:
            camera.set_focal_length(float(focal_mm))
        except Exception:
            pass
        try:
            cam = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            if not cam:
                return
            cam.GetFocalLengthAttr().Set(float(focal_mm))
        except Exception as exc:
            print(f"[BAOEnv] could not set USD focalLength on {path}: {exc}")

    def _create_camera(self) -> None:
        resolution = tuple(
            int(v) for v in self.task_dict.get("camera_resolution", (1024, 1024))
        )
        self.camera = Camera(
            prim_path="/World/Camera",
            translation=_user_to_isaac_pos(
                np.array([self._start_pos[0], ROBOT_HEAD_HEIGHT, self._start_pos[2]])
            ),
            frequency=20,
            resolution=resolution,
        )
        self._apply_focal_length(
            self.camera,
            "/World/Camera",
            float(
                self.task_dict.get(
                    "third_camera_focal",
                    self.task_dict.get("camera_focal", 2.5),
                )
            ),
        )

    def _create_eye_camera(self) -> None:
        """Robot eye camera mounted at the H1 head d435 module."""
        resolution = tuple(
            int(v) for v in self.task_dict.get("camera_resolution", (1024, 1024))
        )
        self.eye_camera = Camera(
            prim_path="/World/RobotEyeCamera",
            translation=_user_to_isaac_pos(
                np.array([self._start_pos[0], ROBOT_HEAD_HEIGHT, self._start_pos[2]])
            ),
            frequency=20,
            resolution=resolution,
        )
        self._apply_focal_length(
            self.eye_camera,
            "/World/RobotEyeCamera",
            float(
                self.task_dict.get(
                    "robot_camera_focal",
                    self.task_dict.get("camera_focal", ROBOT_CAMERA_FOCAL),
                )
            ),
        )

    def _create_lights(self) -> None:
        """Light the room from inside it.

        The room is enclosed (see :meth:`_create_room`), so lights placed
        outside it no longer reach anything: with the inherited dome plus
        distant light the external views rendered at mean=1.8 against 104
        before the ceiling existed, i.e. almost black.  These are interior
        lights instead, and one still points down the channel so the opening
        keeps a visible highlight.
        """
        # A distant light is directional and originates outside the room, so it
        # would be blocked by the ceiling; interior lights are used instead.
        #
        # Measured exposure history, so the values are not guesses:
        #   60000 intensity, 5 m room  -> mean=229.5, 226 unique colours (blown out)
        #    9000 intensity, 16 m room -> mean= 87.5                        (too dark)
        #                                     mean= 80.8 at eye_near
        # The room is now 16 m rather than 5 m, so four small emitters have to
        # cover three times the length and the far surfaces receive very little.
        #
        # Two changes rather than one, because intensity alone is the wrong tool:
        #
        #   * more emitters -- eight along the 16 m instead of four, so the gap
        #     between lights is 2 m rather than 4 m and no stretch of wall sits
        #     far from any light;
        #   * a larger radius -- 0.35 m is effectively a point source whose
        #     falloff is harsh, while 1.0 m spreads the same power over an area,
        #     which both softens shadows and cuts the render noise that made the
        #     frames look grainy;
        #   * intensity tuned by sweeping and measuring, because the response is
        #     NON-linear and linear extrapolation from a single point overshoots.
        #     Twelve thousand was still washed out at mean 220, and scaling
        #     linearly from there suggested ~7000, which measured 220 again.
        #
        # Measured sweep, 800x800, eight 1.0 m emitters, this 16 m room:
        #     light   dome   eye_start mean   std    unique
        #      500      0        70.2        31.5    22910   (too dark)
        #     1500      0       136.7        40.2    33590
        #     1500    100       137.3        39.9    33551   <- chosen
        #     3000    100       179.8        34.9    30589
        #     7000      -       220.0        22.7    48090   (washed out)
        #    12000      -       235.0        15.6    30350   (washed out)
        #
        # The exponent is roughly 0.8, not 1, which is why 7000 did not help.
        #
        # The dome light is NOT the lever it looks like: at light=1500, dome=100
        # and dome=0 differ by 0.6 in the mean, because the dome mostly lights the
        # ceiling and the distance ahead rather than the view down the corridor.
        #
        # Acceptance test: eye_start mean roughly 130-160 AND std above about 30.
        # A high mean with a LOW std is the signature of a washed-out frame -- at
        # mean 235 the std was 15.6 and the blue and green had faded to pale
        # tints, which is what prompted this sweep.
        #
        # Overridable so exposure can be tuned by rendering rather than by
        # editing this file: capture_views.py --light_intensity / --light_radius.
        intensity = float(self.task_dict.get("light_intensity", 1500.0))
        radius = float(self.task_dict.get("light_radius", 1.0))
        dome_intensity = float(self.task_dict.get("dome_intensity", 300.0))

        positions = [
            ("/World/Light_01", np.array([1.5, 2.6, 0.0])),
            ("/World/Light_02", np.array([3.5, 2.6, 0.0])),
            ("/World/Light_03", np.array([5.5, 2.6, 0.0])),
            ("/World/Light_04", np.array([7.5, 2.6, 0.0])),
            ("/World/Light_05", np.array([9.5, 2.6, 0.0])),
            ("/World/Light_06", np.array([11.5, 2.6, 0.0])),
            ("/World/Light_07", np.array([13.5, 2.6, 0.0])),
            ("/World/Light_08", np.array([15.0, 2.6, 0.0])),
        ]
        for path, position in positions:
            light = UsdLux.SphereLight.Define(self.stage, path)
            light.GetIntensityAttr().Set(intensity)
            light.GetRadiusAttr().Set(radius)
            light.AddTranslateOp().Set(Gf.Vec3d(*_user_to_isaac_pos(position)))

        # Ambient fill so no surface is pure black.
        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.GetIntensityAttr().Set(dome_intensity)

    def _load_robot(self) -> None:
        usd_path = self._resolve_robot_usd_path()
        add_reference_to_stage(usd_path=usd_path, prim_path=self.robot_prim_path)
        self.robot_usd_path = usd_path
        # The benchmark is explicitly kinematic: poses are set directly and
        # collision legality is decided analytically.  Dynamic physics is opt-in
        # because the authored visual floor has no collider.
        if not self.task_dict.get("robot_physics", False):
            self._disable_robot_physics()
        self.robot_root = XFormPrim(prim_paths_expr=self.robot_prim_path)
        self._robot_ground_offset = self._compute_robot_ground_offset()
        self._set_robot_pose(self._start_pos, ROBOT_START_YAW_DEG)
        if self.task_dict.get("hide_robot", False):
            for sub_prim in self.stage.Traverse():
                if str(sub_prim.GetPath()).startswith(self.robot_prim_path):
                    try:
                        sub_prim.SetActive(False)
                    except Exception:
                        pass
        self._find_hand_prim()
        self._find_head_camera_link()

    def _compute_robot_ground_offset(self) -> float:
        """Raise the robot so its lowest mesh point sits on the ground (z=0)."""
        try:
            from pxr import Usd, UsdGeom

            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            min_z = float("inf")
            max_z = float("-inf")
            for prim in self.stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(self.robot_prim_path):
                    continue
                bound = cache.ComputeWorldBound(prim)
                range3d = bound.ComputeAlignedRange()
                lo = range3d.GetMin()
                hi = range3d.GetMax()
                if not all(
                    math.isfinite(v) for v in (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
                ):
                    continue
                min_z = min(min_z, lo[2])
                max_z = max(max_z, hi[2])
            if min_z == float("inf"):
                return 0.0
            offset = float(-min_z)
            if math.isfinite(max_z):
                self._robot_measured_height = float(max_z - min_z)
                print(
                    f"[BAOEnv] ROBOT HEIGHT (measured) = "
                    f"{self._robot_measured_height:.3f} m | "
                    f"ground offset = {offset:.4f} m | "
                    f"authored ROBOT_HEAD_HEIGHT = {ROBOT_HEAD_HEIGHT:.3f} m | "
                    f"wall height = {WALL_HEIGHT:.2f} m"
                )
            else:
                print(f"[BAOEnv] robot ground offset = {offset:.4f} m")
            return offset
        except Exception as exc:
            import traceback as _tb

            print(f"[BAOEnv] robot height measurement FAILED: {exc}")
            _tb.print_exc()
            return 0.0

    def _resolve_robot_usd_path(self) -> str:
        candidates: List[str] = []
        override = self.task_dict.get("robot_usd_path") or os.environ.get(
            "EMBODIEDBAO_H1_USD"
        )
        if override:
            candidates.append(str(override))

        assets_root = ""
        try:
            assets_root = get_assets_root_path() or ""
        except Exception:
            assets_root = ""
        if assets_root:
            candidates.append(
                os.path.join(assets_root, "Isaac", "Robots", "Unitree", "H1", "h1.usd")
            )
            candidates.append(
                os.path.join(
                    assets_root, "Isaac", "Robots", "Unitree", "H1", "h1_with_hands.usd"
                )
            )

        isaac_lab_root = os.environ.get("ISAACLAB_ASSETS_DIR", "")
        if isaac_lab_root:
            candidates.append(
                os.path.join(isaac_lab_root, "Isaac", "Robots", "Unitree", "H1", "h1.usd")
            )
            candidates.append(
                os.path.join(
                    isaac_lab_root,
                    "Isaac",
                    "Robots",
                    "Unitree",
                    "H1",
                    "h1_with_hands.usd",
                )
            )

        isaacsim_root = os.environ.get("ISAACSIM_ROOT", "")
        if isaacsim_root:
            candidates.append(
                os.path.join(
                    isaacsim_root, "assets", "Isaac", "Robots", "Unitree", "H1", "h1.usd"
                )
            )
        candidates.append(os.path.join(os.getcwd(), "assets", "H1", "h1.usd"))
        candidates.append(os.path.join(os.getcwd(), "assets", "H1", "h1_with_hands.usd"))
        candidates.append("omniverse://localhost/Isaac/Robots/Unitree/H1/h1.usd")
        candidates.append("omniverse://localhost/Isaac/Robots/Unitree/H1/h1.usda")
        for path in self._discover_h1_assets():
            if path not in candidates:
                candidates.append(path)

        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.isfile(path):
                return path
        for path in seen:
            if path.startswith("omniverse://") or path.startswith("http"):
                return path
        raise FileNotFoundError(
            "Could not locate a Unitree H1 USD asset. Set EMBODIEDBAO_H1_USD to the "
            "USD path, place h1.usd under assets/H1/, or install Isaac Sim/Isaac Lab "
            "assets."
        )

    def _discover_h1_assets(self, max_depth: int = 10) -> List[str]:
        """Search common Isaac Lab / local asset locations for H1 USD files."""
        roots: List[str] = []
        for env_name in ("EMBODIEDBAO_H1_USD_DIR", "ISAACLAB_ASSETS_DIR"):
            root = os.environ.get(env_name, "")
            if root and os.path.isdir(root):
                roots.append(root)
        try:
            import isaaclab_assets

            module_dir = os.path.dirname(os.path.abspath(isaaclab_assets.__file__))
            if module_dir not in roots and os.path.isdir(module_dir):
                roots.append(module_dir)
        except Exception:
            pass
        home = os.path.expanduser("~")
        roots.extend(
            [
                os.path.join(home, "isaaclab"),
                os.path.join(home, ".local", "share", "ov", "pkg"),
            ]
        )
        isaacsim_root = os.environ.get("ISAACSIM_ROOT", "")
        if isaacsim_root:
            roots.append(os.path.join(isaacsim_root, "assets"))
        roots.append(os.path.join(os.getcwd(), "assets"))
        roots = [root for root in roots if root and os.path.isdir(root)]

        found: List[str] = []
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                depth = dirpath[len(root):].count(os.sep)
                if depth >= max_depth:
                    dirnames[:] = []
                    continue
                for name in filenames:
                    lower = name.lower()
                    if not lower.startswith("h1") or not lower.endswith((".usd", ".usda")):
                        continue
                    path = os.path.join(dirpath, name)
                    if path not in found:
                        found.append(path)
        found.sort(
            key=lambda path: (
                0 if "unitree" in path.lower() else 1,
                path.lower(),
            )
        )
        return found

    def _find_hand_prim(self) -> None:
        """Locate a hand prim purely for optional state telemetry."""
        scored: List[Tuple[int, str]] = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            name = prim.GetName().lower()
            if "right_hand" in name or name in ("r_hand", "right_hand"):
                scored.append((2, path))
            elif "hand" in name:
                scored.append((1, path))
        if not scored:
            return
        scored.sort(key=lambda item: -item[0])
        self.hand_prim_path = scored[0][1]
        try:
            self.hand_xform = XFormPrim(prim_paths_expr=self.hand_prim_path)
        except Exception as exc:  # pragma: no cover - runtime Isaac Sim path
            print(f"[BAOEnv] Could not wrap hand prim {self.hand_prim_path}: {exc}")
            self.hand_xform = None

    def _find_head_camera_link(self) -> None:
        """Locate the H1 head camera module (prefer the visual lens mesh)."""
        scored: List[Tuple[int, str]] = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            name = prim.GetName().lower()
            is_visual = "/visuals/" in path.lower()
            if "d435" in name and "rgb" in name:
                scored.append((3 if is_visual else 2, path))
            elif "d435" in name and "imager" in name:
                scored.append((2 if is_visual else 1, path))
            elif "d435" in name and is_visual:
                scored.append((1, path))
        if not scored:
            return
        scored.sort(key=lambda item: -item[0])
        self.head_prim_path = scored[0][1]
        if "/visuals/" in self.head_prim_path.lower():
            self.head_visual_path = self.head_prim_path
        try:
            self.head_xform = XFormPrim(prim_paths_expr=self.head_prim_path)
        except Exception as exc:
            print(f"[BAOEnv] Could not wrap head camera link {self.head_prim_path}: {exc}")
            self.head_xform = None

    def _disable_robot_collisions(self) -> None:
        """Kinematic mode: keep the articulated body but drop its colliders."""
        for sub_prim in self.stage.Traverse():
            path = str(sub_prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            if sub_prim.HasAPI(UsdPhysics.CollisionAPI):
                sub_prim.RemoveAPI(UsdPhysics.CollisionAPI)
            if sub_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                sub_prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

    def _disable_robot_physics(self) -> None:
        """Freeze the robot in its authored pose.

        The robot stays movable kinematically and wall contact is still
        detected by the analytic collision gate, but the physics engine can
        never knock the body over.
        """
        for sub_prim in self.stage.Traverse():
            path = str(sub_prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            for api in (
                UsdPhysics.RigidBodyAPI,
                UsdPhysics.CollisionAPI,
                PhysxSchema.PhysxRigidBodyAPI,
                PhysxSchema.PhysxCollisionAPI,
                PhysxSchema.PhysxArticulationAPI,
                UsdPhysics.ArticulationRootAPI,
            ):
                try:
                    if sub_prim.HasAPI(api):
                        sub_prim.RemoveAPI(api)
                except Exception:
                    pass
            if sub_prim.IsA(UsdPhysics.Joint):
                try:
                    sub_prim.SetActive(False)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def _create_and_bind_glass_material(self, prim_path: str, mat_path: str) -> None:
        """Kept for the optional OmniGlass path; the obstacle wall is opaque now.

        See WALL_COLOR: the panels are solid blue, so this is only used when a
        caller explicitly asks for a translucent wall.
        """
        if not self.task_dict.get("use_omni_glass", False):
            self._create_and_bind_material(
                prim_path,
                mat_path,
                color=WALL_COLOR,
                metallic=0.0,
                roughness=0.5,
                opacity=WALL_OPACITY,
            )
            return
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim {prim_path} not found to bind glass material.")
        sdf_path = Sdf.Path(mat_path)
        mat = UsdShade.Material.Define(self.stage, sdf_path)
        shader = UsdShade.Shader.Define(self.stage, sdf_path.AppendChild("OmniGlass"))
        shader.CreateIdAttr("OmniGlass.mdl")
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.85, 0.92, 0.95)
        )
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.45)
        shader.CreateInput("glass_reflection", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateInput("glass_refraction", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("glass_roughness", Sdf.ValueTypeNames.Float).Set(0.02)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(mat)

    def _create_and_bind_material(
        self,
        prim_path: str,
        mat_path: str,
        color: List[float],
        metallic: float = 1.0,
        roughness: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim {prim_path} not found to bind material.")
        sdf_path = Sdf.Path(mat_path)
        mat = UsdShade.Material.Define(self.stage, sdf_path)
        shader = UsdShade.Shader.Define(self.stage, sdf_path.AppendChild("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(mat)

    # ------------------------------------------------------------------
    # Robot pose / articulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _yaw_quat(yaw_deg: float) -> np.ndarray:
        """Quaternion (w, x, y, z) for yaw around Isaac Sim's +z (up)."""
        half = float(np.radians(yaw_deg)) / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    def _set_robot_pose(self, position: np.ndarray, yaw_deg: float) -> None:
        if self.robot_root is None:
            return
        pos = np.asarray(position, dtype=float).copy()
        pos[1] += float(self.task_dict.get("robot_root_y_offset", 0.0))
        yaw_offset = float(self.task_dict.get("robot_yaw_offset", MODEL_YAW_OFFSET_DEG))
        quat = self._yaw_quat(yaw_deg + yaw_offset)
        pos_isaac = _user_to_isaac_pos(pos)
        pos_isaac[2] += self._robot_ground_offset
        try:
            self.robot_root.set_world_poses(
                positions=np.array([pos_isaac]),
                orientations=np.array([quat]),
            )
        except Exception:
            pass
        self._robot_yaw = float(yaw_deg)

    def _root_position(self) -> np.ndarray:
        if self.robot_root is None:
            return self._start_pos.copy()
        try:
            pos = self.robot_root.get_world_poses()[0][0]
        except Exception:
            return self._start_pos.copy()
        user_pos = _isaac_to_user_pos(np.asarray(pos, dtype=float))
        # The USD root is raised by the ground offset when written, so strip it
        # back out here to keep x/y/z in ground-relative user coordinates.
        user_pos[1] -= self._robot_ground_offset
        return user_pos

    def _init_robot_controller(self) -> bool:
        if not self.task_dict.get("robot_physics", False):
            self._articulation_ok = False
            return False
        try:
            if self._articulation is None:
                import_errors = []
                articulation_class = None
                for module_name in (
                    "isaacsim.core.api.articulations",
                    "isaacsim.core.articulations",
                    "omni.isaac.core.articulations",
                ):
                    try:
                        articulation_class = getattr(
                            __import__(module_name, fromlist=["Articulation"]),
                            "Articulation",
                        )
                        break
                    except Exception as exc:
                        import_errors.append(f"{module_name}: {exc}")
                if articulation_class is None:
                    try:
                        from isaacsim.core.prims import SingleArticulation

                        articulation_class = SingleArticulation
                    except Exception as exc:
                        import_errors.append(
                            f"isaacsim.core.prims.SingleArticulation: {exc}"
                        )
                    try:
                        from isaacsim.core.prims import Articulation

                        articulation_class = Articulation
                    except Exception as exc:
                        import_errors.append(f"isaacsim.core.prims.Articulation: {exc}")
                if articulation_class is None:
                    raise ImportError("; ".join(import_errors))
                Articulation = articulation_class

                try:
                    self._articulation = Articulation(prim_path=self.robot_prim_path)
                except TypeError:
                    self._articulation = Articulation(
                        prim_paths_expr=self.robot_prim_path
                    )
                self._articulation.initialize()
            self._articulation.post_reset()
            self._articulation_ok = True
        except Exception as exc:
            available = []
            if self._articulation is not None:
                available = [
                    n for n in dir(self._articulation) if not n.startswith("_")
                ][:60]
            self._articulation_error = f"{exc} | available={available}"
            print(f"[BAOEnv] Articulation init failed, using kinematic pose only: {exc}")
            self._articulation = None
            self._articulation_ok = False
        return self._articulation_ok

    def _articulation_dof_names(self) -> List[str]:
        art = self._articulation
        for name in ("get_dof_names", "get_joint_names"):
            fn = getattr(art, name, None)
            if callable(fn):
                return [str(n) for n in fn()]
        for name in ("dof_names", "joint_names"):
            val = getattr(art, name, None)
            if val is not None:
                return [str(n) for n in val]
        raise AttributeError("no dof-names accessor on articulation")

    def _articulation_set_targets(
        self, positions: np.ndarray, joint_indices: np.ndarray
    ) -> None:
        art = self._articulation
        pos = np.asarray(positions, dtype=float)
        idx = np.asarray(joint_indices, dtype=int)
        for name in (
            "set_joint_targets",
            "set_dof_targets",
            "set_joint_positions_targets",
            "set_dof_positions_targets",
        ):
            fn = getattr(art, name, None)
            if not callable(fn):
                continue
            try:
                fn(positions=pos, joint_indices=idx)
                return
            except TypeError:
                try:
                    fn(pos, idx)
                    return
                except TypeError:
                    continue
        controller_fn = getattr(art, "get_articulation_controller", None)
        if callable(controller_fn):
            try:
                controller = controller_fn()
                for name in ("set_joint_target_positions", "set_joint_positions"):
                    fn = getattr(controller, name, None)
                    if not callable(fn):
                        continue
                    try:
                        fn(positions=pos, joint_indices=idx)
                        return
                    except TypeError:
                        fn(pos, idx)
                        return
            except Exception:
                pass
        set_pos = getattr(art, "set_joint_positions", None)
        if callable(set_pos):
            try:
                set_pos(positions=pos, joint_indices=idx)
                return
            except TypeError:
                try:
                    set_pos(pos, idx)
                    return
                except TypeError:
                    pass
        apply_action = getattr(art, "apply_action", None)
        if callable(apply_action):
            try:
                apply_action({"joint_positions": pos, "joint_indices": idx})
                return
            except Exception:
                pass
        raise AttributeError("no joint-target setter on articulation")

    def _articulation_joint_positions(self) -> np.ndarray:
        art = self._articulation
        for name in ("get_joint_positions", "get_dof_positions"):
            fn = getattr(art, name, None)
            if callable(fn):
                return np.asarray(fn(), dtype=float).reshape(-1)
        for name in ("joint_positions", "dof_positions"):
            val = getattr(art, name, None)
            if val is not None:
                return np.asarray(val, dtype=float).reshape(-1)
        return np.zeros(0)

    def _set_standing_joint_targets(self) -> None:
        """Pose the robot upright with both arms hanging straight down."""
        if not self._articulation_ok or self._articulation is None:
            return
        try:
            names = self._articulation_dof_names()
            indices: List[int] = []
            positions: List[float] = []
            hang_shoulder = float(
                self.task_dict.get(
                    "arm_hang_shoulder_pitch_rad", ARM_HANG_SHOULDER_PITCH_RAD
                )
            )
            hang_elbow = float(
                self.task_dict.get("arm_hang_elbow_pitch_rad", ARM_HANG_ELBOW_PITCH_RAD)
            )
            for i, name in enumerate(names):
                lower = name.lower()
                if any(key in lower for key in ("hip", "knee")):
                    indices.append(i)
                    positions.append(0.0)
                elif "shoulder_roll" in lower:
                    indices.append(i)
                    positions.append(0.0)
                elif "shoulder_pitch" in lower:
                    indices.append(i)
                    positions.append(hang_shoulder)
                elif "elbow" in lower:
                    indices.append(i)
                    positions.append(hang_elbow)
            if indices:
                self._articulation_set_targets(
                    np.asarray(positions, dtype=float),
                    np.array(indices, dtype=int),
                )
        except Exception as exc:
            print(f"[BAOEnv] standing joint init skipped: {exc}")

    def _analytic_hand_position(self) -> np.ndarray:
        """Approximate hand position from the hanging-arm kinematic model."""
        root = self._root_position()
        yaw_offset = float(self.task_dict.get("robot_yaw_offset", MODEL_YAW_OFFSET_DEG))
        local = np.array([0.10, 0.95, 0.24], dtype=float)
        return root + _rotate_xz(local, np.radians(self._robot_yaw + yaw_offset))

    def _update_camera(self) -> None:
        """Third-person overview camera: head height, looking down the channel."""
        pos = self._root_position()
        forward = _forward_vector(np.radians(self._robot_yaw))
        forward_offset = float(self.task_dict.get("camera_forward_offset", 0.42))
        camera_height = float(self.task_dict.get("camera_height", 1.6))
        eye = np.array([pos[0], camera_height, pos[2]]) + forward * forward_offset
        look = float(self.task_dict.get("camera_look_distance", 2.0))
        target = np.array(
            [pos[0] + forward[0] * look, 0.9, pos[2] + forward[2] * look], dtype=float
        )
        eye_isaac = _user_to_isaac_pos(eye)
        target_isaac = _user_to_isaac_pos(target)
        try:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                up=[0.0, 0.0, 1.0],
                camera_prim_path="/World/Camera",
            )
        except TypeError:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                camera_prim_path="/World/Camera",
            )

    def _update_eye_camera(self) -> None:
        """Point the robot eye camera along robot yaw plus camera offset.

        The original reach-the-ball build forced the look-at point to the
        target ball's height (`target[1] = TARGET_POS[1]`), which pitched the
        head camera downward.  Removing the ball removed that pitch, and a
        camera sitting 0.5 m from a 2.0 m wall at head height, looking
        perfectly level, sees nothing but a featureless grey plane -- the
        opening and the wall are indistinguishable and no floor is visible, so
        there is no visual frame of reference at all.

        Reinstating an explicit downward pitch restores a readable view: floor,
        wall and the channel region all appear in the same frame.
        """
        if self.eye_camera is None:
            return
        eye = self._head_camera_position()
        look_yaw = self._robot_yaw + self._camera_yaw_offset
        distance = float(self.task_dict.get("eye_look_distance", 2.0))
        pitch_deg = float(self.task_dict.get("eye_pitch_deg", EYE_PITCH_DEG))
        # Unit vector pitched down by `pitch_deg` in the vertical plane.
        target = eye + _eye_look_direction(
            np.radians(look_yaw), pitch_deg, distance
        )
        eye_isaac = _user_to_isaac_pos(eye)
        target_isaac = _user_to_isaac_pos(target)
        try:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                up=[0.0, 0.0, 1.0],
                camera_prim_path="/World/RobotEyeCamera",
            )
        except TypeError:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                camera_prim_path="/World/RobotEyeCamera",
            )

    def _head_camera_position(self) -> np.ndarray:
        """Robot eye anchor: on the body, at the real H1 head camera height.

        The reach-the-ball build used 1.9 m because the head had to see a ball
        floating at 1.2 m across the room.  The measured H1 is 1.806 m tall, so
        the honest anchor for a human-like eye is 1.68 m -- about 93% of the
        robot's height, which is where a person's eyes sit.  ``eye_camera_height``
        overrides it.
        """
        root = self._root_position()
        forward = _forward_vector(np.radians(self._robot_yaw))
        height = float(self.task_dict.get("eye_camera_height", EYE_CAMERA_HEIGHT))
        offset = float(self.task_dict.get("eye_forward_offset", 0.0))
        return np.array([root[0], height, root[2]], dtype=float) + forward * offset

    # ------------------------------------------------------------------
    # Public environment interface
    # ------------------------------------------------------------------

    def reset_scene(self) -> np.ndarray:
        """Reset the robot to the start pose and return the first RGB frame."""
        self.world.reset()
        # The authored USD bounds are only meaningful once the physics/app has
        # settled, and world.reset() re-applies the reference pose.  Re-measure
        # here so the reported height is the real one rather than whatever the
        # half-initialised stage showed at construction time.
        self._robot_ground_offset = self._compute_robot_ground_offset()
        if self.task_dict.get("hide_robot", False):
            for sub_prim in self.stage.Traverse():
                if str(sub_prim.GetPath()).startswith(self.robot_prim_path):
                    try:
                        sub_prim.SetActive(False)
                    except Exception:
                        pass
        self.camera.initialize()
        if self.eye_camera is not None:
            self.eye_camera.initialize()
        self._init_robot_controller()
        self._set_robot_pose(self._start_pos, ROBOT_START_YAW_DEG)
        self._camera_yaw_offset = 0.0
        if self._articulation is not None:
            try:
                self._articulation.post_reset()
            except Exception:
                pass
        self._set_standing_joint_targets()
        self._update_camera()
        self._update_eye_camera()
        for _ in range(int(self.task_dict.get("reset_steps", 30))):
            self.world.step(render=True)
        return self.get_camera_image()

    def reset(self) -> np.ndarray:
        """Alias for reset_scene (MirrorBench compatibility)."""
        return self.reset_scene()

    def get_camera_image(self) -> np.ndarray:
        if self.eye_camera is not None:
            return self.eye_camera.get_rgb()
        return self.camera.get_rgb()

    def get_robot_state(self) -> Dict[str, Any]:
        pos = self._root_position()
        hand = self.get_hand_position()
        joints: Dict[str, float] = {}
        if self._articulation_ok and self._articulation is not None:
            try:
                names = self._articulation_dof_names()
                values = self._articulation_joint_positions()
                joints = {name: float(v) for name, v in zip(names, values)}
            except Exception:
                joints = {}
        return {
            "position": pos.tolist(),
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": self._robot_yaw},
            "joint_angles": joints,
            "end_effector_position": hand.tolist(),
            "hand_position": hand.tolist(),
            "torso_rotation": self.get_torso_rotation(),
            "camera_yaw": self._camera_yaw_offset,
            "channel_width": self.get_channel_width(),
            "a_s_ratio": self.get_a_s_ratio(),
            "distance_to_goal": self.get_distance_to_goal(),
            "move_step": float(self._move_step),
        }

    def get_torso_rotation(self) -> float:
        """Return torso yaw in degrees (0 = facing +x, 90 = sideways)."""
        return float(self._robot_yaw)

    def get_abs_torso_rotation(self) -> float:
        """Absolute torso yaw in degrees, folded into [0, 180]."""
        yaw = abs(self.get_torso_rotation()) % 360.0
        return float(360.0 - yaw if yaw > 180.0 else yaw)

    def is_sideways(self, tolerance_deg: float = 0.0) -> bool:
        """True when the torso is rotated enough to present the thin profile."""
        yaw = self.get_abs_torso_rotation()
        return bool(
            SIDEWAYS_YAW_MIN_DEG - tolerance_deg
            <= yaw
            <= SIDEWAYS_YAW_MAX_DEG + tolerance_deg
        )

    def get_hand_position(self) -> np.ndarray:
        if self._articulation_ok and self.hand_xform is not None:
            try:
                pos = self.hand_xform.get_world_poses()[0][0]
                return _isaac_to_user_pos(np.asarray(pos, dtype=float))
            except Exception:
                pass
        return self._analytic_hand_position()

    def get_distance_to_goal(self) -> float:
        """Signed distance along +x to the passage line (positive = not yet)."""
        return float(SUCCESS_X - self._root_position()[0])

    def check_collision_with_wall(self) -> bool:
        """Return True when the robot is currently colliding with a panel."""
        return self._get_wall_collision_info() is not None

    def get_collision_position(self) -> Optional[List[float]]:
        """Return the current collision point (x, y, z) or None."""
        info = self._get_wall_collision_info()
        return list(info["point"]) if info else None

    def check_success(self) -> bool:
        """Success condition: the body centre reaches the x=11 m goal plane."""
        return bool(self._root_position()[0] >= SUCCESS_X)

    def _get_wall_collision_info(self) -> Optional[Dict[str, Any]]:
        return _check_wall_collision(
            self._root_position(),
            np.radians(self._robot_yaw),
            channel_width=self._channel_width,
        )

    def execute_action(self, action: str, n_steps: Optional[int] = None) -> StepResult:
        if n_steps is None:
            n_steps = int(self.task_dict.get("action_steps", 30))
        action = str(action).strip().lower()
        if action not in ACTIONS:
            return StepResult(
                rgb=self.get_camera_image(),
                legal=False,
                feedback=f"unknown action: {action}",
                success=self.check_success(),
            )

        legal, feedback, collision = self._apply_action(action)
        self._update_camera()
        self._update_eye_camera()
        for _ in range(int(n_steps)):
            self.world.step(render=True)

        return StepResult(
            rgb=self.get_camera_image(),
            legal=legal,
            feedback=feedback,
            success=self.check_success(),
            collision=collision,
            state=self.get_robot_state(),
        )

    def step_wait(self, n_steps: int = 10000) -> None:
        for _ in range(n_steps):
            self.world.step(render=True)

    def close(self) -> None:
        if self.sim_app is not None:
            self.sim_app.close()

    # ------------------------------------------------------------------
    # Internal action execution
    # ------------------------------------------------------------------

    def _apply_action(self, action: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        root = self._root_position()
        yaw = np.radians(self._robot_yaw)

        if action in ("forward", "backward", "left", "right"):
            # Egocentric translations: forward/backward follow the robot's
            # facing direction; left/right are relative to the robot.
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
            # A turn is legal only when the whole swept arc stays clear of the
            # panels, measured from the current pose.  Near the wall the
            # shoulders would sweep through the panels, so the agent has to
            # rotate while still in the free space in front of (or behind) the
            # wall -- which is exactly the behaviour the benchmark measures.
            collision = _turn_path_is_clear(
                root,
                self._robot_yaw,
                new_yaw,
                channel_width=self._channel_width,
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

        return False, "invalid action", None


def setup_scene(sim_app: Any = None, task_dict: Optional[Dict[str, Any]] = None) -> BAOEnv:
    """Convenience factory used by main.py: create and return a BAOEnv."""
    return BAOEnv(sim_app=sim_app, task_dict=task_dict)


def _smoke_test() -> None:
    """Minimal headless smoke test (run with the Isaac Sim python)."""
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    env = BAOEnv(simulation_app, task_dict={"headless": True})
    rgb = env.reset_scene()
    print(f"[smoke] reset rgb={rgb.shape} width={env.get_channel_width()} m")
    # Sideways route: rotate 90 degrees in free space, then translate along +z.
    actions = ["turn_left"] * 6 + ["right"] * 14
    for action in actions:
        result = env.execute_action(action)
        print(
            f"[smoke] {action:11s} legal={result.legal} "
            f"feedback={result.feedback} x={env.get_distance_to_goal():+.3f} "
            f"success={result.success}"
        )
    env.close()


if __name__ == "__main__":
    _smoke_test()
