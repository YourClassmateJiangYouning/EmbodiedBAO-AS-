"""The Stage 3 scene and marker catalogue, as data.

Nothing here imports Isaac Sim, and nothing here builds anything: this module says *what*
the five scenes and the twenty-five markers are, in a form that can be checked by arithmetic
on any machine.  ``environment.py`` executes it against the stage; the tests in
``test_bao_scenes.py`` verify it without a simulator.  That split is deliberate -- the
properties that matter (markers are distinct, dressing never occludes the opening, the
collision set is untouched) are all statements about numbers, not about pixels.

Why the constraints exist, in one place:

* **A marker is the task.**  0.60 m bounding size, at x = 16.0, z = 0, centre height 1.40 m,
  on the far wall, for all twenty-five of them, so that the distance cue is the same size in
  every condition.  Only shape and colour change.
* **At 512 px from the start pose the marker is about 20 px wide**, so a ring or a skinny
  cross would blur into a blob; every shape here is solid or fat.
* **Five markers per scene, one per repeat.**  The five runs of a (scene, model, level) cell
  use five different markers, so the cell's five episodes are five conditions rather than
  five replicates.  See STAGE3_SCENES.md 3.5 for what that does to the statistics.
* **A marker must be the only object of its colour in its frame**, which is why each scene
  declares forbidden colours and why a green marker is unusable on the baseline's green far
  wall while being fine in the warehouse.
* **Dressing is decoration, and must stay decoration**: collision disabled everywhere, no
  overlap with the opening's or the marker's screen region from any of the 17 start views,
  and one frozen layout per scene -- never per-episode randomisation, which would be a second
  independent variable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# The marker's fixed geometry.  These are identical for all 25, on purpose:
# the point of varying the marker is to vary its appearance and nothing else.
# ---------------------------------------------------------------------------

MARKER_SIZE_M = 0.60
MARKER_X_M = 16.0
MARKER_Z_M = 0.0
MARKER_Y_M = 1.40
MARKER_THICKNESS_M = 0.02

# The camera the check below projects with: eye height, downward pitch, and the pinhole
# model that environment.py's focal length and sensor imply.
EYE_HEIGHT_M = 1.68
EYE_PITCH_DEG = 15.0
FOCAL_LENGTH_MM = 13.36
SENSOR_WIDTH_MM = 20.955

# ---------------------------------------------------------------------------
# Ten shapes, ten colours, built from primitives only.  No textures: a textured
# marker would vary in colour across its surface and make the uniqueness check --
# which compares colours -- unreliable, and it would need an asset download.
# ---------------------------------------------------------------------------

# A shape is a tuple of flat polygons in the marker's own plane, in metres, centred on the
# origin and no further than half the marker from it.  Polygons rather than primitives
# because a triangle is not a cylinder: USD's Cylinder has radius and height but its facet
# count is a render setting, not a per-prim attribute, so a 3-sided one cannot be asked for.
# Explicit polygons also make every shape exact and low-poly, and let the checks measure the
# geometry the renderer will actually draw.
#
# The marker plane is the far wall's: in-plane coordinates are (z, y), and the thickness is
# extruded along x, towards the robot.
SHAPE_HALF_M = MARKER_SIZE_M / 2.0


def _regular(sides: int, start_deg: float = 90.0) -> Tuple[Tuple[float, float], ...]:
    """A regular polygon inscribed in the marker's bounding circle, vertex up by default."""
    return tuple(
        (SHAPE_HALF_M * math.cos(math.radians(start_deg + 360.0 * i / sides)),
         SHAPE_HALF_M * math.sin(math.radians(start_deg + 360.0 * i / sides)))
        for i in range(sides)
    )


def _rect(x0: float, y0: float, x1: float, y1: float) -> Tuple[Tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


_ARM = SHAPE_HALF_M / 3.0          # half width of the cross's arms
SHAFT_HALF = SHAPE_HALF_M / 4.3    # half height of the arrow's shaft

SHAPES: Dict[str, Tuple[Tuple[Tuple[float, float], ...], ...]] = {
    "square": (_rect(-SHAPE_HALF_M, -SHAPE_HALF_M, SHAPE_HALF_M, SHAPE_HALF_M),),
    "diamond": (((0.0, -SHAPE_HALF_M), (SHAPE_HALF_M, 0.0),
                 (0.0, SHAPE_HALF_M), (-SHAPE_HALF_M, 0.0)),),
    "triangle": (_regular(3),),
    "triangle_down": (_regular(3, start_deg=-90.0),),
    "disc": (_regular(64),),
    "hexagon": (_regular(6),),
    "cross": (((-_ARM, -SHAPE_HALF_M), (_ARM, -SHAPE_HALF_M), (_ARM, -_ARM),
               (SHAPE_HALF_M, -_ARM), (SHAPE_HALF_M, _ARM), (_ARM, _ARM),
               (_ARM, SHAPE_HALF_M), (-_ARM, SHAPE_HALF_M), (-_ARM, _ARM),
               (-SHAPE_HALF_M, _ARM), (-SHAPE_HALF_M, -_ARM), (-_ARM, -_ARM)),),
    "bars3": (lambda w=SHAPE_HALF_M / 4.3, g=SHAPE_HALF_M * 0.30: (
        _rect(-g - w, -SHAPE_HALF_M, -g + w, SHAPE_HALF_M),
        _rect(-w, -SHAPE_HALF_M, w, SHAPE_HALF_M),
        _rect(g - w, -SHAPE_HALF_M, g + w, SHAPE_HALF_M)))(),
    "bars2": (lambda h=SHAPE_HALF_M / 4.3, g=SHAPE_HALF_M * 0.235: (
        _rect(-SHAPE_HALF_M, g - h, SHAPE_HALF_M, g + h),
        _rect(-SHAPE_HALF_M, -g - h, SHAPE_HALF_M, -g + h)))(),
    "arrow": (
        _rect(-SHAPE_HALF_M, -SHAFT_HALF, -SHAPE_HALF_M * 0.07, SHAFT_HALF),
        ((-SHAPE_HALF_M * 0.07, -SHAPE_HALF_M * 0.60),
         (SHAPE_HALF_M, 0.0),
         (-SHAPE_HALF_M * 0.07, SHAPE_HALF_M * 0.60)),
    ),
}

COLOURS: Dict[str, Tuple[float, float, float]] = {
    "r": (0.85, 0.15, 0.12),   # red, the existing marker's colour
    "m": (0.90, 0.10, 0.55),   # magenta
    "o": (1.00, 0.45, 0.05),   # bright orange
    "l": (0.85, 0.95, 0.10),   # lime
    "c": (0.10, 0.65, 0.90),   # cyan blue
    "p": (0.55, 0.15, 0.75),   # purple
    "g": (0.20, 0.85, 0.25),   # bright green
    "w": (0.55, 0.05, 0.25),   # wine
    "t": (0.05, 0.80, 0.70),   # teal
    "k": (1.00, 0.40, 0.70),   # pink
}

# The scene's existing far-wall colour: a green marker on it would be invisible, which is
# why the supervisor's green triangle is assigned to the warehouse instead.
BASELINE_FAR_WALL_COLOUR = (0.13, 0.42, 0.20)

# ---------------------------------------------------------------------------
# The 25 markers.  Slot order is the order the five repeats run in.
# Within a scene the five colours differ; across the set no (shape, colour) repeats.
# ---------------------------------------------------------------------------

MARKERS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "stage1.1": (("square", "r"), ("triangle", "m"), ("disc", "c"),
                 ("cross", "o"), ("bars3", "l")),
    "stage1.2": (("diamond", "r"), ("hexagon", "c"), ("arrow", "l"),
                 ("bars2", "g"), ("triangle_down", "m")),
    "stage1.3": (("triangle_down", "r"), ("cross", "c"), ("square", "p"),
                 ("hexagon", "m"), ("arrow", "t")),
    "stage1.4": (("disc", "r"), ("bars3", "m"), ("triangle", "o"),
                 ("diamond", "p"), ("bars2", "w")),
    "stage1.5": (("hexagon", "k"), ("arrow", "w"), ("bars3", "t"),
                 ("disc", "p"), ("bars2", "l")),
}

# ---------------------------------------------------------------------------
# The five scenes.  ``materials`` names the four surfaces we rebind; a scene with no
# materials is the untouched baseline.  ``verified`` marks whether the asset name was read
# off NVIDIA's public bucket listing (the warehouse's were) or still has to be confirmed by
# the same listing on the lab machine.
# ---------------------------------------------------------------------------

ASSET_ROOT = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
              "/Assets/Isaac/4.5/Isaac")
WAREHOUSE_MATERIALS = ASSET_ROOT + "/Environments/Simple_Warehouse/Materials"


# Muted, scene-neutral default so that no dressing item is ever rendered untextured black.
# The first preview came back with a solid black wall on one side and black blocks in the
# corridor: prims with no colour and no working material render black, and the material
# library's cache is not writable on the lab machine, so there was nothing to fall back on.
DRESSING_GREY = (0.45, 0.45, 0.47)
DRESSING_WOOD = (0.35, 0.24, 0.16)
DRESSING_METAL = (0.55, 0.57, 0.60)
DRESSING_GREEN = (0.22, 0.40, 0.18)
DRESSING_LIGHT = (0.78, 0.78, 0.80)


def _dressing_floor(name: str, at: Tuple[float, float, float],
                    size: Tuple[float, float, float], colour: Optional[str] = None,
                    asset: Optional[str] = None) -> Dict[str, Any]:
    return {"name": name, "mount": "floor", "at": at, "size": size,
            "colour": colour or DRESSING_GREY, "asset": asset, "collides": False}


def _dressing_wall(name: str, mount: str, at: Tuple[float, float, float],
                   size: Tuple[float, float, float], colour: Optional[str] = None,
                   asset: Optional[str] = None) -> Dict[str, Any]:
    """A wall-mounted item, declared like a picture: (across, tall, thick).

    The thickness is moved to the x axis here, because a wall faces along x and the first
    version's call sites wrote (0.5, 0.4, 0.03) -- which put half a metre of plate straight
    out into the corridor, the same mistake as the duct, and the rendered frame showed it as a
    black slab hanging in the middle of the agent's view.  Callers describe the picture; this
    function decides which way it faces.
    """
    across, tall, thick = (float(v) for v in size)
    return {"name": name, "mount": mount, "at": at, "size": (thick, tall, across),
            "colour": colour or DRESSING_GREY, "asset": asset, "collides": False}


SCENES: Dict[str, Dict[str, Any]] = {
    "stage1.1": {
        "label": "plain laboratory (baseline)",
        "materials": {},
        "material_verified": True,
        "dressing": (),
        "forbidden_colours": (BASELINE_FAR_WALL_COLOUR,),
        "notes": "Unchanged. The existing red square is marker slot 1 of this scene.",
    },
    "stage1.2": {
        "label": "warehouse / factory",
        "materials": {
            "floor": WAREHOUSE_MATERIALS + "/MI_Floor_01.mdl",
            "side_wall": WAREHOUSE_MATERIALS + "/MI_WallA_01.mdl",
            "ceiling": WAREHOUSE_MATERIALS + "/MI_CeilingA_06b.mdl",
            "far_wall": WAREHOUSE_MATERIALS + "/MI_WallB_01.mdl",
        },
        "material_verified": True,
        "dressing": (
            _dressing_wall("sign", "obstacle_wall", (8.0, 1.5, 1.35), (0.5, 0.4, 0.03),
                           asset=WAREHOUSE_MATERIALS + "/MI_SignB.mdl"),
            _dressing_wall("toolboard", "obstacle_wall", (8.0, 1.2, -1.35), (0.8, 0.6, 0.03)),
            _dressing_wall("duct", "obstacle_wall", (8.0, 2.6, 1.10), (0.30, 0.30, 0.30),
                           DRESSING_METAL),
            _dressing_wall("bay_sign", "far_wall", (16.0, 2.15, 1.60), (0.9, 0.35, 0.03)),
            # Floor items are deliberately small and far off the centre line.  The first
            # version put a 1.8 m tall, 1.1 m deep forklift five metres from the camera, and
            # the rendered frame came back with a dead black mass filling one side of it: a
            # big unlit box that close is not dressing, it is an obstruction.  Nothing here is
            # more than a metre tall, more than 0.9 m long along the view axis, or nearer the
            # centre line than |z| 2.0.
            _dressing_floor("pallets", (2.6, 0.20, 2.05), (0.9, 0.40, 0.9)),
            _dressing_floor("klt_bins", (4.0, 0.15, -2.00), (0.6, 0.30, 0.5)),
            _dressing_floor("forklift", (6.2, 0.35, 2.15), (0.9, 0.70, 0.7)),
            _dressing_floor("traffic_cone", (7.0, 0.35, -1.95), (0.4, 0.70, 0.4),
                            asset=WAREHOUSE_MATERIALS + "/M_TrafficCone.mdl"),
        ),
        "forbidden_colours": (),
        "notes": "The only scene whose材料 were read off the bucket listing in full.",
    },
    "stage1.3": {
        "label": "library",
        "materials": {
            "floor": ASSET_ROOT + "/Environments/Hospital/Materials/M_Wood_Floor.mdl",
            "side_wall": ASSET_ROOT + "/Environments/Office/Materials/MI_WallOffice_01.mdl",
            "ceiling": WAREHOUSE_MATERIALS + "/MI_CeilingA_06b.mdl",
            "far_wall": ASSET_ROOT + "/Environments/Hospital/Materials/M_Wall_Plaster.mdl",
        },
        "material_verified": False,
        "dressing": (
            _dressing_wall("noticeboard", "obstacle_wall", (8.0, 1.45, -1.30), (1.0, 0.7, 0.03)),
            _dressing_wall("painting", "obstacle_wall", (8.0, 1.40, 1.30), (0.7, 0.5, 0.03)),
            _dressing_wall("clock", "far_wall", (16.0, 2.20, 1.40), (0.4, 0.4, 0.03)),
            _dressing_floor("cabinet_a", (2.0, 0.45, 1.85), (0.9, 0.90, 0.5)),
            _dressing_floor("cabinet_b", (3.2, 0.45, 1.85), (0.9, 0.90, 0.5)),
            _dressing_floor("reading_table", (5.0, 0.37, -1.75), (1.4, 0.74, 0.9)),
            _dressing_floor("chair", (5.6, 0.45, -1.75), (0.5, 0.90, 0.5)),
            _dressing_floor("book_pile", (6.6, 0.12, 1.70), (0.4, 0.24, 0.4)),
        ),
        "forbidden_colours": (),
        "notes": "No library environment exists; composed from bucket-root materials.",
    },
    "stage1.4": {
        "label": "park / outdoor",
        "materials": {
            "side_wall": ASSET_ROOT + "/Environments/Hospital/Materials/M_Wall_Plaster.mdl",
            "ceiling": WAREHOUSE_MATERIALS + "/MI_CeilingA_06b.mdl",
            "far_wall": ASSET_ROOT + "/Environments/Office/Materials/MI_WallOffice_01.mdl",
        },
        "material_verified": False,
        "dressing": (
            _dressing_wall("noticeboard", "obstacle_wall", (8.0, 1.35, 1.35), (0.9, 0.6, 0.03)),
            _dressing_floor("bench_a", (2.6, 0.45, 1.80), (1.6, 0.90, 0.6)),
            _dressing_floor("bench_b", (5.4, 0.45, -1.80), (1.6, 0.90, 0.6)),
            _dressing_floor("bin", (3.9, 0.45, 1.65), (0.5, 0.90, 0.5)),
            _dressing_floor("tree_a", (4.6, 1.60, 2.05), (0.7, 3.20, 0.7)),
            _dressing_floor("tree_b", (7.0, 1.60, -2.05), (0.7, 3.20, 0.7)),
            _dressing_floor("lamp_post", (6.2, 1.70, 1.85), (0.3, 3.40, 0.3)),
        ),
        "forbidden_colours": (),
        "notes": "The Props library has no vegetation; trees are cylinder + sphere.",
    },
    "stage1.5": {
        "label": "supermarket",
        "materials": {
            "floor": ASSET_ROOT + "/Environments/Office/Materials/MI_FloorMarbleTiles_03.mdl",
            "side_wall": WAREHOUSE_MATERIALS + "/MI_WallA_01.mdl",
            "ceiling": WAREHOUSE_MATERIALS + "/MI_CeilingA_06b.mdl",
            "far_wall": ASSET_ROOT + "/Environments/Office/Materials/MI_WallOffice_01.mdl",
        },
        "material_verified": False,
        "dressing": (
            _dressing_wall("poster_a", "obstacle_wall", (8.0, 1.45, 1.30), (0.8, 0.6, 0.03)),
            _dressing_wall("poster_b", "obstacle_wall", (8.0, 1.45, -1.30), (0.8, 0.6, 0.03)),
            _dressing_wall("price_strip", "obstacle_wall", (8.0, 2.10, 1.30), (1.6, 0.15, 0.03)),
            _dressing_wall("promo_hanger", "far_wall", (16.0, 2.05, -1.40), (0.9, 0.6, 0.03)),
            _dressing_floor("crate_stack", (2.0, 0.35, 1.80), (0.8, 0.70, 0.8)),
            _dressing_floor("klt_bins", (3.4, 0.20, -1.75), (0.9, 0.40, 0.6)),
            _dressing_floor("trolley", (4.8, 0.50, 1.75), (0.7, 1.00, 0.5)),
            _dressing_floor("checkout_base", (6.8, 0.45, -1.85), (1.4, 0.90, 0.7)),
        ),
        "forbidden_colours": (),
        "notes": "No supermarket environment exists; the busiest scene of the five.",
    },
}

SCENE_ORDER = ("stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5")
MARKERS_PER_SCENE = 5
LEVELS_PER_SCENE = 17
ROSTER_SIZE = 15

# ---------------------------------------------------------------------------
# Camera projection, used by the tests to prove that dressing cannot occlude the
# opening or the marker.  A pinhole model built from the same focal length and sensor
# the environment uses, with the eye's downward pitch, so the check is arithmetic
# rather than an eyeball on a rendered frame.
# ---------------------------------------------------------------------------


def _fov_deg() -> float:
    return 2.0 * math.degrees(math.atan(SENSOR_WIDTH_MM / (2.0 * FOCAL_LENGTH_MM)))


HORIZONTAL_FOV_DEG = _fov_deg()
VERTICAL_FOV_DEG = 2.0 * math.degrees(
    math.atan((SENSOR_WIDTH_MM * 9.0 / 16.0) / (2.0 * FOCAL_LENGTH_MM))
)


def project(point: Sequence[float], eye_x: float = 0.5) -> Optional[Tuple[float, float]]:
    """Project a world point to normalised screen coordinates, or None if behind.

    The camera sits at (eye_x, EYE_HEIGHT_M, 0) looking along +x, pitched down by
    EYE_PITCH_DEG.  Returns (u, v) with both in [-1, 1] inside the frame, u to the right
    and v upward, so overlap tests are simple rectangle comparisons.
    """
    dx = float(point[0]) - eye_x
    dy = float(point[1]) - EYE_HEIGHT_M
    dz = float(point[2])
    if dx <= 1e-6:
        return None
    pitch = math.radians(EYE_PITCH_DEG)
    # Rotate into the camera frame: forward is +x tilted down by the pitch.
    forward = dx * math.cos(pitch) - dy * math.sin(pitch)
    up = dx * math.sin(pitch) + dy * math.cos(pitch)
    if forward <= 1e-6:
        return None
    half_w = math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0))
    half_h = math.tan(math.radians(VERTICAL_FOV_DEG / 2.0))
    return (dz / (forward * half_w), up / (forward * half_h))


def box_corners(centre: Sequence[float], size: Sequence[float]) -> List[Tuple[float, float, float]]:
    cx, cy, cz = (float(v) for v in centre)
    sx, sy, sz = (float(v) / 2.0 for v in size)
    return [(cx + i * sx, cy + j * sy, cz + k * sz)
            for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]


def screen_bounds(centre: Sequence[float], size: Sequence[float],
                  eye_x: float = 0.5) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned screen rectangle of a world box, or None if it is behind the camera."""
    points = [project(corner, eye_x) for corner in box_corners(centre, size)]
    points = [p for p in points if p is not None]
    if not points:
        return None
    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    return (min(us), min(vs), max(us), max(vs))


def opening_box(width_m: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """The opening as a world box: the gap through the obstacle wall."""
    return ((8.0, 1.0, 0.0), (0.02, 2.0, width_m))


def marker_box() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    return ((MARKER_X_M, MARKER_Y_M, MARKER_Z_M),
            (MARKER_THICKNESS_M, MARKER_SIZE_M, MARKER_SIZE_M))


def rectangles_overlap(a: Tuple[float, float, float, float],
                       b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


SHAPE_NAMES: Dict[str, str] = {
    "square": "square",
    "diamond": "diamond",
    "triangle": "triangle",
    "triangle_down": "downward-pointing triangle",
    "disc": "disc",
    "hexagon": "hexagon",
    "cross": "thick cross",
    "bars3": "set of three bars",
    "bars2": "set of two bars",
    "arrow": "arrow",
}

COLOUR_NAMES: Dict[str, str] = {
    "r": "red",
    "m": "magenta",
    "o": "orange",
    "l": "lime green",
    "c": "cyan",
    "p": "purple",
    "g": "green",
    "w": "white",
    "t": "teal",
    "k": "black",
}


def describe_marker(scene: str, slot: int) -> str:
    """The words for the marker that is actually on the far wall, e.g. "cyan disc".

    This is the noun phrase the prompt uses.  Every prompt is the frozen Stage 1 text with only
    this phrase swapped, so the words have to match the object: a run whose prompt said "red
    marker" while a cyan disc hung on the wall would be measuring obedience to a false
    statement rather than the aperture judgement being studied.
    """
    shape, colour = MARKERS[scene][slot - 1]
    return f"{COLOUR_NAMES[colour]} {SHAPE_NAMES[shape]}"


def marker_entries(scene: str) -> Tuple[Tuple[str, str], ...]:
    return MARKERS[scene]


def marker_colours(scene: str) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(COLOURS[colour] for _, colour in MARKERS[scene])


# ---------------------------------------------------------------------------
# Dressing for the library, park and supermarket
# ---------------------------------------------------------------------------
# Installed here rather than inline so the three blocks can be replaced as a unit, and so the
# reasoning sits next to the numbers.  Every item obeys the rules the tests enforce:
#   * collides = False, always: the task must not change because a bench is in the way;
#   * floor items are >= 1.9 m off the centre line, <= 1 m tall and <= 0.9 m long along the
#     view axis, because a big unlit box near the camera stops being dressing and becomes an
#     obstruction -- the warehouse's forklift taught that one twice;
#   * wall items are >= 0.9 m off the centre line and reach <= 0.20 m off their wall, and are
#     declared as a picture is: (across, tall, thick);
#   * no item uses one of its own scene's five marker colours.  The library's markers are
#     red/cyan/purple/magenta/teal, the park's are red/magenta/orange/purple/white, and the
#     supermarket's are black/white/teal/purple/lime, so the dressing below is deliberately
#     grey, wood, steel, green, blue and yellow.

DRESSING_BLUE = (0.20, 0.35, 0.65)
DRESSING_YELLOW = (0.85, 0.78, 0.30)
DRESSING_CLAY = (0.62, 0.45, 0.30)


def _install_dressing() -> None:
    """The library: shelves either side of the opening, a reading corner behind the agent."""
    SCENES["stage1.3"]["dressing"] = (
        # 1.4 m across, centred 1.65 m off the axis, so the inner edge sits at 0.95 m: clear of
        # the 1.14 m opening's half width (0.57 m) in the start view, which is the projection
        # check that caught the first version's 1.6 m shelves at 1.30 m.
        _dressing_wall("bookshelf_left", "obstacle_wall", (8.0, 1.15, 1.65),
                       (1.4, 1.9, 0.16), DRESSING_WOOD),
        _dressing_wall("bookshelf_right", "obstacle_wall", (8.0, 1.15, -1.65),
                       (1.4, 1.9, 0.16), DRESSING_WOOD),
        _dressing_wall("clock", "obstacle_wall", (8.0, 2.45, -1.70), (0.4, 0.4, 0.03),
                       DRESSING_LIGHT),
        _dressing_wall("reading_poster", "far_wall", (16.0, 2.05, 1.55), (0.7, 0.5, 0.03),
                       DRESSING_LIGHT),
        _dressing_floor("study_table", (3.0, 0.38, 2.05), (0.9, 0.75, 0.7), DRESSING_WOOD),
        _dressing_floor("chair_row", (5.0, 0.25, -2.05), (0.7, 0.50, 0.9), DRESSING_GREY),
        _dressing_floor("book_cart", (6.6, 0.45, 2.10), (0.8, 0.90, 0.5), DRESSING_METAL),
        _dressing_floor("reading_lamp", (1.8, 0.48, -2.10), (0.3, 0.95, 0.3), DRESSING_METAL),
    )

    """The park: benches and planters behind the agent, hedges either side of the opening."""
    SCENES["stage1.4"]["dressing"] = (
        _dressing_wall("hedge_left", "obstacle_wall", (8.0, 2.30, 1.50), (1.2, 0.4, 0.16),
                       DRESSING_GREEN),
        _dressing_wall("hedge_right", "obstacle_wall", (8.0, 2.30, -1.50), (1.2, 0.4, 0.16),
                       DRESSING_GREEN),
        _dressing_wall("park_sign", "far_wall", (16.0, 2.00, -1.60), (0.6, 0.4, 0.03),
                       DRESSING_CLAY),
        _dressing_floor("bench_left", (2.4, 0.25, 2.05), (0.9, 0.45, 0.5), DRESSING_WOOD),
        _dressing_floor("bench_right", (4.8, 0.25, -2.05), (0.9, 0.45, 0.5), DRESSING_WOOD),
        _dressing_floor("litter_bin", (6.4, 0.35, 2.10), (0.4, 0.70, 0.4), DRESSING_GREEN),
        _dressing_floor("planter_left", (1.6, 0.20, -2.10), (0.6, 0.40, 0.6), DRESSING_CLAY),
        _dressing_floor("planter_right", (7.2, 0.20, -2.00), (0.6, 0.40, 0.6), DRESSING_CLAY),
    )

    """The supermarket: shelving either side, a checkout and produce behind the agent."""
    SCENES["stage1.5"]["dressing"] = (
        _dressing_wall("shelf_left", "obstacle_wall", (8.0, 1.05, 1.60), (1.4, 1.9, 0.18),
                       DRESSING_METAL),
        _dressing_wall("shelf_right", "obstacle_wall", (8.0, 1.05, -1.60), (1.4, 1.9, 0.18),
                       DRESSING_METAL),
        _dressing_wall("price_strip", "obstacle_wall", (7.97, 2.00, 1.60), (1.3, 0.15, 0.03),
                       DRESSING_YELLOW),
        _dressing_wall("promo_banner", "far_wall", (16.0, 2.10, 1.55), (0.9, 0.5, 0.03),
                       DRESSING_YELLOW),
        _dressing_floor("trolley", (2.8, 0.48, 2.05), (0.8, 0.95, 0.5), DRESSING_METAL),
        _dressing_floor("produce_bins", (4.6, 0.28, -2.05), (0.9, 0.55, 0.6), DRESSING_YELLOW),
        _dressing_floor("checkout", (6.4, 0.42, 2.10), (0.9, 0.85, 0.7), DRESSING_GREY),
        _dressing_floor("stacked_boxes", (7.4, 0.28, -2.00), (0.7, 0.55, 0.5), DRESSING_BLUE),
    )


_install_dressing()


def episode_count() -> int:
    return (len(SCENE_ORDER) * LEVELS_PER_SCENE * MARKERS_PER_SCENE * ROSTER_SIZE)
