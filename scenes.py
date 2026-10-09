"""The Stage 1 scene and marker catalogue, as data.

Stage numbering, because this is easy to mis-file and was mis-filed once.  **Stage 1 is the
A/S threshold study, and the scenes below belong to it**: the supervisor asked for the site
to look like an everyday place, and the five repeats of a (scene, model, level) cell became
five different markers rather than five identical ones.  **Stage 2** is the memory/note
experiment (``memory_protocol.py``, spec in ``STAGE23_DESIGN.md``).  **Stage 3** is the
follow-up phases of that same memory experiment -- a second passage back through a narrow
opening and a wide-then-narrow control -- and ``STAGE23_DESIGN.md`` §10 lists both under
"explicitly not doing".  It is NOT the scene-variant matrix: those ten variants change the
marker, the far wall, the floor and the prompt, so they are Stage 1 variables, and they
live in ``STAGE1_SCENE_VARIANTS.md``.

The two documents used to be named ``STAGE3_SCENES.md`` and ``STAGE3_SCENE_VARIANTS.md``,
which is what put the wrong stage number into this file's first line and into
``scene_builder.py``'s; they are now ``STAGE1_SCENES.md`` and ``STAGE1_SCENE_VARIANTS.md``.

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
  five replicates.  See STAGE1_SCENES.md 3.5 for what that does to the statistics.
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
MARKER_Z_M = 0.0
MARKER_Y_M = 1.40
MARKER_THICKNESS_M = 0.02

# The far wall's inner face, and where the plate sits relative to it.
#
# environment._create_room builds that wall with its centre at
# ``ROOM_LENGTH_X + thickness/2``, so its INNER face -- the surface the room sees -- is
# exactly at ``ROOM_LENGTH_X`` (16.0).  The plate therefore has to be placed so that its
# BACK face is at or in front of 16.0, not centred on it:
#
#   centred on the wall plane   -> spans 15.99 .. 16.01, burying half its thickness in
#                                  the wall and moving the face the camera sees 0.02 m
#                                  further away than the environment's own marker;
#   back face flush + clearance -> spans 15.97 .. 15.99, which is exactly where the
#                                  environment's marker is.
#
# ``MARKER_CLEARANCE_M`` is that clearance.  It is not a new number: the environment's
# centre line is ``ROOM_LENGTH_X - thickness/2 - 0.01``, i.e. a 0.01 m gap in front of
# the wall, and this reproduces it so that ``stage1.1`` slot 1 is the same plate by the
# same coordinates whichever construction path built it.  Getting this wrong is silent
# in a still frame -- 0.02 m at 15.5 m is under a pixel -- so it is pinned by a test that
# compares built geometry rather than declared coordinates.
MARKER_WALL_X_M = 16.0
MARKER_CLEARANCE_M = 0.01
MARKER_X_M = MARKER_WALL_X_M - MARKER_THICKNESS_M / 2.0 - MARKER_CLEARANCE_M

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
# materials is the untouched baseline.  ``material_verified`` marks whether the asset
# name was read off NVIDIA's public bucket listing (the warehouse's were) or still has to
# be confirmed by the same listing on the lab machine.
#
# WHERE A SCENE'S DRESSING LIVES.  Two of the five carry their layout inline, as the
# `dressing` value right here: 1.1 has none, and 1.2 has its eight items.  The other
# three -- 1.3, 1.4, 1.5 -- have an empty tuple here and are filled in by
# ``_install_dressing`` at the bottom of the module.
#
# That split is not a style choice, it is a record of how the layouts were arrived at.
# The 1.2 list never had to change, so it stayed where it was written.  The other three
# were re-cut twice against rendered previews -- shelves 1.6 m wide at |z| 1.30 m
# projected straight across the opening and had to move out to 1.4 m at 1.65 m -- and
# keeping each layout next to its reasoning was worth more than keeping all five in one
# place.  What is NOT acceptable is having both: an earlier revision carried full inline
# tuples for 1.3, 1.4 and 1.5 as well, which ``_install_dressing`` then overwrote at
# import, so 24 lines of configuration that read exactly like the live layout were dead.
# A scene therefore has its layout in exactly one of the two places, and
# test_bao_scenes.py checks that.
# ---------------------------------------------------------------------------

ASSET_ROOT = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
              "/Assets/Isaac/4.5/Isaac")
WAREHOUSE_MATERIALS = ASSET_ROOT + "/Environments/Simple_Warehouse/Materials"
# Where the vendored prop MESHES live (as opposed to the materials above).  A catalogue
# `asset` on a floor item is a reference to one of these; scene_builder.prop_asset_for()
# resolves it to the file in assets/isaac and refuses anything else -- an .mdl is a material
# and a wrapper layer renders as nothing, both of which have been hit here already.
PROP_ROOT = ASSET_ROOT + "/Props"

# What tools/measure_assets.py measured each vendored mesh to be, in this catalogue's own
# frame (x, height, lateral).  This is DATA, not a note: a prop's real size is not derivable
# here (there is no USD reader on the development machine), so it is recorded from the run and
# a test compares every declared size against it.
#
# It lives beside the catalogue because that is what consumes it, and it is keyed by the
# asset file name a scene's ``asset`` URL ends with.  An entry here is also a statement that
# the mesh can be referenced at all: the two filenames deliberately ABSENT are
# sektion_cabinet_visuals.usd (USD: "Unresolved reference prim path ... <defaultPrim>", no
# default prim, so it references as nothing) and mac_n_cheese_centered.usd (its sublayer
# mac_n_cheese.usd is not vendored).  Both were measured on the workstation and both came back
# with no extent.
PROP_MEASUREMENTS = {
    "small_KLT.usd": (0.1978, 0.1464, 0.2966),
    "small_KLT_visual.usd": (0.1978, 0.1464, 0.2966),
    "pallet.usd": (1.2132, 0.1425, 0.8023),
    "sektion_cabinet_instanceable.usd": (0.6678, 0.7861, 0.7638),
    "beaker_500ml.usd": (0.1621, 0.1492, 0.1789),
    "cone.usd": (1.0, 1.0, 1.0),
    "cube.usd": (1.0, 1.0, 1.0),
    "cylinder.usd": (1.0, 1.0, 1.0),
    "disk.usd": (1.0, 0.0, 1.0),
    "plane.usd": (1.0, 0.0, 1.0),
    "sphere.usd": (1.0, 1.0, 1.0),
    "torus.usd": (1.5, 0.5, 1.5),
}


# Muted, scene-neutral default so that no dressing item is ever rendered untextured black.
# The first preview came back with a solid black wall on one side and black blocks in the
# corridor: prims with no colour and no working material render black, and the material
# library's cache is not writable on the lab machine, so there was nothing to fall back on.
DRESSING_GREY = (0.45, 0.45, 0.47)
DRESSING_WOOD = (0.35, 0.24, 0.16)
DRESSING_METAL = (0.55, 0.57, 0.60)
DRESSING_GREEN = (0.22, 0.40, 0.18)
DRESSING_LIGHT = (0.78, 0.78, 0.80)


# ---------------------------------------------------------------------------
# Mounting: where a wall item's CENTRE has to be, given the surface it hangs on
# ---------------------------------------------------------------------------
# environment builds the obstacle wall with its centre at ``WALL_X`` (8.0) and thickness
# 0.02, so the wall occupies 7.99..8.01 and the surface the corridor sees is at 7.99.  A
# wall item written "at x = 8.0" like the wall itself therefore straddles that surface: a
# 0.03 m plate spans 7.985..8.015, which is 25 mm into the room and 5 mm inside the wall,
# and the 0.16-0.30 m items are 20 mm inside it -- half their thickness buried.
#
# Same fault as the marker had (see MARKER_WALL_X_M above), same answer: the author writes
# the SURFACE and the helper derives the centre.  ``_dressing_wall`` now takes the x it is
# given as the face the item hangs on and subtracts half the item's thickness, so its back
# sits against that face.  A caller no longer has to know that the wall's facing surface is
# not where its centre is.
OBSTACLE_WALL_FACE_X = 8.0 - 0.02 / 2.0          # WALL_X - WALL_THICKNESS/2
# The far wall is built with its centre at ROOM_LENGTH_X + thickness/2, so its inner face is
# exactly ROOM_LENGTH_X (16.0) -- which is what an author writing "on the far wall" means.
FAR_WALL_FACE_X = 16.0
# A millimetre of clearance so two surfaces are never exactly coplanar, which is what a
# renderer z-fights over.  This is not a placement error and not a size change: 1 mm on a
# 0.6 m plate, 0.13 px at the start pose.
MOUNT_CLEARANCE_M = 0.001


def _dressing_floor(name: str, at: Tuple[float, float, float],
                    size: Tuple[float, float, float], colour: Optional[str] = None,
                    asset: Optional[str] = None) -> Dict[str, Any]:
    """A floor item.  ``at`` is (x, height of the BASE, lateral).

    The caller gives the base height, not the centre height, so a bench declared at 0 has
    its feet on the floor whatever it is tall.  ``place_dressing`` translates by the box
    centre, so the centre is derived here; before this, seven items stood 5-25 mm above the
    floor and one -- the supermarket checkout -- had its base 5 mm BELOW it, all because the
    call sites were quietly centre coordinates.
    """
    height = float(size[1])
    return {"name": name, "mount": "floor",
            "at": (float(at[0]), float(at[1]) + height / 2.0, float(at[2])),
            "size": size, "colour": colour or DRESSING_GREY, "asset": asset,
            "collides": False}


def _dressing_wall(name: str, mount: str, at: Tuple[float, float, float],
                   size: Tuple[float, float, float], colour: Optional[str] = None,
                   asset: Optional[str] = None) -> Dict[str, Any]:
    """A wall-mounted item, declared like a picture: (across, tall, thick).

    ``at``'s x is the SURFACE the item is stuck to, and it is used AS the box centre -- so
    writing ``OBSTACLE_WALL_FACE_X`` or ``FAR_WALL_FACE_X`` means "against that face at every
    thickness" rather than "against it provided I also accounted for how thick my item is".
    The thickness is not subtracted here on purpose: an earlier revision did subtract it, and
    because every call site writes the same two constants that made two items of different
    thicknesses hang at different distances from the wall, which is the opposite of what a
    constant is for.

    The thickness IS moved to the x axis, because a wall faces along x and the first
    version's call sites wrote (0.5, 0.4, 0.03) -- which put half a metre of plate straight
    out into the corridor, the same mistake as the duct.  Callers describe the picture; this
    function decides which way it faces.

    ``MOUNT_CLEARANCE_M`` is taken off the front face only, so the item never shares a plane
    with the wall it hangs on (which is what a renderer z-fights over) while its back stays
    inside the wall where it is hidden.
    """
    across, tall, thick = (float(v) for v in size)
    centre_x = float(at[0]) - MOUNT_CLEARANCE_M
    return {"name": name, "mount": mount,
            "at": (centre_x, float(at[1]), float(at[2])),
            "size": (thick, tall, across), "colour": colour or DRESSING_GREY,
            "asset": asset, "collides": False}


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
        # This scene's inline layout below IS the live one -- there is no replacement for
        # 1.2 in _install_dressing(), so these eight items are what gets placed.
        "dressing": (
            # Wall items name the SURFACE they hang on, so these read as x = 7.99 (the
            # obstacle wall's corridor-facing face) and 16.0 (the far wall's inner face).
            # The helper turns that into a box centre.  Writing the wall's own centre here
            # is what buried half of every thick item inside it.
            _dressing_wall("sign", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 1.5, 1.35),
                           (0.5, 0.4, 0.03),
                           asset=WAREHOUSE_MATERIALS + "/MI_SignB.mdl"),
            _dressing_wall("toolboard", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 1.2, -1.35),
                           (0.8, 0.6, 0.03)),
            _dressing_wall("duct", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 2.6, 1.10),
                           (0.30, 0.30, 0.30), DRESSING_METAL),
            _dressing_wall("bay_sign", "far_wall", (FAR_WALL_FACE_X, 2.15, 1.60),
                           (0.9, 0.35, 0.03)),
            # Floor items are deliberately small and far off the centre line.  The first
            # version put a 1.8 m tall, 1.1 m deep forklift five metres from the camera, and
            # the rendered frame came back with a dead black mass filling one side of it: a
            # big unlit box that close is not dressing, it is an obstruction.  Nothing here is
            # more than a metre tall, more than 0.9 m long along the view axis, or nearer the
            # centre line than |z| 2.0.
            # The height argument is the BASE, so it is 0 for everything standing on the
            # floor.  It used to be the box centre -- these read 0.20/0.15/0.35/0.35 -- and
            # because the old helper did not move it, four of these ended up with their
            # bottoms below the floor plane once the helper started adding half the height.
            #
            # The two sizes below are the MEASURED size of the prop they name, in this frame,
            # taken from tools/measure_assets.py.  They were descriptive guesses before, and
            # both were wrong in opposite directions: the pallet was declared SMALLER than a
            # real pallet (0.9 x 0.9 m against 1.2132 x 0.8023 m) and the small load carrier
            # LARGER than a real one (0.6 x 0.5 m against 0.1978 x 0.2966 m).  Because
            # place_dressing references the prop at its own scale, a box that disagrees with
            # its prop is either a mesh hanging outside its declared volume or a box drawn
            # around nothing -- and the occlusion and walking-band checks are written against
            # the box, so they were being satisfied by a number nobody had measured.
            _dressing_floor("pallets", (2.6, 0.0, 2.05), (1.2132, 0.1425, 0.8023),
                            asset=PROP_ROOT + "/Pallet/pallet.usd"),
            _dressing_floor("klt_bins", (4.0, 0.0, -2.00), (0.1978, 0.1464, 0.2966),
                            asset=PROP_ROOT + "/KLT_Bin/small_KLT_visual.usd"),
            _dressing_floor("forklift", (6.2, 0.0, 2.15), (0.9, 0.70, 0.7)),
            _dressing_floor("traffic_cone", (7.0, 0.0, -1.95), (0.4, 0.70, 0.4),
                            asset=WAREHOUSE_MATERIALS + "/M_TrafficCone.mdl"),
        ),
        "forbidden_colours": (),
        "notes": "The only scene whose materials were read off the bucket listing in full.",
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
        # Not here: `_install_dressing` below sets this scene's layout, because it took two
        # rendered previews to get the shelves off the opening's sight line.  An earlier
        # revision also carried a full eight-item tuple at this point, which that function
        # then overwrote at import -- 24 lines of dead configuration reading as if it were
        # the live layout.  See the note above SCENES.
        "dressing": (),
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
        # Set by `_install_dressing` below, as for 1.3.  See the note above SCENES.
        "dressing": (),
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
        # Set by `_install_dressing` below, as for 1.3.  See the note above SCENES.
        "dressing": (),
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


# ---------------------------------------------------------------------------
# Two coordinate conventions, and the one line that joins them
# ---------------------------------------------------------------------------
# Everything in this catalogue is written in USER coordinates: (x, y, z) means
# (along the corridor, height above the floor, sideways from the centre line).  A marker at
# (16.0, 1.40, 0.0) is therefore 1.40 m up the far wall, on the centre line, which is what the
# prompt describes.
#
# The stage is in WORLD coordinates, which are Z-up: (x, y, z) means
# (along the corridor, sideways, height).  environment.py writes its own room in user
# coordinates and passes everything through _add_box, which converts; the room bounds that come
# back out of discover_surfaces prove it.  room_ceiling is authored at y = 3.01 and reported at
# world z = 3.01, and room_side_left is authored at z = -2.51 and reported at world y = -2.51.
#
# scene_builder builds prims directly with UsdGeom, bypassing _add_box, so it has to convert for
# itself.  It did not, and a marker meant for the middle of the far wall at 1.40 m was placed at
# 1.40 m SIDEWAYS and height zero: a plate lying at the foot of the wall, half of it under the
# floor.  Every dressing item was misplaced the same way, which is what put boxes in the agent's
# view, and the surface classifier was reading world boxes with user-coordinate rules, which is
# why the floor material landed on a side wall and the ceiling was never found at all.
#
# The conversion is the same in both directions -- it swaps the last two components -- so these
# two functions are the same operation, named for what the caller means.


def to_world(point: Sequence[float]) -> Tuple[float, float, float]:
    """(x, height, lateral) -> (x, lateral, height), for placing a prim on the stage."""
    x, y, z = (float(v) for v in point)
    return (x, z, y)


def to_user(point: Sequence[float]) -> Tuple[float, float, float]:
    """(x, lateral, height) -> (x, height, lateral), for reading a prim back off the stage."""
    x, y, z = (float(v) for v in point)
    return (x, z, y)


def to_world_size(size: Sequence[float]) -> Tuple[float, float, float]:
    """A box's dimensions in the same two frames: lengths swap with their axes."""
    sx, sy, sz = (float(v) for v in size)
    return (sx, sz, sy)


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

# The word for each colour key, and it has to describe the RGB above, because
# ``describe_marker`` puts this word into the prompt: the task sentence becomes "reach
# the <this> on the far wall", so a wrong word here is a false statement handed to the
# model about the very object it is being asked to walk to.
#
# Two entries were wrong and are the reason this comment exists.  ``w`` is
# (0.55, 0.05, 0.25) -- a dark wine red -- and was called "white"; ``k`` is
# (1.00, 0.40, 0.70) -- pink -- and was called "black".  The false prompts that
# produced are "white set of two bars" (scene 1.4 slot 5), "white arrow" (1.5 slot 2)
# and "black hexagon" (1.5 slot 1).  Nothing caught it: the uniqueness tests compare
# RGB triples, not words, and no test read the rendered phrase.  A test now pins every
# colour word against its key's own value.
#
# ``l`` is "lime" rather than "lime green" so that the analysis in
# tools/parse_agent_logs.py counts the phrase the prompt actually used: it looks the
# marker's colour word up in this same table before searching the model's reasoning.
COLOUR_NAMES: Dict[str, str] = {
    "r": "red",
    "m": "magenta",
    "o": "orange",
    "l": "lime",
    "c": "cyan",
    "p": "purple",
    "g": "green",
    "w": "wine",
    "t": "teal",
    "k": "pink",
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
#     red/cyan/purple/magenta/teal, the park's are red/magenta/orange/purple/wine, and the
#     supermarket's are pink/wine/teal/purple/lime, so the dressing below is deliberately
#     grey, wood, steel, green, blue and yellow.

DRESSING_BLUE = (0.20, 0.35, 0.65)
DRESSING_YELLOW = (0.85, 0.78, 0.30)
DRESSING_CLAY = (0.62, 0.45, 0.30)


def _install_dressing() -> None:
    """The layouts for 1.3, 1.4 and 1.5: the three scenes whose dressing was re-cut.

    Each scene's list is assigned whole, so a scene has its layout either here or inline
    above and never in both.  ``test_bao_scenes.py`` pins that, because the revision that
    had both meant editing the inline list changed nothing at all.
    """
    # The library: shelves either side of the opening, a reading corner behind the agent.
    SCENES["stage1.3"]["dressing"] = (
        # 1.4 m across, centred 1.65 m off the axis, so the inner edge sits at 0.95 m: clear of
        # the 1.14 m opening's half width (0.57 m) in the start view, which is the projection
        # check that caught the first version's 1.6 m shelves at 1.30 m.
        _dressing_wall("bookshelf_left", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 1.15, 1.65),
                       (1.4, 1.9, 0.16), DRESSING_WOOD),
        _dressing_wall("bookshelf_right", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 1.15, -1.65),
                       (1.4, 1.9, 0.16), DRESSING_WOOD),
        _dressing_wall("clock", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 2.45, -1.70), (0.4, 0.4, 0.03),
                       DRESSING_LIGHT),
        _dressing_wall("reading_poster", "far_wall", (FAR_WALL_FACE_X, 2.05, 1.55), (0.7, 0.5, 0.03),
                       DRESSING_LIGHT),
        _dressing_floor("study_table", (3.0, 0.00, 2.05), (0.9, 0.75, 0.7), DRESSING_WOOD),
        _dressing_floor("chair_row", (5.0, 0.00, -2.05), (0.7, 0.50, 0.9), DRESSING_GREY),
        _dressing_floor("book_cart", (6.6, 0.00, 2.10), (0.8, 0.90, 0.5), DRESSING_METAL),
        _dressing_floor("reading_lamp", (1.8, 0.00, -2.10), (0.3, 0.95, 0.3), DRESSING_METAL),
        # The cabinet is FLOOR furniture, not a wall item, and that is a decision rather than a
        # convenience.  It measures 0.6678 x 0.7861 x 0.7638 m, so as a wall item its own
        # half-depth (0.33 m) alone exceeds test_no_dressing_protrudes_into_the_corridor's
        # 0.20 m cap -- and that cap is what the warehouse's duct taught when it hung a black
        # slab across the agent's view.  Rather than relax the rule or squash the mesh, the
        # cabinet stands on the floor, which is a thing cabinets do.
        #
        # |z| is 2.05, not 1.5: the rule is that a floor item whose BASE is above 0.6 m needs
        # |z| >= 1.8, and this is 0.7861 m tall with base 0.  Its lateral half-extent 0.3819
        # then leaves the inner edge at 1.67 m, clear of the 1.14 m opening's half width.
        _dressing_floor("cabinet", (7.0, 0.00, 2.05), (0.6678, 0.7861, 0.7638),
                        DRESSING_WOOD,
                        asset=PROP_ROOT + "/Sektion_Cabinet/sektion_cabinet_instanceable.usd"),
    )

    # The park: benches and planters behind the agent, hedges either side of the opening.
    # (This used to be a bare string literal between two statements -- not a docstring, so
    # Python compiled it and threw it away, and it read as though it documented the code.)
    SCENES["stage1.4"]["dressing"] = (
        _dressing_wall("hedge_left", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 2.30, 1.50), (1.2, 0.4, 0.16),
                       DRESSING_GREEN),
        _dressing_wall("hedge_right", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 2.30, -1.50), (1.2, 0.4, 0.16),
                       DRESSING_GREEN),
        _dressing_wall("park_sign", "far_wall", (FAR_WALL_FACE_X, 2.00, -1.60), (0.6, 0.4, 0.03),
                       DRESSING_CLAY),
        _dressing_floor("bench_left", (2.4, 0.00, 2.05), (0.9, 0.45, 0.5), DRESSING_WOOD),
        _dressing_floor("bench_right", (4.8, 0.00, -2.05), (0.9, 0.45, 0.5), DRESSING_WOOD),
        _dressing_floor("litter_bin", (6.4, 0.00, 2.10), (0.4, 0.70, 0.4), DRESSING_GREEN),
        _dressing_floor("planter_left", (1.6, 0.00, -2.10), (0.6, 0.40, 0.6), DRESSING_CLAY),
        _dressing_floor("planter_right", (7.2, 0.00, -2.00), (0.6, 0.40, 0.6), DRESSING_CLAY),
    )

    # The supermarket: shelving either side, a checkout and produce behind the agent.
    SCENES["stage1.5"]["dressing"] = (
        _dressing_wall("shelf_left", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 1.05, 1.60), (1.4, 1.9, 0.18),
                       DRESSING_METAL),
        _dressing_wall("shelf_right", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 1.05, -1.60), (1.4, 1.9, 0.18),
                       DRESSING_METAL),
        _dressing_wall("price_strip", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 2.00, 1.60), (1.3, 0.15, 0.03),
                       DRESSING_YELLOW),
        _dressing_wall("promo_banner", "far_wall", (FAR_WALL_FACE_X, 2.10, 1.55), (0.9, 0.5, 0.03),
                       DRESSING_YELLOW),
        _dressing_floor("trolley", (2.8, 0.00, 2.05), (0.8, 0.95, 0.5), DRESSING_METAL),
        _dressing_floor("produce_bins", (4.6, 0.00, -2.05), (0.9, 0.55, 0.6), DRESSING_YELLOW),
        _dressing_floor("checkout", (6.4, 0.00, 2.10), (0.9, 0.85, 0.7), DRESSING_GREY),
        _dressing_floor("stacked_boxes", (7.4, 0.00, -2.00), (0.7, 0.55, 0.5), DRESSING_BLUE),
    )


_install_dressing()


def episode_count() -> int:
    return (len(SCENE_ORDER) * LEVELS_PER_SCENE * MARKERS_PER_SCENE * ROSTER_SIZE)
