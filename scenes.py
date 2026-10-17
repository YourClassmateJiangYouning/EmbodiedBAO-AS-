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

* **A marker is the task.**  0.90 m bounding size, at x = 16.0, z = 0, centre height 1.40 m,
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

MARKER_SIZE_M = 0.90
MARKER_Z_M = 0.0
MARKER_Y_M = 1.40
MARKER_THICKNESS_M = 0.02

# 0.90 m rather than 0.60 m, at the request that the marker be legible in frames that measured
# dimmer than the room the size was chosen in.
# The history is in environment._create_goal_marker and it is the reason this is a considered
# change rather than a tweak: 0.30 m was tried and rejected for a different reason (a 0.80 m
# square became a red field at close range), then adopted at 0.60 m because the marker is the
# TASK and has to be identifiable from the start pose at the 512 px input -- at 0.30 m it was
# about 10 px there.  Measured from the actual frames, the plate lands at 13-25 px in a 1024 px
# frame, i.e. 6-12 px at the 512 px the model receives, and its closest pixel sits 47-69 away in
# RGB from the authored colour because the textured scenes render dimmer than the plain baseline
# the 0.60 m was tuned against (eye-frame means: 137.6 for stage1.1, but 81-112 for 1.2-1.5,
# against environment.py's own 130-160 acceptance range).
#
# 0.90 m is 1.5x, so the apparent edge grows 1.5x and the area 2.25x.  It still fits the level 0
# opening, which is 1.14 m wide, so the marker keeps the property the design relies on: it is an
# object that would pass through the aperture rather than one that could not.  All ten shapes
# are derived from SHAPE_HALF_M below, so they scale with it and no shape definition changes.

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
# Ten shapes, ten colours, built from primitives only.  The green is darkened from its first value
# at the owner's request; no colour has been added.  No textures: a textured
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

# THE GAP BETWEEN THE BARS, AND WHY IT HAS TO BE A CONSTANT.
#
# `bars3` is described to the model as "set of three bars" and `bars2` as "set of two bars".  Both were
# SOLID: bars3's three bars overlapped each other by 0.0743 m, so their union was one 0.479 x 0.900
# rectangle, and bars2's two bars were separated by 0.0022 m -- about 0.09 px from the start pose, i.e.
# invisible.  The five conditions that use them (stage1.1/5, stage1.2/4, stage1.4/2, stage1.4/5,
# stage1.5/3) therefore showed the model a solid block and told it "three bars".
#
# 0.05 m is about 2.2 px at the 1024-px frame and the ~15.5 m distance, so the gap survives the
# downscale; test_bao_scenes checks that the bars do not touch.
BAR_GAP_M = 0.05

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
    # g is derived from w and BAR_GAP_M so the three bars cannot touch: the centre bar occupies
    # [-w, w] and the outer bars start at g - w, so the gap is 2*(g - w).  It used to be
    # SHAPE_HALF_M * 0.30, which is smaller than 2*w and made them OVERLAP by 0.0743 m.
    "bars3": (lambda w=SHAPE_HALF_M / 4.3, g=(SHAPE_HALF_M / 4.3) * 2.0 + BAR_GAP_M: (
        _rect(-g - w, -SHAPE_HALF_M, -g + w, SHAPE_HALF_M),
        _rect(-w, -SHAPE_HALF_M, w, SHAPE_HALF_M),
        _rect(g - w, -SHAPE_HALF_M, g + w, SHAPE_HALF_M)))(),
    # g is h plus half the gap, so the two bars are separated by exactly BAR_GAP_M.  It used to be
    # SHAPE_HALF_M * 0.235, which left a 0.0022 m slit -- invisible, so this read as one solid bar.
    "bars2": (lambda h=SHAPE_HALF_M / 4.3, g=(SHAPE_HALF_M / 4.3) + BAR_GAP_M / 2.0: (
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
    "g": (0.10, 0.50, 0.14),   # green, darkened: the owner read the bright version as too light
    "w": (0.55, 0.05, 0.25),   # wine
    "t": (0.05, 0.80, 0.70),   # teal
    "k": (1.00, 0.40, 0.70),   # pink    # There is no eleventh colour.  `x` existed briefly as black, then as a light green
    # called mint, and is now REMOVED: stage1.5's fifth marker uses the palette's existing
    # dark green with a solid shape instead, so the palette is the original ten again and
    # nothing needs an extra key.
}

# The scene's existing far-wall colour: a green marker on it would be invisible, which is
# why the supervisor's green triangle is assigned to the warehouse instead.
BASELINE_FAR_WALL_COLOUR = (0.13, 0.42, 0.20)

# ---------------------------------------------------------------------------
# Each scene's four surfaces, flat-painted.
# ---------------------------------------------------------------------------
# `materials` names the MDL each surface would use, but the previews run with use_mdl=False on
# purpose (this machine's material library cache is not writable, so MDL prims are created without
# their shaders and every surface renders as noise).  The colour a viewer actually sees therefore
# comes from here, and until now it did not: scene_builder held ONE fallback palette for all five
# scenes -- floor (0.55, 0.55, 0.55), side_wall (0.72, 0.73, 0.75), ceiling (0.92, 0.93, 0.95),
# far_wall (0.13, 0.42, 0.20) -- so every scene had a grey floor, a blue-grey corridor and the
# same dark green wall at the far end, and the scenes were indistinguishable from each other
# however different their catalogues were.
#
# The colours are chosen per scene and checked two ways, both by arithmetic:
#
#   * the five FLOORS are pairwise distinct.  The smallest distance between any two of them is
#     0.182 (warehouse vs park), so no two scenes share a floor.
#   * every surface stays clear of that scene's own five MARKER colours, which are the one thing
#     this experiment varies and must never be confusable with the room.  The closest any surface
#     comes to a marker of its own scene is 0.380 (the library's wood floor against its red
#     marker), and the closest a FAR wall comes is 0.457.  The test
#     `test_no_dressing_uses_a_marker_colour` polices the same invariant for dressing.
#
# stage1.1 is the baseline and its far wall keeps BASELINE_FAR_WALL_COLOUR exactly: it is the
# scene the earlier runs were collected on, and the green wall is part of what they measured.
#
# The far walls deliberately stay in the middle of the brightness range (0.20-0.80 on their
# strongest channel).  A very dark or very bright far wall would change how easily the 0.90 m
# marker on it can be seen, and the marker's visibility is the independent variable of the whole
# study.
# ---------------------------------------------------------------------------
# Each scene's surfaces as TEXTURES, because a flat colour is not what was asked for
# ---------------------------------------------------------------------------
# The owner asked for patterned walls rather than tints, and for the two wall surfaces to be
# distinguishable from each other.  These are PNGs already vendored in the repository, bound through
# a UsdPreviewSurface + UsdUVTexture by scene_builder.bind_texture_material -- no MDL, so the
# unwritable material cache that forced the flat-paint default does not apply.
#
#   * SIDE and FAR are always DIFFERENT images, per scene, so the corridor wall and the end wall
#     cannot be confused with each other.
#   * No image is used by two scenes, so the five rooms do not share a wall.
#   * Each was checked to exist in assets/ by basename before being written here; a scene whose
#     texture is missing falls back to SURFACE_COLOURS below, and the report says which happened.
SURFACE_TEXTURES: Dict[str, Dict[str, str]] = {
    # Each pair was chosen by MEASURING the image first (mean luminance, and the standard deviation
    # that says how much pattern it carries), because two candidates turned out to be unusable and
    # one could not be seen until it was on a wall:
    #
    #   T_Bookcase_bc.png                  mean 124  std 82   <- strong pattern, bright
    #   T_Board_bc.png                     mean 154  std 75
    #   T_Building_01_bc.png               mean 182  std 35
    #   T_Building_02_bc.png               mean 148  std 52
    #   T_Cupboard_bc.png                  mean  96  std 86
    #   T_BooksA_bc.png                    mean  97  std 77
    #   T_Billboard_bc.png                 mean 122  std 69
    #   T_Blinds_bc.png                    mean 237  std 18   <- bright but nearly flat
    #   Tex_Wall_Plaster_basecolor.png     mean 184  std  3   <- REJECTED for a wall: std 3 is a
    #   T_Carpet_bc.png                    mean  46  std 35      plain colour, not a pattern
    #
    # and each image is used ONCE across the five scenes, side walls and far walls always different.
    # 1.2: the corrugated atlas read as a plain colour however its repeat was set, because a
    # corrugated profile IS a fine repeating line pattern and averages out at corridor distance.  It
    # is replaced by a real tileable material with a structure large enough to survive -- the ceiling
    # tile grid, whose cell is a whole panel rather than a rib.
    # 1.2: the ceiling tile was still too regular and read as plain from the corridor.  This is a
    # building-facade image with real structure in it.
    "stage1.2": {"side_wall": "T_Building_01_bc.png",                      # panelled facade
                 "far_wall":  "T_Board_bc.png"},                          # boarded partition
    "stage1.3": {"side_wall": "T_Bookcase_bc.png",                        # shelf-lined reading room
                 "far_wall":  "T_Building_02_bc.png"},                    # stone end wall
    # 1.4: the side wall was also an OBJECT ATLAS (`T_Cupboard_bc`) and showed large black cabinet
    # shapes at random sizes.  Replaced by a tileable wood-grain material, which is what a park's
    # boarded fence actually is.
    # 1.4 IS A PARK, so it becomes an outdoor sky box: both walls are sky images from the vendored
    # Hospital/Office texture sets (there is no outdoor sky in the bucket), and the passage is covered
    # in planting.  No leaf or vine texture is vendored anywhere in either asset root -- the flowering
    # shrubs carry MDL-only materials with no PNGs -- so the passage uses the fetched grass ground
    # texture, which is a real green vegetation image and reads as a planted wall.
    "stage1.4": {"side_wall": "T_Sky_Blue.png",                            # open sky
                 "far_wall":  "T_Sky_Clouds_M.png",                       # sky with cloud
                 # THE LAWN, from the bucket's own terrain-with-grass material
                 # (nv_core/.../terrain_with_grass_adv/grassdistance_basecolor.png), fetched for
                 # this.  It is a tileable ground texture rather than an object atlas, which is what
                 # a floor needs.
                 "floor":     "default_texture.png"},
    # `T_BooksA_bc` was here for stage1.5 and the owner called the result "过分花哨", correctly: these
    # `T_*_bc` images are OBJECT ATLASES -- T_BooksA_bc is a photograph of whole shelves of products --
    # so tiling one covers a wall in a chaotic collage of tiny objects.  A wall needs a tileable
    # MATERIAL texture, and the Simple_Warehouse set has real ones (`Tex_*`): plaster, wood grain,
    # ceiling tile, checkered, hospital floor.  stage1.5 gets the ceiling tile, which is a calm grid
    # and reads as a shop's wall/ceiling finish.
    # 1.5 REVERTED, on the owner's report that the calm replacement was worse than what it replaced:
    # the ceiling tile and hospital floor are so regular that the supermarket read as bare.  These are
    # the atlas images that were there before, restored.
    "stage1.5": {"side_wall": "T_BooksA_bc.png",
                 "far_wall":  "T_Billboard_bc.png"},
}
# THE OBSTACLE WALL -- the panels that form the passage, including the two flanking the opening.
# These are the surfaces an eye-camera frame is mostly made of, and they are NOT the side walls: they
# are built by environment._create_wall with `material=("WallPanelMaterial_i", WALL_COLOR)` where
# WALL_COLOR is [0.13, 0.28, 0.72], a saturated blue.  Until this table existed, apply_scene never
# touched them, so every scene's corridor was blue however its side walls were textured -- which is
# exactly what the owner reported twice, and it was a real gap rather than a viewing problem.
#
# A separate texture from both the side walls and the far wall, so the passage, the corridor walls
# and the end wall are three different surfaces to look at.
SURFACE_TEXTURES_PANELS: Dict[str, str] = {
    "stage1.2": "T_ElectricRoom_bc.png",     # mean 139 std 31   plant-room cladding
    "stage1.3": "T_FileCabinet_bc.png",      # mean 123 std 42   card-catalogue panels
    "stage1.4": "default_texture.png",           # planting over the passage, lighter
    "stage1.5": "T_DoorD_bc.png",            # mean 214 std 32   bright shopfront
}
# and the flat colour each panel falls back to when its texture is missing.
SURFACE_COLOUR_PANELS: Dict[str, Tuple[float, float, float]] = {
    "stage1.2": (0.34, 0.33, 0.31),
    "stage1.3": (0.42, 0.34, 0.26),
    "stage1.4": (0.36, 0.37, 0.34),
    "stage1.5": (0.74, 0.74, 0.72),
}
# stage1.1 has no entry on purpose: it is the baseline and keeps no materials at all.

SURFACE_COLOURS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    # bare, tools, no skin at all
    "stage1.1": {"floor": (0.55, 0.55, 0.55), "side_wall": (0.72, 0.73, 0.75),
                 "ceiling": (0.92, 0.93, 0.95), "far_wall": BASELINE_FAR_WALL_COLOUR},
    # loading bay: grey power-float concrete, steel-grey racking and a beige partition
    "stage1.2": {"floor": (0.42, 0.41, 0.39), "side_wall": (0.72, 0.71, 0.66),
                 "ceiling": (0.90, 0.90, 0.91), "far_wall": (0.62, 0.58, 0.50)},
    # reading room: dark oak boards, warm plaster and a deep red-brown end wall
    "stage1.3": {"floor": (0.55, 0.36, 0.22), "side_wall": (0.80, 0.73, 0.60),
                 "ceiling": (0.94, 0.93, 0.90), "far_wall": (0.45, 0.30, 0.33)},
    # park: grass, weathered concrete and clipped hedge at the far end
    "stage1.4": {"floor": (0.33, 0.46, 0.24), "side_wall": (0.60, 0.63, 0.56),
                 "ceiling": (0.88, 0.90, 0.94), "far_wall": (0.20, 0.44, 0.24)},
    # supermarket: pale terrazzo, white walls and an orange promotional end wall
    "stage1.5": {"floor": (0.78, 0.77, 0.74), "side_wall": (0.90, 0.89, 0.87),
                 "ceiling": (0.95, 0.95, 0.96), "far_wall": (0.80, 0.45, 0.18)},
}

# ---------------------------------------------------------------------------
# The 25 markers.  Slot order is the order the five repeats run in.
# Within a scene the five colours differ; across the set no (shape, colour) repeats.
# ---------------------------------------------------------------------------

MARKERS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "stage1.1": (("square", "r"), ("triangle", "m"), ("disc", "c"),
                 ("cross", "o"), ("bars3", "l")),
    "stage1.2": (("diamond", "r"), ("hexagon", "c"), ("arrow", "p"),
                 ("bars2", "g"), ("triangle_down", "m")),
    "stage1.3": (("triangle_down", "r"), ("cross", "c"), ("square", "p"),
                 ("hexagon", "m"), ("arrow", "t")),
    "stage1.4": (("disc", "r"), ("bars3", "m"), ("triangle", "o"),
                 ("diamond", "p"), ("bars2", "w")),
    "stage1.5": (("hexagon", "k"), ("arrow", "w"), ("bars3", "t"),
                 ("disc", "p"), ("square", "g")),
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

# The local root for everything vendored from OUTSIDE Assets/Isaac -- the vegetation and the
# outdoor furniture, fetched by tools/fetch_bucket_set.py.  A catalogue ``asset`` may name a file
# under either root and scene_builder._asset_index() searches both, because these assets are not
# under Assets/Isaac at all and the park had nothing but boxes while only assets/isaac was read.
#
# Relative, like the URLs above: this module deliberately does not import os, and the value is only
# ever used as a prefix that identifies which root an asset belongs to.  scene_builder holds the
# absolute BUCKET_ASSETS it actually walks.
BUCKET_ASSETS = "assets/bucket"

# Assets that are wired into a scene but not yet measured on the machine that has a USD reader.
# This is NOT an exemption from the declared-size check: the check fails any referenced asset that
# is neither in PROP_MEASUREMENTS nor named here, so a new prop still cannot slip through, and it
# prints what is pending on every run so the list cannot be forgotten quietly.  The purpose is to
# let a scene be furnished and committed while the measurement wait is outstanding, instead of
# either blocking the work or writing invented numbers into a table whose whole value is that it
# holds measurements.
PROP_MEASUREMENT_PENDING = (
    # Fetched but never wired: RackLargeEmpty_A1 has no scene yet, so it is measured
    # only if a scene names it.  Kept in the list so the gap is visible.
    "RackLargeEmpty_A1.usd",
)

# Wall items whose declaration deliberately does NOT follow the measured axis order, because the
# asset's thinnest axis is not its local x and place_dressing never rotates a reference.  A wall
# item's slot 0 is its depth along the wall normal, so for these the thickness must be declared in
# slot 0 or the long axis would stand out into the corridor; that in turn means slot 0 cannot also
# equal the measured x.  The declared-size check therefore prints a warning for these instead of
# failing, and the plan is a rotation in place_dressing.
#
# Measuring which items this applies to, across all 63: only `reading_poster` is materially
# affected.  `fountain` (0.1621 x 0.1108 x 0.1621) is near-cubic and its slot 0 is already within
# 4 mm of its thinnest axis, so declaring the measured order would satisfy every rule without a
# rotation; `banana` is long-and-thin with its thinnest axis on slot 2 exactly like the poster, but
# its long axis is only 0.1972 m -- inside the 0.20 m protrusion cap -- so it is legal in either
# order.  That is why the printed warning lists one item and not three.  The two mugs this note used
# to name are gone from the catalogue (see the stage1.5 note in _install_dressing).
PROP_NEEDS_ROTATION = ("FramedPoster.usd",)


def needs_rotation(asset: Optional[str]) -> bool:
    """Does this asset have to be turned before it can hang flat on a wall?

    True for the assets in PROP_NEEDS_ROTATION, i.e. those whose THINNEST axis is not
    their local x AND whose long axis would exceed the 0.20 m protrusion cap if it were
    left facing the corridor.  ``place_dressing`` references a prop with translate and
    scale and never a rotation, so an asset's local x lands on the room's x; a wall item
    whose thin axis is local y would therefore stand its full length out into the
    corridor instead of lying flat against the wall.  That is the framed poster exactly:
    it measures 1.092 m along local x and 0.0524 m along local y.

    The membership list IS the rule; the thin-axis test is its motivation, not a second
    criterion.  The banana is the case that shows why: its thinnest measured axis is
    slot 2, so the motivation alone would list it, but its long axis (0.1972 m) is under
    the cap, so it is legal whichever way it faces and is deliberately absent (see the
    note on its catalogue entry).  ``test_a_declared_size_is_the_size_of_the_prop_it_names``
    checks the consequence -- anything above a 10x depth-to-thinness ratio that is not
    in this list is an error.

    ``asset`` may be a bare file name or a full path; both resolve.  Matching on the suffix rather than
    on ``os.path.basename`` keeps this module free of an os import it does not otherwise need.
    """
    if not asset:
        return False
    text = str(asset)
    return any(text == name or text.endswith("/" + name)
               for name in PROP_NEEDS_ROTATION)

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
    # Measured 2026-10-10 by tools/measure_named_props.py, in the catalogue's
    # (x, height, lateral) frame.  A declared dressing size has to equal the prop's own size,
    # because place_dressing references the prop WITHOUT scaling it.
    #
    # THE FOUR SM_Mug_* ENTRIES WERE REMOVED ON 2026-10-16, deliberately, and they must not come back.
    # They were the only assets in this table whose BOUND is not linear in the prop's scale:
    # tools/probe_prop_bound.py measured bound(scale=2)/bound(scale=1) = 200 on every axis instead of
    # 2, because the bound disagrees with the mesh by about 10.8x -- bound 0.008521 m across against a
    # mesh that measures 0.0923 m.  That non-linearity is what drew a 9.2 cm mug 9.2 METRES across and
    # made an earlier per-axis fit unreleasable (bfd5be7).  Their scenes entries went first; these
    # measurements follow, so that nothing can reference them by accident.  Re-adding an entry here is
    # safe only once the bound agrees with the mesh again.  The measured numbers are kept below so the
    # measurement itself is not lost:
    #   SM_Mug_A2.usd (0.0923, 0.0910, 0.1271)     SM_Mug_B1.usd (0.0925, 0.0919, 0.1365)
    #   SM_Mug_C1.usd (0.0889, 0.1072, 0.1277)     SM_Mug_D1.usd (0.0888, 0.1101, 0.1276)
    "011_banana.usd": (0.1972, 0.0741, 0.0386),
    "005_tomato_soup_can.usd": (0.0677, 0.0677, 0.1019),
    "024_bowl.usd": (0.1612, 0.1615, 0.0550),
    "004_sugar_box.usd": (0.0927, 0.0451, 0.1763),
    # Measured 2026-10-10 by tools/measure_named_props.py.  NOT all in metres: the entries whose
    # numbers are implausible as furniture are in the asset's own centimetres, which is what that
    # tool reports because the assets carry no metersPerUnit metadata.  scenes.ASSET_SCALE holds
    # the factor that turns each one into metres, and the test that compares a declared size with
    # its measurement multiplies by that factor -- so a declaration and its asset are checked in
    # the same units rather than silently differing by 100x.
    "SM_HeavyDutyPackingTable_C02_01.usd": (247.3647, 99.4051, 76.2),
    "SM_Crate_A07_Yellow_01.usd": (60.1459, 17.0007, 40.1773),
    "SM_Crate_A08_Blue_01.usd": (60.5133, 17.0294, 40.5133),
    "sm_whitecorrugatedbox_b12_brown_01.usd": (51.1862, 51.7182, 52.1379),
    "sm_whitecorrugatedbox_b20_brown_01.usd": (29.9008, 30.1949, 62.1582),
    "dolly.usd": (0.8504, 0.4407, 1.2594),
    "rubiks_cube.usd": (0.0721, 0.0721, 0.0721),
    # Measured 2026-10-10, and again NOT all in one unit.  Note the two poster and bench entries:
    # both report metersPerUnit 0.01, and sign_plaza_fountain_01 reports 1.0 while measuring 16.2
    # -- so the metadata cannot be trusted and the factor is decided per asset from the numbers.
    "S_TrafficCone.usd": (0.3357, 0.4628, 0.3339),
    "SM_Book_01.usd": (0.02, 0.205, 0.145),
    "SM_Clock_04.usd": (0.045, 0.15, 0.45),
    "SM_Chair.usd": (0.6001, 0.8772, 0.5993),
    "SM_Armchair.usd": (0.82, 0.79, 1.04),
    "FramedPoster.usd": (109.2, 74.2589, 5.2437),
    "Roxana_DiningBench.usd": (43.3811, 45.3252, 150.0522),
    "sign_plaza_fountain_01.usd": (16.214, 11.0828, 16.214),
    "table01.usd": (0.8109, 0.6403, 0.8109),
    "bench_wrought_iron_02.usd": (1.2955, 0.9895, 0.8583),
    "trashcan_square_01.usd": (0.5855, 1.0112, 0.7957),
    "safety_railing_01.usd": (2.2261, 1.166, 0.0728),
    "shopping_cart_corral_01.usd": (2.0057, 2.3761, 3.7107),
    "RackSmallEmpty_A1.usd": (108.0188, 301.0001, 199.8097),
    # Measured 2026-10-12 for the two replacement assets, by tools/measure_named_props.py.
    # `packing_table` replaces the warehouse's forklift, which has no asset in the library; a heavy
    # packing bench is what that loading bay is actually missing.  `RackSmall_A7` replaces the
    # supermarket's two shelves, which were box primitives.  RackSmall_A7 is a centimetre asset (the
    # tool's own DIVIDE BY 100 warning), hence the ASSET_SCALE entry below.
    "packing_table.usd": (2.4736, 1.0829, 0.7816),
    "Goldflame_Spirea.usd": (31.8724, 20.4208, 31.1931),
    "RackSmall_A7.usd": (108.0188, 301.0001, 199.8097),
    # Measured 2026-10-10 on the lab machine.  Four of these are centimetre assets and three are
    # metres, IN THE SAME BATCH: the shrubs and the conifer are centimetres while the bench, bin,
    # planter, bollard and block are metres.  Nothing about the folder says which, so each was
    # measured rather than assumed, and ASSET_SCALE carries the factor.
    "Barberry.usd": (100.6735, 119.6581, 99.7061),
    "bench_curved_01.usd": (4.0906, 0.4417, 2.1698),
    "trashcan_cylinder_01.usd": (0.722, 1.0411, 0.722),
    "planter_round_02.usd": (0.7062, 0.7726, 0.7062),
    "bollard_01.usd": (0.1393, 1.0028, 0.1393),
    "concrete_block_01.usd": (1.0442, 0.5, 1.0501),
    "Chinese_Juniper.usd": (107.7438, 254.4219, 106.015),
    "Douglas_Fir.usd": (302.2725, 602.6265, 277.0605),
    "Elm_Sapling.usd": (174.2069, 308.7109, 175.0135),
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
DRESSING_BLUE = (0.20, 0.35, 0.65)
DRESSING_YELLOW = (0.85, 0.78, 0.30)
DRESSING_CLAY = (0.62, 0.45, 0.30)

# A DEEPER PALETTE, because seven colours for 52 items left each scene dominated by one or two of
# them: 1.2 used five distinct colours across eleven items and six of those were the same grey, and
# the supermarket used four across twelve, so the dressing read as a single material rather than as
# a warehouse or a shop.  Each one below is checked against EVERY scene's five marker colours, which
# is the constraint that matters -- a dressing prop the colour of a marker would make the task
# ambiguous -- and the closest any of them comes to any marker is 0.373 (GRASS_PALE against the
# warehouse's green).  Two candidates were rejected by that check and are recorded so they are not
# proposed again: a brick red (0.55, 0.16, 0.16) sits 0.142 from the park's wine marker, and the
# safety yellow (0.82, 0.72, 0.18) 0.245 from the baseline's lime square.
DRESSING_STEEL = (0.52, 0.55, 0.58)          # bare galvanised racking
DRESSING_DARK_STEEL = (0.30, 0.32, 0.35)     # painted machinery, trolleys
DRESSING_CONCRETE = (0.62, 0.62, 0.60)       # kerbs, bollards, blockwork
DRESSING_CARD = (0.62, 0.46, 0.30)           # shipping cartons
DRESSING_CARD_PALE = (0.72, 0.58, 0.42)      # paler cartons, packaging
DRESSING_WOOD_PALE = (0.58, 0.44, 0.28)      # light timber: tables, shelves, seating
DRESSING_PLASTIC_BLUE = (0.18, 0.34, 0.58)   # stackable crates and bins
DRESSING_HEDGE = (0.32, 0.54, 0.26)          # clipped hedge (lightened)
DRESSING_FLOWER = (0.68, 0.50, 0.62)         # flowering shrub, pale mauve
DRESSING_GRASS = (0.42, 0.55, 0.28)          # planting and grass
DRESSING_PAPER = (0.86, 0.84, 0.78)          # posters, notice boards, price cards
DRESSING_PLASTER = (0.80, 0.76, 0.70)        # painted plaster fittings
DRESSING_TEAL = (0.15, 0.40, 0.40)           # upholstery
DRESSING_OLIVE = (0.38, 0.40, 0.22)          # outdoor fabric, foliage in shade


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

# Some vendored props are authored in centimetres with no metersPerUnit metadata, so a reference
# to one draws 100x too large: the packing table measures 247.36 in its own units, which is 2.47 m
# of real furniture.  The sizes recorded in PROP_MEASUREMENTS are therefore NOT all in metres, and
# a scene that references such an asset has to say so per item.  The distinction is carried by
# asset_scale on the item and by this constant, rather than by silently dividing somewhere in
# place_dressing, because a factor applied in one hidden place is a factor nobody can audit.
CM_TO_M = 0.01

# The factor that turns each asset's own measurement into metres, keyed by asset basename.
#
# This is NOT what scales the geometry any more.  place_dressing fits every referenced prop to its
# declared box per axis, so the drawn size is the declared size whichever unit the asset is in, and
# ASSET_SCALE has exactly one job left: letting the declared-size check compare a declaration with
# its measurement.  That job is worth keeping, and deleting this as "redundant" would remove the
# only thing standing between a fit and a lie -- if a declared size were wrong, fitting would
# quietly squash the prop to the wrong number and look perfect, whereas this makes the mismatch an
# error with both numbers printed.
#
# It is also a property of the ASSET rather than of the item that places it, which is why it is not
# a parameter on _dressing_floor: the same packing table is centimetre-authored wherever it stands.
ASSET_SCALE = {
    "SM_HeavyDutyPackingTable_C02_01.usd": CM_TO_M,
    "SM_Crate_A07_Yellow_01.usd": CM_TO_M,
    "SM_Crate_A08_Blue_01.usd": CM_TO_M,
    "sm_whitecorrugatedbox_b12_brown_01.usd": CM_TO_M,
    "sm_whitecorrugatedbox_b20_brown_01.usd": CM_TO_M,
    # The vegetation is centimetres while the outdoor furniture beside it is metres, so this table
    # is not per-folder and cannot be guessed from a path.
    "Barberry.usd": CM_TO_M,
    "Chinese_Juniper.usd": CM_TO_M,
    "Douglas_Fir.usd": CM_TO_M,
    "Elm_Sapling.usd": CM_TO_M,
    # A centimetre asset whose metadata even says so -- FramedPoster and Roxana_DiningBench both
    # report metersPerUnit 0.01, while sign_plaza_fountain_01 reports 1.0 and measures 16.2.  The
    # metadata therefore cannot be trusted either way; these three are centimetres by their
    # numbers, which is what the measurement is for.
    "FramedPoster.usd": CM_TO_M,
    "Roxana_DiningBench.usd": CM_TO_M,
    "sign_plaza_fountain_01.usd": CM_TO_M,
    "RackSmallEmpty_A1.usd": CM_TO_M,
    "RackSmall_A7.usd": CM_TO_M,
    "Goldflame_Spirea.usd": CM_TO_M,
}


def _dressing_floor(name: str, at: Tuple[float, float, float],
                    size: Tuple[float, float, float], colour: Optional[str] = None,
                    asset: Optional[str] = None,
                    room_fit: float = 1.0) -> Dict[str, Any]:
    """A floor item.  ``at`` is (x, height of the BASE, lateral).

    The caller gives the base height, not the centre height, so a bench declared at 0 has
    its feet on the floor whatever it is tall.  ``place_dressing`` translates by the box
    centre, so the centre is derived here; before this, seven items stood 5-25 mm above the
    floor and one -- the supermarket checkout -- had its base 5 mm BELOW it, all because the
    call sites were quietly centre coordinates.

    ``size`` is always the real size in METRES, whatever the asset's own unit, and place_dressing
    fits the prop to it.  ``room_fit`` is for the case where the asset is too big for the room
    itself: a measured Douglas fir is 6.03 m tall and this room has a 3 m ceiling, so it is placed
    at a fraction of its measured size.  Recording that fraction separates a deliberate fit from a
    declared size that is simply wrong, which the declared-size check would otherwise have to treat
    identically.
    """
    height = float(size[1])
    return {"name": name, "mount": "floor",
            "at": (float(at[0]), float(at[1]) + height / 2.0, float(at[2])),
            "size": size, "colour": colour or DRESSING_GREY, "asset": asset,
            "room_fit": float(room_fit), "collides": False}


def _dressing_wall(name: str, mount: str, at: Tuple[float, float, float],
                   size: Tuple[float, float, float], colour: Optional[str] = None,
                   asset: Optional[str] = None,
                   room_fit: float = 1.0) -> Dict[str, Any]:
    """A wall-mounted item, standing against a wall.

    ``at``'s x is the SURFACE the item is stuck to, and it is used AS the box centre -- so
    writing ``OBSTACLE_WALL_FACE_X`` or ``FAR_WALL_FACE_X`` means "against that face at every
    thickness" rather than "against it provided I also accounted for how thick my item is".
    The thickness is not subtracted here on purpose: an earlier revision did subtract it, and
    because every call site writes the same two constants that made two items of different
    thicknesses hang at different distances from the wall, which is the opposite of what a
    constant is for.

    ``MOUNT_CLEARANCE_M`` is taken off the front face only, so the item never shares a plane
    with the wall it hangs on (which is what a renderer z-fights over) while its back stays
    inside the wall where it is hidden.

    The size is given in the CATALOGUE's frame -- (x, height, lateral) -- the same frame as
    PROP_MEASUREMENTS and every floor item, and it is stored unchanged.  It used to be
    (across, tall, thick) reversed on the way in, and a reversal cannot be reconciled with a
    measurement table written in one frame for every asset: for a wide item the two orders give
    different numbers and one of them has to win.  One frame everywhere is worth more than a helper
    that reads a picture.

    There was a ``frame="wall"`` parameter to declare a size in (across, tall, thick) instead.  It is
    gone: no call site used it, and a parameter that switches the meaning of an argument is the
    mechanism by which an invisible reversal comes back.
    """
    x_thick, tall, across = (float(v) for v in size)
    # The box CENTRE is derived from the surface, and the comment at OBSTACLE_WALL_FACE_X has claimed
    # this since it was written: "the helper derives the centre ... subtracts half the item's
    # thickness, so its back sits against that face".  It did not -- it subtracted only the 1 mm
    # clearance, so the centre sat ON the surface and half of every wall item was inside the wall.
    # On the 0.02 m obstacle wall that means the shelf and the mugs and the banana all poked through
    # into the destination room, and the physical protrusion into the corridor was size[0]/2 rather
    # than size[0], which is also why the two protrusion checks disagreed by a factor of two.
    centre_x = float(at[0]) - MOUNT_CLEARANCE_M - x_thick / 2.0
    # Slot 0 is the wall-normal depth, so a reader that asks "how far does this protrude" reads
    # size[0] and gets the thickness.  The other two are the height and width within the wall plane.
    #
    # The comment that used to sit here said place_dressing "takes this tuple as already being in the
    # axes the stage uses and does not convert it".  That is false: place_dressing converts every
    # item's size with ``to_world_size``, floor and wall alike, and rule 1 of tools/audit_dressing.py
    # asserts that it does and that there is no mount condition around it.  The claim was left over
    # from a revision in which this helper swapped the outer two components on the way in; it no
    # longer does, and a comment arguing for behaviour the code does not have is how a frame error
    # gets reintroduced.
    return {"name": name, "mount": mount,
            "at": (centre_x, float(at[1]), float(at[2])),
            "size": (x_thick, tall, across), "colour": colour or DRESSING_GREY,
            "asset": asset, "room_fit": float(room_fit), "collides": False}


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
            # Not the warehouse's own MI_Floor_01, and the reason is measured.  That material's
            # albedo is ColorAlbedo 0.145 grey lerped with Textures/T_Floor_01_D.png, whose mean
            # luminance is 63 of 255; the scene rendered its floor band at 29.8 with 31.9 % of it
            # near black, against 106.4 for the baseline.  The warehouse set has no brighter floor
            # to swap to -- MI_Floor_02b samples the SAME T_Floor_01_D.png and its ColorAlbedo
            # (0.006, 0.018, 0.030) is darker still -- so this uses the wood floor already in use
            # by stage1.3, whose basecolour texture measures 167.2.  Wood is also apt for a
            # factory interior, which is why this rather than the white plastic hospital floor.
            "floor": ASSET_ROOT + "/Environments/Hospital/Materials/M_Wood_Floor.mdl",
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
                           (0.03, 0.4, 0.5),
                           asset=WAREHOUSE_MATERIALS + "/MI_SignB.mdl"),
            # A real rack rather than a painted plate: what a toolboard is.  Its size is the
            # measured RackSmallEmpty_A1 (2.16 MB, the mesh layer).
            #
            # This comment used to add that "RackSmall_A7 is 2.2 kB and measures nothing, because
            # the geometry lives in the layer it references" -- and that contradicted the table this
            # very declaration is checked against, where RackSmall_A7.usd carries the SAME measurement
            # as RackSmallEmpty_A1 (108.0188, 301.0001, 199.8097) and is what stage1.5's two shelves
            # are declared from.  Both statements cannot be true.  What is actually the case: the 2.2 kB
            # file is a thin wrapper that references this layer, so measuring it through the reference
            # returns the mesh's size rather than nothing -- which is exactly why tools/measure_assets.py
            # recorded it, and why the declaration passes the per-axis check.
            _dressing_floor("rack", (7.0, 0.00, -1.8376), (0.716165, 1.995631, 1.324738), DRESSING_METAL,
                            BUCKET_ASSETS + "/Assets/ArchVis/Industrial/Shelves/"
                            "RackSmallEmpty_A1.usd", room_fit=0.663),
            _dressing_floor("rack_tall", (7.0, 0.00, 1.8376), (0.716165, 1.995631, 1.324738), DRESSING_STEEL,
                            BUCKET_ASSETS + "/Assets/ArchVis/Industrial/Shelves/"
                            "RackSmallEmpty_A1.usd", room_fit=0.663),
            _dressing_wall("bay_sign", "far_wall", (FAR_WALL_FACE_X, 2.15, 1.60),
                           (0.03, 0.35, 0.9)),
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
            # WAS a forklift with no asset, so it rendered as a plain box.  The library has no
            # forklift and no reachable set was found to contain one; a heavy packing bench is the
            # honest thing for a loading bay and `packing_table.usd` is vendored.  Declared at the
            # bench's measured proportions times room_fit 0.36 -> (0.8905, 0.3898, 0.2814), so the
            # uniform fit fills the box instead of leaving it at a third of its size.
            _dressing_floor("packing_bench", (6.2, 0.0, 2.15), (0.8905, 0.3898, 0.2814), DRESSING_DARK_STEEL,
                            PROP_ROOT + "/PackingTable/packing_table.usd", room_fit=0.36),
            _dressing_floor("traffic_cone", (7.55, 0.0, -1.95), (0.3357, 0.4628, 0.3339),
                            DRESSING_CLAY, BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/"
                            "Environments/Simple_Warehouse/Props/S_TrafficCone.usd"),
            # Real warehouse freight: a second pallet, a plastic crate and a corrugated box.
            # These replace nothing -- they are added, because this scene is a warehouse and three
            # boxes standing against the far wall is what a warehouse looks like.  Sizes are the
            # measured props; the crate and the box are centimetre assets, hence CM_TO_M.
            _dressing_floor("pallet", (12.0, 0.0, 2.04), (1.2132, 0.1425, 0.8023),
                            asset=PROP_ROOT + "/Pallet/pallet.usd"),
            _dressing_floor("crate", (10.2, 0.0, 2.05), (0.6015, 0.1700, 0.4018), DRESSING_WOOD_PALE,
                            PROP_ROOT + "/PackingTable/props/SM_Crate_A07_Yellow_01/"
                            "SM_Crate_A07_Yellow_01.usd"),
            _dressing_floor("carton", (13.4, 0.0, -2.10), (0.2990, 0.3019, 0.6216), DRESSING_CARD,
                            PROP_ROOT + "/PackingTable/props/sm_whitecorrugatedbox_b/"
                            "sm_whitecorrugatedbox_b20_brown_01.usd"),
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
        # Verified 2026-10-10 against the bucket listing, not merely named from memory: every one
        # of this repo's material and prop URLs answered HTTP 200 with a Content-Length, and the
        # remote ETag matched the local file's MD5 byte for byte.  ``material_verified`` records
        # exactly that, so it is now true for all five scenes.
        "material_verified": True,
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
            # The park had NO floor entry, so /World/Ground was never painted and kept whatever
            # material it was born with -- which is why stage1.1 (also unpainted) and stage1.4
            # showed the SAME floor while their catalogues were completely different, and why the
            # park's grass colour in SURFACE_COLOURS had nothing to land on.  "The default ground is
            # the park's outdoor cue" was the old note; a park with a grey tiled floor is not an
            # outdoor cue, it is a missing material.  M_Wood_Floor is the wrong subject for it, so
            # the park keeps a floor ENTRY naming the ground material and paints
            # SURFACE_COLOURS["stage1.4"]["floor"] on /World/Ground.
            "floor": ASSET_ROOT + "/Environments/Hospital/Materials/M_Wood_Floor.mdl",
            "side_wall": ASSET_ROOT + "/Environments/Hospital/Materials/M_Wall_Plaster.mdl",
            "ceiling": WAREHOUSE_MATERIALS + "/MI_CeilingA_06b.mdl",
            "far_wall": ASSET_ROOT + "/Environments/Office/Materials/MI_WallOffice_01.mdl",
        },
        # Verified 2026-10-10; see the note on stage1.3's flag.  The park has no floor material
        # on purpose: it keeps the default ground, which is the "outdoor" cue.
        "material_verified": True,
        # Set by `_install_dressing` below, as for 1.3.  See the note above SCENES.
        "dressing": (),
        "forbidden_colours": (),
        "notes": "The Props library has no vegetation in 1.3; 1.4 uses the bucket vegetation. Its floor names a material now -- without one the ground was never painted and matched stage1.1's.",
    },
    "stage1.5": {
        "label": "supermarket",
        "materials": {
            "floor": ASSET_ROOT + "/Environments/Office/Materials/MI_FloorMarbleTiles_03.mdl",
            "side_wall": WAREHOUSE_MATERIALS + "/MI_WallA_01.mdl",
            "ceiling": WAREHOUSE_MATERIALS + "/MI_CeilingA_06b.mdl",
            "far_wall": ASSET_ROOT + "/Environments/Office/Materials/MI_WallOffice_01.mdl",
        },
        # Verified 2026-10-10; see the note on stage1.3's flag.  Note that this floor material
        # samples Textures/T_Bookcase_bc.png -- a material named for marble tiles whose albedo is
        # a bookcase texture.  That is what NVIDIA's asset contains, byte for byte, so a wooden
        # look underfoot in the supermarket is the asset and not a wiring mistake here.
        "material_verified": True,
        # Set by `_install_dressing` below, as for 1.3.  See the note above SCENES.
        "dressing": (),
        "forbidden_colours": (),
        "notes": "No supermarket environment exists; the busiest scene of the five.",
    },
}

SCENE_ORDER = ("stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5")
MARKERS_PER_SCENE = 5

# The least mean luminance, of 255, that a scene floor's albedo texture may have.  Measured
# rather than chosen: stage1.2's original floor sampled T_Floor_01_D.png at 63.0 and rendered a
# floor band of 29.8 with 31.9 % of it near black, against 106.4 for the light baseline; the wood
# floor that replaced it measures 167.2 and the office marble 123.7.  80 sits between the failure
# and the working materials, so it fails what was actually wrong rather than everything strict.
# test_bao_scenes.test_no_scene_floor_is_covered_by_a_dark_material applies it.
FLOOR_ALBEDO_MIN = 80.0
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
# THE VERTICAL FIELD EQUALS THE HORIZONTAL ONE, because the camera fits HORIZONTALLY and the render is
# square.  This used to compute it from the 16:9 vertical aperture (11.787 mm) and get 47.6 degrees,
# while the renderer actually applied the 20.955 mm HORIZONTAL aperture on both axes. The repository's
# own measurement settles it: a 0.60 m plate at 15.48 m subtends about 25 px at 1024, which is 76.9 deg
# vertical, not 47.6. Every projected size in this module was therefore about 1.78x too large, and
# test_bao_scenes' legibility figures inherited that. The camera now sets filmFit=Horizontal
# explicitly (environment._apply_focal_length), so this is the contract rather than a coincidence.
VERTICAL_FOV_DEG = HORIZONTAL_FOV_DEG


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
#
# These three live at module level with the rest of the colours.  They used to be defined just
# above _install_dressing(), which happens to be BELOW the stage1.2 block that a later edit made
# use of one -- so the catalogue raised NameError on import.  A colour constant is not scoped to
# the function that first needed it.


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
        # A rack, standing on the FLOOR at its measured size (1.080188 x 3.010001 x 1.998097 m).
        # It was a wall item while the placeholder was a 0.16 m plate, and a real rack cannot be
        # one: its depth is 1.08 m against a 0.20 m protrusion cap, and that cap exists because a
        # black slab once hung across the agent's view.  Its height also settles the placement --
        # 3.01 m against a 3.0 m ceiling -- so squeezing it to a bookcase's 1.90 m through room_fit
        # would leave it 0.43 m deep, and a shelf that shallow is not a shelf.  Free-standing
        # racking at |z| >= 1.8 keeps the walking band clear and touches no wall, which is what
        # racking does.  room_fit 0.663 brings the 3.01 m height under the ceiling.
        _dressing_floor("bookshelf_left", (6.4, 0.00, 1.8376), (0.716165, 1.995631, 1.324738), DRESSING_WOOD_PALE, BUCKET_ASSETS + "/Assets/ArchVis/Industrial/Shelves/"
                        "RackSmallEmpty_A1.usd", room_fit=0.663),
        _dressing_floor("bookshelf_right", (6.4, 0.00, -1.8376), (0.716165, 1.995631, 1.324738), DRESSING_WOOD_PALE, BUCKET_ASSETS + "/Assets/ArchVis/Industrial/Shelves/"
                        "RackSmallEmpty_A1.usd", room_fit=0.663),
        # A real clock face, at its measured size: 0.045 m thick, 0.15 m tall, 0.45 m across the
        # wall.  Wall-mounted because a clock is thin: the 0.20 m protrusion cap exists because the
        # warehouse's duct once hung a black slab across the agent's view, and a 45 mm clock is not
        # that.
        _dressing_wall("clock", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 2.45, -1.70),
                       (0.045, 0.15, 0.45), DRESSING_DARK_STEEL,
                       BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Office/Props/"
                       "SM_Clock_04.usd"),
        # The framed poster measures 1.092 (width) x 0.7426 (height) x 0.0524 (thickness), and it is
        # the ONE wall item in the catalogue whose declared order is NOT the measured order: the
        # declaration is (0.0524, 0.7426, 1.092) while every other wall item -- clock, fountain, the
        # three mugs, the banana -- declares the measurement axis for axis.
        #
        # That swap is not a mistake, it is the only way this asset can hang flat with the code as it
        # stands.  A wall item's slot 0 is its depth along the wall normal, and place_dressing applies
        # a reference with translate and scale only -- NEVER a rotation -- so the asset's own local x
        # lands on the room's x.  This asset's thin axis is its local y (0.0524), not its local x
        # (1.092), so slot 0 has to carry the thickness or the plate would stand 1.09 m out into the
        # corridor.  Declaring it in measured order therefore trips the protrusion cap, and declaring
        # it this way is what makes the item work at all.
        #
        # It is listed in PROP_NEEDS_ROTATION, and the check that would otherwise reject it is
        # downgraded to a printed warning for that list only.  The fix is a rotation in
        # place_dressing; it is deferred because it cannot be verified without a rendered frame, and
        # because a quarter turn about the wall normal does not move the corridor geometry -- the
        # declared box, the protrusion and the occlusion checks are all identical either way.  What
        # the rotation WOULD change is what a viewer sees, which is why it is recorded here instead
        # of being quietly worked around.
        _dressing_wall("reading_poster", "far_wall", (FAR_WALL_FACE_X, 2.05, 1.55),
                       (0.0524, 0.7426, 1.092), DRESSING_PAPER,
                       BUCKET_ASSETS + "/Assets/ArchVis/Residential/Decor/Pictures/"
                       "FramedPoster.usd"),
        _dressing_floor("study_table", (3.0, 0.00, 1.90), (2.4736, 0.9941, 0.7620), DRESSING_WOOD_PALE,
                        PROP_ROOT + "/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/"
                        "SM_HeavyDutyPackingTable_C02_01.usd"),
        _dressing_floor("chair", (5.0, 0.00, -2.05), (0.6001, 0.8772, 0.5993), DRESSING_WOOD_PALE,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Office/Props/"
                        "SM_Chair.usd"),
        _dressing_floor("armchair", (5.9, 0.00, -1.95), (0.82, 0.79, 1.04), DRESSING_TEAL,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Office/Props/"
                        "SM_Armchair.usd"),
        _dressing_floor("book_cart", (6.6, 0.00, 1.60), (0.8504, 0.4407, 1.2594), DRESSING_DARK_STEEL,
                        PROP_ROOT + "/Dolly/dolly.usd"),
        # Two books on the reading table, at their measured size.  A book is 0.02 x 0.205 x 0.145 m,
        # so on a 1.7 m bookcase it would be sub-pixel -- and 0.205 m is taller than the bookcase's
        # own shelf depth, so scaling one to fit a shelf would make a cushion rather than a book.
        # On the table they are the size a book is and they read as books.
        _dressing_floor("book_a", (2.6, 0.9941, 1.88), (0.02, 0.205, 0.145), DRESSING_PAPER,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Office/Props/"
                        "SM_Book_01.usd"),
        _dressing_floor("book_b", (3.2, 0.9941, 1.95), (0.02, 0.205, 0.145), DRESSING_PAPER,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Office/Props/"
                        "SM_Book_01.usd"),
        # `reading_lamp` was here: a floor item with no asset, so it was drawn as its clearance box.
        # It is DELETED rather than built, on the owner's instruction that asset-less items go.  A
        # plate is only honest for the wall-mounted boards (their shape IS a rectangle); a floor
        # standard lamp is not a rectangle, so building one would be inventing furniture, and a cube
        # in the corner is worse than no lamp.  It projected to v=-3.36, i.e. outside the eye frame,
        # so nothing visible is lost.  Item count 13 -> 12.
        # A carton of stock waiting to be shelved, which is what a library cart holds.  Its scale
        # comes from ASSET_SCALE.
        _dressing_floor("stock_carton", (1.8, 0.00, 2.10), (0.2990, 0.3019, 0.6216), DRESSING_CARD,
                        PROP_ROOT + "/PackingTable/props/sm_whitecorrugatedbox_b/"
                        "sm_whitecorrugatedbox_b20_brown_01.usd"),
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
        _dressing_floor("cabinet", (7.0, 0.00, 2.05), (0.6678, 0.7861, 0.7638), DRESSING_WOOD_PALE,
                        asset=PROP_ROOT + "/Sektion_Cabinet/sektion_cabinet_instanceable.usd"),
    )

    # The park: benches and planters behind the agent, hedges either side of the opening.
    # (This used to be a bare string literal between two statements -- not a docstring, so
    # Python compiled it and threw it away, and it read as though it documented the code.)
    SCENES["stage1.4"]["dressing"] = (
        # Real vegetation and real park furniture.  These come from two roots that are NOT under
        # Assets/Isaac: Assets/Vegetation holds the trees and shrubs, and the benches, planters and
        # bin come from NVIDIA's Rivermark outdoor environment.  That is why the park was the last
        # scene still made of boxes -- assets/isaac contains no plant of any kind, and
        # tools/fetch_bucket_set.py exists to reach the rest of the bucket.
        #
        # A conifer and a broadleaf shrub as the hedges: a hedge IS a planted shrub, so these are
        # the honest asset for it rather than a green box.  Declared sizes are the shapes these
        # species actually take, and place_dressing fits each prop to its declared box, so the
        # clearance and occlusion checks describe what is drawn.
        # Shrubs stand on the FLOOR.  They were wall items at first, which was wrong twice over:
        # _dressing_wall takes its height as the CENTRE, so both hung 2 m up in mid-air, and a
        # shrub is a ball, so scaling one to a wall plate's 0.16 m thickness either squashes it or
        # -- keeping its proportions as room_fit does -- makes it 1.56 m thick and pushes 0.78 m
        # into the corridor.  A hedge is a planted shrub, and a shrub is a thing in the ground.
        #
        # Both are the Barberry.  Goldflame_Spirea measures 0.20 m tall, so reaching a shrub's
        # 0.90 m would need a room_fit of 4.4 -- an ENLARGEMENT, and room_fit is defined as a
        # fraction in (0, 1], which the declared-size check enforces.  Enlarging is the opposite of
        # what this field is for, so a 1.20 m shrub serves both sides rather than stretching a
        # 20 cm plant into one.
        _dressing_floor("shrub_left", (7.0, 0.00, 1.88), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_right", (7.0, 0.00, -1.88), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        # FOUR MORE, on the owner's note that the park should read as a park.  No grass TEXTURE is
        # vendored (only `Grass_Short_A.usd`, which measured worse than the assets already in use),
        # so the lawn is the painted grass-green floor and the planting is what carries the reading.
        # All four sit at |z| 2.05, outside the |z| >= 1.5 walking band, and use the same vendored
        # Barberry as the two above, at the size that shrub already measured.
        _dressing_floor("shrub_a", (3.2, 0.00, 2.05), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_b", (5.4, 0.00, -2.05), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_c", (10.4, 0.00, 2.05), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_d", (12.4, 0.00, -2.05), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        # EIGHT MORE, to the owner's "增加到 14 个": a planted park reads by density, and four of the
        # six were hidden behind the obstacle wall from this camera.  All at |z| 2.05-2.15, outside the
        # walking band, alternating sides so the two walls both look planted.
        _dressing_floor("shrub_e", (2.2, 0.00, -2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_f", (4.2, 0.00, 2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_g", (6.2, 0.00, -2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_h", (8.2, 0.00, 2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_i", (9.6, 0.00, -2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_j", (11.0, 0.00, 2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_k", (13.0, 0.00, -2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        _dressing_floor("shrub_l", (14.2, 0.00, 2.10), (0.7572, 0.9000, 0.7499), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Barberry.usd", room_fit=0.752143),
        # FLOWERS: `Goldflame_Spirea` is the one flowering shrub vendored and it measures 0.20 m tall,
        # so it is used at ITS OWN size as low flowering ground cover rather than stretched -- room_fit
        # is a fraction in (0,1] and enlarging is the opposite of what it is for.
        _dressing_floor("flowers_a", (3.0, 0.00, -2.15), (0.3187, 0.2042, 0.3119), DRESSING_FLOWER,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Goldflame_Spirea.usd"),
        _dressing_floor("flowers_b", (7.2, 0.00, 2.15), (0.3187, 0.2042, 0.3119), DRESSING_FLOWER,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Goldflame_Spirea.usd"),
        _dressing_floor("flowers_c", (12.0, 0.00, -2.15), (0.3187, 0.2042, 0.3119), DRESSING_FLOWER,
                        BUCKET_ASSETS + "/Assets/Vegetation/Shrub/Goldflame_Spirea.usd"),
        _dressing_wall("park_sign", "far_wall", (FAR_WALL_FACE_X, 2.00, -1.60), (0.03, 0.4, 0.6),
                       DRESSING_CLAY),
        # The park's remaining furniture, all at measured sizes.  Every placement below was computed
        # against the two rules rather than chosen: a floor item needs |z| >= 1.5 and its lateral
        # half-extent must not cross into the walking band, inside a room whose z is +/-2.5.
        _dressing_floor("picnic_table", (5.5, 0.00, 1.95), (1.7315, 0.7580, 0.5471), DRESSING_WOOD_PALE,
                        PROP_ROOT + "/PackingTable/packing_table.usd", room_fit=0.70),
        _dressing_floor("bench_iron", (9.5, 0.00, -1.7497), (0.4338, 0.4533, 1.5005), DRESSING_WOOD_PALE,
                        BUCKET_ASSETS + "/ArchVis/Residential/Furniture/Outdoors/Roxana/Roxana_DiningBench.usd"),
        _dressing_floor("bin_square", (13.0, 0.00, 1.95), (0.5855, 1.0112, 0.7957), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Outdoor/Rivermark/"
                        "dsready_content/nv_content/common_assets/props_general/"
                        "trashcan_square_01/trashcan_square_01.usd"),
        _dressing_floor("railing", (3.5, 0.00, 1.90), (2.2261, 1.166, 0.0728), DRESSING_METAL,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Outdoor/Rivermark/"
                        "dsready_content/nv_content/common_assets/props_general/"
                        "safety_railing_01/safety_railing_01.usd"),
        # The plaza fountain and its sign, on the far wall.  It is a centimetre asset: 16.214 x
        # 11.0828 x 16.214 in its own units, which is 0.1621 x 0.1108 x 0.1621 m of ornament.
        _dressing_wall("fountain", "far_wall", (FAR_WALL_FACE_X, 1.80, -0.60),
                       (0.1621, 0.1108, 0.1621), DRESSING_LIGHT,
                       BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Outdoor/Rivermark/"
                       "dsready_content/nv_content/common_assets/props_general/"
                       "sign_plaza_fountain_01/sign_plaza_fountain_01.usd"),
        # Sizes are the measured assets.  The bench really is 4.09 m long in the asset and 2.17 m
        # across; a park bench that size is a bandstand seat, so it is declared at a bench's
        # proportions and place_dressing fits the prop to that box.  The bin and the planters are
        # declared at their measured sizes because those are already right.
        _dressing_floor("bench_left", (2.4, 0.00, 1.7497), (0.4338, 0.4533, 1.5005), DRESSING_WOOD_PALE,
                        BUCKET_ASSETS + "/ArchVis/Residential/Furniture/Outdoors/Roxana/Roxana_DiningBench.usd"),
        _dressing_floor("bench_right", (2.4, 0.00, -1.7497), (0.4338, 0.4533, 1.5005), DRESSING_WOOD_PALE,
                        BUCKET_ASSETS + "/ArchVis/Residential/Furniture/Outdoors/Roxana/Roxana_DiningBench.usd"),
        _dressing_floor("litter_bin", (6.4, 0.00, 2.10), (0.722, 1.0411, 0.722), DRESSING_CONCRETE,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Outdoor/Rivermark/"
                        "dsready_content/nv_content/common_assets/props_general/"
                        "trashcan_cylinder_01/trashcan_cylinder_01.usd"),
        _dressing_floor("planter_left", (1.6, 0.00, -2.10), (0.7062, 0.7726, 0.7062), DRESSING_CLAY,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Outdoor/Rivermark/"
                        "dsready_content/nv_content/common_assets/props_general/"
                        "planter_round_02/planter_round_02.usd"),
        _dressing_floor("planter_right", (7.2, 0.00, -2.00), (0.7062, 0.7726, 0.7062), DRESSING_CLAY,
                        BUCKET_ASSETS + "/Assets/Isaac/4.5/Isaac/Environments/Outdoor/Rivermark/"
                        "dsready_content/nv_content/common_assets/props_general/"
                        "planter_round_02/planter_round_02.usd"),
        # Three trees along the far wall.  They are the largest thing in any of the five scenes and
        # they are what makes this read as outdoors rather than as a corridor with a bench.
        #
        # The measured conifer is 2.54 m and the measured fir is 6.03 m.  A 6 m tree does not fit a
        # room with a 3 m ceiling -- it would grow straight through it, and the fit would then pull
        # it down to whatever the declared box says, which is a 6 m fir drawn as a 3 m one.  So the
        # heights here are room-appropriate rather than measured, and that is a deliberate choice
        # rather than an accident: the declared box is what the clearance checks clear, and a tree
        # taller than the ceiling has no legal box at all.  Proportions are kept from the
        # measurement (the fir is ~1.09 : 1 across, the juniper ~1.02 : 1).
        # Trees at a corridor's scale, not at a forest's.  The measured conifer is 2.54 m and the
        # fir 6.03 m; declared here at 1.6 m and 1.55 m because the ceiling is 3 m and the first
        # attempt put a 2.91 m sapling in a 16 m corridor, which read as a tree growing through the
        # roof rather than as planting.  room_fit keeps each one's own proportions.
        _dressing_floor("tree_conifer", (12.4, 0.00, 1.9999), (1.0164, 2.4000, 1.0001), DRESSING_HEDGE,
                        BUCKET_ASSETS + "/Assets/Vegetation/Trees/Chinese_Juniper.usd",
                        room_fit=0.943315),
        _dressing_floor("tree_fir", (14.6, 0.00, -1.9713), (1.1537, 2.3000, 1.0574), DRESSING_GRASS,
                        BUCKET_ASSETS + "/Assets/Vegetation/Trees/Douglas_Fir.usd",
                        room_fit=0.381663),
        _dressing_floor("tree_sapling", (10.8, 0.00, -1.8764), (1.2415, 2.2000, 1.2472), DRESSING_GRASS,
                        BUCKET_ASSETS + "/Assets/Vegetation/Trees/Elm_Sapling.usd",
                        room_fit=0.712641),
    )

    # The supermarket: shelving either side, a checkout and produce behind the agent.
    SCENES["stage1.5"]["dressing"] = (
        # These two stay box primitives, and `RackSmall_A7` -- the obvious candidate, vendored and a
        # real shelving rack -- was measured and REJECTED for this position: its own size is
        # (1.08, 3.01, 2.00) m, so fitting it to a wall item's 0.20 m protrusion cap needs room_fit
        # 0.10, which leaves a 0.11 x 0.30 m rack.  Declaring it bigger fails the audit's own
        # "box cannot hold the prop" rule outright (0.68 m of depth against a 0.18 m box).  A wall
        # shelf 2 m deep is not a shelf, so the boxes stay until a rack with a shallow profile is
        # found -- and the report keeps saying they are boxes.
        # WERE wall-mounted box primitives, and could not stay that way: every shallow candidate is
        # deeper than the 0.20 m protrusion cap on its own thinnest axis (RackSmall_A7 measures 2.00 m
        # deep; the corrugated cartons 0.30 m), so the audit refused each substitution with
        # "protrudes ... past the 0.20 m cap" or "box cannot hold the prop".  The owner approved
        # moving them to the FLOOR, which has no protrusion cap, so they are now real racks.
        #
        # RackSmall_A7 measures (1.0802, 3.01, 1.9981) m -- a centimetre asset -- and is declared at
        # room_fit 0.63 for a 1.896 m tall rack, the same fraction its sibling RackSmallEmpty_A1 uses
          # in the warehouse.  Both stand at |z| 1.8706, FLUSH AGAINST THE SIDE WALL, because that is
          # the only centre a 1.2588 m lateral extent allows without pushing 0.0794 m of shelf through
          # it (measured 2026-10-16, SCENE_SIZE_AUDIT.md section 4).  This comment used to claim they
          # sit "outside the |z| >= 1.5 line the walking band requires"; that was read off the ANCHOR.
          # From the INNER edge it is false: 1.8706 - 1.2588/2 = 1.2412, i.e. 0.2588 m inside the band.
          # For an item this wide the two rules cannot both hold -- clearing the wall needs |z| <=
          # 1.8706, clearing the band needs |z| >= 2.1294 -- so the wall wins: a prop crossing a wall is
          # a geometry error, one inside the band is only a narrower corridor.  Nothing collides either
          # way; every dressing item is declared collides=False.
        _dressing_floor("shelf_left", (8.9, 0.0, 1.8706), (0.6805, 1.8963, 1.2588), DRESSING_STEEL,
                        BUCKET_ASSETS + "/Assets/ArchVis/Industrial/Shelves/RackSmall_A7.usd",
                        room_fit=0.63),
        _dressing_floor("shelf_right", (8.9, 0.0, -1.8706), (0.6805, 1.8963, 1.2588), DRESSING_STEEL,
                        BUCKET_ASSETS + "/Assets/ArchVis/Industrial/Shelves/RackSmall_A7.usd",
                        room_fit=0.63),
        _dressing_wall("price_strip", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 2.00, 1.60),
                       (0.03, 0.15, 1.3), DRESSING_PAPER),
        _dressing_wall("promo_banner", "far_wall", (FAR_WALL_FACE_X, 2.10, 1.55),
                       (0.03, 0.5, 0.9), DRESSING_PAPER),
        _dressing_floor("flat_cart", (2.8, 0.00, 1.60), (0.8504, 0.4407, 1.2594), DRESSING_DARK_STEEL,
                        PROP_ROOT + "/Dolly/dolly.usd"),
        # A plastic crate used as the produce bin, and a corrugated box for the stack.  Both are
        # centimetre assets, so ASSET_SCALE scales them; without that the crate draws 60 m long.
        _dressing_floor("produce_bins", (4.6, 0.00, -2.05), (0.6051, 0.1703, 0.4051), DRESSING_PLASTIC_BLUE,
                        PROP_ROOT + "/PackingTable/props/SM_Crate_A08_Blue_01/SM_Crate_A08_Blue_01.usd"),
        _dressing_floor("checkout", (6.4, 0.00, 1.90), (2.4736, 0.9941, 0.7620), DRESSING_STEEL,
                        PROP_ROOT + "/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/"
                        "SM_HeavyDutyPackingTable_C02_01.usd"),
        _dressing_floor("stacked_boxes", (7.4, 0.00, -2.00), (0.5119, 0.5172, 0.5214), DRESSING_CARD_PALE,
                        PROP_ROOT + "/PackingTable/props/sm_whitecorrugatedbox_b/"
                        "sm_whitecorrugatedbox_b12_brown_01.usd"),
        # THE THREE MUGS ARE DELETED, and the reason is a measured defect rather than taste.  They
        # were the only props the fit cannot draw correctly: their meshes' own world size equals their
        # declared box (0.0923 m across for SM_Mug_A2), so the factor should be 1.0, yet every route
        # through ``ComputeRelativeBound`` puts them out by a factor of 100 -- the uniform factor
        # draws them 8.5 mm across where a mug is 92 mm, and a per-axis ``1/native_i`` draws them 9.2
        # METRES across, which is what got that attempt reverted as ``bfd5be7``.  Neither is a scene.
        # See ``fit_reference_to_local_unit`` for the numbers and ``tools/probe_fit_space.py`` for the
        # space comparison; the defect is in the measurement, not in these three assets.
        #
        # Deleting them costs almost nothing on screen: at 8.5 mm they were sub-pixel in every eye
        # frame, which is exactly why the defect went unnoticed for so long.  The shelf goods that are
        # visible -- the banana below -- stay.
        #
        # Their four measurements were removed from PROP_MEASUREMENTS on 2026-10-16 as well, so that
        # nothing can put them back by accident; the numbers are kept in the comment there.
        # A banana against the obstacle wall.  `_dressing_wall` stores `size`
        # UNCHANGED in the catalogue frame (x, height, lateral), and for a wall item
        # slot 0 is the DEPTH ALONG THE WALL NORMAL.  So declaring
        # (0.1972, 0.0741, 0.0386) hangs the banana with its 0.1972 m long axis
        # pointing INTO THE CORRIDOR -- centre x = 7.99 - 0.001 - 0.1972/2 = 7.8904,
        # world x 7.7918..7.9890 -- and only 0.0386 m of it along the wall.  That is
        # 0.1972 m of protrusion, just under the 0.20 m cap, so the layout is legal;
        # this comment used to claim the opposite ("puts the 0.1972 m length ALONG
        # the wall ... holds the 0.0386 m thickness out of the corridor"), which is
        # what the code does NOT do.
        #
        # It is the deliberate PROP_NEEDS_ROTATION exception spelled out below: the
        # measured order is (x, height, lateral) = (0.0741, 0.0386, 0.1972), and
        # place_dressing references a prop with translate and scale only -- never a
        # rotation -- so a wall item whose thin axis is not its local x cannot both
        # declare the measured order and lie flat.  Unlike the poster (1.092 m, five
        # times the cap) the banana fits either way, which is why it is tolerated
        # and only warned about.  (The DRAWN banana is a speck inside this box: that
        # is the separate prop-fit defect in SCENE_SIZE_AUDIT.md, not an orientation
        # question.)
        _dressing_wall("banana", "obstacle_wall", (OBSTACLE_WALL_FACE_X, 1.04 + 0.0386 / 2, 1.98),
                       (0.1972, 0.0741, 0.0386), DRESSING_YELLOW,
                       PROP_ROOT + "/YCB/Axis_Aligned/011_banana.usd"),
    )


_install_dressing()


def episode_count() -> int:
    return (len(SCENE_ORDER) * LEVELS_PER_SCENE * MARKERS_PER_SCENE * ROSTER_SIZE)
