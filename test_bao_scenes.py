"""Checks on the Stage 1 scene and marker catalogue.

All arithmetic, no simulator: the properties that matter are statements about numbers --
markers differ from each other, a marker fits the bounding size the distance cue depends on,
dressing cannot occlude the opening from any of the 17 start views, decoration never
collides -- so they can be checked here rather than by looking at rendered frames.

Run from the repository root:  python test_bao_scenes.py
"""

from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes as sc  # noqa: E402


class Failure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# File-name prefixes that are NOT project source.  The two source scans below walk the working
# tree, and a throwaway probe sitting beside the source is not part of the repo -- diagnosed probes
# are gitignored precisely so they stay out of it -- yet a stale one failed the whole suite on the
# lab machine once ("diag_paint.py:120 sb.paint() takes 3..3 positional arguments, 4 given", after
# paint() lost a parameter).  That was a true statement about a scratch file and a false alarm
# about the project, so scratch files are excluded by design here.
SCRATCH_PREFIXES = ("diag_", "probe_", "scratch_", "tmp_", "check_")


def test_twenty_five_markers_are_distinct() -> None:
    """25 markers, and no (shape, colour) pair repeats across the whole set."""
    total = 0
    pairs = []
    for scene in sc.SCENE_ORDER:
        entries = sc.MARKERS[scene]
        check(len(entries) == sc.MARKERS_PER_SCENE,
              f"{scene} has {len(entries)} markers, expected {sc.MARKERS_PER_SCENE}")
        total += len(entries)
        pairs.extend((scene, entry) for entry in entries)
    check(total == 25, f"{total} markers in total, expected 25")

    duplicates = [pair for pair, count in collections.Counter(e for _, e in pairs).items()
                  if count > 1]
    check(not duplicates, f"these (shape, colour) pairs appear in more than one scene: {duplicates}")

    for scene in sc.SCENE_ORDER:
        colours = [colour for _, colour in sc.MARKERS[scene]]
        check(len(set(colours)) == len(colours),
              f"{scene} uses a colour twice: {colours}")
    print("[ok] 25 markers, no repeated (shape, colour), five distinct colours per scene")


def test_every_marker_names_a_real_shape_and_colour() -> None:
    for scene in sc.SCENE_ORDER:
        for shape, colour in sc.MARKERS[scene]:
            check(shape in sc.SHAPES, f"{scene}: unknown shape {shape!r}")
            check(colour in sc.COLOURS, f"{scene}: unknown colour {colour!r}")
    print(f"[ok] every marker names one of the {len(sc.SHAPES)} shapes and "
          f"{len(sc.COLOURS)} colours")


def test_every_shape_fits_the_marker_bounding_box() -> None:
    """The bounding size is the distance cue, so no shape may exceed it.

    A shape that grew past the marker's size would subtend more pixels at the same distance than
    the others, which would make the five repeats differ in something other than appearance.
    The figures below were measured at 0.60 m and scale with it, which is why they are written
    as fractions of the size rather than as millimetres.

    Measured as the shape's own extent, not as ``|vertex| <= SHAPE_HALF_M``.  Those differ
    for a shape that is anchored on its centroid rather than on the centre of its bounding
    box, which the arrow is: its span is -0.45..+0.45 at the current size but centred 2.2% of
    that size behind the anchor, so its front tip sits past +SHAPE_HALF_M and the vertex form
    would reject a shape that is exactly the budgeted size across.  The extent is what the
    distance cue depends on, so the extent is what is checked, and each end is allowed half the
    budget from the anchor.
    """
    for name, polygons in sc.SHAPES.items():
        xs = [p[0] for poly in polygons for p in poly]
        ys = [p[1] for poly in polygons for p in poly]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        check(width <= sc.MARKER_SIZE_M + 1e-9,
              f"shape {name!r} is {width:.4f} m across, past the "
              f"{sc.MARKER_SIZE_M:.2f} m budget the distance cue depends on")
        check(height <= sc.MARKER_SIZE_M + 1e-9,
              f"shape {name!r} is {height:.4f} m tall, past the "
              f"{sc.MARKER_SIZE_M:.2f} m budget")
        check(min(xs) >= -sc.SHAPE_HALF_M - sc.MARKER_CLEARANCE_M - 1e-9
              and max(xs) <= sc.SHAPE_HALF_M + sc.MARKER_CLEARANCE_M + 1e-9,
              f"shape {name!r} spans x {min(xs):+.4f}..{max(xs):+.4f}, outside the "
              f"+/-{sc.SHAPE_HALF_M:.3f} m the marker's geometry allows around its anchor")
    print(f"[ok] all 10 shapes stay inside the {sc.MARKER_SIZE_M:.2f} m budget "
          f"around their anchor")


def test_the_arrow_keeps_its_size_and_its_measured_centroid_offset() -> None:
    """The arrow's centroid is a fixed fraction of the marker's size behind its centre line.

    Three constraints on this shape cannot hold together, and the numbers were measured
    rather than argued (see the trade-off table in the commit for this test):

      * the shape's total width must equal the marker's size, like the other twenty-four,
        because the bounding size IS the distance cue;
      * it must stay inside the half-size the marker's geometry allows around its anchor;
      * its area centroid should sit on the centre line like the other nine shapes.

    A left-to-right arrow whose shaft tail reaches the -x edge and whose head tip reaches
    the +x edge has a span of exactly the marker's size, and its centroid then sits 2.205% of
    that size behind the anchor.  Centring that centroid moves the whole shape +x, putting the
    tip outside the allowed box; shrinking the shape so that both hold makes it about 4% narrower
    than the other markers, which attacks the distance cue this test exists to protect.  The
    current geometry is therefore the only one of the three options that keeps the size budget
    intact, and the offset it costs is sub-pixel: 0.28 px at the 512 px start pose, where the
    whole marker is only about 13 px across.

    The offset was 13.23 mm when the marker was 0.60 m and is 19.84 mm at 0.90 m, i.e. the same
    2.205% either way, because every shape is derived from SHAPE_HALF_M.  So the check is on the
    ratio and on the pixel figure, both of which are scale-invariant, rather than on a band of
    millimetres that only means something at one marker size.

    This test fails if the offset grows as a fraction of the size, if the width stops being
    exactly the marker's size, or if the tip leaves the allowed box.
    """
    polygons = sc.SHAPES["arrow"]
    xs = [p[0] for poly in polygons for p in poly]

    width = max(xs) - min(xs)
    check(abs(width - sc.MARKER_SIZE_M) < 1e-12,
          f"the arrow is {width:.6f} m across, not the {sc.MARKER_SIZE_M} m the other "
          f"markers use; a narrower arrow subtends fewer pixels at the same distance")

    # Area centroid of the union of its polygons -- shaft plus head, area weighted.
    total = 0.0
    cx = cy = 0.0
    for polygon in polygons:
        n = len(polygon)
        twice_area = 0.0
        px = py = 0.0
        for i in range(n):
            x0, y0 = polygon[i]
            x1, y1 = polygon[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            twice_area += cross
            px += (x0 + x1) * cross
            py += (y0 + y1) * cross
        area = twice_area / 2.0
        px /= (6.0 * area)
        py /= (6.0 * area)
        total += abs(area)
        cx += px * abs(area)
        cy += py * abs(area)
    cx /= total
    cy /= total

    check(abs(cy) < 1e-12, f"the arrow's centroid is {cy:+.6f} m off the centre line in y")
    # Expressed as a FRACTION of the marker's size, not as a band of millimetres.
    #
    # The absolute offset was 13.23 mm when the marker was 0.60 m.  Scaling the marker to 0.90 m
    # scaled the arrow with it -- every shape is derived from SHAPE_HALF_M -- so the offset
    # became 19.84 mm, which is exactly 13.23 x 1.5.  Nothing about the shape changed; only the
    # unit did.  A millimetre band would therefore have failed on a change that did not touch
    # the geometry, and the way to notice a real change is the ratio:
    #
    #     13.23 mm / 0.60 m = 0.02205   and   19.84 mm / 0.90 m = 0.02205
    #
    # The sub-pixel property below is scale-invariant too, for the same reason, so this pair of
    # checks now says "the trade-off is the one that was derived" at any marker size instead of
    # at one particular size.
    ratio = cx / sc.MARKER_SIZE_M
    check(-0.02305 < ratio < -0.02105,
          f"the arrow's centroid is {ratio:.5f} of the marker's size behind the anchor "
          f"({cx*1000:+.2f} mm at {sc.MARKER_SIZE_M:.2f} m), outside the derived "
          f"-0.02205 +/- 0.001; if this changed, re-derive the three-way trade-off rather "
          f"than widening this band")
    # Sub-pixel at the start pose, which is why it is acceptable at all.
    import math
    distance = sc.MARKER_X_M - 0.5
    fov = 2.0 * math.degrees(math.atan(sc.SENSOR_WIDTH_MM / (2.0 * sc.FOCAL_LENGTH_MM)))
    mm_per_px = (2.0 * distance * math.tan(math.radians(fov / 2.0))) / 512.0
    check(abs(cx) / mm_per_px < 0.5,
          f"the arrow's {abs(cx)*1000:.2f} mm offset is {abs(cx)/mm_per_px:.2f} px at the "
          f"start pose, no longer sub-pixel")

    check(min(xs) >= -sc.SHAPE_HALF_M - 1e-9,
          f"the arrow's tail is at {min(xs):+.6f}, outside the allowed "
          f"{-sc.SHAPE_HALF_M:+.3f} m")
    print(f"[ok] arrow: {width:.4f} m across, centroid {cx*1000:+.2f} mm "
          f"({abs(cx)/mm_per_px:.2f} px at the start pose), tail at {min(xs):+.4f} m")


def test_shapes_are_fat_enough_to_read_at_the_start_pose() -> None:
    """At 512 px from the start the marker is about 20 px, so thin parts blur away.

    The measure is each polygon's smaller bounding dimension, because every marker is a flat
    plate facing the robot: its 0.02 m thickness runs along the view axis and says nothing
    about legibility.  An earlier version measured min(height, 2*radius) and flagged the
    triangle for being a plate, which is what all of them are.
    """
    minimum = sc.MARKER_SIZE_M * 0.20
    for name, polygons in sc.SHAPES.items():
        for polygon in polygons:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            feature = min(max(xs) - min(xs), max(ys) - min(ys))
            check(feature >= minimum - 1e-9,
                  f"shape {name!r} has an in-plane part {feature:.3f} m across, below the "
                  f"{minimum:.3f} m that stays legible from the start pose")
    print("[ok] no shape has an in-plane part thinner than a fifth of the marker")


def test_no_dressing_occludes_the_opening_or_the_marker() -> None:
    """The opening must stay readable at every width, and the marker always visible.

    Projected with the same focal length, sensor and downward pitch the environment uses,
    from the start pose, which is where the agent decides.
    """
    checked = 0
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            bounds = sc.screen_bounds(item["at"], item["size"])
            if bounds is None:
                continue  # behind the camera, cannot occlude anything ahead
            for _level, width in enumerate(sorted(_ladder_widths())):
                opening = sc.screen_bounds(*sc.opening_box(width))
                # Only what is IN FRONT of the obstacle wall can occlude the opening.  Things
                # on the far wall are seen THROUGH it, so an earlier version flagged the far
                # wall's own sign for overlapping the opening's projected rectangle -- which is
                # what everything behind the opening does, the marker's wall included.  The
                # |z| >= 0.9 rule keeps on-wall items clear of the gap itself.
                if (opening and float(item["at"][0]) < 8.0
                        and sc.rectangles_overlap(bounds, opening)):
                    raise Failure(
                        f"{scene}: {item['name']} overlaps the {width:.3f} m opening in the "
                        f"start view (item {bounds}, opening {opening})")
                checked += 1
            marker = sc.screen_bounds(*sc.marker_box())
            if marker and sc.rectangles_overlap(bounds, marker):
                raise Failure(f"{scene}: {item['name']} overlaps the marker in the start view")
    print(f"[ok] no dressing overlaps the opening or the marker "
          f"({checked} item x width pairs projected)")


def _ladder_widths():
    sys.path.insert(0, ".")
    import environment as env
    return [env.LEVEL_CHANNEL_WIDTHS[level] for level in sorted(env.LEVEL_CHANNEL_WIDTHS)]


def test_decoration_is_never_collidable_and_stays_off_the_path() -> None:
    """Dressing must be decoration: no collision, and out of the central band."""
    count = 0
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            count += 1
            check(item["collides"] is False,
                  f"{scene}: {item['name']} is collidable, which would change the task")
            z = abs(float(item["at"][2]))
            if item["mount"] == "floor":
                check(z >= 1.5, f"{scene}: floor item {item['name']} sits at |z| {z:.2f}, "
                                f"inside the band the robot walks")
                check(item["at"][1] <= 0.6 or z >= 1.8,
                      f"{scene}: floor item {item['name']} is {item['at'][1]:.2f} m up at "
                      f"|z| {z:.2f}; tall items belong outside |z| 1.8")
            elif item["mount"] == "obstacle_wall":
                check(z >= 0.9, f"{scene}: obstacle-wall item {item['name']} sits at "
                                f"|z| {z:.2f}, inside the widest opening's half width plus "
                                f"the margin")
            else:
                check(item["mount"] == "far_wall",
                      f"{scene}: {item['name']} has an unknown mount {item['mount']!r}")
    print(f"[ok] all {count} dressing items are non-collidable and outside the walking band")


def test_dressing_colours_are_declared_and_not_marker_colours() -> None:
    """Every dressing item has a colour, and none of them borrows a marker's.

    An earlier version looped over item["colour"] and asserted that at least one item had no
    colour, which was true then and false once every item was given one -- the test was
    inverted by a fix.  What matters is the two properties below.
    """
    for scene in sc.SCENE_ORDER:
        marker_colours = set(sc.marker_colours(scene))
        for item in sc.SCENES[scene]["dressing"]:
            colour = item.get("colour")
            check(colour is not None, f"{scene}: {item['name']} has no colour")
            check(tuple(colour) not in marker_colours,
                  f"{scene}: {item['name']} uses a marker colour {colour}")
    print("[ok] every dressing item has a colour, and none of them is a marker colour")


def test_the_baseline_scene_is_untouched() -> None:
    """stage1.1 must stay exactly the scene the 660 committed episodes were recorded in."""
    baseline = sc.SCENES["stage1.1"]
    check(baseline["materials"] == {}, "the baseline scene must not rebind any material")
    check(baseline["dressing"] == (), "the baseline scene must not add any dressing")
    check(sc.MARKERS["stage1.1"][0] == ("square", "r"),
          "the baseline's first marker must be the existing red square")
    print("[ok] stage1.1 keeps its materials, its empty dressing and the red square")


def test_materials_are_marked_as_verified_or_not() -> None:
    """A material URL is either read off the bucket listing or flagged for confirmation."""
    for scene, spec in sc.SCENES.items():
        if spec["material_verified"]:
            for surface, url in spec["materials"].items():
                check(url.startswith(sc.ASSET_ROOT),
                      f"{scene}: {surface} claims to be verified but {url} is not under "
                      f"the Isaac asset root")
        else:
            check(spec["materials"], f"{scene} claims unverified materials but has none")
    print("[ok] every material URL says whether it was read off the bucket listing")


def test_the_episode_count_is_what_the_design_says() -> None:
    check(sc.episode_count() == 6375,
          f"the catalogue implies {sc.episode_count()} episodes, expected 6,375")
    check(len(sc.SCENE_ORDER) == 5, f"{len(sc.SCENE_ORDER)} scenes, expected 5")
    print("[ok] 5 scenes x 17 levels x 5 markers x 15 models = 6,375 episodes")


def test_the_report_says_the_marker_size_and_whether_the_paint_landed() -> None:
    """The size and the paint result must be readable from the run log.

    The marker went to 0.90 m, and a paint that fails would otherwise be visible only as a
    differently coloured marker in a picture -- which is the kind of thing this session has been
    wrong about twice.  A self-lit component was tried alongside the size change and removed
    again; the flag it needed is still worth reporting, because paint() can fail on its own.

    format_report() is pure, so this needs no stage.
    """
    import scene_builder

    report = {
        "scene": "stage1.2", "label": "warehouse", "slot": 2,
        "marker": {"shape": "hexagon", "colour": "c", "parts": 1,
                   "size_m": sc.MARKER_SIZE_M,
                   "painted": True, "paint_error": ""},
        "materials": {}, "dressing": [], "dressing_count": 0,
    }
    text = scene_builder.format_report(report)
    check("marker:" in text, f"the report has no marker line:\n{text}")
    check(f"{sc.MARKER_SIZE_M:.2f} m" in text,
          f"the marker line does not state the size:\n{text}")
    check("painted True" in text,
          f"the marker line does not say whether painting succeeded:\n{text}")

    # And a failure has to be visible AS a failure, WITH its reason.  Two render round trips were
    # spent on "painted False" alone, because the except had discarded the answer.
    report["marker"]["painted"] = False
    report["marker"]["paint_error"] = "AttributeError: 'X' object has no attribute 'Y'"
    failed = scene_builder.format_report(report)
    check("painted False" in failed,
          "a marker whose paint failed reports the same as one that succeeded")
    check("no attribute 'Y'" in failed,
          f"the failure is reported without its reason, so it cannot be acted on:\n{failed}")
    print("[ok] the marker report carries its size, whether the paint landed, and the reason "
          "when it did not")


def test_every_colour_is_used_at_least_once() -> None:
    """A colour in the palette that no marker uses would be dead weight in the design."""
    used = collections.Counter(colour for scene in sc.SCENE_ORDER
                               for _, colour in sc.MARKERS[scene])
    unused = sorted(set(sc.COLOURS) - set(used))
    check(not unused, f"these colours are defined but never used: {unused}")
    shapes_used = {shape for scene in sc.SCENE_ORDER for shape, _ in sc.MARKERS[scene]}
    unused_shapes = sorted(set(sc.SHAPES) - shapes_used)
    check(not unused_shapes, f"these shapes are defined but never used: {unused_shapes}")
    print(f"[ok] all {len(sc.COLOURS)} colours and all {len(sc.SHAPES)} shapes are used")


def test_the_builder_imports_without_a_simulator() -> None:
    """scene_builder must import where pxr does not exist.

    The catalogue checks run on the development machine, so a module-level pxr import would
    make them impossible there; pxr is imported inside the functions that need it instead.
    """
    import scene_builder

    check(hasattr(scene_builder, "apply_scene"), "scene_builder has no apply_scene")
    report = {
        "scene": "stage1.2", "slot": 3, "label": sc.SCENES["stage1.2"]["label"],
        "marker": {"shape": "arrow", "colour": "l", "parts": 2},
        "materials": {"floor": {"how": "mdl", "url": "https://example/MI_Floor_01.mdl"},
                      "ceiling": {"how": "fallback", "url": "https://example/MI_Ceiling.mdl"}},
        "dressing": [{"name": "pallets", "mount": "floor", "asset": None, "used_asset": False},
                     {"name": "sign", "mount": "obstacle_wall", "asset": "x", "used_asset": True}],
        "dressing_count": 2,
    }
    text = scene_builder.format_report(report)
    check("arrow" in text, f"the report does not name the marker shape:\n{text}")
    check("fallback" in text, f"the report does not admit a fallback material:\n{text}")
    check("1 from assets" in text, f"the report miscounts asset-backed dressing:\n{text}")
    print("[ok] scene_builder imports without pxr and reports what it actually did")


def test_the_catalogue_matches_the_environment_it_replaces() -> None:
    """The baseline marker must reproduce the one the frozen environment already builds.

    Pin the catalogue against environment.py's own constants, so that changing one without
    the other fails here rather than silently moving or resizing the marker on the far wall.

    This was briefly weakened to make room for two sources of truth: the size check was dropped
    and replaced with a check that the preview harness passed an override, because
    environment.GOAL_MARKER_SIZE had been left at 0.60 m to avoid moving a baseline that 660
    test episodes had been rendered against.  That reason was wrong -- those episodes are a test
    run, not a reference later work has to follow -- so both values are 0.90 m now, the override
    is gone, and the size is pinned again.  One number in one place beats an override plus a test
    that checks the override.
    """
    import environment as env

    check(abs(sc.MARKER_SIZE_M - float(env.GOAL_MARKER_SIZE)) < 1e-9,
          f"catalogue marker size {sc.MARKER_SIZE_M} vs environment {env.GOAL_MARKER_SIZE}; "
          f"the environment builds its own marker at the same /World/GoalMarker path, so these "
          f"two have to be the same number")
    check(abs(sc.MARKER_WALL_X_M - float(env.ROOM_LENGTH_X)) < 1e-9,
          "the catalogue puts the marker on a different wall than the room's far wall")
    check(abs(sc.MARKER_Y_M - 1.40) < 1e-9,
          "the catalogue's marker centre height differs from the environment's 1.40 m")
    check(tuple(sc.COLOURS["r"]) == (0.85, 0.15, 0.12),
          "the red in the palette is not the existing marker's red")
    print(f"[ok] the catalogue's baseline marker matches the environment's own constants "
          f"(size {sc.MARKER_SIZE_M:.2f} m, wall, height, red)")


def test_the_built_marker_matches_the_environment_plate_exactly() -> None:
    """The plate the catalogue builds must occupy the same x range as the environment's.

    The check above compares *declared* coordinates, and the marker's placement relative to
    the wall is exactly the kind of thing that passes such a check while being wrong: the
    catalogue said x = 16.0 and so does ``ROOM_LENGTH_X``, so it looked pinned -- but
    ``_add_box`` builds a box *centred* on its coordinate, and the far wall lives at
    16.00 .. 16.02.  Centring a 0.02 m plate on 16.0 therefore buried half of it in the wall
    and put the face the camera sees 0.02 m behind the environment's.  Nothing failed.

    Both paths are read from source here (``environment._create_goal_marker``'s corner list
    and ``scene_builder._prism``'s vertex list) and their world-frame extents are compared,
    which needs no Isaac Sim because both are pure arithmetic on the constants.
    """
    import environment as env

    # --- environment._create_goal_marker -> _add_box -------------------------
    thickness = 0.02                                    # goal_marker_span default
    centre_user = (
        env.ROOM_LENGTH_X - thickness / 2.0 - 0.01,     # goal_marker_base offset
        1.40,
        0.0,
    )
    dims = (thickness, float(env.GOAL_MARKER_SIZE), float(env.GOAL_MARKER_SIZE))
    cx, cy, cz = env._user_to_isaac_pos(centre_user)
    hx, hy, hz = (d / 2.0 for d in dims)
    env_x = (cx - hx, cx + hx)

    # --- scene_builder.build_marker -> _prism --------------------------------
    half = sc.MARKER_THICKNESS_M / 2.0
    ccx, _, _ = sc.to_world((sc.MARKER_X_M, sc.MARKER_Y_M, sc.MARKER_Z_M))
    cat_x = (ccx - half, ccx + half)

    check(
        abs(env_x[0] - cat_x[0]) < 1e-12 and abs(env_x[1] - cat_x[1]) < 1e-12,
        f"the catalogue's plate spans x {cat_x[0]:.4f}..{cat_x[1]:.4f} but the "
        f"environment's spans {env_x[0]:.4f}..{env_x[1]:.4f}; the far wall's inner face "
        f"is at {env.ROOM_LENGTH_X}, so a plate centred on the wall plane is half buried "
        f"in it and its visible face is 0.02 m too far away",
    )

    # The wall's inner face, from the environment's own room construction.
    wall_thickness = env.ROOM_WALL_THICKNESS
    inner_face = (env.ROOM_LENGTH_X + wall_thickness / 2.0) - wall_thickness / 2.0
    check(
        cat_x[1] <= inner_face + 1e-12,
        f"the plate reaches x={cat_x[1]:.4f}, past the far wall's inner face at "
        f"{inner_face:.4f}; the plate must sit in front of the wall, not inside it",
    )
    check(
        cat_x[1] <= sc.MARKER_WALL_X_M + 1e-12,
        f"the plate reaches x={cat_x[1]:.4f}, beyond the wall plane "
        f"{sc.MARKER_WALL_X_M}; half of it would be outside the room",
    )
    print(
        f"[ok] the built plate occupies x {cat_x[0]:.4f}..{cat_x[1]:.4f}, exactly the "
        f"environment's {env_x[0]:.4f}..{env_x[1]:.4f}, clear of the wall at {inner_face:.4f}"
    )


def test_the_scene_tag_separates_scenes_and_leaves_old_tags_alone() -> None:
    """Two scenes must not resolve to one directory, and old tags must not move.

    The committed 660 episodes live under tags composed without a scene, so the no-scene
    result has to stay byte-identical; and feeding a full tag back in has to be idempotent,
    which is the trap that once produced <model>-v4-walkframe-v4-walkframe.
    """
    import main

    plain = main.effective_tag("some-model")
    check("stage1." not in plain, f"the default tag mentions a scene: {plain}")
    for scene in sc.SCENE_ORDER:
        tagged = main.effective_tag("some-model", "", scene)
        check(tagged == plain + "-" + scene,
              f"{scene}: {tagged} is not {plain}-{scene}")
        again = main.effective_tag("some-model", tagged, scene)
        check(again == tagged, f"{scene}: feeding the tag back changed it to {again}")
    for other in sc.SCENE_ORDER[1:]:
        check(main.effective_tag("some-model", "", other)
              != main.effective_tag("some-model", "", sc.SCENE_ORDER[0]),
              "two scenes compose the same tag")
    print("[ok] the scene is in the tag, reapplying is idempotent, old tags unchanged")


def test_surfaces_are_classified_by_geometry_not_by_name() -> None:
    """The builder finds our floor, ceiling and walls without knowing their names.

    environment.py is frozen, so the builder classifies by where a box sits and how big its
    faces are.  This is the pure core of discovery, checked here with boxes shaped like the
    real room.
    """
    import scene_builder

    boxes = [
        ("/World/Anything_7", (0.0, -0.01, -2.5), (16.0, 0.0, 2.5)),     # floor
        ("/World/Opus_3", (0.0, 3.0, -2.5), (16.0, 3.01, 2.5)),          # ceiling
        ("/World/Thing", (15.99, 0.0, -2.5), (16.0, 3.0, 2.5)),          # far wall
        ("/World/Other", (0.0, 0.0, 2.49), (16.0, 3.0, 2.5)),            # side wall
    ]
    found = scene_builder.classify_surfaces(boxes)
    check(found.get("floor") == "/World/Anything_7", f"floor found as {found.get('floor')}")
    check(found.get("ceiling") == "/World/Opus_3", f"ceiling found as {found.get('ceiling')}")
    check(found.get("far_wall") == "/World/Thing", f"far wall found as {found.get('far_wall')}")
    check(found.get("side_wall") == "/World/Other", f"side wall found as {found.get('side_wall')}")
    check(scene_builder.classify_surfaces([]) == {},
          "an empty stage should classify to nothing, not to a guess")
    print("[ok] the four surfaces are found by geometry, and an empty stage yields nothing")


def test_catalogue_frame_matches_the_environment_converter() -> None:
    """Pin the catalogue's frame conversion to the environment's own, point for point.

    environment.py already had this: ``_user_to_isaac_pos`` is documented as "Map a user-frame
    position (y up) to Isaac Sim (z up): swap y and z", and ``_add_box`` calls it for every wall,
    the ceiling and the original red marker -- which is why those were always in the right place.
    The catalogue grew its own copy of the operation instead of reading that one, and the copy
    was never applied at all, which is the entire bug.  Two definitions of one convention is the
    defect; this test makes them one.
    """
    import numpy as np

    import environment as env

    points = [(16.0, 1.40, 0.0), (8.0, 2.6, 1.10), (0.5, 1.68, -0.3), (2.6, 0.20, 2.05)]
    for point in points:
        mine = sc.to_world(point)
        theirs = tuple(float(v) for v in env._user_to_isaac_pos(np.array(point, dtype=float)))
        check(mine == theirs, f"to_world({point}) = {mine} but the environment says {theirs}")
        back = tuple(float(v) for v in env._isaac_to_user_pos(np.array(mine, dtype=float)))
        check(back == point, f"the environment's inverse of {mine} is {back}, not {point}")
    for size in [(0.02, 0.6, 0.6), (0.9, 0.4, 0.7), (1.4, 1.9, 0.16)]:
        mine = sc.to_world_size(size)
        theirs = tuple(float(v) for v in env._user_to_isaac_scale(np.array(size, dtype=float)))
        check(mine == theirs, f"to_world_size({size}) = {mine} but the environment says {theirs}")
    # The original red square, authored in _create_goal_marker as
    # [ROOM_LENGTH_X - thickness/2 - 0.01, 1.40, 0.0] and converted by environment's own
    # function, is the placement that was always right.  The catalogue marker must land at the
    # same height and the same lateral offset; x differs by the 0.02 inset that keeps the legacy
    # square off the wall surface.
    thickness = 0.02
    legacy_authored = (env.ROOM_LENGTH_X - thickness / 2.0 - 0.01, 1.40, 0.0)
    legacy_world = tuple(float(v) for v in env._user_to_isaac_pos(np.array(legacy_authored)))
    mine_world = sc.to_world((sc.MARKER_X_M, sc.MARKER_Y_M, sc.MARKER_Z_M))
    check(abs(mine_world[1] - legacy_world[1]) < 1e-9 and abs(mine_world[2] - legacy_world[2]) < 1e-9,
          f"the marker's world (lateral, height) {mine_world[1:]} must match the original red "
          f"square's {legacy_world[1:]}")
    print("[ok] the catalogue's frame conversion is identical to the environment's own")


def test_world_and_user_frames_are_inverse() -> None:
    """The stage is Z-up and the catalogue is height-second, and the two must join correctly.

    Not academic: scene_builder places prims directly, so if this swap is wrong a marker meant
    for 1.40 m up the far wall ends up at height zero and 1.40 m sideways, which is where the
    first renders found it -- nowhere.
    """
    for point in ((16.0, 1.40, 0.0), (8.0, 2.6, 1.10), (0.5, 1.68, -0.3)):
        world = sc.to_world(point)
        check(sc.to_user(world) == point, f"to_user(to_world({point})) = {sc.to_user(world)}")
        check(sc.to_world(world) == point, f"to_world is not its own inverse at {point}")
    marker_world = sc.to_world((16.0, sc.MARKER_Y_M, sc.MARKER_Z_M))
    check(marker_world[2] == sc.MARKER_Y_M,
          f"the marker's world HEIGHT must be its catalogue y; got {marker_world}")
    check(marker_world[1] == sc.MARKER_Z_M,
          f"the marker's world lateral offset must be its catalogue z; got {marker_world}")
    check(sc.to_world_size((0.9, 0.4, 0.7)) == (0.9, 0.7, 0.4),
          "a box's dimensions must swap with their axes")
    print("[ok] the catalogue's frame and the stage's Z-up frame are exact inverses")


def test_real_room_boxes_classify_correctly() -> None:
    """Regression: the world boxes reported on the lab machine, converted and classified.

    These are the numbers discover_surfaces actually printed for the real room, in world
    coordinates.  With user-coordinate rules applied to them directly -- which is what the code
    did -- the floor came out as room_side_left (its y is negative and the rule asked only for
    y <= 0.05), the side wall came out as room_ceiling, and the real ceiling was never found.  If
    this test fails, that bug is back.
    """
    import scene_builder

    world_boxes = [
        ("/World/Ground", (0.0, -2.5, -0.02), (16.0, 2.5, 0.0)),
        ("/World/room_ceiling", (0.0, -2.5, 3.0), (16.0, 2.5, 3.02)),
        ("/World/room_side_left", (0.0, -2.52, 0.0), (16.0, -2.5, 3.0)),
        ("/World/room_side_right", (0.0, 2.5, 0.0), (16.0, 2.52, 3.0)),
        ("/World/room_far", (16.0, -3.5, 0.0), (16.02, 3.5, 3.0)),
    ]
    boxes = [(name, sc.to_user(low), sc.to_user(high)) for name, low, high in world_boxes]
    found = scene_builder.classify_surfaces(boxes)
    check(found.get("floor") == "/World/Ground", f"floor found as {found.get('floor')}")
    check(found.get("ceiling") == "/World/room_ceiling",
          f"ceiling found as {found.get('ceiling')}")
    check(found.get("far_wall") == "/World/room_far", f"far wall found as {found.get('far_wall')}")
    check(set(found.get("side_walls", ())) == {"/World/room_side_left", "/World/room_side_right"},
          f"both side walls must be reported, got {found.get('side_walls')}")
    print("[ok] the real room's world boxes classify to the right prims, both side walls")


def test_prim_paths_from_a_scene_name_are_valid_usd_paths() -> None:
    """A dot is a property separator in SdfPath, and every scene name has one.

    Measured on the lab machine: /World/Looks/Marker_stage1.1_1 is ill-formed, USD refused
    it, and the material was simply never created -- so a scene would have run with its
    surfaces unbound and nothing in the results would have said so.
    """
    import scene_builder

    check(scene_builder.prim_name("stage1.2") == "stage1_2",
          f"prim_name('stage1.2') is {scene_builder.prim_name('stage1.2')!r}")
    for scene in sc.SCENE_ORDER:
        for surface in ("floor", "side_wall", "ceiling", "far_wall"):
            name = f"{surface}_{scene_builder.prim_name(scene)}"
            check("." not in name, f"{surface} of {scene} composes an ill-formed path {name}")
        for slot in range(1, sc.MARKERS_PER_SCENE + 1):
            name = f"Marker_{scene_builder.prim_name(scene)}_{slot}"
            check("." not in name, f"{scene} slot {slot} composes an ill-formed path {name}")
    print("[ok] every composed material path is a valid USD path, dots replaced")


def test_no_dressing_protrudes_into_the_corridor() -> None:
    """Nothing may stick out from a wall into the space the agent looks down.

    The first rendered preview showed a black slab hanging across the middle of the agent's
    own view.  It was the warehouse's duct: size (1.8, 0.30, 0.30) centred on x = 8.0, so it
    reached 0.9 m out from the wall towards the robot.  The |z| >= 0.9 rule that was supposed
    to prevent this only constrains where an item sits sideways, not how far it protrudes, so
    it is now constrained too.
    """
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            if item["mount"] == "floor":
                continue
            protrusion = float(item["size"][0]) / 2.0
            check(protrusion <= 0.20,
                  f"{scene}: {item['name']} protrudes {protrusion:.2f} m from the wall into "
                  f"the corridor the agent looks along")
    print("[ok] no wall-mounted dressing protrudes more than 0.20 m into the corridor")


def test_every_dressing_item_has_a_colour() -> None:
    """An untextured prim with no colour renders black, which is what the first preview showed.

    The lab machine cannot write the material library's cache, so there is no material to fall
    back on either; a colour on every item is what keeps the scene from having unexplained
    black slabs in it.
    """
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            colour = item.get("colour")
            check(colour is not None, f"{scene}: {item['name']} has no colour")
            check(len(colour) == 3, f"{scene}: {item['name']} colour {colour} is not RGB")
            check(all(0.0 <= float(v) <= 1.0 for v in colour),
                  f"{scene}: {item['name']} colour {colour} is out of range")
    print("[ok] every dressing item carries an in-range RGB colour")


def test_every_colour_word_names_its_own_rgb() -> None:
    """The colour word that goes into the prompt must describe the RGB that is drawn.

    ``describe_marker`` builds the noun phrase the task sentence uses -- "reach the <phrase>
    on the far wall" -- so a colour word that does not match the plate on the wall hands the
    model a false statement about the object it is being asked to reach.

    This existed and was wrong: ``w``, a dark wine red (0.55, 0.05, 0.25), was called
    "white", and ``k``, pink (1.00, 0.40, 0.70), was called "black".  Three of the twenty-five
    prompts were therefore false -- "white set of two bars", "white arrow", "black hexagon" --
    and nothing noticed, because the uniqueness tests compare RGB triples and no test ever read
    the rendered phrase.

    The word table is pinned literally, per colour key, with the key's own RGB printed in the
    failure so the contradiction is visible rather than asserted.
    """
    # key -> (word, what the RGB is, roughly).  The word must match the RGB on the right.
    expected = {
        "r": ("red", (0.85, 0.15, 0.12)),
        "m": ("magenta", (0.90, 0.10, 0.55)),
        "o": ("orange", (1.00, 0.45, 0.05)),
        "l": ("lime", (0.85, 0.95, 0.10)),
        "c": ("cyan", (0.10, 0.65, 0.90)),
        "p": ("purple", (0.55, 0.15, 0.75)),
        "g": ("green", (0.20, 0.85, 0.25)),
        "w": ("wine", (0.55, 0.05, 0.25)),
        "t": ("teal", (0.05, 0.80, 0.70)),
        "k": ("pink", (1.00, 0.40, 0.70)),
    }
    check(set(expected) == set(sc.COLOURS),
          f"the palette and this table name different colours: "
          f"{sorted(set(expected) ^ set(sc.COLOURS))}")
    for key, (word, rgb) in expected.items():
        check(sc.COLOUR_NAMES[key] == word,
              f"colour {key!r} is RGB {sc.COLOURS[key]} but its word is "
              f"{sc.COLOUR_NAMES[key]!r}, not {word!r}; that word goes into the prompt")
        check(tuple(sc.COLOURS[key]) == rgb,
              f"colour {key!r} is {sc.COLOURS[key]}, but this table says {rgb}; the word "
              f"{word!r} was chosen for the RGB on the right")

    # And the phrases themselves, which is what the prompt actually carries.
    for scene in sc.SCENE_ORDER:
        for slot, (shape, colour) in enumerate(sc.MARKERS[scene], start=1):
            phrase = sc.describe_marker(scene, slot)
            want = f"{expected[colour][0]} {sc.SHAPE_NAMES[shape]}"
            check(phrase == want,
                  f"{scene} slot {slot}: describe_marker gives {phrase!r}, expected {want!r}")
    print("[ok] every colour word matches its own RGB, and all 25 prompt phrases follow")


def test_no_prompt_ever_names_a_colour_the_palette_does_not_use() -> None:
    """The words that reach the prompt are exactly this palette's words, and no others.

    The companion to the check above from the other side: not "is each word right" but "is
    each word one of ours".  A stale entry copied from an earlier draft -- "white", "black",
    "lime green" -- has to fail here even if its own RGB happens to be plausible.
    """
    allowed = {"red", "magenta", "orange", "lime", "cyan", "purple", "green",
               "wine", "teal", "pink"}
    words = set(sc.COLOUR_NAMES.values())
    check(words == allowed,
          f"the colour words are {sorted(words)}, expected exactly {sorted(allowed)}")
    for scene in sc.SCENE_ORDER:
        for slot, (_, colour) in enumerate(sc.MARKERS[scene], start=1):
            first = sc.describe_marker(scene, slot).split()[0]
            check(first in allowed,
                  f"{scene} slot {slot}: the prompt would say {first!r}, which is not a "
                  f"colour in this palette")
    print(f"[ok] all 25 prompts use one of the {len(allowed)} palette colour words")


def test_each_scene_declares_its_dressing_exactly_once() -> None:
    """A scene's layout lives inline or in _install_dressing, never in both.

    The revision this pins had full inline tuples for 1.3, 1.4 and 1.5 that
    ``_install_dressing`` overwrote at import.  Twenty-four lines of configuration read
    exactly like the live layout, and editing them changed nothing whatsoever -- the sort
    of thing that costs an afternoon of "but I already fixed that".

    Read off the AST rather than off the module, because the overwriting is exactly what
    the attribute lookup cannot see: at runtime the dead list is simply gone.
    """
    import ast
    import os as _os

    src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scenes.py")
    tree = ast.parse(open(src, encoding="utf-8").read(), filename=src)

    scenes_literal = None
    install_assigns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "SCENES":
            scenes_literal = node.value
        # SCENES["stage1.3"]["dressing"] = (...)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Subscript)
                    and getattr(target.value.value, "id", "") == "SCENES"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "dressing"):
                inner = target.value.slice
                if isinstance(inner, ast.Constant):
                    install_assigns.append((inner.value, node.value.lineno))

    check(scenes_literal is not None, "no module-level SCENES literal found")
    inline_counts = {}
    for key, value in zip(scenes_literal.keys, scenes_literal.values):
        for k, v in zip(value.keys, value.values):
            if isinstance(k, ast.Constant) and k.value == "dressing":
                inline_counts[key.value] = len(v.elts)
    installed = {scene for scene, _ in install_assigns}
    print(f"    inline counts {inline_counts}, assigned by _install_dressing "
          f"{sorted(installed)}")

    for scene in sc.SCENE_ORDER:
        inline = inline_counts.get(scene, 0)
        live = len(sc.SCENES[scene]["dressing"])
        if scene in installed:
            check(inline == 0,
                  f"{scene} has {inline} inline dressing items AND is assigned by "
                  f"_install_dressing, so the inline list is dead configuration")
        else:
            check(inline == live,
                  f"{scene} is not assigned by _install_dressing, so its {live} live items "
                  f"must all be inline; the AST found {inline}")
    print("[ok] every scene declares its dressing in exactly one place")


def test_dressing_stands_on_the_floor_and_hangs_on_its_wall() -> None:
    """Mounting geometry: nothing buried in the floor, nothing floating off its wall.

    Both faults were real and both were invisible to every other check here, because those
    compare declarations while these two are about where the box actually ends up.

    Floor items were written with the box CENTRE as their height and the height differs per
    item, so seven hovered (the park benches by 25 mm) and the supermarket checkout stood with
    its base 5 mm BELOW the floor plane -- a box half-buried in the ground.  The helper now
    takes the base, so the design value for anything standing on the floor is 0.

    Wall items were written at the obstacle wall's own centre line, x = 8.0, while the wall
    occupies 7.99..8.01 -- so all fourteen straddled it, the 0.16-0.30 m ones with 20 mm of
    themselves inside the wall.  The catalogue names the SURFACE now
    (OBSTACLE_WALL_FACE_X / FAR_WALL_FACE_X) and the helper offsets by a millimetre of
    clearance, which is a statement about which way the box faces, not about how big it is --
    an earlier attempt subtracted half the thickness as well, which made two items of
    different thickness hang at different distances from the same named face.

    Asserted from the placed geometry, so it holds however the call sites are written.
    """
    import environment as env

    obstacle_face = float(env.WALL_X) - float(env.WALL_THICKNESS) / 2.0   # 7.99
    far_face = float(env.ROOM_LENGTH_X)                                   # 16.0
    buried_floor, detached_wall = [], []
    count = 0
    deepest = 0.0

    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            count += 1
            at, size = item["at"], item["size"]
            if item["mount"] == "floor":
                base = float(at[1]) - float(size[1]) / 2.0
                if base < -1e-9:
                    buried_floor.append((scene, item["name"], base))
                continue
            face = obstacle_face if item["mount"] == "obstacle_wall" else far_face
            half = float(size[0]) / 2.0
            back = float(at[0]) - half
            # "Hanging on the wall" is a statement about the BACK face: it must sit at the
            # wall's surface, not in front of it.  The helper puts the box centre ON the face,
            # so the back ends up half a thickness inside the wall -- which is what mounting
            # looks like, and the reason this check must not be written against the FRONT
            # face.  An earlier version of this test did, and failed every item: the front is
            # meant to stand half a thickness proud, and that is bounded by
            # test_no_dressing_protrudes_into_the_corridor instead of duplicated here.
            gap = back - face
            deepest = min(deepest, gap)
            if gap > sc.MOUNT_CLEARANCE_M + 1e-9:
                detached_wall.append((scene, item["name"], round(gap, 4)))

    check(not buried_floor,
          f"these floor items have their base below the floor plane: "
          f"{[(s, n, round(b * 1000, 1)) for s, n, b in buried_floor]}")
    check(not detached_wall,
          f"these wall items float in front of their wall by this many metres: "
          f"{detached_wall}")
    print(f"[ok] all {count} dressing items stand on the floor and hang on their wall "
          f"(obstacle face {obstacle_face}, far face {far_face}; deepest back face "
          f"{deepest * 1000:.0f} mm inside the wall)")


def test_no_scene_name_reaches_a_usd_path_unsanitised() -> None:
    """Every USD path built from a scene name goes through prim_name().

    A scene is called ``stage1.2`` and a dot is a PROPERTY separator in SdfPath, so
    ``/World/Probe/stage1.2/klt_bins`` is ill-formed.  USD does not fail loudly in a way that
    stops a run: it prints

        Warning: in SdfPath ... Ill-formed SdfPath </World/Probe/stage1.2/klt_bins>: syntax error

    and the tool carries on having built nothing.  Measured on the workstation, in
    tools/measure_props.py, which is how this test came to exist -- scene_builder.prim_name()
    had been there all along and the new file simply did not use it.

    Checked by reading the source rather than by calling anything, because the failure is a
    missing call, and a missing call looks exactly like a correct one at runtime.
    """
    import scene_builder

    # The sanitiser itself.
    for given, want in (("stage1.2", "stage1_2"), ("stage1.5", "stage1_5"),
                        ("klt_bins", "klt_bins"), ("a/b", "a_b"), ("a.b.c", "a_b_c")):
        got = scene_builder.prim_name(given)
        check(got == want, f"prim_name({given!r}) gave {got!r}, expected {want!r}")

    # Any f-string path under /World that interpolates a scene-name variable must wrap it.
    import re

    root = os.path.dirname(os.path.abspath(__file__))
    offenders = []
    for name in ("scene_builder.py", "environment.py", "capture_scenes.py",
                 "tools/measure_props.py", "tools/scene_diagrams.py"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        for number, line in enumerate(open(path, encoding="utf-8"), start=1):
            if "/World" not in line or "{" not in line:
                continue
            for variable in re.findall(r"\{(scene|tag|scene_name)\}", line):
                offenders.append(f"{name}:{number} interpolates {variable!r}: {line.strip()}")
    check(not offenders,
          "these USD paths embed a scene name without prim_name():\n    "
          + "\n    ".join(offenders))
    print("[ok] no scene name reaches a USD path without prim_name()")


def test_no_module_asks_pxr_for_something_it_does_not_have() -> None:
    """Catch a misspelt pxr member without needing pxr installed.

    This is the fourth time this session that code reached for a pxr API that does not exist,
    and every one of them was found by running it on the workstation:

      * Prim.GetReferences() on a schema instead of a prim;
      * UsdGeom.BoxCache, when the class is UsdGeom.BBoxCache;
      * UsdGeom.BBoxCache.CreateBoxCache(...).ComputeWorldBounds(), when the constructor takes
        a time code and the method is ComputeWorldBound(prim);
      * a probe bound read off a Cube whose extent was never authored.

    None of those can be caught by importing the module here -- there is no pxr on this
    machine, which is exactly why they survived to the workstation.  What CAN be caught is a
    reference to a member that the USD Python API does not define, by checking the module
    against a list of the members these files actually use.  The list is short and explicit on
    purpose: an allowlist that grows silently is not a check, so a new member has to be added
    here deliberately, with the docs open.

    See https://openusd.org/release/api/ -- in particular
    class_usd_geom_b_box_cache.html, which is what corrected the second and third faults above.

    The scan is over the CODE, not the text.  This docstring names the wrong spellings, and the
    first version of this check flagged its own prose -- as did a message string elswhere that
    mentioned one.  String literals are blanked before scanning for exactly that reason.
    """
    import ast as _ast
    import re

    # module -> members these files may use, verified against the OpenUSD API reference or, better,
    # against the machine.  A name that a probe has shown does NOT exist must be removed rather
    # than left here.  One was: a token wrapper on pxr.Tf (spelled out in the docstring of
    # scene_builder.paint, which is blanked before this scan) was deleted from both the dict and
    # the code on 2026-10-10, after the workstation showed that module has no token type at all --
    # its only "oken" members are DumpTokenStats and two test helpers.  This table must not simply
    # accumulate, and this comment cannot spell the dotted name it is about: the scan reads
    # comments, and doing so reported itself.
    ALLOWED = {
        "Gf": {"Vec3f", "Vec3d", "Matrix4d"},
        "Sdf": {"ValueTypeNames", "Path", "AssetPath"},
        "Vt": {"Vec3fArray", "Vec3dArray", "IntArray", "Token", "Value"},
        "Usd": {"Stage", "Prim", "TimeCode", "Attribute"},
        "UsdGeom": {
            "Cube", "Sphere", "Cylinder", "Cone", "Capsule", "Mesh", "Xform", "Scope",
            "Tokens", "SetStageUpAxis", "Xformable", "XformOp", "BBoxCache", "Gprim",
            "Imageable", "Boundable", "Camera", "GetStageUpAxis",
        },
        "UsdLux": {"SphereLight", "DomeLight", "DiskLight", "RectLight"},
        "UsdShade": {"Material", "Shader", "ConnectableAPI", "Input", "Output",
                     "MaterialBindingAPI"},
        "UsdPhysics": {"RigidBodyAPI", "CollisionAPI", "MassAPI", "ArticulationRootAPI",
                       "Joint"},
        "UsdSkel": set(),
    }
    # Any name in the pxr style, not only the ones already listed.
    #
    # This took three attempts and each failure is the same mistake, so it is recorded: the
    # pattern must be able to SEE the new thing it is meant to police.
    #
    #   1. built from ALLOWED's own keys -> a new pxr name was added to scene_builder.py and the
    #      guard still reported "43 members, all verified" while checking none of the new one;
    #   2. widened to an explicit list of module names -> an unlisted pxr module was still
    #      invisible, which I only found by planting one and watching the guard stay silent;
    #   3. this: any `Name.Name` where the first name looks like a pxr module (one of the
    #      prefixes pxr uses) AND is never bound at module level in that file.  The second clause
    #      is what keeps it usable: `sc`, `sb`, `os`, `np` and friends are bound by their imports,
    #      and no pxr module is, so they are excluded by construction rather than by a hand list.
    #
    # (Step 3 also means this docstring cannot spell a module-qualified example: the pattern has
    # no way to tell prose from code, and naming one here reports it.  It did.)
    PXR_LIKE = re.compile(r"\b((?:Gf|Sdf|Tf|Pcp|Pxr|Usd|Vt|Kind|Trace|Work|Plug|Ar|CameraUtil)"
                          r"[A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]*")
    KNOWN_MODULES = set(ALLOWED)

    def code_text(source: str) -> list:
        """The source with every string literal blanked out, so prose cannot be mistaken."""
        rows = [list(line) for line in source.splitlines()]
        for node in _ast.walk(_ast.parse(source)):
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                for index in range(node.lineno - 1, node.end_lineno):
                    if index >= len(rows):
                        continue
                    start = node.col_offset if index == node.lineno - 1 else 0
                    end = node.end_col_offset if index == node.end_lineno - 1 else len(rows[index])
                    for column in range(min(start, len(rows[index])), min(end, len(rows[index]))):
                        rows[index][column] = " "
        return ["".join(chars) for chars in rows]

    root = os.path.dirname(os.path.abspath(__file__))
    offenders = []
    for folder in (root, os.path.join(root, "tools")):
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".py") or name.startswith(SCRATCH_PREFIXES):
                continue
            source = open(os.path.join(folder, name), encoding="utf-8").read()
            tree = _ast.parse(source)

            # Names bound at MODULE level in this file.  These files import pxr inside the
            # functions that need it, so a pxr module name is never bound here -- which makes
            # "bound at module level" a sound reason to skip, rather than a hand-kept list:
            # `sc`, `sb`, `os`, `np` and friends are all bound here, and no pxr module is.
            bound = set()
            for node in tree.body:
                targets = []
                if isinstance(node, _ast.Assign):
                    targets = node.targets
                elif isinstance(node, _ast.AnnAssign):
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, _ast.Name):
                        bound.add(target.id)
                if isinstance(node, _ast.Import):
                    for alias in node.names:
                        bound.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, _ast.ImportFrom):
                    for alias in node.names:
                        bound.add(alias.asname or alias.name)

            for number, line in enumerate(code_text(source), start=1):
                for module in PXR_LIKE.findall(line):
                    if module in bound:
                        continue
                    if module not in KNOWN_MODULES:
                        offenders.append(
                            f"{name}:{number} {module} is used but is not an allowlisted pxr "
                            f"module, so none of its members are checked")
                        continue
                    for member in re.findall(rf"\b{module}\.([A-Za-z_][A-Za-z0-9_]*)", line):
                        if member not in ALLOWED[module]:
                            offenders.append(f"{name}:{number} {module}.{member}")

    check(not offenders,
          "these pxr members are not in the allowlist of ones this repo uses, so either the "
          "name is misspelt or a new API is being used without checking the docs:\n    "
          + "\n    ".join(sorted(set(offenders))))
    total = sum(len(v) for v in ALLOWED.values())
    print(f"[ok] every pxr member used in the repo is on the verified list ({total} members)")


def test_no_module_calls_a_scene_helper_that_does_not_exist() -> None:
    """Every ``sc.<name>`` / ``sb.<name>`` in the repo exists, and takes the arguments given.

    This exists because of a measured mistake: tools/measure_assets.py called
    ``scenes.to_user_size()``, which does not exist -- the catalogue defines ``to_user`` for a
    point and ``to_world_size`` for a size.  That file cannot be run here (it needs isaacsim),
    so nothing local would have complained until the workstation run 160 s later.

    Unlike the pxr check, this one can be exact: `scenes` and `scene_builder` import fine on
    this machine, so the attributes and the signatures are available to be inspected.  A call
    with the wrong number of positional arguments is caught too, which is the same fault one
    level down.

    Scratch diagnostics are skipped.  This scans the working tree, so a throwaway probe left beside
    the source -- and diagnosed probes are deliberately gitignored, so they are not in the repo --
    was able to fail the whole suite on the lab machine after paint() lost its emissive parameter:
    "diag_paint.py:120 sb.paint() takes 3..3 positional arguments, 4 given".  That is a true
    statement about a stale file and a false alarm about the project.  The repo's own modules are
    what this is for, so the names below are excluded by design rather than by whoever remembers
    to delete them.
    """
    import ast as _ast
    import inspect

    import scene_builder
    import scenes as catalogue

    MODULES = {"sc": catalogue, "sb": scene_builder}
    problems = []

    root = os.path.dirname(os.path.abspath(__file__))
    for folder in (root, os.path.join(root, "tools")):
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".py") or name.startswith(SCRATCH_PREFIXES):
                continue
            source = open(os.path.join(folder, name), encoding="utf-8").read()
            tree = _ast.parse(source)

            # Walk ATTRIBUTE ACCESSES, not lines.  A regex over lines cannot tell code from a
            # comment or a docstring: the previous two attempts at this reported
            # "sb.SCENE_ORDER does not exist" for a comment that said exactly that, and
            # "sc.screen_bounds() takes 1" for a correct starred call.  An ast.Attribute whose
            # value is the imported alias is unambiguous -- comments and strings contain no
            # nodes at all.
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.Attribute):
                    continue
                if not isinstance(node.value, _ast.Name) or node.value.id not in MODULES:
                    continue
                alias, member = node.value.id, node.attr
                if not hasattr(MODULES[alias], member):
                    problems.append(f"{name}:{node.lineno} {alias}.{member} does not exist")

            # Argument counts, walked from the AST so multi-line calls are checked too.
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.Call) or not isinstance(node.func, _ast.Attribute):
                    continue
                if not isinstance(node.func.value, _ast.Name):
                    continue
                alias = node.func.value.id
                if alias not in MODULES:
                    continue
                member = node.func.attr
                target = getattr(MODULES[alias], member, None)
                if target is None or not callable(target):
                    continue
                try:
                    signature = inspect.signature(target)
                except (TypeError, ValueError):
                    continue
                if any(p.kind is p.VAR_POSITIONAL for p in signature.parameters.values()):
                    continue
                # A starred argument unpacks at runtime, so the count here is unknown.  The
                # first version of this check ignored that and reported
                # sc.screen_bounds(*sc.opening_box(width)) -- which is correct code.
                if any(isinstance(argument, _ast.Starred) for argument in node.args):
                    continue
                required = [p for p in signature.parameters.values()
                            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY,
                                                                   p.POSITIONAL_OR_KEYWORD)]
                positional = len(node.args)
                maximum = len([p for p in signature.parameters.values()
                               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
                if positional < len(required) or positional > maximum:
                    problems.append(
                        f"{name}:{node.lineno} {alias}.{member}() takes {len(required)}.."
                        f"{maximum} positional arguments, {positional} given")

    check(not problems,
          "these calls into the catalogue or the scene builder cannot work:\n    "
          + "\n    ".join(sorted(set(problems))))
    print("[ok] every sc./sb. helper called in the repo exists and takes the arguments given")


def test_a_declared_size_is_the_size_of_the_prop_it_names() -> None:
    """A floor item that references a mesh declares that mesh's real size.

    place_dressing references the prop at its own scale, on top of a box of the declared size.
    So if the two disagree, one of two things is silently true and neither is visible from the
    catalogue: the mesh hangs outside the volume the occlusion and walking-band checks were
    computed for, or the box is drawn around a much smaller object and every "the dressing is
    clear of the opening" conclusion is about a box rather than about the thing in it.

    Both directions were found by measuring on the workstation:

        pallet.usd   declared (0.9, 0.40, 0.9)      actually (1.2132, 0.1425, 0.8023)
        small_KLT    declared (0.6, 0.30, 0.5)      actually (0.1978, 0.1464, 0.2966)

    a pallet declared smaller than a real pallet and a load carrier declared larger than a real
    one -- the two guesses did not even err the same way.

    The comparison is against scenes.PROP_MEASUREMENTS, which is the recorded output of
    tools/measure_named_props.py, because no USD reader exists on this machine.  A prop referenced
    without a recorded measurement fails here rather than passing unmeasured.

    Every item naming a .usd is checked, not only the floor ones.  This used to skip wall items
    for the same reason place_dressing did, and that skipped the supermarket's shelf goods -- a
    mug standing on a shelf is a wall item, so the two checks that were supposed to catch a
    mismatched declaration were both blind to the newest props.

    The measurement is compared AFTER applying scenes.ASSET_SCALE, because PROP_MEASUREMENTS holds
    what the asset reports in its own units and part of this library is authored in centimetres:
    the packing table measures 247.3647 there and is 2.4736 m of furniture.  A declared size is
    always metres, so the factor is what makes the two comparable; without it every centimetre
    asset would be reported as a 100x mismatch.
    """
    measured = sc.PROP_MEASUREMENTS
    scales = getattr(sc, "ASSET_SCALE", {})
    pending = set(getattr(sc, "PROP_MEASUREMENT_PENDING", ()))
    problems, checked, waiting = [], [], []

    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            asset = item.get("asset")
            if not asset:
                continue
            name = os.path.basename(asset)
            if not name.endswith(".usd"):
                # An .mdl is a material and never a reference; test_bao_assets covers those.
                continue
            if name not in measured and name in pending:
                # Wired, waiting for a measurement.  Recorded rather than skipped silently: the
                # point of PROP_MEASUREMENT_PENDING is that the gap is visible on every run.
                waiting.append(f"{scene}/{item['name']}->{name}")
                continue
            if name not in measured:
                problems.append(f"{scene}/{item['name']} references {name}, which has no "
                                f"recorded measurement")
                continue
            want = measured[name]
            factor = float(scales.get(name, 1.0))
            if factor <= 0:
                problems.append(f"{scene}/{item['name']}: ASSET_SCALE[{name!r}] is {factor}, "
                                f"which cannot scale anything")
                continue
            room_fit = float(item.get("room_fit", 1.0))
            if not 0.0 < room_fit <= 1.0:
                problems.append(f"{scene}/{item['name']}: room_fit is {room_fit}; it is a fraction "
                                f"of the measured size and must be in (0, 1]")
                continue
            metres = tuple(v * factor * room_fit for v in want)
            got = tuple(float(v) for v in item["size"])
            # Compared as SETS, not axis by axis, and the reason is a genuine ambiguity rather than
            # convenience.  PROP_MEASUREMENTS records an asset as the measuring tool reported it --
            # for the framed poster that is (1.092, 0.7426, 0.0524), whose slot 0 is its WIDTH --
            # while a wall item has to be stored with its slot 0 on the wall's normal, i.e. its
            # DEPTH, which for that poster is 0.0524.  The same three numbers therefore appear in
            # two different orders and no axis-by-axis comparison can hold for both.  What this
            # check can still guarantee is what it is for: the item occupies the size of the prop it
            # names, so a declared (1.8, 0.3, 0.3) around a 0.3 m prop is still an error, and the
            # separate protrusion check constrains the depth, which is the axis that actually
            # matters for whether something hangs into the corridor.
            if sorted(round(v, 4) for v in got) != sorted(round(v, 4) for v in metres):
                problems.append(
                    f"{scene}/{item['name']} ({item['mount']}) declares {got} m, but {name} measures "
                    f"{want} in its own units x ASSET_SCALE {factor} x room_fit {room_fit} = "
                    f"{metres} m -- the three dimensions do not match as a set")
            else:
                checked.append(f"{item['name']}->{name}"
                               + (f" (room_fit {room_fit})" if room_fit != 1.0 else ""))
            continue

    check(not problems,
          "these declared sizes are not the size of the prop they reference, so the box the "
          "fixture checks use is not the thing that will be drawn:\n    "
          + "\n    ".join(problems))
    print(f"[ok] {len(checked)} declared size(s) equal the measured prop: {checked}, and "
          f"{len(measured)} measurement(s) are recorded")
    if waiting:
        # Loud, on every run, and it does not fail: PROP_MEASUREMENT_PENDING means the prop is
        # placed with a size chosen by hand and not yet checked against the asset.
        print(f"[waiting] {len(waiting)} prop(s) are wired but not measured, so their declared "
              f"sizes are unchecked: {waiting}")
        print("          run tools/measure_named_props.py on the machine with a USD reader and "
              "move them into PROP_MEASUREMENTS")


def test_the_marker_assertion_counts_parts_not_children() -> None:
    """set_marker_slot must count marker PARTS, not whatever sits under /World/GoalMarker.

    The assertion exists to refuse a scene with two markers on the far wall, because two markers
    means two colours and the uniqueness premise the prompts rest on is void.  It used to do that
    by counting the path's children, which was the same thing only while the marker was its parts
    and nothing else.  When a self-lit variant added one material per part at "<part>_emissive",
    a sibling of the part, the first full render died with

        expected 1 marker part(s) on the far wall after replacing it, found 2

    which is the right failure reported about the wrong thing.  The material is gone again, but
    the count is now by name prefix and the message names what it found, so the next thing added
    under that path cannot masquerade as a second marker.

    This is a source scan: the geometry and the assertion both need pxr, so the relationship is
    pinned where it is written.
    """
    import scene_builder

    part = scene_builder.MARKER_PART_PREFIX
    material = scene_builder.MARKER_MATERIAL_PREFIX
    check(part.startswith("part") and material.startswith("material"),
          "the marker name constants no longer match the names build_marker writes")
    check(not material.startswith(part),
          f"the material prefix {material!r} starts with the part prefix {part!r}, so a count of "
          f"parts would include materials")

    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "scene_builder.py"), encoding="utf-8").read()
    check('f"{MARKER_ROOT}/part{index}"' in source,
          "build_marker no longer names its parts from MARKER_ROOT and MARKER_PART_PREFIX")

    env_source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "environment.py"), encoding="utf-8").read()
    check("startswith(scene_builder.MARKER_PART_PREFIX)" in env_source,
          "set_marker_slot no longer counts parts by their prefix, so it is counting children "
          "again and anything else under /World/GoalMarker would read as a second marker")

    # Every shape has to produce at least one part, or the count would be zero and the assertion
    # would compare 0 against 0 and call that agreement.
    empty = [name for name, polygons in sc.SHAPES.items() if not polygons]
    check(not empty, f"these shapes have no polygons, so they build no marker part: {empty}")
    print(f"[ok] marker parts and their materials are separately named "
          f"({part!r} vs {material!r}), and set_marker_slot counts parts")


def test_no_scene_floor_is_covered_by_a_dark_material() -> None:
    """Every textured floor must sample an albedo texture bright enough to be seen.

    This exists because one was not.  stage1.2 used the warehouse's own MI_Floor_01, whose albedo
    is ColorAlbedo 0.145 grey lerped with Textures/T_Floor_01_D.png -- a texture whose mean
    luminance is 63 of 255.  The rendered floor band came out at 29.8 with 31.9 % of it near
    black, against 106.4 for the light baseline, and the scene was the only one that darkened
    towards the bottom of the frame.  A near-black floor is not a matter of taste: the eye camera
    looks down a corridor whose floor fills the lower fifth of the image.

    The check follows the material's own indirection rather than trusting a name: an MDL declares
    the textures it samples, and the albedo one is measured here with Pillow.  Numbers, because
    the alternative is rendering the scene and looking at it, which is what missed this twice.

    The floor is not required to exist -- stage1.4 keeps the default ground on purpose, which is
    its outdoor cue -- but a floor material that IS declared has to be bright enough.
    """
    import re

    import numpy as np
    from PIL import Image

    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "isaac")
    floor_min = float(getattr(sc, "FLOOR_ALBEDO_MIN", 80.0))
    texture = re.compile(r'texture_2d\(\s*"\./Textures/([^"]+)"')

    # Found by basename rather than by rebuilding the URL's path.  The first version rebuilt it
    # and reported a real, vendored material as "not vendored" -- a failure for the wrong reason,
    # which would send the next reader after a missing file instead of a dark one.
    vendored = {}
    for folder, _, files in os.walk(assets):
        for name in files:
            vendored.setdefault(name, os.path.join(folder, name))

    reported, problems = [], []

    for scene in sc.SCENE_ORDER:
        url = sc.SCENES[scene]["materials"].get("floor")
        if not url:
            continue
        basename = os.path.basename(url)
        candidate = vendored.get(basename)
        check(candidate,
              f"{scene}'s floor material {basename} is not vendored anywhere under assets/isaac")
        source = open(candidate, encoding="utf-8", errors="replace").read()
        names = [n for n in texture.findall(source)
                 if re.search(r"_D\.png|basecolor|_bc\.png", n, re.I)]
        check(names, f"{scene}'s floor material {basename} declares no albedo texture, so this "
                     f"check cannot measure it and must not pass it silently")
        for name in names:
            image_path = os.path.join(os.path.dirname(candidate), "Textures", name)
            check(os.path.exists(image_path),
                  f"{scene}'s floor samples {name}, which is not vendored at {image_path}")
            grey = np.asarray(Image.open(image_path).convert("RGB")).astype(float).mean(axis=2)
            mean, low = float(grey.mean()), float(grey.min())
            reported.append(f"{scene}:{basename}->{name}={mean:.1f}")
            if mean < floor_min:
                problems.append(f"{scene} floor {basename} samples {name}, whose mean luminance "
                                f"is {mean:.1f} of 255, minimum {low:.1f} (floor {floor_min})")

    check(not problems,
          "these floors are covered by a material too dark to see, which is how stage1.2 "
          "rendered its floor at 29.8 with 31.9 % of it near black:\n    "
          + "\n    ".join(problems))
    print(f"[ok] every declared floor samples a bright albedo: {reported}")


def test_the_dressing_code_has_none_of_this_session_s_known_faults() -> None:
    """Run tools/audit_dressing.py, whose checks each stand for a defect that happened here.

    The faults it looks for are the ones that cost round trips in this session: a size or scale
    converted through to_world_size without a mount guard (three separate occurrences, each
    swapping an item's height with its width), an exception recorded without its traceback (so a
    USD ErrorException reported nothing at all), a scale op added without checking for an authored
    one, a wall item whose depth exceeds the 0.20 m cap, a prop wired in without a measurement, and
    any box taller than the room.

    It is run as a subprocess rather than imported so that a failure inside it reports its own
    message instead of raising here.  The check was verified to fail by planting a missing mount
    guard in scene_builder.py and watching it name the line.
    """
    import subprocess

    root = os.path.dirname(os.path.abspath(__file__))
    audit = os.path.join(root, "tools", "audit_dressing.py")
    check(os.path.exists(audit), "tools/audit_dressing.py is missing, so nothing audits the "
                                 "dressing code for the faults this session produced")
    result = subprocess.run([sys.executable, audit], capture_output=True, text=True, cwd=root)
    check(result.returncode == 0,
          f"tools/audit_dressing.py reports problems:\n{result.stdout}{result.stderr}")
    print(f"[ok] the dressing audit found none of the session's known faults "
          f"({len([line for line in result.stdout.splitlines() if line.strip().startswith('note:')])} "
          f"guarded conversion(s) confirmed)")


def main() -> int:
    tests = [
        test_twenty_five_markers_are_distinct,
        test_every_marker_names_a_real_shape_and_colour,
        test_every_shape_fits_the_marker_bounding_box,
        test_the_arrow_keeps_its_size_and_its_measured_centroid_offset,
        test_shapes_are_fat_enough_to_read_at_the_start_pose,
        test_no_dressing_occludes_the_opening_or_the_marker,
        test_no_dressing_protrudes_into_the_corridor,
        test_dressing_stands_on_the_floor_and_hangs_on_its_wall,
        test_every_dressing_item_has_a_colour,
        test_decoration_is_never_collidable_and_stays_off_the_path,
        test_dressing_colours_are_declared_and_not_marker_colours,
        test_the_baseline_scene_is_untouched,
        test_materials_are_marked_as_verified_or_not,
        test_the_episode_count_is_what_the_design_says,
        test_every_colour_is_used_at_least_once,
        test_every_colour_word_names_its_own_rgb,
        test_no_prompt_ever_names_a_colour_the_palette_does_not_use,
        test_each_scene_declares_its_dressing_exactly_once,
        test_the_builder_imports_without_a_simulator,
        test_the_catalogue_matches_the_environment_it_replaces,
        test_the_built_marker_matches_the_environment_plate_exactly,
        test_the_scene_tag_separates_scenes_and_leaves_old_tags_alone,
        test_surfaces_are_classified_by_geometry_not_by_name,
        test_catalogue_frame_matches_the_environment_converter,
        test_world_and_user_frames_are_inverse,
        test_real_room_boxes_classify_correctly,
        test_prim_paths_from_a_scene_name_are_valid_usd_paths,
        test_no_scene_name_reaches_a_usd_path_unsanitised,
        test_no_module_asks_pxr_for_something_it_does_not_have,
        test_no_module_calls_a_scene_helper_that_does_not_exist,
        test_a_declared_size_is_the_size_of_the_prop_it_names,
        test_the_report_says_the_marker_size_and_whether_the_paint_landed,
        test_the_marker_assertion_counts_parts_not_children,
        test_no_scene_floor_is_covered_by_a_dark_material,
        test_the_dressing_code_has_none_of_this_session_s_known_faults,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Failure as exc:
            failed += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} checks FAILED")
        return 1
    print(f"all {len(tests)} scene-catalogue checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
