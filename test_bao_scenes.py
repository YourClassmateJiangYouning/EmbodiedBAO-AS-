"""Checks on the Stage 3 scene and marker catalogue.

All arithmetic, no simulator: the properties that matter are statements about numbers --
markers differ from each other, a marker fits the bounding size the distance cue depends on,
dressing cannot occlude the opening from any of the 17 start views, decoration never
collides -- so they can be checked here rather than by looking at rendered frames.

Run from the repository root:  python test_bao_scenes.py
"""

from __future__ import annotations

import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes as sc  # noqa: E402


class Failure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


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

    A shape that grew beyond 0.60 m would subtend more pixels at the same distance than the
    others, which would make the five repeats differ in something other than appearance.
    """
    for name, polygons in sc.SHAPES.items():
        for polygon in polygons:
            for px, py in polygon:
                check(max(abs(px), abs(py)) <= sc.SHAPE_HALF_M + 1e-9,
                      f"shape {name!r} has a vertex at ({px:.3f}, {py:.3f}), past the "
                      f"{sc.SHAPE_HALF_M:.3f} m half-size of the marker's bounding box")
    print("[ok] all 10 shapes stay inside the 0.60 m bounding box")


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
            for level, width in enumerate(sorted(_ladder_widths())):
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
    the other fails here rather than silently moving the marker on the far wall.
    """
    import environment as env

    check(abs(sc.MARKER_SIZE_M - float(env.GOAL_MARKER_SIZE)) < 1e-9,
          f"catalogue marker size {sc.MARKER_SIZE_M} vs environment "
          f"{env.GOAL_MARKER_SIZE}")
    check(abs(sc.MARKER_X_M - float(env.ROOM_LENGTH_X)) < 1e-9,
          "the catalogue puts the marker on a different wall than the room's far wall")
    check(abs(sc.MARKER_Y_M - 1.40) < 1e-9,
          "the catalogue's marker centre height differs from the environment's 1.40 m")
    check(tuple(sc.COLOURS["r"]) == (0.85, 0.15, 0.12),
          "the red in the palette is not the existing marker's red")
    print("[ok] the catalogue's baseline marker matches the environment's own constants")


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


def main() -> int:
    tests = [
        test_twenty_five_markers_are_distinct,
        test_every_marker_names_a_real_shape_and_colour,
        test_every_shape_fits_the_marker_bounding_box,
        test_shapes_are_fat_enough_to_read_at_the_start_pose,
        test_no_dressing_occludes_the_opening_or_the_marker,
        test_no_dressing_protrudes_into_the_corridor,
        test_every_dressing_item_has_a_colour,
        test_decoration_is_never_collidable_and_stays_off_the_path,
        test_dressing_colours_are_declared_and_not_marker_colours,
        test_the_baseline_scene_is_untouched,
        test_materials_are_marked_as_verified_or_not,
        test_the_episode_count_is_what_the_design_says,
        test_every_colour_is_used_at_least_once,
        test_the_builder_imports_without_a_simulator,
        test_the_catalogue_matches_the_environment_it_replaces,
        test_the_scene_tag_separates_scenes_and_leaves_old_tags_alone,
        test_surfaces_are_classified_by_geometry_not_by_name,
        test_prim_paths_from_a_scene_name_are_valid_usd_paths,
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
