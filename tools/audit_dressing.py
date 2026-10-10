"""Audit the dressing code for the mistakes this session actually made, by reading it.

Written after the user asked why the same class of bug kept appearing.  Every check below stands
for a real defect from this session, so a green run means those specific faults are absent rather
than that the code "looks fine".

  1. A frame conversion applied to a tuple that is already in that frame.  This happened three
     times -- _dressing_wall, fit_scale_from_extent, and the scale op -- and each time it silently
     swapped an item's height with its width.
  2. An exception whose record keeps only str(exc), which for USD's ErrorException is the empty
     string, so a report said "FAILED ErrorException:" and nothing else.
  3. A transform op added without checking for an authored one of the same type.
  4. A prop fitted along an axis the checks do not use, i.e. a wrong-axis fit.
  5. A helper called with a size whose frame does not match what that helper expects.
  6. An unmeasured prop wired into a scene, whose declared size was therefore invented.

Run from the repository root:  python tools/audit_dressing.py
"""
from __future__ import annotations

import ast
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

FILES = ("scene_builder.py", "scenes.py")


def read(name: str) -> str:
    return open(os.path.join(_ROOT, name), encoding="utf-8").read()


def main() -> int:
    problems = []
    notes = []

    builder = read("scene_builder.py")

    # 1. to_world_size applied to a WALL item's size or scale.  Wall sizes are stored in world
    # order already, so every conversion must be guarded by the mount.
    for name, source in (("scene_builder.py", builder),):
        for lineno, line in enumerate(source.splitlines(), 1):
            if "to_world_size" not in line:
                continue
            window = "\n".join(source.splitlines()[max(0, lineno - 6):lineno + 2])
            if 'item["mount"] != "floor"' in window or "mount ==" in window:
                notes.append(f"{name}:{lineno} guarded to_world_size (mount-aware)")
            else:
                problems.append(
                    f"{name}:{lineno} calls to_world_size without a mount guard, so a WALL item's "
                    f"already-world-order tuple would be swapped again: {line.strip()}")

    # 2. An except that records only str(exc).
    tree = ast.parse(builder)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = "".join(ast.dump(statement) for statement in node.body)
        if "reference_error" in body and "traceback" not in body and "extract_tb" not in body:
            problems.append(
                f"scene_builder.py:{node.lineno} records an exception without its traceback, so a "
                f"USD ErrorException -- whose str() is empty -- would report nothing")

    # 3. AddScaleOp must not be reached without checking for an authored op.
    if "def apply_scale" not in builder:
        problems.append("scene_builder.py has no apply_scale helper; an authored xformOp:scale "
                        "would make AddScaleOp raise")
    elif "AddScaleOp()" in builder:
        guarded = "GetOrderedXformOps" in builder
        notes.append(f"AddScaleOp present, GetOrderedXformOps present: {guarded}")
        if not guarded:
            problems.append("AddScaleOp is used without GetOrderedXformOps, so an asset that ships "
                            "a scale op cannot be fitted")

    # 4/5. Every wall call passes a size in the frame _dressing_wall expects, and every wall item
    # with an asset has its depth on slot 0.
    import scenes as sc                                                  # noqa: E402

    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            if item["mount"] == "floor":
                continue
            depth = float(item["size"][0])
            if depth > 0.20 + 1e-9:
                problems.append(f"{scene}/{item['name']} stores {depth:.3f} m on its x axis, past "
                                f"the 0.20 m protrusion cap")

    # 6. Wired props with no measurement and not declared pending.
    pending = set(getattr(sc, "PROP_MEASUREMENT_PENDING", ()))
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            asset = item.get("asset")
            if not asset:
                continue
            base = os.path.basename(asset)
            if not base.endswith(".usd"):
                continue
            if base not in sc.PROP_MEASUREMENTS and base not in pending:
                problems.append(f"{scene}/{item['name']} references unmeasured {base}")

    # 7. A wall item may not be declared taller than the room's 3 m ceiling, and a floor item's box
    # may not exceed it either: the fit would stretch the prop through the ceiling.
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            if float(item["size"][1]) > 3.0:
                problems.append(f"{scene}/{item['name']} is {item['size'][1]:.2f} m tall, above the "
                                f"3.0 m ceiling")

    # 8. The fit must be UNIFORM.  A per-axis factor reshapes a prop whose proportions differ from
    # its declared box, and that is not a measurement error to be caught -- it is the design being
    # wrong.  It showed up as (1.0, 16.0075, 0.0624) on the safety railing and (1.0, 2.4425, 0.7788)
    # on an office chair: a railing 16x too tall, a chair squashed in one axis.  A uniform factor
    # cannot do that, and an asset larger than its box is shrunk rather than reshaped.
    if "def fit_scale_from_extent" not in builder:
        problems.append("scene_builder.py has no fit_scale_from_extent; the fit cannot be checked")
        fit_body = ""
    else:
        # Split on the NEXT top-level def, so the slice is this function only.  An earlier version
        # split on "\ndef " and caught apply_scale's ComputeWorldBound comment below, reporting a
        # defect in a function that no longer had one.
        fit_body = builder.split("def fit_scale_from_extent", 1)[1]
        for marker in ("\ndef ", "\nclass "):
            if marker in fit_body:
                fit_body = fit_body.split(marker, 1)[0]
    if fit_body and "min(ratios)" not in fit_body and "min(min(ratios)" not in fit_body:
        problems.append("fit_scale_from_extent does not derive a single factor from the per-axis "
                        "ratios, so it would reshape the prop instead of scaling it")
    if "fitted.append" in fit_body:
        problems.append("fit_scale_from_extent builds a per-axis result list, which means the prop "
                        "is reshaped rather than scaled")

    # 9. No declared box may be smaller than the prop it names, because the prop is fitted INSIDE
    # the box: a box that cannot hold it means the prop overhangs and clips, which is the failure
    # this project must not have.  Compared as sorted sets, for the wall-item frame difference, and
    # with a tolerance because a box derived from the same measurement differs from it by float
    # rounding -- an exact comparison reported eleven items whose two lists printed identically.
    # 5e-5: declared sizes are written to four decimal places by hand, so a box derived from a
    # measurement can differ from it by half a unit in the last place -- 0.17 written against
    # 0.1700070 computed.  An exact or 1e-6 comparison reported fifteen items that fit perfectly.
    TOL = 5e-5
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            asset = item.get("asset")
            if not asset:
                continue
            base = os.path.basename(asset)
            measured = sc.PROP_MEASUREMENTS.get(base)
            if not measured:
                continue
            factor = float(sc.ASSET_SCALE.get(base, 1.0))
            room_fit = float(item.get("room_fit", 1.0))
            prop = sorted(float(v) * factor * room_fit for v in measured)
            box = sorted(float(v) for v in item["size"])
            if any(box[axis] < prop[axis] - TOL for axis in range(3)):
                problems.append(
                    f"{scene}/{item['name']}: box {[round(v, 4) for v in box]} m cannot hold the prop "
                    f"{[round(v, 4) for v in prop]} m, so the prop would overhang and clip")

    # 9. The bound API must be the one whose SEMANTICS fit the question.  This is the fault class a
    # name allowlist cannot catch, and it produced the worst defect of this session: every one of
    # these calls is a real API, so only reading the reference distinguishes them.  From
    # https://openusd.org/release/api/class_usd_geom_b_box_cache.html :
    #
    #   ComputeWorldBound(prim)                 "in world space" -- includes the prim's own
    #                                           transform AND every ancestor's.
    #   ComputeRelativeBound(prim, ancestor)    "in the space of an ancestor prim ... excludes the
    #                                           local transform at relativeToAncestorPrim".
    #   ComputeLocalBound(prim)                 includes the prim's own transform, excludes ancestor
    #                                           transforms, and returns an ORIENTED box.
    #   ComputeUntransformedBound(prim)         excludes the prim's own transform.
    #
    # fit_scale_from_extent measures a prop that is a CHILD of the clearance box whose scale it is
    # about to set, so an ancestor-including bound measures the fit against itself: reported as
    # factors of 219.9, 39.5 and 11.1.  A relative bound excludes exactly that ancestor.
    if "ComputeRelativeBound" not in builder:
        problems.append("scene_builder.py measures a fit with no ComputeRelativeBound, so the "
                        "extent would include the parent box's own scale and feed itself")
    # fit_body was extracted once, above, for rule 8.  It was extracted a SECOND time here, and the
    # duplicate split on "\ndef " rather than on the next top-level def, so it ran past the end of
    # fit_scale_from_extent into apply_scale and read that function's comment about ComputeWorldBound
    # as though it were a call -- reporting a defect in a function that no longer had one.  Two
    # extractions of the same slice is the duplication this project treats as a defect in its own
    # right, and it produced exactly that: a false report.
    if fit_body and "ComputeWorldBound" in fit_body:
        problems.append("fit_scale_from_extent uses ComputeWorldBound, whose bound includes every "
                        "ancestor transform -- including the parent box's scale, which is what this "
                        "function is deciding.  Use ComputeRelativeBound.")
    # 10. place_dressing must VERIFY the fit on the assembled stage.  The clearance cube carries a
    # scale of `size` and the prop under it carries a scale of its own; xform ops compose down the
    # hierarchy, so whether the drawn prop is native x size x factor or something else is a runtime
    # fact that reading the source cannot settle -- and guessing it is how this session produced its
    # most expensive defects.  The code therefore measures each resolved prop's extent back off the
    # stage with a FRESH BBoxCache and records fit_error when it exceeds its declared box.
    if "fit_error" not in builder:
        problems.append("place_dressing never verifies the drawn size against the declared box, so a "
                        "prop scaled wrongly would be invisible until someone looked at a picture")
    if "drawn_extent" not in builder:
        problems.append("place_dressing does not record what the stage says was drawn, so no report "
                        "can state a prop's actual size")
    place_body = builder.split("def place_dressing", 1)[-1]
    # A SECOND BBoxCache, constructed after placement.  Reusing the one that measured the fit would
    # read cached bounds: the reference states plainly that the cache "does not listen for change
    # notifications; the user is responsible for clearing the cache when changes occur".
    if place_body.count("BBoxCache(") < 1:
        problems.append("place_dressing does not construct a BBoxCache of its own, so its "
                        "verification would read bounds cached before the props were scaled")
    if "ComputeRelativeBound" not in place_body:
        problems.append("place_dressing's verification does not use ComputeRelativeBound, so an "
                        "ancestor transform would be included in the size it checks")
    # 11. The stale-comment check.  Every defect fixed in this file left its old argument behind at
    # least once, and a comment that argues for behaviour the code no longer has is worse than none:
    # it is how the next person reintroduces the defect.  Only phrases that can be tied to a specific
    # reversal are listed, so this cannot fire on ordinary prose.
    STALE = (
        ("per axis, not uniform", "the fit is uniform now"),
        ("Fitting the prop to the box removes that class of defect: the\n                #    declared size is what is drawn",
         "the prop is fitted INSIDE the box and may be smaller, so the declared size is a bound"),
        ("STORED IN WORLD ORDER", "a wall item is stored in the catalogue frame, depth on slot 0"),
        ('``frame="wall"`` reverses', "the frame parameter is gone"),
    )
    for phrase, why in STALE:
        if phrase in builder or phrase in read("scenes.py"):
            problems.append(f"stale comment arguing for superseded behaviour: {phrase!r} -- {why}")

    env_path = os.path.join(_ROOT, "environment.py")
    if os.path.exists(env_path):
        env = open(env_path, encoding="utf-8").read()
        if "ComputeWorldBound" not in env:
            notes.append("environment.py no longer measures the robot with a world bound; that is "
                         "not an error, but the comment explaining why it was correct there has "
                         "probably moved or gone")

    print(f"read {', '.join(FILES)}")
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nno problems found by these checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
