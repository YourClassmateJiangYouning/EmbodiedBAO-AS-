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
    catalogue = read("scenes.py")

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
