"""Prove tools/audit_dressing.py fails on the exact structure that shipped and hid every prop.

The state this project actually had was: the referenced prop a CHILD of the clearance box, and the
box hidden with MakeInvisible().  `invisible` is an inherited property, so that hid the prop too and
every frame stayed bare while the report printed RESOLVED fifteen times.  The audit must reject it.

Each fault below is planted by rewriting the source, the audit is run, and the source is restored.
"""
from __future__ import annotations

import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "scene_builder.py"
AUDIT = ROOT / "tools" / "audit_dressing.py"

FAULTS = [
    (
        "prop parented under the clearance box (what shipped, with MakeInvisible)",
        'box_path = f"/World/Dressing/{item[\'name\']}/clearance"',
        "box_path = f\"/World/Dressing/{item['name']}\"",
    ),
    (
        "the item Xform points at the box instead of the item, so nothing carries the scale",
        "item_xform = UsdGeom.Xform.Define(stage, path)",
        "item_xform = UsdGeom.Xform.Define(stage, box_path)",
    ),
    (
        "the clearance box scaled instead of the item Xform",
        "item_xform.AddScaleOp().Set(Gf.Vec3f(*size))",
        "cube.AddScaleOp().Set(Gf.Vec3f(*size))",
    ),
    (
        "the item's translate added after its scale",
        "        UsdGeom.Xformable(item_prim).AddTranslateOp().Set(Gf.Vec3d(*sc.to_world(item[\"at\"])))\n"
        "        item_xform.AddScaleOp().Set(Gf.Vec3f(*size))",
        "        item_xform.AddScaleOp().Set(Gf.Vec3f(*size))\n"
        "        UsdGeom.Xformable(item_prim).AddTranslateOp().Set(Gf.Vec3d(*sc.to_world(item[\"at\"])))",
    ),
    (
        "the fit multiplies by the declared size again (declared/native), which double counts it",
        "    ratios = [1.0 / native[axis] for axis in range(3) if native[axis] > 1e-9]",
        "    want = tuple(float(v) for v in declared)\n"
        "    ratios = [want[axis] / native[axis] for axis in range(3) if native[axis] > 1e-9]",
    ),
    (
        "the fit is per axis instead of uniform",
        "    return (offset, (factor, factor, factor))",
        "    return (offset,\n"
        "            tuple(want[axis] / native[axis] if native[axis] > 1e-9 else 1.0\n"
        "                  for axis in range(3)))",
    ),
    (
        "the fit grows the prop when its declared box is larger than the asset",
        "    if factor > 1.0:\n        factor = 1.0",
        "    pass",
    ),
    (
        "the prop is not offset onto the anchor, so a face-anchored asset is drawn outward",
        "    return (offset, (factor, factor, factor))",
        "    return ((0.0, 0.0, 0.0), (factor, factor, factor))",
    ),
    (
        "the prop's centre left in the asset's own units instead of the parent's",
        "    offset = tuple(-centre[axis] / native[axis] if native[axis] > 1e-9 else 0.0\n"
        "                   for axis in range(3))",
        "    offset = tuple(-centre[axis] if native[axis] > 1e-9 else 0.0\n"
        "                   for axis in range(3))",
    ),
    (
        "the prop's offset applied AFTER its scale, which multiplies the offset by the fit factor",
        "                    apply_translate(proxy_prim, offset)\n"
        "                if any(abs(s - 1.0) > 1e-9 for s in scale):",
        "                    apply_scale(proxy_prim, scale)\n"
        "                if any(abs(s - 1.0) > 1e-9 for s in scale):",
    ),
    (
        "the verification compares only the prop's extent, never its position",
        "        outside = [axis_names[axis] for axis in range(3)\n"
        "                   if float(lo[axis]) < box_lo[axis] - 1e-3\n"
        "                   or float(hi[axis]) > box_hi[axis] + 1e-3]",
        "        outside = []",
    ),
]


def run_audit() -> tuple[int, str]:
    result = subprocess.run([sys.executable, str(AUDIT)], capture_output=True, text=True, cwd=ROOT)
    return result.returncode, (result.stdout + result.stderr).strip()


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    failures = []

    code, output = run_audit()
    print(f"baseline: exit {code}")
    if code != 0:
        print(output)
        print("BASELINE IS NOT CLEAN -- fix that before trusting the planted runs")
        return 1

    for label, old, new in FAULTS:
        if old not in original:
            failures.append(f"{label}: could not plant, pattern not found: {old!r}")
            continue
        TARGET.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            code, output = run_audit()
        finally:
            TARGET.write_text(original, encoding="utf-8")
        status = "CAUGHT" if code != 0 else "MISSED"
        print(f"{status:7} {label}")
        if code == 0:
            failures.append(label)
        else:
            for line in output.splitlines():
                if line.strip().startswith("- "):
                    print(f"          {line.strip()}")

    code, output = run_audit()
    print(f"restored: exit {code}")
    if code != 0:
        failures.append("the audit is not clean after restoring the source")
        print(output)

    if failures:
        print(f"\n{len(failures)} fault(s) were not caught:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nall {len(FAULTS)} planted fault(s) were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
