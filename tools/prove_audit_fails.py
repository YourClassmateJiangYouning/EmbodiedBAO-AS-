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
        "the translate added after the scale",
        "UsdGeom.Xformable(item_prim).AddTranslateOp().Set(Gf.Vec3d(*sc.to_world(item[\"at\"])))",
        "pass",
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
