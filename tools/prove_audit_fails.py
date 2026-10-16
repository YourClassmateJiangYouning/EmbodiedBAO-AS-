"""Prove tools/audit_dressing.py fails on the exact structure that shipped and hid every prop.

The state this project actually had was: the referenced prop a CHILD of the clearance box, and the
box hidden with MakeInvisible().  `invisible` is an inherited property, so that hid the prop too and
every frame stayed bare while the report printed RESOLVED fifteen times.  The audit must reject it.

Each fault below is planted in a THROWAWAY COPY of scene_builder.py under a temporary root that
mirrors this repository, the audit is run against that copy, and the copy is discarded.  The real
scene_builder.py is only ever read, and is hashed before and after so that a change is reported.  An
earlier version wrote each planted fault over the LIVE file and restored it in a `finally`; a SIGKILL,
the OOM killer, or a power cut between the write and the restore left the planted fault in the
repository, and every worker started afterwards imported the broken source.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
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
        "the fit takes a MIN over the ratios again, so only the largest-native axis reaches its "
        "declared box (what shipped: a 14.25 cm pallet drawn 1.7 cm thick)",
        "    scale = tuple(1.0 / native[axis] if native[axis] > 1e-9 else 1.0\n"
        "                  for axis in range(3))",
        "    factor = min(1.0 / native[axis] for axis in range(3) if native[axis] > 1e-9)\n"
        "    scale = (factor, factor, factor)",
    ),
    (
        "the fit caps the factor at 1.0 again, which draws a sub-metre asset at want^2 "
        "(what shipped: a 9.2 cm mug drawn 8.5 mm)",
        "    return (offset, scale)",
        "    scale = tuple(s if s <= 1.0 else 1.0 for s in scale)\n"
        "    return (offset, scale)",
    ),
    (
        "the per-axis normalisation is dropped, so the prop is drawn at its own natural size "
        "instead of filling its declared box",
        "    scale = tuple(1.0 / native[axis] if native[axis] > 1e-9 else 1.0\n"
        "                  for axis in range(3))",
        "    scale = (1.0, 1.0, 1.0)",
    ),
    (
        "the offset is no longer multiplied by the per-axis scale, so a face-anchored prop is "
        "drawn off its anchor",
        "    offset = tuple(-scale[axis] * centre[axis] for axis in range(3))",
        "    offset = tuple(-centre[axis] for axis in range(3))",
    ),
    (
        "the bound is normalised against the largest extent, keeping the proportions but leaving "
        "every axis short of its declared box",
        "    scale = tuple(1.0 / native[axis] if native[axis] > 1e-9 else 1.0\n"
        "                  for axis in range(3))",
        "    _big = max(native[axis] for axis in range(3) if native[axis] > 1e-9)\n"
        "    factor = 1.0 / _big\n"
        "    scale = (factor, factor, factor)",
    ),
    (
        "the prop's offset applied AFTER its scale, which multiplies the offset by the fit factor",
        # The placement code now FORCES the op order instead of relying on two adjacent calls being
        # appended in the right sequence, so the fault has to defeat that.  Swapping the two appends
        # reproduces the real defect: [scale, translate] evaluates as scale*(p + offset) rather than
        # p*scale + offset, and on 2026-10-16 that lifted every prop by 0.5*(1-scale)*height -- half
        # an object for the centimetre assets, which is what floated the racks and the shelves.
        "                if _translate_op is not None:\n"
        "                    _ordered.append(_translate_op)\n"
        "                if _scale_op is not None:\n"
        "                    _ordered.append(_scale_op)",
        "                if _scale_op is not None:\n"
        "                    _ordered.append(_scale_op)\n"
        "                if _translate_op is not None:\n"
        "                    _ordered.append(_translate_op)",
    ),
    (
        "the verification compares only the prop's extent, never its position",
        "        outside = [axis_names[axis] for axis in range(3)\n"
        "                   if float(lo[axis]) < box_lo[axis] - 1e-3\n"
        "                   or float(hi[axis]) > box_hi[axis] + 1e-3]",
        "        outside = []",
    ),
]

_TRACEBACK = "Traceback (most recent call last)"


def build_sandbox() -> pathlib.Path:
    """A temporary repository root that owns its own scene_builder.py.

    tools/audit_dressing.py derives the repository root from its own __file__ (two parents up), so a
    COPY of the audit inside the sandbox reads the sandbox's scene_builder.py.  It is copied rather
    than symlinked precisely so that no symlink resolution can lead it back to the real file.  Every
    other entry is a symlink to the real repository: nothing heavy is copied, and the audit's other
    checks (scenes.py, environment.py, capture_scenes.py, tools/render_dressing.py) still read the real
    files they expect.
    """
    sandbox = pathlib.Path(tempfile.mkdtemp(prefix="prove_audit_fails_"))
    for entry in ROOT.iterdir():
        if entry.name in ("scene_builder.py", "tools", "__pycache__"):
            continue
        os.symlink(entry, sandbox / entry.name)
    tools = sandbox / "tools"
    tools.mkdir()
    shutil.copy2(AUDIT, tools / AUDIT.name)
    for entry in (ROOT / "tools").iterdir():
        if entry.name in (AUDIT.name, "__pycache__"):
            continue
        os.symlink(entry, tools / entry.name)
    return sandbox


def run_audit(sandbox: pathlib.Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(sandbox / "tools" / AUDIT.name)],
        capture_output=True,
        text=True,
        cwd=str(sandbox),
        env=env,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def reported_problems(output: str) -> list:
    """The audit's own named problems, i.e. its `  - <text>` lines.

    A non-zero exit on its own is NOT a detection: an unrelated crash -- an ImportError, or a
    SyntaxError from a fault planted badly -- also exits non-zero, and the old
    `status = "CAUGHT" if code != 0 else "MISSED"` counted it as one.  Only the baseline run guarded
    against that.  The audit prints `N problem(s):` and then one `  - ...` line per problem, so a
    detection must produce at least one such line; a crash produces a traceback and no problem lines.
    """
    return [line.strip()[2:].strip() for line in output.splitlines() if line.startswith("  - ")]


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    before = TARGET.read_bytes()
    failures = []

    sandbox = build_sandbox()
    sandbox_target = sandbox / "scene_builder.py"
    try:
        sandbox_target.write_text(original, encoding="utf-8")
        code, output = run_audit(sandbox)
        print(f"baseline: exit {code}  (audit root {sandbox})")
        if code != 0 or reported_problems(output):
            print(output)
            print("BASELINE IS NOT CLEAN -- fix that before trusting the planted runs")
            failures.append("the audit is not clean on the unmodified source")
        else:
            for label, old, new in FAULTS:
                if old not in original:
                    failures.append(f"{label}: could not plant, pattern not found: {old!r}")
                    continue
                # The fault goes into the COPY.  The real file is not opened for writing anywhere in
                # this process, so no kill between these two lines can leave the repository holding it.
                sandbox_target.write_text(original.replace(old, new, 1), encoding="utf-8")
                code, output = run_audit(sandbox)
                problems = reported_problems(output)
                if code != 0 and problems:
                    status = "CAUGHT"
                elif _TRACEBACK in output:
                    status = "CRASH"
                    failures.append(f"{label}: the audit crashed instead of reporting a problem")
                else:
                    status = "MISSED"
                    failures.append(label)
                print(f"{status:7} {label}  ({len(problems)} problem(s))")
                for problem in problems:
                    print(f"          - {problem}")
                if status == "CRASH":
                    print(output)

            sandbox_target.write_text(original, encoding="utf-8")
            code, output = run_audit(sandbox)
            print(f"restored: exit {code}")
            if code != 0 or reported_problems(output):
                failures.append("the audit is not clean after restoring the source")
                print(output)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    # DIAGNOSTIC, deliberately NOT a failure.  This workspace is edited by several processes at once,
    # so a byte change during the run is far more likely to be a concurrent editor than this tool --
    # and counting it as a failure would mask the fault results the tool exists to report.  The tool
    # has no write path to TARGET at all: every planted fault goes to the sandbox copy.
    after = TARGET.read_bytes()
    if after != before:
        print(f"WARNING: {TARGET} changed while this ran.  This tool never writes it, so check "
              f"`git diff scene_builder.py` for a concurrent editor.")
    else:
        digest = hashlib.sha256(before).hexdigest()[:12]
        print(f"real scene_builder.py: unchanged by this run (sha256 {digest})")

    if failures:
        print(f"\n{len(failures)} fault(s) were not caught:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nall {len(FAULTS)} planted fault(s) were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
