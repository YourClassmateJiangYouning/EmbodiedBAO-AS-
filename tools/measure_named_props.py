"""Measure the named vendored props and print what scenes.py needs.

tools/measure_assets.py measures EVERY vendored mesh, which is right for a full audit and wrong
when what is wanted is four lines to paste.  This measures the props named on the command line
and prints the source lines to write.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/measure_named_props.py SM_Mug_A2 011_banana forklift

The measurement follows tools/measure_assets.py exactly -- reference the asset into a fresh
in-memory stage, bound it, fall back to the prototype when the root is instanceable, and convert
with scenes.to_user -- because that path is the one already verified on this project.  A size
derived independently here would be a second opinion about the same question, and the two would
eventually disagree.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

ASSETS = os.path.join(_ROOT, "assets", "isaac")
# The vegetation and the outdoor furniture are vendored under assets/bucket instead, because they
# are not under Assets/Isaac in NVIDIA's bucket.  Both roots are searched by basename.
ASSET_ROOTS = (ASSETS, os.path.join(_ROOT, "assets", "bucket"))
REPORT = os.path.join(_ROOT, "named_prop_measurements.txt")


def find(name: str):
    """The vendored .usd whose stem matches, preferring the callable layer over colliders."""
    hits = []
    for root in ASSET_ROOTS:
        for folder, _dirs, files in os.walk(root):
            for candidate in files:
                if candidate.endswith(".usd") and os.path.splitext(candidate)[0] == name:
                    hits.append(os.path.join(folder, candidate))
    if not hits:
        return None
    # `_base` before `_inst`: the inst layer is a proxy whose geometry lives in the base layer.
    hits.sort(key=lambda p: ("collision" in p, "_inst" in p, "visual" not in p, len(p)))
    return hits[0]


def main() -> int:
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        return 2

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom

        import scenes as sc

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        lines = []

        def say(text: str = "") -> None:
            lines.append(text)
            print(text, flush=True)

        say(f"# named prop measurements ({len(names)} requested)")
        say()
        for index, name in enumerate(names):
            path = find(name)
            if path is None:
                say(f"{name}: NOT VENDORED -- fetch it first (tools/fetch_assets.py --set ...)")
                say()
                continue
            rel = os.path.relpath(path, ASSETS)
            prim_path = f"/World/A{index}/" + "".join(c if c.isalnum() else "_" for c in rel)
            holder = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
            holder.GetReferences().AddReference(path)

            measured, used = None, "referenced prim"
            if holder.IsValid():
                rng = box.ComputeRelativeBound(holder, holder.GetParent()).ComputeAlignedRange()
                if not rng.IsEmpty():
                    lo, hi = rng.GetMin(), rng.GetMax()
                    measured = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
            if measured is None:
                prototype = holder.GetPrototype()
                if prototype and prototype.IsValid():
                    used = "prototype"
                    rng = box.ComputeRelativeBound(
                        prototype, prototype.GetParent()).ComputeAlignedRange()
                    if not rng.IsEmpty():
                        lo, hi = rng.GetMin(), rng.GetMax()
                        measured = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])

            if measured is None:
                say(f"{name}: UNMEASURABLE (via {used}) -- {rel}")
                say()
                continue

            stage_size = tuple(round(v, 4) for v in measured)
            # to_user, not a "to_user_size": scenes.py defines to_user for a point and
            # to_world_size for a size, and the size conversion is the same two-component swap.
            user = tuple(round(v, 4) for v in sc.to_user(measured))

            # metersPerUnit is read so a size is never copied out of here without its unit, because
            # the library is NOT consistent: the packing table measures 247 and the mugs 0.09.
            # Only stage metadata is used.  The first attempt read a metersPerUnit attribute off the
            # root Layer, which does not exist, and the failure was swallowed by a broad except so
            # every asset silently reported 1.0.  A second attempt reached for a UsdGeom helper the
            # allowlisted-API guard has not verified, and the guard rejected it -- so the size is
            # left in whatever unit the asset authored it in, and an implausible number is called
            # out below instead of being trusted.
            per_unit = 1.0
            try:
                reported = Usd.Stage.Open(path).GetMetadata("metersPerUnit")
                if reported:
                    per_unit = float(reported)
            except Exception as error:  # noqa: BLE001
                say(f"    (stage metadata metersPerUnit unreadable: {type(error).__name__}: {error})")

            say(f"{name}: stage (x,lateral,height) = {stage_size}   metersPerUnit={per_unit}")
            say(f"    catalogue (x,height,lateral) = {user}   via {used}")
            largest = max(user)
            if largest > 10.0:
                say(f"    WARNING: {largest:.1f} is not a plausible size in metres for a prop.  "
                    f"These assets are very likely authored in centimetres with no metersPerUnit "
                    f"metadata, so DIVIDE BY 100: "
                    f"{tuple(round(v / 100.0, 4) for v in user)}")
            say(f"    source: {rel}")
            say(f'    scenes.PROP_MEASUREMENTS["{os.path.basename(path)}"] = {user}')
            say()

        with open(REPORT, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"[written] {REPORT}")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
