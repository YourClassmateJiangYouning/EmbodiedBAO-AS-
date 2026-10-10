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
REPORT = os.path.join(_ROOT, "named_prop_measurements.txt")


def find(name: str):
    """The vendored .usd whose stem matches, preferring the visual layer over colliders."""
    hits = []
    for folder, _dirs, files in os.walk(ASSETS):
        for candidate in files:
            if candidate.endswith(".usd") and os.path.splitext(candidate)[0] == name:
                hits.append(os.path.join(folder, candidate))
    if not hits:
        return None
    hits.sort(key=lambda p: ("collision" in p, "visual" not in p, len(p)))
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

            # metersPerUnit is reported because the first run of this tool did not check it, and
            # the numbers that came back were impossible: a packing table 247 m long, a
            # corrugated box 18 m.  They are centimetres.  The warehouse props (pallet 1.21 m,
            # cabinet 0.67 m) are metres, so the library is NOT consistent and a size copied out
            # of this tool is meaningless without its unit.
            per_unit = 1.0
            try:
                root = Usd.Stage.Open(path).GetRootLayer()
                per_unit = float(root.metersPerUnit)
            except Exception as error:  # noqa: BLE001
                say(f"    (could not read metersPerUnit: {type(error).__name__}: {error})")

            say(f"{name}: stage (x,lateral,height) = {stage_size}   metersPerUnit={per_unit}")
            say(f"    catalogue (x,height,lateral) = {user}   via {used}")
            if abs(per_unit - 1.0) > 1e-9:
                # What the asset measures once it is expressed in the metres the room is built in.
                metres = tuple(round(v * per_unit, 4) for v in user)
                say(f"    THIS ASSET IS IN {'CENTIMETRES' if abs(per_unit - 0.01) < 1e-9 else str(per_unit)}; "
                    f"its size in metres is {metres}")
                say(f"    a reference must be SCALED by {per_unit} or it draws "
                    f"{1.0 / per_unit:.0f}x too large")
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
