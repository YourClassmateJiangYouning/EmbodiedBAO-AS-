"""Measure each referenced prop against the box the item declares.

Why this exists: place_dressing puts the prop reference on a box of the declared size but does
NOT scale the prop.  The declared size is what the occlusion and walking-band checks are
written against, so it must not move; a prop's own authored size is a fact about the asset.
The two only agree by luck, and nothing on the development machine can measure it -- there is
no pxr there.  So this runs on the lab machine, where "is the pallet 1.2 m across or 12 cm?"
can be answered with numbers instead of a squint.

It changes nothing.  It reports, per item, the prop's own size in the layer's units and the
scale that would make it fill the declared box; a scale of 1.0 means it already does.

Two things are deliberate, both learned by running it:

  * it writes to stdout AND to prop_measurement_report.txt.  The first version produced no
    stdout at all -- the run took 162 s, printed nothing, exited 0 -- so a file is kept as a
    second channel, and every step is announced with a line beginning MARK so a silent failure
    can be told from a silent success.  (Running under `python.sh -u` is what showed the
    traceback that found the next fault, so the command below passes it.)
  * it does NOT measure the box.  An earlier version did, and its number was the declaration
    read back to itself, which answers nothing.  What decides the next step is the PROP's size,
    so that is what is printed.

Instanceable assets: small_KLT_visual.usd and friends live under an instanceable model root.
GetPrototype() is used to reach the real geometry when there is one, because a bound computed
on the referencing prim alone would not include the instance proxy's children.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/measure_props.py 2>&1 | tee lab_measure_props.log
    cat prop_measurement_report.txt
"""
from __future__ import annotations

import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

REPORT = os.path.join(_ROOT, "prop_measurement_report.txt")
_lines: list[str] = []


def emit(text: str = "") -> None:
    """Record a line, print it unbuffered, and never let either failure hide the other."""
    _lines.append(text)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def save() -> None:
    try:
        with open(REPORT, "w", encoding="utf-8") as handle:
            handle.write("\n".join(_lines) + "\n")
    except OSError as exc:
        sys.stdout.write(f"could not write {REPORT}: {exc}\n")
        sys.stdout.flush()


def measure(prop_prim, box: object) -> tuple | None:
    """The referenced prop's own axis-aligned size, or None if it has no measurable extent.

    ComputeRelativeBound is used rather than ComputeWorldBound: the parent here is a bare
    /World/Probe xform with no scale, so the relative bound IS the prop's size in the layer's
    units, which is the number to compare with the declared box.
    """
    if prop_prim is None or not prop_prim.IsValid():
        return None
    range_ = box.ComputeRelativeBound(prop_prim, prop_prim.GetParent()).ComputeAlignedRange()
    if range_.IsEmpty():
        return None
    lo, hi = range_.GetMin(), range_.GetMax()
    return (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])


def main() -> int:
    emit("MARK start")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    emit("MARK SimulationApp up")
    try:
        from pxr import Gf, Usd, UsdGeom

        import scene_builder as sb
        import scenes as sc

        emit("MARK imports done")
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        # One cache for the whole run: it is cheap to reuse and expensive to rebuild.
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        targets = []
        for scene in sc.SCENE_ORDER:
            for item in sc.SCENES[scene]["dressing"]:
                local = sb.prop_asset_for(item.get("asset"))
                if local and item["mount"] == "floor":
                    targets.append((scene, item, local))
        emit(f"MARK {len(targets)} target(s): {[t[1] for t in targets]}")
        emit()
        emit(f"{'scene':<9} {'item':<15} {'prop file':<30} {'declared (x,y,z) m':<22} "
             f"{'prop (x,y,z) m':<24} {'scale to fill the box':<24} verdict")
        emit("-" * 150)

        for scene, item, local in targets:
            # The scene is called stage1.2 and a dot is a PROPERTY separator in SdfPath, so
            # "/World/Probe/stage1.2/klt_bins" is ill-formed and USD rejects it -- measured on
            # the workstation, which is the only place this file runs.  scene_builder.prim_name()
            # exists for exactly this and is reused rather than re-derived.
            path = f"/World/Probe/{sb.prim_name(scene)}/{sb.prim_name(item['name'])}"
            emit(f"MARK defining {path}")
            # The probe mirrors place_dressing: the prop is referenced onto an Xform.  It is
            # NOT wrapped in the item's box, because the box contributes nothing to the
            # question -- its size is the declaration, so measuring it would print the
            # declaration back.
            holder = UsdGeom.Xform.Define(stage, path)
            UsdGeom.Xformable(holder).AddTranslateOp().Set(
                Gf.Vec3f(*sc.to_world(item["at"])))
            prop_prim = UsdGeom.Xform.Define(stage, f"{path}/prop").GetPrim()
            prop_prim.GetReferences().AddReference(local)

            emit(f"MARK   prop prim valid: {bool(prop_prim.IsValid())}, "
                 f"instanceable: {prop_prim.IsInstanceable()}")
            prop_size = measure(prop_prim, box)

            # An instanceable model root reports no children of its own, and its bound can come
            # back empty; the prototype is where the geometry actually is.  Try it before
            # declaring the prop unmeasurable, and say which one was used.
            used = "referenced prim"
            if prop_size is None:
                prototype = prop_prim.GetPrototype()
                if prototype and prototype.IsValid():
                    used = "prototype"
                    prop_size = measure(prototype, box)
            emit(f"MARK   measured via {used}")

            declared = sc.to_world_size(item["size"])
            name = os.path.basename(local)
            if prop_size is None:
                emit(f"{scene:<9} {item['name']:<15} {name:<30} "
                     f"{str(tuple(round(v, 3) for v in declared)):<22} "
                     f"{'no measurable extent':<24} "
                     f"{'cannot be scaled from this':<24} PROP UNMEASURABLE")
                continue

            need = tuple(
                (declared[i] / prop_size[i]) if prop_size[i] else float("nan")
                for i in range(3)
            )
            fits = all(0.8 <= r <= 1.25 for r in need if r == r)
            emit(f"{scene:<9} {item['name']:<15} {name:<30} "
                 f"{str(tuple(round(v, 3) for v in declared)):<22} "
                 f"{str(tuple(round(v, 3) for v in prop_size)):<24} "
                 f"{str(tuple(round(v, 3) for v in need)):<24} "
                 f"{'prop fits the box' if fits else 'NEEDS SCALING (or a new declared size)'}")

        emit()
        emit("declared  = the box place_dressing builds, in the stage's Z-up frame")
        emit("prop      = the referenced mesh's own size, in the layer's units")
        emit("scale     = declared / prop: the factor that would make the prop fill the box")
        emit("            1.0 means it already does.  A value far from 1 is a decision:")
        emit("            scale the prop, or change the declared size, or drop the reference.")
        emit("MARK done")
        return 0
    except BaseException:
        emit("MARK EXCEPTION")
        emit(traceback.format_exc())
        return 1
    finally:
        save()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
