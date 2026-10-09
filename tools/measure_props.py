"""Measure each referenced prop against the box the item declares.

Why this exists: place_dressing puts the prop reference on a box of the declared size but does
NOT scale the prop.  The declared size is what the occlusion and walking-band checks are
written against, so it must not move; a prop's own authored size is a fact about the asset.
The two only agree by luck, and nothing on the development machine can measure it -- there is
no pxr here.  So this runs on the lab machine, where "is the pallet 1.2 m across or 12 cm?"
can be answered with numbers instead of a squint.

It does not change anything, and it writes its findings to BOTH stdout and a file, because on
the workstation it once produced no stdout at all: the run took 162 s, printed nothing, and
exited 0.  A file is not at the mercy of Kit's log handling, and every step is marked with a
line beginning "MARK" so a silent failure can be told apart from a silent success.

The file it writes has its OWN name, so redirecting stdout into prop_measurements.txt cannot
have the tool and the shell writing the same file at once.

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


def main() -> int:
    emit("MARK start")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    emit("MARK SimulationApp up")
    try:
        from pxr import Usd, UsdGeom

        import scene_builder as sb
        import scenes as sc

        emit("MARK imports done")
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        targets = []
        for scene in sc.SCENE_ORDER:
            for item in sc.SCENES[scene]["dressing"]:
                local = sb.prop_asset_for(item.get("asset"))
                if local and item["mount"] == "floor":
                    targets.append((scene, item, local))
        emit(f"MARK {len(targets)} target(s): {[t[1] for t in targets]}")
        emit()
        emit(f"{'scene':<9} {'item':<15} {'prop file':<34} {'declared (x,y,z) m':<22} "
             f"{'measured (x,y,z) m':<22} {'ratio x,y,z':<20} verdict")
        emit("-" * 140)

        for scene, item, local in targets:
            # The scene is called stage1.2 and a dot is a PROPERTY separator in SdfPath, so
            # "/World/Probe/stage1.2/klt_bins" is ill-formed and USD rejects it -- measured on
            # the workstation, which is the only place this file runs.  scene_builder.prim_name()
            # exists for exactly this and is reused rather than re-derived.
            path = f"/World/Probe/{sb.prim_name(scene)}/{sb.prim_name(item['name'])}"
            emit(f"MARK defining {path}")
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            prim.GetReferences().AddReference(local)

            box = UsdGeom.BoxCache(UsdGeom.BoxCache.CreateBoxCache(prim))
            rng = box.ComputeWorldBounds().ComputeAlignedRange()
            declared = sc.to_world_size(item["size"])
            if rng.IsEmpty():
                emit(f"{scene:<9} {item['name']:<15} {os.path.basename(local):<34} "
                     f"{str(tuple(round(v, 3) for v in declared)):<22} "
                     f"{'EMPTY':<22} {'-':<20} no geometry -- wrapper or broken layers")
                continue

            lo, hi = rng.GetMin(), rng.GetMax()
            measured = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
            ratios = tuple(
                (measured[i] / declared[i]) if declared[i] else float("nan")
                for i in range(3)
            )
            ok = all(0.8 <= r <= 1.25 for r in ratios if r == r)
            emit(f"{scene:<9} {item['name']:<15} {os.path.basename(local):<34} "
                 f"{str(tuple(round(v, 3) for v in declared)):<22} "
                 f"{str(tuple(round(v, 3) for v in measured)):<22} "
                 f"{str(tuple(round(r, 2) for r in ratios)):<20} "
                 f"{'fits the box' if ok else 'DIFFERS from the box'}")

        emit()
        emit("A ratio near 1.0 means the prop already matches the volume the fixture checks")
        emit("were written against.  A ratio far from 1 means either the prop needs scaling")
        emit("or the declared size needs revisiting -- a decision that needs these numbers.")
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
