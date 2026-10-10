"""Render the dressing props alone, to see whether the references drew anything.

Why this exists: the eye camera looks through a 0.570 m opening while obstacle-wall dressing is
required to sit at |z| >= 0.9, so that dressing is geometrically invisible from the start pose.
Three attempts to photograph it with the external camera produced a flat grey field and then a
corner of the room with the fixtures cropped at the frame edge -- chasing a camera angle around a
room in which the very thing being checked is behind a wall is the wrong instrument.

So this removes the room.  It builds an empty stage, calls the PRODUCTION code path
(scene_builder.place_dressing) for the scene's dressing, adds one camera and one light, and
renders.  No wall, no opening, no occlusion: what appears is what the reference produced.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/render_dressing.py stage1.5

Writes <outdir>/<scene>_dressing.png and prints, per item, whether the reference resolved and the
world bounding box of what was drawn -- so "the prop is there" is a measurement and not an
impression.
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="scene whose dressing to render, e.g. stage1.5")
    parser.add_argument("--outdir", default="dressing_check")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--only", default="",
                        help="comma-separated item names to keep; empty means all")
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import scene_builder
        import scenes as sc
        from pxr import Usd, UsdGeom, UsdLux

        # A camera and the write_png helper from the capture tool, so the pixels are produced by
        # the same code that produces the previews.
        sys.path.insert(0, _ROOT)
        import capture_scenes as capture

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")

        wanted = [n for n in args.only.split(",") if n]
        if wanted:
            original = sc.SCENES[args.scene]["dressing"]
            sc.SCENES[args.scene]["dressing"] = tuple(
                item for item in original if item["name"] in wanted)
            print(f"[dressing] restricted to {wanted}")

        placed = scene_builder.place_dressing(stage, args.scene)

        print(f"{'item':<16} {'mount':<14} {'reference':<34} drawn world box (x,y,z)")
        print("-" * 104)
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for fixture in placed:
            path = f"/World/Dressing/{fixture['name']}"
            prim = stage.GetPrimAtPath(path)
            extent = "-"
            if prim and prim.IsValid():
                try:
                    rng = box.ComputeWorldBounds(prim).ComputeAlignedRange()
                    if not rng.IsEmpty():
                        lo, hi = rng.GetMin(), rng.GetMax()
                        extent = (f"({hi[0] - lo[0]:.4f}, {hi[1] - lo[1]:.4f}, "
                                  f"{hi[2] - lo[2]:.4f}) at "
                                  f"({(lo[0] + hi[0]) / 2:.2f}, {(lo[1] + hi[1]) / 2:.2f}, "
                                  f"{(lo[2] + hi[2]) / 2:.2f})")
                except Exception as error:  # noqa: BLE001
                    extent = f"bound failed: {type(error).__name__}"
            shown = fixture.get("used_asset") or fixture.get("reference_error") or "-"
            print(f"{fixture['name']:<16} {fixture['mount']:<14} {str(shown)[:32]:<34} {extent}")

        # One dome light: this is a visibility check, not a lighting study.
        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr().Set(1000.0)

        # Frame the dressing: the union of every drawn box, with margin.
        lows, highs = [], []
        for fixture in placed:
            prim = stage.GetPrimAtPath(f"/World/Dressing/{fixture['name']}")
            if not prim or not prim.IsValid():
                continue
            rng = box.ComputeWorldBounds(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            lows.append(np.array(rng.GetMin()))
            highs.append(np.array(rng.GetMax()))
        if not lows:
            print("[dressing] nothing was drawn, so there is nothing to photograph")
            return 1
        low = np.min(np.vstack(lows), axis=0)
        high = np.max(np.vstack(highs), axis=0)
        centre = (low + high) / 2.0
        radius = float(np.max(high - low)) / 2.0
        print(f"[dressing] union box {np.round(high - low, 3).tolist()} centred "
              f"{np.round(centre, 3).tolist()}")

        from isaacsim.sensors.camera import Camera

        height_above = 2.4 * radius + 0.4
        camera_position = (centre[0] + radius * 2.2 + 0.3,
                           height_above,
                           centre[2] - radius * 1.6)
        camera = Camera(prim_path="/World/DressingCamera", frequency=20,
                        resolution=(args.width, args.height))
        camera.initialize()
        capture.look_from(camera, target=tuple(centre), position=camera_position)
        print(f"[dressing] camera at {tuple(round(v, 2) for v in camera_position)} "
              f"looking at {tuple(round(float(v), 2) for v in centre)}")

        # The app drives rendering; the capture tool steps the world instead, and there is no
        # environment here.  Several updates, because one is not enough for the product to have
        # a converged frame in it.
        for _ in range(8):
            app.update()
        image = np.asarray(camera.get_rgb())[:, :, :3]
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, f"{args.scene}_dressing.png")
        capture.write_png(out, image)
        grey = image.astype(float).mean(axis=2)
        print(f"[dressing] wrote {out}  mean {grey.mean():.1f}  std {grey.std():.1f}  "
              f"non-background {(np.abs(grey - np.median(grey)) > 8).mean() * 100:.1f}%")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
