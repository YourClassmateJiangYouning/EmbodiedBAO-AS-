"""Identify the dark horizontal bar visible in the rendered eye view.

The captured views all show a thin black horizontal line across the frame, on
the wall panels, across the opening, and on the ceiling.  No horizontal bar was
authored, so it is either a light artefact or a part of the robot in view.  This
locates it by measurement rather than by eye:

  * which image rows are darkest, converted to a world height using the camera's
    height, pitch and vertical field of view;
  * the height range of every prim in the scene, so whatever occupies that
    height is named.

Every line is flushed, because stdout is block-buffered when redirected and a
run that ends without an explicit flush loses all of it.

    /home/ybh/isaacsim/python.sh tools/bar_probe.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def out(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import environment as env

    e = env.BAOEnv(app, task_dict={"headless": True, "start_x": 0.5})
    e.reset_scene()
    e._update_eye_camera()
    for _ in range(60):
        app.update()

    eye_y = float(e._head_camera_position()[1])
    pitch = float(e.task_dict.get("eye_pitch_deg", env.EYE_PITCH_DEG))
    fov = 2.0 * math.degrees(
        math.atan((20.955 / 2.0) / e.eye_camera.get_focal_length())
    )
    wall_distance = env.WALL_X - float(e._root_position()[0])

    out("=" * 70)
    out("CAMERA  height=%.3f m  pitch=%.1f deg  fov=%.1f deg  "
        "distance to wall=%.2f m" % (eye_y, pitch, fov, wall_distance))
    out("=" * 70)

    rgb = np.asarray(e.get_camera_image(), dtype=np.uint8)
    grey = rgb.astype(float).mean(axis=2)
    h, w = grey.shape
    row_mean = grey.mean(axis=1)

    out("DARKEST ROWS (row, brightness, world height on the wall plane)")
    order = list(np.argsort(row_mean)[:10])
    for row in sorted(order):
        # Row 0 is the top of the frame.  Elevation decreases down the frame.
        frac = (row + 0.5) / h
        # Screen y fraction 0..1 top->bottom maps to elevation
        # +(fov/2) - pitch at the top, down by `fov` across the frame.
        elev = (fov / 2.0 - pitch) - frac * fov
        # Height where that ray meets the wall plane.
        height = eye_y + math.tan(math.radians(elev)) * wall_distance
        out("  row %4d  brightness %6.1f  elevation %+6.1f deg  "
            "wall height %.3f m" % (row, row_mean[row], elev, height))

    out("")
    out("PRIM HEIGHT RANGES (isaac z is up), plus x extent")
    out("  %-24s %10s %10s %10s %10s" % ("name", "z_lo", "z_hi", "x_lo", "x_hi"))
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rows = []
    for prim in e.stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(e.robot_prim_path) and not path.startswith("/World/"):
            continue
        if path.count("/") != 2:
            continue
        rg = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = rg.GetMin(), rg.GetMax()
        if not all(
            math.isfinite(v) for v in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        ):
            continue
        rows.append((prim.GetName(), lo[2], hi[2], lo[0], hi[0]))
    for name, z_lo, z_hi, x_lo, x_hi in sorted(rows, key=lambda r: r[1]):
        out("  %-24s %10.3f %10.3f %10.2f %10.2f" % (name, z_lo, z_hi, x_lo, x_hi))

    out("")
    out("ROBOT PRIMS (in case the bar is part of the robot)")
    robot_rows = []
    for prim in e.stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(e.robot_prim_path):
            continue
        rg = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = rg.GetMin(), rg.GetMax()
        if not all(
            math.isfinite(v) for v in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        ):
            continue
        robot_rows.append((path, lo[2], hi[2], lo[0], hi[0], lo[1], hi[1]))
    robot_rows.sort(key=lambda r: -(r[2] - r[1]))
    for path, z_lo, z_hi, x_lo, x_hi, y_lo, y_hi in robot_rows[:15]:
        out("  %-52s z[%.3f %.3f] x[%.2f %.2f] y[%.2f %.2f]"
            % (path, z_lo, z_hi, x_lo, x_hi, y_lo, y_hi))

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
