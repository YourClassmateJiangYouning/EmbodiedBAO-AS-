"""Read the authored mesh points of each box, per axis.

Bounding boxes have been misleading throughout this project, so this reads the
geometry that was actually authored: for each box prim it takes the mesh's
`points` attribute and reports the min/max on each axis.  That is the ground
truth about what shape exists in the scene.

    /home/ybh/isaacsim/python.sh tools/points_probe.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def out(message: str) -> None:
    print(message, flush=True)


NAMES = [
    "WallPanel_0",
    "ChannelEdge_0",
    "ChannelEdge_1",
    "room_far",
    "room_side_left",
    "room_ceiling",
    "Ground",
    "Grid_x_0",
    "Grid_z_0",
]


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import environment as env
    from pxr import UsdGeom

    e = env.BAOEnv(app, task_dict={"headless": True, "start_x": 0.5})
    e.reset_scene()
    for _ in range(30):
        app.update()

    out("expected (user frame, authored):")
    out("  ChannelEdge  x=%.3f  height=%.3f  left_right=%.3f" % (
        env.WALL_THICKNESS * 2.5, env.WALL_HEIGHT, env.CHANNEL_EDGE_THICKNESS))
    out("  WallPanel    x=%.3f  height=%.3f  left_right=%.3f" % (
        env.WALL_THICKNESS, env.WALL_HEIGHT, 2.05))
    out("")
    out("  %-18s %-28s %-28s %-28s" % ("prim", "isaac x span", "isaac y span", "isaac z span"))
    for name in NAMES:
        prim = e.stage.GetPrimAtPath(f"/World/{name}")
        if not prim or not prim.IsValid():
            out("  %-18s MISSING" % name)
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get() if mesh else None
        if not pts:
            # fall back to the xform's own extent
            geo = UsdGeom.Boundable(prim)
            pts = None
        if pts:
            arr = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in pts])
            spans = arr.max(axis=0) - arr.min(axis=0)
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)
            out("  %-18s x[%.3f..%.3f]=%.3f  y[%.3f..%.3f]=%.3f  z[%.3f..%.3f]=%.3f"
                % (name, lo[0], hi[0], spans[0],
                   lo[1], hi[1], spans[1], lo[2], hi[2], spans[2]))
        else:
            out("  %-18s no points attribute (not a mesh)" % name)

    out("")
    out("camera and robot:")
    p, q = e.eye_camera.get_world_pose(camera_axes="world")
    out("  eye camera isaac pos = %s" % np.round(np.asarray(p), 3))
    out("  robot root isaac pos = %s"
        % np.round(np.asarray(e.robot_root.get_world_poses()[0][0]), 3))
    out("  robot root USER pos  = %s" % np.round(e._root_position(), 3))
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
