"""Find where the eye camera's focal length gets lost.

The camera reports a 165.5 degree field of view, which corresponds to a 1.336 mm
focal length -- exactly ROBOT_CAMERA_FOCAL/10.  That value must be coming from
somewhere, so this reads the constant, the task dict, the wrapper getter and the
USD attribute, both right after construction and again after reset_scene().

    /home/ybh/isaacsim/python.sh tools/focal_probe.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def out(message: str) -> None:
    print(message, flush=True)


def report(label: str, e) -> None:
    from pxr import UsdGeom

    cam = UsdGeom.Camera(e.stage.GetPrimAtPath("/World/RobotEyeCamera"))
    usd_focal = cam.GetFocalLengthAttr().Get()
    usd_ap = cam.GetHorizontalApertureAttr().Get()
    wrapper = e.eye_camera.get_focal_length()
    if usd_ap:
        fov = 2.0 * math.degrees(math.atan((float(usd_ap) / 2.0) / float(usd_focal)))
    else:
        fov = float("nan")
    out("%-22s USD focalLength=%-8s hAperture=%-8s wrapper=%-8s -> FOV %.1f deg"
        % (label, usd_focal, usd_ap, wrapper, fov))


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import environment as env

    out("=" * 70)
    out("ROBOT_CAMERA_FOCAL constant = %s" % env.ROBOT_CAMERA_FOCAL)
    task = {"headless": True, "start_x": float(env.ROBOT_START_POS[0])}
    out("task_dict                    = %s" % task)
    out("=" * 70)

    e = env.BAOEnv(app, task_dict=dict(task))
    report("after __init__", e)

    e.reset_scene()
    report("after reset_scene", e)

    for _ in range(60):
        app.update()
    report("after 60 updates", e)

    out("")
    out("resolved value from the same expression _create_eye_camera uses:")
    resolved = float(
        e.task_dict.get("robot_camera_focal", e.task_dict.get("camera_focal",
                        env.ROBOT_CAMERA_FOCAL))
    )
    out("  -> %s" % resolved)
    out("task_dict contents after init: %s" % e.task_dict)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
