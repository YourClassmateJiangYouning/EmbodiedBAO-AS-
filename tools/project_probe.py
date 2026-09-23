"""Print where known world points land in the eye camera image.

The rendered eye view and the analytic frustum disagree so badly that neither
the field of view nor the geometry can be trusted by inference alone.  This
projects a set of world points whose positions are already verified (the two
channel posts at z=+-0.45, the far wall, the floor under the camera) through the
camera's reported pose and field of view, and prints the pixel each one should
occupy.

Comparing those pixels against the actual image says which of the two is wrong:

* if the predicted pixels for the posts are inside the frame but the image shows
  no posts there, the render is not matching the projection;
* if the whole frame maps to a tiny angular range, the effective field of view
  is much narrower than measured;
* if everything projects off-frame, the camera is not where its pose claims.

    /home/ybh/isaacsim/python.sh tools/project_probe.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quat_rotate(quat, vec):
    w, x, y, z = (float(v) for v in quat)
    u = np.array([x, y, z], dtype=float)
    v = np.asarray(vec, dtype=float)
    return 2.0 * float(np.dot(u, v)) * u + (w * w - float(np.dot(u, u))) * v + 2.0 * w * np.cross(u, v)


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import environment as env

    e = env.BAOEnv(app, task_dict={"headless": True, "start_x": 0.5})
    e.reset_scene()
    e._update_eye_camera()
    for _ in range(60):
        app.update()

    pos, quat = e.eye_camera.get_world_pose(camera_axes="world")
    focal = e.eye_camera.get_focal_length()
    width, height = e.eye_camera.get_resolution()
    print("ISAAC pose  pos=%s quat=%s" % (np.round(pos, 4), np.round(quat, 4)), flush=True)
    print("USER  eye   =%s" % (np.round(env._isaac_to_user_pos(np.asarray(pos)), 4),), flush=True)
    print("focal=%.3f  res=%dx%d" % (focal, width, height), flush=True)

    # Camera basis in world coordinates (camera_axes="world": +X fwd, +Z up).
    fwd = quat_rotate(quat, [1.0, 0.0, 0.0])
    up = quat_rotate(quat, [0.0, 0.0, 1.0])
    right = quat_rotate(quat, [0.0, -1.0, 0.0])
    print("fwd=%s  right=%s  up=%s" % (np.round(fwd, 4), np.round(right, 4), np.round(up, 4)), flush=True)

    # World points in USER coordinates, converted to the Isaac frame the pose is in.
    targets = {
        "post_left": [env.WALL_X, 1.0, -env.CHANNEL_WIDTH / 2.0],
        "post_right": [env.WALL_X, 1.0, env.CHANNEL_WIDTH / 2.0],
        "gap_centre": [env.WALL_X, 1.0, 0.0],
        "far_wall_centre": [env.SCENE_SIZE, 1.5, 0.0],
        "floor_under_camera": [env.ROBOT_START_POS[0], 0.0, 0.0],
        "wall_panel_left": [env.WALL_X, 1.0, -1.5],
    }

    for name, user_pt in targets.items():
        world = np.asarray(env._user_to_isaac_pos(np.array(user_pt, dtype=float)), dtype=float)
        rel = world - np.asarray(pos, dtype=float)
        x_c = float(np.dot(rel, fwd))
        y_c = float(np.dot(rel, right))
        z_c = float(np.dot(rel, up))
        if x_c <= 1e-6:
            print("%-22s BEHIND the camera (x_c=%.3f)" % (name, x_c), flush=True)
            continue
        # Pinhole with the sensor width implied by the measured 174 deg field.
        for label, fov_deg in (("fov105", 105.0), ("fov174", 174.0)):
            half = math.radians(fov_deg / 2.0)
            # pixel = centre + (y_c / x_c) / tan(half) * (res/2)
            px = width / 2.0 + (y_c / x_c) / math.tan(half) * (width / 2.0)
            py = height / 2.0 - (z_c / x_c) / math.tan(half) * (height / 2.0)
            inside = 0 <= px < width and 0 <= py < height
            print(
                "%-22s dist=%5.2f  %s -> pixel (%7.1f, %7.1f) %s"
                % (name, x_c, label, px, py, "IN" if inside else "off-frame"),
                flush=True,
            )

    rgb = np.asarray(e.get_camera_image(), dtype=np.uint8)
    print(flush=True)
    print("IMAGE centre=%s  corners=%s" % (rgb[512, 512], rgb[5, 5]), flush=True)
    green = (rgb[:, :, 1].astype(int) - rgb[:, :, 0].astype(int)) > 20
    print("green fraction = %.3f" % float(green.mean()), flush=True)
    # Where is the green, if not everywhere?
    rows = green.mean(axis=1)
    bands = [float(rows[i * 128 : (i + 1) * 128].mean()) for i in range(8)]
    print("green by band (top->bottom):", [round(b, 2) for b in bands], flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
