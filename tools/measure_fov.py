"""Measure the eye camera's ACTUAL field of view from the rendered image.

The analytic frustum calculation and the render disagree badly: at start_x=0.5
the ray probe says 32% of the frame should show the gap and 68% should show
wall, while the captured image is 100% green (the far wall through the gap).
Either the field of view is much narrower than computed, or the geometry is not
where the maths thinks.

This settles it without inference by looking straight down at the floor grid,
whose lines are at known world positions.  The grid spacing is 0.5 m, so the
pixel spacing between lines gives metres-per-pixel, and hence the field of view.

    /home/ybh/isaacsim/python.sh tools/measure_fov.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import environment as env

    # Straight down from a known height, so the floor grid is in view.
    height = 1.68
    task = {"headless": True, "eye_pitch_deg": 89.0, "start_x": 2.0}
    e = env.BAOEnv(app, task_dict=task)
    e.reset_scene()
    e._update_eye_camera()
    for _ in range(60):
        app.update()

    print("CAMERA eye pos =", np.round(e._head_camera_position(), 3), flush=True)
    print("FOCAL get_focal_length() =", e.eye_camera.get_focal_length(), flush=True)
    print("RES =", e.eye_camera.get_resolution(), flush=True)

    rgb = np.asarray(e.get_camera_image(), dtype=np.uint8)
    grey = rgb.mean(axis=2)
    print("IMG mean=%.1f std=%.1f" % (grey.mean(), grey.std()), flush=True)

    # The grid lines are dark on a lighter floor; find dark columns/rows.
    col = grey.mean(axis=0)
    row = grey.mean(axis=1)
    print("col profile min/max = %.1f / %.1f" % (col.min(), col.max()), flush=True)

    def dark_positions(profile: np.ndarray, span: float) -> list:
        """Indices of local minima that are clearly darker than the median."""
        med = float(np.median(profile))
        thresh = med - 0.05 * (med - float(profile.min()) + 1e-6) * 3.0
        idx = []
        for i in range(2, len(profile) - 2):
            window = profile[i - 2 : i + 3]
            if profile[i] == window.min() and profile[i] < thresh:
                if not idx or i - idx[-1] > 5:
                    idx.append(i)
        return idx

    dark_cols = dark_positions(col, 1024)
    dark_rows = dark_positions(row, 1024)
    print("dark columns:", dark_cols[:12], flush=True)
    print("dark rows   :", dark_rows[:12], flush=True)

    spacing_m = 0.5  # the grid step authored in _create_ground_grid

    def fov_from(idx: list, label: str) -> None:
        if len(idx) < 2:
            print(f"{label}: not enough lines detected to measure", flush=True)
            return
        gaps = np.diff(idx).astype(float)
        # Use the median gap: it is the most robust to a missed or extra line.
        gap_px = float(np.median(gaps))
        if gap_px <= 0:
            print(f"{label}: degenerate gap", flush=True)
            return
        m_per_px = spacing_m / gap_px
        frame_m = m_per_px * 1024.0
        # The camera looks straight down from `height`, so the frame half-width
        # in metres is half of frame_m and the half-angle follows.
        half_angle = math.degrees(math.atan((frame_m / 2.0) / height))
        print(
            f"{label}: median gap {gap_px:.1f} px per {spacing_m} m -> "
            f"{m_per_px * 1000:.2f} mm/px -> frame is {frame_m:.2f} m tall -> "
            f"MEASURED FOV = {2 * half_angle:.1f} deg",
            flush=True,
        )

    fov_from(dark_cols, "horizontal")
    fov_from(dark_rows, "vertical")

    # Also measure the gap directly: how many pixels wide is the 0.90 m channel
    # when viewed from the front at a known distance?
    print(flush=True)
    print("Now the same camera looking horizontally at the gap:", flush=True)
    app2 = None
    e.eye_camera.set_world_pose(
        position=env._user_to_isaac_pos(np.array([1.0, 1.68, 0.0])).tolist(),
        orientation=[0.9914, 0.0, 0.1305, 0.0],
        camera_axes="world",
    )
    for _ in range(40):
        app.update()
    rgb = np.asarray(e.get_camera_image(), dtype=np.uint8)
    green = (rgb[:, :, 1].astype(int) - rgb[:, :, 0].astype(int)) > 20
    frac_green = float(green.mean())
    print(
        "at 1.0 m from the wall, green (far wall through the gap) occupies "
        f"{100 * frac_green:.1f}% of the frame",
        flush=True,
    )
    print(
        "  analytic prediction at 1.0 m with a 105 deg view: 46%",
        flush=True,
    )
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
