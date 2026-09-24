"""Measure the eye camera's field of view robustly, and read its intrinsics.

The previous attempt picked dark columns with a loose threshold and reported a
median spacing of 8 px, which is equally consistent with the floor grid at a
174-degree view and with ordinary render noise.  Because the conclusion hinged
on it, this version does not threshold at all:

  * it reads the camera's own projection matrix, which is authoritative;
  * it autocorrelates the image along rows and columns, so the grid period is
    found from the whole signal rather than from picked peaks;
  * it verifies `set_focal_length` by asking the camera to report its field of
    view before and after being given a new focal length.

Any one of these alone can be wrong; together they have to agree.

    /home/ybh/isaacsim/python.sh tools/measure_fov.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def grid_period(profile: np.ndarray) -> float:
    """Dominant spatial period of a 1-D profile, by autocorrelation."""
    x = np.asarray(profile, dtype=float)
    x = x - x.mean()
    if x.size < 8 or float(np.dot(x, x)) < 1e-9:
        return float("nan")
    ac = np.correlate(x, x, mode="full")[x.size - 1 :]
    ac = ac / ac[0]
    # First strong local maximum after the initial descent.
    best_lag, best_val = -1, -1.0
    for lag in range(3, min(len(ac) - 1, 400)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > best_val:
            best_val, best_lag = float(ac[lag]), lag
            break
    return float(best_lag) if best_lag > 0 else float("nan")


def camera_fov(cam) -> float:
    """Field of view from the camera's own intrinsics, if it exposes them."""
    for name in ("get_intrinsics_matrix", "get_intrinsic_matrix"):
        fn = getattr(cam, name, None)
        if not callable(fn):
            continue
        try:
            k = np.asarray(fn(), dtype=float)
            fx = float(k[0, 0])
            w, _ = cam.get_resolution()
            if fx > 0:
                return 2.0 * math.degrees(math.atan((w / 2.0) / fx))
        except Exception as exc:
            print("  %s failed: %s" % (name, exc), flush=True)
    return float("nan")


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import environment as env

    height = 1.68
    e = env.BAOEnv(
        app,
        task_dict={
            "headless": True,
            "eye_pitch_deg": 89.0,
            "start_x": float(env.ROBOT_START_POS[0]),
        }
    )
    e.reset_scene()
    e._update_eye_camera()
    for _ in range(60):
        app.update()

    print("=" * 68, flush=True)
    print("1. What the camera reports about itself", flush=True)
    print("   get_focal_length() =", e.eye_camera.get_focal_length(), flush=True)
    print("   USD focalLength    =", end=" ", flush=True)
    try:
        from pxr import UsdGeom

        cam = UsdGeom.Camera(e.stage.GetPrimAtPath("/World/RobotEyeCamera"))
        print(cam.GetFocalLengthAttr().Get(), flush=True)
        print("   USD hAperture      =", cam.GetHorizontalApertureAttr().Get(), flush=True)
        print("   USD clippingRange  =", cam.GetClippingRangeAttr().Get(), flush=True)
    except Exception as exc:
        print("failed:", exc, flush=True)
    print("   intrinsics FOV     = %.1f deg" % camera_fov(e.eye_camera), flush=True)

    print("=" * 68, flush=True)
    print("2. Grid period by autocorrelation (straight down, 0.5 m grid)", flush=True)
    rgb = np.asarray(e.get_camera_image(), dtype=np.uint8)
    grey = rgb.astype(float).mean(axis=2)
    print("   image mean=%.1f std=%.1f" % (grey.mean(), grey.std()), flush=True)
    col_period = grid_period(grey.mean(axis=0))
    row_period = grid_period(grey.mean(axis=1))
    print("   column period = %.1f px ; row period = %.1f px" % (col_period, row_period), flush=True)
    spacing_m = 0.5
    for label, period in (("horizontal", col_period), ("vertical", row_period)):
        if not math.isfinite(period) or period <= 1:
            print("   %s: no period found" % label, flush=True)
            continue
        m_per_px = spacing_m / period
        frame_m = m_per_px * 1024.0
        fov = 2.0 * math.degrees(math.atan((frame_m / 2.0) / height))
        print(
            "   %s: %.1f px per %.1f m -> frame %.2f m across at %.2f m drop "
            "-> FOV %.1f deg" % (label, period, spacing_m, frame_m, height, fov),
            flush=True,
        )

    print("=" * 68, flush=True)
    print("3. Does changing the focal length change anything?", flush=True)
    before = camera_fov(e.eye_camera)
    e.eye_camera.set_focal_length(30.0)
    for _ in range(30):
        app.update()
    after = camera_fov(e.eye_camera)
    print("   intrinsics FOV before=%.1f  after setting 30 mm=%.1f" % (before, after), flush=True)
    rgb2 = np.asarray(e.get_camera_image(), dtype=np.uint8)
    print(
        "   image changed: mean %.1f -> %.1f  (identical pixels: %s)"
        % (
            grey.mean(),
            rgb2.astype(float).mean(),
            bool(np.array_equal(rgb, rgb2)),
        ),
        flush=True,
    )
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
