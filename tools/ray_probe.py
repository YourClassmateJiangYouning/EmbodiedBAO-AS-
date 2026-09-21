"""Where does each ray from the head camera land?

Ray-casts the eye camera's frustum against the scene's analytic geometry so the
rendered image can be predicted from first principles.  This is the check that
tells us whether a uniform grey frame is expected (the camera is looking at a
large flat surface) or impossible (something else is wrong).

    python tools/ray_probe.py --start-x 1.0 --pitch 15 --focal 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import environment as env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic ray probe for the eye camera.")
    parser.add_argument("--start-x", type=float, default=1.0)
    parser.add_argument("--pitch", type=float, default=env.EYE_PITCH_DEG)
    parser.add_argument("--focal", type=float, default=env.ROBOT_CAMERA_FOCAL)
    parser.add_argument("--height", type=float, default=env.EYE_CAMERA_HEIGHT)
    parser.add_argument("--sensor-width", type=float, default=20.955)
    return parser.parse_args()


def ray_hit(origin: np.ndarray, direction: np.ndarray, channel_width: float) -> str:
    """First scene surface hit along the ray, by analytic intersection."""
    best_t = math.inf
    what = "background"

    def consider(t: float, label: str) -> None:
        nonlocal best_t, what
        if 1e-6 < t < best_t:
            best_t = t
            what = label

    # Ground plane y = 0.
    if abs(direction[1]) > 1e-12:
        t = -origin[1] / direction[1]
        consider(t, "floor")

    # The two wall panels at x = WALL_X, outside |z| < channel/2.
    if abs(direction[0]) > 1e-12:
        t = (env.WALL_X - origin[0]) / direction[0]
        if 1e-6 < t < best_t:
            p = origin + direction * t
            if abs(p[2]) >= channel_width / 2.0 and 0.0 <= p[1] <= env.WALL_HEIGHT:
                best_t = t
                what = "wall panel"

    # The far wall of the room at x = SCENE_SIZE (there is none authored, but
    # the ground ends there; report what is actually beyond it).
    if abs(direction[0]) > 1e-12:
        t = (env.SCENE_SIZE - origin[0]) / direction[0]
        consider(t, "far room edge")

    # Far room floor is the same plane; note the ground ends at x = SCENE_SIZE.
    if what == "floor":
        p = origin + direction * best_t
        if p[0] > env.SCENE_SIZE or p[0] < 0.0 or abs(p[2]) > env.SCENE_SIZE / 2.0:
            what = "floor (outside room)"

    return f"{what} at {best_t:.2f} m"


def main() -> int:
    args = parse_args()
    channel = env.LEVEL_CHANNEL_WIDTHS[0]
    sensor = args.sensor_width
    fov_h = 2.0 * math.degrees(math.atan((sensor / 2.0) / args.focal))

    eye = np.array([args.start_x, args.height, 0.0], dtype=float)
    print(f"camera at {np.round(eye, 3)}   pitch {args.pitch} deg down")
    print(f"focal {args.focal} mm on {sensor} mm sensor -> {fov_h:.1f} deg FOV")
    print(f"wall at x={env.WALL_X}, height {env.WALL_HEIGHT:.2f}, channel {channel:.2f} m")
    print()

    half = math.radians(fov_h / 2.0)
    # Pitch is measured DOWNWARD from horizontal.  u is the fractional position
    # in the frame: +1 is the top edge, -1 the bottom.  A ray above the optical
    # axis has positive elevation, so elevation = -(pitch) + u*half.
    for name, u in (
        ("top of frame    ", 1.0),
        ("upper third     ", 1.0 / 3.0),
        ("centre          ", 0.0),
        ("lower third     ", -1.0 / 3.0),
        ("bottom of frame ", -1.0),
    ):
        elevation = math.radians(-args.pitch) + u * half
        direction = np.array([math.cos(elevation), math.sin(elevation), 0.0])
        print(
            f"{name} elev {math.degrees(elevation):+6.1f} deg -> "
            f"{ray_hit(eye, direction, channel)}"
        )

    print()
    print("Horizontal field at the wall plane:")
    dist = env.WALL_X - args.start_x
    print(f"  visible height there = {2.0 * dist * math.tan(half):.2f} m")
    print(f"  visible width  there = {2.0 * dist * math.tan(half):.2f} m")
    print(f"  wall is {env.WALL_HEIGHT:.2f} m tall, spans |z| <= {env.SCENE_SIZE / 2:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
