"""Where does each ray from the head camera land?

Ray-casts the eye camera's frustum against the scene's analytic geometry so the
rendered image can be predicted from first principles.  This is the check that
tells us whether a uniform grey frame is expected (the camera is looking at a
large flat surface) or impossible (something else is wrong).

    python tools/ray_probe.py --start-x 0.5 --pitch 15 --focal 8
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
    parser.add_argument("--start-x", type=float, default=float(env.ROBOT_START_POS[0]))
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
            if (
                channel_width / 2.0 <= abs(p[2]) <= env.ROOM_WIDTH_Z / 2.0
                and 0.0 <= p[1] <= env.WALL_HEIGHT
            ):
                best_t = t
                what = "wall panel"

    # The green far wall closes the room at x = ROOM_LENGTH_X.
    if abs(direction[0]) > 1e-12:
        t = (env.ROOM_LENGTH_X - origin[0]) / direction[0]
        if 1e-6 < t < best_t:
            p = origin + direction * t
            if (
                0.0 <= p[1] <= env.ROOM_WALL_HEIGHT
                and abs(p[2]) <= env.ROOM_WIDTH_Z / 2.0
            ):
                consider(t, "far wall")

    # Floor bounds use independent longitudinal and lateral room dimensions.
    if what == "floor":
        p = origin + direction * best_t
        if (
            p[0] > env.ROOM_LENGTH_X
            or p[0] < 0.0
            or abs(p[2]) > env.ROOM_WIDTH_Z / 2.0
        ):
            what = "floor (outside room)"

    return f"{what} at {best_t:.2f} m"


def classify(origin: np.ndarray, direction: np.ndarray, channel_width: float) -> str:
    """Which surface does this ray hit first, in the horizontal plane?"""
    if abs(direction[0]) < 1e-12:
        return "behind"
    best_t = math.inf
    what = "nothing"

    # Obstacle wall plane at x = WALL_X.
    t = (env.WALL_X - origin[0]) / direction[0]
    if t > 1e-6:
        p = origin + direction * t
        if 0.0 <= p[1] <= env.WALL_HEIGHT:
            if abs(p[2]) < channel_width / 2.0:
                best_t, what = t, "OPENING"
            else:
                best_t, what = t, "wall panel"

    # Far room wall at x = ROOM_LENGTH_X (only reachable through the opening).
    t = (env.ROOM_LENGTH_X - origin[0]) / direction[0]
    if 1e-6 < t < best_t:
        p = origin + direction * t
        if (
            0.0 <= p[1] <= env.ROOM_WALL_HEIGHT
            and abs(p[2]) <= env.ROOM_WIDTH_Z / 2.0
        ):
            best_t, what = t, "far wall"

    # Side walls stay at z = +-ROOM_WIDTH_Z/2 and extend only along x.
    for sign in (-1.0, 1.0):
        if abs(direction[2]) > 1e-12:
            t = (sign * env.ROOM_WIDTH_Z / 2.0 - origin[2]) / direction[2]
            if 1e-6 < t < best_t:
                p = origin + direction * t
                if (
                    0.0 <= p[0] <= env.ROOM_LENGTH_X
                    and 0.0 <= p[1] <= env.ROOM_WALL_HEIGHT
                ):
                    best_t, what = t, "side wall"
    return what


def frame_occupancy(start_x: float, pitch: float, focal: float, height: float,
                    channel: float, sensor: float) -> dict:
    """Horizontal scan: how much of the frame width each surface occupies."""
    fov = 2.0 * math.degrees(math.atan((sensor / 2.0) / focal))
    eye = np.array([start_x, height, 0.0], dtype=float)
    counts: dict = {}
    samples = 400
    for i in range(samples):
        # -1 .. +1 across the frame; z is to the camera's right.
        u = -1.0 + 2.0 * (i + 0.5) / samples
        yaw = math.radians(u * fov / 2.0)
        # Ray in the horizontal plane.  Pitch only affects where on the wall it
        # lands vertically, which matters for the wall/opening test, so cast the
        # ray with the configured pitch.
        elevation = math.radians(-pitch)
        d = np.array(
            [
                math.cos(elevation) * math.cos(yaw),
                math.sin(elevation),
                math.cos(elevation) * math.sin(yaw),
            ]
        )
        what = classify(eye, d, channel)
        counts[what] = counts.get(what, 0) + 1
    return {k: v / samples for k, v in counts.items()}


def report_occupancy(args: argparse.Namespace, channel: float, sensor: float) -> None:
    print()
    print("Frame occupancy (horizontal scan).  'opening+far' is the share of the")
    print("frame showing the gap and what lies beyond it; the rest is wall, which")
    print("is what lets the agent see WHERE the gap is.  Target: under half.")
    print()
    print(
        f"  {'focal':>6} {'start_x':>8} {'dist':>6} "
        f"{'open+far':>9} {'wall':>7} {'steps':>6}"
    )
    budget = 30
    for focal in (args.focal, 12.0, 16.0):
        for start_x in (
            float(env.ROBOT_START_POS[0]),
            env.WALL_X - 3.0,
            env.WALL_X - 1.5,
            env.WALL_X - env.MOVE_STEP,
        ):
            occ = frame_occupancy(
                start_x, args.pitch, focal, args.height, channel, sensor
            )
            travel = max(0.0, env.SUCCESS_X - start_x)
            steps = int(round(90.0 / env.TURN_STEP_DEG)) + int(
                math.ceil(travel / env.MOVE_STEP)
            )
            share = occ.get("OPENING", 0.0) + occ.get("far wall", 0.0)
            flag = "OK " if (share < 0.5 and steps <= budget) else "   "
            print(
                f"  {flag}{focal:>4.1f} {start_x:>8.1f} {env.WALL_X - start_x:>6.2f} "
                f"{100 * share:>8.0f}% {100 * occ.get('wall panel', 0.0):>6.0f}% "
                f"{steps:>6d}"
            )
    print()
    print("  OK = gap under half the frame and the route fits the budget")
    print(f"  note: steps are computed with MOVE_STEP={env.MOVE_STEP} m")


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
    print(
        f"  wall is {env.WALL_HEIGHT:.2f} m tall, "
        f"spans |z| <= {env.ROOM_WIDTH_Z / 2.0:.1f}"
    )

    report_occupancy(args, channel, sensor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
