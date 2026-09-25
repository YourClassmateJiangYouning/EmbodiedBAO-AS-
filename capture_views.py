"""Render the benchmark scene from several diagnostic viewpoints.

Why this file exists
--------------------
The robot's eye camera is a poor diagnostic: when the scene fails to read it
just reports "a solid grey wall", which says nothing about where the wall, the
opening and the robot actually are.  This script builds the same scene at any
Level and captures it from fixed external cameras so the geometry can be judged
directly.

Implementation note (this was a real bug)
-----------------------------------------
An earlier version moved the *viewport* camera with
``isaacsim.core.utils.viewports.set_camera_view`` and then read
``env.camera.get_rgb()``.  Those are two different objects: the viewport camera
only exists in the GUI, while ``get_rgb`` reads the ``/World/Camera`` sensor
prim.  The sensor never moved, so every "different viewpoint" produced a
byte-identical image.

The fix is to move the sensor that is actually being read, using the documented
``Camera.set_world_pose(position, orientation, camera_axes="world")`` with the
world axes convention ``+X forward, +Z up`` (which matches the project's own
user frame, so no conversion is needed).

Usage:
    /home/ybh/isaacsim/python.sh capture_views.py --level 0
    /home/ybh/isaacsim/python.sh capture_views.py --level 4 --outdir views_L4
    /home/ybh/isaacsim/python.sh capture_views.py --level 0 --eye_height 1.65 --eye_pitch 10

Output (default ``views/``): top.png, iso.png, front.png, behind.png,
eye_start.png, eye_near.png, and measurements.txt with the numeric summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture diagnostic scene views.")
    parser.add_argument("--outdir", type=str, default="views")
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Which A/S level's channel width to build",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--light_intensity",
        type=float,
        default=None,
        help=(
            "Sphere light intensity (default 12000). Exposure is easiest to tune "
            "by rendering: 20000 with eight 1.0 m emitters blew the frame out to "
            "white, while 9000 with four 0.35 m emitters left it at mean 87."
        ),
    )
    parser.add_argument(
        "--light_radius",
        type=float,
        default=None,
        help=(
            "Sphere light radius in metres (default 1.0). Larger radii behave as "
            "area lights: softer falloff and visibly less render noise than the "
            "0.35 m point-like emitters. Prefer raising this over raising "
            "intensity when frames look grainy."
        ),
    )
    parser.add_argument(
        "--dome_intensity",
        type=float,
        default=None,
        help="Ambient dome light intensity (default 300).",
    )
    parser.add_argument(
        "--eye_height",
        type=float,
        default=None,
        help="Head camera height in metres (default: ROBOT_HEAD_HEIGHT)",
    )
    parser.add_argument(
        "--eye_pitch",
        type=float,
        default=None,
        help="Head camera downward pitch in degrees (default: EYE_PITCH_DEG)",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=None,
        help=(
            "Robot eye camera focal length in mm. The inherited default of 1.5 "
            "is roughly a 170-degree fisheye, which renders the channel edges a "
            "few pixels wide; 8.0 is about 44 degrees and much closer to a "
            "person's view."
        ),
    )
    parser.add_argument(
        "--eye_forward_offset",
        type=float,
        default=None,
        help=(
            "Push the eye camera this far forward along the TORSO's facing "
            "direction, in metres (the eye sits on the head, which the torso "
            "carries; only the gaze is pinned to the walking direction). Useful "
            "for testing whether the robot's own body geometry occludes the head "
            "camera."
        ),
    )
    parser.add_argument(
        "--hide_robot",
        action="store_true",
        help=(
            "Hide the robot prim. Comparing the eye view with and without the "
            "robot distinguishes self-occlusion from an unreadable scene."
        ),
    )
    parser.add_argument(
        "--only_eye",
        action="store_true",
        help="Capture only the robot eye view, skipping the external cameras.",
    )
    parser.add_argument(
        "--scan_pitch",
        type=str,
        default="",
        help=(
            "Comma-separated pitch angles to sweep in one run, e.g. '0,10,20,30,45'. "
            "For each angle it reports the strength of vertical edges (the "
            "channel posts) and horizontal edges (the floor grid), which "
            "distinguishes 'looking at a featureless wall' from 'looking at the "
            "floor'."
        ),
    )
    parser.add_argument(
        "--start_x",
        type=float,
        default=None,
        help=(
            "Robot start x in metres (default: ROBOT_START_POS[0]). Standing "
            "further back makes the channel readable, at the cost of travel."
        ),
    )
    parser.add_argument(
        "--move_step",
        type=float,
        default=None,
        help=(
            "Translation per action in metres (default: MOVE_STEP). Must rise "
            "with --start_x or the robot cannot reach the far side in 30 steps."
        ),
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default="{}",
        help='JSON dict merged into the env task dict, e.g. \'{"robot_camera_focal": 8.0}\'',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pure orientation maths -- unit-testable without Isaac Sim
# ---------------------------------------------------------------------------


def look_at_quaternion(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (w, x, y, z) orienting a camera at ``eye`` toward ``target``.

    Built for the documented ``camera_axes="world"`` convention:
    ``+X forward, +Y left, +Z up``.  The rotation matrix columns are the
    camera's local axes expressed in world coordinates, then converted to a
    quaternion.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise ValueError("eye and target coincide; cannot orient the camera")
    forward = forward / norm

    up_world = np.asarray(up, dtype=float)
    up_world = up_world / float(np.linalg.norm(up_world))

    # +Y is the camera's left axis, so left = up x forward.
    left = np.cross(up_world, forward)
    left_norm = float(np.linalg.norm(left))
    if left_norm < 1e-9:
        raise ValueError("up vector is parallel to the view direction")
    left = left / left_norm

    true_up = np.cross(forward, left)
    true_up = true_up / float(np.linalg.norm(true_up))

    # Columns: camera X, Y, Z axes in world coordinates.
    rotation = np.column_stack((forward, left, true_up))
    return _matrix_to_quaternion(rotation)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    m = np.asarray(matrix, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / float(np.linalg.norm(quat))


def edge_profile(arr: np.ndarray) -> tuple:
    """(vertical, horizontal) edge energies of a grey image, normalised.

    A vertical edge (a channel post) shows up as a large column-to-column
    difference; a horizontal edge (a floor grid line) as a large row-to-row
    difference.  Separating them says which kind of surface fills the view.
    """
    grey = np.asarray(arr, dtype=float).mean(axis=2)
    if grey.ndim != 2 or grey.shape[0] < 4 or grey.shape[1] < 4:
        return 0.0, 0.0
    vertical = float(np.abs(np.diff(grey, axis=1)).mean())
    horizontal = float(np.abs(np.diff(grey, axis=0)).mean())
    return vertical, horizontal


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

    # Python block-buffers stdout when it is a pipe, so piping through tee would
    # lose every line below when the process exits.  Flush on write and mirror
    # the report into a file.
    os.makedirs(args.outdir, exist_ok=True)
    measurements_path = os.path.join(args.outdir, "measurements.txt")
    # Explicit UTF-8: the default is the ANSI code page on Windows (cp936 on a
    # Chinese install), which mangles any non-ASCII text in the report.
    with open(measurements_path, "w", encoding="utf-8") as handle:
        handle.write("")

    def say(message: str) -> None:
        print(message, flush=True)
        try:
            with open(measurements_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except OSError:
            pass

    ok = False
    try:
        from PIL import Image

        import environment

        task_dict = {"headless": True, "camera_resolution": (args.width, args.height)}
        if args.eye_height is not None:
            task_dict["eye_camera_height"] = float(args.eye_height)
        if args.eye_pitch is not None:
            task_dict["eye_pitch_deg"] = float(args.eye_pitch)
        if args.focal is not None:
            task_dict["robot_camera_focal"] = float(args.focal)
        if args.start_x is not None:
            task_dict["start_x"] = float(args.start_x)
        if args.move_step is not None:
            task_dict["move_step"] = float(args.move_step)
        if args.eye_forward_offset is not None:
            task_dict["eye_forward_offset"] = float(args.eye_forward_offset)
        if args.light_intensity is not None:
            task_dict["light_intensity"] = float(args.light_intensity)
        if args.light_radius is not None:
            task_dict["light_radius"] = float(args.light_radius)
        if args.dome_intensity is not None:
            task_dict["dome_intensity"] = float(args.dome_intensity)
        if args.hide_robot:
            task_dict["hide_robot"] = True
        try:
            extra = json.loads(args.env_config)
        except ValueError as exc:
            say(f"[views] --env_config is not valid JSON: {exc}")
            extra = {}
        if isinstance(extra, dict):
            task_dict.update(extra)

        env = environment.BAOEnv(simulation_app, task_dict=task_dict)
        width = environment.level_channel_width(args.level)
        env.set_channel_width(width)
        env.reset_scene()

        # ---- report the geometry ------------------------------------------
        try:
            robot = env._root_position()
        except Exception:
            robot = np.array([float("nan")] * 3)
        measured = getattr(env, "_robot_measured_height", None)
        pitch_now = float(env.task_dict.get("eye_pitch_deg", environment.EYE_PITCH_DEG))
        try:
            eye_height_now = float(env._head_camera_position()[1])
        except Exception:
            eye_height_now = float("nan")

        say(f"[views] level {args.level}: channel {width:.2f} m")
        say(f"[views] robot root x={robot[0]:.2f} y={robot[1]:.2f} z={robot[2]:.2f}")
        say(
            f"[views] wall at x={environment.WALL_X}, "
            f"height {environment.WALL_HEIGHT:.2f} m"
        )
        if measured is None:
            say("[views] robot height = n/a (measurement failed)")
        else:
            say(f"[views] robot height = {measured:.3f} m")
            say(
                f"[views] wall/robot height ratio = "
                f"{environment.WALL_HEIGHT / measured:.2f}"
            )
        say(
            f"[views] eye camera: height={eye_height_now:.3f} m, "
            f"pitch={pitch_now:.1f} deg"
        )

        # ---- capture from the sensor that is actually read -----------------
        os.makedirs(args.outdir, exist_ok=True)

        def to_isaac(p):
            return environment._user_to_isaac_pos(np.asarray(p, dtype=float))

        def capture(name: str, eye, target, up=(0.0, 0.0, 1.0)) -> None:
            """Move the /World/Camera sensor to look at the target, then save."""
            eye_isaac = to_isaac(eye)
            target_isaac = to_isaac(target)
            quat = look_at_quaternion(eye_isaac, target_isaac, up=up)
            env.camera.set_world_pose(
                position=eye_isaac.tolist(),
                orientation=quat.tolist(),
                camera_axes="world",
            )
            # Let the render product pick up the new pose before grabbing it.
            for _ in range(20):
                simulation_app.update()
            rgb = env.camera.get_rgb()
            if rgb is None:
                say(f"[views] {name}: no frame returned")
                return
            arr = np.asarray(rgb, dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(args.outdir, f"{name}.png"))
            say(
                f"[views] {name}.png {arr.shape[1]}x{arr.shape[0]} "
                f"mean={arr.mean():.1f} std={arr.std():.1f} "
                f"unique_colors={len(np.unique(arr.reshape(-1, 3), axis=0))}"
            )
            # Per-band profile: shows WHERE structure is, so an unreadable frame
            # can be attributed to a specific part of the view.
            if name.startswith("eye"):
                h = arr.shape[0]
                bands = 8
                parts = []
                for b in range(bands):
                    lo = b * h // bands
                    hi = (b + 1) * h // bands
                    band = arr[lo:hi]
                    parts.append(
                        f"{b}:{band.std():5.1f}/{len(np.unique(band.reshape(-1,3),axis=0)):5d}"
                    )
                say(f"[views] {name} bands (top->bottom) std/colors: " + "  ".join(parts))

        # 1. Floor plan: straight down over the room centre.  NOTE the frame:
        #    to_isaac() swaps y and z, so "looking down" is -Z in the frame this
        #    function receives.  The up hint must therefore be a world axis
        #    perpendicular to -Z, i.e. world +Y.  (Passing (0,0,-1) here is
        #    parallel to the view direction and raises.)
        if not args.only_eye:
            capture(
                "top",
                [environment.ROOM_LENGTH_X / 2.0, 18.0, 0.0],
                [environment.ROOM_LENGTH_X / 2.0, 0.0, 0.0],
                up=(0.0, 1.0, 0.0),
            )

            # 2. Elevated three-quarter view from behind-right of the robot.
            capture(
                "iso",
                [environment.WALL_X - 4.0, 3.2, 2.2],
                [environment.WALL_X, 0.9, 0.0],
            )

            # 3. Wall face-on from the robot's side, showing the opening.
            capture(
                "front",
                [environment.WALL_X - 3.0, 1.3, 0.0],
                [environment.WALL_X, 1.0, 0.0],
            )

            # 4. From behind the wall looking back through the opening.
            capture(
                "behind",
                [environment.WALL_X + 3.0, 1.3, 0.0],
                [environment.WALL_X, 1.0, 0.0],
            )

        # 5. Optional pitch sweep: find the pitch at which the channel becomes
        #    visible, by separating vertical edges (the posts) from horizontal
        #    ones (the floor grid).
        if args.scan_pitch.strip():
            angles = []
            for token in args.scan_pitch.split(","):
                token = token.strip()
                if token:
                    angles.append(float(token))
            say(f"[views] pitch scan: {angles}")
            for pitch in angles:
                env.task_dict["eye_pitch_deg"] = float(pitch)
                env._update_eye_camera()
                for _ in range(20):
                    simulation_app.update()
                arr = np.asarray(env.get_camera_image(), dtype=np.uint8)
                vert, horiz = edge_profile(arr)
                Image.fromarray(arr).save(
                    os.path.join(args.outdir, f"scan_pitch{int(pitch):03d}.png")
                )
                say(
                    f"[views] pitch {pitch:6.1f}  mean={arr.mean():6.1f} "
                    f"std={arr.std():6.1f} "
                    f"unique={len(np.unique(arr.reshape(-1, 3), axis=0)):5d} "
                    f"vert_edge={vert:5.2f}  horiz_edge={horiz:5.2f}"
                )
            # Restore the configured pitch for any later capture.
            env.task_dict["eye_pitch_deg"] = pitch_now
            env._update_eye_camera()

        # 6. The robot's own eye camera at the start pose, then after stepping
        #    toward the channel -- the view changes a lot with distance.
        for label, presses in (("eye_start", 0), ("eye_near", 8)):
            if presses:
                for _ in range(presses):
                    env.execute_action("forward")
            env._update_eye_camera()
            for _ in range(20):
                simulation_app.update()
            try:
                arr = np.asarray(env.get_camera_image(), dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(args.outdir, f"{label}.png"))
                say(
                    f"[views] {label}.png {arr.shape[1]}x{arr.shape[0]} "
                    f"mean={arr.mean():.1f} std={arr.std():.1f} "
                    f"unique_colors={len(np.unique(arr.reshape(-1, 3), axis=0))}"
                )
            except Exception:
                say(f"[views] {label} capture failed:")
                traceback.print_exc()

        say(f"[views] wrote {len(os.listdir(args.outdir))} file(s) to {args.outdir}/")
        ok = True
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
