"""Render the benchmark scene from several diagnostic viewpoints.

The robot's eye camera is what the model sees, but it is a poor tool for
diagnosing whether the scene is *readable*: a view that is mostly a flat grey
wall tells you nothing about where the wall, the opening and the robot are.
This script builds the same scene and captures the same frame from a set of
fixed external cameras so the geometry can be judged directly.

    /home/ybh/isaacsim/python.sh capture_views.py
    /home/ybh/isaacsim/python.sh capture_views.py --level 4 --outdir views

Output (default ``views/``):

    top.png       straight down over the whole 4 m room (floor plan)
    iso.png       behind-right, elevated, looking at the wall from the start side
    front.png     level, from the front, the wall face-on
    behind.png    level, from behind the wall looking back
    eye.png       exactly what the model sees
"""

from __future__ import annotations

import argparse
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
        "--res",
        type=str,
        default="1280x720",
        help="viewport resolution for the external views, e.g. 1280x720",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

    # Python block-buffers stdout when it is a pipe, so every print below would
    # be lost when the process exits if it were piped through tee.  Flush on
    # write and record the measurements to a file as well.
    os.makedirs(args.outdir, exist_ok=True)
    measurements_path = os.path.join(args.outdir, "measurements.txt")
    with open(measurements_path, "w") as handle:
        handle.write("")

    def say(message: str) -> None:
        print(message, flush=True)
        try:
            with open(measurements_path, "a") as handle:
                handle.write(message + "\n")
        except OSError:
            pass

    ok = False
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        from PIL import Image

        import environment

        # --- build the scene ------------------------------------------------
        task_dict = {"headless": True, "camera_resolution": (args.width, args.height)}
        if args.eye_height is not None:
            task_dict["eye_camera_height"] = float(args.eye_height)
        if args.eye_pitch is not None:
            task_dict["eye_pitch_deg"] = float(args.eye_pitch)
        env = environment.BAOEnv(simulation_app, task_dict=task_dict)
        width = environment.level_channel_width(args.level)
        env.set_channel_width(width)
        env.reset_scene()

        try:
            robot = env._root_position()
        except Exception:
            robot = [float("nan")] * 3
        measured = getattr(env, "_robot_measured_height", None)
        pitch_now = float(
            env.task_dict.get("eye_pitch_deg", environment.EYE_PITCH_DEG)
        )
        try:
            eye_height_now = float(env._head_camera_position()[1])
        except Exception:
            eye_height_now = float("nan")

        say(f"[views] level {args.level}: channel {width:.2f} m")
        say(
            f"[views] robot root x={robot[0]:.2f} y={robot[1]:.2f} z={robot[2]:.2f}"
        )
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

        def to_isaac(p):
            return environment._user_to_isaac_pos(np.asarray(p, dtype=float)).tolist()

        def capture(name: str, eye, target, up=(0.0, 0.0, 1.0)) -> None:
            """Point the viewport camera and save one frame."""
            try:
                set_camera_view(
                    eye=eye,
                    target=target,
                    up=list(up),
                    camera_prim_path="/OmniverseKit_Persp",
                )
            except TypeError:
                set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
            # Re-render the shifted view a few times before grabbing it.
            for _ in range(12):
                simulation_app.update()
            rgb = env.camera.get_rgb()
            if rgb is None:
                print(f"[views] {name}: no frame returned")
                return
            path = os.path.join(args.outdir, f"{name}.png")
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
            arr = np.asarray(rgb, dtype=np.uint8)
            print(
                f"[views] {name}.png  mean={arr.mean():.1f}  "
                f"std={arr.std():.1f}  unique_colors={len(np.unique(arr.reshape(-1, 3), axis=0))}"
            )

        # 1. Floor plan: straight down over the room centre.  At 7 m with a
        #    2.5 mm lens the whole 4 m room fits inside the frame.
        capture(
            "top",
            to_isaac([2.0, 7.0, 0.0]),
            to_isaac([2.0, 0.0, 0.0]),
            up=(0.0, 1.0, 0.0),
        )

        # 2. Elevated three-quarter view from behind-right of the robot.
        capture(
            "iso",
            to_isaac([0.3, 2.2, 1.8]),
            to_isaac([2.2, 0.7, 0.2]),
        )

        # 3. Wall face-on from the robot's side, showing the opening.
        capture(
            "front",
            to_isaac([0.2, 1.3, 0.0]),
            to_isaac([2.0, 1.0, 0.0]),
        )

        # 4. From behind the wall looking back through the opening.
        capture(
            "behind",
            to_isaac([3.8, 1.3, 0.0]),
            to_isaac([2.0, 1.0, 0.0]),
        )

        # 5. The robot's own eye camera (what the model is shown), captured at
        #    the start pose and again after stepping toward the channel, since
        #    the view changes a lot with distance to the wall.
        for label, presses in (("eye_start", 0), ("eye_near", 8)):
            if presses:
                for _ in range(presses):
                    env.execute_action("forward")
            try:
                eye_rgb = env.get_camera_image()
                arr = np.asarray(eye_rgb, dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(args.outdir, f"{label}.png"))
                print(
                    f"[views] {label}.png  mean={arr.mean():.1f}  std={arr.std():.1f}  "
                    f"unique_colors={len(np.unique(arr.reshape(-1, 3), axis=0))}"
                )
            except Exception:
                print(f"[views] {label} capture failed:")
                traceback.print_exc()

        print(f"\n[views] wrote {len(os.listdir(args.outdir))} file(s) to {args.outdir}/")
        ok = True
    finally:
        if not ok:
            traceback.print_exc()
        simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
