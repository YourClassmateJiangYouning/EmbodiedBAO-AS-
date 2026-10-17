#!/usr/bin/env python3
"""Render each Stage 1 scene from behind the robot, so the ROBOT, its feet and the marker are all visible.

WHY THIS EXISTS
    capture_scenes.py's "iso" view points its camera at the far wall, and the environment's own
    third-person camera sits 0.42 m IN FRONT of the body -- so neither one shows the robot. The question
    this answers is "is the robot standing on the floor at the right place, or is it sunk into it", and
    that needs a view from behind and above with the feet in frame.

    The camera pose is re-applied after every world.step because the environment re-poses /World/Camera
    onto the robot on each step; without that the frames come back as the robot's own view.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

SCENES = ("stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5")


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    sys.path.insert(0, os.getcwd())
    import omni.usd
    from pxr import UsdGeom

    import environment

    outdir = sys.argv[1] if len(sys.argv) > 1 else "scene_preview"
    only = sys.argv[2] if len(sys.argv) > 2 else ""
    os.makedirs(outdir, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    results = []

    for scene in ([only] if only else SCENES):
        env = environment.setup_scene(
            app, task_dict={"headless": True, "scene": scene, "level": 10}
        )
        try:
            env.reset_scene(episode_id=0)
            state = env.get_robot_state()
            rx = float(state["position"][0])
            rz = float(state["position"][2])
            # Behind and above the robot, offset to one side so the feet and the floor plane are both
            # in frame. User coordinates: (x, height, lateral).
            cam_pos = (min(rx + 1.4, 15.0), 2.7, 1.7)
            cam_target = (rx + 0.2, 0.55, 0.0)

            camera = env.camera
            from environment import _user_to_isaac_pos

            position = _user_to_isaac_pos(np.array(cam_pos, dtype=float))
            target = _user_to_isaac_pos(np.array(cam_target, dtype=float))
            direction = target - position
            direction = direction / (float(np.linalg.norm(direction)) or 1.0)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(direction, up)
            right = right / (float(np.linalg.norm(right)) or 1.0)
            true_up = np.cross(right, direction)
            matrix = np.stack([direction, -right, true_up], axis=1)
            from capture_scenes import _quaternion_from_matrix

            quat = _quaternion_from_matrix(matrix)

            def repose() -> None:
                camera.set_world_pose(
                    position=position.tolist(), orientation=quat.tolist(), camera_axes="world"
                )

            frames = []
            for _ in range(12):
                env.world.step(render=True)
                repose()
                frames.append(np.asarray(camera.get_rgb())[:, :, :3].astype(np.float64))
            image = (np.mean(frames, axis=0)).astype(np.uint8)

            from capture_scenes import write_png

            path = os.path.join(outdir, f"{scene}_robot.png")
            write_png(path, image)
            lift = getattr(env, "_robot_ground_offset", float("nan"))
            results.append((scene, rx, rz, lift, image.mean()))
            print(
                f"[show] {scene}: robot at x={rx:+.3f} lateral={rz:+.3f} "
                f"ground_offset={lift:+.4f} mean={image.mean():.1f} -> {path}",
                flush=True,
            )
        finally:
            env.close()
    for scene, rx, rz, lift, mean in results:
        print(f"PROBE {scene} x={rx:+.3f} z={rz:+.3f} lift={lift:+.4f} mean={mean:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
