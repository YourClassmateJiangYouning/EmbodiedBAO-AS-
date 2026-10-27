"""Stage 3 supplementary runner: G1/R1 bodies, A/S 1.1-0.9, Stage 1 setting.

Run one process per robot:
    /home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 --headless
    /home/ybh/isaacsim/python.sh stage3_runner.py --robot R1 --headless

Smoke:
    /home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 \
        --model deepseek-v4-flash-vision-exp --as 1.0 --limit 1 --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import stage3_config
from robot_profiles import as_width, get_profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 body-size supplementary experiment")
    p.add_argument("--robot", choices=sorted(stage3_config.STAGE3_ROBOTS + ("H1",)), required=True)
    p.add_argument("--scene", default=stage3_config.STAGE3_SCENE)
    p.add_argument("--models", default="", help="comma-separated; default Stage 1 top five")
    p.add_argument("--as", dest="as_values", default="", help="comma-separated A/S values")
    p.add_argument("--limit", type=int, default=0, help="limit cells; 0 = all")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    import environment
    import experiments
    import protocol

    profile = environment.apply_robot_profile(args.robot)
    # The prompt must name the body the agent controls, otherwise the model is
    # told it is an H1 while standing in a G1/R1.  Nothing else in the Stage 1
    # wording is changed.
    protocol.TASK_INSTRUCTION = (
        f"You are a {profile.name} humanoid robot. Your task is to reach the red marker on "
        "the far wall."
    )

    models = tuple(x.strip() for x in args.models.split(",") if x.strip()) or stage3_config.STAGE3_MODELS
    as_values = tuple(float(x) for x in args.as_values.split(",") if x.strip()) or stage3_config.STAGE3_AS_VALUES
    task = {
        "headless": args.headless,
        "scene": args.scene,
        "robot_usd_path": os.path.join(ROOT, profile.usd_path),
        "robot_name": profile.name,
        "camera_resolution": (args.image_size, args.image_size),
        "eye_camera_height": profile.eye_height,
        "eye_pitch_deg": 15.0,
        "action_steps": 30,
    }
    print(
        f"[stage3] robot={profile.key} ({profile.name}) scene={args.scene} "
        f"shoulder={profile.shoulder_width:.3f} height={profile.body_height:.3f} usd={task['robot_usd_path']}",
        flush=True,
    )
    env = environment.setup_scene(app, task_dict=task)

    cells = []
    for model in models:
        for asv in as_values:
            width = as_width(profile, asv)
            cells.append((model, asv, width))
    if args.limit:
        cells = cells[: args.limit]

    print(f"[stage3] {len(cells)} cells; expected full design {stage3_config.STAGE3_EXPECTED}", flush=True)
    for index, (model, asv, width) in enumerate(cells, 1):
        env.set_channel_width(width)
        built = env.get_channel_width()
        ratio = env.get_a_s_ratio()
        tag = f"stage3-{profile.key}-{args.scene}-as{asv:.2f}-{model}".replace("/", "-")
        runner = experiments.BAOExperimentRunner(
            env=env,
            model=model,
            max_steps=args.max_steps,
            episodes_per_level=1,
            tag=tag,
            save_obs=False,
            results_root="results",
            logs_root="logs",
        )
        print(
            f"[stage3] {index}/{len(cells)} {model} A/S={asv:.2f} width={width:.4f} "
            f"built={built:.4f} ratio={ratio:.3f}",
            flush=True,
        )
        episode = runner._run_episode(
            level=int(round(asv * 100)),
            episode_id=0,
            channel_width=built,
            ratio=ratio,
        )
        episode["stage3_robot"] = profile.key
        episode["stage3_robot_name"] = profile.name
        episode["stage3_a_s_requested"] = float(asv)
        episode["stage3_shoulder_width"] = float(profile.shoulder_width)
        episode["stage3_scene"] = args.scene
        runner._save_episode(episode)
        print(
            f"[stage3] done {model} A/S={asv:.2f} passed={episode.get('passed')} "
            f"steps={episode.get('total_steps')}",
            flush=True,
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
