"""Stage 3 supplementary runner: G1/R1 bodies, source-matched Stage 1 episodes.

Each row of reports/stage3_pair_plan.csv is one experiment:
  * Stage 1 source: model, scene, marker, A/S
  * Stage 3 target: G1 or R1, same scene and marker
  * A/S is recomputed against the target robot's shoulder width

Run one source scene per process:
    /home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 --scene stage1.1 --headless
    /home/ybh/isaacsim/python.sh stage3_runner.py --robot R1 --scene stage1.1 --headless

Smoke one cell:
    /home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 --scene stage1.1 \
        --model deepseek-v4-flash-vision-exp --as 1.1 --limit 1 --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import stage3_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 body-size supplementary experiment")
    p.add_argument("--robot", choices=("G1", "R1"), required=True)
    p.add_argument("--scene", default="stage1.1", choices=("stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5"))
    p.add_argument("--plan", default=os.path.join("reports", "stage3_pair_plan.csv"))
    p.add_argument("--model", default="", help="optional filter")
    p.add_argument("--as", dest="as_value", type=float, default=None, help="optional target A/S filter")
    p.add_argument("--limit", type=int, default=0, help="limit rows; 0 = all")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def load_plan(path: str, robot: str, scene: str, model: str, as_value: float | None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["target_robot"] != robot or r["source_scene"] != scene:
                continue
            if model and r["model"] != model:
                continue
            if as_value is not None and abs(float(r["target_a_s"]) - float(as_value)) > 1e-9:
                continue
            rows.append(r)
    return rows


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    import environment
    import experiments
    import protocol

    profile = environment.apply_robot_profile(args.robot)
    protocol.TASK_INSTRUCTION = (
        f"You are a {profile.name} humanoid robot. Your task is to reach the red marker on "
        "the far wall."
    )
    rows = load_plan(args.plan, args.robot, args.scene, args.model, args.as_value)
    if args.limit:
        rows = rows[: args.limit]
    print(
        f"[stage3] robot={profile.key} ({profile.name}) scene={args.scene} "
        f"rows={len(rows)} shoulder={profile.shoulder_width:.3f} height={profile.body_height:.3f}",
        flush=True,
    )
    if not rows:
        print("[stage3] no rows match the filters", flush=True)
        app.close()
        return 2

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
    env = environment.setup_scene(app, task_dict=task)

    for index, row in enumerate(rows, 1):
        model = row["model"]
        source_episode = int(row["source_episode_id"])
        source_marker = int(row["source_marker_slot"])
        target_as = float(row["target_a_s"])
        width = float(row["target_channel_width"])
        env.set_channel_width(width)
        built = env.get_channel_width()
        ratio = env.get_a_s_ratio()
        tag = f"stage3-{profile.key}-{args.scene}-as{target_as:.2f}-{model}-src{source_episode:03d}"
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
            f"[stage3] {index}/{len(rows)} {model} A/S={target_as:.2f} "
            f"source={row['source_scene']} L{row['source_level']} ep{source_episode:03d} "
            f"marker={source_marker} width={width:.4f} built={built:.4f} ratio={ratio:.3f}",
            flush=True,
        )
        # The source marker determines reset_scene's slot: same scene and same
        # marker as the Stage 1 record, but a different body and width.
        episode = runner._run_episode(
            level=int(round(target_as * 100)),
            episode_id=source_marker - 1,
            channel_width=built,
            ratio=ratio,
        )
        episode["stage3_robot"] = profile.key
        episode["stage3_robot_name"] = profile.name
        episode["stage3_shoulder_width"] = float(profile.shoulder_width)
        episode["stage3_a_s_requested"] = float(target_as)
        episode["stage3_scene"] = args.scene
        episode["stage3_pair_id"] = row["pair_id"]
        episode["stage3_source_model"] = model
        episode["stage3_source_scene"] = row["source_scene"]
        episode["stage3_source_level"] = int(row["source_level"])
        episode["stage3_source_a_s"] = float(row["source_a_s"])
        episode["stage3_source_episode_id"] = source_episode
        episode["stage3_source_marker_slot"] = source_marker
        episode["stage3_source_passed"] = row["source_passed"] == "True"
        episode["stage3_source_result"] = row["source_result"]
        episode["stage3_source_actions"] = int(row["source_actions"])
        runner._save_episode(episode)
        print(
            f"[stage3] done {row['pair_id']} {model} A/S={target_as:.2f} "
            f"passed={episode.get('passed')} steps={episode.get('total_steps')}",
            flush=True,
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
