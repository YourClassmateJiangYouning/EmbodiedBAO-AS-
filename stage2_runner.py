"""Run the new Stage 2 test pairs: one blind success memory -> Level 0 / A/S 2.0.

The old ``memory_experiment.py`` is archived and is not imported.  This runner
uses the existing Stage 1 environment/step loop, but sets ``BAOExperimentRunner``'s
opt-in Stage 2 memory fields so the prompt is:

  Stage 1 task/action/state + one blind memory + no-action digest + test actions.

One invocation handles ONE scene (Isaac's scene is constructed once).  Run it
five times for the five scenes in the test plan.  It does not read or write the
Stage 1 checkpoint and does not touch a running sweep.

Example:

    export BOYUE_API_KEY=...
    /home/ybh/isaacsim/python.sh stage2_runner.py \
        --scene stage1.5 --plan reports/stage2_top7_15_test_plan.csv \
        --memory reports/stage2_memory_cards_top7_15_blind.jsonl --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from persistence import sanitize_tag

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
DEFAULT_PLAN = os.path.join(REPORTS, "stage2_top7_15_test_plan.csv")
DEFAULT_MEMORY = os.path.join(REPORTS, "stage2_memory_cards_top7_15_blind.jsonl")


def load_memory(path: str) -> dict:
    cards = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                card = json.loads(line)
                cards[card["memory_id"]] = card
    return cards


def load_plan(path: str, scene: str, model: str | None, models: set[str] | None, memory_id: str | None, limit: int | None):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scene"] != scene:
                continue
            if model and row["model"] != model:
                continue
            if models and row["model"] not in models:
                continue
            if memory_id and row["memory_id"] != memory_id:
                continue
            rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", default=None, help="comma-separated model names")
    parser.add_argument("--memory-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    os.environ["BAO_IMAGE_SIZE"] = str(int(args.image_size))
    os.environ["BAO_LLM_TIMEOUT"] = str(float(args.llm_timeout))

    memories = load_memory(args.memory)
    models = {x.strip() for x in (args.models or "").split(",") if x.strip()} or None
    rows = load_plan(args.plan, args.scene, args.model, models, args.memory_id, args.limit)
    if not rows:
        print(f"[stage2] no rows for scene={args.scene} model={args.model}", flush=True)
        return 2

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})

    import environment
    import ai_agent
    from experiments import BAOExperimentRunner

    # The environment's scene is fixed at construction; one invocation = one scene.
    env = environment.setup_scene(
        app, task_dict={"scene": args.scene, "level": 0, "headless": args.headless}
    )

    try:
        for index, row in enumerate(rows, 1):
            memory = memories.get(row["memory_id"])
            if memory is None:
                print(f"[stage2] missing memory {row['memory_id']}", flush=True)
                continue
            model = row["model"]
            # The plan keeps the filesystem-safe slug (results/ directories use
            # it), but the gateway/model API needs the historical slash spelling.
            # Stage 1's models.json and result model_name both use
            # "Qwen/Qwen3-VL-30B-A3B-Instruct"; sending the dash slug made Qwen
            # produce no response at all.
            api_model = model.replace("Qwen-", "Qwen/", 1) if model.startswith("Qwen-") else model
            marker = int(row["marker_slot"])
            # reset_scene(episode_id) picks slot = episode_id % 5 + 1.
            episode_id = marker - 1
            tag = sanitize_tag(f"stage2-{args.scene}-{row['memory_id']}")
            runner = BAOExperimentRunner(
                env=env,
                model=api_model,
                max_steps=args.max_steps,
                episodes_per_level=1,
                tag=tag,
                save_obs=False,
                results_root="results",
                logs_root="logs",
            )
            runner.stage2_memory_text = memory["memory_text"]
            runner.stage2_memory_id = row["memory_id"]

            width = environment.level_channel_width(0)
            env.set_channel_width(width)
            built = float(env.get_channel_width())
            if abs(built - width) > 1e-9:
                raise RuntimeError(f"level0 built {built} not {width}")
            ratio = environment.a_s_ratio(built)

            print(
                f"[stage2] {index}/{len(rows)} {api_model} scene={args.scene} "
                f"marker={marker} memory={row['memory_id']}",
                flush=True,
            )
            episode = runner._run_episode(
                level=0,
                episode_id=episode_id,
                channel_width=built,
                ratio=ratio,
            )
            # Add the pairing identity to the episode record; these are not used by
            # Stage 1 and do not change the step records.
            episode["stage2_memory_id"] = row["memory_id"]
            episode["stage2_scene"] = args.scene
            episode["stage2_marker_slot"] = marker
            episode["stage2_test_level"] = 0
            episode["stage2_test_as_ratio"] = 2.0
            runner._save_episode(episode)
            runner._save_summary(0, [episode])
            print(
                f"[stage2] done memory={row['memory_id']} passed={episode.get('passed')} "
                f"steps={episode.get('total_steps')}",
                flush=True,
            )
    finally:
        try:
            app.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
