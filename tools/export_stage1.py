"""Export the complete final Stage 1 corpus (15 models, 425 episodes each)."""
from __future__ import annotations
import csv, glob, json, os, shutil, zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exports", "stage1_final_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
TABLES = os.path.join(OUT, "tables")
os.makedirs(TABLES, exist_ok=True)

summary = list(csv.DictReader(open(os.path.join(ROOT, "reports", "stage1_model_summary.csv"))))
models = [r["model"] for r in summary]

# Copy only full v8 Stage 1 result directories for the 15 completed models.
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
copied_dirs = []
for model in models:
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "level*", model, "*v8-actions-flipped-stage1.*"))):
        if not os.path.isdir(path):
            continue
        level = os.path.basename(os.path.dirname(os.path.dirname(path)))
        tag = os.path.basename(path)
        dst = os.path.join(OUT, "results", level, model, tag)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(path, dst)
        copied_dirs.append(dst)

# Copy the matching full-agent-log directories for the same 15 models.
os.makedirs(os.path.join(OUT, "agent_logs"), exist_ok=True)
copied_logs = []
for model in models:
    for path in sorted(glob.glob(os.path.join(ROOT, "logs", f"full_{model}-v8-actions-flipped-stage1.*"))):
        if not os.path.isdir(path):
            continue
        dst = os.path.join(OUT, "agent_logs", os.path.basename(path))
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(path, dst)
        copied_logs.append(dst)

# Tables
for name in ("stage1_model_summary.csv", "stage1_model_scene_metrics.csv",
             "stage1_sfinal_nps_ranked.csv", "stage1_report.md"):
    src = os.path.join(ROOT, "reports", name)
    if os.path.exists(src): shutil.copy2(src, os.path.join(OUT, "tables", name))

# Code/config that define the measurement.
os.makedirs(os.path.join(OUT, "code"), exist_ok=True)
for name in ("main.py", "experiments.py", "protocol.py", "ai_agent.py", "environment.py", "models.json", "STAGE1_SCENES.md", "STAGE1_SCENE_VARIANTS.md"):
    src = os.path.join(ROOT, name)
    if os.path.exists(src): shutil.copy2(src, os.path.join(OUT, "code", name))

episodes = 0
steps = 0
for dirpath, _, files in os.walk(os.path.join(OUT, "results")):
    for name in files:
        if name.startswith("episode_") and name.endswith(".json"):
            if name.endswith("_steps.json"): steps += 1
            else: episodes += 1

with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write(f"""# Stage 1 final export

Generated: {datetime.now().isoformat(timespec='seconds')}

This is the complete final v8-actions-flipped Stage 1 corpus:
15 models x 425 episodes = 6375 episode records, each with a `_steps.json`
sidecar and a matching full agent log.

## Contents

- `results/levelN/<model>/<tag>/episode_*.json`: raw episode records.
- `results/.../episode_*_steps.json`: per-step sidecars (reasoning, response,
  state, feedback).
- `agent_logs/`: full prompts and raw model replies for every episode.
- `tables/stage1_model_summary.csv`: combined per-model metrics and S0/Sfinal/NPS.
- `tables/stage1_model_scene_metrics.csv`: model x scene and ALL.
- `tables/stage1_sfinal_nps_ranked.csv`: final ranking.
- `tables/stage1_report.md`: compact report.
- `code/`: the Stage 1 runner, prompt, environment and model roster.

## Definitions

Narrow = Levels 11-16 (A/S 0.9-0.4). Wide = Levels 0-10 (A/S 2.0-1.0).
S0/Sfinal and NPS follow the formulas in `tables/stage1_report.md`.
The 15 models are the completed v8 corpus, including Qwen3-VL-32B.
""")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"models": models, "result_directories": len(copied_dirs), "log_directories": len(copied_logs), "episode_records": episodes, "step_sidecars": steps}, f, indent=2)

zip_path = os.path.join(ROOT, "exports", os.path.basename(OUT) + ".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(OUT):
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.relpath(p, OUT))
print("OUT", OUT)
print("ZIP", zip_path)
print("models", len(models), "result_dirs", len(copied_dirs), "log_dirs", len(copied_logs), "episodes", episodes, "steps", steps)
