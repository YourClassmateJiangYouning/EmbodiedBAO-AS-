"""Export the final 105-pair Stage 2 corpus as one paper-ready folder + zip.

Run from repo root. Copies (never moves) the 105 result directories and their
105 agent logs, writes pair/model/scene/baseline tables, and includes the blind
memory cards, test plan, prompt protocol, runner and Stage 1 baseline tables.
"""
from __future__ import annotations
import csv, glob, json, os, shutil, sys, zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stage2_protocol import parse_digest
OUT = os.path.join(ROOT, "exports", "stage2_final_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
TABLES = os.path.join(OUT, "tables")
os.makedirs(TABLES, exist_ok=True)

def acts(x):
    if isinstance(x, list):
        return [str(a).strip() for a in x if str(a).strip()]
    return [a.strip() for a in str(x or "").split(",") if a.strip()]

def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]

def read_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

mem = {}
for line in open(os.path.join(ROOT, "reports", "stage2_memory_cards_top7_15.jsonl")):
    c = json.loads(line)
    mem[c["memory_id"]] = c
plan = {r["memory_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "reports", "stage2_top7_15_test_plan.csv")))}

rows = []
result_dirs = []
for path in sorted(glob.glob(os.path.join(ROOT, "results", "level0", "*", "stage2-*", "episode_*.json"))):
    if path.endswith("_steps.json"):
        continue
    d = read_json(path)
    if not d:
        continue
    mid = d.get("stage2_memory_id")
    m = mem.get(mid, {})
    src = acts(m.get("actions"))
    test = acts(d.get("action_sequence"))
    turns = sum(1 for x in test if x in ("turn_left", "turn_right"))
    looks = sum(1 for x in test if x == "look_down")
    tag = d.get("run_tag")
    ep_id = int(d.get("episode_id", 0))
    log = os.path.join(ROOT, "logs", tag, f"level0_episode{ep_id:03d}_agent.txt")
    timeout_lines = 0
    if os.path.exists(log):
        timeout_lines = sum(1 for line in open(log, errors="ignore") if any(k in line for k in ("TimeoutError", "RemoteDisconnected", "REQUEST ERROR", "RETRY")))
    # Same scene / same marker, ordinary Stage 1 Level 0 baseline.
    marker = int(d.get("marker_slot") or 1)
    base_path = os.path.join(ROOT, "results", "level0", d.get("model_name").replace("/", "-"),
                             f"full_{d.get('model_name').replace('/', '-')}-v8-actions-flipped-{d.get('scene')}",
                             f"episode_{marker - 1:03d}.json")
    base = read_json(base_path)
    base_actions = acts(base.get("action_sequence")) if base else []
    base_turns = sum(1 for x in base_actions if x in ("turn_left", "turn_right"))
    row = {
        "pair_id": plan.get(mid, {}).get("pair_id", ""),
        "model": d.get("model_name"),
        "scene": d.get("scene"),
        "marker_slot": marker,
        "memory_id": mid,
        "source_level": m.get("source_level"),
        "source_as_ratio": m.get("source_as_ratio"),
        "source_channel_width_m": m.get("channel_width_m"),
        "source_actions": len(src),
        "test_actions": len(test),
        "passed": d.get("passed"),
        "end_reason": d.get("end_reason"),
        "total_steps": d.get("total_steps"),
        "invalid_response_count": d.get("invalid_response_count"),
        "wall_collision_count": d.get("wall_collision_count"),
        "max_rotation_deg": d.get("max_rotation_deg"),
        "total_rotation": d.get("total_rotation"),
        "turn_count": turns,
        "look_down_count": looks,
        "copy_lcs_fraction": round(lcs(src, test) / max(1, len(src)), 4) if src else 0.0,
        "digest_valid": bool(parse_digest(d.get("stage2_digest") or "").get("digested")),
        "timeout_lines": timeout_lines,
        "baseline_passed": base.get("passed") if base else "",
        "baseline_steps": base.get("total_steps") if base else "",
        "baseline_turns": base_turns if base else "",
        "baseline_max_rotation_deg": base.get("max_rotation_deg") if base else "",
        "turn_delta_vs_baseline": (turns - base_turns) if base else "",
        "action_sequence": ",".join(test),
        "source_action_sequence": ",".join(src),
        "result_path": os.path.relpath(path, ROOT),
        "agent_log": os.path.relpath(log, ROOT) if os.path.exists(log) else "",
    }
    rows.append(row)
    result_dirs.append(os.path.dirname(path))

# Pair table
pair_fields = list(rows[0].keys())
with open(os.path.join(TABLES, "stage2_pair_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pair_fields); w.writeheader(); w.writerows(rows)

# Model summary
by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r)
model_fields = ["model", "n_pairs", "passed", "pass_rate_pct", "mean_steps", "mean_turns", "mean_max_rotation_deg", "mean_copy_lcs", "mean_look_down", "total_wall_collisions", "mean_turn_delta_vs_baseline"]
with open(os.path.join(TABLES, "stage2_model_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=model_fields); w.writeheader()
    for model, rs in sorted(by_model.items()):
        n = len(rs)
        w.writerow({
            "model": model,
            "n_pairs": n,
            "passed": sum(bool(x["passed"]) for x in rs),
            "pass_rate_pct": round(100 * sum(bool(x["passed"]) for x in rs) / n, 2),
            "mean_steps": round(sum(float(x["total_steps"] or 0) for x in rs) / n, 2),
            "mean_turns": round(sum(int(x["turn_count"]) for x in rs) / n, 2),
            "mean_max_rotation_deg": round(sum(float(x["max_rotation_deg"] or 0) for x in rs) / n, 2),
            "mean_copy_lcs": round(sum(float(x["copy_lcs_fraction"]) for x in rs) / n, 3),
            "mean_look_down": round(sum(int(x["look_down_count"]) for x in rs) / n, 2),
            "total_wall_collisions": sum(int(x["wall_collision_count"] or 0) for x in rs),
            "mean_turn_delta_vs_baseline": round(sum(float(x["turn_delta_vs_baseline"] or 0) for x in rs) / n, 2),
        })

# Scene x model
by_sm = {}
for r in rows:
    by_sm.setdefault((r["scene"], r["model"]), []).append(r)
with open(os.path.join(TABLES, "stage2_scene_model_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["scene", "model", "n_pairs", "passed", "pass_rate_pct", "mean_steps", "mean_turns", "mean_max_rotation_deg", "mean_copy_lcs", "mean_turn_delta_vs_baseline"]); w.writeheader()
    for (scene, model), rs in sorted(by_sm.items()):
        n = len(rs)
        w.writerow({
            "scene": scene, "model": model, "n_pairs": n,
            "passed": sum(bool(x["passed"]) for x in rs),
            "pass_rate_pct": round(100 * sum(bool(x["passed"]) for x in rs) / n, 2),
            "mean_steps": round(sum(float(x["total_steps"] or 0) for x in rs) / n, 2),
            "mean_turns": round(sum(int(x["turn_count"]) for x in rs) / n, 2),
            "mean_max_rotation_deg": round(sum(float(x["max_rotation_deg"] or 0) for x in rs) / n, 2),
            "mean_copy_lcs": round(sum(float(x["copy_lcs_fraction"]) for x in rs) / n, 3),
            "mean_turn_delta_vs_baseline": round(sum(float(x["turn_delta_vs_baseline"] or 0) for x in rs) / n, 2),
        })

# Failures
with open(os.path.join(TABLES, "stage2_failed_pairs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pair_fields); w.writeheader()
    w.writerows([r for r in rows if not r["passed"]])

# Copy result dirs and agent logs
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
os.makedirs(os.path.join(OUT, "agent_logs"), exist_ok=True)
for d in sorted(set(result_dirs)):
    model = os.path.basename(os.path.dirname(d))
    tag = os.path.basename(d)
    dst = os.path.join(OUT, "results", model, tag)
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(d, dst)
for r in rows:
    tag = os.path.basename(os.path.dirname(r["result_path"]))
    src = os.path.join(ROOT, "logs", tag)
    dst = os.path.join(OUT, "agent_logs", tag)
    if os.path.isdir(src):
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)

# Memory / plan / code / docs
os.makedirs(os.path.join(OUT, "memory"), exist_ok=True)
for name in ("stage2_memory_cards_top7_15.jsonl", "stage2_memory_cards_top7_15_blind.jsonl",
             "stage2_top7_15_test_plan.csv", "stage2_memory_schema.md", "stage2_smoke_S2M-029.md"):
    src = os.path.join(ROOT, "reports", name)
    if os.path.exists(src): shutil.copy2(src, os.path.join(OUT, "memory", name))
os.makedirs(os.path.join(OUT, "code"), exist_ok=True)
for name in ("STAGE2_DESIGN.md", "stage2_protocol.py", "stage2_runner.py"):
    src = os.path.join(ROOT, name)
    if os.path.exists(src): shutil.copy2(src, os.path.join(OUT, "code", name))
os.makedirs(os.path.join(OUT, "stage1_reference"), exist_ok=True)
for name in ("stage1_model_summary.csv", "stage1_model_scene_metrics.csv", "stage1_sfinal_nps_ranked.csv", "stage1_report.md"):
    src = os.path.join(ROOT, "reports", name)
    if os.path.exists(src): shutil.copy2(src, os.path.join(OUT, "stage1_reference", name))

# README
with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write(f"""# Stage 2 final export

Generated: {datetime.now().isoformat(timespec='seconds')}

## Corpus

- 7 models x 15 narrow-success memories = 105 paired tests.
- Same model, same scene, same marker.
- Source: Levels 11-16 (A/S 0.9-0.4).
- Test: Level 0 / A/S 2.0.
- One blind memory per test; no memory bank.
- The prompt does not state A/S, Level, width, narrow/wide or test.
- Stage 2 uses Step 0 digest (no action) then formal steps 1..30.

## Contents

- `results/`: the 105 episode records and `_steps.json` sidecars.
- `agent_logs/`: full prompts and raw model replies for each pair.
- `tables/stage2_pair_summary.csv`: one row per pair, with source/test action
  counts, copy LCS, turns, max rotation, pass, invalid, timeout lines, and the
  Stage 1 no-memory Level-0 baseline for the same scene/marker.
- `tables/stage2_model_summary.csv`: per-model aggregate.
- `tables/stage2_scene_model_summary.csv`: scene x model.
- `tables/stage2_failed_pairs.csv`: failures.
- `memory/`: blind/full memory cards, test plan, schema, smoke report.
- `code/`: prompt protocol, runner and design.
- `stage1_reference/`: Stage 1 summaries used for baselines.

## Important definitions

- `copy_lcs_fraction`: LCS(source actions, test actions) / len(source actions).
- `turn_count`: number of turn_left/turn_right in the test.
- `baseline_turns`: turns in the ordinary Stage 1 Level-0 episode with the same
  scene and marker, no memory.
- `timeout_lines`: timeout/retry lines in that pair's agent log (not a step count).

One Qwen pair (S2M-105) is a genuine `max_steps` failure, not a timeout.
""")

# Zip
zip_path = os.path.join(ROOT, "exports", os.path.basename(OUT) + ".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for base, dirs, files in os.walk(OUT):
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.relpath(p, OUT))
print("OUT", OUT)
print("ZIP", zip_path)
print("pairs", len(rows), "failed", sum(1 for r in rows if not r["passed"]))
