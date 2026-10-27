"""Build the 50-cell Stage 3 plan from the final Stage 1 corpus.

For each Stage 1 top-five-by-pass-rate model, choose two source conditions per
Stage 1 scene (one for G1, one for R1), covering A/S 1.1/1.0/0.9.  Each source
condition is one concrete Stage 1 episode, with its original scene, marker and
A/S.  The Stage 3 run keeps scene and marker and recomputes the channel width
from the target robot's shoulder width.
"""
from __future__ import annotations
import csv, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [
    "deepseek-v4-flash-vision-exp",
    "gemini-3-flash-preview",
    "mimo-v2.6-pro",
    "claude-haiku-4-5-20251001",
    "gpt-5",
]
SCENES = ["stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5"]
# scene -> (G1 A/S, R1 A/S)
ASSIGN = {
    "stage1.1": (1.1, 0.9),
    "stage1.2": (0.9, 1.1),
    "stage1.3": (1.0, 0.9),
    "stage1.4": (1.1, 0.9),
    "stage1.5": (0.9, 1.1),
}
LEVEL_FOR_AS = {1.1: 9, 1.0: 10, 0.9: 11}
SHOULDER = {"G1": 0.450, "R1": 0.357}

def source_for(model: str, scene: str, asv: float):
    level = LEVEL_FOR_AS[asv]
    pat = os.path.join(ROOT, "results", f"level{level}", model,
                       f"full_{model}-v8-actions-flipped-{scene}*", "episode_*.json")
    paths = [p for p in glob.glob(pat) if not p.endswith("_steps.json")]
    if not paths:
        raise FileNotFoundError(f"no Stage 1 source for {model} {scene} A/S {asv}: {pat}")
    candidates = []
    for p in sorted(paths):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        candidates.append((0 if d.get("passed") else 1, int(d.get("episode_id", 0)), p, d))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, episode_id, path, d = candidates[0]
    return {
        "source_result": os.path.relpath(path, ROOT),
        "source_episode_id": episode_id,
        "source_marker_slot": int(d.get("marker_slot") or 1),
        "source_passed": bool(d.get("passed")),
        "source_level": level,
        "source_a_s": round(float(d.get("a_s_ratio") or asv), 4),
        "source_actions": len(str(d.get("action_sequence") or "").split(",")),
    }

rows = []
for model in MODELS:
    for scene in SCENES:
        for robot, asv in zip(("G1", "R1"), ASSIGN[scene]):
            s = source_for(model, scene, asv)
            width = round(SHOULDER[robot] * asv, 6)
            rows.append({
                "pair_id": f"S3P-{len(rows)+1:03d}",
                "model": model,
                "source_scene": scene,
                "source_level": s["source_level"],
                "source_a_s": s["source_a_s"],
                "source_episode_id": s["source_episode_id"],
                "source_marker_slot": s["source_marker_slot"],
                "source_passed": s["source_passed"],
                "source_actions": s["source_actions"],
                "source_result": s["source_result"],
                "target_robot": robot,
                "target_shoulder_width": SHOULDER[robot],
                "target_channel_width": width,
                "target_a_s": asv,
            })
out = os.path.join(ROOT, "reports", "stage3_pair_plan.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", out, "rows", len(rows))
from collections import Counter
print("model", Counter(r["model"] for r in rows))
print("robot", Counter(r["target_robot"] for r in rows))
print("as", Counter(r["target_a_s"] for r in rows))
print("scene", Counter(r["source_scene"] for r in rows))
