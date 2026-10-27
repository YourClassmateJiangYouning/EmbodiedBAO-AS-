"""Build the final 100-cell Stage 3 plan.

Per model and per body: 10 cells
  A/S 1.3 x3, 1.1 x3, 1.0 x2, 0.9 x2
  each Stage 1 scene exactly x2
  as many distinct markers as possible

G1 keeps its 5 existing cells per model (the current 50-cell plan's G1 rows;
22 records already on disk + 3 pending).  Five new G1 cells fill the missing
A/S/scene counts.  R1 mirrors the final 10 G1 conditions per model, so scene,
A/S, marker and source episode are paired across the two bodies.
"""
from __future__ import annotations
import csv, glob, json, os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [
    "deepseek-v4-flash-vision-exp",
    "gemini-3-flash-preview",
    "mimo-v2.6-pro",
    "claude-haiku-4-5-20251001",
    "gpt-5",
]
SCENES = ["stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5"]
SHOULDER = {"G1": 0.450, "R1": 0.357}
LEVEL = {1.3: 7, 1.1: 9, 1.0: 10, 0.9: 11}
# G1 supplement: one cell for every scene.  A/S fills the existing distribution.
SUPPLEMENT = {
    "stage1.1": (1.3, 2),
    "stage1.2": (1.3, 4),
    "stage1.3": (1.3, 3),
    "stage1.4": (1.0, 2),
    "stage1.5": (1.1, 5),
}

def pick_source(model: str, scene: str, asv: float, marker: int):
    level = LEVEL[asv]
    pat = os.path.join(ROOT, "results", f"level{level}", model,
                       f"full_{model}-v8-actions-flipped-{scene}*", "episode_*.json")
    paths = [p for p in glob.glob(pat) if not p.endswith("_steps.json")]
    if not paths:
        raise FileNotFoundError(f"no Stage 1 source for {model} {scene} A/S {asv}")
    candidates = []
    for p in sorted(paths):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if int(d.get("marker_slot") or 1) != int(marker):
            continue
        candidates.append((0 if d.get("passed") else 1, int(d.get("episode_id", 0)), p, d))
    if not candidates:
        raise FileNotFoundError(f"no marker {marker} source for {model} {scene} A/S {asv}")
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, ep, path, d = candidates[0]
    return {
        "source_result": os.path.relpath(path, ROOT),
        "source_episode_id": ep,
        "source_marker_slot": int(d.get("marker_slot") or 1),
        "source_passed": bool(d.get("passed")),
        "source_level": level,
        "source_a_s": round(float(d.get("a_s_ratio") or asv), 4),
        "source_actions": len(str(d.get("action_sequence") or "").split(",")),
    }

# Existing G1 rows from the original 50-cell plan.
old = [r for r in csv.DictReader(open(os.path.join(ROOT, "reports", "stage3_pair_plan.csv")))
       if r["target_robot"] == "G1"]
old_by_model = {}
for r in old:
    old_by_model.setdefault(r["model"], []).append(r)

# Completion status of existing G1 records.
done = {}
for p in glob.glob(os.path.join(ROOT, "results", "level*", "**", "stage3-*", "episode_*.json"), recursive=True):
    if p.endswith("_steps.json"):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    pid = d.get("stage3_pair_id")
    if pid:
        done[pid] = d

rows = []
for model in MODELS:
    g1_final = []
    # 1. preserve the existing 5 G1 cells per model.
    for r in old_by_model[model]:
        g1_final.append({
            "phase": "existing_g1",
            "source_scene": r["source_scene"],
            "source_a_s": float(r["target_a_s"]),
            "source_level": int(r["source_level"]),
            "source_episode_id": int(r["source_episode_id"]),
            "source_marker_slot": int(r["source_marker_slot"]),
            "source_passed": r["source_passed"] == "True",
            "source_actions": int(r["source_actions"]),
            "source_result": r["source_result"],
            "existing_pair_id": r["pair_id"],
            "already_done": r["pair_id"] in done,
        })
    # 2. add 5 G1 cells: one per scene, chosen A/S and a new marker.
    for scene in SCENES:
        asv, marker = SUPPLEMENT[scene]
        s = pick_source(model, scene, asv, marker)
        g1_final.append({
            "phase": "supplement_g1",
            **s,
            "source_scene": scene,
            "source_a_s": asv,
            "source_level": LEVEL[asv],
            "source_marker_slot": marker,
            "existing_pair_id": "",
            "already_done": False,
        })
    # 3. final 10 G1, then mirror exactly to R1.
    for body in ("G1", "R1"):
        for i, cell in enumerate(sorted(g1_final, key=lambda x: (SCENES.index(x["source_scene"]), -x["source_a_s"]))):
            asv = float(cell["source_a_s"])
            width = round(SHOULDER[body] * asv, 6)
            rows.append({
                "pair_id": f"S3F-{len(rows)+1:03d}",
                "model": model,
                "target_robot": body,
                "target_shoulder_width": SHOULDER[body],
                "target_channel_width": width,
                "target_a_s": asv,
                "source_scene": cell["source_scene"],
                "source_level": cell["source_level"],
                "source_a_s": cell["source_a_s"],
                "source_episode_id": cell["source_episode_id"],
                "source_marker_slot": cell["source_marker_slot"],
                "source_passed": cell["source_passed"],
                "source_actions": cell["source_actions"],
                "source_result": cell["source_result"],
                "phase": cell["phase"] if body == "G1" else "r1_mirror",
                "existing_pair_id": cell["existing_pair_id"] if body == "G1" else "",
                "already_done": cell["already_done"] if body == "G1" else False,
            })

out = os.path.join(ROOT, "reports", "stage3_final100_plan.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("wrote", out, "rows", len(rows))
print("by model", Counter(r["model"] for r in rows))
print("by body", Counter(r["target_robot"] for r in rows))
print("by model/body", Counter((r["model"], r["target_robot"]) for r in rows))
print("by body/A-S", Counter((r["target_robot"], r["target_a_s"]) for r in rows))
print("by model/body/scene", Counter((r["model"], r["target_robot"], r["source_scene"]) for r in rows))
print("markers G1", Counter(r["source_marker_slot"] for r in rows if r["target_robot"] == "G1"))
print("markers R1", Counter(r["source_marker_slot"] for r in rows if r["target_robot"] == "R1"))
print("G1 existing done", sum(1 for r in rows if r["target_robot"] == "G1" and r["phase"] == "existing_g1" and r["already_done"]))
print("G1 existing total", sum(1 for r in rows if r["target_robot"] == "G1" and r["phase"] == "existing_g1"))
