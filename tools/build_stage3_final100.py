"""Build the final 100-cell Stage 3 plan with partial G1 reruns.

Per model and body: 10 cells
  A/S 1.3 x3, 1.1 x3, 1.0 x2, 0.9 x2
  each Stage 1 scene exactly twice
  G1 final markers: two cells per marker (1..5)

G1 keeps 12 existing cells (including 3 pending), reruns 13 existing marker-1
cells with different markers, and adds 5 supplements.  R1 gets 10 new cells per
model, paired to G1 on scene and A/S but never reusing the same source episode.
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
# G1 existing cells: keep these; rerun the others.
KEEP = {
    "deepseek-v4-flash-vision-exp": {"stage1.1", "stage1.2"},
    "gemini-3-flash-preview": {"stage1.1", "stage1.2"},
    "mimo-v2.6-pro": {"stage1.1", "stage1.2"},
    "claude-haiku-4-5-20251001": {"stage1.1", "stage1.2", "stage1.5"},
    "gpt-5": {"stage1.1", "stage1.2", "stage1.5"},
}
# Rerun markers for the dropped existing cell in each scene.
RERUN_MARKER = {
    "deepseek-v4-flash-vision-exp": {"stage1.3": 4, "stage1.4": 5, "stage1.5": 3},
    "gemini-3-flash-preview": {"stage1.3": 4, "stage1.4": 5, "stage1.5": 3},
    "mimo-v2.6-pro": {"stage1.3": 4, "stage1.4": 5, "stage1.5": 3},
    "claude-haiku-4-5-20251001": {"stage1.3": 4, "stage1.4": 5},
    "gpt-5": {"stage1.3": 5, "stage1.4": 3},
}
R1_MARKERS = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
SUPPLEMENT = {
    "stage1.1": (1.3, 2),
    "stage1.2": (1.3, 4),
    "stage1.3": (1.3, 3),
    "stage1.4": (1.0, 2),
    "stage1.5": (1.1, 5),
}

def source_for(model: str, scene: str, asv: float, marker: int):
    level = LEVEL[asv]
    pat = os.path.join(ROOT, "results", f"level{level}", model,
                       f"full_{model}-v8-actions-flipped-{scene}*", "episode_*.json")
    paths = [p for p in glob.glob(pat) if not p.endswith("_steps.json")]
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

def pick_r1(model: str, scene: str, asv: float, marker: int, g1_source: str):
    """Same scene/A/S but a different source episode and a chosen marker."""
    level = LEVEL[asv]
    pat = os.path.join(ROOT, "results", f"level{level}", model,
                       f"full_{model}-v8-actions-flipped-{scene}*", "episode_*.json")
    paths = [p for p in glob.glob(pat) if not p.endswith("_steps.json")]
    candidates = []
    for p in sorted(paths):
        rel = os.path.relpath(p, ROOT)
        if rel == g1_source:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if int(d.get("marker_slot") or 1) != int(marker):
            continue
        candidates.append((0 if d.get("passed") else 1, int(d.get("episode_id", 0)), p, d))
    if not candidates:
        raise FileNotFoundError(f"no R1 marker {marker} source for {model} {scene} A/S {asv}")
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

def row(model, body, scene, asv, s, phase, existing_pair="", already_done=False):
    return {
        "model": model, "target_robot": body,
        "target_shoulder_width": SHOULDER[body],
        "target_channel_width": round(SHOULDER[body] * asv, 6),
        "target_a_s": asv,
        "source_scene": scene, "source_level": LEVEL[asv],
        "source_a_s": s["source_a_s"], "source_episode_id": s["source_episode_id"],
        "source_marker_slot": s["source_marker_slot"],
        "source_passed": s["source_passed"], "source_actions": s["source_actions"],
        "source_result": s["source_result"],
        "phase": phase, "existing_pair_id": existing_pair, "already_done": already_done,
    }

old = {r["pair_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "reports", "stage3_pair_plan.csv"))) if r["target_robot"] == "G1"}
done = {}
for p in glob.glob(os.path.join(ROOT, "results", "level*", "**", "stage3-*", "episode_*.json"), recursive=True):
    if p.endswith("_steps.json"): continue
    try: d = json.load(open(p))
    except Exception: continue
    pid = d.get("stage3_pair_id")
    if pid: done[pid] = d

rows = []
for model in MODELS:
    g1 = []
    # Preserve existing cells.
    for r in [x for x in old.values() if x["model"] == model]:
        if r["source_scene"] in KEEP[model]:
            g1.append(row(model, "G1", r["source_scene"], float(r["target_a_s"]),
                           {k: r[k] for k in ("source_result", "source_episode_id", "source_marker_slot", "source_passed", "source_a_s", "source_actions")},
                           "existing_keep", r["pair_id"], r["pair_id"] in done))
    # Rerun the dropped marker-1 cells with a different marker.
    for r in [x for x in old.values() if x["model"] == model]:
        scene = r["source_scene"]
        if scene in KEEP[model]:
            continue
        marker = RERUN_MARKER[model][scene]
        s = source_for(model, scene, float(r["target_a_s"]), marker)
        g1.append(row(model, "G1", scene, float(r["target_a_s"]), s, "existing_rerun", r["pair_id"], False))
    # Add five supplements.
    for scene, (asv, marker) in SUPPLEMENT.items():
        s = source_for(model, scene, asv, marker)
        g1.append(row(model, "G1", scene, asv, s, "supplement_g1"))
    # Unique G1 source check.
    assert len({x["source_result"] for x in g1}) == 10, (model, "G1 duplicate source")
    # Mirror only scene/A/S, not source: R1 must use a different source episode.
    ordered = sorted(g1, key=lambda x: (SCENES.index(x["source_scene"]), -x["target_a_s"]))
    for i, cell in enumerate(ordered):
        s = pick_r1(model, cell["source_scene"], cell["target_a_s"], R1_MARKERS[i], cell["source_result"])
        rows.append(row(model, "R1", cell["source_scene"], cell["target_a_s"], s, "r1_unique"))
        rows.append(row(model, "G1", cell["source_scene"], cell["target_a_s"], {k: cell[k] for k in ("source_result", "source_episode_id", "source_marker_slot", "source_passed", "source_a_s", "source_actions")}, cell["phase"], cell["existing_pair_id"], cell["already_done"]))
# Reorder within model/body.
rows.sort(key=lambda r: (MODELS.index(r["model"]), r["target_robot"], SCENES.index(r["source_scene"]), -r["target_a_s"]))
for i, r in enumerate(rows, 1):
    r["pair_id"] = f"S3F-{i:03d}"
out = os.path.join(ROOT, "reports", "stage3_final100_plan.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", out, len(rows))
print("by model/body", Counter((r["model"], r["target_robot"]) for r in rows))
print("by body/A-S", Counter((r["target_robot"], r["target_a_s"]) for r in rows))
print("by model/body/scene", Counter((r["model"], r["target_robot"], r["source_scene"]) for r in rows))
print("G1 markers", Counter(r["source_marker_slot"] for r in rows if r["target_robot"] == "G1"))
print("R1 markers", Counter(r["source_marker_slot"] for r in rows if r["target_robot"] == "R1"))
print("G1 phases", Counter(r["phase"] for r in rows if r["target_robot"] == "G1"))
print("G1 preserved done", sum(1 for r in rows if r["target_robot"] == "G1" and r["phase"] == "existing_keep" and r["already_done"] == "True"))
# Uniqueness across all 100.
print("duplicate source rows", [(k,v) for k,v in Counter(r["source_result"] for r in rows).items() if v>1])
