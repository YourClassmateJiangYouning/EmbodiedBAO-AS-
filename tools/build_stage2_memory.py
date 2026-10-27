"""Build the top-7 Stage 2 memory cards and test plan.

Selection
---------
Top 7 models by the recomputed Stage 1 Sfinal (ties by NPS):
gemini-3-flash-preview, deepseek-v4-flash-vision-exp, mimo-v2.6-pro,
claude-haiku-4-5-20251001, gemini-2.5-flash, gpt-5,
Qwen/Qwen3-VL-30B-A3B-Instruct.

For each model: 15 narrow successful episodes (Levels 11-16, A/S 0.9-0.4),
3 per scene.  Every selected source becomes one memory card and one wide
Level 0 / A/S 2.0 test in the SAME scene and marker slot.  This is 105 paired
tests in total (7 x 15), one own-memory card per test, not a 15-card bank.

A memory card contains the successful action sequence and the step-by-step
reasoning/feedback, plus the geometry and outcome metadata.  Raw camera
images are not stored by the Stage 1 runner and are not invented here.

Outputs:
  reports/stage2_memory_cards_top7_15.jsonl
  reports/stage2_top7_15_test_plan.csv
  reports/stage2_memory_schema.md
"""
from __future__ import annotations
import collections, csv, glob, json, os, re, sys
sys.path.insert(0, ".")
import memory_metrics as mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
NARROW_LEVELS = list(range(11, 17))

def load_results():
    by = collections.defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, "results", "**", "episode_*.json"), recursive=True):
        if path.endswith("_steps.json") or "v8-actions-flipped" not in path:
            continue
        try:
            d = json.load(open(path))
        except (OSError, ValueError):
            continue
        if "level" not in d or len(d.get("steps") or []) == 0:
            continue
        d["_path"] = path
        # The result directory name is the same key used by stage1_model_summary.csv;
        # model_name in the JSON can be the dashed gateway spelling (Qwen/...).
        model_key = path.split(os.sep)[-3]
        by[model_key].append(d)
    return by

def top7():
    rows = list(csv.DictReader(open(os.path.join(REPORTS, "stage1_model_summary.csv"))))
    rows.sort(key=lambda r: (-float(r["stage1_sfinal"]), -float(r["nps"]), r["model"]))
    return rows[:7]

def action_list(e):
    a = e.get("action_sequence") or []
    return [x.strip() for x in a.split(",") if x.strip()] if isinstance(a, str) else list(a)

def success_rows(e):
    a = action_list(e)
    turns = sum(1 for x in a if x in ("turn_left", "turn_right"))
    looks = sum(1 for x in a if x == "look_down")
    req = 15 * mm.min_turns(float(e.get("channel_width") or 0))
    return {
        "actions": a, "turns": turns, "looks": looks, "theta_req": req,
        "turned": turns > 0,
        "timely_look": looks > 0,
    }

def choose_15(model, episodes):
    """3 per scene; spread over levels where possible."""
    by_scene = collections.defaultdict(list)
    for e in episodes:
        if int(e["level"]) in NARROW_LEVELS and e.get("passed") and success_rows(e)["turned"]:
            by_scene[e.get("scene")].append(e)
    for v in by_scene.values():
        v.sort(key=lambda e: (int(e["level"]), int(e.get("episode_id") or 0)))
    selected = []
    leftovers = []
    for scene in ("stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5"):
        pool = by_scene.get(scene, [])
        # one per level, walking 11..16; the first three available levels per scene
        chosen = []
        for level in NARROW_LEVELS:
            for e in pool:
                if int(e["level"]) == level and e not in chosen:
                    chosen.append(e)
                    break
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            for e in pool:
                if e not in chosen:
                    chosen.append(e)
                if len(chosen) == 3:
                    break
        selected.extend(chosen[:3])
        leftovers.extend([e for e in pool if e not in chosen])
    # fill any shortfall from leftovers, never from another model
    for e in leftovers:
        if len(selected) >= 15:
            break
        if e not in selected:
            selected.append(e)
    return selected[:15]

def memory_card(model, e, idx):
    m = success_rows(e)
    width = float(e.get("channel_width") or 0)
    steps = e.get("steps") or []
    traj = []
    for s in steps:
        traj.append({
            "step": s.get("step"),
            "action": s.get("action"),
            "feedback": s.get("feedback"),
            "reasoning": s.get("reasoning", ""),
            "position_x": s.get("position_x"),
            "position_z": s.get("position_z"),
            "torso_rotation": s.get("torso_rotation"),
            "collision": s.get("collision"),
            "scene_description": s.get("scene_description", ""),
        })
    text = [
        f"Successful narrow-pass memory from the SAME model, scene and marker.",
        f"model: {model}",
        f"scene: {e.get('scene')}; marker slot: {e.get('marker_slot')}",
        f"source A/S: {2.0 - 0.1*int(e['level']):.1f}; level: {e['level']}; channel width: {width:.3f} m",
        f"required rotation: {m['theta_req']:.0f} degrees; max rotation reached: {float(e.get('max_rotation_deg') or 0):.0f} degrees",
        f"outcome: passed in {e.get('total_steps') or len(steps)} steps; actions: {', '.join(m['actions'])}",
        "step-by-step:",
    ]
    for s in traj:
        text.append(
            f"- step {s['step']}: {s['action']} -> {s['feedback']} | reasoning: {s['reasoning']}"
        )
    return {
        "memory_id": f"S2M-{idx:03d}",
        "model": model,
        "scene": e.get("scene"),
        "marker_slot": e.get("marker_slot"),
        "source_level": int(e["level"]),
        "source_as_ratio": round(2.0 - 0.1*int(e["level"]), 1),
        "channel_width_m": width,
        "required_turns": mm.min_turns(width) if width else 0,
        "theta_req_deg": m["theta_req"],
        "outcome": "passed",
        "total_steps": int(e.get("total_steps") or len(steps)),
        "max_rotation_deg": float(e.get("max_rotation_deg") or 0),
        "look_down_count": m["looks"],
        "actions": m["actions"],
        "steps": traj,
        "final_position_x": e.get("final_position_x"),
        "final_position_z": e.get("final_position_z"),
        "final_torso_rotation": e.get("final_torso_rotation"),
        "wall_collision_count": int(e.get("wall_collision_count") or 0),
        "memory_text": "\n".join(text),
    }

_CONDITION_TERMS = re.compile(r"(?i)\b(?:narrow(?:est)?|wide|a/s|level\s*\d+|channel(?:\s*width)?|width|required(?:\s*rotation)?|shoulder(?:\s*width)?|opening(?:\s*width)?)\b")
_ANGLE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:°|deg(?:rees)?)\b", re.I)
_METERS = re.compile(r"\b\d+(?:\.\d+)?\s*m\b", re.I)
_DEGREES = re.compile(r"\bdegrees?\b", re.I)


def _redact_condition(text):
    if not text:
        return ""
    text = _CONDITION_TERMS.sub("the passage", str(text))
    text = _ANGLE.sub("the rotation", text)
    text = _METERS.sub("the distance", text)
    text = _DEGREES.sub("the rotation", text)
    return text


def blind_card(card):
    """What the tested model sees: only scene, actions and the source reasoning."""
    steps = []
    for s in card["steps"]:
        steps.append({
            "step": s.get("step"), "action": s.get("action"),
            "feedback": s.get("feedback"), "reasoning": s.get("reasoning", ""),
            "position_x": s.get("position_x"), "position_z": s.get("position_z"),
            "torso_rotation": s.get("torso_rotation"), "collision": s.get("collision"),
            "scene_description": s.get("scene_description", ""),
        })
    text = [f"Scene: {card['scene']}",
            f"Actions: {', '.join(card['actions'])}", "Steps:"]
    for s in steps:
        text.append(f"- step {s['step']}: {s['action']} -> {s['feedback']} | reasoning: {s['reasoning']}")
    return {
        "memory_id": card["memory_id"], "scene": card["scene"],
        "marker_slot": card["marker_slot"], "actions": card["actions"],
        "steps": steps, "final_position_x": card["final_position_x"],
        "final_position_z": card["final_position_z"],
        "wall_collision_count": card["wall_collision_count"],
        "memory_text": "\n".join(text),
    }


def main():
    by = load_results()
    ranked = top7()
    cards, plan = [], []
    for r in ranked:
        model = r["model"]
        eps = by.get(model, [])
        selected = choose_15(model, eps)
        print(f"{model:38s} selected {len(selected)}")
        for e in selected:
            cards.append(memory_card(model, e, len(cards)+1))
            plan.append({
                "pair_id": f"S2P-{len(plan)+1:03d}",
                "model": model,
                "memory_id": cards[-1]["memory_id"],
                "scene": cards[-1]["scene"],
                "marker_slot": cards[-1]["marker_slot"],
                "narrow_level": cards[-1]["source_level"],
                "narrow_as_ratio": cards[-1]["source_as_ratio"],
                "narrow_episode_id": e.get("episode_id"),
                "wide_level": 0,
                "wide_as_ratio": 2.0,
                "wide_episode_id": "TBD",
                "condition": "own_narrow_success_memory",
            })
    with open(os.path.join(REPORTS, "stage2_memory_cards_top7_15.jsonl"), "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(REPORTS, "stage2_memory_cards_top7_15_blind.jsonl"), "w") as f:
        for c in cards:
            f.write(json.dumps(blind_card(c), ensure_ascii=False) + "\n")
    with open(os.path.join(REPORTS, "stage2_top7_15_test_plan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plan[0].keys()))
        w.writeheader(); w.writerows(plan)
    schema = """# Stage 2 memory schema

Each line in `stage2_memory_cards_top7_15.jsonl` is one successful narrow-pass
memory.  A test uses exactly that memory card; there is no 15-card memory bank.

Fields:
- identity: `memory_id`, `model`, `scene`, `marker_slot`
- source condition: `source_level`, `source_as_ratio`, `channel_width_m`
- geometry: `required_turns`, `theta_req_deg`
- outcome: `outcome`, `total_steps`, `max_rotation_deg`, `look_down_count`,
  `final_position_x`, `final_position_z`, `final_torso_rotation`,
  `wall_collision_count`
- behaviour: `actions` (ordered action sequence)
- reasoning: `steps` (per-step action, feedback, reasoning, pose, collision,
  scene_description)
- `memory_text`: a prompt-ready rendering of the same information

The test plan (`stage2_top7_15_test_plan.csv`) pairs each memory with the SAME
scene and marker slot at Level 0 / A/S 2.0.  `wide_episode_id` is TBD because
Stage 2 has not been run.

Raw camera frames are not included: the Stage 1 runner did not save them.
"""
    open(os.path.join(REPORTS, "stage2_memory_schema.md"), "w").write(schema)
    print("cards", len(cards), "plan", len(plan))

if __name__ == "__main__":
    main()
