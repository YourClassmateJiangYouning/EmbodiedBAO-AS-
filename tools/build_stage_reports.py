"""Build the Stage 1 per-scene tables and the Stage 1 report.

Run from the repository root:

    python3 tools/build_stage_reports.py

Outputs (under reports/):
  stage1_model_scene_metrics.csv   model x scene, five scenes plus ALL
  stage1_model_summary.csv         one row per completed model
  stage1_report.md                  compact human-readable Stage 1 report

The completed corpus is every v8-actions-flipped model with exactly 425
episodes.  Qwen3-VL-32B is still running and is deliberately excluded from the
completed-model totals; its partial progress is reported separately.

Stage 1 score: the repository does not contain the earlier composite scoring
function, so the previously recorded values are carried in ``stage1_score_prior``
for the eleven models that had one.  New models are left blank rather than
inventing a number.  All raw components used by that score are in the table.

Stage 2 is built separately by tools/build_stage2_memory.py; see STAGE2_DESIGN.md. The old 12-attempt/note scenario is archived and is not used.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import os
import sys
import math

sys.path.insert(0, ".")
import memory_metrics as mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")

# Recorded before the current run, from the earlier Stage 1 scoring pass.
STAGE1_SCORE_PRIOR = {
    "Qwen-Qwen3-VL-8B-Instruct": 9.1,
    "Qwen-Qwen3-VL-30B-A3B-Instruct": 14.5,
    "gpt-4o-mini": 9.2,
    "deepseek-v4-flash-vision-exp": 20.0,
    "glm-4.6v": 9.4,
    "mimo-v2.6-pro": 20.0,
    "gemini-2.5-flash": 15.6,
    "glm-4.5v": 10.1,
    "gemini-3-flash-preview": 20.0,
    "kimi-kimi-k2.6": 9.1,
    "claude-haiku-4-5-20251001": 16.6,
}
NPS_PRIOR = {
    "deepseek-v4-flash-vision-exp": 40.9,
    "mimo-v2.6-pro": 34.7,
    "gemini-3-flash-preview": 66.1,
    "Qwen-Qwen3-VL-30B-A3B-Instruct": 4.6,
    "gemini-2.5-flash": 6.2,
    "claude-haiku-4-5-20251001": 6.9,
}
STAGE2_TOP7 = (
    "gemini-3-flash-preview",
    "deepseek-v4-flash-vision-exp",
    "mimo-v2.6-pro",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
    "gpt-5",
    "Qwen-Qwen3-VL-30B-A3B-Instruct",
)
NARROW_LEVELS = range(11, 17)  # A/S 0.9 down to 0.4
LEVELS = range(17)


def load_completed():
    by_model = collections.defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, "results", "**", "episode_*.json"), recursive=True):
        if path.endswith("_steps.json") or "v8-actions-flipped" not in path:
            continue
        parts = path.split(os.sep)
        if len(parts) < 4:
            continue
        model = parts[-4] if parts[-4] not in ("results",) else parts[-4]
        # Path is results/levelN/model/tag/episode.json
        model = parts[-3]
        try:
            episode = json.load(open(path))
        except (OSError, ValueError):
            continue
        if "level" not in episode:
            continue
        episode["_path"] = path
        by_model[model].append(episode)
    return by_model


def level_numbers(episodes):
    return {int(e["level"]) for e in episodes}


def episode_metrics(episode):
    level = int(episode["level"])
    width = float(episode.get("channel_width") or 0.0)
    try:
        required_turns = mm.min_turns(width) if width else 0
        optimal = mm.optimal_steps(width) if width else 0
    except Exception:
        required_turns, optimal = 0, 0
    actions = episode.get("action_sequence") or []
    if isinstance(actions, str):
        # The runner stores this as a comma-separated string, not a list.
        actions = [a.strip() for a in actions.split(",") if a.strip()]
    else:
        actions = list(actions)
    turn_count = sum(1 for a in actions if a in ("turn_left", "turn_right"))
    total_rotation = float(episode.get("total_rotation") or 0.0)
    max_rotation = float(episode.get("max_rotation_deg") or 0.0)
    steps = int(episode.get("total_steps") or episode.get("max_steps") or len(episode.get("steps") or []))
    return {
        "level": level,
        "passed": bool(episode.get("passed")),
        "wide": level <= 10,
        "narrow": level in NARROW_LEVELS,
        "turned": turn_count > 0,
        "looked": "look_down" in actions,
        "invalid": int(episode.get("invalid_response_count") or 0),
        "steps": steps,
        "max_rotation": max_rotation,
        "total_rotation": total_rotation,
        "required_rotation": 15.0 * required_turns,
        "over_rotated": total_rotation > 15.0 * required_turns + 1e-9,
        "extra_steps": max(0, steps - optimal) if episode.get("passed") else None,
        # Raw fields used by the recovered Stage 1 / NPS formulas.
        "actions": actions,
        "n_look": sum(1 for a in actions if a == "look_down"),
        "n_turn": turn_count,
        "width": width,
        "required_turns": required_turns,
        "optimal": optimal,
        "collisions": int(episode.get("wall_collision_count") or 0),
    }


def aggregate(rows):
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    wide = [r for r in rows if r["wide"]]
    narrow = [r for r in rows if r["narrow"]]
    turns = [r for r in rows if r["turned"]]
    looks = [r for r in rows if r["looked"]]
    over = [r for r in rows if r["over_rotated"]]
    extras = [r["extra_steps"] for r in rows if r["extra_steps"] is not None]
    return {
        "n": n,
        "passed": passed,
        "pass_pct": 100.0 * passed / n if n else 0.0,
        "wide_n": len(wide),
        "wide_passed": sum(r["passed"] for r in wide),
        "wide_pass_pct": 100.0 * sum(r["passed"] for r in wide) / len(wide) if wide else 0.0,
        "narrow_n": len(narrow),
        "narrow_passed": sum(r["passed"] for r in narrow),
        "narrow_pass_pct": 100.0 * sum(r["passed"] for r in narrow) / len(narrow) if narrow else 0.0,
        "turned": len(turns),
        "turned_pct": 100.0 * len(turns) / n if n else 0.0,
        "looked_pct": 100.0 * len(looks) / n if n else 0.0,
        "invalid": sum(r["invalid"] for r in rows),
        "mean_steps": sum(r["steps"] for r in rows) / n if n else 0.0,
        "mean_max_rot": sum(r["max_rotation"] for r in rows) / n if n else 0.0,
        "over_rotated_pct": 100.0 * len(over) / n if n else 0.0,
        "mean_extra_steps": sum(extras) / len(extras) if extras else 0.0,
    }


def _look_quality(req: float, n_look: int, actions: list) -> float:
    """Stage 1 look quality from the prior scoring note."""
    if req <= 0:
        return 0.95 if n_look else 1.0
    if n_look == 0:
        return 0.90
    if n_look > 2:
        return 0.80
    turns = [i for i, a in enumerate(actions) if a in ("turn_left", "turn_right")]
    looks = [i for i, a in enumerate(actions) if a == "look_down"]
    if not turns or (looks and looks[0] < turns[0]):
        return 1.00
    return 0.95


def _nps_look_quality(req: float, n_look: int, actions: list) -> float:
    """NPS look-quality from the prior NPS note."""
    if n_look == 0:
        return 0.90
    if n_look > 2:
        return 0.80
    turns = [i for i, a in enumerate(actions) if a in ("turn_left", "turn_right")]
    looks = [i for i, a in enumerate(actions) if a == "look_down"]
    if not turns or (looks and looks[0] < turns[0]):
        return 1.00
    return 0.95


def _episode_score(e, for_nps: bool = False) -> float:
    width = float(e.get("width") or 0.0)
    req = 15.0 * float(e.get("required_turns") or 0)
    opt = float(e.get("optimal") or 0)
    actions = e.get("actions") or []
    n_turn = int(e.get("n_turn") or 0)
    n_look = int(e.get("n_look") or 0)
    max_rot = float(e.get("max_rotation") or 0.0)
    over = max(0.0, max_rot - req) / 15.0
    under = max(0.0, req - max_rot) / 15.0
    extra = max(0, n_turn - int(e.get("required_turns") or 0))
    excess = max(0, int(e.get("steps") or 0) - int(opt))
    if for_nps:
        q_rot = math.exp(-0.20 * over - 0.03 * over * over)
        q_extra = math.exp(-0.12 * extra)
        q_eff = min(1.0, opt / max(1.0, float(e.get("steps") or 0)))
        q_look = _nps_look_quality(req, n_look, actions)
        return q_rot * q_extra * q_eff * q_look
    q_rot = math.exp(-0.20 * over - 0.03 * over * over - 0.20 * under)
    q_turn = math.exp(-0.12 * extra)
    q_look = _look_quality(req, n_look, actions)
    q_eff = math.exp(-0.02 * (excess / opt if opt else 0.0))
    q_col = math.exp(-0.05 * float(e.get("collisions") or 0))
    return q_rot * q_turn * q_look * q_eff * q_col


def stage1_scores(episodes) -> dict:
    rows = [episode_metrics(e) for e in episodes]
    # S0: difficulty-weighted mean of per-level mean q, passed only.
    num = den = 0.0
    for level in sorted({r["level"] for r in rows}):
        level_rows = [r for r in rows if r["level"] == level]
        req = 15.0 * float(level_rows[0]["required_turns"])
        w = 0.05 + 1.0 * (req / 90.0)
        mean_s = sum(100.0 * (1 if r["passed"] else 0) * _episode_score(r) for r in level_rows) / len(level_rows)
        num += w * mean_s
        den += w
    s0 = num / den if den else 0.0
    wide = [r for r in rows if r["required_turns"] == 0]
    narrow = [r for r in rows if r["required_turns"] > 0]
    r_wide = (sum(1 for r in wide if r["n_turn"]) / len(wide)) if wide else 0.0
    r_look = (sum(1 for r in narrow if r["n_look"]) / len(narrow)) if narrow else 0.0
    if r_wide > 0.10 or r_look < 0.10:
        s_final = min(s0, 20.0)
    elif r_wide > 0.05 or r_look < 0.25:
        s_final = min(s0, 50.0)
    else:
        s_final = s0
    # NPS: narrow-only quality; P_n * Q_n * (1 - O_n).
    passed = [r for r in rows if r["narrow"] and r["passed"]]
    p_n = len(passed) / len(narrow) if narrow else 0.0
    qs = []
    over_count = 0
    for r in passed:
        qs.append(_episode_score(r, for_nps=True))
        req = 15.0 * float(r["required_turns"])
        if max(0.0, float(r["max_rotation"]) - req) > 1e-9:
            over_count += 1
    q_n = sum(qs) / len(qs) if qs else 0.0
    o_n = over_count / len(passed) if passed else 0.0
    nps = 100.0 * p_n * q_n * (1.0 - o_n)
    return {
        "stage1_s0": s0,
        "stage1_sfinal": s_final,
        "nps": nps,
        "r_wide": r_wide,
        "r_look": r_look,
        "p_n": p_n,
        "q_n": q_n,
        "o_n": o_n,
    }


def main():
    os.makedirs(REPORTS, exist_ok=True)
    by_model = load_completed()
    complete = {m: es for m, es in by_model.items() if len(es) == 425 and len(level_numbers(es)) == 17}
    partial = {m: es for m, es in by_model.items() if m not in complete}

    rows = []
    for model in sorted(complete):
        all_rows = [episode_metrics(e) for e in complete[model]]
        for scene in ("stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5"):
            scene_rows = [episode_metrics(e) for e in complete[model] if e.get("scene") == scene]
            if not scene_rows:
                continue
            a = aggregate(scene_rows)
            rows.append({"model": model, "scene": scene, **a, **stage1_scores([e for e in complete[model] if e.get("scene") == scene])})
        rows.append({"model": model, "scene": "ALL", **aggregate(all_rows), **stage1_scores(complete[model])})

    scene_csv = os.path.join(REPORTS, "stage1_model_scene_metrics.csv")
    fields = ["model", "scene", "n", "passed", "pass_pct", "wide_n", "wide_passed", "wide_pass_pct",
              "narrow_n", "narrow_passed", "narrow_pass_pct", "turned", "turned_pct",
              "looked_pct", "invalid", "mean_steps", "mean_max_rot", "over_rotated_pct",
              "mean_extra_steps", "stage1_s0", "stage1_sfinal", "nps", "r_wide", "r_look",
              "p_n", "q_n", "o_n"]
    with open(scene_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(row[k], 3) if isinstance(row[k], float) else row[k]) for k in fields})

    summary_csv = os.path.join(REPORTS, "stage1_model_summary.csv")
    summary_fields = ["model", "n", "passed", "pass_pct", "wide_passed", "wide_pass_pct",
                      "narrow_passed", "narrow_pass_pct", "turned_pct", "looked_pct", "invalid",
                      "mean_steps", "mean_max_rot", "over_rotated_pct", "mean_extra_steps",
                      "stage1_s0", "stage1_sfinal", "nps", "r_wide", "r_look", "p_n", "q_n", "o_n",
                      "stage1_score_prior", "nps_prior", "stage2_eligible"]
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for model in sorted(complete):
            a = next(r for r in rows if r["model"] == model and r["scene"] == "ALL")
            writer.writerow({
                "model": model,
                "n": a["n"],
                "passed": a["passed"],
                "pass_pct": round(a["pass_pct"], 3),
                "wide_passed": a["wide_passed"],
                "wide_pass_pct": round(a["wide_pass_pct"], 3),
                "narrow_passed": a["narrow_passed"],
                "narrow_pass_pct": round(a["narrow_pass_pct"], 3),
                "turned_pct": round(a["turned_pct"], 3),
                "looked_pct": round(a["looked_pct"], 3),
                "invalid": a["invalid"],
                "mean_steps": round(a["mean_steps"], 3),
                "mean_max_rot": round(a["mean_max_rot"], 3),
                "over_rotated_pct": round(a["over_rotated_pct"], 3),
                "mean_extra_steps": round(a["mean_extra_steps"], 3),
                "stage1_s0": round(a["stage1_s0"], 3),
                "stage1_sfinal": round(a["stage1_sfinal"], 3),
                "nps": round(a["nps"], 3),
                "r_wide": round(a["r_wide"], 4),
                "r_look": round(a["r_look"], 4),
                "p_n": round(a["p_n"], 4),
                "q_n": round(a["q_n"], 4),
                "o_n": round(a["o_n"], 4),
                "stage1_score_prior": STAGE1_SCORE_PRIOR.get(model, ""),
                "nps_prior": NPS_PRIOR.get(model, ""),
                "stage2_eligible": model in STAGE2_TOP7,
            })

    # Compact markdown report.
    md = os.path.join(REPORTS, "stage1_report.md")
    with open(md, "w") as handle:
        handle.write("# Stage 1 data report\n\n")
        handle.write(f"Completed models: {len(complete)} (425 episodes each).\n\n")
        handle.write("S0/Sfinal and NPS are recomputed from the formulas recovered from the prior session; "
                     "the earlier recorded values are retained as `stage1_score_prior`/`nps_prior` for comparison. "
                     "The old recorded values differ slightly because the earlier pass used a slightly different component definition; "
                     "the new columns are the consistent recomputation used for the new models.\n\n")
        handle.write("Narrow = levels 11-16 (A/S 0.9-0.4). Wide = levels 0-10 (A/S 2.0-1.0).\n\n")
        handle.write("## Recovered formulas\n\n")
        handle.write("Stage-1 S0 (per-episode quality, difficulty weighted):\n\n")
        handle.write("```\n")
        handle.write("over  = max(0, theta_max - theta_req) / 15\n")
        handle.write("under = max(0, theta_req - theta_max) / 15\n")
        handle.write("extra = max(0, n_turn - min_turns(W))\n")
        handle.write("excess = max(0, total_steps - optimal_steps(W))\n")
        handle.write("q_rot   = exp(-0.20*over - 0.03*over^2 - 0.20*under)\n")
        handle.write("q_turn  = exp(-0.12*extra)\n")
        handle.write("q_look  = 1.00 timely, 0.95 late, 0.90 none, 0.80 >2 looks\n")
        handle.write("q_eff   = exp(-0.02*excess/optimal)\n")
        handle.write("q_col   = exp(-0.05*wall_collisions)\n")
        handle.write("s_i     = 100 * passed * q_rot*q_turn*q_look*q_eff*q_col\n")
        handle.write("w_l     = 0.05 + 1.0*(theta_req/90)\n")
        handle.write("S0      = sum_l w_l * mean(s_i in level l) / sum_l w_l\n")
        handle.write("```\n\n")
        handle.write("Model gates: if wide-turn rate > 0.10 or narrow look rate < 0.10, Sfinal=min(S0,20); "
                     "elif wide-turn > 0.05 or look < 0.25, Sfinal=min(S0,50).\n\n")
        handle.write("NPS (narrow-only; not Net Promoter Score):\n\n")
        handle.write("```\n")
        handle.write("P_n = narrow passes / 150\n")
        handle.write("q_rot   = exp(-0.20*over - 0.03*over^2)\n")
        handle.write("q_extra = exp(-0.12*max(0,n_turn-min_turns))\n")
        handle.write("q_eff   = min(1, optimal_steps/total_steps)\n")
        handle.write("q_look  = 1.00 timely, 0.95 late, 0.90 none, 0.80 >2 looks\n")
        handle.write("Q_n     = mean(q_rot*q_extra*q_eff*q_look) over passed narrow\n")
        handle.write("O_n     = fraction of passed narrow with over > 0\n")
        handle.write("NPS     = 100 * P_n * Q_n * (1 - O_n)\n")
        handle.write("```\n\n")
        handle.write("Eligibility: n_narrow_pass >= 10 and NPS >= 20.\n\n")
        handle.write("## Completed models, combined (ALL scenes)\n\n")
        handle.write("| model | pass% | wide% | narrow | narrow% | turns% | look% | invalid | S0 | Sfinal | NPS | Stage2 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for model in sorted(complete):
            r = next(x for x in rows if x["model"] == model and x["scene"] == "ALL")
            handle.write(
                f"| {model} | {r['pass_pct']:.1f} | {r['wide_pass_pct']:.1f} | "
                f"{r['narrow_passed']}/{r['narrow_n']} | {r['narrow_pass_pct']:.1f} | "
                f"{r['turned_pct']:.1f} | {r['looked_pct']:.1f} | {r['invalid']} | "
                f"{r['stage1_s0']:.1f} | {r['stage1_sfinal']:.1f} | {r['nps']:.1f} | "
                f"{'YES' if r['narrow_passed'] >= 10 and r['nps'] >= 20 else ''} |\n"
            )
        handle.write("\n## Five scenes per model\n\n")
        handle.write("The full model x scene table is in `stage1_model_scene_metrics.csv` (5 scene rows + ALL).\n\n")
        handle.write("## Stage 2\n\n")
        handle.write("Stage 2 is the new single-memory/wide-A/S-2.0 experiment. The old 12-attempt/note scenario is archived and is not used.\n\n")
        handle.write("- design: `STAGE2_DESIGN.md`\n")
        handle.write("- memory cards: `stage2_memory_cards_top7_15_blind.jsonl`\n")
        handle.write("- test plan: `stage2_top7_15_test_plan.csv`\n")
        handle.write("- schema: `stage2_memory_schema.md`\n\n")
        if partial:
            handle.write("## Partial model excluded\n\n")
            for model, es in sorted(partial.items()):
                handle.write(f"- {model}: {len(es)}/425 episodes (running; not in completed comparisons)\n")

    print(f"complete models: {len(complete)}")
    for model in sorted(complete):
        r = next(x for x in rows if x["model"] == model and x["scene"] == "ALL")
        print(f"  {model:38s} pass={r['pass_pct']:5.1f}% narrow={r['narrow_passed']:3d}/{r['narrow_n']:3d} invalid={r['invalid']:3d}")
    print(f"wrote {scene_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
