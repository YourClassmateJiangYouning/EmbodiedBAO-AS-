"""Export the Stage 2 wide tables: one row per round, one row per run.

The column names for everything the two experiments share are the ones the Stage 1
exporter uses (``final_position_x``, ``wall_collisions``, ...), and both tables also
carry ``model`` and ``family``, so the three CSVs -- 660 episodes, Stage 2 rounds,
Stage 2 runs -- can be concatenated or joined without renaming anything.

Written from the round records on disk, so it can be re-run while a sweep is in
flight and after it, and the paper's figures are generated from the runs table alone.

    python lab_logs/export_memory_table.py
    python lab_logs/export_memory_table.py --model qwen3-vl-32b-instruct
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory_metrics as mm  # noqa: E402
import memory_protocol as mp  # noqa: E402

FAMILY = {
    "qwen3-vl-235b-a22b-instruct": "Qwen",
    "qwen3-vl-32b-instruct": "Qwen",
    "qwen-vl-max": "Qwen",
    "gemini-2.5-pro": "Gemini",
    "gemini-2.5-flash": "Gemini",
    "gpt-4.1": "OpenAI",
    "gpt-4o": "OpenAI",
    "gpt-4o-mini": "OpenAI",
    "claude-sonnet-4-6": "Anthropic",
    "deepseek-v4.1-flash": "DeepSeek",
    "glm-4.6v": "Zhipu",
}

# Stage 1 column names are kept for the shared fields, so the tables line up.
ROUND_COLUMNS = [
    "model", "family", "tag", "memory_mode", "run", "round", "phase",
    "a_s_ratio", "channel_width", "passed", "passed_sideways", "total_steps",
    "end_reason", "n_forward", "n_backward", "n_lateral", "n_turn", "n_look_down",
    "n_glance", "turned", "total_rotation", "max_rotation_deg",
    "passage_rotation_deg", "final_torso_rotation", "first_turn_step", "first_turn_x",
    "first_sideways_step", "final_position_x", "final_position_z", "wall_collisions",
    "invalid_response_count", "reached_door", "optimal_steps", "gap", "excess",
    "strategy_label", "note_chars", "memory_injected_chars", "total_llm_time_ms",
    "note_llm_time_ms", "action_sequence",
]

RUN_COLUMNS = [
    "model", "family", "tag", "memory_mode", "run", "rounds", "d",
    "first_pass_round", "acquired", "terminal_state", "curve_label", "curve_I",
    "curve_D", "curve_S", "curve_rho", "improving_rounds", "jump_round",
    "valued_rounds", "set_index", "approach_latency", "never_approached",
    "excess_total", "invalid_response_count",
] + ["probe_rotation_%d" % i for i in range(1, mp.ROUNDS_PROBE + 1)] + [
    "probe_turn_%d" % i for i in range(1, mp.ROUNDS_PROBE + 1)
] + [
    "probe_look_down_%d" % i for i in range(1, mp.ROUNDS_PROBE + 1)
]


def load_records(root: str, model_filter: str = "") -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "run*_round*.json"))):
        if path.endswith("_steps.json"):
            continue
        parts = path.replace(os.sep, "/").split("/")
        model = parts[-3]
        if model_filter and model != model_filter:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue
        record["_path"] = path
        records.append(record)
    return records


def sidecar_actions(path: str) -> str:
    sidecar = path.replace(".json", "_steps.json")
    try:
        with open(sidecar, encoding="utf-8") as handle:
            return ",".join(str(step.get("action", "?")) for step in json.load(handle))
    except Exception:
        return ""


def round_row(record: dict) -> dict:
    width = mm.SHOULDER_WIDTH_M * float(record.get("a_s_ratio", 0.0))
    row = {name: "" for name in ROUND_COLUMNS}
    row.update(
        {
            "model": record.get("model", ""),
            "family": FAMILY.get(record.get("model", ""), record.get("model", "")),
            "tag": record.get("tag", ""),
            "memory_mode": record.get("memory_mode", ""),
            "run": record.get("run", ""),
            "round": record.get("round", ""),
            "phase": record.get("phase", ""),
            "a_s_ratio": record.get("a_s_ratio", ""),
            "channel_width": round(width, 6),
            "passed": bool(record.get("passed")),
            "passed_sideways": bool(record.get("passed_sideways")),
            "total_steps": record.get("total_steps", ""),
            "end_reason": record.get("end_reason", ""),
            "n_forward": record.get("n_forward", ""),
            "n_backward": record.get("n_backward", ""),
            "n_lateral": record.get("n_lateral", ""),
            "n_turn": record.get("n_turn", ""),
            "n_look_down": record.get("n_look_down", ""),
            "n_glance": record.get("n_glance", ""),
            "turned": bool(record.get("turned")),
            "total_rotation": record.get("total_rotation", ""),
            "max_rotation_deg": record.get("max_rotation_deg", ""),
            "passage_rotation_deg": record.get("passage_rotation_deg", ""),
            "final_torso_rotation": record.get("final_torso_rotation", ""),
            "first_turn_step": record.get("first_turn_step", ""),
            "first_turn_x": record.get("first_turn_x", ""),
            "first_sideways_step": record.get("first_sideways_step", ""),
            "final_position_x": record.get("final_x", ""),
            "final_position_z": record.get("final_z", ""),
            "wall_collisions": record.get("wall_collision_count", ""),
            "invalid_response_count": record.get("invalid_response_count", ""),
            "reached_door": bool(record.get("reached_door")),
            "optimal_steps": record.get("optimal_steps", ""),
            # Empty rather than 0 or -1 when the attempt never got near the wall: an
            # imputed gap is a number the analysis would happily average.
            "gap": "" if record.get("gap") is None else record.get("gap"),
            "excess": record.get("excess", ""),
            "strategy_label": record.get("strategy_label", ""),
            "note_chars": record.get("note_chars", 0),
            "memory_injected_chars": record.get("memory_injected_chars", 0),
            "total_llm_time_ms": round(float(record.get("total_llm_time_ms", 0.0)), 1),
            "note_llm_time_ms": round(float(record.get("note_llm_time_ms", 0.0)), 1),
            "action_sequence": sidecar_actions(record["_path"]),
        }
    )
    return row


def run_rows(records: list) -> list:
    grouped: dict = collections.defaultdict(list)
    for record in records:
        grouped[(record.get("model", ""), record.get("tag", ""), int(record.get("run", 0)))].append(
            record
        )
    rows = []
    for (model, tag, run) in sorted(grouped):
        rows_for_run = grouped[(model, tag, run)]
        summary = mp.summarise_run(rows_for_run)
        curve = dict(summary["curve"])
        row = {name: "" for name in RUN_COLUMNS}
        row.update(
            {
                "model": model,
                "family": FAMILY.get(model, model),
                "tag": tag,
                "memory_mode": rows_for_run[0].get("memory_mode", ""),
                "run": run,
                "rounds": summary["rounds"],
                "d": summary["d"],
                "first_pass_round": "" if summary["first_pass_round"] is None
                else summary["first_pass_round"],
                "acquired": summary["acquired"],
                "terminal_state": summary["terminal_state"],
                "curve_label": curve.get("label", ""),
                "curve_I": curve.get("I", ""),
                "curve_D": curve.get("D", ""),
                "curve_S": curve.get("S", ""),
                "curve_rho": curve.get("rho", ""),
                "improving_rounds": curve.get("improving_rounds", ""),
                "jump_round": "" if curve.get("jump_round") is None else curve["jump_round"],
                "valued_rounds": curve.get("valued_rounds", ""),
                "set_index": curve.get("set_index", ""),
                "approach_latency": curve.get("approach_latency", ""),
                "never_approached": curve.get("never_approached", ""),
                "excess_total": sum(
                    int(r.get("excess", 0) or 0) for r in rows_for_run if r.get("passed")
                ),
                "invalid_response_count": sum(
                    int(r.get("invalid_response_count", 0) or 0) for r in rows_for_run
                ),
            }
        )
        for index, value in enumerate(summary["probe_rotation_deg"], start=1):
            row["probe_rotation_%d" % index] = round(float(value), 3)
        for index, value in enumerate(summary["probe_turn"], start=1):
            row["probe_turn_%d" % index] = int(value)
        for index, value in enumerate(summary["probe_look_down"], start=1):
            row["probe_look_down_%d" % index] = int(value)
        rows.append(row)
    return rows


def write_csv(path: str, columns: list, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=os.path.join("results", "memory"))
    parser.add_argument("--model", default="")
    parser.add_argument("--rounds-out", default=os.path.join(
        "lab_logs", "embodiedbao_stage2_rounds.csv"))
    parser.add_argument("--runs-out", default=os.path.join(
        "lab_logs", "embodiedbao_stage2_runs.csv"))
    args = parser.parse_args()

    records = load_records(args.root, args.model)
    if not records:
        print(f"no round records under {args.root}")
        return 1
    rounds = sorted(
        (round_row(r) for r in records),
        key=lambda row: (row["model"], row["memory_mode"], int(row["run"]), int(row["round"])),
    )
    runs = run_rows(records)
    write_csv(args.rounds_out, ROUND_COLUMNS, rounds)
    write_csv(args.runs_out, RUN_COLUMNS, runs)
    fallbacks = [r for r in records if int(r.get("invalid_response_count", 0) or 0)
                 >= int(r.get("total_steps", 0) or 0) > 0]
    print(
        "wrote %s (%d rounds) and %s (%d runs)"
        % (args.rounds_out, len(rounds), args.runs_out, len(runs))
    )
    if fallbacks:
        print(
            "  !! WARNING: %d round(s) are entirely fallback steps -- the model never "
            "answered, and their gaps and labels are meaningless"
            % len(fallbacks)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
