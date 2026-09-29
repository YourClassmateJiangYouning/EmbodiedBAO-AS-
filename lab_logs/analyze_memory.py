"""Per-round table of the Stage 2 experiment, read live from the records on disk.

Answers, per attempt: what happened (passed or not, in how many steps), whether it
rotated and how much, what its best posture at the wall was (gap), what kind of
attempt it was, and what it wrote to itself.  Then aggregates the things Stage 2 is
scored on: the turn rate, the share of attempts that turned *enough*, where the
first turn happened, and how many of the 12 learning attempts succeeded.

It reads whatever is on disk at the moment, so it can be run while a sweep is
running and re-run as often as wanted; `--watch` refreshes it in place.

    python3 lab_logs/analyze_memory.py
    python3 lab_logs/analyze_memory.py --model qwen3-vl-32b-instruct
    python3 lab_logs/analyze_memory.py --watch 60
    python3 lab_logs/analyze_memory.py --reasoning --full-notes
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import time

LEARNING_ROUNDS = 12
ROUNDS_PER_RUN = 17
REASONING = re.compile(r'"reasoning":\s*"((?:[^"\\]|\\.)*)"')


def load_records(root: str):
    """Every finished round under ``root``, grouped by (model, tag) and ordered."""
    grouped: dict = collections.defaultdict(list)
    for path in glob.glob(os.path.join(root, "*", "*", "run*_round*.json")):
        if path.endswith("_steps.json"):
            continue
        parts = path.replace(os.sep, "/").split("/")
        model, tag = parts[-3], parts[-2]
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue  # half-written or quarantined; it will appear next time
        record["_path"] = path
        grouped[(model, tag)].append(record)
    for key in grouped:
        grouped[key].sort(key=lambda r: (int(r.get("run", 0)), int(r.get("round", 0))))
    return grouped


def fmt_gap(value) -> str:
    if value is None:
        return "   n/a "
    return "%+7.4f" % float(value)


def note_of(record: dict) -> str:
    return str(record.get("note_text") or "").strip()


def last_reasoning(record: dict, logs_root: str, width: int) -> str:
    """The model's own last reasoned step of an attempt, from its agent log."""
    tag = str(record.get("tag") or "")
    name = os.path.basename(record["_path"]).replace(".json", "_agent.txt")
    log = os.path.join(logs_root, tag, name)
    if not os.path.exists(log):
        return ""
    try:
        with open(log, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except Exception:
        return ""
    found = [r for r in REASONING.findall(text) if r != "<brief reasoning>"]
    if not found:
        return ""
    snippet = " ".join(found[-1].split())
    return snippet[: width - 1] + "\u2026" if len(snippet) > width else snippet


def summarise(records) -> dict:
    learning = [r for r in records if int(r.get("round", 0)) <= LEARNING_ROUNDS]
    probe = [r for r in records if int(r.get("round", 0)) > LEARNING_ROUNDS]
    turned = [r for r in records if int(r.get("n_turn", 0)) >= 1]
    enough = [
        r
        for r in records
        if int(r.get("n_turn", 0))
        >= (5 if int(r.get("round", 0)) <= LEARNING_ROUNDS else 4)
    ]
    sideways = [r for r in records if int(r.get("n_lateral", 0)) >= 3]
    look_down = [r for r in records if int(r.get("n_look_down", 0)) >= 1]
    bands = collections.Counter()
    for record in records:
        first = record.get("first_turn_x")
        if record.get("first_turn_step") is None or first is None:
            bands["never"] += 1
        elif float(first) < 4.0:
            bands["x<4"] += 1
        elif float(first) < 7.0:
            bands["mid"] += 1
        else:
            bands["x>=7"] += 1
    mean = lambda values: (sum(values) / len(values)) if values else 0.0  # noqa: E731
    return {
        "rounds": len(records),
        "passed": sum(1 for r in records if r.get("passed")),
        "d": sum(1 for r in learning if r.get("passed")),
        "learning": len(learning),
        "probe": len(probe),
        "turned": len(turned),
        "enough": len(enough),
        "sideways": len(sideways),
        "look_down": len(look_down),
        "mean_turns": mean([float(r.get("n_turn", 0)) for r in records]),
        "mean_max_rot": mean([float(r.get("max_rotation_deg", 0.0)) for r in records]),
        "mean_steps": mean([float(r.get("total_steps", 0)) for r in records]),
        "bands": bands,
    }


def print_tag(model: str, tag: str, records, args) -> None:
    mode = "rolling" if tag.endswith("-roll") else "cumulative"
    print()
    print("=" * 132)
    print(f"{model}   tag {tag}   ({mode} memory)   {len(records)} round(s) on disk")
    print("=" * 132)
    header = (
        f"{'run':>3} {'rnd':>3} {'ph':>2} {'A/S':>5} {'pass':>4} {'steps':>5} "
        f"{'turn':>4} {'lat':>3} {'lkd':>3} {'gap':>8} {'label':<10} "
        f"{'maxrot':>6} {'first@x':>7} {'note':>6}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        first = record.get("first_turn_x")
        print(
            "%3d %3d %2s %5.2f %4s %5d %4d %3d %3d %8s %-10s %6.1f %7s %5dch"
            % (
                int(record.get("run", 0)),
                int(record.get("round", 0)),
                str(record.get("phase", "?")),
                float(record.get("a_s_ratio", 0.0)),
                "P" if record.get("passed") else ".",
                int(record.get("total_steps", 0)),
                int(record.get("n_turn", 0)),
                int(record.get("n_lateral", 0)),
                int(record.get("n_look_down", 0)),
                fmt_gap(record.get("gap")),
                str(record.get("strategy_label", "?")),
                float(record.get("max_rotation_deg", 0.0)),
                "never" if first is None else "%.2f" % float(first),
                int(record.get("note_chars", 0)),
            )
        )

    stats = summarise(records)
    print(
        "\n  learning: d=%d/%d passed | probe: %d round(s) | turns: mean %.1f, "
        "turned %.0f%%, enough %.0f%%, lateral>=3 %.0f%%, look_down %.0f%%"
        % (
            stats["d"],
            stats["learning"],
            stats["probe"],
            stats["mean_turns"],
            100.0 * stats["turned"] / max(stats["rounds"], 1),
            100.0 * stats["enough"] / max(stats["rounds"], 1),
            100.0 * stats["sideways"] / max(stats["rounds"], 1),
            100.0 * stats["look_down"] / max(stats["rounds"], 1),
        )
    )
    print(
        "  first turn at: never %d, x<4 %.0f%% (%d), 4<=x<7 %d, x>=7 %d | "
        "mean max rotation %.1f deg | mean steps %.1f"
        % (
            stats["bands"]["never"],
            100.0 * stats["bands"]["x<4"] / max(stats["rounds"], 1),
            stats["bands"]["x<4"],
            stats["bands"]["mid"],
            stats["bands"]["x>=7"],
            stats["mean_max_rot"],
            stats["mean_steps"],
        )
    )

    if args.notes:
        print("\n  what it wrote to itself:")
        for record in records:
            note = note_of(record)
            if not note:
                continue
            shown = note if args.full_notes else (
                note[: args.width - 1] + "\u2026" if len(note) > args.width else note
            )
            outcome = (
                f"passed in {int(record.get('total_steps', 0))}"
                if record.get("passed")
                else f"failed, {int(record.get('total_steps', 0))} steps"
            )
            print(
                "    run %d round %02d (%s): %s"
                % (int(record.get("run", 0)), int(record.get("round", 0)), outcome, shown)
            )
    if args.reasoning:
        print("\n  last reasoned step of each attempt:")
        for record in records:
            snippet = last_reasoning(record, args.logs, args.width)
            if snippet:
                print(
                    "    run %d round %02d: %s"
                    % (int(record.get("run", 0)), int(record.get("round", 0)), snippet)
                )


def render(args) -> int:
    grouped = load_records(args.root)
    if not grouped:
        print(f"no round records under {args.root} yet")
        return 1
    models = sorted({model for model, _ in grouped})
    for model in models:
        if args.model and model != args.model:
            continue
        for _, tag in sorted(key for key in grouped if key[0] == model):
            print_tag(model, tag, grouped[(model, tag)], args)
    if args.model and args.model not in models:
        print(f"{args.model} has no round records under {args.root}")
        return 1
    print()
    print(
        "  reading: %s   |  %s"
        % (os.path.abspath(args.root), time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    print("  P = passed, ph = A learning (A/S 0.80) or B probe (A/S 1.10), "
          "turn/lat/lkd = turn / lateral / look_down actions")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=os.path.join("results", "memory"))
    parser.add_argument("--logs", default="logs",
                        help="where the per-attempt agent logs live")
    parser.add_argument("--model", default="", help="only this model")
    parser.add_argument("--notes", dest="notes", action="store_true", default=True)
    parser.add_argument("--no-notes", dest="notes", action="store_false")
    parser.add_argument("--full-notes", action="store_true")
    parser.add_argument("--reasoning", action="store_true",
                        help="also print the model's last reasoned step per attempt")
    parser.add_argument("--width", type=int, default=220, help="snippet width")
    parser.add_argument("--watch", type=int, default=0,
                        help="refresh every N seconds instead of printing once")
    args = parser.parse_args()
    if args.watch > 0:
        try:
            while True:
                if sys.stdout.isatty():
                    os.system("clear")
                render(args)
                sys.stdout.flush()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
        return 0
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
