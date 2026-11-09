"""Stage 4B: does a narrow-gap success memory change behaviour at a wide gap?

Mirrors Stage 2's design with the digest removed (Stage 2's digest sub-call was built with a prompt that
omitted the memory, so it is treated as never having existed):

    baseline  A/S 2.0, mode A, no memory
    memory    A/S 2.0, mode B, the model's own successful narrow-gap episode as a blind action transcript

The measure is behavioural: how much does the model turn at a gap it can walk straight through?
The simulator needed no rotation at A/S 2.0 and passes 105/105 either way, so pass rate carries no signal.

Usage:
    python stage4/memory_runner.py --dry-run
    python stage4/memory_runner.py --models gemini-3-flash-preview
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import core  # noqa: E402
import replay_runner as runner  # noqa: E402

SOURCE_ROOT = os.path.join(HERE, "sessions_r1")      # the Stage-4A sessions that hold the narrow successes
OUT_ROOT = os.path.join(HERE, "sessions_b")          # this experiment's sessions
TEST_A_S = 2.0                                       # wide gap: no rotation needed
LEVEL_LABELS = {0.8: "08", 0.6: "06"}


def memory_sources():
    """Every successful narrow-gap episode, keyed by (model, marker) with its source level.

    The action transcript of a success is what Stage 2 fed back; the level it came from is kept in the key so a
    cell that succeeded twice (at 0.8 and at 0.6) yields two independent memory cards.
    """
    old = ("the opening is narrower than your body", "turning here would strike the wall panels")
    found = []
    if not os.path.isdir(SOURCE_ROOT):
        return found
    for directory in sorted(os.listdir(SOURCE_ROOT)):
        path = os.path.join(SOURCE_ROOT, directory)
        steps_file = os.path.join(path, "steps.jsonl")
        manifest = os.path.join(path, "manifest.csv")
        if not (os.path.isdir(path) and os.path.isfile(steps_file) and os.path.isfile(manifest)):
            continue
        with open(steps_file, encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle if line.strip()]
        if any(any(marker in (row.get("feedback") or "") for marker in old) for row in steps):
            continue
        with open(manifest, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        episode = rows[-1]
        if str(episode.get("a_s")) not in ("0.8", "0.6"):
            continue
        if str(episode.get("passed")).lower() != "true":
            continue
        found.append({
            "model": episode.get("model"),
            "marker": episode.get("marker"),
            "level": float(episode.get("a_s")),
            "session": directory,
            "episode_id": episode.get("episode_id"),
        })
    return found


def descriptor_for(slot: str) -> str:
    return core.MARKER_PRESETS[int(slot[1:]) - 1]


def test_id(spec: dict, condition: str) -> str:
    """Unique directory per cell AND per memory source.

    The first version keyed on spec["level"], which is always the TEST level (2.0), so the memory card from
    0.8 and the one from 0.6 landed in the same directory and the second episode was appended to the first
    (22 steps in one session instead of 11 in each).
    """
    tag = "base" if condition == "baseline" else LEVEL_LABELS.get(spec.get("source_level"), "src")
    return "b-%s-%s-%s-%s-%s" % (
        condition,
        spec["model"].split("/")[0].replace("-", "")[:18],
        tag,
        spec["marker"],
        "test2",
    )


def build_card(source: dict) -> dict:
    """Rebuild the blind memory card from the source episode's own steps."""
    session = core.Stage4Session(sessions_root=SOURCE_ROOT)
    session.open_session(source["session"], {
        "scene": "R1", "marker": source["marker"], "marker_descriptor": descriptor_for(source["marker"]),
        "mode": "A", "model": source["model"], "a_s": source["level"],
    })
    episode = next((e for e in session.episodes if e.get("episode_id") == source["episode_id"]), None)
    if episode is None:
        raise RuntimeError("source episode %s not found in %s" % (source["episode_id"], source["session"]))
    return session.build_memory_from_episode(
        episode, expect_scene=episode.get("scene"), expect_marker=source["marker"])


def plan():
    """(spec, condition, source) triples: one baseline per cell, one memory run per card."""
    sources = memory_sources()
    cells = {}
    for source in sources:
        cells.setdefault((source["model"], source["marker"]), []).append(source)
    todo = []
    for (model, marker), cards in sorted(cells.items()):
        base = {"model": model, "marker": marker, "level": TEST_A_S}
        todo.append((base, "baseline", None))
        for card in sorted(cards, key=lambda c: c["level"]):
            todo.append(({"model": model, "marker": marker, "level": TEST_A_S, "source_level": card["level"]},
                         "memory", card))
    return todo


def run_one(spec: dict, condition: str, source: dict, verbose: bool = False) -> dict:
    session = core.Stage4Session(sessions_root=OUT_ROOT)
    session.open_session(test_id(spec, condition), {
        "scene": "R1", "marker": spec["marker"], "marker_descriptor": descriptor_for(spec["marker"]),
        "mode": "B" if condition == "memory" else "A",
        "model": spec["model"], "a_s": TEST_A_S,
        "max_steps": runner.MAX_STEPS, "disable_thinking": True,
        "shoulder_m": runner.S, "move_step": runner.STEP,
    })
    if condition == "memory":
        session.memory = build_card(source)
    session.start_episode()

    x, theta = runner.START_X, 0.0
    for step in range(runner.MAX_STEPS):
        session.capture_from_bank(runner.BANK)
        session.call_model()
        record = session.history[-1]
        action = record.get("action")
        if action is None:
            x, theta, feedback, collision = x, theta, runner.FEEDBACK_INVALID, False
        else:
            x, theta, feedback, blocked, collision = runner.apply_action(action, x, theta, TEST_A_S)
            session.position_x, session.torso_rotation = x, theta
        session.action_executed(feedback=feedback, collision=collision, position_x=x,
                                torso_rotation=theta, note="stage4b", advance_odometer=False)
        if x >= runner.SUCCESS_X - 1e-9 or session.episode is None:
            break
    if session.episode is not None:
        session.finish(passed=x >= runner.SUCCESS_X - 1e-9,
                       end_reason="passed" if x >= runner.SUCCESS_X - 1e-9 else "max_steps")

    steps = session.history
    turns = sum(1 for row in steps if (row.get("action") or "").startswith("turn"))
    rotations = [abs(float(row.get("torso_rotation") or 0)) for row in steps]
    folded = [min(r % 360, 360 - r % 360) for r in rotations] or [0.0]
    return {
        "model": spec["model"], "marker": spec["marker"], "condition": condition,
        "source_level": spec.get("source_level"),
        "passed": x >= runner.SUCCESS_X - 1e-9,
        "steps": len(steps), "turns": turns, "max_rotation": max(folded),
        "turned": turns > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    todo = plan()
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        todo = [item for item in todo if item[0]["model"] in wanted]
    if args.limit:
        todo = todo[:args.limit]

    print("cells:", len(todo))
    print("  baseline:", sum(1 for _, c, _ in todo if c == "baseline"))
    print("  memory  :", sum(1 for _, c, _ in todo if c == "memory"))
    if args.dry_run:
        for spec, condition, source in todo:
            print("  %-8s %-30s %-4s A/S %.1f %s" % (
                condition, spec["model"][:30], spec["marker"], spec["level"],
                ("<- %.1f" % source["level"]) if source else ""))
        return 0

    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for index, (spec, condition, source) in enumerate(todo, 1):
        print("[%3d/%3d] %-8s %-30s %-4s" % (index, len(todo), condition, spec["model"][:30], spec["marker"]))
        try:
            result = run_one(spec, condition, source)
        except Exception as error:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            result = {"model": spec["model"], "marker": spec["marker"], "condition": condition,
                      "error": "%s: %s" % (type(error).__name__, error)}
        results.append(result)
        print("      -> turned=%-5s turns=%-2s max_rot=%-5s steps=%-2s passed=%s"
              % (result.get("turned"), result.get("turns"), result.get("max_rotation"),
                 result.get("steps"), result.get("passed")))

    summary = os.path.join(OUT_ROOT, "stage4b_summary.json")
    with open(summary, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print("\nsummary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
