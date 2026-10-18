"""Turn a run's model I/O into the table the reasoning analysis needs.

WHERE THE THINKING ACTUALLY LIVES
    The authoritative per-step record is
    ``results/level{N}/{model}/{tag}/episode_{id:03d}_steps.json``: every step in it carries its own
    ``reasoning``, ``response``, ``scene_description`` and ``confidence``, with the step number the
    runner assigned.  This tool reads that file FIRST.

WHY THE LOG IS ONLY A FALLBACK
    ``logs/{tag}/level{N}_episode{id:03d}_agent.txt`` records the prompt and the reply, but its
    layout changed.  In OLD logs the model client wrote each reply and the runner wrote the prompt
    plus the same reply again: every successful step appeared twice, and the prompt came AFTER the
    reply it produced.  The current runner writes the prompt BEFORE the call and the client writes
    the reply, so each step appears once and in order.

    The fallback below handles both: it keeps prompt-bearing blocks for the old layout, and when
    there are none (the current layout) it keeps the unique replies and ignores the bare prompt
    blocks.  The sidecar above remains the primary source either way.

The marker a run used is derivable rather than recorded: ``slot = (episode_id % 5) + 1``, the same
mapping ``scenes.describe_marker`` expects.  So each reply is attributed to the object that was
actually on the wall.

    python3 tools/parse_agent_logs.py --tag full_Qwen-Qwen3-VL-8B-Instruct-v8-actions-flipped-stage1.1
    python3 tools/parse_agent_logs.py --tag full_Qwen-Qwen3-VL-8B-Instruct-v8-actions-flipped-stage1.1 --csv logs_reasoning.csv
    python3 tools/parse_agent_logs.py --results results --scene stage1.1

Prints, per model/tag: how often its own words mention the marker's colour and shape, whether it
named the opening, and a few verbatim examples.  A model that never names its marker is not looking
at it, whatever its feet are doing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import scenes as sc
except Exception:  # noqa: BLE001
    sc = None

RESPONSE = re.compile(r"MODEL RESPONSE:\s*")
NAME = re.compile(r"level(\d+)_episode(\d+)_agent\.txt$")
SCENE = re.compile(r"(stage1\.\d+)")
PROMPT_MARKER = "You are a Unitree H1"
SEPARATOR = re.compile(r"^-{40,}\s*$", re.MULTILINE)


def extract_responses(text: str) -> list:
    """Every JSON object that follows MODEL RESPONSE:, found by brace balance.

    A regex is not enough, and this was a real bug rather than a theoretical one: the replies are
    usually pretty-printed over many lines, so a pattern anchored on a newline before the closing
    brace passes every hand check -- and silently returns nothing for a reply that arrived on one
    line, which models do sometimes do.  Counting braces handles both.
    """
    found = []
    for match in RESPONSE.finditer(text):
        start = text.find("{", match.end())
        if start < 0:
            continue
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start:index + 1])
                    break
    return found


def extract_prompted_responses(text: str) -> list:
    """Replies from the runner's blocks only: a block is kept when it carries a prompt.

    The current log layout is

        MODEL RESPONSE: <client copy>
        ----------------------------------------
        <prompt>
        MODEL RESPONSE: <runner copy>
        ----------------------------------------

    so the client copy has no prompt in its block and the runner copy does.  Keeping the prompted
    blocks removes the duplicate without having to compare reply bodies, which is what makes it safe
    when a model legitimately repeats the same action twice.
    """
    found = []
    for block in SEPARATOR.split(text):
        if PROMPT_MARKER not in block:
            continue
        found.extend(extract_responses(block))
    return found


def _canonical(blob: str) -> str:
    try:
        return json.dumps(json.loads(blob), sort_keys=True, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - an unparseable reply is compared by its text
        return blob.strip()


def _dedupe(blobs: list) -> list:
    """Remove exact duplicate replies, for logs that predate the prompt-block layout."""
    seen = set()
    kept = []
    for blob in blobs:
        key = _canonical(blob)
        if key in seen:
            continue
        seen.add(key)
        kept.append(blob)
    return kept

# The inverse of scenes.COLOUR_NAMES, and it has to be kept in step with it: the marker
# phrase is built from that table, the first word of the phrase is looked up here, and the
# result is the letter searched for in the model's own reasoning.  "white"->w and
# "black"->k were wrong in both places -- w is a wine red and k is pink -- so the analysis
# searched for the word "white" in a run whose prompt had said "white set of two bars"
# while a dark red plate was on the wall.  Both tables now agree with the RGB values, and
# test_bao_scenes.py checks that agreement rather than this comment.
COLOURS = {"red": "r", "magenta": "m", "orange": "o", "lime": "l", "cyan": "c",
           "purple": "p", "green": "g", "wine": "w", "teal": "t", "pink": "k"}


def marker_for(scene: str, episode: int) -> str:
    if sc is None:
        return f"slot{(episode % 5) + 1}"
    try:
        return sc.describe_marker(scene, (episode % 5) + 1)
    except Exception:  # noqa: BLE001
        return f"slot{(episode % 5) + 1}"


def scene_from_tag(tag: str, fallback: str = "") -> str:
    match = SCENE.search(tag or "")
    return match.group(1) if match else (fallback or "stage1.1")


def model_from_tag(tag: str) -> str:
    """The model part of a tag, with the protocol suffix removed when it is a known one."""
    for suffix in ("-v8-actions-flipped", "-v7-state-axes", "-v6-marker-ahead", "-v5-12widths"):
        if suffix in tag:
            return tag.split(suffix)[0]
    return tag


def model_for_tag(level: int, tag: str, results_root: str) -> str:
    """The model DIRECTORY for one tag, or the tag prefix when the directory is absent.

    The tag's leading text is a run label (``full_``, ``cost_``, ...), not the model.  The
    results directory has the real model slug, so it wins whenever it exists; only a log with no
    results tree falls back to stripping the protocol suffix from the tag.
    """
    pattern = os.path.join(results_root, f"level{level}", "*", tag)
    matches = sorted(path for path in glob.glob(pattern) if os.path.isdir(path))
    if matches:
        return os.path.basename(os.path.dirname(matches[0]))
    return model_from_tag(tag)


def _row(level: int, episode: int, step: int, marker: str, reply: dict) -> dict:
    return {
        "level": level,
        "episode": episode,
        "step": step,
        "marker": marker,
        "action": str(reply.get("action", "")),
        "confidence": reply.get("confidence", ""),
        "scene_description": str(reply.get("scene_description", "")),
        "reasoning": str(reply.get("reasoning", "")),
    }


def parse_sidecar(path: str, level: int, episode: int, scene: str) -> list:
    """Read one ``episode_*_steps.json`` -- the canonical per-step record."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    steps = payload.get("steps") if isinstance(payload, dict) else payload
    if not isinstance(steps, list):
        return []
    marker = marker_for(scene, episode)
    rows = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        rows.append(
            _row(
                level,
                episode,
                int(step.get("step", index)),
                marker,
                step,
            )
        )
    return rows


def parse_file(path: str, scene: str) -> list:
    """Every step of one episode from the agent log, with the duplicate copy removed."""
    match = NAME.search(os.path.basename(path))
    if not match:
        return []
    level, episode = int(match.group(1)), int(match.group(2))
    marker = marker_for(scene, episode)
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    blobs = extract_prompted_responses(text)
    if not blobs:
        # An older log without the runner's prompt block: every reply is present, possibly twice.
        blobs = _dedupe(extract_responses(text))
    rows = []
    for index, blob in enumerate(blobs):
        try:
            reply = json.loads(blob)
        except Exception:  # noqa: BLE001
            continue
        rows.append(_row(level, episode, index, marker, reply))
    return rows


def sidecar_for(tag: str, level: int, episode: int, results_root: str) -> str:
    """Find the sidecar for one episode, across the model directory (the slug varies)."""
    pattern = os.path.join(
        results_root,
        f"level{level}",
        "*",
        tag,
        f"episode_{episode:03d}_steps.json",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a run's step-level model I/O.")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--scene", default="",
                        help="fallback scene when the tag does not name one (default stage1.1)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()

    pattern = os.path.join(args.logs, args.tag or "*", "level*_episode*_agent.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no agent logs matched {pattern}")
        return 1
    print(f"{len(files)} episode log(s) under {args.logs}/")

    by_model = {}
    all_rows = []
    from_steps = 0
    from_logs = 0
    for path in files:
        match = NAME.search(os.path.basename(path))
        if not match:
            continue
        level, episode = int(match.group(1)), int(match.group(2))
        tag = os.path.basename(os.path.dirname(path))
        scene = scene_from_tag(tag, args.scene)
        sidecar = sidecar_for(tag, level, episode, args.results)
        model = model_for_tag(level, tag, args.results)
        if sidecar:
            rows = parse_sidecar(sidecar, level, episode, scene)
            from_steps += 1
        else:
            rows = parse_file(path, scene)
            from_logs += 1
        if not rows:
            continue
        by_model.setdefault(model, []).extend(rows)
        all_rows.extend(rows)

    print(
        f"source: {from_steps} episode(s) from _steps.json, "
        f"{from_logs} episode(s) from the agent log fallback"
    )

    for model in sorted(by_model):
        rows = by_model[model]
        named_colour = named_shape = named_opening = 0
        for row in rows:
            words = (row["scene_description"] + " " + row["reasoning"]).lower()
            marker = row["marker"].lower()
            colour = COLOURS.get(marker.split()[0], "")
            shape_words = [w for w in ("square", "triangle", "disc", "hexagon", "cross",
                                       "bars", "arrow", "diamond") if w in marker]
            if colour and colour in words:
                named_colour += 1
            if any(w in words for w in shape_words):
                named_shape += 1
            if "opening" in words or "gap" in words:
                named_opening += 1
        n = len(rows)
        print(f"\n=== {model}   {n} step(s) ===")
        print(f"  提到自己标志物的颜色: {named_colour}/{n} = {named_colour / n:.1%}")
        print(f"  提到自己标志物的形状: {named_shape}/{n} = {named_shape / n:.1%}")
        print(f"  提到开口/缝隙      : {named_opening}/{n} = {named_opening / n:.1%}")
        shown = 0
        for row in rows:
            if shown >= args.examples:
                break
            if row["reasoning"]:
                print(f"  [{row['marker']} L{row['level']} ep{row['episode']} step{row['step']}]"
                      f" {row['action']} conf={row['confidence']}")
                print(f"     scene: {row['scene_description'][:200]}")
                print(f"     think: {row['reasoning'][:200]}")
                shown += 1

    if args.csv:
        if not all_rows:
            print("no rows to write")
            return 1
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
