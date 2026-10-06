"""Turn the raw model I/O logs into the table the reasoning analysis needs.

Where the thinking actually lives: main.py's own output hierarchy documents
``logs/{tag}/level{level}_episode{id:03d}_agent.txt`` as "raw model I/O", and that is what it is
-- every step's full prompt and the model's complete JSON reply, including ``scene_description``,
``reasoning`` and ``confidence``.  The per-episode JSON and the flat CSV under results/ hold only
behaviour, which is why searching there for the reasoning finds nothing.  Nothing was ever lost;
it was in logs/ the whole time.

The marker a run used is derivable rather than recorded: the environment advances the slot once
per episode, ``slot = (episode_id % 5) + 1``, which is the same mapping scenes.describe_marker
expects.  So each reply can be attributed to the object that was actually on the wall.

    python3 tools/parse_agent_logs.py                       # every stage1.1 run found
    python3 tools/parse_agent_logs.py --tag gemini-2.5-flash-v7-state-axes-stage1.1
    python3 tools/parse_agent_logs.py --csv logs_reasoning.csv

Prints, per model: how often its own words mention the marker's colour and shape, whether it
named the opening, and a few verbatim examples.  A model that never names its marker is not
looking at it, whatever its feet are doing.
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

COLOURS = {"red": "r", "magenta": "m", "orange": "o", "lime": "l", "cyan": "c",
           "purple": "p", "green": "g", "white": "w", "teal": "t", "black": "k"}


def marker_for(scene: str, episode: int) -> str:
    if sc is None:
        return f"slot{(episode % 5) + 1}"
    try:
        return sc.describe_marker(scene, (episode % 5) + 1)
    except Exception:  # noqa: BLE001
        return f"slot{(episode % 5) + 1}"


def parse_file(path: str, scene: str) -> list:
    """Every step of one episode, with the model's own words."""
    match = NAME.search(os.path.basename(path))
    if not match:
        return []
    level, episode = int(match.group(1)), int(match.group(2))
    marker = marker_for(scene, episode)
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    rows = []
    for index, blob in enumerate(extract_responses(text)):
        try:
            reply = json.loads(blob)
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "level": level, "episode": episode, "step": index, "marker": marker,
            "action": str(reply.get("action", "")),
            "confidence": reply.get("confidence", ""),
            "scene_description": str(reply.get("scene_description", "")),
            "reasoning": str(reply.get("reasoning", "")),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse the raw model I/O logs.")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--scene", default="stage1.1")
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
    for path in files:
        tag = os.path.basename(os.path.dirname(path))
        model = tag.split("-v7-state")[0]
        rows = parse_file(path, args.scene)
        if not rows:
            continue
        by_model.setdefault(model, []).extend(rows)
        all_rows.extend(rows)

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
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
