"""Export the 660-episode wide table used by the figures.

One row per episode, with the fields the paper needs, so the figures can be
regenerated from the CSV alone and the numbers behind every plotted point can be
checked by hand.  Written next to the archive, which is gitignored.
"""

import csv
import json
import tarfile

ARCHIVE = "lab_logs/bao_v7_all.tgz"
OUT = "lab_logs/embodiedbao_v7_episodes.csv"

COLUMNS = [
    "model", "family", "level", "a_s_ratio", "channel_width", "episode_id",
    "passed", "passed_sideways", "total_steps", "n_forward", "n_backward",
    "n_lateral", "n_turn", "n_look_down", "n_glance", "max_rotation_deg",
    "passage_rotation_deg", "first_turn_step", "first_turn_x", "final_position_x",
    "final_position_z", "wall_collisions", "invalid_response_count",
    "end_reason", "action_sequence",
]

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


def main() -> None:
    tf = tarfile.open(ARCHIVE)
    records = []
    steps = {}
    for member in tf.getmembers():
        name = member.name
        if name.endswith(".json") and "/episode_" in name and "_steps" not in name:
            records.append(json.load(tf.extractfile(member)))
        elif name.endswith("_steps.json") and "/episode_" in name:
            parts = name.split("/")
            steps[(parts[2], int(parts[1].replace("level", "")),
                   int(parts[4].split("_")[1]))] = json.load(tf.extractfile(member))

    rows = []
    for record in sorted(records, key=lambda r: (r["model_name"], r["level"], r["episode_id"])):
        key = (record["model_name"], record["level"], record["episode_id"])
        sidecar = steps[key]
        actions = [s["action"] for s in sidecar]
        first_turn = record.get("first_turn_step")
        first_turn_x = ""
        if first_turn is not None and first_turn < len(sidecar):
            first_turn_x = round(sidecar[first_turn]["position_x"], 3)
        rows.append({
            "model": record["model_name"],
            "family": FAMILY.get(record["model_name"], "?"),
            "level": record["level"],
            "a_s_ratio": record["a_s_ratio"],
            "channel_width": record["channel_width"],
            "episode_id": record["episode_id"],
            "passed": int(record["passed"]),
            "passed_sideways": int(record["passed_sideways"]),
            "total_steps": record["total_steps"],
            "n_forward": actions.count("forward"),
            "n_backward": actions.count("backward"),
            "n_lateral": actions.count("left") + actions.count("right"),
            "n_turn": actions.count("turn_left") + actions.count("turn_right"),
            "n_look_down": actions.count("look_down"),
            "n_glance": actions.count("look_left") + actions.count("look_right"),
            "max_rotation_deg": record["max_rotation_deg"],
            "passage_rotation_deg": "" if record.get("passage_rotation_deg") is None
            else record["passage_rotation_deg"],
            "first_turn_step": "" if first_turn is None else first_turn,
            "first_turn_x": first_turn_x,
            "final_position_x": record["final_position_x"],
            "final_position_z": record["final_position_z"],
            "wall_collisions": record["wall_collision_count"],
            "invalid_response_count": record["invalid_response_count"],
            "end_reason": record["end_reason"],
            "action_sequence": record["action_sequence"],
        })

    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %s: %d rows, %d columns" % (OUT, len(rows), len(COLUMNS)))


if __name__ == "__main__":
    main()
