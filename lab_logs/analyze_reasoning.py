"""Stage-1 side analysis: what the models' own reasoning says.

Reads the transferred archive in memory (logs/ carries every step's prompt and the
model's reply, reasoning included) and reports:

  * inventory and verbosity
  * topic rates, as a percentage of logged replies.  Every reply is written to the
    log TWICE (once by the model client, once by the runner's adapter), so the rate
    is hits / reply-instances: both sides count the duplication, which makes the
    ratio correct without assuming how many copies there are.
  * WHERE the model first turned, in metres, which is the human-comparison core.
    The wall is at x = 8.0 and the last free position is x = 7.25, so a turn that
    starts before x = 7.0 was taken while the body was still clear of the wall --
    the model decided to rotate before it could have been blocked.  A turn that
    starts at 7.25 or later came after the wall stopped it.  Humans do the first.
  * two reasoning patterns measured in earlier protocols: the "align myself /
    sidestep" plan (v5) and the "I have overshot the target" belief (v6).
"""

import collections
import json
import re
import tarfile

ARCHIVE = "lab_logs/bao_v7_all.tgz"
REASONING = re.compile(r'"reasoning":\s*"((?:[^"\\]|\\.)*)"')

TOPICS = {
    "lateralPlan": r"sidestep|shift (my|to the)|move (slightly )?(left|right) to",
    "rotation": r"rotat|turn my torso|torso rotation|turn (left|right)",
    "ownWidth": r"how wide|my width|shoulder|body width|am i (too )?wide",
    "fitNarrow": r"fit through|fit in|wide enough|narrow|too wide|squeeze|clearance",
    "blocked": r"blocked|collision|collide|stuck",
    "budget": r"budget|steps left|remaining steps|running out of",
    "overshoot": r"overshoot",
    "alignSelf": r"align myself|align the|realign",
}

WALL_X = 8.0
LAST_FREE_X = 7.25          # one 0.75 m stride short of the wall
CLEAR_X = 7.0               # turning before this means never having been blocked


def main() -> None:
    tf = tarfile.open(ARCHIVE)
    calls = collections.defaultdict(collections.Counter)
    files = 0
    for member in tf.getmembers():
        name = member.name
        if not (name.startswith("logs/") and name.endswith("_agent.txt")):
            continue
        files += 1
        text = tf.extractfile(member).read().decode("utf-8", "replace")
        model = name.split("/")[1].split("-v7-state-axes")[0]
        counter = calls[model]
        for found in REASONING.findall(text):
            if found == "<brief reasoning>":
                continue
            counter["raw"] += 1
            counter["chars"] += len(found)
            low = found.lower()
            for topic, pattern in TOPICS.items():
                if re.search(pattern, low):
                    counter[topic] += 1

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

    print("episode logs: %d   episodes: %d   steps: %d"
          % (files, len(records), sum(r["total_steps"] for r in records)))
    print()
    print("replies are logged twice, so rate = hits / reply-instances:")
    print("%-30s %9s %9s %8s | %7s %8s %8s %8s %9s %7s %7s %9s"
          % ("model", "replies", "steps", "chars", "lateral", "rotation",
             "ownWidth", "fitNarrow", "blocked", "budget", "over", "alignSelf"))
    for model in sorted(calls):
        counter = calls[model]
        n = counter["raw"] or 1
        step_total = sum(r["total_steps"] for r in records if r["model_name"] == model)
        rate = lambda key: 100.0 * counter[key] / n  # noqa: E731
        print("%-30s %9d %9d %8.0f | %6.1f%% %7.1f%% %7.1f%% %7.1f%% %8.1f%% %6.1f%% %6.1f%% %8.1f%%"
              % (model, counter["raw"], step_total,
                 counter["chars"] / n, rate("lateralPlan"), rate("rotation"),
                 rate("ownWidth"), rate("fitNarrow"), rate("blocked"),
                 rate("budget"), rate("overshoot"), rate("alignSelf")))

    print()
    print("WHERE the first turn happened, by level (x in metres):")
    print("  clear  = first turn at x < %.2f (turned before the wall could block)" % CLEAR_X)
    print("  atWall = first turn at x >= %.2f (turned only because the wall stopped it)" % LAST_FREE_X)
    print()
    print("%-30s %s" % ("model", "  ".join("%-22s" % ("L%d" % lv) for lv in (9, 10, 11))))
    for model in sorted(calls):
        cells = []
        for level in (9, 10, 11):
            clear = wall = never = 0
            for record in records:
                if record["model_name"] != model or record["level"] != level:
                    continue
                if record.get("first_turn_step") is None:
                    never += 1
                    continue
                sidecar = steps.get((model, level, record["episode_id"])) or []
                index = record["first_turn_step"]
                x = sidecar[index]["position_x"] if index < len(sidecar) else None
                if x is None:
                    never += 1
                elif x < CLEAR_X:
                    clear += 1
                else:
                    wall += 1
            cells.append("clear=%d wall=%d none=%d" % (clear, wall, never))
        print("%-30s %s" % (model, "  ".join("%-22s" % c for c in cells)))

    print()
    print("L11 only, per episode: x at the first turn / x when it first got through")
    for model in sorted(set(r["model_name"] for r in records)):
        rows = []
        for record in records:
            if record["model_name"] != model or record["level"] != 11:
                continue
            sidecar = steps.get((model, 11, record["episode_id"])) or []
            index = record.get("first_turn_step")
            x = sidecar[index]["position_x"] if index is not None and index < len(sidecar) else None
            rows.append("%s@%s" % ("never" if x is None else "%.2f" % x,
                                   "P" if record["passed"] else "f"))
        print("  %-30s %s" % (model, "  ".join(rows)))


if __name__ == "__main__":
    main()
