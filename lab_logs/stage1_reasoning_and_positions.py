"""Stage 1 reasoning mentions and first-turn positions, from the archive.

Two claims in the draft are load-bearing and both were wrong before this script existed:
how often the models' own reasoning mentions the opening, and where the first turn
happens.  The second contradicted the draft's "reactive" framing outright -- the turns
are early (median 2.0 m, 5.8 m before the wall) and only 3% happen at the wall, so the
deficit is in how much the model rotates, not in when it decides to.

Run from the repository root:  python lab_logs/stage1_reasoning_and_positions.py
"""

import collections
import json
import re
import sys
import tarfile

sys.path.insert(0, ".")
import memory_metrics as mm  # noqa: E402

REASONING = re.compile(r'"reasoning":\s*"((?:[^"\\]|\\.)*)"')
MIN_LEN = 20


def reasonings(text):
    found = [r for r in REASONING.findall(text) if len(r.strip()) >= MIN_LEN]
    return [" ".join(r.split()) for r in found]


tf = tarfile.open("lab_logs/bao_v7_all.tgz")
logs, records, steps_by_key = {}, {}, {}
AGENT = re.compile(r"level(\d+)_episode(\d+)_agent\.txt$")
for member in tf.getmembers():
    name = member.name
    parts = name.split("/")
    basename = parts[-1]
    if name.startswith("logs/") and basename.endswith("_agent.txt"):
        match = AGENT.search(basename)
        if match:
            model = parts[1].split("-v7-state-axes")[0]
            key = (model, int(match.group(1)), int(match.group(2)))
            logs[key] = tf.extractfile(member).read().decode("utf-8", "replace")
    elif basename.endswith("_steps.json") and len(parts) > 4:
        steps_by_key[(parts[2], int(parts[1].replace("level", "")),
                      int(basename.replace(".json", "").split("_")[1]))] = (
            json.load(tf.extractfile(member)))
    elif basename.endswith(".json") and len(parts) > 4 and "/episode_" in name:
        record = json.load(tf.extractfile(member))
        records[(record["model_name"], int(record["level"]), int(record["episode_id"]))] = record

episodes = list(records.items())
print("episodes %d | agent logs %d | matched %d" % (
    len(episodes), len(logs), sum(1 for key, _ in episodes if key in logs)))

by_level = collections.defaultdict(lambda: {"n": 0, "reason": 0, "open": 0, "turn": 0, "shoulder": 0})
overall = {"n": 0, "reason": 0, "open": 0}
for key, record in episodes:
    level = key[1]
    text = logs.get(key, "")
    rs = reasonings(text)
    joined = " ".join(rs).lower()
    by_level[level]["n"] += 1
    overall["n"] += 1
    if rs:
        by_level[level]["reason"] += 1
        overall["reason"] += 1
        if "opening" in joined or "gap" in joined or "aperture" in joined:
            by_level[level]["open"] += 1
            overall["open"] += 1
        if "turn" in joined or "rotate" in joined or "rotation" in joined:
            by_level[level]["turn"] += 1
        if "shoulder" in joined or "width" in joined or "wide" in joined:
            by_level[level]["shoulder"] += 1

print("reasonings extracted: %d episodes" % overall["reason"])
print("episodes whose reasoning mentions the opening/aperture/gap: %d/%d = %.0f%%" % (
    overall["open"], overall["reason"], 100.0 * overall["open"] / max(overall["reason"], 1)))
print()
print("%-5s %-6s %-18s %-18s %-18s" % ("lvl", "n", "mentions opening", "mentions turning", "mentions width"))
for level in sorted(by_level):
    row = by_level[level]
    if not row["reason"]:
        continue
    print("%-5d %-6d %-18s %-18s %-18s" % (
        level, row["n"],
        "%d (%.0f%%)" % (row["open"], 100.0 * row["open"] / row["reason"]),
        "%d (%.0f%%)" % (row["turn"], 100.0 * row["turn"] / row["reason"]),
        "%d (%.0f%%)" % (row["shoulder"], 100.0 * row["shoulder"] / row["reason"])))

print()
print("FIRST TURN POSITION (from the sidecars, x in metres; wall at 8.0, door zone starts at 7.0)")
buckets = collections.Counter()
positions = []
for key, record in episodes:
    step = record.get("first_turn_step")
    if step is None:
        buckets["never turned"] += 1
        continue
    sidecar = steps_by_key.get(key) or []
    if step < len(sidecar):
        x = float(sidecar[step].get("position_x") or 0.0)
        positions.append(x)
        buckets["x < 4" if x < 4 else "4 <= x < 7" if x < 7 else "x >= 7 (at the wall)"] += 1
for name, count in buckets.most_common():
    print("  %-22s %4d  (%.0f%%)" % (name, count, 100.0 * count / len(episodes)))
if positions:
    print("  mean x of the first turn: %.2f m (n=%d), median %.2f m" % (
        sum(positions) / len(positions), len(positions), sorted(positions)[len(positions) // 2]))
print("  steps from the first turn to the wall (x=8.0): %.1f m" % (
    8.0 - (sum(positions) / len(positions)) if positions else 0.0))
