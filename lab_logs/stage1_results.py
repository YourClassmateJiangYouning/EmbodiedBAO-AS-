"""Stage 1 results table, recomputed from lab_logs/bao_v7_all.tgz.

Every number the paper draft's results section quotes is produced here rather than
typed in, so the tables can be re-derived after any change to the archive or the code.

Run from the repository root:  python lab_logs/stage1_results.py
"""

import collections
import json
import sys
import tarfile

sys.path.insert(0, ".")
import memory_metrics as mm  # noqa: E402

tf = tarfile.open("lab_logs/bao_v7_all.tgz")
records, sidecars = {}, {}
for member in tf.getmembers():
    name = member.name
    if not name.endswith(".json"):
        continue
    parts = name.split("/")
    if "_steps" in name and len(parts) > 4:
        sidecars[(parts[2], int(parts[1].replace("level", "")),
                  int(parts[4].replace(".json", "").split("_")[1]))] = json.load(tf.extractfile(member))
    elif "/episode_" in name and len(parts) > 4:
        record = json.load(tf.extractfile(member))
        records[(record["model_name"], int(record["level"]), int(record["episode_id"]))] = record

episodes = []
for key, record in records.items():
    steps = sidecars.get(key, [])
    actions = [s.get("action") for s in steps]
    record["_turns"] = sum(1 for a in actions if a in ("turn_left", "turn_right"))
    record["_lateral"] = sum(1 for a in actions if a in ("left", "right"))
    record["_forward"] = sum(1 for a in actions if a == "forward")
    record["_look_down"] = sum(1 for a in actions if a == "look_down")
    record["_steps_list"] = steps
    episodes.append(record)

open_by_level = {}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


print("=" * 108)
print("PER LEVEL")
print("%-5s %-6s %-8s %-8s %-9s %-8s %-9s %-8s %-8s %-9s" % (
    "lvl", "A/S", "width", "pass", "needs", "steps", "maxrot", "turneq", "turned%", "sideways%"))
for level in sorted({e["level"] for e in episodes}):
    rows = [e for e in episodes if e["level"] == level]
    width = rows[0]["channel_width"]
    passes = [r for r in rows if r.get("passed")]
    print("%-5d %-6.2f %-8.3f %-8s %-9d %-8.1f %-9.1f %-8.1f %-8.0f %-9.0f" % (
        level, rows[0]["a_s_ratio"], width,
        "%d/%d" % (len(passes), len(rows)), mm.min_turns(width),
        mean(r["total_steps"] for r in rows),
        mean(float(r.get("max_rotation_deg") or 0) for r in rows),
        mean(r["total_steps"] - mm.optimal_steps(width) for r in rows),
        100.0 * mean(1 if r["_turns"] else 0 for r in rows),
        100.0 * len([r for r in passes if r.get("passed_sideways")]) / max(len(passes), 1),
    ))
    open_by_level[level] = 100.0 * len(passes) / len(rows)

print()
print("=" * 108)
print("PER MODEL  (L11 = A/S 0.90 narrowest, L10 = A/S 1.00)")
print("%-32s %-8s %-9s %-10s %-9s %-9s %-9s" % (
    "model", "overall", "L11", "L10", "maxrot", "turned%", "steps"))
for model in sorted({e["model_name"] for e in episodes}):
    mine = [e for e in episodes if e["model_name"] == model]
    l11 = [e for e in mine if e["level"] == 11]
    l10 = [e for e in mine if e["level"] == 10]
    print("%-32s %-8.0f %-9s %-10s %-9.1f %-9.0f %-9.1f" % (
        model,
        100.0 * mean(1 if e.get("passed") else 0 for e in mine),
        "%d/%d" % (sum(1 for e in l11 if e.get("passed")), len(l11)),
        "%d/%d" % (sum(1 for e in l10 if e.get("passed")), len(l10)),
        mean(float(e.get("max_rotation_deg") or 0) for e in mine),
        100.0 * mean(1 if e["_turns"] else 0 for e in mine),
        mean(e["total_steps"] for e in mine)))

print()
print("=" * 108)
print("FAILURES AT L11")
l11_fails = [e for e in episodes if e["level"] == 11 and not e.get("passed")]
buckets = collections.Counter()
for e in l11_fails:
    n = e["_turns"]
    buckets["never turned" if n == 0 else "1-3" if n <= 3 else "4-6" if n <= 6 else ">=7"] += 1
print("n = %d | turns: %s" % (len(l11_fails), dict(buckets)))
print("held a 45-135 deg posture at some point: %d | never did: %d" % (
    sum(1 for e in l11_fails if mm.is_sideways_yaw(float(e.get("max_rotation_deg") or 0))),
    sum(1 for e in l11_fails if not mm.is_sideways_yaw(float(e.get("max_rotation_deg") or 0)))))
print("first turn step: %s" % (
    collections.Counter(
        "none" if e.get("first_turn_step") is None
        else "step %d" % (int(e["first_turn_step"]) // 3 * 3)
        for e in l11_fails).most_common(6)))


def first_turn_x(record):
    """Where the first turn happened, from the per-step sidecar.

    The episode record has no ``first_turn_x``; the x of a step lives in its
    sidecar.  Reading the record's field quietly reported "none" for all 660
    episodes, which is the kind of silence that survives a review.  Derived the
    same way ``model_report.py`` and ``stage1_reasoning_and_positions.py`` derive
    it, so three scripts cannot disagree about the number the paper quotes.
    """
    step = record.get("first_turn_step")
    steps = record.get("_steps_list") or []
    if step is None or not 0 <= int(step) < len(steps):
        return None
    return float(steps[int(step)].get("position_x") or 0.0)


turn_x = [first_turn_x(e) for e in episodes if first_turn_x(e) is not None]
print("where the first turn happened (x, m): %s" % (
    collections.Counter(
        "none" if first_turn_x(e) is None
        else "x<4" if first_turn_x(e) < 4 else "4<=x<7" if first_turn_x(e) < 7
        else "x>=7 (at the wall)" for e in episodes).most_common()))
if turn_x:
    late = sum(1 for x in turn_x if x >= 7.0)
    print("  mean x of the first turn: %.2f m (n=%d), median %.2f m" % (
        mean(turn_x), len(turn_x), sorted(turn_x)[len(turn_x) // 2]))
    # Both denominators, because the paper quotes the first one and a reader who
    # takes the second for it would see 7% where the text says 3%.
    print("  first turn at x >= 7.0 m (the wall zone): %d | %.1f%% of all %d episodes | "
          "%.0f%% of the %d that turned at all" % (
              late, 100.0 * late / len(episodes), len(episodes),
              100.0 * late / len(turn_x), len(turn_x)))

print()
print("=" * 108)
print("SPURIOUS ROTATION AT WIDTHS THAT NEED NONE (A/S >= 1.10, min_turns == 0)")
wide = [e for e in episodes if mm.min_turns(e["channel_width"]) == 0]
print("episodes %d | turned at least once %d (%.0f%%) | passed %d (%.1f%%)" % (
    len(wide), sum(1 for e in wide if e["_turns"]),
    100.0 * mean(1 if e["_turns"] else 0 for e in wide),
    sum(1 for e in wide if e.get("passed")), 100.0 * mean(1 if e.get("passed") else 0 for e in wide)))
turned_wide = [e for e in wide if e["_turns"]]
print("  of the ones that turned: pass %.1f%% vs %.1f%% for the ones that did not" % (
    100.0 * mean(1 if e.get("passed") else 0 for e in turned_wide),
    100.0 * mean(1 if e.get("passed") else 0 for e in wide if not e["_turns"])))

print()
print("=" * 108)
print("EFFORT: steps vs optimal")
print("mean steps %.1f | mean optimal %.1f | mean excess %.1f" % (
    mean(e["total_steps"] for e in episodes),
    mean(mm.optimal_steps(e["channel_width"]) for e in episodes),
    mean(e["total_steps"] - mm.optimal_steps(e["channel_width"]) for e in episodes)))
for level in sorted(open_by_level):
    rows = [e for e in episodes if e["level"] == level]
    passes = [e for e in rows if e.get("passed")]
    if passes:
        print("  level %2d: passing episodes ran %.1f steps against an optimum of %d" % (
            level, mean(e["total_steps"] for e in passes),
            mm.optimal_steps(rows[0]["channel_width"])))

print()
print("=" * 108)
print("LOOK ACTIONS (perception) and ACTION MIX")
actions = collections.Counter()
for e in episodes:
    actions.update(s.get("action") for s in e["_steps_list"])
total = sum(actions.values())
for name, count in actions.most_common():
    print("  %-14s %6d  %5.1f%%" % (name, count, 100.0 * count / total))
print("episodes that ever looked down: %d/%d (%.0f%%)" % (
    sum(1 for e in episodes if e["_look_down"]), len(episodes),
    100.0 * mean(1 if e["_look_down"] else 0 for e in episodes)))
print("mean look actions per episode: %.2f" % mean(
    e["_look_down"] + sum(1 for s in e["_steps_list"] if s.get("action") in ("look_left", "look_right"))
    for e in episodes))
