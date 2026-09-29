"""Which torso yaws actually traverse each Level?  Measured, not derived.

The geometry says the width a yawed body needs peaks at 21.1 degrees, so a body rotated
15-40 degrees needs MORE room than an unrotated one and should fail at exactly flush
width (A/S 1.00).  This checks that against the episodes that actually passed: at
A/S 1.00, 37 of the 44 passes crossed at 0 degrees and 7 at 45 degrees or more, and
none between 12 and 42 -- while the 9 of 11 failures that reached the wall did so at
15 or 30 degrees, exactly the range the geometry calls worst.  At A/S 1.10, by
contrast, 15 and 30 degrees do pass, which is what makes this a test of the geometry
rather than of the model.

Run from the repository root:  python lab_logs/check_passage_yaws.py
"""

import collections
import json
import sys
import tarfile

sys.path.insert(0, ".")
import memory_metrics as mm  # noqa: E402

tf = tarfile.open("lab_logs/bao_v7_all.tgz")
records = {}
for member in tf.getmembers():
    name = member.name
    parts = name.split("/")
    if name.endswith(".json") and "/episode_" in name and "_steps" not in name and len(parts) > 4:
        record = json.load(tf.extractfile(member))
        records[(record["model_name"], int(record["level"]), int(record["episode_id"]))] = record

by_level = collections.defaultdict(list)
for (model, level, episode), record in records.items():
    by_level[level].append(record)

print("required width at each yaw: ", end="")
print(", ".join("%d deg -> %.3f" % (yaw, mm.needed_width(yaw)) for yaw in (0, 15, 21, 30, 45, 60, 90)))
print()

for level in (9, 10, 11):
    rows = by_level[level]
    width = rows[0]["channel_width"]
    passes = [r for r in rows if r.get("passed")]
    fails = [r for r in rows if not r.get("passed")]
    print("Level %d  A/S %.2f  width %.3f m  fits if needed <= %.3f" % (
        level, rows[0]["a_s_ratio"], width, width))
    for label, group in (("passed", passes), ("failed", fails)):
        yaws = collections.Counter(
            "none" if r.get("passage_rotation_deg") is None
            else "%.1f" % float(r["passage_rotation_deg"]) for r in group)
        maxes = collections.Counter("%.0f" % float(r.get("max_rotation_deg") or 0) for r in group)
        print("   %-6s n=%-3d passage yaw at the wall: %s" % (label, len(group), dict(yaws)))
        print("          max yaw reached:             %s" % dict(sorted(maxes.items(), key=lambda kv: float(kv[0]))))
    # The decisive test: a pass whose body was at 15-40 degrees would falsify the table.
    contradicting = [
        r for r in passes
        if r.get("passage_rotation_deg") is not None
        and 12.0 < mm.fold_yaw(float(r["passage_rotation_deg"])) < 42.0
    ]
    print("   passes with a 12-42 degree passage yaw (would contradict the geometry): %d"
          % len(contradicting))
    for r in contradicting[:4]:
        print("      %s ep%d passage=%.1f max=%.1f steps=%d" % (
            r["model_name"], r["episode_id"], float(r["passage_rotation_deg"]),
            float(r.get("max_rotation_deg") or 0), r["total_steps"]))
    print()
