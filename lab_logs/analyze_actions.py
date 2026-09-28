"""Action-level analysis: the curves for rotation, look_down and lateral movement.

Reads the transferred archive and answers, over 660 episodes / 10123 steps:

  1. per-A/S curves (pooled over 11 models, 55 episodes per level) for how often
     each action family is used, so shape can be compared with the human studies
     (which report rotation onset near A/S = 1.25-1.30)
  2. the same curves per model, to see who follows the pooled shape
  3. transition probabilities: what follows look_down, a turn, a lateral step,
     against the base rate of each action
  4. the functional chain that would make look_down useful:
     blocked -> look down at own body -> rotate -> get through
     including where the first look_down sits relative to the first collision and
     the first turn, and the pass rate with and without it
"""

import collections
import json
import tarfile

ARCHIVE = "lab_logs/bao_v7_all.tgz"
FAMILIES = {
    "forward": ("forward",),
    "backward": ("backward",),
    "lateral": ("left", "right"),
    "turn": ("turn_left", "turn_right"),
    "look_down": ("look_down",),
    "glance": ("look_left", "look_right"),
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

    seqs = {}
    for record in records:
        key = (record["model_name"], record["level"], record["episode_id"])
        sidecar = steps.get(key)
        if sidecar:
            seqs[key] = [s["action"] for s in sidecar]

    print("episodes %d, with sidecar %d, steps %d"
          % (len(records), len(seqs), sum(len(v) for v in seqs.values())))

    # ---- 1. pooled curves by A/S -------------------------------------------
    print()
    print("== 每局平均使用次数（11 模型合计，每级 55 局）==")
    header = ["A/S", "pass%"] + list(FAMILIES) + ["有转身的局%", "有低头的局%", "有侧移的局%"]
    print("%-6s %-7s %s" % ("", "", " ".join("%-9s" % h for h in header[2:-3])))
    print("%-6s %-7s %-9s %-9s %-9s %-9s %-9s %-9s %-12s %-12s %-12s"
          % ("A/S", "pass%", "forward", "backward", "lateral", "turn", "look_down",
             "glance", "turn_eps%", "lookdown_eps%", "lateral_eps%"))
    by_level = collections.defaultdict(list)
    for record in records:
        by_level[record["level"]].append(record)
    for level in sorted(by_level):
        eps = by_level[level]
        n = len(eps)
        counts = {name: 0 for name in FAMILIES}
        users = {name: 0 for name in FAMILIES}
        for record in eps:
            key = (record["model_name"], level, record["episode_id"])
            actions = seqs.get(key, [])
            for name, members in FAMILIES.items():
                hits = sum(1 for a in actions if a in members)
                counts[name] += hits
                users[name] += 1 if hits else 0
        print("%-6.1f %-7.1f %-9.1f %-9.1f %-9.1f %-9.1f %-9.2f %-9.1f %-12.1f %-12.1f %-12.1f"
              % (eps[0]["a_s_ratio"], 100.0 * sum(e["passed"] for e in eps) / n,
                 counts["forward"] / n, counts["backward"] / n, counts["lateral"] / n,
                 counts["turn"] / n, counts["look_down"] / n, counts["glance"] / n,
                 100.0 * users["turn"] / n, 100.0 * users["look_down"] / n,
                 100.0 * users["lateral"] / n))

    # ---- 2. per model ------------------------------------------------------
    print()
    print("== 每局平均次数（按模型，全 12 级合计 60 局）==")
    print("%-30s %-8s %-9s %-9s %-9s %-9s %-9s"
          % ("model", "turn", "look_down", "lateral", "glance", "forward", "backward"))
    by_model = collections.defaultdict(list)
    for record in records:
        by_model[record["model_name"]].append(record)
    for model in sorted(by_model):
        eps = by_model[model]
        total = collections.Counter()
        for record in eps:
            actions = seqs.get((model, record["level"], record["episode_id"]), [])
            for name, members in FAMILIES.items():
                total[name] += sum(1 for a in actions if a in members)
        print("%-30s %-8.2f %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f"
              % (model, total["turn"] / len(eps), total["look_down"] / len(eps),
                 total["lateral"] / len(eps), total["glance"] / len(eps),
                 total["forward"] / len(eps), total["backward"] / len(eps)))

    # ---- 3. transitions ----------------------------------------------------
    base = collections.Counter()
    after = collections.defaultdict(collections.Counter)
    for actions in seqs.values():
        for a in actions:
            base[a] += 1
        for a, b in zip(actions, actions[1:]):
            after[a][b] += 1
    total_actions = sum(base.values())
    print()
    print("== 下一步动作的分布（行=当前动作，只列 n>=80 的）==")
    print("base: turn %.1f%%  lateral %.1f%%  look_down %.1f%%  forward %.1f%%"
          % (100.0 * sum(base[a] for a in FAMILIES["turn"]) / total_actions,
             100.0 * sum(base[a] for a in FAMILIES["lateral"]) / total_actions,
             100.0 * base["look_down"] / total_actions,
             100.0 * base["forward"] / total_actions))
    print("%-12s %-7s %-11s %-11s %-11s %-11s" % ("after", "n", "->turn%", "->lateral%", "->look_down%", "->forward%"))
    for action, counter in sorted(after.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(counter.values())
        if n < 80:
            continue
        print("%-12s %-7d %-11.1f %-11.1f %-11.1f %-11.1f"
              % (action, n,
                 100.0 * sum(counter[a] for a in FAMILIES["turn"]) / n,
                 100.0 * sum(counter[a] for a in FAMILIES["lateral"]) / n,
                 100.0 * counter["look_down"] / n,
                 100.0 * counter["forward"] / n))

    # ---- 4. is look_down functional? ---------------------------------------
    print()
    print("== look_down 的功能性 ==")
    used = [r for r in records if "look_down" in seqs.get((r["model_name"], r["level"], r["episode_id"]), [])]
    unused = [r for r in records if r not in used and (r["model_name"], r["level"], r["episode_id"]) in seqs]
    print("用了 look_down 的局: %d  通过率 %5.1f%%" % (len(used), 100.0 * sum(r["passed"] for r in used) / max(1, len(used))))
    print("没用的局        : %d  通过率 %5.1f%%" % (len(unused), 100.0 * sum(r["passed"] for r in unused) / max(1, len(unused))))

    chain = collections.Counter()
    for record in records:
        key = (record["model_name"], record["level"], record["episode_id"])
        sidecar = steps.get(key)
        if not sidecar:
            continue
        actions = [s["action"] for s in sidecar]
        first_block = next((s["step"] for s in sidecar if s.get("collision")), None)
        first_look = next((i for i, a in enumerate(actions) if a == "look_down"), None)
        first_turn = record.get("first_turn_step")
        if first_look is None:
            chain["never looks down"] += 1
            continue
        if first_block is None:
            chain["looks down before any block"] += 1
        elif first_look > first_block:
            chain["looks down AFTER being blocked"] += 1
        else:
            chain["looks down BEFORE first block"] += 1
        if first_turn is not None:
            chain["look_down then later turns" if first_turn > first_look else "turns before looking down"] += 1
        else:
            chain["looks down but NEVER turns"] += 1
        if record["passed"]:
            chain["look_down and passes"] += 1
    for key in ("never looks down", "looks down before any block", "looks down BEFORE first block",
                "looks down AFTER being blocked", "look_down then later turns",
                "turns before looking down", "looks down but NEVER turns", "look_down and passes"):
        print("  %-34s %d" % (key, chain[key]))

    # ---- 5. turn / lateral timing relative to the first block ---------------
    print()
    print("== 首次转身位置（x，全部 660 局；墙在 8.0，最后自由位置 7.25）==")
    buckets = collections.Counter()
    for record in records:
        key = (record["model_name"], record["level"], record["episode_id"])
        sidecar = steps.get(key)
        turn_step = record.get("first_turn_step")
        if not sidecar or turn_step is None:
            buckets["never turns"] += 1
            continue
        x = sidecar[turn_step]["position_x"] if turn_step < len(sidecar) else None
        if x is None:
            buckets["never turns"] += 1
        elif x < 4.0:
            buckets["x < 4.0  (early, unprompted)"] += 1
        elif x < 7.0:
            buckets["4.0 <= x < 7.0  (mid)"] += 1
        else:
            buckets["x >= 7.0  (at the wall)"] += 1
    for key in ("never turns", "x < 4.0  (early, unprompted)", "4.0 <= x < 7.0  (mid)",
                "x >= 7.0  (at the wall)"):
        print("  %-32s %3d / 660" % (key, buckets[key]))


if __name__ == "__main__":
    main()
