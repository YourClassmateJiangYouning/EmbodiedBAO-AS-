"""Did the action list's explanation of turning actually reach behaviour?

The action list every model is given says, in the turn_left/turn_right entries,
that turning "changes how wide your body is across the opening" and that "you
cannot turn once your shoulders are inside the opening".  Those two sentences state
the mechanism and the timing rule, so the question this answers is not whether they
are there -- they are -- but whether models use them.

Measured per episode, over the 660-episode v7 archive, by reading each episode's own
log (``logs/<tag>/level<N>_episode<MMM>_agent.txt``) and cross-referencing the
record's outcome and action sequence:

  * how often an episode's reasoning mentions the opening, the shoulders, the
    body's width, or the timing rule;
  * split by outcome, and, at A/S 0.9 where turning is the only way through, split
    by whether the episode turned at all.

Presence per episode is used rather than a per-reply rate, because every reply is
logged twice (client and adapter) and a presence count does not care.
"""

from __future__ import annotations

import collections
import json
import re
import tarfile

ARCHIVE = "lab_logs/bao_v7_all.tgz"
REASONING = re.compile(r'"reasoning":\s*"((?:[^"\\]|\\.)*)"')
LOG_NAME = re.compile(r"logs/(?P<tag>[^/]+)/level(?P<level>\d+)_episode(?P<episode>\d+)_agent\.txt$")

# Phrases that come from, or paraphrase, the action list's own explanation.
PATTERNS = {
    "opening": r"opening",
    "shoulder": r"shoulder",
    "width/wide": r"\bwidth\b|\bwide\b|how wide",
    "timing rule": (
        r"before (?:i|my|the) (?:shoulder|body|torso|reach|enter|get)|"
        r"before entering|while (?:there is|i have) room|need(?:s)? room|"
        r"room to (?:turn|rotate)|once (?:my|the) shoulder"
    ),
    "fit/too wide": r"fit through|fit in|too wide|wide enough|narrow|squeeze|clearance",
    "blocked": r"blocked|collision|collide|stuck",
}


def episode_texts(tar: tarfile.TarFile, name: str) -> str:
    member = tar.extractfile(name)
    if member is None:
        return ""
    return member.read().decode("utf-8", "replace")


def main() -> None:
    tar = tarfile.open(ARCHIVE)

    records: dict = {}
    for member in tar.getmembers():
        name = member.name
        if name.endswith(".json") and "/episode_" in name and not name.endswith("_steps.json"):
            record = json.load(tar.extractfile(member))
            parts = name.split("/")
            key = (int(parts[1].replace("level", "")), record["model_name"], record["episode_id"])
            records[key] = record

    # (level, model, episode) -> the set of pattern names its reasoning matched
    matched: dict = {}
    for member in tar.getmembers():
        found = LOG_NAME.match(member.name)
        if not found:
            continue
        level = int(found.group("level"))
        episode = int(found.group("episode"))
        model = found.group("tag").split("-v7-state-axes")[0]
        text = episode_texts(tar, member.name)
        hits = set()
        for reasoning in REASONING.findall(text):
            if reasoning == "<brief reasoning>":
                continue
            low = reasoning.lower()
            for label, pattern in PATTERNS.items():
                if re.search(pattern, low):
                    hits.add(label)
        matched[(level, model, episode)] = hits

    def turned(record: dict) -> bool:
        sequence = record.get("action_sequence") or ""
        return "turn_left" in sequence or "turn_right" in sequence

    def report(title: str, keys: list) -> None:
        if not keys:
            return
        print()
        print(f"=== {title} (n={len(keys)}) ===")
        header = "  %-22s %6s %7s %8s %9s %9s %9s" % (
            "group", "n", "opening", "shoulder", "width", "timing", "fit"
        )
        print(header)
        groups = collections.defaultdict(list)
        for key in keys:
            level, model, episode = key
            record = records.get(key)
            if record is None:
                continue
            if record["passed"]:
                groups["passed"].append(key)
            else:
                groups["failed"].append(key)
                groups["failed, never turned" if not turned(record) else "failed, turned"].append(key)
        groups["all"] = keys
        for name in ("all", "passed", "failed", "failed, turned", "failed, never turned"):
            group = groups.get(name)
            if not group:
                continue
            counts = collections.Counter()
            for key in group:
                for hit in matched.get(key, ()):  # noqa: B007
                    counts[hit] += 1
            share = lambda label: 100.0 * counts[label] / len(group)  # noqa: E731
            print("  %-22s %6d %6.0f%% %7.0f%% %7.0f%% %8.0f%% %8.0f%% %8.0f%%" % (
                name, len(group), share("opening"), share("shoulder"),
                share("width/wide"), share("timing rule"), share("fit/too wide"),
                share("blocked"),
            ))

    all_keys = sorted(records)
    level11 = [key for key in all_keys if key[0] == 11]
    level10 = [key for key in all_keys if key[0] == 10]
    report("every Level (A/S 2.0 to 0.9)", all_keys)
    report("Level 10, A/S 1.00, the exact geometric limit", level10)
    report("Level 11, A/S 0.90, where turning is the only way through", level11)

    print()
    print("reading of the action list, at Level 11, per model:")
    print("  %-32s %8s %8s %10s %10s" % ("model", "turned", "passed", "mentions", "timing rule"))
    for model in sorted(set(key[1] for key in level11)):
        group = [key for key in level11 if key[1] == model]
        turned_n = sum(1 for key in group if turned(records[key]))
        passed_n = sum(1 for key in group if records[key]["passed"])
        mentions = sum(1 for key in group if matched.get(key))
        timing = sum(1 for key in group if "timing rule" in matched.get(key, ()))
        print("  %-32s %8d %8d %10d %10d" % (model, turned_n, passed_n, mentions, timing))


if __name__ == "__main__":
    main()
