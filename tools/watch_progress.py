#!/usr/bin/env python3
"""Follow a running sweep and print ONE LINE per finished episode, as it finishes.

Why this and not `tail -f`: the per-job logs carry progress too, but a log line is text that has to be
parsed and can be interleaved between jobs.  The episode RECORD is the thing that actually landed, so
this watches `results/` and reads each record as it appears -- which means the line it prints carries
the outcome (passed, end reason, steps) and cannot disagree with what was saved.

Existing records are counted but not printed, so the output is forward progress rather than a dump of
history.  Pass --replay to print the history once before following.
"""
import argparse
import datetime
import glob
import json
import os
import re
import time

# The scene is the ONLY component of a run tag that can be recovered from the tag itself, because it is
# the only component whose spelling is fixed.  `run_tag` is composed as
# "<base>-<PROTOCOL_TAG>-<scene>-<request-param-suffix>" (main.py:compose_run_tag), so the LAST
# hyphen-separated field is the request-parameter suffix whenever the model has one -- splitting on "-"
# and taking the last field printed "nothinking" as the scene of
# "full_glm-4.6v-v7-state-axes-stage1.5-nothinking", whose real scene is "stage1.5".  A stage scene is
# always one of stage1.1..stage1.5, so match that explicitly instead.
STAGE_SCENE_RE = re.compile(r"stage1\.[1-5]")
# Fields an episode record may carry the scene in.  Checked, in order, BEFORE the tag, so a record that
# states its scene is believed over anything inferred from text.
SCENE_FIELDS = ("scene", "scene_name", "scene_id")


def stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def scene_of(record: dict, tag: str = "") -> str:
    """The scene of an episode, or "?" when it cannot be determined.

    Preference order: a scene field on the record, then the explicit ``stage1.N`` in the run tag.  "?"
    is returned only when neither yields a stage scene -- e.g. the pre-scene tags such as
    "gemini-2.5-pro-v7-state-axes", which genuinely name no scene.
    """
    for field in SCENE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    match = STAGE_SCENE_RE.search(tag)
    return match.group(0) if match else "?"


def episode_files():
    return [f for f in glob.glob("results/level*/*/*/episode_*.json")
            if not f.endswith("_steps.json")]


def summary_files():
    return glob.glob("results/level*/*/*/summary_*.json")


def describe(path: str) -> str:
    """One line for one episode record: model, scene, level, slot, outcome."""
    try:
        with open(path) as handle:
            d = json.load(handle)
    except Exception as exc:  # a record being written right now can be half-flushed
        return f"(unreadable: {type(exc).__name__})"
    tag = str(d.get("run_tag") or "")
    scene = scene_of(d, tag)
    model = str(d.get("model_name") or "?")
    passed = "PASS" if d.get("passed") else "fail"
    sideways = "sideways" if d.get("passed_sideways") else "-"
    return (f"{model:32s} {scene:9s} level {d.get('level', '?'):>2} "
            f"ep {d.get('episode_id', '?'):>2}  {passed:4s} {sideways:8s} "
            f"steps {d.get('total_steps', '?'):>2}  end={d.get('end_reason', '?')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="One line per finished episode, live.")
    ap.add_argument("--every", type=float, default=4.0, help="seconds between scans (default 4)")
    ap.add_argument("--replay", action="store_true", help="print existing records first")
    ap.add_argument("--heartbeat", type=float, default=60.0,
                    help="seconds between progress summaries (0 disables, default 60)")
    args = ap.parse_args()

    seen = set()
    existing = sorted(episode_files())
    seen.update(existing)
    seen.update(summary_files())

    n_pass = sum(1 for f in existing if _passed(f))
    print(f"{stamp()}  watching results/  --  {len(existing)} episode(s) already on disk, "
          f"{n_pass} passed", flush=True)
    if args.replay:
        for f in existing:
            print(f"{stamp()}  REPLAY   {describe(f)}", flush=True)
    print(f"{stamp()}  waiting for new records every {args.every:g}s  (Ctrl-C to stop watching; "
          f"the sweep keeps running)", flush=True)

    last_beat = time.time()
    try:
        while True:
            time.sleep(args.every)
            for f in sorted(episode_files()):
                if f in seen:
                    continue
                seen.add(f)
                flag = "PASS" if _passed(f) else "fail"
                if flag == "PASS":
                    n_pass += 1
                print(f"{stamp()}  EPISODE  {describe(f)}", flush=True)
            for f in sorted(summary_files()):
                if f in seen:
                    continue
                seen.add(f)
                print(f"{stamp()}  LEVEL    finished unit: {f}", flush=True)
            if args.heartbeat and time.time() - last_beat >= args.heartbeat:
                last_beat = time.time()
                total = len(episode_files())
                print(f"{stamp()}  ----  {total} episode(s) on disk, {n_pass} passed, "
                      f"{len(summary_files())} level unit(s) done  ----", flush=True)
    except KeyboardInterrupt:
        print(f"\n{stamp()}  stopped watching ({len(episode_files())} episode(s) on disk, "
              f"{n_pass} passed).  The sweep itself was not touched.", flush=True)
    return 0


def _passed(path: str) -> bool:
    try:
        with open(path) as handle:
            return bool(json.load(handle).get("passed"))
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
