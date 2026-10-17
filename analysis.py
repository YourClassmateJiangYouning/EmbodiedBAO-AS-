"""A/S threshold analysis for EmbodiedBAO.

Consumes the per-episode JSON files written by ``experiments.py``
(``results/level{level}/{model}/episode_*.json``) and reports, for each model,
how its behaviour changes as the channel narrows.

Headline metrics (per Level, and per model overall):

* **Pass rate**       -- fraction of episodes that reached the success plane
  (x = 8.75 m, one stride past the obstacle wall).
* **Sideways rate**   -- fraction of episodes that passed while the torso was
  rotated into the sideways band (45-135 degrees).
* **Sideways threshold** -- the widest A/S ratio (largest channel) at which the
  model still chose to pass sideways.  This is the number to compare against
  the human threshold of **1.30** (Warren & Whang, 1987).  A model that only
  goes sideways once the channel is already narrower than its shoulders has
  threshold < 1.0, i.e. it reacts to contact instead of anticipating it.
* **First turn step** -- mean step index of the first turn, i.e. how early the
  agent starts re-orienting.
* **Mean pass steps**  -- average steps needed for a successful passage.

The classification of a model mirrors the affordance-perception question:

    anticipatory   threshold >= 1.30   rotates before the channel is tight
    borderline     1.00 <= threshold < 1.30   rotates while still passable
    reactive       threshold <  1.00   only rotates after being blocked

Outputs: a per-model JSON report, a CSV table, a Markdown table, and (unless
``--no_plot``) a threshold plot with the human 1.30 reference marked.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from persistence import atomic_write_json, atomic_write_text
# The sideways band is scored by the protocol, so it is defined there and only
# there.  This module carried its own 45/135 copy, which meant the offline report
# and the live runner could in principle disagree about whether an episode passed
# sideways.  protocol.py is pure text and imports nothing, so reaching for it here
# costs nothing and keeps the report free of the Isaac Sim dependency.  The alias
# is this module's historical name for the predicate, kept because
# test_bao_integration.py refers to it.
from protocol import is_sideways_yaw as _sideways_yaw
# The ladder's length is the default for --levels, so it is read from its definition rather than
# written down: a hard-coded [0..5] silently stopped reporting at level 5 and the tool still looked
# healthy.  It lives in environment.py, next to the widths themselves.
from environment import LEVEL_CHANNEL_WIDTHS

HUMAN_THRESHOLD = 1.30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episodes(
    results_root: str, level: int, model: str, tag: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load per-episode JSON files for one model/Level, optionally one run tag."""
    model_dir = os.path.join(results_root, f"level{level}", model)
    if tag:
        paths = sorted(glob.glob(os.path.join(model_dir, tag, "episode_*.json")))
    else:
        # Preserve legacy direct files.  For tagged runs, analyse only the most
        # recently modified run directory rather than mixing incompatible runs.
        paths = sorted(glob.glob(os.path.join(model_dir, "episode_*.json")))
        if not paths:
            run_dirs = [
                path
                for path in glob.glob(os.path.join(model_dir, "*"))
                if os.path.isdir(path)
            ]
            if run_dirs:
                latest = max(run_dirs, key=os.path.getmtime)
                paths = sorted(glob.glob(os.path.join(latest, "episode_*.json")))
    # Exclude the per-step sidecars written next to the episode records.
    paths = [p for p in paths if not p.endswith("_steps.json")]
    if not paths:
        # Legacy layout: results/level{level}/{model}/round*/episode_*.json
        paths = sorted(
            glob.glob(os.path.join(model_dir, "round*", "episode_*.json"))
        )
        paths = [p for p in paths if not p.endswith("_steps.json")]

    episodes: List[Dict[str, Any]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            # One damaged record must not take the whole analysis down: warn,
            # name the file, and carry on with the records that do parse.
            print(f"[analysis] WARNING: skipping unreadable {path} ({exc})")
            continue
        if not episode.get("steps"):
            sidecar = path[: -len(".json")] + "_steps.json"
            if os.path.exists(sidecar):
                try:
                    with open(sidecar, "r", encoding="utf-8") as handle:
                        episode["steps"] = json.load(handle)
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[analysis] WARNING: skipping unreadable {sidecar} ({exc})")
        episodes.append(episode)
    episodes.sort(key=lambda ep: int(ep.get("episode_id", 0)))
    return episodes


def _holds_episode_records(directory: str) -> bool:
    """True when ``directory`` directly contains at least one episode record.

    ``episode_*_steps.json`` sidecars live next to the records, and
    ``episode_*.json`` matches them too, so they are filtered out: a directory
    holding only sidecars has nothing to analyse.
    """
    for path in glob.glob(os.path.join(directory, "episode_*.json")):
        if not path.endswith("_steps.json"):
            return True
    return False


def discover_models(results_root: str) -> List[str]:
    """List models that have episode results under any Level.

    Three layouts are recognised, because all three exist in the wild:

    * ``level{n}/{model}/{tag}/episode_*.json`` -- what ``main.py`` writes for
      every tagged run, and therefore every run ``run_all_models.sh`` makes;
    * ``level{n}/{model}/episode_*.json`` -- untagged runs from before the tag
      became a directory component;
    * ``level{n}/{model}/round*/episode_*.json`` -- older still.

    The tagged form was missing here, which made a finished sweep invisible:
    ``python analysis.py`` printed "No episode results found" and exited 1
    while every episode sat on disk under its run tag.
    """
    models: set[str] = set()
    for level_dir in glob.glob(os.path.join(results_root, "level*")):
        if not os.path.isdir(level_dir):
            continue
        for name in sorted(os.listdir(level_dir)):
            model_dir = os.path.join(level_dir, name)
            if not os.path.isdir(model_dir):
                continue
            if _holds_episode_records(model_dir) or glob.glob(
                os.path.join(model_dir, "round*", "episode_*.json")
            ):
                models.add(name)
                continue
            # Tagged layout: one directory per run tag, named by the tag.
            for tag_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
                if os.path.isdir(tag_dir) and _holds_episode_records(tag_dir):
                    models.add(name)
                    break
    return sorted(models)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def episode_passed_sideways(episode: Dict[str, Any]) -> bool:
    """Fall back to the step trace when the field is missing (legacy data)."""
    if "passed_sideways" in episode:
        return bool(episode["passed_sideways"])
    if not episode.get("passed"):
        return False
    steps = episode.get("steps", [])
    if not steps:
        return False
    return _sideways_yaw(steps[-1].get("torso_rotation", 0.0))


def episode_passed(episode: Dict[str, Any]) -> bool:
    if "passed" in episode:
        return bool(episode["passed"])
    return bool(episode.get("success"))


def episode_first_turn_step(episode: Dict[str, Any]) -> Optional[int]:
    if episode.get("first_turn_step") is not None:
        return int(episode["first_turn_step"])
    for step in episode.get("steps", []):
        if str(step.get("action")) in ("turn_left", "turn_right"):
            return int(step.get("step", 0))
    return None


def episode_had_no_valid_call(episode: Dict[str, Any]) -> bool:
    """True when not one step produced a usable action.

    Such an episode is an infrastructure failure -- an HTTP error, an exhausted
    quota, a dead key -- and the runner records it exactly like a model that could
    not solve the task: ``passed=False``, ``end_reason=max_steps``.  Counted as a
    model failure it changes the headline number (measured on the committed corpus:
    215 of 3465 episodes, every one of them an HTTP 4xx/5xx in the agent log), so
    the rates exclude it and the count is reported separately.  A model whose whole
    run failed this way has no measured ability at all, which is not the same as
    scoring zero.
    """
    steps = int(episode.get("total_steps") or 0)
    invalid = int(episode.get("invalid_response_count") or 0)
    return steps > 0 and invalid >= steps


def summarize_level(level: int, episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the A/S metrics for one Level, over the episodes that produced data."""
    # Episodes in which every call failed are not measurements of the model; drop
    # them from the rates and report how many were dropped.  See
    # episode_had_no_valid_call.
    excluded = [ep for ep in episodes if episode_had_no_valid_call(ep)]
    usable = [ep for ep in episodes if not episode_had_no_valid_call(ep)]
    if not usable:
        return {
            "level": int(level),
            "episodes": 0,
            "excluded_no_valid_call": len(excluded),
        }
    episodes = usable

    passed = [episode_passed(ep) for ep in episodes]
    sideways = [episode_passed_sideways(ep) for ep in episodes]
    first_turns = [
        step for step in (episode_first_turn_step(ep) for ep in episodes) if step is not None
    ]
    success_steps = [
        int(ep.get("total_steps", len(ep.get("steps", []))))
        for ep in episodes
        if episode_passed(ep)
    ]
    sideways_success_steps = [
        int(ep.get("total_steps", len(ep.get("steps", []))))
        for ep in episodes
        if episode_passed(ep) and episode_passed_sideways(ep)
    ]

    return {
        "level": int(level),
        "channel_width": float(episodes[0].get("channel_width", float("nan"))),
        "a_s_ratio": float(episodes[0].get("a_s_ratio", float("nan"))),
        "episodes": len(episodes),
        "excluded_no_valid_call": len(excluded),
        "pass_rate": float(np.mean(passed)),
        "passed_count": int(sum(passed)),
        "sideways_rate": float(np.mean(sideways)),
        "passed_sideways_count": int(sum(sideways)),
        "turned_rate": float(
            np.mean([episode_first_turn_step(ep) is not None for ep in episodes])
        ),
        "turned_count": int(
            sum(1 for ep in episodes if episode_first_turn_step(ep) is not None)
        ),
        "first_turn_step_mean": (float(np.mean(first_turns)) if first_turns else None),
        "first_turn_step_min": (int(min(first_turns)) if first_turns else None),
        "avg_success_steps": (
            float(np.mean(success_steps)) if success_steps else None
        ),
        "avg_sideways_success_steps": (
            float(np.mean(sideways_success_steps)) if sideways_success_steps else None
        ),
        "avg_total_rotation_deg": float(
            np.mean(
                [
                    float(
                        ep.get(
                            "total_rotation",
                            sum(
                                15.0
                                for s in ep.get("steps", [])
                                if str(s.get("action")) in ("turn_left", "turn_right")
                            ),
                        )
                    )
                    for ep in episodes
                ]
            )
        ),
        # The shoulder rotation at the moment of passage, for the episodes that
        # crossed the wall plane.  This is the graded measurement: the human
        # reference is a curve of rotation angle against A/S, not a boundary.
        # Legacy records have no such field, so it stays None rather than 0.0 --
        # reporting a mean of 0 degrees would look like "never rotated".
        "avg_passage_rotation_deg": (
            float(
                np.mean(
                    [
                        float(ep["passage_rotation_deg"])
                        for ep in episodes
                        if ep.get("passage_rotation_deg") is not None
                    ]
                )
            )
            if any(ep.get("passage_rotation_deg") is not None for ep in episodes)
            else None
        ),
        "passage_rotation_deg_count": int(
            sum(1 for ep in episodes if ep.get("passage_rotation_deg") is not None)
        ),
    }


# A level at or above this A/S is "wide": the human A/S_crit is 1.30, so rotation is
# not required there.  Sideways or turning behaviour observed at a wide level is the
# background against which a threshold has to stand out.
WIDE_A_S = 1.0
# One-sided binomial significance level for calling a level's rate above background.
THRESHOLD_ALPHA = 0.05
# A threshold may not rest on a single episode: with 5 episodes per Level a lone
# chance sideways pass is 20%, and at the measured 5.7% per-episode wide-level
# sideways rate the probability of at least one such pass across the 11 wide Levels
# is about 96%.  Two passes are the minimum evidence for "this width made the model
# rotate".
THRESHOLD_MIN_COUNT = 2
# Used as the null for the WIDEST level, which has no wider level to compare with.
PRIOR_RATE = 0.05


def _binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if n <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return sum(
        math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        for i in range(k, n + 1)
    )


def _pooled_rate(
    summaries: Sequence[Dict[str, Any]], count_key: str
) -> Optional[float]:
    """Pooled rate of ``count_key`` over ``summaries``, or None if there are none.

    Laplace-smoothed, so a model with no events in the reference set does not make
    any later single event automatically significant.
    """
    hits = trials = 0
    for summary in summaries:
        n = int(summary.get("episodes", 0))
        if n <= 0:
            continue
        hits += int(summary.get(count_key, 0))
        trials += n
    if trials <= 0:
        return None
    return (hits + 0.5) / (trials + 1.0)


def _wide_background(
    level_summaries: Dict[int, Dict[str, Any]], count_key: str
) -> Optional[float]:
    """Pooled wide-Level (A/S >= WIDE_A_S) rate, for the report."""
    return _pooled_rate(
        [
            summary
            for summary in level_summaries.values()
            if summary.get("a_s_ratio") is not None
            and float(summary["a_s_ratio"]) >= WIDE_A_S
        ],
        count_key,
    )


def _significant_threshold(
    level_summaries: Dict[int, Dict[str, Any]], count_key: str
) -> Optional[float]:
    """Widest A/S whose count is significantly above the wider-Level background.

    The null for a candidate Level is the pooled rate at STRICTLY WIDER openings
    (excluding the candidate, so a Level cannot raise its own background); for the
    widest Level, which has none, it is PRIOR_RATE.  The old rule took the widest
    Level with ANY event, so one chance sideways pass at A/S 2.0 defined a model's
    threshold.
    """
    levels = [
        summary
        for summary in level_summaries.values()
        if int(summary.get("episodes", 0)) > 0
        and summary.get("a_s_ratio") is not None
    ]
    # A threshold is a CONTRAST between widths; with one Level there is nothing to
    # compare against, and a policy that simply always rotates would otherwise look
    # like a threshold at whatever that single width happens to be.
    if len(levels) < 2:
        return None
    candidates = []
    for summary in levels:
        n = int(summary.get("episodes", 0))
        k = int(summary.get(count_key, 0))
        if k < THRESHOLD_MIN_COUNT:
            continue
        a_s = float(summary["a_s_ratio"])
        background = _pooled_rate(
            [other for other in levels if float(other["a_s_ratio"]) > a_s],
            count_key,
        )
        if background is None:
            # The widest Level has nothing wider to compare with.  The null is then
            # the model's own behaviour at the OTHER wide Levels (A/S >= WIDE_A_S),
            # excluding the candidate: a model that rotates at every wide opening has
            # no threshold, it has a habit, and comparing only against a fixed prior
            # would call that habit significant.
            background = _pooled_rate(
                [
                    other
                    for other in levels
                    if other is not summary
                    and float(other["a_s_ratio"]) >= WIDE_A_S
                ],
                count_key,
            )
        if background is None:
            background = PRIOR_RATE
        candidates.append((float(_binomial_tail(k, n, background)), summary))
    if not candidates:
        return None
    # HOLM, NOT SEVENTEEN INDEPENDENT TESTS AT alpha = 0.05.
    #
    # The threshold is searched for over every Level, so 17 one-sided tests at 0.05 give a family-wise
    # chance of at least one false positive of 1 - 0.95**17 ~= 0.58 -- and the rule then reported the
    # WIDEST significant Level, which is precisely the rung most likely to be that false positive (a
    # wide opening where one chance sideways pass stands out against a near-zero background).  Holm
    # controls the family-wise error rate without assuming the tests are independent, which suits a
    # ladder whose neighbouring rungs are anything but independent.
    #
    # Step-down: sort the p-values ascending and require p_i <= alpha / (m - i); the first failure ends
    # the family, because every later (larger) p-value would face a stricter threshold.
    candidates.sort(key=lambda pair: pair[0])
    family = len(candidates)
    survivors = []
    for rank, (p_value, summary) in enumerate(candidates):
        if p_value <= THRESHOLD_ALPHA / (family - rank):
            survivors.append(summary)
        else:
            break
    if not survivors:
        return None
    return float(max(survivors, key=lambda s: float(s["a_s_ratio"]))["a_s_ratio"])


def sideways_threshold(level_summaries: Dict[int, Dict[str, Any]]) -> Optional[float]:
    """Widest A/S ratio at which the model still passed sideways, above noise.

    Requires at least THRESHOLD_MIN_COUNT sideways passes at the Level AND a
    one-sided binomial p < THRESHOLD_ALPHA against the model's own wide-Level
    sideways rate.  The previous rule took the widest Level with ANY sideways pass,
    which at the measured 5.7% per-episode wide-Level rate is a ~96% false
    positive across the 11 wide Levels; it scored gemini-2.5-pro at A/S 2.0
    ("anticipatory") on 4 sideways passes in the widest opening.
    """
    return _significant_threshold(level_summaries, "passed_sideways_count")


def turned_threshold(level_summaries: Dict[int, Dict[str, Any]]) -> Optional[float]:
    """Widest A/S ratio at which the model rotated its torso at all, above noise.

    Distinct from ``sideways_threshold``, which additionally requires the episode
    to have SUCCEEDED.  The human reference (Warren & Whang 1987, 1.30) measures
    whether the shoulders were rotated, not whether the person got through, so
    this is the closer analogue: it separates the decision from the execution and
    cannot be depressed by a model that rotates correctly and then fails to
    follow through.

    Same significance rule as ``sideways_threshold``: a single collision-recovery
    turn at the widest opening is not a threshold.  The random baseline used to
    score 2.0 here purely from the wide-Level turn rate (0.23-0.29, no trend).
    """
    return _significant_threshold(level_summaries, "turned_count")


def rotation_gradedness(
    level_summaries: Dict[int, Dict[str, Any]]
) -> Optional[str]:
    """Did the rotation rate actually vary with A/S, or was it a fixed policy?

    A threshold only means something if the behaviour changes across the ladder.
    A model that always rotates 90 degrees scores ``turned_rate`` = 1.0 at every
    width -- including widths it could walk straight through -- and has no
    threshold at all, just a habit; one that never rotates has no threshold either.
    The animal studies control for exactly this ("a simple preference for the
    conveniently sized opening", Lenkei et al. 2019), and with a binary per-Level
    rate it is cheap to check here.

    Returns "graded", "flat_always", "flat_never", or None when there is nothing
    to judge.
    """
    rates = [
        float(summary.get("turned_rate", 0.0))
        for summary in level_summaries.values()
        if int(summary.get("episodes", 0)) > 0
    ]
    if not rates:
        return None
    if min(rates) >= 0.99:
        return "flat_always"
    if max(rates) <= 0.01:
        return "flat_never"
    return "graded"


def anticipation_class(threshold: Optional[float]) -> str:
    """Bucket a threshold against the human reference of 1.30."""
    if threshold is None:
        return "no_sideways"
    if threshold >= HUMAN_THRESHOLD:
        return "anticipatory"
    if threshold >= 1.0:
        return "borderline"
    return "reactive"


def _current_protocol_tags(
    results_root: str, model: str, levels: Sequence[int]
) -> List[str]:
    """Every run tag under ``results/level*/{model}`` that is the current protocol.

    The tag carries the protocol version (``protocol.PROTOCOL_TAG``, appended by
    ``main.effective_tag``) and the scene, so a tag is one scene of the current
    protocol.  Matched by CONTAINMENT, not by suffix: a tag can also carry a
    request-parameter suffix after the protocol tag ("...-<protocol>-effortnone",
    for models whose extended thinking is switched off), and an endswith test
    classified every one of those as a different protocol.
    """
    from protocol import PROTOCOL_TAG

    tags: set = set()
    for level in levels:
        for path in glob.glob(
            os.path.join(results_root, f"level{level}", model, "*")
        ):
            if not os.path.isdir(path) or os.path.basename(path).startswith("round"):
                continue
            name = os.path.basename(path)
            if f"-{PROTOCOL_TAG}" in name or name == PROTOCOL_TAG:
                tags.add(name)
    return sorted(tags)


def _episode_count(results_root: str, model: str, tag: str, levels: Sequence[int]) -> int:
    """How many episode records a tag actually holds, across the requested Levels."""
    total = 0
    for level in levels:
        for path in glob.glob(
            os.path.join(results_root, f"level{level}", model, f"*{tag}*.json")
        ):
            if not path.endswith("_steps.json"):
                total += 1
    return total


def choose_tag(
    results_root: str, model: str, levels: Sequence[int]
) -> Optional[str]:
    """The ONE tag to report for ``model``, or None when it has nothing.

    POOLING EVERY MATCHING TAG DOUBLE-COUNTS.  Two tags that both carry the current protocol tag --
    say ``<model>-v8-actions-flipped`` and ``<model>-v8-actions-flipped-stage1.1`` -- are two runs of
    the same protocol, and adding their episodes together turns five conditions into ten while
    presenting them as ten independent observations.  The README is explicit that the five markers are
    five CONDITIONS rather than replicates, so inflating n is not cosmetic: every rate, and every test
    built on it, runs on the wrong denominator.

    So a model contributes ONE tag: the one holding the most episode records over the requested
    Levels, and the most recently written directory when they tie.  A caller that genuinely wants a
    different tag passes --tag explicitly, which analyze_model still honours.
    """
    tags = _current_protocol_tags(results_root, model, levels)
    if not tags:
        return None
    if len(tags) == 1:
        return tags[0]

    def rank(name: str) -> Any:
        directory = os.path.join(results_root, f"level{levels[0]}", model, name) \
            if levels else os.path.join(results_root, model, name)
        try:
            mtime = os.path.getmtime(directory)
        except OSError:
            mtime = 0.0
        # Most episodes wins; then the SHORTEST name, because a suffix such as "-effortnone" records
        # a request parameter rather than a different experiment, and the plain tag is the one a
        # reader expects to see quoted; then the newest directory.  Without the length term the choice
        # between two equally-sized tags would come down to mtimes, which differ per checkout.
        return (_episode_count(results_root, model, name, levels), -len(name), mtime)

    return max(tags, key=rank)


def analyze_model(
    results_root: str,
    model: str,
    levels: Sequence[int],
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the full per-model report across every requested Level.

    With ``tag`` given, that one run tag is analysed.  With ``tag is None`` every
    current-protocol tag is pooled.  It used to pick the single most recently
    modified tag directory, which meant a default run covered one scene of five and
    a partially re-run (or merely ``touch``ed) tag could beat a complete one.
    """
    selected_tags: List[str] = []
    if tag is not None:
        selected_tags = [tag]
    else:
        tagged_dirs = [
            path
            for level in levels
            for path in glob.glob(
                os.path.join(results_root, f"level{level}", model, "*")
            )
            if os.path.isdir(path) and not os.path.basename(path).startswith("round")
        ]
        if tagged_dirs:
            current = _current_protocol_tags(results_root, model, levels)
            if current:
                # ONE TAG PER MODEL.  See choose_tag: adding two tags of the same protocol together
                # presents five marker CONDITIONS as ten independent observations.
                chosen = choose_tag(results_root, model, levels)
                selected_tags = [chosen] if chosen else []
                if len(current) > 1:
                    ignored = [t for t in current if t != chosen]
                    print(
                        f"[analysis] {model}: analysing tag {chosen!r}; "
                        f"NOT pooling {len(ignored)} other current-protocol tag(s) "
                        f"({', '.join(ignored)}) -- they are repeat runs, not replicates. "
                        f"Pass --tag to pick one explicitly."
                    )
                else:
                    print(f"[analysis] {model}: analysing tag {current[0]!r}")
            else:
                # Nothing from the current protocol: fall back to the newest tag so
                # an old tree still reports, but say plainly that it is not this
                # protocol's data.
                from protocol import PROTOCOL_TAG

                fallback = os.path.basename(max(tagged_dirs, key=os.path.getmtime))
                selected_tags = [fallback]
                print(
                    f"[analysis] WARNING: no {model} results tagged with the "
                    f"current protocol ({PROTOCOL_TAG}); using {fallback!r}. "
                    "Run main.py with a fresh tag before comparing."
                )

    def level_episodes(level: int) -> List[Dict[str, Any]]:
        if not selected_tags:
            # Untagged / legacy layout: let load_episodes pick its own directory.
            return load_episodes(results_root, level, model, tag=None)
        episodes: List[Dict[str, Any]] = []
        for selected in selected_tags:
            episodes.extend(load_episodes(results_root, level, model, tag=selected))
        return episodes

    # Load each Level once: pooling several tags made the old double load
    # (per-Level summary, then the overall aggregate) multiply by the tag count.
    loaded: Dict[int, List[Dict[str, Any]]] = {
        int(level): level_episodes(level) for level in levels
    }
    per_level: Dict[int, Dict[str, Any]] = {}
    for level in levels:
        episodes = loaded[int(level)]
        if not episodes:
            continue
        per_level[int(level)] = summarize_level(int(level), episodes)

    threshold = sideways_threshold(per_level)
    turned = turned_threshold(per_level)
    every_episode = [ep for level in levels for ep in loaded[int(level)]]
    # Episodes whose every call failed are not a measurement of the model; see
    # episode_had_no_valid_call.  Reported, never scored.
    excluded_no_valid_call = sum(
        1 for ep in every_episode if episode_had_no_valid_call(ep)
    )
    all_episodes = [
        ep for ep in every_episode if not episode_had_no_valid_call(ep)
    ]
    total_passed = sum(1 for ep in all_episodes if episode_passed(ep))
    total_sideways = sum(1 for ep in all_episodes if episode_passed_sideways(ep))

    return {
        "model": model,
        "tags": list(selected_tags),
        "levels": per_level,
        "excluded_no_valid_call": excluded_no_valid_call,
        "sideways_threshold": threshold,
        "turned_threshold": turned,
        # The model's own wide-Level behaviour, which the thresholds above are tested
        # against; reported so a reader can see the noise floor the rule removes.
        "sideways_background_rate": (
            _wide_background(per_level, "passed_sideways_count") if per_level else None
        ),
        "turned_background_rate": (
            _wide_background(per_level, "turned_count") if per_level else None
        ),
        "threshold_rule": {
            "wide_a_s": WIDE_A_S,
            "alpha": THRESHOLD_ALPHA,
            "min_count": THRESHOLD_MIN_COUNT,
        },
        "human_reference_threshold": HUMAN_THRESHOLD,
        "threshold_gap_vs_human": (
            None if threshold is None else float(threshold - HUMAN_THRESHOLD)
        ),
        "threshold_class": anticipation_class(threshold),
        "turned_threshold_class": anticipation_class(turned),
        "rotation_gradedness": rotation_gradedness(per_level),
        "turned_threshold_gap_vs_human": (
            None if turned is None else float(turned - HUMAN_THRESHOLD)
        ),
        "overall": {
            "episodes": len(all_episodes),
            "excluded_no_valid_call": excluded_no_valid_call,
            "passed_count": total_passed,
            "pass_rate": (total_passed / len(all_episodes)) if all_episodes else 0.0,
            "passed_sideways_count": total_sideways,
            "sideways_rate": (
                total_sideways / len(all_episodes) if all_episodes else 0.0
            ),
            "first_turn_step_mean": (
                float(
                    np.mean(
                        [
                            step
                            for step in (
                                episode_first_turn_step(ep) for ep in all_episodes
                            )
                            if step is not None
                        ]
                    )
                )
                if any(episode_first_turn_step(ep) is not None for ep in all_episodes)
                else None
            ),
            "avg_success_steps": (
                float(
                    np.mean(
                        [
                            int(ep.get("total_steps", 0))
                            for ep in all_episodes
                            if episode_passed(ep)
                        ]
                    )
                )
                if total_passed
                else None
            ),
            "total_wall_collisions": int(
                sum(int(ep.get("wall_collision_count", 0)) for ep in all_episodes)
            ),
            "total_invalid_responses": int(
                sum(
                    int(ep.get("invalid_response_count", 0)) for ep in every_episode
                )
            ),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_markdown_table(
    reports: Sequence[Dict[str, Any]], levels: Sequence[int]
) -> str:
    """Per-(model, level) table plus a threshold summary per model."""
    header = [
        "Model",
        "Level",
        "A/S",
        "Width (m)",
        "Pass %",
        "Turned %",
        "Sideways %",
        "1st Turn",
        "Rot@Pass",
        "Avg Pass Steps",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for report in reports:
        model = report["model"]
        for level in levels:
            summary = report["levels"].get(int(level))
            # A Level whose every call failed has no measured metric fields, only
            # episodes=0 and the excluded count (see summarize_level).  It must be
            # skipped, not indexed: the report crashed with KeyError: 'a_s_ratio'
            # on the real corpus (kimi, all 85 episodes API-failed).
            if not summary or not summary.get("episodes"):
                continue
            first_turn = summary.get("first_turn_step_mean")
            avg_steps = summary.get("avg_success_steps")
            rot_at_pass = summary.get("avg_passage_rotation_deg")
            values = [
                str(model),
                str(level),
                f"{summary['a_s_ratio']:.2f}",
                f"{summary['channel_width']:.2f}",
                f"{100.0 * summary['pass_rate']:.1f}",
                f"{100.0 * summary['turned_rate']:.1f}",
                f"{100.0 * summary['sideways_rate']:.1f}",
                "-" if first_turn is None else f"{first_turn:.1f}",
                "-" if rot_at_pass is None else f"{rot_at_pass:.1f}",
                "-" if avg_steps is None else f"{avg_steps:.2f}",
            ]
            lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    threshold_header = [
        "Model",
        "Turned Threshold (A/S)",
        "Sideways Threshold (A/S)",
        "Human Reference",
        "Gap vs Human",
        "Class",
        "Graded?",
    ]
    lines.append("| " + " | ".join(threshold_header) + " |")
    lines.append("|" + "---|" * len(threshold_header))
    for report in reports:
        threshold = report["sideways_threshold"]
        turned = report.get("turned_threshold")
        gap = report["threshold_gap_vs_human"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(report["model"]),
                    "-" if turned is None else f"{turned:.2f}",
                    "-" if threshold is None else f"{threshold:.2f}",
                    f"{HUMAN_THRESHOLD:.2f}",
                    "-" if gap is None else f"{gap:+.2f}",
                    str(report["threshold_class"]),
                    str(report.get("rotation_gradedness") or "-"),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Turned Threshold is the widest A/S at which the torso was rotated at all;"
    )
    lines.append(
        "Sideways Threshold additionally requires the episode to have succeeded."
    )
    lines.append(
        "The human reference of 1.30 measures whether the shoulders were rotated,"
    )
    lines.append("so Turned Threshold is the closer analogue.")
    lines.append(
        "Graded? is flat_always / flat_never when the rotation rate does not vary"
    )
    lines.append(
        "with A/S at all: a fixed policy has no threshold, whatever the number says."
    )
    return "\n".join(lines)


def table_rows(
    reports: Sequence[Dict[str, Any]], levels: Sequence[int]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in reports:
        for level in levels:
            summary = report["levels"].get(int(level))
            # Same guard as format_markdown_table: an all-failed Level has no metric
            # fields and must be skipped rather than indexed.
            if not summary or not summary.get("episodes"):
                continue
            rows.append(
                {
                    "model": report["model"],
                    "level": level,
                    "channel_width": summary["channel_width"],
                    "a_s_ratio": summary["a_s_ratio"],
                    "episodes": summary["episodes"],
                    "pass_rate": summary["pass_rate"],
                    "sideways_rate": summary["sideways_rate"],
                    "turned_rate": summary["turned_rate"],
                    "passed_sideways_count": summary["passed_sideways_count"],
                    "first_turn_step_mean": summary["first_turn_step_mean"],
                    "avg_success_steps": summary["avg_success_steps"],
                    "avg_total_rotation_deg": summary["avg_total_rotation_deg"],
                    "avg_passage_rotation_deg": summary.get(
                        "avg_passage_rotation_deg"
                    ),
                }
            )
        rows.append(
            {
                "model": report["model"],
                "level": "ALL",
                "sideways_threshold": report["sideways_threshold"],
                "turned_threshold": report.get("turned_threshold"),
                "threshold_class": report["threshold_class"],
                "turned_threshold_class": report.get("turned_threshold_class"),
                "overall_pass_rate": report["overall"]["pass_rate"],
                "overall_sideways_rate": report["overall"]["sideways_rate"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def plot_thresholds(
    reports: Sequence[Dict[str, Any]], levels: Sequence[int], out_path: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    for report in reports:
        xs = [
            report["levels"][int(level)]["a_s_ratio"]
            for level in levels
            if int(level) in report["levels"]
        ]
        ys = [
            100.0 * report["levels"][int(level)]["sideways_rate"]
            for level in levels
            if int(level) in report["levels"]
        ]
        if not xs:
            continue
        order = np.argsort(xs)
        plt.plot(
            np.asarray(xs)[order],
            np.asarray(ys)[order],
            marker="o",
            label=str(report["model"]),
        )
        threshold = report["sideways_threshold"]
        if threshold is not None:
            plt.axvline(
                threshold,
                linestyle=":",
                alpha=0.6,
                label=f"{report['model']} threshold {threshold:.2f}",
            )

    plt.axvline(
        HUMAN_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"human threshold {HUMAN_THRESHOLD:.2f}",
    )
    plt.axvline(1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    plt.xlabel("A/S ratio (channel width / shoulder width)")
    plt.ylabel("Sideways passage rate (%)")
    plt.title("Sideways passage vs channel-to-shoulder ratio")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_results_root() -> str:
    return os.path.join(_project_root(), "results")


def _default_out_dir() -> str:
    return os.path.join(_project_root(), "analysis")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the EmbodiedBAO A/S threshold results."
    )
    # Defaults are anchored to the repository so `python analysis.py` reports
    # on the run that was just produced, wherever it is invoked from.
    parser.add_argument("--results_root", type=str, default=_default_results_root())
    parser.add_argument(
        "--levels", type=int, nargs="+", default=None,
        help=(
            "Levels to report on; default: the WHOLE ladder.  This used to default to "
            "[0, 1, 2, 3, 4, 5], so `python analysis.py` -- the command the README documents -- "
            "skipped levels 6-16, which is exactly where the threshold is: the wide levels are all "
            "passable without rotating, so a default report that stops at 5 cannot show a threshold "
            "at all.  The ladder has %d levels." % len(LEVEL_CHANNEL_WIDTHS)
        ),
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None, help="Model names; default: all"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Analyze one run tag; default: latest tagged run (or legacy files)",
    )
    parser.add_argument("--out_dir", type=str, default=_default_out_dir())
    parser.add_argument("--no_plot", action="store_true", help="Skip matplotlib plots")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.levels is None:
        # The whole ladder, not a prefix of it.  See the --levels help: defaulting to 0-5 hid the
        # only levels where a threshold can appear.
        args.levels = list(range(len(LEVEL_CHANNEL_WIDTHS)))
    models = args.models or discover_models(args.results_root)
    if not models:
        print(f"No episode results found under {args.results_root}.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    reports = [
        analyze_model(args.results_root, model, args.levels, tag=args.tag)
        for model in models
    ]
    reports = [r for r in reports if r["levels"]]
    if not reports:
        print("No analyzable episodes.")
        return 1

    for report in reports:
        path = os.path.join(
            args.out_dir,
            f"threshold_{report['model'].replace('/', '-')}.json",
        )
        # Atomic: a report that is half-written is worse than a stale one, and
        # these files are what the paper's tables are generated from.
        atomic_write_json(path, report)

    markdown = format_markdown_table(reports, args.levels)
    for name in ("threshold_table.md", "threshold_table.txt"):
        atomic_write_text(os.path.join(args.out_dir, name), markdown + "\n")

    rows = table_rows(reports, args.levels)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(
            os.path.join(args.out_dir, "threshold_table.csv"), buffer.getvalue()
        )

    if not args.no_plot:
        try:
            plot_thresholds(
                reports, args.levels, os.path.join(args.out_dir, "thresholds.png")
            )
        except Exception as exc:
            print(f"[analysis] plotting failed: {exc}")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
