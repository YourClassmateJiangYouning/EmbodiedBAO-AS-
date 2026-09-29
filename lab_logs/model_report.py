"""Per-model and cross-model analysis of the Stage 1 sweep (660 episodes).

For every model: its pass rate and turn rate at each A/S, the actions it took, when it
looked down at its own body and whether that came with success, what its own reasoning
talks about, and representative quotes.  Then the eleven of them together.

Writes three things, all recomputed from lab_logs/bao_v7_all.tgz rather than typed:

  lab_logs/figures/stage1_per_model.pdf    one page per model, plus the cross-model pages
  lab_logs/figures/stage1_per_model/*.png  the same pages as images, for quick reading
  lab_logs/PER_MODEL_ANALYSIS.md           the same analysis as text, for the write-up
  lab_logs/model_level_table.csv           model x level metrics, one row per cell

Run from the repository root:  python lab_logs/model_report.py
"""

import collections
import csv
import json
import os
import re
import sys
import tarfile
import textwrap

sys.path.insert(0, ".")
import memory_metrics as mm  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ARCHIVE = "lab_logs/bao_v7_all.tgz"
OUTDIR = "lab_logs/figures"
PNGDIR = os.path.join(OUTDIR, "stage1_per_model")
PDF_PATH = os.path.join(OUTDIR, "stage1_per_model.pdf")
MD_PATH = "lab_logs/PER_MODEL_ANALYSIS.md"
CSV_PATH = "lab_logs/model_level_table.csv"

# Same journal block as make_figures.py / make_memory_figures.py.
TEXT_WIDTH_CM = 12.2
FONT_PT = 8
RASTER_DPI = 600
FONT_SANS = ["Arial", "Helvetica", "DejaVu Sans"]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000"]
CM = 1.0 / 2.54
WALL_X = 8.0
DOOR_X = 7.0
LEARNING_LEVELS = (10, 11)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT_SANS,
    "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT,
    "xtick.labelsize": FONT_PT - 1, "ytick.labelsize": FONT_PT - 1,
    "legend.fontsize": FONT_PT - 1, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "lines.markersize": 3.5, "legend.frameon": False,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

REASONING = re.compile(r'"reasoning":\s*"((?:[^"\\]|\\.)*)"')
AGENT = re.compile(r"level(\d+)_episode(\d+)_agent\.txt$")
TOPICS = {
    "opening": r"opening|aperture|\bgap\b",
    "turn": r"rotat|turn my torso|turn (left|right)|angle",
    "width": r"shoulder|body width|how wide|my width|narrow|too wide|clearance|fit through|fit in",
    "blocked": r"blocked|collision|collide|stuck",
    "budget": r"budget|steps left|remaining steps|running out",
    "align": r"align|centre|center|straight|midline",
    "sideways": r"sideways|side-on|profile|shoulder-first",
}


def load():
    tf = tarfile.open(ARCHIVE)
    records, steps, logs = {}, {}, {}
    for member in tf.getmembers():
        name = member.name
        parts = name.split("/")
        base = parts[-1]
        if name.startswith("logs/") and base.endswith("_agent.txt"):
            match = AGENT.search(base)
            if match:
                logs[(parts[1].split("-v7-state-axes")[0], int(match.group(1)),
                      int(match.group(2)))] = tf.extractfile(member).read().decode("utf-8", "replace")
        elif base.endswith("_steps.json") and len(parts) > 4:
            key = (parts[2], int(parts[1].replace("level", "")),
                   int(base.replace(".json", "").split("_")[1]))
            steps[key] = json.load(tf.extractfile(member))
        elif base.endswith(".json") and "/episode_" in name and "_steps" not in name and len(parts) > 4:
            record = json.load(tf.extractfile(member))
            records[(record["model_name"], int(record["level"]), int(record["episode_id"]))] = record

    episodes = []
    for key, record in records.items():
        sidecar = steps.get(key, [])
        actions = [s.get("action") for s in sidecar]
        positions = [float(s.get("position_x") or 0.0) for s in sidecar]
        turns = [i for i, a in enumerate(actions) if a in ("turn_left", "turn_right")]
        looks = [i for i, a in enumerate(actions) if a == "look_down"]
        reasoning = " ".join(
            " ".join(r.split()) for r in REASONING.findall(logs.get(key, ""))
            if len(r.strip()) >= 20 and r != "<brief reasoning>")
        episodes.append({
            "model": record["model_name"], "level": int(record["level"]),
            "episode": int(record["episode_id"]), "record": record,
            "width": float(record["channel_width"]), "a_s": float(record["a_s_ratio"]),
            "passed": bool(record.get("passed")), "steps": int(record["total_steps"]),
            "max_rot": float(record.get("max_rotation_deg") or 0.0),
            "passage_rot": record.get("passage_rotation_deg"),
            "excess": int(record["total_steps"] - mm.optimal_steps(float(record["channel_width"]))),
            "n_turn": len(turns), "n_look_down": len(looks), "n_lateral": sum(
                1 for a in actions if a in ("left", "right")),
            "n_forward": sum(1 for a in actions if a == "forward"),
            "first_turn_step": turns[0] if turns else None,
            "first_turn_x": positions[turns[0]] if turns else None,
            "first_look_step": looks[0] if looks else None,
            "first_look_x": positions[looks[0]] if looks else None,
            "looked_down": bool(looks),
            "looked_down_early": bool(looks) and positions[looks[0]] < 4.0,
            "reached_door": any(p >= DOOR_X for p in positions),
            "actions": actions, "reasoning": reasoning,
            "label": mm.strategy_label(
                sum(1 for a in actions if a in ("left", "right")), len(turns),
                positions[turns[0]] if turns else None),
        })
    return episodes


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def per_level(episodes):
    table = {}
    for level in sorted({e["level"] for e in episodes}):
        rows = [e for e in episodes if e["level"] == level]
        passes = [e for e in rows if e["passed"]]
        table[level] = {
            "a_s": rows[0]["a_s"], "width": rows[0]["width"], "n": len(rows),
            "passed": len(passes), "pass_pct": pct(len(passes), len(rows)),
            "needs_turns": mm.min_turns(rows[0]["width"]),
            "turn_pct": pct(sum(1 for e in rows if e["n_turn"]), len(rows)),
            "sideways_pass_pct": pct(sum(1 for e in passes if e["passage_rot"] is not None
                                         and mm.is_sideways_yaw(float(e["passage_rot"]))),
                                     max(len(passes), 1)),
            "mean_steps": mean([e["steps"] for e in rows]),
            "mean_max_rot": mean([e["max_rot"] for e in rows]),
            "mean_excess": mean([e["excess"] for e in rows]),
            "look_pct": pct(sum(1 for e in rows if e["looked_down"]), len(rows)),
            "look_n": sum(1 for e in rows if e["looked_down"]),
            "n_nolook": sum(1 for e in rows if not e["looked_down"]),
            "pass_if_look": pct(sum(1 for e in rows if e["looked_down"] and e["passed"]),
                                max(sum(1 for e in rows if e["looked_down"]), 1)),
            "pass_if_no_look": pct(sum(1 for e in rows if not e["looked_down"] and e["passed"]),
                                   max(sum(1 for e in rows if not e["looked_down"]), 1)),
        }
    return table


def model_summary(episodes):
    wide = [e for e in episodes if mm.min_turns(e["width"]) == 0]
    narrow = [e for e in episodes if e["level"] in LEARNING_LEVELS]
    turned = [e for e in episodes if e["n_turn"]]
    labels = collections.Counter(e["label"] for e in episodes)
    action_mix = collections.Counter()
    for e in episodes:
        action_mix.update(e["actions"])
    total_actions = sum(action_mix.values()) or 1
    topics = collections.Counter()
    topic_hits = collections.Counter()
    for e in episodes:
        low = e["reasoning"].lower()
        for name, pattern in TOPICS.items():
            topics[name] += len(re.findall(pattern, low))
            if re.search(pattern, low):
                topic_hits[name] += 1
    looked = [e for e in episodes if e["looked_down"]]
    # Pooling every aperture would confound this: looking down is concentrated at the
    # narrow Levels, where the pass rate is low anyway.  The comparison that means
    # something is within an aperture, then averaged over the apertures that have both
    # groups present.
    matched, weights = [], []
    for level in sorted({e["level"] for e in episodes}):
        rows = [e for e in episodes if e["level"] == level]
        look = [e for e in rows if e["looked_down"]]
        nolook = [e for e in rows if not e["looked_down"]]
        if look and nolook:
            matched.append(pct(sum(1 for e in look if e["passed"]), len(look))
                           - pct(sum(1 for e in nolook if e["passed"]), len(nolook)))
            weights.append(len(rows))
    return {
        "n": len(episodes),
        "pass_pct": pct(sum(1 for e in episodes if e["passed"]), len(episodes)),
        "wide_turn_pct": pct(sum(1 for e in wide if e["n_turn"]), len(wide)),
        "wide_pass_pct": pct(sum(1 for e in wide if e["passed"]), len(wide)),
        "narrow_pass": "%d/%d" % (sum(1 for e in narrow if e["passed"]), len(narrow)),
        "turned_pct": pct(len(turned), len(episodes)),
        "first_turn_x": mean([e["first_turn_x"] for e in turned]),
        "mean_max_rot": mean([e["max_rot"] for e in episodes]),
        "mean_steps": mean([e["steps"] for e in episodes]),
        "mean_excess": mean([e["excess"] for e in episodes]),
        "look_pct": pct(len(looked), len(episodes)),
        "look_pass": pct(sum(1 for e in looked if e["passed"]), max(len(looked), 1)),
        "no_look_pass": pct(sum(1 for e in episodes if not e["looked_down"] and e["passed"]),
                            max(sum(1 for e in episodes if not e["looked_down"]), 1)),
        "look_first_x": mean([e["first_look_x"] for e in looked]),
        # Level-matched: weighted mean of (pass rate with inspection - without), and how
        # many apertures went each way.
        "look_matched_diff": (sum(d * w for d, w in zip(matched, weights)) / sum(weights)
                              if weights else 0.0),
        "look_levels_pos": sum(1 for d in matched if d > 0),
        "look_levels_neg": sum(1 for d in matched if d < 0),
        "look_levels_n": len(matched),
        "labels": labels,
        "action_mix": {k: 100.0 * v / total_actions for k, v in action_mix.most_common()},
        "topic_hits": {k: pct(v, len(episodes)) for k, v in topic_hits.items()},
    }


def quotes(episodes, limit=4):
    """Reasoning worth reading: a wide pass, a narrow pass, a narrow failure."""
    picked = []
    for wanted in ({"level": 0, "passed": True}, {"level": 11, "passed": True},
                   {"level": 11, "passed": False}, {"level": 10, "passed": False}):
        for e in episodes:
            if e["level"] == wanted["level"] and e["passed"] == wanted["passed"] and e["reasoning"]:
                snippet = " ".join(e["reasoning"].split())
                picked.append((e, snippet[:400] + ("…" if len(snippet) > 400 else "")))
                break
    return picked[:limit]


def draw_text(axis, paragraphs, width=76, fontsize=FONT_PT - 1):
    """Wrapped multi-line text in an empty axes.

    matplotlib's own wrap only breaks at the figure edge, which is what made the first
    version of this report unreadable: sentences ran past the page and the block grew
    upward into the table above it.  Wrapping to a measured character count and drawing
    one text object with a fixed linespacing keeps the block inside its own row.
    """
    body = "\n".join(textwrap.fill(p, width=width) for p in paragraphs)
    axis.text(0.0, 1.0, body, fontsize=fontsize, va="top", ha="left",
              linespacing=1.55, transform=axis.transAxes)
    axis.axis("off")


def model_page(pdf, model, episodes, summary, table):
    figure = plt.figure(figsize=(TEXT_WIDTH_CM * CM, 23.0 * CM))
    figure.suptitle("%s   —   %d episodes, overall pass %.0f%%" % (
        model, summary["n"], summary["pass_pct"]), fontsize=FONT_PT + 2, y=0.985)
    grid = figure.add_gridspec(
        4, 2, height_ratios=[2.9, 2.9, 7.4, 7.6], hspace=0.78, wspace=0.42,
        top=0.938, bottom=0.012, left=0.11, right=0.98)

    levels = sorted(table)
    ratios = [table[l]["a_s"] for l in levels]
    axis = figure.add_subplot(grid[0, 0])
    axis.plot(ratios, [table[l]["pass_pct"] for l in levels], "o-", color=PALETTE[0])
    axis.axvline(1.3, color="0.45", lw=0.6, ls="--")
    axis.set_ylim(-5, 112)
    axis.set_ylabel("pass rate (%)")
    axis.set_xlabel("A/S")
    axis.set_title("outcome vs aperture")
    axis.invert_xaxis()

    axis = figure.add_subplot(grid[0, 1])
    axis.plot(ratios, [table[l]["turn_pct"] for l in levels], "s-", color=PALETTE[1],
              label="turned (%)")
    axis.plot(ratios, [table[l]["mean_max_rot"] for l in levels], "^-", color=PALETTE[2],
              label="mean max rotation (deg)")
    axis.plot(ratios, [table[l]["mean_steps"] for l in levels], "d-", color=PALETTE[3],
              label="mean steps")
    axis.set_xlabel("A/S")
    axis.set_ylim(-5, 118)
    axis.set_title("behaviour vs aperture")
    axis.legend(loc="center left", fontsize=FONT_PT - 3, ncol=1, handlelength=1.2,
                framealpha=1.0, facecolor="white", edgecolor="none", borderaxespad=0.3)
    axis.invert_xaxis()

    axis = figure.add_subplot(grid[1, 0])
    looked = [e for e in episodes if e["looked_down"]]
    bars = [table[l]["pass_if_look"] if table[l]["look_n"] else 0 for l in levels]
    nobars = [table[l]["pass_if_no_look"] if table[l]["n_nolook"] else 0 for l in levels]
    xs = range(len(levels))
    axis.bar([x - 0.2 for x in xs], bars, width=0.4, color=PALETTE[0], label="inspected body")
    axis.bar([x + 0.2 for x in xs], nobars, width=0.4, color=PALETTE[4], label="never inspected")
    axis.set_xticks(list(xs))
    axis.set_xticklabels(["%.1f" % table[l]["a_s"] for l in levels], rotation=90,
                         fontsize=FONT_PT - 2)
    axis.set_ylabel("pass rate (%)")
    axis.set_ylim(0, 118)
    axis.set_title("self-inspection and outcome, within each aperture")
    axis.legend(loc="lower left", fontsize=FONT_PT - 2, handlelength=1.0,
                borderaxespad=0.2)

    axis = figure.add_subplot(grid[1, 1])
    early = sum(1 for e in looked if e["looked_down_early"])
    door = sum(1 for e in looked if e["first_look_x"] is not None and e["first_look_x"] >= DOOR_X)
    mid = len(looked) - early - door
    axis.bar([0, 1, 2], [early, mid, door], color=[PALETTE[0], PALETTE[5], PALETTE[1]])
    axis.set_xticks([0, 1, 2])
    axis.set_xticklabels(["before x=4", "4-7 m", "at the door"], fontsize=FONT_PT - 1)
    axis.set_ylabel("episodes")
    axis.set_ylim(0, max(early, mid, door, 1) * 1.25)
    axis.set_title("when it inspected itself (mean x %.1f m)" % summary["look_first_x"])

    axis = figure.add_subplot(grid[2, :])
    axis.axis("off")
    axis.set_title("per aperture   (side% = share of that aperture's passes made sideways;"
                   "  '–' = no episode of that kind)",
                   fontsize=FONT_PT - 2, pad=6)
    header = ["A/S", "width", "pass", "turn%", "side%", "steps", "maxrot", "excess",
              "look%", "pass|look", "pass|nolook"]
    body = [[
        "%.1f" % table[l]["a_s"], "%.3f" % table[l]["width"],
        "%d/%d" % (table[l]["passed"], table[l]["n"]), "%.0f" % table[l]["turn_pct"],
        "%.0f" % table[l]["sideways_pass_pct"], "%.1f" % table[l]["mean_steps"],
        "%.0f" % table[l]["mean_max_rot"], "%.1f" % table[l]["mean_excess"],
        "%.0f" % table[l]["look_pct"],
        "–" if table[l]["look_n"] == 0 else "%.0f" % table[l]["pass_if_look"],
        "–" if table[l]["n_nolook"] == 0 else "%.0f" % table[l]["pass_if_no_look"],
    ] for l in levels]
    rendered = axis.table(cellText=body, colLabels=header, loc="center",
                          cellLoc="center", colLoc="center", bbox=[0.0, 0.0, 1.0, 1.0],
                          colWidths=[0.075, 0.085, 0.075, 0.075, 0.075, 0.075, 0.09,
                                     0.09, 0.075, 0.12, 0.13])
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(FONT_PT - 2)

    axis = figure.add_subplot(grid[3, :])
    topics = summary["topic_hits"]
    draw_text(axis, [
        "Columns: pass|look / pass|nolook are this model's pass rate inside that aperture "
        "when it inspected its own body and when it did not; look% is how often it did.",
        "Reasoning mentions (share of episodes): " + ", ".join(
            "%s %.0f%%" % (k, v) for k, v in sorted(topics.items(), key=lambda kv: -kv[1])),
        "Actions: " + ", ".join("%s %.1f%%" % (k, v)
                                for k, v in list(summary["action_mix"].items())[:6]),
        "Strategy labels: " + ", ".join("%s %.0f%%" % (k, pct(v, summary["n"]))
                                        for k, v in summary["labels"].most_common()),
        "It turned in %.0f%% of episodes (first turn at x = %.1f m on average), and in %.0f%% "
        "of the episodes at apertures that need no turn at all; its passes waste %.1f steps "
        "over the optimum on average." % (
            summary["turned_pct"], summary["first_turn_x"], summary["wide_turn_pct"],
            summary["mean_excess"]),
        "Inspecting its own body, compared inside each aperture: %+.1f percentage points on "
        "the pass rate (%d of %d apertures positive)." % (
            summary["look_matched_diff"], summary["look_levels_pos"], summary["look_levels_n"]),
    ])
    pdf.savefig(figure)
    figure.savefig(os.path.join(PNGDIR, "%s.png" % model.replace("/", "_")), dpi=150)
    plt.close(figure)


def cross_page_plots(pdf, summaries, tables):
    """The eleven models' curves and the two correlations, on a page of their own."""
    models = sorted(summaries)
    figure = plt.figure(figsize=(TEXT_WIDTH_CM * CM, 17.5 * CM))
    figure.suptitle("The eleven models together", fontsize=FONT_PT + 2, y=0.975)
    grid = figure.add_gridspec(2, 2, hspace=1.05, wspace=0.42, top=0.925, bottom=0.175,
                               left=0.11, right=0.98)
    handles = []

    axis = figure.add_subplot(grid[0, 0])
    for index, model in enumerate(models):
        table = tables[model]
        levels = sorted(table)
        line, = axis.plot([table[l]["a_s"] for l in levels],
                          [table[l]["pass_pct"] for l in levels], "o-",
                          color=PALETTE[index % len(PALETTE)], markersize=2.4,
                          label=model.replace("-instruct", ""))
        handles.append(line)
    axis.axvline(1.3, color="0.45", lw=0.6, ls="--")
    axis.set_xlabel("A/S")
    axis.set_ylabel("pass rate (%)")
    axis.set_title("pass rate per aperture")
    axis.set_ylim(-6, 110)
    axis.invert_xaxis()

    axis = figure.add_subplot(grid[0, 1])
    for index, model in enumerate(models):
        table = tables[model]
        levels = sorted(table)
        axis.plot([table[l]["a_s"] for l in levels], [table[l]["turn_pct"] for l in levels],
                  "s-", color=PALETTE[index % len(PALETTE)], markersize=2.4)
    axis.set_xlabel("A/S")
    axis.set_ylabel("episodes with a turn (%)")
    axis.set_title("turn rate per aperture (flat = a fixed policy)")
    axis.set_ylim(-6, 110)
    axis.invert_xaxis()

    # Only the extreme models are labelled in the scatters: naming all eleven put the
    # labels on top of each other, which is what made the first version unreadable.
    for slot, (key, xlabel, title, colour) in enumerate([
        ("wide_turn_pct", "turn rate where no turn is needed (%)",
         "spurious rotation vs success", PALETTE[0]),
        ("look_pct", "episodes that inspected their own body (%)",
         "self-inspection vs success", PALETTE[2]),
    ]):
        axis = figure.add_subplot(grid[1, slot])
        xs = [summaries[m][key] for m in models]
        ys = [summaries[m]["pass_pct"] for m in models]
        axis.plot(xs, ys, "o", color=colour, markersize=3.5)
        interesting = {min(range(len(xs)), key=lambda i: xs[i]),
                       max(range(len(xs)), key=lambda i: xs[i]),
                       min(range(len(ys)), key=lambda i: ys[i]),
                       max(range(len(ys)), key=lambda i: ys[i])}
        for index in sorted(interesting):
            axis.annotate(models[index].replace("-instruct", "")[:16], (xs[index], ys[index]),
                          fontsize=FONT_PT - 3, xytext=(3, 3), textcoords="offset points")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("overall pass rate (%)")
        axis.set_title(title)
        axis.margins(0.16)

    figure.legend(handles, [h.get_label() for h in handles], loc="lower center",
                  ncol=4, fontsize=FONT_PT - 2, frameon=False)
    pdf.savefig(figure)
    figure.savefig(os.path.join(PNGDIR, "_cross_plots.png"), dpi=150)
    plt.close(figure)


def cross_page_table(pdf, summaries):
    """The table of every model and the synthesis, on a page of their own."""
    models = sorted(summaries, key=lambda m: -summaries[m]["pass_pct"])
    figure = plt.figure(figsize=(TEXT_WIDTH_CM * CM, 15.5 * CM))
    grid = figure.add_gridspec(2, 1, height_ratios=[7.4, 5.6], hspace=0.30,
                               top=0.905, bottom=0.03, left=0.015, right=0.985)

    axis = figure.add_subplot(grid[0])
    axis.axis("off")
    axis.set_title("Every model, sorted by overall pass rate\n"
                   "narrow = pass rate at the two apertures that need rotation (A/S 1.0, 0.9);"
                   "   wide turn% = turning where none is needed",
                   fontsize=FONT_PT, pad=6)
    header = ["model", "pass%", "narrow", "turn%", "wide\nturn%", "first\nturn x",
              "look%", "pass|\nlook", "pass|\nnolook"]
    body = [[m.replace("-instruct", "")[:16],
             "%.0f" % s["pass_pct"], s["narrow_pass"], "%.0f" % s["turned_pct"],
             "%.0f" % s["wide_turn_pct"], "%.2f" % s["first_turn_x"],
             "%.0f" % s["look_pct"],
             "–" if s["look_pct"] == 0 else "%.0f" % s["look_pass"],
             "%.0f" % s["no_look_pass"]]
            for m, s in ((m, summaries[m]) for m in models)]
    rendered = axis.table(cellText=body, colLabels=header, loc="center", cellLoc="center",
                          colLoc="center", bbox=[0.0, 0.0, 1.0, 1.0],
                          colWidths=[0.17, 0.08, 0.11, 0.09, 0.10, 0.11, 0.09, 0.12, 0.12])
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(FONT_PT - 2)

    axis = figure.add_subplot(grid[1])
    wide_turn = mean([summaries[m]["wide_turn_pct"] for m in models])
    matched = mean([summaries[m]["look_matched_diff"] for m in models])
    positive = sum(1 for m in models if summaries[m]["look_matched_diff"] > 0)
    turners = sorted(models, key=lambda m: -summaries[m]["wide_turn_pct"])[:2]
    abstainers = sorted(models, key=lambda m: summaries[m]["wide_turn_pct"])[:2]
    draw_text(axis, [
        "Pass rate spans %.0f-%.0f%%, and the two apertures that need rotation already "
        "separate the models (%s)." % (
            min(s["pass_pct"] for s in summaries.values()),
            max(s["pass_pct"] for s in summaries.values()),
            ", ".join("%s %s" % (m.replace("-instruct", "")[:14], summaries[m]["narrow_pass"])
                      for m in models[:3])),
        "Turning where no turn is needed averages %.0f%% and ranges from %.0f%% (%s) to %.0f%% "
        "(%s), so the models differ by more than an order of magnitude on a decision the "
        "aperture does not require at all." % (
            wide_turn, summaries[abstainers[0]]["wide_turn_pct"],
            abstainers[0].replace("-instruct", "")[:14],
            summaries[turners[0]]["wide_turn_pct"], turners[0].replace("-instruct", "")[:14]),
        "Inspecting its own body, compared inside each aperture rather than pooled (pooling "
        "would be confounded, because looking down clusters at the narrow Levels): %+.1f "
        "percentage points on average, positive in %d of 11 models, so inspection does not "
        "buy success by itself." % (matched, positive),
        "The reasoning names the opening, the turn and the model's own width; what fails is "
        "the amount of rotation and the alignment, not knowing that a turn is needed.",
    ])
    pdf.savefig(figure)
    figure.savefig(os.path.join(PNGDIR, "_cross_table.png"), dpi=150)
    plt.close(figure)


def write_markdown(episodes, summaries, tables):
    lines = [
        "# Stage 1 逐模型分析（自动生成，勿手改）",
        "",
        "数据：`lab_logs/bao_v7_all.tgz`（660 集 = 11 模型 × 12 档 A/S × 5 次）。",
        "复算：`python lab_logs/model_report.py`（本文件由它生成，所有数字现算）。",
        "",
        "指标说明：**turn%** 该档中至少转身一次的集占比；**side%** 该模型在该档的通过里"
        "以 45°–135° 侧身姿态完成的占比；**excess** = 实际步数 − 该宽度最优步数；"
        "**look%** 至少低头看过自己一次的集占比；**pass|look / pass|nolook** 低头过的集/"
        "没低头的集的通过率。",
        "",
        "---",
        "",
        "## 跨模型总表",
        "",
        "| 模型 | 通过率 | A/S 1.0+0.9 | 转身集 | 无需转身处仍转 | 首次转身x | 平均最大转角 | 平均步数 | excess | 低头率 | pass\\|look | pass\\|nolook |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, s in sorted(summaries.items(), key=lambda kv: -kv[1]["pass_pct"]):
        lines.append("| %s | %.0f%% | %s | %.0f%% | %.0f%% | %.2f m | %.0f° | %.1f | %.1f | %.0f%% | %.0f%% | %.0f%% |" % (
            model, s["pass_pct"], s["narrow_pass"], s["turned_pct"], s["wide_turn_pct"],
            s["first_turn_x"], s["mean_max_rot"], s["mean_steps"], s["mean_excess"],
            s["look_pct"], s["look_pass"], s["no_look_pass"]))
    lines += ["", "---", ""]

    for model in sorted(summaries, key=lambda m: -summaries[m]["pass_pct"]):
        s, table = summaries[model], tables[model]
        lines += [
            "## %s" % model,
            "",
            "**总体**：%d 集，通过率 **%.0f%%**；两个需要转身的档（A/S 1.0 与 0.9）%s。"
            "转身集 %.0f%%，首次转身平均在 x = %.2f m；**在不需要转身的宽度上仍转身的比例 %.0f%%**。"
            "平均最大转角 %.0f°，平均 %.1f 步（比最优多 %.1f 步）。"
            "低头看过自己 %.0f%%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）："
            "低头与不低头的通过率之差为 **%+.1f 个百分点**（%d/%d 个档位上低头更好）。" % (
                s["n"], s["pass_pct"], s["narrow_pass"], s["turned_pct"],
                s["first_turn_x"], s["wide_turn_pct"], s["mean_max_rot"],
                s["mean_steps"], s["mean_excess"], s["look_pct"],
                s["look_matched_diff"], s["look_levels_pos"], s["look_levels_n"]),
            "",
            "**策略标签分布**：" + "，".join(
                "%s %.0f%%" % (k, pct(v, s["n"])) for k, v in s["labels"].most_common()),
            "",
            "**动作构成**：" + "，".join("%s %.1f%%" % (k, v)
                                       for k, v in list(s["action_mix"].items())[:6]),
            "",
            "**推理提到的主题**（占该模型集数的比例）：" + "，".join(
                "%s %.0f%%" % (k, v) for k, v in sorted(s["topic_hits"].items(), key=lambda kv: -kv[1])),
            "",
            "| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\\|look | pass\\|nolook |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for level in sorted(table):
            row = table[level]
            look_cell = "–" if row["look_n"] == 0 else "%.0f" % row["pass_if_look"]
            nolook_cell = "–" if row["n_nolook"] == 0 else "%.0f" % row["pass_if_no_look"]
            lines.append("| %.1f | %.3f | %d/%d | %.0f | %.0f | %.1f | %.0f | %.1f | %.0f | %s | %s |" % (
                row["a_s"], row["width"], row["passed"], row["n"], row["turn_pct"],
                row["sideways_pass_pct"], row["mean_steps"], row["mean_max_rot"],
                row["mean_excess"], row["look_pct"], look_cell, nolook_cell))
        picked = quotes([e for e in episodes if e["model"] == model])
        if picked:
            lines += ["", "**它自己的原话**：", ""]
            for episode, snippet in picked:
                lines.append("- （A/S %.2f，%s，%d 步）「%s」" % (
                    episode["a_s"], "通过" if episode["passed"] else "失败",
                    episode["steps"], snippet))
        lines += ["", "---", ""]

    with open(MD_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_csv(tables):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "level", "a_s", "channel_width", "episodes", "passed",
                         "pass_pct", "min_turns", "turn_pct", "sideways_pass_pct",
                         "mean_steps", "mean_max_rotation_deg", "mean_excess",
                         "look_pct", "pass_if_look", "pass_if_no_look"])
        for model in sorted(tables):
            for level in sorted(tables[model]):
                row = tables[model][level]
                writer.writerow([model, level, row["a_s"], row["width"], row["n"],
                                 row["passed"], "%.1f" % row["pass_pct"], row["needs_turns"],
                                 "%.1f" % row["turn_pct"], "%.1f" % row["sideways_pass_pct"],
                                 "%.2f" % row["mean_steps"], "%.1f" % row["mean_max_rot"],
                                 "%.2f" % row["mean_excess"], "%.1f" % row["look_pct"],
                                 "%.1f" % row["pass_if_look"], "%.1f" % row["pass_if_no_look"]])


def main():
    os.makedirs(PNGDIR, exist_ok=True)
    episodes = load()
    models = sorted({e["model"] for e in episodes})
    tables = {m: per_level([e for e in episodes if e["model"] == m]) for m in models}
    summaries = {m: model_summary([e for e in episodes if e["model"] == m]) for m in models}
    with PdfPages(PDF_PATH) as pdf:
        cross_page_plots(pdf, summaries, tables)
        cross_page_table(pdf, summaries)
        for model in models:
            model_page(pdf, model, [e for e in episodes if e["model"] == model],
                       summaries[model], tables[model])
    write_markdown(episodes, summaries, tables)
    write_csv(tables)
    print("wrote %s (%d models), %s, %s, and per-model PNGs in %s" % (
        PDF_PATH, len(models), MD_PATH, CSV_PATH, PNGDIR))


if __name__ == "__main__":
    main()
