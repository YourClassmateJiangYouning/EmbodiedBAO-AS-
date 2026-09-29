"""Figures for the EmbodiedBAO Stage-2 paper, sized for Springer LNCS/CCIS.

JOURNAL CONFIGURATION -- the same block as the Stage-1 figure script, so the two
papers' figures are drawn to one specification.  Where each value comes from is stated
in that script and repeated here for the same reason: it can be corrected in one place.

  TEXT_WIDTH_CM 12.2   LNCS single-column text area (LNCS is single column, body 10 pt)
  FONT_PT 8            stays legible at final size and never below ~6 pt
  RASTER_DPI 600       Springer wants >=600 dpi for combination art; the PDF is vector
  FORMATS              PDF (submit) + PNG (preview)
  FONT_SANS            Arial/Helvetica, DejaVu Sans as the portable fallback
  PALETTE              Okabe-Ito, colour-blind safe; series also differ in line style

Every figure is drawn from the two CSVs the exporter writes, not from the records, so
each plotted number can be checked by hand in the table.

    python lab_logs/export_memory_table.py
    python lab_logs/make_memory_figures.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH_CM = 12.2
FONT_PT = 8
RASTER_DPI = 600
FORMATS = ("pdf", "png")
FONT_SANS = ["Arial", "Helvetica", "DejaVu Sans"]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000"]
OUTDIR = "lab_logs/figures"
CM = 1.0 / 2.54
LEARNING_ROUNDS = 12
PROBE_ROUNDS = 5
SIDEWAYS_MIN_DEG = 45.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_SANS,
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "axes.titlesize": FONT_PT,
    "xtick.labelsize": FONT_PT - 1,
    "ytick.labelsize": FONT_PT - 1,
    "legend.fontsize": FONT_PT - 1,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODE_STYLE = {"cumulative": "-", "rolling": "--"}
MODE_ORDER = ("cumulative", "rolling")
CURVE_ORDER = ("insight", "gradual", "oscillating", "perseveration")
CURVE_TITLE = {
    "insight": "sudden (insight)",
    "gradual": "gradual",
    "oscillating": "oscillating",
    "perseveration": "perseveration",
}


def read_csv(path: str) -> list:
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing; run lab_logs/export_memory_table.py first"
        )
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save(figure, name: str) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    for fmt in FORMATS:
        path = os.path.join(OUTDIR, f"{name}.{fmt}")
        figure.savefig(path, format=fmt, dpi=RASTER_DPI)
    plt.close(figure)
    print(f"wrote {OUTDIR}/{name}.{{{','.join(FORMATS)}}}")


def models_of(rounds: list) -> list:
    return sorted({row["model"] for row in rounds})


def colour_for(models: list, model: str) -> str:
    return PALETTE[models.index(model) % len(PALETTE)]


def figure_learning_curves(rounds: list, models: list) -> None:
    """Running pass rate within a run, averaged over runs, per memory mode.

    The running rate rather than the raw outcome because a single attempt says little
    and 17 points per run would hide the shape; averaging the running rate over the
    runs of a model keeps every run's contribution equal.
    """
    figure, axes = plt.subplots(
        1, len(MODE_ORDER), figsize=(TEXT_WIDTH_CM * CM, 2.5), sharey=True
    )
    for axis, mode in zip(axes, MODE_ORDER):
        by_model: dict = collections.defaultdict(list)
        for row in rounds:
            if row["memory_mode"] != mode:
                continue
            by_model[row["model"]].append(row)
        for model in models:
            rows = by_model.get(model)
            if not rows:
                continue
            runs: dict = collections.defaultdict(list)
            for row in rows:
                runs[row["run"]].append(row)
            curves = []
            for run in runs:
                ordered = sorted(runs[run], key=lambda r: int(r["round"]))
                hits = 0
                running = []
                for index, row in enumerate(ordered[:LEARNING_ROUNDS], start=1):
                    hits += 1 if row["passed"] == "True" else 0
                    running.append(hits / index)
                curves.append(running)
            if not curves:
                continue
            mean = [
                sum(curve[i] for curve in curves if i < len(curve))
                / max(sum(1 for curve in curves if i < len(curve)), 1)
                for i in range(LEARNING_ROUNDS)
            ]
            axis.plot(
                range(1, len(mean) + 1), mean,
                MODE_STYLE[mode], color=colour_for(models, model), label=model,
            )
        axis.set_title(f"{mode} memory")
        axis.set_xlabel("attempt within the run")
        axis.set_xlim(1, LEARNING_ROUNDS)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks([1, 4, 8, 12])
    axes[0].set_ylabel("running pass rate")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(figure, "stage2_learning_curves")


def figure_gap_by_class(runs: list, rounds: list) -> None:
    """The gap trajectory of every run, grouped by the verdict it received."""
    by_run: dict = collections.defaultdict(dict)
    for row in rounds:
        by_run[(row["tag"], row["run"])][int(row["round"])] = number(row["gap"])
    figure, axes = plt.subplots(
        1, len(CURVE_ORDER), figsize=(TEXT_WIDTH_CM * CM, 2.4), sharey=True
    )
    for axis, label in zip(axes, CURVE_ORDER):
        grouped = [row for row in runs if row["curve_label"] == label]
        for row in grouped:
            series = by_run.get((row["tag"], row["run"]), {})
            xs = [r for r in range(1, LEARNING_ROUNDS + 1) if series.get(r) is not None]
            ys = [series[r] for r in xs]
            if xs:
                axis.plot(xs, ys, "-", color="#999999", linewidth=0.5, alpha=0.8)
        valued = [
            [by_run.get((row["tag"], row["run"]), {}).get(r) for r in range(1, LEARNING_ROUNDS + 1)]
            for row in grouped
        ]
        mean = []
        for index in range(LEARNING_ROUNDS):
            column = [series[index] for series in valued if series[index] is not None]
            mean.append(sum(column) / len(column) if column else None)
        xs = [i + 1 for i, value in enumerate(mean) if value is not None]
        ys = [value for value in mean if value is not None]
        if xs:
            axis.plot(xs, ys, "-o", color=PALETTE[1], linewidth=1.4)
        axis.axhline(0.0, color="#000000", linewidth=0.6, linestyle=":")
        axis.set_title(f"{CURVE_TITLE[label]} (n={len(grouped)})")
        axis.set_xlabel("attempt within the run")
        axis.set_xlim(1, LEARNING_ROUNDS)
        axis.set_xticks([1, 4, 8, 12])
    axes[0].set_ylabel("gap to fitting (m)")
    axes[0].text(
        0.04, 0.06, "0 = exactly fits", transform=axes[0].transAxes, fontsize=FONT_PT - 1
    )
    save(figure, "stage2_gap_by_class")


def figure_probe_decay(runs: list, models: list) -> None:
    """Torso rotation in the wide passage, attempt by attempt.

    This is the habit measurement: a model that learned to turn in the narrow passage
    keeps turning where it does not need to, and the rotation it spends is the trace.
    """
    figure, axis = plt.subplots(figsize=(TEXT_WIDTH_CM * CM * 0.62, 2.6))
    for mode in MODE_ORDER:
        for model in models:
            rows = [r for r in runs if r["model"] == model and r["memory_mode"] == mode]
            if not rows:
                continue
            series = []
            for index in range(1, PROBE_ROUNDS + 1):
                values = [
                    number(r.get(f"probe_rotation_{index}")) for r in rows
                ]
                values = [v for v in values if v is not None]
                series.append(sum(values) / len(values) if values else None)
            xs = [i + 1 for i, value in enumerate(series) if value is not None]
            ys = [value for value in series if value is not None]
            if xs:
                axis.plot(
                    [LEARNING_ROUNDS + x for x in xs], ys, MODE_STYLE[mode] + "o",
                    color=colour_for(models, model), markersize=3.0,
                    label=f"{model} ({mode[:4]})",
                )
    axis.axhline(SIDEWAYS_MIN_DEG, color="#000000", linewidth=0.6, linestyle=":")
    axis.text(
        LEARNING_ROUNDS + 0.15, SIDEWAYS_MIN_DEG + 3.0, "sideways band starts (45°)",
        fontsize=FONT_PT - 1,
    )
    axis.set_xlabel("attempt in the wide passage (A/S 1.10)")
    axis.set_ylabel("largest torso rotation (deg)")
    axis.set_xticks([LEARNING_ROUNDS + i for i in range(1, PROBE_ROUNDS + 1)])
    axis.set_xlim(LEARNING_ROUNDS + 0.7, LEARNING_ROUNDS + PROBE_ROUNDS + 0.3)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
    save(figure, "stage2_probe_decay")


def figure_d_and_turns(runs: list, rounds: list, models: list) -> None:
    """How many of the 12 narrow attempts succeeded, beside how often it turned."""
    figure, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_CM * CM, 2.9), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    left, right = axes
    width = 0.36
    for offset, mode in zip((-width / 2, width / 2), MODE_ORDER):
        means, lows, highs = [], [], []
        for model in models:
            values = [
                number(r["d"], 0.0)
                for r in runs
                if r["model"] == model and r["memory_mode"] == mode
            ]
            if values:
                means.append(sum(values) / len(values))
                lows.append(min(values))
                highs.append(max(values))
            else:
                means.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
        xs = [i + offset for i in range(len(models))]
        left.bar(
            xs, means, width=width, color=[colour_for(models, m) for m in models],
            edgecolor="black", linewidth=0.4,
            label=mode, hatch="" if mode == "cumulative" else "///",
        )
        left.errorbar(
            xs, means,
            yerr=[[m - low for m, low in zip(means, lows)],
                  [high - m for m, high in zip(means, highs)]],
            fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5,
        )
    learning = [row for row in rounds if int(row["round"]) <= LEARNING_ROUNDS]
    probe = [row for row in rounds if int(row["round"]) > LEARNING_ROUNDS]

    def rates(rows, model):
        mine = [r for r in rows if r["model"] == model]
        if not mine:
            return None, None
        turned = sum(1 for r in mine if number(r.get("n_turn"), 0.0) >= 1) / len(mine)
        enough = sum(
            1
            for r in mine
            if number(r.get("n_turn"), 0.0)
            >= (5 if int(r["round"]) <= LEARNING_ROUNDS else 4)
        ) / len(mine)
        return turned, enough

    xs = range(len(models))
    turned_learning = [rates(learning, m)[0] or 0.0 for m in models]
    turned_probe = [rates(probe, m)[0] or 0.0 for m in models]
    right.plot(list(xs), turned_learning, "o-", color=PALETTE[0], label="turned, narrow")
    right.plot(list(xs), turned_probe, "s--", color=PALETTE[2], label="turned, wide")
    right.set_ylim(-0.03, 1.03)
    right.set_ylabel("share of attempts with a turn")
    right.legend(loc="lower left")

    for axis in axes:
        axis.set_xticks(list(xs))
        axis.set_xticklabels(
            [m.replace("-instruct", "") for m in models], rotation=90, fontsize=FONT_PT - 2
        )
    left.set_ylabel("successful narrow attempts (d, of 12)")
    left.set_ylim(0, LEARNING_ROUNDS + 0.5)
    left.legend(loc="upper left", ncol=2)
    save(figure, "stage2_d_and_turns")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rounds", default=os.path.join(
        "lab_logs", "embodiedbao_stage2_rounds.csv"))
    parser.add_argument("--runs", default=os.path.join(
        "lab_logs", "embodiedbao_stage2_runs.csv"))
    args = parser.parse_args()
    rounds = read_csv(args.rounds)
    runs = read_csv(args.runs)
    if not rounds or not runs:
        print("the tables are empty; nothing to plot")
        return 1
    models = models_of(rounds)
    figure_learning_curves(rounds, models)
    figure_gap_by_class(runs, rounds)
    figure_probe_decay(runs, models)
    figure_d_and_turns(runs, rounds, models)
    print(f"figures from {len(rounds)} rounds and {len(runs)} runs, {len(models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
