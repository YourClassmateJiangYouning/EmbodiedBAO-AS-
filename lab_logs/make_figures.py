"""Figures for the EmbodiedBAO Stage-1 paper, sized for Springer LNCS/CCIS.

JOURNAL CONFIGURATION -- every number a template can dictate lives here.
Where each value comes from is stated so it can be corrected in one place:

  TEXT_WIDTH_CM 12.2   LNCS single-column text area (LNCS is single column, body
                       10 pt).  Every figure is drawn at this width so it can be
                       scaled to 100% in the template.
  FONT_PT 8            Springer asks for figure lettering that stays legible at
                       final size and never smaller than ~6 pt; 8 pt at 100%
                       scale leaves margin.
  RASTER_DPI 600       Springer line art wants >=1200 dpi as raster, halftone
                       >=300 dpi, combinations >=600 dpi.  These are pure vector
                       line drawings, so the PDF is the submission artefact and
                       600 dpi PNGs are provided only for quick viewing.
  FORMATS              PDF (vector, the one to submit) + PNG (preview).
  FONT_SANS            Arial/Helvetica per Springer's sans-serif requirement,
                       with DejaVu Sans as the portable fallback.
  PALETTE              Okabe-Ito, colour-blind safe; every series also differs in
                       line style or marker so the figures survive greyscale print.

If the CCPR author kit states different numbers, change only this block.
"""

import collections
import json
import math
import os
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# JOURNAL CONFIGURATION
# --------------------------------------------------------------------------
TEXT_WIDTH_CM = 12.2
FONT_PT = 8
RASTER_DPI = 600
FORMATS = ("pdf", "png")
FONT_SANS = ["Arial", "Helvetica", "DejaVu Sans"]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000"]
OUTDIR = "lab_logs/figures"
HUMAN_LO, HUMAN_HI = 1.10, 1.30   # human rotation onset across studies: Franchak et al.
                                  # 2012 (0.5 cm apparatus) report 1.10, Keizer et al. 2013
                                  # healthy controls 1.25, Warren & Whang 1987 / Higuchi
                                  # et al. 2006 report 1.2-1.3.  Shading the whole range
                                  # rather than one number is the honest reading of the
                                  # literature; a single 1.30 is the coarse-step estimate.
MODEL_KNEE = 1.00
WALL_X, LAST_FREE_X = 8.0, 7.25
CM = 1.0 / 2.54

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

FAMILY = {
    "qwen3-vl-235b-a22b-instruct": "Qwen", "qwen3-vl-32b-instruct": "Qwen",
    "qwen-vl-max": "Qwen", "gemini-2.5-pro": "Gemini", "gemini-2.5-flash": "Gemini",
    "gpt-4.1": "OpenAI", "gpt-4o": "OpenAI", "gpt-4o-mini": "OpenAI",
    "claude-sonnet-4-6": "Anthropic", "deepseek-v4.1-flash": "DeepSeek",
    "glm-4.6v": "Zhipu",
}
FAMILY_COLOR = {"Qwen": PALETTE[0], "Gemini": PALETTE[1], "OpenAI": PALETTE[2],
                "Anthropic": PALETTE[3], "DeepSeek": PALETTE[4], "Zhipu": PALETTE[5]}


def load():
    tf = tarfile.open("lab_logs/bao_v7_all.tgz")
    records, steps = [], {}
    for member in tf.getmembers():
        name = member.name
        if name.endswith(".json") and "/episode_" in name and "_steps" not in name:
            records.append(json.load(tf.extractfile(member)))
        elif name.endswith("_steps.json") and "/episode_" in name:
            parts = name.split("/")
            steps[(parts[2], int(parts[1].replace("level", "")),
                   int(parts[4].split("_")[1]))] = json.load(tf.extractfile(member))
    return records, steps


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)) *
                    sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den else 0.0


def save(fig, stem):
    os.makedirs(OUTDIR, exist_ok=True)
    for fmt in FORMATS:
        path = os.path.join(OUTDIR, "%s.%s" % (stem, fmt))
        # A PDF records the wall-clock time it was written in its /CreationDate,
        # so two renderings of identical data never produce identical bytes and
        # every re-run shows up as a modification in `git status` even though
        # nothing about the figure changed.  Committing the PDFs is only useful if
        # they are reproducible, so that one field is dropped.  The PNGs are
        # already byte-stable and are left alone.
        extra = {} if fmt == "png" else {"metadata": {"CreationDate": None}}
        fig.savefig(path, dpi=RASTER_DPI if fmt == "png" else None, format=fmt, **extra)
        print("wrote", path)
    plt.close(fig)


def fig1(records, seqs):
    """Behaviour against aperture: pass rate, then the three action families.

    The three action series share one axis because they share one unit (actions
    per episode); an earlier version put lateral movement and a percentage on the
    same axis, which is not a comparison a reader can make.
    """
    levels = sorted(set(r["level"] for r in records))
    as_ratio = [next(r["a_s_ratio"] for r in records if r["level"] == lv) for lv in levels]
    counts = [sum(1 for r in records if r["level"] == lv) for lv in levels]
    passed = [100.0 * sum(r["passed"] for r in records if r["level"] == lv) / k
              for lv, k in zip(levels, counts)]
    turns, looks, laterals, look_eps = [], [], [], []
    for lv, k in zip(levels, counts):
        eps = [r for r in records if r["level"] == lv]
        c = collections.Counter()
        users = 0
        for r in eps:
            actions = seqs[(r["model_name"], lv, r["episode_id"])]
            c["turn"] += sum(1 for a in actions if a.startswith("turn"))
            c["look"] += actions.count("look_down")
            c["lat"] += sum(1 for a in actions if a in ("left", "right"))
            users += 1 if "look_down" in actions else 0
        turns.append(c["turn"] / k)
        looks.append(c["look"] / k)
        laterals.append(c["lat"] / k)
        look_eps.append(100.0 * users / k)

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_CM * CM * 0.86, 4.3),
                             sharex=True, gridspec_kw={"hspace": 0.14})
    for ax in axes:
        ax.axvspan(0.85, MODEL_KNEE, color="0.93", zorder=0)
        ax.axvspan(HUMAN_LO, HUMAN_HI, color="#009E73", alpha=0.22, zorder=0)
        ax.axvline(MODEL_KNEE, color="0.35", lw=0.6, ls="--", zorder=1)
        ax.set_xlim(2.05, 0.85)
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].plot(as_ratio, passed, "o-", color=PALETTE[0])
    axes[0].set_ylabel("pass rate (%)")
    axes[0].set_ylim(-5, 108)
    axes[0].text(HUMAN_HI + 0.01, 30, "human rotation onset\n1.1 - 1.3\n(studies differ)",
                 fontsize=FONT_PT - 2, ha="right", va="center", color="#00674F")
    axes[0].text(MODEL_KNEE - 0.02, 8, "model knee\nA/S = 1.0", fontsize=FONT_PT - 2,
                 ha="right", va="center", color="0.3")

    axes[1].plot(as_ratio, turns, "s-", color=PALETTE[1], label="torso rotation")
    axes[1].plot(as_ratio, looks, "^-", color=PALETTE[2], label="self-inspection")
    axes[1].plot(as_ratio, laterals, "d-", color=PALETTE[3], label="lateral movement")
    axes[1].set_ylabel("actions per episode")
    axes[1].set_xlabel("aperture ratio A/S  (opening / shoulder width)")
    axes[1].set_ylim(-0.25, 5.4)
    axes[1].legend(loc="upper left", ncol=3, handlelength=1.3, columnspacing=0.9)

    fig.align_ylabels(axes)
    save(fig, "fig1_behaviour_vs_aperture")


def fig2(records, steps):
    """Where the first torso rotation happens: the missing mid-approach band."""
    bins = collections.Counter()
    for r in records:
        sidecar = steps.get((r["model_name"], r["level"], r["episode_id"]))
        turn = r.get("first_turn_step")
        if not sidecar or turn is None or turn >= len(sidecar):
            bins["never"] += 1
        else:
            x = sidecar[turn]["position_x"]
            if x < 4.0:
                bins["early\nx < 4.0"] += 1
            elif x < 7.0:
                bins["mid\n4.0-7.0"] += 1
            else:
                bins["at wall\nx $\\geq$ 7.0"] += 1
    order = ["never", "early\nx < 4.0", "mid\n4.0-7.0", "at wall\nx $\\geq$ 7.0"]
    values = [bins[k] for k in order]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_CM * CM * 0.56, 2.6))
    colors = [PALETTE[7], PALETTE[1], PALETTE[2], PALETTE[3]]
    bars = ax.bar(range(len(order)), values, color=colors, width=0.6)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 8,
                "%d\n(%.1f%%)" % (v, 100.0 * v / total),
                ha="center", va="bottom", fontsize=FONT_PT - 1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("episodes (of %d)" % total)
    ax.set_ylim(0, max(values) * 1.26)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.annotate("where a body-scaled affordance\naccount puts the decision",
                xy=(2, 12), xytext=(1.45, 150), fontsize=FONT_PT - 2, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.35"))
    save(fig, "fig2_first_rotation_position")


def fig3(records):
    """Model archetypes: rotation at wide apertures versus rotation scaling."""
    stats = {}
    for model in sorted(set(r["model_name"] for r in records)):
        eps = [r for r in records if r["model_name"] == model]
        wide = [r for r in eps if r["level"] <= 5]
        narrow = [r for r in eps if r["level"] >= 9]
        mt = lambda group: sum(sum(1 for a in r["action_sequence"].split(",")  # noqa: E731
                                   if a.startswith("turn")) for r in group) / len(group)
        stats[model] = {
            "wide": mt(wide), "narrow": mt(narrow),
            "rho": spearman([r["a_s_ratio"] for r in eps],
                            [r["max_rotation_deg"] for r in eps]),
            "pass": 100.0 * sum(r["passed"] for r in eps) / len(eps),
        }

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_CM * CM * 0.66, 3.1))
    ax.axhspan(-0.254, 0.254, color="0.93", zorder=0)
    ax.axhline(0, color="0.6", lw=0.5, zorder=1)
    ax.axvline(2.0, color="0.6", lw=0.5, ls=":", zorder=1)
    for model, s in stats.items():
        ax.scatter(s["wide"], s["rho"], s=10 + 0.26 * s["pass"],
                   color=FAMILY_COLOR[FAMILY[model]], zorder=3,
                   edgecolor="white", linewidth=0.4)
    offsets = {
        "claude-sonnet-4-6": (0.14, 0.00), "gpt-4o": (0.14, -0.01),
        "gemini-2.5-flash": (0.14, -0.04), "gpt-4o-mini": (0.14, 0.03),
        "qwen3-vl-235b-a22b-instruct": (0.14, 0.02),
        "qwen3-vl-32b-instruct": (0.14, 0.045), "glm-4.6v": (0.14, -0.085),
        "deepseek-v4.1-flash": (0.14, 0.02), "gemini-2.5-pro": (-0.12, -0.075),
        "gpt-4.1": (0.14, -0.055), "qwen-vl-max": (-0.12, -0.06),
    }
    for model, s in stats.items():
        short = (model.replace("-instruct", "")
                 .replace("qwen3-vl-235b-a22b", "qwen3-vl-235B"))
        dx, dy = offsets.get(model, (0.14, 0.02))
        ax.annotate(short, (s["wide"], s["rho"]),
                    xytext=(s["wide"] + dx, s["rho"] + dy),
                    fontsize=FONT_PT - 2,
                    ha="left" if dx > 0 else "right")
    ax.set_xlim(-0.15, 4.35)
    ax.set_ylim(-0.83, 0.36)
    ax.set_xlabel("torso rotation at wide apertures\n(actions per episode, A/S 2.0-1.5)")
    ax.set_ylabel(r"Spearman $\rho$ (A/S, max rotation)")
    ax.text(0.02, 0.965, r"grey band: $|\rho| < 0.254$ ($p > 0.05$, $n = 60$)",
            transform=ax.transAxes, fontsize=FONT_PT - 2, va="top", color="0.25")
    ax.grid(color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=FAMILY_COLOR[f], label=f) for f in FAMILY_COLOR]
    ax.legend(handles=handles, loc="lower right", ncol=2, handlelength=0.8,
              columnspacing=0.7, labelspacing=0.25)
    save(fig, "fig3_model_archetypes")


def fig4(records):
    """The two necessary conditions at the narrowest aperture, A/S = 0.9."""
    l11 = [r for r in records if r["level"] == 11]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_CM * CM, 2.7),
                             gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.33})

    ax = axes[0]
    ax.axvspan(54, 195, color="0.93", zorder=0)
    ax.axhspan(-0.12, 0.12, color="0.93", zorder=0)
    for r in l11:
        ax.scatter(abs(r["max_rotation_deg"]), abs(r["final_position_z"]),
                   marker="o" if r["passed"] else "x",
                   color=PALETTE[2] if r["passed"] else PALETTE[1],
                   s=24 if r["passed"] else 15, zorder=3, linewidth=1.0)
    ax.axvline(54, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel("max torso rotation reached (deg)")
    ax.set_ylabel("lateral offset at the end (m)")
    ax.set_xlim(-8, 200)
    ax.set_ylim(-0.22, 1.95)
    ax.text(60, 1.80, r"needs $\geq 54^\circ$", fontsize=FONT_PT - 2, color="0.25")
    ax.text(2, 0.20, "needs to end on the centreline", fontsize=FONT_PT - 2, color="0.25")
    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[2], label="passed"),
               plt.Line2D([], [], marker="x", ls="", color=PALETTE[1], label="failed")]
    ax.legend(handles=handles, loc="lower right", handlelength=0.8, labelspacing=0.25)
    ax.grid(color="0.92", lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    grid = collections.Counter()
    for r in l11:
        key = (abs(r["max_rotation_deg"]) >= 54, abs(r["final_position_z"]) < 0.1,
               r["passed"])
        grid[key] += 1
    labels = ["no rot\n+ off", "no rot\ncentred", "rotated\n+ off", "rotated\ncentred"]
    passed = [grid[(a, b, True)] for a in (False, True) for b in (False, True)]
    failed = [grid[(a, b, False)] for a in (False, True) for b in (False, True)]
    x = list(range(4))
    ax.bar(x, failed, color=PALETTE[1], width=0.6, label="failed")
    ax.bar(x, passed, bottom=failed, color=PALETTE[2], width=0.6, label="passed")
    for i, (f, p) in enumerate(zip(failed, passed)):
        ax.text(i, f + p + 0.5, "%d/%d" % (p, f + p), ha="center", fontsize=FONT_PT - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("episodes (of %d)" % len(l11))
    ax.set_ylim(0, 27)
    ax.legend(loc="upper left", handlelength=0.9, labelspacing=0.25)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig4_narrowest_aperture_factors")


def main() -> None:
    records, steps = load()
    seqs = {k: [s["action"] for s in steps[k]] for k in steps}
    print("episodes:", len(records))
    fig1(records, seqs)
    fig2(records, steps)
    fig3(records)
    fig4(records)


if __name__ == "__main__":
    main()
