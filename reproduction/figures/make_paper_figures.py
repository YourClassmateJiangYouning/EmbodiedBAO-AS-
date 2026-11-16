"""Paper figures for EmbodiedBAO, drawn in the style of the LAM plotting tutorial.

Style learned from D:\\数据\\画图教程\\figure (1).ipynb (see analysis/plot_style.py):
Times New Roman, fixed six-colour palette, dashed grid, black-edged bars with
alpha 0.7, annotated single-hue heatmaps, no titles, 500-dpi transparent export.

Data are read from the tables already produced by the analysis scripts; nothing
is recomputed from raw episodes except the small failure-anatomy and first-turn
histograms.  Outputs go to analysis/figures_paper/.

Run:
    python analysis/make_paper_figures.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402

OUT = os.path.join(HERE, "v8")
FIG = os.path.join(HERE, "figures_paper")
ps.setup(9.0)

SHORT = {
    "Qwen-Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B",
    "Qwen-Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
    "Qwen-Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "claude-haiku-4-5-20251001": "Claude-Haiku",
    "deepseek-v4-flash-vision-exp": "DeepSeek-V4",
    "gemini-2.5-flash": "Gemini-2.5",
    "gemini-3-flash-preview": "Gemini-3",
    "glm-4.5v": "GLM-4.5V",
    "glm-4.6v": "GLM-4.6V",
    "gpt-4.1": "GPT-4.1",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5": "GPT-5",
    "kimi-kimi-k2.5": "Kimi-K2.5",
    "kimi-kimi-k2.6": "Kimi-K2.6",
    "mimo-v2.6-pro": "MiMo-v2.6",
}


def short(name: str) -> str:
    return SHORT.get(name, str(name).split("/")[-1][:18])


def wilson(k: int, n: int, z: float = 1.959963984540054):
    p = k / n if n else 0.0
    den = 1 + z * z / max(1, n)
    centre = (p + z * z / (2 * max(1, n))) / den
    half = z / den * math.sqrt(p * (1 - p) / max(1, n) + z * z / (4 * max(1, n) ** 2))
    return max(0.0, centre - half), min(1.0, centre + half)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load():
    d = {}
    d["ep"] = pd.read_csv(os.path.join(OUT, "episodes.csv.gz"))
    d["ep"]["a_s"] = d["ep"]["a_s"].round(1)
    with open(os.path.join(OUT, "final_findings.json"), encoding="utf-8") as fh:
        d["findings"] = json.load(fh)
    d["table"] = pd.read_csv(os.path.join(OUT, "final_model_table.csv"))
    d["pairs"] = pd.read_csv(os.path.join(OUT, "stage2_deep_pairs.csv"))
    d["s2m"] = pd.read_csv(os.path.join(OUT, "stage2_deep_model.csv"))
    d["scores"] = pd.read_csv(os.path.join(OUT, "scores_model.csv"))
    d["comps"] = pd.read_csv(os.path.join(OUT, "scores_components.csv"))
    d["ft"] = pd.read_csv(os.path.join(OUT, "first_turn.csv"))
    d["models"] = sorted(d["ep"]["model"].unique())
    return d


# ----------------------------------------------------------------------------
# Figure 1 - A/S response curves
# ----------------------------------------------------------------------------
def fig1(d):
    curve = pd.DataFrame(d["findings"]["curve"]).sort_values("a_s", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    for ax, col, ylabel, tag in (
        (axes[0], "pass_rate", "Pass rate", "(a)"),
        (axes[1], "rot30_rate", r"$P(\mathrm{rotation} \geq 30^\circ)$", "(b)"),
    ):
        key = "passed" if col == "pass_rate" else "rot30"
        for m in d["models"]:
            sub = d["ep"][d["ep"]["model"] == m]
            if key == "rot30":
                sub = sub.assign(rot30=(sub["max_rotation_deg"] >= 30 - 1e-9).astype(float))
            g = sub.groupby("a_s")[key].mean().sort_index(ascending=False)
            ax.plot(g.index, g.values, color=ps.LIGHT_GREY, lw=0.6, ls="--", zorder=1)
        # human band and references
        ax.axvspan(1.10, 1.30, color=ps.HUMAN_BAND, alpha=0.20, lw=0, zorder=0)
        for hv in (1.10, 1.25, 1.30):
            ax.axvline(hv, color="#b8860b", ls=":", lw=0.6, zorder=1)
        ax.plot(curve["a_s"], curve[col], color=ps.PALETTE[0], marker="o", ms=3.4,
                lw=1.8, zorder=3, label="15 models pooled (n = 6,375)")
        ax.axvline(1.0, color=ps.INK, ls="--", lw=0.7, zorder=1)
        ax.set_xlim(2.02, 0.36)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("A/S  (aperture width / shoulder width)")
        ax.set_ylabel(ylabel)
        ps.style_axes(ax)
        ps.panel_label(ax, tag)
    axes[0].text(1.205, 0.55, "human rotation onset\n1.10 - 1.30", color="#8a6d0b",
                 rotation=90, ha="center", va="center", fontsize=8)
    axes[1].text(1.24, 0.80, "no threshold\nbefore A/S = 1.0", color=ps.INK,
                 ha="center", va="center", fontsize=7.5)
    # reusable legend in a clean place
    handles = [Line2D([], [], color=ps.PALETTE[0], marker="o", ms=3.4, lw=1.8),
               Line2D([], [], color=ps.LIGHT_GREY, lw=0.6, ls="--"),
               Patch(facecolor=ps.HUMAN_BAND, alpha=0.20, label="human band")]
    axes[0].legend(handles, ["15 models pooled (n = 6,375)", "individual models",
                             "human rotation-onset band"],
                   loc="lower left", frameon=False, fontsize=7.2)
    fig.tight_layout(w_pad=1.4)
    ps.save(fig, "fig1_response_curves")


# ----------------------------------------------------------------------------
# Figure 2 - rotation amplitude vs A/S
# ----------------------------------------------------------------------------
def fig2(d):
    curve = pd.DataFrame(d["findings"]["curve"]).sort_values("a_s", ascending=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for m in d["models"]:
        sub = d["ep"][d["ep"]["model"] == m]
        g = sub.groupby("a_s")["max_rotation_deg"].mean().sort_index(ascending=False)
        ax.plot(g.index, g.values, color=ps.LIGHT_GREY, lw=0.6, ls="--", zorder=1)
    ax.plot(curve["a_s"], curve["mean_max_rot"], color=ps.PALETTE[0], marker="o",
            ms=3.4, lw=1.8, zorder=3, label="pooled mean rotation")
    # geometric requirement per level (A/S 2.0..1.0 -> 0 deg; 0.9 -> 60; 0.8/0.7 -> 75; 0.6/0.5/0.4 -> 90)
    req_by_as = {2.0: 0, 1.9: 0, 1.8: 0, 1.7: 0, 1.6: 0, 1.5: 0, 1.4: 0, 1.3: 0,
                 1.2: 0, 1.1: 0, 1.0: 0, 0.9: 60, 0.8: 75, 0.7: 75, 0.6: 90,
                 0.5: 90, 0.4: 90}
    lvl = curve["a_s"].to_numpy()
    reqv = [req_by_as.get(round(float(lv), 1), 0) for lv in lvl]
    ax.plot(lvl, reqv, color=ps.INK, ls=":", lw=1.0, marker="x", ms=4.0,
            label="geometric requirement")
    ax.axvspan(1.10, 1.30, color=ps.HUMAN_BAND, alpha=0.20, lw=0, zorder=0)
    ax.axvline(1.0, color=ps.INK, ls="--", lw=0.7)
    ax.set_xlim(2.02, 0.36)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("A/S  (aperture width / shoulder width)")
    ax.set_ylabel("max torso rotation (deg)")
    ps.style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=7.0)
    ax.text(1.2, 20, "humans turn\nalready here", color="#8a6d0b", fontsize=7,
            ha="center")
    fig.tight_layout()
    ps.save(fig, "fig2_rotation_amplitude")


# ----------------------------------------------------------------------------
# Figure 3 - per-model thresholds + pass-rate confound
# ----------------------------------------------------------------------------
def fig3(d):
    t = d["table"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    graded = t[t["group"] == "graded"].sort_values("pi_rot30")
    for i, (_, r) in enumerate(graded.iterrows()):
        ax.plot([r["pi_rot30"], 1.0], [i, i], color=ps.LIGHT_GREY, lw=0.7, zorder=1)
        ax.scatter(r["pi_rot30"], i, s=26, color=ps.PALETTE[0], edgecolor=ps.INK,
                   lw=0.4, zorder=3)
    ax.axvspan(1.10, 1.30, color=ps.HUMAN_BAND, alpha=0.22, lw=0, zorder=0)
    ax.axvline(1.0, color=ps.INK, ls=":", lw=0.7)
    pooled = float(d["findings"]["pi_rot30"])
    ax.axvline(pooled, color=ps.PALETTE[0], ls="--", lw=1.0)
    ax.set_yticks(range(len(graded)))
    ax.set_yticklabels([short(m) for m in graded["model"]], fontsize=7.2)
    ax.set_xlim(0.40, 1.42)
    ax.set_ylim(-0.7, len(graded) - 0.3)
    ax.set_xlabel(r"A/S at which $P(\mathrm{rotation} \geq 30^\circ) = 0.5$")
    ps.style_axes(ax)
    ps.panel_label(ax, "(a)")
    ax.text(1.20, len(graded) - 0.55, "humans\n1.10 - 1.30", color="#8a6d0b",
            fontsize=7, ha="center", va="top")
    ax.text(pooled - 0.015, -0.55, f"pooled {pooled:.2f}", color=ps.PALETTE[0],
            fontsize=7, ha="right")

    ax = axes[1]
    colors = {"graded": ps.PALETTE[0], "habitual": ps.PALETTE[1],
              "no_rotation": ps.LIGHT_GREY}
    for _, r in t.iterrows():
        ax.scatter(r["wide_turn_rate"], 100 * r["pass_narrow"],
                   s=30, color=colors.get(r["group"], ps.LIGHT_GREY),
                   edgecolor=ps.INK, lw=0.4, zorder=3)
    show = t[(t["pass_narrow"] > 0.05) | (t["wide_turn_rate"] > 0.20)]
    offsets = {"MiMo-v2.6": (4, 2), "DeepSeek-V4": (-8, 6), "Claude-Haiku": (4, 5),
               "Qwen3-VL-30B": (-38, -12), "Gemini-2.5": (6, -9),
               "GPT-4.1": (4, 3), "Gemini-3": (-52, -12)}
    for _, r in show.iterrows():
        nm = short(r["model"])
        dx, dy = offsets.get(nm, (3, 2))
        ax.annotate(nm, (r["wide_turn_rate"], 100 * r["pass_narrow"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6.2)
    ax.axvline(0.25, color=ps.GREY, ls=":", lw=0.7)
    ax.set_xlabel("turn rate at wide A/S $\\geq$ 1.5   (fixed policy)")
    ax.set_ylabel("pass rate at narrow A/S $\\leq$ 0.9 (%)")
    ps.style_axes(ax)
    ps.panel_label(ax, "(b)")
    ax.text(0.98, 0.05, r"Spearman $\rho$ = 0.54,  p = 0.04", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.2)
    handles = [Line2D([], [], marker="o", ls="", color=colors[k], markeredgecolor=ps.INK,
                      label=k.replace("_", " ")) for k in ("graded", "habitual", "no_rotation")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.6)
    ps.save(fig, "fig3_thresholds_confound")


# ----------------------------------------------------------------------------
# Figure 4 - failure anatomy at A/S = 0.9
# ----------------------------------------------------------------------------
def fig4(d):
    g = d["ep"][d["ep"]["level"] == 11]
    cats = {"never\nturned": 0, "turned\n<45°": 0, "turned\n45-60°": 0,
            "reached\n60° (required)": 0, "passed": 0}
    for _, r in g.iterrows():
        if r["passed"]:
            cats["passed"] += 1
        elif r["max_rotation_deg"] >= 60 - 1e-9:
            cats["reached\n60° (required)"] += 1
        elif r["max_rotation_deg"] >= 45 - 1e-9:
            cats["turned\n45-60°"] += 1
        elif r["max_rotation_deg"] > 0:
            cats["turned\n<45°"] += 1
        else:
            cats["never\nturned"] += 1
    labels = list(cats.keys())
    vals = list(cats.values())
    colors = [ps.LIGHT_GREY, ps.PALETTE[2], ps.PALETTE[4], ps.PALETTE[1], ps.PALETTE[0]]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", lw=0.5,
                  alpha=0.75, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center",
                fontsize=7.0)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_xlabel("A/S = 0.9, all 15 models (375 episodes)")
    ax.set_ylabel("episodes")
    ps.style_axes(ax, grid_axis="y")
    ps.panel_label(ax, "")
    ax.set_title("41 / 273 failures reached the required 60°", fontsize=8.0)
    fig.tight_layout()
    ps.save(fig, "fig4_failure_anatomy")


# ----------------------------------------------------------------------------
# Figure 5 - Stage 2 memory transfer (bar with Wilson CI)
# ----------------------------------------------------------------------------
def fig5(d):
    m = d["s2m"].copy()
    m["short"] = m["model"].map(short)
    m = m.sort_values("delta_turn")
    y = np.arange(len(m))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    ax = axes[0]
    base = 100 * m["model_lvl0_turn"].to_numpy()
    mem = 100 * m["turn_rate"].to_numpy()
    lo, hi = [], []
    for rate, n in zip(m["turn_rate"], m["n"]):
        a, b = wilson(int(round(rate * n)), int(n))
        lo.append(100 * (rate - a))
        hi.append(100 * (b - rate))
    ax.barh(y - 0.19, base, height=0.36, color=ps.LIGHT_GREY, edgecolor="black",
            lw=0.5, alpha=0.75, label="Stage 1, no memory")
    ax.barh(y + 0.19, mem, height=0.36, color=ps.PALETTE[0], edgecolor="black",
            lw=0.5, alpha=0.75, xerr=[lo, hi], capsize=2.5,
            error_kw=dict(lw=0.7, capthick=0.7), label="after one narrow memory")
    ax.set_yticks(y)
    ax.set_yticklabels(m["short"], fontsize=7.2)
    ax.set_xlabel("P(turn) at A/S = 2.0 (%)")
    ax.set_xlim(0, 108)
    ps.style_axes(ax, grid_axis="x")
    ps.panel_label(ax, "(a)")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax = axes[1]
    ax.barh(y - 0.19, m["model_lvl0_rot"], height=0.36, color=ps.LIGHT_GREY,
            edgecolor="black", lw=0.5, alpha=0.75)
    ax.barh(y + 0.19, m["mean_max_rot"], height=0.36, color=ps.PALETTE[1],
            edgecolor="black", lw=0.5, alpha=0.75)
    ax.set_xlabel("mean max torso rotation (deg) at A/S = 2.0")
    ax.set_xlim(0, 95)
    ps.style_axes(ax, grid_axis="x")
    ps.panel_label(ax, "(b)")
    fig.tight_layout(w_pad=1.5)
    ps.save(fig, "fig5_stage2_transfer")


# ----------------------------------------------------------------------------
# Figure 6 - where Stage 1 turns happen
# ----------------------------------------------------------------------------
def fig6(d):
    ft = d["ft"]
    narrow = ft[ft["a_s"] < 1.0]
    fig, ax = plt.subplots(figsize=(3.7, 2.6))
    bins = np.arange(0.75, 8.5, 0.5)
    counts, edges, bars = ax.hist(narrow["first_turn_x"].dropna(), bins=bins,
                                  color=ps.PALETTE[0], edgecolor="black", lw=0.4,
                                  alpha=0.75)
    for b, c in zip(bars, counts):
        if c > 100:
            ax.text(b.get_x() + b.get_width() / 2, c + 18, f"{int(c)}", ha="center",
                    fontsize=6.6)
    ax.axvline(8.0, color=ps.PALETTE[1], ls="--", lw=1.0)
    ax.text(7.9, 0.90 * ax.get_ylim()[1], "wall  x = 8.0", color=ps.PALETTE[1],
            rotation=90, ha="right", va="top", fontsize=7)
    ax.set_xlabel("robot x at the first successful turn (m), narrow A/S $\\leq$ 0.9")
    ax.set_ylabel("episodes")
    ax.set_xlim(0.75, 8.4)
    ps.style_axes(ax, grid_axis="y")
    med = float(narrow["first_turn_x"].median())
    ax.annotate(f"median = {med:.2f} m\n(0.75 m before the wall)",
                xy=(med, 0.55 * counts.max()), xytext=(3.6, 0.62 * counts.max()),
                fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.6, color=ps.INK))
    fig.tight_layout()
    ps.save(fig, "fig6_first_turn_position")


# ----------------------------------------------------------------------------
# Figure 7 - Stage 2 deep analysis (2x2)
# ----------------------------------------------------------------------------
def fig7(d):
    p = d["pairs"]
    with open(os.path.join(OUT, "stage2_deep_findings.json"), encoding="utf-8") as fh:
        deep = json.load(fh)
    imit = deep.get("imitators", ["mimo-v2.6-pro", "deepseek-v4-flash-vision-exp",
                                  "gemini-3-flash-preview"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    rng = np.random.default_rng(3)
    # (a) source vs test turn count
    ax = axes[0, 0]
    is_im = p["model"].isin(imit)
    ax.scatter(p.loc[is_im, "src_n_turns"] + rng.normal(0, 0.07, is_im.sum()),
               p.loc[is_im, "turns"] + rng.normal(0, 0.08, is_im.sum()), s=15,
               color=ps.PALETTE[0], edgecolor="black", lw=0.25, alpha=0.85,
               label="imitators", zorder=3)
    ax.scatter(p.loc[~is_im, "src_n_turns"] + rng.normal(0, 0.07, (~is_im).sum()),
               p.loc[~is_im, "turns"] + rng.normal(0, 0.08, (~is_im).sum()), s=12,
               color=ps.LIGHT_GREY, edgecolor="black", lw=0.2, alpha=0.8,
               label="others", zorder=2)
    lim = [3.4, 8.4]
    ax.plot(lim, lim, color=ps.INK, ls=":", lw=0.8)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(-0.5, 8.5)
    ax.set_xlabel("source memory: turn count")
    ax.set_ylabel("test at A/S = 2.0: turn count")
    ps.style_axes(ax)
    ps.panel_label(ax, "(a)")
    ax.text(0.98, 0.05, "91.1% exact turn-count match", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8)
    ax.legend(loc="lower right", frameon=False, fontsize=6.6)
    # (b) the written plan vs the body
    ax = axes[0, 1]
    straight = p["plan_class"] == "straight"
    turned = p["turned"] == 1
    cats = ["plan says\nstraight", "other plan"]
    turned_v = [int((straight & turned).sum()), int(((~straight) & turned).sum())]
    still_v = [int((straight & ~turned).sum()), int(((~straight) & ~turned).sum())]
    x = np.arange(2)
    ax.bar(x - 0.17, turned_v, 0.34, color=ps.PALETTE[1], edgecolor="black", lw=0.5,
           alpha=0.75, label="turned in test")
    ax.bar(x + 0.17, still_v, 0.34, color=ps.LIGHT_GREY, edgecolor="black", lw=0.5,
           alpha=0.75, label="did not turn")
    for xi, (a_, b_) in enumerate(zip(turned_v, still_v)):
        ax.text(xi - 0.17, a_ + 1.2, str(a_), ha="center", fontsize=6.8)
        ax.text(xi + 0.17, b_ + 1.2, str(b_), ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=7)
    ax.set_ylabel("pairs")
    ax.set_ylim(0, 66)
    ps.style_axes(ax, grid_axis="y")
    ps.panel_label(ax, "(b)")
    ax.text(0.03, 0.95, "digest mentions turning in 0/105 tests", transform=ax.transAxes,
            va="top", fontsize=6.6)
    ax.legend(loc="upper right", frameon=False, fontsize=6.6)
    # (c) first-turn position: test vs source (imitators)
    ax = axes[1, 0]
    bins = np.arange(0.75, 8.5, 0.5)
    ax.hist(p.loc[is_im, "first_turn_x"].dropna(), bins=bins, color=ps.PALETTE[1],
            edgecolor="black", lw=0.4, alpha=0.65, label="test (A/S = 2.0)")
    ax.hist(p.loc[is_im, "src_first_turn_x"].dropna(), bins=bins, color=ps.PALETTE[0],
            edgecolor="black", lw=0.4, alpha=0.45, label="source (narrow)")
    ax.axvline(8.0, color=ps.INK, ls="--", lw=0.8)
    ax.set_xlabel("robot x at the first turn (m)")
    ax.set_ylabel("pairs")
    ps.style_axes(ax, grid_axis="y")
    ps.panel_label(ax, "(c)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.6)
    # (d) cost of the unnecessary turn
    ax = axes[1, 1]
    t_steps = float(p.loc[turned, "total_steps"].mean())
    s_steps = float(p.loc[~turned, "total_steps"].mean())
    bars = ax.bar([0, 1], [t_steps, s_steps], width=0.5, color=[ps.PALETTE[1], ps.LIGHT_GREY],
                  edgecolor="black", lw=0.5, alpha=0.75)
    ax.axhline(11, color=ps.INK, ls=":", lw=0.8)
    ax.text(0.5, 12.4, "optimal = 11 steps", fontsize=6.6, ha="center", va="bottom")
    for xi, (v, lab) in enumerate(((t_steps, "pass 98.2%"), (s_steps, "pass 100%"))):
        ax.text(xi, v + 0.5, f"{v:.1f} steps\n{lab}", ha="center", fontsize=6.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["turned", "did not turn"], fontsize=7)
    ax.set_ylabel("mean steps at A/S = 2.0")
    ax.set_ylim(0, 21.5)
    ps.style_axes(ax, grid_axis="y")
    ps.panel_label(ax, "(d)")
    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    ps.save(fig, "fig7_stage2_deep")


# ----------------------------------------------------------------------------
# Figure 8 - scoring models
# ----------------------------------------------------------------------------
def fig8(d):
    s = d["scores"].copy()
    s["short"] = s["model"].map(short)
    s = s.sort_values("BAO_Q")
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    ax.barh(y - 0.19, s["Sfinal_doc"], height=0.36, color=ps.LIGHT_GREY,
            edgecolor="black", lw=0.5, alpha=0.8, label="documented $S_{final}$ (gate at 20)")
    ax.barh(y + 0.19, s["BAO_Q"], height=0.36, color=ps.PALETTE[0],
            edgecolor="black", lw=0.5, alpha=0.8, label="BAO-Q (this work)")
    ax.axvline(20, color=ps.PALETTE[1], ls="--", lw=1.0)
    ax.text(19.3, len(s) - 0.2, "ceiling = 20", color=ps.PALETTE[1], rotation=90,
            ha="right", va="top", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(s["short"], fontsize=7.0)
    ax.set_xlabel("score (0 - 20, human = 20)")
    ax.set_xlim(0, 24)
    ps.style_axes(ax, grid_axis="x")
    ps.panel_label(ax, "(a)")
    ax.legend(loc="lower right", frameon=False, fontsize=6.8)
    # component heatmap
    ax = axes[1]
    comps = d["comps"].set_index("model").loc[s["model"]]
    fields = ["r_margin", "r_graded", "r_anticip", "r_condpass", "r_fixed",
              "reason_specificity", "reason_ratio", "reason_consistency",
              "timely_self_check", "rotation_sufficiency", "collision_factor",
              "effort_factor", "recovery"]
    names = ["margin", "graded", "anticip.", "cond. pass", "not fixed", "spec.",
             "ratio", "know-act", "self-check", "rot. suff.", "collision",
             "effort", "recovery"]
    mat = comps[fields].astype(float).to_numpy()
    ps.heatmap(ax, mat, [short(m) for m in comps.index], names, cmap="Oranges",
               vmin=0.0, vmax=1.0, fmt="{:.2f}", annot_size=4.3, rotate=35)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.tick_params(axis="x", labelsize=6.4)
    ps.panel_label(ax, "(b)")
    fig.tight_layout(w_pad=1.4)
    ps.save(fig, "fig8_scores")


# ----------------------------------------------------------------------------
# Figure 9 - radar of the 5 Stage-2 models
# ----------------------------------------------------------------------------
def fig9(d):
    comps = d["comps"].set_index("model")
    deep_models = ["mimo-v2.6-pro", "deepseek-v4-flash-vision-exp",
                   "claude-haiku-4-5-20251001", "gpt-5"]
    axes_names = [r"HBAS/20", r"RBA/100", r"CLC/100", r"1$-$MSS/100", r"not fixed"]
    vals = []
    for m in deep_models:
        r = comps.loc[m]
        vals.append([r["HBAS"] / 20, r["RBA"] / 100, r["CLC"] / 100,
                     1 - r["MSS"] / 100, r["r_fixed"]])
    angles = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(3.9, 3.4), subplot_kw=dict(polar=True))
    for i, (m, v) in enumerate(zip(deep_models, vals)):
        vv = v + v[:1]
        ax.plot(angles, vv, color=ps.PALETTE[i], marker="o", ms=3.2, lw=1.5,
                label=short(m))
        ax.fill(angles, vv, color=ps.PALETTE[i], alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_names, fontsize=7.6)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=5.6, color=ps.GREY)
    ax.set_ylim(0, 1.05)
    ax.grid(color=ps.GRID["color"], ls="--", lw=0.6)
    ax.spines["polar"].set_color(ps.LIGHT_GREY)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), frameon=False, fontsize=7)
    fig.tight_layout()
    ps.save(fig, "fig9_radar_profiles")


def main() -> int:
    d = load()
    fig1(d)
    fig2(d)
    fig3(d)
    fig4(d)
    fig5(d)
    fig6(d)
    fig7(d)
    fig8(d)
    fig9(d)
    print("done -> analysis/figures_paper/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
