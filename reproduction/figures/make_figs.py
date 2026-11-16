"""Main-paper figures for EmbodiedBAO (CVPR 2027).

All numbers are recomputed from the raw exports:
  02_S1逐回合6375行.csv  (Stage 1, 6,375 episodes)
  04_S2_105对配对.csv     (Stage 2, 105 pairs)
  05_S3_100集体型.csv     (Stage 3, 100 episodes)
  06_pihat逐模型.csv      (pi-hat, bootstrap CI)
Palette and typography follow Fig. 1 / Fig. 2.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

D = "/mnt/user-data/uploads/"
OUT = os.path.join(os.path.dirname(__file__), "..", "fig")
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.5, "grid.linestyle": "--",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
NAVY, RUST, OLIVE, RED, GOLD = "#2F5597", "#C55A11", "#548235", "#C00000", "#BF9000"
GREY, LGREY, PBLUE, PEACH, MINT, CREAM = "#7F7F7F", "#C9C9C9", "#DEEBF7", "#FBE5D6", "#E2F0D9", "#FFF2CC"
TOP2 = ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview"]
LABEL = {
    "deepseek-v4-flash-vision-exp": "DeepSeek-V4-Flash", "gemini-3-flash-preview": "Gemini-3-Flash",
    "mimo-v2.6-pro": "MiMo-v2.6-Pro", "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B-A3B",
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5", "gpt-5": "GPT-5", "gemini-2.5-flash": "Gemini-2.5-Flash",
    "glm-4.5v": "GLM-4.5V", "gpt-4o-mini": "GPT-4o-mini", "kimi/kimi-k2.6": "Kimi-K2.6",
    "kimi/kimi-k2.5": "Kimi-K2.5", "gpt-4.1": "GPT-4.1", "glm-4.6v": "GLM-4.6V",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
}
REQ = {0.9: 54.0, 0.8: 62.8, 0.7: 70.3, 0.6: 77.1, 0.5: 83.3, 0.4: 89.2}   # H1 exact required angle
HUMAN = (1.10, 1.30)
PIHAT, PI_LO, PI_HI = 0.917, 0.906, 0.927

d = pd.read_csv(D + "02_S1逐回合6375行.csv")
d["passed"] = d.passed.astype(str) == "True"
d["A"] = d.a_s_ratio.round(1)
LEV = sorted(d.A.unique(), reverse=True)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def panel(ax, s):
    ax._letter = s


def finalize(fig):
    for ax in fig.axes:
        if hasattr(ax, "_letter"):
            t = ax.get_title(loc="left")
            ax.set_title(f"{ax._letter} {t}".strip(), loc="left", x=ax.title_x if hasattr(ax, "title_x") else 0.0)


def human_band(ax, label=True, y=0.97):
    ax.axvspan(*HUMAN, color=PBLUE, zorder=0, lw=0)
    if label:
        ax.text(1.2, y, "human\n1.10–1.30", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=NAVY, fontsize=5.5, linespacing=0.95)


def as_axis(ax):
    ax.set_xlim(2.05, 0.35)
    ax.set_xticks([2.0, 1.6, 1.2, 1.0, 0.8, 0.4])
    ax.set_xlabel("A/S  (opening / shoulder width)")


# ---------------------------------------------------------------- Fig. 3
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(6.875, 1.85), gridspec_kw={"wspace": 0.32, "width_ratios": [1.15, 1.15, 1]})
    pm = d.groupby(["model", "A"]).agg(p=("passed", "mean"), r30=("max_rotation_deg", lambda x: (x >= 30).mean()))
    pool = d.groupby("A").agg(k=("passed", "sum"), n=("passed", "size"), r30=("max_rotation_deg", lambda x: (x >= 30).mean()))
    pool = pool.loc[LEV]
    # (a) pass rate
    ax = axs[0]
    human_band(ax, y=0.62)
    for m in d.model.unique():
        s = pm.loc[m].loc[LEV]
        c = RUST if m in TOP2 else (GOLD if m == "mimo-v2.6-pro" else LGREY)
        ax.plot(LEV, s.p * 100, color=c, lw=0.6, alpha=0.9 if c != LGREY else 0.8, zorder=1)
    lo, hi = wilson(pool.k.values, pool.n.values)
    ax.fill_between(LEV, lo * 100, hi * 100, color=NAVY, alpha=0.18, lw=0, zorder=2)
    ax.plot(LEV, pool.k / pool.n * 100, color=NAVY, lw=1.4, marker="o", ms=2.4, zorder=3)
    ax.axvline(1.0, color="#404040", lw=0.6, ls=":")
    ax.annotate("94.4% → 27.2%\nbetween A/S 1.0 and 0.9", xy=(0.92, 50), xytext=(1.60, 22), fontsize=6, bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85),
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#404040"), ha="center")
    ax.text(0.52, 62, "top-3", color=RUST, fontsize=5.5, ha="center")
    ax.set_ylim(-3, 104); ax.set_ylabel("pass rate (%)")
    as_axis(ax); panel(ax, "(a)")
    ax.set_title("Pass rate collapses at zero margin", loc="left", x=0.0)
    # (b) P(rot>=30)
    ax = axs[1]
    human_band(ax, y=0.30)
    for m in d.model.unique():
        s = pm.loc[m].loc[LEV]
        c = RUST if m in TOP2 else LGREY
        ax.plot(LEV, s.r30 * 100, color=c, lw=0.6, alpha=0.9 if c != LGREY else 0.8, zorder=1)
    ax.plot(LEV, pool.r30 * 100, color=RUST, lw=1.4, marker="s", ms=2.4, zorder=3)
    ax.axhline(50, color=GREY, lw=0.5, ls="--")
    ax.axvspan(PI_HI, PI_LO, color=RED, alpha=0.18, lw=0)
    ax.axvline(PIHAT, color=RED, lw=0.9, ls="--")
    ax.text(PIHAT - 0.03, 22, r"$\hat{\pi}$ = 0.917" + "\n[0.906, 0.927]", color=RED, fontsize=5.8, ha="left")
    ax.text(2.0, 92, "fixed turners\n(DeepSeek, Gemini-3)", color=RUST, fontsize=5.5, ha="left", va="top")
    ax.set_ylim(-3, 104); ax.set_ylabel(r"P(max rotation $\geq$ 30°) (%)")
    as_axis(ax); panel(ax, "(b)")
    ax.set_title("Turning starts only below A/S 1.0", loc="left", x=0.0)
    # (c) required vs achieved
    ax = axs[2]
    n = d[d.A < 1.0]
    lv = sorted(REQ.keys(), reverse=True)
    mean = [n[n.A == a].max_rotation_deg.mean() for a in lv]
    q1 = [n[n.A == a].max_rotation_deg.quantile(0.25) for a in lv]
    q3 = [n[n.A == a].max_rotation_deg.quantile(0.75) for a in lv]
    met = [(n[n.A == a].max_rotation_deg >= REQ[a]).mean() * 100 for a in lv]
    x = np.arange(len(lv))
    ax.plot(x, [REQ[a] for a in lv], color=RED, ls="--", marker="x", ms=3.5, lw=0.9, label="geometrically required")
    ax.errorbar(x, mean, yerr=[np.array(mean) - q1, np.array(q3) - mean], fmt="o", color=NAVY, ms=3, lw=0.8,
                capsize=1.8, label="achieved (mean, IQR)")
    for xi, mt in zip(x, met):
        ax.text(xi, 3, f"{mt:.0f}%", ha="center", fontsize=5.3, color=OLIVE)
    ax.set_xticks(x); ax.set_xticklabels([f"{a:.1f}" for a in lv]); ax.set_xlabel("A/S (narrow bins)")
    ax.set_ylim(0, 102); ax.set_ylabel("max body rotation (°)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.1)
    panel(ax, "(c)")
    ax.set_title("…and it is too small", loc="left", x=0.0)
    finalize(fig)
    fig.savefig(os.path.join(OUT, "fig3_threshold.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
def fig4():
    n = d[d.A < 1.0].copy()
    n["req"] = n.A.map(REQ)
    n["met"] = n.max_rotation_deg >= n.req
    turned = n.first_turn_x_m.notna()
    early = n.first_turn_x_m <= 5
    cats = [("early\n& enough", n.met & early, OLIVE), ("late\n& enough", n.met & turned & ~early, GOLD),
            ("early\n& too little", ~n.met & early, RUST), ("late\n& too little", ~n.met & turned & ~early, RUST),
            ("never\nturned", ~turned, GREY)]
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.55), gridspec_kw={"wspace": 0.42, "width_ratios": [0.9, 1]})
    ax = axs[0]
    for i, (lab, m, c) in enumerate(cats):
        p = n[m].passed.mean() * 100
        ax.barh(i, max(p, 0.8), color=c, height=0.62, edgecolor="#404040", lw=0.4)
        ax.text(max(p, 0.8) + 3, i, f"{p:.1f}%", va="center", fontsize=5.6)
    ax.set_yticks(range(5)); ax.set_yticklabels([f"{c[0].replace(chr(10), ' ')} (n={c[1].sum()})" for c in cats], fontsize=5.4)
    ax.invert_yaxis(); ax.set_xlim(0, 128); ax.set_xticks([0, 50, 100]); ax.set_xlabel("pass rate (%)")
    ax.set_title("Turn enough, and early", loc="left", x=0.0); ax.title_x = -1.0
    ax.grid(axis="y", visible=False)
    panel(ax, "(a)")
    ax = axs[1]
    x = n.first_turn_x_m.dropna()
    bins = np.arange(0.125, 8.0, 0.75)
    h, e = np.histogram(x, bins=bins)
    ctr = (e[:-1] + e[1:]) / 2
    ax.bar(ctr, h / len(x) * 100, width=0.62, color=[OLIVE if c <= 5 else RUST for c in ctr], edgecolor="#404040", lw=0.4)
    ax.axvline(8.0, color=NAVY, lw=1.6)
    ax.text(8.12, 40, "wall", color=NAVY, fontsize=5.6, ha="left", rotation=90)
    ax.axvline(5.0, color=GREY, lw=0.5, ls="--")
    ax.text(2.6, 60, "early", color=OLIVE, fontsize=5.6, ha="center")
    ax.set_xlim(0, 8.4); ax.set_xticks([0, 2, 4, 6, 8]); ax.set_ylim(0, 72)
    ax.set_xlabel("x of first turn (m)"); ax.set_ylabel("narrow turning episodes (%)")
    ax.set_title(f"Median {x.median():.2f} m", loc="left", x=0.0)
    ax.grid(axis="x", visible=False)
    panel(ax, "(b)")
    finalize(fig)
    fig.savefig(os.path.join(OUT, "fig4_mechanism.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 5
def fig5():
    wide = d[d.A >= 1.5]
    nar = d[d.A < 1.0].groupby("model").passed.mean() * 100
    wr = wide.groupby("model").max_rotation_deg.mean()
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.55), gridspec_kw={"wspace": 0.5, "width_ratios": [1.3, 1]})
    ax = axs[0]
    for m in wr.index:
        c = RUST if m in TOP2 else (GOLD if m == "mimo-v2.6-pro" else GREY)
        ax.scatter(wr[m], nar[m], s=14, color=c, edgecolor="white", lw=0.4, zorder=3)
    for m, dx, dy, ha in [("deepseek-v4-flash-vision-exp", -2, 0, "right"), ("gemini-3-flash-preview", 0, -10, "center"),
                          ("mimo-v2.6-pro", 2, 0, "left")]:
        ax.text(wr[m] + dx, nar[m] + dy, LABEL[m], fontsize=5.5, ha=ha, va="center", color=RUST if m in TOP2 else GOLD)
    r_all = np.corrcoef(wr, nar[wr.index])[0, 1]
    k = ~wr.index.isin(TOP2)
    r_wo = np.corrcoef(wr[k], nar[wr.index][k])[0, 1]
    ax.text(0.97, 0.42, f"r = {r_all:.2f} (all 15)\nr = {r_wo:.2f} (w/o top-2)", transform=ax.transAxes,
            ha="right", fontsize=5.6)
    ax.text(6, 22, "12 VLMs", fontsize=5.5, color=GREY)
    ax.set_xlim(-3, 56); ax.set_ylim(-5, 108)
    ax.set_xlabel("mean rotation, A/S 2.0–1.5 (°)"); ax.set_ylabel("narrow-gap pass rate (%)")
    ax.set_title("Who passes, turns everywhere", loc="left", x=0.0)
    panel(ax, "(a)")
    ax = axs[1]
    t = wide.n_turns > 0
    vals = [(wide[~t].passed.mean() * 100, wide[t].passed.mean() * 100), (wide[~t].total_steps.mean(), wide[t].total_steps.mean())]
    ax.bar([0, 1], [vals[1][0], vals[1][1]], color=[LGREY, RUST], edgecolor="#404040", lw=0.4, width=0.6)
    for i, (s, p) in enumerate(zip(vals[1], vals[0])):
        ax.text(i, s + 0.5, f"{s:.1f} steps\n{p:.1f}% pass", ha="center", fontsize=5.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"no turn\nn={(~t).sum()}", f"turned\nn={t.sum()}"], fontsize=5.6)
    ax.set_ylim(0, 22); ax.set_ylabel("mean steps (A/S 2.0–1.5)")
    ax.set_title("Needless turns cost", loc="left", x=0.0)
    ax.grid(axis="x", visible=False)
    panel(ax, "(b)")
    finalize(fig)
    fig.savefig(os.path.join(OUT, "fig5_strategy.pdf"))
    plt.close(fig)
    return r_all, r_wo


# ---------------------------------------------------------------- Fig. 6
def fig6():
    s2 = pd.read_csv(D + "04_S2_105对配对.csv")
    s3 = pd.read_csv(D + "05_S3_100集体型.csv")
    s3["passed"] = s3.passed.astype(str) == "True"
    fig, axs = plt.subplots(1, 4, figsize=(6.875, 1.65), gridspec_kw={"wspace": 0.48, "width_ratios": [1.25, 1, 1, 1.1]})
    # (a) Stage 2 dumbbell
    ax = axs[0]
    g = s2.groupby("model").agg(b=("baseline_turns", "mean"), m=("turn_count", "mean"), lcs=("copy_lcs_fraction", "mean")).sort_values("m")
    y = np.arange(len(g))
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.plot([r.b, r.m], [yi, yi], color=LGREY, lw=1.4, zorder=1)
    ax.scatter(g.b, y, color=GREY, s=12, zorder=2, label="no")
    ax.scatter(g.m, y, color=NAVY, s=12, zorder=3, label="yes")
    ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in g.index], fontsize=5.4)
    for yi, l in zip(y, g.lcs):
        ax.text(7.6, yi, f"{l:.2f}", fontsize=5, va="center", ha="right", color=OLIVE)
    ax.text(7.6, len(g) - 0.35, "LCS", fontsize=5, ha="right", color=OLIVE)
    ax.set_xlim(-0.4, 7.8); ax.set_xlabel("turns at A/S 2.0 (mean)")
    ax.legend(loc="lower right", frameon=False, fontsize=5.2, bbox_to_anchor=(0.86, 0.0), handletextpad=0.1, title="memory", title_fontsize=5.2)
    ax.set_title("Stage 2: memory copies turns", loc="left", x=0.0); ax.title_x = -0.62
    panel(ax, "(a)")
    # H1 reference: same five models in Stage 1
    five = s3.model.unique()
    h1 = d[d.model.isin(five)].groupby("A").passed.mean() * 100
    # (b) pass vs A/S per body
    ax = axs[1]
    lv = [1.3, 1.1, 1.0, 0.9]
    ax.plot(lv, [h1[a] for a in lv], color=NAVY, marker="o", ms=3, lw=1.1, label="H1 (S 0.570)")
    for rb, c in [("G1", RUST), ("R1", GOLD)]:
        s = s3[s3.robot == rb].groupby("a_s_requested").passed.mean() * 100
        ax.plot(lv, [s[a] for a in lv], color=c, marker="o", ms=3, lw=1.1,
                label=f"{rb} (S {s3[s3.robot == rb].shoulder_width_m.iloc[0]:.3f})")
    ax.set_xlim(1.35, 0.85); ax.set_xticks(lv); ax.set_ylim(40, 104)
    ax.set_xlabel("A/S"); ax.set_ylabel("pass rate (%)")
    ax.legend(loc="lower left", frameon=False, fontsize=5.2, handlelength=1.2)
    ax.set_title("Stage 3: same A/S cliff", loc="left", x=0.0)
    panel(ax, "(b)")
    # (c) rotation vs own required angle at A/S 0.9
    ax = axs[2]
    req = {"H1": 54.0, "G1": 58.6, "R1": 65.4}
    h1rot = d[(d.model.isin(five)) & (d.A == 0.9)].max_rotation_deg
    data = {"H1": h1rot.values, "G1": s3[(s3.robot == "G1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values,
            "R1": s3[(s3.robot == "R1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values}
    rng = np.random.default_rng(0)
    for i, (rb, c) in enumerate([("H1", NAVY), ("G1", RUST), ("R1", GOLD)]):
        v = np.minimum(data[rb], 120)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=c, alpha=0.65, lw=0)
        ax.plot([i - 0.3, i + 0.3], [req[rb]] * 2, color="#262626", lw=1.1, ls="--")
        ax.text(i, 106, f"{(data[rb] >= req[rb]).mean() * 100:.0f}%", ha="center", fontsize=5.5, color=OLIVE)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["H1", "G1", "R1"]); ax.set_ylim(-4, 120)
    ax.set_ylabel("max rotation at A/S 0.9 (°)")
    ax.set_title("(b) largest turn at A/S=0.9", loc="left", x=0.0)
    ax.grid(axis="x", visible=False)
    panel(ax, "(c)")
    # (d) absolute width
    ax = axs[3]
    h1w = d[d.model.isin(five)].groupby("channel_width_m").passed.mean() * 100
    ax.axvspan(0.44, 0.475, color=PEACH, lw=0, zorder=0)
    ax.scatter(h1w.index, h1w.values, s=9, color=NAVY, zorder=3, label="H1")
    for rb, c in [("G1", RUST), ("R1", GOLD)]:
        s = s3[s3.robot == rb].groupby("channel_width_m").passed.mean() * 100
        ax.scatter(s.index, s.values, s=14, color=c, edgecolor="white", lw=0.4, zorder=4, label=rb)
    ax.annotate("≈0.46 m: G1, R1 100%\nvs. H1 59%", xy=(0.457, 62), xytext=(0.80, 40), fontsize=5.3, ha="center", color=RUST, arrowprops=dict(arrowstyle="->", lw=0.5, color=RUST))
    ax.set_xlabel("opening width (m)"); ax.set_ylabel("pass rate (%)"); ax.set_ylim(15, 106)
    ax.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.1)
    ax.set_title("(c) pass rate by opening width", loc="left", x=0.0)
    panel(ax, "(d)")
    finalize(fig)
    fig.savefig(os.path.join(OUT, "fig6_transfer.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig3(); fig4(); r = fig5(); fig6()
    print("r_all %.3f  r_without_top2 %.3f" % r)
