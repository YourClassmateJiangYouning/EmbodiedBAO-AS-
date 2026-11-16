"""Redrawn main-text figures (v2). All numbers computed from the raw exports."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
U = "/mnt/user-data/uploads/"
MAIN = "../fig/main/"; SUPP = "../fig/supp/"
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
    "grid.linestyle": "--", "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
NAVY, RUST, GOLD, GREY, LG, PEACH = "#2F5597", "#C55A11", "#BF9000", "#8C8C8C", "#C9C9C9", "#FBE5D6"
GREEN, RED, PURPLE = "#548235", "#C00000", "#8E6BB0"
TOP2 = ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview"]
LAB = {"deepseek-v4-flash-vision-exp": "DeepSeek-V4", "gemini-3-flash-preview": "Gemini-3", "mimo-v2.6-pro": "MiMo-v2.6",
       "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B", "claude-haiku-4-5-20251001": "Claude-Haiku", "gpt-5": "GPT-5",
       "gemini-2.5-flash": "Gemini-2.5", "glm-4.5v": "GLM-4.5V", "gpt-4o-mini": "GPT-4o-mini", "kimi/kimi-k2.6": "Kimi-K2.6",
       "kimi/kimi-k2.5": "Kimi-K2.5", "gpt-4.1": "GPT-4.1", "glm-4.6v": "GLM-4.6V", "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
       "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B"}
S, D = 0.570, 0.220
REQ = {0.9: 54.0, 0.8: 62.8, 0.7: 70.3, 0.6: 77.1, 0.5: 83.3, 0.4: 89.2}

d = pd.read_csv(U + "02_S1逐回合6375行.csv"); d["passed"] = d.passed.astype(str) == "True"; d["A"] = d.a_s_ratio.round(1)
LEV = sorted(d.A.unique(), reverse=True)
MODELS = list(d.model.unique())

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def save(fig, name):
    fig.savefig(name); plt.close(fig)

# ---------------- C01 threshold cliff (figure*, 6.9 in)
def c01():
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45), gridspec_kw={"wspace": 0.28})
    pm = d.groupby(["model", "A"]).passed.mean().unstack()
    pool = d.groupby("A").agg(k=("passed", "sum"), n=("passed", "size"), r30=("max_rotation_deg", lambda x: (x >= 30).mean()))
    pool = pool.loc[LEV]
    ax = axs[0]
    ax.axvspan(1.10, 1.30, color="#FFF2CC", lw=0, zorder=0)
    ax.text(1.20, 101, "human onset\n1.10–1.30", ha="center", va="bottom", fontsize=6.3, color=GOLD, bbox=dict(fc="white", ec="none", pad=0.5))
    for m in MODELS:
        if m in TOP2 or m == "mimo-v2.6-pro": continue
        ax.plot(LEV, pm.loc[m].reindex(LEV) * 100, color=LG, lw=0.5, zorder=1)
    for m, c in [("mimo-v2.6-pro", GOLD)] + [(m, RUST) for m in TOP2]:
        ax.plot(LEV, pm.loc[m].reindex(LEV) * 100, color=c, lw=0.7, alpha=0.9, zorder=2)
    lo, hi = wilson(pool.k.values, pool.n.values)
    ax.fill_between(LEV, lo * 100, hi * 100, color=NAVY, alpha=0.2, lw=0, zorder=2)
    ax.plot(LEV, pool.k / pool.n * 100, "-o", color=NAVY, lw=1.6, ms=3, zorder=3)
    ax.axvline(1.0, color="#262626", lw=0.6, ls=":")
    ax.annotate("94.4% at A/S 1.0\n27.2% at A/S 0.9", xy=(0.93, 28), xytext=(0.66, 44), fontsize=6.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#404040"))
    ax.set_xlim(2.05, 0.35); ax.set_ylim(-3, 112); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks([2.0, 1.6, 1.2, 1.0, 0.8, 0.4]); ax.set_xlabel("A/S (opening / shoulder width)"); ax.set_ylabel("pass rate (%)")
    ax.set_title("(a) Pass rate: flat, then a cliff at zero margin", loc="left", x=0)
    ax.legend(handles=[Line2D([], [], color=NAVY, marker="o", ms=3, lw=1.5, label="pooled, 95% CI"),
                       Line2D([], [], color=LG, lw=0.6, label="each model"),
                       Line2D([], [], color=RUST, lw=0.7, label="DeepSeek / Gemini-3")],
              frameon=False, loc="lower left", fontsize=6)
    ax = axs[1]
    ax.axvspan(1.10, 1.30, color="#FFF2CC", lw=0, zorder=0)
    p30m = d.groupby(["model", "A"]).max_rotation_deg.apply(lambda x: (x >= 30).mean()).unstack()
    for m in MODELS:
        if m in TOP2: continue
        ax.plot(LEV, p30m.loc[m].reindex(LEV) * 100, color=LG, lw=0.5, zorder=1)
    ax.plot(LEV, pool.r30 * 100, "-s", color=RUST, lw=1.6, ms=3, zorder=3)
    ax.axvline(1.0, color="#262626", lw=0.6, ls=":")
    ax.axhline(50, color=GREY, lw=0.5, ls="--")
    ax.axvline(0.917, color=RED, lw=0.9, ls="--")
    ax.text(1.45, 84, "π̂ = 0.917 [0.906, 0.927]", color=RED, fontsize=6.5, ha="left")
    ax.set_xlim(2.05, 0.35); ax.set_ylim(-3, 105); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks([2.0, 1.6, 1.2, 1.0, 0.8, 0.4]); ax.set_xlabel("A/S (opening / shoulder width)")
    ax.set_ylabel(r"episodes with max rotation $\geq$30° (%)")
    ax.set_title("(b) Turning starts only below zero margin", loc="left", x=0)
    ax.legend(handles=[Line2D([], [], color=RUST, marker="s", ms=3, lw=1.6, label="pooled"),
                       Line2D([], [], color=LG, lw=0.6, label="each model")], frameon=False, loc="upper left", fontsize=6)
    save(fig, MAIN + "c01_threshold.pdf")

# ---------------- C05 necessity and timing (figure, column 3.3 in)
def c05():
    n = d[d.A < 1].copy(); n["req"] = n.A.map(REQ); n["met"] = n.max_rotation_deg >= n.req
    early = n.first_turn_x_m <= 5; turned = n.first_turn_x_m.notna()
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"wspace": 0.25, "width_ratios": [0.8, 1.6]})
    ax = axs[0]
    cats = [("inadequate", ~n.met), ("adequate", n.met)]
    for i, (lab, m) in enumerate(cats):
        p = n[m].passed.mean() * 100; c = GREY if lab == "inadequate" else NAVY
        ax.bar(i, max(p, 0.8), color=c, width=0.6, edgecolor="#404040", lw=0.4)
        ax.text(i, max(p, 0.8) + 3, f"{p:.1f}%\nn={m.sum():,}", ha="center", fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["rotation <\nrequired", "rotation ≥\nrequired"], fontsize=6.3)
    ax.set_ylim(0, 115); ax.set_ylabel("pass rate (%)"); ax.grid(axis="x", visible=False)
    ax.set_title("(a) Necessary condition", loc="left", x=-0.3)
    ax = axs[1]
    cats = [("early\nOK", n.met & early, NAVY), ("late\nOK", n.met & turned & ~early, GOLD),
            ("early\nfail", ~n.met & early, GREY), ("late\nfail", ~n.met & turned & ~early, GREY),
            ("never", ~turned, GREY)]
    for i, (lab, m, c) in enumerate(cats):
        p = n[m].passed.mean() * 100
        ax.bar(i, max(p, 0.8), color=c, width=0.65, edgecolor="#404040", lw=0.4)
        ax.text(i, max(p, 0.8) + 3, f"{p:.1f}%\nn={m.sum():,}", ha="center", fontsize=5.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([c[0] for c in cats], fontsize=5.8)
    ax.set_ylim(0, 115); ax.set_ylabel("pass rate (%)"); ax.grid(axis="x", visible=False)
    ax.set_title("(b) Timing among narrow episodes", loc="left", x=-0.2)
    save(fig, SUPP + "c05_mechanism_full.pdf")

# ---------------- C07 strategy scatter (column)
def c07():
    W = d[d.A >= 1.0 - 1e-9]; N = d[d.A < 1.0]
    wr = W.groupby("model").max_rotation_deg.mean(); nar = N.groupby("model").passed.mean() * 100
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for m in wr.index:
        c = RUST if m in TOP2 else (GOLD if m == "mimo-v2.6-pro" else GREY)
        ax.scatter(wr[m], nar[m], s=22 if c != GREY else 16, color=c, edgecolor="white", lw=0.5, zorder=3)
    offs = {"deepseek-v4-flash-vision-exp": (-1.5, -7, "right"), "gemini-3-flash-preview": (-1.5, 5, "right"),
            "mimo-v2.6-pro": (1.5, 0, "left")}
    for m, (dx, dy, ha) in offs.items():
        ax.text(wr[m] + dx, nar[m] + dy, LAB[m], fontsize=6.5, ha=ha, va="center", color=RUST if m in TOP2 else GOLD)
    r_all = np.corrcoef(wr, nar[wr.index])[0, 1]; k = ~wr.index.isin(TOP2)
    r_wo = np.corrcoef(wr[k], nar[wr.index][k])[0, 1]
    ax.text(0.98, 0.30, f"Pearson r = {r_all:.2f} (15 models)\nr = {r_wo:.2f} without the top two",
            transform=ax.transAxes, ha="right", fontsize=6.5, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(-2, 56); ax.set_ylim(-5, 108)
    ax.set_xlabel("mean maximum body rotation, wide bins A/S ≥ 1.0 (°)")
    ax.set_ylabel("narrow-gap pass rate, A/S ≤ 0.9 (%)")
    save(fig, MAIN + "c07_strategy.pdf")

# ---------------- C13 stage-2 transfer (figure*-style, 6.9 in)
def c13():
    s2 = pd.read_csv(U + "04_S2_105对配对.csv")
    s2["passed"] = s2.passed.astype(str) == "True"
    g = s2.groupby("model").agg(b=("baseline_turns", "mean"), m=("turn_count", "mean"), lcs=("copy_lcs_fraction", "mean")).sort_values("m")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"wspace": 0.55, "width_ratios": [1.4, 1]})
    ax = axs[0]; y = np.arange(len(g))
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.plot([r.b, r.m], [yi, yi], color=LG, lw=1.6, zorder=1)
    ax.scatter(g.b, y, s=20, color=GREY, zorder=2, label="no memory")
    ax.scatter(g.m, y, s=20, color=NAVY, zorder=3, label="with memory")
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.text(r.m + 0.25, yi, f"{r.m - r.b:+.1f}", va="center", fontsize=6, color=NAVY)
    ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in g.index], fontsize=6.6)
    ax.set_xlim(-0.3, 7.2); ax.set_xlabel("turns at A/S 2.0 (mean per episode)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)
    ax.set_title("(a) Turns: no memory vs. own memory", loc="left", x=-0.6)
    ax = axs[1]
    ax.barh(y, g.lcs, color=PURPLE, edgecolor="#404040", lw=0.3, height=0.6)
    for yi, v in zip(y, g.lcs): ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in g.index], fontsize=5.8); ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color=RUST, lw=0.7, ls="--")
    ax.set_xlabel("action-sequence LCS vs. source memory")
    ax.set_title("(b) How closely the test copies the memory", loc="left", x=-0.1)
    ax.grid(axis="y", visible=False)
    save(fig, MAIN + "c13_stage2.pdf")

# ---------------- C15 C16 C17 body swap (three panels for figure*)
def body():
    s3 = pd.read_csv(U + "05_S3_100集体型.csv"); s3["passed"] = s3.passed.astype(str) == "True"
    five = ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview", "mimo-v2.6-pro", "claude-haiku-4-5-20251001", "gpt-5"]
    h1 = d[d.model.isin(five)]
    lv = [1.3, 1.1, 1.0, 0.9]
    # C15
    fig, ax = plt.subplots(figsize=(2.25, 2.2))
    hp = h1.groupby("A").passed.mean().reindex(lv) * 100
    ax.plot(lv, hp.values, "-o", color=NAVY, ms=3.5, lw=1.4, label="H1 (S 0.570 m), n=125 per bin")
    for rb, c, lab in [("G1", RUST, "G1 (S 0.450 m), n=10"), ("R1", GOLD, "R1 (S 0.357 m), n=10")]:
        sub = s3[s3.robot == rb].groupby("a_s_requested").passed.mean().reindex(lv) * 100
        ax.plot(lv, sub.values, "-o", color=c, ms=3.5, lw=1.4, label=lab)
    ax.set_xlim(1.35, 0.85); ax.set_xticks(lv); ax.set_ylim(40, 104)
    ax.set_xlabel("A/S"); ax.set_ylabel("pass rate (%)")
    ax.legend(frameon=False, loc="lower left", fontsize=5.6, handlelength=1.5)
    ax.set_title("(a) Same threshold for all bodies", loc="left", x=-0.2)
    save(fig, MAIN + "c15_body_pass.pdf")
    # C16
    fig, ax = plt.subplots(figsize=(2.25, 2.2))
    req = {"H1": 54.0, "G1": 58.6, "R1": 65.4}
    h1rot = d[d.model.isin(five) & (d.A == 0.9)].max_rotation_deg.values
    data = {"H1": h1rot, "G1": s3[(s3.robot == "G1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values,
            "R1": s3[(s3.robot == "R1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values}
    rng = np.random.default_rng(0)
    for i, (rb, c) in enumerate([("H1", NAVY), ("G1", RUST), ("R1", GOLD)]):
        v = data[rb]
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=7, color=c, alpha=0.7, lw=0)
        ax.plot([i - 0.32, i + 0.32], [req[rb]] * 2, color="#262626", lw=1.2, ls="--")
        ax.text(i + 0.36, req[rb], f"req. {req[rb]:.1f}°\nmet {(v >= req[rb]).mean()*100:.0f}%", fontsize=5.5, va="center")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["H1\n0.570 m", "G1\n0.450 m", "R1\n0.357 m"], fontsize=6.2)
    ax.set_xlim(-0.5, 2.9); ax.set_ylim(-4, 102); ax.set_ylabel("max rotation at A/S 0.9 (°)")
    ax.set_title("(b) Turn size is not rescaled", loc="left", x=-0.2)
    ax.grid(axis="x", visible=False)
    save(fig, MAIN + "c16_body_rot.pdf")
    # C17
    fig, ax = plt.subplots(figsize=(2.25, 2.2))
    h = h1.groupby("channel_width_m").passed.mean() * 100
    ax.axvspan(0.44, 0.475, color=PEACH, lw=0, zorder=0)
    ax.scatter(h.index, h.values, s=10, color=NAVY, zorder=3)
    for rb, c in [("G1", RUST), ("R1", GOLD)]:
        g = s3[s3.robot == rb].groupby("channel_width_m").passed.mean() * 100
        ax.scatter(g.index, g.values, s=16, color=c, edgecolor="white", lw=0.4, zorder=4)
    h46 = h.loc[min(h.index, key=lambda w: abs(w - 0.456))]
    ax.annotate(f"≈0.46 m:\nG1, R1 100%\nH1 (same five) {h46:.0f}%", xy=(0.456, h46), xytext=(0.66, 28),
                fontsize=5.8, color=RUST, ha="left", arrowprops=dict(arrowstyle="->", lw=0.5, color=RUST))
    ax.legend(handles=[Line2D([], [], color=NAVY, marker="o", ls="", ms=3.5, label="H1 (same five models)"),
                       Line2D([], [], color=RUST, marker="o", ls="", ms=3.5, label="G1"),
                       Line2D([], [], color=GOLD, marker="o", ls="", ms=3.5, label="R1")],
              frameon=False, loc="center right", fontsize=5.6, handletextpad=0.2)
    ax.set_xlabel("physical opening width (m)"); ax.set_ylabel("pass rate (%)"); ax.set_ylim(15, 110)
    ax.set_title("(c) pass rate by opening width", loc="left", x=-0.2)
    save(fig, MAIN + "c17_abs_width.pdf")

# ---------------- C22 rotation heatmaps (figure*)
def c22():
    mr = d.groupby(["model", "A"]).max_rotation_deg.mean().unstack()
    p30 = d.groupby(["model", "A"]).max_rotation_deg.apply(lambda x: (x >= 30).mean() * 100).unstack()
    order = d.groupby("model").passed.mean().sort_values(ascending=False).index
    order = [m for m in order if m in mr.index]
    mr = mr.loc[order, LEV]; p30 = p30.loc[order, LEV]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.0), gridspec_kw={"wspace": 0.08})
    for k, (ax, mat, cmap, vmax, title, fmt) in enumerate([
        (axs[0], mr, "Blues", 90, "(a) mean maximum body rotation (°)", "{:.0f}"),
        (axs[1], p30, "Reds", 100, r"(b) share of episodes with max rotation ≥ 30° (%)", "{:.0f}")]):
        im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=4.6,
                        color="white" if v > 0.62 * vmax else "#1A1A1A")
        ax.set_xticks(range(len(LEV))); ax.set_xticklabels([f"{a:.1f}" for a in LEV], rotation=90, fontsize=5.6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LAB[m] for m in order] if k == 0 else [], fontsize=6.2)
        ax.axvline(LEV.index(0.9) - 0.5, color="#262626", lw=0.9, ls="--")
        ax.grid(False); ax.set_title(title, loc="left", x=0, fontsize=7.6)
        ax.set_xlabel("A/S (wide bins 2.0–1.0 | narrow 0.9–0.4)", fontsize=6.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cb.ax.tick_params(labelsize=5.5)
    save(fig, SUPP + "c22_rot_heat.pdf")

# ---------------- C47 feasibility mask (column, taller)
def c47():
    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    th = np.linspace(0, 90, 2000)
    W = np.sqrt((S * np.cos(np.deg2rad(th))) ** 2 + 0) if False else S * np.abs(np.cos(np.deg2rad(th))) + D * np.abs(np.sin(np.deg2rad(th)))
    x_levels = np.linspace(0.36, 1.0, 400)
    theta_star = []
    for x in x_levels:
        idx = np.where((th >= 21.1) & (W <= x * S))[0]
        theta_star.append(th[idx[0]] if len(idx) else 90)
    theta_star = np.array(theta_star)
    ax.axvspan(1.0, 2.1, color="#F3D27A", alpha=0.45, lw=0)
    ax.fill_between(x_levels, theta_star, 90, color="#F3D27A", alpha=0.45, lw=0)
    ax.plot(x_levels, theta_star, color="#7F6000", lw=0.9)
    ax.text(0.73, 22, "blocked:\nθ < θ*(A/S)", ha="center", fontsize=6.3, color="#404040")
    ax.text(1.55, 76, "frontal passage feasible", ha="center", fontsize=6.3, color="#7F6000")
    mean = d.groupby("A").max_rotation_deg.mean()
    ax.plot(mean.index, mean.values, "-o", color=NAVY, ms=2.8, lw=1.1, label="measured mean maximum rotation")
    ax.axvline(1.0, color="#262626", lw=0.6, ls="--")
    ax.set_xlim(2.05, 0.35); ax.set_ylim(0, 92)
    ax.set_xticks([2.0, 1.6, 1.2, 1.0, 0.8, 0.6, 0.4])
    ax.set_xlabel("A/S"); ax.set_ylabel("body rotation θ (°)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.8)
    save(fig, SUPP + "c47_mask.pdf")

if __name__ == "__main__":
    c01(); c05(); c07(); c13(); body(); c22(); c47()
    print("v2 figures written")
