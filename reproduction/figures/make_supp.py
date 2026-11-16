"""Supplementary figures for EmbodiedBAO. Every number is computed from the raw exports."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

D = "/mnt/user-data/uploads/"
OUT = os.path.join(os.path.dirname(__file__), "..", "fig", "supp")
os.makedirs(OUT, exist_ok=True)
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 6.8,
    "axes.titlesize": 7.2, "axes.labelsize": 6.8, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.5, "grid.linestyle": "--",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
NAVY, RUST, OLIVE, RED, GOLD = "#2F5597", "#C55A11", "#548235", "#C00000", "#BF9000"
GREY, LGREY, PBLUE, PEACH = "#7F7F7F", "#C9C9C9", "#DEEBF7", "#FBE5D6"
HEAT = LinearSegmentedColormap.from_list("h", ["#FFFFFF", "#9DC3E6", "#2F5597"])
HEATR = LinearSegmentedColormap.from_list("r", ["#FFFFFF", "#F4B183", "#C55A11"])
TOP2 = ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview"]
LABEL = {
    "deepseek-v4-flash-vision-exp": "DeepSeek-V4", "gemini-3-flash-preview": "Gemini-3",
    "mimo-v2.6-pro": "MiMo-v2.6", "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B",
    "claude-haiku-4-5-20251001": "Claude-Haiku", "gpt-5": "GPT-5", "gemini-2.5-flash": "Gemini-2.5",
    "glm-4.5v": "GLM-4.5V", "gpt-4o-mini": "GPT-4o-mini", "kimi/kimi-k2.6": "Kimi-K2.6",
    "kimi/kimi-k2.5": "Kimi-K2.5", "gpt-4.1": "GPT-4.1", "glm-4.6v": "GLM-4.6V",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
}
SCENE = {"stage1.1": "Lab", "stage1.2": "Warehouse", "stage1.3": "Library", "stage1.4": "Park", "stage1.5": "Supermarket"}
REQ = {0.9: 54.0, 0.8: 62.8, 0.7: 70.3, 0.6: 77.1, 0.5: 83.3, 0.4: 89.2}
S, DD = 0.570, 0.220

d = pd.read_csv(D + "02_S1逐回合6375行.csv")
d["passed"] = d.passed.astype(str) == "True"
d["A"] = d.a_s_ratio.round(1)
LEV = sorted(d.A.unique(), reverse=True)
MODELS = [m for m in LABEL if m in set(d.model)]
MODELS = sorted(MODELS, key=lambda m: -d[(d.model == m) & (d.A < 1)].passed.mean())
prof = pd.read_csv(D + "03_S1逐模型画像.csv")
pih = pd.read_csv(D + "06_pihat逐模型.csv").set_index("model")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def ax_letter(ax, s, title):
    ax.set_title(f"{s} {title}", loc="left", x=0.0)


# ------------------------------------------------ C02: model x level pass rate (values)
def c02():
    pm = d.groupby(["model", "A"]).passed.mean().unstack().loc[MODELS, LEV] * 100
    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    im = ax.imshow(pm.values, cmap=HEAT, vmin=0, vmax=100, aspect="auto")
    for i in range(pm.shape[0]):
        for j in range(pm.shape[1]):
            v = pm.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.9,
                    color="white" if v > 60 else "#262626")
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([LABEL[m] for m in MODELS], fontsize=5.8)
    ax.set_xticks(range(len(LEV))); ax.set_xticklabels([f"{a:.1f}" for a in LEV], fontsize=5.2, rotation=0)
    ax.axvline(LEV.index(1.0) + 0.5, color="#262626", lw=0.8, ls="--")
    ax.axvline(LEV.index(0.9) - 0.5, color=RUST, lw=0.9, ls="--")
    ax.text(LEV.index(1.0), -0.9, "zero margin", ha="center", fontsize=5.6)
    ax.text(LEV.index(0.9) + 0.2, -0.9, "rotation required", ha="left", fontsize=5.6, color=RUST)
    ax.set_xlabel("A/S (17 bins, n = 25 per cell)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("pass rate (%)", fontsize=5.8)
    ax_letter(ax, "", "Pass rate by model and A/S bin (Stage 1, 6,375 episodes)")
    save(fig, "s01_pass_heatmap")


# ------------------------------------------------ C22: rotation heatmaps (mean and P>=30)
def c22():
    mr = d.groupby(["model", "A"]).max_rotation_deg.mean().unstack().loc[MODELS, LEV]
    p30 = d.groupby(["model", "A"]).max_rotation_deg.apply(lambda x: (x >= 30).mean() * 100).unstack().loc[MODELS, LEV]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.2), gridspec_kw={"wspace": 0.55})
    for ax, mat, cm, vmax, t, fmt in [(axs[0], mr, HEAT, 90, "mean max body rotation (°)", "{:.0f}"),
                                       (axs[1], p30, HEATR, 100, r"P(max rotation $\geq$ 30°) (%)", "{:.0f}")]:
        im = ax.imshow(mat.values, cmap=cm, vmin=0, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=3.9,
                        color="white" if v > 0.6 * vmax else "#262626")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([LABEL[m] for m in MODELS] if ax is axs[0] else [], fontsize=5.4)
        ax.set_xticks(range(len(LEV))); ax.set_xticklabels([f"{a:.1f}" for a in LEV], fontsize=4.6, rotation=90)
        ax.axvline(LEV.index(0.9) - 0.5, color="#262626", lw=0.7, ls="--")
        ax.grid(False)
        ax.set_title(t, fontsize=6.6)
    axs[0].set_xlabel("A/S"); axs[1].set_xlabel("A/S")
    save(fig, "s02_rotation_heatmaps")


# ------------------------------------------------ C08 forest + C30 interpolation
def c08():
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.45})
    ax = axs[0]
    rows = pih.reset_index()
    rows = rows[(rows["class"] == "onset") & (~rows.model.str.startswith("ALL"))].sort_values("pi_rot30", ascending=False)
    y = np.arange(len(rows))
    ax.axvspan(1.10, 1.30, color=GOLD, alpha=0.18, lw=0)
    ax.axvline(0.917, color=NAVY, ls="--", lw=0.8)
    ax.errorbar(rows.pi_rot30, y, xerr=[rows.pi_rot30 - rows.pi_rot30_lo, rows.pi_rot30_hi - rows.pi_rot30],
                fmt="o", ms=3.2, color=NAVY, ecolor=GREY, elinewidth=0.9, capsize=1.6)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in rows.model], fontsize=5.8)
    ax.set_xlim(0.4, 1.12); ax.set_xlabel(r"$\hat{\pi}$: A/S where P(rotation $\geq$ 30°) = 0.5")
    ax.text(1.115, len(rows) - 1, "human\n1.10–1.30", fontsize=5.4, ha="right", color=GOLD, va="top")
    ax.text(0.925, -0.9, r"pooled 0.917 [0.906, 0.927]", fontsize=5.4, color=NAVY)
    ax.set_ylim(-1.5, len(rows) - 0.4)
    ax_letter(ax, "(a)", "Per-model onset (10 models with a defined onset)")
    ax = axs[1]
    lv = [1.0, 0.9]
    p = d[d.A.isin(lv)].groupby("A").apply(lambda g: (g.max_rotation_deg >= 30).mean() * 100)
    ax.plot([1.0, 0.9], [p[1.0], p[0.9]], color=NAVY, lw=1.4, marker="o", ms=4)
    ax.axhline(50, color=RUST, ls="--", lw=0.8)
    t = (50 - p[1.0]) / (p[0.9] - p[1.0]); pihat = 1.0 + (0.9 - 1.0) * t
    ax.plot([pihat], [50], "o", color=RUST, ms=4.5)
    ax.annotate(f"π̂ = {pihat:.3f}", xy=(pihat, 50), xytext=(pihat + 0.07, 72),
                fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color=RUST), color=RUST)
    ax.text(1.0, p[1.0] - 9, f"{p[1.0]:.1f}%", ha="center", fontsize=5.8, color=NAVY)
    ax.text(0.9, p[0.9] + 4, f"{p[0.9]:.1f}%", ha="center", fontsize=5.8, color=NAVY)
    ax.set_xlim(1.05, 0.85); ax.set_ylim(0, 85)
    ax.set_xlabel("A/S"); ax.set_ylabel("P(max rotation ≥ 30°) (%)")
    ax_letter(ax, "(b)", "Linear interpolation")
    save(fig, "s03_pihat")


# ------------------------------------------------ C29 3x6 small multiples
def c29():
    fig, axs = plt.subplots(3, 6, figsize=(6.9, 4.4), sharex=True, sharey=True)
    for k, ax in enumerate(axs.flat):
        if k < len(MODELS):
            m = MODELS[k]
            s = d[d.model == m].groupby("A").agg(p=("passed", "mean"), r=("max_rotation_deg", lambda x: (x >= 30).mean()))
            s = s.loc[LEV]
            ax.plot(LEV, s.p * 100, color=NAVY, lw=1.0, label="pass")
            ax.plot(LEV, s.r * 100, color=RUST, lw=0.9, ls="--", label=r"P(rot$\geq$30°)")
            ax.set_title(LABEL[m], fontsize=5.8, pad=2)
        elif k == len(MODELS):
            s = d.groupby("A").agg(p=("passed", "mean"), r=("max_rotation_deg", lambda x: (x >= 30).mean())).loc[LEV]
            ax.plot(LEV, s.p * 100, color=NAVY, lw=1.4); ax.plot(LEV, s.r * 100, color=RUST, lw=1.1, ls="--")
            ax.set_title("pooled (15 models)", fontsize=5.8, pad=2)
        else:
            ax.axis("off"); continue
        ax.axvspan(1.10, 1.30, color=GOLD, alpha=0.18, lw=0)
        ax.set_xlim(2.05, 0.35); ax.set_ylim(-4, 104)
        ax.set_xticks([2.0, 1.0, 0.4]); ax.set_yticks([0, 50, 100]); ax.tick_params(labelsize=4.8)
    for ax in axs[-1]:
        ax.set_xlabel("A/S", fontsize=5.6)
    axs[0, 0].legend(frameon=False, loc="lower left", fontsize=5)
    fig.text(0.5, 1.0, "Per-model pass rate (solid) and rotation probability (dashed) versus A/S", ha="center", va="top", fontsize=7.2)
    save(fig, "s04_small_multiples")


# ------------------------------------------------ C31 violins by level + C43 CDFs
def c31_c43():
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.4), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.35})
    ax = axs[0]
    data = [d[d.A == a].max_rotation_deg.values for a in LEV]
    parts = ax.violinplot(data, positions=range(len(LEV)), vert=False, widths=0.85, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(RUST if LEV[i] < 1 else NAVY); pc.set_alpha(0.55); pc.set_edgecolor("#404040"); pc.set_linewidth(0.4)
    ax.scatter([np.mean(x) for x in data], range(len(LEV)), s=6, color="white", edgecolor="#262626", lw=0.4, zorder=3)
    ax.set_yticks(range(len(LEV))); ax.set_yticklabels([f"{a:.1f}" for a in LEV], fontsize=5.2)
    ax.invert_yaxis(); ax.set_xlabel("max body rotation per episode (°)"); ax.set_ylabel("A/S")
    ax.set_xlim(-5, 185); ax.grid(axis="y", visible=False)
    ax_letter(ax, "(a)", "Rotation distribution, 17 bins (dot = mean)")
    ax = axs[1]
    for lab, sub, c in [("A/S ≥ 1.0", d[d.A >= 1.0], NAVY), ("A/S < 1.0", d[d.A < 1.0], RUST)]:
        x = np.sort(sub.max_rotation_deg.values); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, lw=1.1, label=lab)
    ax.set_xlabel("max body rotation (°)"); ax.set_ylabel("cumulative fraction")
    ax.legend(frameon=False, loc="lower right")
    ax_letter(ax, "(b)", "Empirical CDF of rotation by regime")
    save(fig, "s05_rotation_distribution")


# ------------------------------------------------ C25 action composition + C41 metric bars
def c25_c41():
    def cat(a):
        if a == "forward": return "forward"
        if a in ("turn_left", "turn_right"): return "turn"
        if a.startswith("look"): return "look"
        return "side"
    rows = []
    for _, r in d.iterrows():
        acts = str(r.action_sequence).split(",")
        for a in acts:
            rows.append((r.A, cat(a)))
    x = pd.DataFrame(rows, columns=["A", "cat"])
    bands = [("A/S 2.0–1.5", lambda a: a >= 1.5), ("A/S 1.4–1.1", lambda a: 1.1 <= a <= 1.4),
             ("A/S 1.0", lambda a: abs(a - 1.0) < 1e-9), ("A/S 0.9", lambda a: abs(a - 0.9) < 1e-9),
             ("A/S 0.8–0.6", lambda a: 0.6 <= a <= 0.8), ("A/S 0.5–0.4", lambda a: a <= 0.5)]
    cols = {"forward": NAVY, "turn": RUST, "look": "#8E6BB0", "side": GOLD}
    fig, axs = plt.subplots(1, 6, figsize=(6.9, 2.5), gridspec_kw={"wspace": 0.25})
    axes = list(axs)
    for ax, (lab, f) in zip(axes, bands):
        sub = x[x.A.apply(f)]
        sh = sub.cat.value_counts(normalize=True).reindex(["forward", "turn", "look", "side"]).fillna(0) * 100
        bot = 0
        for k, v in sh.items():
            ax.bar(0, v, bottom=bot, color=cols[k], width=0.6, edgecolor="white", lw=0.4, label=k if lab == bands[0][0] else None)
            if v > 4: ax.text(0, bot + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=5.4, color="white")
            bot += v
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 100); ax.set_xticks([])
        ax.set_title(lab, fontsize=5.6); ax.grid(False)
        if ax is not axes[0]: ax.set_yticklabels([])
    axes[0].set_ylabel("action share (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=5.4)
    fig.subplots_adjust(bottom=0.24)
    fig.suptitle("(a) Action composition by A/S band (all Stage-1 steps)", fontsize=7.2, x=0.02, ha="left", y=1.04)
    save(fig, "s06a_action_composition")

    # C41 metric grouped bars per model (from the per-model profile)
    p = prof.set_index("model").loc[MODELS]
    metrics = [("pass_pct", "overall pass", NAVY), ("narrow_pass_pct", "narrow pass", RUST),
               ("narrow_met_req_pct", "rotation adequate", GOLD), ("early_turn_pct_in_turns", "early turn", "#8E6BB0"),
               ("look_down_ep_pct", "look_down", "#5FB3B3")]
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    w = 0.16
    x0 = np.arange(len(MODELS))
    for k, (col, lab, c) in enumerate(metrics):
        ax.bar(x0 + (k - 2) * w, p[col].values, width=w, color=c, edgecolor="#404040", lw=0.3, label=lab)
    ax.set_xticks(x0); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=40, ha="right", fontsize=5.4)
    ax.set_ylabel("% of episodes / turns"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=5, loc="upper right", fontsize=5.4)
    ax.grid(axis="x", visible=False)
    ax_letter(ax, "(b)", "Metric profile per model")
    save(fig, "s06b_metric_profile")


# ------------------------------------------------ C40 per-scene curves + C12 robustness
def c40_c12():
    d["scene_lab"] = d.scene.map(SCENE)
    fig, axs = plt.subplots(1, 5, figsize=(6.9, 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, sc in zip(axs, ["Lab", "Warehouse", "Library", "Park", "Supermarket"]):
        sub = d[d.scene_lab == sc]
        for m in MODELS:
            s = sub[sub.model == m].groupby("A").passed.mean().reindex(LEV) * 100
            ax.plot(LEV, s.values, color=LGREY, lw=0.4)
        s = sub.groupby("A").passed.mean().reindex(LEV) * 100
        ax.plot(LEV, s.values, color=NAVY, lw=1.3)
        ax.axvspan(1.10, 1.30, color=GOLD, alpha=0.18, lw=0)
        ax.set_xlim(2.05, 0.35); ax.set_xticks([2.0, 1.0, 0.4]); ax.set_title(("(a) " if sc == "Lab" else "") + sc, fontsize=6.2, pad=2, loc="left" if sc == "Lab" else "center")
        ax.set_xlabel("A/S", fontsize=5.6)
    axs[0].set_ylabel("pass rate (%)")
    fig.suptitle("Per-scene pass rate (navy: scene mean; grey: each model)", fontsize=7, y=1.03)
    save(fig, "s07a_per_scene")

    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.1), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    scene_n = d[d.A < 1].groupby("scene_lab").passed.mean() * 100
    scene_w = d[d.A >= 1].groupby("scene_lab").passed.mean() * 100
    order = ["Lab", "Warehouse", "Library", "Park", "Supermarket"]
    xx = np.arange(5)
    ax.bar(xx - 0.2, scene_w[order].values, width=0.38, color=NAVY, label="A/S ≥ 1.0")
    ax.bar(xx + 0.2, scene_n[order].values, width=0.38, color=RUST, label="A/S < 1.0")
    for i, s in enumerate(order):
        ax.text(i + 0.2, scene_n[s] + 1.5, f"{scene_n[s]:.1f}", ha="center", fontsize=5.2)
    ax.set_xticks(xx); ax.set_xticklabels(order, fontsize=5.4); ax.set_ylim(0, 115); ax.set_ylabel("pass rate (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=5.4)
    ax.grid(axis="x", visible=False)
    ax_letter(ax, "(b)", "Five scene skins")
    ax = axs[1]
    mk = d.groupby("marker_slot").passed.mean() * 100
    mkn = d[d.A < 1].groupby("marker_slot").passed.mean() * 100
    ax.plot(mk.index, mk.values, "o-", color=NAVY, ms=3, lw=1, label="all bins")
    ax.plot(mkn.index, mkn.values, "s-", color=RUST, ms=3, lw=1, label="narrow")
    ax.set_xticks(range(1, 6)); ax.set_xlabel("marker slot (5 per scene)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="center right", fontsize=5.4)
    ax_letter(ax, "(c)", "Five target slots")
    save(fig, "s07b_robustness")


# ------------------------------------------------ C38 transfer matrix + C44 paired differences
def c38_c44():
    s2 = pd.read_csv(D + "04_S2_105对配对.csv")
    s2["passed"] = s2.passed.astype(str) == "True"
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1.45, 1, 1, 1], "wspace": 0.5})
    ax = axs[0]
    mat = s2.pivot_table(index="source_level", columns="model", values="delta_turns_vs_baseline", aggfunc="mean")
    mat = mat.sort_index(ascending=False)
    cols = [m for m in MODELS if m in mat.columns]
    mat = mat[cols]
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:+.1f}", ha="center", va="center", fontsize=4.8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([LABEL[m] for m in cols], rotation=40, ha="right", fontsize=5)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels([f"L{l}" for l in mat.index], fontsize=5.4)
    ax.grid(False)
    ax_letter(ax, "(a)", "Δ turns by source bin (rows) × model")
    ax = axs[1]
    t = s2.groupby("model").apply(lambda g: pd.Series({"b": g.baseline_turns.mean(), "m": g.turn_count.mean()}))
    ax.scatter(s2.baseline_turns + np.random.default_rng(1).uniform(-0.12, 0.12, len(s2)),
               s2.turn_count + np.random.default_rng(2).uniform(-0.12, 0.12, len(s2)), s=7, color=NAVY, alpha=0.6, lw=0)
    ax.plot([0, 8], [0, 8], ":", color="#262626", lw=0.7)
    ax.set_xlabel("turns without memory"); ax.set_ylabel("turns with memory")
    ax.text(7.8, 0.3, f"{(s2.turn_count > s2.baseline_turns).sum()} up / {(s2.turn_count < s2.baseline_turns).sum()} down", ha="right", fontsize=5.6)
    ax_letter(ax, "(b)", "105 pairs")
    ax = axs[2]
    dlt = s2.turn_delta_vs_baseline
    ax.hist(dlt, bins=np.arange(-0.5, 8, 1), color=RUST, edgecolor="#404040", lw=0.4)
    ax.set_xlabel("Δ turns (memory − baseline)"); ax.set_ylabel("pairs")
    ax_letter(ax, "(c)", f"mean {dlt.mean():+.2f}")
    ax = axs[3]
    s3 = pd.read_csv(D + "05_S3_100集体型.csv")
    k = ["model", "scene", "a_s_requested"]
    pv = s3.pivot_table(index=k, columns="robot", values="max_rotation_deg", aggfunc="first").dropna()
    mean_pair = (pv.G1 + pv.R1) / 2
    diff = pv.G1 - pv.R1
    ax.scatter(mean_pair, diff, s=9, color=GOLD, edgecolor="#404040", lw=0.3)
    ax.axhline(0, color="#262626", ls="--", lw=0.6)
    ax.axhline(diff.mean(), color=RUST, ls=":", lw=0.8)
    ax.set_xlabel("mean rotation of G1/R1 pair (°)"); ax.set_ylabel("G1 − R1 rotation (°)")
    ax_letter(ax, "(d)", f"body swap, {len(pv)} pairs")
    save(fig, "s08_transfer_paired")


# ------------------------------------------------ C42 rank correlation + C10 look_down scatter
def c42_c10():
    cols = {"pass_pct": "overall pass", "narrow_pass_pct": "narrow pass", "narrow_met_req_pct": "rotation adequate",
            "early_turn_pct_in_turns": "early turn", "wide_rot_deg": "wide rotation", "look_down_ep_pct": "look_down"}
    p = prof.set_index("model")[list(cols)].rename(columns=cols)
    rho = p.corr(method="spearman")
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.4), gridspec_kw={"width_ratios": [1.2, 1, 1], "wspace": 0.55})
    ax = axs[0]
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(rho)):
        for j in range(len(rho)):
            ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", fontsize=4.9)
    ax.set_xticks(range(len(rho))); ax.set_xticklabels(rho.columns, rotation=40, ha="right", fontsize=5)
    ax.set_yticks(range(len(rho))); ax.set_yticklabels(rho.index, fontsize=5.4); ax.grid(False)
    ax_letter(ax, "(a)", "Spearman ρ across 15 models")
    ax = axs[1]
    ax.scatter(prof.pass_pct.rank(ascending=False), prof.narrow_met_req_pct.rank(ascending=False), s=10, color=NAVY)
    ax.plot([1, 15], [1, 15], ":", color="#262626", lw=0.6)
    ax.set_xlabel("rank by overall pass"); ax.set_ylabel("rank by rotation adequacy"); ax.invert_xaxis(); ax.invert_yaxis()
    ax_letter(ax, "(b)", "Ranking mismatch")
    ax = axs[2]
    ax.scatter(prof.look_down_ep_pct, prof.narrow_pass_pct, s=12, color="#8E6BB0", edgecolor="white", lw=0.3)
    r = np.corrcoef(prof.look_down_ep_pct, prof.narrow_pass_pct)[0, 1]
    ax.text(97, 8, f"r = {r:.2f}", ha="right", fontsize=5.8)
    ax.set_xlabel("episodes with look_down (%)"); ax.set_ylabel("narrow pass rate (%)")
    ax_letter(ax, "(c)", "Looking down ≠ passing")
    save(fig, "s09_rank_lookdown")


# ------------------------------------------------ C18 Stage 3 per model + C19 data quality
def c18_c19():
    s3 = pd.read_csv(D + "05_S3_100集体型.csv"); s3["passed"] = s3.passed.astype(str) == "True"
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.2), gridspec_kw={"width_ratios": [1.5, 1, 1, 1], "wspace": 0.5})
    ax = axs[0]
    ms = list(s3.model.unique())
    g = s3.groupby(["model", "robot"]).passed.agg(["mean", "size"])
    xx = np.arange(len(ms))
    for k, (rb, c) in enumerate([("G1", RUST), ("R1", GOLD)]):
        vals = [g.loc[(m, rb), "mean"] * 100 for m in ms]
        ns = [g.loc[(m, rb), "size"] for m in ms]
        ax.bar(xx + (k - 0.5) * 0.38, vals, width=0.36, color=c, edgecolor="#404040", lw=0.3, label=f"{rb} (n={ns[0]})")
    ax.set_xticks(xx); ax.set_xticklabels([LABEL[m] for m in ms], rotation=30, ha="right", fontsize=5.2)
    ax.set_ylim(0, 115); ax.set_ylabel("pass rate (%)"); ax.legend(frameon=False, ncol=2, fontsize=5.2, loc="upper center")
    ax.grid(axis="x", visible=False)
    ax_letter(ax, "(a)", "Body swap, per model × body")
    dq = d.groupby("model").agg(inv=("invalid_responses", "sum"), col=("wall_collisions", lambda x: (x > 0).mean() * 100),
                                steps=("total_steps", "mean"))
    dq = dq.loc[MODELS]
    names = [LABEL[m] for m in dq.index]
    for ax, col, lab, c, t in [(axs[1], "inv", "invalid replies", RUST, "(b)"),
                               (axs[2], "col", "episodes with collision (%)", GOLD, "(c)"),
                               (axs[3], "steps", "mean steps", NAVY, "(d)")]:
        ax.barh(range(len(dq)), dq[col].values, color=c, edgecolor="#404040", lw=0.3, height=0.7)
        ax.set_yticks(range(len(dq))); ax.set_yticklabels(names if ax is axs[1] else [], fontsize=5.1)
        ax.invert_yaxis(); ax.set_xlabel(lab, fontsize=6); ax.grid(axis="y", visible=False)
        ax_letter(ax, t, "")
    save(fig, "s10_stage3_quality")


# ------------------------------------------------ C47 feasibility mask + C45 adequacy band
def c47_c45():
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
    ax = axs[0]
    th = np.linspace(0, 90, 400)
    ratio = (S * np.abs(np.cos(np.deg2rad(th))) + DD * np.abs(np.sin(np.deg2rad(th)))) / S
    ax.axvspan(1.0, 2.1, color=GOLD, alpha=0.35, lw=0)
    ax.fill_between(ratio, th, 90, color=GOLD, alpha=0.35, lw=0)
    ax.plot(ratio, th, color="#7F6000", lw=0.8)
    ax.text(1.55, 80, "passable for θ ≥ θ*(A/S)", fontsize=6, color="#7F6000", ha="center")
    ax.text(0.72, 22, "blocked", fontsize=6.2, color="#404040", ha="center")
    mean = d.groupby("A").max_rotation_deg.mean()
    ax.plot(mean.index, mean.values, "o-", color=NAVY, ms=2.6, lw=1.1, label="measured mean max rotation")
    ax.axvline(1.0, color="#262626", ls="--", lw=0.6)
    ax.set_xlim(2.05, 0.35); ax.set_ylim(0, 92); ax.set_xticks([2.0, 1.5, 1.0, 0.8, 0.6, 0.4])
    ax.set_xlabel("A/S"); ax.set_ylabel("body rotation θ (°)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.6)
    ax_letter(ax, "(a)", "Feasibility mask and where agents operate")
    ax = axs[1]
    nar = d[d.A < 1].copy(); nar["req"] = nar.A.map(REQ); nar["met"] = nar.max_rotation_deg >= nar.req
    lv = sorted(REQ, reverse=True)
    rng = np.random.default_rng(3)
    met, pas, mlo, mhi, plo, phi = [], [], [], [], [], []
    for a in lv:
        sub = nar[nar.A == a]
        mm = sub.met.values.astype(float); pp = sub.passed.values.astype(float)
        bm = [rng.choice(mm, len(mm)).mean() * 100 for _ in range(1000)]
        bp = [rng.choice(pp, len(pp)).mean() * 100 for _ in range(1000)]
        met.append(mm.mean() * 100); pas.append(pp.mean() * 100)
        mlo.append(np.percentile(bm, 2.5)); mhi.append(np.percentile(bm, 97.5))
        plo.append(np.percentile(bp, 2.5)); phi.append(np.percentile(bp, 97.5))
    ax.fill_between(lv, mlo, mhi, color=NAVY, alpha=0.18, lw=0)
    ax.plot(lv, met, "o-", color=NAVY, ms=3, lw=1.1, label="rotation ≥ required")
    ax.fill_between(lv, plo, phi, color=RUST, alpha=0.18, lw=0)
    ax.plot(lv, pas, "s-", color=RUST, ms=3, lw=1.1, label="pass")
    ax.set_xlim(0.95, 0.35); ax.set_xlabel("A/S (narrow bins)"); ax.set_ylabel("% of narrow episodes")
    ax.set_ylim(0, 60); ax.legend(frameon=False, loc="upper right", fontsize=5.6)
    ax_letter(ax, "(b)", "Adequacy vs pass, 95% bootstrap")
    save(fig, "s11_mask_adequacy")


if __name__ == "__main__":
    for fn in [c02, c22, c08, c29, c31_c43, c25_c41, c40_c12, c38_c44, c42_c10, c18_c19, c47_c45]:
        fn()
    print("ok", sorted(os.listdir(OUT)))


def c44_paired():
    """Paired differences (memory and body swap), panels b-d of the former transfer figure."""
    s2 = pd.read_csv(D + "04_S2_105对配对.csv")
    s2["passed"] = s2.passed.astype(str) == "True"
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.3), gridspec_kw={"wspace": 0.5})
    ax = axs[0]
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(2)
    ax.scatter(s2.baseline_turns + rng1.uniform(-0.12, 0.12, len(s2)),
               s2.turn_count + rng2.uniform(-0.12, 0.12, len(s2)), s=7, color=NAVY, alpha=0.6, lw=0)
    ax.plot([0, 8], [0, 8], ":", color="#262626", lw=0.7)
    ax.set_xlabel("turns without memory"); ax.set_ylabel("turns with memory")
    ax.text(7.8, 0.3, f"{(s2.turn_count > s2.baseline_turns).sum()} up / {(s2.turn_count < s2.baseline_turns).sum()} down",
            ha="right", fontsize=5.6)
    ax_letter(ax, "(a)", "105 memory pairs")
    ax = axs[1]
    dlt = s2.turn_delta_vs_baseline
    ax.hist(dlt, bins=np.arange(-0.5, 8, 1), color=RUST, edgecolor="#404040", lw=0.4)
    ax.set_xlabel("Δ turns (memory − baseline)"); ax.set_ylabel("pairs")
    ax_letter(ax, "(b)", f"mean {dlt.mean():+.2f}")
    ax = axs[2]
    s3 = pd.read_csv(D + "05_S3_100集体型.csv")
    k = ["model", "scene", "a_s_requested"]
    pv = s3.pivot_table(index=k, columns="robot", values="max_rotation_deg", aggfunc="first").dropna()
    ax.scatter((pv.G1 + pv.R1) / 2, pv.G1 - pv.R1, s=9, color=GOLD, edgecolor="#404040", lw=0.3)
    ax.axhline(0, color="#262626", ls="--", lw=0.6)
    ax.axhline((pv.G1 - pv.R1).mean(), color=RUST, ls=":", lw=0.8)
    ax.set_xlabel("mean rotation of G1/R1 pair (°)"); ax.set_ylabel("G1 − R1 rotation (°)")
    ax_letter(ax, "(c)", f"body swap, {len(pv)} matched pairs")
    save(fig, "s08b_paired")


if __name__ == "__main__":
    c44_paired()
    print("paired ok")
