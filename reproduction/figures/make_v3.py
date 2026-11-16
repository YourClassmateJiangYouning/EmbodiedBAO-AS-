"""v3 redraws: policy violins (main Fig 7), body-swap panels (main Fig 8), feasibility mask (main Fig 2)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from make_main_v2 import (d, LAB, TOP2, MAIN, SUPP, U, S, D as DD, REQ, NAVY, RUST, GOLD, GREY, LG, PEACH, GREEN, RED, PURPLE,
                          save, LEV)
import make_main_v2 as base
plt.rcParams.update({"axes.grid": True})

def policy_violin():
    W = d[d.A >= 1.0 - 1e-9]; N = d[d.A < 1.0]
    nar = N.groupby("model").passed.mean() * 100
    wr = W.groupby("model").max_rotation_deg.mean()
    order = list(nar.sort_values(ascending=True).index)   # bottom at top of plot
    fig, ax = plt.subplots(figsize=(3.3, 3.05))
    for i, m in enumerate(order):
        vals = W[W.model == m].max_rotation_deg.values
        c = RUST if m in TOP2 else (GOLD if m == "mimo-v2.6-pro" else GREY)
        if np.unique(vals).size > 1:
            parts = ax.violinplot([vals], positions=[i], vert=False, widths=0.85, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor("#404040"); pc.set_linewidth(0.4)
        ax.scatter([vals.mean()], [i], s=8, color="#262626", zorder=4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[m] for m in order], fontsize=5.8)
    ax.set_xlim(-6, 100); ax.set_xticks([0, 30, 60, 90]); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("max body rotation per episode, wide bins A/S ≥ 1.0 (°)", fontsize=6.6)
    ax.grid(axis="y", visible=False)
    for i, m in enumerate(order):
        ax.text(103, i, f"{wr[m]:.1f}°", va="center", fontsize=5.4, ha="left", color="#262626")
        ax.text(120, i, f"{nar[m]:.0f}%", va="center", fontsize=5.4, ha="left",
                color=RUST if m in TOP2 else (GOLD if m == "mimo-v2.6-pro" else "#595959"))
    ax.text(103, len(order) - 0.35, "mean", fontsize=5.4, ha="left", color="#262626", weight="bold")
    ax.text(120, len(order) - 0.35, "narrow\npass", fontsize=5.2, ha="left", color="#262626", weight="bold", va="bottom")
    ax.set_xlim(-6, 135)
    ax.set_title("Wide-bin rotation vs. narrow-gap success", loc="left", x=-0.35, fontsize=7.2)
    save(fig, MAIN + "c07_strategy.pdf")

def body_v3():
    s3 = pd.read_csv(U + "05_S3_100集体型.csv"); s3["passed"] = s3.passed.astype(str) == "True"
    five = ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview", "mimo-v2.6-pro", "claude-haiku-4-5-20251001", "gpt-5"]
    h1 = d[d.model.isin(five)]; lv = [1.3, 1.1, 1.0, 0.9]
    # (a) pass vs A/S
    fig, ax = plt.subplots(figsize=(2.3, 2.35))
    hp = h1.groupby("A").passed.mean().reindex(lv) * 100
    ax.plot(lv, hp.values, "-o", color=NAVY, ms=3.5, lw=1.4, label="H1 (n=125/bin)")
    for rb, c in [("G1", RUST), ("R1", GOLD)]:
        sub = s3[s3.robot == rb].groupby("a_s_requested").passed.mean().reindex(lv) * 100
        ax.plot(lv, sub.values, "-o", color=c, ms=3.5, lw=1.4, label=f"{rb} (n=10/bin)")
    ax.axvspan(1.10, 1.30, color="#FFF2CC", lw=0, zorder=0)
    ax.set_xlim(1.35, 0.85); ax.set_xticks(lv); ax.set_ylim(40, 104)
    ax.set_xlabel("A/S"); ax.set_ylabel("pass rate (%)")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="lower left", fontsize=5.4, handlelength=1.4)
    ax.set_title("(a) Same A/S threshold", loc="left", x=-0.25)
    save(fig, MAIN + "c15_body_pass.pdf")
    # (b) strip + box, requirement drawn as dashed line, numbers under the axis
    fig, ax = plt.subplots(figsize=(2.3, 2.35))
    req = {"H1": 54.0, "G1": 58.6, "R1": 65.4}
    data = {"H1": h1[(h1.A == 0.9)].max_rotation_deg.values,
            "G1": s3[(s3.robot == "G1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values,
            "R1": s3[(s3.robot == "R1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values}
    rng = np.random.default_rng(0)
    for i, (rb, c) in enumerate([("H1", NAVY), ("G1", RUST), ("R1", GOLD)]):
        v = data[rb]
        ax.boxplot([v], positions=[i], widths=0.42, showfliers=False, medianprops=dict(color="#262626", lw=0.9),
                   boxprops=dict(color=c, lw=0.8), whiskerprops=dict(color=c, lw=0.6), capprops=dict(lw=0))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color=c, alpha=0.6, lw=0, zorder=3)
        ax.plot([i - 0.42, i + 0.42], [req[rb]] * 2, color="#262626", lw=1.0, ls="--", zorder=4)
        ax.text(i, -17, f"req. {req[rb]:.1f}°\nmet {(v >= req[rb]).mean()*100:.0f}%", ha="center", va="top", fontsize=5.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["H1 (0.570 m)", "G1 (0.450 m)", "R1 (0.357 m)"], fontsize=5.6)
    ax.tick_params(axis="x", pad=2)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-2, 100); ax.set_ylabel("max rotation at A/S 0.9 (°)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) largest turn at A/S=0.9", loc="left", x=-0.25)
    save(fig, MAIN + "c16_body_rot.pdf")
    # (c) absolute width
    fig, ax = plt.subplots(figsize=(2.3, 2.35))
    h = h1.groupby("channel_width_m").passed.mean() * 100
    ax.axvspan(0.44, 0.475, color=PEACH, lw=0, zorder=0)
    ax.scatter(h.index, h.values, s=10, color=NAVY, zorder=3)
    for rb, c in [("G1", RUST), ("R1", GOLD)]:
        g = s3[s3.robot == rb].groupby("channel_width_m").passed.mean() * 100
        ax.scatter(g.index, g.values, s=16, color=c, edgecolor="white", lw=0.4, zorder=4)
    h46 = h.loc[min(h.index, key=lambda w: abs(w - 0.456))]
    ax.annotate(f"≈0.46 m\nG1, R1: 100%\nH1 (same five): {h46:.0f}%", xy=(0.456, h46), xytext=(0.62, 22),
                fontsize=5.5, color=RUST, ha="left", arrowprops=dict(arrowstyle="->", lw=0.5, color=RUST),
                bbox=dict(fc="white", ec="none", pad=1))
    ax.legend(handles=[Line2D([], [], color=NAVY, marker="o", ls="", ms=3.5, label="H1 (same five)"),
                       Line2D([], [], color=RUST, marker="o", ls="", ms=3.5, label="G1"),
                       Line2D([], [], color=GOLD, marker="o", ls="", ms=3.5, label="R1")],
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="center right", fontsize=5.3, handletextpad=0.2)
    ax.set_xlabel("opening width (m)"); ax.set_ylabel("pass rate (%)"); ax.set_ylim(15, 110)
    ax.set_title("(c) pass rate by opening width", loc="left", x=-0.25)
    save(fig, MAIN + "c17_abs_width.pdf")

def mask_v3():
    fig, ax = plt.subplots(figsize=(3.3, 1.7))
    th = np.linspace(0, 90, 2000)
    W = S * np.abs(np.cos(np.deg2rad(th))) + DD * np.abs(np.sin(np.deg2rad(th)))
    xs = np.linspace(0.36, 1.0, 400)
    star = []
    for x in xs:
        idx = np.where((th >= 21.1) & (W <= x * S))[0]
        star.append(th[idx[0]] if len(idx) else 90)
    star = np.array(star)
    ax.axvspan(1.0, 2.1, color="#F4D58D", alpha=0.35, lw=0)
    ax.fill_between(xs, star, 90, color="#F4D58D", alpha=0.45, lw=0, label="passable: θ ≥ θ*(A/S)")
    ax.plot(xs, star, color="#7F6000", lw=1.0)
    mean = d.groupby("A").max_rotation_deg.mean()
    sd = d.groupby("A").max_rotation_deg.std() / np.sqrt(d.groupby("A").size())
    ax.fill_between(mean.index, (mean - 1.96 * sd), (mean + 1.96 * sd), color=NAVY, alpha=0.18, lw=0)
    ax.plot(mean.index, mean.values, "-o", color=NAVY, ms=2.6, lw=1.1, label="measured mean ± 95% CI")
    ax.axvline(1.0, color="#262626", lw=0.6, ls="--")
    ax.text(1.04, 6, "zero margin", fontsize=5.6, color="#262626", ha="left")
    ax.text(0.80, 22, "blocked", fontsize=6, color="#404040", ha="center")
    ax.text(1.55, 80, "frontal passage feasible", fontsize=6, color="#7F6000", ha="center")
    ax.set_xlim(2.05, 0.35); ax.set_ylim(0, 92)
    ax.set_xticks([2.0, 1.6, 1.2, 1.0, 0.8, 0.6, 0.4])
    ax.set_xlabel("A/S"); ax.set_ylabel("body rotation θ (°)")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="lower right", fontsize=5.6)
    save(fig, SUPP + "c47_mask.pdf")

if __name__ == "__main__":
    policy_violin(); body_v3(); mask_v3(); print("v3 written")
