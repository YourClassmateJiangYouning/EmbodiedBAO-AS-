"""v4 palette refresh for main Figs 2, 4, 7, 8 (data from the v8 exports)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
U = "/mnt/user-data/uploads/"; MAIN = "../fig/main/"
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E9E4DA", "grid.linewidth": 0.5, "grid.linestyle": "--", "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
# palette: deep navy, teal, sand, apricot, coral, plum
NAVY, TEAL, SAND, APRI, CORAL, PLUM, INK = "#1D3557", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#8E6BB0", "#262626"
GREYV = "#B8B8B8"
TOP2 = ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview"]
LAB = {"deepseek-v4-flash-vision-exp": "DeepSeek-V4", "gemini-3-flash-preview": "Gemini-3", "mimo-v2.6-pro": "MiMo-v2.6",
       "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B", "claude-haiku-4-5-20251001": "Claude-Haiku", "gpt-5": "GPT-5",
       "gemini-2.5-flash": "Gemini-2.5", "glm-4.5v": "GLM-4.5V", "gpt-4o-mini": "GPT-4o-mini", "kimi/kimi-k2.6": "Kimi-K2.6",
       "kimi/kimi-k2.5": "Kimi-K2.5", "gpt-4.1": "GPT-4.1", "glm-4.6v": "GLM-4.6V", "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
       "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B"}
S, DD = 0.570, 0.220
d = pd.read_csv(U + "02_S1逐回合6375行.csv"); d["passed"] = d.passed.astype(str) == "True"; d["A"] = d.a_s_ratio.round(1)
LEV = sorted(d.A.unique(), reverse=True)
wide_c = LinearSegmentedColormap.from_list("ink", ["#FBF6EC", SAND, APRI, CORAL, "#7B2D26"])
cool_c = LinearSegmentedColormap.from_list("sea", ["#F4F9F8", "#A8DADC", TEAL, NAVY])

def fig2_surface():
    order = d.groupby("model").passed.mean().sort_values(ascending=False).index.tolist()
    Z = d.groupby(["model", "A"]).max_rotation_deg.mean().unstack().loc[order, LEV].values
    X, Y = np.meshgrid(np.arange(len(LEV)), np.arange(len(order)))
    fig = plt.figure(figsize=(3.9, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(0, 90)
    surf = ax.plot_surface(X, Y, Z, cmap=wide_c, norm=norm, linewidth=0.15, edgecolor="white", alpha=0.97, antialiased=True)
    ax.contour(X, Y, Z, zdir="z", offset=0, levels=8, cmap=cool_c, linewidths=0.6)
    ax.set_xticks(range(0, len(LEV), 3)); ax.set_xticklabels([f"{LEV[i]:.1f}" for i in range(0, len(LEV), 3)], fontsize=5.5)
    ax.set_yticks(range(0, len(order), 3)); ax.set_yticklabels([LAB[order[i]] for i in range(0, len(order), 3)], fontsize=5.2)
    ax.set_zticks([0, 30, 60, 90]); ax.tick_params(labelsize=5.5, pad=0)
    ax.set_xlabel("A/S", labelpad=-2); ax.set_zlabel("mean max rotation (°)", labelpad=-1)
    ax.view_init(elev=28, azim=-58)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor("#D9D4C7")
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, aspect=12); cb.set_label("°", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    fig.savefig(MAIN + "fancy_surface.pdf"); plt.close(fig)

def episode_trace(row):
    acts = str(row.action_sequence).split(",")
    x, rot, z = 0.5, 0.0, 0.0
    xs, rs, ts, turns = [x], [rot], [0], []
    for k, a in enumerate(acts, 1):
        if a == "forward": x += 0.75
        elif a == "backward": x -= 0.75
        elif a == "left": z += 0.75
        elif a == "right": z -= 0.75
        elif a == "turn_left": rot += 15; turns.append(k)
        elif a == "turn_right": rot -= 15; turns.append(k)
        xs.append(x); rs.append(abs(rot)); ts.append(k)
    return np.array(ts), np.array(xs), np.array(rs), turns

def pick():
    sel = {}
    p = d[(d.passed) & (d.A.isin([0.6, 0.7, 0.8])) & (d.first_turn_x_m <= 5) & (d.n_turns > 0) & (d.max_rotation_deg >= 60)]
    sel["early pass"] = p.sort_values("total_steps").iloc[len(p)//2]
    f = d[(~d.passed) & (d.A == 0.6) & (d.first_turn_x_m >= 7) & (d.wall_collisions >= 3)]
    sel["reactive failure"] = f.iloc[len(f)//2]
    i = d[(~d.passed) & (d.A == 0.9) & (d.max_rotation_deg.between(30, 50))]
    sel["insufficient rotation"] = i.iloc[len(i)//2]
    w = d[(d.passed) & (d.A == 2.0) & (d.max_rotation_deg >= 60)]
    sel["wide-gap pass"] = w.iloc[len(w)//2]
    return sel

def fig4_traj():
    sel = pick()
    fig = plt.figure(figsize=(3.9, 3.5))
    for k, (name, row) in enumerate(sel.items()):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ts, xs, rs, turns = episode_trace(row)
        ok = bool(row.passed)
        pts = np.array([ts, xs, rs]).T
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        cols = cool_c(np.linspace(0.25, 1, len(segs))) if ok else wide_c(np.linspace(0.2, 0.95, len(segs)))
        ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=1.6))
        ax.scatter(ts[turns] if turns else [], xs[turns] if turns else [], rs[turns] if turns else [], s=5, color=APRI, depthshade=False)
        ax.scatter([ts[-1]], [xs[-1]], [rs[-1]], s=16, color=TEAL if ok else CORAL, depthshade=False, edgecolor="white", lw=0.4)
        ax.set_xlim(0, 32); ax.set_ylim(0, 9); ax.set_zlim(0, 100)
        ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 4, 8]); ax.set_zticks([0, 45, 90])
        ax.tick_params(labelsize=4.8, pad=-2)
        ax.set_xlabel("step", fontsize=5.2, labelpad=-6); ax.set_ylabel("x (m)", fontsize=5.2, labelpad=-6); ax.set_zlabel("rot (°)", fontsize=5.2, labelpad=-6)
        ax.view_init(elev=22, azim=-55)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor("#D9D4C7")
        verdict = "pass" if ok else "fail"
        short = {"early pass": "early pass", "reactive failure": "reactive fail",
                 "insufficient rotation": "under-rotation", "wide-gap pass": "wide-gap pass"}[name]
        ax.set_title(f"{short}\nA/S {row.A:.1f} · {int(row.total_steps)} steps", fontsize=5.8, pad=-4,
                     color=TEAL if ok else CORAL)
    fig.subplots_adjust(wspace=0.18, hspace=0.42, left=0.02, right=0.98, top=0.95, bottom=0.02)
    fig.savefig(MAIN + "fancy_traj.pdf"); plt.close(fig)
    return {k: (r.A, bool(r.passed), int(r.total_steps), int(r.n_turns)) for k, r in sel.items()}

from mpl_toolkits.mplot3d.art3d import Line3DCollection
def fig7_violin():
    W = d[d.A >= 1.0 - 1e-9]; N = d[d.A < 1.0]
    nar = N.groupby("model").passed.mean() * 100
    wr = W.groupby("model").max_rotation_deg.mean()
    order = list(nar.sort_values(ascending=True).index)
    fig, ax = plt.subplots(figsize=(3.3, 3.35))
    ax.axvspan(104, 136, color="#F4F1EA", lw=0, zorder=0)
    for i, m in enumerate(order):
        vals = W[W.model == m].max_rotation_deg.values
        c = CORAL if m in TOP2 else (SAND if m == "mimo-v2.6-pro" else (TEAL if nar[m] >= 10 else GREYV))
        if np.unique(vals).size > 1:
            parts = ax.violinplot([vals], positions=[i], vert=False, widths=0.86, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(c); pc.set_alpha(0.75); pc.set_edgecolor(INK); pc.set_linewidth(0.35)
        ax.scatter([vals.mean()], [i], s=9, color=NAVY, zorder=4, edgecolor="white", lw=0.4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[m] for m in order], fontsize=5.8)
    ax.set_xlim(-6, 135); ax.set_xticks([0, 30, 60, 90]); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("max body rotation per episode, wide bins A/S ≥ 1.0 (°)", fontsize=6.6)
    ax.grid(axis="y", visible=False)
    for i, m in enumerate(order):
        ax.text(103, i, f"{wr[m]:.1f}°", va="center", fontsize=5.4, color=NAVY)
        ax.text(120, i, f"{nar[m]:.0f}%", va="center", fontsize=5.4, weight="bold" if m in TOP2 else "normal",
                color=CORAL if m in TOP2 else (INK if nar[m] >= 10 else "#7A7A7A"))
    ax.text(103, len(order) - 0.3, "mean", fontsize=5.3, color=NAVY, weight="bold")
    ax.text(120, len(order) - 0.3, "narrow pass", fontsize=5.1, color=NAVY, weight="bold")
    ax.legend(handles=[Line2D([], [], color=CORAL, lw=5, alpha=0.75, label="top two (pass narrow gaps)"),
                       Line2D([], [], color=SAND, lw=5, alpha=0.75, label="MiMo-v2.6"),
                       Line2D([], [], color=TEAL, lw=5, alpha=0.75, label="other ≥10%"),
                       Line2D([], [], color=GREYV, lw=5, alpha=0.75, label="other <10%")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=5.0, handlelength=1.2, columnspacing=0.8)
    ax.set_title("Wide-bin rotation vs. narrow-gap success", loc="left", x=-0.35, fontsize=7.2)
    fig.savefig(MAIN + "c07_strategy.pdf"); plt.close(fig)

def fig8_body():
    s3 = pd.read_csv(U + "05_S3_100集体型.csv"); s3["passed"] = s3.passed.astype(str) == "True"
    five = ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview", "mimo-v2.6-pro", "claude-haiku-4-5-20251001", "gpt-5"]
    h1 = d[d.model.isin(five)]; lv = [1.3, 1.1, 1.0, 0.9]
    C = {"H1": NAVY, "G1": CORAL, "R1": "#E0A526"}
    fig, ax = plt.subplots(figsize=(2.3, 2.35))
    ax.axvspan(1.10, 1.30, color="#FFF0C2", lw=0, zorder=0)
    hp = h1.groupby("A").passed.mean().reindex(lv) * 100
    ax.plot(lv, hp.values, "-o", color=C["H1"], ms=3.8, lw=1.6, label="H1 (n=125/bin)", zorder=3)
    for rb in ["G1", "R1"]:
        sub = s3[s3.robot == rb].groupby("a_s_requested").passed.mean().reindex(lv) * 100
        ax.plot(lv, sub.values, "-o", color=C[rb], ms=3.8, lw=1.6, label=f"{rb} (n=10/bin)", zorder=3)
    ax.set_xlim(1.35, 0.85); ax.set_xticks(lv); ax.set_ylim(40, 104)
    ax.set_xlabel("A/S"); ax.set_ylabel("pass rate (%)")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="lower left", fontsize=5.4, handlelength=1.4)
    ax.set_title("(a) Same A/S threshold", loc="left", x=-0.25)
    fig.savefig(MAIN + "c15_body_pass.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(2.3, 2.35))
    req = {"H1": 54.0, "G1": 58.6, "R1": 65.4}
    data = {"H1": h1[(h1.A == 0.9)].max_rotation_deg.values,
            "G1": s3[(s3.robot == "G1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values,
            "R1": s3[(s3.robot == "R1") & (s3.a_s_requested == 0.9)].max_rotation_deg.values}
    rng = np.random.default_rng(0)
    for i, rb in enumerate(["H1", "G1", "R1"]):
        v = data[rb]
        ax.boxplot([v], positions=[i], widths=0.44, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=C[rb], lw=0.7), capprops=dict(lw=0),
                   boxprops=dict(color=C[rb], lw=0.8, facecolor=C[rb] + "33"))
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=7, color=C[rb], alpha=0.85, lw=0, zorder=3)
        ax.plot([i - 0.44, i + 0.44], [req[rb]] * 2, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
        ax.text(i, -17, f"req. {req[rb]:.1f}°\nmet {(v >= req[rb]).mean()*100:.0f}%", ha="center", va="top", fontsize=5.5, color=INK)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["H1 (0.570 m)", "G1 (0.450 m)", "R1 (0.357 m)"], fontsize=5.6)
    ax.tick_params(axis="x", pad=2); ax.set_xlim(-0.6, 2.6); ax.set_ylim(-2, 100)
    ax.set_ylabel("max rotation at A/S 0.9 (°)"); ax.grid(axis="x", visible=False)
    ax.set_title("(b) largest turn at A/S=0.9", loc="left", x=-0.25)
    fig.savefig(MAIN + "c16_body_rot.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(2.3, 2.35))
    ax.axvspan(0.44, 0.475, color="#FFE3D3", lw=0, zorder=0)
    h = h1.groupby("channel_width_m").passed.mean() * 100
    ax.scatter(h.index, h.values, s=12, color=NAVY, zorder=3, edgecolor="white", lw=0.3)
    for rb in ["G1", "R1"]:
        g = s3[s3.robot == rb].groupby("channel_width_m").passed.mean() * 100
        ax.scatter(g.index, g.values, s=18, color=C[rb], edgecolor="white", lw=0.4, zorder=4)
    h46 = h.loc[min(h.index, key=lambda w: abs(w - 0.456))]
    ax.annotate(f"≈0.46 m\nG1, R1: 100%\nH1 (same five): {h46:.0f}%", xy=(0.456, h46), xytext=(0.62, 22),
                fontsize=5.5, color=CORAL, ha="left", arrowprops=dict(arrowstyle="->", lw=0.6, color=CORAL),
                bbox=dict(fc="white", ec="none", pad=1))
    ax.legend(handles=[Line2D([], [], color=NAVY, marker="o", ls="", ms=3.8, label="H1 (same five)"),
                       Line2D([], [], color=C["G1"], marker="o", ls="", ms=3.8, label="G1"),
                       Line2D([], [], color=C["R1"], marker="o", ls="", ms=3.8, label="R1")],
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="center right", fontsize=5.3, handletextpad=0.2)
    ax.set_xlabel("opening width (m)"); ax.set_ylabel("pass rate (%)"); ax.set_ylim(15, 110)
    ax.set_title("(c) pass rate by opening width", loc="left", x=-0.25)
    fig.savefig(MAIN + "c17_abs_width.pdf"); plt.close(fig)

if __name__ == "__main__":
    fig2_surface(); info = fig4_traj(); fig7_violin(); fig8_body()
    print("v4 written"); print(info)
