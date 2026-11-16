"""Figure 6 (memory experiment, turns at A/S=2.0), redrawn taller for the figure layout.
Values are the per-model mean turns from Table S4 (no memory / one own narrow success in memory).
Same data as the previous fig/main/c13a_turns.pdf; only the aspect ratio changes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix",
                     "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

MODELS = ["MiMo-v2.6-Pro", "DeepSeek-V4-Flash", "Gemini-3-Flash", "Gemini-2.5-Flash",
          "Qwen3-VL-30B-A3B", "Claude-Haiku-4.5", "GPT-5"]
BASE = [0.00, 1.20, 3.07, 0.00, 0.00, 0.00, 0.00]
MEM = [5.27, 5.60, 5.27, 1.60, 0.60, 0.27, 0.00]
GREY, BLUE = "#BFBFBF", "#2F5597"

fig, ax = plt.subplots(figsize=(3.22, 4.67))
y = list(range(len(MODELS)))[::-1]
h = 0.36
ax.barh([v + h / 2 for v in y], BASE, height=h, color=GREY, label="without memory")
ax.barh([v - h / 2 for v in y], MEM, height=h, color=BLUE, label="with memory")
for yy, b, m in zip(y, BASE, MEM):
    ax.text(b + 0.08, yy + h / 2, f"{b:.1f}", va="center", fontsize=6.5, color="#404040")
    ax.text(m + 0.08, yy - h / 2, f"{m:.1f}", va="center", fontsize=6.5, color="#404040")
ax.set_yticks(y)
ax.set_yticklabels(MODELS, fontsize=6.8)
ax.set_xlim(0, 7)
ax.set_xticks([0, 2, 4, 6])
ax.set_xlabel(r"turns at $A/S{=}2.0$ (mean per episode)", fontsize=7)
ax.tick_params(length=2, width=0.6, labelsize=6.8)
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.3)
fig.savefig("fig/main/c13a_turns_tall.pdf")
print("written")
