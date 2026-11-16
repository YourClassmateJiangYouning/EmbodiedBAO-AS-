"""LaTeX tables for the supplement, generated from the raw exports."""
import os
import numpy as np, pandas as pd
D = "/mnt/user-data/uploads/"
OUT = os.path.join(os.path.dirname(__file__), "..", "sec", "supp")
os.makedirs(OUT, exist_ok=True)
L = {"deepseek-v4-flash-vision-exp": "DeepSeek-V4-Flash", "gemini-3-flash-preview": "Gemini-3-Flash",
     "mimo-v2.6-pro": "MiMo-v2.6-Pro", "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B-A3B",
     "claude-haiku-4-5-20251001": "Claude-Haiku-4.5", "gpt-5": "GPT-5", "gemini-2.5-flash": "Gemini-2.5-Flash",
     "glm-4.5v": "GLM-4.5V", "gpt-4o-mini": "GPT-4o-mini", "kimi/kimi-k2.6": "Kimi-K2.6", "kimi/kimi-k2.5": "Kimi-K2.5",
     "gpt-4.1": "GPT-4.1", "glm-4.6v": "GLM-4.6V", "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
     "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B"}
d = pd.read_csv(D + "02_S1逐回合6375行.csv"); d["passed"] = d.passed.astype(str) == "True"
d["A"] = d.a_s_ratio.round(1)

# Table S-config: calls per model (08)
c = pd.read_csv(D + "08_模型调用配置.csv")
rows = []
for _, r in c.sort_values("total_calls", ascending=False).iterrows():
    rows.append(f"{r.label} & \\texttt{{{r.api_model_name}}} & {r.origin} & {r.thinking_mode.replace("_", chr(92)+"_")} & {int(r.stage1_calls):,} & {int(r.stage2_calls):,} & {int(r.stage3_calls):,} & {int(r.total_calls):,} \\\\")
rows.append(f"\\midrule\nTotal & & & & {int(c.stage1_calls.sum()):,} & {int(c.stage2_calls.sum()):,} & {int(c.stage3_calls.sum()):,} & {int(c.total_calls.sum()):,} \\\\")
open(os.path.join(OUT, "tab_config.tex"), "w").write("\n".join(rows) + "\n")

# Table S-profile: per-model profile (03) with extra columns
p = pd.read_csv(D + "03_S1逐模型画像.csv").set_index("model")
rows = []
for m in sorted(L, key=lambda k: -p.loc[k, "narrow_pass_pct"] if k in p.index else 0):
    if m not in p.index: continue
    r = p.loc[m]
    rows.append(f"{L[m]} & {r.pass_pct:.1f} & {r.wide_pass_pct:.1f} & {r.narrow_pass_pct:.1f} & {r.look_down_step_pct:.2f} & {r.look_down_ep_pct:.1f} & {r.early_turn_pct_in_turns:.1f} & {r.collision_ep_pct:.1f} & {int(r.invalid_steps)} & {r.mean_steps:.1f} \\\\")
open(os.path.join(OUT, "tab_profile.tex"), "w").write("\n".join(rows) + "\n")

# Table S-stage2: per model memory transfer (04)
s2 = pd.read_csv(D + "04_S2_105对配对.csv")
for col in ["passed", "baseline_passed"]:
    s2[col] = s2[col].astype(str) == "True"
rows = []
g = s2.groupby("model")
for m, x in sorted(g, key=lambda kv: -kv[1].turn_delta_vs_baseline.mean()):
    rows.append(f"{L.get(m, m)} & {len(x)} & {x.baseline_passed.sum()} & {x.passed.sum()} & {x.baseline_turns.mean():.2f} & {x.turn_count.mean():.2f} & {x.turn_delta_vs_baseline.mean():+.2f} & {x.baseline_max_rotation_deg.mean():.1f} & {x.max_rotation_deg.mean():.1f} & {x.copy_lcs_fraction.mean():.2f} \\\\")
tot = s2
rows.append(f"\\midrule\nAll & {len(tot)} & {tot.baseline_passed.sum()} & {tot.passed.sum()} & {tot.baseline_turns.mean():.2f} & {tot.turn_count.mean():.2f} & {tot.turn_delta_vs_baseline.mean():+.2f} & {tot.baseline_max_rotation_deg.mean():.1f} & {tot.max_rotation_deg.mean():.1f} & {tot.copy_lcs_fraction.mean():.2f} \\\\")
open(os.path.join(OUT, "tab_stage2.tex"), "w").write("\n".join(rows) + "\n")

# Table S-stage3: per model x body (05)
s3 = pd.read_csv(D + "05_S3_100集体型.csv"); s3["passed"] = s3.passed.astype(str) == "True"
req = {"G1": {0.9: 58.6}, "R1": {0.9: 65.4}}
rows = []
for m in s3.model.unique():
    for rb in ["G1", "R1"]:
        x = s3[(s3.model == m) & (s3.robot == rb)]
        a9 = x[x.a_s_requested == 0.9]
        rq = req[rb][0.9]
        rows.append(f"{L[m]} & {rb} & {len(x)} & {x.passed.sum()} & {100*x.passed.mean():.0f} & {a9.max_rotation_deg.mean():.1f} & {100*(a9.max_rotation_deg>=rq).mean():.0f} \\\\")
x = s3
g1=x[(x.robot=='G1')&(x.a_s_requested==0.9)]; r1=x[(x.robot=='R1')&(x.a_s_requested==0.9)]
rows.append(f"\\midrule\nG1 total & G1 & {int((x.robot=='G1').sum())} & {int(x[x.robot=='G1'].passed.sum())} & {100*x[x.robot=='G1'].passed.mean():.0f} & {g1.max_rotation_deg.mean():.1f} & {100*(g1.max_rotation_deg>=req['G1'][0.9]).mean():.0f} \\\\")
rows.append(f"R1 total & R1 & {int((x.robot=='R1').sum())} & {int(x[x.robot=='R1'].passed.sum())} & {100*x[x.robot=='R1'].passed.mean():.0f} & {r1.max_rotation_deg.mean():.1f} & {100*(r1.max_rotation_deg>=req['R1'][0.9]).mean():.0f} \\\\")
open(os.path.join(OUT, "tab_stage3.tex"), "w").write("\n".join(rows) + "\n")

# Table S-sensitivity counts for scenes and invalid exclusion
v = d[d.invalid_responses == 0]
txt = (f"{v[v.A>=1].passed.mean()*100:.1f} & {v[v.A<1].passed.mean()*100:.1f} & {len(v)} \\\\")
open(os.path.join(OUT, "tab_invalid.tex"), "w").write(txt + "\n")
print("tables written:", sorted(os.listdir(OUT)))
print(open(os.path.join(OUT, "tab_invalid.tex")).read())

# Sensitivity of the rotation onset to the turning criterion (recomputed)
def onset(th):
    pr = d.groupby("A").max_rotation_deg.apply(lambda x: (x >= th).mean())
    levels = sorted(pr.index, reverse=True)
    for a, b in zip(levels[:-1], levels[1:]):
        if pr[a] < 0.5 <= pr[b]:
            return a + (b - a) * (0.5 - pr[a]) / (pr[b] - pr[a]), pr
    return None, pr
rows = []
for th, lab in [(15, r"$\theta_{\max}\geq15^\circ$ (one turn action)"), (30, r"$\theta_{\max}\geq30^\circ$ (two turn actions; main)"), (45, r"$\theta_{\max}\geq45^\circ$")]:
    v, pr = onset(th)
    rows.append(f"{lab} & {('%.3f' % v) if v is not None else 'no crossing of 0.5'} \\\\")
# pass-rate 50% point
pp = d.groupby("A").passed.mean()
lv = sorted(pp.index, reverse=True)
for a, b in zip(lv[:-1], lv[1:]):
    if pp[a] >= 0.5 > pp[b]:
        v = a + (b - a) * (0.5 - pp[a]) / (pp[b] - pp[a]); break
rows.append(f"Pass rate $=50\\%$ & {v:.3f} \\\\")
open(os.path.join(OUT, "tab_sens.tex"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
