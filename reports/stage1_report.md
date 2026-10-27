# Stage 1 data report

Completed models: 15 (425 episodes each).

S0/Sfinal and NPS are recomputed from the formulas recovered from the prior session; the earlier recorded values are retained as `stage1_score_prior`/`nps_prior` for comparison. The old recorded values differ slightly because the earlier pass used a slightly different component definition; the new columns are the consistent recomputation used for the new models.

Narrow = levels 11-16 (A/S 0.9-0.4). Wide = levels 0-10 (A/S 2.0-1.0).

## Recovered formulas

Stage-1 S0 (per-episode quality, difficulty weighted):

```
over  = max(0, theta_max - theta_req) / 15
under = max(0, theta_req - theta_max) / 15
extra = max(0, n_turn - min_turns(W))
excess = max(0, total_steps - optimal_steps(W))
q_rot   = exp(-0.20*over - 0.03*over^2 - 0.20*under)
q_turn  = exp(-0.12*extra)
q_look  = 1.00 timely, 0.95 late, 0.90 none, 0.80 >2 looks
q_eff   = exp(-0.02*excess/optimal)
q_col   = exp(-0.05*wall_collisions)
s_i     = 100 * passed * q_rot*q_turn*q_look*q_eff*q_col
w_l     = 0.05 + 1.0*(theta_req/90)
S0      = sum_l w_l * mean(s_i in level l) / sum_l w_l
```

Model gates: if wide-turn rate > 0.10 or narrow look rate < 0.10, Sfinal=min(S0,20); elif wide-turn > 0.05 or look < 0.25, Sfinal=min(S0,50).

NPS (narrow-only; not Net Promoter Score):

```
P_n = narrow passes / 150
q_rot   = exp(-0.20*over - 0.03*over^2)
q_extra = exp(-0.12*max(0,n_turn-min_turns))
q_eff   = min(1, optimal_steps/total_steps)
q_look  = 1.00 timely, 0.95 late, 0.90 none, 0.80 >2 looks
Q_n     = mean(q_rot*q_extra*q_eff*q_look) over passed narrow
O_n     = fraction of passed narrow with over > 0
NPS     = 100 * P_n * Q_n * (1 - O_n)
```

Eligibility: n_narrow_pass >= 10 and NPS >= 20.

## Completed models, combined (ALL scenes)

| model | pass% | wide% | narrow | narrow% | turns% | look% | invalid | S0 | Sfinal | NPS | Stage2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen-Qwen3-VL-30B-A3B-Instruct | 68.5 | 99.3 | 18/150 | 12.0 | 28.0 | 12.7 | 3 | 14.5 | 14.5 | 3.3 |  |
| Qwen-Qwen3-VL-32B-Instruct | 59.8 | 92.4 | 0/150 | 0.0 | 20.5 | 34.4 | 8 | 7.8 | 7.8 | 0.0 |  |
| Qwen-Qwen3-VL-8B-Instruct | 60.9 | 93.1 | 3/150 | 2.0 | 34.6 | 37.9 | 21 | 9.1 | 9.1 | 0.6 |  |
| claude-haiku-4-5-20251001 | 68.5 | 97.5 | 23/150 | 15.3 | 36.9 | 24.7 | 0 | 16.6 | 16.6 | 5.4 |  |
| deepseek-v4-flash-vision-exp | 99.8 | 100.0 | 149/150 | 99.3 | 64.7 | 14.6 | 15 | 72.9 | 20.0 | 37.3 | YES |
| gemini-2.5-flash | 67.3 | 97.5 | 18/150 | 12.0 | 32.0 | 23.3 | 1 | 15.6 | 15.6 | 4.5 |  |
| gemini-3-flash-preview | 98.8 | 100.0 | 145/150 | 96.7 | 96.9 | 99.8 | 4 | 74.3 | 20.0 | 54.2 | YES |
| glm-4.5v | 65.9 | 100.0 | 5/150 | 3.3 | 45.9 | 52.7 | 0 | 10.2 | 10.2 | 1.5 |  |
| glm-4.6v | 63.8 | 97.8 | 2/150 | 1.3 | 15.1 | 15.5 | 24 | 9.4 | 9.4 | 0.4 |  |
| gpt-4.1 | 64.0 | 95.3 | 10/150 | 6.7 | 55.1 | 71.5 | 56 | 10.5 | 10.5 | 2.4 |  |
| gpt-4o-mini | 65.2 | 100.0 | 2/150 | 1.3 | 20.7 | 0.0 | 11 | 9.2 | 9.2 | 0.0 |  |
| gpt-5 | 68.5 | 100.0 | 16/150 | 10.7 | 21.9 | 5.9 | 0 | 15.4 | 15.4 | 6.5 |  |
| kimi-kimi-k2.5 | 64.9 | 100.0 | 1/150 | 0.7 | 31.5 | 45.2 | 38 | 9.3 | 9.3 | 0.5 |  |
| kimi-kimi-k2.6 | 65.2 | 100.0 | 2/150 | 1.3 | 34.6 | 38.8 | 0 | 9.1 | 9.1 | 0.0 |  |
| mimo-v2.6-pro | 86.1 | 100.0 | 91/150 | 60.7 | 44.5 | 70.6 | 22 | 51.0 | 20.0 | 28.3 | YES |

## Five scenes per model

The full model x scene table is in `stage1_model_scene_metrics.csv` (5 scene rows + ALL).

## Stage 2

Stage 2 is the new single-memory/wide-A/S-2.0 experiment. The old 12-attempt/note scenario is archived and is not used.

- design: `STAGE2_DESIGN.md`
- memory cards: `stage2_memory_cards_top7_15_blind.jsonl`
- test plan: `stage2_top7_15_test_plan.csv`
- schema: `stage2_memory_schema.md`

