# v8 语料表

论文里每一个数字都来自这里的表。原始工作目录 `数据图片与教程/最终数据/` 是 1.5 GB（大部分是场景照片
和渲染帧），不能进版本库；它里面的表本身很小，所以复制到这里。**除了对三个逐步骤表做 gzip 之外没有任何
变换**，克隆后解压即可得到论文计算所用的同一批字节。

来源运行：`stage1_final_20261027_154706`（Stage 1）、`stage2_final_20261027_154234`（Stage 2）、
`stage3_final100_20261101`（Stage 3）。

## 内容

| 文件 | 行数 | 是什么 |
| :-- | ---: | :-- |
| `S1_episodes.csv` | 6,375 | Stage 1 逐 episode：15 模型 × 17 档 A/S × 5 标志物 × 5 场景 |
| `S1_steps.csv.gz` | — | Stage 1 逐步骤动作与状态（159 MB，gzip 后 14 MB） |
| `S1_model_summary.csv` | 15 | 每模型汇总 |
| `S1_model_scene_metrics.csv` | 90 | 模型 × 场景 |
| `S1_per_level_pass_rates.csv` | 255 | 模型 × 档位的通过率（长表） |
| `S1_per_level_pass_pct_wide.csv` | 15 | 同上，档位为列（画断崖用这张） |
| `S1_rotation_profile.csv` | 15 | 每模型宽档/窄档的平均转角与差值 |
| `S1_run_arguments.csv` | — | 每模型调用 API 时传的参数 |
| `S1_sfinal_nps_ranked.csv` | — | 窄档通过率排序 |
| `S2_episodes.csv` | 105 | 记忆实验逐 episode |
| `S2_pairs.csv` | — | 105 组配对的基线/记忆差 |
| `S2_steps.csv.gz` | — | 记忆实验逐步骤 |
| `S2_memory_cards_blind.csv` | — | 记忆卡（盲化版，实际喂给模型的） |
| `S2_memory_cards_full.csv` | — | 记忆卡（完整版，含被盲化掉的句子） |
| `S2_model_summary.csv` / `S2_scene_model_summary.csv` | — | 汇总 |
| `S2_test_plan.csv` / `S2_failed_pairs.csv` | — | 测试计划与失败配对 |
| `S3_episodes.csv` | 100 | 换身体实验：H1 / G1 / R1 各若干，5 个模型 |
| `S3_steps.csv.gz` / `S3_failures.csv` / `S3_input_plan.csv` / `S3_timeout_audit.csv` | — | 同上 |
| `00_*.csv` | — | 索引、模型名映射、源文件校验和 |

## 已核对的规模

```
S1_episodes.csv   6,375 行   15 个模型
S2_episodes.csv     105 行    7 个模型
S3_episodes.csv     100 行    5 个模型
```

Stage 1 的 15 个模型：Qwen3-VL-8B/32B/30B-A3B、Claude-Haiku-4.5、DeepSeek-V4-Flash-vision-exp、
Gemini-2.5-Flash、Gemini-3-Flash、GLM-4.5V、GLM-4.6V、GPT-4.1、GPT-4o-mini、GPT-5、Kimi-K2.5、
Kimi-K2.6、MiMo-v2.6-Pro。

## 注意

`lab_logs/` 里另有一份 **v7** 数据（`embodiedbao_v7_episodes.csv`，11 个模型、660 条），那是更早的一次
扫描，模型名（DeepSeek-V4.1-Flash、Gemini-2.5-Pro、Claude-Sonnet-4-6）与 v8 不同，**不是论文正文用的
这份**。两者不要混用。
