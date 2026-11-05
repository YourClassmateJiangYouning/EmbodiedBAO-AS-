# Stage 3 超时与未完成情况整理

数据来源：`results/**/stage3-*/episode_*.json`，共 100 条。

## 总览

| 指标 | 数量 |
|---|---:|
| 总 episode | 100 |
| passed | 76 |
| failed | 24 |
| invalid = 0 的干净 episode | 71 |
| 至少出现一次超时的 episode | 29 |
| 失败且由超时导致 | 17 |
| 失败但没有超时 | 7 |
| 通过但过程中有超时 | 12 |

> “失败且由超时导致”定义为：`passed=False`，且至少一个 invalid step 的 reasoning 明确包含 `timed out` / `urlopen error`。

## 按模型统计

| 模型 | n | passed | timeout-affected | failed + timeout |
|---|---:|---:|---:|---:|
| claude-haiku-4-5-20251001 | 20 | 15 | 3 | 2 |
| deepseek-v4-flash-vision-exp | 20 | 18 | 9 | 2 |
| gemini-3-flash-preview | 20 | 14 | 6 | 6 |
| gpt-5 | 20 | 14 | 4 | 3 |
| mimo-v2.6-pro | 20 | 15 | 7 | 4 |

## 按机器人统计

| 机器人 | n | passed | timeout-affected | failed + timeout |
|---|---:|---:|---:|---:|
| G1 | 50 | 36 | 14 | 12 |
| R1 | 50 | 40 | 15 | 5 |

## 失败且由超时导致的 episode

| Pair | 模型 | 机器人 | 场景 | A/S | steps | invalid | timeout invalid | 结束原因 |
|---|---|---|---|---:|---:|---:|---:|---|
| S3F-010 | deepseek-v4-flash-vision-exp | G1 | stage1.5 | 0.9 | 30 | 19 | 19 | max_steps |
| S3F-020 | deepseek-v4-flash-vision-exp | R1 | stage1.5 | 0.9 | 30 | 21 | 21 | max_steps |
| S3F-027 | gemini-3-flash-preview | G1 | stage1.4 | 1.1 | 30 | 25 | 24 | max_steps |
| S3F-028 | gemini-3-flash-preview | G1 | stage1.4 | 1.0 | 30 | 19 | 19 | max_steps |
| S3F-029 | gemini-3-flash-preview | G1 | stage1.5 | 1.1 | 30 | 28 | 27 | max_steps |
| S3F-030 | gemini-3-flash-preview | G1 | stage1.5 | 0.9 | 30 | 25 | 25 | max_steps |
| S3F-031 | gemini-3-flash-preview | R1 | stage1.1 | 1.3 | 30 | 15 | 15 | max_steps |
| S3F-032 | gemini-3-flash-preview | R1 | stage1.1 | 1.1 | 30 | 18 | 18 | max_steps |
| S3F-049 | mimo-v2.6-pro | G1 | stage1.5 | 1.1 | 30 | 19 | 19 | max_steps |
| S3F-050 | mimo-v2.6-pro | G1 | stage1.5 | 0.9 | 30 | 29 | 29 | max_steps |
| S3F-053 | mimo-v2.6-pro | R1 | stage1.2 | 1.3 | 30 | 12 | 12 | max_steps |
| S3F-054 | mimo-v2.6-pro | R1 | stage1.2 | 0.9 | 30 | 14 | 14 | max_steps |
| S3F-069 | claude-haiku-4-5-20251001 | G1 | stage1.5 | 1.1 | 30 | 24 | 24 | max_steps |
| S3F-070 | claude-haiku-4-5-20251001 | G1 | stage1.5 | 0.9 | 30 | 3 | 3 | max_steps |
| S3F-087 | gpt-5 | G1 | stage1.4 | 1.1 | 30 | 21 | 21 | max_steps |
| S3F-088 | gpt-5 | G1 | stage1.4 | 1.0 | 30 | 24 | 23 | max_steps |
| S3F-090 | gpt-5 | G1 | stage1.5 | 0.9 | 30 | 15 | 15 | max_steps |

## 失败但没有超时记录的 episode

| Pair | 模型 | 机器人 | 场景 | A/S | steps | invalid | 结束原因 |
|---|---|---|---|---:|---:|---:|---|
| S3F-060 | mimo-v2.6-pro | R1 | stage1.5 | 0.9 | 30 | 0 | max_steps |
| S3F-074 | claude-haiku-4-5-20251001 | R1 | stage1.2 | 0.9 | 30 | 0 | max_steps |
| S3F-080 | claude-haiku-4-5-20251001 | R1 | stage1.5 | 0.9 | 30 | 0 | max_steps |
| S3F-094 | gpt-5 | R1 | stage1.2 | 0.9 | 30 | 0 | max_steps |
| S3F-100 | gpt-5 | R1 | stage1.5 | 0.9 | 30 | 0 | max_steps |
| S3P-033 | claude-haiku-4-5-20251001 | G1 | stage1.2 | 0.9 | 30 | 0 | max_steps |
| S3P-043 | gpt-5 | G1 | stage1.2 | 0.9 | 30 | 0 | max_steps |

## 通过但过程中出现超时的 episode

| Pair | 模型 | 机器人 | 场景 | A/S | timeout invalid |
|---|---|---|---|---:|---:|
| S3F-009 | deepseek-v4-flash-vision-exp | G1 | stage1.5 | 1.1 | 7 |
| S3F-011 | deepseek-v4-flash-vision-exp | R1 | stage1.1 | 1.3 | 2 |
| S3F-013 | deepseek-v4-flash-vision-exp | R1 | stage1.2 | 1.3 | 8 |
| S3F-016 | deepseek-v4-flash-vision-exp | R1 | stage1.3 | 1.0 | 4 |
| S3F-017 | deepseek-v4-flash-vision-exp | R1 | stage1.4 | 1.1 | 14 |
| S3F-018 | deepseek-v4-flash-vision-exp | R1 | stage1.4 | 1.0 | 6 |
| S3F-019 | deepseek-v4-flash-vision-exp | R1 | stage1.5 | 1.1 | 9 |
| S3F-047 | mimo-v2.6-pro | G1 | stage1.4 | 1.1 | 2 |
| S3F-051 | mimo-v2.6-pro | R1 | stage1.1 | 1.3 | 16 |
| S3F-052 | mimo-v2.6-pro | R1 | stage1.1 | 1.1 | 8 |
| S3F-072 | claude-haiku-4-5-20251001 | R1 | stage1.1 | 1.1 | 1 |
| S3F-091 | gpt-5 | R1 | stage1.1 | 1.3 | 12 |
