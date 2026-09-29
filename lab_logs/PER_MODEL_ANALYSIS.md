# Stage 1 逐模型分析（自动生成，勿手改）

数据：`lab_logs/bao_v7_all.tgz`（660 集 = 11 模型 × 12 档 A/S × 5 次）。
复算：`python lab_logs/model_report.py`（本文件由它生成，所有数字现算）。

指标说明：**turn%** 该档中至少转身一次的集占比；**side%** 该模型在该档的通过里以 45°–135° 侧身姿态完成的占比；**excess** = 实际步数 − 该宽度最优步数；**look%** 至少低头看过自己一次的集占比；**pass|look / pass|nolook** 低头过的集/没低头的集的通过率。

---

## 跨模型总表

| 模型 | 类型 | 通过率 | A/S 1.0+0.9 | 转身集 | 无需转身处仍转 | 首次转身x | 平均最大转角 | 平均步数 | excess | 低头率 | pass\|look | pass\|nolook |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-sonnet-4-6 | 按需转身型 | 95% | 7/10 | 40% | 35% | 2.31 m | 12° | 13.7 | 2.4 | 42% | 88% | 100% |
| gemini-2.5-flash | 直立派 | 93% | 6/10 | 7% | 0% | 7.06 m | 4° | 12.6 | 1.3 | 8% | 60% | 96% |
| deepseek-v4.1-flash | 按需转身型 | 92% | 5/10 | 18% | 11% | 3.77 m | 4° | 13.3 | 2.0 | 53% | 84% | 100% |
| glm-4.6v | 直立派 | 92% | 6/10 | 8% | 4% | 5.00 m | 6° | 13.1 | 1.7 | 10% | 50% | 96% |
| gpt-4o | 按需转身型 | 92% | 6/10 | 42% | 36% | 1.64 m | 15° | 15.0 | 3.7 | 70% | 88% | 100% |
| gpt-4o-mini | 直立派 | 92% | 5/10 | 7% | 0% | 7.25 m | 4° | 12.6 | 1.2 | 0% | 0% | 92% |
| gemini-2.5-pro | 固定策略型 | 90% | 5/10 | 98% | 98% | 1.99 m | 51° | 17.9 | 6.6 | 95% | 89% | 100% |
| gpt-4.1 | 固定策略型 | 88% | 3/10 | 77% | 75% | 1.71 m | 12° | 15.4 | 4.1 | 90% | 87% | 100% |
| qwen3-vl-235b-a22b-instruct | 固定策略型 | 78% | 3/10 | 77% | 75% | 1.82 m | 22° | 17.8 | 6.4 | 83% | 76% | 90% |
| qwen-vl-max | 半按需型 | 77% | 4/10 | 50% | 49% | 2.52 m | 14° | 16.7 | 5.3 | 55% | 58% | 100% |
| qwen3-vl-32b-instruct | 半按需型 | 58% | 2/10 | 45% | 45% | 1.22 m | 10° | 20.6 | 9.3 | 70% | 48% | 83% |

---

## claude-sonnet-4-6

**类型：按需转身型**。会随宽度改变是否转身（宽档转身率仅 35%，首次转身平均在 x = 2.31 m），是几种类型里最接近「判断」而非「固定策略」的一种；两档合计通过 7/10。

**总体**：60 集，通过率 **95%**；两个需要转身的档（A/S 1.0 与 0.9）7/10。转身集 40%，首次转身平均在 x = 2.31 m；**在不需要转身的宽度上仍转身的比例 35%**。平均最大转角 12°，平均 13.7 步（比最优多 2.4 步）。低头看过自己 42%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **+0.0 个百分点**（0/9 个档位上低头更好）。

**策略标签分布**：FRONTAL 60%，ROT_EARLY 35%，ROT_MID 3%，ROT_LATE 2%

**动作构成**：forward 84.9%，turn_left 5.2%，look_down 3.8%，turn_right 2.7%，right 1.8%，left 1.3%

**推理提到的主题**（占该模型集数的比例）：opening 100%，align 100%，turn 92%，width 70%，sideways 40%，blocked 28%，budget 3%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.9 | 1.083 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.8 | 1.026 | 5/5 | 0 | 0 | 11.2 | 0 | 0.2 | 20 | 100 | 100 |
| 1.7 | 0.969 | 5/5 | 20 | 0 | 11.6 | 3 | 0.6 | 20 | 100 | 100 |
| 1.6 | 0.912 | 5/5 | 20 | 0 | 11.4 | 3 | 0.4 | 20 | 100 | 100 |
| 1.5 | 0.855 | 5/5 | 20 | 0 | 11.8 | 3 | 0.8 | 60 | 100 | 100 |
| 1.4 | 0.798 | 5/5 | 40 | 0 | 12.2 | 6 | 1.2 | 60 | 100 | 100 |
| 1.3 | 0.741 | 5/5 | 40 | 20 | 12.0 | 12 | 1.0 | 20 | 100 | 100 |
| 1.2 | 0.684 | 5/5 | 80 | 0 | 13.0 | 18 | 2.0 | 60 | 100 | 100 |
| 1.1 | 0.627 | 5/5 | 80 | 40 | 13.4 | 27 | 2.4 | 60 | 100 | 100 |
| 1.0 | 0.570 | 5/5 | 80 | 0 | 16.6 | 21 | 5.6 | 80 | 100 | 100 |
| 0.9 | 0.513 | 2/5 | 100 | 100 | 29.2 | 45 | 14.2 | 100 | 40 | – |

**它自己的原话**：

- （A/S 2.00，通过，11 步）「The red marker is straight ahead and centered. There appear to be two blue obstacles with a gap between them. I'm currently at x=0.5, z=0.0, so I'm centered on my starting line. The gap between the blue blocks appears to be centered, so I should be able to walk straight through. I'll move forward toward the red marker. The red marker is straight ahead and centered. There appear to be two blue obst…」
- （A/S 0.90，通过，29 步）「I can see the red marker straight ahead through a narrow gap between two blue panels. I'm currently at x=0.5, z=0.0, so I'm centered on my walking line. The gap appears to be centered as well. I need to walk forward toward the red marker. The gap looks narrow, so I should check if I need to turn my torso to fit through. Let me first look down to assess my body width relative to the opening. I can …」
- （A/S 0.90，失败，30 步）「I can see a blue wall blocking the corridor with a narrow opening in the center where the red marker is visible. The opening appears to be centered on my walking line. I need to walk forward through this opening. The opening looks narrow, so I should check if my body width fits. My torso rotation is 0 degrees, meaning I'm facing straight ahead. I should walk forward toward the opening since it app…」

---

## gemini-2.5-flash

**类型：直立派**。几乎不转身（全部集里仅 7% 转身），靠正面通过；它的分主要来自 A/S 1.0 的零容差正面通过，到 A/S 0.9（必须 ≥60°）就基本全败，两档合计 6/10。　另：从不看自己的身体（8%）。

**总体**：60 集，通过率 **93%**；两个需要转身的档（A/S 1.0 与 0.9）6/10。转身集 7%，首次转身平均在 x = 7.06 m；**在不需要转身的宽度上仍转身的比例 0%**。平均最大转角 4°，平均 12.6 步（比最优多 1.3 步）。低头看过自己 8%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **-8.3 个百分点**（0/4 个档位上低头更好）。

**策略标签分布**：FRONTAL 93%，ROT_LATE 5%，ROT_MID 2%

**动作构成**：forward 89.2%，turn_right 2.5%，turn_left 2.2%，look_down 2.1%，left 1.5%，right 1.2%

**推理提到的主题**（占该模型集数的比例）：opening 100%，align 92%，turn 32%，blocked 22%，width 12%，sideways 3%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 0 | 0 | 11.4 | 0 | 0.4 | 0 | – | 100 |
| 1.9 | 1.083 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.8 | 1.026 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.7 | 0.969 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.6 | 0.912 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.5 | 0.855 | 5/5 | 0 | 0 | 11.2 | 0 | 0.2 | 20 | 100 | 100 |
| 1.4 | 0.798 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.3 | 0.741 | 5/5 | 0 | 0 | 11.2 | 0 | 0.2 | 20 | 100 | 100 |
| 1.2 | 0.684 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.1 | 0.627 | 5/5 | 0 | 0 | 11.2 | 0 | 0.2 | 20 | 100 | 100 |
| 1.0 | 0.570 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 0.9 | 0.513 | 1/5 | 80 | 100 | 29.4 | 51 | 14.4 | 40 | 0 | 33 |

**它自己的原话**：

- （A/S 2.00，通过，11 步）「The goal is to reach the red marker. The red marker is visible straight ahead, between the two blue obstacles. I need to move forward to get closer to the opening and then pass through it. My current position is 0.5m forward, so I am still some distance from the obstacles. The goal is to reach the red marker. The red marker is visible straight ahead, between the two blue obstacles. I need to move …」
- （A/S 0.90，通过，27 步）「The goal is to reach the red marker. The current view shows the red marker straight ahead through an opening. The most direct path is to move forward. The goal is to reach the red marker. The current view shows the red marker straight ahead through an opening. The most direct path is to move forward. The red marker is still straight ahead and visible through the opening. The previous step moved me…」
- （A/S 0.90，失败，30 步）「The goal is to reach the red marker on the far wall. The current view shows the red marker directly ahead, centered in the opening. The path is clear, so the most direct action is to move forward. The goal is to reach the red marker on the far wall. The current view shows the red marker directly ahead, centered in the opening. The path is clear, so the most direct action is to move forward. The go…」

---

## deepseek-v4.1-flash

**类型：按需转身型**。会随宽度改变是否转身（宽档转身率仅 11%，首次转身平均在 x = 3.77 m），是几种类型里最接近「判断」而非「固定策略」的一种；两档合计通过 5/10。

**总体**：60 集，通过率 **92%**；两个需要转身的档（A/S 1.0 与 0.9）5/10。转身集 18%，首次转身平均在 x = 3.77 m；**在不需要转身的宽度上仍转身的比例 11%**。平均最大转角 4°，平均 13.3 步（比最优多 2.0 步）。低头看过自己 53%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **+0.0 个百分点**（0/11 个档位上低头更好）。

**策略标签分布**：FRONTAL 82%，ROT_EARLY 10%，ROT_LATE 8%

**动作构成**：forward 86.1%，look_down 5.1%，turn_left 3.0%，turn_right 2.0%，right 1.6%，left 1.6%

**推理提到的主题**（占该模型集数的比例）：opening 100%，align 100%，width 80%，turn 43%，blocked 32%，sideways 10%，budget 3%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 0 | 0 | 12.0 | 0 | 1.0 | 80 | 100 | 100 |
| 1.9 | 1.083 | 5/5 | 0 | 0 | 11.6 | 0 | 0.6 | 60 | 100 | 100 |
| 1.8 | 1.026 | 5/5 | 20 | 0 | 11.4 | 3 | 0.4 | 20 | 100 | 100 |
| 1.7 | 0.969 | 5/5 | 20 | 0 | 12.0 | 6 | 1.0 | 60 | 100 | 100 |
| 1.6 | 0.912 | 5/5 | 40 | 0 | 12.4 | 9 | 1.4 | 60 | 100 | 100 |
| 1.5 | 0.855 | 5/5 | 0 | 0 | 11.6 | 0 | 0.6 | 60 | 100 | 100 |
| 1.4 | 0.798 | 5/5 | 20 | 0 | 11.8 | 3 | 0.8 | 40 | 100 | 100 |
| 1.3 | 0.741 | 5/5 | 20 | 0 | 12.0 | 3 | 1.0 | 20 | 100 | 100 |
| 1.2 | 0.684 | 5/5 | 0 | 0 | 11.4 | 0 | 0.4 | 40 | 100 | 100 |
| 1.1 | 0.627 | 5/5 | 0 | 0 | 11.8 | 0 | 0.8 | 60 | 100 | 100 |
| 1.0 | 0.570 | 5/5 | 0 | 0 | 11.6 | 0 | 0.6 | 40 | 100 | 100 |
| 0.9 | 0.513 | 0/5 | 100 | 0 | 30.0 | 27 | 15.0 | 100 | 0 | – |

**它自己的原话**：

- （A/S 2.00，通过，13 步）「The red marker is straight ahead through the gap between the blue panels. My torso is at 0 degrees, so my body is aligned with the walking direction. The opening looks narrow, so I should check my body width with look_down before entering, but first I need to get closer. Walking forward moves me toward the marker along the fixed walking direction. The red marker is straight ahead through the gap b…」
- （A/S 0.90，失败，30 步）「The red marker is straight ahead through the narrow opening. I am currently centered on the walking line (z=0) and aligned with the gap, so walking forward moves me toward the marker without needing to sidestep or turn. The red marker is straight ahead through the narrow opening. I am currently centered on the walking line (z=0) and aligned with the gap, so walking forward moves me toward the mark…」

---

## glm-4.6v

**类型：直立派**。几乎不转身（全部集里仅 8% 转身），靠正面通过；它的分主要来自 A/S 1.0 的零容差正面通过，到 A/S 0.9（必须 ≥60°）就基本全败，两档合计 6/10。　另：从不看自己的身体（10%）。

**总体**：60 集，通过率 **92%**；两个需要转身的档（A/S 1.0 与 0.9）6/10。转身集 8%，首次转身平均在 x = 5.00 m；**在不需要转身的宽度上仍转身的比例 4%**。平均最大转角 6°，平均 13.1 步（比最优多 1.7 步）。低头看过自己 10%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **+8.3 个百分点**（1/3 个档位上低头更好）。

**策略标签分布**：FRONTAL 90%，ROT_LATE 5%，ROT_EARLY 3%，SIDEWAYS 2%

**动作构成**：forward 93.7%，turn_right 2.3%，left 1.5%，turn_left 1.1%，look_down 0.8%，right 0.5%

**推理提到的主题**（占该模型集数的比例）：align 65%，opening 43%，width 27%，blocked 13%，turn 12%，sideways 3%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.9 | 1.083 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.8 | 1.026 | 5/5 | 20 | 0 | 13.4 | 36 | 2.4 | 0 | – | 100 |
| 1.7 | 0.969 | 5/5 | 0 | 0 | 11.6 | 0 | 0.6 | 20 | 100 | 100 |
| 1.6 | 0.912 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.5 | 0.855 | 5/5 | 20 | 0 | 11.2 | 3 | 0.2 | 0 | – | 100 |
| 1.4 | 0.798 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.3 | 0.741 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.2 | 0.684 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.1 | 0.627 | 4/5 | 0 | 0 | 14.8 | 0 | 3.8 | 0 | – | 80 |
| 1.0 | 0.570 | 5/5 | 0 | 0 | 11.2 | 0 | 0.2 | 20 | 100 | 100 |
| 0.9 | 0.513 | 1/5 | 60 | 100 | 28.4 | 27 | 13.4 | 80 | 25 | 0 |

**它自己的原话**：

- （A/S 2.00，通过，11 步）「The red marker is clearly visible at the end of the corridor, directly in front of me. My walking direction is fixed toward this marker, and there are no obstacles blocking my path. The blue panels on the sides do not obstruct my forward movement. Therefore, the most efficient action is to continue moving forward to reach the red marker. The red marker is clearly visible at the end of the corridor…」
- （A/S 0.90，通过，22 步）「My goal is to reach the red marker on the far wall. The current view shows a clear path directly ahead through the center of the corridor, between the two blue obstacles. Since the path is unobstructed and my walking direction is fixed toward the red marker, the most efficient action is to move forward to progress toward the target. No sidestepping or turning is necessary at this stage as the path…」
- （A/S 0.90，失败，30 步）「I need to reach the red marker on the far wall. Currently, there is a blue barrier blocking my path directly ahead. Since I cannot walk through the barrier, I need to navigate around it. The barrier appears to be wide enough that I should be able to walk around it by moving to the left or right. Given my current position and the layout of the corridor, moving to the left seems like the most direct…」

---

## gpt-4o

**类型：按需转身型**。会随宽度改变是否转身（宽档转身率仅 36%，首次转身平均在 x = 1.64 m），是几种类型里最接近「判断」而非「固定策略」的一种；两档合计通过 6/10。

**总体**：60 集，通过率 **92%**；两个需要转身的档（A/S 1.0 与 0.9）6/10。转身集 42%，首次转身平均在 x = 1.64 m；**在不需要转身的宽度上仍转身的比例 36%**。平均最大转角 15°，平均 15.0 步（比最优多 3.7 步）。低头看过自己 70%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **-5.0 个百分点**（0/10 个档位上低头更好）。

**策略标签分布**：FRONTAL 58%，ROT_EARLY 38%，SIDEWAYS 3%

**动作构成**：forward 77.4%，look_down 10.9%，turn_left 6.8%，look_left 1.7%，left 1.0%，look_right 1.0%

**推理提到的主题**（占该模型集数的比例）：opening 88%，width 87%，align 85%，turn 45%，blocked 15%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 0 | 0 | 12.2 | 0 | 1.2 | 40 | 100 | 100 |
| 1.9 | 1.083 | 5/5 | 20 | 20 | 12.2 | 9 | 1.2 | 60 | 100 | 100 |
| 1.8 | 1.026 | 5/5 | 40 | 0 | 14.8 | 9 | 3.8 | 80 | 100 | 100 |
| 1.7 | 0.969 | 5/5 | 0 | 0 | 11.4 | 0 | 0.4 | 40 | 100 | 100 |
| 1.6 | 0.912 | 5/5 | 40 | 0 | 13.0 | 6 | 2.0 | 60 | 100 | 100 |
| 1.5 | 0.855 | 5/5 | 20 | 20 | 14.8 | 9 | 3.8 | 80 | 100 | 100 |
| 1.4 | 0.798 | 4/5 | 40 | 0 | 15.4 | 6 | 4.4 | 40 | 50 | 100 |
| 1.3 | 0.741 | 5/5 | 40 | 0 | 12.4 | 9 | 1.4 | 80 | 100 | 100 |
| 1.2 | 0.684 | 5/5 | 60 | 20 | 15.4 | 27 | 4.4 | 80 | 100 | 100 |
| 1.1 | 0.627 | 5/5 | 60 | 0 | 13.8 | 12 | 2.8 | 80 | 100 | 100 |
| 1.0 | 0.570 | 3/5 | 80 | 67 | 21.8 | 30 | 10.8 | 100 | 60 | – |
| 0.9 | 0.513 | 3/5 | 100 | 100 | 22.8 | 63 | 7.8 | 100 | 60 | – |

**它自己的原话**：

- （A/S 2.00，通过，12 步）「The red marker is visible straight ahead, and there is a narrow passage between the blue obstacles. Moving forward will bring me closer to the red marker. The red marker is visible straight ahead, and there is a narrow passage between the blue obstacles. Moving forward will bring me closer to the red marker. The narrow passage between the blue obstacles requires me to check my width to ensure I ca…」
- （A/S 0.90，通过，16 步）「The red marker is directly ahead, and the path appears clear. Moving forward will bring me closer to the target. The red marker is directly ahead, and the path appears clear. Moving forward will bring me closer to the target. The red marker is visible directly ahead, but the blue barriers create a narrow opening that may require checking my width before proceeding. I will glance down to assess my …」
- （A/S 0.90，失败，30 步）「The red marker is visible straight ahead, and the opening between the blue obstacles is aligned with the walking direction. Moving forward will bring me closer to the marker without obstruction. The red marker is visible straight ahead, and the opening between the blue obstacles is aligned with the walking direction. Moving forward will bring me closer to the marker without obstruction. The openin…」
- （A/S 1.00，失败，30 步）「The red marker is visible straight ahead, and there are no obstacles blocking the path. Moving forward will bring me closer to the goal. The red marker is visible straight ahead, and there are no obstacles blocking the path. Moving forward will bring me closer to the goal. The blue obstacle blocks the direct path to the red marker. I need to determine the width of the openings to decide if I can p…」

---

## gpt-4o-mini

**类型：直立派**。几乎不转身（全部集里仅 7% 转身），靠正面通过；它的分主要来自 A/S 1.0 的零容差正面通过，到 A/S 0.9（必须 ≥60°）就基本全败，两档合计 5/10。　另：从不看自己的身体（0%）。

**总体**：60 集，通过率 **92%**；两个需要转身的档（A/S 1.0 与 0.9）5/10。转身集 7%，首次转身平均在 x = 7.25 m；**在不需要转身的宽度上仍转身的比例 0%**。平均最大转角 4°，平均 12.6 步（比最优多 1.2 步）。低头看过自己 0%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **+0.0 个百分点**（0/0 个档位上低头更好）。

**策略标签分布**：FRONTAL 92%，ROT_LATE 7%，SIDEWAYS 2%

**动作构成**：forward 89.7%，left 3.6%，right 2.8%，turn_left 2.5%，turn_right 1.3%，look_left 0.1%

**推理提到的主题**（占该模型集数的比例）：align 47%，blocked 8%，turn 7%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.9 | 1.083 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.8 | 1.026 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.7 | 0.969 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.6 | 0.912 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.5 | 0.855 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.4 | 0.798 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.3 | 0.741 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.2 | 0.684 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.1 | 0.627 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 1.0 | 0.570 | 5/5 | 0 | 0 | 11.0 | 0 | 0.0 | 0 | – | 100 |
| 0.9 | 0.513 | 0/5 | 80 | 0 | 30.0 | 48 | 15.0 | 0 | – | 0 |

**它自己的原话**：

- （A/S 2.00，通过，11 步）「The red marker is straight ahead, and the goal is to reach it. The best action is to move forward. The red marker is straight ahead, and the goal is to reach it. The best action is to move forward. The red marker is still straight ahead, and I need to continue moving towards it. The best action is to move forward again. The red marker is still straight ahead, and I need to continue moving towards …」
- （A/S 0.90，失败，30 步）「The red marker is directly ahead, and I am currently positioned 0.5 meters away from it. The best action to take is to move forward towards the marker. The red marker is directly ahead, and I am currently positioned 0.5 meters away from it. The best action to take is to move forward towards the marker. I am currently 1.25 meters away from the red marker, which is directly ahead. The best action is…」

---

## gemini-2.5-pro

**类型：固定策略型**。在不需要转身的宽度上仍有 98% 的集转身、平均最大转角 51°；它读的不是开口宽度，而是把「转身」当成固定动作，代价是平均多花 6.6 步，而且在需要转身的两档只通过 5/10。　另：频繁自检（95% 的集低头看自己）；步数浪费大（比最优多 6.6 步）。

**总体**：60 集，通过率 **90%**；两个需要转身的档（A/S 1.0 与 0.9）5/10。转身集 98%，首次转身平均在 x = 1.99 m；**在不需要转身的宽度上仍转身的比例 98%**。平均最大转角 51°，平均 17.9 步（比最优多 6.6 步）。低头看过自己 95%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **+0.0 个百分点**（0/3 个档位上低头更好）。

**策略标签分布**：ROT_EARLY 98%，FRONTAL 2%

**动作构成**：forward 61.5%，turn_right 20.9%，look_down 8.8%，turn_left 7.1%，right 0.8%，left 0.6%

**推理提到的主题**（占该模型集数的比例）：opening 100%，width 100%，turn 98%，sideways 98%，align 95%，blocked 70%，budget 2%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 80 | 40 | 14.8 | 30 | 3.8 | 100 | 100 | – |
| 1.9 | 1.083 | 5/5 | 100 | 40 | 14.8 | 39 | 3.8 | 80 | 100 | 100 |
| 1.8 | 1.026 | 5/5 | 100 | 60 | 15.6 | 45 | 4.6 | 100 | 100 | – |
| 1.7 | 0.969 | 5/5 | 100 | 80 | 17.6 | 60 | 6.6 | 100 | 100 | – |
| 1.6 | 0.912 | 4/5 | 100 | 75 | 18.8 | 60 | 7.8 | 100 | 80 | – |
| 1.5 | 0.855 | 5/5 | 100 | 60 | 16.6 | 48 | 5.6 | 100 | 100 | – |
| 1.4 | 0.798 | 5/5 | 100 | 60 | 17.0 | 57 | 6.0 | 100 | 100 | – |
| 1.3 | 0.741 | 5/5 | 100 | 40 | 18.6 | 57 | 7.6 | 80 | 100 | 100 |
| 1.2 | 0.684 | 5/5 | 100 | 80 | 16.2 | 54 | 5.2 | 100 | 100 | – |
| 1.1 | 0.627 | 5/5 | 100 | 40 | 15.6 | 48 | 4.6 | 80 | 100 | 100 |
| 1.0 | 0.570 | 4/5 | 100 | 50 | 22.2 | 51 | 11.2 | 100 | 80 | – |
| 0.9 | 0.513 | 1/5 | 100 | 100 | 27.6 | 60 | 12.6 | 100 | 20 | – |

**它自己的原话**：

- （A/S 2.00，通过，12 步）「My objective is to reach the red marker. The path is obstructed by two blue blocks with a gap between them. I am currently aligned with the gap. I will move forward to get closer and better assess if I can pass through it. My objective is to reach the red marker. The path is obstructed by two blue blocks with a gap between them. I am currently aligned with the gap. I will move forward to get close…」
- （A/S 0.90，通过，18 步）「My goal is to reach the red marker. There is a narrow opening between two blue obstacles that I must pass through. The opening is directly in front of me. I will move forward to get a better view and assess if I can fit through. My goal is to reach the red marker. There is a narrow opening between two blue obstacles that I must pass through. The opening is directly in front of me. I will move forw…」
- （A/S 0.90，失败，30 步）「My goal is to reach the red marker. There is a narrow gap between two blue obstacles that I must pass through. I am currently aligned with the gap but am some distance away. I will move forward to get closer to the obstacle to better assess the situation and plan my passage through the gap. My goal is to reach the red marker. There is a narrow gap between two blue obstacles that I must pass throug…」
- （A/S 1.00，失败，30 步）「My goal is to reach the red marker, which is straight ahead. There is a narrow gap between two blue obstacles that I must pass through. I will continue to move forward to get closer to the gap so I can better assess if I need to turn my body to fit through. My goal is to reach the red marker, which is straight ahead. There is a narrow gap between two blue obstacles that I must pass through. I will…」

---

## gpt-4.1

**类型：固定策略型**。在不需要转身的宽度上仍有 75% 的集转身、平均最大转角 12°；它读的不是开口宽度，而是把「转身」当成固定动作，代价是平均多花 4.1 步，而且在需要转身的两档只通过 3/10。　另：频繁自检（90% 的集低头看自己）。

**总体**：60 集，通过率 **88%**；两个需要转身的档（A/S 1.0 与 0.9）3/10。转身集 77%，首次转身平均在 x = 1.71 m；**在不需要转身的宽度上仍转身的比例 75%**。平均最大转角 12°，平均 15.4 步（比最优多 4.1 步）。低头看过自己 90%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **+0.0 个百分点**（0/5 个档位上低头更好）。

**策略标签分布**：ROT_EARLY 70%，FRONTAL 23%，SIDEWAYS 7%

**动作构成**：forward 76.2%，look_down 13.2%，turn_left 5.1%，right 1.6%，left 1.3%，look_left 1.3%

**推理提到的主题**（占该模型集数的比例）：opening 100%，align 98%，width 90%，turn 77%，blocked 23%，sideways 2%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 5/5 | 100 | 0 | 13.4 | 15 | 2.4 | 100 | 100 | – |
| 1.9 | 1.083 | 5/5 | 80 | 0 | 13.4 | 12 | 2.4 | 100 | 100 | – |
| 1.8 | 1.026 | 5/5 | 80 | 0 | 13.8 | 12 | 2.8 | 100 | 100 | – |
| 1.7 | 0.969 | 5/5 | 100 | 0 | 13.6 | 15 | 2.6 | 100 | 100 | – |
| 1.6 | 0.912 | 5/5 | 60 | 0 | 13.0 | 9 | 2.0 | 80 | 100 | 100 |
| 1.5 | 0.855 | 5/5 | 40 | 0 | 12.2 | 6 | 1.2 | 60 | 100 | 100 |
| 1.4 | 0.798 | 5/5 | 100 | 0 | 13.6 | 15 | 2.6 | 100 | 100 | – |
| 1.3 | 0.741 | 5/5 | 60 | 0 | 12.6 | 9 | 1.6 | 80 | 100 | 100 |
| 1.2 | 0.684 | 5/5 | 80 | 0 | 13.2 | 12 | 2.2 | 80 | 100 | 100 |
| 1.1 | 0.627 | 5/5 | 40 | 0 | 12.6 | 6 | 1.6 | 80 | 100 | 100 |
| 1.0 | 0.570 | 3/5 | 80 | 0 | 23.6 | 12 | 12.6 | 100 | 60 | – |
| 0.9 | 0.513 | 0/5 | 100 | 0 | 30.0 | 15 | 15.0 | 100 | 0 | – |

**它自己的原话**：

- （A/S 2.00，通过，13 步）「The path to the red marker is straight ahead, but there is a narrow opening between two blue barriers. I need to approach the opening and may need to check my width or orientation to fit through. The path to the red marker is straight ahead, but there is a narrow opening between two blue barriers. I need to approach the opening and may need to check my width or orientation to fit through. I am app…」
- （A/S 0.90，失败，30 步）「The path ahead is clear and the opening in the barriers is aligned with my current walking direction. Moving forward will bring me closer to the red marker and the opening. The path ahead is clear and the opening in the barriers is aligned with my current walking direction. Moving forward will bring me closer to the red marker and the opening. The opening in the blue barriers is directly ahead and…」
- （A/S 1.00，失败，30 步）「The path to the red marker is mostly clear, but there is a narrow opening between two blue barriers ahead. I need to approach the opening to assess if I can fit through or if I need to rotate my torso. The path to the red marker is mostly clear, but there is a narrow opening between two blue barriers ahead. I need to approach the opening to assess if I can fit through or if I need to rotate my tor…」

---

## qwen3-vl-235b-a22b-instruct

**类型：固定策略型**。在不需要转身的宽度上仍有 75% 的集转身、平均最大转角 22°；它读的不是开口宽度，而是把「转身」当成固定动作，代价是平均多花 6.4 步，而且在需要转身的两档只通过 3/10。　另：频繁自检（83% 的集低头看自己）；步数浪费大（比最优多 6.4 步）；总体通过率偏低（78%）。

**总体**：60 集，通过率 **78%**；两个需要转身的档（A/S 1.0 与 0.9）3/10。转身集 77%，首次转身平均在 x = 1.82 m；**在不需要转身的宽度上仍转身的比例 75%**。平均最大转角 22°，平均 17.8 步（比最优多 6.4 步）。低头看过自己 83%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **-18.1 个百分点**（0/6 个档位上低头更好）。

**策略标签分布**：ROT_EARLY 73%，FRONTAL 23%，ROT_LATE 2%，SIDEWAYS 2%

**动作构成**：forward 68.0%，look_down 7.4%，turn_right 7.1%，turn_left 5.6%，look_left 4.1%，backward 2.9%

**推理提到的主题**（占该模型集数的比例）：align 100%，opening 97%，turn 97%，width 93%，blocked 85%，sideways 37%，budget 5%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 4/5 | 80 | 0 | 17.2 | 12 | 6.2 | 80 | 75 | 100 |
| 1.9 | 1.083 | 5/5 | 60 | 0 | 14.4 | 9 | 3.4 | 100 | 100 | – |
| 1.8 | 1.026 | 4/5 | 80 | 0 | 17.0 | 18 | 6.0 | 60 | 67 | 100 |
| 1.7 | 0.969 | 5/5 | 60 | 0 | 12.6 | 9 | 1.6 | 20 | 100 | 100 |
| 1.6 | 0.912 | 5/5 | 60 | 0 | 13.6 | 9 | 2.6 | 100 | 100 | – |
| 1.5 | 0.855 | 4/5 | 60 | 0 | 17.2 | 9 | 6.2 | 100 | 80 | – |
| 1.4 | 0.798 | 4/5 | 100 | 0 | 18.6 | 21 | 7.6 | 100 | 80 | – |
| 1.3 | 0.741 | 4/5 | 100 | 0 | 17.6 | 45 | 6.6 | 100 | 80 | – |
| 1.2 | 0.684 | 5/5 | 60 | 0 | 15.2 | 9 | 4.2 | 80 | 100 | 100 |
| 1.1 | 0.627 | 4/5 | 80 | 25 | 18.0 | 51 | 7.0 | 100 | 80 | – |
| 1.0 | 0.570 | 3/5 | 80 | 33 | 21.8 | 33 | 10.8 | 80 | 50 | 100 |
| 0.9 | 0.513 | 0/5 | 100 | 0 | 30.0 | 36 | 15.0 | 80 | 0 | 0 |

**它自己的原话**：

- （A/S 2.00，通过，17 步）「I am positioned at x=0.5, meaning I have already moved forward slightly. The red marker is directly ahead, but there are blue obstacles on either side that may restrict torso rotation once I get closer. Since my torso is currently at 0 degrees and I need to ensure I can pass through the opening without collision, I should first glance down to assess my body width relative to the gap. This will hel…」
- （A/S 0.90，失败，30 步）「I am aligned with the red marker and the green opening ahead. Since I need to reach the red marker, the most direct action is to move forward. There are no immediate obstacles in my path, and turning or sidestepping would be unnecessary unless I encounter a problem later. Forward movement will bring me closer to the goal. I am aligned with the red marker and the green opening ahead. Since I need t…」
- （A/S 1.00，失败，30 步）「I am currently positioned at x=0.5, z=0, facing the red marker. Since the path appears clear and I need to reach the far wall, the most efficient action is to move forward. Turning or sidestepping is unnecessary unless an obstacle blocks my path, which is not evident from the current view. I will proceed forward to close the distance to the red marker. I am currently positioned at x=0.5, z=0, faci…」

---

## qwen-vl-max

**类型：半按需型**。转身与宽度的关系较弱：宽档仍有 49% 转身、首次转身平均 x = 2.52 m，介于判断与固定策略之间；两档合计通过 4/10。　另：步数浪费大（比最优多 5.3 步）；总体通过率偏低（77%）。

**总体**：60 集，通过率 **77%**；两个需要转身的档（A/S 1.0 与 0.9）4/10。转身集 50%，首次转身平均在 x = 2.52 m；**在不需要转身的宽度上仍转身的比例 49%**。平均最大转角 14°，平均 16.7 步（比最优多 5.3 步）。低头看过自己 55%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **-40.0 个百分点**（0/10 个档位上低头更好）。

**策略标签分布**：FRONTAL 47%，ROT_EARLY 40%，SIDEWAYS 8%，ROT_MID 3%，ROT_LATE 2%

**动作构成**：forward 72.4%，look_down 8.7%，turn_left 4.5%，right 3.5%，turn_right 3.1%，left 2.6%

**推理提到的主题**（占该模型集数的比例）：align 100%，turn 67%，opening 65%，width 63%，blocked 60%，sideways 20%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 4/5 | 40 | 0 | 15.2 | 24 | 4.2 | 40 | 50 | 100 |
| 1.9 | 1.083 | 5/5 | 80 | 0 | 13.8 | 18 | 2.8 | 100 | 100 | – |
| 1.8 | 1.026 | 5/5 | 20 | 0 | 12.4 | 3 | 1.4 | 80 | 100 | 100 |
| 1.7 | 0.969 | 4/5 | 80 | 0 | 17.4 | 24 | 6.4 | 40 | 50 | 100 |
| 1.6 | 0.912 | 3/5 | 60 | 0 | 19.0 | 15 | 8.0 | 60 | 33 | 100 |
| 1.5 | 0.855 | 4/5 | 80 | 0 | 16.2 | 27 | 5.2 | 60 | 67 | 100 |
| 1.4 | 0.798 | 4/5 | 20 | 0 | 16.6 | 3 | 5.6 | 40 | 50 | 100 |
| 1.3 | 0.741 | 5/5 | 20 | 0 | 11.8 | 3 | 0.8 | 20 | 100 | 100 |
| 1.2 | 0.684 | 4/5 | 40 | 0 | 16.0 | 9 | 5.0 | 40 | 50 | 100 |
| 1.1 | 0.627 | 4/5 | 60 | 0 | 15.6 | 9 | 4.6 | 40 | 50 | 100 |
| 1.0 | 0.570 | 4/5 | 40 | 25 | 16.2 | 15 | 5.2 | 40 | 50 | 100 |
| 0.9 | 0.513 | 0/5 | 60 | 0 | 30.0 | 21 | 15.0 | 100 | 0 | – |

**它自己的原话**：

- （A/S 2.00，通过，11 步）「The red marker is directly ahead, centered in the corridor. There are no obstacles in the immediate path, so the optimal action is to move forward toward the marker. No turning or sidestepping is needed at this stage. The red marker is directly ahead, centered in the corridor. There are no obstacles in the immediate path, so the optimal action is to move forward toward the marker. No turning or si…」
- （A/S 0.90，失败，30 步）「The red marker is visible ahead, centered in the corridor. The robot is already aligned with the path to the marker, so no lateral movement or turning is needed. The next step should be to move forward to progress toward the goal. The red marker is visible ahead, centered in the corridor. The robot is already aligned with the path to the marker, so no lateral movement or turning is needed. The nex…」
- （A/S 1.00，失败，30 步）「The red marker is visible and centered in the view, indicating that the robot is aligned with the path to the target. No obstacles are blocking the direct path forward. The next logical step is to move closer to the marker by walking forward. The red marker is visible and centered in the view, indicating that the robot is aligned with the path to the target. No obstacles are blocking the direct pa…」

---

## qwen3-vl-32b-instruct

**类型：半按需型**。转身与宽度的关系较弱：宽档仍有 45% 转身、首次转身平均 x = 1.22 m，介于判断与固定策略之间；两档合计通过 2/10。　另：步数浪费大（比最优多 9.3 步）；总体通过率偏低（58%）。

**总体**：60 集，通过率 **58%**；两个需要转身的档（A/S 1.0 与 0.9）2/10。转身集 45%，首次转身平均在 x = 1.22 m；**在不需要转身的宽度上仍转身的比例 45%**。平均最大转角 10°，平均 20.6 步（比最优多 9.3 步）。低头看过自己 70%；**同档位内**比较（排除『窄档本来就更容易低头、通过率也天然更低』这一混淆）：低头与不低头的通过率之差为 **-37.0 个百分点**（1/9 个档位上低头更好）。

**策略标签分布**：FRONTAL 43%，ROT_EARLY 32%，SIDEWAYS 23%，ROT_LATE 2%

**动作构成**：forward 70.3%，look_down 9.4%，right 4.5%，left 3.8%，look_left 3.3%，turn_left 2.7%

**推理提到的主题**（占该模型集数的比例）：align 100%，opening 87%，width 83%，blocked 80%，turn 68%，sideways 32%，budget 3%

| A/S | 宽度(m) | 通过 | turn% | side% | 步数 | maxrot | excess | look% | pass\|look | pass\|nolook |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.140 | 4/5 | 40 | 0 | 19.6 | 6 | 8.6 | 80 | 75 | 100 |
| 1.9 | 1.083 | 3/5 | 20 | 0 | 19.2 | 3 | 8.2 | 60 | 33 | 100 |
| 1.8 | 1.026 | 2/5 | 0 | 0 | 22.6 | 0 | 11.6 | 80 | 25 | 100 |
| 1.7 | 0.969 | 5/5 | 40 | 0 | 12.2 | 6 | 1.2 | 40 | 100 | 100 |
| 1.6 | 0.912 | 3/5 | 80 | 0 | 23.0 | 15 | 12.0 | 100 | 60 | – |
| 1.5 | 0.855 | 3/5 | 100 | 0 | 21.4 | 21 | 10.4 | 60 | 67 | 50 |
| 1.4 | 0.798 | 3/5 | 40 | 0 | 18.6 | 6 | 7.6 | 40 | 0 | 100 |
| 1.3 | 0.741 | 3/5 | 20 | 0 | 20.0 | 6 | 9.0 | 40 | 50 | 67 |
| 1.2 | 0.684 | 4/5 | 40 | 0 | 16.0 | 6 | 5.0 | 100 | 80 | – |
| 1.1 | 0.627 | 3/5 | 60 | 0 | 20.4 | 12 | 9.4 | 80 | 50 | 100 |
| 1.0 | 0.570 | 2/5 | 60 | 50 | 24.8 | 21 | 13.8 | 60 | 33 | 50 |
| 0.9 | 0.513 | 0/5 | 40 | 0 | 30.0 | 18 | 15.0 | 100 | 0 | – |

**它自己的原话**：

- （A/S 2.00，通过，14 步）「The red marker is visible at the far end, but blocked by two blue barriers. Since forward movement is blocked, I must determine if I can sidestep around one of the barriers. To do so safely, I need to assess my width relative to the opening between the barrier and the wall. Looking down will help me see how wide my body is and whether I can fit through the gap. The red marker is visible at the far…」
- （A/S 0.90，失败，30 步）「The red marker is visible ahead, but the path is blocked by two blue barriers. The robot must navigate through the gap between them. Since the robot's current position is centered (z=0.0) and the gap appears to be centered as well, moving forward should allow passage through the opening. No turning or sidestepping is needed yet, as the robot is already aligned with the gap. The red marker is visib…」
- （A/S 1.00，失败，30 步）「The red marker is visible at the end of the corridor, but blocked by two blue barriers. Since the robot's walking direction is fixed toward the marker, and forward movement will take it directly toward the barriers, I must first determine if I can pass between them. To assess the gap width relative to my body width, I should look down to see how wide I am, then glance left and right to estimate th…」

---

