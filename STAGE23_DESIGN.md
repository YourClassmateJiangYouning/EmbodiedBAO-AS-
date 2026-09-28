# Stage 2 / Stage 3 设计文档（EmbodiedBAO 记忆与习惯实验）

本文件是唯一权威规格。对话里讨论过的内容如果和这里冲突，以这里为准。
代码实现依据本文件；实现完成后本文件随代码一起提交。

---

## 0. 一句话

让每个模型在**窄到正面必不可能通过**的通道（A/S 0.80）里反复尝试 **12 回合**，
每回合结束由它**自己写一条笔记**留给下一次；然后把它放到**宽到不用转身**的通道
（A/S 1.10）走 **5 回合**，看它**会不会惯性转身**、**多快能改过来**。

---

## 1. 场景与几何（为什么是 0.80 和 1.10）

身体在开口处的投影宽度：

```
needed(theta) = 0.570*|cos theta| + 0.220*|sin theta|
              = 0.6110 * cos(theta - 21.1 deg)
```

| 级别 | 通道宽 W | 通过所需最小转角 | 实际要转 | 正面能过吗 |
| :--- | :--- | :--- | :--- | :--- |
| **A/S 0.80（学习段）** | **0.456 m** | **62.83 deg** | **5 次 turn（75 deg）** | **不能** |
| **A/S 1.10（探针段）** | **0.627 m** | 0 deg | 0 次 | **能**（单侧余量 2.85 cm） |

两个关键推论：

1. **在 A/S 0.80 上，成功 <=> 必然执行了"转够 63 度后穿过"**（唯一通路；正面不行、
   侧移不改变正面所需宽度、房间封闭绕不过去）。所以"成功次数"就是"转身通过动作被强化的次数"。
2. **在 A/S 1.10 上任何转角都能过**（最大需求 0.6110 < 0.627），所以探针里唯一有信息量的
   因变量是**发出多少转身**，不是成败。

**理论最优步数按通道宽度算**，不是一律 16：

```
optimal_steps(W) = ceil((SUCCESS_X - START_X) / MOVE_STEP) + min_turns(W)
                 = 11 + min_turns(W)
min_turns(W)     = 满足 needed(theta) <= W 的最小 15 度倍数 theta 的转身次数
```

| 级别 | W | min_turns | optimal_steps |
| :--- | :--- | :--- | :--- |
| A/S 0.80（学习段） | 0.456 | 5 | **16** |
| A/S 1.10（探针段） | 0.627 | 0 | **11** |
| A/S 0.90（对照） | 0.513 | 4 | **15** |

这张表是**广度优先搜索实测出来的**，不是算术推断：`lab_logs/verify_optimal_steps.py`
在真实动作空间里搜出最短无碰撞路径，0.456 → 16（11 forward + 5 turn）、0.627 → 11
（11 forward + 0 turn）、0.513 → 15（11 forward + 4 turn），与公式逐项一致。

效率指标定义为**每一回合都有值**（失败的回合跑满 30 步上限，所以也有值）：

```
excess(round) = total_steps(round) - optimal_steps(W(round))     # 失败回合 = 30 - optimal
```

失败回合的 excess 会恒等于 30 − optimal（学习段 14，探针段 19），它**只说明这回合失败了**，
不说明效率。所以 §7 的 Q2（"excess 有没有降到 ≤2"）**只在通过的回合里问**；`total_steps`
和 `optimal_steps` 都要单独存列，两种口径事后都能重算。

---

## 2. 实验结构

```
每个模型 x 6 次独立复现（无跨 run 记忆）
  run 1-4：累积笔记模式（cumulative）
  run 5-6：滚动笔记模式（rolling）

  每次复现内部：
    阶段 A：A/S 0.80，固定 12 回合（每回合 <= 30 步）
    阶段 B：A/S 1.10，固定  5 回合（每回合 <= 30 步）

11 个模型 x 6 run x 17 回合 = 1,122 个回合记录
```

固定不变量（对全部模型、全部 run 完全相同）：

| 项 | 值 |
| :--- | :--- |
| 学习段级别 / 回合数 | A/S 0.80 / **12 回合**，无论成败都跑满 |
| 探针段级别 / 回合数 | A/S 1.10 / **5 回合** |
| 每回合步数上限 | 30 |
| 回合编号 | 连续 1-17（不设倒计时提示） |
| 图像 | 512 px |
| 动作空间 / 场景 / 状态块 / 坐标系说明 / 任务句 | **与 Stage 1 逐字相同** |

成功与失败都不改变流程：成功了照样进入下一回合（这是"成功后是否改进"的数据来源）；
12 回合里 0 次成功的 run 照样走探针（它就是 d = 0 的天然对照）。

---

## 3. 记忆机制（本实验的核心装置）

```
(1) 回合内：模型看得见本回合之前的所有步骤（动作 + 环境反馈 + 每步 240 字自述）
           —— 与 Stage 1 完全相同（HISTORY_LIMIT=None, HISTORY_REASONING_CHARS=240）

(2) 回合末：一次额外调用（只看文本、不看图）
           输入 = 该回合的完整记录（每步动作 + 反馈 + 完整未截断的推理）
           输出 = 它自己写的一段纯文本笔记（长度不限，原样保存）

(3) 下一回合：只看得见记忆块
           —— 看不到上一回合的任何动作或推理
```

### 3.1 累积模式（run 1-4）的记忆块

```
Previous attempts (oldest first):
- round 1: failed (did not reach the red marker within 30 steps)
  your note: "<它自己写的话>"
- round 2: passed (reached the red marker in 22 steps)
  your note: "<它自己写的话>"
```

回合末指令：

```
Write a short note to yourself for your next attempt. Your note is the only thing
you will be able to read about this attempt next time; the record above will not be
shown to you again.
```

### 3.2 滚动模式（run 5-6）的记忆块

```
Your note from your previous attempt:
"<它自己重写过的那一条>"
```

**没有回合编号、没有成功/失败记录、没有任何历史。** 它唯一的记忆就是自己写的那句话。
回合末指令：

```
Here is your previous note and the complete record of this attempt. Rewrite your
note to yourself for your next attempt - it will replace the previous one. Your note
is the only thing you will be able to read about this attempt next time; the record
above will not be shown to you again.
```

### 3.3 两种模式的关系

差别是**"环境是否提供事实日志 + 笔记是否累积"**，即
**外置记忆（offloading）vs 自写痕迹的反复重写（Bartlett 式 serial reproduction）**。

对抗性推论：累积模式下模型**知道自己成功了几次**；滚动模式下不知道（除非它自己写进笔记）。
所以"习惯强度"究竟取决于**练了多少次**，还是**它知道自己练了多少次**，可以被这两组分开。

第 1 回合时记忆块整块不出现（两种模式相同）。

---

## 4. 三份提示词（定稿）

### 4.1 提示词 A：行动阶段（每回合开始）

```
You are a Unitree H1 humanoid robot. Your task is to reach the red marker on the far wall.

Available actions (each action is one discrete step):
<与 Stage 1 的 action_options_string 逐字相同，八字未改>

Your walking direction is fixed: it always points straight ahead at the far wall, ...
<ACTION_FRAME_NOTE，与 Stage 1 逐字相同>

<记忆块：累积模式见 3.1，滚动模式见 3.2；第 1 回合整块省略>

Current state:
- position (x, y, z): [...]
  x is how far forward you have walked from where you started, ...（与 Stage 1 相同）
- torso rotation (degrees): 0.0
- head camera sideways offset from straight ahead (degrees): ...
- head camera extra downward pitch (degrees): ...
- step limit for this episode: 30

Action history for this attempt (0 step(s) already taken, oldest first). This is your
own record of this attempt: use it to notice what you have already tried and whether
it worked.
<本回合之前的步骤，与 Stage 1 的渲染方式相同>

Reply with exactly one JSON object:
{"scene_description": "<what the current camera image shows>", "reasoning": "<brief reasoning>", "action": "<action>", "confidence": 0.0}
```

与 Stage 1 的差异**只有两处**：

1. 新增记忆块（位置：坐标系说明之后、Current state 之前）。
2. `Action history for this episode` -> `for this attempt`（连同后面那句里的 episode）。

### 4.2 提示词 B：笔记调用（回合结束，新增）

```
You are a Unitree H1 humanoid robot. You have just finished one attempt at your task:
reaching the red marker on the far wall.

Complete record of this attempt:
- step 0: forward -> executed | your reasoning: "<完整未截断>"
- step 1: turn_left -> executed | your reasoning: "<完整未截断>"
- step 2: forward -> blocked by obstacle or room boundary (body) | your reasoning: "<完整未截断>"
...
- the attempt ended: failed (did not reach the red marker within 30 steps)

<累积模式> Write a short note to yourself for your next attempt. Your note is the only
thing you will be able to read about this attempt next time; the record above will not
be shown to you again.

<滚动模式> Here is your previous note and the complete record of this attempt. Rewrite
your note to yourself for your next attempt - it will replace the previous one. Your
note is the only thing you will be able to read about this attempt next time; the record
above will not be shown to you again.
```

- 输出**纯文本**，不是 JSON；长度不限；原样保存。
- `the attempt ended:` 那一行由环境填（成功/失败 + 步数），不是它写的。
- **没有任何字段要求**（不写 what_went_wrong / what_to_try_next / confidence）。

### 4.3 提示词 C：探针阶段（第 13-17 回合）

**与提示词 A 完全相同，一字不改。** 只是环境的通道变成 1.10，提示词里不提这件事。

### 4.4 全局禁忌（实现时必须加测试钉住）

任务句、记忆块、结果行、笔记指令里**不得出现**：

```
opening, channel, wide, narrow, shoulder, sideways, turn to fit, you should,
A/S, 0.80, 0.90, 1.10, 0.57, 0.456, 0.627
```

**不得出现任何归因**（"你被挡住了"、"开口太窄"、"你应该转身"）。
结果行只写"是否在 30 步内到达红方块"。

---

## 5. 记录字段

**每回合**：

```
model, run, memory_mode(cumulative|rolling), phase(A|B), round(1-17),
a_s_ratio, channel_width, passed, total_steps, end_reason,
action_sequence, n_forward, n_backward, n_lateral, n_turn, n_look_down, n_glance,
max_rotation_deg, passage_rotation_deg, first_turn_step, first_turn_x,
final_x, final_z, wall_collisions, invalid_response_count,
reached_door, optimal_steps, gap, excess, strategy_label,
note_text, note_chars, memory_injected_chars
```

`gap` 允许是 `null`（这一回合没有一步 x >= 7.0，见 6.2）。`reached_door` / `optimal_steps`
必须落盘：前者是缺口的可见性，后者让两种 excess 口径事后都能重算。

**每 run 派生**：

```
d（阶段 A 成功次数 0-12）, first_pass_round, terminal_state（最后 3 回合成败模式）,
approach_latency（6.2 里被跳过的回合数）, gap_defined_rounds（1-12 里有 gap 值的回合数）,
probe_rotation[1..5], probe_turn[1..5], probe_look_down[1..5],
note_drift[1..16]（相邻笔记文本相似度）, excess_series[]
```

---

## 6. 预注册判据（先写进方法再跑）

### 6.1 策略标签（每回合，按优先级）

| 标签 | 规则 |
| :--- | :--- |
| SIDEWAYS | 横移 >= 3 且转身 <= 2 |
| ROT_LATE | 首次转身 x >= 7.0 |
| ROT_MID | 4.0 <= 首次转身 x < 7.0 |
| ROT_EARLY | 首次转身 x < 4.0 |
| FRONTAL | 转身 <= 1 且横移 <= 2 |
| MIXED | 其余 |

**按表格自上而下第一个匹配者胜出**，并且：`ROT_*` 三条都要求"转身次数 ≥ 1"；一回合
0 次转身时 `first_turn_x = None`，直接跳到 FRONTAL，不得比较 `None`。这个优先级有一个
必须写进方法里的后果：**只要转身 ≥ 1 次，就一定落在某条 `ROT_*` 上，所以 FRONTAL 实际
等价于"转身次数 = 0 且横移 ≤ 2"**。这是 §6.1 表格的字面含义，不是新增自由度。

计数口径：横移 = `n_lateral`（left + right），转身 = `n_turn`（turn_left + turn_right），
`look_*` 不计入任何一类。

### 6.2 三个曲线判据（基于 gap）

```
gap(round) = min over steps with x >= 7.0 of ( needed(theta) - W(round) )
             None，如果这一回合没有任何一步 x >= 7.0
I = gap(第一个有值的回合) - min(gap 的全部有值回合)
D = max single-round drop（只在有值的相邻回合之间算）
S = D / I
```

**三个必须先钉死的细节：**

1. **减的是这个回合的通道宽，不是写死的 0.456。** 0.456 只对 A/S 0.80 成立。
2. **gap 只在学习段（第 1–12 回合）算。** 探针段 W = 0.627 > needed 的最大值 0.6110，
   所以探针段 gap 恒 ≤ 0 且不携带任何信息；`I`、`D`、`S` 一律只用第 1–12 回合。
3. **没走到墙就是"没有值"，不许补数。** `gap = None`，同时记一个布尔列 `reached_door`。
   不补值的理由是实测的：Stage 1 全 660 局里只有 8 局（1.2%）从没到过 x ≥ 7.0，
   47 个 A/S 0.9 失败局里只有 1 局，而**没有任何一局从没发出过 forward**——"原地转圈"
   不是常见失败形态，常见的是"走到墙前、从不转身"（47 个失败局里 39 个）。
   A/S 0.80 更难，但这个比例的数量级不变，所以 `I` 的分母几乎总是存在。

   配套规则：`I` 用**第一个有 gap 值的回合**当基线，被跳过的回合数单独记为
   `approach_latency` 报出来；如果一个 run 12 个回合全都没有 gap 值（从没接近过墙），
   就按"定势"归类并打上 `never_approached` 标记，不参与 `S` 的比较。

   **敏感性臂（不需要重跑实验）**：把缺失回合补成 `needed_max - W = 0.6110 - 0.456 = 0.155`
   （该回合可能出现的最大 gap，因此是对"顿悟"最不利的补法），重算一遍 `S` 并报出分类是否改变。

| 判定 | 条件 |
| :--- | :--- |
| **顿悟** | S >= 0.6 且突变后连续 >= 2 回合 gap <= 0 且策略标签在突变回合质变 |
| **渐悟** | S <= 0.35 且改进回合数 >= 3 且 rho(round, gap) <= -0.7 |
| **定势** | I <= 0，或（SetIndex >= 0.6 且 gap 无趋势） |
| **振荡** | 其余 |

`SetIndex = P(strategy(r+1) == strategy(r) | round r failed)`；随机基线 1/6。
阈值做敏感性分析（S 取 0.5 / 0.6 / 0.7 各算一遍）。

### 6.3 "学会"的描述性标签（不用于筛选）

```
acquired = (d >= 2) and (最后一次成功在最后 5 回合内)
```

---

## 7. 分析计划

| # | 问题 | 方法 |
| :--- | :--- | :--- |
| Q1 | 顿悟 / 渐悟 / 定势？ | 1,122 个回合的 gap 曲线分类；报 S、rho、SetIndex 连续量 + 标签 |
| Q2 | 成功后会不会优化？ | excess 在成功后各回合的斜率；有没有出现 excess <= 2；笔记里有没有提到"更快/更少步数" |
| Q3 | 习惯有多强、多快消退？ | 探针第 1 回合转角（主指标）vs Stage 1 的 L9 基线；5 回合衰减斜率；probe_look_down（会不会主动看世界推翻规则） |
| Q4 | 累积 vs 滚动 | 11 个模型内的配对比较（Wilcoxon 符号秩，n=11）：探针第 1 回合转角、衰减斜率、d 的组内方差、笔记语义漂移率 |

补充（不需额外实验）：把 66 个 (模型, run) 摊开，
`probe_rotation ~ d + first_pass_round`，看**强化次数**还是**学习速度**在解释探针行为。

---

## 8. 与 Stage 1 的对照基线（已实测，免费）

| 量 | Stage 1 实测 |
| :--- | :--- |
| A/S 1.10 的转角 | 平均 **15.0 deg**；仅 **4/55（7.3%）** >= 45 deg；1.3 次转身/局 |
| A/S 0.90 的通过率 | 8/55 = 14.5%（Stage 2 用更难的 0.80，预期更低） |
| 首次转身位置 | 从不转 57.4% / x<4.0 38.3% / 中途 1.1% / 墙边 3.2% |
| 低头 -> 转身 | 25.2% vs 基率 8.7%（**2.9x**） |
| 总体 | 11 模型 568/660 = 86.1% |

---

## 9. 规模与成本

| | 数量 |
| :--- | :--- |
| 每次复现 | 17 回合 + 16 次笔记调用 ≈ **380 次模型调用** |
| 每模型（6 run） | ≈ **2,300 次** |
| 11 模型合计 | ≈ **25,000 次调用** |
| 机时估算 | 快模型 5 s/次 ≈ 35 h；deepseek 20 s/次 ≈ 13 h -> **约 2-3 个晚上** |

---

## 10. 明确不做的事

- 兜底臂（教它转身）
- 回穿阶段（探针后再回窄通道）
- 逆序对照（先宽后窄）
- 冻结记忆
- 记忆格式字段（what_went_wrong 之类）
- 依据学习速度筛选样本
- 改动任务句或动作菜单
- 告知通道宽度或"你被挡住了"

---

## 11. 实现要点

| 项 | 内容 |
| :--- | :--- |
| 协议 tag | `v8-memory-a08-a11`（记忆条件后缀 `-cum` / `-roll`） |
| 运行器 | 外层 `for run in 1..6` -> `for round in 1..17`；第 13 回合起 `set_channel_width(1.10)` |
| 每回合 | 新建 AgentAdapter（历史清空），提示词只注入记忆块 |
| 笔记调用 | 新增文本调用路径（无图、不解析 JSON、原样取 content） |
| 代码位置 | **新文件 `stage23.py`**（Stage 2 的纯函数与记录成形），**新文件 `test_stage23.py`**（它的测试）。Stage 1 的 `experiments.py` / `environment.py` / `protocol.py` / `analysis.py` 在实现纯函数阶段一行不动 |
| 纯函数 | `needed(theta)` / `gap_for(steps, W)` / `optimal_steps(W)` / `min_turns(W)` / `strategy_label(round_record)` / `excess(total_steps, W)`，全部进 `test_stage23.py`，不依赖 Isaac |
| 常量来源 | `stage23.py` **不许在模块顶层 `import environment`**（导入顺序坑：会把 `environment._HAS_ISAAC_SIM` 永久锁成 `False`）。它自己定义 `BODY_LENGTH = 0.570` / `BODY_WIDTH = 0.220` / `MOVE_STEP = 0.75` / `SUCCESS_X = 8.75` / `START_X = 0.5` / `TURN_STEP_DEG = 15.0`，并由 `test_stage23.py`（测试进程里可以 import environment）断言它们与 `environment`、`protocol` 里的同名值逐个相等，防止漂移 |
| 日志 | `logs/{tag}/run{R}_round{NN}_note.txt`（笔记调用的输入与输出分开存） |
| 护栏 | `--max_calls`、断点续跑、每回合原子落盘 |
| 测试 | 提示词禁忌词测试（4.4 节）；`gap <= 0 <=> analytic_pass_check` 一致性测试；`optimal_steps` 对 0.456 / 0.627 / 0.513 分别等于 16 / 11 / 15（与 BFS 实测一致） |

## 12. 已实测的支撑数据

写进方法里之前先跑出来的数，都不需要重跑实验：

| 量 | 实测值 | 出处 |
| :--- | :--- | :--- |
| 从没走到 x ≥ 7.0 的局 | 8/660 = **1.2%**（A/S 0.9 的 47 个失败局里只有 1 局） | `lab_logs/analyze_door_reach.py` 扫 660 个 per-step sidecar |
| 从没发出过 `forward` 的局 | **0/660** | 同上 |
| A/S 0.9 失败局的形态 | 47 个里 39 个"走到了墙前但从不转身"，1 个"没走到墙前" | 同上 |
| A/S 0.9 的失败局动作总量 | forward 795、look_down 153、turn_left 119、left 101、turn_right 101、right 90、backward 20 | 同上 |
| 最短通过步数（观测，通过局） | 第 0–10 级全部 **11**；第 11 级（A/S 0.9）**16** | 660 条 episode 记录 |
| 最短通过步数（BFS，真实动作空间） | 0.456 → **16**；0.627 → **11**；0.513 → **15** | `lab_logs/verify_optimal_steps.py` |
| 失败的回合 | 全部恰好跑满 **30** 步（92/92） | 660 条 episode 记录 |

两点必须写进结果：

1. **第 11 级观测最短 16 步，而 BFS 证明 15 步就够**（11 forward + 4 turn）。也就是说
   A/S 0.9 上 55 局里没有任何一个模型找到最优路径，全都多花了一步。
2. 失败回合的 `total_steps` 恒为 30，所以 `excess` 在失败回合上恒为 `30 - optimal_steps(W)`，
   它不携带效率信息，只能用来区分成败——Q2 必须只在通过回合里问。
