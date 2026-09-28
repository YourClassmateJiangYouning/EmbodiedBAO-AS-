# 交接提示词：EmbodiedBAO Stage 2/3（记忆与习惯实验）

> 用法：把下面 `---` 之间的全部内容整段复制，粘贴到新会话的第一条消息里。

---

你现在接手一个 Isaac Sim 具身智能实验项目的第二阶段。这个新会话没有任何历史上下文，下面是你需要的全部信息。**先读完再动手，不要凭直觉补全我没写的东西。**

## 0. 我的工作纪律（硬性要求，违反等于白干）

1. **任何结论都必须有实测依据。** 不许猜、不许推断后当成事实、不许"一般来说"。没有数据就说不确定，或者去测。这是我上一轮明确发过火的事。
2. **给我的命令必须是单行、可直接复制、不含任何 `<占位符>`。** 多行命令和占位符我已经粘错过两次。
3. **不要和 MirrorBench 对比。** 这是我自己的实验，不服务任何别人的 benchmark 对齐。
4. **画图严格按 Springer LNCS/CCIS 模板**（字体、线宽、字号、图注、单双栏宽度）。
5. **删除任何文件前先打印文件清单给我看**，我说可以再删。你上一轮一条无条件 `rm` 删掉过三个可用的实验记录。
6. 实验室机器是**共享**的，同一时刻只能跑**一个** Isaac Sim 实例。绝对不许 `pkill -f isaac` / `pkill -f main.py`（会杀掉别人的仿真），要杀就按记录的 PID 杀。
7. 用中文回答我。
8. **区分"结论"和"实现选择"，这是最重要的一条。** 涉及数据的判断（谁更好、有没有趋势、原因是什么）必须有实测依据，不许猜。但**实现层面的歧义不要停下来问我**——某个函数在边界情况返回什么、代码放哪个文件、某个字段叫什么，这类问题按设计文档**最字面的读法**实现，在代码注释和汇报里写明你选的是哪一种，并把这一类问题**攒成一批**在同一次汇报里一起给我看。上一轮你把三个实现选择当成三份要签字的科学决策、一行代码没写就整体停住了：那是交接的错，不是你的错。只有当文档**自相矛盾到两种读法会导致不同的科学结论**时，才停下来问。

## 0.1 设计文档已经补过一轮（2025 后续修订）

上一轮暴露出 `STAGE23_DESIGN.md` 有三处含糊，已经在文档里就地改掉了，以现在的版本为准：

- **§1**：`optimal_steps` 改为按宽度算（A/S 0.80 → 16，A/S 1.10 → 11，A/S 0.90 → 15，全部经真实动作空间 BFS 实测验证），`excess = total_steps − optimal_steps(W)`，失败回合因跑满 30 步也有值；Q2 的"excess ≤ 2"只在通过回合里问。
- **§6.1**：策略标签的优先级后果写明了（转身 ≥ 1 次必定落在某条 `ROT_*` 上，所以 FRONTAL 等价于"0 次转身且横移 ≤ 2"）；0 次转身时 `first_turn_x = None`，不许比较。
- **§6.2**：`gap` 减的是**本回合的通道宽**而不是写死的 0.456；**只在学习段（1–12 回合）算**；没走到墙（没有任何一步 x ≥ 7.0）时 `gap = None`，**不许补数**，配 `reached_door` 列，`I` 用第一个有值的回合当基线并报出 `approach_latency`，另有一个补 0.155 的敏感性臂。
- **§5**：新增 `reached_door` / `optimal_steps` / `approach_latency` / `gap_defined_rounds` 字段。
- **§11**：明确新建 `stage23.py` + `test_stage23.py`，实现纯函数阶段 Stage 1 的四个模块一行不动；`stage23.py` 顶层不许 `import environment`（导入顺序坑）。
- **§12（新增）**：这些修订背后的实测数据（1.2% 从没到墙、0/660 从没 forward、A/S 0.9 观测最短 16 步而 BFS 证明 15 步够，等等）。

## 1. 工作目录与仓库（先做这四件收拾工作）

工作目录：`D:\EmInAI\EmbodiedBAO-Stage23`

当前这个目录的状态（已经查清，不用再猜）：

- 它是**旧仓库 `D:\EmInAI\EmbodiedBAO(AS)` 的工作树副本**，分支 `master`，HEAD `ccd67ea`，`origin` 还指向旧仓库 `https://github.com/YourClassmateJiangYouning/EmbodiedBAO-AS-.git`。
- 里面嵌套了一个 `EmbodiedBAO-Stage23\EmbodiedBAO-Stage23\`，那是**新仓库的空克隆**：remote 是 `https://github.com/YourClassmateJiangYouning/EmbodiedBAO-Stage23.git`，只有一个 `982af6a Initial commit`，内容仅为 GitHub 自动生成的 `.gitattributes`。
- 新仓库的默认分支是 `main`，且已经有那个 initial commit；本工作树的 `master` 与它**没有共同历史**。
- 旧仓库里新写了两个文件**还没拷过来**：`README.md`（已重写，末尾新增"Lab operations"实验操作手册）和 `STAGE23_DESIGN.md`（唯一权威规格）。本目录里的 `README.md` 是旧版本。

收拾步骤（按顺序，每步做完告诉我）：

1. `git remote set-url origin https://github.com/YourClassmateJiangYouning/EmbodiedBAO-Stage23.git`
2. 删掉嵌套目录（那个空克隆，`.git` 里只有一个 initial commit，删除无损失）：先 `Get-ChildItem -LiteralPath 'D:\EmInAI\EmbodiedBAO-Stage23\EmbodiedBAO-Stage23' -Recurse -Force | Measure-Object` 报数给我，再删。
3. ~~从旧仓库拷文件~~ **这一步已经做完了**：`README.md`（含新的 "Lab operations" 一节）、`STAGE23_DESIGN.md`（权威规格）和本提示词 `HANDOFF_STAGE23.md` 都已经在新仓库目录里了。所以待办只剩上面的 1、2、4。
4. 首次推送要 `git push --force origin master:main`（会覆盖那个只有 `.gitattributes` 的 initial commit，无损失）。

## 2. 这个项目是什么

一个用 Isaac Sim 4.5 跑的具身导航基准：一个 Unitree H1 人形机器人（运动学、非物理）在封闭房间里走向远端红方块，中途要穿过墙上的一个通道开口。核心自变量是**通道宽度与身体肩宽的比值 A/S**，因为"人能不能正面走过去"取决于 `needed(θ) = 0.570·|cosθ| + 0.220·|sinθ| = 0.6110·cos(θ−21.1°)`（0.570 × 0.220 是身体矩形足迹；A/S 的分母就是 0.570）。

**Stage 1（A/S 阈值实验）已经跑完并定稿**，11 个模型 × 12 个宽度 × 5 次重复 = 660 局，568/660 = 86.1%。关键实测结果：

- 每级通过率：A/S 2.0→1.1 平坦在 90.9–98.2%，A/S 1.0 掉到 80.0%，**A/S 0.9 只有 14.5%**（8/55）。
- 平均转身次数：平坦段 0.6–1.3 → A/S 1.0 的 2.3 → A/S 0.9 的 4.8；平均最大偏航 8–15° → 16.6° → 37.4°；横移局占比 3.6–10.9% → 20% → **69.1%**。
- 首次转身位置分布：从不转 57.4%、x<4.0 占 38.3%、**中段 4.0–7.0 只占 1.1%**、x≥7.0 占 3.2%——人类那种"接近通道时才侧身"的中段带几乎是空的。
- A/S 0.9 上 8 次成功全部是"转了 ≥60° 且最终居中"；偏心的 23 局 0 成功。
- `look_down` 之后下一步转身的概率 25.2%，基线 8.7%（**2.9 倍**）。
- 模型个性示例（Spearman ρ(A/S, 最大偏航)，n=60）：claude-sonnet-4-6 −0.725、gpt-4o −0.552、gemini-2.5-flash −0.425、deepseek −0.270、gemini-2.5-pro −0.132、gpt-4.1 +0.091。

## 3. 你现在要实现的是 Stage 2/3

**一句话**：让每个模型在窄到正面绝不可能通过的通道（A/S 0.80）里反复试 12 回合，每回合结束**由它自己写一条笔记**留给下一次；然后把它放到宽到不用转身的通道（A/S 1.10）走 5 回合，看它会不会惯性转身、多快能改过来。

- 几何：A/S 0.80 → 通道宽 0.456 m，通过最少需要 62.83° 转角，必须 5 次 `turn`（75°）；A/S 1.10 → 0.627 m，任何转角都能过（最大需求 0.6110 < 0.627），所以探针段唯一有信息量的因变量是**发出多少转身**，不是成败。
- 理论最优步数 = 16（11 次 forward + 5 次 turn），定义 `excess = steps − 16`。
- 结构：每模型 6 次独立复现（run 1–4 = 累积笔记 `cumulative`，run 5–6 = 滚动笔记 `rolling`）；每次复现 = 阶段 A（A/S 0.80，固定 12 回合）+ 阶段 B（A/S 1.10，固定 5 回合）；每回合 ≤30 步；合计 11 × 6 × 17 = 1,122 个回合记录，约 25,000 次模型调用，2–3 个晚上。
- 记忆机制：回合内看本回合全部步骤（与 Stage 1 相同）；回合末一次**纯文本**调用让它写笔记；下一回合**只能看到记忆块**，看不到上一回合任何动作或推理。累积模式给全部结果行 + 全部笔记；滚动模式只给它自己重写过的那一条（无回合编号、无成败日志）。
- 协议 tag：`v8-memory-a08-a11`，记忆条件后缀 `-cum` / `-roll`。
- 明确不做：兜底臂、回穿阶段、逆序对照、冻结记忆、记忆格式字段、按学习速度筛样本、改任务句或动作菜单、告诉它通道宽度或"你被挡住了"。

**上面只是摘要。** 几何推导、三份提示词逐字稿、记忆块逐字稿、全局禁忌词、记录字段、预注册判据（顿悟/渐悟/定势的 S、ρ、SetIndex 阈值）、分析计划、成本表、实现要点，全部在 `STAGE23_DESIGN.md` 里，**它是唯一权威**；如果和本提示词或与我口头说的冲突，以它为准。

## 4. 实现顺序（按 `STAGE23_DESIGN.md` §11）

1. **纯函数 + 测试先做**：`needed(theta)`、`gap_for(...)`、`optimal_steps()`、`strategy_label(...)`，全部加进离线几何测试（不依赖 Isaac Sim）。
2. 回合循环：外层 `for run in 1..6` → `for round in 1..17`，第 13 回合起把通道切到 A/S 1.10。每回合新建 `AgentAdapter`（历史清空），提示词只注入记忆块。
3. 记忆注入 + 两种模式的记忆块渲染。
4. 笔记调用路径：新增文本调用（无图、不解析 JSON、原样取 `content`）。
5. 护栏：`--max_calls`、断点续跑、每回合原子落盘、笔记输入输出分开存 `logs/{tag}/run{R}_round{NN}_note.txt`。
6. 测试：提示词禁忌词测试（§4.4 的 13 个词一个都不许出现）、`gap <= 0 ⟺ analytic_pass_check` 一致性测试。

## 5. 代码库事实（别重新发明，这些都已经验证过）

文件角色：`environment.py`（Isaac 场景、运动学、解析碰撞门、成功判定）、`protocol.py`（动作空间与统一提示词的唯一来源）、`experiments.py`（回合循环、Level 阶梯、记录成形、断点续跑）、`ai_agent.py`（OpenAI 兼容 MLLM 客户端 + `random` 基线）、`main.py`（CLI、CSV 导出、阈值报告）、`analysis.py`（指标与画图）、`persistence.py`（原子写、tag 清洗、损坏文件隔离）、`capture_views.py`（固定视角渲染）、`tools/`（一次性测量探针）。

关键常量与语义：

- 动作空间 9 个（含 `look_down`）。**行走坐标系**：`forward`/`backward` = ±x，`left`/`right` = ∓z；`action_delta(action, move_step)` 是唯一来源；`turn_*` 只旋转躯干 15°，不改变行走方向、不改变视野。
- 视线钉在行走方向上：`look_yaw = self._camera_yaw_offset`（**故意不带躯干偏航**），`look_down` 只影响一帧，每个动作后清空。
- `SUCCESS_X = 8.75`、`WALL_X = 8.0`、`MOVE_STEP = 0.75`、房间 16 × 5 m、眼高 1.68 m、图像 512 px。
- 12 级宽度阶梯 `_APERTURE_RATIOS = (2.0 … 0.9 步长 0.1)`，Level 0 最宽（A/S 2.0），Level 11 = 0.9。
- 碰撞是解析的定向矩形 SAT，**无膨胀**：`BODY_CLEARANCE = 0`，`OVERLAP_TOLERANCE = 2e-7`。这决定了 A/S 1.00 恰好可通过，不能改。
- `HISTORY_LIMIT = None`（不截断），`HISTORY_REASONING_CHARS = 240`。
- 提示词历史行格式：`- step {n}: {action} -> {feedback} | your reasoning: {clipped}`。
- 任务句必须逐字保持：`You are a Unitree H1 humanoid robot. Your task is to reach the red marker on the far wall.`（**不许改**，不许告诉它要穿通道——实验的核心是"无意中"发现）。

环境与模型：

- 网关 `http://35.220.164.252:3888/v1`，key 环境变量 `BOYUE_API_KEY`；`BAO_LLM_TIMEOUT=90`、`BAO_MAX_RETRIES=3`。
- `models.json` 里 11 个模型；`claude-sonnet-4-5-20250929` 已被换成 `claude-sonnet-4-6`（旧名字在 `excluded` 里，原因是 `/v1/models` 列了它但账号没有 Bedrock 通道）。
- 推理预算已经实测并写进 `ai_agent.MODEL_REQUEST_PARAMS`：deepseek `reasoning_effort="none"`（12,820 → 1,224 tok）、glm `thinking={"type":"disabled"}`（12.9 → 6.4 s）。**动 `ai_agent.py` 之后必须跑一次带参数的模型端到端**，否则会出现 `TypeError: unexpected keyword argument` 让整局变成 30 步兜底（`invalid_response_count = 30`）。

## 6. 两个必须先读的坑

- **导入顺序坑**：`import environment` 必须在 `SimulationApp` 之后。先导入会把 `environment._HAS_ISAAC_SIM` 永久锁成 `False`，于是脚本什么都不建、**退出码还是 0**。`test_bao_geometry.py` 里有 AST 检查钉住它，所以跑 sweep 前必须让这套测试通过。
- **命令行粘贴坑**：实验室机器上的命令永远单行。`[1] 12345` 那个数字是 bash 的**包装子 shell**，不是脚本进程；上一轮按它排除 PID，把新 sweep 打死了、旧的还在跑。

## 7. 第一步请做这些，然后停下来向我汇报

1. 读 `STAGE23_DESIGN.md`（权威规格）和 `README.md` 的 "Lab operations" 一节。
2. 做完第 1 节的四件收拾工作。
3. 跑三个离线测试确认环境没坏：`python test_bao_geometry.py`、`python test_bao_integration.py`、`python test_bao_persistence.py`（当前基线是 35/19/13 项全过）。这三个不需要 Isaac Sim。
4. 把 `STAGE23_DESIGN.md` §11 的纯函数和它们的测试实现出来，跑通。
5. 汇报：你改了什么、测试结果、你对设计文档里任何一处**不同意或发现矛盾**的地方（如果有的话，先问我，别自己改设计）。

不要一次把整个 Stage 2 写完。先给我纯函数 + 测试，我看到测试通过再让你写回合循环。

---
