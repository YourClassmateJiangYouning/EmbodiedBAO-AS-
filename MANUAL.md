# 项目手册（**每一轮对话开始前先读这个文件**）

> 这是给我（AI 助手）看的操作手册，不是给外人看的介绍。项目介绍在 `README.md`。
> **每次和用户对话前先通读一遍本文件**，尤其是「数据住在哪」和「我犯过的错」两节。

---

## 0. 一句话现状

在 Isaac Sim 4.5 里让 15 个 MLLM 驱动 Unitree H1 穿越窄通道，测 A/S 阈值（A/S = 开口宽 / 肩宽 0.570 m）。
**Stage 1 已冻结并扩展**（17 档、15 模型、5 场景、每场景 5 个标志物）。实验室场景（`stage1.1`）的全量 sweep **正在实验室机器上跑**。

---

## 1. 数据住在哪（★最重要★ 我为此浪费了好几轮）

| 我要找的东西 | **正确位置** | 说明 |
| :--- | :--- | :--- |
| **模型的思考过程 / 原话**（`scene_description`、`reasoning`、`confidence`） | **`logs/{tag}/level{level}_episode{id:03d}_agent.txt`** | **每步的完整提示词 + 模型完整回答** ✓ 每个文件 30+ 处 `scene_description` ✓ |
| 每回合结果（通过/转角/首次转身步/碰撞） | `results/level{N}/{model}/{tag}/episode_{id:03d}.json` | **只有行为，没有思考** |
| 每一步行为（动作/位置/旋转/耗时） | `results/level{N}/{model}/{tag}/episode_{id:03d}_steps.json` | **只有行为，没有思考** |
| 档位汇总 | `results/level{N}/{model}/{tag}/summary_{tag}.json` | 通过率、侧身率、平均转角等 |
| 扁平 CSV（分析用） | `results/{model}/level{N}_{tag}_{时间戳}.csv` | 20 列行为数据 |
| 断点续跑状态 | `results/{model}/checkpoint_{tag}.json` | |
| 运行时间线 | `sweep_*.log`、`logs/STATUS.txt`、`run_progress.txt` | |
| 生效参数 | `logs/{tag}/args.json` | |

**教训**：`main.py` 的文档字符串里**一开始就写着** `logs/{tag}/…_agent.txt  raw model I/O`。
**找任何东西之前，先读 `main.py` 顶部的输出结构说明，以及写它的那个函数，不要在 `results/` 里乱翻。**

**标志物归因规则**：环境每回合推进一个槽位，`slot = (episode_id % 5) + 1`，对应 `scenes.describe_marker(scene, slot)`。

---

## 2. 目录地图

| 路径 | 是什么 | 能改吗 |
| :--- | :--- | :--- |
| `environment.py` | Isaac 场景：房间 / 障碍墙 / 开口 / 标志物 / 相机 / 机器人 | **Stage 1 冻结，只增不改** |
| `protocol.py` | 提示词与动作空间（9 个动作）。`build_prompt()` 里标志物名词短语会按本回合替换 | 谨慎 |
| `scenes.py` | **Stage 1 扩展的纯数据**：10 形状 / 10 颜色 / 25 标志物 / 5 场景材质与陈设 / `describe_marker()` | 可改，**但不能动 `stage1.1`**（见 §5） |
| `scene_builder.py` | 把 `scenes.py` 应用到 stage：材质解析（优先用仓库内本地文件）/ 标志物网格 / 陈设 | 可改 |
| `capture_scenes.py` | 渲染预览。`--scene --level --outdir --parts --hide_robot --flat_paint` | 可改 |
| `main.py` | 实验入口。**顶部注释写明全部输出结构** | 谨慎 |
| `experiments.py` | runner：调模型、记 history、写记录。`_append_log()` 写 `logs/{tag}/…_agent.txt` | 谨慎 |
| `ai_agent.py` | 模型客户端与响应解析（`parse_response` 取出 action/reasoning/scene_description/confidence） | 谨慎 |
| `run_all_models.sh` | 全量 sweep 驱动（`SCENE=` 选场景；`--all-levels --episodes 5 --resume`） | 可改 |
| `status.sh` | 一条命令看：在跑什么 / 进度 / 报错 / 花费 | 可改 |
| `watch.sh` | 实时追踪：每完成一档跳一行，每完成一个模型跳一个横幅 | 可改 |
| `tools/parse_agent_logs.py` | **把 `logs/` 里的思考过程解析成表 + 统计**（提到标志物颜色/形状/开口的比例） | 可改 |
| `tools/verify_stage1_lab.py` | 冒烟验收：标志物轮换 / 提示词命名 / 额度 / 坏回复 / traceback | 可改 |
| `tools/check_field_names.py`、`tools/check_names.py` | 静态审计：字段名 / 未定义名 | 可改 |
| `tools/fetch_assets.py`、`tools/shrink_textures.py`、`tools/shrink_tree.py` | 素材抓取 / 降采样（**坏图只跳过不中断**） | 可改 |
| `assets/isaac/` | 随仓库走的素材（**136 MB / 880 文件**：Simple_Warehouse + Hospital + Office） | 只增 |
| `test_bao_*.py` | 手写测试套件：geometry 37 / integration 20 / persistence 13 / memory 29 / memory_runner 17 / scenes 18 / **assets 3** | 必须全过 |
| `lab_logs/` | Stage 1 分析与出图脚本、`figures/` 成品图、`bao_v7_all.tgz` 归档 | 冻结 |
| `results/`、`logs/` | 运行产物（**不入库**） | — |

---

## 3. 硬规矩（用户反复强调，违反会被骂）

1. **一切结论必须有依据**：不猜。跑代码 / 看日志 / 看图 / **读代码**。说「很可能」之前先去找证据。
2. **找东西先读代码**：文件在哪、字段叫什么、写到哪 —— `main.py` 顶部和写它的函数里都写着。
3. **给的命令必须可直接复制**：单行、无 `<占位符>`（用户会照抄，`kill <93233>` 那次就是教训）。
4. **模型原话是实验结论的一部分**，不是"定性补充"：推理文本是结果变量。
5. **不能动正在跑的场景**：`main.py` 每换一个模型就重读 `scenes.py` → 改 `stage1.1` 会让 15 个模型内部不一致。
6. **删除/移动前先打印**。**不要用 `pkill -f isaac`**（共享机器，别人也在跑）；用 `pkill -f 'main.py --model'` 这种精确匹配。
7. **PDF 里不能有中文**（matplotlib 无 CJK 字形）；中文只进 Markdown/docx。
8. **`stage1.1` 就是"原实验室、无遮挡"的基线**，用户当它是在跑的那一轮，不要装修它。

---

## 4. 命令速查

```bash
# 进度 / 在跑什么 / 报错 / 花费
cd ~/EmbodiedBAO-AS- && bash status.sh
cd ~/EmbodiedBAO-AS- && bash watch.sh

# 只看到哪一档（最简）
cd ~/EmbodiedBAO-AS- && grep -E 'pass_rate=' sweep_1.1.log | tail -20

# 确认标志物轮换 + 提示词同步（必须出现 5 行、颜色各不相同）
cd ~/EmbodiedBAO-AS- && grep -c 'prompt now says' sweep_1.1.log; grep -m6 'prompt now says' sweep_1.1.log

# ★ 思考过程统计（每模型：提到自己标志物颜色/形状/开口的比例 + 原文例子）
cd ~/EmbodiedBAO-AS- && python3 tools/parse_agent_logs.py --csv logs_reasoning.csv | tail -60

# 冒烟验收
cd ~/EmbodiedBAO-AS- && python3 tools/verify_stage1_lab.py --log smoke_1.1.log --model gpt-4o-mini

# 全部测试
cd ~/EmbodiedBAO-AS- && for s in test_bao_geometry test_bao_integration test_bao_persistence test_bao_memory test_bao_memory_runner test_bao_scenes test_bao_assets; do printf '%-26s ' $s; python3 $s.py 2>/dev/null | tail -1; done

# 渲染预览（会再起一个 Isaac，注意显存）
cd ~/EmbodiedBAO-AS- && /home/ybh/isaacsim/python.sh capture_scenes.py --scene stage1.2 --level 10 --outdir prev_1.2 --parts materials,marker

# 停（精确）
pkill -f 'main.py --model'
```

**实验室机器**：仓库 `/home/eai/EmbodiedBAO-AS-/`；Isaac 启动器 `/home/ybh/isaacsim/python.sh`（headless）；
用户会把图放到 `C:\Users\asus\Desktop\科研狗之人机心理学\实验图片\`，**我可以直接用 read_image 读**。

---

## 5. 几何与规模（实测，别再重算）

- `needed(θ) = 0.570|cosθ| + 0.220|sinθ|`；峰值 0.611 m @ 21.1°；**任何姿态都过不去的下限 = 0.220 m（A/S 0.386）**。
- 17 档：A/S 2.0→0.9 步长 0.1，再 0.8/0.7/0.6/0.5/0.4（宽 1.140 … 0.228 m；需 0/…/75/75/90/90/90°）。
  **A/S 2.0 的通道宽 = 1.140 m**（不是 2.28 ✗ 我算错过一次）。
- 成功判定 x ≥ 8.75；步长 0.75 m；转身 15°；眼高 1.68 m / 俯仰 15° / 焦距 13.36 mm / 水平 FOV ≈76°。
- 规模：15 模型 × 17 档 × 5 标志物 = **1,275 集/场景**（全 5 场景 6,375 集）。
- **实测成本与时间**：约 $0.036/集 → 单场景 ≈ **$46 / ¥330**；**单模型 2.5–9 小时** → 15 模型 ≈ **50–60 小时**（gemini-2.5-pro 单独跑了 9 小时 ✗）。

---

## 6. 15 个模型与网关

网关 `http://35.220.164.252:3888/v1`，key **从环境变量读，绝不写进任何文件**（见下方事故记录）。
启动前 `export BOYUE_API_KEY='...'`，`run_all_models.sh` 和 `tools/check_credit.py` 都从环境里取。
另外要用 `BAO_DISABLE_PROXY=1`。

> **事故记录（2026-10-08）**：这一节原来把真实的 key 明文写在文件里，而 `MANUAL.md` 被提交并推到了
> 公开仓库 `github.com/YourClassmateJiangYouning/EmbodiedBAO-AS-`。此后账户额度从正常一路掉到
> `$0.0025`（10-07 21:13 首次 403 quota），22:06 起 key 变成 401 Invalid token。
> **公开仓库里的 key 必须视为已泄露**：删掉这一行不能把它从 git 历史里去掉，任何人 clone 都能翻出来。
> 正确处理是**换 key**，并且从此只用环境变量。仓库里其他脚本、日志、`results/` 一律不得出现明文 key。
名单在 `models.json`：qwen3-vl-235b / qwen3-vl-32b / qwen-vl-max / gemini-2.5-pro / gemini-2.5-flash / gpt-4.1 / gpt-4o / gpt-4o-mini / claude-sonnet-4-6 / kimi-k2.5 / deepseek-v4.1-flash / glm-4.6v / grok-4.3 / doubao-seed-2-0-pro-260215 / mimo-v2.5。
`mimo-v2.5` 需要 `reasoning_effort="none"`（已写入 `ai_agent.py` 的 `MODEL_REQUEST_PARAMS`）。
**已排除**：MiniMax 全系（认色不稳）、Mistral 全系（HTTP 500）、Meta 全系（500）、kimi-k2-thinking（无视觉）、doubao-seed-2-1-pro（180 s 超时）。

---

## 7. 场景搭建进度（Stage 1 扩展）

| 场景 | 材质 | 陈设 | 状态 |
| :--- | :--- | :--- | :--- |
| `stage1.1` 实验室 | **无** | **无** | **正在跑的基线，一个字都不许改** |
| `stage1.2` 仓库 | Simple_Warehouse 四件套 | 8 件 | 数据就绪；渲染时出现**未定位的深色遮挡** ✗ |
| `stage1.3` 图书馆 | 木地板 / 白墙 / 吊顶 / 抹灰 | 8 件（书架/书车/阅读灯…） | 数据就绪，未渲染验证 |
| `stage1.4` 公园 | 抹灰 / 吊顶 / 白墙 + **地面纯色**（没抓到草地贴图） | 8 件（长椅/绿篱/花坛…） | 数据就绪 |
| `stage1.5` 超市 | 瓷砖 / 金属板 / 吊顶 / 白墙 | 8 件（货架/价签/收银台…） | 数据就绪 |

- 陈设硬约束（`test_bao_scenes.py` 强制）：全部 `collides=False`；地面物 `|z| ≥ 1.9`、≤1 m 高、≤0.9 m 长；
  墙挂件 `|z| ≥ 0.9`、伸出 ≤0.20 m、按「(横向, 纵向, 厚度)」声明；每件必须有颜色；**不得用本场景 5 个标志物之一的颜色**。
- 宽墙件要离中线够远，否则会**压住开口投影** ✓（图书馆书架就栽在这 ✓ 现在 1.4 m 宽 @ z=±1.65 ✓）。
- 素材抓取流程：`python tools/fetch_assets.py --list Environments/Hospital` → `--set …` → `tools/shrink_tree.py`（**一张坏图不能中断整轮**）。
- **渲染前先确认显存**：实验室那台还有别人的 IsaacLab 占 5.5 GB ✗。

---

## 8. 我犯过的错（别重犯）

| # | 错 | 教训 |
| :-: | :--- | :--- |
| 1 | **在跑不了 `pxr` 的机器上猜 pxr API**，连错三次（手写着色器图两次、把 `.mdl` 当 stage 资产一次） | 不确定能不能跑通的 pxr 调用，**先用不依赖猜测的写法**（`displayColor` / 纯色平涂），并在提交信息里标注"未验证" |
| 2 | **找东西不读代码，靠猜目录**（在 `results/` 里翻了四轮找思考过程，其实在 `logs/`） | 先读 `main.py` 顶部输出结构 + 写它的函数 |
| 3 | **说"不影响结论"**（把推理文本当定性补充） | 推理文本是**结果变量**，丢了就是数据损失 |
| 4 | 在**本机**用 PowerShell 字符串改 UTF-8 文件 / 传多行 `-m` | 用 `edit`/`write` 工具；多行提交用 `COMMIT_MSG_*.txt` + `git commit -F` |
| 5 | 用 `grep '^\['`、`2>/dev/null`、`head -30` **把真正的报错过滤掉** | 看日志先 `tail -60` 全文，再定向 grep |
| 6 | `capture_scenes.py` 给 `SimulationApp` 传 `width/height` → 渲染目标 256² 被放大成噪点 | **照 `main.py` 的写法：只传 `headless`** |
| 7 | 以为"渲染 4 帧"能替代相机初始化 | 相机标注器在 **`reset_scene()`** 里初始化（`reset()` 只是别名） |
| 8 | 标志物轮换挂在 `reset()` 上 → 运行器调的是 `reset_scene()` → **一次都没换** | 逻辑要挂在**真正被调用的**入口上；用日志验证而非想当然 |
| 9 | 路径写错导致误删/误提交（`assets\isaac\Outdoor` vs `assets\isaac\Environments\Outdoor`）→ 仓库一度 574 MB | 删之前 **`Get-ChildItem` 确认路径**，提交前看 `git ls-files` 的体积 |
| 10 | 降采样被**一张坏图**打断整轮 | 批量处理要**逐文件 try/except** |

---

## 9. 会话进展日志（**每次做完事就追加，别只写"下一步"**）

### 2026-10-08：三个模型作废的根因 + 五处代码修复

**这一轮的经过**：15 模型 sweep 跑到第 13 个时发现后面三个（kimi、deepseek、glm）每档都是 0.000，
包括最宽档 A/S 2.0（1.14 m，早期模型 100% 通过）。查原始日志后确认是**三种完全不同的原因**，
其中两个与模型能力无关：

| 模型 | 真因 | 责任 |
| :--- | :--- | :--- |
| kimi/kimi-k2.5 | 模型名被洗成 `kimi-kimi-k2.5` → **503 model_not_found**，150 步全 invalid | **我的代码** |
| deepseek-v4.1-flash | **403 token quota is not enough** | 账户余额 |
| glm-4.6v | **401 Unauthorized: Invalid token** | key 失效 |
| claude-sonnet-4-6 | 第 16 档 150/150 invalid（其余档 0/150）→ 那个 0.0 是故障不是几何 | 待查 |

时间线：10-07 21:13 首次出现 403 配额，22:06 出现 401，之后 key 一直无效。

**检查 key 的正确方式**（`tools/check_credit.py` 已改）：它以前在 401 时也打印一堆网关响应头，
看着像"连上了"，现在先判层（key 被拒 / 额度为空 / 个别模型），且只在**真正被应答或限流**的行上
才打印 ratelimit/quota 头。手动一发最便宜请求同样有效：

```bash
curl -s -m 30 -o /tmp/gw.json -w "HTTP %{http_code}\n" -X POST http://35.220.164.252:3888/v1/chat/completions \
  -H "Authorization: Bearer $BOYUE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ok"}],"max_tokens":5}'; head -c 200 /tmp/gw.json
```

**五处修复（提交 00ad2a3）**

1. **模型名不再被清洗**：`experiments.py` 里 `self.model` 保持原样给网关用，新增 `self.model_slug`
   只给路径用。`_result_dir` 改用 slug。`test_bao_persistence.py` 原本断言 `runner.model` 不含分隔符，
   那正是 bug 被写成了需求，现在反过来断言。
2. **`effective_tag` 幂等**：带参数后缀时（`-effortnone` / `-nothinking`）以前会重复追加协议标签，
   产生 `…-stage1.1-…-stage1.1-effortnone`。现在反复剥离自己追加过的每一段，再按固定顺序重建。
   `tools/check_tags.py` 验证全部 15 条目稳定且单一后缀。
3. **坏运行会自己停**：`_note_episode_health`。全部步骤无效的回合，第一、二次打印醒目横幅并附上
   最后一条请求错误，**第三次连续出现就抛错中止该模型**（sweep 记为失败并继续下一个）。
   会答但答错的模型不受影响——那是结果，不是故障。
4. **解析宽容化**：只试第一个花括号（前面散文里有个 `{` 就全丢）、动作名必须小写完全一致、
   只认 `action` 键、`NaN` 置信度会让整条回复作废（连动作一起丢）。四条都改了，
   `test_bao_parsing.py` 9 项。丢掉一条正确回复会被记成 invalid 步并计 0 分，所以这里值得宽容。
5. **`check_credit.py` 不再误导**（见上）。

**新增测试套件**：`test_bao_parsing.py`（9 项）、`test_bao_health.py`（5 项）、`tools/check_tags.py`。
九个套件共 **151 项全过**：geometry 37 / integration 20 / persistence 13 / memory 29 /
memory_runner 17 / scenes 18 / assets 3 / parsing 9 / health 5。

**审计工具的一处修正**：`tools/check_field_names.py` 把同义词表的**键**（`move_forward`、`walk`…）
当成"协议里不存在的动作名"报错。已加豁免：字典的键、其值是合法动作名，那是别名表不是动作引用。

**这一轮做对的**：先停 sweep 再查（虽然第一次停错了，见下），先读代码不猜，用原始日志定因。

**这一轮做错的**：`pkill -f 'main.py --model'` **停不掉 sweep**——外层 `run_all_models.sh` 是循环，
子进程一死它就拉起下一个模型（豆包就是这么起来的）。必须先 `pkill -f 'run_all_models.sh'`。
已写成 `stop.sh`，用 PID 精确停止，并且会把还活着的 Isaac 进程按命令行分类（我们 / 同事的
`/workspace/isaaclab` / 未知），同事那个绝不碰。

### 2026-10-06：找到思考过程 + 场景数据层完成

**① 模型的思考过程一直在，只是我找错了地方** ✗→✓
`logs/{tag}/level{N}_episode{ID:03d}_agent.txt` 里是**每步的完整提示词 + 模型完整 JSON 回答**（`scene_description` / `reasoning` / `confidence`）✓
每个文件 30+ 处 `scene_description` ✓ 本轮 15 个模型全都有 ✓。
我错在：`main.py` 顶部注释里**写着**这个路径 ✓ 我却在 `results/` 里翻了四轮 ✓ 还错误地说"思考没存" ✗。
**教训：找东西先读 `main.py` 顶部输出结构 + 写它的那个函数。**

**② 新增 `tools/parse_agent_logs.py`** ✓ 把 `logs/` 解析成表并统计 ✓
`python3 tools/parse_agent_logs.py --csv logs_reasoning.csv` → 43,721 步 ✓ 每模型"提到自己标志物颜色/形状/开口"的比例 + 原文例子 ✓。
**已得的结论**（实验室场景 ✓）：
- 五种标志物之间的**颜色串色 = 0.0%** ✓，**形状对角线完胜**（disc 4026 / bars 3944 / triangle 3938 / cross 3928 / square 4832 ✓）→ **模型确实识别出了不同标志物** ✓✓
- 提示词替换**零残留**（`red marker` 计数 0 ✓ `reach the magenta triangle…` 73/73 ✓）✓
- 但**部分模型的推理里仍有 `red` 命名惯性** ✗（提示词已写 magenta ✓ 它仍说 red ✓）→ 这是**发现** ✓ 论文可写 ✓
- `green`（开口后的绿墙 ✓ 38–43%）与 `white`（7–9%）是**与提示词无关的场景描述** ✓ = 模型真在看画面 ✓
- **"说不说开口"差 700 倍** ✓：gpt-4o-mini 0.1% ✗ / gpt-4o 59.2% / qwen 三家 73–76% ✓ → 很可能是"是否把可通行性当问题"这条因果链 ✓
- 形状表里 `arrow` 是**常数列**（每行 5000+ ✗）→ 与标志物无关 ✓ 分析时要剔除并标注 ✓

**③ 场景数据层做完了（1.2/1.3/1.4/1.5）** ✓ 视觉层一步没做 ✗
- `test_bao_scenes.py` 18 项 ✓（几何/开口投影遮挡/颜色避让/不碰撞/伸出≤0.20 m/地面物≤1 m 高）✓
- **新增 `test_bao_assets.py` 3 项** ✓：每张面的材质必须在仓库里存在 ✓ 不许两张面共用一个材质文件 ✓ **`stage1.1` 必须仍然无材质无陈设** ✓✓（守住正在跑的那轮 ✓）
- 1.3 图书馆 → `M_Wood_Floor` / `MI_WallOffice_01` / `MI_CeilingA_06b` / `M_Wall_Plaster` ✓
- 1.4 公园 → `M_Wall_Plaster` / `MI_WallOffice_01` / `MI_CeilingA_06b` + **地面纯色** ✗（没抓到草地贴图 ✓ 待补 ✓）
- 1.5 超市 → `MI_FloorMarbleTiles_03` / `MI_WallA_01` / `MI_CeilingA_06b` / `MI_WallOffice_01` ✓
- 素材 `assets/isaac` = **136 MB / 880 文件** ✓（Simple_Warehouse + Hospital + Office ✓ 随仓库走 ✓ 克隆即可渲染 ✓）
- 宽墙件必须离中线够远 ✓ 否则**压住开口投影** ✗（图书馆书架 1.6 m @ z=1.30 → 内边缘 0.50 m < 开口半宽 0.57 m ✗ 被抓 ✓ 现改 1.4 m @ z=±1.65 ✓）

**④ 新增/修好的工具** ✓
- `tools/fetch_assets.py` ✓ 抓取+降采样；**`--list` 现在区分 mesh 与 material** ✓
  **道具是 `.usd` 不是 `.mdl`** ✗ → 扩展名过滤不含 USD 时会**只抓贴图、mesh 全跳过** ✗ 已修 ✓
- `tools/shrink_tree.py` ✓ **一张坏图不能中断整轮**（`Outdoor` 那次 3.9 GB 原图就是这么来的 ✗）
- `tools/parse_agent_logs.py` ✓（见 ②）· `tools/verify_stage1_lab.py` ✓ · `status.sh` / `watch.sh` ✓

**⑤ 我在这一轮犯的错（写下来防止重犯）**
猜 pxr API 三次 ✗ · 找错目录四轮 ✗ · 说"思考不影响结论" ✗ · 抓素材前不看体积（`Outdoor` 1.9 GB ✗）· 删目录路径写错致 574 MB 进仓库 ✗ · 用 PowerShell 字符串改 UTF-8 ✗ · 用 `head -30`/`2>/dev/null` 把报错过滤掉 ✗ · 脚本里键名不一致导致输出空表 ✗

### 下一步（**除标注外都不需要仿真器**）

1. **把陈设从"彩色方块"换成真实道具** ✓（在抓 ✓）：`scene_builder.place_dressing` 目前**只造方块** ✗，
   `asset` 字段只当注释 ✓ → 要改成：`asset` 解析到仓库内的 `.usd` 时**加引用** ✓ 并按声明的尺寸缩放 ✓（或用 `UsdGeom.Boundable` 的 extent 算缩放 ✓）✓ 抓完后做 ✓
2. **1.4 的草地** ✗：`Environments/Outdoor` 体积大 ✗ → 需要给 `fetch_assets.py` 加**按关键词过滤**（只抓 grass/ground 相关键 ✓）
3. **需要上机（等显存）**：① 二分定位 1.2 的**深色遮挡**（`--parts marker` → `--parts materials,marker` ✓ 各 4 分钟 ✓）② 修 `discover_surfaces` 认不出**天花板**（诊断代码已写好 ✓ 会打印 stage 上最大的盒子 ✓）③ 四个场景各渲一张图给我看 ✓
4. 全部验证通过后，按需跑其它场景 sweep（每个 ≈ ¥330 / 2.5 天 ✓）
