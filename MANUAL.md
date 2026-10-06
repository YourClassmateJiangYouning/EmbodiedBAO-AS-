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

网关 `http://35.220.164.252:3888/v1`，key `REDACTED`，**用 `BAO_DISABLE_PROXY=1`**。
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

## 9. 下一轮对话该做什么（随进度更新）

1. 跑 `python3 tools/parse_agent_logs.py --csv logs_reasoning.csv` → 得**每模型"看见标志物"的统计** + 原文例子。
2. 用 `summary_*.json` + CSV 出**15 模型 × 17 档通过率矩阵**、阈值表，以及**5 个标志物是否改变行为**的检验（每标志物 255 集）。
3. sweep 跑完后：解决 `stage1.2` 的**深色遮挡**（用 `--parts marker` / `--parts materials,marker` 二分 ✓ 各约 4 分钟），再渲染 1.3/1.4/1.5 验证场景。
4. 场景验证通过后，按需要跑其它场景的 sweep（每个 ≈ ¥330 / 2.5 天）。
