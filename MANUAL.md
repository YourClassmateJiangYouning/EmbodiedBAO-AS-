# 项目手册（**每一轮对话开始前先读这个文件**）

> 这是给我（AI 助手）看的操作手册，不是给外人看的介绍。项目介绍在 `README.md`。
> **每次和用户对话前先通读一遍本文件**，尤其是「数据住在哪」和「我犯过的错」两节。
>
> **本文件最后校对于 2026-10-08**，校对的依据是**代码**，不是上一版手册。如果本文件与代码冲突，
> **以代码为准**，并回来改本文件。上一版有几处已经落后于代码，见 §10「已更正的旧说法」。

---

## 0. 一句话现状

在 Isaac Sim 4.5 里让 **15 个** MLLM 驱动 Unitree H1 穿越窄通道，测 A/S 阈值
（A/S = 开口宽 / 肩宽 0.570 m）。

- **Stage 1（A/S 阈值）= 当前主战场**：**17 档**（A/S 2.0 → 0.4）、**5 个场景皮肤**、
  每场景 **25 个标志物**（5 场景 × 5）。规模 **15 × 17 × 5 × 5 = 6,375 集**。
  基线场景 `stage1.1`（无材质无陈设）的全量 sweep 正在实验室机器上跑。
  Stage 1 另有一组 **10 个单变量变体**（标记大小/形色、远墙距离、地面中线、墙高、提示词、中文），
  见 `STAGE1_SCENE_VARIANTS.md`，**未实现**。
- **Stage 2（记忆/笔记）**：代码与测试**已完成**，数据**零**（`results/memory/` 不存在）。
- **Stage 3**：`STAGE23_DESIGN.md` §10 把"回穿阶段、逆序对照"等列为**明确不做**，
  所以 Stage 3 目前**只是文档里的名字，没有代码也没有计划**。**它和场景变体矩阵是两回事。**

**术语约定（我搞错过两次，写死在这里）**：
1. **换皮肤/搭场景属于 Stage 1**，不是 Stage 3。
2. 那两个设计文档的文件名历史上带 `STAGE3_` 前缀，**已改名**：
   `STAGE3_SCENES.md` → **`STAGE1_SCENES.md`**，`STAGE3_SCENE_VARIANTS.md` → **`STAGE1_SCENE_VARIANTS.md`**。
3. `STAGE23_DESIGN.md` 名字是对的：它的 Stage 2/3 **都是记忆实验**（Stage 3 = 回穿/逆序那几个后续），
   **和场景变体无关**。我一度把变体矩阵也标成 Stage 3，那是我造的错，已改掉。

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
| **场景实际绑到了哪个 prim** | **`logs/{tag}/args.json` 旁边的 scene report**（`scene_builder.format_report`） | 场景材质/标志物/陈设各自实际解析成了什么；"这次真的用了 vendored 贴图还是退化成了纯色"只能从这里读 |
| **Stage 2 回合记录** | `results/memory/{model}/{tag}/` | 与 Stage 1 分开的前缀：`analysis.py` 只找 `results/level*`，所以永远不会把 Stage 2 记录当阈值集 |
| **Stage 2 笔记调用** | `logs/{tag}/run{R}_round{NN}_note.txt` | 提示词 + 回复 + 实际用的系统提示词，一个文件自证 |

**教训**：`main.py` 的文档字符串里**一开始就写着** `logs/{tag}/…_agent.txt  raw model I/O`。
**找任何东西之前，先读 `main.py` 顶部的输出结构说明，以及写它的那个函数，不要在 `results/` 里乱翻。**

**标志物归因规则**：环境每回合推进一个槽位。`environment.reset_scene()` 里先
`self._episode_index += 1`，再 `set_marker_slot((self._episode_index - 1) % 5 + 1)`，
**等价于 `slot = (episode_id % 5) + 1`**（已实测：两者对 episode_id 0–9 逐项相同），
对应 `scenes.describe_marker(scene, slot)`。`tools/parse_agent_logs.py` 用同一映射反推。

**提示词里的标志物名词**由 `protocol.set_marker_descriptor()` 替换（`environment.py` 的
`set_marker_slot` 里调用），短语来自 `scenes.describe_marker`。

---

## 2. 目录地图

| 路径 | 是什么 | 能改吗 |
| :--- | :--- | :--- |
| `environment.py` | Isaac 场景：房间 / 障碍墙 / 开口 / 标志物 / 相机 / 机器人 / 解析式碰撞门 | **Stage 1 冻结，只增不改** |
| `protocol.py` | 提示词与动作空间（**9 个**动作，含 `look_down`）。`build_prompt()` 里标志物名词短语会按本回合替换。**侧身角度带（45–135°）的唯一实现在这里** | 谨慎 |
| `scenes.py` | **Stage 1 场景目录，纯数据**：10 形状 / 10 颜色 / 25 标志物 / 5 场景的材质与陈设 / `describe_marker()` / `to_world`·`to_user` 坐标转换 | 可改，**但不能动 `stage1.1`**（见 §5） |
| `scene_builder.py` | 把 `scenes.py` 应用到 stage：材质解析（优先用仓库内本地文件）/ 标志物网格 / 陈设 / 表面分类。**返回 report，说明实际解析成了什么** | 可改 |
| `capture_scenes.py` | 渲染预览。`--scene --level --outdir --slots` | 可改 |
| `main.py` | 实验入口。**顶部注释写明全部输出结构** | 谨慎 |
| `experiments.py` | runner：调模型、记 history、写记录。`_append_log()` 写 `logs/{tag}/…_agent.txt`。坏运行会自己停（`_note_episode_health`） | 谨慎 |
| `ai_agent.py` | 模型客户端与响应解析（`parse_action_json` / `parse_action_text` 取出 action/reasoning/scene_description/confidence；**没有 `parse_response` 这个函数**，手册曾写错） | 谨慎 |
| `memory_metrics.py` / `memory_protocol.py` / `memory_experiment.py` | Stage 2：纯算术指标 / 跑法与提示词 / 回合循环与笔记调用 | 可改（Stage 2 未跑，改了没有数据要重跑） |
| `run_all_models.sh` | 全量 sweep 驱动（`SCENE=` 选场景；`--all-levels --episodes 5 --resume`） | 可改 |
| `run_all_models_memory.sh` | Stage 2 的 sweep 驱动 | 可改 |
| `status.sh` / `watch.sh` / `stop.sh` | 一条命令看状态 / 实时追踪 / **按记录的 PID 精确停止**（见第 6 条硬规矩） | 可改 |
| `tools/parse_agent_logs.py` | **把 `logs/` 里的思考过程解析成表 + 统计**（提到标志物颜色/形状/开口的比例） | 可改 |
| `tools/verify_stage1_lab.py` | 冒烟验收：标志物轮换 / 提示词命名 / 额度 / 坏回复 / traceback | 可改 |
| `tools/check_field_names.py`、`tools/check_names.py` | 静态审计：字段名 / 未定义名（后者在 geometry 套件里跑） | 可改 |
| `tools/fetch_assets.py`、`tools/shrink_textures.py`、`tools/shrink_tree.py` | 素材抓取 / 降采样（**坏图只跳过不中断**） | 可改 |
| `assets/isaac/` | 随仓库走的素材（**实测 899 文件 / 138.5 MB**：Simple_Warehouse + Hospital + Office） | 只增 |
| `test_bao_*.py` | 手写测试套件，**实测 9 套件共 167 项全过**：geometry 37 / scenes **32** / memory 29 / integration 20 / memory_runner 17 / persistence 13 / parsing 9 / health 5 / assets **5** | 必须全过 |
| `STAGE1_SCENES.md` | Stage 1 场景与 25 标志物的规格（原 `STAGE3_SCENES.md`） | 可改 |
| `STAGE1_SCENE_VARIANTS.md` | Stage 1 的 10 个单变量变体规格（原 `STAGE3_SCENE_VARIANTS.md`） | 可改 |
| `STAGE23_DESIGN.md` | Stage 2/3 记忆实验的**唯一权威规格** | 可改 |
| `lab_logs/` | Stage 1 分析与出图脚本、`figures/` 成品图、`bao_v7_all.tgz` 归档 | 冻结 |
| `results/`、`logs/` | 运行产物（**不入库**） | — |

---

## 3. 硬规矩（用户反复强调，违反会被骂）

1. **一切结论必须有依据**：不猜。跑代码 / 看日志 / 看图 / **读代码**。说「很可能」之前先去找证据。
2. **找东西先读代码**：文件在哪、字段叫什么、写到哪 —— `main.py` 顶部和写它的函数里都写着。
3. **给的命令必须可直接复制**：单行、无 `<占位符>`（用户会照抄，`kill <93233>` 那次就是教训）。
4. **模型原话是实验结论的一部分**，不是"定性补充"：推理文本是结果变量。
5. **不能动正在跑的场景**：`main.py` 每换一个模型就重读 `scenes.py` → 改 `stage1.1` 会让 15 个模型内部不一致。
6. **停 sweep 用 `bash stop.sh`，不要用 `pkill -f 'main.py --model'`。** 外层 `run_all_models.sh` 是循环，
   子进程一死它就拉起下一个模型（豆包就是这么起来的）。要精确匹配就先杀循环本身
   （`pkill -f 'run_all_models.sh'`）。**任何情况下不要用 `pkill -f isaac`**：共享机器，别人也在跑。
   删除/移动前先打印文件列表。
7. **PDF 里不能有中文**（matplotlib 无 CJK 字形）；中文只进 Markdown/docx。
8. **`stage1.1` 就是"原实验室、无遮挡"的基线**，用户当它是在跑的那一轮，不要装修它。
   `test_bao_assets.py` 会强制它保持无材质无陈设。
9. **结论写进文档前先跑代码验证。** 差值型/计数型断言（"公式改了"、"少了 4 次"）必须先用脚本枚举
   再落笔；本手册的 §10 记录了两次因为跳过这一步而写错，其中一次还编出了具体数字。
10. **文档与代码冲突时以代码为准**，并回来改文档 —— 手册 §10 就是为此存在的。

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

# 全部测试（9 套件，实测 167 项）
# ★ 必须用 Isaac 解释器：系统 python3 没有 Pillow，test_bao_persistence 会 FAIL（实测）
cd ~/EmbodiedBAO-AS- && for s in test_bao_geometry test_bao_integration test_bao_memory test_bao_scenes test_bao_memory_runner test_bao_persistence test_bao_parsing test_bao_health test_bao_assets; do printf '%-28s ' $s; /home/ybh/isaacsim/python.sh $s.py 2>/dev/null | tail -1; done

# 渲染预览（会再起一个 Isaac，注意显存）
cd ~/EmbodiedBAO-AS- && /home/ybh/isaacsim/python.sh capture_scenes.py --scene stage1.2 --level 10 --outdir prev_1.2 --slots 1,2,3,4,5

# 停（精确，别用 pkill -f 'main.py --model'）
cd ~/EmbodiedBAO-AS- && bash stop.sh

# Stage 2（记忆实验）：便宜的先跑，不花钱的那条不需要 key
cd ~/EmbodiedBAO-AS- && /home/ybh/isaacsim/python.sh memory_experiment.py --model random --runs 1 --headless --max_calls 40
cd ~/EmbodiedBAO-AS- && /home/ybh/isaacsim/python.sh memory_experiment.py --model qwen3-vl-32b-instruct --runs 1-6 --headless --resume
```

**实验室机器**：仓库 `/home/eai/EmbodiedBAO-AS-/`；Isaac 启动器 `/home/ybh/isaacsim/python.sh`（headless）；
用户会把图放到 `C:\Users\asus\Desktop\科研狗之人机心理学\实验图片\`，**我可以直接用 read_image 读**。

---

## 5. 几何与规模（实测，别再重算）

- `needed(θ) = 0.570|cosθ| + 0.220|sinθ|`；峰值 0.611 m @ 21.1°；**任何姿态都过不去的下限 = 0.220 m（A/S 0.386）**。
- **17 档**，宽度由 `round(0.570 × ratio, 6)` 推导（`environment.LEVEL_CHANNEL_WIDTHS`），
  实测逐档如下（"最少转身"由 `memory_metrics.min_turns` 算出，不是查表）：

  | 档 | A/S | 通道宽 | 最少转身 | 最少步数 |
  | ---: | ---: | ---: | ---: | ---: |
  | 0–10 | 2.0 → 1.0 | 1.140 → 0.570 m | 0° | 11 |
  | 11 | 0.9 | 0.513 m | **60°**（4 次） | 15 |
  | 12 | 0.8 | 0.456 m | **75°**（5 次） | 16 |
  | 13 | 0.7 | 0.399 m | **75°**（5 次） | 16 |
  | 14 | 0.6 | 0.342 m | **90°**（6 次） | 17 |
  | 15 | 0.5 | 0.285 m | **90°**（6 次） | 17 |
  | 16 | 0.4 | 0.228 m | **90°**（6 次） | 17 |

  **A/S 2.0 的通道宽 = 1.140 m**（不是 2.28 ✗ 我算错过一次）。
  **A/S 0.9 是 4 次转身（60°），不是 5 次**：45° 时身体需要 0.5586 m > 0.513 m，过不去；
  60° 需要 0.4755 m ≤ 0.513 m，能过。`STAGE1_SCENES.md` §1 的表写成 5 次（75°）✗，
  `lab_logs/verify_optimal_steps.py` 的广度优先搜索实测 **15 步**（11 forward + 4 turn）✓，与代码一致。
- 成功判定 x ≥ 8.75；步长 0.75 m；转身 15°；眼高 1.68 m / 俯仰 15° / 焦距 13.36 mm / 传感器 20.955 mm
  → **水平 FOV 76.2°、垂直 47.6°**（`scenes.HORIZONTAL_FOV_DEG`，不是"≈76° / ≈52°"）。
- 规模：15 模型 × 17 档 × 5 标志物 = **1,275 集/场景**；全 5 场景 **6,375 集**
  （`scenes.episode_count()`，测试断言等于 6375）。
- **实测成本与时间**：约 $0.036/集 → 单场景 ≈ **$46 / ¥330**；**单模型 2.5–9 小时** → 15 模型 ≈ **50–60 小时**
  （gemini-2.5-pro 单独跑了 9 小时 ✗）。

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

## 7. 场景搭建进度（Stage 1 皮肤；依据 = 代码实测，不是上一版手册）

| 场景 | 材质（代码里的条数） | 陈设 | 状态 |
| :--- | :--- | :--- | :--- |
| `stage1.1` 实验室 | **无** | **无** | **正在跑的基线，一个字都不许改** |
| `stage1.2` 仓库 | **4 条，已核对**（唯一整套借用素材的场景） | 8 件 | 有"未定位深色遮挡"待查 ✗ |
| `stage1.3` 图书馆 | 4 条，**`material_verified=False`** ✗ | 8 件 | 未渲染验证 |
| `stage1.4` 公园 | **3 条**，未核对 ✗（**没有地面材质** —— 草地贴图没抓到） | 8 件 | 未渲染验证 |
| `stage1.5` 超市 | 4 条，未核对 ✗ | 8 件 | 未渲染验证 |

- **陈设全部还是方块，不是真道具**：`scene_builder.place_dressing` 明确**不使用** `asset` 字段
  （`.mdl` 是材质不是 stage 资产，USD 回 "Cannot determine file format"），report 里每件都记 `used_asset=False`。
  换成真实道具属于 **Stage 1 的场景皮肤工作**（不是 Stage 3）；`.usd` 道具的接引用见
  `tools/fetch_assets.py` 与 `STAGE1_SCENES.md` §2.2。
- **陈设硬约束**（`test_bao_scenes.py` 强制，以测试为准）：全部 `collides=False`；
  地面物 **`|z| ≥ 1.5`**、高 ≤1.0 m（>0.6 m 时须 `|z| ≥ 1.8`）；墙挂件 `|z| ≥ 0.9`、伸出 ≤0.20 m、
  按「(横向, 纵向, 厚度)」声明；每件必须有颜色；**不得用本场景 5 个标志物之一的颜色**。
  （上一版手册把地面物写成 `|z| ≥ 1.9` ✗，代码与测试都是 1.5。）
- **布局只存在于一处**：1.1 与 1.2 内联在 `scenes.py` 的 `SCENES` 里；1.3/1.4/1.5 由
  `_install_dressing()`（模块导入时执行）赋值。**不要两处都写** —— 之前两处都有，内联那份是死的，
  改它完全无效。`test_bao_scenes.py` 现在从 AST 检查这一点。
- 宽墙件要离中线够远，否则会**压住开口投影** ✓（图书馆书架就栽在这 ✓ 现在 1.4 m 宽 @ z=±1.65 ✓）。
- 素材抓取流程：`python tools/fetch_assets.py --list Environments/Hospital` → `--set …` → `tools/shrink_tree.py`（**一张坏图不能中断整轮**）。
- **渲染前先确认显存**：实验室那台还有别人的 IsaacLab 占 5.5 GB ✗。
- **未完成的三件事**：① 11 条材质 URL 待用桶列表核对（上表 ✗ 的三行）；
  ② `tools/check_marker_uniqueness.py` **不存在**（`STAGE1_SCENES.md` §4 检查②自己承认了），
  在它出现之前，标志物唯一性只能靠几何/配置层检查 + 看图；
  ③ **陈设正在从方块换成真实 `.usd` 道具**（进行中）：`scene_builder.prop_asset_for()` 已能把目录里的
  `asset` 解析为**仓库内**的 `.usd`，`place_dressing()` 会在方块之上加引用。**已接 3 件**（均按实测尺寸 1:1）：

  | 场景 | 陈设 | 道具 | 实测尺寸（catalogue 系） |
  | :--- | :--- | :--- | :--- |
  | `stage1.2` | `pallets` | `Pallet/pallet.usd` | (1.2132, 0.1425, 0.8023) |
  | `stage1.2` | `klt_bins` | `KLT_Bin/small_KLT_visual.usd` | (0.1978, 0.1464, 0.2966) |
  | `stage1.3` | `cabinet` | `Sektion_Cabinet/sektion_cabinet_instanceable.usd` | (0.6678, 0.7861, 0.7638) |

### 7.1 道具引用的四条硬事实（2026-10-09 实测，`tools/measure_assets.py`）

| 事实 | 依据 |
| :--- | :--- |
| **几何在 `*_visual*.usd` 层** | `small_KLT.usd` 6.6 KB vs `small_KLT_visual.usd` 180 KB。接引用要指 visual 层 |
| **但 `sektion_cabinet_visuals.usd` 不可引用** | 日志：`Unresolved reference prim path <defaultPrim>`。**能用的恰是 `_instanceable` 那个**（实测 0.6678×0.7861×0.7638）。所以含 `instanceable` 的名字按原样使用 |
| **`mac_n_cheese_centered.usd` 不可引用** | 日志：`Could not load sublayer mac_n_cheese.usd; skipping` —— sublayer 兄弟文件未入库 |
| **`.mdl` 不是道具** | `MI_SignB.mdl` / `M_TrafficCone.mdl` 是材质，`prop_asset_for` 明确拒绝，这两件**按设计保持方块** |

**声明尺寸必须有出处**：`scenes.PROP_MEASUREMENTS` 记录实测值，`test_a_declared_size_is_the_size_of_the_prop_it_names`
强制"声明 == 实测"。此前两处声明都错了且**方向相反**：托盘被写小（0.9×0.9 vs 真 1.2132×0.8023），
料箱被写大（0.6×0.5 vs 真 0.1978×0.2966）。

**柜子为何在地面而不在墙上**：它深 0.6678 m，半深 0.33 m 已超 `test_no_dressing_protrudes_into_the_corridor`
的 0.20 m 上限（该上限源自旧通风管把黑块横在机器人视野中央）。**不放宽规矩、也不压扁网格**，改为地面独立家具。

新增工具：**`tools/measure_assets.py`**（量所有 vendored 资产）、**`tools/measure_props.py`**（量某场景已接道具 vs 声明）。

### 7.2 首批 1024 px 实拍审查（2026-10-08，远程渲染；完整记录见 SESSION_RECORD.md §I）

**已核实（像素实测，不是眼估）**：

| 项 | 结果 |
| :--- | :--- |
| 标志物是否在绿远墙上 | **25/25 在**。收紧色相窗后 **24/25** 的 x 中心落在 501–514，紧贴图像中轴 **512**；唯一例外 `stage1.3_slot1` 被书架等场景物体污染。宽 13–25 px、高 21–31 px，与 0.60 m / 15.48 m 下预期（≈25 px）吻合 |
| **颜色 bug 端到端验证** | `stage1.4` slot5、`stage1.5` slot2 实测**酒红**；`stage1.5` slot1 实测**亮粉**（修复前提示词会说 white/black） |
| 颜色 vs 声明 | 25 个中 **23 个通道顺序完全一致**；另 2 个是我的采样窗混入背景，收紧后亦正确 |
| 图像尺寸 | **渲染 1024×1024**，送模型前降采样到 **512**（`BAO_IMAGE_SIZE`） |

**两个现象已量化，都不是分辨率问题**：

| 现象 | 实测 | 结论 |
| :--- | :--- | :--- |
| 画面"发糊/发麻" | 噪声 `mean\|Δ邻\|`：1024 px **6.3–9.3** → 512 px **5.5–7.5**（≈2.2–2.9%） | **是渲染噪声，不是分辨率**。来源 `RaytracedLighting` + `spp=32` + 无降噪；可调 `spp`/降噪/`light_radius` |
| `stage1.2` 地面近黑 | 下半均值 **50.8** vs 基线 113.3（暗 55%） | 旧账**仍在**。**未定因**：可能 `MI_Floor_01` 本身深色，也可能回退成纯色 |

**重渲时必须 `tee prev_render.log`**：`stage1.2` 地面暗的**唯一判据**是日志里
`[scene] floor ... how=mdl-local|mdl|fallback-paint|no-surface-given` 那一行。这批图**只有 PNG、没有日志**，
所以现在**无法定因，不许猜**。

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

### 8.1 第七轮起新增的错（**这些曾只写在 SESSION_RECORD，是本手册的漏项**）

用户在 2026-10-10 指出：错误只记进了 `SESSION_RECORD.md`，`MANUAL.md §8` 长期停在上面那 10 条。
下面是从 §E / §G4 / §H5 / §I5 / §J3 / §K3 / §L4 汇总进来的部分，**按病根归类**，细节在 SESSION_RECORD 对应小节。

| # | 错 | 真相 | 怎么发现的 |
| :-: | :--- | :--- | :--- |
| 11 | 认为"陈设的放置几何"已由测试保证 | 测试比对的是**声明**，不是**放置后**的位置 | 写 STEP 2 审计时逐件算 `at`/`size` |
| 12 | 用对称性判据去判**非中心对称**形状是否"贴墙" | 43/44 件判为"偏离"，其中 **4 个是误报** | 换成质心判据后 25/25 正确 |
| 13 | 用**色相窗口**找蓝色墙上的标志物 | 把蓝色远墙本身算成了青色标志物（69025 px） | 打印命中像素数，量级不对 |
| 14 | 用 `sat>60` 当阈值 | 得到一个 700 px 的"标志物" | 同上 |
| 15 | 只找 `Assign` 节点做 AST 审计 | 实际是 `AnnAssign` → 得出"全为空"的错误结论 | 改用正确节点类型 |
| 16 | 手改 `stage1.2` 四个地面件为 y=0.0 | 那些位置本是**中心**，改成 0 把物件从 0.20 m 压到地面 | 后续脚本输出显示 base 为负 |
| 17 | 写脚本"把中心转成底面"时对 y 减 `height/2` | **helper 已经加了 `height/2`** → 抵消，出现**负底面**（`pallets −0.20`） | 恢复脚本显示 base 全部低于地板 |
| 18 | 新测试断言 `front ≤ face + clearance` | 方向反了：helper 把**中心**放在墙面，**正面按设计就该凸出半个厚度** → 14 件全部误报 | 看失败列表，全是墙件且规律 |
| 19 | 认为"量道具的比值"已经回答了问题 | 那个数只说明了**方块 == 声明**，对道具一无所知 | 读回代码发现量的是方块 |
| 20 | `UsdGeom.BoxCache` | 正确是 **`UsdGeom.BBoxCache`**；构造要 `(time, purposes)`，方法是 `ComputeWorldBound(prim)` | 上机 AttributeError；**改前查了官方文档** |
| 21 | 桶列表脚本报 "Pallet/Forklift/Food 全部 ABSENT" | **前缀多了前导斜杠**，S3 前缀不匹配任何键，**返回 200 + 0 个 CommonPrefixes 而不报错** | PowerShell 直接请求同一 URL 得 `KeyCount=22`，与 Python 矛盾 |
| 22 | 用 `delimiter=/` 查"每类有哪些 .usd" | delimiter 只返回子目录，不返回文件 | 去掉后 Pallet 列出 8 个 .usd |
| 23 | 猜 pxr API 三次（`Prim.GetReferences()` 是否存在、`Define` 返回 schema 还是 prim、同一行反复改） | 正解：`Define` 返回 **schema** → `.GetPrim()` → `prim.GetReferences().AddReference(path)` | 不确定就重写，最后用显式三步 + `except` 兜底 |
| 24 | 解析 `small_KLT_visual.usd` 时选中 `small_KLT_visual_collision.usd` | 名字已含 `visual` 时应**原样使用** | 打印解析结果 |
| 25 | 把"**更大的文件**"当作"更好的文件"，把柜子升级到 `sektion_cabinet_visuals.usd` | 该文件**没有 defaultPrim**（`Unresolved reference prim path <defaultPrim>`），**能用的恰是 `_instanceable` 那个** | 上机实测 + USD 警告；`tools/measure_assets.py` |
| 26 | 差点给 `shelf_left/right` 写 `(1.2132, 0.6678, 0.7638)` | `_dressing_wall` 约定是 **(across, tall, thick)** 且 `thick` 落 x 轴；柜子深 0.6678 → 半深 0.33 超 0.20 上限，**必然违规** | 读测试判据后**全部回退，没有硬推** |
| 27 | 差点按 `\|z\| ≥ 1.5` 放柜子 | 规矩是**底面高于 0.6 m 的地面件要 `\|z\| ≥ 1.8`**；柜子高 0.7861 → 必须 1.8 | 读 `test_decoration_is_never_collidable_and_stays_off_the_path` |
| 28 | 表里 `declared` 用**用户坐标系**，道具用**世界坐标系** | 两个坐标系互比 → 表里出现 `(0.6, 0.5, 0.3)` 而源文件是 `(0.6, 0.30, 0.5)` | 对照 `scenes.py` 源码 |
| 29 | 调 `sc.to_user_size()` | **该函数不存在**（只有 `to_user` 与 `to_world_size`） | 新守卫 `test_no_module_calls_a_scene_helper_that_does_not_exist` |
| 30 | `measure_props.py` 构造的路径含 `stage1.2` | `.` 是 **SdfPath 的属性分隔符** → 路径非法，只打一条警告就什么都没量 | 上机日志；修法是复用 `scene_builder.prim_name()` |
| 31 | 我的守卫**本身错了三次**：①取全文所有别名对每行套用 ②把**注释**当代码 ③正则数参数不认**星号解包** | 最终改为**遍历 AST 属性访问**（`Attribute(value=Name('sc'))`），注释/字符串/星号都产生不了这种节点 | 每次误报都打印出被误判的行 |
| 32 | 把核查用的临时脚本和 PNG 一起提交，**其中含一个已判定给出错误结论的工具** | 把已知会误导人的工具放进仓库，**比不放更糟** | 自己 review 提交内容时发现；已移除并加 `.gitignore` |
| 33 | 一直以"**保护已跑的 660 集基线**"为由，不肯改 `environment.GOAL_MARKER_SIZE`，还为此加了一套"覆盖机制 + 检查覆盖的测试" | 用户明确说：**那 660 集只是测试，不需要让后续构造与它同步**。于是恢复单一真值来源，删掉覆盖 | 用户指出 |

### 8.2 由上面这些错提炼的规则（**比单条错误更重要**）

| 规则 | 来源 |
| :--- | :--- |
| **判据本身要先验证**：任何"检查/审计/阈値"在使用前，先证明它**选中**的是它声称要选的东西 | 12–15、19、31 |
| **变换的代数要先写完再动手**：谁在什么时刻加/减了多少，逐位验证 | 16–18 |
| **坐标系要么都换算、要么都不换算**，绝不一边一个 | 28 |
| **外部服务的"空结果"必须先证伪再当事实**（200 + 空列表既可能是"真没有"，也可能是"查错了"） | 21、22 |
| **调用前查文档**，尤其本机跑不了的 API；并且给新写的 API 调用加**静态守卫** | 20、23、29 |
| **自己写的守卫要能抓住它存在的那个 bug**（否则等于装饰） | 14、31 |
| **"更大的/更完整的文件"不等于"更对的入口"** | 24、25 |
| **临时脚本与派生图片不入库**；已知会误导的工具绝不入库 | 32 |
| **不要拿"已经跑过的数据"当作不能改动的理由**——先问它是不是只是测试 | 33 |

---

## 9. 会话进展日志（**每次做完事就追加，别只写"下一步"**）

> **读这一节的方式**：它是**带日期的历史快照**，不是当前状态。里面的数字（测试项数、素材体积、
> 完成度）写的是**那一天**的事实，之后代码会变。**要当前值就去跑代码**。
> 下面 §10 列出了本手册已被代码推翻的说法。

### 2026-10-08（第二轮）：文件名纠错 + 场景目录里的两个真 bug

**这一轮的起因**：用户指出**换皮肤属于 Stage 1，不是 Stage 3**，我上一轮跟着文件名叫错了。

**① 两个设计文档改名**（只改文件名和标题，正文未动；git 记为 rename）：
`STAGE3_SCENES.md` → `STAGE1_SCENES.md`，`STAGE3_SCENE_VARIANTS.md` → `STAGE1_SCENE_VARIANTS.md`。
`STAGE23_DESIGN.md` 名字本来就是对的。所有引用点（`scenes.py` / `scene_builder.py` / `tools/check_field_names.py`）已同步。

为什么 `SCENE_VARIANTS` 也要改：它列的 10 个变体全是**标记大小/形色、远墙距离、地面中线、墙高、
提示词措辞与语言**，都是 Stage 1 阈值研究的自变量。
**它不该叫 Stage 3**：`STAGE23_DESIGN.md` §10 里的 Stage 3 指的是记忆实验的回穿/逆序后续，
且被列为"明确不做"。我一度把两者都叫 Stage 3，那是我造的错，已改掉。

**② 找到并修掉一个会进提示词的真 bug**：`scenes.COLOUR_NAMES` 里
`w`（RGB 0.55, 0.05, 0.25 酒红）写成 **"white"**、`k`（1.00, 0.40, 0.70 亮粉）写成 **"black"**。
`describe_marker` 生成的短语直接进任务句（"reach the … on the far wall"），所以有 **3 条假提示词**：
`white set of two bars`(1.4 slot5)、`white arrow`(1.5 slot2)、`black hexagon`(1.5 slot1)。
**没有任何测试读过渲染出的短语**，唯一性测试比的是 RGB 三元组，所以一直没被抓到。
已修，并加了两条测试；**实测回退到旧值能被抓住**。`tools/parse_agent_logs.py` 的同错颜色表一并修正。

**③ 24 行死配置**：`scenes.py` 里 1.3/1.4/1.5 的内联 `dressing` 被 `_install_dressing()`
（导入时执行）覆盖，改它完全无效。实测（AST 比对）确认后删除，改为指向唯一来源；
新增测试从 AST 检查"一个场景的布局只存在于一处"。（我第一次写这个审计脚本时只找了 `Assign`，
实际是 `AnnAssign`，于是得出"全为空"的错误结论，改用正确写法后才拿到真结果。）

**④ 我自己在文件里犯的编辑错误**：删 `_install_dressing` 文档尾的换行导致两行粘连、以及误改了
`material_verified` 的值。两处都是**读回文件时发现**的。教训：`edit` 之后必须读回。

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
当时九个套件共 **151 项全过**：geometry 37 / integration 20 / persistence 13 / memory 29 /
memory_runner 17 / scenes 18 / assets 3 / parsing 9 / health 5。
（**当前值是 159 项，scenes 已增至 26** —— 见 §2 与 §10，那里的数字以代码为准。）

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
- `test_bao_scenes.py` 当时 18 项 ✓（**现在 24 项**）（几何/开口投影遮挡/颜色避让/不碰撞/伸出≤0.20 m/地面物≤1 m 高）✓
- **新增 `test_bao_assets.py` 3 项** ✓：每张面的材质必须在仓库里存在 ✓ 不许两张面共用一个材质文件 ✓ **`stage1.1` 必须仍然无材质无陈设** ✓✓（守住正在跑的那轮 ✓）
- 1.3 图书馆 → `M_Wood_Floor` / `MI_WallOffice_01` / `MI_CeilingA_06b` / `M_Wall_Plaster` ✓
- 1.4 公园 → `M_Wall_Plaster` / `MI_WallOffice_01` / `MI_CeilingA_06b` + **地面纯色** ✗（没抓到草地贴图 ✓ 待补 ✓）
- 1.5 超市 → `MI_FloorMarbleTiles_03` / `MI_WallA_01` / `MI_CeilingA_06b` / `MI_WallOffice_01` ✓
- 素材 `assets/isaac` = **136 MB / 880 文件**（当时实测；**现在实测 899 文件 / 138.5 MB**）✓（Simple_Warehouse + Hospital + Office ✓ 随仓库走 ✓ 克隆即可渲染 ✓）
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

---

## 10. 已更正的旧说法（**这一节是防记忆污染用的**）

2026-10-08 把本手册逐条对过代码，以下是**上一版写错、现已改正**的说法。
如果你在别处（旧对话、旧提交信息、别的 md）看到左边这列，**它是错的**：

| 旧说法（错） | 代码/实测（对） | 依据 |
| :--- | :--- | :--- |
| 换皮肤 / 场景构建属于 **Stage 3** | 属于 **Stage 1**（A/S 阈值研究） | 用户更正；`scenes.py` 头注释 |
| 场景变体矩阵是 **Stage 3** | **错，也是我造的错**。变体属于 **Stage 1**；`STAGE23_DESIGN.md` §10 里的 Stage 3 是记忆实验的回穿/逆序后续，且明确不做 | `STAGE23_DESIGN.md:492-501` |
| 文档名 `STAGE3_SCENES.md` / `STAGE3_SCENE_VARIANTS.md` | 已改名为 **`STAGE1_SCENES.md`** / **`STAGE1_SCENE_VARIANTS.md`** | `git mv`，引用点已同步 |
| `ai_agent.parse_response` 取出 action/reasoning | **没有这个函数**。实际是 `parse_action_json` / `parse_action_text` | `grep 'def parse_' ai_agent.py` |
| 陈设地面物 `\|z\| ≥ 1.9` | **`\|z\| ≥ 1.5`**（>0.6 m 高时须 `\|z\| ≥ 1.8`） | `test_bao_scenes.py` 的断言 |
| `test_bao_scenes.py` 18 项、合计 151 项 | **scenes 26 项，9 套件合计 159 项**（`grep -c '^def test_' test_bao_*.py` 求和） | 实测跑完九个套件 |
| 素材 136 MB / 880 文件 | **138.5 MB / 899 文件** | `Get-ChildItem -Recurse assets/isaac` |
| A/S 0.9 需 5 次转身（75°） | **4 次（60°）**。45° 需 0.5586 m > 0.513 m 过不去；60° 需 0.4755 m 能过 | `memory_metrics.min_turns(0.513)==4`；`lab_logs/verify_optimal_steps.py` BFS 实测 15 步 |
| 垂直 FOV ≈52° | **47.6°**（水平 76.2°；焦距 13.36 mm / 传感器 20.955 mm） | `scenes.VERTICAL_FOV_DEG` |
| 动作 8 个 | **9 个**（含 `look_down`） | `protocol.ACTIONS` |
| 一台机器上"深度 12 档 / 11 模型 / 每模型 60 集" | 那是 **v7 归档**的历史事实（`lab_logs/bao_v7_all.tgz`，660 集）。**当前代码是 17 档 / 15 模型 / 每模型每场景 85 集** | `main.py` 的 `PROTOCOL_LEVELS`；`models.json` |
| 标志物 slot 公式与代码"不一致" | **两者恒等**：`((i+1)-1)%5+1 ≡ i%5+1`。这条是我自己一度写错的判断，已实测推翻 | 对 episode_id 0–9 枚举比对 |

**我写这些错误时的共同点**：都是**没跑代码就写结论**。§3 第 9 条硬规矩就是为此加的。
