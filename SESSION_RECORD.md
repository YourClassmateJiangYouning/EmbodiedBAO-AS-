# 对话记录与查证台账

> 用途（准则 7）：每一轮对话**发现的问题**与**已查证的事实**都写在这里，避免上下文丢失后错乱。
> 规则：只写**有依据**的条目，每条注明依据（文件:行 / 命令输出 / git 记录）。禁止写入推测。
> 最后更新：2026-10-08 会话（代码结构梳理 + 内部清理）

---

## A. 本轮（2026-10-08）代码清理：已改动的文件与理由

改动前基线：9 个测试套件共 **154 项全过**（实测：assets 3 / geometry 37 / health 5 /
integration 20 / memory 29 / memory_runner 17 / parsing 9 / persistence 13 / scenes 21）。
改动后同一命令复测：**9 套件全过，0 失败**；`python tools/check_names.py` 输出
`no unbound names in 10 files`。

| # | 文件 | 改动 | 依据 |
| :-: | :--- | :--- | :--- |
| 1 | `experiments.py` | `__init__` 新增 `self.history`，`_run_episode` 把 agent 的 history 绑上去 | `_note_episode_health` 读 `getattr(self, "history", [])`，而 runner 从未定义该属性 → 分支恒为空。测试没暴露是因为 `test_bao_health.py:37` 的 stub 自己设了 `history` |
| 2 | `ai_agent.py` | `_AGENT_CACHE` 注解 `Tuple[str,str,str,str]` → 五元组 | `create_agent` 实际用 5 个元素建键（4 个 env + `str(log_file)`），实测 `typing.get_args` 得 4 |
| 3 | `experiments.py` | 删掉循环外重复的 `self._save_summary` | `run_level` 内 `_save_summary` 出现 2 次，第二次内容是最后一次 episode 的重复 |
| 4 | `main.py` | `_export_csv` 结果记入 `csv_paths`，Level 结束不再重复导出；保留"整档被 resume 跳过"时的补写 | `run_experiment` 内 `_export_csv` 出现 3 次（闭包 + 回调 + 循环尾），回调每次 episode 已写 |
| 5 | `experiments.py` | 删除本文件内 `parse_action_text`，改调 `ai_agent.parse_action_text` | 两份 AST 结构相同（实测 `shape()` 比较为 True），字节数 17 vs 21 行 |
| 6 | `protocol.py` | 新增 `SIDEWAYS_YAW_MIN_DEG/MAX_DEG`、`fold_yaw()`、`is_sideways_yaw()` 作为唯一实现 | 原本三份：`experiments._abs_yaw`、`memory_experiment._abs_yaw`（与前者逐字节相同）、`environment.get_abs_torso_rotation`；角度带 45/135 另有三处硬编码 |
| 7 | `experiments.py` | `_abs_yaw` / `_is_sideways_yaw` 改为从 `protocol` 导入的别名 | `test_bao_memory.py:1243` 用 `experiments._is_sideways_yaw` 逐 0.1° 比对 Stage 2 谓词，名字必须保留 |
| 8 | `environment.py` | 重新导出 `protocol` 的角度带；`get_abs_torso_rotation`/`is_sideways` 委托 | `test_bao_memory.py:1231` 从 `environment` 导入这两个常量 |
| 9 | `analysis.py` | `_sideways_yaw` 改为 `protocol.is_sideways_yaw` 的别名，删除硬编码 45/135 | `analysis.py` 原为 `return 45.0 <= yaw <= 135.0`（独立第三份） |
| 10 | `memory_experiment.py` | 删除 `_abs_yaw`，改用 `memory_metrics.fold_yaw` | 两者逐字节相同 |
| 11 | `memory_metrics.py` | **刻意保留**自有 45/135 与 `fold_yaw` | `test_bao_memory.py:1154` 断言该模块只 import `math`/`typing`；`:1168` 断言它没有 `np`。导入 protocol 会破坏"SimulationApp 之前可加载"的契约 |
| 12 | `ai_agent.py` / `experiments.py` | 删除全链路 `options` 参数 | `AgentAPI.get_action` 的 docstring 自己写明"does **not** inject them"；`protocol.build_prompt` 已把动作菜单渲染进 prompt |
| 13 | `ai_agent.py` | 删除 `build_prompt` 包装函数 | 全仓库无调用者（grep 仅命中自身）；它另建一份 state 字典，与 `protocol.build_prompt` 重复 |
| 14 | `environment.py` | `_oriented_rects_overlap` 四段 → 一个 `for axis in (...)` | **实测等价**：3375 组姿态对旧实现逐位比较，0 处不一致 |
| 15 | `environment.py` | `_check_room_boundary` 抽出 `half_along_body` / `half_across_body` | **实测等价**：245 个位姿比对，0 处不一致 |
| 16 | `environment.py` | `_set_standing_joint_targets` 五个分支 → 规则表 | 五个分支都只做"记 index + 记角度"；嵌套深度 6 → 5 |
| 17 | `main.py` | `save_episodes_csv` 20 个字段名从两个元组推导 | 原名在表头列表、字典键、取值名各写一遍 |
| 18 | 多处 | 清理未使用导入（含 `test_bao_memory.py` 未定义的 `Dict`） | `ruff check --select F,E9` 输出 |

**未处理、留给用户决定**：
- `results/` 与 `lab_logs/extracted/results/` 各 346 条 episode，README 已声明二者逐字节相同（`README.md:679`）。是否删除未获授权，未动。
- `tools/` 下 17 条 ruff 提示（未使用导入等），未动：不在本轮"核心模块"范围。

---

## B. 已查证的事实（含与文档的不一致）

### B1. 数据规模

| 事实 | 数值 | 依据 |
| :--- | :--- | :--- |
| 已入库的 v7 归档 | 660 条 episode + 660 条 sidecar，11 个模型 × 60 | 解包 `lab_logs/bao_v7_all.tgz` 实测 |
| v7 归档档位 | 只有 `level0`–`level11`（每档 55 条） | 同上；即 A/S 2.0–0.9 的 12 档 |
| 本机 `results/` | 346 条 episode，无 v7 记录 | `Get-ChildItem -Recurse results` 实测；最新记录 2026-09-25 |
| Stage 2 数据 | **不存在** `results/memory/` | 实测该路径不存在 |
| 代码支持的档位 | **17 档（level 0–16）** | `main.py:315` `PROTOCOL_LEVELS = tuple(range(17))`；`main.py:351` help 文本 "0-16" |
| 名册规模 | **15 个模型** | `models.json` `models` 数组实测 15 项 |

### B2. 文档与代码的不一致（**发现的问题**）

| # | 位置 | 文档写的 | 代码实际 | 说明 |
| :-: | :--- | :--- | :--- | :--- |
| D1 | `main.py:5` | `--all-levels  # the 12-width A/S series` | `PROTOCOL_LEVELS = range(17)` | 模块自己的 docstring 过时 |
| D2 | `run_all_models.sh:6` | "60-episode ceiling per model -- the 12-width A/S series" | 17 档 × 5 = 85 | 脚本注释过时；实际跑几档由 `main.py --all-levels` 决定 |
| D3 | `README.md` 通篇 | 12 档 / 60 集 / 11 模型；无任何场景皮肤（`stage1.2`–`stage1.5`）内容 | 代码有 5 场景、25 标志物、17 档；`models.json` 15 模型 | README 最近未随扩档更新（`README.md` 全文 grep `stage1.[2-5]`、`Stage 3` 零命中） |
| D4 | ~~`MANUAL.md:31` slot 公式与代码不符~~ | — | — | **本条为我的误判，已推翻（见 E 节）。实测两者恒等。** 保留删除线是为了记住这个错误 |
| D5 | `MANUAL.md:55` | 测试项数 geometry 37 / integration 20 / persistence 13 / memory 29 / memory_runner 17 / scenes 18 / assets 3 = 137（另处写 151） | 实测 scenes **21**，合计 **154** | 项数已增长 |
| D6 | `MANUAL.md:114` | "15 模型 × 17 档 × 5 标志物 = **1,275 集/场景**" | 15×17×5 = 1275 ✓（该条正确） | 记录以免误判 |

### B3. 坐标系（已逐条对上代码，供后续建模直接引用）
- **user 坐标系**（`environment.py` 模块头部注释）：`x` 沿走廊、`y` 高度、`z` 横向。见 `environment.py:5-8`。
- **world 坐标系**（Isaac stage）：Z-up，`x` 沿走廊、`y` 横向、`z` 高度。见 `scenes.py:331`。
- **转换函数**（`scenes.py:348-363`）：`to_world((x,y,z)) -> (x,z,y)`；`to_user` 同式反用；`to_world_size` 交换后两分量。`environment._user_to_isaac_pos` / `_isaac_to_user_pos` 是同一操作（`environment.py:402-411`）。
- **横向符号**：`environment.py:378` `HEADING_RIGHT = [0,0,+1]`，即 **+z 为机器人右手方向**。
  由此：README `README.md:57` 写 "z : lateral"（未给符号）与 MANUAL「z 正 = 左」并无直接冲突（MANUAL 那句话主语是 H1 的 URDF/几何约定），但**两者表述不一致**，引用时必须以 `HEADING_RIGHT` 为准。
- **动作空间平移**：`_HEADING_DELTAS`（`environment.py:380-385`）只含 forward/backward/left/right，与 torso yaw 无关（walking frame）。
- **身体碰撞盒**：`environment.py:504`、`574-580`；yaw 旋转用 `np.radians`，`_rotate_xz`（`environment.py:340-344`）绕 +y 轴。
- **注意 `main.py` 里的重复常量**：`main.py:315` 独立写 `tuple(range(17))`，注释说明不能 import `environment`（会提前锁死 `_HAS_ISAAC_SIM`）。这是**有意的重复**，`test_bao_geometry.py` 钉住一致性。

### B4. 代码分层（以代码为准）

| 层 | 文件 | 是否 import `environment` | 依据 |
| :--- | :--- | :--- | :--- |
| 纯文本协议层 | `protocol.py` | 否（只 import `typing`） | `protocol.py:31-33`；`test_bao_memory.py:1199-1216` 拿它与其它模块一起做图遍历 |
| 纯算术层 | `memory_metrics.py` | 否（只 `math`/`typing`） | `test_bao_memory.py:1154` 白名单断言 |
| 持久化层 | `persistence.py` | 否（`json`/`os`/`time`） | `persistence.py:23-28` |
| 仿真层 | `environment.py` | 自身 | 顶层 try import isaacsim |
| Stage 1 运行器 | `experiments.py` | **是**（顶层） | `experiments.py:69` |
| Stage 2 运行器 | `memory_experiment.py` | 否（延迟） | `test_bao_memory.py:1174` 图遍历断言 |
| 入口 | `main.py` | 否（延迟到 `run_experiment` 内） | `main.py:686-688` |

---

## C. 待用户确认的未决问题（不擅自处理）

1. `main.py:5` 与 `run_all_models.sh:6` 的 "12-width/60-episode" 注释是否改为 17 档/85 集。
2. `README.md` 是否补 Stage 3（`stage1.2`–`stage1.5`、15 模型、17 档）。
3. `MANUAL.md:55` 与 §9 的测试项数（scenes 18、合计 137/151）是否更新为实测的 scenes 21、合计 154。
4. `results/` 与 `lab_logs/extracted/results/` 的 346 条重复记录是否清理（README 已声明二者逐字节相同）。
5. **（F 节新增）** `STAGE3_SCENES.md` / `STAGE3_SCENE_VARIANTS.md` 是否改名为 `STAGE1_*`，以及正文标题里的 "Stage 3" 是否跟着改。
6. **（F 节新增）** 11 条 `material_verified=False` 的材质 URL 何时用桶列表核对。
7. **（F 节新增）** 是否补 `tools/check_marker_uniqueness.py`（`STAGE3_SCENES.md` §4 检查②自己承认尚未实现）。

---

## E. 我在本轮犯的错（准则 4 / 10，必须留痕）

| # | 错误 | 真相 | 我怎么发现的 | 教训 |
| :-: | :--- | :--- | :--- | :--- |
| E1 | 在 `SESSION_RECORD.md` 初稿里断言 `MANUAL.md:31` 的 slot 公式与 `environment.py:2037-2039` **结果不同**（并编造了 "episode_id=4 → 手册 5 / 代码 4" 这个具体例子） | **两者恒等**：`((i+1)-1) % 5 + 1 ≡ i % 5 + 1`。实测 episode_id 0–9 逐项一致，无一例外 | 写下断言后按要求实测（`tmp_slot_check.py`），结果全为 "yes"，随即推翻并更正 D4 | 看到"先 +1 再 -1"就以为语义变化，是**没有动笔算**。差值型断言必须先用代码枚举验证，再写进记录 |
| E2 | 上一轮清理时把 `memory_experiment.py` 的模块级 `import sys` 删掉过一次 | 该文件 `main()` 的 `except BaseException` 分支里用到 `sys.stdout.flush()`；模块级那个是**冗余**（局部另有 import），删模块级后靠局部 import 仍可运行，但我在删除前没有确认局部 import 的存在，属于顺序颠倒 | 复查 `Select-String '\bsys\b'` 看到 845 行有局部 `import sys`，确认两条路径都不会 `NameError` | 删导入前先列出该名字的**全部**使用点及其作用域，再决定删哪一个 |

**未犯但差点犯的**：E1 的错误例子若未实测就交付，会直接违反准则 10。此处记录以示该流程有效。

---

## F. 第二轮：Stage 1 场景（皮肤）工作 —— 术语更正与已修问题

### F0. 术语（用户更正，以此为准）

皮肤/场景构建属于 **Stage 1**（A/S 阈值研究），**不是 Stage 3**。Stage 3 是场景变体矩阵
（`STAGE3_SCENE_VARIANTS.md`：标记变小/换形色/远墙更近/中线/墙高/提示词/中文等 10 个变体）。
`STAGE3_*.md` 这两个**文件名**是早先归档时的误称，代码与测试现已统一写 Stage 1。

### F1. 查证到的场景构建要求（以代码为权威，文档仅作设计记录）

| 项 | 规格 | 依据 |
| :--- | :--- | :--- |
| 规模 | 5 场景 × 17 档 × 5 标志物 × 15 模型 = **6,375 集** | `scenes.py:577` `episode_count()`；`test_bao_scenes.py` 断言 == 6375 |
| 5 场景 | `stage1.1` 实验室（基线）/ `1.2` 仓库 / `1.3` 图书馆 / `1.4` 公园 / `1.5` 超市 | `scenes.py:300` `SCENE_ORDER` |
| 标志物共同规格 | 外接 **0.60 m**、x=16.0、z=0、中心高 **1.40 m**、厚 0.02 m、不参与碰撞 | `scenes.py:39-43` |
| 25 个标志物 | 每场景 5 个，同场景 5 色互不相同，全局 25 个 (形状,颜色) 组合不重复 | `scenes.py:135`；`test_bao_scenes.py:30` |
| 10 形状 | 全部**显式多边形网格**（USD `Cylinder` 无法指定边数，故三角形/六边形不能用图元做） | `scenes.py:58-64` |
| 形状粗细 | 每个多边形面内最小边 ≥ 0.20×0.60 = **0.12 m**（512 px 起点约 20 px，细则糊） | `test_bao_scenes.py:85` |
| 4 张面材质 | `materials` 键 = floor / side_wall / ceiling / far_wall | `scenes.py` 各场景；`scene_builder.apply_scene` |
| 陈设硬规则 | ① `collides=False` ② 遮挡墙挂件 \|z\|≥0.9 ③ 地面物 \|z\|≥1.5、高≤1.0（>0.6 时须 \|z\|≥1.8）④ 颜色不得等于本场景标志物颜色 ⑤ 布局冻结，不逐集随机 | `test_bao_scenes.py:135-174` |
| 不遮挡 | 陈设投影包围盒不得与开口或标志物屏幕区域相交（起点姿态 × 17 档） | `test_bao_scenes.py:97` |
| 基线冻结 | `stage1.1` 必须无材质、无陈设、首个标志物为红方块 | `test_bao_scenes.py:177` |
| 坐标系 | 目录用 user 帧 `(x, 高度, 横向)`；stage 是 Z-up `(x, 横向, 高度)`；`to_world`/`to_user` 交换后两分量 | `scenes.py:348-357` |
| 标签含场景 | `effective_tag(model, tag, scene)` 幂等，防两个场景写进同一目录 | `main.py:124`；`test_bao_scenes.py:264` |

### F2. 本轮修掉的问题

| # | 问题 | 影响 | 依据 / 验证 |
| :-: | :--- | :--- | :--- |
| 1 | `scenes.COLOUR_NAMES` 里 `w`(0.55,0.05,0.25 酒红) 写成 **"white"**、`k`(1.00,0.40,0.70 亮粉) 写成 **"black"**、`l` 写成 "lime green" | `describe_marker` 生成的短语**直接进入任务句**，导致 3 条**假提示词**：`white set of two bars`(1.4 slot5)、`white arrow`(1.5 slot2)、`black hexagon`(1.5 slot1) | 实测 `tmp_colour_audit.py` 输出 25 条短语对照设计表；修后加两条测试，并实测**回退到旧值能被抓住** |
| 2 | `scenes.py` 中 1.3/1.4/1.5 的内联 `dressing` 被 `_install_dressing()`（`scenes.py:574` 导入时执行）覆盖 | **24 行死配置**，读起来就是活布局，改它完全无效 | AST 比对（`tmp_dressing_audit.py`）：1.3/1.4/1.5 运行时值 ≠ 内联值；1.2 相同故保留内联 |
| 3 | `_install_dressing` 内两个裸字符串（原"docstring"位置错位） | 不是 docstring，被 Python 编译后丢弃，读起来像在注释代码 | 改为 `#` 注释 |
| 4 | `tools/parse_agent_logs.py:67` 的 `COLOURS` 与 `scenes.COLOUR_NAMES` 同错 | 分析脚本会去找提示词里从未出现过的 "white" | 同步修正为 wine/pink/lime |
| 5 | 14 处 "Stage 3" 指代场景工作 | 术语与 Stage 编号冲突 | 改为 Stage 1，`scenes.py` 头注明编号分工 |
| 6 | `scenes.py:233` 中英混杂 `whose材料 were` | — | 改为 `whose materials were` |
| 7 | 我自己的编辑失误：`_install_dressing` 文档尾删换行导致 27/28 行粘连 | 文档结构损坏 | 读回文件发现并修复；**记录以示 edit 后必须读回** |

### F3. 仍未完成（代码实测状态，非文档声称）

| 缺口 | 实测 | 依据 |
| :--- | :--- | :--- |
| 材质未核对 | `material_verified=False`：1.3、1.4、1.5 共 **11 条材质 URL** 待用桶列表核对 | `scenes.py` 各场景；实测打印 |
| 唯一性脚本不存在 | `tools/check_marker_uniqueness.py` **实测不存在**（该文档已自行更正过这一条） | `Test-Path` 输出 False |
| 陈设**素材**未生效 | 32 件陈设全部是**方块**；`place_dressing` 明确不使用 `asset`（`used_asset=False`）。`.mdl` 是材质不是 stage 资产，2 个真实 stage 资产未接入 | `scene_builder.py:224-253` |
| 渲染验证未做 | 1.3/1.4/1.5 未渲染；1.2 曾有"未定位深色遮挡" | `MANUAL.md` §7（该节已落后于 10-08 的 6 个提交） |

### F4. 未获授权、我没有动的东西

- ~~`STAGE3_SCENES.md` / `STAGE3_SCENE_VARIANTS.md` 两个**文件名**未改~~ → **已获授权改名，见 G 节**。
- `MANUAL.md`、`README.md` 未改 → **已获授权更新，见 G 节**。

---

## G. 第三轮：文件名纠错 + README/MANUAL 更新（用户授权）

### G1. 改名（`git mv`，git 记为 rename，正文内容保留）

| 原名 | 新名 | 理由 |
| :--- | :--- | :--- |
| `STAGE3_SCENES.md` | **`STAGE1_SCENES.md`** | 场景皮肤属于 Stage 1 |
| `STAGE3_SCENE_VARIANTS.md` | **`STAGE1_SCENE_VARIANTS.md`** | 10 个变体（标记大小/形色、远墙距离、中线、墙高、提示词、中文）全是 Stage 1 自变量 |
| `STAGE23_DESIGN.md` | **不改** | 名字正确：它的 Stage 2/3 **都是记忆实验** |

**关键的查证（推翻了我自己上一轮的命名）**：`STAGE23_DESIGN.md` §10「明确不做的事」列出
**回穿阶段（探针后再回窄通道）、逆序对照（先宽后窄）**——这才是该文件标题里的 Stage 3，
**与场景变体矩阵无关**。我上一轮把变体矩阵也标成 Stage 3，那是**我造的错**，已全部改掉。

引用点已同步：`scenes.py`、`scene_builder.py`、`tools/check_field_names.py`、两份 md 的交叉引用。

### G2. 文档更新的依据（全部实测，非回忆）

| 项 | 实测值 | 怎么得到的 |
| :--- | :--- | :--- |
| 档位 | **17 档**，A/S 2.0→0.4 | `environment.LEVEL_CHANNEL_WIDTHS` 逐档打印 |
| 每档宽度 | 由 `round(0.570 × ratio, 6)` 推导 | 同上 |
| 每档最少转身/最少步数 | 见 MANUAL §5 表 | `memory_metrics.min_turns` / `optimal_steps` 逐档计算 |
| **A/S 0.9 是 4 次（60°）不是 5 次（75°）** | 45° 需 0.5586 m > 0.513 m ✗；60° 需 0.4755 m ✓ | `needed_width` 逐 15° 打印 |
| 模型名册 | **15 个**，已排除 **25 条** | `models.json` |
| 场景 | 5 个，材质条数 **1.1:0 / 1.2:4 已核对 / 1.3:4 未核对 / 1.4:3 未核对 / 1.5:4 未核对** | `scenes.SCENES` 逐场景打印 |
| 陈设 | 每场景 8 件（1.1 为 0），共 32 件，**全部是方块** | 同上 + `scene_builder.place_dressing` |
| 25 条提示词短语 | 逐条打印核对 | `describe_marker` |
| 垂直 FOV | **47.6°**（水平 76.2°） | `scenes.VERTICAL_FOV_DEG` |
| 素材体积 | **899 文件 / 138.5 MB** | `Get-ChildItem -Recurse assets/isaac` |
| 测试项数 | **9 套件 157 项全过** | 逐个 `def test_` 计数并求和；scenes 的 24 项**全部注册**在 `main()` 里（AST 核对） |

### G3. 本轮在文档里改正的错误说法

`README.md`：开篇"1.58 下到 0.79"→17 档；12 档表→17 档表 + 每档最少转身/最少步数；
"Eight discrete actions"→**Nine**；`z : lateral` 补上 **+z = 机器人的右**并加两坐标系说明；
新增"Five scene skins, twenty-five markers"整节（含量化统计后果）；Protocol 12 档/60 集→17 档/85 集；
Files 表补 `scenes.py`/`scene_builder.py`/`capture_scenes.py`/`test_bao_scenes.py`/`test_bao_assets.py`/
`status.sh`/`watch.sh`/`stop.sh`/`MANUAL.md`/两份 STAGE1 文档；"11 models"→15；
"12 Levels × 5"→"17 Levels × 5"；checkpoint 60→85；Testing 段 5 套件→9 套件（含实测项数）；
Known issues 新增 5 条（11 条材质未核对、陈设是方块、无唯一性脚本、非基线场景未渲染）；
Out of scope 补 Stage 1 变体集。

`MANUAL.md`：§0 现状与术语（写明我搞错过两次）；§1 数据位置补场景 report / Stage 2 路径 / 标志物
slot 实测；§2 目录地图补 6 个文件、修 `ai_agent.parse_response`（**该函数不存在**，实为
`parse_action_json`）、素材体积、测试项数；§3 硬规矩 6 改为用 `stop.sh`，新增第 9、10 条；
§4 命令速查补全 9 套件、Stage 2 命令、渲染参数；§5 换 17 档完整表并修正 A/S 0.9 与垂直 FOV；
§7 场景进度改为代码实测状态（材质核对情况、陈设是方块、布局只存一处、两项未完成）；
§9 会话日志加"这是历史快照"的读法说明；**新增 §10「已更正的旧说法」**（11 条对照表）。

### G4. 我在本轮犯的错（留痕）

| # | 错误 | 真相 | 怎么发现 |
| :-: | :--- | :--- | :--- |
| G4-1 | 把场景**变体矩阵**也标成 Stage 3 | `STAGE23_DESIGN.md` §10 里的 Stage 3 是记忆实验的回穿/逆序后续，且"明确不做"。变体属于 Stage 1 | 写 `scenes.py` 头注释时发现该文件自称覆盖 Stage 3，去查 §10 才发现撞名，随后全局改正 |
| G4-2 | 在 README/MANUAL 写"合计 **156** 项" | 实测 **157**（37+29+24+20+17+13+9+5+3） | 最后统计时按套件求和，发现与 156 差 1；再逐文件 `grep -c '^def test_'` 复核得 157 |
| G4-3 | 编辑 `scenes.py` 时误改 `material_verified` 的值、并删掉一个换行导致两行粘连 | 已恢复原值与换行 | 每次都读回文件，两处都是读回时发现 |

**共同教训**：G4-1 是**命名前没查该名词在别处的既定含义**；G4-2 是**算总数时凭记忆相加没复核**。
两条都已写进 MANUAL §3 的第 9 条硬规矩。

---

## H. 第四轮：场景构建（Stage 1 皮肤）—— 分步进行，每步记录并自审

### H0. 本机能力边界（决定了任务怎么分解，实测）

| 能力 | 状态 | 依据 |
| :--- | :--- | :--- |
| `isaacsim` / `pxr` / `omni.kit` | **全部不可用** | 三次 `python -c "import …"` 均 ModuleNotFoundError |
| NVIDIA 素材桶 | **可达** | `Invoke-WebRequest` → **HTTP 200**，2249 字节 |
| Isaac 启动器 | 在实验室机器 `/home/ybh/isaacsim/python.sh` | MANUAL §4 |

**结论**：**渲染/真正构建 USD 场景必须上机**（`capture_scenes.py` 要 `SimulationApp`）。
本机能做且可验证的是：几何与坐标审计、材质 URL 核对与下载、纯算术测试、
以及**不需要渲染就能证明的等价性**。下面按这个边界推进。

### H1. STEP 1：stage1.1 标志物与冻结基线的一致性 —— 发现并修掉一个真差异

**做法**：把两条构建路径**从源码读出**并算出各自的世界坐标包围盒（纯算术，不需要 Isaac Sim）：
路径 A = `environment._create_goal_marker` → `_add_box`（8 角点）；路径 B = `scene_builder._prism` → `build_marker`。

**远墙几何**（`environment._create_room` 的 `room_far`）：中心 `ROOM_LENGTH_X + thickness/2` = 16.01，厚 0.02
→ **内表面正好在 x = 16.00**。

**审计结果（修复前）**：

| | 路径 A 环境 | 路径 B scene_builder |
| :--- | ---: | ---: |
| x 范围（user 帧） | 15.97 – 15.99 | 15.99 – **16.01** |
| 可见正面 x | 15.99 | 16.01（**后移 0.02 m**） |
| 与墙关系 | 完全在墙前，留 0.01 m 间隙 | **埋进墙内 0.01 m**（占其 0.02 m 厚度的一半） |

**根因**：`scenes.MARKER_X_M` 写成 `16.0`（= 远墙平面），而 `_add_box`/`_prism` 都是**以该坐标为几何中心**。
环境自己的中心线是 `ROOM_LENGTH_X - thickness/2 - 0.01`，即**留 0.01 m 间隙**，那个 `- 0.01` 就是为此存在的。

**为什么没被测出来**：`test_bao_scenes.py` 只断言 **声明坐标**（`MARKER_X_M == ROOM_LENGTH_X`），
两条路径声明都对，差别在**实际几何**上，所以断言通过而几何不一致。

**修法**（`scenes.py`）：新增 `MARKER_WALL_X_M = 16.0`、`MARKER_CLEARANCE_M = 0.01`，
并把 `MARKER_X_M` 改为推导式 `MARKER_WALL_X_M - MARKER_THICKNESS_M/2 - MARKER_CLEARANCE_M` = **15.98**。
修后实测：路径 B 的 x 范围 **15.9700 – 15.9900**，与路径 A **逐位相同**。

**新增测试** `test_the_built_marker_matches_the_environment_plate_exactly`：
断言**建出的几何**逐位相同、背面不越过墙内表面、整体不越出墙平面。
（原测试只断言声明坐标，保留并改为断言 `MARKER_WALL_X_M`。）

**为什么不改冻结的 `environment.py`**：该文件是 Stage 1 冻结件（MANUAL §2「只增不改」），
而且环境侧的行为本来正确；错的是目录侧的坐标，所以修目录侧。

### H2. STEP 1 自审（我不能上机，所以必须说清哪部分没验证）

| 项 | 状态 |
| :--- | :--- |
| 两条路径的**包围盒一致** | ✅ **已证明**（逐位相同，纯算术） |
| 新常量与测试 | ✅ 实测：scenes 套件 24 → **25 项全过**；全套 9 套件 **158 项全过** |
| 渲染后实际长什么样 | ❌ **未验证**，本机无 Isaac Sim。0.02 m 在 15.5 m 处不足一个像素，**肉眼也未必能看出来**，所以这个差异不能靠看图发现，只能靠几何审计 |
| 远墙内表面是否真在 16.00 | ⚠️ **由源码推导**，未在渲染中实测。但 `test_bao_scenes.py` 的 `test_the_catalogue_frame_matches_the_environment_converter` 与 `test_real_room_boxes_classify_correctly` 已用真实 box 列表钉住坐标转换 |
| 是否需要重新核对**其它**几何 | ⚠️ **未做**。本轮只审了标志物；陈设、材质面、开口柱尚未逐项审计 |

### H3. STEP 1b（用户提问）：标志物是否真在绿色远墙上？坐标错乱是否已改正？

**判据（全部来自源码）**：远墙 `room_far` 中心 `ROOM_LENGTH_X + 0.02/2 = 16.01`，厚 0.02，
**内表面 x = 16.00**，颜色 `ROOM_FAR_WALL_COLOR = [0.13, 0.42, 0.20]`（绿）。

**实测结果（修复后）**：

| 项 | 实测 | 结论 |
| :--- | :--- | :--- |
| 标志物世界坐标（x, 横向, 高度） | **(15.98, 0.00, 1.40)** | 在远墙上 |
| x 范围 | 15.9700 – 15.9900 | 完全在墙内表面（16.00）**之前**，不埋进墙 |
| 横向范围 | −0.3000 – +0.3000，中心 0 | **居中于中线** |
| 高度范围 | 1.1000 – 1.7000，中心 1.40 | **1.40 m 高处**，不是高度 0 |
| 顶端 1.70 ≤ 障碍墙高 2.0 | 通过 | 可从开口看见 |

**"坐标错乱"的症状与实测对比**（`scenes.py` 记录的旧 bug：标志物被放到 1.40 m **横向**、高度 0）：

| | 若转换被跳过 | 实测 |
| :--- | :--- | :--- |
| 高度 z | 0.0000 | **1.4000** ✓ |
| 横向 y | 1.4000 | **0.0000** ✓ |

→ **转换已正确应用**（`to_world` / `_user_to_isaac_pos` 都是交换后两分量，实测两者输出一致）。

**全部 25 个标志物复验：25/25 落在远墙同一位置**（每个回合换槽位，所以必须全查）。

**钉住此事的既有回归测试**（不是我这次脚本，是仓库里的）：
- `test_catalogue_frame_matches_the_environment_converter` —— 目录的帧转换与 `environment._user_to_isaac_pos` 逐点相同
- `test_world_and_user_frames_are_inverse` —— 两个方向互逆；**其 docstring 明确写着这个 bug**（"a marker meant for 1.40 m up the far wall ends up at height zero and 1.40 m sideways"）
- `test_real_room_boxes_classify_correctly` —— 用实验室机器真实打印的 world box 数值做回归
- `test_the_built_marker_matches_the_environment_plate_exactly` —— **本轮新增**，钉“建出的几何”而非声明坐标

### H4. STEP 1c：顺带查出的第二个问题 —— `arrow` 的形心偏移（**已量化，决定不改几何**）

**发现**：计算 10 个形状的面积形心，9 个都在原点（|偏移| ≤ 4e-17 = 浮点噪声），
**只有 `arrow` 偏了 −13.23 mm**（`scenes.py:142-147`：轴从 −0.30 起，头到 +0.30 止）。

**三个约束无法同时满足**（数字实测，非推断）：

| 方案 | 总宽 | 在 ±0.30 内 | 形心偏移 |
| :--- | ---: | :--- | ---: |
| **A 保持现状** | **0.600000 m** | ✓ | 13.23 mm |
| B 形心归零 | 0.600000 m | **✗ 超框 13.23 mm** | 0 |
| D x 向缩 4.22% 兼顾 | **0.574662 m ✗** | ✓ | 0 |

**屏幕尺度**：起点姿态 47.42 mm/px，13.23 mm = **0.28 px**；标志物本身在该姿态只有 **12.7 px** 宽。

**决定：不改几何。** 理由：A 是唯一保持"总宽 0.60 m 与其余 24 个一致"的解，而那个尺寸**就是距离线索**
（`STAGE1_SCENES.md:145`）；代价 0.28 px 属亚像素。B 会让尖端出框，D 会让箭头比别的标志物窄 4.22%。

**改的是测试**（把隐含假设写明并钉住）：
1. `test_every_shape_fits_the_marker_bounding_box` 原用 `|vertex| <= SHAPE_HALF_M`，这**隐含"围绕形心居中的包围盒"**；
   改为直接量**形状自身跨度 ≤ 0.60 m**，并允许锚点两侧各半格预算 + 目录的间隙常量。
2. **新增** `test_the_arrow_keeps_its_size_and_its_measured_centroid_offset`：钉住
   **总宽恰为 0.600000 m**、**形心偏移在 12.0–14.0 mm 带内**、**该偏移在起点姿态 < 0.5 px**、尾部不越 −0.30。
   docstring 完整记录三约束冲突与三个方案的数字，避免后人重新试错。

### H5. 本轮我犯的错（留痕）

| # | 错误 | 真相 | 怎么发现 |
| :-: | :--- | :--- | :--- |
| H5-1 | 第一版"25 个标志物是否在墙上"的脚本报 **4 个 NO**（triangle / triangle_down） | **我的判据错**：我要求横向跨度对称（`ys[0]+ys[1]≈0`），但正三角形不是中心对称图形，其**外接圆圆心在形心**，跨度自然是 −0.15..+0.30 | 看到 4 个 NO 全是三角形，改用**面积形心**判据复验，得 25/25 ✓。**若当时直接下"4 个标志物位置错了"的结论就是误报** |
| H5-2 | 上一轮把场景变体矩阵标成 Stage 3（见 G4-1） | 已改 | — |

**H5-1 的教训**：判据本身也要被怀疑。对称性只对中心对称图形成立，用它当通用判据必然误判。

---

## I. 第五轮：1024 px 实拍图审查（远程渲染的首批真实画面）

### I0. 这批图是什么（先核实，避免用错基准）

| 项 | 实测 | 依据 |
| :--- | :--- | :--- |
| 位置 | `C:\Users\asus\Desktop\科研狗之人机心理学\stage\` | 列目录 |
| 完整批次 | `prev/` **60 张** = 5 场景 × (slot1–5 + sheet) × (eye + iso) | 实测 5×6×2=60 |
| 其它批次 | `prev_stage1.2`…`1.5`、`all_stage1.1`、`all_stage1.2` 各 **12 张**（只渲 slot1–3） | 同上 |
| 图像尺寸 | **1024×1024**（不是 512） | `stage1.1_slot1_eye.png` 实测 |
| 与当前代码一致性 | **一致**：`capture_scenes.py:236-240` 确实写 `_sheet_eye` 与 `_sheet_iso` 两个文件 | 读源码 |

**我在本轮犯的错（又一次"没读代码就推断"）**：只看 `capture_scenes.py:16` 的 docstring（写单个 `<scene>_sheet.png`）就断言"这批图是更旧代码渲的"。**实际是 docstring 过时，脚本行为一致**。已修正该 docstring。

### I1. 标志物：**25/25 在绿远墙上、横向居中、颜色与声明相符**

| 判据 | 实测 |
| :--- | :--- |
| 位置 | 收紧色相窗（±20°）后，**25 个里 24 个**的 x 中心落在 **501–514**，紧贴图像中轴 **512**；宽 13–41 px |
| 唯一例外 | `stage1.3_slot1`（红倒三角）测得中心 496、跨 2–990 —— **被场景物体污染**（图书馆的书架/陈设与红色同色相），非标志物错位 |
| 尺寸 | 干净样本宽 **13–25 px**、高 21–31 px，与 0.60 m 在 1024 px / 15.48 m 下的预期（≈25 px）吻合 |
| 背景 | 标志物后面是**绿远墙**（`stage1.1_sheet_eye` 肉眼可见绿区在开口内） |
| 颜色通道顺序 | **25 个中 23 个与声明完全一致** |
| 另 2 个 | `stage1.2_slot4`、`stage1.5_slot1` 首次测"顺序不符"，但跨度 **90×42**（远大于 25 px）说明采样窗混入背景；**收紧后两者都居中且通道正确**，`stage1.5_sheet_eye` 肉眼亦确认 slot1=亮粉、slot4=紫 |

**这一批是颜色 bug 的第一次端到端验证**：

| 槽位 | 声明 | 声明 RGB | 实测 RGB | 通道顺序 |
| :--- | :--- | :--- | :--- | :--- |
| `stage1.4` slot5 | `w` 酒红 | (140, 13, 64) RBG | (134, 102, 121) RBG | **一致** |
| `stage1.5` slot2 | `w` 酒红 | (140, 13, 64) RBG | (139, 53, 105) RBG | **一致** |
| `stage1.5` slot1 | `k` 亮粉 | (255, 102, 178) RBG | (169, 139, 158) RBG | **一致** |

未修 `COLOUR_NAMES` 时这三处提示词会说 "white"/"black"，而墙上是酒红/亮粉；**现在词与物一致**。

### I2. 你报告的"模糊"= **渲染噪声**，不是分辨率

`mean |相邻像素差|`，越小越干净：

| 场景 | 1024 px | **512 px（模型实际收到）** | 降幅 |
| :--- | ---: | ---: | ---: |
| stage1.1 | 9.34 | **7.49** | 20% |
| stage1.2 | 6.35 | **5.49** | 13% |
| stage1.3 | 8.00 | **6.50** | 17% |
| stage1.4 | 7.28 | **6.05** | 17% |
| stage1.5 | 7.97 | **6.65** | 17% |

- 噪声 **5.5–7.5 / 255 ≈ 2.2–2.9%**，均匀分布在受光面 → 视觉是"发麻、发糊"。
- 512 px 比 1024 px 干净 13–20%（降采样平均掉噪声）。
- 来源：`RaytracedLighting` + `spp=32`、无降噪器。`environment.py:1283-1284` 已记录加大发光半径就是为削减 grainy。
- **分辨率本身无问题**；可调项：提高 `spp`、开降噪、或加大 `light_radius`。

### I3. `stage1.2` 下半很暗（下半均值 50.8 vs 基线 113.3）

| 场景 | 上半 | 下半 | 上−下 |
| :--- | ---: | ---: | ---: |
| stage1.1 | 156.1 | 113.3 | +42.8 |
| **stage1.2** | 110.7 | **50.8** | +59.9 |
| stage1.3 | 93.4 | 105.9 | −12.5 |
| stage1.4 | 112.8 | 90.5 | +22.3 |
| stage1.5 | 118.2 | 100.5 | +17.7 |

- `stage1.2` 地面比基线暗 **55%**，MANUAL §7 的"未定位深色遮挡"旧账**仍在**。
- **尚未定因**：可能 `MI_Floor_01` 本身是深色仓库地面（正常），也可能是 `how=fallback-paint`（失败回退）。
  判据是那次渲染的 `prev_render.log`，**它不在这批文件里**（实测该目录只有 PNG）。
  → **需重渲一次并保留日志**才能定因。**不猜。**

### I4. 你说的"物品贴图还没有，只是正方体"：**与代码一致，不是缺陷**

`scene_builder.place_dressing`（`scene_builder.py:224-257`）**只造 `UsdGeom.Cube`**，明确不使用 `asset` 字段
（`.mdl` 是材质、不是 stage 资产），报告里每件记 `used_asset=False`。32 件陈设**设计上就是方块**。
换真实 `.usd` 道具属**未实现的 Stage 1 皮肤工作**（README "Known issues" 已列）。

### I5. 本轮我犯的错（**第 3 次同类错误**，必须记）

| # | 错误 | 真相 | 怎么发现 |
| :-: | :--- | :--- | :--- |
| I5-1 | 用 `capture_scenes.py:16` 的 docstring 推断"这批图是旧代码渲的" | 脚本 `:236-240` 本来就写两个 sheet 文件，**docstring 过时** | 读源码 `:232-240` 后更正，并修 docstring |
| I5-2 | 用"色相窗口"判颜色，把**蓝色遮挡墙**当成青色标志物，报出"blob 69025 px 覆盖整幅图" | 判据错：蓝墙 (32,64,128) 色相 ≈220°，与青色 198.8° 只差 21° | 改用逐行颜色游程看像素，再换成"色相紧匹配 + 落在中轴 + 尺寸 ≤60 px"三重条件 |
| I5-3 | 用 `sat>60` 判标志物，把最饱和的蓝墙算进去，报出"标志物宽 700 px" | 判据错，蓝墙是画面里最饱和的东西 | 同上 |

**共同教训（第 3 次）**：**判据设计本身就是最易错的一步**。三次都是"没先检验判据、直接采信输出"。
下次做像素分析前，**先用逐行游程或直接看图确认判据给出的对象是对的**，再谈数量。

### I6. 仍需验证（需重渲或补取证）

1. `stage1.2` 地面暗 = 材质本身还是回退 → **保留 `prev_render.log` 重渲**。
2. 材质绑定结果（`mdl-local` / `mdl` / `fallback-paint` / `no-surface-given`）→ 同上。
3. level 16 窄档可辨性 → 本批只有 level 0（日志缺失，无法从图确认档位）。
4. 陈设**真实渲染位置**是否逐件与声明一致 → 代数层已证不遮挡，渲染层未逐件核对。



