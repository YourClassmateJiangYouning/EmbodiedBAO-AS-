# REMOTE_WORKORDER.md — 给远程电脑上的 AI 的工作令

> 这份文档是**操作指令**，不是背景介绍。读完它你应该能立刻开始工作，并且**不再重复
> 2026-10-10/11 这一轮在另一台机器上已经犯过的所有错误**。
>
> 每一节末尾都有【依据】，说明结论来自代码、官方文档还是实测日志。**没有依据的断言一律不要信，
> 包括本文件里的**——包括这一句。

---

## 0. 你的任务是什么

把 **Stage 1 的五个场景皮肤**做到"可交付"：每个场景的陈设（dressing）是真实道具、
尺寸正确、不穿模、不进入机器人行走带，并且**用文本报告证明**，而不是靠看图判断。

**当前状态（提交 `adadf5b`）**：

| 场景 | 陈设件数 | 接了真实 `.usd` 道具 |
| :--- | ---: | ---: |
| `stage1.1` 实验室 | 0 | 0（**基线，一个字都不许改**） |
| `stage1.2` 仓库 | 11 | 9 |
| `stage1.3` 图书馆 | 13 | 12 |
| `stage1.4` 公园 | 16 | 15 |
| `stage1.5` 超市 | 12 | 8 |
| **合计** | **52** | **44** |

**已知仍未解决的三件事**（按优先级）：

1. **平涂纯色墙面仍出现高频花纹**。`use_mdl=False` 时四个墙面是纯色 `displayColor`，
   纯色不可能产生噪点，所以来源很可能是**光照**：8 个 `SphereLight` 位于 2.6 m 高
   （天花板 3.0 m）、沿走廊每 2 m 一个。**未确认，不要当成结论。**
2. **等比适配让部分道具偏细**：`railing` 被画成 2.4 mm 厚、`bench_left` 2 cm 厚、
   `tree_conifer` 只有 0.43 m 宽（声明 1.02 m）。原因是 `fill_scale_for_unit_parent`
   取"最紧那一轴"的比例并封顶 1.0，**保持比例** 的代价就是**填不满盒子**。
3. `park_sign`、`reading_lamp`、`shelf_left/right`、`price_strip`、`promo_banner`
   仍是纯几何体（`NO ASSET`），因为库里没有对应道具。

**资产根的状态（实测，不要听凭印象）**：`assets/isaac` 与 `assets/bucket` **都已提交**，
分别 1090 与 94 个文件（`git ls-files`）。`.gitignore` 用 `assets/*` 屏蔽整个 `assets/`，
靠 `!assets/isaac/` 与 `!assets/bucket/` 两条**例外**放行——这两条例外曾经缺失，
症状是"实验室机器上 `0 from assets`，而本机测量正常"。**开工前跑一次
`git ls-files assets | wc -l` 确认是 1192。**

### 远程机器的 git 状态（**已实测，不是推测**）

在 `eai@eai:~/EmbodiedBAO-AS-` 上实测 2026-10-11：

```
git config --get user.name      -> 空
git config --get user.email     -> 空
git ls-remote origin            -> READ-OK      （仓库公开，匿名可读）
git config --get credential.helper -> 空
```

**结论**：这台机器**能拉，但没有身份、也没有推送凭据**。

- **身份**：两行命令即可解决，不需要任何凭据。
- **推送凭据是必需的，不是可选的**：数据流只有一个方向——**这台机器 → GitHub → 人类的机器**。
  只有这里产生的提交才带着这里做出的工作；别的机器推送的是它自己的代码。
  **所以"改成只做本地提交、由人类代推"是错的**——那条路把工作留在了这台机器上，人类拉不到。
  （本手册先前写过那个错误建议，已改正。）
  - **推荐 SSH**：`ssh-keygen` 生成密钥 → 人类把公钥加到 GitHub → `git remote set-url origin
    git@github.com:...`。私钥不离开这台机器，没有明文 token。
  - 或 **PAT**：`git config credential.helper store`，首推时经 URL 传入 `ghp_...`。
    代价是 token 明文落在 `~/.git-credentials`。
- 详细步骤（含确切命令）见 `REMOTE_KICKOFF.md` 第 0.5 步。
- 推送后**确认远端收到**：`git ls-remote origin master` 的哈希应等于 `git rev-parse master`。
  **不要只看 `git push` 的退出码。**
- **任何情况下都不要把 token、密码、私钥写进文件、提交、日志或对话。**
  `.gitignore` 已屏蔽 `.git-credentials`/`.netrc`/`*.pem`/`*.key`/`id_rsa*`/`.env` 等模式，
  但**屏蔽不是许可**——公开历史里的 token 只能吊销，不能"改掉"。
- 凭据尚未配好时：把工作做完并**本地提交**，报告里写明"**推送凭据未配置，N 个提交待推**"，
  **不要反复重试 `git push`**。


---

## 1. 先做这五件事，做完再动任何代码

```bash
cd ~/EmbodiedBAO-AS-          # 远程的工作目录
git pull --ff-only origin master
git log --oneline -3          # 确认 HEAD 至少是 adadf5b

# ① 全套离线检查必须 197 项全过、0 失败
for f in test_bao_*.py; do python $f 2>/dev/null | tail -1; done

# ② 陈设代码的语义审核（必须 "no problems found"）
python tools/audit_dressing.py

# ③ 证明审核脚本本身有效（必须 "all 4 planted fault(s) were caught"）
python tools/prove_audit_fails.py

# ④ 字段名守卫
python tools/check_field_names.py

# ⑤ 纸面环境不适用；到这一步才能开 Isaac Sim
```

**另外先确认资产根完整**——这两个数不对，后面所有 `RESOLVED` 都会变成 `NO ASSET`：

```bash
git ls-files assets | wc -l        # 必须是 1192
ls assets/isaac | head -3          # 必须非空
ls assets/bucket | head -3         # 必须非空
```

**任何一步失败，先修它，不要往下走。** 这一轮最大的时间浪费就是"带着已知失败继续往前"。

【依据】`test_bao_scenes.py` 等 **9** 个测试文件，共 **197** 项；
`tools/audit_dressing.py`（11 条语义规则）；`tools/prove_audit_fails.py`（向源码植入 4 个真实故障
并验证审核会失败）。

---

## 2. 我犯过的错 —— 精确目录（**按病根归类，不按时间**）

这一节是本文件最重要的部分。**每一行的"症状"我都亲眼见过，"代价"是我实际花掉的时间。**

### 2.A USD API：名字对、语义错（最难发现，代价最大）

| # | 我以为 | 实际 | 症状 | 我为此付出的代价 |
| :-: | :--- | :--- | :--- | :--- |
| A1 | `prim.SetActive(False)` 只是"不显示" | **停用会把整个子树从舞台上移除**（"All scene description for prims beneath the deactivated prim will not be composed"） | 43 件已解析的道具**全部不存在**，而报告逐件打印 `RESOLVED` | 4 轮以上；这是"画面光秃秃"的第一层原因 |
| A2 | 改用 `MakeInvisible()` 就好了 | **可见性是继承属性**：`invisible` 的定义是"the prim and all of its **descendants** will be invisible" | 换了隐藏方式，**画面没有任何变化**，因为 prop 仍是盒子的子节点 | 又 2 轮 |
| A3 | `ComputeWorldBound(prim)` 量的是"这个 prim 的尺寸" | 它是 **world space**，**包含全部祖先变换** | 被测 prim 的父级正是"即将被缩放的那个盒子"→ 度量与施加互相污染 → fit 因子出现 **219.9 / 39.5 / 11.1** | 3 轮，且中途误判为"测量逻辑有 bug" |
| A4 | `ComputeRelativeBound(prop, Xform)` 得到的是**米** | 它**排除该祖先自身的变换**；那个 Xform 承载声明缩放 → 数值回到**父级局部单位**，在每个项最紧的轴上饱和到 **1.0** | 报告出现 3 条**假** `FITS BADLY`（喷泉"1.0 m"实际 0.16 m） | 1 轮，靠算术自证：`0.1108/0.1621 = 0.6835` |
| A5 | `AddScaleOp()` 总能加一个缩放 | **已存在同名 op 时会抛异常**，而错误信息里 `ErrorException` 的 `str()` 是**空字符串** | 工作台与一个纸箱直接 `FAILED ErrorException:`（无任何信息） | 1 轮诊断 + 1 轮修复 |
| A6 | 先 `AddScaleOp` 再 `AddTranslateOp`，顺序无所谓 | 每个 `AddXformOp` 都比前一个"**更局部**"，列表**逆序复合** → scale 先加则**平移先被应用再乘缩放** | 每个盒子中心变成 `size ⊙ at`：木箱应在 x 10.2，实际 6.14 | 1 轮（审查者发现） |
| A7 | `AddTranslateOp` 可以传 `Gf.Vec3f` | 它默认 `PrecisionDouble` → `double3` 属性；`Vt` **没有** `Vec3f↔Vec3d` 转换 | 可能静默不写入 → 所有盒子落在原点 | 未验证（已改为 `Vec3d`） |

**教训（请照做）**：
- 用到的**每一个** pxr 调用，在**调用点旁边**写一句"它的语义是什么、依据是哪份文档"。
- **界限规则**：*"我自己是否位于我正在度量的那个变换之内？"* —— `environment.py` 里量机器人
  离地高度**可以**用 `ComputeWorldBound`（问的就是世界坐标下的最低点），
  `scene_builder` 里量 prop **不可以**（prop 就在被判定的缩放之内）。
  同一个 API，两种用法，一对一错，**区别不在 API 而在调用者的位置**。
- 官方参考：<https://openusd.org/release/api/class_usd_geom_b_box_cache.html>（包围盒四个方法的语义）、
  `UsdGeomImageable`（可见性继承）、`UsdGeomXformable`（AddXformOp 顺序）。

### 2.B 坐标框架：两个框架，我混淆了六次以上

项目有**两个**框架，所有尺寸/位置类 bug 几乎都源自这里：

| 框架 | 三个分量 | 谁在用 |
| :--- | :--- | :--- |
| **目录框架**（作者用） | `(x, 高度, 横向)` | `scenes.py` 的每个 `size` / `at`、`PROP_MEASUREMENTS`、`_dressing_floor` / `_dressing_wall` |
| **世界框架**（舞台用，**Z-up**） | `(x, 横向, 高度)` | USD 舞台、`sc.to_world` / `sc.to_world_size` 的结果 |
| **相机辅助框架** | `(x, 横向, 高度)` | `capture_scenes.py` 的 `--iso_x/--iso_height/--iso_z`（见 `--iso_height` 帮助文本："under the 3 m ceiling"） |

| # | 错 | 症状 | 代价 |
| :-: | :--- | :--- | :--- |
| B1 | 同一批资产里**单位不统一**：`Barberry`/`Douglas_Fir` 是**厘米**，`bench_curved_01`/`planter` 是**米**；`FramedPoster` 报 `metersPerUnit=0.01`，而 `sign_plaza_fountain_01` 报 **1.0 却量出 16.2** | 元数据**不可信**，必须逐件实测 | 2 轮 |
| B2 | 声明值与实测值**框架不同**却直接比 | 表里出现 `(0.6, 0.5, 0.3)` 而源码是 `(0.6, 0.30, 0.5)` | 1 轮 |
| B3 | `_dressing_wall` 内部交换外两项、`place_dressing` 又对已存储值再换一次 | **两次抵消** → 存储顺序停在"测量表的顺序"，而凸出判据把 slot 0 当**墙法向**读；同一组数字处在两个顺序里 | **6 次尝试对齐，每次报出不同的正确项** |
| B4 | 墙面项按"平面内顺序无所谓"免于框架转换 | Cube 的 scale 分量 1 落在世界 y（横向）、分量 2 落在世界 z（高度）→ 海报声明 `(0.0524, 0.7426, 1.092)` 被画成 **1.092 m 高**、0.74 m 宽 | 1 轮（审查者发现） |
| B5 | 按"舞台是 Z-up 所以 z 是高度"去改**相机**元组 | 相机辅助用 `(x, 横向, 高度)`，方向**正好相反** | 1 轮，且我把正确代码改错了 |
| B6 | 把"更大的文件"当"更好的文件" | `sektion_cabinet_visuals.usd` **没有 defaultPrim**；能用的恰是 `_instanceable` 那个 | 1 轮 |

**教训（请照做）**：
- **一个框架、一次转换。** 任何"隐式交换"都是缺陷来源。`scenes.py` 现在只有一个框架
  `(x, 高, 横)`，`place_dressing` 里**只有一处** `to_world_size`，`tools/audit_dressing.py`
  检查这一点。
- **墙面项**：`size[0]` = 墙法向（厚度），凸出判据读 `size[0]`；另两轴是墙平面内的宽与高。
- 尺寸**不要手写**，从实测值算：`声明 = 实测 × ASSET_SCALE × room_fit`，且 `room_fit ∈ (0,1]`。

### 2.C 设计：等比 vs 逐轴（这是**设计错误**，不是实现 bug）

| # | 错 | 真相 |
| :-: | :--- | :--- |
| C1 | `fit_scale_from_extent` 返回**逐轴**因子 | 任何"比例与声明盒子不同"的资产都会被**重塑**。报告里出现 `(1.0, 16.0075, 0.0624)`——护栏 16 倍高或 16 倍薄，两种读法都不是护栏 |
| C2 | 因子取 `声明/原生`，但 prop 在**已带声明缩放的盒子之下** | 最终尺寸 ≈ **声明值的平方**（多乘了一次 0.05~3 之间的数）|
| C3 | 结构：prop 是清除盒的**子节点** | 盒子必须被隐藏，而**两种隐藏方式都作用于子树** → 鱼与熊掌不可兼得 |

**现在的正确设计（改之前先读懂）**：

```
/World/Dressing/<name>              Xform —— 项本体；先 AddTranslateOp，后 AddScaleOp(声明尺寸)
/World/Dressing/<name>/clearance    Cube  —— 单位立方（size=1），在已缩放的 Xform 下恰为声明体积
/World/Dressing/<name>/prop         Xform —— 被引用的网格，与 clearance 是【兄弟】
```

- **兄弟关系**使"隐藏盒子"不可能波及 prop。
- 缩放放在**项 Xform** 上，两个子节点共用。
- `fill_scale_for_unit_parent(prop)` 返回 **`1/native` 的等比最小值**（封顶 1.0），
  **不接受声明尺寸**：prop 要填满的是**父级局部空间的 1 个单位**。
- 结果：尺寸精确等于声明 → 这正是 `[drawn ... vs declared ...]` 两列相等所验证的。

### 2.D 验证：我的检查**假通过**（这一类比 API 错误更危险）

| # | 错 | 真相 |
| :-: | :--- | :--- |
| D1 | `audit_dressing.py` **在含全部 5 个严重缺陷的版本上打印 "no problems found"** | 它只断言**字符串存在**（`"fit_error" in builder`），不断言代码**可达**。那段复核代码因为父级被停用而**从未执行**，却满足了所有存在性检查 |
| D2 | 加了一条"复核绘制尺寸"的代码 | 依赖 `GetPrimAtPath(.../prop)`，而该路径**已被停用移除** → 每次 `continue` → **一次都没运行过**，却打印"通过" |
| D3 | 测试把 `NO ASSET` 当正常 | `park_sign` 等一直是盒子，测试却全绿 |
| D4 | 审核规则读**原始文本** | **五次**被自己的散文绊倒：docstring 写出错拼的名字、注释引用已废除的写法、注释里写出要检测的 API 名 |
| D5 | 断言 `MakeInvisible` 存在 | 缺陷正是"存在、但作用在**错误的 prim** 上"。**存在性检查抓不到结构错误** |
| D6 | 复核测量相对**项 Xform** | 排除该祖先变换 → 数值在父级局部单位 → 3 条假 `FITS BADLY` |

**教训（请照做）**：

1. **每条检查都要先证明它会失败。** 本仓库有 `tools/prove_audit_fails.py`：向源码植入真实故障、
   跑审核、确认报错、再还原。**新增检查时把故障加进那个列表。**
2. **区分"存在"与"可达/正确"**：检查**调用**而不是**名字**；检查**结构**（谁是谁的子节点、
   op 的顺序）而不是**某个函数被调用过**。
3. **"无法运行的检查"必须显式说出来**——静默跳过看起来和通过一模一样。
   （例：套件里 ruff 检查在 ruff 缺失时打印 `[skip] ... were NOT checked`。）
4. **剥离散文再扫描**：只清 **docstring 与注释**（AST + tokenize），**保留其他字符串字面量**——
   因为 `record["fit_error"]` 的键**是代码**。我用正则清掉全部字符串时，检查就报"字段不存在"
   而它就在代码里。
5. **切函数体用 AST**（`lineno`/`end_lineno`），不要用字符串 `split("\ndef ")`：它会切过函数末尾、
   也会被**缩进的嵌套函数**提前截断。

### 2.E 流程与习惯（用户反复强调，违反会浪费整轮）

| # | 错 | 教训 |
| :-: | :--- | :--- |
| E1 | **改完不提交就往下走**（连续多轮，用户反复追问） | **每轮最后一步固定是提交**。**暂存要逐文件列名（`git add <文件…>`），不要用 `-A`** —— 实验室那台机器的工作树里有八十多个未跟踪的日志与旧目录，`-A` 会把它们扫进公开仓库（详见 §5）。提交前用 `git diff --cached --stat` 核对，并确认暂存区之外没有改动 |
| E2 | 只领先两个提交就让你 `git push`，而本地还有未提交改动 | 你拉到的和我说的话不一致。**push 前确认工作树干净** |
| E3 | 给用户的命令**加了 `grep`**，把诊断输出过滤掉了（`grep -E 'code stamp|tree'` 匹配到 `tree_conifer/tree_fir/tree_sapling`，而 `[tree]` 行没显示） | **一个结果会被阅读者自己的过滤器藏起来的诊断，注定被误读。** 诊断命令**不要带管道**，或同时写文件 |
| E4 | 上机前没看 `--help` 就断言参数行为 | 先 `--help`，或**真的调用一次 `parse_args`**（参数解析不依赖 Isaac，可本地验证） |
| E5 | 陈旧注释与文档留在代码里 | "为代码已不具备的行为辩护的注释**比没有注释更糟**"，它是下一个人重新引入缺陷的路径。`audit_dressing.py` 会扫 4 个此类短语 + "clearance box deactivated" 这个说法 |
| E6 | 文档与代码脱节（MANUAL 写"全是方块、接了 3 件"，实际 52 件、44 件） | 现在有 `test_the_manual_states_the_scene_facts_the_code_holds` 强制 MANUAL 的表格与目录一致 |
| E7 | 用 PowerShell 字符串改 UTF-8 文件 / 传多行 `-m` | 用 `edit`/`write` 工具；多行提交用消息文件 + `git commit -F` |
| E8 | 一次删/移动前不确认路径 | 先 `Get-ChildItem` / `ls` 确认**绝对路径**是目标 |
| E9 | 批量处理被**一张坏图**打断整轮 | 逐文件 `try/except` |
| E10 | 靠**看图**判断"道具在不在/尺寸对不对" | 见 §3 |

### 2.F 尺寸与规则：几个我算错过的具体数

| 量 | 值 | 说明 |
| :--- | :--- | :--- |
| 房间 | x ∈ [0, 16]，z ∈ [−2.5, 2.5]，天花板 3.0 m | 障碍墙在 x=8.0，厚 0.02，**内表面 7.99** |
| 开口 | 0.570 m（level 0） | 眼平相机**只能**从这个开口看侧厅 |
| 远墙内表面 | x = 16.0 | `FAR_WALL_FACE_X` |
| 地面件 | `\|z\| ≥ 1.5`；**底面高于 0.6 m 时 `\|z\| ≥ 1.8`** | 两条都要满足 |
| 墙挂件 | `\|z\| ≥ 0.9`、**凸出 ≤ 0.20 m** | 凸出 = `size[0] − MOUNT_CLEARANCE_M` |
| 墙面项定位 | 盒**中心** = 面 − 间隙 − `size[0]/2` | 背面贴面；写错会让半个道具埋进墙（0.02 m 的墙会被穿透） |
| 挂载坐标 | `at[0]` 是**表面**，不是墙心 | |
| 标志物 | 0.90 m，x = 15.98，高度 1.40 | 颜色词是**自变量**，`stage1.4` slot 1 = `r` = `(0.85, 0.15, 0.12)` |
| `stage1.1` | **无材质、无陈设，一个字都不许改** | 基线 |

**禁用颜色**：任何陈设不得使用本场景 5 个标志物之一的颜色。

---

## 3. 判断"到底画出来没有"的正确方法（**别再看图猜**）

这一轮我在"看画面判断"上浪费的时间**远超其他所有错误之和**，原因是三层症状叠加，
每一层都让下一层更难判断。**按下面的顺序，全部是文本**：

```bash
# ① 逐件：解析成功 / 失败 / 无资产，以及**舞台实测的绘制尺寸**
python capture_scenes.py --scene stage1.4 --level 0 --slots 1 --dressing_report

# ② prim 树：每件的 active / 继承可见性 / 是否有 material 或 displayColor
#    ★ 不要加管道！★
python capture_scenes.py --scene stage1.4 --level 0 --slots 1 --stage_tree
```

**判读表**：

| `[scene] dressing` 行 | `[tree]` 行 | 结论 |
| :--- | :--- | :--- |
| `RESOLVED X` | `<name>/prop active=True visible=inherited` | **正常**，道具在舞台上且可见 |
| `RESOLVED X` | `prop` 行缺失或 `active=False` | **严重**：某个祖先被停用 |
| `RESOLVED X` | `prop visible=invisible` | 某个祖先被设为不可见（**可见性继承**） |
| `FAILED <原因>` | 该 `prop` 行不存在 | 引用失败，看到的是清除盒 |
| `NO ASSET` | 只有 `clearance` 子节点 | 设计如此，看到的是清除盒 |
| `FITS BADLY` | —— | `drawn` 超过 `declared` → **穿模，必须修** |

**另外两条已内建的保护**：
- `place_dressing` 在放置完毕后用**新建的** `BBoxCache`（官方文档明确：它**不监听变更**，
  复用会读到旧缓存）量回每件 prop，相对**无变换的** `/World/Dressing` 根 → 得到**世界尺寸**，
  与 `declared_extent` 直接可比，写进 `[drawn ... vs declared ...]`。
- `capture_scenes.py` 打印并写入 **code stamp**（git HEAD + 两个关键模块的 mtime）到
  `slot_report.txt`。**图片可能是旧代码生成的**——这一条曾经让我把旧图当证据。

---

## 4. 现在请你做的具体工作（按顺序，每步都要报告依据）

### STEP 1 — 确认结构无误（只跑命令，不改代码）

```bash
cd ~/EmbodiedBAO-AS- && git pull --ff-only origin master
python capture_scenes.py --scene stage1.4 --level 0 --slots 1 --stage_tree
```

把 `[tree]` 输出贴出来。**检查点**：每件已解析的道具都应有
`<name>/prop active=True visible=inherited material=...` 或 `displayColor`。

**如果 `prop` 行不是 `active=True visible=inherited`，停下，报告我。**

### STEP 2 — 定位"纯色墙面仍有花纹"的来源

`use_mdl` 现在是 **False**（默认平涂），四个墙面用 `paint()` 涂纯色：
floor `(0.55,0.55,0.55)`、side_wall `(0.72,0.73,0.75)`、ceiling `(0.92,0.93,0.94)`、
far_wall `(0.13,0.42,0.20)`。**纯色不可能产生高频花纹。**

最可疑的是光照：`environment.py` 的 `_setup_lights` 在 `/World/Light_01..08` 放了 8 个
`SphereLight`，高度 2.6 m、沿 x 每 2 m 一个（天花板 3.0 m）。

**做法**：在**不改默认行为**的前提下做一次对照——临时把 `SphereLight` 的强度降为 0（或只留
`DomeLight`），渲染同一机位，比较。**报告两件事**：花纹是否消失、
`[scene] <surface> flat-paint -> <prim>` 四行是否都是 `flat-paint`。

**不要凭猜改光照参数**：先拿到"花纹是否随光源消失"这个事实。

### STEP 3 — 处理"道具偏细"

根因：`fill_scale_for_unit_parent` 取**最紧轴**的比例并封顶 1.0 → 保持比例、填不满盒子。
现在的后果：

| 项 | 声明 | 实际绘制 | 视觉 |
| :--- | :--- | :--- | :--- |
| `railing` | (2.2261, 0.0728, 1.166) | (2.2261, **0.0024**, 0.6107) | 像一张纸 |
| `bench_left` | (1.7181, 0.9113, 0.1855) | (1.7181, 0.4834, **0.02**) | 极薄 |
| `tree_conifer` | (1.0164, 1.0001, 2.4) | (**0.4304**, 0.4167, 2.4) | 细柱 |

**两条可行路线，选一条并说明理由**：
- **(a) 改声明值去接近资产比例**：用 `PROP_MEASUREMENTS` 里的实测比例反算声明值，
  使最紧轴不再是"吃紧的那一轴"。**代价**：道具尺寸不再是"现实尺寸"。
- **(b) 换素材**：找固有比例更接近目标的资产（库里 `props_general` 有 27 个、
  `props_poles` 有 200 个可供检索）。

**不要引入逐轴适配**——那会让比例失真（见 §2.C）。**也不要把封顶从 1.0 放开**：
放大曾把 20 cm 的灌木变成 1.2 m 的"树"。

### STEP 4 — 补齐 `NO ASSET` 的项（若素材存在）

当前仍是盒子的：`park_sign`（公园）、`reading_lamp`（图书馆）、`shelf_left/right`、
`price_strip`、`promo_banner`（超市）。**只在库里确实有合适素材时才接**；
没有就保留盒子，并在 README 写明"该道具在 NVIDIA 公开素材库中不存在"。

**接线流程（顺序不可颠倒）**：

```bash
# ① 抓取（会顺带降采样贴图）
python tools/fetch_bucket_set.py --long-edge 512 --set <bucket 前缀>
# ② 测量（★必须在接线之前★，否则声明值就是猜的）
python tools/measure_named_props.py <资产名...>
#    输出里若出现 "DIVIDE BY 100" 警告，说明是厘米制资产
# ③ 把实测值写进 scenes.PROP_MEASUREMENTS 和 scenes.ASSET_SCALE
# ④ 声明值 = 实测(米) × room_fit，room_fit ∈ (0,1]，等比
# ⑤ 本地跑全套 + audit，再提交
```

**为什么顺序不可颠倒**：`place_dressing` 会把道具**强制适配到声明盒子**，
所以声明值如果是猜的，画出来就是那个猜测——这一轮我曾把 6 m 的冷杉压进 2.9 m 的盒子。

---

## 5. 提交纪律（**硬要求**）

```bash
# 改完必须做，且顺序固定
python tools/audit_dressing.py            # 必须 no problems found
python tools/prove_audit_fails.py         # 必须 all 4 planted fault(s) were caught
python tools/verify_docs.py               # 必须 all hold
for f in test_bao_*.py; do python $f; done  # 必须 0 失败
git status --short                        # 先看清有哪些改动
git add <你改的那几个文件>                  # ★ 逐条列名，不要 -A / -u / .
git diff --cached --stat                  # 确认暂存的正是预期内容
git status --short                        # 暂存区之外应为空
git commit -F <消息文件>                   # 多行消息走文件，不要 -m
git push origin master
git rev-parse master && git ls-remote origin master   # 两个哈希必须相同
```

**为什么不用 `git add -A`**：远程那台机器的 `git status --short` 里有**八十多个未跟踪的日志、
探针脚本和旧渲染目录**（几周积累）。`.gitignore` 覆盖了大部分但**不是全部**，
而在这个仓库是公开的前提下，`git add -A` 会把剩下的全部扫进一个提交。
**判据：`git diff --cached --stat` 里每个文件你都能说出为什么它在里面。**

**提交信息里必须写清**：改了什么、**为什么**、**依据是什么**、以及**哪些结论还没验证**。
本仓库的历史提交信息是主要的知识载体之一，**不要写"fix bug"**。

---

## 6. 你**不要**做的事

1. **不要改 `stage1.1`**（基线，无材质无陈设）。
2. **不要引入隐式坐标交换**：一个框架、一次转换。
3. **不要用逐轴缩放**适配道具。
4. **不要放开 `room_fit` 的上限**去放大道具。
5. **不要凭声明值判定尺寸正确**——判据是**舞台实测**的 `[drawn ... vs declared ...]`。
6. **不要靠图片判断"道具在不在"**——用 §3 的两条文本命令。
7. **不要给诊断命令加 `grep`**（E3）。
8. **不要在审核/检查失败的情况下提交**。
9. **不要留下为旧行为辩护的注释**。
10. **不要在没有测量数据时接线**。

---

## 7. 关键文件与它们的职责

| 文件 | 职责 | 注意 |
| :--- | :--- | :--- |
| `scenes.py` | **Stage 1 场景目录，纯数据**：5 场景的材质与陈设、10 形状/10 颜色/25 标志物、`PROP_MEASUREMENTS`、`ASSET_SCALE`、坐标转换 | 可改，**不能动 `stage1.1`** |
| `scene_builder.py` | 把目录应用到 live USD 舞台：材质解析、标志物网格、**陈设放置与适配**、表面分类、报告 | 结构见 §2.C，改前先读 |
| `environment.py` | 房间几何、灯光、机器人、相机、`setup_scene` | 房间常量在 §2.F |
| `capture_scenes.py` | 预览与三个文本诊断：`--dressing_report` / `--stage_tree`；code stamp | 诊断**别加管道** |
| `tools/audit_dressing.py` | **11 条语义规则**（框架、可达性、结构、拟合、默认值、陈旧注释） | 新增检查要加进 `prove_audit_fails.py` |
| `tools/prove_audit_fails.py` | 向源码植入 4 个真实故障并验证审核会失败 | **证明检查有效性的唯一手段** |
| `test_bao_scenes.py` 等 **9** 个 | **197 项**离线检查 | 提交前必须全过 |
| `MANUAL.md` | 交接手册（含更早的错误目录） | 场景表格由测试强制与代码一致 |

---

## 8. 一分钟版（如果只读一段）

> **一个框架、一次转换；量尺寸用相对度量、并问清"我是否在自己度量的变换之内"；
> 隐藏一个 prim 会连带它的子树，所以被引用的道具必须是**兄弟**而非子节点；
> 适配只缩放、不重塑；每条检查都要先证明它会失败；
> 诊断不要加管道；改完必须提交。
> **判断"画出来没有"用 `--dressing_report` 与 `--stage_tree` 的文本，不要靠看图。**
