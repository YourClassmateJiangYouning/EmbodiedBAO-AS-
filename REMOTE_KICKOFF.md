# REMOTE_KICKOFF.md — 粘到远程电脑的 DeepSeek Harness 里的启动提示词

> 用法：把下面 `---` 之间的**全部内容**复制，粘贴到远程电脑的 DSH 会话里作为第一条消息。
> 它是自包含的：不需要这台机器上的任何上下文。

---

你在 `~/EmbodiedBAO-AS-` 工作，这是 **EmbodiedBAO** 项目：一个 Isaac Sim 4.5 具身智能基准，
机器人是 Unitree H1，自变量 A/S = 通道宽度 / 肩宽。Stage 1 是"场景皮肤"：五个场景各一套
材质与陈设，17 个关卡 × 5 场景 × 25 标志物 = 6375 个 episode。

**你的任务**：把 Stage 1 五个场景的陈设做到可交付——真实道具、尺寸正确、不穿模、
不进入机器人行走带，**并且用文本报告证明**，而不是靠看图判断。

## 第一步（不要跳过，不要改代码）

```bash
cd ~/EmbodiedBAO-AS- && git pull --ff-only origin master && git log --oneline -3
cat REMOTE_WORKORDER.md
```

**`REMOTE_WORKORDER.md` 是主指令**，包含：当前状态、我犯过的**全部错误目录**（按病根归类，
含症状与代价）、坐标系与几何常量、判断"道具到底画出来没有"的正确方法、以及四步具体工作。

**先完整读完它，再动手。** 它里面每一节末尾的【依据】说明了结论来自代码、官方文档还是实测日志。

## 第二步：确认基线干净

```bash
cd ~/EmbodiedBAO-AS-
python tools/audit_dressing.py          # 必须: no problems found
python tools/prove_audit_fails.py       # 必须: all 4 planted fault(s) were caught
python tools/check_field_names.py       # 必须: 3 个 ok
for f in test_bao_*.py; do echo "== $f"; python $f 2>/dev/null | tail -3; done   # 必须: 0 失败，共 173 项
```

**任何一步失败，先修它，不要往下走。**

## 第三步：按 REMOTE_WORKORDER.md 的 §4 执行

顺序是 **STEP 1 确认结构 → STEP 2 定位光照花纹 → STEP 3 处理道具偏细 → STEP 4 补齐 NO ASSET**。
每一步都给出了命令、判读表和依据。

## 必须遵守的硬规矩

1. **`stage1.1` 是基线，无材质无陈设，一个字都不许改。**
2. **一个坐标框架、一次转换。** 项目有目录框架 `(x, 高度, 横向)` 与舞台框架（Z-up）
   `(x, 横向, 高度)`；**隐式交换是缺陷来源**。相机辅助用的是 `(x, 横向, 高度)`。
3. **尺寸判据是舞台实测，不是声明值。** 报告里的 `[drawn ... vs declared ...]` 才是事实。
4. **适配只缩放、不重塑**（等比，取最紧轴，封顶 1.0）。不要引入逐轴缩放。
5. **接线前必须先测量。** `place_dressing` 会把道具强制适配到声明盒子，
   所以声明值若是猜的，画出来就是那个猜测。顺序：抓取 → 测量 → 写表 → 声明 → 验证。
6. **每条新检查都要先证明它会失败**：把故障加进 `tools/prove_audit_fails.py`。
7. **诊断命令不要加 `grep`/管道**（我曾用 `grep -E 'tree'` 把诊断输出过滤掉，
   看起来完全像分支没执行）。要看全文，或看它写出的文件。
8. **每轮最后一步固定是提交**：`git add -A && git status --short`（必须为空）→
   `git commit -F <消息文件>`。提交信息必须写清**改了什么、为什么、依据是什么、哪些还没验证**。
9. **不要靠图片判断"道具在不在"**——用 `--dressing_report` 与 `--stage_tree` 的文本。
10. **不要在检查失败的情况下提交**；**不要留下为旧行为辩护的注释**。

## 判断"画出来没有"的两条文本命令（这是重点）

```bash
# 逐件：解析成功/失败/无资产 + 舞台实测的绘制尺寸
python capture_scenes.py --scene stage1.4 --level 0 --slots 1 --dressing_report

# prim 树：每件的 active / 继承可见性 / 是否有 material 或 displayColor
# ★ 不要加管道 ★
python capture_scenes.py --scene stage1.4 --level 0 --slots 1 --stage_tree
```

判读表在 `REMOTE_WORKORDER.md` §3。核心事实：**隐藏一个 prim 会连带隐藏它的整个子树**
（`SetActive(False)` 是组合移除，`MakeInvisible()` 是**继承**属性），所以被引用的道具
在设计上必须是清除盒的**兄弟**，不能是子节点——这一点曾经让 43 件道具从每一帧里消失，
而报告逐件打印 `RESOLVED`。

## 六个已知未解决问题（不要当成已完成）

1. 平涂纯色墙面仍出现高频花纹——纯色不可能产生噪点，最可疑是 8 个 `SphereLight`（2.6 m 高，
   天花板 3.0 m）。**未确认。**
2. 等比适配导致 `railing` 被画成 2.4 mm 厚、`bench_left` 2 cm、`tree_conifer` 只有 0.43 m 宽。**未修。**
3. `park_sign`、`reading_lamp`、`shelf_left/right`、`price_strip`、`promo_banner` 仍是纯几何体。**未修。**
4. `--iso_target`（世界框架）与 `--iso_x/--iso_height/--iso_z`（相机辅助框架）在
   `capture_scenes.look_from` 里被当作同一框架使用——`iso_target` 少了 `to_world` 转换。
   默认值恰好合理，所以未产生可见症状。**已知未修。**
5. `shopping_cart_corral_01` 已抓取但**故意未接线**：实测 2.006 × 2.376 × 3.711 m，
   在 5 m 宽的房间里需要的 `|z|` 区间只有 2 毫米宽。**已知未接。**
6. **资产根已全部提交**（实测：`git ls-files assets/isaac` = 1090 个文件，
   `git ls-files assets/bucket` = 95 个，共 1185 个；工作树干净）。
   `.gitignore` 用 `assets/*` 屏蔽整个目录，靠 `!assets/isaac/` 与 `!assets/bucket/` 两条**例外**
   放行——这两条例外曾经缺失过，导致实验室机器上"0 from assets"而本机测量一切正常。
   仍然请先跑一次 `git ls-files assets | wc -l` 确认你在远程拿到的是 1185，不是 0。
   （本节先前那条"资产根没有进仓库"的说明是**错的**，已改正。）

## 报告方式

每完成一步，报告：**做了什么命令 → 输出里的关键行 → 你的判读 → 依据是什么**。
**不要报告"应该没问题了"**；报告**命令与数字**。若某个结论只是推测，明说"未验证"。

---

## 附：如果远程机器上还没有仓库

```bash
git clone https://github.com/YourClassmateJiangYouning/EmbodiedBAO-AS-.git ~/EmbodiedBAO-AS-
cd ~/EmbodiedBAO-AS- && git log --oneline -1
```

Isaac Sim 的 Python 解释器在本项目里一直用 `/home/ybh/isaacsim/python.sh`。
**先确认它存在**：`ls -la /home/ybh/isaacsim/python.sh`。
若路径不同，先找到 Isaac Sim 自带的 `python.sh`，再用它跑所有需要 `pxr`/`isaacsim` 的命令。
**纯 Python 的检查（`tools/audit_dressing.py`、`tools/prove_audit_fails.py`、`test_bao_*.py`）
用系统 `python` 即可，不需要 Isaac Sim。**
