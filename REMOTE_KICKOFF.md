# REMOTE_KICKOFF.md — 粘到远程电脑的 DeepSeek Harness 里的启动提示词

> 用法：把下面 `---` 之间的**全部内容**复制，粘贴到远程电脑的 DSH 会话里作为**第一条消息**。
> 它是自包含的，不需要本机的任何上下文。
>
> 本文件的**可验证声明**由 `tools/verify_docs.py` 校验（已接入测试套件）。

---

你在 `~/EmbodiedBAO-AS-` 工作，这是 **EmbodiedBAO** 项目：一个 Isaac Sim 4.5 具身智能基准，
机器人是 Unitree H1，自变量 A/S = 通道宽度 / 肩宽。Stage 1 是"场景皮肤"：五个场景各一套材质与
陈设，17 个关卡 × 5 场景 × 25 标志物 = 6375 个 episode。

# 你的任务：构建场景

把 **Stage 1 五个场景的陈设**做到可交付：**真实道具、尺寸正确、不穿模、不进入机器人行走带**，
并且**用文本报告证明**，而不是靠看图判断。

**先完整读 `REMOTE_WORKORDER.md`**（主指令：我犯过的全部错误目录、坐标框架、几何常量、
文本判据、四步工作、硬规矩）。它每节末尾的【依据】说明结论来自代码、官方文档还是实测日志。
**本文件是开工流程；工作令是知识。**

---

## 第 0 步：环境自检（**全部通过才继续**）

```bash
cd ~/EmbodiedBAO-AS- && git pull --ff-only origin master && git log --oneline -3
git ls-files assets | wc -l        # 必须 1192，否则道具会全部变成 NO ASSET
ls assets/isaac | head -3          # 必须非空
ls assets/bucket | head -3         # 必须非空

# 纯 Python 检查，不需要 Isaac Sim
python tools/audit_dressing.py     # 必须: no problems found
python tools/prove_audit_fails.py  # 必须: all 4 planted fault(s) were caught
python tools/verify_docs.py        # 必须: all hold
python tools/check_field_names.py  # 必须: 3 个 ok
for f in test_bao_*.py; do echo "== $f"; python $f 2>/dev/null | tail -3; done
                                   # 必须: 0 失败，合计 193 项
```

**任何一步失败，先修它，不要往下走。**

### 第 0.5 步：git 状态与**推送凭据**（已实测；这是你必须先解决的一件事）

2026-10-11 在 `eai@eai:~/EmbodiedBAO-AS-` 实测：

```
git config --get user.name          -> 空
git config --get user.email         -> 空
git ls-remote origin                -> READ-OK    （仓库公开，匿名可读）
git config --get credential.helper  -> 空
```

**这台机器能拉，但没有身份、也没有推送凭据。**

#### 为什么凭据是必需的（不要误以为可以让别处代推）

**数据流只有一个方向：这台机器 → GitHub → 人类的机器。**
**只有这台机器上产生的提交，才能带着这台机器做出的场景工作进到 GitHub。**
别的机器推送的是它自己的代码，与这里的工作无关。所以**推送必须在这台机器上完成**，
**没有凭据就必须先配好，而不是"改成只做本地提交"**。
（本节先前写过"只做本地提交、由人类代推"，**那是把方向搞反了**，已改正。）

#### 第一步：配置身份（不需要凭据）

```bash
cd ~/EmbodiedBAO-AS-
git config user.name  "YourClassmateJiangYouning"
git config user.email "jiangyongxing910tt@gmail.com"
git config --get user.name && git config --get user.email     # 确认非空
```

（这两个值取自本仓库既有提交 `git log -1 --format='%an <%ae>'`，保证历史作者一致。）

#### 第二步：配置推送凭据（**已在这台机器上配置完成并验证通过**）

2026-10-11 实测通过，**不要再重新生成密钥**，直接使用：

```
~/.ssh/id_ed25519_bao        专用密钥（ed25519，空口令）
公钥注释                      embodiedbao-lab
公钥指纹                      SHA256:E0+S4xlRtYugcyTZFyxStkAD8j8TbnHI/V8o8bwYbPk
~/.ssh/config                Host github.com -> IdentityFile ~/.ssh/id_ed25519_bao,
                             IdentitiesOnly yes
验证结果                      ssh -T git@github.com
                             -> "Hi YourClassmateJiangYouning! You've successfully authenticated"
```

**注意**：这台机器上还有另一个密钥 `~/.ssh/id_ed25519`，其公钥注释是 **`flahm0104@gmail.com`**
——**与这个仓库无关**，属于别人留下的东西。**不要使用它、也不要删它**：
若它注册在别人的 GitHub 账号下，用它推送会把提交挂到别人的账号下，并且让那台机器上多一个
能写这个仓库的人。`~/.ssh/config` 里的 `IdentitiesOnly yes` 就是为了确保只用 `_bao`。

确认当前状态：

```bash
ssh -T git@github.com            # 出现 "Hi YourClassmateJiangYouning!" 即正常（exit code 1 属正常）
cd ~/EmbodiedBAO-AS- && git remote -v      # 应是 git@github.com:YourClassmateJiangYouning/EmbodiedBAO-AS-.git
```

若 `~/.ssh/id_ed25519_bao` 或 `~/.ssh/config` 丢失，按下面重建：

```bash
ssh-keygen -t ed25519 -C "embodiedbao-lab" -f ~/.ssh/id_ed25519_bao -N ""
cat ~/.ssh/id_ed25519_bao.pub        # 交给人类加到 GitHub -> Settings -> SSH and GPG keys
cat >> ~/.ssh/config <<'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_bao
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

**通用规则（两种方式都适用）**

- **推送前先确认工作树干净**：`git status --short` 必须为空（若有改动，那是还没提交的工作）。
- 推送成功后**确认远端真的收到了**：`git ls-remote origin master` 的哈希应与
  `git rev-parse master` 相同。**不要只看 `git push` 的退出码。**
- 每轮工作结束时报告：`git log --oneline -3` 与 `git ls-remote origin master` 的哈希，
  这样人类能确认他拉到的正是你做的那一版。
- **绝对不要**把 token、密码、私钥写进任何文件、提交、日志或对话。`.gitignore` 已屏蔽
  `.git-credentials`/`.netrc`/`*.pem`/`*.key`/`id_rsa*`/`.env` 等，但屏蔽不是许可：
  公开历史里的 token 只能吊销，不能"改掉"。
- **若凭据失效**：先把工作做完并**本地提交**，在报告里明确写"**推送失败，N 个提交待推**"
  并附 `git log --oneline -5` 与**原始报错全文**。**不要反复重试 `git push`，也不要索取凭据。**

---

## 工作内容（按优先级，每步只做一件，做完报告再前进）

### 任务 1 — 用文本确认 52 件陈设的结构（**不改代码**）

```bash
python capture_scenes.py --scene stage1.4 --level 0 --slots 1 --stage_tree
```

**不要加管道**（我曾用 `grep -E 'tree'` 把诊断输出过滤掉，看起来完全像分支没执行）。
把 `[tree]` 输出贴出来。**检查点**：每件已解析的道具都应有
`<name>/prop active=True visible=inherited`，且带 `material=...` 或 `displayColor`。

对 `stage1.2` / `stage1.3` / `stage1.5` 各跑一次。

**如果某件 `prop` 不是 `active=True visible=inherited`，停下并报告。**
判读表在 `REMOTE_WORKORDER.md` §3。

### 任务 2 — 用文本确认尺寸正确

```bash
python capture_scenes.py --scene stage1.4 --level 0 --slots 1 --dressing_report
```

**检查点**：每行的 `[drawn (...) vs declared (...)]` 里，`drawn` 的两个非约束轴应接近
`declared`，且**任何轴都不能超过 `declared`**（超过 = 穿模，会打印 `FITS BADLY`）。
把有 `FITS BADLY` 或 `FAILED` 的行原样贴出来。

### 任务 3 — 定位"平涂纯色墙面仍有高频花纹"的来源

`use_mdl` 默认 **False**（平涂），四个墙面用 `paint()` 涂纯色：
floor `(0.55,0.55,0.55)`、side_wall `(0.72,0.73,0.75)`、ceiling `(0.92,0.93,0.95)`、
far_wall `(0.13,0.42,0.20)`。**纯色不可能产生高频花纹。**

最可疑的是光照：`environment.py` 在 `/World/Light_01..08` 放了 8 个 `SphereLight`，
高度 2.6 m（天花板 3.0 m）、沿 x 每 2 m 一个。

**做一次对照**（不改默认值）：临时把这 8 个灯的强度设为 0（或只留 `DomeLight`），
渲染同一机位，比较。**报告两件事**：花纹是否消失；`[scene] <surface> flat-paint -> <prim>`
四行是否都是 `flat-paint`。

**先拿事实，不要凭猜调光照参数。**

### 任务 4 — 处理"道具偏细"

根因：`fill_scale_for_unit_parent` 取**最紧轴**的比例并封顶 1.0 → **保持比例**的代价是**填不满盒子**。

| 项 | 声明 | 实际绘制 | 视觉 |
| :--- | :--- | :--- | :--- |
| `railing` | (2.2261, 0.0728, 1.166) | (2.2261, **0.0024**, 0.6107) | 像一张纸 |
| `bench_left` | (1.7181, 0.9113, 0.1855) | (1.7181, 0.4834, **0.02**) | 极薄 |
| `tree_conifer` | (1.0164, 1.0001, 2.4) | (**0.4304**, 0.4167, 2.4) | 细柱 |

**两条路线，选一条并说明理由**：
- **(a) 按资产实测比例重算声明值**——用 `PROP_MEASUREMENTS` 反算，使最紧轴不再吃紧。
  **代价**：道具尺寸不再是"现实尺寸"。
- **(b) 换素材**——找固有比例更接近目标的资产（库里 `props_general` 有 27 个、
  `props_poles` 有 200 个可检索）。

**不要引入逐轴适配**（会让比例失真）；**不要把封顶从 1.0 放开**（放大曾把 20 cm 的灌木变成 1.2 m）。

### 任务 5 — 补齐仍是盒子的项（**只在库里有合适素材时**）

当前是纯几何体的：`park_sign`（公园）、`reading_lamp`（图书馆）、`shelf_left/right`、
`price_strip`、`promo_banner`（超市）。没有对应素材就**保留盒子**，并在 README 写明
"该素材在 NVIDIA 公开素材库中不存在"。

**接线顺序不可颠倒**（`place_dressing` 会把道具**强制适配到声明盒子**，声明值若是猜的，
画出来就是那个猜测——我曾把 6 m 的冷杉压进 2.9 m 的盒子）：

```bash
python tools/fetch_bucket_set.py --long-edge 512 --set <bucket 前缀>   # ① 抓取
python tools/measure_named_props.py <资产名...>                        # ② 测量（★必须先做★）
#    出现 "DIVIDE BY 100" 警告 = 厘米制资产
# ③ 实测值写进 scenes.PROP_MEASUREMENTS 与 scenes.ASSET_SCALE
# ④ 声明值 = 实测(米) × room_fit，room_fit ∈ (0,1]，等比
# ⑤ 本地全套检查 → 提交
```

---

## 硬规矩（违反会浪费整轮）

0. **暂存只加你真正改动的文件，不要用 `git add -A`。**
   这台机器的 `git status --short` 里有**八十多个未跟踪的日志、探针脚本和旧渲染目录**
   （积累了几周）。`.gitignore` 已覆盖其中大部分，但**不是全部**，而且
   **在这棵工作树上 `git add -A` 会把剩下的全部扫进一个提交——这个仓库是公开的。**
   正确做法：

   ```bash
   git status --short              # 看清有哪些改动
   git add <你改的那几个文件>        # 逐条列名，不用 -A / -u / .
   git diff --cached --stat        # 确认暂存的正是预期内容
   git status --short              # 确认没有意外被暂存的东西
   git commit -F <消息文件>
   ```

   判断标准：**`git diff --cached --stat` 里出现的每个文件，你都能说出为什么它在里面。**
   若出现一个你说不出原因的文件，`git reset <那个文件>` 再继续。

1. **`stage1.1` 是基线（无材质无陈设），一个字都不许改。**
2. **一个坐标框架、一次转换。** 目录框架 `(x, 高度, 横向)`；舞台框架（Z-up）`(x, 横向, 高度)`；
   相机辅助用 `(x, 横向, 高度)`。**隐式交换是缺陷来源。**
3. **尺寸判据是舞台实测**（`[drawn ... vs declared ...]`），不是声明值。
4. **适配只缩放、不重塑**（等比，取最紧轴，封顶 1.0）。
5. **接线前必须先测量。**
6. **每条新检查都要先证明它会失败**：把故障加进 `tools/prove_audit_fails.py`。
7. **诊断命令不要加 `grep`/管道。**
8. **每轮最后一步固定是提交**：`git add <你改的文件>`（**不是 `-A`**，见规矩 0）→
   `git diff --cached --stat` 确认 → `git status --short`（暂存区之外应为空）→
   `git commit -F <消息文件>`。提交信息必须写清**改了什么、为什么、依据是什么、哪些还没验证**。
   **推送后核对** `git rev-parse master` 与 `git ls-remote origin master` 的哈希一致。
9. **不要靠图片判断"道具在不在/尺寸对不对"**——用任务 1、2 的文本命令。
10. **不要在检查失败的情况下提交；不要留下为旧行为辩护的注释；不要提交任何凭据。**

**最该记住的一句**：**一个无法运行的检查，看起来和一个通过的检查一模一样。**

---

## 六个已知未解决问题（**不要当成已完成**）

1. 平涂纯色墙面仍有高频花纹 —— 纯色不可能产生噪点，最可疑是 8 个 `SphereLight`。**未确认。**
2. 等比适配导致 `railing` 2.4 mm 厚、`bench_left` 2 cm、`tree_conifer` 0.43 m 宽。**未修。**
3. 5 件仍是纯几何体（见任务 5）。**未修。**
4. `--iso_target`（世界框架）与 `--iso_x/--iso_height/--iso_z`（相机辅助框架）在
   `capture_scenes.look_from` 里被当作同一框架使用——`iso_target` 少了 `to_world` 转换。
   默认值恰好合理，所以未产生可见症状。**已知未修。**
5. `shopping_cart_corral_01` 已抓取但**故意未接线**：实测 2.006 × 2.376 × 3.711 m，
   在 5 m 宽房间里需要的 `|z|` 区间只有 2 毫米宽。**已知未接。**
6. 资产根**已全部提交**（`git ls-files assets` = 1192）。`.gitignore` 用 `assets/*` 屏蔽整个
   目录，靠 `!assets/isaac/` 与 `!assets/bucket/` 两条**例外**放行——这两条例外曾经缺失，
   症状是"实验室机器上 `0 from assets`，而本机测量正常"。

---

## 报告方式

每完成一步，报告：**做了什么命令 → 输出里的关键行 → 你的判读 → 依据是什么**。

**不要报告"应该没问题了"**；报告**命令与数字**。若某个结论只是推测，明说"未验证"。
若卡住，贴**原始报错全文**，不要只贴最后一行。

---

## 附：如果远程还没有仓库或解释器路径不同

```bash
git clone https://github.com/YourClassmateJiangYouning/EmbodiedBAO-AS-.git ~/EmbodiedBAO-AS-
cd ~/EmbodiedBAO-AS- && git log --oneline -1 && git ls-files assets | wc -l   # 1192
```

Isaac Sim 的解释器本项目一直用 `/home/ybh/isaacsim/python.sh`。**先确认它存在**：
`ls -la /home/ybh/isaacsim/python.sh`。若不同，先找到 Isaac Sim 自带的 `python.sh`，
再用它跑所有需要 `pxr`/`isaacsim` 的命令。

**纯 Python 的检查用系统 `python` 即可，不需要 Isaac Sim**：
`tools/audit_dressing.py`、`tools/prove_audit_fails.py`、`tools/verify_docs.py`、
`tools/check_field_names.py`、`test_bao_*.py`。
