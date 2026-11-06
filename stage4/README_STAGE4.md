# Stage 4 闭环实验操作台 · 使用说明

> 按 `STAGE4_HANDOFF_SPEC.md` 实现。模型不控制机器人；软件负责**发请求、记录、导出**，动作由实验人员执行。

---

## 1. 怎么打开

### 1.1 启动

在仓库根目录 `D:\EmInAI\EmbodiedBAO(AS)` 打开一个终端：

```bat
python stage4\run.py
```

浏览器会自动打开 **http://127.0.0.1:8765/**。如果 8765 被占用：

```bat
python stage4\run.py 8790
```

> 只需要 Python 3.9+ 和浏览器，**不需要安装任何第三方包**。
> 装了 Pillow 会自动把发给模型的图压成 1024 JPEG（和之前实验一致）；没装也能跑，只是发送原图、稍慢。

### 1.2 设置 API Key（二选一）

- **方式 A（推荐，一次设置）**：在终端里

  ```bat
  setx BOYUE_API_KEY "sk-你的密钥"
  ```

  然后**重开一个新终端**再启动。之后窗口里不用再填。

- **方式 B**：每次在窗口「配置」页的 **API Key** 输入框里粘贴，然后点「建立 / 打开会话」。

> Key 只保存在内存里，**不会写进任何实验文件**。

---

## 2. 一次实验的完整操作顺序

窗口从上到下：状态栏 → 左「当前照片」右「模型输出」→ 按钮行 → 执行结果 → 日志。

| 步骤 | 按钮 | 说明 |
| :-- | :-- | :-- |
| ① | 配置页填好 → **建立 / 打开会话** | 填 Session ID、场景、目标物、Mode、模型、A/S |
| ② | **Start Episode** | 开一个新 episode，清空 history |
| ③ | **Capture / 下一步照片** | 选一张相机拍的照片（1024×1024）。原图按字节保存，不裁剪不压缩 |
| ④ | **Call Model** | 发 prompt，等模型回 action |
| ⑤ | 看右侧 **Model Output** | action / confidence / latency / reasoning / raw response |
| ⑥ | 填「执行结果」→ **Action Executed** | 记录这次动作的真实结果（executed / blocked / collision / skipped）+ 碰撞 + 位置 |
| ⑦ | 回到 ③ 拍下一张 | 循环，最多 30 步 |
| ⑧ | **Mark Passed / Mark Failed** | 身体完整穿过且无碰撞 = Passed |
| ⑨ | **导出 CSV** | 写 `steps.csv` / `episodes.csv` / `summary.csv` |

其他按钮：

- **Digest（仅 B）**：Stage 4B 在主循环前读一遍记忆，**不消耗 step**
- **Retry API**：上一次调用超时/无效时重发（会记成新的一次调用，旧记录保留）
- **Stop**：中止当前 episode，记为 `stopped`

---

## 2.5 现场尺寸：肩宽与开口宽度

真实场景按 **肩宽 S = 70 cm** 设计，最宽开口 140 cm 正好是 **A/S = 2.0**。

| A/S | 开口宽度 | 现场怎么摆 |
| ---: | ---: | :-- |
| 2.0 | **140 cm** | 最宽，直走可通过 |
| 1.3 | **91 cm** | 约身体 1.3 倍 |
| 1.1 | **77 cm** | 略宽于身体 |
| 1.0 | **70 cm** | 正好等于身体宽度 |
| 0.9 | **63 cm** | 窄于身体，必须转身（或侧身）才过 |

换算公式：`开口宽度 = A/S × 肩宽`

这两项都能在窗口的「配置 / 汇总」页改：

- **肩宽 S（cm）**：默认 70。改成别的值，上面整张表和记进数据的 `channel_width_m` 会一起变。
- **步长（m）**：默认 0.75（一次 forward 走的距离）。**保持 0.75 才能和 Stage 1–3 的动作描述逐字一致**；
  真机如果走不了 0.75 m，就在这里填真实值，prompt 里的 "walk 75cm" 会自动变成对应数字。

窗口顶部状态栏实时显示 `A/S 2.0 · 开口 1.40 m / 140.0 cm · 步长 0.75 m`，配置页还有一张
**开口宽度对照表**（当前档位高亮 ◀），现场照着调隔板即可。

> ⚠️ **一个必须确认的前提**：A/S 的分母是"设计肩宽"70 cm，不是机器人实测宽度。
> 如果真实机器人明显窄于 70 cm，那么 63 cm 的开口（A/S 0.9）实际可能直接走得过去，
> 就测不到"必须转身"的现象。实验前请量一下机器人的实际肩宽：
> 若与 70 cm 相差较大，应在论文中明确说明 A/S 的分母定义，并考虑让 S = 实测肩宽。

---

## 3. Stage 4B 的记忆怎么给

Mode 选 **B** 之后：

1. 先跑完 Stage 4A 的 **A/S = 0.9** episode 并 **Mark Passed**；
2. 切到 Mode B 的 episode；
3. 用接口挂记忆（见下方 curl 示例），或直接编辑 `memory/*.json`。

记忆会自动**盲处理**：只保留「动作序列 + 每步反馈 + 模型自己的 reasoning」，**不含 A/S、宽度、narrow/wide**。

```bash
curl -X POST http://127.0.0.1:8765/api/memory \
  -H "Content-Type: application/json" \
  -d "{\"source\":\"episode\",\"session_id\":\"s4-20260920-A\",\"episode_id\":\"S4E_xxxx\"}"
```

---

## 4. 数据落在哪

```text
stage4/sessions/<Session ID>/
├─ config.json        # 场景/模型/A/S/宽度/时间/endpoint
├─ manifest.csv       # 每个 episode 一行：passed、步数、invalid 数
├─ episodes.jsonl     # episode 全量记录
├─ steps.jsonl        # 每步全量记录（action、feedback、位置、延迟、invalid、reasoning、原始响应）
├─ images/            # 原图，不裁剪不压缩
├─ prompts/           # 每次实际发送的 prompt 原文
├─ responses/         # 每次调用的所有 attempt（超时/无效也保留）
├─ memory/            # 盲化后的记忆卡
└─ exports/
   ├─ steps.csv
   ├─ episodes.csv
   └─ summary.csv     # 按模型 / 按 A/S 汇总
```

---

## 5. 界面每一栏是什么

**顶部状态栏**：Mode / Session / Model / Scene / Marker / **A/S** / Episode / Step / Status
（A/S 只在这里显示，**不会进 prompt**）

**右侧 5 个标签页**

| 标签 | 内容 |
| :-- | :-- |
| Model Output | action、confidence、latency、invalid、attempts、payload 大小、reasoning、场景描述、原始响应 |
| Prompt | 本次**逐字发送**的完整 prompt |
| History | 每步动作/反馈/位置/碰撞/延迟，点一行看原始记录 |
| Memory | 记忆卡与 digest（仅 B） |
| 配置 / 汇总 | 会话参数、导出按钮、按模型和按 A/S 的汇总表 |

---

## 6. 需要知道的三条硬规则

1. **A/S、开口宽度、narrow/wide 永远不会出现在 prompt 里** —— 它们只在 `config.json` 和界面上。
2. **timeout / invalid 不会补动作** —— 记录 `action: null, invalid: true`，原始响应完整保留。
3. **reasoning 是模型自己返回的字段**（`reasoning` / `reasoning_content`），不是模型的隐藏思维链。

---

## 6.5 两个实测出来的参数（重要）

这两项不是猜的，是用真实 prompt 测出来的：

### 关闭隐藏思考 `reasoning_effort: "none"`（默认开启）

| 模型 | 默认（思考开）| 关掉思考 |
| :-- | :-- | :-- |
| **gemini-3-flash-preview** | 6.9 s，**840 思考 token**；有一次 1151 token 把 1200 预算吃光，JSON 被截断成 `finish_reason=length` | **2.7 s**，思考 0 token，JSON 完整 |
| **deepseek-v4-flash-vision-exp** | 1200/1200、3000/3000、8000/8000 **全部被思考吃光，正文 0 字符**，永远拿不到 action | **1.8 s**，JSON 完整 |
| claude-haiku-4-5 | 本来就不思考（0 token）| 无变化 |

> **所以默认勾选「关闭隐藏思考」**，否则 deepseek **一个合法动作都拿不到**。
> 想复现 Stage 1 那种"思考开"的设置，把配置页的勾去掉，并在 `config.json` 里可以看到实际发送的参数
> （`request_params` 也会写进每一步的记录，方便事后核对）。

### `max_tokens = 2000`

思考关掉后，正文约 400–900 字符，2000 足够；旧值 1200 在思考开时会被截断。

---

## 6.6 状态含义

| 状态 | 含义 | 下一步 |
| :-- | :-- | :-- |
| `READY` | 可以拍摄 / 调用 | — |
| `CALLING_MODEL` | 正在请求（最长 = 超时 × 重试次数）| 等待 |
| `WAITING_FOR_ACTION` | 模型已返回合法动作，等实验人员执行 | 填结果 → **Action Executed** |
| `INVALID` | 超时 / 截断 / 动作不在枚举内 | 可 **Retry API**，或直接记 `skipped` |
| `ACTION_DONE` | 本步已记录 | 拍下一张 |
| `PASSED` / `FAILED` | episode 结束 | **Start Episode** 开下一个 |

---

## 7. 常见问题

| 现象 | 处理 |
| :-- | :-- |
| 打开页面是空白 | 确认终端里显示 `Stage 4 操作台已启动`，用 `http://127.0.0.1:8765/` 而不是 `file://` |
| `Call Model` 报 401 | Key 没填或填错，重新粘贴后点「建立 / 打开会话」 |
| 一直 `CALLING_MODEL` | 模型超时（默认 90 s，可在配置页改）；超时后状态会变 `INVALID`，按 **Retry API** |
| 想换模型/A/S | 改配置 → 重新「建立 / 打开会话」（新 Session ID，避免混数据） |
| 端口被占 | `python stage4\run.py 8790` |
| 想重跑某个 episode | 不删数据，用新 Session ID 重建会话即可 |

---

## 8. 最小跑通（最低配置：1 个 marker）

```text
Stage 4A：5 A/S × 3 模型 = 15 episodes
Stage 4B：3 模型 = 3 episodes（先有 A/S 0.9 的成功 episode 才能做记忆）
```
