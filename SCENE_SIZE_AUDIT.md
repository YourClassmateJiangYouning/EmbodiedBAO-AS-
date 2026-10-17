# 场景尺寸审查 — Stage 1 的五个场景（2026-10-16）

对 `scenes.py` 里 Stage 1 的全部场景尺寸做一次逐项算术审查，目标是**尽可能多地找出不合理的尺寸**：
既包括"声明值本身就不可能"的（轴写反、穿墙、重叠、悬空），也包括"声明合理但画出来不对"的
（fit 因子），以及固定几何常量之间的互相矛盾。

结论：**声明的数值大多是对的，问题几乎都在"声明 → 绘制"这一步，以及若干条从未被任何检查覆盖的
几何约束上。** 下面每一条都能在没有 Isaac Sim 的机器上复现（`scenes.py` 可独立导入，绘制结果取自
已落盘的实测日志）。

---

## 0. 范围、数据来源与方法

| 项 | 值 |
| :--- | :--- |
| 审查的场景 | `scenes.py` `SCENES` 的 `stage1.1`–`stage1.5` |
| stage1.1 | 无 dressing（0 项） |
| stage1.2 / 1.3 / 1.4 / 1.5 | 11 / 12 / 31 / 12 项，合计 **66** 项（其中 61 项引用 `.usd`，5 块广告牌无资产） |
| 房间几何（`environment.py:83-106, 953-1017`） | `x∈[0,16]`、`z∈[-2.5, 2.5]`、`y∈[0, 3.0]`；障碍墙 `x∈[7.99, 8.01]`、高 2.0 m |
| 绘制逻辑（`scene_builder.py:948-1293`） | 每项一个声明盒（隐藏的 clearance cube）+ 引用网格，统一缩放放进盒子 |
| 实测绘制尺寸 | `diag_logs/markers_stage1.{2,3,4,5}.log` 的 `[drawn … vs declared …]` 行 |
| 数据对齐 | 脚本逐项比对日志里的 declared 与当前 `scenes.py` 的 `to_world_size(size)`：**61/61 完全一致**，日志对当前代码有效 |
| 复现 | 见文末 §11 |

绘制公式（`scene_builder.py:441-555` `fit_reference_to_local_unit` + `:1026,1092,1105`）：

```
item Xform 的 scale   = to_world_size(声明 size)          # 已含声明尺寸
prop  的 scale        = factor = min(1/native_i)，上限 1.0  # 不含声明尺寸
=> drawn_i = declared_i × native_i × min(1, 1/native_max)
```

`native_i` 是资产自身的包围盒（米或厘米，取决于资产），这就是下面 §1 的根因。

---

## 1. 系统性问题：61 个道具画出来全都远小于声明值

> 这条在 `STAGE1_DRESSING_FINDINGS.md` 的 F2/F3 里已被记录为"未修"。这里补的是**当前 HEAD 的完整
> 量化**，以及**渲染出来的绝对尺寸**——因为"fill 0.02"不直观，"一本书 0.4 mm 厚"才直观。

**61 项中 34 项至少有一个轴小于声明值的 50%**；worst-fill ≥ 0.9 的只有 `stacked_boxes`（0.982）一项。

### 1.1 画出来以后物理上荒谬的尺寸（世界坐标，米）

| 场景 | 项 | 资产 | 声明世界 (x,y,z) | **实际画出来 (x,y,z)** | 物体常识 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1.3 | `book_a`/`book_b` | `SM_Book_01` | 0.020 × 0.145 × 0.205 | **0.0004 × 0.021 × 0.042** | 书厚 0.4 mm、高 4.2 cm |
| 1.5 | `banana` | `011_banana` | 0.197 × 0.039 × 0.074 | **0.039 × 0.0015 × 0.0055** | 香蕉宽 1.5 mm、高 5.5 mm |
| 1.4 | `railing` | `safety_railing_01` | 2.226 × 0.073 × 1.166 | **2.226 × 0.0024 × 0.611** | 栏杆厚 2.4 mm |
| 1.5 | `mug_a2/b1/c1` | `SM_Mug_*` | ≈0.09 × 0.13 × 0.09 | **≈0.008 × 0.016 × 0.008** | 9 cm 杯子画成 8.5 mm |
| 1.3 | `clock` | `SM_Clock_04` | 0.045 × 0.450 × 0.150 | **0.002 × 0.2025 × 0.0225** | 钟面 3:1 被压成 9:1 |
| 1.3 | `reading_poster` | `FramedPoster` | 0.052 × 1.092 × 0.743 | **0.052 × 0.052 × 0.505** | 1.09 m 海报画成 5 cm 竖条（另见 §2） |
| 1.2 | `pallets`/`pallet_stack` | `pallet` | 1.213 × 0.802 × 0.143 | **1.213 × 0.531 × 0.0167** | 托盘厚 1.7 cm（真实 14.3 cm） |
| 1.2 | `klt_bins` | `small_KLT_visual` | 0.198 × 0.297 × 0.146 | **0.039 × 0.088 × 0.021** | 料箱高 2.1 cm |
| 1.2 | `crate` / 1.5 `produce_bins` | `SM_Crate_*` | 0.602 × 0.402 × 0.170 | **0.602 × 0.268 × 0.048** | 周转箱高 4.8 cm |
| 1.2 | `forklift` | `packing_table` | 0.891 × 0.281 × 0.390 | **0.891 × 0.089 × 0.171** | 89 cm 长、**8.9 cm 宽**的薄片 |
| 1.4 | `bench_iron/left/right` | `Roxana_DiningBench` | 0.434 × 1.500 × 0.453 | **0.125 × 1.500 × 0.137** | 长椅高 13.7 cm |
| 1.3/1.5 | `study_table`/`checkout` | `SM_HeavyDutyPackingTable` | 2.474 × 0.762 × 0.994 | **2.474 × 0.235 × 0.400** | 桌子进深 23 cm、高 40 cm |
| 1.3/1.5 | `book_cart`/`trolley` | `dolly` | 0.850 × 1.259 × 0.441 | **0.574 × 1.259 × 0.154** | 推车高 15 cm |
| 1.4 | `picnic_table` | `packing_table` | 1.732 × 0.547 × 0.758 | **1.732 × 0.173 × 0.332** | 野餐桌进深 17 cm、高 33 cm |
| 1.2 | `traffic_cone` | `S_TrafficCone` | 0.336 × 0.334 × 0.463 | **0.113 × 0.112 × 0.214** | 锥高 21 cm |
| 1.4 | `tree_conifer/fir/sapling` | 三种树 | x 1.016/1.154/1.242 | **x 0.430/0.579/0.701** | 2.4 m 的树只有 43 cm 粗 |
| 1.2/1.3 | `rack(_tall)`/`bookshelf_*` | `RackSmallEmpty_A1` | 0.716 × 1.325 × 1.996 | **0.257 × 0.879 × 1.996** | 货架进深 26 cm、宽 88 cm |
| 1.5 | `shelf_left/right` | `RackSmall_A7` | 0.681 × 1.259 × 1.896 | **0.244 × 0.836 × 1.896** | 同上 |

**机制**：声明值本来就≈测量值（`declared_i ∝ native_i`），于是

* 资产尺度 < 1 m 时（书、杯子、香蕉、钟…）：`factor` 被 1.0 截断，`drawn_i = declared_i × native_i`
  ——也就是**把米数当比例乘了一遍**，尺寸被"平方"，所以 9 cm 的杯子变 8.5 mm；
* 资产尺度 > 1 m 时：`drawn_i = declared_i × native_i / native_max`——只有 `native` 最大的那个轴
  恰好等于声明值，其余轴按资产比例缩小，于是**长宽高比例被平方**（钟面 3:1 → 9:1）。

**还有 3 项跑出了声明盒**（`bench_iron`、`bench_left`、`bench_right`）：日志里 `FITS BADLY: its
geometry leaves the declared box on axis ['z']`，且底座落在 `z = -0.0029 m`——**沉进地板 2.9 mm**。
只有这 3 项会让 `fit_error` 非空，其余 58 项虽然画得离谱，检查全部是绿的。

**需要更正的一处旧结论**：`STAGE1_DRESSING_FINDINGS.md` F6 说"25/43 项出盒、三棵树穿天花板
(+1.1 m)"。那是更早的 `min(declared/native)` 版本的测量结果，`bfd5be7` 已回退到 `min(1/native)`；
当前日志里树高就是 2.4/2.3/2.2 m，在盒内。**当前的问题不是"太大穿顶"，而是"一律太小"。**

---

## 2. 新发现：`reading_poster` 的宽度轴与厚度轴写反了

`STAGE1_DRESSING_FINDINGS.md` F4 指出壁挂件的尺寸坐标系自相矛盾，并明确写"**哪种顺序是有意的，本文件
不决定**"。逐轴比对可以把这件事定下来：

**逐轴**比较"声明 `(slot0, slot1, slot2)`"与"测量 × `ASSET_SCALE` × `room_fit` 的
`(x, height, lateral)`"：

| 壁挂项 | 声明 | 测量×换算 | 逐轴 |
| :--- | :--- | :--- | :--- |
| `clock` | (0.045, 0.15, 0.45) | (0.045, 0.15, 0.45) | ✅ 同序 |
| `fountain` | (0.1621, 0.1108, 0.1621) | (0.1621, 0.1108, 0.1621) | ✅ 同序 |
| `mug_a2/b1/c1` | (0.0923, 0.091, 0.1271) … | 同值 | ✅ 同序 |
| `banana` | (0.1972, 0.0741, 0.0386) | (0.1972, 0.0741, 0.0386) | ✅ 同序（但这个顺序本身错，见 §3） |
| **`reading_poster`** | **(0.0524, 0.7426, 1.092)** | **(1.092, 0.7426, 0.0524)** | ❌ **slot 0 与 slot 2 对调** |

也就是说约定就是 `_dressing_wall` 文档写的那样——`size` 与 `PROP_MEASUREMENTS` 同序，helper
**不做任何交换**。`reading_poster` 是唯一的例外，因为它的注释（`scenes.py:1087-1091`）声称
"The wall helper takes (across, tall, thick) and STORES (thick, tall, across) … an argument tuple A
becomes (A[2], A[1], A[0])"——这正是 `_dressing_wall` 在某个旧版本里做过、现在**已经不做的**事
（`scenes.py:651, 672` 原样保存 `(x_thick, tall, across)`）。

**为什么一直没被发现**：`test_a_declared_size_is_the_size_of_the_prop_it_names`
（`test_bao_scenes.py:1281-1294`）用 **sorted** 比较三个轴。对调 slot0/slot2 后 sorted 三元组完全相同，
所以声明盒"装得下道具"的断言恒真；而 `tools/audit_dressing.py` 的 wall-frame 检查只看
`to_world_size` 有没有被额外调用，不看数值是否与测量同序。**结果就是 §1 表里那块
52 mm × 52 mm × 505 mm 的"海报"。**

---

## 3. 新发现：壁挂件从不旋转，深度轴永远朝走廊

`place_dressing` 对道具只做 translate + scale（`:1025-1026, 1098-1105`），**没有 rotate**。因此：

* 壁挂件的世界 `x`（= 声明 `size[0]`）必须容纳资产**自身 local x** 的尺寸；
* 资产 local x 恰好是薄的（钟 45 mm、杯子 92 mm）时看起来正常；
* 资产 local x 是长的，就会**垂直插出墙面**：

| 项 | `size[0]` 声明深度 | 伸出量（− 1 mm 间隙） | 200 mm 上限 | 备注 |
| :--- | ---: | ---: | :--- | :--- |
| `banana` | **0.1972** | 196.2 mm | 擦线通过 | local x 是香蕉**长度**；in-plane 只剩 38.6 mm，画出来宽 1.5 mm |
| `fountain` | 0.1621 | 161.1 mm | 通过 | 三个轴的 x/z 相等，无所谓朝向，但"喷泉"是 16 cm 壁挂片 |
| `mug_*` | 0.0923 | 91.3 mm | 通过 | 杯子近立方，朝向影响小 |

`FramedPoster` 的 local y（52.4 mm）才是薄轴，local x 是 1092 mm——所以按 §2 的"同序"约定声明
`(0.0524, 0.7426, 1.092)` 本身就意味着把 1052 mm 的海报横轴塞进 52 mm 的 x 盒，再被 fit 缩成 5 cm。
**海报要在墙上平贴，必须要一次 90° 旋转，而代码里没有。**

---

## 4. 新发现：声明盒（和实际几何）穿透侧墙 `z = ±2.5`

房间宽 5.0 m，侧墙内表面在 `z = ±2.5`（`environment.py:990-1001`）。
**没有任何检查覆盖侧向边界**：`tools/audit_dressing.py:213-219` 只断言 `size[1] > 3.0`（天花板）。

### 4.1 声明盒超出侧墙（`|z| + size[2]/2 > 2.5`）

| 场景 | 项 | 锚点 z | 侧向尺寸 | 外沿 | 超出 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 1.2 | `rack_tall` | 2.35 | 1.3247 | **3.0124** | **+0.512 m** |
| 1.4 | `bench_iron` | −2.05 | 1.5005 | **−2.8002** | **+0.300 m** |
| 1.4 | `bench_left` | 2.05 | 1.5005 | **2.8002** | **+0.300 m** |
| 1.4 | `bench_right` | −2.05 | 1.5005 | **−2.8002** | **+0.300 m** |
| 1.4 | `tree_sapling` | −2.20 | 1.2472 | **−2.8236** | **+0.324 m** |
| 1.3 | `bookshelf_left` | 2.05 | 1.3247 | 2.7124 | +0.212 m |
| 1.3 | `bookshelf_right` | −2.05 | 1.3247 | −2.7124 | +0.212 m |
| 1.5 | `shelf_left` | 1.95 | 1.2588 | 2.5794 | +0.079 m |
| 1.5 | `shelf_right` | −1.95 | 1.2588 | −2.5794 | +0.079 m |
| 1.4 | `tree_fir` | −2.05 | 1.0574 | −2.5787 | +0.079 m |
| 1.4 | `tree_conifer` | 2.05 | 1.0001 | 2.5500 | +0.050 m |
| 1.2 | `rack` | −1.85 | 1.3247 | −2.5124 | +0.012 m |
| 1.2 | `pallet_stack` | 2.10 | 0.8023 | 2.5011 | +0.001 m |

### 4.2 实际画出来的几何也确实穿墙的（来自日志世界包围盒）

* `bench_left` z 到 **+2.8003**，`bench_right`/`bench_iron` 到 **−2.8003**（虽然只有 13.7 cm 高）；
* `tree_sapling` y 到 **−2.5809**；
* **`shrub_h` 横跨障碍墙**：世界盒 `x 7.8815 … 8.5185`——一棵灌木种在 `x=8.0` 的障碍墙里，
  两半分处走廊和里屋；
* `picnic_table` x 到 6.37、`carton`、`bin_square` 等都在墙内，正常。

---

## 5. 新发现：同一场景内 20 处声明盒互相重叠

（`drawn` 通常比盒子小，所以部分重叠不显形；但布局算术是用这些盒子做的，重叠意味着"净空"被重复计算。）

| 场景 | 重叠对 | 重叠量 (x,y,z) |
| :--- | :--- | :--- |
| 1.2 | `rack` ↔ `traffic_cone` | (0.336, 0.334, 0.463) —— 完全叠放 |
| 1.2 | `rack_tall` ↔ `forklift` | (0.003, 0.281, 0.390) |
| 1.3 | `bookshelf_left` ↔ `book_cart` | (0.583, 0.842, 0.441) |
| 1.3 | `bookshelf_left` ↔ `cabinet` | (0.092, 0.764, 0.786) |
| 1.3 | `bookshelf_right` ↔ `armchair` | (0.268, 1.040, 0.790) |
| 1.3 | `study_table` ↔ `stock_carton` | (0.186, 0.192, 0.302) |
| 1.3 | `book_cart` ↔ `cabinet` | (0.359, 0.562, 0.441) |
| 1.4 | `shrub_left` ↔ `flowers_b` | (0.319, 0.231, 0.204) |
| 1.4 | `shrub_left` ↔ `litter_bin` | (0.140, 0.486, 0.900) |
| 1.4 | `shrub_right` ↔ `planter_right` | (0.532, 0.578, 0.773) |
| 1.4 | `shrub_a` ↔ `railing` | (0.757, 0.073, 0.900) |
| 1.4 | `shrub_c` ↔ `shrub_j` | (0.157, 0.700, 0.900) |
| 1.4 | `shrub_d` ↔ `shrub_k` | (0.157, 0.700, 0.900) |
| 1.4 | `shrub_d` ↔ `flowers_c` | (0.138, 0.312, 0.204) |
| 1.4 | `shrub_e` ↔ `bench_right` | (0.396, 0.750, 0.453) |
| 1.4 | `shrub_e` ↔ `planter_left` | (0.132, 0.706, 0.773) |
| 1.4 | `shrub_f` ↔ `railing` | (0.757, 0.073, 0.900) |
| 1.4 | `shrub_i` ↔ `bench_iron` | (0.434, 0.750, 0.453) |
| 1.4 | `picnic_table` ↔ `litter_bin` | (0.327, 0.485, 0.758) |
| 1.4 | `bin_square` ↔ `tree_conifer` | (0.201, 0.796, 1.011) |

`shrub_c`(x=10.4) 与 `shrub_j`(x=11.0) 间距只有 0.60 m，而单个灌木声明宽 0.757 m——所以每两棵就叠 0.157 m。
`stage1.5` 是唯一没有声明盒重叠的场景。

---

## 6. 新发现：走道占用——测试只查锚点，不查半宽

`test_decoration_is_never_collidable_and_stays_off_the_path`（`test_bao_scenes.py:267-273`）只断言
`|at.z| >= 1.5`，**不检查盒子内边缘**。内边缘进入 `|z| < 1.5` 的项：

| 场景 | 项 | 锚点 z | 侧向尺寸 | **内边缘 \|z\|** |
| :--- | :--- | ---: | ---: | ---: |
| 1.3 | `book_cart` | 1.60 | 1.2594 | **0.970** |
| 1.5 | `trolley` | 1.60 | 1.2594 | **0.970** |
| 1.2 | `rack` | −1.85 | 1.3247 | 1.188 |
| 1.3 | `study_table` | 1.60 | 0.7620 | 1.219 |
| 1.4 | `bench_iron/left/right` | ±2.05 | 1.5005 | 1.300 |
| 1.5 | `shelf_left/right` | ±1.95 | 1.2588 | 1.321 |
| 1.3 | `bookshelf_left/right` | ±2.05 | 1.3247 | 1.388 |
| 1.3 | `armchair` | −1.95 | 1.0400 | 1.430 |
| 1.4 | `shrub_left/right` | ±1.85 | 0.7499 | 1.475 |

另外 `_install_dressing` 的注释（`scenes.py:1039-1041`）说 "floor items are **>= 1.9 m off the centre
line**"，但 `study_table`/`book_cart`/`trolley` 在 1.60、`shrub_left/right` 在 1.85，都与该规则矛盾。

---

## 7. 新发现："放在货架/桌子上"的高度算术是错的，物体悬在空中

### 7.1 超市：货架顶面不是 1.04 m

`scenes.py:1322-1327` 的注释说：

> The shelf's own `at` height is a CENTRE (1.9 m tall, so its middle is 0.95 m), so its **top face is
> 0.95 + 0.09 = 1.04 m** and these sit on that …

但 `shelf_left/right` 的声明是 `(0.6805, 1.8963, 1.2588)`、base = 0 → 中心 0.9482、**顶面 1.8963 m**
（日志 drawn 也确认高 1.8963）。那个 `+0.09` 是把某个 0.18 m 的尺寸当成了板厚。

后果：

* `mug_a2/b1/c1` 与 `banana` 按"底面 1.04 m"摆放（日志里 mug 的世界盒 z = 1.0400…1.0483），
  比真实货架顶**低 0.85 m**；
* 它们挂在**障碍墙**上（`x ≈ 7.94`），而货架在 `x = 8.9`——**沿 x 相距 0.96 m**，也不在货架上方；
* 所以"货架上的商品"实际上是走廊里 1.04 m 高的壁挂小物。

### 7.2 图书馆：书悬空 0.59 m

`book_a`/`book_b` 的 base 声明为 0.9941（= `study_table` 的声明高度，即"桌面"）。但 `study_table`
实际只画到 **0.3995 m** 高（§1），于是两本书的底面在 `z = 0.9941`、桌面在 0.3995——**书悬空
0.5946 m**。

**没有任何检查把物体 `at.y` 与"它声称站在其上的那个东西的顶面"比较。**

---

## 8. 名字与尺寸不符（语义层面的"不合理尺寸"）

| 场景 | 项 | 声明/绘制 | 问题 |
| :--- | :--- | :--- | :--- |
| 1.2 | `forklift` | 声明 0.891×0.281×0.390，`room_fit=0.36`，画出来 0.891×0.089×0.171 | 名字是叉车，实际是一台被缩到 **39 cm 高**的包装台，且画出来只有 **8.9 cm 宽**；39 cm 低于任何工作台面 |
| 1.5 | `trolley` | `dolly.usd`，画出来 1.259 m 长、**0.154 m 高** | 超市"购物车"是 15 cm 高的平板小车 |
| 1.3 | `chair_row` | 一把 `SM_Chair` (0.600×0.877×0.599) | 名字是"一排椅子"，实际一把 |
| 1.2 | `pallet_stack` | 一个 `pallet`（0.143 m 厚，画出 0.017 m） | 名字是"托盘堆"，实际一块托盘 |
| 1.3/1.5 | `study_table`/`checkout` | 2.474 m 长的工业包装台 | 作为"书桌"过长（是普通书桌的 2.5 倍）；作为收银台尚可 |
| 1.4 | `bench_iron/left/right` | `Roxana_DiningBench` 的长轴沿 z（跨走廊方向） | 长椅横着放，1.50 m 的长度因此**穿过侧墙 0.30 m**（§4） |

---

## 9. 固定几何常量之间的不一致

| 位置 | 问题 |
| :--- | :--- |
| `environment.py:170-172` | `CHANNEL_WIDTH = 0.90` 注释写 "Level 0 default"，但 `LEVEL_CHANNEL_WIDTHS[0] = 0.57 × 2.0 = **1.14**`（`environment.py:224,291`）。默认值比 Level 0 窄 **0.24 m（21%）**；`PANEL_WIDTH = (5.0-0.90)/2 = 2.05`，而 Level 0 的墙板应是 `(5.0-1.14)/2 = 1.93`。任何走到默认值的路径（`_panel_boxes(None)`、`capture_views.py` 的回退）会建出**错误的开口** |
| `environment.py:168-169` | 障碍墙 `WALL_HEIGHT = 2.0`，房间 `ROOM_WALL_HEIGHT = 3.0`，且没有过梁（`_panel_boxes` 只建两侧墙板）：**墙顶上方留了一条 5 m 宽、1 m 高的全宽空隙**。`scenes.opening_box` 的高度也写 2.0，所以可能是刻意的——但"开口"实际是"2 m 隔断下方的空隙"，不是"整面墙上的洞" |
| `scenes.py:54-57` vs `environment.py:165` | marker 0.90 m；Level 0 开口 1.14 m，marker 能穿过——一致（记录在案，无问题） |
| `environment.py:167-169, 493` | 障碍墙 2 m 高、暗色边柱也是 2 m 高——一致（无问题） |

---

## 10. 测量表本身可疑的条目

| 条目 | 情况 |
| :--- | :--- |
| `RackSmall_A7.usd` | `PROP_MEASUREMENTS` 与 `RackSmallEmpty_A1.usd` **完全相同** `(108.0188, 301.0001, 199.8097)`。文件是 2.2 kB 的薄层，二进制里可见 `RackSmallEmpty_A1` + `../Shelves/` 的引用——所以**数值合法**，但 `scenes.py:713` 的注释写 "RackSmall_A7 is 2.2 kB and measures nothing"，与表里"有测量值"**自相矛盾**，属该被清理的陈旧注释 |
| `SM_Clock_04.usd = (0.045, 0.15, 0.45)` | 资产自身就是 45 cm 宽 × 15 cm 高（3:1 横向），作为壁钟偏扁；值来自实测，但建议复核是否为轴的读法 |
| `sign_plaza_fountain_01.usd` | 名为"广场喷泉"，声明 `(0.1621, 0.1108, 0.1621)`——16 cm 的壁挂饰片；值合法，命名误导 |
| `stage1.2` 的 `sign`/`bay_sign` | `asset` 字段放的是 `MI_SignB.mdl`（材质而非 `.usd`），`place_dressing` 当作无资产处理（`:1157-1162` 有记录） |
| 已测量但从未使用的条目 | `table01`、`bench_wrought_iron_02`、`bench_curved_01`、`shopping_cart_corral_01`、`SM_Mug_D1`、`005_tomato_soup_can`、`024_bowl`、`004_sugar_box`、`rubiks_cube`、`bollard_01`、`concrete_block_01`、`small_KLT`、七个 Shapes 原型 |

---

## 11. 建议补上的最小检查（每条都对应上面一个真实缺陷）

1. **逐轴**（不是 sorted）比对 `item["size"]` 与 `PROP_MEASUREMENTS × ASSET_SCALE × room_fit`；
   壁挂件按 `(thick, tall, across) = (x, height, lateral)` 同序。→ 抓 §2。
2. **侧向/地板/天花板/远端墙**的盒包含检查，declared 与 drawn 各一次。→ 抓 §4。
3. **同场景声明盒两两重叠**检查。→ 抓 §5。
4. 走道检查改用**内边缘** `|z| − size[2]/2 >= 1.5`（而不是锚点）。→ 抓 §6。
5. **堆叠一致性**：`at.y` 必须等于所声明支撑物的顶面高度（±1 mm）。→ 抓 §7。
6. 壁挂件的 `size[0]` 必须等于资产的**最薄轴**，否则要求显式旋转。→ 抓 §2/§3。
7. 绘制尺寸必须等于声明尺寸：fit 因子必须带上 `declared`（`min(declared_i/native_i)`），
   或在 `place_dressing` 里对 `drawn vs declared` 直接断言。→ 抓 §1。

---

## 12. 复现命令

```bash
cd /home/eai/EmbodiedBAO-AS-

# scenes.py 可独立导入，逐项打印声明/测量/换算
python3 - <<'PY'
import scenes as sc, os
for s in sc.SCENE_ORDER:
    for it in sc.SCENES[s]["dressing"]:
        b = os.path.basename(it["asset"]) if it.get("asset") else None
        print(s, it["name"], it["mount"], it["size"], b,
              sc.PROP_MEASUREMENTS.get(b), sc.ASSET_SCALE.get(b), it.get("room_fit"))
PY

# 实测的 drawn vs declared（当前 HEAD 的代码与这批日志等价）
grep -h "drawn .* vs declared" diag_logs/markers_stage1.{2,3,4,5}.log

# 现有检查为什么全绿
python3 tools/audit_dressing.py
```

---

### 附：本次审查相对已有文档的增量

| 编号 | 结论 | 是否已有 |
| :--- | :--- | :--- |
| §1 | fit 因子导致全部道具偏小；给出 34/61 项 <50% 与绝对渲染尺寸 | F2/F3 已记录机制；**绝对尺寸清单与"当前版本"修正是新增** |
| §2 | `reading_poster` 的 slot0/slot2 写反，且约定已由逐轴证据确定 | F4 提出疑问但**明确未决定**；**本报告给出定论与唯一违规项** |
| §3 | 壁挂件无旋转，深度轴朝走廊 | F4 提到 banana 197 mm；**"无旋转 + 最薄轴规则"的推广是新增** |
| §4 | 声明盒/实际几何穿透侧墙 | **新增**（F6 只讲天花板，且已过时） |
| §5 | 同场景声明盒 20 处重叠 | **新增** |
| §6 | 走道检查只查锚点、不查半宽；与注释规则矛盾 | **新增** |
| §7 | 货架顶面 1.04 m 算错；书悬空 0.59 m | **新增** |
| §8 | 名称与尺寸不符 | 部分（forklift 注释）已知，**其余新增** |
| §9 | `CHANNEL_WIDTH` 默认 0.90 ≠ Level 0 的 1.14；2 m 墙在 3 m 房 | **新增** |
| §10 | `RackSmall_A7` 注释与测量表矛盾等 | **新增** |
