# STAGE 1 dressing — findings from the text-only verification (2026-10-11, session on the remote machine)

Code verified against: `git rev-parse --short HEAD` = **ae25b0b** ("Collect every instruction the human
gave into one protocol the next session must follow"), working tree clean before and after every run.

Interpreter: `/home/ybh/isaacsim/python.sh` (Isaac Sim 4.5, Python 3.10.15, PIL 12.3.0, numpy 1.26.0).
The `python` on `PATH` here is miniconda Python 3.13 with **no PIL**, and one suite
(`test_no_scene_floor_is_covered_by_a_dark_material`) needs Pillow.  See finding F0.

Every number in this file is copied from a command's own output.  Raw logs are under `diag_logs/`
(`*.log`, already covered by `.gitignore`), one file per command per scene.

---

## STEP 1 — environment and baseline self-check

| # | command | key output | reading | basis |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `git pull --ff-only origin master` | `已经是最新的。` | remote `master` already contained in the local branch | the pull's own line; `git log --oneline -1` = `ae25b0b` |
| 2 | `git ls-files assets \| wc -l` | `1184` | the asset root is complete | required value is 1184 (`REMOTE_WORKORDER.md` §0) |
| 3 | `git ls-files assets/isaac \| wc -l` / `assets/bucket` | `1090` / `95` | both vendored roots are tracked | 1090 + 94 = 1184 |
| 4 | `ls assets/isaac \| head -3` | `Environments` `Props` `Simple_Warehouse` | non-empty | the directory lists |
| 5 | `ls assets/bucket \| head -3` | `Assets` | non-empty | the directory lists |
| 6 | `python tools/audit_dressing.py` | `no problems found by these checks`, exit 0 | the 11 semantic rules pass | the tool's own line |
| 7 | `python tools/prove_audit_fails.py` | `all 4 planted fault(s) were caught`, exit 0 | the audit is a live check, not a decorative one | it plants each fault, sees the audit fail, restores (`CAUGHT` lines name all four) |
| 8 | `python tools/verify_docs.py` | `the handoff documents' checkable claims all hold`, exit 0 | 20 backticked paths and the counted claims hold | the tool's own line |
| 9 | `python tools/check_field_names.py` | three `ok:` lines, exit 0 | no field/action name outside a schema | the tool's three lines |
| 10 | `for f in test_bao_*.py; ... python $f` | `1/39 checks FAILED` in `test_bao_scenes.py` (only that suite fails; 174 checks total: 5+37+5+20+29+17+9+13+39) | **this interpreter cannot run one check** | the `[ERROR] ... ModuleNotFoundError: No module named 'PIL'` line |
| 11 | same loop with `/home/ybh/isaacsim/python.sh` | all nine suites pass: `all 5/37/5/20/29/17/9/13/39 ... passed`, 174 = 174 | **baseline is green under the interpreter the suites were written for** | the nine `all N ... passed` lines; `test_bao_persistence.py` also stops printing its `[skip]` |
| 12 | `ssh -T git@github.com` | `Hi YourClassmateJiangYouning! ...`, exit 0 | push identity is the intended one | `ssh -v` names `/home/eai/.ssh/id_ed25519_bao` as the offered key |
| 13 | `git status --short` | (empty) | nothing was modified before or during STEP 1–3 | the command's empty output |

**F0 — one check is interpreter-bound, and the suites are meant for Isaac's Python.**
`test_bao_persistence.py` says so in its own docstring: "`python3` has no PIL, while
`/home/ybh/isaacsim/python.sh` has Pillow 12.3.0 ... Run the suites that way and this check runs;
run them under python3 and it reports `[skip]`."  `test_bao_scenes.py` has no such guard, so under
the no-PIL interpreter its floor-albedo check is an `[ERROR]` rather than a `[skip]` and the suite
prints `1/39 checks FAILED`.  Under Isaac's Python all 174 pass.  No file was changed for this.

---

## STEP 2 — structure of the 52 dressing items, in text (`--stage_tree`)

Four runs, one per dressed scene (`stage1.1` has no dressing by design and was not touched):

```bash
/home/ybh/isaacsim/python.sh capture_scenes.py --scene <scene> --level 0 --slots 1 --stage_tree
```

**Result: every one of the 47 resolved props is `active=True visible=inherited`, and every one has a
bound material or a `displayColor` somewhere in its subtree.  The 5 items with no asset are boxes by
design and their `clearance` box is `visible=inherited` (visible), which is what makes them the item.**

Per-scene counts, from the `[scene] dressing:` line and the `[tree]` lines in the same run:

| scene | items | resolved `.usd` | `NO ASSET` | `prop` lines with `active=True visible=inherited` |
| :--- | ---: | ---: | ---: | ---: |
| `stage1.2` warehouse | 11 | 8 | 3 (`sign`, `bay_sign`, `forklift`) | 8 / 8 |
| `stage1.3` library | 13 | 12 | 1 (`reading_lamp`) | 12 / 12 |
| `stage1.4` park | 16 | 15 | 1 (`park_sign`) | 15 / 15 |
| `stage1.5` supermarket | 12 | 8 | 4 (`shelf_left`, `shelf_right`, `price_strip`, `promo_banner`) | 8 / 8 |
| **total** | **52** | **43** | **9** | **43 / 43** |

**F1 — the handoff documents' resolved-asset counts are wrong for `stage1.2` (9 claimed, 8 measured)
and for the total (44 claimed, 43 measured).**  `REMOTE_WORKORDER.md` §0 and
`IMPLEMENTATION_PROTOCOL.md` §1 both carry the table `11 items / 9 from assets`.  The run above prints
`11 items, 8 from assets`.  Two independent counts agree on 8: the `[scene] dressing:` summary line
and the eight `RESOLVED` lines.  `MANUAL.md`'s own table says 8 for `stage1.2`, so the manual and the
code agree and the two newer documents are the ones that drifted.
**Why no test caught it:** `test_the_manual_states_the_scene_facts_the_code_holds` checks `MANUAL.md`
only; nothing checks `REMOTE_WORKORDER.md`/`IMPLEMENTATION_PROTOCOL.md`'s tables.  The `verify_docs.py`
claim that "docs state the total 173" is about the test count, not the asset counts.

Material evidence per scene (one line per item, from the `[tree]` dump):

- `stage1.2`: `traffic_cone` -> `S_TrafficCone ... material=M_TrafficCone`; `pallets`, `klt_bins`,
  `crate`, `carton`, `rack`, `rack_tall` carry a `Geometry`+`Materials` or `Looks`/`Visuals` child,
  which is where their bindings live (the same alternation appears in `stage1.4`).
- `stage1.3`: `reading_poster` binds `MatteWhite`/`SolPoster`/`Frame` per part; `chair_row` ->
  `material=MI_Chair`; `armchair` -> `MI_Sofa`; `book_a`/`book_b` -> `MI_BooksA`;
  `book_cart` -> `OmniPBR`; `clock`/`study_table`/`stock_carton`/`cabinet` via `Geometry`/`Materials`.
- `stage1.4`: `tree_conifer` and `tree_fir` -> `material=Pine_bark` with `displayColor`;
  `tree_sapling` -> `bark3` + `American_Beech_leaf`; the rest via instancer `Looks` subtrees.
- `stage1.5`: `mug_a2`/`mug_b1`/`mug_c1` -> `M_Mug_A2`/`M_Mug_B1`/`M_Mug_C1`;
  `banana` -> `OmniPBR`; `trolley` -> `OmniPBR`; `produce_bins`/`checkout`/`stacked_boxes` via
  `Geometry`+`Materials`.

A separate on-stage probe (`/tmp/probe_props.py`, temporary, not committed; output in
`diag_logs/probe_props_stage1.*.log`) walks each `/World/Dressing/<name>/prop` subtree and counts
every prim with a resolvable bound material and every prim with a non-empty
`primvars:displayColor`.  Counts per item (mats / displayColor prims), the ones that decide the
question being the zeros:

| scene | item | mats | displayColor prims | reading |
| :--- | :--- | ---: | ---: | :--- |
| 1.2 | `rack`, `rack_tall` | 3 | 3 | bound |
| 1.2 | `klt_bins` | 3 | 3 | bound |
| 1.2 | `carton` | 4 | 4 | bound |
| 1.2 | `pallets`, `pallet_stack`, `traffic_cone` | 1 | 0 | bound via an instancer's `Looks`/`Materials` scope |
| 1.2 | `crate` | 1 | 1 | bound |
| 1.3 | `reading_poster` | 4 | 3 | bound per part |
| 1.3 | `clock`, `book_a`, `book_b` | 2 / 1 / 1 | 0 | bound |
| 1.3 | `study_table`, `stock_carton` | 4 | 4 | bound |
| 1.3 | `book_cart` | 10 | 0 | bound |
| 1.3 | `cabinet` | **0** | **0** | the referenced cabinet prims are all `link`-style with no binding and no displayColor of their own; the item still has `colour=DRESSING_GREY` on its `clearance` box, but that box is `invisible` once a prop resolves |
| 1.4 | `tree_conifer`, `tree_fir`, `tree_sapling` | 3 / 7 / 2 | 3 / 8 / 2 | bound (`Pine_bark`, `bark3`, `American_Beech_leaf`) |
| 1.4 | `shrub_left`, `shrub_right` | 3 | 3 | bound |
| 1.4 | `picnic_table`, `bench_iron`, `bin_square`, `railing`, `fountain`, `bench_left`, `bench_right`, `litter_bin`, `planter_left`, `planter_right` | **0** | **0** | same as `cabinet`: the mesh is present and scaled, but nothing in the subtree carries a binding or a displayColor |
| 1.5 | `mug_a2`, `mug_b1`, `mug_c1` | 1 | 0 | bound (`M_Mug_A2`/`M_Mug_B1`/`M_Mug_C1`) |
| 1.5 | `banana`, `trolley` | 1 / 10 | 0 | bound (`OmniPBR`) |
| 1.5 | `produce_bins`, `checkout`, `stacked_boxes` | 1 / 4 / 3 | 1 / 4 / 3 | bound |

The 13 zeros are **not** evidence of an invisible prop — the tree dump shows the mesh prims — but
they are evidence that the material question cannot be answered by this probe alone: the binding for
these assets lives on a prototype/instancer scope that
`UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()` does not resolve when the prim is walked
directly.  The tree dump remains the primary evidence, and it shows a `Looks`, `Materials` or
`Geometry` child for every one of the 43.  The one item whose subtree has no material anywhere in
the tree dump either is `cabinet` (its children are `sektion`, `rootJoint`, `door_*_link`,
`drawer_*`), which is reported as-is: whether it renders grey, black or textured is **unverified**
here, because that decision is in the asset and in `use_mdl=False` handling, not in the catalogue.

---

## STEP 3 — sizes, in text (`--dressing_report`)

```bash
/home/ybh/isaacsim/python.sh capture_scenes.py --scene <scene> --level 0 --slots 1 --dressing_report
```

**Result: no `FITS BADLY`, and no line in any of the four reports has a `drawn` component greater than
its `declared` component.  The clipping criterion holds on all 43 resolved props in all four scenes.**

```
grep -c "FITS BADLY" diag_logs/report_stage1.2.log -> 0
                     report_stage1.3.log           -> 0
                     report_stage1.4.log           -> 0
                     report_stage1.5.log           -> 0
```

and a parser over the same four logs re-reads every `[drawn (...) vs declared (...)]` pair and tests
`drawn[i] > declared[i]` component by component (tolerance 1e-9):

```
resolved prop(s) parsed = 43; drawn > declared on any axis = 0
```

43 is the same count the four `--stage_tree` runs produced, so the two text channels agree on which
props exist.

**What that does and does not prove.**  It does prove the stated requirement (`drawn` never exceeds
`declared`).  It does **not** prove the props are the right size, and the reason is structural: the
fit factor is `min over axes of 1/native`, capped at 1.0, and the declared scale then multiplies it,
so `drawn = declared x min(1/native) x native`.  On the axis where `native` is largest the drawn
extent is exactly the declared one, and every other axis is strictly smaller.  **`drawn <= declared`
is therefore a property of the arithmetic, not an independent measurement.**  A clipping check that
cannot fail is exactly the class of check this project has already been burned by
(`REMOTE_WORKORDER.md` §2.D), so it is reported as such here.

---

**F2 — the measured defect the reports do show: every prop is drawn smaller than its box

`drawn / declared` per axis, for all 43 resolved props, computed from the `[drawn ... vs declared ...]`
lines above (no new measurement, just the division):

| item | scene | axis where it exactly fills | worst axis (`drawn/declared`) |
| :--- | :--- | :--- | :--- |
| `book_a`, `book_b` | 1.3 | height | **0.020** (4 mm thick against a 20 mm box) |
| `railing` | 1.4 | length | **0.033** (2.4 mm thick against 72.8 mm) |
| `banana` | 1.5 | x | **0.039** |
| `clock` | 1.3 | height | **0.044** |
| `reading_poster` | 1.3 | x | **0.048** (52 mm wide and 505 mm tall against a 1092 x 743 mm frame) |
| `mug_a2`, `mug_b1`, `mug_c1` | 1.5 | height | **0.089 / 0.091** (a 9 cm mug drawn 8.5 mm wide) |
| `bench_left`, `bench_right` | 1.4 | length | **0.108** (20 mm thick against 185 mm) |
| `pallet_stack`, `pallets` | 1.2 | x | **0.117** |
| `klt_bins` | 1.2 | y | 0.197 |
| `crate` | 1.2 | x | 0.283 |
| `traffic_cone` | 1.2 | y | 0.334 |
| `rack`, `rack_tall`, `bookshelf_left`, `bookshelf_right` | 1.2, 1.3 | height | 0.359 |
| `checkout`, `study_table` | 1.5, 1.3 | x | 0.308 |
| `trolley`, `book_cart` | 1.5, 1.3 | y | 0.350 |
| `tree_conifer` | 1.4 | height | 0.417 |

The pattern is uniform: on exactly one axis the ratio is 1.000, on the others it is that axis's
`declared/native` divided by the largest `declared/native`.  The defect is the one the human already
described as "道具偏细" (`IMPLEMENTATION_PROTOCOL.md` §9.2), now with a number for every item.

**Mechanism, with the code and the units.**  `scene_builder.fill_scale_for_unit_parent` returns
`factor = min(min(1/native_i), 1.0)`, where `native` is the prop's own bound relative to its parent,
and never receives the declared size.  That is correct only if the parent is a **unit** cube.  It is
not: since `a549ecd` the tree is

```
/World/Dressing/<name>            Xform   <- translate + scale of the DECLARED size (to_world_size)
/World/Dressing/<name>/clearance  Cube    <- unit cube, hidden when a prop resolves
/World/Dressing/<name>/prop       Xform   <- the referenced mesh, scaled by the factor above
```

so the prop must fill its parent's local space, which is the declared size, not 1.  Multiplying gives
`drawn = native x factor x declared = declared x min(1/native) x native`, i.e. a prop at 2 %–88 % of
the declared box depending on its proportions.**

**F3 — three places in the tree state an arrangement that no longer exists.**
`fill_scale_for_unit_parent`'s docstring says "the prop at `<box>/prop` is a CHILD of the clearance
Cube, and that Cube already carries a scale op of the declared size" (line 384) and explains the
factor as "1/native: no declaration enters into it" on that basis.  Since `a549ecd` the prop is a
**sibling** of the box and the scale lives on the item Xform, which is why the `1/native` factor is
now the wrong one.  The same stale ancestor is used as the justification for choosing
`proxy_prim.GetParent()` as the frame in the measurement loop (lines 418–422: "Passing the clearance
box as the ancestor ... asks for the bound of a prim in the space of an ancestor that is not an
ancestor").  This is the "comment defending behaviour the code no longer has" that
`IMPLEMENTATION_PROTOCOL.md` §8.9 and `audit_dressing.py`'s phrase scan exist for; the phrase list
does not contain this one, and no test reads prose for this claim.

**F4 — the declared size sits in a different frame from the one the code says it is in.**
`scenes._dressing_wall` unpacks `x_thick, tall, across = size` and stores the tuple unchanged, so a
caller must pass `(wall-normal, in-plane, in-plane)`.  Every call site instead passes the catalogue
tuple `(x, height, lateral)` from `PROP_MEASUREMENTS`; `place_dressing` then converts it with
`to_world_size` again (line 534).  Measured consequences, all from the tables above:
`fountain` declared world `(0.1621, 0.1621, 0.1108)` equals its measured size; `reading_poster`
declared world `(0.0524, 1.092, 0.7426)` equals `to_world_size(measured)`; `banana` declared world
`(0.1972, 0.0386, 0.0741)` equals `to_world_size(measured)` too.  For the banana that puts the
**0.1972 m** axis on `size[0]`, which `_dressing_wall` documents as the wall normal — a 197 mm
protrusion from the wall, and the drawn x is 0.0389 m only because of F2.  The docstring at
`_dressing_wall` (lines 484–489) says the opposite of the code in two consecutive sentences
("converting a wall item through `to_world_size` swapped its width into the world's height" versus
the conversion `place_dressing` actually performs).  **Which of the two orders is intended is not
decided by this file; the contradiction is recorded, not resolved.**  The banana's own comment
(lines 1089–1093) asserts a third order (`A -> (A[2], A[1], A[0])`) that the helper does not
implement.  **Unverified:** whether the wall items are visually rotated by 90 degrees in the wall
plane; the two candidate orders differ only for items whose in-plane sides differ, and no text
report prints orientation.

**F5 — the report cannot see a prop that is off-centre.**  `place_dressing`'s verification computes
`drawn` as `hi - lo` from a bound range; it never compares `lo`/`hi` with the declared box's own
span, so a prop of the right size standing half a metre away from where it was declared would still
print a green line.  **F6** below is the measurement that closes this, and it shows the gap is not
hypothetical.

---

**F6 — 25 of the 43 resolved props stand outside their own declared box; three trees pierce the ceiling

The probe (`/tmp/probe_bounds.py`, temporary, not committed; output in
`diag_logs/probe_bounds_stage1.*.log`) reads, for each resolved prop: the item Xform's authored
`xformOp:translate`/`xformOp:scale`, the prop's bound **relative to its own item Xform** (its local
space), and the prop's bound relative to `/World/Dressing` (world metres).  The algebra is checked in
the same run: for every item
`world_centre == to_world(at) + declared_size * local_centre` held to 2 mm, so the local measurement
is the one the renderer uses and the offsets are not a measuring artefact.

The recipe that turns the two into a verdict: the declared box is centred on `at` with the declared
size, and the drawn box is `[at + decl*local_lo, at + decl*local_hi]`, so a negative `local_lo` or a
`local_hi` beyond the local half-extent of the unit cube puts drawn geometry outside the box the
clearance, occlusion and walking-band checks were written against.

```
local bounds measured for 43 item(s) across the four scenes
25 of 43 measured item(s) leave their declared box
```

| scene | item | axis out | prop z-range (m) | declared box z (m) | by |
| :--- | :--- | :--- | :--- | :--- | ---: |
| `stage1.4` | `tree_conifer` | z | 1.1738 .. **3.5738** | 0.0 .. 2.4 | **+1.17 m** |
| `stage1.4` | `tree_fir` | z | 1.1500 .. **3.4500** | 0.0 .. 2.3 | **+1.15 m** |
| `stage1.4` | `tree_sapling` | z | 1.0998 .. **3.2998** | 0.0 .. 2.2 | **+1.10 m** |
| `stage1.2` | `rack`, `rack_tall` | z | 0.9978 .. **2.9934** | 0.0 .. 1.9956 | +1.00 m |
| `stage1.3` | `bookshelf_left`, `bookshelf_right` | z | 0.9978 .. **2.9934** | 0.0 .. 1.9956 | +1.00 m |
| `stage1.4` | `bench_left`, `bench_right` | x | (z in box) | 1.5410 .. 3.2590 | +0.86 m along x |
| `stage1.4` | `litter_bin` | z | 0.5285 .. 1.5696 | 0.0 .. 1.0411 | +0.53 m |
| `stage1.4` | `bin_square` | z | 0.5056 .. 1.5168 | 0.0 .. 1.0112 | +0.51 m |
| `stage1.3` | `chair_row` | z | 0.4386 .. 1.2081 | 0.0 .. 0.8772 | +0.33 m |
| `stage1.4` | `shrub_left`, `shrub_right` | z | 0.4048 .. 1.3048 | 0.0 .. 0.9000 | +0.40 m |
| `stage1.3` | `armchair` | z | 0.3950 .. 0.9951 | 0.0 .. 0.7900 | +0.21 m |
| `stage1.4` | `planter_left`, `planter_right` | z | 0.3848 .. 0.9817 | 0.0 .. 0.7726 | +0.21 m |
| `stage1.5` | `stacked_boxes` | z | 0.2586 .. 0.7717 | 0.0 .. 0.5172 | +0.25 m |
| `stage1.4` | `picnic_table` | z | 0.3166 .. 0.7266 | 0.0 .. 0.6403 | +0.09 m |
| `stage1.4` | `bench_iron` | z | 0.4958 .. 1.2516 | 0.0 .. 0.9895 | +0.26 m |
| `stage1.4` | `fountain` | z | 1.8000 .. 1.8757 | 1.7446 .. 1.8554 | +0.02 m |
| `stage1.3`, `stage1.5` | `book_cart`, `trolley` | y | (z in box) | 1.0394 .. 1.4801 | +0.13 m along y |
| `stage1.2` | `pallets`, `pallet_stack` | x | (z in box) | 1.9934 .. 3.2066 | +0.0016 m along x |

**The ceiling interaction is the one the human named.**  The room is 3.0 m high and
`tree_conifer` reaches **3.5738 m**, `tree_fir` 3.45 m and `tree_sapling` 3.2998 m — all three stand
through the ceiling, and all three also float: the conifer's lowest geometry is 1.1738 m above the
floor.  The comment in `scenes.py` that says the trees were cut to "1.6 m and 1.55 m because the
ceiling is 3 m" refers to the declared heights (2.4/2.3/2.2 m), which are not what the stage draws.

**Why the props leave their boxes: the assets are not authored on the unit cube.**  The probe
`/tmp/probe_asset.py` reads each asset's own root bound in its own units:

```
Chinese_Juniper        x[-54.3605, 53.3833] y[-52.1413, 53.8737] z[-2.7802, 251.6417]
Barberry               x[-50.3363, 50.3372] y[-49.1925, 50.5137] z[-6.0049, 113.6533]
RackSmallEmpty_A1      x[-54.0094, 54.0094] y[-99.9048, 99.9050] z[0.0000, 301.0001]
bench_curved_01        x[0.0000, 4.0906]   y[-0.3706, 1.7992]   z[-0.0005, 0.4412]
safety_railing_01      x[-1.1130, 1.1131]  y[-0.0372, 0.0357]   z[-0.1008, 1.0652]
trashcan_cylinder_01   x[-0.3610, 0.3610]  y[-0.3610, 0.3610]  z[0.0080, 1.0491]
SM_Mug_A2              x[-0.0462, 0.0462]  y[-0.0462, 0.0810]  z[0.0000, 0.0910]
table01                x[-0.4054, 0.4054]  y[-0.4054, 0.4054]  z[-0.0055, 0.6348]
```

Five of the eight are anchored at a **boundary**, not at the geometry's centre: the trees, the
shrubs and the rack sit with their feet on their own `z`/`y = 0` plane, and `bench_curved_01` starts
at `x = 0`.  `place_dressing` multiplies such a bound by the declared size and places the result
around the item's origin, so a base-anchored prop is drawn from the origin upward instead of being
centred on it, and a prop that starts at `x = 0` is drawn from `at` forward instead of centred on
`at`.  Measured, the displacement along an axis is `declared_i * local_centre_i` (0.5 x declared for
a base-anchored asset).  That is exactly the offsets in the table above: conifer `2.4 x 0.4891 =
1.174`, rack `1.9956 x 0.5 = 0.998`, bench_left `1.7181 x 0.5 = 0.859`, shrub `0.9 x 0.4498 = 0.405`.

**Consequence for the acceptance criterion.**  The report's `drawn <= declared` (STEP 3) is a
statement about **extents only**, and it holds.  The statement the human actually needs —
"the prop is inside the volume the checks cleared" — is **false for 25 of the 43 resolved props**,
and three of them are inside the ceiling.  The two statements are not in conflict: an extent is
`hi - lo`, and all of these props do have an extent no larger than their declared size on each axis;
they are simply not *centred* on it.  This is why F5's positional check is required in addition to
STEP 3's extent check.

**Unverified:** whether the ceiling actually intersects the tree meshes in the rendered frame (the
bounds prove the geometry enters `z > 3.0`; the ceiling prim's own extent has not been measured
here), and how many of the 25 violations are visible from the eye camera rather than only from an
external one.

---

## What was not done, and why

- No code, scene, or default value was changed.  STEP 1's gate is green under Isaac's Python
  (F0 is an interpreter mismatch, not a code failure), but the three text readers disagree with
  three separate claims the project was holding:
  * the handoff documents' asset counts (F1);
  * STEP 3's clipping criterion, which is satisfied by the arithmetic and therefore cannot fail
    (F2), while the property it is standing in for is false for 25 of 43 props (F6);
  * the "props are the size they were declared" reading of `drawn` (F2), which is true per axis on
    one axis and 2 %–88 % on the others.
  The next action is therefore a decision rather than an edit: fixing F2 changes every drawn size in
  four scenes, F6 changes where every base-anchored prop stands, and F4 decides whether wall items
  are oriented as measured.
- STEP 4 (locating the high-frequency pattern on flat-painted walls), STEP 5 (the "too thin" props)
  and STEP 6 (the five boxes) were not started.  STEP 5's root cause is already localised in F2 with
  a number per item; F6 is the larger version of the same root cause (the prop's local placement is
  neither centred nor on the unit cube).
- No commit has been made yet in this session; this file is the record the human asked for, and the
  raw command outputs it cites are the four `diag_logs/tree_*`, four `diag_logs/report_*`, four
  `diag_logs/probe_props_*` and four `diag_logs/probe_bounds_*` files (`*.log`, git-ignored).

---

# Second pass: what was fixed, and what the remaining "too thin" items actually are

Code state: `git rev-parse --short HEAD` = the commit before this one plus the working-tree changes
listed in the commit that follows this file.

## FIXED and verified on the stage

**The offset (F6).** `fit_reference_to_local_unit` now returns `((-centre/native), (factor, factor,
factor))` and `place_dressing` applies the translate BEFORE the scale on the prop, so the prop's own
centre lands on the item's anchor.  Verified by rebuilding all four scenes and measuring every prop's
world box against its declared box (`diag_logs/verify_*`, and the `[world box ...]` and
`[fills ...]` fields now printed by every run):

| scene | props inside their declared box | outside | `fit_error` |
| :--- | ---: | ---: | ---: |
| `stage1.2` | 8 | 0 | 0 |
| `stage1.3` | 12 | 0 | 0 |
| `stage1.4` | 15 | 0 | 0 |
| `stage1.5` | 8 | 0 | 0 |

Before this change the same measurement said **25 of 43 props left their box**, the conifer reached
z = 3.5738 m in a 3.0 m room and the rack 2.9934 m against a declared 1.9956 m.  Now the conifer's
world box is `0.0 .. 2.4` in z and sits on the floor.

**The scale was left at `1/native` on purpose, and that was measured, not assumed.**  A version that
used `declared/native` drew the conifer 5.76 m tall for a 2.4 m box.  The two-point calibration on the
real stage (metersPerUnit 0.01, item Xform scaled to (1.0164, 1.0001, 2.4)): the referenced juniper
measures 109.51 m across at prop scale 1 and 219.02 m at scale 2.  Scale is linear, the measured bound
is already in the item's world metres, and `1/native` is the multiplier.

**The audit had a rule that could not see what it named.**  Its second op-order rule searched for the
PROP's calls (`apply_translate`/`apply_scale`) while claiming to check the ITEM's translate/scale, so
it was a duplicate of the rule above it and the fault it was supposed to catch was reported MISSED.
It now searches `.AddTranslateOp` and `item_xform.AddScaleOp`.  `tools/prove_audit_fails.py` grew from
4 to 11 planted faults, including the two that shipped (the missing offset and the unit of the offset);
all 11 are now caught.

**A text criterion for "too thin", so it never needs a picture.**  Every resolved prop now prints
`[fills (x, y, z) of its box]` -- the drawn extent divided by the declared extent, per axis -- and the
report prints one summary line naming the items whose worst axis is below 0.9.

## The remaining "too thin" items are a declaration problem, and here is the number for every one

`[scene] dressing` lines, `stage1.4` (worst fill ratio first): `railing 0.033`, `bench_left 0.108`,
`bench_right 0.108`, `tree_conifer 0.417`, `tree_fir 0.46`, `tree_sapling 0.564`, `bin_square 0.579`,
`picnic_table 0.64`, `bench_iron 0.663`, `fountain 0.684`, `litter_bin 0.694`, `planter_left 0.706`,
`planter_right 0.706`, `shrub_left 0.833`, `shrub_right 0.833`.

The cause is arithmetic and fully determined: the factor is `min(1/native)` and the parent Xform
multiplies it by the declared size, so `drawn = measured_native x min(1/native) x want`.  The tight
axis fills to `want` exactly and **every other axis is `want_axis x (native_min/native_axis)`**.  A box
whose proportions are not the asset's therefore cannot be filled under a uniform fit.  The fill ratio
above is exactly `native_min/native_axis` for each item, which is why it is the same number for all
three axes of a given asset when the box and the asset agree in shape (the mug: 0.716/0.673/0.696 --
they differ only by the asset's own aspect).

Consequences, so the decision is concrete:

* `railing`: drawn 2.2261 x **0.0024** x 0.6107 m against a declared 2.2261 x 0.0728 x 1.166.  The
  asset's own measurement is 2.2261 x 1.166 x 0.0728, so the declared box has the railing's HEIGHT
  where its THICKNESS is and the other way round, which is what leaves 2.4 mm.
* `bench_left`: drawn 1.7181 x 0.4834 x **0.02** against a declared 1.7181 x 0.9113 x 0.1855.  The
  asset measures 4.0906 x 2.1698 x 0.4417 (a shallow curved bench: 4.09 m long, 2.17 m across,
  0.44 m high), so its own proportions are 1 : 0.53 : 0.108 and the thin axis is a property of the
  asset, not of the fit.
* `tree_conifer`: drawn 0.4304 x 0.4167 x 2.4 against a declared 1.0164 x 1.0001 x 2.4; the asset is
  107.74 x 106.02 x 254.42 in its own units, i.e. much narrower than wide, so a 1.02 m-wide box
  cannot be filled on every axis at once.

`scenes.PROP_MEASUREMENTS` therefore already contains the data needed to decide: for an item that
should be as wide as it is tall, the declared box has to be `measured x k` for one `k`, not three.
The two routes are the ones `IMPLEMENTATION_PROTOCOL.md` §7 already names -- (a) restate the declared
sizes as the measured shape times one factor, or (b) swap in assets whose natural proportions match
the intended box.  Either changes the declared sizes of most items in four scenes, so it is recorded
here with the numbers rather than applied unilaterally.

**What was tried and disproved, so it is not tried again.**  A rotation is not the fix: applying
`rotateXYZ(90,0,0)` (and the other axis-aligned 90-degree rotations) to the existing props was
measured on the stage, and none of them makes a swapped box fill (railing `X90` gives
2.2261 x 0.0381 x 0.0382, `Z90` gives 0.0728 x 0.0728 x 0.6107).  A `declared/native` factor is not
the fix either: it drew the conifer at 5.76 m in a 2.4 m box.  What the fill column shows is a
property of the declaration, not of the placing code.

---

# STEP 4 -- where the pattern on flat-painted surfaces comes from

Method: `capture_scenes.py` gained three optional flags (`--light_intensity`, `--light_radius`,
`--dome_intensity`) that are passed through to `environment._setup_lights` via `task_dict`, the same
route `capture_views.py --light_intensity` already used.  An unset flag leaves `environment.py`'s own
default in force, so the flags cannot become a second source of truth for the tuned exposure.
Four renders of `stage1.4`, one slot, same camera and same scene in every run; then the frames are
measured with a temporary script (not committed) that reports the mean, the mean absolute
4-neighbour difference (`hf`), the same after 2/4/8x box downsampling, and the median `std/mean` of
48x48 patches whose mean is above 20.

**The surfaces are flat paint in every run.**  Each of the four configurations prints
`[scene] side_wall flat-paint`, `[scene] ceiling flat-paint`, `[scene] far_wall flat-paint`.  The
`floor` line is absent for `stage1.4` on purpose -- it keeps the default ground, which is its outdoor
cue.  With `use_mdl False` these are a single `displayColor` each, so the surface itself carries no
spatial variation whatever.

| render | `--light_intensity` | `--dome_intensity` | eye mean | eye hf | eye hf after 8x | eye patch cv (median) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default` | 1500 | 300 | 133.45 | 13.631 | 6.298 | **0.1444** |
| `nodome` | 1500 | 0 | 132.19 | 13.359 | 6.208 | **0.1461** |
| `nosphere` | 0 | 300 | **1.91** | 2.259 | 1.044 | (1 patch above 20) |
| `nodome_nosphere` | 0 | 0 | **0.00** | 0.000 | 0.000 | (none) |

Same runs, external camera: `default` mean 160.58, hf 3.859, patch cv **0.0025**.

**What that establishes.**

  1. **The pattern is structured, not per-pixel dithering.**  In the default eye frame `hf` is 6.298
     after eight halvings (1024 px -> 8 px), i.e. only about half of it averages away.  A pattern that
     survived to 8 px is a several-pixel-scale structure, which is what "high-frequency pattern"
     looked like; per-pixel noise would have gone to near zero.
  2. **It scales with the lighting, not with the distance to the surfaces.**  The same surfaces from
     the external camera, with the same lights, have a patch cv of 0.0025 against the eye camera's
     0.1444 -- a factor of 58.  The eye camera is the near one (0.5 m into a 16 m corridor, looking
     down it), so the non-uniformity grows with proximity to the lit surface.
  3. **The dome is not the source.**  Setting `--dome_intensity 0` changes the eye frame by 1.3 in
     mean and 0.27 in hf, while the external frame's mean falls from 160.58 to 32.23: the dome lights
     the far view and the ceiling, and removing it leaves the eye frame's pattern at 0.1461.
  4. **With the eight SphereLights at 0 the eye frame is essentially black** (mean 1.91, and 0.00 with
     the dome also off), so "does the pattern disappear when the SphereLights are off" cannot be
     answered by looking: there is no light left to see anything by.  That is a property of this
     room's exposure -- the corridor is lit entirely by the eight sphere lights plus a 300-intensity
     dome -- and it means the pattern has to be attributed by varying a light PARAMETER, not by
     switching lighting off.

**Not yet established.**  Which parameter of the SphereLights produces the structure (radius, count,
or the per-pixel sampling of an area light near a surface), and whether the same structure appears in
the other three dressed scenes.  Both are being measured rather than guessed; the runs for
`--light_radius 0.0`, `--light_radius 4.0` and `--light_intensity 500` are in `diag_logs/lights_r0`,
`lights_r4`, `lights_dim500`.

## STEP 4 continued -- the pattern is temporal, and it is not the assets

Three further measurements, each aimed at a candidate cause.

**It is temporal, not a texture.**  Render the same pose twice in one run and difference the frames
(`diag_logs/noise`, `diag_logs/noise.log`):

```
eye frame 0 vs 1: mean abs diff = 17.470629  max = 224.0
eye frame 1 vs 2: mean abs diff = 16.035284  max = 204.0
identical frames: False
iso frame 0 vs 1: mean abs diff = 12.417139
iso frame 1 vs 2: mean abs diff =  2.558236
```

A static scene, a static camera, and consecutive frames that differ by 16-17 levels on average with
pixels changing by up to 224.  A `displayColor` cannot do that; the renderer is producing a different
image each time.  In a 60-frame sequence the successive difference alternates between 0.000 and
11-16 (`diag_logs/converge.log`: frame 21 = 0.000, frame 26 = 11.917, frame 31 = 0.000), so frames
come back unchanged on some reads and fully different on others.

**It does not converge.**  A static scene must converge to one image if the renderer accumulates.
It does not (`diag_logs/settle.log`): with a reference taken after the 3 render steps the capture
path uses, the mean absolute change is still 15.2 after 10 more steps, 14.9 after 20, 15.4 after 40
and 15.6 after 80.  Turning the temporal reconstruction off (`/rtx/post/dlss/execMode = 0`, which
already read 0) moved the frame-to-frame difference from 15.079 to 14.435 (`diag_logs/aa.log`).

**It is spread over the whole frame and is not the dressing.**  The change is 12-19 levels in every
cell of a 4x4 grid, so it is not concentrated on any one surface.  Rendering `stage1.4` with
`--parts marker,materials` (no dressing at all) removes every MDL error in the log and changes the
eye frame's patch cv only from 0.1355 to 0.1298, its `hf` only from 11.617 to 11.263.  The eight MDL
compile failures the runs print -- `bark3`, `TreeBark_10`, `Pine_needles`, `Barberry_Mat`,
`American_Beech_leaf`, `SimPBR` -- belong to the vegetation props and are visible only when the
dressing is in the frame; they are real (this machine's material library cache is not writable, which
is the mechanism the project already recorded) but they are not what puts a pattern on the walls.

**What the pattern is, in one line.**  The four room surfaces are a single flat colour each, and the
frame still changes by ~15 levels per frame with the lights on, over the whole image, from the near
camera and not from the far one (patch cv 0.1355 against 0.0028), with the dressing removed, and
without converging in 80 render steps.  That is an undersampled render of a scene lit by eight 1 m
sphere lights 2.6 m up with a 300-intensity dome: the per-pixel estimate has not settled.  The two
knobs that would settle it are samples per pixel and the light setup, and both live in the render
configuration rather than in the scene skin.

**Not established, and deliberately not guessed.**  The exact render setting whose default leaves it
undersampled; whether `--mdl` (which the project keeps off because the material cache is not
writable) is what a production render would use; and whether the other three scenes show the same
magnitude.  None of those were measured here.

---

# Third pass: the replacement route, measured, and what it actually supports

The owner chose route (b) -- replace the worst-fitting props with better assets.  This is what
measuring that route found, and it does not support the replacement it was meant to.

## Every current asset already matches its declared box

For each weak item, the current asset and every candidate were fitted into that item's declared box
with the uniform factor, and the resulting world box was compared with the declared box
(`diag_logs/replace2_stage1.*`).  `worst_ratio_dev` is the largest per-axis disagreement:

| item | declared world (m) | current asset | dev | best candidate | dev |
| :--- | :--- | :--- | ---: | :--- | ---: |
| `crate` | 0.6015, 0.4018, 0.17 | `SM_Crate_A07_Yellow_01` | **0.000** | `SM_Crate_A07_01_1` (same mesh) | 0.000 |
| `carton` | 0.299, 0.6216, 0.3019 | `sm_whitecorrugatedbox_b20_brown_01` | **0.000** | `...b04_brown_01` cube | 0.519 |
| `book_a` | 0.02, 0.145, 0.205 | `SM_Book_01` | **0.000** | `...b04_brown_01` cube | 0.903 |
| `railing` | 2.2261, 0.0728, 1.166 | `safety_railing_01` | **0.001** | `bike_rack_01` | 0.542 |
| `bench_left` | 1.7181, 0.9113, 0.1855 | `bench_curved_01` | **0.000** | `Roxana_DiningBench` | 0.897 |
| `tree_conifer` | 1.0164, 1.0001, 2.4 | `Chinese_Juniper` | **0.000** | `Grass_Short_A` | 0.948 |
| `shrub_left` | 0.7572, 0.7499, 0.9 | `Barberry` | **0.000** | `Goldflame_Spirea` | 0.461 |
| `bin_square` | 0.5855, 0.7957, 1.0112 | `trashcan_square_01` | **0.000** | `dumpster_lrg_01` | 0.403 |

Every current asset fits its box exactly; every candidate measured is worse.  Six candidates were
fetched from the public bucket for this test (`dumpster_lrg_01`, `dumpster_lrg_02`,
`dumpster_blue_sm_01`, `bike_rack_01`, `concrete_block_02`, `lamppost_single_01`), measured, and
removed again because none belongs in the project.  Two assets the local library holds cannot be
used at all: `SM_BooksSet_11_167` references an `SM_BooksSet_11.usd` that is not vendored, and
`RackLong_A2` references missing cardboards and pallets.

## Why the fill ratios are low, and why that is not an asset defect

With a uniform factor `k = min(1/native)` and a parent scale of `want`, the drawn box is
`want_i x native_i x k`.  The axis with the largest `native` therefore fills its declaration
exactly, and every other axis fills `native_i / native_max`.  A box whose proportions are looser
than the asset's cannot be filled on those axes by ANY uniform multiple -- and forcing it would
require a per-axis factor, which reshapes the prop and is forbidden.

That is what the "too thin" numbers are: the loose side of a box, not a small prop.  The tight axis
is at the declared size by construction, so `railing` really is 2.2261 m long and 1.166 m in the
direction its 1.166 m declaration describes; what reads as 2.4 mm is the box's 0.0728 m thickness
axis, which the asset's own 0.0728 m measurement fills to `0.0728 x native_thin/native_max`.

Verified for the record after this analysis: all four scenes report `FITS BADLY` 0 times and the
same fill ratios as before, and the nine suites pass 174 of 174.

## A stale comment removed on the way

`scenes._dressing_wall` carried the claim that "place_dressing takes this tuple as already being in
the axes the stage uses and does not convert it".  It does convert it -- `to_world_size`, for both
mounts, which is what `tools/audit_dressing.py` rule 1 asserts and what
`test_no_dressing_protrudes_into_the_corridor` reads.  The comment was left over from a revision in
which that helper swapped the outer two components on the way in; it no longer does, so the comment
now describes the single conversion that exists.  No behaviour changed: the working tree differs from
the verified commit only in comments, docstrings and one audit message.

---

# STEP 4 closed, and the five previews

## The mechanism, with the number that identifies it

RaytracedLighting -- the mode every previous preview ran in -- does not converge for a STATIC scene.
Two renders of the same pose differ by 14.3-15.9 grey levels on average, with individual pixels
changing by up to 224, and in a flat-painted wall (one `displayColor`, which cannot carry a pattern)
every 16x16 block has a standard deviation of 8-20 levels, so no part of a wall is ever a flat
colour.  Measured per scene (`diag_logs/pt_stage1.*`):

| scene | RaytracedLighting | PathTracing 1 spp |
| :--- | ---: | ---: |
| `stage1.1` | 14.253 | 0.217 |
| `stage1.2` | 14.365 | 0.232 |
| `stage1.3` | 15.523 | 0.258 |
| `stage1.4` | 14.611 | 0.235 |
| `stage1.5` | 15.884 | 0.242 |

(numbers are the mean absolute difference between successive renders of one pose)

## Why the preview does not simply switch to PathTracing

It was tried, and it cannot be done safely from this file:

  * `environment.setup_scene()` RESETS `/rtx/rendermode`.  Setting the mode, then building the scene,
    reads back `RaytracedLighting` -- so a setting applied before the scene exists is silently
    discarded, the preview comes back noisy, and the script still believes it chose a clean mode
    (`diag_logs/capture_repeat.log`).
  * Re-applying it AFTER the scene exists segfaults the renderer (`diag_logs/verify_mode.log`:
    `Fatal Python error: Segmentation fault` in `world.step`).

So the preview AVERAGES FRAMES instead, which needs no renderer change.  The noise is zero-mean, so
the mean converges as 1/sqrt(n), and that was measured rather than assumed -- median 16x16 block
standard deviation of a `stage1.4` eye frame (`diag_logs/avg_n.log`):

```
 1 frame 14.29   2 frames 10.81   4 frames 8.57   8 frames 6.67
16 frames 5.30  32 frames 4.10  64 frames 3.19
```

`--frames` defaults to 32 (about 1.5 s per camera at 1024x1024); `--frames 1` reproduces the old
single-frame preview.  The eye frame also used to be read with NO render step at all, which is the
noisiest possible read; it now settles and averages.

## The previews, and what changed in them

Five scenes rendered at `--level 0 --slots 1` into `scene_preview/`, code stamp `git 55adda2`.  The
same measurement on the delivered images:

| image | 16x16 block std (median) | before |
| :--- | ---: | ---: |
| `stage1.4_slot1_eye.png` | **4.11** | 13.86 |
| `stage1.2_slot1_eye.png` | **4.03** | -- |
| `stage1.5_slot1_eye.png` | **4.18** | -- |

and the flat-block floor (the 1st percentile) went from 9.76 to 2.32, so the walls are now close to
uniform.

**One new thing the legible images show, recorded and NOT explained.**  The park's dressing renders
RED in `stage1.4_slot1_eye.png` (the railing, the litter bin, the curved benches), while `scenes.py`
declares DRESSING_METAL `(0.55, 0.57, 0.6)`, DRESSING_GREEN `(0.22, 0.40, 0.18)` and DRESSING_WOOD
`(0.35, 0.24, 0.16)`.  A stage-side probe confirms the declared colour IS authored on each item's
clearance box and that several props (`picnic_table`, `bench_iron`, `bin_square`, `railing`,
`bench_left/right`, `litter_bin`) carry **no** `displayColor` and no bound material anywhere under
`/prop` (`diag_logs/colour_stage1.4.log`) -- so nothing in the catalogue decides what those props are
drawn in.  The two attempts to measure the drawn colour directly (isolate a prop, difference it
against an empty room) failed, one on a masking mistake and one on a renderer crash, so the colour a
material-less prop falls back to is **unverified**.  What IS verified is that the declared dressing
colours reach only the clearance box, which is invisible whenever a prop resolves.

## The external camera does not show the dressing

`stage1.*_slot1_iso.png` is a wash: the external camera stands at (-2.0, 2.2, 1.2) looking down 18 m
of corridor, so the near side wall fills the frame and the room is a small pale rectangle in the
middle.  The site's own comment says the eye camera cannot see the dressing either -- it looks
through a 0.570 m opening at items required to sit at |z| >= 0.9 -- and that is why `--iso_orbit`
exists.  Four alternative poses were measured (`diag_logs/iso2.log`, `iso3.log`); none framed the
dressing usefully, and the search was stopped rather than continued, so the delivered external views
are the default ones and are known to be uninformative.  The eye views are the ones to read.
