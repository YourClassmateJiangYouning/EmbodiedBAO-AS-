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
| 2 | `git ls-files assets \| wc -l` | `1185` | the asset root is complete | required value is 1185 (`REMOTE_WORKORDER.md` §0) |
| 3 | `git ls-files assets/isaac \| wc -l` / `assets/bucket` | `1090` / `95` | both vendored roots are tracked | 1090 + 95 = 1185 |
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
