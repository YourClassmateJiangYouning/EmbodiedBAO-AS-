# AUDIT_DECISIONS.md — findings that were deliberately NOT changed, and why

This file exists so the **next** review does not re-discover the same items and either (a) re-report
them as fresh defects, or (b) "fix" them in a way the owner already rejected.

Every entry below was examined, was reported by an automated reviewer, and is **either not a defect,
or a defect whose fix was declined**. The reason is recorded with the evidence that settled it.

If you are reviewing this repository: **check this file before reporting a finding.** If the item is
here and you disagree, say which evidence is wrong — do not silently re-report it.

Reviewed 2026-10-17 by the agent that made the A–G fixes. Sources: the 90-item review in the previous
conversation, plus a second pass in which the agent **measured** rather than reasoned.

---

## 1. Not defects — the review's claim was measured and is false

### C1 · "the marker shapes are not the same size (3.7× area difference) vs README's '0.90 m'"

**Verdict: the sizes ARE comparable; nothing changed.** The owner looked at the renders and said the
markers look about the same size, and that was correct.

The **maximum linear dimension** of all ten shapes is **0.779–0.900 m** (`square` 0.900, `disc` 0.900,
`hexagon` 0.779 × 0.900, `triangle` 0.779 × 0.675, `bars2` 0.900 × 0.421, `arrow` 0.900 × 0.540). The
README's "0.90 m bounding size" is therefore true as stated.

The width/height figures that make it look like a factor of 3.7 are **areas**, and they differ because
the shapes are fatter or thinner at the same extent — a 0.9 m bar is 0.209 m thick while a 0.9 m square
is 0.9 m thick. Area is the wrong measure for "does the distance cue look the same size": apparent size
tracks linear extent.

**Two measurement errors were made here, and both are worth remembering:**

1. The first measurement used only `SHAPES[name][0]` — the **first sub-polygon**. `arrow` has two parts
   (shaft + head) so it measured as 0.418 × 0.209 and the agent concluded "only arrow is undersized".
   `bars2`/`bars3` also have 2–3 parts. **A shape is the union of all its parts.**
2. On the strength of that, the agent told the owner the review's numbers were wrong. **They were
   right** (0.421, 0.540, 0.479, 3.72× all reproduce exactly once all parts are measured). The agent's
   correction was itself the error.

### C4b · "the walking-band check silently skips 15 wide floor items"

**Verdict: not silent; nothing changed.** `test_bao_scenes.py` collects the skipped items in
`unplaceable` and prints all fifteen by name, with the lateral width, the band rule and the remedy:

```
[layout] 15 floor item(s) have a declared box too wide for this 5 m room to keep out of the |z| 1.5
         walking band at ANY anchor, so their z is unchecked and the fix is a rotation in
         place_dressing or a smaller declaration: ["stage1.2/rack (lateral 1.3247 m needs …']
```

The items genuinely cannot satisfy both the band and the side wall (width > 1.0 m in a 5 m room at a
1.5 m band), so they are recorded rather than failed. The suite also prints the nineteen known overlaps
the same way.

### G6 · "`scenes._dressing_wall`'s docstring contradicts its implementation"

**CORRECTED 2026-10-17: this entry was WRONG, and so was my verdict.** The docstring says the surface
"is used AS the box centre" and that "the thickness is not subtracted here on purpose", while
`scenes.py:711` computes `centre_x = at[0] - MOUNT_CLEARANCE_M - x_thick / 2.0` -- it DOES subtract the
half-thickness. The inline comment immediately above that line even says the old claim was false. The
code was changed after I read it and the top docstring was not updated, so my "self-consistent" verdict
was true only of the version I looked at. The reviewer was right.

### #12 · "the prompt could change with Level and no test would notice"

**Verdict: the premise is wrong.** `protocol.build_prompt` takes `state, history, max_steps,
move_step, memory_block, attempt_wording` — **it has no `level` parameter at all**, so a prompt cannot
vary by Level. The README's "no `level` argument" is literally true. The weaker remaining claim (that
no test compares two Levels' prompts) is a coverage gap, not a defect.

### #29 · "`_add_box`'s six faces all wind inward and it has no doubleSided"

**Verdict: false.** `_add_box` **does** set `doubleSided` (one occurrence, same as `_prism`).

### #32 · "`hide_wall` is undone by `set_channel_width`"

**Verdict: false.** `set_channel_width` calls `_remove_wall()` and `_create_wall()`, not
`_create_wall` from the `hide_wall` path; `hide_wall` appears exactly once in the file, in `__init__`.

### #63 · "no test instantiates a real `BAOEnv`"

**CORRECTED 2026-10-17: the finding was RIGHT and my "false" was wrong.** I counted with
`grep -c 'BAOEnv('`, which also matches `FakeBAOEnv(`. Measured properly: `test_bao_integration.py` has
21 `BAOEnv(` occurrences and **all 21 are `FakeBAOEnv(`** -- zero real ones. `check_success`,
`reset_scene`'s marker rule and `execute_action` were only ever tested against the stand-in.

There is now `test_a_real_baoenv_scores_the_way_the_fake_does`, which builds a real `BAOEnv` under the
Isaac interpreter and SKIPS (rather than passes) when the simulator is absent.

### #82 · "`verify_docs.py` hard-codes `ISAAC_FILES = 1090`"

**Verdict: confirmed as a fragility issue and FIXED, but the value was correct.** The measured count
was 1090, so the check passed; the problem was that adding one asset would turn a *documentation*
check red. `ISAAC_FILES` is now measured with `git ls-files`, and the three
`total == 1192` / `isaac == ISAAC_FILES` / `bucket == 102` assertions became `> 0`.

---

## 2. Real findings whose fix was DECLINED by the owner

These are known defects. They are **not** bugs of omission; the owner chose the behaviour.

### B3 · five markers treated as five independent Bernoulli trials

**Declined.** The owner's position: *"保持现状，就是一共25个标志物，相当于同样的标志物不重复"* — the 25
markers are 25 CONDITIONS and are deliberately treated as non-repeating. The binomial test stays.

Consequence to be aware of when writing up: `_binomial_tail(k, n, p)` assumes independent draws, and
the five episodes of a Level are five different marker conditions, so the p-value is optimistic. If a
reviewer wants this changed, the change is a methodological decision for the owner, **not a bug fix** —
do not "fix" it silently.

### B5 · an all-invalid Level crashes `plot_thresholds` with `KeyError`

**Declined.** The owner: *"全invaild我会重新测试，不会将invaild搞到绘图里面，不会如此愚蠢"* — a Level
with every call failed will be re-run, so such a Level will not reach the plot.

The latent crash is real: `analysis.summarize_level` returns only
`{'level', 'episodes', 'excluded_no_valid_call'}` for such a Level, and `plot_thresholds` indexes
`a_s_ratio` / `sideways_rate` unconditionally. **If the owner's re-run policy ever lapses, this is the
symptom.** It is recorded, not fixed.

### B6 · `experiments.summarize_level` and `analysis.summarize_level` disagree

**Declined, same reason as B5.** Measured on the same input: the first reports `pass_rate = 0.5,
episodes = 2`, the second `pass_rate = 1.0, episodes = 1` (it drops all-invalid episodes). So the live
threshold table in `main.py` and the offline report can print different numbers for the same run.

The owner will re-run any Level containing invalid episodes, so the two should agree in practice.
**They are two implementations of one rule, and that remains a maintenance hazard** — if it is ever
"fixed", decide which definition wins before touching either.

---

## 3. Reviewed items that are real but were already correct

Recorded so they are not re-reported.

| # | Claim | State |
|---|---|---|
| F3 | README action table swapped left/right | **already corrected** to `right (-z) / left (+z)` |
| F6 | `verify_docs` asserts the check total is 176 | **already corrected** to compare against the measured total |
| C3 | the leak test inspects a prompt the experiment never sends | **fixed**; adding `camera_yaw` immediately exposed a real `sideways` leak, now removed from the prompt |

---

## 4. Items still unverified

Listed honestly. Neither confirmed nor denied. If you have the environment to check them, do so before
reporting them as defects.

`#20` (binomial implementation details), `#22` (see B6 — confirmed but declined), `#25`,
`#37` (build_plate inset), `#39` (bound count double-counting), `#41` (scene_parts does not gate the
marker), `#43` (resolved — see E5), `#46`, `#47`, `#48` (ai_agent cache/JSON-mode/repair-hint),
`#60` (light test literals — **resolved**: constants extracted, see the light test),
`#68` (assets test strength), `#85` (repository size).

---

## 5. Process notes for the next reviewer

1. **Measure the union, not the first element.** Two of this round's three worst errors came from
   inspecting `SHAPES[name][0]` and treating it as the shape.
2. **A negative claim ("file X has no Y") needs a positive search.** Three of the review's findings
   were wrong this way (#29, #32) -- and #63 was the reverse: my own check used a substring match that
   counted `FakeBAOEnv(` as `BAOEnv(`, so I "refuted" a finding that was correct.
3. **Separate "the value is wrong" from "the value is hard-coded".** #82 was the second, not the first,
   and the fix differs.
4. **Check whether a "dead" function is a documented future interface.** `analytic_pass_check` has zero
   callers and is named by two Stage 2 design documents; deleting it was wrong and it was restored.
5. **An intermittent CUDA failure is not a statement about the machine.** Two renders died with
   `no CUDA-capable device`, and the GPUs were fine minutes before and after. Retry before concluding.
