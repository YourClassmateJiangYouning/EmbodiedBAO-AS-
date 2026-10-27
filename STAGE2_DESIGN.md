# Stage 2 — single success memory → wide A/S 2.0

The old repeated-attempt / self-note experiment (`STAGE23_DESIGN.md`,
`memory_experiment.py`, `memory_protocol.py`, its scene variants and its 100-pair
plan) is archived under `_archive/stage2_legacy_20261025/` and is **not used**.

## Design

Source models: the **top 7 by Stage 1 Sfinal** (ties by NPS):

1. gemini-3-flash-preview
2. deepseek-v4-flash-vision-exp
3. mimo-v2.6-pro
4. claude-haiku-4-5-20251001
5. gemini-2.5-flash
6. gpt-5
7. Qwen/Qwen3-VL-30B-A3B-Instruct

For each model, select **15 narrow successful episodes** (Levels 11-16,
A/S 0.9-0.4), 3 per scene where available.  That is **7 × 15 = 105** source
memories and 105 paired tests, not 15 memories per model in one bank.

Each test keeps the **same model, same scene and same marker slot**, and changes
only the opening to **Level 0 / A/S 2.0**.  The model sees exactly one source
memory for its test; it is never told that the source was narrow, that the test is
wide, or what the A/S values are.  Stage 1 already omits Level/A-S from the prompt;
the test prompt must keep doing so (`protocol.build_prompt` has no Level argument).

## What the model sees

- `stage2_memory_cards_top7_15_blind.jsonl`:
  - scene, marker slot
  - ordered action sequence
  - per-step action, feedback, reasoning, pose, collision, scene_description
  - no A/S, level, width, required rotation, narrow/wide or model name
  - the wrapper may say this is a successful traversal by the same model in the
    same scene and marker; that is now intentional
- `stage2_memory_cards_top7_15.jsonl` is the **internal full record** for audit.

The source reasoning is **unchanged**.  Stage 1 never told the model A/S, so the
source reasoning contains no A/S value.  Words such as "narrow", "opening" or
"width" are the model's own visual observations and are retained.

## Prompts

The tested model gets the normal Stage 1 role/task/action menu/state, plus one
memory block:

```text
A previous successful traversal by the same model in this same scene and marker:
<blind memory: Scene / Actions / Steps>
```

Then a **Step 0 no-action digest call** (before the first real action; no
environment step):

```text
Before taking any action, read the prior successful experience and the current
view. Do not choose an action yet. Return:
{"digested": true, "memory_summary": "...", "plan": "...", "cues": ["..."]}
```

No environment step is consumed by the digest.  The digest is then carried into
the first real test prompt as a short reflection block.  The formal actions are
numbered **1..30**, not 0..29.  The test uses:

```python
from stage2_protocol import build_digest_prompt, build_test_prompt
digest_prompt = build_digest_prompt(state, blind_memory_text, max_steps=30)
# model returns JSON; parse_digest() stores it
test_prompt = build_test_prompt(state, [], blind_memory_text,
                                digest=digest_text, max_steps=30)
```

The task strings, action menu, state block, image size (1024) and response format
are the same as Stage 1.  The test prompt must not add "wide", "A/S 2.0",
"narrow", "test" or the source Level.

## Observations

The digest and the test use the same 1024 camera frame and the same state fields
as Stage 1.  No width/A-S value is added to either prompt.

## Measures

Per test:

- `passed`, `total_steps`
- `turn_count`, `total_rotation`, `max_rotation_deg`
- `look_down_count`
- `action_copy_rate` / LCS against the source action sequence
- `turn_match` against the source turn count

Compare each run with:

1. the same model's no-memory Level 0 baseline;
2. the same model's Stage 1 Level 0 behaviour.

## Files

- `stage2_memory_cards_top7_15_blind.jsonl` — prompt-visible memory cards
- `stage2_memory_cards_top7_15.jsonl` — full internal records
- `stage2_top7_15_test_plan.csv` — 105 same-scene/same-marker wide tests
- `stage2_memory_schema.md` — field definitions
- `stage2_protocol.py` — prompt builder for the digest and test steps
- `reports/stage2_sample_prompt_S2M-029.md` — a rendered DeepSeek example

## Runner

The old `memory_experiment.py` is archived.  The new runner is
`stage2_runner.py`: it reads the test plan, loads the blind memory card, calls
`stage2_protocol.build_digest_prompt` once (no environment step), then runs the
ordinary Stage 1 step loop with `build_test_prompt`.  It writes a separate
`stage2-<scene>-<memory_id>` tag under `results/` and `logs/`; it does not touch
the Stage 1 checkpoint or a running sweep.

Run one scene per invocation:

```bash
export BOYUE_API_KEY=...
/home/ybh/isaacsim/python.sh stage2_runner.py \
    --scene stage1.5 \
    --plan reports/stage2_top7_15_test_plan.csv \
    --memory reports/stage2_memory_cards_top7_15_blind.jsonl \
    --headless
```

Do not launch it while a large Stage 1 sweep is using the same gateway; the
runner is code-ready but has not been smoke-tested in Isaac yet.
