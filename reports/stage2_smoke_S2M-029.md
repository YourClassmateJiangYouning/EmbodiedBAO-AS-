# Stage 2 smoke — S2M-029 (DeepSeek)

## Pair

- model: deepseek-v4-flash-vision-exp
- scene: stage1.5
- marker: 1
- memory: S2M-029
- source: level 12 / A/S 0.8 / width 0.456 m
- test: level 0 / A/S 2.0 / width 1.14 m
- run tag: `stage2-stage1.5-S2M-029`
- result: `results/level0/deepseek-v4-flash-vision-exp/stage2-stage1.5-S2M-029/episode_000.json`

## Step 0 digest — PASS

The model returned a valid no-action digest:

```json
{
  "digested": true,
  "memory_summary": "...",
  "plan": "...",
  "cues": ["..."]
}
```

No action was taken in Step 0; the environment did not move.

## Formal test — PASS, but copied the memory

```text
passed              true
total_steps         17
invalid_response    0
timeouts            0
max_rotation        90.0 degrees
total_rotation      90.0 degrees
```

Test actions:

```text
forward, forward, forward,
turn_left x6,
forward x8
```

Source actions:

```text
forward, forward, forward,
turn_left x6,
forward x8
```

**Action sequence is exactly the source memory: copy rate 100%.**

## Same-scene / same-marker Stage 1 wide baseline

The ordinary Stage 1 Episode 000 in this scene/marker, with no memory:

```text
passed        true
steps         11
max_rotation  0.0
turns         0
actions       forward x11
```

So the model at **wide A/S 2.0**:

- no memory: 0 turns, 11 steps;
- with the narrow-success memory: 6 turns, 90°, 17 steps.

That is the intended Stage 2 signal.

## Caveat

This is **one smoke pair**, not the experiment result. It shows:

1. the pipeline works;
2. Step 0 digest works;
3. one success memory can induce the exact narrow-turn habit in the same scene at wide A/S 2.0.

The formal result needs all 105 pairs and the no-memory/control comparisons.
