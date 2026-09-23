# EmbodiedBAO — Body-as-Obstacle Benchmark

Does an MLLM-driven humanoid rotate its body **before** a passage becomes too
narrow to walk through, the way humans do?

Warren & Whang (1987) showed that people start turning sideways once a gap is
about **1.30×** their shoulder width — they anticipate the constraint instead of
waiting to be blocked. This benchmark puts a Unitree H1 in front of a wall with
a single vertical opening and sweeps that ratio from 1.58 down to 0.79,
measuring where the agent's body-scale affordance perception sits.

**The only thing that changes between Levels is the physical opening width. The
prompt is identical for every Level, and it never mentions the width, the body
size, the A/S ratio, or the need to turn.**

## The A/S ladder

A/S = channel width / shoulder width (0.57 m).

| Level | Channel | A/S | Frontal passage | Design intent |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.90 m | 1.58 | easy | baseline |
| 1 | 0.80 m | 1.40 | works | near the human threshold |
| 2 | 0.74 m | 1.30 | tight | humans turn here |
| 3 | 0.68 m | 1.19 | grazing | below the human threshold |
| 4 | 0.57 m | 1.00 | impossible | must turn |
| 5 | 0.45 m | 0.79 | impossible | turn required |

Levels 0–3 can be walked through facing forward. Levels 4–5 cannot: the shoulders
are wider than the gap, so the agent must rotate to present its 0.22 m torso
thickness — and it must do so **before** reaching the wall, because the turn gate
samples the robot's current pose and rejects any rotation that would sweep the
shoulders through a panel.

## Scene

```
x : forward   (5 m room; wall at x = 3.0, robot starts at x = 0.5)
y : up        (ground at y = 0)
z : lateral   (opening centred at z = 0, room spans z in [-2.5, 2.5])
```

| Element | Value |
| :--- | :--- |
| Room | 5.0 × 5.0 m, **fully enclosed**: four walls 3.0 m tall plus a ceiling |
| Obstacle wall | plane `x = 3.0`, 2.0 m tall, 0.02 m thick, spanning the room |
| Opening | vertical, centred at `z = 0`, floor to the top of the wall |
| Channel edge posts | 0.05 m wide, full wall height, one on each side, placed **outside** the opening |
| Robot start | `(0.5, 0, 0)`, facing `+x` |
| Success | body centre reaches **`x > 3.5`** |
| Eye camera | head height 1.68 m, pitched 15° down, 76° field of view |

### Surface colours

Every surface the agent can see has its own colour, so that "aimed at the
opening" and "aimed at a panel" cannot be confused:

| Surface | Colour |
| :--- | :--- |
| Obstacle wall | opaque blue, `[0.13, 0.28, 0.72]`, opacity 1.0 |
| Far wall (seen *through* the opening) | green, `[0.13, 0.42, 0.20]` |
| Side and near walls | light grey, `[0.72, 0.73, 0.75]` |
| Ceiling | white, `[0.92, 0.93, 0.95]` |
| Floor | grey with a 0.5 m reference grid |
| Channel edge posts | near-black, `[0.10, 0.11, 0.13]` |

The obstacle wall is deliberately **opaque**. A translucent panel made the
channel hard to read at close range; a saturated opaque surface renders the
opening as a clean silhouette.

## Action space

Eight discrete actions:

| Action | Effect |
| :--- | :--- |
| `forward` / `backward` | move **0.20 m** along the torso's facing direction |
| `left` / `right` | move **0.20 m** along the torso's own left / right |
| `turn_left` / `turn_right` | rotate the torso (and its head camera) **15°** |
| `look_left` / `look_right` | rotate the head camera **30°**, body unchanged |

Movement is **egocentric**: after turning sideways, `forward` walks along the new
facing direction. Reaching the goal therefore requires composing rotation and
translation, which is what the benchmark is measuring.

The translation step is 0.20 m because the walk is 3.0 m: 15 moves plus a 90°
turn (6 moves) fits the 30-step budget with 9 spare.

## Protocol

* 6 Levels × **10 episodes** = **60 scored episodes** per model.
* Each episode: at most **30 steps**. Episodes end on success or exhaustion.
* Wall collisions are **recorded but never terminate** an episode.
* Every episode starts fresh from the same pose; Levels are independent.
* Every Level uses the same prompt — see `protocol.py`.

### What the model is told each step

The prompt contains the task, the eight actions with their real distances, a note
that movement is egocentric, the robot's own position and torso rotation, the step
limit, and **the full action history for the episode so far**. Each history entry
is `step N: <action> -> <feedback> | your reasoning: <the agent's own reasoning>`,
listed oldest first, and the block explicitly invites the agent to use it to
notice what it has already tried and whether it worked.

The prompt never contains the channel width, the body dimensions, the A/S ratio,
or any hint that a turn may be needed. `build_prompt()` takes no `level`
parameter, so leaking the geometry structurally is not possible.

The history is the agent's own within-episode memory, not a sliding window, and
it is not truncated. The action distances are derived from `MOVE_STEP` rather
than hard-coded: an earlier version advertised "move forward 5cm" while
`MOVE_STEP` was 0.20 m, which would have made every distance judgement the agent
made wrong by a factor of four.

## Recorded data

Per episode (`results/level{level}/{model}/episode_{id:03d}.json`):

```
episode_id, level, channel_width, a_s_ratio,
passed, passed_sideways, total_rotation, first_turn_step,
total_steps, action_sequence
```

Per step (`episode_{id:03d}_steps.json`, and the flat CSV):

```
step, action, torso_rotation, position_x, position_z, collision, step_success
```

## Metrics

| Metric | Definition |
| :--- | :--- |
| **Sideways rate** | fraction of episodes that passed while the torso was rotated 45–135° |
| **Sideways threshold** | widest A/S at which a sideways passage was observed — compare against the human **1.30** |
| **First turn step** | step index of the first rotation action |
| **Mean pass steps** | average steps needed for a successful passage |

The threshold classifies a model:

| Class | Threshold | Reading |
| :--- | :--- | :--- |
| `anticipatory` | ≥ 1.30 | turns before the gap is tight, like a human |
| `borderline` | 1.00 – 1.30 | turns while still passable |
| `reactive` | < 1.00 | only turns after being blocked |
| `no_sideways` | — | never passed sideways |

## Files

| File | Role |
| :--- | :--- |
| `environment.py` | Isaac Sim scene, kinematics, analytic collision gate, success test |
| `protocol.py` | single source of truth for the action space and the unified prompt |
| `experiments.py` | episode loop, Level ladder, record shaping, checkpoints |
| `ai_agent.py` | unified OpenAI-compatible MLLM client + `random` baseline |
| `main.py` | CLI entry point, CSV export, threshold report |
| `analysis.py` | per-Level metrics, A/S threshold, reports and plots |
| `capture_views.py` | render the scene from fixed viewpoints (diagnostics) |
| `test_bao_geometry.py` | offline geometry/protocol verification (no Isaac Sim needed) |
| `test_bao_integration.py` | end-to-end runner + analysis test against a mock environment |
| `run_diagnostics.sh` | one-shot environment check on a new machine |
| `tools/` | measurement probes: field of view, mesh points, ray casting, box sizes |

## Running

Requires Isaac Sim. Use its bundled Python:

```bash
$ISAACSIM_ROOT/python.sh main.py --model gemini-2.5-pro --all-levels --image_size 512
# Windows
%ISAACSIM_ROOT%\python.bat main.py --model gemini-2.5-pro --all-levels --image_size 512
```

Common invocations:

```bash
# a single Level, two episodes
python main.py --model gemini-2.5-pro --level 0 --episodes 2

# offline smoke test, no API key required
python main.py --model random --level 0 --episodes 2

# resume a run whose tag is already recorded
python main.py --model gemini-2.5-pro --all-levels --image_size 512 --tag gemini-v1 --resume
```

### Useful flags

| Flag | Purpose |
| :--- | :--- |
| `--image_size N` | downscale the camera frame to N×N before sending it. Latency was ~32 s per step at 1024 and ~6 s at 512. |
| `--llm_timeout S` | per-request timeout (default 90 s; 60 s lost 9 of 30 steps to timeouts) |
| `--eye_height M` / `--eye_pitch D` | move the head camera without editing code |
| `--start_x M` / `--move_step M` | start distance and translation step; the two must move together or the route will not fit the budget |
| `--save_obs` | save the camera frame for every step under `logs/{tag}/obs/` |
| `--tag NAME` | stable name for logs, results and the checkpoint. Recommended for any run you may need to resume. |

`--resume` skips episodes already recorded in
`results/{model}/checkpoint_{tag}.json`. Changing the prompt, the action
definitions or the scene invalidates old results, so use a fresh tag after any
such change.

Then analyse:

```bash
python analysis.py --results_root results
```

Outputs `analysis/threshold_table.{md,csv}`, one JSON report per model, and a
`thresholds.png` plot with the human 1.30 reference line.

## Verification without Isaac Sim

Everything that can be decided without a renderer runs on plain Python:

```bash
python test_bao_geometry.py     # 30 checks: ladder, collision gate, routes, colours, prompt
python test_bao_integration.py  # 11 checks: full protocol against a mock environment
```

The geometry suite re-derives the collision model independently and pins the
properties that make the ladder meaningful:

* the declared frontal/sideways feasibility of every Level,
* the body footprint stays centred on the root pose under rotation,
* the sideways route reaches past the success plane at every Level,
* a frontal walk succeeds at Levels 0–3 and is blocked at 4–5,
* turns from a wall-fouling pose are always rejected,
* the clearance boundary for a full 0→90° turn sits in front of the wall,
* the robot has enough free run from its start pose to complete that turn,
* the channel edge posts do not narrow the opening,
* box axes map so that a 2.0 m tall post gets a 2.0 m vertical extent,
* every pair of visible surfaces differs enough to be told apart,
* the prompt states the same step length that the agent actually moves,
* the prompt is identical, deterministic, and leak-free.

## Diagnostics on a new machine

```bash
bash run_diagnostics.sh 2>&1 | tee diagnose.log
```

Runs the offline suites, probes the Isaac Sim import, captures the scene from
several viewpoints, and writes everything to `diagnose.log`. Each probe under
`tools/` answers one question with a measurement rather than an inference:

| Probe | Answers |
| :--- | :--- |
| `tools/measure_fov.py` | the eye camera's true field of view, three independent ways |
| `tools/project_probe.py` | which pixel a known world point should occupy |
| `tools/ray_probe.py` | what each part of the frame should hit, and frame occupancy per Level |
| `tools/points_probe.py` | the authored mesh points, per axis, for every box |
| `tools/focal_probe.py` | where the camera focal length is set and what it resolves to |
| `tools/bar_probe.py` | which world height a dark image row corresponds to |
| `tools/check_names.py` | unbound names across the package, run as part of the geometry suite |

`tools/check_names.py` is a small AST scope-chain checker; it runs inside
`test_bao_geometry.py`, which is why a name error fails the suite rather than
waiting for Isaac Sim.

`capture_views.py` renders the scene from fixed viewpoints — a floor plan, an
elevated three-quarter view, the wall face-on, behind the wall, and the robot's
own eye before and after approaching:

```bash
python capture_views.py --level 0 --outdir views
```

## Environment variables

| Variable | Purpose |
| :--- | :--- |
| `BOYUE_API_KEY` / `TAOTOKEN_API_KEY` / `OPENAI_API_KEY` | API key (first match wins) |
| `BOYUE_BASE_URL` / `TAOTOKEN_BASE_URL` / `OPENAI_BASE_URL` | endpoint override |
| `BAO_IMAGE_SIZE` | same as `--image_size` |
| `BAO_LLM_TIMEOUT`, `BAO_MAX_RETRIES` | request timeout / retry count |
| `BAO_DISABLE_PROXY=1` | clear proxy variables before calling the API |
| `EMBODIEDBAO_H1_USD` | explicit path to the H1 USD asset |

## Notes on the physics model

The robot is kinematic: actions teleport the root pose and every candidate pose is
gated by an analytic collision test. The torso footprint is an oriented rectangle
(0.22 m × 0.57 m) tested against the two wall panels with an exact
separating-axis test, inflated by a 2 mm skin so that A/S = 1.00 is a real pinch
point rather than a zero-clearance squeeze. Rotation checks sweep the whole 15°
arc, and a pose that already fouls the wall cannot rotate free — so the agent
cannot teleport through the wall one 15° hop at a time.

Every box in the scene is an explicit `UsdGeom.Mesh` whose eight corners are
authored directly in metres, so local geometry, world size and reported bounds
are the same number by construction. Earlier versions used `FixedCuboid` and then
a `UsdGeom.Cube` with an extent plus a scale op; in both cases the reported size
and the rendered size disagreed, which is how the channel posts once rendered as
a 2 m horizontal bar instead of a vertical post.

## Known issues

* **Model behaviour on Level 0.** Both `qwen-vl-max` and `gemini-2.5-pro` walked
  cleanly to `x = 2.30` and then stopped advancing, spending the remaining steps
  alternating turns while reporting no collisions. Both episodes of a two-episode
  run stopped at the same `x` and the same final yaw, so it is reproducible.
  Whether that is the phenomenon under study or an artefact of the observation is
  unresolved; the frames saved with `--save_obs` are what will settle it.
* **API latency.** Measured per-step latency at the same settings has varied by
  more than 5× between runs: `qwen-vl-max` at 512 px averaged 5.7–8.0 s, while
  `gemini-2.5-pro` at the same 512 px averaged a median of 31.9 s over 60 steps
  (max 82.1 s) — yet a text-only call to that same model returned in 6.0 s, and
  an earlier `gemini-2.5-pro` run at 512 px completed 50 steps in about 10
  minutes. The cost is therefore dominated by the model and by API load rather
  than by the image, and **this has not been isolated by a controlled A/B test**:
  doing so (same model and prompt at 512 vs 1024) is the measurement that would
  settle whether the resolution can be raised without cost.

## Out of scope (future work)

The current study is the A/S threshold only. Follow-ups once the threshold is
known: soft-material edges (does the agent try to squeeze through?), strategy
persistence (does a wide channel still trigger a turn after priming?), and
insight-versus-gradual learning curves across repeated episodes.
