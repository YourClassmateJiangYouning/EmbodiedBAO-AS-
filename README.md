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
| 0 | 0.90 m | 1.58 | easy, 0.165 m of slack | baseline |
| 1 | 0.80 m | 1.40 | 0.115 m of slack | near the human threshold |
| 2 | 0.74 m | 1.30 | 0.085 m of slack | humans turn here |
| 3 | 0.68 m | 1.19 | 0.055 m of slack | below the human threshold |
| 4 | 0.57 m | 1.00 | exactly touching: only dead centre | the geometric limit |
| 5 | 0.45 m | 0.79 | impossible unturned | rotation required (≥75°) |

The slack column is measured, not derived: it is the largest lateral offset from
which a straight walk still reaches the goal (`tools/check_heading_frame.py`).
Level 4 is why it is quoted: shoulder and channel are equal, so an aligned body
fits but has no room for error, while a torso rotated 45° or more clears it by up
to 0.175 m. Level 5 is narrower than the shoulders at every yaw below 75°, so it
is the Level that requires the rotation this benchmark measures — and the
rotation must happen **before** reaching the wall, because the turn gate samples
the robot's current pose and rejects any rotation that would sweep the shoulders
through a panel.

## Scene

```
x : forward   (16 m room; wall at x = 8.0, robot starts at x = 0.5)
y : up        (ground at y = 0)
z : lateral   (opening centred at z = 0, room spans z in [-2.5, 2.5])
```

| Element | Value |
| :--- | :--- |
| Room | 16.0 m long (`x`) × 5.0 m wide (`z`), **fully enclosed**: floor and ceiling are 16 × 5 m; side walls extend only along `x` |
| Far wall | green wall at `x = 16.0`, spanning the 5.0 m room width |
| Obstacle wall | plane `x = 8.0`, 2.0 m tall, 0.02 m thick, spanning the room width |
| Opening | vertical, centred at `z = 0`, floor to the top of the wall |
| Channel edge posts | 0.05 m wide, full wall height, one on each side, placed **outside** the opening |
| Robot start | `(0.5, 0, 0)`, facing `+x` |
| Success | body centre reaches **`x >= 11.0`** |
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
| `forward` / `backward` | walk **0.75 m** toward / away from the far wall |
| `left` / `right` | sidestep **0.75 m** to the walker's left / right |
| `turn_left` / `turn_right` | rotate the torso **15°**; view unchanged |
| `look_left` / `look_right` | turn the head camera **30°** off the walking direction |

Movement is in the **walking frame**, not the body frame: the walking direction
always points at the far wall, and a torso rotation does not steer. `turn_*`
rotates the torso *relative to* that walking direction, which is the shoulder
rotation the human aperture literature measures (Warren & Whang 1987) — passing a
tight opening means rotating the torso and continuing to walk forward.

This is a deliberate reversal of the earlier egocentric framing, and it is not
cosmetic. With body-frame translation a 0.75 m step taken at 15° slides the body
0.19 m sideways, more than any Level's channel can absorb, so only 0° and 90°
could traverse at *any* width: rotating was all-or-nothing, and a model that
rotated correctly could still fail purely on the translation frame. Measured side
by side in `tools/check_heading_frame.py`; the shipped frame is:

| Level | A/S | torso yaws that traverse |
| :--- | :--- | :--- |
| 0–3 | 1.58–1.19 | every yaw |
| 4 | 1.00 | 0°, or 45°–90° |
| 5 | 0.79 | 75°, 90° |

So the rotation a model adopts is graded in 15° steps, which is the quantity that
can be compared with the human threshold of 1.30.

The **gaze** is pinned to the walking direction: rotating the torso changes the
body's footprint and nothing the agent can see. That matches how a person crosses
a narrow opening — eyes on the opening, shoulders rotated — and it keeps the
channel in view at the 75–90° rotations Levels 4 and 5 ask for, which a
torso-mounted camera could not (the channel subtends 76°). The torso angle is
reported to the agent as a number, which is the proprioception a person has.

The translation step is 0.75 m, approximately an adult walking step. From the
`x = 0.5` start, ten forward translations reach the obstacle plane at `x = 8.0`,
and four more reach the inclusive success plane at `x = 11.0`. Level 5 needs five
15° turns (75°) plus those fourteen steps, so the intended route fits within the
30-step budget.

## Protocol

* 6 Levels × **10 episodes** = **60 scored episodes** per model.
* Each episode: at most **30 steps**. Episodes end on success or exhaustion.
* Wall collisions are **recorded but never terminate** an episode.
* Every episode starts fresh from the same pose; Levels are independent.
* Every Level uses the same prompt — see `protocol.py`.

### What the model is told each step

The prompt contains the task, the eight actions with their real distances and their
mechanical effects, a note that the walking direction is fixed at the far wall and
that the eyes stay on it when the torso turns, the robot's own position, torso
rotation and **head-camera offset**, the step limit, and **the full action history
for the episode so far**. Each history entry is
`step N: <action> -> <feedback> | your reasoning: <the agent's own reasoning>`,
listed oldest first, and the block explicitly invites the agent to use it to
notice what it has already tried and whether it worked.

The head-camera offset is reported because the two rotation families are not the
same thing and nothing in a single frame distinguishes them. `turn_left` /
`turn_right` rotate the body, and the head camera turns with it, so the facing and
the view change together. `look_left` / `look_right` rotate only the head, leaving
an offset that **persists** and rides along when the body later turns. An agent
that had called `look_left` three times would otherwise be judging its alignment
from a view rotated 90 degrees with nothing in the prompt saying so — being tested
on guessing the interface rather than on judging its own body. The environment
already produced this value in `get_robot_state()`; the prompt simply never
rendered it.

The prompt never contains the channel width, the body dimensions, the A/S ratio,
or any hint that a turn may be needed. `build_prompt()` takes no `level`
parameter, so leaking the geometry structurally is not possible. The action
descriptions mention turning and the head camera, but only to document what the
controls do; the task statement itself carries no advice about the solution, and
the test suite checks those two properties separately.

The history is the agent's own within-episode memory, not a sliding window, and
it is not truncated. The action distances are derived from `MOVE_STEP` rather
than hard-coded, so the prompt always advertises the configured 0.75 m movement
instead of drifting from the environment.

## Recorded data

Per episode (`results/level{level}/{model}/{tag}/episode_{id:03d}.json`):

```
episode_id, level, channel_width, a_s_ratio,
passed, passed_sideways, total_rotation, first_turn_step,
total_steps, action_sequence
```

Per step (`episode_{id:03d}_steps.json`, and the flat CSV):

```
step, action, torso_rotation, position_x, position_z, collision, step_success
```

Episodes, sidecars, summaries, run settings and the threshold reports are all
written **atomically** (temporary file + `os.replace`), so a crash, an OOM kill
or Ctrl-C cannot leave a half-written record behind for the next `--resume` to
choke on. The checkpoint is written *after* its episode record, so an
interruption costs a re-run rather than the episode; and if a record or the
checkpoint is unreadable, the damaged file is renamed to
`*.corrupt-<timestamp>` and the episode is re-run instead of being silently
dropped from the sample.

Paths are anchored to the repository rather than to the working directory, so
`python /path/to/main.py` cannot scatter `results/` into whatever directory the
shell happened to be in; `BAO_OUTPUT_ROOT` relocates every artefact at once.
A run also writes `logs/{tag}/args.json` (the effective settings, including the
resolved tag and output paths) and `run_progress.txt` (a greppable timeline),
and the flat CSV is refreshed after **every** episode, so an interrupted Level
is still exported.

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
| `persistence.py` | atomic writes, tag sanitising, corrupt-file quarantine: the durability layer every writer goes through |
| `test_bao_geometry.py` | offline geometry/protocol verification (no Isaac Sim needed) |
| `test_bao_integration.py` | end-to-end runner + analysis test against a mock environment |
| `test_bao_persistence.py` | offline checks that an interrupted run cannot lose or corrupt collected data |
| `models.json` | the model roster: which models are tested, and why the others were excluded |
| `run_all_models.sh` | sweep the roster sequentially, resuming each model by tag |
| `verify_professor_machine.sh` | one-shot machine check: suites, passability, Isaac probe, scripted Level-5 run, rendered views |
| `download.py` | fetch the H1 USD asset if `assets/` is empty |
| `tools/` | measurement probes used while building the scene; `check_names.py` also runs inside the geometry suite |

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

# smoke test (no API key required, but Isaac Sim is still required)
python main.py --model random --level 0 --episodes 2

# resume a run whose tag is already recorded
python main.py --model gemini-2.5-pro --all-levels --image_size 512 --tag gemini-v1 --resume
```

### The whole roster (11 models)

```bash
export BOYUE_API_KEY='...'
ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models.sh 2>&1 | tee sweep.log
```

`models.json` is the single definition of the roster: the script reads it, gives
each model its own tag, and runs 6 Levels × 10 episodes × 30 steps per model
with `--resume`, so a second invocation continues instead of starting over. It
refuses to start if the interpreter cannot import `isaacsim` — rather than
failing 11 times and looking like a finished sweep — and it exits non-zero if
any model did.

Budget **days, not hours**: up to 1800 model calls per model. Run it inside
`tmux` or `nohup`, because a dropped SSH session would kill the sweep. Every
episode is written as it is scored, so an interruption costs at most the episode
in flight. Afterwards:

```bash
python analysis.py                     # every model found under results/
python analysis.py --tag gemini-v1     # one exact run
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
# analyze one exact run without mixing tags
python analysis.py --results_root results --tag gemini-v1
```

Outputs `analysis/threshold_table.{md,csv}`, one JSON report per model, and a
`thresholds.png` plot with the human 1.30 reference line.

## Verification without Isaac Sim

Everything that can be decided without a renderer runs on plain Python:

```bash
python test_bao_geometry.py     # 32 checks: ladder, collision gate, routes, colours, prompt
python test_bao_integration.py  # 16 checks: full protocol, tagged runs, CLI flags, against a mock environment
python test_bao_persistence.py  # 13 checks: interrupt and corruption safety of every artefact written
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
ISAAC_PY=/home/ybh/isaacsim/python.sh bash verify_professor_machine.sh 2>&1 | tee professor_verify.log
```

Runs the offline suites (geometry, integration, persistence), the passability
diagnostic, an Isaac Sim import probe, a scripted Level-5 traversal and the
rendered Level-0/Level-5 views, and reports each section as OK or FAILED without
aborting early. `PROFESSOR_TEST_CONTEXT.md` lists the scene constants it checks
and the visual acceptance criteria.

Each probe under `tools/` answers one question with a measurement rather than an
inference, and each exists because a guess had already been wrong once:

| Probe | Answers |
| :--- | :--- |
| `tools/measure_fov.py` | the eye camera's true field of view, three independent ways |
| `tools/project_probe.py` | which pixel a known world point should occupy |
| `tools/ray_probe.py` | what each part of the frame should hit, and frame occupancy per Level |
| `tools/points_probe.py` | the authored mesh points, per axis, for every box |
| `tools/focal_probe.py` | where the camera focal length is set and what it resolves to |
| `tools/bar_probe.py` | which world height a dark image row corresponds to |
| `tools/image_format_probe.py` | which image payload formats the gateway accepts |
| `tools/verify_models.py` | which models can serve a text+image request at all |
| `tools/check_credit.py` | whether a gateway failure is the key, the account balance, or one model |
| `tools/passability_probe.py` | whether each Level is reachable; single-pose legality, so prefer the next one |
| `tools/reachability_search.py` | the same question by breadth-first search over the real actions |
| `tools/memory_test.py` | whether a model carries state across API calls (standalone, no Isaac Sim) |
| `tools/dry_run_save.py` | what the runner actually writes to disk, field by field, without Isaac Sim |
| `tools/check_names.py` | unbound names across the package, run as part of the geometry suite |

`tools/check_names.py` is a small AST scope-chain checker; it runs inside
`test_bao_geometry.py`, which is why a name error fails the suite rather than
waiting for Isaac Sim.

`tools/memory_test.py` imports nothing from the project — only the standard
library — so it can be run anywhere with an API key, without Isaac Sim.

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
| `BAO_OUTPUT_ROOT` | write `results/`, `logs/` and `run_progress.txt` under this directory instead of the repository (e.g. a large scratch disk) |
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

* **Earlier-layout model behaviour on Level 0.** Runs recorded before the room
  was extended to 16 × 5 m showed both `qwen-vl-max` and `gemini-2.5-pro`
  stopping before the obstacle and alternating turns while reporting no
  collisions. Those coordinates are not comparable with the current layout;
  fresh runs with a new tag are required.
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
