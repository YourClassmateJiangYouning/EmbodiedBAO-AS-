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

A/S = channel width / shoulder width (0.57 m). The ladder is the **12-width
aperture series of the human study this benchmark follows**: A/S = 2.0 down to 0.9
in steps of 0.1, five episodes per width (Warren & Whang 1987; Keizer et al. 2013
used the same 12 ratios × 3 trials). Sampling at 0.1 is what localises the
threshold: the reference band 1.25–1.30 falls between two Levels.

Widest first, so a sweep runs from trivially passable toward rotation-required:

| A/S | Channel | Frontal passage |
| :--- | :--- | :--- |
| 2.0 → 1.4 (Levels 0–6) | 1.140 → 0.798 m | easy |
| 1.3 (Level 7) | 0.741 m | the human reference ratio |
| 1.2 (Level 8) | 0.684 m | just below it |
| 1.1 (Level 9) | 0.627 m | tight |
| 1.0 (Level 10) | 0.570 m | exactly flush: only dead centre |
| 0.9 (Level 11) | 0.513 m | **impossible unturned** — needs ≥60° of rotation |

The slack is measured, not derived: at A/S = 1.0 the aligned body has zero
tolerance, and the earlier coarse ladder showed the same effect at its own flush
Level (`tools/check_heading_frame.py` prints the whole series). A/S = 0.9 is
narrower than the shoulders at every yaw below 54°, so it is the Level that
requires the rotation this benchmark measures — and the rotation must happen
**before** reaching the wall, because the turn gate samples the robot's current
pose and rejects any rotation that would sweep the shoulders through a panel.

Why 12 points and not 6: a fixed policy ("always turn 90°") produces a high
rotation rate at *every* width, including the ones that need no rotation at all.
Only a curve with enough points can show that the rate does not vary with A/S, and
that check is reported as `rotation_gradedness` in the threshold table.

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
| Success | body centre reaches **`x >= 8.75`** (one stride past the wall) |
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
| `look_left` / `look_right` | one-frame glance **30°** off the walking direction |
| `look_down` | one-frame glance **45°** down: the only view of its own body |

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
| 0–9 | 2.0–1.1 | every yaw |
| 10 | 1.0 | 0°, or 45°–90° |
| 11 | 0.9 | 60°–90° |

So the rotation a model adopts is graded in 15° steps, which is the quantity that
can be compared with the human threshold of 1.30.

The **gaze** is pinned to the walking direction: rotating the torso changes the
body's footprint and nothing the agent can see. That matches how a person crosses
a narrow opening — eyes on the opening, shoulders rotated — and it keeps the
channel in view at the large rotations this ladder asks for, which a torso-mounted
camera could not: the channel subtends 76° and would leave the field of view. The
torso angle is reported to the agent as a number, which is the proprioception a
person has.

The eye is at body centre 1.68 m up, pitched 15° down, so the agent's own body is
**outside the frame**. `look_down` exists for that: one 45° downward glance, which
is how a person checks their own width — and it was verified on the lab machine
rather than assumed. `tools/check_look_down_view.py` captures the glance twice,
with the robot's prims active and with them deactivated: 84.5% of the frame changes
when the robot is hidden, so the body is in view; the body region has real
structure (pixel std 33.5, not a flat close-up blur); and the remaining 15.5% is
structured room (std 23.3), so there is a scale reference around it. Eyeballed on
the rendered frame: shoulders and below, much as a person sees looking down.
Frames land in `look_down_check/`.

The translation step is 0.75 m, approximately an adult walking step. From the
`x = 0.5` start, ten forward translations reach the obstacle plane at `x = 8.0`,
and **one more** clears the wall: success is scored at `x = 8.75`, because the task
is to get through the opening and the body's largest half-extent is 0.306 m, so at
8.75 the whole body is on the far side at any torso angle. The narrowest Level
(A/S = 0.9) needs four 15° turns (60°, the measured minimum) plus those eleven
steps, so the narrowest route is 15 of the 30 steps.

## Protocol

* 12 Levels × **5 episodes** = **60 scored episodes** per model: the reference's
  12-width aperture series with five repetitions per width.
* Each episode: at most **30 steps**. Episodes end on success or exhaustion.
* Wall collisions are **recorded but never terminate** an episode.
* Every episode starts fresh from the same pose; Levels are independent, and the
  agent carries no memory between episodes.
* Every Level uses the same prompt — see `protocol.py`.

### What the model is told each step

The prompt contains the task, the nine actions with their real distances and their
mechanical effects, a note that the walking direction is fixed at the far wall and
that the eyes stay on it when the torso turns, the robot's own position, torso
rotation and both **head-camera offsets**, the step limit, and **the full action
history
for the episode so far**. Each history entry is
`step N: <action> -> <feedback> | your reasoning: <the agent's own reasoning>`,
listed oldest first, and the block explicitly invites the agent to use it to
notice what it has already tried and whether it worked.

The head-camera offsets are reported because the two rotation families are not the
same thing and nothing in a single frame distinguishes them. `turn_left` /
`turn_right` rotate the torso and leave the view **unchanged**: the eyes are pinned
to the walking direction. `look_left` / `look_right` / `look_down` give a single
glance on one axis, which **clears on the next action** and does not accumulate. An
agent that had just glanced would otherwise read a view 30° off its path, or 45°
down at its own feet, as if it were the normal forward view — being tested on
guessing the interface rather than on judging its own body.

The prompt never contains the channel width, the body dimensions, the A/S ratio,
or any hint that a turn may be needed. `build_prompt()` takes no `level`
parameter, so leaking the geometry structurally is not possible. The action
descriptions mention turning and the head camera, but only to document what the
controls do; the task statement itself carries no advice about the solution, and
the test suite checks those two properties separately.

**The task is stated as a destination**, not as an instruction about the obstacle:
"reach the red marker on the far wall". That mirrors the studies this benchmark
follows — Keizer et al. sent participants to a table beyond the aperture and
presented the aperture as meaningless panels; Lenkei et al. sent dogs to their
owner through an opening — so the obstacle has to be discovered and judged rather
than announced. Telling the agent to "pass through the opening" would hand it the
fact that there is an opening to fit through. The marker is visible **only**
through the opening (sight lines to it are blocked by the 2 m obstacle wall
everywhere else), so obeying the instruction requires finding and using it.

The task statement is a bare destination and nothing more: it must not mention the
corridor, where the agent starts, or that there is a passage at all, because those
are the things the agent is supposed to discover. The connection between the goal
and the control that reaches it lives in the **action list** instead, since that is
where the controls are documented: `forward` is "walk 75cm straight ahead, in your
walking direction, which always takes you toward the red marker on the far wall",
and the walking-frame note says the fixed walking direction is where the marker is.
That is the v6 change, and it is not a hint about the obstacle — the width of the
opening, its distance and the body's own width all stay hidden, so the agent still
has to judge from the image whether it fits. It was made because the v5 logs showed
what happens without it: the agent read "reach the red marker" in the task and "your
walking direction points at the far wall" in the action list, never joined the two,
and spent the episode trying to *align* itself with something it could not connect
to any action, usually with a sidestep that put its body outside the opening.

v7 adds the other half, in the state block: which axis each control moves along.
`x` is the forward axis and `z` is the lateral one, and the raw tuple
`position (x, y, z)` was being read as image coordinates, where x is horizontal.
That is measured, not assumed — the v6 run reported "I am currently at x=1.25,
meaning I am significantly to the right of the centerline" and answered every
forward step with `left` steps until the room's side wall blocked the third one,
then tried to fix its "width" by rotating its torso to 90 degrees. The same class of
statement as the walking-frame note, and again not a hint about the answer.

v6 and v7 are wording-only: the task statement is byte-identical to v5, and the
ladder, the action mechanics, the episode length and the success plane are
unchanged, so the protocols are directly comparable.

Note the goal marker sits at `x = 16.0` while success is scored at `x = 8.75`:
the agent is scored as soon as it is clear of the wall, which is what "get through"
means physically, and it never has to reach the marker itself.

The history is the agent's own within-episode memory, not a sliding window, and
it is not truncated. The action distances are derived from `MOVE_STEP` rather
than hard-coded, so the prompt always advertises the configured 0.75 m movement
instead of drifting from the environment.

## Recorded data

Per episode (`results/level{level}/{model}/{tag}/episode_{id:03d}.json`):

```
episode_id, level, channel_width, a_s_ratio,
passed, passed_sideways, passage_rotation_deg, total_rotation, first_turn_step,
total_steps, action_sequence
```

`passage_rotation_deg` is the torso angle at the step where the body reached the
wall plane, which is where the passage is scored. It is recorded as an angle and
not only as the 45–135° boolean because the human comparison is a *curve* —
shoulder rotation against aperture width — and scoring the orientation at the end
of the episode instead would call a person who straightened up after getting
through "never turned sideways".

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
| **Turned rate** | fraction of episodes in which the torso was successfully rotated at all |
| **Turned threshold** | widest A/S at which the torso was rotated — **the metric to compare against the human 1.30** |
| **Sideways rate** | fraction of episodes that passed while the torso was rotated 45–135° at the wall plane |
| **Sideways threshold** | widest A/S at which a sideways *passage* was observed; also requires success |
| **Rotation at passage** | mean torso angle at the wall plane, the graded version of the threshold |
| **First turn step** | step index of the first successful rotation action |
| **Mean pass steps** | average steps needed for a successful passage |

The human reference (Warren & Whang 1987) counts rotated shoulders, not completed
passages, so **Turned threshold** is the closer analogue: it separates the decision
from the execution and cannot be depressed by a model that rotates correctly and
then fails to follow through. `Sideways threshold` is reported alongside it for
continuity with the earlier runs.

Either threshold classifies a model:

| Class | Threshold | Reading |
| :--- | :--- | :--- |
| `anticipatory` | ≥ 1.30 | turns before the gap is tight, like a human |
| `borderline` | 1.00 – 1.30 | turns while still passable |
| `reactive` | < 1.00 | only turns after being blocked |
| `no_sideways` | — | never rotated |

## Files

| File | Role |
| :--- | :--- |
| `environment.py` | Isaac Sim scene, kinematics, analytic collision gate, success test |
| `protocol.py` | single source of truth for the action space and the unified prompt |
| `experiments.py` | episode loop, Level ladder, record shaping, checkpoints |
| `ai_agent.py` | unified OpenAI-compatible MLLM client + `random` baseline |
| `main.py` | CLI entry point, CSV export, threshold report |
| `analysis.py` | per-Level metrics, A/S threshold, reports and plots |
| `memory_metrics.py` | Stage 2 measures: how wide the body is across an opening, how close a round came to fitting, actions wasted, attempt labels, and the pre-registered insight/gradual/perseveration criteria. Standard library only, no Isaac Sim, no API |
| `memory_protocol.py` | Stage 2 protocol: the run plan (6 runs x 17 rounds), the three prompts, the memory block, and the round record |
| `memory_experiment.py` | Stage 2 runner: the round loop, the note call between attempts, the call budget and resume. Run it as `python memory_experiment.py --model <name> --resume` |
| `test_bao_memory.py` | offline checks for both of the above, against an independent projection of the body rectangle and the simulator's own collision gate |
| `test_bao_memory_runner.py` | offline end-to-end checks of the runner against a mock environment and a scripted agent: the memory, the note calls, the phase change, resume |
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
| `STAGE23_DESIGN.md` | authoritative spec for the memory/habit experiment (Stage 2 and 3): geometry, run structure, the three prompts, record fields, pre-registered criteria |
| `lab_logs/` | retrieved lab data: the v7 and v5 archives every analysis reads, the derived episode table, the figures and the analysis scripts (see "Reproducing this repository") |

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
each model its own protocol-versioned tag, and runs 12 Levels × 5 episodes × 30
steps per model
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

**Run this before every sweep**, because `run_all_models.sh`'s own preflight
cannot cover it:

```bash
python tools/verify_models.py          # one real text+image request per candidate
```

The preflight in `main.py` asks `/v1/models` for a status code, which proves the
endpoint and the key are alive but says nothing about whether a *named* model has a
working channel behind it. Measured twice, at a cost of about four hours each:

* `claude-sonnet-4-5-20250929` is listed by `/v1/models` and still fails every
  call — HTTP 500 `get_channel_failed` ("no available channel for this model in
  group auto"), then HTTP 400 `Access to Bedrock models is not allowed for this
  account`. It is now excluded; `claude-sonnet-4-6` answers and sees.
* The runner does not treat a failed call as fatal. It records the episode with
  `end_reason=max_steps`, `passed=false` and `invalid_response_count=30`, at the
  start position — which looks exactly like a model that refused to move, and is
  why `invalid_response_count` has to be checked before any analysis.

`tools/verify_models.py` catches both cases because it sends a real request
carrying a solid red image and requires the model to name the colour.

### The repeated-attempt experiment (Stage 2)

`STAGE23_DESIGN.md` is the spec. Each model is run 6 times; a run is 12 attempts at
an opening it cannot walk through facing forward (A/S 0.80) followed by 5 at one it
can (A/S 1.10), with a text-only note call between attempts that is the only thing
it can read about the attempt next time. 17 attempts and 16 note calls per run,
about 2,300 model calls per model for all six runs.

```bash
ISAAC_PY=/home/ybh/isaacsim/python.sh
$ISAAC_PY memory_experiment.py --model qwen3-vl-32b-instruct --runs 1-6 --headless --resume
```

Cheap checks before committing to the full thing — the first needs no API key, and
both stop cleanly at the budget with every finished attempt on disk:

```bash
$ISAAC_PY memory_experiment.py --model random --runs 1 --headless --max_calls 40
$ISAAC_PY memory_experiment.py --model qwen3-vl-32b-instruct --runs 1 --headless --max_calls 60
```

To leave it running across a dropped SSH session:

```bash
cd ~/EmbodiedBAO-AS- && export BOYUE_API_KEY='sk-...' && nohup bash -c '/home/ybh/isaacsim/python.sh memory_experiment.py --model qwen3-vl-32b-instruct --runs 1-6 --headless --resume' > memory_sweep.log 2>&1 & echo $! > memory_sweep.pid
```

Round records, step sidecars and a checkpoint live under
`results/memory/{model}/{tag}/`, keyed by the tag `v8-memory-a08-a11-cum` for runs
1-4 and `-roll` for runs 5-6; note prompts and replies live under
`logs/{tag}/run{NN}_round{NN}_note.txt`. The `results/memory/` prefix is deliberate:
Stage 1's `analysis.py` looks for `results/level*`, so it can never mistake a Stage 2
record for a threshold episode.

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

## Lab operations: running a sweep, watching it, and getting the data back

Everything here was learned by getting it wrong on the lab machine at least once.
On that machine the repository is `~/EmbodiedBAO-AS-` and the Isaac launcher is
`/home/ybh/isaacsim/python.sh`. `results/`, `logs/`, `analysis/` and
`run_progress.txt` are all gitignored, so **nothing leaves the machine on its own**:
a sweep that is never retrieved is a sweep that has to be run again. Retrieve it
before analysing anything on this side.

### Start it so a dropped SSH session cannot kill it

`tmux` is documented above; the `nohup` form is the same idea for a
non-interactive session. Single line, so it can be pasted whole:

```bash
cd ~/EmbodiedBAO-AS- && export BOYUE_API_KEY='sk-...' && nohup bash -c 'ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models.sh' > sweep.log 2>&1 & echo $! > sweep.pid
```

`$!` is the PID of the `nohup`'d shell — that is what belongs in `sweep.pid`, and it
is *not* the same number bash prints as the job id in its own `[1] 12345` line. That
line is the wrapper subshell, and stopping the sweep by job id, or by excluding that
job id from a wider kill, has already killed the wrong thing here once.

**Never use `pkill -f isaac`, `pkill -f main.py` or `pkill -f python.sh` on that
machine.** It is shared: one such command takes out somebody else's simulation, and
Isaac Sim does not survive losing its process group politely. Kill by the recorded
PID, and only after checking what it is:

```bash
cat sweep.pid && ps -o pid,ppid,etime,cmd -p "$(cat sweep.pid)"
```

Only one Isaac Sim instance can run at a time on that box, so two sweeps in
parallel do not go twice as fast — they collide.

### Is it still running, and where is it?

```bash
pgrep -af 'run_all_models'          # the sweep loop itself is alive
pgrep -af 'main.py'                 # the model process inside it is alive
tail -n 20 sweep.log                # which model it is on right now
```

Per-model progress, read from the checkpoints `--resume` uses (a finished model
reports 60 = 12 Levels x 5 episodes; duplicates in the list are separate tags and
are listed by tag, not by model):

```bash
python3 -c "import json,glob,os; [print(os.path.basename(p)[11:-5], len(json.load(open(p))['completed'])) for p in sorted(glob.glob('results/*/checkpoint_*.json'))]"
```

Is the newest record recent, and is what it recorded real? A record that is old, or
one whose `invalid_response_count` is 30, means the model is not actually being
asked anything — 30 fallback steps at the start position look exactly like a model
that refused to move:

```bash
python3 -c "import glob,json,os,time; p=max(glob.glob('results/level*/*/*/episode_*.json'), key=os.path.getmtime); d=json.load(open(p)); print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p))), p, d['passed'], d['end_reason'], 'invalid=' + str(d['invalid_response_count']))"
```

How many scored episodes exist on disk, across layouts:

```bash
python3 -c "import glob; print(len([p for p in glob.glob('results/level*/*/*/episode_*.json') if not p.endswith('_steps.json')]))"
```

`episode_NNN.json` and `episode_NNN_steps.json` both match `episode_*.json`, so a
count that does not exclude `_steps.json` is exactly twice the number of episodes.
The per-step sidecar is also a different shape from the record: it is a list of step
objects, not one dict.

A checkpoint that says an episode is done while its record file is missing is
handled in the runner rather than trusted (`experiments.py`, the resume path): that
episode is re-run instead of being counted as missing data. So a disk count below
the checkpoint count is worth investigating, but it is not automatically lost data.

### Getting the data back

On the lab machine, one archive containing everything a report needs — records,
per-step sidecars, raw model I/O, the Level summaries, the analysis output and the
timeline:

```bash
cd ~/EmbodiedBAO-AS- && tar czf bao_results_$(date +%Y%m%d-%H%M).tgz results logs analysis run_progress.txt && md5sum bao_results_*.tgz
```

Copy it back with the MD5 that `tar`+`md5sum` just printed, from the Windows
workspace root (the address is whatever you already `ssh` to):

```powershell
scp 'eai@<lab-host>:~/EmbodiedBAO-AS-/bao_results_*.tgz' .
Get-FileHash -Algorithm MD5 .\bao_results_*.tgz
```

The remote path is quoted so the glob is expanded by the lab machine's shell, not by
PowerShell; the archive name carries the timestamp of the run, so it is not
reproducible by hand and is safest left as a glob.

`Get-FileHash` and `md5sum` report the same digest for the same bytes, so the two
lines are comparable character for character. **Check them before extracting**: a
truncated `scp` produces a `.tgz` that still extracts part of the way, and a partial
sweep is much harder to spot after the fact than a mismatched hash.

Unpack into `lab_logs/extracted/` (gitignored — the archives themselves are
committed, the unpacked tree is not) rather than into
the repository root, so the lab's own `results/` cannot be confused with a local
run's, and analyse from there:

```bash
mkdir -p lab_logs/extracted && tar xzf bao_results_*.tgz -C lab_logs/extracted && ls lab_logs/extracted
```

### Traps that have already cost a sweep

| Trap | Symptom | What it actually is |
| :--- | :--- | :--- |
| `import environment` before `SimulationApp` | the script builds nothing and **exits 0** | `environment._HAS_ISAAC_SIM` latches to `False` for the life of the process; pinned by the AST check in `test_bao_geometry.py`, which is why that suite must pass before a sweep |
| Killing by job id | the new sweep loop dies, the old one keeps running | `[1] 12345` is the wrapper subshell, not the script; use the PID you recorded |
| `pkill -f isaac` | someone else's simulation dies | shared machine, one Isaac instance at a time |
| a model listed by `/v1/models` but with no channel | 30 identical fallback steps, `passed=false`, `end_reason=max_steps` at the start position | run `tools/verify_models.py` *before* the sweep; measured twice, about four hours each |
| relative paths | a `--tag` run cannot see the other's checkpoint | always `cd ~/EmbodiedBAO-AS-` first; `--resume` keys on `results/{model}/checkpoint_{tag}.json` |
| deleting "stale" files next to a diagnostic | good records gone | print the file list, read it, *then* delete; this has already destroyed three usable deepseek episodes |
| pasting a multi-line command with `<placeholders>` | the shell errors or runs half of it | single line, real values, one command per paste |
| copying the Windows working tree to the lab | `bash: /usr/bin/env: bad interpreter` | `core.autocrlf` rewrites `*.sh` to CRLF, so the shebang stops being a shebang; `.gitattributes` pins `*.sh text eol=lf` because this transfer path is used |
| counting `episode_*.json` | double the true episode count | `_steps.json` matches too |

## Reproducing this repository

A clone contains the code **and the collected data of every scored experiment**, so
re-deriving the analysis needs neither Isaac Sim nor an API key:

| Committed | What it holds |
| :--- | :--- |
| `lab_logs/bao_v7_all.tgz` | the 660-episode v7 sweep — 11 models x 12 Levels x 5 episodes — with the per-step sidecars, the raw model I/O, `summary_v7.txt`, `run_progress.txt` and `git_head_v7.txt` |
| `lab_logs/bao_v5_logs.tgz` | the 346-episode v5-era runs on the earlier layouts and prompts |
| `lab_logs/embodiedbao_v7_episodes.csv` | the 660 rows every figure and table is derived from |
| `lab_logs/*.py` | `export_table.py`, `analyze_actions.py`, `analyze_reasoning.py`, `make_figures.py`; all four read the archive directly rather than the unpacked tree |
| `lab_logs/figures/` | fig1–fig4, PDF and 600 dpi PNG |
| `lab_logs/summary.txt` | the written analysis summary |

Left out on purpose: `lab_logs/extracted/` (a partial unpacking of the v5 archive,
60 MB) and `lab_logs/bao_v7_qwen.tgz` (a strict subset of the v7 archive). Both were
checked against the committed archives **key by key and byte by byte** (SHA-256)
before being left out:

| Copy | Records | Identical to an archive | Conflicting | Unique to this copy |
| :--- | :--- | :--- | :--- | :--- |
| `results/` (loose, this machine) | 346 | 346/346 vs `bao_v5_logs.tgz` | 0 | 0 |
| `lab_logs/extracted/` | 346 | 346/346 vs `bao_v5_logs.tgz` | 0 | 0 |
| `lab_logs/bao_v7_qwen.tgz` | 180 | 180/180 vs `bao_v7_all.tgz` | 0 | 0 |

The per-step sidecars match the same way (346/346 identical again). So committing
the two archives and nothing else drops nothing.

### From a fresh clone, offline

```bash
python test_bao_geometry.py; python test_bao_integration.py; python test_bao_persistence.py; python test_bao_memory.py; python test_bao_memory_runner.py
python lab_logs/export_table.py
python lab_logs/analyze_actions.py
python lab_logs/analyze_reasoning.py
python lab_logs/analyze_door_reach.py lab_logs/bao_v7_all.tgz
python lab_logs/verify_optimal_steps.py
python lab_logs/make_figures.py
```

The first line is the 35 + 19 + 13 offline checks; the rest rebuild the episode
table, the statistics and the figures from inside the archive. Re-running them
leaves `git status --short` clean, which is what makes this repository reproducible
rather than merely archived.

What a clone still cannot reproduce is a **new sweep**: that needs the Isaac Sim
machine, the API key, and model versions that may since have changed. `models.json`,
the run tags and `git_head_v7.txt` are what record which models, which settings and
which commit produced each dataset.

**Before `bao_v7_all.tgz` was committed, the entire 660-episode Stage 1 existed in
exactly one file on this machine**: the loose `results/` tree holds v5-era runs
only, zero v7 records. Losing that one archive would have lost Stage 1 outright. It
is versioned now — do not delete it, and do not regenerate it in place.

## Verification without Isaac Sim

Everything that can be decided without a renderer runs on plain Python:

```bash
python test_bao_geometry.py     # 37 checks: ladder, collision gate, routes, colours, prompt
python test_bao_integration.py  # 20 checks: full protocol, tagged runs, CLI flags, against a mock environment
python test_bao_persistence.py  # 13 checks: interrupt and corruption safety of every artefact written
python test_bao_memory.py       # 26 checks: the Stage 2 measures and prompts, against an independent model and the real gate
python test_bao_memory_runner.py # 9 checks: the Stage 2 runner end to end against a mock environment and a scripted agent
```

The geometry suite re-derives the collision model independently and pins the
properties that make the ladder meaningful:

* the ladder is the reference 12-width A/S series (2.0 → 0.9 by 0.1) and every
  width is that ratio times the body's own shoulder width,
* the declared frontal feasibility of every Level, asserted against A/S >= 1.0
  rather than from a table,
* the body footprint stays centred on the root pose under rotation,
* the rotate-then-walk-forward route reaches past the success plane at every
  Level, with the minimum rotation derived per Level, and one turn less than that
  minimum is asserted to fail,
* a frontal walk succeeds wherever A/S >= 1.0 and is blocked at the narrowest
  Level,
* turns from a wall-fouling pose are always rejected,
* the clearance boundary for a full 0→90° turn sits in front of the wall,
* the robot has enough free run from its start pose to complete that turn,
* the channel edge posts do not narrow the opening,
* box axes map so that a 2.0 m tall post gets a 2.0 m vertical extent,
* every pair of visible surfaces differs enough to be told apart,
* the prompt states the same step length that the agent actually moves,
* look_down is a camera glance that does not translate the robot,
* the prompt is identical, deterministic, and leak-free.

## Diagnostics on a new machine

```bash
ISAAC_PY=/home/ybh/isaacsim/python.sh bash verify_professor_machine.sh 2>&1 | tee professor_verify.log
```

Runs the offline suites (geometry, integration, persistence, Stage 2 measures), the passability
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
| `BAO_MODEL_PARAMS` | JSON per-model request overrides, merged over the built-in map (see below) |
| `BAO_DISABLE_PROXY=1` | clear proxy variables before calling the API |
| `BAO_OUTPUT_ROOT` | write `results/`, `logs/` and `run_progress.txt` under this directory instead of the repository (e.g. a large scratch disk) |
| `EMBODIEDBAO_H1_USD` | explicit path to the H1 USD asset |

## Reasoning budget: a measured, recorded configuration

Some models spend most of a call thinking before they answer, and that is the whole
cost of a sweep. `tools/probe_reasoning.py` measures that with the real prompt and a
512 px frame, because the gateway answers **HTTP 200 for every unknown parameter**,
so a key being accepted proves nothing — only the token count and the latency show
whether it had an effect:

| model | default | with thinking off | knob |
| :--- | :--- | :--- | :--- |
| `deepseek-v4.1-flash` | 86.8 s / 12,820 tok | **20.1 s / 1,224 tok** | `reasoning_effort="none"` |
| `glm-4.6v` | 12.9 s / 1,595 tok | **6.4 s / 1,397 tok** | `thinking={"type":"disabled"}` |
| `qwen3-vl-32b-instruct` | 3.9 s / 1,342 tok | unchanged | — (no extended thinking) |
| `qwen3-vl-235b-a22b-instruct` | 3.3 s / 1,331 tok | unchanged | — |
| `qwen-vl-max` | 3.5 s / 1,304 tok | unchanged | (`reasoning_effort="low"` → HTTP 400) |

The Qwen family already answers without deliberation, so turning thinking off for
the models that do deliberate makes the roster **more** comparable, not less. The
overrides live in `ai_agent.MODEL_REQUEST_PARAMS`, can be overridden with
`BAO_MODEL_PARAMS`, and are written into each run's `logs/{tag}/args.json` so the
report cannot disagree with what was sent. Both request paths are *meant* to apply
them, so that behaviour does not depend on whether an unrelated package is
installed — but only the built-in HTTP client is verified here, because the `openai`
package is not installed on the machine these runs happen on. That path is covered
by `test_parameterised_models_survive_the_compat_client`; the SDK path passes the
same values as call kwargs and would need `extra_body` if its `create()` rejects
unknown names.

They change what the model *does*, so they are a reported configuration: a run with
them removed is the ablation, and `deepseek-v4.1-flash` at its default setting costs
about 43 h for one sweep instead of about 7 h.

Two traps this configuration has already sprung, both worth knowing before trusting
the table above:

1. `tools/probe_reasoning.py` measures the gateway with a **raw HTTP request**, which
   is not the path the runner uses. The two agree because they send the same JSON,
   so a parameter can measure perfectly and still be unusable in a sweep. The only
   test that covers the runner's own path is
   `test_parameterised_models_survive_the_compat_client`, and it exists because it
   did not: the standard-library client declared four keyword arguments, the runner
   passed a fifth, and **every call to the only two parameterised models on the
   roster** died locally with `TypeError: ... unexpected keyword argument
   'reasoning_effort'` before reaching the network. The nine parameterless models
   were unaffected, so no amount of smoke-testing them would have found it.
2. A model with parameters is therefore the risky case, not the exotic one. Run the
   parameterised models once, end to end, after any change to `ai_agent.py`, and
   check the recorded `invalid_response_count` before analysing: 30 means the whole
   episode was fallback steps at the start position.

## Notes on the physics model

The robot is kinematic: actions teleport the root pose and every candidate pose is
gated by an analytic collision test. The torso footprint is an oriented rectangle
(0.22 m × 0.57 m) tested against the two wall panels with an exact
separating-axis test, with **no inflation**: `BODY_CLEARANCE` is 0, and a
`OVERLAP_TOLERANCE` of 2e-7 m is subtracted from each projection so that boxes
touching exactly count as clear. That is what makes A/S = 1.00 a width an aligned
body can actually pass; an earlier 2 mm skin made that Level demand 0.574 m
while still printing 1.00, so it was geometrically impossible and every model
scored zero there for a reason that had nothing to do with its behaviour.
Rotation checks sweep the whole 15° arc, and a pose that already fouls the wall
cannot rotate free — so the agent cannot teleport through the wall one 15° hop at
a time.

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

Of those, strategy persistence and the insight-versus-gradual learning curves are
now specified, and their prompts and pre-registered criteria fixed, in
`STAGE23_DESIGN.md`. That document is the authority for everything it covers; the
threshold study described above is its baseline, not its subject.
