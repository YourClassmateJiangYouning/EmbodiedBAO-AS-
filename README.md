# EmbodiedBAO — Body-as-Obstacle Benchmark

Does an MLLM-driven humanoid rotate its body **before** a passage becomes too
narrow to walk through, the way humans do?

Warren & Whang (1987) showed that people start turning sideways once a gap is
about **1.30×** their shoulder width — they anticipate the constraint instead of
waiting to be blocked. This benchmark puts a Unitree H1 in front of a wall with
a single vertical opening and sweeps that ratio from 1.58 down to 0.79,
measuring where the agent's body-scale affordance perception sits.

**The only thing that changes between Levels is the physical opening width. The
prompt is identical, and it never mentions the width, the body size, or the
need to turn.**

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

Levels 0–3 can be walked through facing forward. Levels 4–5 cannot: the
shoulders are wider than the gap, so the agent must rotate to present its
0.22 m torso thickness — and it must do so **before** reaching the wall, because
once the shoulders are in the wall plane the turn gate rejects the rotation.

## Scene

```
x : forward   (4 m room; robot starts at x = 1.5, wall at x = 2.0)
y : up        (ground at y = 0)
z : lateral   (opening centred at z = 0)
```

* Transparent wall: plane `x = 2.0`, 2.0 m tall, 0.02 m thick, spanning the room.
* Opening: vertical, centred at `z = 0`, floor to the top of the wall.
* Robot: `(1.5, 0, 0)`, facing `+x` (shoulder width 0.57 m, torso thickness 0.22 m).
* Camera: the robot's head view, 1024×1024, follows the robot root.
* Success: the body centre reaches **`x > 2.5`** (the whole body is through).

## Action space

Eight discrete actions, 5 cm or 15° each:

| Action | Effect |
| :--- | :--- |
| `forward` / `backward` | move 5 cm along the torso's facing direction |
| `left` / `right` | move 5 cm along the torso's own left / right |
| `turn_left` / `turn_right` | rotate the torso (and its head camera) 15° |
| `look_left` / `look_right` | rotate the head camera 30°, body unchanged |

Movement is **egocentric**: after turning sideways, `forward` walks along the
new facing direction. Reaching the goal therefore requires composing rotation
and translation, which is exactly the behaviour under test.

## Protocol

* 6 Levels × **20 episodes** = **120 scored episodes** per model.
* Each episode: at most **30 steps**. Episodes end on success or step exhaustion.
* Wall collisions are **recorded but never terminate** an episode.
* Every episode starts fresh from the same pose; Levels are independent, so
  there is no cross-episode memory and no guided tutorial.
* Every Level uses the same prompt — see `protocol.py`.

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
| `test_bao_geometry.py` | offline geometry/protocol verification (no Isaac Sim needed) |
| `test_bao_integration.py` | end-to-end runner + analysis test against a mock environment |

## Running

Requires Isaac Sim. Use its bundled Python:

```bash
$ISAACSIM_ROOT/python.sh main.py --model gemini-2.5-pro --all-levels
# Windows
%ISAACSIM_ROOT%\python.bat main.py --model gemini-2.5-pro --all-levels
```

Common invocations:

```bash
python main.py --model gpt-4o --level 2              # a single Level
python main.py --model gpt-4o --levels 4 5           # the narrow Levels only
python main.py --model random --level 0 --episodes 2 # offline smoke test, no API key
python main.py --model gemini-2.5-pro --all-levels --resume
```

`--resume` skips episodes already recorded in
`results/{model}/checkpoint_{tag}.json`. The default tag is a timestamp; pass
`--tag NAME` to reuse one. Note that changing the prompt, the action
definitions or the scene invalidates old results, so use a fresh tag after any
such change.

Then analyse:

```bash
python analysis.py --results_root results
```

Outputs `analysis/threshold_table.{md,csv}`, one JSON report per model, and a
`thresholds.png` plot with the human 1.30 reference line.

## Verification without Isaac Sim

The parts that decide whether the experiment is valid run on plain Python:

```bash
python test_bao_geometry.py    # 13 checks: ladder, collision gate, routes, prompt
python test_bao_integration.py #  9 checks: full protocol against a mock env
```

The geometry suite re-derives the collision model independently (an exact
oriented-box sweep against an analytic slab intersection) and pins the
properties that make the ladder meaningful:

* the declared frontal/sideways feasibility of every Level,
* the body footprint stays centred on the root pose under rotation,
* the sideways route actually reaches `x > 2.5` at every Level,
* a frontal walk succeeds at Levels 0–3 and is blocked at 4–5,
* turns from a wall-fouling pose are always rejected,
* the clearance boundary for a full 0→90° turn sits in front of the wall,
* the prompt is byte-identical and contains no geometry leaks.

The integration suite drives the real runner against a kinematic mock built
from the same geometry helpers, and checks the record contract, collision
handling, CSV columns, threshold recovery, checkpoint resume, invalid-response
handling, and a full six-Level run feeding `analysis.py`.

## Environment variables

| Variable | Purpose |
| :--- | :--- |
| `BOYUE_API_KEY` / `TAOTOKEN_API_KEY` / `OPENAI_API_KEY` | API key (first match wins) |
| `BOYUE_BASE_URL` / `TAOTOKEN_BASE_URL` / `OPENAI_BASE_URL` | endpoint override |
| `EMBODIEDBAO_H1_USD` | explicit path to the H1 USD asset |
| `BAO_LLM_TIMEOUT`, `BAO_MAX_RETRIES` | request timeout / retry count |
| `BAO_DISABLE_PROXY=1` | clear proxy variables before calling the API |

## Notes on the physics model

The robot is kinematic: actions teleport the root pose and every candidate pose
is gated by an analytic collision test. The torso footprint is an oriented
rectangle (0.22 m × 0.57 m) tested against the two wall panels with an exact
separating-axis test, inflated by a 2 mm skin so that A/S = 1.00 is a real
pinch point rather than a zero-clearance squeeze. Rotation checks sweep the
whole 15° arc, and a pose that already fouls the wall cannot rotate free — so
the agent cannot teleport through the wall one 15° hop at a time.

## Out of scope (future work)

The current study is the A/S threshold only. Follow-ups once the threshold is
known: soft-material edges (does the agent try to squeeze through?), strategy
persistence (does a wide channel still trigger a turn after priming?), and
insight-versus-gradual learning curves across repeated episodes.
