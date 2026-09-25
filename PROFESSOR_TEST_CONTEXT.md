# Professor-machine verification context

## Purpose

This revision changes the EmbodiedBAO scene from a short square room to a long,
fully enclosed passage while preserving the lateral channel geometry used by the
A/S experiment.

## Expected revision behaviour

| Item | Expected value |
|---|---:|
| Room length along `x` | `16.0 m` |
| Room width along `z` | `5.0 m` (`z = -2.5 .. +2.5`) |
| Room height | `3.0 m` |
| Robot start | `(0.5, 0, 0)` |
| Translation per locomotion action | `0.75 m` |
| Obstacle wall | `x = 8.0 m` |
| Success condition | body centre `x >= 8.75 m` (one stride past the wall) |
| Green far wall | `x = 16.0 m` |
| Episode budget | `30 actions` |

The distance arithmetic is intentional:

- `(8.0 - 0.5) / 0.75 = 10` forward translations from the start to the obstacle plane;
- 1 more clears the wall: success is at `x = 8.75`, one stride past the wall plane;
- `(16.0 - 8.75) / 0.75 = 9.67`, so the room still extends about ten steps past the goal;
- the narrowest route is five 15-degree turns (75 degrees) plus eleven
  translations, or `16/30` actions.

Actions are in the **walking frame**: the walking direction always points at the
far wall, so `forward` advances along world `+x` whether the torso is turned or
not, and `turn_left`/`turn_right` rotate the torso relative to that direction. The
turn does not steer; it changes how much of the opening the body occupies, which
is the shoulder rotation the human aperture literature measures. Measured
consequences per Level, and the side-by-side comparison with the earlier
body-frame convention, are in `tools/check_heading_frame.py`.

## Geometry that must not change

The ladder is the reference aperture series, 12 widths from A/S = 2.0 down to 0.9
in steps of 0.1 (Warren & Whang 1987; Keizer et al. 2013), which at a 0.57 m
shoulder is:

`1.140, 1.083, 1.026, 0.969, 0.912, 0.855, 0.798, 0.741, 0.684, 0.627, 0.570, 0.513 m`

Level 0 is the widest, so a sweep runs toward the narrowest, and the human
reference band 1.25-1.30 is bracketed by Levels 7 (1.3) and 8 (1.2).

The H1 analytic body remains `0.57 m` shoulder width by `0.22 m` torso thickness,
with no inflation: the gate tests the body exactly, and a 2e-7 m tolerance in the
separating-axis comparison keeps boxes that touch exactly clear. Therefore every
Level with A/S >= 1.0 permits a frontal passage (Level 10, A/S = 1.0 exactly:
shoulder and channel are equal, so an aligned body fits with no aim tolerance to
spare), while Level 11 (A/S = 0.9) does not at any torso angle below 54 degrees,
so it needs at least 60 degrees on the agent's 15-degree lattice.

## Important implementation details

- Ground, side walls, ceiling and analytic far boundary all extend to `x=16`.
- Side boundaries remain at `z=±2.5`.
- The obstacle panels derive their span from the 5 m room width, not its 16 m length.
- Four interior sphere lights are distributed at `x=2, 6, 10, 14` so both sides
  of the opaque obstacle are lit.
- Collision-gated translations sample the complete path; 0.75 m actions cannot
  teleport through the 0.02 m obstacle.
- Turn collision checks sample the swept rotation at intervals no larger than 1 degree.
- `passed_sideways` is scored at the obstacle plane (`x=8`), not at the later goal.
- Experiment episode files are isolated by run tag:
  `results/level{level}/{model}/{tag}/episode_NNN.json`.
- `analysis.py --tag TAG` analyzes one exact run. Without `--tag`, it selects the
  latest tagged run consistently across Levels.
- If `--move_step` is overridden, the same distance is shown in the model prompt.

## Automated verification

From the repository root on the professor's Linux workstation:

```bash
chmod +x verify_professor_machine.sh
ISAAC_PY=/home/ybh/isaacsim/python.sh \
  bash verify_professor_machine.sh 2>&1 | tee professor_verify.log
```

If the Isaac launcher is elsewhere, replace `ISAAC_PY` accordingly.

The script performs:

1. revision and working-tree capture;
2. exact scene-constant arithmetic checks;
3. the offline geometry and integration suites;
4. the passability diagnostic;
5. an Isaac Sim import probe;
6. a real scripted Level-5 traversal through Isaac Sim;
7. rendered Level-0 and Level-5 diagnostic views.

Return these artifacts:

- `professor_verify.log`
- `professor_views/level0/`
- `professor_views/level5/`

## Optional real-model smoke test

Only after the automated and visual checks pass:

```bash
export BOYUE_API_KEY='...'
ISAAC_PY=/home/ybh/isaacsim/python.sh \
RUN_MODEL_SMOKE=1 MODEL=gemini-2.5-pro \
  bash verify_professor_machine.sh 2>&1 | tee professor_model_smoke.log
```

This runs one Level-5 episode with a fresh timestamped tag and saves observations.
Do not reuse pre-layout-change tags or results.

## Running the scored sweep

Only after the automated and visual checks pass, start the 11-model roster.
`models.json` is the single definition of the roster; `run_all_models.sh` reads
it, gives every model its own tag, and resumes each model from its checkpoint.

```bash
export BOYUE_API_KEY='...'
tmux new -s bao
ISAAC_PY=/home/ybh/isaacsim/python.sh \
  bash run_all_models.sh 2>&1 | tee sweep.log
# Ctrl-b d to detach; tmux attach -t bao to come back
```

The script refuses to start if that interpreter cannot import `isaacsim`, so a
wrong `ISAAC_PY` fails immediately instead of 11 times. Budget **days** for the
whole roster: up to 1800 model calls per model. Each episode record is written
atomically the moment it is scored, and `results/{model}/checkpoint_{tag}.json`
is updated after its episode, so an interruption loses at most the episode in
flight; re-running the same command continues from there.

Afterwards, on the same machine:

```bash
python3 analysis.py --results_root results
```

`analysis.py` discovers every model under `results/`, follows the tagged layout
that `main.py` writes, and prints the A/S threshold table.

Return these artifacts for the scored sweep (about 660 episode records and 660
agent logs for 11 models — a single archive is easiest):

```bash
tar czf bao_results.tgz results logs analysis run_progress.txt
```

- `results/` — episode records, per-step sidecars, Level summaries, per-invocation
  flat CSVs, and `{model}/checkpoint_{tag}.json`;
- `logs/` — the raw model I/O per episode, plus `logs/{tag}/args.json` with the
  effective settings of each run;
- `analysis/` — the threshold reports and plot, if the analysis was run there;
- `run_progress.txt` — the timeline, including timestamps and any interruption.

If a model's run was interrupted, re-running `run_all_models.sh` on that machine
first is cheaper than analysing a partial Level: it resumes from the checkpoint
and only re-runs what is missing.

## Visual acceptance criteria

Inspect the generated PNGs and verify:

- the top view shows a 16-by-5 enclosed room and full-length ceiling;
- the blue obstacle is at `x=8`, with the green far wall at `x=16`;
- `front.png` clearly shows the channel and both dark vertical edge posts;
- `behind.png` is behind the obstacle and looks back through the channel;
- the opening is small but distinguishable in `eye_start.png`;
- it is clear in `eye_near.png`;
- the wall-to-goal and goal-to-far-wall regions are lit rather than black;
- Level 5 is visibly narrower than Level 0;
- there is no unexpected widening in the lateral direction.

## Current local verification status

On the development machine, without Isaac Sim:

- geometry/protocol suite: `35/35` passed;
- integration suite: `19/19` passed;
- persistence suite: `13/13` passed;
- Python compilation and `git diff --check`: passed;
- passability diagnostic: all Levels reachable; Level 4 is passable straight
  (0.000 m of aim tolerance) or turned at 45 degrees and above, and Level 5 needs
  at least 75 degrees of torso rotation (measured; 60 degrees is not enough).

Real USD authoring and rendered brightness still require the professor-machine run.

`verify_professor_machine.sh` runs all three offline suites, so the persistence
checks are exercised on the professor's machine too. A sweep must be started
inside `tmux` or `nohup`: it runs for days, and a dropped SSH session would kill
it. Each episode is written atomically as it is scored, so an interruption costs
at most the episode in flight, and `run_all_models.sh` resumes from the
checkpoint when re-run with the same command.
