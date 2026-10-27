# Stage 3 — body-size supplementary experiment

## Question

Stages 1 and 2 used one body, the Unitree H1 (1.806 m measured height, 0.570 m
analytic shoulder width).  Stage 3 asks whether the aperture behaviour survives
when only the body changes.

There is no old Stage 3 implementation to delete.  Historical notes used
"Stage 3" for the scene-skin work, but that is Stage 1 (`STAGE1_SCENES.md`).
The old repeated-attempt memory experiment is archived under
`_archive/stage2_legacy_20261025/`.

## Robots

| key | robot | official size | body height used | shoulder width used | asset |
|---|---|---|---|---|---|
| G1 | Unitree G1 | 1320 x 450 x 200 mm, 35 kg | 1.323 m | 0.450 m | `assets/G1/configuration/g1_base.usd` |
| R1 | Unitree R1 | 1230 x 357 x 190 mm, 27-29 kg | 1.230 m | 0.357 m | `assets/R1/configuration/r1_base.usd` |

Official sources:

- https://www.unitree.com/g1/
- https://www.unitree.com/R1/
- https://github.com/unitreerobotics/unitree_ros

The official URDFs are converted with `tools/convert_unitree_urdf.py`.  The base
payload is used directly; the separate sensor configuration is not part of the
kinematic body experiment.

## Held constant from Stage 1

- Same room, wall, obstacle location, success plane and 30-step limit.
- Same action set, 0.75 m translation, 15 degree torso turn, camera glances.
- Same five Stage 1 scenes and their 25 markers.
- Same prompt except the body name: "You are a Unitree G1/R1 humanoid robot".
- No A/S, width, shoulder width, narrow/wide or turn instruction in the prompt.
- No memory and no Step 0 digest.

## What changes

- Robot body geometry and USD.
- Camera height (G1 1.230 m, R1 1.144 m).
- Analytic shoulder width / torso thickness used by the collision gate.
- Channel width: `width = target_robot_shoulder_width * A/S`.

### Definition of shoulder width

Stage 3 uses the same definition as Stage 1: the manufacturer's official
standing body **width** (the Y dimension in the standing table), not the span
between the two shoulder joints.

| robot | official standing size | width used | thickness used |
|---|---|---|---:|---:|
| H1 | 1800 x 570 x 220 mm | 0.570 m | 0.220 m |
| G1 | 1320 x 450 x 200 mm | 0.450 m | 0.200 m |
| R1 | 1230 x 357 x 190 mm | 0.357 m | 0.190 m |

The URDF shoulder-pitch joints are about 0.200 m apart on G1 and 0.171 m on R1;
that is a joint-layout distance, not the body width that must clear the opening.
The converted URDF lateral bounding boxes (G1 0.371 m, R1 0.375 m) are the
arm-down mesh extents and also are not used as the A/S denominator.  This is a
deliberate consistency choice: Stage 1's 0.570 m H1 value is the official width,
so G1/R1 must use their official widths too.

## Selection

The five models are the Stage 1 overall pass-rate top five:

| rank | model | pass rate |
|---:|---|---:|
| 1 | deepseek-v4-flash-vision-exp | 99.765% |
| 2 | gemini-3-flash-preview | 98.824% |
| 3 | mimo-v2.6-pro | 86.118% |
| 4 | claude-haiku-4-5-20251001 | 68.471% |
| 5 | gpt-5 | 68.471% |

`claude` and `gpt-5` are tied at 291/425; both are included.  Gemini-2.5-flash
is sixth at 67.294% and is not in this set.

Final design (2026-10-27): 5 models x 2 bodies x 10 cells = 100 episodes.

Per model and per body:

```
A/S 1.3 x3
A/S 1.1 x3
A/S 1.0 x2
A/S 0.9 x2
= 10 cells
```

Every Stage 1 scene appears exactly twice per model and body.  G1 keeps its
existing five cells per model (22 records already on disk + 3 pending) and adds
five cells: A/S 1.3 at stage1.1/1.2/1.3, A/S 1.0 at stage1.4, A/S 1.1 at
stage1.5.  The five supplements use markers 2/4/3/2/5, so the final G1 set uses
all five marker slots.  R1 mirrors the final ten G1 conditions exactly: same
scene, A/S, marker and source episode; only the body and width change.

The frozen plan is `reports/stage3_final100_plan.csv`.

## Step length and distance

Stage 3 keeps the Stage 1 step length (0.75 m), start x (0.5 m), wall x (8.0 m)
and success plane (8.75 m).  It does not rescale the room.

Reason: the experiment changes one variable, shoulder width.  Scaling step
length with height would also change the locomotion control and the number of
steps; scaling the room as well would change the scene.  Either would make a
difference in A/S impossible to separate from a difference in layout.

The smaller robots therefore take a larger step relative to their height.  That
is a deliberate fixed-control choice, not a claim that 0.75 m is the G1/R1
walking gait.  The camera height and the body collision rectangle do change,
because those are the body variables under study.

A secondary scaled-layout sensitivity is possible later, but is not part of the
50-cell primary design.

## Code

- `robot_profiles.py`: body geometry, camera anchor, USD path.
- `stage3_config.py`: selected models, robots, A/S range.
- `tools/build_stage3_plan.py`: builds the frozen 50-cell source plan.
- `stage3_runner.py`: runs one source scene per process.
- `tools/convert_unitree_urdf.py`: URDF -> USD conversion.

Run one scene:

```bash
/home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 --scene stage1.1 --headless
```

Run one planned cell:

```bash
/home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 --scene stage1.1 \
  --model deepseek-v4-flash-vision-exp --as 1.1 --limit 1 --headless
```

## Smoke status

The G1 and R1 asset paths load and complete a Stage 3 episode at A/S 1.0:

| robot | measured height | shoulder | A/S | result |
|---|---:|---:|---:|---|
| G1 | 1.323 m | 0.450 m | 1.00 | passed, 17 steps, 90 degrees |
| R1 | 1.230 m | 0.357 m | 1.00 | passed, 17 steps, 90 degrees |

These are pipeline smoke tests, not the 50-cell result.
