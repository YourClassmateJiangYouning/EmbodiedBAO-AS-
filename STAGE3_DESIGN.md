# Stage 3 — body-size supplementary experiment

## What changed

Stages 1 and 2 used a single body, the Unitree H1.  Its measured height was
1.806 m and its analytic shoulder width was 0.570 m.  That leaves a substantive
limitation: the observed A/S threshold could be a property of the H1 rather than
of body-scale affordance perception.

Stage 3 repeats the same body-as-obstacle setting with two smaller Unitree
humanoids and tests whether the A/S threshold stays in the same place.

There was no committed old Stage 3 implementation.  Historical notes used
"Stage 3" for the scene-skin work, but that work is Stage 1 (`STAGE1_SCENES.md`).
The old repeated-attempt memory experiment is archived under
`_archive/stage2_legacy_20261025/`.  This document defines the new Stage 3.

## Robots

| key | robot | official standing size | body height used | shoulder width used | source |
|---|---|---|---|---|---|
| G1 | Unitree G1 | 1320 x 450 x 200 mm, 35 kg | 1.323 m (URDF bbox) | 0.450 m | official G1 page, Unitree ROS g1_description |
| R1 | Unitree R1 | 1230 x 357 x 190 mm, 27-29 kg | 1.230 m (URDF bbox) | 0.357 m | official R1 page, Unitree ROS r1_description |

References:

- https://www.unitree.com/g1/
- https://www.unitree.com/R1/
- https://github.com/unitreerobotics/unitree_ros

Assets are converted from the official 23-DOF G1 URDF and the R1 URDF with
`tools/convert_unitree_urdf.py` into `assets/G1/configuration/g1_base.usd` and
`assets/R1/configuration/r1_base.usd`.  The base payload is used directly; the
separate sensor configuration is not part of the kinematic body experiment.

## Held constant from Stage 1

- Same room, wall, channel, success plane and 30-step budget.
- Same action menu, 0.75 m translation, 15 degree torso turn and camera glances.
- Same Stage 1.1 scene and marker selection.
- Same prompt except the robot name: "You are a Unitree G1/R1 humanoid robot".
- No A/S, level, width, shoulder width, narrow/wide or turn instruction appears
  in the prompt.
- No Stage 2 memory and no Step 0 digest.

## Design

```
5 models x 5 A/S values x 2 robots = 50 episodes
A/S = 1.10, 1.05, 1.00, 0.95, 0.90
```

Models are the Stage 1 top five by Sfinal/NPS:

1. gemini-3-flash-preview
2. deepseek-v4-flash-vision-exp
3. mimo-v2.6-pro
4. claude-haiku-4-5-20251001
5. gemini-2.5-flash

A/S is channel width divided by that robot's shoulder width.  The same scene and
marker are used across robots; one episode per cell.

## Code

- `robot_profiles.py`: body geometry, camera anchor, USD path.
- `stage3_config.py`: models, robots, A/S values.
- `stage3_runner.py`: one process per robot; explicit channel width per cell.
- `tools/convert_unitree_urdf.py`: offline URDF -> USD conversion.

Run:

```bash
/home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 --headless
/home/ybh/isaacsim/python.sh stage3_runner.py --robot R1 --headless
```

Smoke one cell:

```bash
/home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 \
  --model deepseek-v4-flash-vision-exp --as 1.0 --limit 1 --headless
```

## Smoke status

Both bodies load and complete a Stage 3 episode at A/S 1.0:

| robot | height | width | A/S | result |
|---|---:|---:|---:|---|
| G1 | 1.323 m | 0.450 m | 1.00 | passed, 17 steps, 90 degrees |
| R1 | 1.230 m | 0.357 m | 1.00 | passed, 17 steps, 90 degrees |

These are pipeline smoke tests, not the 50-cell result.

## Interpretation

If the A/S threshold is body-independent, the G1 and R1 rotation-onset curves
should fall near the same normalized A/S values as H1.  If the curves shift with
body size, the threshold is not a body-scale invariant.  The 50-cell design is
deliberately narrow (1.1-0.9) because that is where the H1 threshold sits.
