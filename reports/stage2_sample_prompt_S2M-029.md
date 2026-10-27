# Stage 2 sample prompt — S2M-029 (DeepSeek)

## Step 0: digest

```text
You are a Unitree H1 humanoid robot. Your task is to reach the red marker on the far wall.

Available actions (each action is one discrete step):
{"action": "forward"} - walk 75cm straight ahead, in your walking direction, which always takes you toward the red marker on the far wall. Turning your torso does not change where forward takes you
{"action": "backward"} - walk 75cm backwards, away from the red marker on the far wall: the same walking direction as forward, reversed
{"action": "left"} - sidestep 75cm to your left, off the line you are walking along, without changing your walking direction
{"action": "right"} - sidestep 75cm to your right, off the line you are walking along, without changing your walking direction
{"action": "turn_left"} - rotate your torso 15 degrees to the left. This does NOT change your walking direction, so forward still takes you toward the red marker, and it does NOT change what you can see: your eyes keep looking straight ahead along your walking direction. It changes how wide your body is. Rotating needs room, so you cannot turn once your shoulders are inside the opening
{"action": "turn_right"} - rotate your torso 15 degrees to the right. This does NOT change your walking direction, so forward still takes you toward the red marker, and it does NOT change what you can see: your eyes keep looking straight ahead along your walking direction. It changes how wide your body is. Rotating needs room, so you cannot turn once your shoulders are inside the opening
{"action": "look_left"} - glance to the left with your head camera, 30 degrees off your walking direction: you see that view now, and your gaze returns to straight ahead as soon as you do anything else. It does not accumulate, so looking twice is not looking 60 degrees. Your body and your walking direction are unaffected
{"action": "look_right"} - glance to the right with your head camera, 30 degrees off your walking direction: you see that view now, and your gaze returns to straight ahead as soon as you do anything else. It does not accumulate, so looking twice is not looking 60 degrees. Your body and your walking direction are unaffected
{"action": "look_down"} - glance downwards with your head camera, 45 degrees below your usual view: you see the floor in front of you and your own body. Your gaze returns to its usual direction as soon as you do anything else. Use it to see how wide you are, which you cannot otherwise see. Your torso and your walking direction are unaffected

Your walking direction is fixed: it always points straight ahead at the far wall, which is where the red marker you are heading for is. forward/backward/left/right are defined relative to that walking direction, so they behave the same way however your torso is turned.
turn_left/turn_right rotate your torso 15 degrees relative to that walking direction. The direction you walk does not change, and neither does your view: your eyes look straight ahead along your walking direction whatever your torso is doing. Turning changes how your body is oriented; it does not steer you and it does not change what you see.
look_left/look_right are a single glance 30 degrees to the left or right of your walking direction, and look_down is a single glance 45 degrees below it. Your torso and your walking direction are unaffected, each glance clears as soon as you do anything else, and glances do not accumulate: two glances in the same direction do not add up. You cannot look further to the side than 30 degrees, and you cannot look up.

A previous successful traversal by the same model in this same scene and marker:

Scene: stage1.5
Actions: forward, forward, forward, turn_left, turn_left, turn_left, turn_left, turn_left, turn_left, forward, forward, forward, forward, forward, forward, forward, forward
Steps:
- step 0: forward -> executed | reasoning: The robot is aligned with the pink hexagon, and the midline path ahead is unobstructed. Moving forward reduces distance to the target directly.
- step 1: forward -> executed | reasoning: The robot is aligned with the pink target and the central path is clear. Continuing forward directly reduces the distance to the pink hexagon without needing to steer or sidestep.
- step 2: forward -> executed | reasoning: The pink hexagon is visible through the doorway straight ahead along the fixed walking direction, and the central floor path is unobstructed. Continuing forward directly reduces the distance to the target; no sidestep or turn is needed yet.
- step 3: turn_left -> executed | reasoning: The pink hexagon is centered along the fixed walking direction, so no sidestep is needed. However, the opening ahead appears narrow, and the robot's torso is currently broadside. Since turning is only possible before the shoulders enter the opening, it is safer to begin rotating the torso now while still in open space; the walking direction and view remain unchanged.
- step 4: turn_left -> executed | reasoning: The doorway ahead is narrow, and the torso is only rotated 15 degrees, so the body may still be too wide to pass safely. Since rotation must happen before the shoulders enter the opening, continue turning left toward edge-on alignment while there is still room, without changing the fixed forward walking direction.
- step 5: turn_left -> executed | reasoning: The body is still only rotated 30 degrees from the walking direction, so its shoulders remain broadside to the narrow opening. Turning must happen before the shoulders enter the doorway, so continue rotating toward edge-on (90 degrees) to fit through, then proceed forward.
- step 6: turn_left -> executed | reasoning: The torso is currently rotated 45 degrees from the walking direction, but the doorway ahead is narrow. Since turning is only possible before the shoulders enter the opening, continue rotating left toward edge-on alignment so the robot can pass through without catching its shoulders.
- step 7: turn_left -> executed | reasoning: The target is straight ahead along the fixed walking direction, but the opening appears narrow. Turning must be completed before the shoulders enter the opening, so continue rotating left toward 90 degrees to minimize body width and safely pass through.
- step 8: turn_left -> executed | reasoning: The torso is rotated 75 degrees, close to edge-on for the narrow doorway. One more turn_left reaches 90 degrees and minimizes shoulder width before entering the opening, so turning now is safest before moving forward.
- step 9: forward -> executed | reasoning: The torso is already at the safest 90-degree edge-on orientation for the narrow opening, and the pink hexagon remains aligned straight ahead. Moving forward now advances directly toward the target without changing the careful alignment.
- step 10: forward -> executed | reasoning: The robot is already rotated 90 degrees edge-on for the narrow opening, and the pink hexagon remains aligned straight ahead. Continuing forward advances directly toward the target without needing a sidestep or further turns.
- step 11: forward -> executed | reasoning: The target remains aligned straight ahead, and the torso is already rotated 90 degrees edge-on for the narrow opening. Continuing forward advances directly toward the pink hexagon without needing further turning or sidestepping.
- step 12: forward -> executed | reasoning: The robot is oriented 90 degrees edge-on to fit through the narrow doorway and the pink hexagon remains directly ahead along the fixed walking direction. The path forward is clear, so continuing forward will bring the robot closer to the target.
- step 13: forward -> executed | reasoning: The torso is already rotated 90 degrees edge-on to fit through the narrow opening, and the pink hexagon remains aligned straight ahead. Continuing forward advances directly toward the target without needing further turns or sidesteps.
- step 14: forward -> executed | reasoning: The torso is already rotated 90 degrees edge-on for the narrow opening, and the pink hexagon remains aligned straight ahead along the fixed walking direction. Continuing forward advances directly toward the target without needing further turns or sidesteps.
- step 15: forward -> executed | reasoning: The robot is already rotated edge-on at 90 degrees for the narrow opening, and the pink hexagon remains straight ahead along the fixed walking direction. Continuing forward directly reduces the remaining distance to the target without needing turns or sidesteps.
- step 16: forward -> executed | reasoning: The pink hexagon remains directly ahead along the fixed walking direction, and the torso is already turned 90 degrees edge-on for any narrow opening. Continuing forward is the most direct way to reduce the remaining distance to the target.

Current state:
- position (x, y, z): [0.500, 0.000, 0.000]
  x is your forward position along the walkway, and changes only when you use forward/backward; y is always 0; z is how far to the right (negative) or left (positive) of the line you started on you are, and changes only when you use left/right
- torso rotation (degrees): 0.0
- head camera yaw offset from straight ahead (degrees): 0.0 (0 = looking straight ahead down your walking direction; NEGATIVE is toward your left and POSITIVE toward your right -- a look_left reports a negative number -- and the glance clears on your next action)
- step limit for this episode: 30

Step 0 (digest; do NOT choose an action). Read the prior successful experience and the current view. Return exactly one JSON object: {"digested": true, "memory_summary": "one or two sentences", "plan": "how you will approach this scene", "cues": ["what you will look for"]}
```

## Step 1: formal action test

```text
You are a Unitree H1 humanoid robot. Your task is to reach the red marker on the far wall.

Available actions (each action is one discrete step):
{"action": "forward"} - walk 75cm straight ahead, in your walking direction, which always takes you toward the red marker on the far wall. Turning your torso does not change where forward takes you
{"action": "backward"} - walk 75cm backwards, away from the red marker on the far wall: the same walking direction as forward, reversed
{"action": "left"} - sidestep 75cm to your left, off the line you are walking along, without changing your walking direction
{"action": "right"} - sidestep 75cm to your right, off the line you are walking along, without changing your walking direction
{"action": "turn_left"} - rotate your torso 15 degrees to the left. This does NOT change your walking direction, so forward still takes you toward the red marker, and it does NOT change what you can see: your eyes keep looking straight ahead along your walking direction. It changes how wide your body is. Rotating needs room, so you cannot turn once your shoulders are inside the opening
{"action": "turn_right"} - rotate your torso 15 degrees to the right. This does NOT change your walking direction, so forward still takes you toward the red marker, and it does NOT change what you can see: your eyes keep looking straight ahead along your walking direction. It changes how wide your body is. Rotating needs room, so you cannot turn once your shoulders are inside the opening
{"action": "look_left"} - glance to the left with your head camera, 30 degrees off your walking direction: you see that view now, and your gaze returns to straight ahead as soon as you do anything else. It does not accumulate, so looking twice is not looking 60 degrees. Your body and your walking direction are unaffected
{"action": "look_right"} - glance to the right with your head camera, 30 degrees off your walking direction: you see that view now, and your gaze returns to straight ahead as soon as you do anything else. It does not accumulate, so looking twice is not looking 60 degrees. Your body and your walking direction are unaffected
{"action": "look_down"} - glance downwards with your head camera, 45 degrees below your usual view: you see the floor in front of you and your own body. Your gaze returns to its usual direction as soon as you do anything else. Use it to see how wide you are, which you cannot otherwise see. Your torso and your walking direction are unaffected

Your walking direction is fixed: it always points straight ahead at the far wall, which is where the red marker you are heading for is. forward/backward/left/right are defined relative to that walking direction, so they behave the same way however your torso is turned.
turn_left/turn_right rotate your torso 15 degrees relative to that walking direction. The direction you walk does not change, and neither does your view: your eyes look straight ahead along your walking direction whatever your torso is doing. Turning changes how your body is oriented; it does not steer you and it does not change what you see.
look_left/look_right are a single glance 30 degrees to the left or right of your walking direction, and look_down is a single glance 45 degrees below it. Your torso and your walking direction are unaffected, each glance clears as soon as you do anything else, and glances do not accumulate: two glances in the same direction do not add up. You cannot look further to the side than 30 degrees, and you cannot look up.

A previous successful traversal by the same model in this same scene and marker:

Scene: stage1.5
Actions: forward, forward, forward, turn_left, turn_left, turn_left, turn_left, turn_left, turn_left, forward, forward, forward, forward, forward, forward, forward, forward
Steps:
- step 0: forward -> executed | reasoning: The robot is aligned with the pink hexagon, and the midline path ahead is unobstructed. Moving forward reduces distance to the target directly.
- step 1: forward -> executed | reasoning: The robot is aligned with the pink target and the central path is clear. Continuing forward directly reduces the distance to the pink hexagon without needing to steer or sidestep.
- step 2: forward -> executed | reasoning: The pink hexagon is visible through the doorway straight ahead along the fixed walking direction, and the central floor path is unobstructed. Continuing forward directly reduces the distance to the target; no sidestep or turn is needed yet.
- step 3: turn_left -> executed | reasoning: The pink hexagon is centered along the fixed walking direction, so no sidestep is needed. However, the opening ahead appears narrow, and the robot's torso is currently broadside. Since turning is only possible before the shoulders enter the opening, it is safer to begin rotating the torso now while still in open space; the walking direction and view remain unchanged.
- step 4: turn_left -> executed | reasoning: The doorway ahead is narrow, and the torso is only rotated 15 degrees, so the body may still be too wide to pass safely. Since rotation must happen before the shoulders enter the opening, continue turning left toward edge-on alignment while there is still room, without changing the fixed forward walking direction.
- step 5: turn_left -> executed | reasoning: The body is still only rotated 30 degrees from the walking direction, so its shoulders remain broadside to the narrow opening. Turning must happen before the shoulders enter the doorway, so continue rotating toward edge-on (90 degrees) to fit through, then proceed forward.
- step 6: turn_left -> executed | reasoning: The torso is currently rotated 45 degrees from the walking direction, but the doorway ahead is narrow. Since turning is only possible before the shoulders enter the opening, continue rotating left toward edge-on alignment so the robot can pass through without catching its shoulders.
- step 7: turn_left -> executed | reasoning: The target is straight ahead along the fixed walking direction, but the opening appears narrow. Turning must be completed before the shoulders enter the opening, so continue rotating left toward 90 degrees to minimize body width and safely pass through.
- step 8: turn_left -> executed | reasoning: The torso is rotated 75 degrees, close to edge-on for the narrow doorway. One more turn_left reaches 90 degrees and minimizes shoulder width before entering the opening, so turning now is safest before moving forward.
- step 9: forward -> executed | reasoning: The torso is already at the safest 90-degree edge-on orientation for the narrow opening, and the pink hexagon remains aligned straight ahead. Moving forward now advances directly toward the target without changing the careful alignment.
- step 10: forward -> executed | reasoning: The robot is already rotated 90 degrees edge-on for the narrow opening, and the pink hexagon remains aligned straight ahead. Continuing forward advances directly toward the target without needing a sidestep or further turns.
- step 11: forward -> executed | reasoning: The target remains aligned straight ahead, and the torso is already rotated 90 degrees edge-on for the narrow opening. Continuing forward advances directly toward the pink hexagon without needing further turning or sidestepping.
- step 12: forward -> executed | reasoning: The robot is oriented 90 degrees edge-on to fit through the narrow doorway and the pink hexagon remains directly ahead along the fixed walking direction. The path forward is clear, so continuing forward will bring the robot closer to the target.
- step 13: forward -> executed | reasoning: The torso is already rotated 90 degrees edge-on to fit through the narrow opening, and the pink hexagon remains aligned straight ahead. Continuing forward advances directly toward the target without needing further turns or sidesteps.
- step 14: forward -> executed | reasoning: The torso is already rotated 90 degrees edge-on for the narrow opening, and the pink hexagon remains aligned straight ahead along the fixed walking direction. Continuing forward advances directly toward the target without needing further turns or sidesteps.
- step 15: forward -> executed | reasoning: The robot is already rotated edge-on at 90 degrees for the narrow opening, and the pink hexagon remains straight ahead along the fixed walking direction. Continuing forward directly reduces the remaining distance to the target without needing turns or sidesteps.
- step 16: forward -> executed | reasoning: The pink hexagon remains directly ahead along the fixed walking direction, and the torso is already turned 90 degrees edge-on for any narrow opening. Continuing forward is the most direct way to reduce the remaining distance to the target.

Your reflection on that experience:

digested

Current state:
- position (x, y, z): [0.500, 0.000, 0.000]
  x is your forward position along the walkway, and changes only when you use forward/backward; y is always 0; z is how far to the right (negative) or left (positive) of the line you started on you are, and changes only when you use left/right
- torso rotation (degrees): 0.0
- head camera yaw offset from straight ahead (degrees): 0.0 (0 = looking straight ahead down your walking direction; NEGATIVE is toward your left and POSITIVE toward your right -- a look_left reports a negative number -- and the glance clears on your next action)
- step limit for this episode: 30

Reply with exactly one JSON object:
{"scene_description": "<what the current camera image shows>", "reasoning": "<brief reasoning>", "action": "<action>", "confidence": 0.0}
```
