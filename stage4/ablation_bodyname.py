"""Ablation: is DeepSeek's weak turning in the real corridor caused by what it is told it is?

DeepSeek turns 39.5 degrees at wide gaps in the simulator but 0-9 degrees in the real scene, and the only
differences between the two prompts are the action menu (5 vs 9), the body name and the images.  The body name
is the one that plausibly changes the prior: "a Unitree H1 humanoid robot" invites rotating the shoulders,
"a small wheeled robot" invites driving straight through.

So: replay the three baseline cells that produced 0/3 turns, with the body named as the simulator named it.
If it starts turning, the naming explains the divergence.  If it still drives straight, the difference is the
images.

    python stage4/ablation_bodyname.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import core  # noqa: E402
import memory_runner as mem  # noqa: E402
import replay_runner as runner  # noqa: E402

OUT = os.path.join(HERE, "sessions_b")
HUMANOID = "a Unitree H1 humanoid robot"
CELLS = [("deepseek-v4-flash-vision-exp", "M2"), ("deepseek-v4-flash-vision-exp", "M4"),
         ("deepseek-v4-flash-vision-exp", "M5")]


def main() -> int:
    for model, marker in CELLS:
        session_id = "abl-h1-baseline-deepseek-%s-test2" % marker
        session = core.Stage4Session(sessions_root=OUT)
        session.open_session(session_id, {
            "scene": "R1", "marker": marker, "marker_descriptor": mem.descriptor_for(marker),
            "mode": "A", "model": model, "a_s": mem.TEST_A_S,
            "max_steps": runner.MAX_STEPS, "disable_thinking": True,
            "shoulder_m": runner.S, "move_step": runner.STEP,
            "body_name": HUMANOID,
        })
        session.start_episode()
        print("[%s] body=%r" % (marker, session.config.get("body_name")))

        x, theta = runner.START_X, 0.0
        for step in range(runner.MAX_STEPS):
            session.capture_from_bank(runner.BANK)
            session.call_model()
            record = session.history[-1]
            action = record.get("action")
            if action is None:
                x, theta, feedback, collision = x, theta, runner.FEEDBACK_INVALID, False
            else:
                x, theta, feedback, blocked, collision = runner.apply_action(
                    action, x, theta, mem.TEST_A_S)
                session.position_x, session.torso_rotation = x, theta
            session.action_executed(feedback=feedback, collision=collision, position_x=x,
                                    torso_rotation=theta, note="ablation-bodyname", advance_odometer=False)
            if x >= runner.SUCCESS_X - 1e-9 or session.episode is None:
                break
        if session.episode is not None:
            session.finish(passed=x >= runner.SUCCESS_X - 1e-9,
                           end_reason="passed" if x >= runner.SUCCESS_X - 1e-9 else "max_steps")

        turns = sum(1 for row in session.history if (row.get("action") or "").startswith("turn"))
        rotations = [abs(float(row.get("torso_rotation") or 0)) for row in session.history] or [0.0]
        folded = [min(r % 360, 360 - r % 360) for r in rotations]
        print("   turns=%d  max_rotation=%.0f  steps=%d"
              % (turns, max(folded), len(session.history)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
