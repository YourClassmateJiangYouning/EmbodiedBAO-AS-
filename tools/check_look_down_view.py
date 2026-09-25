"""Does look_down actually show the agent its own body?

The action exists because the agent has no proprioceptive body schema and the eye
sits at body centre pitched only 15 degrees down (see environment.CAMERA_PITCH_
STEP_DEG).  Geometry says the hands hang about 70 degrees below the eye while the
frame's lower edge is at 53, so the body is probably still out of frame -- but
"probably" is not good enough for an action that costs one of 30 steps.

Method: capture the look_down frame twice in ONE Isaac session, once with the
robot's own prims active and once with them deactivated, and compare the frames.
If they are identical the agent sees no part of itself and the action is inert; if
the lower band changes, it does see itself.  Saving both PNGs lets the difference
be eyeballed as well as measured.

    /home/ybh/isaacsim/python.sh tools/check_look_down_view.py
    /home/ybh/isaacsim/python.sh tools/check_look_down_view.py --level 10 --x 6.0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=10, help="Ladder Level to build")
    parser.add_argument(
        "--x", type=float, default=6.0, help="Robot x position for the glance"
    )
    parser.add_argument("--outdir", type=str, default="look_down_check")
    args = parser.parse_args()

    # SimulationApp FIRST, then environment.  The other order runs
    # isaacsim.core.api before the app exists, which latches
    # environment._HAS_ISAAC_SIM to False forever: the app started, the scene was
    # never built, the script did nothing and exited zero.  That is exactly what
    # the first version of this tool did.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import environment as env_module

        channel = env_module.level_channel_width(args.level)
        if args.level not in env_module.LEVEL_CHANNEL_WIDTHS:
            print(
                f"warning: level {args.level} is not in the ladder, "
                f"using its width anyway"
            )
        env = env_module.setup_scene(
            app,
            task_dict={"headless": True, "channel_width": channel, "hide_robot": False},
        )
        env.reset_scene()
        env._set_robot_pose(np.array([args.x, 0.0, 0.0], dtype=float), 0.0)

        # 1. the normal forward view, for reference
        forward = np.asarray(env.get_camera_image())
        # 2. the glance with the robot visible
        visible = np.asarray(env.execute_action("look_down").rgb)

        # 3. the same glance with the robot's own prims switched off.  Wrapped:
        #    deactivating the robot can invalidate whatever execute_action touches,
        #    and losing the whole report to that would be worse than losing the
        #    comparison -- the visible frame alone answers the question by eye.
        hidden = None
        hide_note = ""
        try:
            hidden_paths = []
            for prim in env.stage.Traverse():
                if str(prim.GetPath()).startswith(env.robot_prim_path):
                    hidden_paths.append(str(prim.GetPath()))
                    try:
                        prim.SetActive(False)
                    except Exception as exc:
                        hide_note = f"{type(exc).__name__}: {exc}"
            print(
                f"[look_down] deactivated {len(hidden_paths)} robot prim(s): "
                f"{', '.join(hidden_paths[:5])}"
                + (" ..." if len(hidden_paths) > 5 else "")
            )
            hidden = np.asarray(env.execute_action("look_down").rgb)
        except Exception as exc:
            hide_note = f"{type(exc).__name__}: {exc}"

        # Everything below runs BEFORE app.close(): printing after shutdown proved
        # fragile, and the first run of this tool lost its whole report that way.
        report(args, forward=forward, visible=visible, hidden=hidden, note=hide_note)
    finally:
        app.close()
    return 0


def report(args, forward, visible, hidden, note: str) -> None:
    """Save the frames and print the comparison.  Runs before app.close()."""
    if visible.ndim == 3 and visible.shape[2] == 4:
        visible = visible[:, :, :3]
    if forward.ndim == 3 and forward.shape[2] == 4:
        forward = forward[:, :, :3]
    if hidden is not None and hidden.ndim == 3 and hidden.shape[2] == 4:
        hidden = hidden[:, :, :3]

    os.makedirs(args.outdir, exist_ok=True)
    from PIL import Image

    Image.fromarray(np.asarray(forward, dtype=np.uint8)).save(
        os.path.join(args.outdir, "forward.png")
    )
    Image.fromarray(np.asarray(visible, dtype=np.uint8)).save(
        os.path.join(args.outdir, "look_down_visible.png")
    )
    if hidden is not None:
        Image.fromarray(np.asarray(hidden, dtype=np.uint8)).save(
            os.path.join(args.outdir, "look_down_hidden.png")
        )

    height = int(visible.shape[0])
    third = max(1, height // 3)
    print()
    print(
        f"[look_down] glance frame {visible.shape[1]}x{height}; forward frame mean "
        f"brightness {forward.mean():.1f}, glance frame {visible.mean():.1f}"
    )
    print(
        f"[look_down] forward vs glance differ by "
        f"{np.abs(forward.astype(int) - visible.astype(int)).mean():.2f} on average "
        f"(must be large: the glance has to change the view)"
    )
    if hidden is None:
        print(
            f"[look_down] could not capture the robot-hidden frame ({note}), so the "
            f"comparison is skipped.  look_down_visible.png is written: judge it by "
            f"eye."
        )
        print(f"[look_down] frames in {os.path.abspath(args.outdir)}/")
        return

    diff = np.abs(visible.astype(int) - hidden.astype(int)).max(axis=2)
    print(f"[look_down] mean |visible - hidden| = {diff.mean():.2f}, "
          f"pixels differing by >10: {100.0 * (diff > 10).mean():.2f}%")

    # Structure, to tell a readable silhouette from a close-up blur.  Pixels that
    # change when the robot is hidden ARE the robot; the rest is the scene behind
    # it.  If the scene part has structure, the agent is looking at its own body
    # against a visible room -- which is what makes the glance usable as a scale
    # reference.  If the scene part is flat, the body fills the frame at point-blank
    # range and the glance shows nothing comparable to the opening.
    body = diff > 10
    print(
        f"[look_down] structure (pixel std): forward {forward.std():.1f}, "
        f"glance {visible.std():.1f}"
    )
    if body.any():
        print(
            f"[look_down] body pixels are {100.0 * body.mean():.1f}% of the frame, "
            f"std {visible[body].std():.1f}"
        )
    if (~body).any():
        scene_std = float(visible[~body].std())
        print(
            f"[look_down] the rest of the frame (scene behind you) is "
            f"{100.0 * (~body).mean():.1f}% of pixels, std {scene_std:.1f}"
        )
        print(
            f"[look_down] readability: "
            + (
                "the scene is visible alongside the body, so sizes could be "
                "compared"
                if scene_std > 8.0
                else "the scene part is nearly flat, so this is a close-up of your "
                "own surface with no room context -- looking at the PNG is the "
                "decisive check"
            )
        )
    print()
    print(f"{'band':>8} {'mean diff':>10} {'% pixels >10':>13}")
    for label, sl in (
        ("upper", slice(0, third)),
        ("middle", slice(third, 2 * third)),
        ("lower", slice(2 * third, height)),
    ):
        band = diff[sl]
        print(f"{label:>8} {band.mean():>10.2f} {100.0 * (band > 10).mean():>12.2f}%")
    print()
    lower_change = 100.0 * (diff[2 * third :] > 10).mean()
    if lower_change > 1.0:
        print(
            f"VERDICT: the robot's own body IS in the look_down frame "
            f"({lower_change:.2f}% of the lower band changes when it is hidden). "
            f"The glance is usable."
        )
    else:
        print(
            "VERDICT: the look_down frame is the same with and without the robot "
            "prims, so the agent sees NO part of its own body in it. The glance "
            "shows only the floor, and cannot serve as a body-size reference: "
            "either pitch further down / move the eye back, or drop the action and "
            "the prompt sentence that promises it."
        )
    print(f"Frames written to {os.path.abspath(args.outdir)}/ for eyeballing.")


if __name__ == "__main__":
    raise SystemExit(main())
