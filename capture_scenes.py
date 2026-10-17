"""Render one Stage 1 scene so it can be looked at.

Why this exists: the scene design is a pile of material URLs, marker polygons and dressing
coordinates, and the only way to know whether it looks like a supermarket rather than a
slightly differently painted laboratory is to render it.  This is also how the marker
uniqueness claim gets eyeballed before the 6,375-episode sweep is trusted to it.

One scene per process, because SimulationApp is a singleton -- the same reason
capture_views.py takes one level at a time.  Run it five times, or use the loop in the
docstring.

Output, under ``--outdir``:

* ``<scene>_slot<N>_eye.png``  what the model sees at the start pose.  Rendered at
  ``--width`` (1024 by default); the pipeline downscales to ``BAO_IMAGE_SIZE`` (512) before
  sending, so this file is twice the model's resolution and twice as noisy.
* ``<scene>_slot<N>_iso.png``  an external view, for judging the room rather than the task
* ``<scene>_sheet_eye.png``     the five markers side by side, as the model sees them
* ``<scene>_sheet_iso.png``     the same five, from the external camera

Use ``--level 0`` to judge materials and dressing: it is the widest opening (1.140 m, about
24 px at the start pose).  The default ``--level 10`` is the flush Level, whose opening is
0.570 m and only about 12 px there, which is too narrow a slot to see a room through.

Images are written with a small PNG encoder rather than Pillow, so that this runs under any
interpreter.  Measured on the workstation: its system ``python3`` has no Pillow at all,
while the interpreter that runs the sweeps (``/home/ybh/isaacsim/python.sh``) has Pillow
12.3.0.  Anything that must work under both therefore cannot rely on Pillow -- this file,
and ai_agent.encode_image, whose Pillow import is unconditional and is the reason the
sweeps have to run under the Isaac interpreter rather than under python3.

Usage:
    ISAAC_PY=/home/ybh/isaacsim/python.sh
    for s in stage1.1 stage1.2 stage1.3 stage1.4 stage1.5; do
      $ISAAC_PY capture_scenes.py --scene $s --level 0 --outdir preview --slots 1,2,3,4,5; done
"""

from __future__ import annotations

from typing import Any

import argparse
import math
import os
import struct
import traceback
import zlib

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one Stage 1 scene.")
    # Not derived from scenes.py here: importing it before SimulationApp is the trap
    # capture_views.py documents, so the name is validated after the app is up.
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--level", type=int, default=10,
                        help="A/S level whose opening to build (10 is A/S 1.00)")
    parser.add_argument("--slots", type=str, default="1,2,3,4,5")
    parser.add_argument("--outdir", type=str, default="scene_preview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--iso_x", type=float, default=-2.0,
                        help="where the external camera stands along x; negative is behind the "
                             "agent's start, outside the open end of the corridor")
    parser.add_argument("--iso_height", type=float, default=2.2,
                        help="external camera height.  Keep it under the 3 m ceiling, or the "
                             "ceiling is the only thing in the picture")
    parser.add_argument("--iso_z", type=float, default=1.2,
                        help="external camera offset sideways.  Keep it inside the 2.5 m half "
                             "width, or a side wall blocks the view")
    parser.add_argument(
        "--hide_robot", action="store_true",
        help="Hide the robot, using the environment's own hide_robot flag.  The lab's frames "
             "are unchanged by every scene fix, which means whatever dominates them does not "
             "depend on the scene; the robot's own body sitting in front of the eye camera is "
             "the first thing to rule out, and this rules it in or out in one render.",
    )
    parser.add_argument(
        "--parts", type=str, default="materials,marker,dressing",
        help="Which parts of the scene to apply.  Bisecting a wrong-looking frame is two "
             "renders with this: 'marker' alone shows the task geometry with nothing added, "
             "then add materials, then dressing, and the step where the view breaks is the "
             "part responsible.  Guessing at it from a finished frame wasted several rounds.",
    )
    parser.add_argument(
        "--flat_paint", action="store_true",
        help="Kept for compatibility; flat paint is now the DEFAULT.  See --mdl to opt in to "
             "materials, and the note below about which default is the safe one.",
    )
    parser.add_argument(
        "--mdl", action="store_true",
        help="Use the scene's MDL/材質 materials instead of flat paint.  OFF by default, and that "
             "default is deliberate: on the lab machine the material library's cache is not "
             "writable, so MDL prims are created without their shaders and every surface that "
             "receives one renders as salt-and-pepper noise -- a preview then looks broken and "
             "cannot be read at all.  This flag used to be inverted (materials unless --flat_paint "
             "was passed), which meant the noise was the default and several previews were "
             "misjudged because of it.",
    )
    parser.add_argument(
        "--iso_target", type=float, nargs=3, default=(10.0, 1.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World point the external camera looks at, default the far end of the corridor. "
             "The eye camera cannot see the obstacle-wall dressing at all: it looks through a "
             "0.570 m opening while those items are required to sit at |z| >= 0.9, so they are "
             "behind the wall panels from that pose.  Verified for the supermarket's shelf goods, "
             "which project 8-16 px into the eye frame at x 612..691 and measure wall-grey there -- "
             "the nearest pixel to the authored yellow was 154,161,171.  Checking them needs a "
             "close view aimed at them: --iso_x 9.4 --iso_height 1.4 --iso_z 3.4 "
             "--iso_target 7.99 1.05 1.6.",
    )
    parser.add_argument(
        "--iso_orbit", type=float, nargs="*", default=None,
        help="Extra camera angles, in degrees around --iso_target, rendered in addition to the "
             "plain iso view and written as <scene>_slot<N>_orbit<angle>.png.  One camera cannot "
             "show this room: the dressing sits at |z| >= 1.5 while the eye looks through a 0.570 m "
             "opening, so the side spaces hold every tree, bench and shrub the eye camera can never "
             "see.  Orbiting samples each side in one run instead of one guessed viewpoint per "
             "round trip, which is what several rounds were wasted on.",
    )
    parser.add_argument(
        "--iso_orbit_radius", type=float, default=9.0,
        help="Distance from --iso_target for the orbit views.  Large enough to stand outside the "
             "5 m wide room, so the orbit does not put the camera inside a wall.",
    )
    parser.add_argument(
        "--iso_orbit_height", type=float, default=2.2,
        help="Orbit camera height, as for --iso_height: under the 3 m ceiling or the ceiling is "
             "all there is to see.",
    )
    parser.add_argument(
        "--stage_tree", action="store_true",
        help="Print the prim tree under /World/Dressing with each prim's active state, inherited "
             "visibility and any displayColor or bound material, then exit without rendering.  Text "
             "only: it settles whether a prop is on the stage and visible, which a frame full of "
             "lights and noisy surfaces cannot.  RUN THIS WITHOUT A PIPE: `| grep tree` keeps the "
             "dressing lines whose item names contain 'tree' (tree_conifer, tree_fir, tree_sapling) "
             "and hides the [tree] lines this prints, so the output looks like the branch never ran.  "
             "Kit also buffers stdout, so the file is written and flushed as well.",
    )
    parser.add_argument(
        "--dressing_report", action="store_true",
        help="Print, per dressing item, whether its prop reference resolved, failed, or there was "
             "no asset -- then exit without rendering.  A resolved prop has its clearance box made "
             "invisible, so an item that resolves and then fails to draw is simply absent with no "
             "fallback to see; 'the picture is empty' cannot distinguish that from 'the reference "
             "failed', and this can.",
    )
    # The three light parameters environment._setup_lights already reads out of task_dict, exposed
    # here because a diagnostic that cannot change one variable cannot attribute a frame to it.  The
    # default is None, NOT the light's own value: passing None leaves the key out of task_dict so
    # environment.py's default is what applies, and this flag can therefore never become a second
    # source of truth for the exposure the scene was tuned to.
    parser.add_argument(
        "--light_intensity", type=float, default=None,
        help="Intensity for the eight SphereLights at /World/Light_01..08.  Leave unset to use "
             "environment.py's tuned default (1500).  0 turns them off, which is how the "
             "high-frequency pattern on flat-painted walls is attributed to lighting or to the "
             "surfaces: with use_mdl False the four surfaces are a single displayColor each, and a "
             "solid colour cannot produce a pattern.",
    )
    parser.add_argument(
        "--light_radius", type=float, default=None,
        help="Radius for the same eight SphereLights.  Unset leaves environment.py's default (1.0).",
    )
    parser.add_argument(
        "--frames", type=int, default=32,
        help="Render this many frames of each pose and average them.  RaytracedLighting does not "
             "converge for a static scene, so one frame carries 14+ grey levels of per-pixel noise "
             "even on a flat-painted wall; the noise is zero-mean, so averaging n frames reduces it "
             "as 1/sqrt(n) (measured on this machine: block std 14.29 at 1 frame, 5.30 at 16, 4.10 "
             "at 32, 3.19 at 64).  Default 32, about 1.5 s per camera.  --frames 1 reproduces the "
             "old single-frame preview.",
    )
    parser.add_argument(
        "--raytracing", action="store_true",
        help="Name Isaac's default RaytracedLighting explicitly.  It IS the default and the mode "
             "this preview renders in; PathTracing was measured to be cleaner (successive-frame "
             "jitter 0.22-0.26 against 14.3-15.9) but cannot be selected safely from here, because "
             "environment.setup_scene() resets /rtx/rendermode and re-applying it after the scene "
             "exists segfaults the renderer.  The averaging that --frames controls is what makes a "
             "RaytracedLighting frame legible instead.",
    )
    parser.add_argument(
        "--dome_intensity", type=float, default=None,
        help="Intensity for /World/DomeLight (the ambient fill).  Unset leaves environment.py's "
             "default (300).",
    )
    return parser.parse_args()


def average_frames(env: Any, camera: Any, frames: int) -> np.ndarray:
    """The mean of `frames` renders of the CURRENT pose, one frame at a time.

    RaytracedLighting does not converge for a static scene, so a single frame carries per-pixel noise
    of 14+ grey levels even on a flat-painted wall.  The noise is zero-mean, so averaging n frames
    reduces it as 1/sqrt(n): measured on this machine, the median 16x16 block standard deviation of a
    stage1.4 eye frame falls 14.29 (1 frame) -> 8.57 (4) -> 5.30 (16) -> 4.10 (32) -> 3.19 (64).
    Thirty-two frames is the preview's default, about 1.5 s per camera at 1024x1024.

    A frame is read from `camera` when it is not None (the external camera) and from `env` otherwise
    (the robot eye camera, which the environment reads through its own annotator).
    """
    total = None
    for _ in range(max(1, int(frames))):
        env.world.step(render=True)
        if camera is None:
            image = np.asarray(env.get_camera_image())[:, :, :3].astype(np.float64)
        else:
            image = np.asarray(camera.get_rgb())[:, :, :3].astype(np.float64)
        total = image if total is None else total + image
    return total / max(1, int(frames))


def write_png(path: str, rgb: np.ndarray) -> None:
    """An 8-bit RGB PNG, filter 0 on every row.  No Pillow, no image library at all."""
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"expected HxWx3, got {rgb.shape}")
    raw = b"".join(b"\x00" + rgb[row].astype(np.uint8).tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", header))
        handle.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        handle.write(chunk(b"IEND", b""))


def tile(images, columns: int, gap: int = 4, background: int = 255) -> np.ndarray:
    """Lay equally sized images out in a grid with a light gap between them."""
    rows = (len(images) + columns - 1) // columns
    height, width = images[0].shape[:2]
    canvas = np.full((rows * height + (rows + 1) * gap,
                      columns * width + (columns + 1) * gap, 3), background, np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = gap + row * (height + gap)
        x = gap + column * (width + gap)
        canvas[y:y + height, x:x + width] = image[:, :, :3]
    return canvas


def look_from(camera, target, position) -> None:
    """Point an external sensor at the scene from an explicit world position.

    It used to place the sensor from (target, distance, height) as
    (target_x - 0.55 d, height, target_z + 0.55 d).  With the defaults that is
    (-0.6, 7.0, 7.6): above the 3 m ceiling and outside the 5 m wide room, so it looked at the
    back of a side wall and every iso frame came out flat grey.  An explicit position cannot do
    that, and the caller can read where the camera is.
    """
    position = np.array(position, dtype=float)
    direction = np.array(target, dtype=float) - position
    norm = float(np.linalg.norm(direction)) or 1.0
    direction = direction / norm
    # Build a rotation whose +X is the view direction and +Z is world up.
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(direction, up)
    right = right / (float(np.linalg.norm(right)) or 1.0)
    true_up = np.cross(right, direction)
    matrix = np.stack([direction, -right, true_up], axis=1)
    quat = _quaternion_from_matrix(matrix)
    camera.set_world_pose(position=position.tolist(), orientation=quat.tolist(),
                          camera_axes="world")


def _quaternion_from_matrix(matrix: np.ndarray):
    """Rotation matrix to (w, x, y, z), the convention Isaac's set_world_pose expects."""
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax([matrix[0, 0], matrix[1, 1], matrix[2, 2]]))
        if index == 0:
            scale = (1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) ** 0.5 * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = (1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) ** 0.5 * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = (1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) ** 0.5 * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    return np.array([w, x, y, z], dtype=float)


def build_stamp() -> str:
    """An identifier for the code that produced a frame: git HEAD plus the key modules' mtimes.

    Written because a stale directory of previews is indistinguishable from a fresh run, and that
    ambiguity was read as evidence about code the images had never executed.  It goes on stdout and
    into slot_report.txt, so any frame can be dated against the checkout that made it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    head = "unknown"
    try:
        import subprocess

        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        pass
    parts = []
    for name in ("scene_builder.py", "scenes.py"):
        path = os.path.join(here, name)
        try:
            parts.append(f"{name}@{int(os.path.getmtime(path))}")
        except OSError:
            parts.append(f"{name}@missing")
    return f"git {head} " + " ".join(parts)


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    slots = [int(part) for part in args.slots.split(",") if part.strip()]

    from isaacsim import SimulationApp

    # Exactly what main.py does: {"headless": ...} and nothing else.  Passing width/height
    # here set the RENDER target to 256x256 -- half of the 512 asked for -- which the camera
    # then upscaled to its own 1024, and the frames came back as salt-and-pepper noise.  The
    # lab log said so in plain words: "DLSS increasing input dimensions: Render resolution of
    # (256, 256) is below minimal input resolution of 300".  The camera's own resolution is set
    # by the environment, which is why the runner never needed to pass anything.
    app = SimulationApp({"headless": True})
    try:
        import carb

        # RENDER MODE AND FRAME AVERAGING.
        #
        # RaytracedLighting, the default, does not converge for a STATIC scene.  Measured on all
        # five scenes, two renders of the same pose differ by 14.3-15.9 grey levels on average with
        # pixels changing by up to 224 (diag_logs/pt_stage1.*), and in a flat-painted wall -- which
        # is one displayColor and cannot carry a pattern -- every 16x16 block has a standard
        # deviation of 8-20 levels.  That is what the "high-frequency pattern" was.
        #
        # PathTracing removes it (0.22-0.26 for the same measurement) but cannot be switched on
        # safely from here: environment.setup_scene() resets /rtx/rendermode, and re-applying it
        # after the scene exists segfaults the renderer (diag_logs/verify_mode.log).  Setting it
        # before the scene is built is silently discarded (diag_logs/capture_repeat.log).
        #
        # So the preview AVERAGES FRAMES instead, which needs no renderer change and works in the
        # default mode.  The noise is zero-mean, so the mean converges as 1/sqrt(n) and that was
        # measured rather than assumed (diag_logs/avg_n.log, 16x16 block standard deviation):
        #
        #     1 frame 14.29   2 frames 10.81   4 frames 8.57   8 frames 6.67
        #    16 frames  5.30  32 frames 4.10  64 frames 3.19
        #
        # 32 frames is the default: 4.10 against 14.29 for one frame, at about 1.5 s per camera.
        # --frames 1 reproduces the old single-frame preview.
        def apply_render_mode() -> None:
            # Kept for --raytracing: it is the default already, so this only ever NAMES the mode.
            if args.raytracing:
                carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

        apply_render_mode()

        import environment
        import scenes

        if args.scene not in scenes.SCENES:
            print(f"unknown scene {args.scene!r}; expected one of {list(scenes.SCENES)}")
            return 1

        task = {"headless": True, "scene": args.scene, "level": args.level,
                "image_size": args.width,
                "use_mdl": bool(args.mdl),
                "hide_robot": bool(args.hide_robot),
                "scene_parts": [p.strip() for p in args.parts.split(",") if p.strip()]}
        # Only the light keys the caller actually gave are added, so an unset flag leaves
        # environment.py's own default in force instead of shadowing it with a copy.
        for key, value in (("light_intensity", args.light_intensity),
                           ("light_radius", args.light_radius),
                           ("dome_intensity", args.dome_intensity)):
            if value is not None:
                task[key] = value
        env = environment.setup_scene(app, task_dict=task)
        # A BUILD STAMP, so a picture can be dated.  A stale directory of previews from an earlier
        # checkout is indistinguishable from a fresh run of the current code, and that ambiguity cost
        # real time: the images were read as evidence about code they had never executed.  The stamp
        # names git HEAD and the mtime of the two modules that decide what a frame contains.
        stamp = build_stamp()
        print(f"[preview] code stamp {stamp}")
        print(f"[preview] scene {args.scene} ({scenes.SCENES[args.scene]['label']}) "
              f"level {args.level} width "
              f"{environment.LEVEL_CHANNEL_WIDTHS[args.level]:.3f} m")

        if args.stage_tree:
            # TEXT ONLY, and it answers the one question that a picture kept being asked and could not
            # answer: is the prop actually on the stage, visible, and in front of the camera?  A frame
            # full of lights and surfaces makes a small prop hard to find even when it is there, and
            # "I cannot see it" has been read as "it is not drawn" more than once.  This prints the
            # prim tree under /World/Dressing with each prim's active state, its inherited visibility,
            # and whether it carries a display color or a bound material -- none of which requires
            # rendering, so it is cheap to run.
            #
            # Imported HERE rather than at module scope: isaacsim.core and pxr are importable only
            # after SimulationApp has started, which this file's own header documents as the trap that
            # cost a round trip.
            from pxr import UsdGeom, UsdShade

            # Collected in a list AND printed as it goes, then written to a flushed file.  Kit buffers
            # stdout: a full render once reported five successful paints for twenty-five slot renders
            # because the process ended before the buffer reached the log.  A diagnostic that can be
            # lost to a buffer is the same as one that never ran, which is exactly how the previous
            # --stage_tree invocation was read.
            tree_lines: list = []

            def emit(line: str) -> None:
                tree_lines.append(line)
                print(line, flush=True)

            def walk(prim, depth: int) -> None:
                if depth > 3:
                    return
                imageable = UsdGeom.Imageable(prim)
                visible = imageable.ComputeVisibility() if imageable else "n/a"
                colour = prim.GetAttribute("primvars:displayColor")
                bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                has_colour = bool(colour and colour.HasAuthoredValue())
                extras = []
                if has_colour:
                    extras.append("displayColor")
                if bound and bound.GetPrim().IsValid():
                    extras.append(f"material={bound.GetPrim().GetName()}")
                emit(f"[tree] {'  ' * depth}{prim.GetName()} active={prim.IsActive()} "
                     f"visible={visible} {' '.join(extras) if extras else '-'}")
                for child in prim.GetChildren():
                    walk(child, depth + 1)

            root = env.stage.GetPrimAtPath("/World/Dressing")
            if root and root.IsValid():
                walk(root, 0)
            else:
                emit("[tree] /World/Dressing does not exist on the stage -- nothing was placed")
            for light_path in ("/World/DomeLight", "/World/Light_01"):
                light_prim = env.stage.GetPrimAtPath(light_path)
                emit(f"[tree] {light_path} present={bool(light_prim and light_prim.IsValid())}")
            # Written and flushed, so the answer survives a buffered or truncated stdout.
            with open(os.path.join(args.outdir, "stage_tree.txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(tree_lines) + "\n")
            print(f"[tree] wrote {len(tree_lines)} line(s) to "
                  f"{os.path.join(args.outdir, 'stage_tree.txt')}", flush=True)
            return 0

        if args.dressing_report:
            # Text only, and it exists because images did not answer the question.  An item whose prop
            # resolves has its clearance box made INVISIBLE, so if the prop itself then fails to draw
            # for any reason the item is absent from the frame with no fallback box to notice --
            # "nothing in the picture" therefore has two opposite causes and a picture cannot tell
            # them apart.  place_dressing records which happened per item and this prints it.  (This
            # comment used to say the box was DEACTIVATED, which is what the code did and what made
            # every resolved prop invisible; it is kept accurate because it was one of several stale
            # explanations that made these reports hard to trust.)
            report = (env.scene_report or {}).get("dressing") or []
            print(f"[dressing] {len(report)} item(s) from the scene report")
            print(f"{'item':<16} {'mount':<14} {'asset':<34} result")
            print("-" * 92)
            ok = failed = missing = 0
            for entry in report:
                used = entry.get("used_asset")
                error = entry.get("reference_error")
                if used:
                    ok += 1
                    result = f"RESOLVED {used}"
                    fit = entry.get("fit_scale") or ()
                    if any(abs(float(v) - 1.0) > 1e-9 for v in fit):
                        result += f"  fit={tuple(round(float(v), 4) for v in fit)}"
                elif error:
                    failed += 1
                    result = f"FAILED {error}"
                else:
                    missing += 1
                    result = "NO ASSET (drawn as its clearance box)"
                print(f"{entry['name']:<16} {entry['mount']:<14} "
                      f"{str(entry.get('asset') or '-')[-32:]:<34} {result}")
            print()
            print(f"[dressing] resolved {ok}, failed {failed}, no asset {missing}")
            return 0

        eye_images, iso_images = [], []
        for slot in slots:
            # ORDER MATTERS, and the first version had it backwards.  reset() is what
            # initialises the camera annotators AND what advances the marker for the next
            # episode, so calling set_marker_slot() before it meant reset() immediately put a
            # different marker in place: every image was of slot+1 while being labelled slot.
            # Reset first, then set the slot explicitly, then read.
            env.reset()
            env.set_marker_slot(slot)
            # Let the renderer draw this pose before reading it, then average: the eye frame used to
            # be read with NO step at all, and one frame of a non-converging renderer is the noisiest
            # thing this file can capture.
            for _ in range(3):
                env.world.step(render=True)
            eye = average_frames(env, None, args.frames).astype(np.uint8)
            eye_images.append(eye)
            write_png(os.path.join(args.outdir, f"{args.scene}_slot{slot}_eye.png"), eye)
            iso_camera = getattr(env, "camera", None)
            if iso_camera is not None:
                # Move the camera AFTER the reset, not before: reset() calls _update_camera(),
                # which puts the external camera back where the environment wants it, so the
                # earlier order silently threw the iso pose away.
                # Stand just outside the open end of the corridor, a little above eye height and
                # a little to one side, looking down it.  Inside the room in both z and y, which
                # is what the old placement was not.
                look_from(iso_camera, target=tuple(args.iso_target),
                          position=(args.iso_x, args.iso_height, args.iso_z))
                iso = average_frames(env, iso_camera, args.frames).astype(np.uint8)
                iso_images.append(iso)
                write_png(os.path.join(args.outdir, f"{args.scene}_slot{slot}_iso.png"), iso)

                # Extra views around the target, because one camera cannot show this room.
                #
                # The dressing is deliberately OUT of the walking band (|z| >= 1.5) while the eye
                # camera looks through a 0.570 m opening, so from the start pose the side spaces --
                # where every tree, bench and shrub stands -- are not visible at all.  That is a
                # property of the geometry, not of the camera, and it is why a preview kept coming
                # back looking empty while the log said 11 items were placed.
                #
                # Orbiting the target by full turns samples every side.  Angles are degrees around
                # the target; each view is written as <scene>_slot<N>_orbit<angle>.png.
                for angle in getattr(args, "iso_orbit", None) or []:
                    radians = math.radians(float(angle))
                    target = tuple(args.iso_target)
                    position = (target[0] + args.iso_orbit_radius * math.cos(radians),
                                args.iso_orbit_height,
                                target[2] + args.iso_orbit_radius * math.sin(radians))
                    look_from(iso_camera, target=tuple(target), position=position)
                    for _ in range(3):
                        env.world.step(render=True)
                    orbit = np.asarray(iso_camera.get_rgb())[:, :, :3]
                    name = f"{args.scene}_slot{slot}_orbit{int(float(angle))}.png"
                    write_png(os.path.join(args.outdir, name), orbit)
                    print(f"[preview] orbit {angle} deg from "
                          f"{tuple(round(v, 2) for v in position)}", flush=True)
            # One line per rendered slot, appended to a file this script controls.
            #
            # Not stdout, and that is the point.  A full render reported "painted True" 5 times
            # for 25 slot renders, because Kit's stdout loses what the process did not flush
            # before app.close() -- the same failure this repo hit once already with a
            # measurement tool.  So whether all 25 rebuilds succeeded could not be established
            # from the log; a file written and flushed here can be.
            marker_info = (env.scene_report or {}).get("marker") or {}
            dressing = (env.scene_report or {}).get("dressing") or []
            resolved = sum(1 for item in dressing if item.get("used_asset"))
            no_asset = sum(1 for item in dressing
                           if not item.get("used_asset") and not item.get("reference_error"))
            with open(os.path.join(args.outdir, "slot_report.txt"), "a", encoding="utf-8") as log:
                log.write(
                    f"{args.scene} slot={slot} stamp={stamp} "
                    f"marker={marker_info.get('shape')}/{marker_info.get('colour')} "
                    f"parts={marker_info.get('parts')} "
                    f"size={marker_info.get('size_m')} "
                    f"painted_ok={marker_info.get('painted')} "
                    f"dressing_resolved={resolved}/{len(dressing)} "
                    f"dressing_no_asset={no_asset} "
                    f"error={marker_info.get('paint_error') or '-'}\n"
                )
            print(f"[preview] slot {slot}: {scenes.MARKERS[args.scene][slot - 1]}")

        write_png(os.path.join(args.outdir, f"{args.scene}_sheet_eye.png"),
                  tile(eye_images, columns=len(eye_images)))
        if iso_images:
            write_png(os.path.join(args.outdir, f"{args.scene}_sheet_iso.png"),
                      tile(iso_images, columns=len(iso_images)))
        print(f"[preview] wrote {len(os.listdir(args.outdir))} files to {args.outdir}/")
        # THE LINEARITY GUARD IS AUDITED HERE, IN THE PATH THAT ACTUALLY RUNS IT.
        #
        # NON_LINEAR_BOUND_ASSETS and NON_LINEAR_BOUND_UNMEASURED are written only while a prop is
        # being placed, and the only thing that ever READ them was a unit test that does not run the
        # placement -- so on its own that check was vacuously true: an empty set, always, on every
        # checkout.  A render is what exercises the probe on all 58 props, so the audit belongs at the
        # end of one.
        import scene_builder as _sb
        _bad = sorted(getattr(_sb, "NON_LINEAR_BOUND_ASSETS", ()))
        _unmeasured = int(getattr(_sb, "NON_LINEAR_BOUND_UNMEASURED", 0))
        print(
            f"[preview] linearity probe: {len(_bad)} non-linear asset(s), "
            f"{_unmeasured} unmeasurable probe(s)"
        )
        if _bad or _unmeasured:
            print(
                f"[preview] FAILED: the per-axis fit cannot be trusted -- non-linear: {_bad or 'none'}"
                f", unmeasured probe(s): {_unmeasured}. Every size in this report is suspect."
            )
            return 1
        return 0
    except Exception:  # noqa: BLE001
        # Print before closing: env.close() in the finally-path is what used to swallow
        # the traceback and leave a failed run looking like a successful one.
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
