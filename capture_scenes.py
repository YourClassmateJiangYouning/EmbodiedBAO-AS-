"""Render one Stage 3 scene so it can be looked at.

Why this exists: the scene design is a pile of material URLs, marker polygons and dressing
coordinates, and the only way to know whether it looks like a supermarket rather than a
slightly differently painted laboratory is to render it.  This is also how the marker
uniqueness claim gets eyeballed before the 6,375-episode sweep is trusted to it.

One scene per process, because SimulationApp is a singleton -- the same reason
capture_views.py takes one level at a time.  Run it five times, or use the loop in the
docstring.

Output, under ``--outdir``:

* ``<scene>_slot<N>_eye.png``  what the model sees at the start pose (512 px, the real input)
* ``<scene>_slot<N>_iso.png``  an external view, for judging the room rather than the task
* ``<scene>_sheet.png``        the five markers side by side, eye view above iso view

Images are written with a small PNG encoder rather than Pillow, which is not installed on
the lab machine -- the persistence suite already fails there for that reason.

Usage:
    ISAAC_PY=/home/ybh/isaacsim/python.sh
    for s in stage1.1 stage1.2 stage1.3 stage1.4 stage1.5; do
      $ISAAC_PY capture_scenes.py --scene $s --level 10 --outdir preview; done
"""

from __future__ import annotations

import argparse
import os
import struct
import traceback
import zlib

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one Stage 3 scene.")
    # Not derived from scenes.py here: importing it before SimulationApp is the trap
    # capture_views.py documents, so the name is validated after the app is up.
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--level", type=int, default=10,
                        help="A/S level whose opening to build (10 is A/S 1.00)")
    parser.add_argument("--slots", type=str, default="1,2,3,4,5")
    parser.add_argument("--outdir", type=str, default="scene_preview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--iso_distance", type=float, default=12.0)
    parser.add_argument("--iso_height", type=float, default=7.0)
    return parser.parse_args()


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


def look_from(camera, target, distance: float, height: float) -> None:
    """Point an external sensor at the scene, using the documented world-axes call."""
    position = np.array([target[0] - distance * 0.55, height, distance * 0.55], dtype=float)
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


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    slots = [int(part) for part in args.slots.split(",") if part.strip()]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
    try:
        import environment
        import scenes

        if args.scene not in scenes.SCENES:
            print(f"unknown scene {args.scene!r}; expected one of {list(scenes.SCENES)}")
            return 1

        task = {"headless": True, "scene": args.scene, "level": args.level,
                "image_size": args.width}
        env = environment.setup_scene(app, task_dict=task)
        print(f"[preview] scene {args.scene} ({scenes.SCENES[args.scene]['label']}) "
              f"level {args.level} width "
              f"{environment.LEVEL_CHANNEL_WIDTHS[args.level]:.3f} m")

        eye_images, iso_images = [], []
        for slot in slots:
            env.set_marker_slot(slot)
            # The RGB annotator is only attached once the renderer has run, and get_rgb()
            # raises AttributeError on None until then -- measured on the lab machine.  The
            # runner steps the world before it reads anything, which is why it never hit this.
            for _ in range(4):
                env.world.step(render=True)
            eye = np.asarray(env.get_camera_image())[:, :, :3]
            eye_images.append(eye)
            write_png(os.path.join(args.outdir, f"{args.scene}_slot{slot}_eye.png"), eye)
            iso_camera = getattr(env, "camera", None)
            if iso_camera is not None:
                look_from(iso_camera, target=(6.0, 0.0, 1.0),
                          distance=args.iso_distance, height=args.iso_height)
                for _ in range(4):
                    env.world.step(render=True)
                iso = np.asarray(iso_camera.get_rgb())[:, :, :3]
                iso_images.append(iso)
                write_png(os.path.join(args.outdir, f"{args.scene}_slot{slot}_iso.png"), iso)
            print(f"[preview] slot {slot}: {scenes.MARKERS[args.scene][slot - 1]}")

        write_png(os.path.join(args.outdir, f"{args.scene}_sheet_eye.png"),
                  tile(eye_images, columns=len(eye_images)))
        if iso_images:
            write_png(os.path.join(args.outdir, f"{args.scene}_sheet_iso.png"),
                      tile(iso_images, columns=len(iso_images)))
        print(f"[preview] wrote {len(os.listdir(args.outdir))} files to {args.outdir}/")
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
