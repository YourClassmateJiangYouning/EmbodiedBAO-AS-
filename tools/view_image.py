"""Look at a rendered PNG when all you have is a terminal.

Two modes, because there are two different questions:

  --ascii   print a coarse luminance rendering.  Enough to see the layout -- where
            the body silhouette is, whether there is floor or wall around it --
            without any image viewer or file transfer.
  model     send the frame to a vision model on the roster and print its answer.
            This is the sharper test for a frame the AGENT has to read: the model
            answers the same question the experiment depends on, namely whether
            the robot's own body and the room are both visible in it.

No Isaac Sim and no environment import: PIL for the pixels, urllib for the API.

    python tools/view_image.py look_down_check/look_down_visible.png --ascii
    python tools/view_image.py look_down_check/forward.png look_down_check/look_down_visible.png --model qwen3-vl-32b-instruct
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"
RAMP = " .:-=+*#%@"

DEFAULT_QUESTION = (
    "This frame comes from a 1.68 m tall humanoid robot's head camera. "
    "Describe what you see in one or two sentences. Then answer these explicitly: "
    "(1) is any part of the robot's OWN body visible, and which part? "
    "(2) is any of the room visible -- floor, walls, or an opening? "
    "(3) could you estimate how wide the robot is from this frame? "
    "Be concrete about what is where in the image."
)


def ascii_render(path: str, width: int, height: int) -> str:
    from PIL import Image

    image = Image.open(path).convert("L")
    # Box-average down to the requested size: PIL's default resample is fine for
    # this, and averaging is what makes the blocks readable rather than noisy.
    small = image.resize((width, height), Image.BOX)
    pixels = list(small.getdata())
    lines = []
    for row in range(height):
        chunk = pixels[row * width : (row + 1) * width]
        lines.append(
            "".join(RAMP[min(len(RAMP) - 1, int(value) * len(RAMP) // 256)] for value in chunk)
        )
    return "\n".join(lines)


def image_data_url(path: str) -> str:
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    extension = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "image/jpeg" if extension in ("jpg", "jpeg") else f"image/{extension}"
    return f"data:{mime};base64,{encoded}"


def ask_model(
    path: str, model: str, base_url: str, key: str, question: str, timeout: float
) -> str:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_data_url(path)}},
                ],
            }
        ],
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return f"HTTP {exc.code}: {exc.reason}; body={detail}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    choice = (data.get("choices") or [{}])[0]
    return ((choice.get("message") or {}).get("content") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", help="PNG/JPEG files to look at")
    parser.add_argument("--ascii", action="store_true", help="print a luminance map")
    parser.add_argument("--width", type=int, default=96, help="ascii columns")
    parser.add_argument("--height", type=int, default=36, help="ascii rows")
    parser.add_argument("--model", default=None, help="vision model to describe them")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    missing = [path for path in args.images if not os.path.exists(path)]
    if missing:
        print(f"no such file(s): {', '.join(missing)}")
        return 1

    if args.ascii or not args.model:
        for path in args.images:
            print("=" * 72)
            print(f"{path}   ({args.width}x{args.height} luminance, ' ' = dark, "
                  f"'@' = bright)")
            print("=" * 72)
            print(ascii_render(path, args.width, args.height))
            print()
        if not args.model:
            print("For a description instead of a luminance map, add --model <name>.")
            return 0

    key = (
        os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        print("No API key: set BOYUE_API_KEY to use --model.")
        return 1
    base_url = args.base_url or os.environ.get("BOYUE_BASE_URL") or DEFAULT_BASE_URL
    question = args.question or DEFAULT_QUESTION
    for path in args.images:
        print("=" * 72)
        print(f"{path}  described by {args.model}")
        print("=" * 72)
        print(ask_model(path, args.model, base_url, key, question, args.timeout))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
