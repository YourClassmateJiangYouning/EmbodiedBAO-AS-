"""Stage 4 core: session state, prompt assembly, model calls, recording and export.

Stage 4 runs a *closed loop* against real photographs rather than a simulator: the operator captures a
1024x1024 frame, the model returns one action, the operator performs it, and the next frame is captured.  This
module owns everything behind the window; `server.py` only moves JSON between the window and this object.

Three rules from the handover spec drive the structure:

1. **The prompt must be the Stage 1 prompt.**  Rather than re-typing the wording, `build_prompt` is imported
   from the repository's `protocol.py`; `set_marker_descriptor` names the real marker ("red triangle" /
   "green square").  A/S, channel width, "narrow", "wide" and the level never enter the prompt -- they exist
   only in the session config and the operator's window.

2. **Nothing is invented.**  A timeout, an unparseable reply or an action outside the enum is recorded as
   invalid with the raw response kept; the code never substitutes `forward`, never rewrites an action, and
   never records a failed call as executed.  Every retry is logged separately.

3. **The raw material is kept.**  Original image bytes, the exact prompt text, and every raw response are
   written to disk next to the parsed fields, so a later analysis can re-check any number.

Layout per session, as specified:

    stage4/sessions/<session_id>/
        config.json  manifest.csv  episodes.jsonl  steps.jsonl
        images/  prompts/  responses/  exports/
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from protocol import (  # noqa: E402
    ACTIONS,
    MOVE_STEP as PROTOCOL_MOVE_STEP,
    SYSTEM_PROMPT,
    build_prompt,
    set_marker_descriptor,
)

DEFAULT_ENDPOINT = "http://35.220.164.252:3888/v1"
DEFAULT_MODELS = [
    "gemini-3-flash-preview",
    "deepseek-v4-flash-vision-exp",
    "claude-haiku-4-5-20251001",
]
A_S_LEVELS = [2.0, 1.2, 1.0, 0.8, 0.6]

# The five markers printed for the real corridor, in the words the prompt must use.  These are physical
# boards, so the prompt names them exactly; there is no scene table to look them up from.
MARKER_PRESETS = [
    "green rectangle",
    "red circle",
    "yellow arrow",
    "blue triangle",
    "purple four-pointed star",
]

# Stage 3 changed exactly one thing in the frozen Stage 1 prompt: the body name, so "You are a Unitree G1
# humanoid robot" matched the body actually being tested (stage3_runner.py:68).  The real corridor uses a low
# wheeled platform, so the same substitution is made here -- one noun phrase, every other word frozen.
DEFAULT_BODY_NAME = "a small wheeled robot"

# The real-scene look_down frame is a single photograph, so it cannot show the body at every torso angle the
# way the simulator's renderer does.  Rather than let the prompt promise a view the image does not give, the
# description says exactly what the frame contains -- the body seen from its own front, i.e. the width being
# presented and the thickness behind it -- and points at the torso angle in the state as the thing that says
# how that width is oriented.  Every other word stays as frozen Stage 1 text.
LOOK_DOWN_DESCRIPTION_REAL = (
    "glance downwards with your head camera, 45 degrees below your usual view: you see the floor in front of "
    "you and your own body as seen from your own front. That view gives you the two dimensions you cannot "
    "otherwise see: the width you present to whatever is ahead, and the thickness of your body behind it. "
    "The view itself does not change when you rotate your torso -- it always shows the body from your own "
    "front -- so read the torso rotation reported in your state to know which way that width is now facing. "
    "Your gaze returns to its usual direction as soon as you do anything else. Use it to see how wide you "
    "are, which you cannot otherwise see. Your torso and your walking direction are "
    "unaffected by the glance itself"
)

# The real scene has photographs only for the forward walk, the torso turn and one downward glance.  left and
# right would need laterally offset photographs, and look_left/look_right would need 30-degree aside views;
# neither exists, so the action set is the five the frame bank can answer.  In Stage 1 these five cover about
# 85% of all decisions (forward 73.1%, turn_left+turn_right 11.5%, look_down 5.0%).
REAL_ACTION_SET = ["forward", "backward", "turn_left", "turn_right", "look_down"]

# The plate-and-wheels photograph stands in for the body; the marker name selects the bank folder that holds
# that target's forward sequence.
BANK_MARKER_DIRS = {
    "red circle": "红色",
    "green rectangle": "绿色",
    "blue triangle": "蓝色",
    "yellow arrow": "黄色",
    "purple four-pointed star": "紫色",
}

# The frozen frame note minus the clauses about actions that are no longer offered.
ACTION_FRAME_NOTE_REAL = (
    "Your walking direction is fixed: it always points straight ahead at the far wall, which is where the red "
    "marker you are heading for is. forward/backward are defined relative to that walking direction, so they "
    "behave the same way however your torso is turned.\n"
    "turn_left/turn_right rotate your torso 15 degrees relative to that walking direction. The direction you "
    "walk does not change, and neither does your view: your eyes look straight ahead along your walking "
    "direction whatever your torso is doing. Turning changes how your body is oriented; it does not steer you "
    "and it does not change what you see.\n"
    "look_down is a single glance 45 degrees below your usual view. Your torso and your walking direction are "
    "unaffected, and the glance clears as soon as you do anything else. You cannot look up."
)

# THE SCENE MUST MATCH THE SIMULATOR, NOT THE HANDOVER DOCUMENT.
#
# The handover spec says the robot starts 4.0 m from the opening, but the simulator that produced Stages 1-3
# does not: ROBOT_START_POS is [0.5, 0, 0] and WALL_X is 8.0, so the start distance is 7.5 m and a straight
# walk to the opening takes 10 steps of 0.75 m -- which the recorded data confirms, since the median passed
# episode at every wide level is exactly 11 steps (10 forward + the step that carries it through).
#
# Every number below is copied from the simulator, with the constant it comes from:
SIM_GEOMETRY: Dict[str, Any] = {
    "start_x_m": 0.5,               # ROBOT_START_POS = [0.5, 0.0, 0.0]
    "wall_x_m": 8.0,                # WALL_X
    "start_distance_m": 7.5,        # 8.0 - 0.5
    "straight_steps_to_wall": 10,   # 7.5 / 0.75
    "move_step_m": 0.75,            # protocol.MOVE_STEP
    "turn_step_deg": 15.0,          # TURN_STEP_DEG
    "camera_turn_step_deg": 30.0,   # CAMERA_TURN_STEP_DEG (look_left / look_right)
    "camera_pitch_step_deg": 45.0,  # CAMERA_PITCH_STEP_DEG (look_down, on top of the 15 deg eye pitch)
    "eye_height_m": 1.68,           # EYE_CAMERA_HEIGHT
    "eye_pitch_deg": 15.0,          # EYE_PITCH_DEG, the camera's normal downward tilt
    "shoulder_width_m": 0.570,      # ROBOT_SHOULDER_WIDTH (robot_profiles.py)
    "torso_thickness_m": 0.220,     # ROBOT_TORSO_THICKNESS -- the width after a 90 deg turn
    "start_yaw_deg": 0.0,           # ROBOT_START_YAW_DEG
    "max_steps": 30,
}

ROBOT_SHOULDER_M = SIM_GEOMETRY["shoulder_width_m"]
MAX_STEPS = 30
IMAGE_SIZE = 1024
# Distance covered by one forward/backward/left/right action, in metres.  Kept at the protocol's 0.75 so the
# action descriptions in the prompt stay word-for-word identical to Stages 1-3; a session may override it,
# and the value actually used is written to config.json and into every step record.
MOVE_STEP = 0.75
API_TEMPERATURE = 0.0
# Measured, not guessed.  At 1200 tokens gemini-3-flash-preview once reported finish_reason=length with
# 1151 reasoning tokens and only 45 characters of visible text, and deepseek-v4-flash-vision-exp consumed
# 1200/1200 and then 3000/3000 and 8000/8000 on reasoning while returning zero characters -- its reply never
# reached the JSON.  With reasoning_effort="none" both return a complete action in under 3 s.  2000 keeps
# headroom for a long scene_description on the models that still think (claude does not).
API_MAX_TOKENS = 2000
DEFAULT_TIMEOUT_S = 90.0
NETWORK_RETRIES = 2               # timeout / 5xx
FORMAT_RETRIES = 1                # invalid JSON / missing action / action not in enum

STATUSES = (
    "IDLE", "READY", "CAPTURING", "CALLING_MODEL", "WAITING_FOR_ACTION",
    "ACTION_DONE", "INVALID", "PASSED", "FAILED", "STOPPED", "ERROR",
)
# Formats a phone or a screenshot can arrive in.
INBOX_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

# Per-model request extras.  These mirror what the completed sweeps needed: without the override the model
# reasons at length and either times out or burns the token budget on deliberation instead of an action.
#
# The Stage 4 entries were measured on the real prompts with the probe in tools-style form:
#   gemini-3-flash-preview        default 6.9 s, 840 reasoning tokens, finish=stop   -> reasoning_effort=none
#                                 2.7 s, 0 reasoning tokens, complete JSON            (the switch IS honoured)
#                                 "thinking: disabled" is NOT honoured by this model (still 840 tokens)
#   deepseek-v4-flash-vision-exp  default: 1200/1200, 3000/3000 and 8000/8000 reasoning tokens, 0 characters
#                                 -> reasoning_effort=none -> complete JSON in 1.8 s
#   claude-haiku-4-5-20251001     no hidden reasoning at all; the switch is a no-op and is not sent
# "thinking: {type: disabled}" works for deepseek too but not for gemini, so the standard OpenAI spelling is
# the one used everywhere -- one knob, same meaning for every model.
MODEL_REQUEST_PARAMS: Dict[str, Dict[str, Any]] = {
    "gpt-5": {"reasoning_effort": "none"},
    "glm-4.6v": {"thinking": {"type": "disabled"}},
    "glm-4.5v": {"thinking": {"type": "disabled"}},
    "gemini-3-flash-preview": {"reasoning_effort": "none"},
    "deepseek-v4-flash-vision-exp": {"reasoning_effort": "none"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid.uuid4().hex[:12])


# --------------------------------------------------------------------------------------------- helpers

def channel_width_m(a_s: float, shoulder_m: float = ROBOT_SHOULDER_M) -> float:
    """Opening width for an A/S (the only place the geometry is computed)."""
    return round(a_s * shoulder_m, 4)


def encode_for_api(image_path: str, size: int = IMAGE_SIZE, quality: int = 95) -> Tuple[str, int]:
    """Return (data URL, payload bytes) for the API, at `size` x `size`, as JPEG.

    The stored image stays untouched; this is only what goes over the wire.  The completed sweeps sent
    1024 px JPEG at quality 95, so the same encoder is used here and payload size is reported in the step
    record -- a large payload is the first thing to check when latency looks wrong.

    A non-square frame is CENTRE-CROPPED to a square before shrinking, never stretched.  A phone photograph of
    a monitor is often 4:3, and squashing it to 1:1 would reshape the opening the model has to judge; cropping
    keeps the geometry honest and only throws away edges.  capture() records the real dimensions so the window
    can warn that a crop happened.
    """
    raw = open(image_path, "rb").read()
    try:
        from PIL import Image
    except Exception:
        # No Pillow: send the file as it is.  Slower, but never silently crops or drops the frame.
        mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        encoded = base64.b64encode(raw).decode("ascii")
        return "data:%s;base64,%s" % (mime, encoded), len(raw)

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = image.size
    if width != height:
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    return "data:image/jpeg;base64,%s" % encoded, len(payload)


def parse_action(text: Any) -> Optional[Dict[str, Any]]:
    """Parse the reply into a dict, or None if it is not a usable action.

    Any code fence is stripped first, because several models wrap their JSON in ```json no matter what the
    prompt asks; that is cosmetic and must not be counted as an invalid reply.
    """
    if not text:
        return None
    cleaned = str(text).strip()
    import re

    cleaned = re.sub(r"(?m)^\s*(?:```|~~~)[a-zA-Z]*\s*$", "", cleaned).replace("```", "")
    candidates: List[str] = []
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(cleaned)):
            if cleaned[index] == "{":
                depth += 1
            elif cleaned[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start:index + 1])
                    break
        start = cleaned.find("{", start + 1)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------------- session

class Stage4Session:
    """One operator session: the window drives this object through the closed loop."""

    def __init__(self, sessions_root: str = None):
        self.sessions_root = sessions_root or os.path.join(ROOT, "stage4", "sessions")
        os.makedirs(self.sessions_root, exist_ok=True)

        self.session_id: Optional[str] = None
        self.dir: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.api_key: str = os.environ.get("BOYUE_API_KEY", "").strip()
        self.endpoint: str = os.environ.get("BOYUE_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")

        self.status = "IDLE"
        self.episode: Optional[Dict[str, Any]] = None      # current episode record
        self.episode_index = 0
        self.total_episodes = 0
        self.current_image: Optional[Dict[str, Any]] = None
        self.last_call: Optional[Dict[str, Any]] = None    # last model call, for the window
        self.history: List[Dict[str, Any]] = []            # steps of the current episode
        self.steps: List[Dict[str, Any]] = []              # every step of every episode this session
        self.episodes: List[Dict[str, Any]] = []
        self.memory: Optional[Dict[str, Any]] = None
        self.digest: Optional[Dict[str, Any]] = None
        # the session's own odometer, so the frame bank does not depend on what the operator typed
        self.position_x: float = 0.5
        self.position_z: float = 0.0
        self.torso_rotation: float = 0.0
        self.log_lines: List[str] = []

    # -- plumbing -------------------------------------------------------------------------------
    def _request_params(self) -> Dict[str, Any]:
        """The per-model extras actually sent, so the data records what the model was asked with.

        `disable_thinking` defaults to true because that is the only setting under which
        deepseek-v4-flash-vision-exp returns a parseable action at any token budget.  Setting it to false in
        the session config reproduces the models' default behaviour (hidden reasoning on), which is what a
        comparison against an earlier sweep would need -- the choice is written into config.json either way.
        """
        if not self.config.get("disable_thinking", True):
            return {}
        return dict(MODEL_REQUEST_PARAMS.get(self.config.get("model"), {}))

    def log(self, message: str) -> None:
        stamp = now_iso()
        self.log_lines.append("%s  %s" % (stamp, message))
        if len(self.log_lines) > 400:
            self.log_lines = self.log_lines[-400:]

    def _paths(self) -> Dict[str, str]:
        base = self.dir
        return {
            "config": os.path.join(base, "config.json"),
            "manifest": os.path.join(base, "manifest.csv"),
            "episodes": os.path.join(base, "episodes.jsonl"),
            "steps": os.path.join(base, "steps.jsonl"),
            "images": os.path.join(base, "images"),
            "prompts": os.path.join(base, "prompts"),
            "responses": os.path.join(base, "responses"),
            "exports": os.path.join(base, "exports"),
            "memory": os.path.join(base, "memory"),
        }

    def open_session(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create (or reopen) a session directory and write its config."""
        self.session_id = session_id.strip() or "session"
        self.dir = os.path.join(self.sessions_root, self.session_id)
        paths = self._paths()
        for key in ("images", "prompts", "responses", "exports", "memory"):
            os.makedirs(paths[key], exist_ok=True)

        a_s = float(config.get("a_s", 1.0))
        shoulder_m = float(config.get("shoulder_m") or ROBOT_SHOULDER_M)
        move_step = float(config.get("move_step") or MOVE_STEP)
        self.config = {
            "session_id": self.session_id,
            "scene": config.get("scene", "R1"),
            "marker": config.get("marker", "M1"),
            "marker_descriptor": config.get("marker_descriptor", "red circle"),
            "body_name": config.get("body_name") or DEFAULT_BODY_NAME,
            "mode": config.get("mode", "A"),
            "model": config.get("model", DEFAULT_MODELS[0]),
            "a_s": a_s,
            "robot_shoulder_m": shoulder_m,
            "move_step_m": move_step,
            "channel_width_m": channel_width_m(a_s, shoulder_m),
            "max_steps": int(config.get("max_steps", MAX_STEPS)),
            "image_size": IMAGE_SIZE,
            "memory_id": config.get("memory_id") or None,
            "disable_thinking": bool(config.get("disable_thinking", True)),
            "api_max_tokens": API_MAX_TOKENS,
            "request_params": self._request_params(),
            "sim_geometry": SIM_GEOMETRY,
            "started_at": now_iso(),
        }
        if config.get("api_key"):
            self.api_key = str(config["api_key"]).strip()
        if config.get("endpoint"):
            self.endpoint = str(config["endpoint"]).rstrip("/")

        with open(paths["config"], "w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, ensure_ascii=False)

        # reload previous work so reopening a session does not lose its history
        self.steps = read_jsonl(paths["steps"])
        self.episodes = read_jsonl(paths["episodes"])
        # a reopened session starts with no episode in flight: keeping the old one would block Start Episode
        # with "an episode is still open" after the operator had already closed that episode's window
        self.episode = None
        self.history = []
        self.current_image = None
        self.last_call = None
        self.memory = None
        self.digest = None
        self.episode_index = 0
        if not os.path.exists(paths["manifest"]):
            with open(paths["manifest"], "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow([
                    "episode_id", "mode", "model", "scene", "marker", "a_s", "channel_width_m",
                    "memory_id", "started_at", "finished_at", "passed", "total_steps",
                    "end_reason", "invalid_count",
                ])
        self.status = "READY"
        self.log("session opened: %s" % self.session_id)
        return self.state()

    def _require(self) -> None:
        if not self.session_id:
            raise RuntimeError("no session is open")

    # -- memory ---------------------------------------------------------------------------------
    def build_memory_from_episode(self, episode: Dict[str, Any], memory_id: str = None) -> Dict[str, Any]:
        """Turn a passed narrow-gap episode into the blind memory Stage 4B feeds back.

        Blind means: the action sequence, the per-step feedback and the model's own reasoning, and nothing
        that reveals the geometry -- no A/S, no channel width, no level, no "narrow"/"wide".  The source A/S
        stays in this record for the operator, and `prompt_text()` never renders it.
        """
        episode_id = episode.get("episode_id")
        steps = [s for s in self.steps if s.get("episode_id") == episode_id]
        actions = [s.get("action") for s in steps if s.get("action")]
        lines = []
        for index, step in enumerate(steps):
            if not step.get("action"):
                continue
            line = "%d. %s -> %s" % (index + 1, step["action"], step.get("feedback") or "executed")
            reasoning = (step.get("reasoning") or "").strip()
            if reasoning:
                line += "  |  %s" % reasoning[:180]
            lines.append(line)
        return {
            "memory_id": memory_id or new_id("S4M"),
            "source_episode_id": episode_id,
            "source_scene": episode.get("scene"),
            "source_marker": episode.get("marker"),
            "source_a_s": episode.get("a_s"),
            "actions": actions,
            "memory_text": "\n".join(lines),
            "created_at": now_iso(),
        }

    def load_memory_file(self, path: str) -> Dict[str, Any]:
        """Load a memory card written by an earlier stage (JSON or JSONL, first record used)."""
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
        if text.startswith("{"):
            record = json.loads(text.splitlines()[0])
        else:
            record = json.loads(text)
        record.setdefault("memory_id", new_id("S4M"))
        record.setdefault("memory_text", "")
        return record

    def memory_prompt_block(self) -> Optional[str]:
        if not self.memory:
            return None
        header = (
            "A previous successful traversal by the same expert in this same scene and marker:\n"
        )
        return header + str(self.memory.get("memory_text", "")).strip()

    # -- episode loop ---------------------------------------------------------------------------
    def start_episode(self) -> Dict[str, Any]:
        self._require()
        if self.episode and not self.episode.get("finished_at"):
            raise RuntimeError("an episode is still open; finish or stop it first")

        mode = self.config.get("mode", "A")
        set_marker_descriptor(self.config.get("marker_descriptor"))
        self.episode_index += 1
        self.episode = {
            "episode_id": new_id("S4E"),
            "run_id": new_id("S4R"),
            "mode": mode,
            "model": self.config.get("model"),
            "scene": self.config.get("scene"),
            "marker": self.config.get("marker"),
            "marker_descriptor": self.config.get("marker_descriptor"),
            "a_s": self.config.get("a_s"),
            "channel_width_m": self.config.get("channel_width_m"),
            "memory_id": (self.memory or {}).get("memory_id"),
            "started_at": now_iso(),
            "finished_at": None,
            "passed": None,
            "total_steps": 0,
            "end_reason": None,
            "invalid_count": 0,
            "digest": None,
        }
        self.history = []
        self.current_image = None
        self.last_call = None
        self.digest = None
        self.position_x = float(SIM_GEOMETRY["start_x_m"])
        self.position_z = 0.0
        self.torso_rotation = 0.0
        self.status = "READY"
        self.log("episode started: %s (mode %s, model %s, A/S %.1f)"
                 % (self.episode["episode_id"], mode, self.episode["model"], self.episode["a_s"]))
        return self.state()

    def state_block_placeholder(self) -> None:  # pragma: no cover - kept for symmetry with the docs
        return None

    def capture_from_inbox(self, inbox_dir: str) -> Dict[str, Any]:
        """Take the newest not-yet-used photo from a folder and make it the current frame.

        Sending a frame through a phone is slow enough that finding the file in a dialog 1,080 times is the
        real cost of the experiment, so the operator can drop the delivery folder's contents into one place and
        take them in order.  The set of already-used paths is persisted in config.json, so a restart does not
        re-import a frame that was already recorded.
        """
        if not inbox_dir or not os.path.isdir(inbox_dir):
            raise RuntimeError("收件箱目录不存在: %s" % inbox_dir)
        used = set(self.config.get("used_sources") or [])
        candidates = []
        for name in os.listdir(inbox_dir):
            if not name.lower().endswith(INBOX_EXTENSIONS):
                continue
            path = os.path.join(inbox_dir, name)
            if not os.path.isfile(path) or path in used:
                continue
            candidates.append((os.path.getmtime(path), path))
        if not candidates:
            raise RuntimeError("收件箱里没有未使用的新照片（%s）" % inbox_dir)
        candidates.sort()
        path = candidates[-1][1]
        with open(path, "rb") as handle:
            raw = handle.read()
        self.config.setdefault("used_sources", []).append(path)
        with open(self._paths()["config"], "w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, ensure_ascii=False)
        return self.capture(os.path.basename(path), raw, "inbox:" + path)

    def bank_frame_for_state(self, bank_root: str) -> str:
        """The one photograph that matches the current state, or a clear error naming what is missing.

        The x index comes from the session's own odometer (start_x + n * step), not from what the operator typed,
        so a forgotten position field cannot silently feed the wrong frame.
        """
        root = (bank_root or "").strip().strip('"')
        if not root or not os.path.isdir(root):
            raise RuntimeError("photo bank directory not found: %s" % (root or "(empty)"))
        last = self.history[-1] if self.history else None
        if last and last.get("action") == "look_down":
            glance = os.path.join(root, "_look_down", "look_down.PNG")
            if not os.path.isfile(glance):
                raise RuntimeError("missing downward-glance photograph: %s" % glance)
            return glance
        colour = BANK_MARKER_DIRS.get(self.config.get("marker_descriptor") or "")
        if colour is None:
            raise RuntimeError(
                "no bank folder is mapped to marker %r; known markers: %s"
                % (self.config.get("marker_descriptor"), sorted(BANK_MARKER_DIRS)))
        start = float(SIM_GEOMETRY["start_x_m"])
        step = float(self.config.get("move_step_m") or SIM_GEOMETRY["move_step_m"])
        index = int(round((self.position_x - start) / step))
        index = max(0, min(10, index))
        folder = "%s%s" % ("%.1f" % float(self.config.get("a_s")), colour)
        frame = os.path.join(root, folder, "%d.PNG" % (index + 1))
        if not os.path.isfile(frame):
            raise RuntimeError("missing forward frame: %s (folder %s)" % (frame, folder))
        return frame

    def capture_from_bank(self, bank_root: str) -> Dict[str, Any]:
        """Load the photograph the state calls for and make it the current frame."""
        frame = self.bank_frame_for_state(bank_root)
        with open(frame, "rb") as handle:
            raw = handle.read()
        # the bank is the archive, so record a pointer and a hash rather than a 7 MB duplicate per step
        state = self.capture(os.path.basename(frame), raw, "bank:" + frame,
                             store_copy=False, reference_path=frame)
        if self.current_image is not None:
            self.current_image["bank_frame"] = frame
        self.log("bank frame: %s (x=%.2f, torso=%.0f)" % (
            frame, self.position_x, self.torso_rotation))
        return state

    def capture(self, filename: str, raw_bytes: bytes, source_note: str = "",
                store_copy: bool = True, reference_path: str = "") -> Dict[str, Any]:
        """Store the operator's photograph and make it the current frame.

        The bytes are written exactly as received -- no crop, no resize, no annotation -- and only the
        API payload is re-encoded later.
        """
        self._require()
        if not self.episode:
            raise RuntimeError("start an episode before capturing")
        paths = self._paths()
        image_id = new_id("IMG")
        extension = os.path.splitext(filename or "frame.png")[1].lower() or ".png"
        stored_name = "%s%s" % (image_id, extension)
        if store_copy:
            stored_path = os.path.join(paths["images"], stored_name)
            with open(stored_path, "wb") as handle:
                handle.write(raw_bytes)
        else:
            # reference mode: the pixels stay in the photo bank, this record keeps the pointer and a hash
            stored_path = reference_path or ""

        import hashlib

        record = {
            "image_id": image_id,
            "path": (os.path.relpath(stored_path, self.dir).replace("\\", "/")
                     if store_copy else stored_path),
            "abs_path": stored_path,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "stored_copy": bool(store_copy),
            "captured_at": now_iso(),
            "bytes": len(raw_bytes),
            "source": source_note or "upload",
            "step": len(self.history),
            "episode_id": self.episode["episode_id"],
        }
        # Record the real pixel size.  The API payload is forced to 1024x1024, and a non-square frame would be
        # squashed to get there -- silently.  Storing the dimensions lets the window warn instead.
        try:
            from PIL import Image
            with Image.open(io.BytesIO(raw_bytes)) as picture:
                record["width"], record["height"] = picture.size
        except Exception:
            record["width"] = record["height"] = None
        self.current_image = record
        self.status = "READY"
        self.log("captured %s (%d bytes, %sx%s)" % (
            image_id, len(raw_bytes), record["width"], record["height"]))
        if record["width"] and (record["width"] != IMAGE_SIZE or record["height"] != IMAGE_SIZE):
            self.log("WARNING: frame is %sx%s, not %dx%d -- it will be resized for the API"
                     % (record["width"], record["height"], IMAGE_SIZE, IMAGE_SIZE))
        return self.state()

    def _state_block(self) -> Dict[str, Any]:
        """The `state` dict `build_prompt` expects.

        The origin is the start line: x counts metres walked forward from where the robot began, z counts
        metres to the left, so the numbers the operator types are the ones on the tape measure.  The prompt
        already explains x as "your forward position along the walkway", which is exactly this.

        The position comes from the session odometer, NOT from `history[-1]["position"]`: action_executed stores
        `position_x` / `position_z`, so reading the absent `position` key silently froze the prompt at the start
        coordinates for every step of every episode -- the agent walked without ever seeing its x change.  The
        odometer is the single source of truth, advanced either by action_executed or, in replay, by the caller.
        """
        last = self.history[-1] if self.history else {}
        return {
            # 0.5 m is the simulator's ROBOT_START_POS, not 0: the model must see the same absolute
            # coordinates it saw in Stages 1-3, so the numbers cannot be re-zeroed here.
            "position": [float(self.position_x), 0.0, float(self.position_z)],
            "torso_rotation": float(last.get("torso_rotation", self.torso_rotation)),
            "camera_yaw": 0.0,
            "camera_pitch": 0.0,
        }

    def _history_block(self) -> List[Dict[str, Any]]:
        return [
            {
                "step": item.get("step"),
                "action": item.get("action"),
                "feedback": item.get("feedback"),
                "reasoning": item.get("reasoning"),
            }
            for item in self.history
        ]

    def _apply_body_name(self) -> None:
        """Name the body that is actually in the corridor, exactly as Stage 3 did for G1 and R1.

        Only the opening noun phrase changes.  The sentence still contains "the red marker", so the existing
        marker substitution in build_prompt keeps replacing it, and every other word of the frozen Stage 1
        text is untouched.
        """
        import protocol

        body = (self.config.get("body_name") or DEFAULT_BODY_NAME).strip()
        protocol.TASK_INSTRUCTION = (
            "You are %s. Your task is to reach the red marker on the far wall." % body
        )

    def _action_options(self, move_step: float) -> str:
        """The action menu for the real scene: only the actions the frame bank can answer.

        build_prompt takes its options from the module-level ACTION_OPTIONS_STRING when move_step is None, and
        regenerates them from action_descriptions() otherwise.  Regenerating would lose the look_down override,
        so the string is always built here -- with the configured step length and the override -- and passed
        down by setting the module global.  Every other action's text is the frozen Stage 1 wording.
        """
        import protocol

        descriptions = dict(protocol.action_descriptions(move_step))
        descriptions["look_down"] = LOOK_DOWN_DESCRIPTION_REAL
        return "\n".join(
            '{"action": "%s"} - %s' % (name, descriptions[name]) for name in REAL_ACTION_SET
        )

    def _apply_real_scene_actions(self) -> None:
        """Restrict the offered actions to the five the photographs can answer.

        The frame note is replaced too, because the frozen one describes look_left/look_right and left/right,
        which are no longer in the menu; a note that explains absent actions is worse than no note.
        """
        import protocol

        protocol.ACTION_FRAME_NOTE = ACTION_FRAME_NOTE_REAL
        self.action_set = list(REAL_ACTION_SET)

    # Two sentences inside build_prompt's state block name actions the real scene no longer offers.  They are
    # rewritten after the prompt is built, with the expected text asserted first so a change in protocol.py
    # fails loudly instead of silently leaving a stale sentence in the prompt.
    REMOVED_ACTION_REWRITES = [
        (
            "z is how far to the right (negative) or left (positive) of the line you started on you are, and "
            "changes only when you use left/right",
            "z is how far to the right (negative) or left (positive) of the line you started on you are; it "
            "does not change in this task",
        ),
        (
            " (0 = looking straight ahead down your walking direction; NEGATIVE is toward your left and "
            "POSITIVE toward your right -- a look_left reports a negative number -- and the glance clears on "
            "your next action)",
            " (0 = looking straight ahead down your walking direction; you have no side glance in this task)",
        ),
    ]

    def _trim_removed_action_mentions(self, prompt: str) -> str:
        for old, new in self.REMOVED_ACTION_REWRITES:
            if old not in prompt:
                raise RuntimeError(
                    "prompt no longer contains a sentence that must be rewritten for the real-scene action "
                    "set: %r -- protocol.py changed, update REMOVED_ACTION_REWRITES" % old[:60])
            prompt = prompt.replace(old, new)
        return prompt

    def build_current_prompt(self, with_memory: bool = None) -> str:
        """The exact text that will be sent for the next action."""
        mode = self.config.get("mode", "A")
        use_memory = (mode == "B") if with_memory is None else with_memory
        self._apply_body_name()
        self._apply_real_scene_actions()
        # Set the marker here as well as in call_model, so a preview can never show a prompt that names a
        # different object from the one the next request will name.
        set_marker_descriptor(self.config.get("marker_descriptor"))
        memory_block = None
        if use_memory and self.memory:
            memory_block = self.memory_prompt_block()
            if self.digest:
                memory_block += (
                    "\n\nYour earlier digest of that memory:\n"
                    + json.dumps(self.digest, ensure_ascii=False)
                )
        # 0.75 m is the protocol's own step, and build_prompt already words the action list for it.  Passing
        # None in that case keeps the prompt byte-identical to Stages 1-3; a session that really walks a
        # different distance must say so, or the action descriptions would lie about what "forward" covers.
        step = float(self.config.get("move_step_m") or PROTOCOL_MOVE_STEP)
        # Always pass the options through the module global (move_step=None), because the look_down override
        # lives in that string; build_prompt would otherwise regenerate the frozen wording when the step
        # length differs from the protocol default.
        import protocol
        protocol.ACTION_OPTIONS_STRING = self._action_options(step)
        prompt = build_prompt(
            state=self._state_block(),
            history=self._history_block(),
            max_steps=self.config.get("max_steps", MAX_STEPS),
            move_step=None,
            memory_block=memory_block,
        )
        return self._trim_removed_action_mentions(prompt)

    def build_digest_prompt(self) -> str:
        """Digest call for Stage 4B: read the memory, choose no action, consume no step."""
        memory_text = self.memory_prompt_block() or ""
        return (
            memory_text
            + "\n\nBefore taking any action, read the memory and the current view. Do not choose an action "
              "yet. Return exactly one JSON object:\n"
              '{"digested": true, "memory_summary": "...", "plan": "...", "cues": ["..."]}'
        )

    # -- model call -----------------------------------------------------------------------------
    def _post(self, prompt: str, image_path: str, timeout: float) -> Dict[str, Any]:
        data_url, payload_bytes = encode_for_api(image_path, self.config.get("image_size", IMAGE_SIZE))
        body = {
            "model": self.config.get("model"),
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "temperature": API_TEMPERATURE,
            "max_tokens": API_MAX_TOKENS,
        }
        body.update(self._request_params())
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(body, allow_nan=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
            latency_ms = int((time.time() - started) * 1000)
            try:
                parsed = json.loads(raw)
                message = (parsed.get("choices") or [{}])[0].get("message") or {}
                choice = (parsed.get("choices") or [{}])[0]
                content = message.get("content") or ""
                reasoning_content = message.get("reasoning_content") or ""
                finish_reason = choice.get("finish_reason")
                usage = parsed.get("usage") or {}
                reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            except Exception:
                content, reasoning_content = raw, ""
                finish_reason, reasoning_tokens = None, None
            return {
                "ok": True, "raw": raw, "content": content,
                "reasoning_content": reasoning_content,
                "finish_reason": finish_reason, "reasoning_tokens": reasoning_tokens,
                "latency_ms": latency_ms, "payload_bytes": payload_bytes, "error": None,
            }
        except urllib.error.HTTPError as error:
            raw = ""
            try:
                raw = error.read().decode("utf-8", "replace")
            except Exception:
                pass
            return {
                "ok": False, "raw": raw, "content": "", "reasoning_content": "",
                "latency_ms": int((time.time() - started) * 1000), "payload_bytes": payload_bytes,
                "error": "HTTP %s" % error.code, "http_status": error.code,
            }
        except Exception as error:  # noqa: BLE001 -- timeout, DNS, TLS, all recorded verbatim
            return {
                "ok": False, "raw": "", "content": "", "reasoning_content": "",
                "latency_ms": int((time.time() - started) * 1000), "payload_bytes": payload_bytes,
                "error": "%s: %s" % (type(error).__name__, error), "http_status": None,
            }

    def _call_with_retries(self, prompt: str, image_path: str, prompt_name: str) -> Dict[str, Any]:
        """Call once, retry per the spec, and keep every attempt.

        A network failure (timeout / 5xx) is retried up to NETWORK_RETRIES times; a reply that arrives but
        cannot be parsed into a legal action is retried FORMAT_RETRIES times with the same prompt.  If it
        still fails the step is recorded as invalid with `action: null` -- never guessed.
        """
        paths = self._paths()
        attempts: List[Dict[str, Any]] = []
        timeout = float(self.config.get("timeout_s", DEFAULT_TIMEOUT_S))
        deadline_calls = NETWORK_RETRIES + FORMAT_RETRIES + 1

        for attempt_index in range(deadline_calls):
            result = self._post(prompt, image_path, timeout)
            parsed = parse_action(result.get("content") or "") if result.get("ok") else None
            action = None
            if parsed:
                candidate = str(parsed.get("action") or "").strip()
                if candidate in getattr(self, "action_set", ACTIONS):
                    action = candidate
            attempt = {
                "attempt": attempt_index + 1,
                "started_at": now_iso(),
                "latency_ms": result.get("latency_ms"),
                "payload_bytes": result.get("payload_bytes"),
                "finish_reason": result.get("finish_reason"),
                "reasoning_tokens": result.get("reasoning_tokens"),
                "error": result.get("error"),
                "raw_response": result.get("raw"),
                "content": result.get("content"),
                "reasoning_content": result.get("reasoning_content"),
                "parsed": parsed,
                "action": action,
            }
            # A truncated reply is worth naming exactly: it is the one failure whose cause is a token budget
            # rather than the model's choice, and the operator can raise max_tokens instead of retrying blind.
            if result.get("finish_reason") == "length" and not action:
                attempt["error"] = "truncated (finish_reason=length, reasoning_tokens=%s)" % (
                    result.get("reasoning_tokens"))
            attempts.append(attempt)
            if action:
                return {"ok": True, "attempt": attempt, "attempts": attempts, "parsed": parsed}

            network_failure = (not result.get("ok")) or (
                result.get("http_status") and int(result["http_status"]) >= 500)
            if network_failure:
                if attempt_index >= NETWORK_RETRIES:
                    break
            else:
                # The reply arrived but was unusable: at most FORMAT_RETRIES extra tries with the same prompt.
                if attempt_index >= NETWORK_RETRIES + FORMAT_RETRIES:
                    break
            self.log("attempt %d unusable (%s); retrying" % (attempt_index + 1, result.get("error") or "bad reply"))

        return {"ok": False, "attempt": attempts[-1] if attempts else None,
                "attempts": attempts, "parsed": None}

    def call_model(self) -> Dict[str, Any]:
        """Send the next-step prompt and record the step (feedback filled in later)."""
        self._require()
        if not self.episode:
            raise RuntimeError("start an episode first")
        if not self.current_image:
            raise RuntimeError("capture a frame before calling the model")
        if len(self.history) >= int(self.config.get("max_steps", MAX_STEPS)):
            raise RuntimeError("step limit reached; finish the episode")

        paths = self._paths()
        set_marker_descriptor(self.config.get("marker_descriptor"))
        prompt = self.build_current_prompt()
        step_index = len(self.history)

        prompt_id = new_id("PRM")
        prompt_file = os.path.join(paths["prompts"], "%s.txt" % prompt_id)
        with open(prompt_file, "w", encoding="utf-8") as handle:
            handle.write(prompt)

        self.status = "CALLING_MODEL"
        call = self._call_with_retries(prompt, self.current_image["abs_path"], prompt_id)

        response_id = new_id("RSP")
        response_file = os.path.join(paths["responses"], "%s.json" % response_id)
        with open(response_file, "w", encoding="utf-8") as handle:
            json.dump(call.get("attempts", []), handle, indent=2, ensure_ascii=False)

        attempt = call.get("attempt") or {}
        parsed = call.get("parsed") or {}
        action = attempt.get("action")
        invalid = not bool(action)
        if invalid:
            self.episode["invalid_count"] += 1

        record = {
            "episode_id": self.episode["episode_id"],
            "step": step_index,
            "call_index": len(self.history) + 1,
            "model": self.config.get("model"),
            "a_s": self.config.get("a_s"),
            "image_id": self.current_image["image_id"],
            "image_path": self.current_image["path"],
            "prompt_path": os.path.relpath(prompt_file, self.dir).replace("\\", "/"),
            "response_path": os.path.relpath(response_file, self.dir).replace("\\", "/"),
            "action": action,
            "confidence": (parsed.get("confidence") if isinstance(parsed, dict) else None),
            "reasoning": (parsed.get("reasoning") if isinstance(parsed, dict) else None),
            "scene_description": (parsed.get("scene_description") if isinstance(parsed, dict) else None),
            "reasoning_content": attempt.get("reasoning_content") or "",
            "raw_response": attempt.get("raw_response") or "",
            "attempts": len(call.get("attempts", [])),
            "feedback": None,
            "collision": None,
            "position_x": None,
            "position_z": None,
            "torso_rotation": self._state_block().get("torso_rotation"),
            "latency_ms": attempt.get("latency_ms"),
            "payload_bytes": attempt.get("payload_bytes"),
            "finish_reason": attempt.get("finish_reason"),
            "reasoning_tokens": attempt.get("reasoning_tokens"),
            "request_params": self._request_params(),
            "max_tokens": API_MAX_TOKENS,
            "move_step_m": self.config.get("move_step_m"),
            "robot_shoulder_m": self.config.get("robot_shoulder_m"),
            "channel_width_m": self.config.get("channel_width_m"),
            "invalid": invalid,
            "error": attempt.get("error"),
            "timestamp": now_iso(),
        }
        append_jsonl(paths["steps"], record)
        self.steps.append(record)
        self.history.append(record)
        self.episode["total_steps"] = len(self.history)
        self.last_call = {"record": record, "attempts": call.get("attempts", []), "prompt": prompt}
        self.status = "INVALID" if invalid else "WAITING_FOR_ACTION"
        self.log("step %d: %s (%.1f s)%s" % (
            step_index, action or "INVALID", (record["latency_ms"] or 0) / 1000.0,
            "" if not invalid else " -- " + str(record.get("error"))))
        return self.state()

    def action_executed(
        self,
        feedback: str,
        collision: bool = False,
        position_x: Optional[float] = None,
        position_z: Optional[float] = None,
        torso_rotation: Optional[float] = None,
        passed: Optional[bool] = None,
        note: str = "",
        advance_odometer: bool = True,
    ) -> Dict[str, Any]:
        """Record what the operator did with the action, and optionally mark the episode finished."""
        self._require()
        if not self.history:
            raise RuntimeError("there is no pending step")
        record = self.history[-1]
        if record.get("feedback"):
            raise RuntimeError("this step is already recorded")

        # Advance the odometer from the action itself, so the window's frame bank follows the real sequence.
        # A caller that simulates the geometry (the replay runner) owns x and theta and passes False, otherwise
        # the position would advance twice -- once there and once here -- and the photographs would run ahead
        # of the state the model is told.
        if advance_odometer:
            step = float(self.config.get("move_step_m") or SIM_GEOMETRY["move_step_m"])
            action = record.get("action")
            if action == "forward":
                self.position_x += step
            elif action == "backward":
                self.position_x -= step
            if action == "turn_left":
                self.torso_rotation = (self.torso_rotation + 15.0) % 360.0
            elif action == "turn_right":
                self.torso_rotation = (self.torso_rotation - 15.0) % 360.0

        record["feedback"] = feedback or "executed"
        record["collision"] = bool(collision)
        if position_x is not None:
            record["position_x"] = float(position_x)
        else:
            record["position_x"] = round(self.position_x, 3)
        if position_z is not None:
            record["position_z"] = float(position_z)
        if torso_rotation is not None:
            record["torso_rotation"] = float(torso_rotation)
        record["operator_note"] = note or ""
        record["recorded_at"] = now_iso()
        self._rewrite_steps()
        self.status = "ACTION_DONE"

        if passed is True:
            self.finish(passed=True, end_reason="passed")
        elif passed is False:
            self.finish(passed=False, end_reason="failed")
        elif len(self.history) >= int(self.config.get("max_steps", MAX_STEPS)):
            self.finish(passed=False, end_reason="max_steps")
        return self.state()

    def _rewrite_steps(self) -> None:
        """Rewrite steps.jsonl after an edit, so the file on disk always matches memory.

        The spec asks for one append-only log; an operator editing the last step is the one case where the
        line has to be replaced, and appending a duplicate would make every later count wrong.
        """
        paths = self._paths()
        with open(paths["steps"], "w", encoding="utf-8") as handle:
            for row in self.steps:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- digest (Stage 4B) ----------------------------------------------------------------------
    def run_digest(self) -> Dict[str, Any]:
        """Ask the model to read the memory before the loop; consumes no step."""
        self._require()
        if self.config.get("mode") != "B":
            raise RuntimeError("digest only runs in mode B")
        if not self.memory:
            raise RuntimeError("no memory selected for this episode")
        if not self.current_image:
            raise RuntimeError("capture the first frame before the digest")
        prompt = self.build_digest_prompt()
        paths = self._paths()
        prompt_file = os.path.join(paths["prompts"], "%s_digest.txt" % new_id("PRM"))
        with open(prompt_file, "w", encoding="utf-8") as handle:
            handle.write(prompt)
        result = self._post(prompt, self.current_image["abs_path"],
                            float(self.config.get("timeout_s", DEFAULT_TIMEOUT_S)))
        parsed = parse_action(result.get("content") or "") if result.get("ok") else None
        valid = bool(parsed and parsed.get("digested") in (True, "true", 1))
        self.digest = {
            "valid": valid,
            "raw": (parsed if valid else None),
            "content": result.get("content"),
            "error": result.get("error"),
            "latency_ms": result.get("latency_ms"),
            "prompt_path": os.path.relpath(prompt_file, self.dir),
            "created_at": now_iso(),
        }
        if self.episode:
            self.episode["digest"] = self.digest
        self.log("digest %s (%.1f s)" % ("ok" if valid else "INVALID",
                                         (result.get("latency_ms") or 0) / 1000.0))
        return self.state()

    # -- finishing and export -------------------------------------------------------------------
    def finish(self, passed: bool, end_reason: str) -> Dict[str, Any]:
        self._require()
        if not self.episode:
            raise RuntimeError("no episode is open")
        self.episode["finished_at"] = now_iso()
        self.episode["passed"] = bool(passed)
        self.episode["end_reason"] = end_reason or ("passed" if passed else "failed")
        self.episode["total_steps"] = len(self.history)
        self.episode["steps"] = len(self.history)
        append_jsonl(self._paths()["episodes"], self.episode)
        self._append_manifest()
        self.episodes.append(self.episode)
        self.episode = None
        self.status = "PASSED" if passed else "FAILED"
        self.log("episode finished: %s" % ("passed" if passed else "failed"))
        return self.state()

    def _append_manifest(self) -> None:
        paths = self._paths()
        episode = self.episode or {}
        with open(paths["manifest"], "a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([
                episode.get("episode_id"), episode.get("mode"), episode.get("model"),
                episode.get("scene"), episode.get("marker"), episode.get("a_s"),
                episode.get("channel_width_m"), episode.get("memory_id"),
                episode.get("started_at"), episode.get("finished_at"),
                episode.get("passed"), episode.get("total_steps"),
                episode.get("end_reason"), episode.get("invalid_count"),
            ])

    def stop(self) -> Dict[str, Any]:
        if self.episode:
            self.finish(passed=False, end_reason="stopped")
        else:
            self.status = "STOPPED"
        return self.state()

    def summary(self) -> List[Dict[str, Any]]:
        """Per-model and per-A/S aggregates for the window's summary panel."""
        finished = [e for e in self.episodes if e.get("finished_at")]
        rows: List[Dict[str, Any]] = []
        for key_name in ("model", "a_s"):
            values = sorted({str(e.get(key_name)) for e in finished})
            for value in values:
                group = [e for e in finished if str(e.get(key_name)) == value]
                if not group:
                    continue
                turns = 0
                invalid = 0
                for episode in group:
                    for step in self.steps:
                        if step.get("episode_id") == episode.get("episode_id"):
                            turns += 1 if str(step.get("action", "")).startswith("turn_") else 0
                            invalid += 1 if step.get("invalid") else 0
                rows.append({
                    "by": key_name,
                    "value": value,
                    "episodes": len(group),
                    "passed": sum(1 for e in group if e.get("passed")),
                    "mean_steps": round(sum(e.get("total_steps") or 0 for e in group) / len(group), 2),
                    "turns": turns,
                    "invalid": invalid,
                })
        return rows

    def export(self) -> Dict[str, Any]:
        """Write the CSVs the spec lists, next to the raw logs."""
        self._require()
        paths = self._paths()
        os.makedirs(paths["exports"], exist_ok=True)

        steps_csv = os.path.join(paths["exports"], "steps.csv")
        fields = [
            "episode_id", "step", "call_index", "model", "a_s", "image_id", "action", "confidence",
            "feedback", "collision", "position_x", "position_z", "torso_rotation", "latency_ms",
            "invalid", "error", "attempts", "reasoning", "scene_description",
        ]
        with open(steps_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.steps:
                writer.writerow(row)

        episodes_csv = os.path.join(paths["exports"], "episodes.csv")
        episode_fields = [
            "episode_id", "mode", "model", "scene", "marker", "a_s", "channel_width_m", "memory_id",
            "started_at", "finished_at", "passed", "total_steps", "end_reason", "invalid_count",
        ]
        with open(episodes_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=episode_fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.episodes:
                writer.writerow(row)

        summary_csv = os.path.join(paths["exports"], "summary.csv")
        rows = self.summary()
        with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["by", "value", "episodes", "passed", "mean_steps", "turns", "invalid"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        self.log("exported to %s" % paths["exports"])
        return {"ok": True, "directory": paths["exports"],
                "files": [steps_csv, episodes_csv, summary_csv], "summary": rows,
                "state": self.state()}

    # -- window state ---------------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        episode = self.episode or {}
        last = self.last_call or {}
        record = last.get("record") or {}
        return {
            "status": self.status,
            "session_id": self.session_id,
            "session_dir": self.dir,
            "config": self.config,
            "sim_geometry": SIM_GEOMETRY,
            "geometry": {
                "robot_shoulder_m": self.config.get("robot_shoulder_m", ROBOT_SHOULDER_M),
                "move_step_m": self.config.get("move_step_m", MOVE_STEP),
                "a_s": self.config.get("a_s"),
                "channel_width_m": self.config.get("channel_width_m"),
                "channel_width_cm": (round(float(self.config.get("channel_width_m") or 0) * 100, 1)
                                     if self.config.get("channel_width_m") is not None else None),
                "table": [
                    {"a_s": level,
                     "channel_width_cm": round(level * float(
                         self.config.get("robot_shoulder_m", ROBOT_SHOULDER_M)) * 100, 1)}
                    for level in A_S_LEVELS
                ],
            },
            "models": DEFAULT_MODELS,
            "a_s_levels": A_S_LEVELS,
            "actions": ACTIONS,
            "episode": {
                "episode_id": episode.get("episode_id"),
                "mode": episode.get("mode") or self.config.get("mode"),
                "model": episode.get("model") or self.config.get("model"),
                "scene": episode.get("scene") or self.config.get("scene"),
                "marker": episode.get("marker") or self.config.get("marker"),
                "a_s": episode.get("a_s") if episode else self.config.get("a_s"),
                "channel_width_m": episode.get("channel_width_m") if episode else self.config.get("channel_width_m"),
                "index": self.episode_index if episode else None,
                "step": len(self.history),
                "max_steps": self.config.get("max_steps", MAX_STEPS),
                "invalid_count": episode.get("invalid_count", 0),
            },
            "image": ({
                "image_id": self.current_image.get("image_id"),
                "captured_at": self.current_image.get("captured_at"),
                "path": self.current_image.get("path"),
                "bytes": self.current_image.get("bytes"),
                "width": self.current_image.get("width"),
                "height": self.current_image.get("height"),
                "source": self.current_image.get("source"),
                "expected_size": IMAGE_SIZE,
                "size_ok": (self.current_image.get("width") == IMAGE_SIZE
                            and self.current_image.get("height") == IMAGE_SIZE),
            } if self.current_image else None),
            "last_call": ({
                "action": record.get("action"),
                "confidence": record.get("confidence"),
                "latency_ms": record.get("latency_ms"),
                "invalid": record.get("invalid"),
                "error": record.get("error"),
                "reasoning": record.get("reasoning"),
                "reasoning_content": record.get("reasoning_content"),
                "scene_description": record.get("scene_description"),
                "raw_response": record.get("raw_response"),
                "attempts": record.get("attempts"),
                "payload_bytes": record.get("payload_bytes"),
                "finish_reason": record.get("finish_reason"),
                "reasoning_tokens": record.get("reasoning_tokens"),
                "prompt": last.get("prompt"),
                "pending_feedback": not bool(record.get("feedback")),
            } if record else None),
            "history": [
                {
                    "step": item.get("step"), "action": item.get("action"),
                    "feedback": item.get("feedback"), "collision": item.get("collision"),
                    "position_x": item.get("position_x"), "position_z": item.get("position_z"),
                    "torso_rotation": item.get("torso_rotation"),
                    "latency_ms": item.get("latency_ms"), "invalid": item.get("invalid"),
                    "reasoning": item.get("reasoning"), "prompt_path": item.get("prompt_path"),
                    "raw_response": item.get("raw_response"),
                }
                for item in self.history
            ],
            "memory": self.memory,
            "digest": self.digest,
            "episodes_finished": len(self.episodes),
            "summary": self.summary(),
            "log": self.log_lines[-60:],
            "api_key_set": bool(self.api_key),
            "endpoint": self.endpoint,
        }
