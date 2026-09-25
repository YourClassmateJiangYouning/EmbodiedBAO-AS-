"""Unified multimodal LLM interface for EmbodiedBAO.

All models are called through a single OpenAI-compatible endpoint (for example
an API aggregator such as Taotoken or a lab inference server), so switching
models is just a matter of passing a different ``model_name``.  The endpoint
and key are read from environment variables, in priority order:

    API key   : BOYUE_API_KEY > TAOTOKEN_API_KEY > OPENAI_API_KEY
    base URL  : BOYUE_BASE_URL > TAOTOKEN_BASE_URL > OPENAI_BASE_URL

When only ``BOYUE_API_KEY`` is set, the lab server default
``http://35.220.164.252:3888/v1/`` is used automatically.

The action space and the task prompt live in ``protocol.py`` so that this
client and the experiment runner can never disagree about them.  A ``random``
model is also provided as a baseline, matching MirrorBench's ``AgentRandom``.

Reference: MirrorBench ``agent.py`` (base64 image encoding, OpenAI client,
temperature 0, retry loop).

Per-model request overrides
---------------------------
Some models spend most of a call thinking before they answer, which is the whole
cost of a sweep.  deepseek-v4.1-flash measured 86.8 s and 12,820 tokens per call
at the default setting, against 20.1 s and 1,224 tokens with
``reasoning_effort="none"`` (tools/probe_reasoning.py, real prompt + 512 px frame,
n=2 per variant).  ``MODEL_REQUEST_PARAMS`` records such overrides per model name
so that a run cannot silently use a different configuration from the one it
reports: the effective parameters are logged into the run's args.json.

The gateway answers HTTP 200 for every unknown parameter, so a key being accepted
proves nothing -- only the token count and the latency show whether it had an
effect.  That is why the probe measures both.
"""

from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import random
import re
import time
import traceback
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environment import ACTIONS
from protocol import (
    ACTION_NAMES_TEXT,
    ACTION_OPTIONS_STRING,
    SYSTEM_PROMPT,
)


SUPPORTED_MODELS: List[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3.5-sonnet",
    "gemini-2.5-pro",
    "qwen-vl-max",
    "qwen2.5-vl-72b",
    "qwen2.5-vl-7b",
    "internvl3.5-4b",
    "llava-1.6-7b",
]

MODEL_ALIASES: Dict[str, str] = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4omini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude3.5sonnet": "claude-3.5-sonnet",
    "claude-3.5-sonnet": "claude-3.5-sonnet",
    "gemini2.5pro": "gemini-2.5-pro",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "qwenvlmax": "qwen-vl-max",
    "qwen-vl-max": "qwen-vl-max",
    "qwen2.5vl72b": "qwen2.5-vl-72b",
    "qwen2.5-vl-72b": "qwen2.5-vl-72b",
    "qwen2.5vl7b": "qwen2.5-vl-7b",
    "qwen2.5-vl-7b": "qwen2.5-vl-7b",
    "internvl3.54b": "internvl3.5-4b",
    "internvl3.5-4b": "internvl3.5-4b",
    "llava1.67b": "llava-1.6-7b",
    "llava-1.6-7b": "llava-1.6-7b",
}

BOYUE_DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1/"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Extra request parameters per model, keyed by the name the gateway is given.
# Empty unless a measurement in tools/probe_reasoning.py justified an entry: this
# changes what the model DOES, not just how fast it runs, so an entry belongs here
# only with the numbers next to it.
#
#   deepseek-v4.1-flash: default 86.8 s / 12,820 tokens per call, versus 20.1 s /
#   1,224 tokens with reasoning_effort="none" (n=2, real prompt + 512 px frame).
#   Without it a 60-episode run is ~43 h; see the probe's table for the variants.
MODEL_REQUEST_PARAMS: Dict[str, Dict[str, Any]] = {
    "deepseek-v4.1-flash": {"reasoning_effort": "none"},
    # glm-4.6v honours the Zhipu spelling: 12.9 s / 1,595 tokens at the default,
    # 6.4 s / 1,397 tokens with thinking disabled (n=4 each, same probe).
    "glm-4.6v": {"thinking": {"type": "disabled"}},
}


def request_params_for(model_name: str) -> Dict[str, Any]:
    """Extra request parameters for one model.

    Merged from MODEL_REQUEST_PARAMS and an optional ``BAO_MODEL_PARAMS`` JSON
    environment override, so an experiment can be re-run under a different
    configuration without editing code -- and, because the runner records the
    result in args.json, without the report disagreeing with what was sent.
    """
    params: Dict[str, Any] = dict(MODEL_REQUEST_PARAMS.get(model_name, {}))
    raw = os.environ.get("BAO_MODEL_PARAMS")
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                extra = override.get(model_name)
                if isinstance(extra, dict):
                    params.update(extra)
        except ValueError as exc:
            print(f"[ai_agent] BAO_MODEL_PARAMS is not valid JSON ({exc}); ignored")
    return params


def disable_proxy() -> None:
    """Remove proxy environment variables (mirrors utils.disable_proxy)."""
    for name in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(name, None)


def normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower().replace("_", "").replace(" ", "")
    return MODEL_ALIASES.get(key, str(model_name).strip())


def encode_image(image: Any) -> str:
    """Encode a numpy RGB image (or image path) as a base64 JPEG data URL.

    Honours ``BAO_IMAGE_SIZE``: when set, the frame is downscaled to that square
    size before encoding.  A recorded 1024x1024 run averaged 31 s per model call
    and lost 9 of its 30 steps to timeouts, so being able to shrink the payload
    without a code change matters for throughput.
    """
    if isinstance(image, str):
        with open(image, "rb") as handle:
            data = base64.b64encode(handle.read()).decode("utf-8")
        mime_type = mimetypes.guess_type(image)[0] or "application/octet-stream"
        if not mime_type.startswith("image/"):
            raise ValueError(f"image path has an unsupported media type: {image}")
        return f"data:{mime_type};base64,{data}"
    from PIL import Image

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = Image.fromarray(image).convert("RGB")
    else:
        image = Image.fromarray(image)

    target = os.environ.get("BAO_IMAGE_SIZE", "").strip()
    if target:
        try:
            side = max(64, int(float(target)))
            if image.size != (side, side):
                if side > max(image.size):
                    # Upscaling adds no information and only costs tokens, so say
                    # so rather than letting it look like a resolution increase.
                    print(
                        f"[ai_agent] BAO_IMAGE_SIZE={side} exceeds the camera's "
                        f"{image.size[0]}px; upsampling adds no detail. Raise the "
                        f"camera resolution instead.",
                        flush=True,
                    )
                image = image.resize((side, side), Image.LANCZOS)
        except ValueError:
            pass

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def parse_action_json(text: Any) -> Optional[Dict[str, Any]]:
    """Parse the model response into {"action", "confidence", "reasoning"}."""
    if text is None:
        return None
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        return None
    try:
        data, _end = json.JSONDecoder(
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            )
        ).raw_decode(cleaned[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if not isinstance(action, str) or action not in ACTIONS:
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    scene = data.get("scene_description", "")
    reasoning = data.get("reasoning", "")
    return {
        "action": action,
        "confidence": confidence,
        "reasoning": str(reasoning),
        "scene_description": str(scene),
    }


def parse_action_text(text: Any) -> Optional[str]:
    """Fallback parser for non-JSON responses (e.g. 'Choice: [3]')."""
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    match = re.search(r"Choice:?[\n\s]*\[?(\d+)\]?", cleaned, re.IGNORECASE)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(ACTIONS):
            return ACTIONS[index - 1]
    lowered = cleaned.lower()
    for action in sorted(ACTIONS, key=len, reverse=True):
        if action in lowered:
            return action
    return None


def build_prompt(context: Optional[Dict[str, Any]] = None) -> str:
    """Build the prompt for one decision step.

    Delegates to ``protocol.build_prompt`` so there is exactly one prompt in
    the project.  ``context`` may carry ``position``, ``torso_rotation`` (or
    ``yaw``), ``camera_yaw``, ``history`` and ``max_steps``.  The channel geometry
    is deliberately never part of the prompt.

    ``camera_yaw`` is forwarded because the gaze is pinned to the walking
    direction and look_left/look_right offset it: dropping the field would omit
    the one line that tells the agent where it is looking, and nothing in a
    single frame reveals it.  (Nothing calls this wrapper today; it exists as the
    documented entry point, and it is fixed rather than deleted because a caller
    silently losing that line is exactly the failure it is meant to prevent.)
    """
    from protocol import build_prompt as _build_prompt

    context = dict(context or {})
    if "torso_rotation" not in context and "yaw" in context:
        context["torso_rotation"] = context["yaw"]
    state = {
        "position": context.get("position"),
        "torso_rotation": context.get("torso_rotation"),
        "camera_yaw": context.get("camera_yaw"),
    }
    return _build_prompt(
        state=state,
        history=context.get("history"),
        max_steps=int(context.get("max_steps", 30)),
    )


class _OpenAICompatMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _OpenAICompatChoice:
    def __init__(self, content: str) -> None:
        self.message = _OpenAICompatMessage(content)


class _OpenAICompatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_OpenAICompatChoice(content)]


class _OpenAICompatCompletions:
    """Minimal chat.completions implementation using only the standard library."""

    def __init__(self, client: "_OpenAICompatClient") -> None:
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> _OpenAICompatResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # Per-model overrides (e.g. thinking off) must reach BOTH request paths:
        # this one is used when the openai package is missing, so applying them
        # only to the openai client would make the model's behaviour depend on the
        # environment the sweep happens to run in.
        payload.update(request_params_for(model))
        request = urllib.request.Request(
            self._client._base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._client._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._client._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code}: {exc.reason}; body={body[:2000]}"
            ) from exc
        content = data["choices"][0]["message"]["content"]
        return _OpenAICompatResponse(content)


class _OpenAICompatChat:
    def __init__(self, client: "_OpenAICompatClient") -> None:
        self.completions = _OpenAICompatCompletions(client)


class _OpenAICompatClient:
    def __init__(self, api_key: str, base_url: str, timeout: float) -> None:
        self._api_key = api_key
        self._base_url = str(base_url).rstrip("/")
        self._timeout = timeout
        self.chat = _OpenAICompatChat(self)


class AgentAPI:
    """OpenAI-compatible client for all supported MLLMs."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.0,
        max_retries: Optional[int] = None,
        use_json_mode: bool = False,
        log_file: Optional[str] = None,
    ) -> None:
        self.model_name = normalize_model_name(model_name)
        if self.model_name not in SUPPORTED_MODELS:
            print(f"[ai_agent] Warning: model '{self.model_name}' is not in the known list.")
        self.api_key = (
            api_key
            or os.environ.get("BOYUE_API_KEY")
            or os.environ.get("TAOTOKEN_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "No API key found. Set BOYUE_API_KEY, TAOTOKEN_API_KEY, or "
                "OPENAI_API_KEY in the environment."
            )
        if not self.api_key.isascii():
            raise ValueError(
                "API key contains non-ASCII characters. Replace the placeholder "
                "'sk-your-real-token' with the actual key and export it again."
            )
        env_base_url = (
            os.environ.get("BOYUE_BASE_URL")
            or os.environ.get("TAOTOKEN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        if base_url:
            self.base_url = base_url
        elif env_base_url:
            self.base_url = env_base_url
        elif os.environ.get("BOYUE_API_KEY"):
            self.base_url = BOYUE_DEFAULT_BASE_URL
        else:
            self.base_url = OPENAI_DEFAULT_BASE_URL
        # 90 s rather than 60: a recorded run lost 9 of 30 steps to timeouts at
        # the old default, each costing a full retry cycle.
        self.timeout = (
            float(timeout)
            if timeout is not None
            else float(os.environ.get("BAO_LLM_TIMEOUT", "90"))
        )
        if not math.isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError(f"timeout must be a positive finite number, got {self.timeout}")
        self.temperature = temperature
        # 3 rather than 1: the gateway intermittently returns 502 and has been
        # measured swinging between 1.8 s and 55.8 s for identical requests, so a
        # single retry leaves too many steps recorded as 'invalid'.  That count is
        # not a property of the model, and it pollutes the action statistics.
        retry_env = os.environ.get("BAO_MAX_RETRIES", "3")
        self.max_retries = (
            int(max_retries) if max_retries is not None else int(retry_env)
        )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        self.use_json_mode = bool(use_json_mode)
        self._json_mode_enabled = self.use_json_mode
        self.log_file = log_file

        if os.environ.get("BAO_DISABLE_PROXY", "0") == "1":
            disable_proxy()

        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        except Exception as exc:
            print(
                f"[ai_agent] openai client unavailable ({exc}); "
                "using built-in HTTP client"
            )
            self.client = _OpenAICompatClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_action(
        self,
        image: Optional[np.ndarray],
        prompt: str,
        state: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        options: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {"action", "confidence", "reasoning"} for the current view.

        ``history``/``state``/``options`` exist so the documented call signature
        stays stable for callers that pass them, but this client does **not**
        inject them: the within-episode memory is already rendered into
        ``prompt`` by ``protocol.build_prompt``, and adding it twice would
        duplicate the block.  Pass the history to the prompt builder, not here.
        """
        messages = self._build_messages(image, prompt)
        repair_hint = (
            "Your previous output was not valid. Respond with exactly one JSON object: "
            f'{{"scene_description": "what you see", '
            f'"reasoning": "short text", "action": "one of {ACTION_NAMES_TEXT}", '
            '"confidence": 0.0-1.0}.'
        )

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                messages = messages + [{"role": "user", "content": repair_hint}]
                self._log(f"RETRY {attempt}/{self.max_retries}: {last_error}")
                time.sleep(min(2 ** (attempt - 1), 4))
            try:
                response = self._request(messages)
                parsed = parse_action_json(response)
                if parsed is not None:
                    action_name = parsed.get("action")
                    scene = parsed.get("scene_description")
                    reasoning = parsed.get("reasoning")
                    if action_name in ACTIONS:
                        if not (isinstance(scene, str) and scene.strip()):
                            self._log(
                                "MODEL RESPONSE (no scene_description, accepted):\n"
                                + str(response)
                            )
                        else:
                            self._log(f"MODEL RESPONSE:\n{response}")
                        if isinstance(reasoning, str) and reasoning.strip():
                            parsed.setdefault("reasoning", reasoning)
                        return parsed
                    self._log(
                        "MODEL RESPONSE (missing scene_description or action):\n"
                        + str(response)
                    )
                    last_error = "response missing scene_description or a valid action"
                    continue
                self._log(f"MODEL RESPONSE (unparseable):\n{response}")
                last_error = "response was not a valid JSON or action text"
            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=8)
                )
                self._log(f"REQUEST ERROR: {last_error}")

        self._log(f"FALLBACK (no valid response after {self.max_retries + 1} attempts)")
        return {
            "action": None,
            "confidence": 0.0,
            "reasoning": f"failed to obtain a valid action after retries: {last_error}",
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self, image: Optional[np.ndarray], prompt: str
    ) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(image)},
                }
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _request(self, messages: List[Dict[str, Any]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        # Same override as the compat path above; see request_params_for.
        kwargs.update(request_params_for(self.model_name))
        if self._json_mode_enabled:
            try:
                completion = self.client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
                return completion.choices[0].message.content or ""
            except Exception:
                # Some aggregator endpoints reject response_format. Disable it
                # for the rest of the process instead of retrying JSON mode on
                # every request.
                self._json_mode_enabled = False
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def _log(self, text: str) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


class RandomAgent:
    """Random baseline, mirroring MirrorBench's AgentRandom."""

    def get_action(
        self,
        image: Optional[np.ndarray] = None,
        prompt: str = "",
        state: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        options: Optional[str] = None,
    ) -> Dict[str, Any]:
        action = random.choice(ACTIONS)
        return {
            "action": action,
            "confidence": 1.0 / len(ACTIONS),
            "reasoning": "random baseline",
        }


class AIAgent(AgentAPI):
    """Alias used by main.py: AIAgent(model=..., **kwargs)."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model_name=model, **kwargs)


_AGENT_CACHE: Dict[Tuple[str, str, str, str], AgentAPI] = {}


def create_agent(
    model_name: Optional[str] = None,
    model: Optional[str] = None,
    log_file: Optional[str] = None,
) -> Any:
    """Factory used by experiments.py; also callable as get_agent."""
    if model_name is None:
        model_name = model
    if model_name is None:
        raise ValueError("create_agent requires model_name or model")
    normalized = normalize_model_name(model_name)
    if normalized == "random":
        return RandomAgent()
    key = (
        normalized,
        os.environ.get("BOYUE_BASE_URL", ""),
        os.environ.get("TAOTOKEN_BASE_URL", ""),
        os.environ.get("OPENAI_BASE_URL", ""),
        str(log_file),
    )
    agent = _AGENT_CACHE.get(key)
    if agent is None:
        agent = AgentAPI(model_name=model_name, log_file=log_file)
        _AGENT_CACHE[key] = agent
    return agent


get_agent = create_agent


def get_action(
    image: Optional[np.ndarray],
    prompt: str,
    model_name: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    options: Optional[str] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Module-level convenience wrapper used by experiments.py."""
    model = model_name or os.environ.get("BAO_MODEL", "gpt-4o")
    agent = create_agent(model_name=model, log_file=log_file)
    return agent.get_action(
        image=image,
        prompt=prompt,
        state=state,
        history=history,
        options=options,
    )
