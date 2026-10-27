"""New Stage 2 prompt protocol: one blind success memory + a no-action digest step.

This module does NOT implement the simulator loop. It only builds the prompts.
The old ``memory_protocol.py`` (12 attempts + self-written notes) is archived and
must not be used.

What the tested model is told:
  * its Stage 1 role/task/action menu/current state (same strings as Stage 1);
  * that one prior trajectory is a successful traversal by the same model in the
    same scene and marker slot;
  * NOT the A/S, Level, width, narrow/wide, or that the upcoming test is wide.

The memory file is already blind (no A/S, Level, width, model, narrow/wide).  The
source reasoning is unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import protocol

DIGEST_INSTRUCTION = (
    "\n\nStep 0 (digest; no action is allowed). Read the prior successful "
    "experience and the current view. Do not output an action, action name, "
    "confidence, or scene_description. Return exactly one JSON object: "
    '{"digested": true, '
    '"memory_summary": "one or two sentences", '
    '"plan": "how you will approach this scene", '
    '"cues": ["what you will look for"]}'
)


def memory_block(memory_text: str, digest: Optional[str] = None) -> str:
    parts = [
        "A previous successful traversal by the same model in this same scene and marker:",
        memory_text.strip(),
    ]
    if digest:
        parts.extend(["Your reflection on that experience:", digest.strip()])
    return "\n\n".join(p for p in parts if p)


def build_digest_prompt(state: Dict[str, Any], memory_text: str, max_steps: int = 30) -> str:
    base = protocol.build_prompt(
        state=state, history=[], max_steps=max_steps,
        memory_block=memory_block(memory_text),
    )
    # The digest is not an action.  Remove BOTH the Stage 1 action-response
    # instruction AND the action menu: with the menu present, DeepSeek answered
    # the digest with an action JSON instead of the requested digest.  The task,
    # state, memory and response-format section are kept.
    instruction = protocol.RESPONSE_FORMAT_INSTRUCTION
    if base.endswith(instruction):
        base = base[: -len(instruction)].rstrip()
    start = base.find("Available actions")
    state_at = base.find("Current state:")
    if start != -1 and state_at != -1 and state_at > start:
        base = base[:start] + base[state_at:]
    return base + DIGEST_INSTRUCTION


def build_test_prompt(state: Dict[str, Any], history: Sequence[Dict[str, Any]],
                      memory_text: str, digest: Optional[str] = None,
                      max_steps: int = 30) -> str:
    return protocol.build_prompt(
        state=state, history=history, max_steps=max_steps,
        memory_block=memory_block(memory_text, digest=digest),
    )


def _strip_json_fence(text: str) -> str:
    """Accept a plain object or one fenced with ```json ... ```."""
    s = str(text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def parse_digest(text: str) -> Dict[str, Any]:
    """Parse the no-action digest; invalid/action replies are marked not digested."""
    try:
        obj = json.loads(_strip_json_fence(text))
        if isinstance(obj, dict) and obj.get("digested") is True:
            if isinstance(obj.get("memory_summary"), str) and isinstance(obj.get("plan"), str):
                return obj
        return {"digested": False, "raw": text, "parsed": obj}
    except Exception:
        return {"digested": False, "raw": text}
