"""Parsing checks for the model reply, including the shapes that used to be thrown away.

A discarded reply is recorded as an invalid step.  It scores zero, it looks exactly like a model
that cannot solve the task, and it costs a full retry cycle first, so the difference between
"the model answered wrongly" and "we could not read the answer" has to be measured rather than
assumed.

    python3 test_bao_parsing.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai_agent  # noqa: E402


class Failure(Exception):
    pass


def check(condition, message: str) -> None:
    if not condition:
        raise Failure(message)


def test_plain_json() -> None:
    got = ai_agent.parse_action_json(
        '{"scene_description": "a corridor", "reasoning": "go", "action": "forward",'
        ' "confidence": 0.8}')
    check(got and got["action"] == "forward", f"plain JSON failed: {got}")
    check(got["confidence"] == 0.8, f"confidence not read: {got}")
    check(got["scene_description"] == "a corridor", f"scene not read: {got}")
    print("[ok] plain JSON")


def test_fenced_json() -> None:
    for fence in ("```json", "```", "~~~json", "~~~"):
        text = f"{fence}\n{{\"action\": \"turn_left\", \"reasoning\": \"narrow\"}}\n{fence}"
        got = ai_agent.parse_action_json(text)
        check(got and got["action"] == "turn_left", f"fence {fence!r} failed: {got}")
    print("[ok] fenced JSON, in both the backtick and the tilde spelling")


def test_brace_before_the_object() -> None:
    """A brace in the prose used to end the search."""
    text = ('I will answer in the form {"action": ...} as instructed.\n'
            'Here it is: {"action": "forward", "confidence": 0.5}')
    got = ai_agent.parse_action_json(text)
    check(got and got["action"] == "forward", f"brace-before-object failed: {got}")
    print("[ok] a brace in the prose does not hide the real object")


def test_action_spelling() -> None:
    cases = {
        "Forward": "forward", "FORWARD": "forward", " forward ": "forward",
        "move_forward": "forward", "move forward": "forward", "walk": "forward",
        "advance": "forward", "rotate left": "turn_left", "turn-left": "turn_left",
        "Turn_Left": "turn_left", "backwards": "backward", "glance down": "look_down",
        "look-right": "look_right", "sidestep left": "left",
    }
    for given, want in cases.items():
        got = ai_agent.parse_action_json('{"action": "%s"}' % given)
        resolved = got["action"] if got else None
        check(resolved == want, f"action {given!r} resolved to {resolved!r}, wanted {want!r}")
    print("[ok] case, spacing and common paraphrases of the nine action names")


def test_action_under_another_key() -> None:
    for key in ("action", "act", "choice", "next_action", "move"):
        got = ai_agent.parse_action_json('{"%s": "right", "confidence": 0.2}' % key)
        check(got and got["action"] == "right", f"key {key!r} not accepted: {got}")
    print("[ok] the action is found under act / choice / next_action / move as well")


def test_alternative_field_names() -> None:
    got = ai_agent.parse_action_json('{"action": "forward", "scene": "a hallway",'
                                     ' "thought": "the gap is wide"}')
    check(got and got["scene_description"] == "a hallway", f"scene alias failed: {got}")
    check(got["reasoning"] == "the gap is wide", f"thought alias failed: {got}")
    print("[ok] scene / thought are accepted for scene_description / reasoning")


def test_confidence_is_clamped() -> None:
    import json as _json

    for given, want in ((2.5, 1.0), (-3, 0.0), ("0.4", 0.4), ("abc", 0.0), (None, 0.0)):
        blob = _json.dumps({"action": "forward", "confidence": given})
        got = ai_agent.parse_action_json(blob)
        check(got and got["confidence"] == want,
              f"confidence {given!r} became {got and got['confidence']!r}, wanted {want!r}")
    got = ai_agent.parse_action_json('{"action": "forward", "confidence": NaN}')
    check(got and got["confidence"] == 0.0, f"NaN confidence not neutralised: {got}")
    print("[ok] confidence is finite and clamped to 0..1")


def test_refusals_are_not_actions() -> None:
    for text in ("", "   ", None, "I cannot do that.", "The image is unclear.",
                 '{"action": "fly"}', '{"reasoning": "no action here"}', "[1, 2, 3]"):
        check(ai_agent.parse_action_json(text) is None,
              f"should not have parsed an action from {text!r}")
    print("[ok] refusals, empty replies and unknown actions stay unparsed")


def test_text_fallback_still_works() -> None:
    check(ai_agent.parse_action_text("Choice: [3]") == ai_agent.ACTIONS[2],
          "Choice: [3] fallback broke")
    check(ai_agent.parse_action_text("I will go forward now") == "forward",
          "prose fallback broke")
    check(ai_agent.parse_action_text("") is None, "empty text should not parse")
    print("[ok] the Choice: [n] and prose fallbacks still behave")


def main() -> int:
    tests = [
        test_plain_json,
        test_fenced_json,
        test_brace_before_the_object,
        test_action_spelling,
        test_action_under_another_key,
        test_alternative_field_names,
        test_confidence_is_clamped,
        test_refusals_are_not_actions,
        test_text_fallback_still_works,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Failure as failure:
            failures += 1
            print(f"[FAIL] {test.__name__}: {failure}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[ERROR] {test.__name__}: {exc!r}")
    if failures:
        print(f"\n{failures}/{len(tests)} checks FAILED")
        return 1
    print(f"\nall {len(tests)} parsing checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
