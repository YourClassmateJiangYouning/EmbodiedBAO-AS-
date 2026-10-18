"""Parsing checks for the model reply, including the shapes that used to be thrown away.

A discarded reply is recorded as an invalid step.  It scores zero, it looks exactly like a model
that cannot solve the task, and it costs a full retry cycle first, so the difference between
"the model answered wrongly" and "we could not read the answer" has to be measured rather than
assumed.

    python3 test_bao_parsing.py
"""

from __future__ import annotations

import json
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
        # The reversed word order.  "turn left" resolves correctly through ACTIONS;
        # "left turn" used to fall through to the substring scan, match "left" and
        # execute a 0.75 m SIDESTEP where the model asked for a 15-degree rotation.
        "left turn": "turn_left", "left_turn": "turn_left", "leftturn": "turn_left",
        "Left Turn": "turn_left", "turn to the left": "turn_left",
        "right turn": "turn_right", "right_turn": "turn_right",
        "turn to the right": "turn_right",
        # ... while "left" on its own is still the sidestep.
        "left": "left", "sidestep left": "left", "strafe left": "left",
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
    # THE PROSE FALLBACK HAS TO RESOLVE WORD ORDER THE SAME WAY THE JSON PARSER DOES.
    #
    # It scanned only the nine canonical names, longest first, so "turn right" matched "right" -- a
    # 0.75 m SIDESTEP -- before turn_right was ever considered, and "look left" matched "left" the same
    # way. That is the bug ACTION_SYNONYMS exists to fix, but only parse_action_json was routed through
    # it. Each pair below is a rotation against the sidestep of the same word, so a regression shows up
    # as the wrong PHYSICAL action rather than a parse failure.
    for text, expected in (
        ("turn right", "turn_right"),
        ("turn left", "turn_left"),
        ("I will turn left now", "turn_left"),
        ("left turn", "turn_left"),
        ("right turn", "turn_right"),
        ("look left", "look_left"),
        ("look right", "look_right"),
        ("look down at my body", "look_down"),
    ):
        check(
            ai_agent.parse_action_text(text) == expected,
            f"the prose fallback read {text!r} as "
            f"{ai_agent.parse_action_text(text)!r}, not {expected!r}: a rotation would be executed as "
            "a sidestep",
        )
    # ... while the bare words are still the sidesteps, which is what they mean.
    check(ai_agent.parse_action_text("left") == "left", "a bare 'left' stopped being a sidestep")
    check(ai_agent.parse_action_text("right") == "right", "a bare 'right' stopped being a sidestep")
    print("[ok] the Choice: [n] and prose fallbacks still behave, including word order")


def test_json_mode_only_downgrades_on_a_parameter_rejection() -> None:
    """A timeout must not be treated as "this endpoint rejects response_format".

    The JSON-mode fallback disabled response_format on ANY exception and then immediately resent the
    request without it. A timeout, a 502 or a depleted balance all took that path, so one attempt cost
    two HTTP calls, the process was switched out of JSON mode for the rest of the run, and the retry
    changed the request shape halfway through. The classification is what decides.
    """

    class _Error(Exception):
        def __init__(self, message: str, status: Any = None) -> None:
            super().__init__(message)
            self.status_code = status

    rejections = (
        _Error("unsupported parameter: response_format"),
        _Error("invalid parameter", 400),
        _Error("unprocessable entity", 422),
        _Error("unrecognized request argument: json_object"),
    )
    transient = (
        _Error("Request timed out"),
        _Error("502 Bad Gateway", 502),
        _Error("Insufficient balance", 403),
        _Error("", None),
    )
    for error in rejections:
        check(
            ai_agent._is_response_format_rejection(error),
            f"{error!r} is a parameter rejection and should downgrade JSON mode",
        )
    for error in transient:
        check(
            not ai_agent._is_response_format_rejection(error),
            f"{error!r} is transient and must NOT downgrade JSON mode: it would resend the request "
            "and double the cost of every failed step",
        )
    # AND THE WIRING, WHICH THE CLASSIFIER TEST ABOVE DOES NOT REACH.  A behavioural test would need a
    # fake client that raises through the OpenAI wrapper; the property that matters is that _request
    # consults the classifier before downgrading, and its absence is detectable at the source.
    import inspect

    source = inspect.getsource(ai_agent.AgentAPI._request)
    check(
        "_is_response_format_rejection" in source,
        "the JSON-mode fallback no longer consults _is_response_format_rejection, so a timeout or a "
        "502 disables JSON mode for the process and resends the request: two calls per failed step",
    )
    print("[ok] only a parameter rejection downgrades JSON mode, and _request asks before doing it")


def test_a_non_finite_request_parameter_is_refused() -> None:
    """NaN is not JSON, and the encoder used to emit it anyway.

    json.dumps writes a bare NaN token for a non-finite float, which no JSON parser accepts. A real
    Level 0 run produced exactly that from the gateway: HTTP 400 "JSON decode error", which cost a
    retry and looked like a model failure. The encoder now refuses, so the fault names itself here
    instead of being a server-side parse error two layers away.
    """
    check(
        json.loads(ai_agent._dumps_request({"temperature": 0.0, "model": "x"}))
        == {"temperature": 0.0, "model": "x"},
        "a normal request body is no longer round-trippable",
    )
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            ai_agent._dumps_request({"temperature": bad})
        except ValueError as exc:
            check(
                "not JSON" in str(exc) or "JSON" in str(exc),
                f"the refusal does not explain itself: {exc}",
            )
        else:
            check(False, f"_dumps_request sent {bad!r}, which JSON cannot represent")
    print("[ok] a non-finite request parameter is refused instead of sent as NaN")


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
        test_json_mode_only_downgrades_on_a_parameter_rejection,
        test_a_non_finite_request_parameter_is_refused,
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
