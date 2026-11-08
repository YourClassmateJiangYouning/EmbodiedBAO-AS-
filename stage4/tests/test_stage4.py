"""Regression tests for the Stage 4 loop.

Every test here pins a defect that actually shipped and was invisible in a run log: the request envelope
missing the system message, the prompt reporting a stale torso angle, the odometer advancing on a blocked
move, a retry budget four times the constant, and a step built on top of an unrecorded one.  The fake
transport means no API key and no network are needed.

Run:  python -m pytest stage4/tests/test_stage4.py -q
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE4 = os.path.dirname(HERE)
REPO = os.path.dirname(STAGE4)
for path in (STAGE4, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)

import core  # noqa: E402
import protocol  # noqa: E402


@pytest.fixture()
def session(tmp_path):
    made = core.Stage4Session(sessions_root=str(tmp_path / "sessions"))
    made.open_session("t", {
        "scene": "R1", "marker": "M2", "marker_descriptor": "red circle", "mode": "A",
        "model": "gemini-3-flash-preview", "a_s": 1.2, "max_steps": 30,
    })
    made.start_episode()
    return made


@pytest.fixture()
def png_file(tmp_path):
    """A real 32x32 PNG: encode_for_api opens the file before the transport is ever reached."""
    from PIL import Image as PILImage
    path = os.path.join(str(tmp_path), "frame.png")
    PILImage.new("RGB", (32, 32), (200, 30, 30)).save(path)
    return path


def fake_image(session, path=None):
    """Point the session at a frame without touching the bank."""
    image = path or png_file
    session.current_image = {
        "image_id": "IMG_test", "abs_path": image, "path": "", "width": 1024, "height": 1024,
        "sha256": "", "source": "test",
    }
    return session


def append_step(session, action):
    session.history.append({
        "step": len(session.history), "action": action, "feedback": None,
        "torso_rotation": 0.0, "position_x": session.position_x,
    })
    return session.history[-1]


# ------------------------------------------------------------------ the request envelope
def test_request_envelope_matches_stage1(session, monkeypatch, png_file):
    """system message, response_format, and no max_tokens -- the three ways Stage 4 diverged."""
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"action": "forward"}'},
                                            "finish_reason": "stop"}]}).encode()

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(core.urllib.request, "urlopen", fake_urlopen)
    session._post("prompt text", png_file, 5.0)

    body = captured["body"]
    roles = [message["role"] for message in body["messages"]]
    assert roles == ["system", "user"], "the request must carry the same system message Stage 1-3 sent"
    assert body["messages"][0]["content"] == protocol.SYSTEM_PROMPT
    assert body.get("response_format") == {"type": "json_object"}
    assert "max_tokens" not in body, "Stage 1-3 used the provider default; a fixed budget breaks parity"
    assert body["temperature"] == core.API_TEMPERATURE


def test_json_mode_dropped_only_on_a_parameter_rejection(session, monkeypatch, png_file):
    """A 400 that names response_format disables JSON mode; a 401 must not."""
    calls = []

    def rejecting(request, timeout=None):
        calls.append(json.loads(request.data.decode("utf-8")))
        raise core.urllib.error.HTTPError(
            "url", 400, "Bad Request", {}, __import__("io").BytesIO(
                b'{"error": "unsupported_parameter: response_format"}'))

    monkeypatch.setattr(core.urllib.request, "urlopen", rejecting)
    session._post("p", png_file, 5.0)
    assert len(calls) == 2, "the rejection must resend once without response_format"
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert session.json_mode_supported is False

    session2 = session
    session2.json_mode_supported = True
    calls.clear()

    def unauthorised(request, timeout=None):
        calls.append(json.loads(request.data.decode("utf-8")))
        raise core.urllib.error.HTTPError("url", 401, "Unauthorized", {},
                                          __import__("io").BytesIO(b'{"error": "bad key"}'))

    monkeypatch.setattr(core.urllib.request, "urlopen", unauthorised)
    result = session2._post("p", png_file, 5.0)
    assert len(calls) == 1, "a 401 is not a parameter rejection and must not be resent"
    assert result["ok"] is False
    assert session2.json_mode_supported is True


# ------------------------------------------------------------------ the prompt and the odometer
def test_prompt_reports_the_odometer_torso_angle(session):
    """Two turns left must read 30 in the record AND in the prompt that is sent next."""
    fake_image(session)
    for _ in range(2):
        append_step(session, "turn_left")
        session.action_executed(feedback="executed")
    prompt = session.build_current_prompt()
    assert "torso rotation (degrees): 30.0" in prompt
    assert abs(session.torso_rotation - 30.0) < 1e-9
    assert abs(session.history[-1]["torso_rotation"] - 30.0) < 1e-9


def test_torso_angle_is_folded_like_the_simulator(session):
    """environment reports get_abs_torso_rotation() in [0, 180]; 345 would mean 'almost a full turn'."""
    for raw, expected in ((15.0, 15.0), (-15.0, 15.0), (-90.0, 90.0), (195.0, 165.0), (345.0, 15.0)):
        session.torso_rotation = raw
        session.history = [{"step": 0, "action": "turn_right", "feedback": "executed", "torso_rotation": raw}]
        prompt = session.build_current_prompt()
        assert "torso rotation (degrees): %.1f" % expected in prompt, raw


def test_blocked_move_does_not_advance_the_odometer(session):
    fake_image(session)
    start = session.position_x
    append_step(session, "forward")
    session.action_executed(feedback="blocked", collision=True)
    assert session.position_x == start, "a move the operator marked blocked did not happen"
    assert abs(session.position_x - start) < 1e-9


def test_operator_measurement_overrides_the_odometer(session):
    fake_image(session)
    append_step(session, "forward")
    session.action_executed(feedback="executed", position_x=6.4, torso_rotation=20.0)
    assert abs(session.position_x - 6.4) < 1e-9
    prompt = session.build_current_prompt()
    assert "position (x, y, z): [6.400" in prompt
    assert "torso rotation (degrees): 20.0" in prompt


def test_glance_is_one_frame_and_is_announced(session):
    """The frame after look_down is the downward view, so the state must report pitch 45 there and 0 after."""
    fake_image(session)
    append_step(session, "look_down")
    session.action_executed(feedback="executed")
    assert "extra downward pitch (degrees): 45.0" in session.build_current_prompt()
    append_step(session, "forward")
    session.action_executed(feedback="executed")
    assert "extra downward pitch (degrees): 0.0" in session.build_current_prompt()


# ------------------------------------------------------------------ the action menu
def test_menu_offers_only_the_five_photograph_backed_actions(session):
    fake_image(session)
    prompt = session.build_current_prompt()
    import re
    assert re.findall(r'\{"action": "(\w+)"\}', prompt) == core.REAL_ACTION_SET
    for removed in ("look_left", "look_right", "left/right"):
        assert removed not in prompt


def test_prompt_does_not_leak_the_apparatus(session):
    fake_image(session)
    prompt = session.build_current_prompt()
    for banned in ("A/S", "aperture", "channel width", "1.2 m", "0.684", "narrow opening"):
        assert banned.lower() not in prompt.lower(), banned


# ------------------------------------------------------------------ retries and the pending guard
def test_retry_budgets_match_the_constants(session, monkeypatch, png_file):
    """A network failure: 1 + NETWORK_RETRIES calls.  A format failure: 1 + FORMAT_RETRIES calls."""
    calls = {"n": 0}

    def network_error(request, timeout=None):
        calls["n"] += 1
        raise ValueError("simulated network failure")

    monkeypatch.setattr(core.urllib.request, "urlopen", network_error)
    session._call_with_retries("p", png_file, "test")
    assert calls["n"] == 1 + core.NETWORK_RETRIES, calls["n"]

    calls["n"] = 0

    def bad_reply(request, timeout=None):
        calls["n"] += 1

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "not json at all"},
                                                "finish_reason": "stop"}]}).encode()

        return FakeResponse()

    monkeypatch.setattr(core.urllib.request, "urlopen", bad_reply)
    session._call_with_retries("p", png_file, "test")
    assert calls["n"] == 1 + core.FORMAT_RETRIES, calls["n"]


def test_a_fatal_client_error_is_not_retried(session, monkeypatch, png_file):
    calls = {"n": 0}

    def unauthorised(request, timeout=None):
        calls["n"] += 1
        raise core.urllib.error.HTTPError("url", 401, "Unauthorized", {},
                                          __import__("io").BytesIO(b'{"error": "bad key"}'))

    monkeypatch.setattr(core.urllib.request, "urlopen", unauthorised)
    session._call_with_retries("p", png_file, "test")
    assert calls["n"] == 1, "a wrong key cannot be fixed by resending it"


def test_cannot_build_a_step_on_an_unrecorded_one(session):
    fake_image(session)
    append_step(session, "forward")          # deliberately left without feedback
    with pytest.raises(RuntimeError, match="not been recorded"):
        session.call_model()


# ------------------------------------------------------------------ storage and provenance
def test_config_records_the_parameters_that_were_sent(tmp_path):
    """config.json must not claim {} while the calls carried reasoning_effort."""
    made = core.Stage4Session(sessions_root=str(tmp_path / "s"))
    made.open_session("p", {"a_s": 1.2, "model": "deepseek-v4-flash-vision-exp",
                            "marker_descriptor": "red circle", "mode": "A"})
    with open(os.path.join(made.dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["request_params"] == {"reasoning_effort": "none"}
    assert config["timeout_s"] > 0


def test_memory_loader_reads_the_tool_s_own_output(tmp_path, session):
    path = os.path.join(session.dir, "memory", "card.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"memory_id": "M1", "memory_text": "walked through"}, handle, indent=2)
    assert session.load_memory_file(path)["memory_text"] == "walked through"


def test_inbox_deduplication_survives_a_reopen(tmp_path, session):
    session.config["used_sources"] = ["a.png"]
    session._write_config() if hasattr(session, "_write_config") else None
    with open(os.path.join(session.dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(session.config, handle, indent=2)
    reopened = core.Stage4Session(sessions_root=session.sessions_root)
    reopened.open_session(session.session_id, {"a_s": 1.0, "model": "gpt-5", "marker_descriptor": "blue triangle"})
    assert reopened.config.get("used_sources") == ["a.png"]


# ------------------------------------------------------------------ replay geometry
def test_success_on_the_last_step_is_recorded_as_a_pass(tmp_path):
    """The replay must not record an arrival on step 30 as a max-steps failure."""
    made = core.Stage4Session(sessions_root=str(tmp_path / "s2"))
    made.open_session("last", {"a_s": 1.2, "model": "gpt-5", "marker_descriptor": "red circle",
                               "mode": "A", "max_steps": 1})
    made.start_episode()
    made.history = [{"step": 0, "action": "forward", "feedback": None,
                     "torso_rotation": 0.0, "position_x": 8.75}]
    made.action_executed(feedback="executed", position_x=8.75, passed=True, advance_odometer=False)
    assert made.episodes[-1]["passed"] is True
    assert made.episodes[-1]["end_reason"] == "passed"


def test_frame_bank_maps_state_to_frame(tmp_path, session):
    """forward advances the frame, a turn does not, and a glance takes the downward photograph."""
    import replay_runner

    bank = os.path.join(tmp_path, "bank")
    folder = os.path.join(bank, "1.2红色")
    os.makedirs(folder)
    for index in range(1, 12):
        open(os.path.join(folder, "%d.PNG" % index), "wb").write(b"x")
    glance = os.path.join(bank, "_look_down")
    os.makedirs(glance)
    open(os.path.join(glance, "look_down.PNG"), "wb").write(b"x")

    session.config["a_s"] = 1.2
    session.config["marker_descriptor"] = "red circle"
    session.position_x = 0.5
    session.history = []
    assert os.path.basename(session.bank_frame_for_state(bank)) == "1.PNG"

    session.position_x = 1.25
    assert os.path.basename(session.bank_frame_for_state(bank)) == "2.PNG"

    session.history = [{"step": 0, "action": "turn_left", "feedback": "executed"}]
    assert os.path.basename(session.bank_frame_for_state(bank)) == "2.PNG", "a turn must not change the frame"

    session.history = [{"step": 1, "action": "look_down", "feedback": "executed"}]
    assert os.path.basename(session.bank_frame_for_state(bank)) == "look_down.PNG"


def test_replay_width_formula_matches_the_paper():
    import math
    import replay_runner
    for degrees in (0, 15, 30, 45, 60, 75, 90):
        expected = core.SIM_GEOMETRY["shoulder_width_m"] * abs(math.cos(math.radians(degrees))) + \
                   core.SIM_GEOMETRY["torso_thickness_m"] * abs(math.sin(math.radians(degrees)))
        assert abs(replay_runner.width_at(degrees) - expected) < 1e-9
    assert abs(replay_runner.width_at(90) - core.SIM_GEOMETRY["torso_thickness_m"]) < 1e-9
