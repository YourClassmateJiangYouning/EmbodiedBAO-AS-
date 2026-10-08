"""Checks on the runner's self-defence: it must stop a model whose every call is failing.

An episode in which all thirty calls failed is stored and scored exactly like an episode in which
the model could not solve the task.  In one sweep that ambiguity cost twelve hours on a model
whose model name had been mangled, nine on one whose quota had run out, and three on one whose
key had been revoked.  These checks pin the behaviour that would have ended all three inside
three episodes.

    python3 test_bao_health.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiments  # noqa: E402


class Failure(Exception):
    pass


def check(condition, message: str) -> None:
    if not condition:
        raise Failure(message)


class Stub:
    """Just enough of the runner for the health check: a streak, a model, a history."""

    def __init__(self, feedback: str = "REQUEST ERROR: HTTP 503 no channel"):
        self._all_invalid_streak = 0
        self.model = "kimi/kimi-k2.5"
        self.history = [{"feedback": feedback}]

    def episode(self, steps: int, invalid: int):
        experiments.BAOExperimentRunner._note_episode_health(
            self, {"total_steps": steps, "invalid_response_count": invalid}
        )


def test_warns_before_it_stops() -> None:
    stub = Stub()
    stub.episode(30, 30)  # first: warn, keep going
    check(stub._all_invalid_streak == 1, f"streak after first: {stub._all_invalid_streak}")
    stub.episode(30, 30)  # second: warn, keep going
    check(stub._all_invalid_streak == 2, f"streak after second: {stub._all_invalid_streak}")
    print("[ok] the first two hopeless episodes warn and continue")


def test_stops_on_the_limit() -> None:
    stub = Stub()
    raised = False
    try:
        for _ in range(experiments.INVALID_EPISODE_LIMIT):
            stub.episode(30, 30)
    except RuntimeError as exc:
        raised = True
        check("invalid" in str(exc), f"the message should say why: {exc}")
    check(raised, f"it did not stop after {experiments.INVALID_EPISODE_LIMIT} hopeless episodes")
    print(f"[ok] it stops on episode {experiments.INVALID_EPISODE_LIMIT} of the same failure")


def test_a_healthy_episode_clears_the_streak() -> None:
    stub = Stub()
    stub.episode(30, 30)
    stub.episode(30, 30)
    stub.episode(12, 0)
    check(stub._all_invalid_streak == 0, f"streak not cleared: {stub._all_invalid_streak}")
    stub.episode(30, 30)
    check(stub._all_invalid_streak == 1, "the count should restart from one")
    print("[ok] one healthy episode clears the streak, and it restarts from one")


def test_a_hard_episode_is_not_a_broken_one() -> None:
    """Zero passes but valid replies is a result, not a fault, and must never stop a run."""
    stub = Stub()
    for _ in range(10):
        stub.episode(30, 0)  # 30 steps, no invalid replies: the agent acted and did not pass
    check(stub._all_invalid_streak == 0, "a failing-but-answering model must not be stopped")
    print("[ok] a model that answers and fails is left alone")


def test_partial_invalidity_does_not_stop() -> None:
    stub = Stub()
    for _ in range(10):
        stub.episode(30, 29)  # almost all invalid, but not all
    check(stub._all_invalid_streak == 0, "only an entirely unusable episode counts as hopeless")
    print("[ok] only an entirely unusable episode counts")


def main() -> int:
    tests = [
        test_warns_before_it_stops,
        test_stops_on_the_limit,
        test_a_healthy_episode_clears_the_streak,
        test_a_hard_episode_is_not_a_broken_one,
        test_partial_invalidity_does_not_stop,
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
    print(f"\nall {len(tests)} health checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
