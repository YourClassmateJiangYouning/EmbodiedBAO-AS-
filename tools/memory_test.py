"""A ten-turn arithmetic memory test for an MLLM.

Purpose
-------
Check whether a model carries state across API calls within one conversation.
Each turn performs one operation on the result of the previous turn, so the
correct answer at turn N depends on what the model actually said at turn N-1.

That chaining is the whole point.  A model that forgets the running value cannot
silently recover: it operates on a wrong base, the error propagates, and the
divergence is visible immediately.  A test built on pre-computed answers would
instead reward a model that simply guessed a plausible-looking number.

Design
------
* Turn 0 asks the MODEL to choose the starting integer in [100, 999].
* Ten turns then apply x3, +17, -8, /2, x2, +113, -40, /3, x7, -19.
  All integer arithmetic, so every turn has one unambiguous right answer.
* The conversation is sent as a real multi-turn message list: prior user prompts
  and the model's own prior replies as assistant turns.  This exercises actual
  conversational context, which is what "does it remember" means for an API.
* At each turn the check compares the model's number against the value implied by
  ITS OWN previous reply, so the test measures self-consistency rather than
  agreement with an ideal chain it has already left.

Usage
-----
    python tools/memory_test.py --model gemini-2.5-pro
    python tools/memory_test.py --model qwen-vl-max --trials 3
    python tools/memory_test.py --model gemini-2.5-pro --max-turns 10 --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# The operation chain
# ---------------------------------------------------------------------------
# Mixed operations on purpose.  Multiplication and division make a "roughly
# right" recollection diverge, while addition and subtraction make a
# re-randomised value look plausible at a glance -- together they separate
# "tracked the value" from "produced a believable number".
OPERATIONS = [
    ("x3", lambda v: v * 3, "multiply by 3"),
    ("+17", lambda v: v + 17, "add 17"),
    ("-8", lambda v: v - 8, "subtract 8"),
    ("/2", lambda v: v // 2, "divide by 2, rounding down if it is odd"),
    ("x2", lambda v: v * 2, "multiply by 2"),
    ("+113", lambda v: v + 113, "add 113"),
    ("-40", lambda v: v - 40, "subtract 40"),
    ("/3", lambda v: v // 3, "divide by 3, rounding down"),
    ("x7", lambda v: v * 7, "multiply by 7"),
    ("-19", lambda v: v - 19, "subtract 19"),
]


def extract_int(text: str) -> Optional[int]:
    """Pull the model's answer out of a reply.

    Prefers an explicit ``answer: N`` form and otherwise falls back to the LAST
    integer in the reply, because models tend to restate the input before giving
    the result.
    """
    if not text:
        return None
    found = re.findall(r"answer\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    if found:
        return int(found[-1])
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


def run_trial(agent, trial_index: int, max_turns: int, verbose: bool) -> Dict[str, Any]:
    """Run one chained trial. Returns a structured record."""
    messages: List[Dict[str, str]] = []
    turns: List[Dict[str, Any]] = []

    def ask(prompt: str) -> str:
        messages.append({"role": "user", "content": prompt})
        reply = agent._request(messages)
        messages.append({"role": "assistant", "content": reply or ""})
        return reply or ""

    # ---- Turn 0: the model picks the starting value itself -----------------
    first_prompt = (
        "Pick a random integer between 100 and 999. "
        "Reply with exactly: answer: <the number>"
    )
    reply = ask(first_prompt)
    start_value = extract_int(reply)
    turns.append(
        {
            "turn": 0,
            "operation": "pick",
            "description": "pick a random integer in [100, 999]",
            "prompt": first_prompt,
            "reply": reply,
            "expected": None,
            "got": start_value,
            "correct": start_value is not None,
        }
    )
    if verbose:
        print(f"    turn  0 pick  -> {start_value}")

    if start_value is None:
        return {
            "trial": trial_index,
            "start": None,
            "turns": turns,
            "correct_turns": 0,
            "first_failure": 0,
            "ok": False,
            "note": "the model never produced a starting number",
        }

    running = start_value
    correct_turns = 0
    first_failure: Optional[int] = None

    for index, (label, fn, description) in enumerate(
        OPERATIONS[:max_turns], start=1
    ):
        expected = fn(running)
        prompt = (
            f"Now {description} the number from your previous answer. "
            f"Reply with exactly: answer: <the result>"
        )
        reply = ask(prompt)
        got = extract_int(reply)
        correct = got == expected

        if correct:
            correct_turns += 1
            running = expected
        else:
            if first_failure is None:
                first_failure = index
            # Continue from the model's own answer when it gave one, so the
            # chain measures whether it stays self-consistent after a slip
            # rather than whether it can rejoin an abandoned ideal chain.
            if got is not None:
                running = got

        turns.append(
            {
                "turn": index,
                "operation": label,
                "description": description,
                "prompt": prompt,
                "reply": reply,
                "expected": expected,
                "got": got,
                "correct": correct,
            }
        )
        if verbose:
            mark = "ok " if correct else "BAD"
            print(
                f"    turn {index:>2} {label:<4} {mark} "
                f"expected {expected:>12}  got {str(got):>12}"
            )

    return {
        "trial": trial_index,
        "start": start_value,
        "turns": turns,
        "correct_turns": correct_turns,
        "first_failure": first_failure,
        "ok": correct_turns == max_turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ten-turn arithmetic memory test for an MLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=int, default=1, help="how many chains")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    import ai_agent

    agent = ai_agent.AgentAPI(model_name=args.model, temperature=0.0)
    verbose = not args.quiet
    total_turns = min(args.max_turns, len(OPERATIONS))

    print("=" * 74)
    print(f"memory test  model={agent.model_name}  trials={args.trials}  "
          f"turns={total_turns}")
    print("turn 0 the model picks a number; every later turn operates on its own")
    print("previous answer, so a forgotten value cannot be recovered from.")
    print("=" * 74)

    records: List[Dict[str, Any]] = []
    for trial in range(args.trials):
        print(f"\n--- trial {trial + 1}/{args.trials} ---")
        record = run_trial(agent, trial, total_turns, verbose)
        records.append(record)
        if record["first_failure"] is None:
            verdict = "perfect"
        else:
            verdict = f"first failure at turn {record['first_failure']}"
        print(f"  -> {record['correct_turns']}/{total_turns} correct, {verdict}")

    perfect = sum(1 for r in records if r["ok"])
    total_correct = sum(r["correct_turns"] for r in records)
    grand_total = total_turns * len(records)
    failures = [r["first_failure"] for r in records if r["first_failure"]]

    print()
    print("=" * 74)
    print(f"perfect trials          : {perfect}/{len(records)}")
    print(f"turns correct           : {total_correct}/{grand_total}")
    print(f"first-failure positions : {failures if failures else 'none'}")
    print("=" * 74)
    print("reading: failing at turn 1-2 means the value was never carried at all;")
    print("failing only later means drift, or the model re-picked a number.")

    report = {
        "model": agent.model_name,
        "trials": len(records),
        "turns_per_trial": total_turns,
        "perfect_trials": perfect,
        "turns_correct": total_correct,
        "turns_total": grand_total,
        "records": records,
    }
    out_path = args.out or os.path.join(
        "analysis", f"memory_test_{agent.model_name.replace('/', '_')}.json"
    )
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
