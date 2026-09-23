"""A ten-turn arithmetic memory test for an MLLM -- standalone, no Isaac Sim.

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
* Multiplication and division make a vaguely-remembered value diverge, while the
  additions and subtractions make a re-randomised value still look plausible --
  together they separate "tracked the value" from "produced a believable number".
* The conversation is sent as a real multi-turn message list: prior user prompts
  and the model's own prior replies as assistant turns.  This exercises actual
  conversational context, which is what "does it remember" means for an API.
* After a wrong turn the chain continues from the model's own answer, so later
  turns measure self-consistency rather than re-joining an ideal chain.

This script talks HTTP directly and imports nothing from the project, so it runs
anywhere with Python 3.8+ and network access -- no Isaac Sim, no numpy.

Usage
-----
    export BOYUE_API_KEY='...'                 # or TAOTOKEN_API_KEY / OPENAI_API_KEY
    python tools/memory_test.py --model gemini-2.5-pro
    python tools/memory_test.py --model qwen-vl-max --trials 3
    python tools/memory_test.py --model gemini-2.5-pro --list-models

On Windows PowerShell:
    $env:BOYUE_API_KEY='...'
    python tools\\memory_test.py --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"

# ---------------------------------------------------------------------------
# The operation chain
# ---------------------------------------------------------------------------
OPERATIONS: List[Tuple[str, Any, str]] = [
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


class ChatClient:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data)
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def list_models(self) -> List[str]:
        try:
            body = self._post("/models")
        except Exception as exc:
            raise RuntimeError(f"could not list models: {exc}") from exc
        entries = body.get("data") or []
        return sorted(str(e.get("id")) for e in entries if e.get("id"))

    def ask(self, messages: List[Dict[str, str]], max_retries: int = 1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                body = self._post("/chat/completions", payload)
                choices = body.get("choices") or []
                if choices:
                    return (choices[0].get("message") or {}).get("content") or ""
                return ""
            except Exception as exc:  # noqa: BLE001 - report and retry
                last_error = exc
                if attempt < max_retries:
                    time.sleep(2.0)
        raise RuntimeError(f"request failed after {max_retries + 1} attempt(s): {last_error}")


def resolve_config(args: argparse.Namespace) -> Tuple[str, str]:
    api_key = (
        args.api_key
        or os.environ.get("BOYUE_API_KEY")
        or os.environ.get("TAOTOKEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "No API key. Set BOYUE_API_KEY (or TAOTOKEN_API_KEY / OPENAI_API_KEY), "
            "or pass --api-key."
        )
    if not api_key.isascii():
        raise SystemExit(
            "The API key contains non-ASCII characters -- it looks like a "
            "placeholder such as a Chinese 'your-key-here' string."
        )
    base_url = (
        args.base_url
        or os.environ.get("BOYUE_BASE_URL")
        or os.environ.get("TAOTOKEN_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return api_key, base_url


def run_trial(
    client: ChatClient, trial_index: int, max_turns: int, verbose: bool
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    turns: List[Dict[str, Any]] = []

    def ask(prompt: str) -> str:
        messages.append({"role": "user", "content": prompt})
        started = time.time()
        reply = client.ask(messages)
        latency = time.time() - started
        messages.append({"role": "assistant", "content": reply})
        turns_latency.append(latency)
        return reply

    turns_latency: List[float] = []

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
            "prompt": first_prompt,
            "reply": reply,
            "expected": None,
            "got": start_value,
            "correct": start_value is not None,
            "latency_s": round(turns_latency[-1], 2),
        }
    )
    if verbose:
        print(f"    turn  0 pick   -> {start_value}   [{turns_latency[-1]:.1f}s]")

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

    for index, (label, fn, description) in enumerate(OPERATIONS[:max_turns], start=1):
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
            # chain measures self-consistency after a slip rather than whether
            # it can rejoin a chain it has already left.
            if got is not None:
                running = got

        turns.append(
            {
                "turn": index,
                "operation": label,
                "prompt": prompt,
                "reply": reply,
                "expected": expected,
                "got": got,
                "correct": correct,
                "latency_s": round(turns_latency[-1], 2),
            }
        )
        if verbose:
            mark = "ok " if correct else "BAD"
            print(
                f"    turn {index:>2} {label:<4} {mark} "
                f"expected {expected:>12}  got {str(got):>12}   "
                f"[{turns_latency[-1]:.1f}s]"
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
        description="Ten-turn arithmetic memory test for an MLLM (standalone)."
    )
    parser.add_argument("--model", help="model name, e.g. gemini-2.5-pro")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the endpoint's model ids and exit (checks the key works)",
    )
    args = parser.parse_args()

    api_key, base_url = resolve_config(args)
    timeout = args.timeout or float(os.environ.get("BAO_LLM_TIMEOUT", "90"))
    model = args.model or "gemini-2.5-pro"
    client = ChatClient(model, base_url, api_key, timeout)

    if args.list_models:
        print(f"endpoint {base_url}")
        for name in client.list_models():
            print(f"  {name}")
        return 0

    verbose = not args.quiet
    total_turns = min(args.max_turns, len(OPERATIONS))

    print("=" * 74)
    print(f"memory test  model={model}  trials={args.trials}  turns={total_turns}")
    print("turn 0 the model picks a number; every later turn operates on its own")
    print("previous answer, so a forgotten value cannot be recovered from.")
    print("=" * 74)

    records: List[Dict[str, Any]] = []
    for trial in range(args.trials):
        print(f"\n--- trial {trial + 1}/{args.trials} ---")
        record = run_trial(client, trial, total_turns, verbose)
        records.append(record)
        verdict = (
            "perfect"
            if record["first_failure"] is None
            else f"first failure at turn {record['first_failure']}"
        )
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
        "model": model,
        "base_url": base_url,
        "trials": len(records),
        "turns_per_trial": total_turns,
        "perfect_trials": perfect,
        "turns_correct": total_correct,
        "turns_total": grand_total,
        "records": records,
    }
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    out_path = args.out or os.path.join("analysis", f"memory_test_{safe}.json")
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
