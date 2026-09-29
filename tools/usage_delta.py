"""Measure what a sweep costs, from the panel's own accounting.

The gateway exposes OpenAI's compatibility billing endpoints but no per-request price,
so the only honest way to know what a run cost is to read the meter before and after:

    python tools/usage_delta.py --before            # before starting a sweep
    ...                                              # run the sweep
    python tools/usage_delta.py --after --rounds 102 --budget-rmb 100

``--after`` prints the delta, converts it, and extrapolates from what the delta covered
to a whole model and the whole roster, which is the number to take to whoever is paying.

Two things this tool refuses to guess:

  * **The unit.**  ``total_usage`` follows OpenAI's convention of hundredths of a
    dollar, but a reseller panel can define its own unit, so ``--unit`` selects the
    reading and ``--status`` prints both rather than picking one quietly.
  * **What the panel's "$" is worth.**  ``--rate`` is the RMB per panel-dollar, and
    ``--nominal`` says the panel bills 1:1 in RMB.  Neither is discoverable from the
    API; the recharge record is the authority.

A delta of zero is reported as a finding, not as a cheap run: it means either nothing
ran or every call was refused, and a refused call is free -- which is exactly how the
first pilot produced a round that looked like data while costing nothing.

    python tools/usage_delta.py --status           # meter reading, no baseline
    python tools/usage_delta.py --before           # record the baseline
    python tools/usage_delta.py --after --rounds 17
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"
DEFAULT_BASELINE = "usage_baseline.json"
ROUNDS_PER_RUN = 17
RUNS_PER_MODEL = 6
ROSTER_MODELS = 11
# The panel's own numbers, so a reading that has not moved is visible at a glance.
STATE_HINT = (
    "A zero delta means nothing was billed: either the work did not run, or every "
    "request was refused.  Check the report for rounds that are entirely fallback "
    "steps before believing a cheap sweep."
)


def fingerprint(key: str) -> str:
    return f"{key[:7]}...{key[-4:]} ({len(key)} chars)" if len(key) > 14 else "(short key)"


def request_json(base_url: str, key: str, path: str, timeout: float = 20.0):
    request = urllib.request.Request(
        base_url.rstrip("/") + path, headers={"Authorization": "Bearer " + key}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace")), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        return None, f"HTTP {exc.code}: {body}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def read_meter(base_url: str, key: str):
    """The two numbers the panel will give: the granted limit and the usage."""
    subscription, error = request_json(base_url, key, "/dashboard/billing/subscription")
    if error:
        return None, error
    usage, error = request_json(base_url, key, "/dashboard/billing/usage")
    if error:
        return None, error
    try:
        limit = float(subscription.get("hard_limit_usd"))
        used = float(usage.get("total_usage"))
    except (AttributeError, TypeError, ValueError):
        return None, f"unexpected bodies: {subscription} {usage}"
    return {"limit": limit, "used": used}, None


def spent_in_usd(used_delta: float, unit: str) -> float:
    return used_delta / 100.0 if unit == "cents" else used_delta


def rmb(usd: float, rate: float, nominal: bool) -> float:
    return usd * (1.0 if nominal else rate)


def show_status(meter: dict, key: str, rate: float, nominal: bool) -> None:
    limit, used = meter["limit"], meter["used"]
    print(f"key      : {fingerprint(key)}")
    print(f"limit    : {limit:.5f}   used: {used:.4f}")
    for unit in ("cents", "dollars"):
        spent = spent_in_usd(used, unit)
        remaining = limit - spent
        print(
            "  if total_usage is %-7s : spent %12.4f  remaining %12.4f  "
            "= RMB %10.2f spent, RMB %10.2f left"
            % (unit, spent, remaining, rmb(spent, rate, nominal), rmb(remaining, rate, nominal))
        )
    print(
        "  the two readings differ by 100x; the recharge record says which is real"
        + (" (--nominal: 1 panel-dollar = 1 RMB)" if nominal else f" (rate {rate} RMB per dollar)")
    )


def load_baseline(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default=os.environ.get("BAO_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--key", default=os.environ.get("BOYUE_API_KEY", ""))
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--unit", choices=("cents", "dollars"), default="cents",
                        help="what total_usage counts in (OpenAI convention: cents)")
    parser.add_argument("--rate", type=float, default=7.2,
                        help="RMB per panel-dollar, for a panel that bills in dollars")
    parser.add_argument("--nominal", action="store_true",
                        help="the panel bills 1:1 in RMB, so no exchange rate applies")
    parser.add_argument("--rounds", type=int, default=0,
                        help="rounded attempts the delta covers, for extrapolation")
    parser.add_argument("--budget-rmb", type=float, default=0.0,
                        help="money available, to say how far it goes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="print the meter and stop")
    group.add_argument("--before", action="store_true", help="record the baseline")
    group.add_argument("--after", action="store_true", help="report the delta since it")
    args = parser.parse_args()

    if not args.key:
        print("set BOYUE_API_KEY or pass --key", file=sys.stderr)
        return 2
    meter, error = read_meter(args.base_url, args.key)
    if error:
        print(f"could not read the meter: {error}", file=sys.stderr)
        return 1

    if args.status:
        print(f"endpoint : {args.base_url}")
        show_status(meter, args.key, args.rate, args.nominal)
        return 0

    if args.before:
        record = {
            "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "endpoint": args.base_url,
            "key": fingerprint(args.key),
            "limit": meter["limit"],
            "used": meter["used"],
            "unit": args.unit,
        }
        with open(args.baseline, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        print(
            "baseline: used %.4f (%s) against a limit of %.5f, recorded %s"
            % (meter["used"], args.unit, meter["limit"], record["recorded_at"])
        )
        print(f"written to {args.baseline}; run --after when the sweep stops")
        return 0

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print(f"no baseline at {args.baseline}; run --before first", file=sys.stderr)
        return 1
    if baseline.get("key") != fingerprint(args.key):
        print(
            "WARNING: the baseline was recorded with a different key (%s, now %s); the "
            "delta would mix two accounts" % (baseline.get("key"), fingerprint(args.key))
        )

    used_delta = meter["used"] - float(baseline["used"])
    spent = spent_in_usd(used_delta, args.unit)
    spent_rmb = rmb(spent, args.rate, args.nominal)
    print(f"endpoint : {args.base_url}")
    print(f"key      : {fingerprint(args.key)}")
    print(f"baseline : {baseline['recorded_at']}  used {baseline['used']:.4f}")
    print(f"now      : used {meter['used']:.4f}   (limit {meter['limit']:.5f})")
    print(
        "delta    : %.4f %s = %.4f panel-dollars = RMB %.2f"
        % (used_delta, args.unit, spent, spent_rmb)
    )

    if used_delta <= 0:
        print()
        print(STATE_HINT)
        return 0

    if args.rounds > 0:
        per_round = spent / args.rounds
        per_run = per_round * ROUNDS_PER_RUN
        per_model = per_run * RUNS_PER_MODEL
        roster = per_model * ROSTER_MODELS
        print()
        print("  %d attempts cost %.4f panel-dollars (RMB %.2f)" % (args.rounds, spent, spent_rmb))
        print("  per attempt  : %.5f  (RMB %.4f)" % (per_round, rmb(per_round, args.rate, args.nominal)))
        print("  per run (17) : %.4f  (RMB %.2f)" % (per_run, rmb(per_run, args.rate, args.nominal)))
        print(
            "  per model (6 runs = 102 attempts): %.3f  (RMB %.2f)"
            % (per_model, rmb(per_model, args.rate, args.nominal))
        )
        print(
            "  the %d-model roster: %.2f  (RMB %.2f)"
            % (ROSTER_MODELS, roster, rmb(roster, args.rate, args.nominal))
        )
        if args.budget_rmb > 0:
            remaining_rmb = args.budget_rmb - spent_rmb
            print()
            print("  budget %.2f RMB - spent %.2f RMB = %.2f RMB left" % (
                args.budget_rmb, spent_rmb, remaining_rmb))
            if per_model > 0:
                cost_per_model_rmb = rmb(per_model, args.rate, args.nominal)
                print(
                    "  that pays for about %.1f more model(s) at this rate"
                    % (remaining_rmb / cost_per_model_rmb)
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
