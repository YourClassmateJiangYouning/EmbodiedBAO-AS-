#!/usr/bin/env bash
# Run the nine offline suites and summarise them, with a non-zero exit if ANY suite fails.
#
# WHY THIS EXISTS
#   The suites were runnable only one at a time, and two of them (test_bao_scenes, test_bao_persistence)
#   print a warning rather than failing when they are skipped, so "all green" was something the operator
#   had to assemble by eye from nine terminal outputs.  A summary that cannot return a non-zero status is
#   not a summary; this one does.
#
# WHICH INTERPRETER
#   test_bao_geometry, test_bao_scenes, test_bao_integration, test_bao_assets and test_bao_health import
#   environment.py and therefore need the Isaac Sim python.  The rest are pure Python.  Running everything
#   under the Isaac interpreter is correct for both, and it is the default here.
#
#   ISAAC_PY=/home/ybh/isaacsim/python.sh ./run_all_tests.sh
#   ISAAC_PY=python3 ./run_all_tests.sh          # only the pure-Python suites will pass
#
# READ-ONLY CHECKOUT
#   test_bao_integration and test_bao_persistence write under _test_tmp/.  On a read-only tree they fail
#   with OSError: [Errno 30] Read-only file system, which is an environment problem and not a defect.
#   The script detects that and says so rather than reporting it as a failure of the code.
set -u

PY="${ISAAC_PY:-/home/ybh/isaacsim/python.sh}"
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "${PY%% *}" ]; then
    echo "run_all_tests.sh: interpreter not found: $PY" >&2
    echo "  set ISAAC_PY to the Isaac Sim python.sh, or to python3 for the pure suites only" >&2
    exit 2
fi

cd "$(dirname "$0")" || exit 2

SUITES=(
    test_bao_geometry.py
    test_bao_scenes.py
    test_bao_integration.py
    test_bao_persistence.py
    test_bao_memory.py
    test_bao_memory_runner.py
    test_bao_parsing.py
    test_bao_health.py
    test_bao_assets.py
)

FAILED=()
SKIPPED_ONLY=()
RO_ONLY=()
for suite in "${SUITES[@]}"; do
    if [ ! -f "$suite" ]; then
        echo "  MISSING  $suite"
        FAILED+=("$suite (missing)")
        continue
    fi
    out="$("$PY" "$suite" 2>&1)"
    status=$?
    last="$(printf '%s\n' "$out" | grep -vE '^\s*$' | tail -1)"
    printf '  %-30s %s\n' "$suite" "${last:-<no output>}"
    # CLASSIFY BY WHAT THE OUTPUT SAYS, NOT ONLY WHEN THE EXIT STATUS IS NON-ZERO.  The SKIPPED branch
    # used to live inside `if [ "$status" -ne 0 ]`, but every suite that can skip RETURNS 0 when it does
    # -- "43/44 passed, 1 SKIPPED" is a success -- so SKIPPED_ONLY was always empty and the exit 3
    # "run was incomplete" path was unreachable.  A run with two skipped checks reported "every suite
    # passed".
    if printf '%s' "$out" | grep -q 'Read-only file system'; then
        RO_ONLY+=("$suite")
    elif [ "$status" -eq 0 ] && printf '%s' "$out" | grep -qE '[0-9]+ SKIPPED|SKIPPED \(not run\)'; then
        SKIPPED_ONLY+=("$suite")
    elif [ "$status" -ne 0 ]; then
        if printf '%s' "$out" | grep -qE 'SKIPPED'; then
            SKIPPED_ONLY+=("$suite")
        else
            FAILED+=("$suite")
        fi
    fi
done

echo
if [ "${#RO_ONLY[@]}" -gt 0 ]; then
    echo "  read-only checkout: $(printf '%s ' "${RO_ONLY[@]}")could not write _test_tmp/."
    echo "  That is the filesystem, not the code. Re-run on a writable checkout before reading anything"
    echo "  into these results -- 16 integration and 11 persistence checks cannot pass without it."
fi
if [ "${#SKIPPED_ONLY[@]}" -gt 0 ]; then
    echo "  skipped only: $(printf '%s ' "${SKIPPED_ONLY[@]}")-- see the suite's own SKIPPED line."
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "  FAILED: $(printf '%s ' "${FAILED[@]}")"
    exit 1
fi
if [ "${#RO_ONLY[@]}" -gt 0 ] || [ "${#SKIPPED_ONLY[@]}" -gt 0 ]; then
    echo "  NO SUITE FAILED, but the run was incomplete (see above)."
    exit 3
fi
echo "  every suite passed"
exit 0
