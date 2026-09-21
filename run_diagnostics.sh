#!/usr/bin/env bash
#
# One-shot diagnostic for the EmbodiedBAO A/S benchmark on the lab machine.
#
# Run it like this so everything lands in diagnose.log as well as on screen:
#
#     cd ~/EmbodiedBAO-AS- && bash run_diagnostics.sh 2>&1 | tee diagnose.log
#
# The script never aborts early: each section reports OK/FAILED and continues,
# so one run gives a complete picture instead of one grep pattern at a time.
# Send back the whole of diagnose.log.

set -u

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ISAAC_PY="${ISAAC_PY:-/home/ybh/isaacsim/python.sh}"
PY="${PY:-python3}"

cd "$REPO_DIR" || exit 1

section() {
    echo
    echo "======================================================================"
    echo "== $*"
    echo "======================================================================"
}

run() {
    # run <label> <command...>  -- never aborts; always reports a verdict
    local label="$1"
    shift
    echo
    echo "--- \$ $*"
    "$@"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "--- [$label] OK"
    else
        echo "--- [$label] FAILED (exit $rc)"
    fi
    return 0
}

section "0. environment"
echo "date          : $(date)"
echo "host          : $(hostname)"
echo "user          : $(whoami)"
echo "pwd           : $(pwd)"
echo "DISPLAY       : ${DISPLAY:-<unset>}"
echo "ISAAC_PY      : $ISAAC_PY"
echo "ISAAC_PY ok   : $([ -x "$ISAAC_PY" ] && echo yes || echo NO)"
echo "python3       : $(command -v "$PY" || echo MISSING)"
echo "conda         : ${CONDA_DEFAULT_ENV:-<none>}"
echo "ISAACSIM_ROOT : ${ISAACSIM_ROOT:-<unset>}"
echo "BOYUE_API_KEY : $([ -n "${BOYUE_API_KEY:-}" ] && echo set || echo unset)"
echo "API base url  : ${BOYUE_BASE_URL:-<unset, code default is the lab server>}"

section "1. other Isaac processes (do NOT kill these)"
ps -eo pid,user,etime,cmd 2>/dev/null | grep -E "[k]it/python/bin/python3|[p]ython.sh"
echo "(empty above means no Isaac Sim is running right now)"

section "2. git revision"
run "git-log" git log --oneline -3
echo
echo "uncommitted changes:"
git status --short

section "3. H1 asset"
if [ -d assets/H1 ]; then
    ls -la assets/H1/
else
    echo "no assets/H1 directory"
fi

section "4. offline suites (no Isaac Sim needed)"
run "geometry-tests" "$PY" test_bao_geometry.py
run "integration-tests" "$PY" test_bao_integration.py

section "5. Isaac Sim import probe (about 3 minutes)"
run "isaac-probe" "$ISAAC_PY" -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
print('APP STARTED')
try:
    import environment
    print('environment._HAS_ISAAC_SIM =', environment._HAS_ISAAC_SIM)
except Exception:
    import traceback
    traceback.print_exc()
app.close()
"

section "6. scene capture, level 0 (about 3-4 minutes)"
run "capture-level0" "$ISAAC_PY" capture_views.py --level 0 --outdir views_L0
echo
echo "files produced:"
ls -la views_L0/ 2>/dev/null || echo "  (views_L0 missing)"

section "7. key measurements"
echo "Robot height, camera placement and image readability:"
echo

section "DONE"
echo "run this to send the whole thing back:"
echo "    cat ~/EmbodiedBAO-AS-/diagnose.log"
