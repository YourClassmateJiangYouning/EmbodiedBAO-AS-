#!/usr/bin/env bash
# Verify the 16 m x 5 m EmbodiedBAO scene on an Isaac Sim workstation.
#
# Usage:
#   ISAAC_PY=/home/ybh/isaacsim/python.sh \
#     bash verify_professor_machine.sh 2>&1 | tee professor_verify.log
#
# Optional environment variables:
#   PY=python3              plain Python for offline tests
#   ISAAC_PY=/path/python.sh
#   RUN_MODEL_SMOKE=1       run one real model episode after diagnostics
#   MODEL=gpt-4o           model used by the optional smoke run
#
# Send back professor_verify.log and the professor_views/ directory.

set -u
set -o pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
ISAAC_PY="${ISAAC_PY:-/home/ybh/isaacsim/python.sh}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-0}"
MODEL="${MODEL:-gpt-4o}"
FAILURES=0

cd "$REPO_DIR" || exit 1

section() {
    echo
    echo "======================================================================"
    echo "== $*"
    echo "======================================================================"
}

run() {
    local label="$1"
    shift
    echo
    printf '%s' '--- $'
    printf ' %q' "$@"
    echo
    "$@"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "--- [$label] OK"
    else
        echo "--- [$label] FAILED (exit $rc)"
        FAILURES=$((FAILURES + 1))
    fi
    return 0
}

section "0. machine and repository"
echo "date       : $(date)"
echo "host       : $(hostname)"
echo "user       : $(whoami)"
echo "repo       : $REPO_DIR"
echo "DISPLAY    : ${DISPLAY:-<unset>}"
echo "plain py   : $(command -v "$PY" 2>/dev/null || echo MISSING)"
echo "Isaac py   : $ISAAC_PY"
echo "Isaac py ok: $([ -x "$ISAAC_PY" ] && echo yes || echo NO)"
echo "API key set: $([ -n "${BOYUE_API_KEY:-${TAOTOKEN_API_KEY:-${OPENAI_API_KEY:-}}}" ] && echo yes || echo no)"
run "git-revision" git log -1 --oneline
run "clean-status" git status --short

section "1. expected scene constants"
run "scene-constants" "$PY" -c '
import environment as e
assert e.ROOM_LENGTH_X == 16.0
assert e.ROOM_WIDTH_Z == 5.0
assert e.ROBOT_START_POS[0] == 0.5
assert e.MOVE_STEP == 0.75
assert e.WALL_X == 8.0
assert e.SUCCESS_X == 11.0
assert abs((e.WALL_X-e.ROBOT_START_POS[0])/e.MOVE_STEP - 10.0) < 1e-12
assert abs((e.SUCCESS_X-e.WALL_X)/e.MOVE_STEP - 4.0) < 1e-12
assert abs((e.ROOM_LENGTH_X-e.SUCCESS_X)/e.MOVE_STEP - 20/3) < 1e-12
print("room=16x5, start=.5, wall=8, goal>=11, far wall=16, step=.75")
print("10 forward moves to wall; 4 more to goal; 6.67-step run-out")
'

section "2. offline regression suites"
run "geometry-tests" "$PY" test_bao_geometry.py
run "integration-tests" "$PY" test_bao_integration.py
run "passability-probe" "$PY" tools/passability_probe.py

section "3. Isaac Sim import"
if [ ! -x "$ISAAC_PY" ]; then
    echo "Isaac launcher is missing or not executable: $ISAAC_PY"
    FAILURES=$((FAILURES + 1))
else
    run "isaac-import" "$ISAAC_PY" -c '
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import environment
    assert environment._HAS_ISAAC_SIM
    print("Isaac Sim and environment imported successfully")
finally:
    app.close()
'

    section "4. real Isaac scene and scripted traversal"
    run "isaac-smoke" "$ISAAC_PY" environment.py

    section "5. rendered diagnostics"
    rm -rf professor_views
    run "capture-level0" "$ISAAC_PY" capture_views.py --level 0 --outdir professor_views/level0
    run "capture-level5" "$ISAAC_PY" capture_views.py --level 5 --outdir professor_views/level5
    echo
    echo "Rendered files:"
    find professor_views -maxdepth 2 -type f -printf '%p  %s bytes\n' 2>/dev/null | sort || true
fi

if [ "$RUN_MODEL_SMOKE" = "1" ]; then
    section "6. optional one-episode model smoke test"
    if [ ! -x "$ISAAC_PY" ]; then
        echo "Skipped: Isaac launcher missing"
        FAILURES=$((FAILURES + 1))
    elif [ -z "${BOYUE_API_KEY:-${TAOTOKEN_API_KEY:-${OPENAI_API_KEY:-}}}" ]; then
        echo "Skipped: no supported API key environment variable is set"
        FAILURES=$((FAILURES + 1))
    else
        TAG="professor-smoke-$(date +%Y%m%d-%H%M%S)"
        run "model-smoke" "$ISAAC_PY" main.py \
            --model "$MODEL" --level 5 --episodes 1 --max_steps 30 \
            --headless --save_obs --tag "$TAG"
    fi
fi

section "7. manual acceptance checklist"
cat <<'EOF'
Open professor_views/level0 and professor_views/level5 and confirm:
  [ ] top.png shows a 16 m long, 5 m wide enclosed room.
  [ ] the blue obstacle wall is at the longitudinal midpoint x=8.
  [ ] the green far wall closes the room at x=16.
  [ ] the ceiling covers the complete floor; there is no open strip overhead.
  [ ] front.png clearly shows the channel and two dark vertical edge posts.
  [ ] behind.png is truly behind the obstacle and looks back through the channel.
  [ ] eye_start.png shows the distant opening (small but distinguishable).
  [ ] eye_near.png shows the opening clearly at approach distance.
  [ ] the wall-behind-goal region and green far wall are not black or blown out.
  [ ] Level 5 remains visibly narrower than Level 0.
EOF

section "RESULT"
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL AUTOMATED CHECKS PASSED"
    echo "Return professor_verify.log plus professor_views/ for visual review."
    exit 0
fi

echo "$FAILURES automated check(s) failed"
echo "Return the complete professor_verify.log; do not start scored experiments yet."
exit 1
