#!/usr/bin/env bash
# Run the stage 1 laboratory sweep unattended: smoke one model, verify it, and only then
# launch all fifteen.
#
# Why a script and not a sequence of commands typed by hand: the decision to spend a day and a
# few hundred yuan has to be made by something that does not get bored.  The smoke run is one
# model over all seventeen levels, the verifier reads its log and its results, and the full
# sweep starts only if the verifier says the run is sound.  If it does not, this stops and leaves
# the evidence in logs/STATUS.txt.
#
# Usage, from the repository root, on the machine that has Isaac Sim:
#
#   export BOYUE_API_KEY='sk-...'
#   export ISAAC_PY=/home/ybh/isaacsim/python.sh
#   nohup bash run_stage1_lab.sh > run_stage1_lab.out 2>&1 &
#
# Then watch:  tail -f logs/STATUS.txt
#
# Everything is resumable: run_all_models.sh is invoked with --resume, so re-running this after
# an interruption continues rather than restarting.

set -uo pipefail

cd "$(dirname "$0")" || exit 1

SCENE="${SCENE:-stage1.1}"
SMOKE_MODEL="${SMOKE_MODEL:-gpt-4o-mini}"
ISAAC_PY="${ISAAC_PY:-/home/ybh/isaacsim/python.sh}"
export SCENE ISAAC_PY
export BAO_DISABLE_PROXY="${BAO_DISABLE_PROXY:-1}"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs
STATUS="logs/STATUS.txt"
SMOKE_LOG="logs/smoke-${SCENE}-${STAMP}.log"
SWEEP_LOG="logs/sweep-${SCENE}-${STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

: > "$STATUS"
log "run_stage1_lab: scene=$SCENE smoke_model=$SMOKE_MODEL"
log "isaac=$ISAAC_PY  python=$(command -v python3)"
log "git: $(git log --oneline -1 2>/dev/null)"

if [ -z "${BOYUE_API_KEY:-}" ]; then
    log "FAIL: BOYUE_API_KEY is not set.  Nothing was started."
    exit 1
fi
if [ ! -x "$ISAAC_PY" ]; then
    log "WARNING: $ISAAC_PY is not executable; the runner may still find Isaac another way."
fi

# 1. Usage baseline, so the cost of this sweep is measured rather than estimated.
if [ -f tools/usage_delta.py ]; then
    python3 tools/usage_delta.py --before >> "$STATUS" 2>&1 && log "usage baseline recorded"
fi

# 2. Smoke: one model, all seventeen levels, five markers each.
log "smoke starting -> $SMOKE_LOG"
bash run_all_models.sh "$SMOKE_MODEL" > "$SMOKE_LOG" 2>&1
SMOKE_RC=$?
log "smoke finished rc=$SMOKE_RC"

# 3. Verify before spending a day.
python3 tools/verify_stage1_lab.py --log "$SMOKE_LOG" --model "$SMOKE_MODEL" \
    > "logs/verify-${STAMP}.txt" 2>&1
VERIFY_RC=$?
tail -25 "logs/verify-${STAMP}.txt" | tee -a "$STATUS"

if [ "$VERIFY_RC" -ne 0 ]; then
    log "FAIL: the verifier rejected the smoke run.  The full sweep was NOT started."
    log "read: logs/verify-${STAMP}.txt   and   $SMOKE_LOG"
    log "the last lines of the smoke log:"
    tail -15 "$SMOKE_LOG" | tee -a "$STATUS"
    exit 1
fi

# 4. Full sweep: all fifteen models, detached, with the usage delta recorded when it ends.
log "verify passed; launching the full sweep -> $SWEEP_LOG"
nohup bash -c "bash run_all_models.sh; RC=\$?; echo \"[sweep rc=\$RC]\" >> '$STATUS'; \
    python3 tools/usage_delta.py --after >> '$STATUS' 2>&1; \
    python3 tools/verify_stage1_lab.py --log '$SWEEP_LOG' --model gpt-4o-mini \
        > logs/verify-sweep-${STAMP}.txt 2>&1; \
    echo \"[post-sweep verification in logs/verify-sweep-${STAMP}.txt]\" >> '$STATUS'" \
    > "$SWEEP_LOG" 2>&1 &
echo $! > "logs/sweep-${STAMP}.pid"

log "full sweep pid $(cat "logs/sweep-${STAMP}.pid")"
log "monitor with:  tail -f logs/STATUS.txt"
log "progress with: grep -c '\"passed\"' $SWEEP_LOG"
log "stop with:     kill \$(cat logs/sweep-${STAMP}.pid)   (never pkill -f isaac)"
log "done. This script exits; the sweep keeps running."
