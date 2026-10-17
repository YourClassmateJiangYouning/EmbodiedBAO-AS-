#!/usr/bin/env bash
# Queue mimo-v2.6-pro and gemini-2.5-flash BEHIND whatever sweep is running now.
# Nothing here starts until the current dispatcher is gone.  Written 2026-10-15, after a direct
# start doubled the worker count on one GPU and the kernel OOM killer took the machine down.
set -u
cd "$(dirname "$0")"

export BOYUE_API_KEY="$(cat ~/.bao_key)"
LOCK="logs/parallel/.dispatcher.lock"
LOG="logs/queue_mimo_gemini.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "waiter started, pid $$"

# ---- 1. WAIT for the running dispatcher, read from its OWN lock file ----------
# Deliberately NOT pgrep/pkill: this script's command line contains the very strings those patterns
# would match, which is exactly how a pkill killed its own shell three times on 2026-10-15.
while :; do
    [ -e "$LOCK" ] || { say "no dispatcher lock: nothing is running"; break; }
    holder="$(cat "$LOCK" 2>/dev/null || true)"
    if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
        sleep 60
        continue
    fi
    say "lock held by '${holder:-?}', which is gone; treating it as stale"
    break
done

# ---- 2. let the finished workers release their ~18 GB each --------------------
say "waiting 60 s for the finished workers to release memory"
sleep 60
free_gb="$(free -g | awk 'NR==2 {print $7}')"
say "available memory: ${free_gb:-?} GB"
case "${free_gb:-x}" in
    ''|*[!0-9]*) say "could not read free memory; continuing anyway" ;;
    *) [ "$free_gb" -ge 70 ] || { say "ABORT: only ${free_gb} GB available, four Isaac need ~76 GB"; exit 1; } ;;
esac

# ---- 3. pre-flight: a model that cannot see must not burn hours --------------
MODELS=()
for m in mimo-v2.6-pro gemini-2.5-flash; do
    say "pre-flight $m"
    if python3 tools/check_model.py --model "$m" >>"$LOG" 2>&1; then
        say "  $m READY"
        MODELS+=("$m")
    else
        say "  $m FAILED pre-flight, leaving it out (details in $LOG)"
    fi
done
[ "${#MODELS[@]}" -gt 0 ] || { say "ABORT: no model passed pre-flight"; exit 1; }

# ---- 4. exactly ONE dispatcher, for the survivors -----------------------------
say "starting one dispatcher: ${MODELS[*]}"
exec bash run_parallel_sweep.sh "${MODELS[@]}"
