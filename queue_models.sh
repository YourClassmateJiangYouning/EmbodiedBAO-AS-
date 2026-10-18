#!/usr/bin/env bash
# Wait for the dispatcher running now, then start ONE dispatcher covering several models, in the order
# given.
#
# WHY ONE CALL, NOT ONE CALL PER MODEL.
#
#   run_parallel_sweep.sh parallelises over (model x scene) jobs against a fixed pool of PARALLEL Isaac
#   instances, and MAX_PARALLEL caps that at 4. It builds the job list as
#
#       for m in MODELS: for s in SCENES: JOBS+=("m|s")
#
#   so the pool fills in model order and a model's scenes only start as the previous model's scenes
#   finish:
#
#       slots 1-4   A/1.1  A/1.2  A/1.3  A/1.4
#       next        A/1.5  B/1.1  B/1.2  B/1.3
#       next        B/1.4  B/1.5  C/1.1  C/1.2   ...
#
#   which keeps four slots busy and starts the models in the order asked for. An earlier version of this
#   script called the sweep once per model and waited for each call to return: that has NO overlap -- a
#   finished model leaves three slots idle -- which is not what a queue is for.
#
#   The INSTANCE COUNT DOES NOT GROW WITH THE NUMBER OF MODELS. It is MAX_PARALLEL always. What grows is
#   the job list. That is why passing five models here is safe and calling the sweep five times in
#   parallel would not be.
#
# WHY IT WAITS RATHER THAN STARTS NOW:
#
#   The dispatcher running now holds the flock, so a second dispatcher would be refused. This script
#   waits for it to disappear (its pid file is the authority, validated against /proc before use), waits
#   for the workers to release their ~18 GB each, and then starts the pool.
#
#   --dry-run  prints the plan and exits without waiting or starting anything.
#
# USAGE
#   bash queue_models.sh --dry-run A B C
#   nohup setsid bash queue_models.sh A B C > logs/queue_models_$(date +%H%M).log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1

LOCK="logs/parallel/.dispatcher.lock"
PIDFILE="logs/parallel/.dispatcher.pid"
LOG="logs/queue_models.log"
PARALLEL="${PARALLEL:-4}"

DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "usage: bash queue_models.sh [--dry-run] <model> [model...]" >&2
    exit 2
fi

mkdir -p logs/parallel
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

dispatcher_pid() {
    local src pid cmd
    for src in "$PIDFILE" "$LOCK"; do
        [ -r "$src" ] || continue
        pid="$(tr -dc '0-9' < "$src" 2>/dev/null)"
        [ -n "$pid" ] || continue
        [ -r "/proc/$pid/cmdline" ] || continue
        cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmd" in
            *run_parallel_sweep.sh*) printf '%s\n' "$pid"; return 0 ;;
        esac
    done
    return 1
}

say "queue: ${#MODELS[@]} model(s); PARALLEL=$PARALLEL (fixed instance count, NOT one per model)"
for i in "${!MODELS[@]}"; do say "  $((i + 1)). ${MODELS[$i]}"; done
say "job order: model order, then scene order; the pool stays at $PARALLEL instances throughout"
say "boundary looks like: <last model's 1.5> + <next model's 1.1 1.2 1.3>"

CURRENT="$(dispatcher_pid || true)"
if [ -n "$DRY" ]; then
    if [ -n "$CURRENT" ]; then
        say "--dry-run: dispatcher $CURRENT is running; this queue would wait for it, then start the"
        say "           ${#MODELS[@]}-model pool above. Nothing was waited for and nothing was started."
    else
        say "--dry-run: nothing is running; the pool above would start immediately."
    fi
    exit 0
fi

if [ -n "$CURRENT" ]; then
    say "waiting for dispatcher $CURRENT to finish"
    while kill -0 "$CURRENT" 2>/dev/null; do sleep 60; done
    say "dispatcher $CURRENT is gone; waiting 120 s for its workers to release memory"
    sleep 120
else
    say "no dispatcher running; starting the pool immediately"
fi

free_gb="$(free -g | awk 'NR==2 {print $7}')"
say "free memory now: ${free_gb:-?} GB"

before="$(find results -name 'episode_*.json' ! -name '*_steps.json' 2>/dev/null | wc -l)"
say "=== START ${#MODELS[@]}-model pool ==="
PARALLEL="$PARALLEL" bash run_parallel_sweep.sh "${MODELS[@]}"
status=$?
after="$(find results -name 'episode_*.json' ! -name '*_steps.json' 2>/dev/null | wc -l)"
produced=$((after - before))
say "=== queue finished: exit=$status, $produced episode(s) written ==="
if [ "$status" -eq 0 ] && [ "$produced" -eq 0 ]; then
    say "WARNING: exit 0 but NO episodes -- treating the queue as a failure"
    exit 1
fi
exit "$status"
