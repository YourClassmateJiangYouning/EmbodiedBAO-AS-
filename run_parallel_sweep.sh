#!/usr/bin/env bash
# Run (model x scene) jobs in parallel, resumable, one log per job.
#
# WHY PARALLEL: main.py starts Isaac Sim every time, which measured about 150 s before a single
# episode runs.  Five scenes sequentially pay that five times per model; run concurrently they pay it
# once.  The machine has 32 cores, 98 GB free RAM and one RTX 5090 with 32 GB, so PARALLEL=4 leaves
# headroom -- raise it only after watching nvidia-smi.
#
# SAFETY: every job passes --resume, so re-running this script skips episodes already finished.  A
# crashed job costs the episode in flight, not the run.
#
# USAGE
#   export BOYUE_API_KEY="$(cat ~/.bao_key)"
#   bash run_parallel_sweep.sh                       # gpt-4o-mini and glm-4.6v, all 5 scenes
#   bash run_parallel_sweep.sh gpt-4o-mini           # one model
#   PARALLEL=5 EPISODES=5 bash run_parallel_sweep.sh gpt-4o-mini glm-4.6v
set -u
cd "$(dirname "$0")"

ISAAC_PY="${ISAAC_PY:-/home/ybh/isaacsim/python.sh}"
# WORKER BUDGET -- READ THIS BEFORE RAISING IT.
#
# PARALLEL is the number of Isaac Sim instances THIS SCRIPT will run, and it must be a budget for the
# WHOLE MACHINE, not for one invocation.  On 2026-10-15 it was treated as per-invocation: three
# dispatchers were started, each with the default 4, so up to 12 Isaac instances ran at once.  Each
# Isaac instance holds roughly 16-19 GB resident, so 12 of them asked for ~200 GB on a 125 GB machine.
# The kernel OOM killer then killed workers mid-level (dmesg: "Out of memory: Killed process ... python3
# total-vm:167779044kB, anon-rss:18336504kB", 78 "oom" lines in the job logs), the load average reached
# 104 on 32 cores, and nvidia-smi briefly lost contact with the driver.
#
# The arithmetic that matters: 4 x ~19 GB = ~76 GB, which leaves room for the OS on a 125 GB box.
# 8 x ~19 GB = ~152 GB, which does not.  So the cap is 4 and the lock below enforces one dispatcher.
MAX_PARALLEL=4
# DEFAULT 3, not 4, and MAX 4.  The cap is what the arithmetic allows
# (4 x ~19 GB = ~76 GB of 125 GB), but xargs starts a replacement the moment a worker EXITS, and a
# worker that has just exited has not necessarily had its pages reclaimed yet.  In that overlap window
# 4 new + 1 lingering is 5 x ~19 GB = ~95 GB, which measured close enough to the edge on 2026-10-15
# that the dispatcher was killed and the lock left behind.  Three leaves ~19 GB of headroom for the
# overlap.  Raise to 4 with PARALLEL=4 if the machine is otherwise idle and the logs show memory
# plateauing well below the total.
PARALLEL="${PARALLEL:-3}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-30}"
TAG_PREFIX="${TAG_PREFIX:-full}"

# ===========================================================================
# ALL VALIDATION RUNS BEFORE THE DETACH, ON PURPOSE.
#
# It used to run after `exec setsid "$SELF" "$@"`, and util-linux setsid FORKS when the caller is
# already a process-group leader.  The original process then exits 0 and the child's refusal is lost:
# `PARALLEL=99 bash run_parallel_sweep.sh; echo $?` printed 0 while nothing had started.  Every check
# that can refuse therefore sits above the detach, where the exit status still belongs to the caller.
# The detached re-exec runs them again; they are cheap and idempotent.
# ===========================================================================

# ---- SCENES ------------------------------------------------------------------------------------
# An explicitly empty list is a mistake, not "use all five".
#
# `SCENES=(${SCENES:-stage1.1 ...})` treats an empty value as unset, so an `${#SCENES[@]} -eq 0` guard
# placed after it can never fire: empty had already become five scenes.  On 2026-10-15
# `SCENES="" bash run_parallel_sweep.sh` launched five real gpt-4o-mini jobs for exactly that reason.
#
# TEST THE RAW VALUE, BEFORE the `:-` substitution, so "set but empty" is refused.
if [ "${SCENES+set}" = "set" ] && [ -z "${SCENES}" ]; then
    echo "SCENES is set but empty.  Name the scenes explicitly, e.g. SCENES=\"stage1.5\"," >&2
    echo "or unset it to use all five.  Refusing to guess." >&2
    exit 1
fi
SCENES=(${SCENES:-stage1.1 stage1.2 stage1.3 stage1.4 stage1.5})
# Set but naming no scene at all (`SCENES=" "`, `SCENES=","`) expands to zero words.  This guard is
# reachable now; before, the substitution above shadowed it.
if [ "${#SCENES[@]}" -eq 0 ]; then
    echo "SCENES is set but names no scene.  Refusing to start with an empty job list." >&2
    exit 1
fi

# ---- PARALLEL ----------------------------------------------------------------------------------
# PARALLEL must be a POSITIVE INTEGER no greater than MAX_PARALLEL.
#
# The old check was only `[ "$PARALLEL" -gt "$MAX_PARALLEL" ]`, and every one of these slipped past it:
#   * PARALLEL=0    -- 0 is not greater than 4, and GNU `xargs -P 0` means UNLIMITED concurrency,
#                      i.e. one Isaac Sim instance per job at ~19 GB each;
#   * PARALLEL=abc  -- `[ abc -gt 4 ]` prints "integer expression expected" and returns 2, which `if`
#                      reads as false; xargs then aborts on the invalid -P value;
#   * PARALLEL=-3   -- also not greater than 4, and a negative -P is not valid.
# Match on digits FIRST so a non-numeric value is rejected with our own message, never by xargs.
case "$PARALLEL" in
    ''|*[!0-9]*)
        echo "PARALLEL must be a positive integer no greater than $MAX_PARALLEL (got '$PARALLEL')." >&2
        echo "GNU xargs reads PARALLEL=0 as 'no limit' -- one Isaac Sim instance per job, ~19 GB" >&2
        echo "each -- and rejects a non-numeric or negative value.  Refusing to start." >&2
        exit 1
        ;;
esac
if [ "$PARALLEL" -lt 1 ] || [ "$PARALLEL" -gt "$MAX_PARALLEL" ]; then
    echo "PARALLEL=$PARALLEL is outside the machine budget of 1..$MAX_PARALLEL." >&2
    if [ "$PARALLEL" -gt "$MAX_PARALLEL" ]; then
        echo "Each Isaac instance holds ~16-19 GB resident; $PARALLEL of them would ask for ~$(( PARALLEL * 19 )) GB" >&2
        echo "on a 125 GB machine and the kernel would start killing workers.  Refusing to start." >&2
    else
        echo "PARALLEL must be at least 1: 0 means 'no limit' to xargs, not 'no workers'." >&2
    fi
    exit 1
fi

# ---- environment and interpreter ---------------------------------------------------------------
if [ -z "${BOYUE_API_KEY:-}${TAOTOKEN_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
    echo "no API key in the environment (BOYUE_API_KEY / TAOTOKEN_API_KEY / OPENAI_API_KEY)" >&2
    exit 1
fi
if [ ! -x "$ISAAC_PY" ]; then
    echo "ISAAC_PY is not executable: $ISAAC_PY" >&2
    exit 1
fi

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(gpt-4o-mini glm-4.6v)
fi

# ===========================================================================
# ONE DISPATCHER AT A TIME, ENFORCED BY THE KERNEL RATHER THAN BY A PID CHECK.
#
# A second dispatcher multiplies the worker count by its own PARALLEL, which is exactly how the
# 2026-10-15 OOM happened.
#
# WHY NOT `kill -0 "$OTHER"`: kill(2) resolves a pid only in the CALLER's PID namespace.  A dispatcher
# started normally is invisible from a shell with a PID namespace of its own (bwrap --unshare-pid), so
# `kill -0` failed there, the code concluded the dispatcher was DEAD, deleted a live lock, took it, and
# added PARALLEL more Isaac instances on top of the running ones.  The safety device caused the OOM it
# exists to prevent.
#
# flock(2) has no such problem: the lock belongs to the open file description, and the kernel releases
# it when the holding process dies, in ANY namespace.  The lock file's CONTENTS carry no authority --
# an absent or unreadable pid is never taken as proof of death, because the pid file is only a
# convenience for stop.sh and for the one-transition case described below.
#
# NOTHING HERE REMOVES THE LOCK FILE.  Deleting it lets the next dispatcher create a second inode and
# lock that, so two dispatchers could each hold "the" lock.  fd 9 stays open for the life of this
# process; the kernel drops the lock at exit, however the process dies.
# ===========================================================================
LOCK="logs/parallel/.dispatcher.lock"
PIDFILE="logs/parallel/.dispatcher.pid"
mkdir -p logs/parallel || { echo "cannot create logs/parallel.  Refusing to start." >&2; exit 1; }

if ! command -v flock >/dev/null 2>&1; then
    echo "flock (util-linux) is not installed, so the one-dispatcher rule cannot be enforced." >&2
    echo "Refusing to start rather than run without the lock that prevents an OOM." >&2
    exit 1
fi
if ! touch "$LOCK" 2>/dev/null; then
    echo "cannot create the dispatcher lock $LOCK.  Refusing to start." >&2
    exit 1
fi

# A dispatcher recorded in the pid file (or, from the old layout, in the lock file) that is still
# alive.  This is a SECONDARY check, needed only while an old-style dispatcher may still be running --
# the old version of this script holds no flock, so flock alone cannot see it.  It is deliberately
# one-directional: it can only ever say "another dispatcher is alive", never "this one is dead".  A
# pid that is absent, unreadable, or not this script is NOT evidence of death; flock is the authority.
other_dispatcher_pid() {
    local source pid cmd
    for source in "$PIDFILE" "$LOCK"; do
        [ -r "$source" ] || continue
        pid="$(tr -dc '0-9' < "$source" 2>/dev/null)"
        [ -n "$pid" ] || continue
        [ "$pid" != "$$" ] || continue
        [ -r "/proc/$pid/cmdline" ] || continue
        cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmd" in
            *run_parallel_sweep.sh*)
                printf '%s\n' "$pid"
                return 0
                ;;
        esac
    done
    return 1
}

refuse_if_busy() {
    local other
    other="$(other_dispatcher_pid || true)"
    echo "another dispatcher is already running${other:+ (pid $other)}." >&2
    echo "Its PARALLEL workers are part of the same machine budget, so a second one would double" >&2
    echo "the load.  Wait for it, or pass more models to the one that is running.  Refusing." >&2
    exit 1
}

# PRE-DETACH PROBE.  The authoritative acquisition is below, after the detach; that one cannot report
# a refusal to the caller, because setsid forks.  This probe can, so a busy dispatcher is refused with
# exit 1 here.  The probe's subshell releases the lock as it exits; the acquisition below is what
# actually holds it, and if the microsecond between them is lost the acquire still refuses.
if [ "${BAO_SWEEP_DETACHED:-0}" != "1" ]; then
    if ! ( exec 8>>"$LOCK"; flock -n 8 ) 2>/dev/null; then
        refuse_if_busy
    fi
    other_dispatcher_pid >/dev/null && refuse_if_busy
fi

# START IT DETACHED, so a dying caller cannot take the sweep with it.
#
# A sweep runs for hours and the thing that started it does not live that long: an SSH session ends,
# a terminal closes, a supervisor is torn down.  setsid puts the script in its own session with no
# controlling terminal, which is what survives that.  Detaching here means the caller does not have to
# remember -- the original operator practice was `setsid nohup ... &`, and a shell function that
# forgets it is a sweep that dies at an arbitrary point with no error anywhere.
#
# Re-exec once, guarded by an environment variable so the second run cannot loop.
#
# WHAT THIS COMMENT USED TO CLAIM, AND WHY IT WAS WRONG: it said dispatchers started this way "died
# within 10-20 minutes while workers kept running as orphans", citing 2026-10-15.  That was my
# misreading: I checked `kill -0 <pid>` from inside a sandboxed shell whose PID namespace did not
# contain the dispatcher, read the inevitable failure as "the process is dead", and wrote the
# mechanism down.  The dispatcher in question (pid 2208586, started 13:14) was alive the whole time --
# 23 minutes later it was still dispatching, with its four workers in its own session.  Every
# dispatcher that actually stopped that day was stopped by an explicit kill of mine, or by the kernel
# OOM killer during the 12:52-13:02 over-subscription.  Detaching is still worth doing, for the SSH
# reason above, but nothing about sandbox teardown was ever observed.
if [ "${BAO_SWEEP_DETACHED:-0}" != "1" ]; then
    export BAO_SWEEP_DETACHED=1
    # Absolute path: setsid searches PATH for a bare name, so "./run_parallel_sweep.sh" style
    # self-exec fails with "No such file or directory" even though the file is right there.
    SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    if command -v setsid >/dev/null 2>&1; then
        exec setsid "$SELF" "$@"
    fi
    echo "warning: setsid not found; running in the caller's session, which an SSH drop would kill" >&2
fi

# AUTHORITATIVE ACQUISITION.  fd 9 is opened once and held; flock -n fails immediately if any other
# process holds the lock, wherever that process lives in the pid namespace tree.
exec 9>>"$LOCK" || { echo "cannot open the dispatcher lock $LOCK.  Refusing to start." >&2; exit 1; }
if ! flock -n 9; then
    refuse_if_busy
fi
# Transitional guard: a dispatcher started by the old version of this script holds no flock, so it is
# still refused by its recorded pid if that pid is visible and is this script.
other_dispatcher_pid >/dev/null && refuse_if_busy

if ! echo "$$" > "$PIDFILE"; then
    echo "cannot write the dispatcher pid file $PIDFILE.  Refusing to start." >&2
    exit 1
fi

XARGS_PID=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    # Stop the xargs FEEDER first: while it lives it keeps starting queued jobs, which is why killing
    # the workers alone never stopped a sweep.  In-flight workers are left to stop.sh (or to finish).
    if [ -n "$XARGS_PID" ] && kill -0 "$XARGS_PID" 2>/dev/null; then
        kill "$XARGS_PID" 2>/dev/null || true
    fi
    # Remove the pid file ONLY if it still holds OUR pid.  The old trap removed whatever pid it found,
    # so a dispatcher that lost the race deleted the winner's lock.
    if [ "$(cat "$PIDFILE" 2>/dev/null || true)" = "$$" ]; then
        rm -f "$PIDFILE"
    fi
    # The lock file is deliberately left in place; closing fd 9 releases the flock.
    exec 9>&- 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

JOBS=()
for m in "${MODELS[@]}"; do
    for s in "${SCENES[@]}"; do
        JOBS+=("${m}|${s}")
    done
done

echo "=== $(date) ==="
echo "models : ${MODELS[*]}"
echo "scenes : ${SCENES[*]}"
echo "jobs   : ${#JOBS[@]}   parallel: ${PARALLEL}   episodes/level: ${EPISODES}   max_steps: ${MAX_STEPS}"
echo

worker() {
    local job="$1"
    local model="${job%%|*}"
    local scene="${job##*|}"
    local safe="${model//\//_}"
    local log="logs/parallel/${safe}_${scene}.log"
    local t0
    t0=$(date +%s)
    echo "START ${model} ${scene}  $(date +%H:%M:%S)"
    "$ISAAC_PY" main.py \
        --model "$model" --scene "$scene" --all-levels \
        --episodes "$EPISODES" --max_steps "$MAX_STEPS" \
        --headless --resume --tag "${TAG_PREFIX}_${safe}" \
        > "$log" 2>&1
    local rc=$?
    echo "END   ${model} ${scene}  exit=${rc}  $(( $(date +%s) - t0 ))s  log=${log}"
    return $rc
}
export -f worker
export ISAAC_PY EPISODES MAX_STEPS TAG_PREFIX

# The feeder runs in the background so the cleanup trap can kill it by pid when this dispatcher is
# signalled; `wait` keeps the dispatcher alive for the whole sweep.
printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'worker "$@"' _ {} &
XARGS_PID=$!
wait "$XARGS_PID"
XARGS_RC=$?

echo
if [ "$XARGS_RC" -ne 0 ]; then
    echo "note: xargs exited $XARGS_RC; xargs reports 123 when a job failed, so check the per-job logs"
fi
echo "=== all jobs dispatched, $(date) ==="
echo "per-job logs : logs/parallel/"
echo "results      : results/level*/<model>/<tag>-*/"
echo "summaries    : $(find results -name 'summary_*.json' 2>/dev/null | wc -l) so far"
echo "re-run this exact command to resume anything that failed"

# Propagate the job status.  Before this the script ended on the `echo` above, so the
# EXIT trap captured $? = 0 and cleanup exited 0 whatever the jobs did: xargs reports
# 123 when any worker fails, and a whole sweep could fail while the launcher -- and
# anything reading its status -- saw success.  `exit` is the last command, so the trap
# sees XARGS_RC and re-exits with it.
exit "${XARGS_RC:-0}"
