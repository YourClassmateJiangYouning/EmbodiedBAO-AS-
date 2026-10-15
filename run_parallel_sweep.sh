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
SCENES=(${SCENES:-stage1.1 stage1.2 stage1.3 stage1.4 stage1.5})

# `SCENES=""` used to fall through ${SCENES:-...}, which treats empty as unset, so "no scenes" silently
# became "all five scenes".  On 2026-10-15 I used SCENES="" to test this script safely and it launched
# five real gpt-4o-mini jobs instead -- a completed model, so pure waste, and it overwrote the running
# dispatcher's lock file.  An explicitly empty list is a mistake; refuse it rather than reinterpret it.
if [ "${SCENES+set}" = "set" ] && [ "${#SCENES[@]}" -eq 0 ]; then
    echo "SCENES is set but empty.  Name the scenes explicitly, e.g. SCENES=\"stage1.5\"," >&2
    echo "or unset it to use all five.  Refusing to guess." >&2
    exit 1
fi

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

if [ "$PARALLEL" -gt "$MAX_PARALLEL" ]; then
    echo "PARALLEL=$PARALLEL exceeds the machine budget of $MAX_PARALLEL." >&2
    echo "Each Isaac instance holds ~16-19 GB resident; $PARALLEL of them would ask for ~$(( PARALLEL * 19 )) GB" >&2
    echo "on a 125 GB machine and the kernel would start killing workers.  Refusing to start." >&2
    exit 1
fi

# ONE DISPATCHER AT A TIME, enforced rather than remembered.  A second dispatcher multiplies the
# worker count by its own PARALLEL, which is exactly how the OOM happened.
LOCK="logs/parallel/.dispatcher.lock"
mkdir -p logs/parallel
if [ -e "$LOCK" ]; then
    OTHER="$(cat "$LOCK" 2>/dev/null || echo '')"
    if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
        echo "another dispatcher is already running (pid $OTHER)." >&2
        echo "Its PARALLEL workers are part of the same machine budget, so a second one would double" >&2
        echo "the load.  Wait for it, or pass more models to the one that is running.  Refusing." >&2
        exit 1
    fi
    echo "removing a stale dispatcher lock (pid ${OTHER:-unknown} is gone)"
    rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

JOBS=()
for m in "${MODELS[@]}"; do
    for s in "${SCENES[@]}"; do
        JOBS+=("${m}|${s}")
    done
done

mkdir -p logs/parallel
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

printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'worker "$@"' _ {}

echo
echo "=== all jobs dispatched, $(date) ==="
echo "per-job logs : logs/parallel/"
echo "results      : results/level*/<model>/<tag>-*/"
echo "summaries    : $(find results -name 'summary_*.json' 2>/dev/null | wc -l) so far"
echo "re-run this exact command to resume anything that failed"
