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
PARALLEL="${PARALLEL:-4}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-30}"
TAG_PREFIX="${TAG_PREFIX:-full}"
SCENES=(${SCENES:-stage1.1 stage1.2 stage1.3 stage1.4 stage1.5})

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
