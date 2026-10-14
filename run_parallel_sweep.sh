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
