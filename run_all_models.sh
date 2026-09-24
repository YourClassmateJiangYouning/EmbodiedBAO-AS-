#!/usr/bin/env bash
# Run the full model roster, one model at a time.
#
# Each model gets its own tag so a run can be resumed with --resume and one
# model's results can never be mistaken for another's.  Scale: 6 Levels x 10
# episodes x 30 steps = a 60-episode, 1800-call ceiling per model.
#
# Usage (on the Isaac Sim workstation):
#     export BOYUE_API_KEY='...'
#     ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models.sh
#     ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models.sh qwen-vl-max
#
# Re-running skips episodes already recorded in the model's checkpoint, so an
# interrupted sweep continues rather than starting over.
#
# This runs for a long time -- budget days, not hours (11 models x up to 1800
# calls each).  Start it inside tmux or nohup, or a dropped SSH session will
# kill the sweep mid-model:
#     tmux new -s bao
#     ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models.sh 2>&1 | tee sweep.log
#     # Ctrl-b d to detach
#
# Every episode is written as it is scored, so an interruption loses at most the
# episode in flight; re-running with the same command resumes from the
# checkpoint.  HEADLESS=0 runs Isaac Sim with a window instead of headless.
set -u

cd "$(dirname "$0")"

if [ -z "${BOYUE_API_KEY:-}" ] && [ -z "${TAOTOKEN_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "no API key in the environment (BOYUE_API_KEY / TAOTOKEN_API_KEY / OPENAI_API_KEY)" >&2
    exit 1
fi

PLAIN_PY="${PY:-python3}"
# ISAAC_PY is the name the README and verify_professor_machine.sh use;
# ISAACSIM_PYTHON is accepted so older invocations keep working.
ISAAC_PY="${ISAAC_PY:-${ISAACSIM_PYTHON:-/home/ybh/isaacsim/python.sh}}"
HEADLESS="${HEADLESS:-1}"

if [ ! -x "$ISAAC_PY" ]; then
    echo "Isaac launcher is missing or not executable: $ISAAC_PY" >&2
    echo "Set ISAAC_PY=/path/to/isaacsim/python.sh and re-run." >&2
    exit 1
fi

# Fail before the first model instead of after the eleventh.  Without this, a
# wrong interpreter makes every model exit non-zero with "No module named
# isaacsim" and the sweep still looks like it finished.
if ! "$ISAAC_PY" -c 'from isaacsim import SimulationApp' >/dev/null 2>&1; then
    echo "cannot import isaacsim with: $ISAAC_PY" >&2
    echo "That interpreter is not the Isaac Sim one; nothing has been run." >&2
    exit 1
fi

# Read the roster from models.json so the list has exactly one definition.
mapfile -t MODELS < <(
    "$PLAIN_PY" - <<'PY'
import json
with open("models.json", encoding="utf-8") as handle:
    for entry in json.load(handle)["models"]:
        print(entry["runner"])
PY
)

# Tag each model with the SAME function main.py applies, so the sweep and a
# manual `main.py --tag ...` agree on where results live.  An earlier version
# rewrote the name in shell (`tr '/.' '--'`), which turned dots into dashes:
# gemini-2.5-pro became gemini-2-5-pro here but main.py's sanitize_tag keeps the
# dot, so a manual run and a sweep run of the same model would use different
# directories and --resume could not find the other's checkpoint.
model_tag() {
    "$PLAIN_PY" - "$1" <<'PY'
import sys
sys.path.insert(0, ".")
from persistence import sanitize_tag
print(sanitize_tag(sys.argv[1]))
PY
}

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "no models to run: models.json listed none and no names were given" >&2
    exit 1
fi

HEADLESS_FLAG=""
if [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
fi

echo "roster: ${#MODELS[@]} model(s)"
echo "isaac  : $ISAAC_PY"
echo "plain  : $PLAIN_PY"
echo "started: $(date)"
FAILED=""

for model in "${MODELS[@]}"; do
    tag="$(model_tag "$model")"
    echo
    echo "=============================================================="
    echo "model: $model    tag: $tag    $(date)"
    echo "=============================================================="
    # $HEADLESS_FLAG is deliberately unquoted: when HEADLESS=0 it is empty and
    # must expand to no argument at all.  (An empty array expansion would do
    # the same, but it trips `set -u` on bash older than 4.4.)
    "$ISAAC_PY" main.py \
        --model "$model" \
        --all-levels \
        --episodes 10 \
        --max_steps 30 \
        --image_size 512 \
        --tag "$tag" \
        $HEADLESS_FLAG \
        --resume
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "!! $model exited with status $status; continuing to the next model" >&2
        FAILED="$FAILED $model"
    fi
done

echo
echo "finished: $(date)"
if [ -n "$FAILED" ]; then
    echo "these model(s) exited non-zero:$FAILED" >&2
    echo "summaries under analysis/; re-run this script to resume them" >&2
    exit 1
fi

echo "all models finished. results under results/, summaries under analysis/"
echo "analyse with: $PLAIN_PY analysis.py --results_root results"
