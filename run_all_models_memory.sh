#!/usr/bin/env bash
# Run the Stage 2 repeated-attempt experiment over the whole model roster.
#
# One model at a time, six runs each: 12 attempts at A/S 0.80 then 5 at A/S 1.10,
# with a self-written note between attempts.  STAGE23_DESIGN.md is the spec.
#
# Usage (on the Isaac Sim workstation):
#     export BOYUE_API_KEY='...'
#     ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models_memory.sh
#     ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models_memory.sh qwen3-vl-32b-instruct
#
# Re-running skips rounds already recorded in the model's checkpoint, so an
# interrupted sweep continues rather than starting over.  Budget about two to three
# nights for the roster: each model is 6 x (17 attempts + 16 note calls), and a note
# call is the most expensive single call in the experiment because it carries the
# whole attempt record.
#
# Start it inside tmux or nohup, or a dropped SSH session ends the sweep:
#     tmux new -s bao2
#     ISAAC_PY=/home/ybh/isaacsim/python.sh bash run_all_models_memory.sh 2>&1 | tee memory_sweep.log
#     # Ctrl-b d to detach
set -u

cd "$(dirname "$0")"

if [ -z "${BOYUE_API_KEY:-}" ] && [ -z "${TAOTOKEN_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "no API key in the environment (BOYUE_API_KEY / TAOTOKEN_API_KEY / OPENAI_API_KEY)" >&2
    exit 1
fi

PLAIN_PY="${PY:-python3}"
ISAAC_PY="${ISAAC_PY:-${ISAACSIM_PYTHON:-/home/ybh/isaacsim/python.sh}}"
HEADLESS="${HEADLESS:-1}"

if [ ! -x "$ISAAC_PY" ]; then
    echo "Isaac launcher is missing or not executable: $ISAAC_PY" >&2
    echo "Set ISAAC_PY=/path/to/isaacsim/python.sh and re-run." >&2
    exit 1
fi

# Fail before the first model instead of after the eleventh.  Without this, a wrong
# interpreter makes every model exit non-zero with "No module named isaacsim" and
# the sweep still looks like it finished.
if ! "$ISAAC_PY" -c 'from isaacsim import SimulationApp' >/dev/null 2>&1; then
    echo "cannot import isaacsim with: $ISAAC_PY" >&2
    echo "That interpreter is not the Isaac Sim one; nothing has been run." >&2
    exit 1
fi

# The roster comes from models.json so the list has exactly one definition.
mapfile -t MODELS < <(
    "$PLAIN_PY" - <<'PY'
import json
with open("models.json", encoding="utf-8") as handle:
    for entry in json.load(handle)["models"]:
        print(entry["runner"])
PY
)

# The tag carries the MODEL NAME, the protocol version and the per-model request
# parameters.  All three matter, and the model name is not redundant with the
# results path: note files and per-step agent logs live under ``logs/{tag}/``, which
# is not model-scoped, so a shared stem would make every model in the sweep
# overwrite the previous model's notes -- and the notes are the manipulation.
# Rather than re-implement the composition in shell -- which went wrong twice on the
# threshold side, once with dots and once with a doubled suffix -- this asks the
# runner that owns it, so the printed stem IS what will be written.
model_stem() {
    "$PLAIN_PY" - "$1" <<'PY'
import sys
sys.path.insert(0, ".")
from memory_experiment import default_tag_for_model
print(default_tag_for_model(sys.argv[1]))
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

echo "roster : ${#MODELS[@]} model(s), 6 runs x 17 attempts each"
echo "isaac  : $ISAAC_PY"
echo "started: $(date)"
FAILED=""

for model in "${MODELS[@]}"; do
    stem="$(model_stem "$model")"
    echo
    echo "=============================================================="
    echo "model: $model    tag stem: $stem    $(date)"
    echo "=============================================================="
    # $HEADLESS_FLAG is deliberately unquoted: when HEADLESS=0 it is empty and must
    # expand to no argument at all.
    "$ISAAC_PY" memory_experiment.py \
        --model "$model" \
        --runs 1-6 \
        --tag "$stem" \
        --image_size 512 \
        --max_steps 30 \
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
    echo "re-run this script to resume them; finished rounds are not repeated" >&2
    exit 1
fi

echo "all models finished. records under results/memory/, notes under logs/"
echo "check progress with the command in the README's Stage 2 section"
