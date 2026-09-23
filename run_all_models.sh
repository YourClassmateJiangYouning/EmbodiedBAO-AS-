#!/usr/bin/env bash
# Run the full model roster, one model at a time.
#
# Each model gets its own tag so a run can be resumed with --resume and one
# model's results can never be mistaken for another's.  Scale: 6 Levels x 10
# episodes x 30 steps = a 60-episode, 1800-call ceiling per model.
#
# Usage:
#     export BOYUE_API_KEY='...'
#     bash run_all_models.sh              # every model in models.json
#     bash run_all_models.sh qwen-vl-max  # just one
#
# Re-running skips episodes already recorded in the model's checkpoint, so an
# interrupted sweep continues rather than starting over.
set -u

cd "$(dirname "$0")"

if [ -z "${BOYUE_API_KEY:-}" ] && [ -z "${TAOTOKEN_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "no API key in the environment (BOYUE_API_KEY / TAOTOKEN_API_KEY / OPENAI_API_KEY)" >&2
    exit 1
fi

PYTHON="${ISAACSIM_PYTHON:-python3}"

# Read the roster from models.json so the list has exactly one definition.
mapfile -t MODELS < <(
    python3 - <<'PY'
import json
with open("models.json", encoding="utf-8") as handle:
    for entry in json.load(handle)["models"]:
        print(entry["runner"])
PY
)

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
fi

echo "roster: ${#MODELS[@]} model(s)"
for model in "${MODELS[@]}"; do
    tag="$(printf '%s' "$model" | tr '/.' '--' | tr -cd 'A-Za-z0-9_-')"
    echo
    echo "=============================================================="
    echo "model: $model    tag: $tag"
    echo "=============================================================="
    "$PYTHON" main.py \
        --model "$model" \
        --all-levels \
        --episodes 10 \
        --max_steps 30 \
        --image_size 512 \
        --tag "$tag" \
        --resume
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "!! $model exited with status $status; continuing to the next model" >&2
    fi
done

echo
echo "all models finished. results under results/, summaries under analysis/"
