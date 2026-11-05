#!/usr/bin/env bash
set -u
cd /home/eai/EmbodiedBAO-AS-
export BOYUE_API_KEY="$(cat ~/.bao_key)"
export BOYUE_BASE_URL="http://35.220.164.252:3888/v1"
export BAO_LLM_TIMEOUT=120
export BAO_MAX_RETRIES=1
PLAN=reports/stage3_final100_resume_plan.csv
LOG=logs/stage3_remaining68_queue.log
: > "$LOG"
# Only robot/scene pairs that actually have remaining rows.  G1 stage1.1-1.3
# have no remaining rows and used to waste ~10 min each rebuilding empty scenes.
for pair in \
  "G1:stage1.4" "G1:stage1.5" \
  "R1:stage1.1" "R1:stage1.2" "R1:stage1.3" "R1:stage1.4" "R1:stage1.5"
do
  robot="${pair%%:*}"; scene="${pair##*:}"
  echo "===== START $robot $scene $(date '+%F %T') =====" | tee -a "$LOG"
  /home/ybh/isaacsim/python.sh stage3_runner.py \
    --robot "$robot" --scene "$scene" --plan "$PLAN" \
    --image-size 1024 --max-steps 30 --headless >> "$LOG" 2>&1
  echo "===== END $robot $scene rc=$? $(date '+%F %T') =====" | tee -a "$LOG"
done
echo "===== ALL DONE $(date '+%F %T') =====" | tee -a "$LOG"
