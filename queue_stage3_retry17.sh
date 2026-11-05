#!/usr/bin/env bash
set -u
cd /home/eai/EmbodiedBAO-AS-
export BOYUE_API_KEY="$(cat ~/.bao_key)"
export BOYUE_BASE_URL="http://35.220.164.252:3888/v1"
export BAO_LLM_TIMEOUT=300
export BAO_MAX_RETRIES=1
PLAN=reports/stage3_retry17_plan.csv
LOG=logs/stage3_retry17_queue.log
: > "$LOG"
for pair in "G1:stage1.4" "G1:stage1.5" "R1:stage1.1" "R1:stage1.2" "R1:stage1.5"; do
  robot="${pair%%:*}"; scene="${pair##*:}"
  python3 - "$robot" "$scene" <<'PY' || continue
import csv,sys
r=[x for x in csv.DictReader(open('reports/stage3_retry17_plan.csv')) if x['target_robot']==sys.argv[1] and x['source_scene']==sys.argv[2]]
raise SystemExit(0 if r else 1)
PY
  echo "===== START $robot $scene $(date '+%F %T') =====" | tee -a "$LOG"
  /home/ybh/isaacsim/python.sh stage3_runner.py --robot "$robot" --scene "$scene" --model "" --plan "$PLAN" --image-size 1024 --max-steps 30 --headless >> "$LOG" 2>&1
  echo "===== END $robot $scene rc=$? $(date '+%F %T') =====" | tee -a "$LOG"
done
echo "===== ALL DONE $(date '+%F %T') =====" | tee -a "$LOG"
