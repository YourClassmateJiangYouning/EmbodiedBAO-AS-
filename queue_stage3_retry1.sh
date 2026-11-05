#!/usr/bin/env bash
set -u
cd /home/eai/EmbodiedBAO-AS-
export BOYUE_API_KEY="$(cat ~/.bao_key)"
export BOYUE_BASE_URL="http://35.220.164.252:3888/v1"
export BAO_LLM_TIMEOUT=600
export BAO_MAX_RETRIES=1
LOG=logs/stage3_retry_S3F-090_queue.log
: > "$LOG"
echo "===== START S3F-090 $(date '+%F %T') =====" | tee -a "$LOG"
/home/ybh/isaacsim/python.sh stage3_runner.py --robot G1 --scene stage1.5 --model gpt-5 --plan reports/stage3_retry1_S3F-090_plan.csv --image-size 1024 --max-steps 30 --headless >> "$LOG" 2>&1
echo "===== END S3F-090 rc=$? $(date '+%F %T') =====" | tee -a "$LOG"
