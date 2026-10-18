#!/usr/bin/env bash
# Full 17-level run for one model.
#
# TWO THINGS THIS GETS RIGHT THAT THE FIRST VERSION DID NOT.
#   1. The log is actually written.  The first version defined LOG="logs/qwen-<stamp>.log",
#      printed it, and never redirected anything into it, so the only log was whatever the caller's
#      own redirect happened to be -- and the message named a file that did not exist.
#   2. The PID IT RECORDS IS THE PROCESS-GROUP LEADER.  setsid puts the CHILD into a new group, so the
#      pid of the `setsid` command itself is not in that group; recording it made
#      `kill -- -<pid>` fail with "no such process group".  __pgid is resolved after the fact.
set -u
cd "$(dirname "$0")"
export BOYUE_API_KEY="${BOYUE_API_KEY:-$(cat ~/.bao_key)}"
export BOYUE_BASE_URL="${BOYUE_BASE_URL:-http://35.220.164.252:3888/v1}"
export BAO_DISABLE_PROXY=1
# UNBUFFERED, OR A KILLED RUN LOSES ITS LAST LINES.  experiments.py used not to flush its
# per-episode progress print, so a SIGTERM left the log ending mid-episode and the finished
# episode's summary line was missing. That print flushes now; this makes the whole process
# unbuffered so no other stale buffer can do the same.
export PYTHONUNBUFFERED=1
MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SAFE="$(printf '%s' "$MODEL" | tr '/' '-')"
LOG="logs/run-${SAFE}-${STAMP}.log"
PIDFILE="logs/run-${SAFE}-${STAMP}.pid"

printf 'model  = %s\nlog    = %s\npidfile= %s\n' "$MODEL" "$LOG" "$PIDFILE"
setsid /home/ybh/isaacsim/python.sh main.py \
    --model "$MODEL" \
    --all-levels \
    --episodes 5 \
    --max_steps 30 \
    --headless --resume \
    > "$LOG" 2>&1 &
sleep 8
# The group leader is the python.sh child of setsid, not setsid's own pid.
LEADER="$(ps -eo pid,pgid,args | grep -F "python.sh main.py --model $MODEL" | grep -v grep | awk '{print $1}' | head -1)"
PGID="$(ps -o pgid= -p "${LEADER:-0}" 2>/dev/null | tr -d ' ')"
printf '%s\n' "${PGID:-$LEADER}" > "$PIDFILE"
printf 'pgid   = %s   (stop with: kill -- -%s)\n' "${PGID:-?}" "${PGID:-?}"
printf 'trace  : tail -f run_progress.txt\ntail   : tail -f %s\n' "$LOG"
