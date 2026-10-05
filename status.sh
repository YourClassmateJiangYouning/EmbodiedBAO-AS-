#!/usr/bin/env bash
# One command that answers: what is running, how far has it got, how does it look, and is
# anything broken.  Written for a sweep that runs overnight, so it assumes nothing about which
# log or which model and simply reports what is there.
#
#   bash status.sh            # the full picture
#   watch -n 60 bash status.sh   # refresh every minute
#
# Deliberately plain: no arrays, no [[ ]], no process substitution, because it was written on a
# machine with no bash to test it on and it has to work the first time on the one that has.

cd "$(dirname "$0")" || exit 1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
echo

echo "--- running ---"
RUNNING=$(pgrep -af 'main.py --model' 2>/dev/null)
if [ -n "$RUNNING" ]; then
    echo "$RUNNING" | sed 's/^/  /'
    echo "  count: $(pgrep -cf 'main.py --model' 2>/dev/null)"
else
    echo "  no main.py process is running"
fi
echo

LOG=$(ls -t sweep_*.log smoke_*.log 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
    echo "--- log ---"
    echo "  no sweep_*.log or smoke_*.log in $(pwd)"
    exit 0
fi
echo "--- newest log: $LOG ($(wc -l < "$LOG" | tr -d ' ') lines, updated $(date -r "$LOG" '+%H:%M:%S' 2>/dev/null)) ---"
tail -4 "$LOG" | sed 's/^/  /'
echo

echo "--- progress: levels completed, with the pass rate ladder ---"
DONE=$(grep -c 'level .* done' "$LOG" 2>/dev/null)
echo "  $DONE level(s) finished in this log"
grep -E 'level [0-9]+ done' "$LOG" 2>/dev/null | tail -20 | sed 's/^/  /'
echo

echo "--- problems ---"
echo "  quota/rate-limit lines : $(grep -icE 'quota|429|402|insufficient|exceeded your current' "$LOG" 2>/dev/null)"
echo "  tracebacks             : $(grep -c 'Traceback (most recent call last)' "$LOG" 2>/dev/null)"
echo "  non-zero invalid counts: $(grep -cE 'invalid_response_count[^0-9]*[1-9]' "$LOG" 2>/dev/null)"
echo "  marker swaps           : $(grep -c 'prompt now says' "$LOG" 2>/dev/null)"
echo

echo "--- results on disk, per model (levels x scenes) ---"
if [ -d results ]; then
    for d in results/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        levels=$(ls "$d"/level*.csv 2>/dev/null | wc -l | tr -d ' ')
        printf '  %-36s %2s level file(s)\n' "$name" "$levels"
    done
    total=$(ls results/*/level*.csv 2>/dev/null | wc -l | tr -d ' ')
    echo "  total level files: $total   (15 models x 17 levels = 255 when finished)"
else
    echo "  no results/ directory yet"
fi
echo

echo "--- the newest finished levels, most recent model last ---"
ls -t results/*/level*.csv 2>/dev/null | head -6 | sed 's/^/  /'
echo
echo "To stop everything:  pkill -f 'main.py --model'"
echo "To resume later:     the runner uses --resume, so re-running it continues."
