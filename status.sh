#!/usr/bin/env bash
# One command that answers: what is running, how far has it got, how does it look, and is
# anything broken.  Written for a sweep that runs overnight, so it assumes nothing about which
# log or which model and simply reports what is there.
#
#   bash status.sh               # the full picture
#   watch -n 60 bash status.sh   # refresh every minute
#
# Deliberately plain: no arrays, no [[ ]], no process substitution.
#
# FIVE THINGS THIS USED TO GET WRONG, all of them making it a window onto the wrong thing:
#
#   1. It read `ls -t sweep_*.log smoke_*.log`. The real logs are logs/parallel/<model>_<scene>.log
#      and logs/<something>_sweep_*.log, so the only matches were September leftovers and every
#      number in the "problems" block was taken from a stale file. It reads every log this campaign
#      wrote now, newest first, and says how many it looked at.
#   2. `grep -c 'level N done'` counted the WHOLE HISTORY of run_progress.txt, which is appended to
#      across runs: it reported 653 finished levels for a 17-level sweep. It counts distinct
#      (job, level) pairs from the CURRENT session now -- see the awk below.
#   3. It hard-coded `stage1.1`. A five-scene sweep has five tags and it reported one, mixed with
#      the September campaign's files. The scenes are discovered from the tags instead.
#   4. `pgrep -cf 'main.py --model'` matches its own command line, so it reported a phantom worker.
#      It asks for the Isaac interpreter's python, which cannot appear in the pattern.
#   5. `exit 0` when there was no log meant the rest of the report never printed.

cd "$(dirname "$0")" || exit 1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
echo

# --- what is running ----------------------------------------------------------------------------
# The worker is `<isaacsim>/kit/python/bin/python3 main.py ...`. Asking for that exact string cannot
# match this script, whose own command line contains "status.sh" and the word pgrep but not that path.
echo "--- running ---"
RUNNING=$(pgrep -af '/kit/python/bin/python3 main\.py' 2>/dev/null)
if [ -n "$RUNNING" ]; then
    echo "$RUNNING" | cut -c1-140 | sed 's/^/  /'
    echo "  workers: $(pgrep -cf '/kit/python/bin/python3 main\.py' 2>/dev/null)"
else
    echo "  no Isaac worker is running"
fi
DISPATCH=$(pgrep -af 'run_parallel_sweep\.sh' 2>/dev/null | grep -v 'grep\|status.sh' | head -1)
[ -n "$DISPATCH" ] && echo "  dispatcher: $(echo "$DISPATCH" | cut -c1-120)"
echo

# --- the logs this campaign wrote ---------------------------------------------------------------
echo "--- logs ---"
# Newest first; both layouts. Nothing is assumed about a fixed name.
LOGS=$(ls -t logs/parallel/*.log logs/*.log 2>/dev/null | head -10)
if [ -z "$LOGS" ]; then
    echo "  no logs under logs/ or logs/parallel/"
else
    echo "$LOGS" | head -5 | while read -r one; do
        printf '  %-58s %6s lines  %s\n' "$one" \
            "$(wc -l < "$one" 2>/dev/null | tr -d ' ')" \
            "$(date -r "$one" '+%H:%M:%S' 2>/dev/null)"
    done
    echo "  (searching $(echo "$LOGS" | wc -l | tr -d ' ') log file(s) below)"
fi
echo

# --- progress, per job ---------------------------------------------------------------------------
echo "--- progress: distinct (job, level) pairs from THIS session ---"
PROGRESS="$(pwd)/run_progress.txt"
[ -f "$PROGRESS" ] || PROGRESS="$(dirname "$(pwd)")/run_progress.txt"
if [ ! -f "$PROGRESS" ]; then
    echo "  (no run_progress.txt at $PROGRESS -- main.py writes it next to results/)"
else
    # Count a level once per JOB. A parallel sweep interleaves several jobs in one file, so a plain
    # `grep -c` adds their levels together, and an appended-to file also counts every earlier run.
    # The job tag is the [bracketed] prefix main.py writes; lines from before that prefix existed
    # count against the run whose "main start" line is nearest above them.
    awk '
        /level [0-9]+ done/ {
            line = $0
            # The JOB label is the bracketed prefix main.py writes on every durable line. Do NOT try to
            # read one off the "main start" line: that line carries `levels=[0, 1, ...]`, so the first
            # bracket on it is the LEVEL LIST and the counter was keyed on that.
            job = "(untagged)"
            if (match(line, /\[[^]]*stage1\.[0-9][^]]*\]/)) { job = substr(line, RSTART, RLENGTH) }
            else if (match(line, /\[[A-Za-z0-9_.-]+\]/)) { job = substr(line, RSTART, RLENGTH) }
            if (match(line, /level [0-9]+ done/)) { lvl = substr(line, RSTART, RLENGTH) }
            key = job " " lvl
            if (!(key in seen)) {
                seen[key] = 1
                count[job]++
            }
        }
        END {
            total = 0
            for (job in count) {
                printf "  %-52s %2d level(s)\n", job, count[job]
                total += count[job]
            }
            if (total == 0) print "  no level has finished in this file yet"
            else printf "  %-52s %2d pair(s) total\n", "TOTAL", total
        }
    ' "$PROGRESS"
    echo
    echo "  most recent 8 lines:"
    tail -8 "$PROGRESS" | cut -c1-130 | sed 's/^/    /'
fi
echo

# --- problems ------------------------------------------------------------------------------------
echo "--- problems (across the log files listed above) ---"
if [ -n "$LOGS" ]; then
    # `grep -h PATTERN $LOGS | wc -l` counts MATCHES across all files. `grep -hc` would print one number
    # per file, and piping that to wc -l counts FILES -- which reads as a flat 10 whatever happened.
    #
    # The quota pattern must be SPECIFIC: a bare "quota" matched Isaac's own extension name
    # "_bad_quota" and reported seven account problems that were not there.
    echo "  quota/rate-limit lines : $(grep -hiE 'HTTP 429|HTTP 402|insufficient_quota|exceeded your current quota|rate.?limit exceeded' $LOGS 2>/dev/null | wc -l | tr -d ' ')"
    echo "  tracebacks             : $(grep -h 'Traceback (most recent call last)' $LOGS 2>/dev/null | wc -l | tr -d ' ')"
    echo "  HTTP 400 bad JSON      : $(grep -h 'HTTP 400' $LOGS 2>/dev/null | wc -l | tr -d ' ')"
    echo "  REQUEST ERROR          : $(grep -h 'REQUEST ERROR' $LOGS 2>/dev/null | wc -l | tr -d ' ')"
    echo "  marker swaps           : $(grep -h 'prompt now says' $LOGS 2>/dev/null | wc -l | tr -d ' ')"
else
    echo "  no logs to search"
fi
echo

# --- results on disk -----------------------------------------------------------------------------
echo "--- results on disk (this protocol only) ---"
if [ -d results ]; then
    # Discover the scenes from the TAG NAMES, not by assuming stage1.1. `find -name '*stage1.*'`
    # returns DIRECTORIES; the scene has to be extracted from each name, and only the bare
    # "stageN.M" may survive, because a tag can carry the scene twice (…-stage1.1-…-stage1.1-…).
    SCENES=$(find results -maxdepth 3 -type d -name '*stage1.*' 2>/dev/null \
        | sed 's/.*[^0-9]\(stage1\.[0-9][0-9]*\).*/\1/' \
        | grep -E '^stage1\.[0-9]+$' | sort -u | tr '\n' ' ')
    if [ -z "$SCENES" ]; then
        echo "  no scene-tagged result directory found"
    else
        echo "  scenes found: $SCENES"
        for one_scene in $SCENES; do
            n=$(find results -maxdepth 3 -type d -name "*${one_scene}*" 2>/dev/null | wc -l | tr -d ' ')
            e=$(find results -maxdepth 3 -type d -name "*${one_scene}*" 2>/dev/null \
                | while read -r tagdir; do
                      ls "$tagdir"/episode_*.json 2>/dev/null | grep -v _steps | wc -l
                  done | awk '{s+=$1} END {print s+0}')
            printf '    %-10s %3s tag dir(s)  %5s episode(s)\n' "$one_scene" "$n" "$e"
        done
        echo
        echo "  per model (episode records, all scenes):"
        # EPISODES LIVE UNDER results/level<N>/<model>/<tag>/, not under results/<model>/ -- the
        # top-level model directory holds only checkpoints and CSVs, so counting there reported zero
        # for every model however much data existed.
        for d in results/level*/; do
            [ -d "$d" ] || continue
            for m in "$d"*/; do
                [ -d "$m" ] || continue
                printf '%s\n' "$(basename "$m")"
            done
        done | sort -u | while read -r name; do
            e=$(find results/level* -maxdepth 2 -type d -name "$name" 2>/dev/null \
                | while read -r modeldir; do
                      ls "$modeldir"/*/episode_*.json 2>/dev/null | grep -v _steps | wc -l
                  done | awk '{s+=$1} END {print s+0}')
            [ "$e" = "0" ] && continue
            printf '    %-42s %4s episode(s)\n' "$name" "$e"
        done
    fi
else
    echo "  no results/ directory yet"
fi
echo

echo "--- newest result files ---"
find results -name 'episode_*.json' -newermt '-1 day' 2>/dev/null | grep -v _steps \
    | xargs ls -t 2>/dev/null | head -6 | sed 's/^/  /'
echo
echo "To stop a SWEEP:     bash stop.sh   (kills the recorded dispatcher pid and its process group)"
echo "To stop a DIRECT run: bash stop.sh cannot; see its output for the pid, or:"
echo "                     kill -- -\$(ps -o pgid= -p <pid of main.py> | tr -d ' ')"
echo "To resume later:     the runner uses --resume, so re-running it continues."
