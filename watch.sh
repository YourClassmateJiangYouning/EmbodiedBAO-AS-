#!/usr/bin/env bash
# Follow a sweep and print one line per finished level, one banner per finished model, and
# anything that looks like a failure the moment it appears.
#
#   bash watch.sh                 # follows the newest sweep_*.log or smoke_*.log
#   bash watch.sh sweep_1.1.log   # follows a specific one
#
# Each line carries the pass rate, so the ladder can be watched as it is built: the wide
# apertures should pass, the narrow ones should fail, and the point where it flips is the
# measurement.  A model banner every seventeen levels means one model of fifteen is complete,
# which at the measured four minutes per level is roughly every seventy minutes.
#
# It reads the log rather than the results because the log is what the runner writes while it
# works; tail -F keeps following even if the file is recreated.

cd "$(dirname "$0")" || exit 1

LOG="$1"
if [ -z "$LOG" ]; then
    LOG=$(ls -t sweep_*.log smoke_*.log 2>/dev/null | head -1)
fi
if [ -z "$LOG" ]; then
    echo "no sweep_*.log or smoke_*.log here yet; pass one explicitly: bash watch.sh <log>"
    exit 1
fi

echo "watching $LOG   (Ctrl-C stops watching; it does NOT stop the experiment)"
echo "one line per finished level, one banner per finished model, failures as they happen"
echo

tail -n 3 -F "$LOG" 2>/dev/null | awk '
    /main start: model=/ {
        model = $0
        sub(/.*model=/, "", model)
        sub(/[ \t].*/, "", model)
        levels = 0
        printf "\n>>> MODEL START  %s\n", model
        fflush()
    }
    /level [0-9]+ done/ {
        levels++
        printf "    %-34s %2d/17  %s\n", model, levels, $0
        fflush()
        if (levels == 17) {
            printf ">>> MODEL DONE   %s  (17/17 levels)\n\n", model
            fflush()
        }
    }
    /Traceback \(most recent call last\)/ {
        printf "    !! TRACEBACK    %s\n", $0
        fflush()
    }
    /quota|rate.?limit|exceeded your current|HTTP 429|HTTP 402|401 Unauthorized/ {
        printf "    !! API PROBLEM  %s\n", $0
        fflush()
    }
    /invalid_response_count": [1-9]/ {
        printf "    !! BAD RESPONSE %s\n", $0
        fflush()
    }
'
