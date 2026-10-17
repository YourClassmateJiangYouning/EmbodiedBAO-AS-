#!/usr/bin/env bash
# Stop the sweep completely: the dispatcher first, then the runners.
#
# Why this exists: killing main.py is not enough.  Two things feed the runners -- run_all_models.sh is
# a bash loop that walks the roster and starts the next model as soon as the previous child exits, and
# run_parallel_sweep.sh feeds jobs through `xargs -P`, which starts the next queued job the moment a
# worker exits.  Killing only the workers makes the feeder launch the following job, which is how
# doubao started after a command that was supposed to stop everything.  The feeder has to die first.
#
# The live dispatcher is run_parallel_sweep.sh.  The old `ours` pattern was
# 'run_all_models.sh|main.py --model', which matches NEITHER it NOR its xargs feeder, so `bash stop.sh`
# killed the in-flight workers and left the sweep running.  The dispatcher is now stopped by the pid it
# recorded (logs/parallel/.dispatcher.pid, falling back to the old pid-in-the-lock-file layout), not by
# a pattern.
#
# It also refuses to touch anything that is not ours.  The machine is shared: someone has an
# IsaacLab job under /workspace/isaaclab that has been running for days, and killing it would be
# destroying a colleague's work.  Only processes whose command line mentions this repository are
# reported as ours.
#
#   bash stop.sh            # stop everything of ours
#   bash stop.sh --dry-run  # just show what would be stopped

cd "$(dirname "$0")" || exit 1
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY=1

LOCK="logs/parallel/.dispatcher.lock"
PIDFILE="logs/parallel/.dispatcher.pid"

ours() { pgrep -af 'run_parallel_sweep\.sh|run_all_models\.sh|main\.py --model' 2>/dev/null; }

# The pid the dispatcher recorded for itself.  A pid file is trusted ONLY after its command line is
# checked, because a stale file can name an unrelated process once the pid has been reused -- and this
# script's whole point is that it does not kill things that are not ours.  An absent or unreadable file
# is not evidence of anything; it just means there is no pid to act on.
dispatcher_pid() {
    local source pid cmd
    for source in "$PIDFILE" "$LOCK"; do
        [ -r "$source" ] || continue
        pid="$(tr -dc '0-9' < "$source" 2>/dev/null)"
        [ -n "$pid" ] || continue
        [ -r "/proc/$pid/cmdline" ] || continue
        cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmd" in
            *run_parallel_sweep.sh*)
                printf '%s\n' "$pid"
                return 0
                ;;
        esac
    done
    return 1
}

DPID="$(dispatcher_pid || true)"

echo "=== before ==="
ours || echo "  (nothing of ours is running)"
if [ -n "$DPID" ]; then
    echo "  dispatcher: pid $DPID  $(tr '\0' ' ' < "/proc/$DPID/cmdline" 2>/dev/null | cut -c1-100)"
else
    echo "  dispatcher: no live run_parallel_sweep.sh pid recorded ($PIDFILE / $LOCK absent, empty or not ours)"
fi
echo

if [ -n "$DRY" ]; then
    echo "--dry-run: nothing was stopped"
    exit 0
fi

# 1. The dispatcher, FIRST, by the pid it recorded.  While it lives its `xargs -P` feeder keeps
#    starting queued jobs, so killing the workers alone changes nothing.
if [ -n "$DPID" ]; then
    # If the dispatcher is a process-GROUP leader it was detached with setsid, and the whole sweep --
    # the xargs feeder plus every worker it started -- is in that group.  The pgid == pid guard means
    # we can never signal the caller's shell, a colleague's job, or another session's group.
    DPGID="$(ps -o pgid= -p "$DPID" 2>/dev/null | tr -d ' ')"
    if [ -n "$DPGID" ] && [ "$DPGID" = "$DPID" ]; then
        kill -TERM -- "-$DPGID" 2>/dev/null \
            && echo "stopped the dispatcher: pid $DPID and its process group $DPGID (xargs feeder + workers)" \
            || echo "could not signal the dispatcher group $DPGID (already gone?)"
    else
        kill -TERM "$DPID" 2>/dev/null \
            && echo "stopped the dispatcher: pid $DPID (run_parallel_sweep.sh; its cleanup kills the xargs feeder)" \
            || echo "could not signal the dispatcher pid $DPID (already gone?)"
    fi
else
    # NO PATTERN FALLBACK.  This used to `pkill -f 'run_all_models.sh'` and `pkill -f 'main.py
    # --model'` here.  On a shared machine that matches EVERY user's process whose command line
    # merely mentions the string -- it has killed this operator's own shell three times -- and it
    # does not even stop the sweep, whose xargs feeder carries neither string.  So when no pid is
    # recorded, this script now REPORTS and stops; it does not guess and it does not kill.
    echo "no dispatcher pid recorded; NOT falling back to a name pattern (see the comment in this file)"
    echo "  the dispatcher, if one is still dispatching, is one of:"
    pgrep -af 'run_parallel_sweep\.sh' 2>/dev/null | sed 's/^/    /' || echo "    (none)"
    echo "  its xargs feeder, which the pattern would never have matched:"
    pgrep -af 'xargs -P' 2>/dev/null | sed 's/^/    /' || echo "    (none)"
    echo "  kill one of THOSE by pid, then remove the stale pid file: rm -f $PIDFILE"
fi
sleep 2

# 4. Report, and list any surviving Isaac process with its command line, ours or not, so a
#    leftover can be killed by pid and a colleague's job can be recognised as not ours.
echo
echo "=== after ==="
ours || echo "  nothing of ours is running"

echo
echo "=== isaac processes still alive ==="
for pid in $(pgrep -f 'isaacsim' 2>/dev/null); do
    line=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)
    case "$line" in
        *EmbodiedBAO-AS-*) tag="OURS" ;;
        *isaaclab*)        tag="colleague (leave alone)" ;;
        *)                 tag="unknown" ;;
    esac
    printf '  %-8s %-8s %s\n' "$pid" "$tag" "$line"
done

echo
echo "If a line above says OURS, kill it by pid:   kill <pid>"
echo "Never pkill -f isaac on this machine."
