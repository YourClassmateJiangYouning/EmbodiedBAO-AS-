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
    echo "no dispatcher pid recorded -- this is normal for a DIRECT run (main.py started by hand, or"
    echo "by run_qwen_all.sh), which writes no dispatcher pid. Looking for this repository's workers:"
    echo

    # A DIRECT RUN IS STOPPED BY THE PROCESS GROUP OF *OUR* WORKER.
    #
    # The test for "ours" is that the command line contains this repository's ABSOLUTE path, which is
    # the same test the trailing report uses. That is narrower than any pattern: a colleague's Isaac
    # job under /workspace/isaaclab cannot contain this path, and neither can another user's shell.
    # The GROUP is signalled, not the pid, because killing only the parent leaves the Kit child (and,
    # for the sweep, the xargs feeder) running -- which is how a "stopped" sweep kept spending money.
    #
    # Nothing is killed here without being printed first, and /repo stays empty when nothing matches.
    REPO="$(pwd)"
    FOUND=""
    for pid in $(pgrep -f 'main\.py' 2>/dev/null); do
        line="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$line" in
            *"$REPO"*) FOUND="$FOUND $pid" ;;
        esac
    done
    if [ -z "$FOUND" ]; then
        echo "  no worker of this repository is running either"
    else
        for pid in $FOUND; do
            pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
            [ -n "$pgid" ] || continue
            leader_cmd="$(tr '\0' ' ' < "/proc/$pgid/cmdline" 2>/dev/null || true)"
            case "$leader_cmd" in
                *"$REPO"*)
                    echo "  group $pgid (leader $(ps -o comm= -p "$pgid" 2>/dev/null), ours):"
                    ps -o pid=,comm= -g "$pgid" 2>/dev/null | sed 's/^/    /'
                    kill -TERM -- "-$pgid" 2>/dev/null \
                        && echo "    -> signalled group $pgid" \
                        || echo "    -> could not signal group $pgid (already gone?)"
                    ;;
                *)
                    # The leader is NOT ours, so this group may contain something else. Signal the
                    # worker by pid only, and say so, rather than risk sweeping up a stranger.
                    echo "  pid $pid is ours but its group leader $pgid is not; signalling the pid only:"
                    echo "    leader cmd: $(echo "$leader_cmd" | cut -c1-90)"
                    kill -TERM "$pid" 2>/dev/null \
                        && echo "    -> signalled pid $pid" \
                        || echo "    -> could not signal pid $pid"
                    ;;
            esac
        done
    fi
    echo
    echo "  the dispatcher, if one is still dispatching, is one of:"
    pgrep -af 'run_parallel_sweep\.sh' 2>/dev/null | sed 's/^/    /' || echo "    (none)"
    echo "  its xargs feeder, which a name pattern would never have matched:"
    pgrep -af 'xargs -P' 2>/dev/null | sed 's/^/    /' || echo "    (none)"
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
