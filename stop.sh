#!/usr/bin/env bash
# Stop the sweep completely: the loop first, then the runners.
#
# Why this exists: killing main.py is not enough.  run_all_models.sh is a bash loop that walks
# the roster and starts the next model as soon as the previous child exits, so killing only the
# runner makes the loop launch the following model -- which is exactly how doubao started after
# a command that was supposed to stop everything.  The loop has to die first.
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

ours() { pgrep -af 'run_all_models.sh|main.py --model' 2>/dev/null; }

echo "=== before ==="
ours || echo "  (nothing of ours is running)"
echo

if [ -n "$DRY" ]; then
    echo "--dry-run: nothing was stopped"
    exit 0
fi

# 1. The loop, so it cannot start the next model while we work.
pkill -f 'run_all_models.sh' 2>/dev/null && echo "stopped the roster loop" || echo "no roster loop found"
sleep 2

# 2. The runners themselves.
pkill -f 'main.py --model' 2>/dev/null && echo "stopped the main.py runner(s)" || echo "no runner found"
sleep 6

# 3. Report, and list any surviving Isaac process with its command line, ours or not, so a
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
