"""How narrow can the opening get before no torso angle fits, and what to add next.

The question this answers, from the supervisor: "work out the width at which the robot
cannot get through however it turns".  The answer is not the shoulder width and not the
A/S threshold already measured -- it is the body's SMALLEST projected width over all
orientations, which is its thickness, because the projection of a 0.570 x 0.220 rectangle
across an opening falls from 0.570 at 0 degrees to 0.220 at 90 degrees.

Everything below is computed from the same functions the experiment uses
(``memory_metrics.needed_width`` / ``min_turns`` / ``optimal_steps``), so a change to the
body model changes this table too.

Run from the repository root:  python tools/ladder_proposal.py
"""

import sys

sys.path.insert(0, ".")
import memory_metrics as mm  # noqa: E402

SHOULDER = mm.SHOULDER_WIDTH_M
THICKNESS = mm.TORSO_THICKNESS_M
TURN = mm.TURN_STEP_DEG

EXISTING = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9]
PROPOSED = EXISTING + [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.38,
                       0.35, 0.3]

# Measured, not assumed: the v7 sweep made 10,123 model calls for 660 episodes, and the
# gateway meter read 17,813 calls for $42.07 across v7 and v5 (tools/usage_delta.py).
CALLS_PER_EPISODE = 10123 / 660
USD_PER_CALL = 42.07 / 17813
USD_TO_RMB = 7.2
SECONDS_PER_CALL = 5.0


def impassable_width() -> float:
    """Narrowest the body ever gets: the projection at 90 degrees, the thickness."""
    return min(mm.needed_width(deg) for deg in range(0, 181))


def main() -> int:
    floor = impassable_width()
    print("=" * 78)
    print("1. THE WIDTH THAT CANNOT BE PASSED AT ANY ORIENTATION")
    print("=" * 78)
    print(f"   body: {SHOULDER:.3f} m across the shoulders, {THICKNESS:.3f} m thick")
    print(f"   projection at  0 deg: {mm.needed_width(0):.3f} m   (shoulders first)")
    print(f"   projection at 90 deg: {mm.needed_width(90):.3f} m   (thickness first)")
    print(f"   smallest over ALL angles: {floor:.3f} m at 90 deg")
    print()
    print(f"   IMPASSABLE AT EVERY TORSO ANGLE:  W < {floor:.3f} m "
          f"(A/S < {floor / SHOULDER:.3f})")
    print()
    print("   Rotation is not the limit: swinging the torso needs a swept circle of")
    print(f"   {2 * 0.5 * (SHOULDER ** 2 + THICKNESS ** 2) ** 0.5:.3f} m, but the room in front of the wall")
    print("   is 5 m across, so the body can always turn to 90 degrees BEFORE the")
    print(f"   opening and walk in sideways.  The binding floor is therefore {floor:.3f} m,")
    print("   and it equals the torso thickness because that is the only dimension left.")
    print("   The start pose is already centred on the opening (z = 0), so no lateral")
    print("   step is needed; the lateral step is 0.75 m, far coarser than the")
    print("   sub-centimetre alignment a tight opening would demand, so a model that")
    print("   strafes near a narrow opening cannot recover the line.")

    print()
    print("=" * 78)
    print("2. REQUIRED ROTATION, BY CHANNEL WIDTH (15 degree steps)")
    print("=" * 78)
    print(f"   {'channel width':>18}  {'turns':>5}  {'angle':>6}  {'optimal steps':>13}")
    # min_turns(W) = k exactly when every earlier pose fails and pose k fits, so the
    # interval for k is [needed(k), min over j < k of needed(j)).  Printing the plateaus
    # in the other order labelled the 0.220-0.360 band as 5 turns when it needs 6 -- the
    # band belongs to the angle that fits, not to the one below it.
    earlier_best = float("inf")
    for turns in range(0, 7):
        required = mm.needed_width(turns * TURN)
        if turns == 0:
            print(f"   {'W >= %.3f' % required:>18}  {0:5d}  {0:5.0f}d  "
                  f"{mm.optimal_steps(required):13d}")
        elif earlier_best > required:
            print(f"   {'%.3f <= W < %.3f' % (required, earlier_best):>18}  {turns:5d}  "
                  f"{turns * TURN:5.0f}d  {11 + turns:13d}")
        else:
            print(f"   {'(no width)':>18}  {turns:5d}  {turns * TURN:5.0f}d  "
                  f"{'--':>13}   never the fewest")
        earlier_best = min(earlier_best, required)
    print(f"   {'W < %.3f' % floor:>18}  {'--':>5}  {'--':>6}  {'--':>13}   "
          "never fits at any angle")
    print()
    print(f"   1 and 2 turns are never the fewest: the projection rises to "
          f"{max(mm.needed_width(d) for d in range(0, 91)):.3f} m at")
    print("   21.1 deg before it falls, so a small rotation makes the body wider, not")
    print("   narrower.  45, 60, 75 and 90 degrees are the only useful angles, so the")
    print("   whole below-1.0 ladder is four plateaus and not a continuum.")

    print()
    print("=" * 78)
    print("3. PROPOSED LADDER (existing 12 levels + 13)")
    print("=" * 78)
    print(f"   {'A/S':>5}  {'width':>7}  {'turns':>5}  {'angle':>6}  {'optimal':>7}  verdict")
    for ratio in PROPOSED:
        width = round(SHOULDER * ratio, 6)
        try:
            turns = mm.min_turns(width)
        except ValueError:
            print(f"   {ratio:5.2f}  {width:7.3f}  {'--':>5}  {'--':>6}  {'--':>7}  "
                  "IMPOSSIBLE at any angle")
            continue
        note = ""
        if width - floor < 0.02:
            note = "  <- within 2 cm of the floor: no margin"
        print(f"   {ratio:5.2f}  {width:7.3f}  {turns:5d}  {turns * TURN:5.0f}d  "
              f"{mm.optimal_steps(width):7d}{note}")

    print()
    print("=" * 78)
    print("4. WHAT 10,000 EPISODES WOULD TAKE")
    print("=" * 78)
    for label, episodes in (("Stage 1 today", 660), ("one scene x 25 levels x 9", 2475),
                            ("four scenes x 25 x 9", 9900)):
        calls = episodes * CALLS_PER_EPISODE
        usd = calls * USD_PER_CALL
        print(f"   {label:26} {episodes:6d} episodes  {calls:9.0f} calls  "
              f"${usd:7.2f}  RMB {usd * USD_TO_RMB:8.0f}  ~{calls * SECONDS_PER_CALL / 86400:4.1f} days serial")
    print()
    print(f"   basis: {CALLS_PER_EPISODE:.2f} calls/episode and ${USD_PER_CALL:.6f}/call,")
    print("   both measured from the committed sweeps, not estimated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
