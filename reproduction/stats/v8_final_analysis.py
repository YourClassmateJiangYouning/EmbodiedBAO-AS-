"""Final v8 analysis: all quantities needed for the paper's results section.

Inputs (read-only): 数据图片与教程/最终数据/*.csv
Outputs: analysis/v8_final/*.csv, findings.json, report_tables.md

Stages
  S1  6,375 episodes, 17 A/S levels, 15 models
  S2  105 paired memory-transfer tests, 7 models
  S3  100 body-size episodes, 5 models x {G1, R1} x 4 A/S values

Methods implemented here
  * required torso angle per (body, A/S): smallest theta with
      S*cos(theta) + D*sin(theta) <= A/S*S          (exact, 0.01 deg grid)
  * pi-hat: A/S at which P(max_rotation >= 30 deg) crosses 0.5, by linear
    interpolation between the two adjacent rungs of the ladder; percentile
    bootstrap CI resampling EPISODES WITHIN LEVEL; a logistic fit is reported
    as a sensitivity analysis (same 50% crossing, 17-point fit)
  * pass-rate crossing at 0.5 by the same interpolation
  * necessity: max_rotation >= required angle -> pass, and the converse
  * timing: early = first turn at x <= 5.0 m, late = x > 5.0 m
  * wide-turn cost, per-scene / per-marker robustness, model profiles
  * S2 paired deltas vs the matched no-memory benchmark
  * S3 per-body results + body-invariance comparisons + robot-specific
    required angles at A/S 0.9
  * reasoning-text features from S1_steps (opening / body / turn / ratio /
    cannot-fit, anticipatory mention before first blocked step)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL = os.path.join(ROOT, "数据图片与教程", "最终数据")
OUT = os.path.join(HERE, "v8_final")
os.makedirs(OUT, exist_ok=True)

HUMAN = {"franchak2012": 1.10, "keizer2013_hc": 1.25, "warren1987": 1.30}
BODY = {  # key: (shoulder width S, torso thickness D)
    "H1": (0.570, 0.220),
    "G1": (0.450, 0.200),
    "R1": (0.357, 0.190),
}
NARROW_LEVELS = list(range(11, 17))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def required_angle(a_s: float, body: str = "H1") -> float:
    """Smallest torso yaw at which the body fits an A/S aperture (degrees)."""
    S, D = BODY[body]
    ratio = float(a_s)
    if ratio >= 1.0 - 1e-9:
        return 0.0
    best = None
    for theta_deg in np.arange(0.0, 90.0 + 1e-9, 0.01):
        th = math.radians(theta_deg)
        if math.cos(th) + (D / S) * math.sin(th) <= ratio + 1e-12:
            best = float(theta_deg)
            break
    return 90.0 if best is None else best


def interp_crossing(curve: pd.DataFrame, value_col: str, target: float = 0.5):
    """Linear interpolation of the A/S at which a monotone curve crosses `target`.

    `curve` must be sorted by descending A/S.
    """
    a = curve["a_s_ratio"].to_numpy(float)
    v = curve[value_col].to_numpy(float)
    for i in range(1, len(v)):
        lo, hi = v[i - 1], v[i]
        if (hi - target) * (lo - target) <= 0 and abs(hi - lo) > 1e-12:
            frac = (target - lo) / (hi - lo)
            return float(a[i - 1] + frac * (a[i] - a[i - 1]))
    return None


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 80):
    y = np.asarray(y, float)
    if y.min() == y.max():
        return None
    X = np.column_stack([np.ones_like(x, dtype=float), np.asarray(x, float)])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        z = eta + (y - p) / w
        A = X.T @ (X * w[:, None]) + ridge * np.eye(2)
        try:
            new = np.linalg.solve(A, X.T @ (w * z))
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    if not np.all(np.isfinite(beta)) or abs(beta[1]) < 1e-12:
        return None
    return float(beta[0]), float(beta[1])


def x_at_p(fit, p: float):
    if fit is None:
        return None
    a, b = fit
    return (math.log(p / (1 - p)) - a) / b


def pi_hat_bootstrap(sub: pd.DataFrame, boot: int = 2000, seed: int = 20261107):
    """pi-hat for one model (or pooled): point + percentile bootstrap CI."""
    levels = sorted(sub["a_s_ratio"].unique(), reverse=True)
    curve = (sub.groupby("level")
                .agg(a_s_ratio=("a_s_ratio", "first"),
                     n=("rot30", "size"), rot30=("rot30", "mean"),
                     passed=("passed", "mean"), max_rot=("max_rotation_deg", "mean"))
                .reset_index().sort_values("a_s_ratio", ascending=False))
    pi_rot = interp_crossing(curve, "rot30")
    pi_pass = interp_crossing(curve, "passed")
    fit = fit_logistic(sub["a_s_ratio"].to_numpy(float), sub["rot30"].to_numpy(float))
    pi_rot_logit = x_at_p(fit, 0.5)
    rng = np.random.default_rng(seed)
    parts = [(lv, g[["rot30", "passed"]].to_numpy(float))
             for lv, g in sub.groupby("level")]
    boot_rot, boot_pass, missing = [], [], 0
    for _ in range(boot):
        rows = []
        for lv, arr in parts:
            idx = rng.integers(0, len(arr), len(arr))
            s = arr[idx]
            rows.append((lv, float(s[:, 0].mean()), float(s[:, 1].mean())))
        bcurve = pd.DataFrame(rows, columns=["level", "rot30", "passed"])
        bcurve["a_s_ratio"] = bcurve["level"].map(
            dict(zip(curve["level"], curve["a_s_ratio"])))
        bcurve = bcurve.sort_values("a_s_ratio", ascending=False)
        r = interp_crossing(bcurve, "rot30")
        p = interp_crossing(bcurve, "passed")
        if r is None:
            missing += 1
        else:
            boot_rot.append(r)
        if p is not None:
            boot_pass.append(p)
    out = {
        "pi_rot30": pi_rot, "pi_pass50": pi_pass, "pi_rot30_logit": pi_rot_logit,
        "n": int(len(sub)),
        "pi_rot30_lo": float(np.percentile(boot_rot, 2.5)) if boot_rot else None,
        "pi_rot30_hi": float(np.percentile(boot_rot, 97.5)) if boot_rot else None,
        "pi_rot30_boot_missing": missing,
        "pi_pass50_lo": float(np.percentile(boot_pass, 2.5)) if boot_pass else None,
        "pi_pass50_hi": float(np.percentile(boot_pass, 97.5)) if boot_pass else None,
    }
    return out, curve


def source_turn(seq: str) -> int:
    return sum(1 for a in str(seq).split(",") if a.strip() in ("turn_left", "turn_right"))


# ---------------------------------------------------------------------------
def main() -> int:
    findings = {}
    tables = {}

    S1 = pd.read_csv(os.path.join(FINAL, "S1_episodes.csv"))
    S1["rot30"] = (S1["max_rotation_deg"] >= 30 - 1e-9).astype(int)
    S1["rot15"] = (S1["max_rotation_deg"] >= 15 - 1e-9).astype(int)
    S1["rot45"] = (S1["max_rotation_deg"] >= 45 - 1e-9).astype(int)
    S1["turned"] = (S1["n_turn_left"] + S1["n_turn_right"] > 0).astype(int)
    S1["req_deg"] = [required_angle(a, "H1") for a in S1["a_s_ratio"]]
    S1["met_req"] = (S1["max_rotation_deg"] >= S1["req_deg"] - 1e-9).astype(int)
    S1["narrow"] = S1["level"].isin(NARROW_LEVELS)
    S1["wide_easy"] = S1["a_s_ratio"] >= 1.5
    S1["early_turn"] = np.where(S1["first_turn_pos_m"].notna(),
                                (S1["first_turn_pos_m"] <= 5.0).astype(float), np.nan)
    models = sorted(S1["model"].unique())
    print(f"[S1] {len(S1)} episodes, {len(models)} models, {S1['level'].nunique()} levels")

    # ---- corpus-level per-level table ---------------------------------------
    lvl = []
    for lv, g in S1.groupby("level"):
        passed = int(g["passed"].sum())
        lo, hi = wilson(passed, len(g))
        lvl.append({
            "level": int(lv), "a_s_ratio": float(g["a_s_ratio"].iloc[0]),
            "channel_width_m": float(g["channel_width_m"].iloc[0]),
            "n": len(g), "passed": passed, "pass_pct": 100 * passed / len(g),
            "pass_lo": 100 * lo, "pass_hi": 100 * hi,
            "turned_pct": 100 * g["turned"].mean(),
            "rot30_pct": 100 * g["rot30"].mean(),
            "sideways_pct": 100 * g["passed_sideways"].mean(),
            "mean_max_rot": float(g["max_rotation_deg"].mean()),
            "mean_turns": float(g["n_turn_left"].mean() + g["n_turn_right"].mean()),
            "mean_steps": float(g["total_steps"].mean()),
            "look_down_pct": 100 * (g["n_look_down"] > 0).mean(),
            "wall_collisions_mean": float(g["wall_collisions"].mean()),
            "req_deg": float(g["req_deg"].iloc[0]),
            "met_req_pct": 100 * g["met_req"].mean(),
            "within_15deg_met_pct": 100 * (g["max_rotation_deg"] >= g["req_deg"] - 15).mean(),
        })
    level_tab = pd.DataFrame(lvl)
    tables["s1_level_pooled"] = level_tab
    level_tab.to_csv(os.path.join(OUT, "s1_level_pooled.csv"), index=False)

    # ---- per model x level --------------------------------------------------
    ml = (S1.groupby(["model", "level"])
            .agg(a_s_ratio=("a_s_ratio", "first"), n=("passed", "size"),
                 passed=("passed", "sum"), pass_pct=("passed", "mean"),
                 turned=("turned", "mean"), rot30=("rot30", "mean"),
                 sideways=("passed_sideways", "mean"),
                 max_rot=("max_rotation_deg", "mean"),
                 steps=("total_steps", "mean"),
                 first_turn_pos=("first_turn_pos_m", "median"),
                 look_down=("n_look_down", "mean"),
                 met_req=("met_req", "mean"))
            .reset_index())
    ml["pass_pct"] = 100 * ml["pass_pct"]
    ml["turned"] = 100 * ml["turned"]
    ml["rot30"] = 100 * ml["rot30"]
    ml["sideways"] = 100 * ml["sideways"]
    ml["met_req"] = 100 * ml["met_req"]
    ml.to_csv(os.path.join(OUT, "s1_model_level.csv"), index=False)
    tables["s1_model_level"] = ml

    # ---- thresholds ---------------------------------------------------------
    pi_rows = []
    pooled, pooled_curve = pi_hat_bootstrap(S1, boot=2000)
    pooled["model"] = "ALL (pooled)"
    pi_rows.append(pooled)
    curves = {"ALL": pooled_curve}
    for m in models:
        sub = S1[S1["model"] == m]
        res, curve = pi_hat_bootstrap(sub, boot=1500)
        res["model"] = m
        pi_rows.append(res)
        curves[m] = curve
    pi_tab = pd.DataFrame(pi_rows)
    for human, val in HUMAN.items():
        pi_tab[f"gap_vs_{human}_pp"] = 100 * (pi_tab["pi_rot30"] - val) / val
        pi_tab[f"gap_vs_{human}_abs"] = pi_tab["pi_rot30"] - val
    pi_tab.to_csv(os.path.join(OUT, "pi_hat.csv"), index=False)
    tables["pi_hat"] = pi_tab
    findings["pi_hat"] = {
        "pooled": {k: v for k, v in pooled.items()},
        "human": HUMAN,
        "pooled_gap_vs_human_pct": {k: 100 * (pooled["pi_rot30"] - v) / v for k, v in HUMAN.items()},
    }
    # rot15 / rot45 crossings of the pooled curve (sensitivity to the angle criterion)
    alt = {}
    for col in ("rot15", "rot30", "rot45"):
        curve = (S1.groupby("level").agg(a_s_ratio=("a_s_ratio", "first"),
                                         v=(col, "mean")).reset_index()
                 .sort_values("a_s_ratio", ascending=False))
        alt[col] = interp_crossing(curve.rename(columns={"v": col}), col)
    findings["pi_hat_angle_sensitivity"] = alt

    # ---- necessity: met required angle -> pass ------------------------------
    narrow = S1[S1["narrow"]].copy()
    nec = (narrow.groupby("met_req")
           .agg(n=("passed", "size"), passed=("passed", "sum"),
                pass_pct=("passed", "mean")).reset_index())
    nec["pass_pct"] *= 100
    tables["s1_necessity"] = nec
    findings["necessity"] = {
        "narrow_n": int(len(narrow)),
        "met_n": int(narrow["met_req"].sum()),
        "met_pass": int(narrow.loc[narrow["met_req"] == 1, "passed"].sum()),
        "met_pass_pct": float(100 * narrow.loc[narrow["met_req"] == 1, "passed"].mean()),
        "notmet_n": int((narrow["met_req"] == 0).sum()),
        "notmet_pass": int(narrow.loc[narrow["met_req"] == 0, "passed"].sum()),
        "notmet_pass_pct": float(100 * narrow.loc[narrow["met_req"] == 0, "passed"].mean()),
    }
    # A/S 0.4 failure rotation distribution
    g04 = S1[S1["level"] == 16]
    fail04 = g04[g04["passed"] == 0]
    bins = [(-0.01, 0.01, "0"), (0.01, 30, "0-30"), (30, 45, "30-45"),
            (45, 60, "45-60"), (60, 90.1, "60-90")]
    dist = {}
    for lo, hi, name in bins:
        dist[name] = int(((fail04["max_rotation_deg"] > lo) &
                          (fail04["max_rotation_deg"] <= hi)).sum())
    findings["narrowest_failure_rotation"] = {
        "level16_n": int(len(g04)), "failed": int(len(fail04)),
        "turned_ge4_rate_in_fail": float((fail04["n_turn_left"] + fail04["n_turn_right"] >= 4).mean()),
        "rotation_bins": dist,
        "required_deg": float(g04["req_deg"].iloc[0]),
    }
    # per-level met rate vs pass rate (for the mechanism figure)
    mech = (narrow.groupby("level")
            .agg(a_s_ratio=("a_s_ratio", "first"), n=("passed", "size"),
                 met_pct=("met_req", "mean"), pass_pct=("passed", "mean"),
                 mean_rot=("max_rotation_deg", "mean"), req_deg=("req_deg", "first"))
            .reset_index())
    for c in ("met_pct", "pass_pct"):
        mech[c] = 100 * mech[c]
    tables["s1_mechanism"] = mech
    mech.to_csv(os.path.join(OUT, "s1_mechanism_by_level.csv"), index=False)

    # ---- timing: early vs late x met ---------------------------------------
    t = narrow.dropna(subset=["first_turn_pos_m"]).copy()
    t["timing"] = np.where(t["first_turn_pos_m"] <= 5.0, "early(<=5m)", "late(>5m)")
    timing = (t.groupby(["timing", "met_req"])
              .agg(n=("passed", "size"), passed=("passed", "sum"),
                   pass_pct=("passed", "mean"),
                   steps=("total_steps", "mean"),
                   collisions=("wall_collisions", "mean"),
                   turns_mean=("n_turn_left", lambda s: float("nan")))
              .reset_index())
    timing["pass_pct"] *= 100
    # add the no-turn group
    no_turn = narrow[narrow["first_turn_pos_m"].isna()]
    no_turn_row = pd.DataFrame([{
        "timing": "never", "met_req": 0, "n": len(no_turn),
        "passed": int(no_turn["passed"].sum()),
        "pass_pct": 100 * no_turn["passed"].mean(),
        "steps": float(no_turn["total_steps"].mean()),
        "collisions": float(no_turn["wall_collisions"].mean()),
        "turns_mean": 0.0,
    }])
    timing = pd.concat([timing, no_turn_row], ignore_index=True)
    timing["n_turns"] = timing["turns_mean"]
    tables["s1_timing"] = timing
    timing.to_csv(os.path.join(OUT, "s1_timing.csv"), index=False)
    e_met = timing[(timing["timing"] == "early(<=5m)") & (timing["met_req"] == 1)].iloc[0]
    l_met = timing[(timing["timing"] == "late(>5m)") & (timing["met_req"] == 1)].iloc[0]
    findings["timing"] = {
        "early_met_n": int(e_met["n"]), "early_met_pass_pct": float(e_met["pass_pct"]),
        "late_met_n": int(l_met["n"]), "late_met_pass_pct": float(l_met["pass_pct"]),
        "late_met_fail_n": int(l_met["n"] - l_met["passed"]),
    }
    late_fail = t[(t["first_turn_pos_m"] > 5.0) & (t["met_req"] == 1) & (t["passed"] == 0)]
    early_pass = t[(t["first_turn_pos_m"] <= 5.0) & (t["met_req"] == 1) & (t["passed"] == 1)]
    findings["timing_late_fail"] = {
        "n": int(len(late_fail)),
        "all_max_steps": bool((late_fail["end_reason"] == "max_steps").all()),
        "mean_collisions": float(late_fail["wall_collisions"].mean()),
        "mean_turns": float((late_fail["n_turn_left"] + late_fail["n_turn_right"]).mean()),
        "mean_total_rotation": float(late_fail["total_rotation_deg"].mean()),
        "mean_steps": float(late_fail["total_steps"].mean()),
        "earlypass_mean_steps": float(early_pass["total_steps"].mean()),
        "earlypass_mean_collisions": float(early_pass["wall_collisions"].mean()),
    }
    # first-turn position buckets
    ft = narrow.dropna(subset=["first_turn_pos_m"])
    buckets = {
        "<=3m": int((ft["first_turn_pos_m"] <= 3).sum()),
        "3-5m": int(((ft["first_turn_pos_m"] > 3) & (ft["first_turn_pos_m"] <= 5)).sum()),
        "5-7m": int(((ft["first_turn_pos_m"] > 5) & (ft["first_turn_pos_m"] < 7)).sum()),
        ">=7m": int((ft["first_turn_pos_m"] >= 7).sum()),
        "never": int(narrow["first_turn_pos_m"].isna().sum()),
    }
    findings["first_turn_position"] = {
        "buckets": buckets, "n_with_turn": int(len(ft)),
        "median_m": float(ft["first_turn_pos_m"].median()),
        "mean_m": float(ft["first_turn_pos_m"].mean()),
        "share_ge7": float((ft["first_turn_pos_m"] >= 7).mean()),
        "share_le3": float((ft["first_turn_pos_m"] <= 3).mean()),
    }

    # ---- wide-turn cost -----------------------------------------------------
    width_rows = []
    for name, mask in (("L0-L5 (A/S 2.0-1.5)", S1["a_s_ratio"] >= 1.5),
                       ("L0-L10 (A/S 2.0-1.0)", S1["a_s_ratio"] >= 1.0 - 1e-9)):
        g = S1[mask]
        turned, still = g[g["turned"] == 1], g[g["turned"] == 0]
        width_rows.append({
            "group": name, "n": len(g),
            "turned_n": len(turned), "turned_pct": 100 * g["turned"].mean(),
            "turned_pass_pct": 100 * turned["passed"].mean() if len(turned) else None,
            "still_pass_pct": 100 * still["passed"].mean() if len(still) else None,
            "turned_steps": float(turned["total_steps"].mean()) if len(turned) else None,
            "still_steps": float(still["total_steps"].mean()) if len(still) else None,
        })
    wide_tab = pd.DataFrame(width_rows)
    tables["s1_wide_turn_cost"] = wide_tab
    wide_tab.to_csv(os.path.join(OUT, "s1_wide_turn_cost.csv"), index=False)
    findings["wide_turn_cost"] = width_rows

    # ---- per-model profile --------------------------------------------------
    prof = []
    for m in models:
        g = S1[S1["model"] == m]
        wide = g[g["a_s_ratio"] >= 1.0 - 1e-9]
        nar = g[g["narrow"]]
        ft_m = g.dropna(subset=["first_turn_pos_m"])
        prof.append({
            "model": m,
            "n": len(g),
            "pass_pct": 100 * g["passed"].mean(),
            "wide_pass_pct": 100 * wide["passed"].mean(),
            "narrow_pass_pct": 100 * nar["passed"].mean(),
            "wide_rot_deg": float(wide["max_rotation_deg"].mean()),
            "narrow_rot_deg": float(nar["max_rotation_deg"].mean()),
            "adaptivity_deg": float(nar["max_rotation_deg"].mean() - wide["max_rotation_deg"].mean()),
            "first_turn_median_m": float(ft_m["first_turn_pos_m"].median()) if len(ft_m) else np.nan,
            "look_down_step_pct": 100 * g["n_look_down"].sum() / max(1, g["total_steps"].sum()),
            "look_down_ep_pct": 100 * (g["n_look_down"] > 0).mean(),
            "mean_steps": float(g["total_steps"].mean()),
            "invalid_steps": int(g["invalid_responses"].sum()),
            "collision_ep_pct": 100 * (g["wall_collisions"] > 0).mean(),
            "early_turn_pct_in_turns": 100 * (ft_m["first_turn_pos_m"] <= 5.0).mean() if len(ft_m) else np.nan,
            "narrow_turn_pct": 100 * nar["turned"].mean(),
            "narrow_met_req_pct": 100 * nar["met_req"].mean(),
        })
    prof_tab = pd.DataFrame(prof).sort_values("pass_pct", ascending=False)
    prof_tab.to_csv(os.path.join(OUT, "s1_model_profile.csv"), index=False)
    tables["s1_model_profile"] = prof_tab

    # ---- correlation: wide rotation vs narrow pass --------------------------
    x = prof_tab["wide_rot_deg"].to_numpy(float)
    y = prof_tab["narrow_pass_pct"].to_numpy(float)
    r_all = float(np.corrcoef(x, y)[0, 1])
    keep = ~prof_tab["model"].isin(["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview"])
    r_without = float(np.corrcoef(x[keep], y[keep])[0, 1])
    findings["strategy_correlation"] = {
        "r_all15": r_all, "r_without_top2": r_without,
        "top2_models": ["deepseek-v4-flash-vision-exp", "gemini-3-flash-preview"],
    }

    # ---- per scene / per marker robustness ----------------------------------
    scene_tab = (S1.groupby(["model", "scene"])
                 .agg(pass_pct=("passed", "mean"), narrow_pass=("narrow", "mean"),
                      wide_pass=("wide_easy", "mean")).reset_index())
    scene_tab["pass_pct"] *= 100
    tables["s1_model_scene"] = scene_tab
    scene_wide = (S1.groupby("scene")
                  .agg(n=("passed", "size"), pass_pct=("passed", "mean"),
                       narrow_n=("narrow", "sum"),
                       narrow_pass=("passed", lambda s: np.nan))
                  .reset_index())
    scene_rows = []
    for sc, g in S1.groupby("scene"):
        scene_rows.append({
            "scene": sc, "n": len(g), "pass_pct": 100 * g["passed"].mean(),
            "narrow_n": int(g["narrow"].sum()),
            "narrow_pass_pct": 100 * g.loc[g["narrow"], "passed"].mean(),
            "wide_pass_pct": 100 * g.loc[~g["narrow"], "passed"].mean(),
        })
    scene_tab2 = pd.DataFrame(scene_rows)
    tables["s1_scene"] = scene_tab2
    scene_tab2.to_csv(os.path.join(OUT, "s1_scene_robustness.csv"), index=False)
    marker_rows = []
    for mk, g in S1.groupby("marker_slot"):
        marker_rows.append({
            "marker_slot": int(mk), "n": len(g), "pass_pct": 100 * g["passed"].mean(),
            "narrow_pass_pct": 100 * g.loc[g["narrow"], "passed"].mean(),
            "wide_pass_pct": 100 * g.loc[~g["narrow"], "passed"].mean(),
        })
    marker_tab = pd.DataFrame(marker_rows).sort_values("marker_slot")
    tables["s1_marker"] = marker_tab
    marker_tab.to_csv(os.path.join(OUT, "s1_marker_robustness.csv"), index=False)
    findings["robustness"] = {
        "scene_narrow_pass_range_pct": [float(scene_tab2["narrow_pass_pct"].min()),
                                         float(scene_tab2["narrow_pass_pct"].max())],
        "marker_pass_range_pct": [float(marker_tab["pass_pct"].min()),
                                   float(marker_tab["pass_pct"].max())],
        "model_scene_pass_sd_pct": float(
            scene_tab.groupby("model")["pass_pct"].std().mean()),
    }

    # ---- reasoning-text features (S1 steps) --------------------------------
    steps = pd.read_csv(os.path.join(FINAL, "S1_steps.csv"),
                        usecols=["model", "scene", "level", "episode_id", "step", "action",
                                 "position_x", "collision", "reasoning"])
    steps["reasoning"] = steps["reasoning"].fillna("")
    import re
    PAT = {
        "opening": r"\b(opening|gap|aperture|doorway|passage|slit|channel)\b",
        "body": r"\b(shoulder|shoulders|my body|torso|body width|my width|body depth|profile)\b",
        "turn": r"\b(turn|turning|rotate|rotating|rotation|sideways|yaw)\b",
        "ratio": r"(wider than my|narrower than my|wider than the|relative to my|times my|"
                 r"as wide as my|as wide as the|shoulders? span|body depth|"
                 r"one grid (square|cell)|grid square|fraction of my|compared to my)",
        "cannot_fit": r"\b(cannot|can'?t|unable to|will not|won'?t) (fit|pass|get through|go through)\b",
        "narrow_word": r"\b(narrow|tight|too small|won'?t fit|doesn'?t fit|barely)\b",
    }
    comp = {k: re.compile(v, re.I) for k, v in PAT.items()}
    feat = []
    for (mdl, sc, lv, eid), g in steps.groupby(["model", "scene", "level", "episode_id"]):
        text = " ".join(g["reasoning"].astype(str))
        # first blocked step (translation collision)
        blocked = g[(g["collision"] == 1) & (~g["action"].isin(["turn_left", "turn_right"]))]
        first_block = int(blocked["step"].min()) if len(blocked) else None
        pre = g if first_block is None else g[g["step"] < first_block]
        pre_text = " ".join(pre["reasoning"].astype(str)).lower()
        row = {"model": mdl, "scene": sc, "level": int(lv), "episode_id": int(eid),
               "first_block_step": first_block,
               "pre_steps": int(len(pre))}
        for k, cre in comp.items():
            row[f"any_{k}"] = int(bool(cre.search(text)))
            row[f"pre_{k}"] = int(bool(cre.search(pre_text)))
        feat.append(row)
    fdf = pd.DataFrame(feat)
    fdf = fdf.merge(S1[["model", "scene", "level", "episode_id", "passed", "narrow",
                        "met_req", "turned", "max_rotation_deg", "req_deg"]],
                    on=["model", "scene", "level", "episode_id"], how="left")
    fdf.to_csv(os.path.join(OUT, "s1_reasoning_features.csv"), index=False)
    fr = {}
    for band, mask in (("wide", fdf["level"] <= 10), ("narrow", fdf["narrow"])):
        g = fdf[mask]
        fr[band] = {f"any_{k}_pct": float(100 * g[f"any_{k}"].mean()) for k in PAT}
        fr[band]["n"] = int(len(g))
    fr["narrow_mention_turn_but_fail"] = int(((fdf["narrow"]) & (fdf["any_turn"] == 1)
                                              & (fdf["passed"] == 0)).sum())
    fr["narrow_mention_turn"] = int(((fdf["narrow"]) & (fdf["any_turn"] == 1)).sum())
    fr["narrow_mention_body"] = int(((fdf["narrow"]) & (fdf["any_body"] == 1)).sum())
    fr["narrow_ratio"] = int(((fdf["narrow"]) & (fdf["any_ratio"] == 1)).sum())
    fr["narrow_n"] = int(fdf["narrow"].sum())
    fr["narrow_talk_turn_but_underrotate"] = int(
        ((fdf["narrow"]) & (fdf["any_turn"] == 1) & (fdf["met_req"] == 0)).sum())
    fr["pre_talk_turn_pct_in_turning_narrow"] = float(
        100 * fdf[(fdf["narrow"]) & (fdf["turned"] == 1)]["pre_turn"].mean())
    fr["pre_talk_turn_pct_in_narrow"] = float(100 * fdf[fdf["narrow"]]["pre_turn"].mean())
    findings["reasoning"] = fr

    # ---- data quality -------------------------------------------------------
    dq = {
        "s1_episodes": int(len(S1)),
        "s1_steps": int(len(steps)),
        "s1_invalid_steps": int(S1["invalid_responses"].sum()),
        "s1_invalid_episodes": int((S1["invalid_responses"] > 0).sum()),
        "s1_collision_episodes": int((S1["wall_collisions"] > 0).sum()),
        "s1_no_forward_episodes": int((S1["n_forward"] == 0).sum()),
        "s1_mean_steps": float(S1["total_steps"].mean()),
        "s1_total_llm_hours": float(S1["total_llm_time_ms"].sum() / 3.6e6),
    }
    tables["s1_data_quality"] = pd.DataFrame([dq])
    findings["data_quality"] = dq

    # =====================================================================
    # S2
    # =====================================================================
    S2 = pd.read_csv(os.path.join(FINAL, "S2_pairs.csv"))
    S2["digest_valid"] = S2["digest_valid"].astype(str).str.lower().isin(["true", "1"])
    S2["src_turns"] = S2["source_actions"].map(source_turn)
    s2_n = len(S2)
    s2_pass = int(S2["passed"].sum())
    s2_base = int((S2["baseline_passed"].astype(str).str.lower().isin(["true", "1"])).sum())
    deltas = S2["turn_count"] - S2["baseline_turns"]
    up = int(((S2["turn_count"] > S2["baseline_turns"])).sum())
    down = int(((S2["turn_count"] < S2["baseline_turns"])).sum())
    sign_p = float(sum(math.comb(up + down, i) for i in range(up, up + down + 1)) / 2 ** (up + down)) if (up + down) else None
    s2_rows = []
    for m, g in S2.groupby("model"):
        d = g["turn_count"] - g["baseline_turns"]
        rng = np.random.default_rng(7)
        boot = [float(np.mean(d.to_numpy()[rng.integers(0, len(d), len(d))])) for _ in range(3000)]
        s2_rows.append({
            "model": m, "n": len(g), "passed": int(g["passed"].sum()),
            "pass_pct": 100 * g["passed"].mean(),
            "baseline_pass_pct": 100 * g["baseline_passed"].astype(str).str.lower().isin(["true", "1"]).mean(),
            "turns_mean": float(g["turn_count"].mean()),
            "baseline_turns_mean": float(g["baseline_turns"].mean()),
            "delta_turns_mean": float(d.mean()),
            "delta_turns_lo": float(np.percentile(boot, 2.5)),
            "delta_turns_hi": float(np.percentile(boot, 97.5)),
            "delta_turns_median": float(d.median()),
            "lcs_mean": float(g["copy_lcs_fraction"].mean()),
            "lcs_median": float(g["copy_lcs_fraction"].median()),
            "max_rot_mean": float(g["max_rotation_deg"].mean()),
            "baseline_rot_mean": float(g["baseline_max_rotation_deg"].mean()),
            "steps_mean": float(g["total_steps"].mean()),
            "baseline_steps_mean": float(g["baseline_steps"].mean()),
            "look_down_mean": float(g["look_down_count"].mean()),
            "digest_valid_pct": 100 * g["digest_valid"].mean(),
            "timeout_lines": int(g.get("timeout_lines", pd.Series([0])).sum()),
        })
    s2_model = pd.DataFrame(s2_rows).sort_values("delta_turns_mean", ascending=False)
    tables["s2_model"] = s2_model
    s2_model.to_csv(os.path.join(OUT, "s2_model.csv"), index=False)
    S2.to_csv(os.path.join(OUT, "s2_pairs_enriched.csv"), index=False)
    findings["s2"] = {
        "n_pairs": s2_n, "passed": s2_pass, "pass_pct": 100 * s2_pass / s2_n,
        "baseline_passed": s2_base, "baseline_pass_pct": 100 * s2_base / s2_n,
        "delta_turns_mean": float(deltas.mean()),
        "delta_turns_median": float(deltas.median()),
        "pairs_more_turns": up, "pairs_fewer_turns": down, "sign_test_p": sign_p,
        "lcs_mean": float(S2["copy_lcs_fraction"].mean()),
        "lcs_median": float(S2["copy_lcs_fraction"].median()),
        "lcs_gt_0_5_pct": 100 * (S2["copy_lcs_fraction"] > 0.5).mean(),
        "max_rot_mean": float(S2["max_rotation_deg"].mean()),
        "baseline_rot_mean": float(S2["baseline_max_rotation_deg"].mean()),
        "steps_mean": float(S2["total_steps"].mean()),
        "baseline_steps_mean": float(S2["baseline_steps"].mean()),
        "digest_valid": int(S2["digest_valid"].sum()),
        "source_level_counts": S2["source_level"].value_counts().sort_index().to_dict(),
        "failed_pairs": S2.loc[S2["passed"] == 0, ["pair_id", "model", "end_reason", "total_steps"]].to_dict("records"),
    }
    # digest text: does it ever mention turning? (raw stage2 episodes)
    try:
        raw_s2 = os.path.join(ROOT, "数据图片与教程", "数据",
                              "stage2_final_20261027_154234", "results")
        import glob
        n_digest, n_turn, n_straight = 0, 0, 0
        for path in glob.glob(os.path.join(raw_s2, "*", "*", "episode_*.json")):
            if "_steps" in os.path.basename(path):
                continue
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            dg = str(rec.get("stage2_digest") or "")
            if not dg:
                continue
            n_digest += 1
            if re.search(r"\b(turn|turning|rotate|rotat|sideways)\b", dg, re.I):
                n_turn += 1
            if re.search(r"(walk straight|straight ahead|straight forward|move forward|path is clear|no need)",
                         dg, re.I):
                n_straight += 1
        findings["s2"]["digest_n"] = n_digest
        findings["s2"]["digest_mentions_turn"] = n_turn
        findings["s2"]["digest_mentions_straight"] = n_straight
    except Exception as exc:
        findings["s2"]["digest_error"] = repr(exc)

    # =====================================================================
    # S3
    # =====================================================================
    S3 = pd.read_csv(os.path.join(FINAL, "S3_episodes.csv"))
    S3["req_deg"] = [required_angle(a, r) for a, r in zip(S3["a_s_requested"], S3["robot"])]
    S3["met_req"] = (S3["max_rotation_deg"] >= S3["req_deg"] - 1e-9).astype(int)
    s3_by_robot_as = (S3.groupby(["robot", "a_s_requested"])
                      .agg(n=("passed", "size"), passed=("passed", "sum"),
                           pass_pct=("passed", "mean"),
                           mean_rot=("max_rotation_deg", "mean"),
                           met_pct=("met_req", "mean"),
                           steps=("total_steps", "mean"),
                           req_deg=("req_deg", "first"))
                      .reset_index())
    s3_by_robot_as["pass_pct"] *= 100
    s3_by_robot_as["met_pct"] *= 100
    tables["s3_robot_as"] = s3_by_robot_as
    s3_by_robot_as.to_csv(os.path.join(OUT, "s3_robot_as.csv"), index=False)
    s3_model = (S3.groupby(["model", "robot"])
                .agg(n=("passed", "size"), passed=("passed", "sum"),
                     pass_pct=("passed", "mean"), mean_rot=("max_rotation_deg", "mean"),
                     timeout_invalid=("timeout_invalid_count", "sum"))
                .reset_index())
    s3_model["pass_pct"] *= 100
    s3_model.to_csv(os.path.join(OUT, "s3_model_robot.csv"), index=False)
    tables["s3_model"] = s3_model
    s3_09 = S3[S3["a_s_requested"] == 0.9]
    s3_09_rows = []
    for robot, g in s3_09.groupby("robot"):
        s3_09_rows.append({
            "robot": robot, "n": len(g), "passed": int(g["passed"].sum()),
            "pass_pct": 100 * g["passed"].mean(),
            "req_deg": float(g["req_deg"].iloc[0]),
            "mean_rot": float(g["max_rotation_deg"].mean()),
            "median_rot": float(g["max_rotation_deg"].median()),
            "met_req_pct": 100 * g["met_req"].mean(),
            "pass_given_met_pct": 100 * g.loc[g["met_req"] == 1, "passed"].mean() if (g["met_req"] == 1).any() else None,
            "failures": int((g["passed"] == 0).sum()),
            "failures_maxrot_le15": int(((g["passed"] == 0) & (g["max_rotation_deg"] <= 15)).sum()),
        })
    s3_09_tab = pd.DataFrame(s3_09_rows)
    tables["s3_as09"] = s3_09_tab
    s3_09_tab.to_csv(os.path.join(OUT, "s3_as09_by_robot.csv"), index=False)
    # H1 reference at the same A/S, restricted to the same five models
    five = sorted(S3["model"].unique())
    h1 = S1[S1["model"].isin(five) & S1["a_s_ratio"].round(1).isin([1.3, 1.1, 1.0, 0.9])]
    h1_rows = []
    for a_s, g in h1.groupby(h1["a_s_ratio"].round(1)):
        h1_rows.append({"robot": "H1", "a_s_requested": float(a_s), "n": len(g),
                        "passed": int(g["passed"].sum()),
                        "pass_pct": 100 * g["passed"].mean(),
                        "mean_rot": float(g["max_rotation_deg"].mean()),
                        "req_deg": float(g["req_deg"].iloc[0]),
                        "met_pct": 100 * g["met_req"].mean()})
    h1_tab = pd.DataFrame(h1_rows)
    body_compare = pd.concat(
        [h1_tab, s3_by_robot_as.rename(columns={"a_s_requested": "a_s_requested"})],
        ignore_index=True, sort=False)
    body_compare.to_csv(os.path.join(OUT, "s3_body_compare.csv"), index=False)
    tables["s3_body_compare"] = body_compare
    # robot-vs-robot tests at A/S 0.9 and overall
    from math import comb
    def fisher_two_sided(a, b, c, d):
        # a=pass robot1, b=fail robot1, c=pass robot2, d=fail robot2
        n = a + b + c + d
        row1, col1 = a + b, a + c
        from math import lgamma
        def hyper(x):
            return math.exp(lgamma(row1 + 1) - lgamma(x + 1) - lgamma(row1 - x + 1)
                            + lgamma(n - row1 + 1) - lgamma(col1 - x + 1)
                            - lgamma(n - row1 - col1 + x + 1) - (lgamma(n + 1)
                            - lgamma(col1 + 1) - lgamma(n - col1 + 1)))
        p_obs = hyper(a)
        return float(sum(hyper(x) for x in range(max(0, col1 - (n - row1)),
                                                min(row1, col1) + 1) if hyper(x) <= p_obs + 1e-12))
    g1_09 = S3[(S3["robot"] == "G1") & (S3["a_s_requested"] == 0.9)]
    r1_09 = S3[(S3["robot"] == "R1") & (S3["a_s_requested"] == 0.9)]
    g1_all = S3[S3["robot"] == "G1"]
    r1_all = S3[S3["robot"] == "R1"]
    findings["s3"] = {
        "n": int(len(S3)), "passed": int(S3["passed"].sum()),
        "pass_pct": 100 * S3["passed"].mean(),
        "by_robot": {r: {"n": int(len(g)), "passed": int(g["passed"].sum()),
                         "pass_pct": 100 * g["passed"].mean()}
                     for r, g in S3.groupby("robot")},
        "by_as": {float(k): {"n": int(len(g)), "passed": int(g["passed"].sum()),
                              "pass_pct": 100 * g["passed"].mean()}
                  for k, g in S3.groupby("a_s_requested")},
        "all_failures_at_09": bool((S3[S3["passed"] == 0]["a_s_requested"] == 0.9).all()),
        "g1_vs_r1_all_p": fisher_two_sided(int(g1_all["passed"].sum()),
                                            int((g1_all["passed"] == 0).sum()),
                                            int(r1_all["passed"].sum()),
                                            int((r1_all["passed"] == 0).sum())),
        "g1_vs_r1_09_p": fisher_two_sided(int(g1_09["passed"].sum()),
                                           int((g1_09["passed"] == 0).sum()),
                                           int(r1_09["passed"].sum()),
                                           int((r1_09["passed"] == 0).sum())),
        "failures": S3[S3["passed"] == 0][["pair_id", "model", "robot", "scene",
                                            "a_s_requested", "total_steps", "end_reason",
                                            "max_rotation_deg", "timeout_invalid_count"]].to_dict("records"),
        "timeout_invalid_total": int(S3["timeout_invalid_count"].sum()),
        "timeout_episodes": int((S3["timeout_invalid_count"] > 0).sum()),
        "total_llm_hours": float(S3["total_llm_time_ms"].sum() / 3.6e6),
    }
    # required angle table for the three bodies
    req_tab = []
    for body in BODY:
        for a_s in (1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            req_tab.append({"robot": body, "a_s": a_s, "required_deg": required_angle(a_s, body)})
    req_tab = pd.DataFrame(req_tab)
    req_tab.to_csv(os.path.join(OUT, "required_angles.csv"), index=False)
    tables["required_angles"] = req_tab

    # ---- human comparison summary ------------------------------------------
    findings["human_comparison"] = {
        "pi_hat_pooled": pooled["pi_rot30"],
        "pi_hat_ci": [pooled["pi_rot30_lo"], pooled["pi_rot30_hi"]],
        "collapse_boundary": "between A/S 1.0 and 0.9 (pass 94.4% -> 27.2%)",
        "pass_at_1.0_pct": float(100 * S1[S1["level"] == 10]["passed"].mean()),
        "pass_at_0.9_pct": float(100 * S1[S1["level"] == 11]["passed"].mean()),
        "human": HUMAN,
        "relative_gap_vs_human_pct_using_pihat": {
            k: 100 * (pooled["pi_rot30"] - v) / v for k, v in HUMAN.items()},
        "relative_gap_vs_human_pct_using_boundary": {
            k: {"1.0_minus_h": 100 * (1.0 - v) / v, "1.0_minus_0.1": None} for k, v in HUMAN.items()},
    }

    # ---- write JSON + tables markdown --------------------------------------
    with open(os.path.join(OUT, "findings.json"), "w", encoding="utf-8") as fh:
        json.dump(findings, fh, ensure_ascii=False, indent=1, default=float)

    lines = ["# v8 最终分析表\n"]
    for name, tab in tables.items():
        lines.append(f"\n## {name}\n")
        try:
            lines.append(tab.round(4).to_markdown(index=False))
        except Exception:
            lines.append(tab.to_string(index=False))
        lines.append("")
    lines.append("\n## findings\n\n```json\n")
    lines.append(json.dumps(findings, ensure_ascii=False, indent=1, default=float))
    lines.append("\n```\n")
    with open(os.path.join(OUT, "report_tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(json.dumps({
        "pi_hat_pooled": findings["pi_hat"]["pooled"],
        "necessity": findings["necessity"],
        "timing": findings["timing"],
        "first_turn_position": findings["first_turn_position"],
        "wide_turn_cost": findings["wide_turn_cost"],
        "strategy_correlation": findings["strategy_correlation"],
        "s2": {k: v for k, v in findings["s2"].items() if k != "failed_pairs"},
        "s3": {k: v for k, v in findings["s3"].items() if k != "failures"},
        "robustness": findings["robustness"],
        "data_quality": findings["data_quality"],
    }, ensure_ascii=False, indent=1, default=float))
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
