#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

# Ensure local import works when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from btce_tables import core

def _as_float_list(xs):
    return [float(x) for x in xs]

def _timing_only(df, theta):
    # Use user_preexfil objective to compute timing stats, but we will only export timing fields.
    m = core.user_level_metrics(
        df, score_col="P_Detect", theta=theta,
        label_col="y_killchain", attack_def="killchain", timing_scope="label",
        f1_mode="user_preexfil"
    )
    # Clamp deltas to be non-negative for reporting (avoid negative deltas in tables).
    dm = float(m.get("Delta mean (days)", np.nan))
    dmed = float(m.get("Delta median (days)", np.nan))
    mttd = float(m.get("MTTD (days)", np.nan))
    return max(dm, 0.0) if not np.isnan(dm) else np.nan, max(dmed, 0.0) if not np.isnan(dmed) else np.nan, mttd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", required=True, type=float)
    ap.add_argument("--n_users", type=int, default=4000)
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--ft", type=int, default=3)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    thresholds = core.ensure_iterable_thresholds(args.thresholds)

    rows = []
    for seed in range(args.n_runs):
        # Behavioral
        core.set_seed(seed)
        core.globals()["NUM_USERS"] = args.n_users
        core.globals()["BEHAVIOR_MODE"] = "behavioral"
        df_b = core.run_simulation(args.ft)
        df_b = core.add_killchain_labels(df_b)
        df_b["P_Detect"] = df_b["P_Detect"].astype(float)

        # Rational (same seed so the target cohort aligns under your code path)
        core.set_seed(seed)
        core.globals()["NUM_USERS"] = args.n_users
        core.globals()["BEHAVIOR_MODE"] = "rational"
        df_r = core.run_simulation(args.ft)
        df_r = core.add_killchain_labels(df_r)
        df_r["P_Detect"] = df_r["P_Detect"].astype(float)

        for th in thresholds:
            dm_b, dmed_b, mttd_b = _timing_only(df_b, th)
            dm_r, dmed_r, mttd_r = _timing_only(df_r, th)

            rows.append({
                "seed": seed,
                "theta": float(th),
                "Delta_mean_beh": dm_b,
                "Delta_median_beh": dmed_b,
                "MTTD_beh": mttd_b,
                "Delta_mean_rat": dm_r,
                "Delta_median_rat": dmed_r,
                "MTTD_rat": mttd_r,
                "Delta_mean_diff": (dm_r - dm_b) if (not np.isnan(dm_b) and not np.isnan(dm_r)) else np.nan,
                "Delta_median_diff": (dmed_r - dmed_b) if (not np.isnan(dmed_b) and not np.isnan(dmed_r)) else np.nan,
                "MTTD_diff": (mttd_r - mttd_b) if (not np.isnan(mttd_b) and not np.isnan(mttd_r)) else np.nan,
            })

    df_out = pd.DataFrame(rows)
    cols = ["Delta_mean_beh","Delta_median_beh","MTTD_beh","Delta_mean_rat","Delta_median_rat","MTTD_rat","Delta_mean_diff","Delta_median_diff","MTTD_diff"]
    df_mean = df_out.groupby("theta", as_index=False)[cols].mean()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_mean.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
