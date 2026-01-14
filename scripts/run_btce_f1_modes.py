#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

from pathlib import Path

# Add parent directory to path so btce_tables can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import btce_tables
import btce_tables.core as core
# ... rest of imports ...


def _as_float_list(xs):
    return [float(x) for x in xs]

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
        core.set_seed(seed)
        # NEW (works):
        core.set_globals(NUM_USERS=args.n_users)
        core.set_globals(BEHAVIOR_MODE = "behavioral")

        df = core.run_simulation(args.ft)
        df = core.add_killchain_labels(df)

        # Evaluate on all users (consistent with paper tables); metrics are user-level on IsTarget population
        df["P_Detect"] = df["P_Detect"].astype(float)

        for th in thresholds:
            m_period = core.user_level_metrics(
                df, score_col="P_Detect", theta=th,
                label_col="y_killchain", attack_def="killchain", timing_scope="label",
                f1_mode="period"
            )
            m_userever = core.user_level_metrics(
                df, score_col="P_Detect", theta=th,
                label_col="y_killchain", attack_def="killchain", timing_scope="label",
                f1_mode="user_ever"
            )
            rows.append({
                "seed": seed,
                "theta": float(th),
                "Pre-Exfil (%)": float(m_period.get("Pre-Exfil (%)", np.nan)),
                "MTTD (days)": float(m_period.get("MTTD (days)", np.nan)),
                "F1 (Period)": float(m_period.get("F1", np.nan)),
                "F1 (UserEver)": float(m_userever.get("F1", np.nan)),
            })

    df_out = pd.DataFrame(rows)
    # Mean over runs
    df_mean = df_out.groupby("theta", as_index=False)[["Pre-Exfil (%)","MTTD (days)","F1 (Period)","F1 (UserEver)"]].mean()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_mean.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
