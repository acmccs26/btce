#!/usr/bin/env python3
"""
run_holmes_lite.py — Evaluate HOLMES-lite baseline alongside BTCE (Behavioral)
on the same simulation, splits, and thresholds.

Outputs a CSV with columns:
  theta, Method, Pre-Exfil (%), MTTD (days), User FP (%), Mal FN (%), F1

Usage:
  python scripts/run_holmes_lite.py \
    --thresholds 0.75 0.85 0.90 \
    --n_users 4000 --n_runs 10 --ft 3 \
    --out results/table_holmes_lite.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from btce_tables import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", required=True, type=float)
    ap.add_argument("--n_users", type=int, default=4000)
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--ft", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=5,
                    help="HOLMES-lite lookback window (days)")
    ap.add_argument("--decay", type=float, default=0.8,
                    help="HOLMES-lite exponential decay factor")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    thresholds = core.ensure_iterable_thresholds(args.thresholds)
    rows = []

    for seed in range(args.n_runs):
        print(f"[run {seed+1}/{args.n_runs}]", flush=True)

        # ----- Behavioral simulation -----
        core.set_seed(seed)
        core.set_globals(NUM_USERS=args.n_users)
        core.set_globals(BEHAVIOR_MODE="behavioral")

        df_btce = core.run_simulation(args.ft)
        df_btce = core.add_killchain_labels(df_btce)

        # Action-derived labels (same as run_one)
        df_btce["y_stage"] = (
            df_btce["Action"].astype(str).isin(["RECON", "EXFIL"]).astype(int)
        )
        df_btce["P_Detect"] = df_btce["P_Detect"].astype(float)

        # ----- User split (deterministic, matches run_one) -----
        tr_u, te_u = core.split_users_stratified_60_40(
            df_btce, train_frac=0.6, seed=seed
        )

        df_te = df_btce[df_btce["AgentID"].isin(te_u)].copy()

        # ----- BTCE Behavioral (reference) -----
        for th in thresholds:
            m_btce_early = core.user_level_metrics(
                df_te,
                score_col="P_Detect",
                theta=th,
                label_col="y_stage",
                attack_def="exfil",
                timing_scope="any",
                f1_mode="user_preexfil",
            )
            rows.append({
                "seed": seed,
                "Method": "BTCE-Behavioral",
                **m_btce_early,
            })

        # ----- HOLMES-lite -----
        holmes_rows, _, _ = core.eval_holmes_lite(
            df_btce,
            te_u,
            thresholds,
            signal_cols=tuple(core.SIG_COLS),
            lookback=args.lookback,
            decay=args.decay,
            zscore_mode="train_users",
            train_users=tr_u,
            label_col="y_stage",
            attack_def="exfil",
            timing_scope="any",
        )

        # eval_holmes_lite returns list of (method_str, metrics_dict)
        # Two entries per threshold: UserEarly, then UserEver.
        # We keep only UserEarly to match the BTCE rows above.
        # theta is already inside m_dict from user_level_metrics.
        for method_str, m_dict in holmes_rows:
            if "UserEarly" not in method_str:
                continue
            rows.append({
                "seed": seed,
                "Method": "HOLMESLite",
                **m_dict,
            })

    # ----- Aggregate across runs -----
    df_all = pd.DataFrame(rows)

    metric_cols = [
        "Pre-Exfil (%)",
        "Delta mean (days)",
        "Delta median (days)",
        "MTTD (days)",
        "User FP (%)",
        "Mal FN (%)",
        "F1",
    ]

    df_mean = (
        df_all
        .groupby(["Method", "theta"], as_index=False)[metric_cols]
        .mean()
        .round(4)
    )

    # ----- Save -----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_mean.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
    print(df_mean.to_string(index=False))


if __name__ == "__main__":
    main()
