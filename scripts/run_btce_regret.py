#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from btce_tables import core

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=4000)
    ap.add_argument("--ft", type=int, default=3)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--regret_n_mc", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows = []
    for seed in range(args.n_runs):
        core.set_seed(seed)
        core.set_globals(NUM_USERS=args.n_users)
        df = core.run_simulation(args.ft, log_regret=True, regret_n_mc=args.regret_n_mc, regret_seed=seed)

        # Average regret per day for targets vs non-targets
        df["IsTarget"] = df["IsTarget"].astype(int)
        g = (df.groupby(["Day", "IsTarget"])["Regret"]
               .mean()
               .reset_index()
               .pivot(index="Day", columns="IsTarget", values="Regret")
               .rename(columns={0: "Regret_NonTarget", 1: "Regret_Target"}))

        g = g.reset_index()
        g["seed"] = seed
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out_mean = (out.groupby("Day", as_index=False)[["Regret_Target","Regret_NonTarget"]]
                  .mean()
                  .fillna(0.0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_mean.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
