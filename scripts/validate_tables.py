#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

# Ensure local import works when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from btce_tables import core

def _as_float_list(xs):
    return [float(x) for x in xs]

def _load_csv(path):
    df = pd.read_csv(path)
    # normalize float columns
    for c in df.columns:
        if c != "theta":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference directory containing the 3 table csvs")
    ap.add_argument("--gen", required=True, help="Generated results directory containing the 3 table csvs")
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()

    files = [
        "table1_btce_f1_period_vs_userever.csv",
        "table2_beh_vs_rat_timing.csv",
        "table3_beh_vs_transformer_userearly.csv",
    ]

    ok = True
    for fn in files:
        ref_path = os.path.join(args.ref, fn)
        gen_path = os.path.join(args.gen, fn)
        if not os.path.exists(ref_path):
            print(f"Missing reference: {ref_path}")
            ok = False
            continue
        if not os.path.exists(gen_path):
            print(f"Missing generated: {gen_path}")
            ok = False
            continue

        ref = _load_csv(ref_path).set_index("theta")
        gen = _load_csv(gen_path).set_index("theta")

        # align
        idx = sorted(set(ref.index) & set(gen.index))
        if not idx:
            print(f"{fn}: no shared thresholds")
            ok = False
            continue
        ref = ref.loc[idx]
        gen = gen.loc[idx]

        # relative error tolerance (fallback absolute for near-zero)
        for col in ref.columns:
            r = ref[col].to_numpy()
            g = gen[col].to_numpy()
            denom = np.maximum(np.abs(r), 1e-9)
            rel = np.abs(g - r) / denom
            if np.nanmax(rel) > args.tolerance:
                print(f"{fn}: FAIL col={col} max_rel={np.nanmax(rel):.4f} tol={args.tolerance}")
                ok = False

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
