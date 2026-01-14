#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

# Ensure local import works when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from btce_tables import core

def _as_float_list(xs):
    return [float(x) for x in xs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", required=True, type=float)
    ap.add_argument("--n_users", type=int, default=4000)
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--ft", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)  # cuda/cpu
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--poison_rate", type=float, default=0.0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    thresholds = core.ensure_iterable_thresholds(args.thresholds)

    rows = []
    device = args.device

    for seed in range(args.n_runs):
        # build the BTCE dataframe (behavioral) and split users
        core.set_seed(seed)
        core.set_globals(NUM_USERS=args.n_users)
        core.set_globals(BEHAVIOR_MODE = "behavioral")
        df = core.run_simulation(args.ft)
        df = core.add_killchain_labels(df)
        df["P_Detect"] = df["P_Detect"].astype(float)

        tr_u, te_u = core.split_users_stratified_60_40(df, train_frac=0.6, seed=seed)

        # Transformer baseline: use user-level IsTarget labels (NO y_killchain as label)
        DENY = {"s_exfil", "killchain"}
        feat_cols = core.infer_transformer_feature_cols(df, drop_cols=DENY)

        model = core.train_transformer_ubs(
            df, feat_cols, tr_u,
            n_bins=args.n_bins, epochs=args.epochs, batch=args.batch,
            poison_rate=args.poison_rate, seed=seed, device=device,
            label_mode="user", user_label_col="IsTarget",
            prefix_train=True, min_prefix=3
        )
        scores_te = core.score_transformer(
            df, feat_cols, te_u, model,
            n_bins=args.n_bins, batch=args.batch, device=device,
            label_mode="user", user_label_col="IsTarget"
        )
        # Merge into df_te for BTCE-style metric fn if desired
        df_te = df[df["AgentID"].isin(te_u)].merge(scores_te, on=["AgentID","Day"], how="left")
        df_te["score"] = df_te["score"].fillna(0.0)

        for th in thresholds:
            m_btce = core.user_level_metrics(
                df_te, score_col="P_Detect", theta=th,
                label_col="y_killchain", attack_def="killchain", timing_scope="label",
                f1_mode="user_preexfil"
            )
            # Transformer metrics: use user-level labels directly, attack_def exfil timing_scope any
            m_tr = core.user_level_metrics(
                df_te.rename(columns={"score":"TransformerScore"}),
                score_col="TransformerScore", theta=th,
                label_col="IsTarget", attack_def="exfil", timing_scope="any",
                f1_mode="user_preexfil"
            )

            rows.append({
                "seed": seed,
                "theta": float(th),
                "BTCE_PreExfil": float(m_btce.get("Pre-Exfil (%)", np.nan)),
                "BTCE_MTTD": float(m_btce.get("MTTD (days)", np.nan)),
                "BTCE_UserFP": float(m_btce.get("User FP (%)", np.nan)),
                "BTCE_MalFN": float(m_btce.get("Mal FN (%)", np.nan)),
                "Transformer_PreExfil": float(m_tr.get("Pre-Exfil (%)", np.nan)),
                "Transformer_MTTD": float(m_tr.get("MTTD (days)", np.nan)),
                "Transformer_UserFP": float(m_tr.get("User FP (%)", np.nan)),
                "Transformer_MalFN": float(m_tr.get("Mal FN (%)", np.nan)),
            })

    df_out = pd.DataFrame(rows)
    cols = ["BTCE_PreExfil","BTCE_MTTD","BTCE_UserFP","BTCE_MalFN",
            "Transformer_PreExfil","Transformer_MTTD","Transformer_UserFP","Transformer_MalFN"]
    df_mean = df_out.groupby("theta", as_index=False)[cols].mean()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_mean.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
