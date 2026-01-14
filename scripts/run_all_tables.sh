#!/usr/bin/env bash
set -euo pipefail

THRESHOLDS=${THRESHOLDS:-"0.75 0.85 0.90"}
N_USERS=${N_USERS:-4000}
N_RUNS=${N_RUNS:-10}
FT=${FT:-3}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-64}
DEVICE=${DEVICE:-cuda}

mkdir -p results

python scripts/run_btce_f1_modes.py --thresholds ${THRESHOLDS} --n_users ${N_USERS} --n_runs ${N_RUNS} --ft ${FT} --out results/table1_btce_f1_period_vs_userever.csv
python scripts/run_btce_beh_vs_rat.py --thresholds ${THRESHOLDS} --n_users ${N_USERS} --n_runs ${N_RUNS} --ft ${FT} --out results/table2_beh_vs_rat_timing.csv
python scripts/run_btce_vs_transformer.py --thresholds ${THRESHOLDS} --n_users ${N_USERS} --n_runs ${N_RUNS} --ft ${FT} --epochs ${EPOCHS} --batch ${BATCH} --device ${DEVICE} --out results/table3_beh_vs_transformer_userearly.csv
