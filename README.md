# BTCE empirical table reproduction scripts

This folder contains standalone Python scripts that reproduce the three empirical tables described in the Open Science Appendix:

1. `scripts\run_btce_f1_modes.bat` -> `results/table1_btce_f1_period_vs_userever.csv`
2. `scripts\run_btce_beh_vs_rat.bat` -> `results/table2_beh_vs_rat_timing.csv`
3. `scripts\run_btce_vs_transformer.bat` -> `results/table3_beh_vs_transformer_userearly.csv`
4. `scripts\run_btce_regret.bat` -> `results/regret_timeseries_ft3.csv` and `results/regret_timeseries_ft9.csv`
5. `scripts\run_btce_merger_day_sweep.bat` -> `results\merger_sweep_regret_ft3.csv` and `results\merger_sweep_regret_ft9.csv`, `results\merger_sweep_metrics_ft3.csv` and `results\merger_sweep_metrics_ft9.csv`

## Quickstart

```sh
python.exe -m venv .venv 
.\.venv\Scripts\activate
pip install -r requirements.txt
.\scripts\run_all_tables.sh
.\scripts\run_btce_merger_day_sweep.sh
```

## Notes

- Table 3 trains the Transformer-UBS baseline using **user-level labels** (`IsTarget`) rather than any BTCE-derived per-period labels.
- Delta mean/median reported by Table 2 are clamped to be non-negative to avoid negative values in tables.

