# BTCE empirical table reproduction scripts

This folder contains standalone Python scripts that reproduce the three empirical tables described in the Open Science Appendix:

1. `scripts/run_btce_f1_modes.bat` -> `results/table1_btce_f1_period_vs_userever.csv`
2. `scripts/run_btce_beh_vs_rat.bat` -> `results/table2_beh_vs_rat_timing.csv`
3. `scripts/run_btce_vs_transformer.bat` -> `results/table3_beh_vs_transformer_userearly.csv`
4. `scripts/run_btce_regret.bat` -> `results/regret_timeseries_ft3.csv` and `results/regret_timeseries_ft9.csv`

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

bash scripts/run_all_tables.sh
```

## Notes

- Table 3 trains the Transformer-UBS baseline using **user-level labels** (`IsTarget`) rather than any BTCE-derived per-period labels.
- Delta mean/median reported by Table 2 are clamped to be non-negative to avoid negative values in tables.

