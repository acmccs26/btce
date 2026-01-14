@echo off
REM run_all_tables.bat - Windows batch script to generate all empirical tables

setlocal enabledelayedexpansion

REM Configuration (can be overridden via environment variables)
set "THRESHOLDS=%THRESHOLDS:0.75 0.85 0.90%"
set "N_USERS=%N_USERS:4000%"
set "N_RUNS=%N_RUNS:10%"
set "FT=%FT:3%"
set "EPOCHS=%EPOCHS:20%"
set "BATCH=%BATCH:64%"
set "DEVICE=%DEVICE:cuda%"

REM Create results directory
if not exist "results" mkdir results

echo.
echo ====================================================================
echo Running BTCE Table 1: Period-F1 vs UserEver-F1
echo ====================================================================
python scripts/run_btce_f1_modes.py ^
  --thresholds !THRESHOLDS! ^
  --n_users !N_USERS! ^
  --n_runs !N_RUNS! ^
  --ft !FT! ^
  --out results/table1_btce_f1_period_vs_userever.csv

if errorlevel 1 (
  echo ERROR: Table 1 failed. Exiting.
  exit /b 1
)

echo.
echo ====================================================================
echo Running BTCE Table 2: Behavioral vs Rational Timing
echo ====================================================================
python scripts\run_btce_beh_vs_rat.py ^
  --thresholds !THRESHOLDS! ^
  --n_users !N_USERS! ^
  --n_runs !N_RUNS! ^
  --ft !FT! ^
  --out results/table2_beh_vs_rat_timing.csv

if errorlevel 1 (
  echo ERROR: Table 2 failed. Exiting.
  exit /b 1
)

echo.
echo ====================================================================
echo Running BTCE Table 3: BTCE vs Transformer-UBS (UserEarly)
echo ====================================================================
python scripts\run_btce_vs_transformer.py ^
  --thresholds !THRESHOLDS! ^
  --n_users !N_USERS! ^
  --n_runs !N_RUNS! ^
  --ft !FT! ^
  --epochs !EPOCHS! ^
  --batch !BATCH! ^
  --device !DEVICE! ^
  --out results/table3_beh_vs_transformer_userearly.csv

if errorlevel 1 (
  echo ERROR: Table 3 failed. Exiting.
  exit /b 1
)

echo.
echo ====================================================================
echo All tables generated successfully!
echo Results saved to: results/
echo ====================================================================
echo.
echo Table 1: results/table1_btce_f1_period_vs_userever.csv
echo Table 2: results/table2_beh_vs_rat_timing.csv
echo Table 3: results/table3_beh_vs_transformer_userearly.csv
echo.

endlocal
exit /b 0
