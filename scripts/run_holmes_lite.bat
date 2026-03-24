@echo off
REM run_holmes_lite.bat - Evaluate HOLMES-lite baseline vs BTCE

setlocal enabledelayedexpansion

REM Configuration (override via environment variables)
set "THRESHOLDS=%THRESHOLDS:0.75 0.85 0.90%"
set "N_USERS=%N_USERS:4000%"
set "N_RUNS=%N_RUNS:10%"
set "FT=%FT:3%"
set "DEVICE=%DEVICE:cuda%"

REM Create results directory
if not exist "results" mkdir results

echo.
echo ====================================================================
echo Running HOLMES-lite vs BTCE-Behavioral comparison
echo ====================================================================
python scripts\run_holmes_lite.py ^
  --thresholds !THRESHOLDS! ^
  --n_users !N_USERS! ^
  --n_runs !N_RUNS! ^
  --ft !FT! ^
  --lookback 5 ^
  --decay 0.8 ^
  --out results/table_holmes_lite.csv

if errorlevel 1 (
  echo ERROR: HOLMES-lite evaluation failed. Exiting.
  exit /b 1
)

echo.
echo ====================================================================
echo HOLMES-lite evaluation complete.
echo Results saved to: results/table_holmes_lite.csv
echo ====================================================================

endlocal
exit /b 0
