@echo off
REM run_regret_timeseries.bat - Generate regret (one-step deviation gain) time-series CSVs

setlocal enabledelayedexpansion

REM ----------------------------
REM Defaults (override by setting env vars before running)
REM ----------------------------
if not defined N_USERS     set "N_USERS=4000"
if not defined N_RUNS      set "N_RUNS=3"
if not defined FT1         set "FT1=3"
if not defined FT2         set "FT2=9"
if not defined REGRET_N_MC set "REGRET_N_MC=3"

REM Output directory
if not exist "results" mkdir results

echo.
echo ====================================================================
echo Regret time-series (one-step deviation gain) for BTCE
echo ====================================================================
echo N_USERS     = !N_USERS!
echo N_RUNS      = !N_RUNS!
echo REGRET_N_MC = !REGRET_N_MC!
echo ====================================================================
echo.

REM ----------------------------
REM f = 3/9
REM ----------------------------
echo.
echo --------------------------------------------------------------------
echo Running regret time-series for f = !FT1!/9
echo --------------------------------------------------------------------
python scripts\run_btce_regret.py ^
  --n_users !N_USERS! ^
  --n_runs !N_RUNS! ^
  --ft !FT1! ^
  --regret_n_mc !REGRET_N_MC! ^
  --out results\regret_timeseries_ft!FT1!.csv

if errorlevel 1 (
  echo ERROR: Regret time-series failed for f=!FT1!/9. Exiting.
  exit /b 1
)

REM ----------------------------
REM f = 9/9 (stress / worst case)
REM ----------------------------
echo.
echo --------------------------------------------------------------------
echo Running regret time-series for f = !FT2!/9
echo --------------------------------------------------------------------
python scripts\run_btce_regret.py ^
  --n_users !N_USERS! ^
  --n_runs !N_RUNS! ^
  --ft !FT2! ^
  --regret_n_mc !REGRET_N_MC! ^
  --out results\regret_timeseries_ft!FT2!.csv

if errorlevel 1 (
  echo ERROR: Regret time-series failed for f=!FT2!/9. Exiting.
  exit /b 1
)

echo.
echo ====================================================================
echo Done.
echo Outputs:
echo   results\regret_timeseries_ft!FT1!.csv
echo   results\regret_timeseries_ft!FT2!.csv
echo ====================================================================
echo.

endlocal
exit /b 0
