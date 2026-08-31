@echo off
setlocal enabledelayedexpansion

set SMALL=
set SMOKE=

for %%a in (%*) do (
    if "%%a"=="--small" set SMALL=--small
    if "%%a"=="--smoke" set SMOKE=--smoke
)

if not defined VIRTUAL_ENV if not defined CONDA_DEFAULT_ENV (
    echo Activate your environment first (e.g., venv\Scripts\activate)
    exit /b 1
)

mkdir results\raw 2>nul
mkdir models 2>nul
mkdir logs 2>nul
mkdir figures 2>nul
mkdir results\archives 2>nul

if defined SMOKE echo === SMOKE TEST ===
echo === Running experiment pipeline ===

python experiments\run_baselines.py %SMALL% %SMOKE%
python experiments\train_ppo.py %SMALL% %SMOKE%
python experiments\run_ga.py %SMALL% %SMOKE%
python experiments\run_hybrid.py %SMALL% %SMOKE%
python experiments\run_sensitivity.py %SMALL% %SMOKE%

echo === Running notebooks ===
for %%n in (
    "01_exploration"
    "02_heuristic_baselines"
    "03_ga_tuning"
    "04_drl_training"
    "05_final_evaluation"
    "06_visualisations"
) do (
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 ^
        "notebooks\%%~n.ipynb" --output "%%~n.ipynb" 2>nul
    if errorlevel 1 (
        echo WARNING: %%~n.ipynb failed
    )
)

echo === Archiving results ==
set TIMESTAMP=%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%
tar -czf "results\archives\results_%TIMESTAMP: =0%.tar.gz" results\raw models logs figures
echo Saved: results\archives\results_%TIMESTAMP: =0%.tar.gz

echo === ALL DONE ===