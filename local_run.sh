#!/usr/bin/env bash
set -euo pipefail

EXTRA="${1:-}"

source /home/dopedino/miniforge3/bin/activate dissertation
cd /home/dopedino/Documents/Dissertation

echo "=== Running local experiment pipeline (small configs only) ==="
mkdir -p results/raw models logs figures

SECONDS=0
python experiments/run_baselines.py --small $EXTRA
echo "  baselines: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$SECONDS

SECONDS=0
python experiments/train_ppo.py --small $EXTRA
echo "  train_ppo: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_ga.py --small $EXTRA
echo "  run_ga: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_hybrid.py --small $EXTRA
echo "  run_hybrid: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_sensitivity.py --small $EXTRA
echo "  run_sensitivity: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

echo "=== ALL DONE ==="
echo "Total time: $(($TOTAL/60))m $(($TOTAL%60))s"