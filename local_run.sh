#!/usr/bin/env bash
set -euo pipefail

SMALL=""
EXTRA=""

for arg in "$@"; do
    case "$arg" in
        --small) SMALL="--small" ;;
        *) EXTRA="$EXTRA $arg" ;;
    esac
done

# Use already-activated environment (cloud_run.sh or manual activation)
# Fallback: try local conda env name 'dissertation'
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    source /home/dopedino/miniforge3/bin/activate dissertation 2>/dev/null || true
fi

if [ -n "$SMALL" ]; then
    echo "=== Running local experiment pipeline (small configs) ==="
else
    echo "=== Running local experiment pipeline (FULL configs) ==="
fi
mkdir -p results/raw models logs figures

SECONDS=0
python experiments/run_baselines.py $SMALL $EXTRA
echo "  baselines: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$SECONDS

SECONDS=0
python experiments/train_ppo.py $SMALL $EXTRA
echo "  train_ppo: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_ga.py $SMALL $EXTRA
echo "  run_ga: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_hybrid.py $SMALL $EXTRA
echo "  run_hybrid: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_sensitivity.py $SMALL $EXTRA
echo "  run_sensitivity: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

echo "=== ALL DONE ==="
echo "Total time: $(($TOTAL/60))m $(($TOTAL%60))s"