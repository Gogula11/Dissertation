#!/usr/bin/env bash
set -euo pipefail

SMALL=""
SMOKE=""

for arg in "$@"; do
    case "$arg" in
        --small) SMALL="--small" ;;
        --smoke) SMOKE="--smoke" ;;
    esac
done

# Environment setup
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    if [ -d venv ]; then
        source venv/bin/activate
    elif [ -d /home/dopedino/miniforge3 ]; then
        source /home/dopedino/miniforge3/bin/activate dissertation 2>/dev/null || true
    fi
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$SMOKE" ]; then
    echo "=== SMOKE TEST ==="
fi
echo "=== Running experiment pipeline ==="
mkdir -p results/raw models logs figures

TOTAL=0

SECONDS=0
python experiments/run_baselines.py $SMALL $SMOKE
echo "  baselines: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/train_ppo.py $SMALL $SMOKE
echo "  train_ppo: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_ga.py $SMALL $SMOKE
echo "  run_ga: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_hybrid.py $SMALL $SMOKE
echo "  run_hybrid: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

SECONDS=0
python experiments/run_sensitivity.py $SMALL $SMOKE
echo "  run_sensitivity: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

# Run notebooks to regenerate figures
echo "=== Running notebooks ==="
SECONDS=0
for nb in notebooks/01_exploration.ipynb notebooks/02_heuristic_baselines.ipynb \
          notebooks/03_ga_tuning.ipynb notebooks/04_drl_training.ipynb \
          notebooks/05_final_evaluation.ipynb notebooks/06_visualisations.ipynb; do
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
        "$nb" --output "$(basename "$nb")" 2>&1 | tail -1
done
echo "  notebooks: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
TOTAL=$((TOTAL + SECONDS))

# Archive results
TIMESTAMP=$(date +%Y%m%d_%H%M)
echo "=== Archiving results ==="
tar czf "results_${TIMESTAMP}.tar.gz" results/raw/ models/ logs/ figures/
echo "Saved: results_${TIMESTAMP}.tar.gz"

echo "=== ALL DONE ==="
echo "Total time: $(($TOTAL/60))m $(($TOTAL%60))s"