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

SECONDS=0
python experiments/run_baselines.py $SMALL $SMOKE
T_BASELINES=$SECONDS
echo "  baselines: ${T_BASELINES}s ($((T_BASELINES/60))m $((T_BASELINES%60))s)"

SECONDS=0
python experiments/train_ppo.py $SMALL $SMOKE
T_TRAIN=$SECONDS
echo "  train_ppo: ${T_TRAIN}s ($((T_TRAIN/60))m $((T_TRAIN%60))s)"

SECONDS=0
python experiments/run_ga.py $SMALL $SMOKE
T_GA=$SECONDS
echo "  run_ga: ${T_GA}s ($((T_GA/60))m $((T_GA%60))s)"

SECONDS=0
python experiments/run_hybrid.py $SMALL $SMOKE
T_HYBRID=$SECONDS
echo "  run_hybrid: ${T_HYBRID}s ($((T_HYBRID/60))m $((T_HYBRID%60))s)"

SECONDS=0
python experiments/run_sensitivity.py $SMALL $SMOKE
T_SENS=$SECONDS
echo "  run_sensitivity: ${T_SENS}s ($((T_SENS/60))m $((T_SENS%60))s)"

# Run notebooks to regenerate figures
echo "=== Running notebooks ==="
SECONDS=0
for nb in notebooks/01_exploration.ipynb notebooks/02_heuristic_baselines.ipynb \
          notebooks/03_ga_tuning.ipynb notebooks/04_drl_training.ipynb \
          notebooks/05_final_evaluation.ipynb notebooks/06_visualisations.ipynb; do
    log=logs/nb_$(basename "$nb" .ipynb).log
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
        "$nb" --output "$(basename "$nb")" 2>&1 | tail -1 \
        || { echo "  WARNING: $nb failed — tail of $log:";
             jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
                 "$nb" --output "$(basename "$nb")" > "$log" 2>&1;
             tail -25 "$log" | sed 's/^/    /'; }
done
T_NOTEBOOKS=$SECONDS
echo "  notebooks: ${T_NOTEBOOKS}s ($((T_NOTEBOOKS/60))m $((T_NOTEBOOKS%60))s)"

# Archive results
TIMESTAMP=$(date +%Y%m%d_%H%M)
echo "=== Archiving results ==="
tar czf "results_${TIMESTAMP}.tar.gz" results/raw/ models/ logs/ figures/
echo "Saved: results_${TIMESTAMP}.tar.gz"

echo "=== ALL DONE ==="

TOTAL=$((T_BASELINES + T_TRAIN + T_GA + T_HYBRID + T_SENS + T_NOTEBOOKS))
echo "=== Timing Summary ==="
printf "  baselines:       %2ds (%dh %2dm %2ds)\n" "$T_BASELINES"  "$((T_BASELINES/3600))"  "$(((T_BASELINES%3600)/60))"  "$((T_BASELINES%60))"
printf "  train_ppo:       %2ds (%dh %2dm %2ds)\n" "$T_TRAIN"      "$((T_TRAIN/3600))"      "$(((T_TRAIN%3600)/60))"      "$((T_TRAIN%60))"
printf "  run_ga:          %2ds (%dh %2dm %2ds)\n" "$T_GA"         "$((T_GA/3600))"         "$(((T_GA%3600)/60))"         "$((T_GA%60))"
printf "  run_hybrid:      %2ds (%dh %2dm %2ds)\n" "$T_HYBRID"     "$((T_HYBRID/3600))"     "$(((T_HYBRID%3600)/60))"     "$((T_HYBRID%60))"
printf "  run_sensitivity: %2ds (%dh %2dm %2ds)\n" "$T_SENS"       "$((T_SENS/3600))"       "$(((T_SENS%3600)/60))"       "$((T_SENS%60))"
printf "  notebooks:       %2ds (%dh %2dm %2ds)\n" "$T_NOTEBOOKS"  "$((T_NOTEBOOKS/3600))"  "$(((T_NOTEBOOKS%3600)/60))"  "$((T_NOTEBOOKS%60))"
echo "  -------------------------------"
printf "  Total:           %2ds (%dh %2dm %2ds)\n" "$TOTAL"  "$((TOTAL/3600))"  "$(((TOTAL%3600)/60))"  "$((TOTAL%60))"