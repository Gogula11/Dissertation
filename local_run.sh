#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-all}"
SMOKE="${2:-}"

source /home/dopedino/miniforge3/bin/activate dissertation
cd /home/dopedino/Documents/Dissertation

run_profile() {
    local p=$1
    echo "=== PROFILE: $p ==="
    mkdir -p results/raw models logs figures

    SECONDS=0
    python experiments/run_baselines.py --profile "$p" $SMOKE
    echo "  baselines: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
    PROFILE_TOTAL=$SECONDS

    SECONDS=0
    python experiments/train_ppo.py --profile "$p" $SMOKE
    echo "  train_ppo: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
    PROFILE_TOTAL=$((PROFILE_TOTAL + SECONDS))

    SECONDS=0
    python experiments/run_ga.py --profile "$p" $SMOKE
    echo "  run_ga: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
    PROFILE_TOTAL=$((PROFILE_TOTAL + SECONDS))

    SECONDS=0
    python experiments/run_hybrid.py --profile "$p" $SMOKE
    echo "  run_hybrid: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
    PROFILE_TOTAL=$((PROFILE_TOTAL + SECONDS))

    SECONDS=0
    python experiments/run_sensitivity.py --profile "$p" $SMOKE
    echo "  run_sensitivity: ${SECONDS}s ($(($SECONDS/60))m $(($SECONDS%60))s)"
    PROFILE_TOTAL=$((PROFILE_TOTAL + SECONDS))

    echo "  PROFILE TOTAL: $(($PROFILE_TOTAL/60))m $(($PROFILE_TOTAL%60))s"
}

OVERALL_START=$SECONDS

case "$PROFILE" in
    all)
        run_profile baseline
        run_profile realistic
        ;;
    baseline|realistic)
        run_profile "$PROFILE"
        ;;
    *)
        echo "Usage: $0 [baseline|realistic|all]"
        exit 1
        ;;
esac

OVERALL_END=$((SECONDS - OVERALL_START))
echo "=== ALL DONE ==="
echo "Total time: $(($OVERALL_END/60))m $(($OVERALL_END%60))s"