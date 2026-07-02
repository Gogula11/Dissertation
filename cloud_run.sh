#!/usr/bin/env bash
# cloud_run.sh — Full experiment pipeline for cloud VM
# Usage: bash cloud_run.sh [profile]
#   profile: "baseline" (default), "realistic", or "all"
set -euo pipefail

PROFILE="${1:-all}"
REPO_URL="https://github.com/Gogula11/Dissertation.git"

echo "=== Cloud Run: profile=$PROFILE ==="

# --- Setup ---
cd ~
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip screen

if [ ! -d Dissertation ]; then
    git clone "$REPO_URL" Dissertation
fi
cd Dissertation
git pull

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --- Verify ---
python -m pytest tests/ -q

# --- Run ---
run_profile() {
    local p=$1
    echo "=== PROFILE: $p ==="
    mkdir -p results/raw models logs figures
    python experiments/run_baselines.py --profile "$p"
    python experiments/train_ppo.py --profile "$p"
    python experiments/run_ga.py --profile "$p"
    python experiments/run_hybrid.py --profile "$p"
    python experiments/run_sensitivity.py --profile "$p"
}

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

# --- Archive results ---
tar czf "results_$(date +%Y%m%d_%H%M).tar.gz" results/raw/ models/ logs/
echo "=== DONE ==="
echo "Results: ~/Dissertation/results_$(date +%Y%m%d)*.tar.gz"
