#!/usr/bin/env bash
# cloud_run.sh — Full experiment pipeline for cloud VM
set -euo pipefail

SMOKE="${1:-}"
REPO_URL="https://github.com/Gogula11/Dissertation.git"

echo "=== Cloud Run ==="

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

python -m pytest tests/ -q

mkdir -p results/raw models logs figures
python experiments/run_baselines.py $SMOKE
python experiments/train_ppo.py $SMOKE
python experiments/run_ga.py $SMOKE
python experiments/run_hybrid.py $SMOKE
python experiments/run_sensitivity.py $SMOKE

tar czf "results_$(date +%Y%m%d_%H%M).tar.gz" results/raw/ models/ logs/
echo "=== DONE ==="
echo "Results: ~/Dissertation/results_$(date +%Y%m%d)*.tar.gz"