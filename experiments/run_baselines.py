"""
Run SPT and nearest-neighbour baselines across all instance configs.
Saves to results/raw/baselines.json
Run from project root: python experiments/run_baselines.py
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.heuristics import spt, nearest_neighbour_greedy
from src.evaluator import evaluate

ALPHA = 0.5
N_SEEDS = 30


def run():
    results = {cfg["label"]: {"spt": [], "nn_greedy": []} for cfg in INSTANCE_CONFIGS}

    for cfg in INSTANCE_CONFIGS:
        print(f"Running baselines: {cfg['label']}")
        for seed in range(N_SEEDS):
            inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
            for name, fn in [("spt", spt), ("nn_greedy", nearest_neighbour_greedy)]:
                sigma = fn(inst)
                ev = evaluate(sigma, inst, alpha=ALPHA)
                results[cfg["label"]][name].append({
                    "seed": seed,
                    "composite":          ev["composite"],
                    "weighted_tardiness": ev["weighted_tardiness"],
                    "setup_cost":         ev["setup_cost"],
                    "makespan":           ev["makespan"],
                })

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/raw/baselines.json")


if __name__ == "__main__":
    run()
