"""
Run SPT and nearest-neighbour baselines across all instance configs.
Saves to results/raw/baselines.json
Run from project root: python experiments/run_baselines.py [--profile baseline|realistic]
"""

import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.heuristics import spt, nearest_neighbour_greedy
from src.evaluator import evaluate

ALPHA = 0.5
N_SEEDS = 50


def run(profile="baseline"):
    results = {cfg["label"]: {"spt": [], "nn_greedy": []} for cfg in INSTANCE_CONFIGS}

    for cfg in INSTANCE_CONFIGS:
        print(f"Running baselines [{profile}]: {cfg['label']}")
        for seed in range(N_SEEDS):
            inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed, profile=profile)
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
    path = f"results/raw/baselines_{profile}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="baseline", choices=["baseline", "realistic"])
    args = parser.parse_args()
    run(profile=args.profile)
