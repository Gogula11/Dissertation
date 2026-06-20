"""
Run plain GA for all instance configs, 30 seeds each, parallelised.
Run from project root: python experiments/run_ga.py
Do NOT run this from a notebook — multiprocessing + DEAP global state = problems.
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import Pool, get_context
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.ga import run_ga

ALPHA = 0.5
N_SEEDS = 30
# Lock these down after tuning in notebook 03
GA_PARAMS = {"n_gen": 200, "pop_size": 100, "cx_prob": 0.9, "mut_prob": 0.2}


def run_one(args):
    cfg, seed = args
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
    result = run_ga(inst, **GA_PARAMS, alpha=ALPHA, seed=seed)
    return cfg["label"], {
        "seed":               seed,
        "composite":          result["best_fitness"],
        "weighted_tardiness": result["weighted_tardiness"],
        "setup_cost":         result["setup_cost"],
        "makespan":           result["makespan"],
    }


def run():
    tasks = [(cfg, seed) for cfg in INSTANCE_CONFIGS for seed in range(N_SEEDS)]
    results = {cfg["label"]: [] for cfg in INSTANCE_CONFIGS}

    with get_context("spawn").Pool() as pool:
        for label, data in pool.imap_unordered(run_one, tasks):
            results[label].append(data)
            print(f"  Done: {label} seed={data['seed']}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/ga.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/raw/ga.json")


if __name__ == "__main__":
    run()
