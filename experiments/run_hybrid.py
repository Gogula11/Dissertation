"""
Run hybrid GA+DRL for all instance configs, 30 seeds each, parallelised.
Requires a trained PPO model at models/ppo_hyperheuristic.zip
Run from project root: python experiments/run_hybrid.py
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import get_context
from stable_baselines3 import PPO
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.drl_agent import run_hybrid

ALPHA = 0.5
N_SEEDS = 30
MODEL_PATH = "models/ppo_hyperheuristic"
TOTAL_GENS = 50

_worker_model = None


def init_worker():
    global _worker_model
    _worker_model = PPO.load(MODEL_PATH)


def run_one(args):
    cfg, seed = args
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
    result = run_hybrid(inst, _worker_model, seed=seed, total_gens=TOTAL_GENS, alpha=ALPHA)
    return cfg["label"], {
        "seed":               seed,
        "composite":          result["composite"],
        "weighted_tardiness": result["weighted_tardiness"],
        "setup_cost":         result["setup_cost"],
        "makespan":           result["makespan"],
    }


def run():
    tasks = [(cfg, seed) for cfg in INSTANCE_CONFIGS for seed in range(N_SEEDS)]
    results = {cfg["label"]: [] for cfg in INSTANCE_CONFIGS}

    with get_context("spawn").Pool(initializer=init_worker) as pool:
        for label, data in pool.imap_unordered(run_one, tasks):
            results[label].append(data)
            print(f"  Done: {label} seed={data['seed']} composite={data['composite']:.2f}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/hybrid.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/raw/hybrid.json")


if __name__ == "__main__":
    run()
