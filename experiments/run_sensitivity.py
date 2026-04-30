"""
Sensitivity analysis: rerun GA + Hybrid on medium configs with α∈{0.3, 0.5, 0.7}, parallelised.
Run from project root: python experiments/run_sensitivity.py
Requires trained PPO model at models/ppo_hyperheuristic.zip
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import get_context
from stable_baselines3 import PPO
from src.instance_generator import generate_instance
from src.ga import run_ga
from src.drl_agent import run_hybrid

ALPHAS = [0.3, 0.5, 0.7]
N_SEEDS = 10
CONFIGS = [{"n": 20, "m": 2, "label": "medium_2m"},
           {"n": 20, "m": 3, "label": "medium_3m"}]
MODEL_PATH = "models/ppo_hyperheuristic"
TOTAL_GENS = 50

_worker_model = None


def init_worker():
    global _worker_model
    _worker_model = PPO.load(MODEL_PATH)


def run_one(args):
    cfg, seed, alpha = args
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
    ga_result = run_ga(inst, alpha=alpha, seed=seed, n_gen=TOTAL_GENS)
    hybrid_result = run_hybrid(inst, _worker_model, seed=seed, total_gens=TOTAL_GENS, alpha=alpha)
    return {
        "cfg_label": cfg["label"],
        "seed": seed,
        "alpha": alpha,
        "ga_composite": ga_result["best_fitness"],
        "ga_weighted_tardiness": ga_result["weighted_tardiness"],
        "ga_setup_cost": ga_result["setup_cost"],
        "hybrid_composite": hybrid_result["composite"],
        "hybrid_weighted_tardiness": hybrid_result["weighted_tardiness"],
        "hybrid_setup_cost": hybrid_result["setup_cost"],
    }


def run():
    tasks = [(cfg, seed, alpha) for cfg in CONFIGS for seed in range(N_SEEDS) for alpha in ALPHAS]
    results = {}

    with get_context("spawn").Pool(initializer=init_worker) as pool:
        for entry in pool.imap_unordered(run_one, tasks):
            label = entry.pop("cfg_label")
            results.setdefault(label, []).append(entry)
            print(f"  {label} seed={entry['seed']} α={entry['alpha']}  "
                  f"GA={entry['ga_composite']:.3f}  Hybrid={entry['hybrid_composite']:.3f}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/raw/sensitivity.json")


if __name__ == "__main__":
    run()
