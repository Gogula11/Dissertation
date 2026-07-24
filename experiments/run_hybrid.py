"""
Run hybrid GA+DRL for all instance configs, 30 seeds each, parallelised.
Requires trained PPO model at models/ppo_hyperheuristic.zip
Run from project root: python experiments/run_hybrid.py [--smoke] [--small]
"""

import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import get_context
from stable_baselines3 import PPO
from src.instance_generator import generate_instance, INSTANCE_CONFIGS, INSTANCE_CONFIGS_SMALL
from src.drl_agent import run_hybrid

ALPHA = 0.5
N_SEEDS = 50
TOTAL_GENS = 300

_worker_model = None


def _init_worker(model_path):
    global _worker_model
    _worker_model = PPO.load(model_path, device="cpu")


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
    model_path = "models/ppo_hyperheuristic"
    tasks = [(cfg, seed) for cfg in _CFG_LIST for seed in range(N_SEEDS)]
    results = {cfg["label"]: [] for cfg in _CFG_LIST}

    if _SMOKE:
        _init_worker(model_path)
        for label, data in map(run_one, tasks):
            results[label].append(data)
            print(f"  Done: {label} seed={data['seed']} composite={data['composite']:.2f}")
    else:
        with get_context("spawn").Pool(initializer=_init_worker, initargs=(model_path,)) as pool:
            for label, data in pool.imap_unordered(run_one, tasks):
                results[label].append(data)
                print(f"  Done: {label} seed={data['seed']} composite={data['composite']:.2f}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/hybrid.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/raw/hybrid.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (tiny config, 3 seeds, n_gen=10)")
    parser.add_argument("--small", action="store_true", help="Only configs with n <= 50")
    args = parser.parse_args()
    _SMOKE = args.smoke
    _CFG_LIST = INSTANCE_CONFIGS_SMALL if args.small else INSTANCE_CONFIGS
    if _SMOKE:
        N_SEEDS = 3
        _CFG_LIST = [c for c in _CFG_LIST if c["label"] == "tiny_1m"]
        TOTAL_GENS = 10
        print("[SMOKE] Overriding hybrid params")
    run()