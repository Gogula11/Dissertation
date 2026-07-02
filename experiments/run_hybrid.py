"""
Run hybrid GA+DRL for all instance configs, 30 seeds each, parallelised.
Requires a trained PPO model at models/ppo_hyperheuristic_{profile}.zip
Run from project root: python experiments/run_hybrid.py [--profile baseline|realistic]
"""

import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import get_context
from stable_baselines3 import PPO
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.drl_agent import run_hybrid

ALPHA = 0.5
N_SEEDS = 50
TOTAL_GENS = 300

_worker_model = None


def _init_worker(model_path):
    global _worker_model
    _worker_model = PPO.load(model_path, device="cpu")


def run_one(args):
    cfg, seed, profile = args
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed, profile=profile)
    result = run_hybrid(inst, _worker_model, seed=seed, total_gens=TOTAL_GENS, alpha=ALPHA)
    return cfg["label"], {
        "seed":               seed,
        "composite":          result["composite"],
        "weighted_tardiness": result["weighted_tardiness"],
        "setup_cost":         result["setup_cost"],
        "makespan":           result["makespan"],
    }


def run(profile="baseline"):
    if _SMOKE:
        _run_sequential(profile)
        return
    model_path = f"models/ppo_hyperheuristic_{profile}"
    tasks = [(cfg, seed, profile) for cfg in INSTANCE_CONFIGS for seed in range(N_SEEDS)]
    results = {cfg["label"]: [] for cfg in INSTANCE_CONFIGS}

    with get_context("spawn").Pool(initializer=_init_worker, initargs=(model_path,)) as pool:
        for label, data in pool.imap_unordered(run_one, tasks):
            results[label].append(data)
            print(f"  Done [{profile}]: {label} seed={data['seed']} composite={data['composite']:.2f}")

    os.makedirs("results/raw", exist_ok=True)
    path = f"results/raw/hybrid_{profile}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


def _run_sequential(profile):
    model_path = f"models/ppo_hyperheuristic_{profile}"
    model = PPO.load(model_path, device="cpu")
    os.makedirs("results/raw", exist_ok=True)
    results = {}
    for cfg in _CFG_LIST:
        for seed in range(N_SEEDS):
            inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed, profile=profile)
            result = run_hybrid(inst, model, seed=seed, total_gens=TOTAL_GENS, alpha=ALPHA)
            results.setdefault(cfg["label"], []).append({
                "seed": seed,
                "composite": result["composite"],
                "weighted_tardiness": result["weighted_tardiness"],
                "setup_cost": result["setup_cost"],
                "makespan": result["makespan"],
            })
            print(f"  Done [{profile}]: {cfg['label']} seed={seed} composite={result['composite']:.2f}")
    path = f"results/raw/hybrid_{profile}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="baseline", choices=["baseline", "realistic"])
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (tiny config, 3 seeds, n_gen=10)")
    args = parser.parse_args()
    _SMOKE = args.smoke
    _CFG_LIST = [c for c in INSTANCE_CONFIGS if c["label"] == "tiny_2m"] if _SMOKE else INSTANCE_CONFIGS
    if _SMOKE:
        N_SEEDS = 3
        TOTAL_GENS = 10
        print("[SMOKE] Overriding hybrid params")
    run(profile=args.profile)
