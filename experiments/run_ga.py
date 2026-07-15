"""
Run plain GA for all instance configs, 30 seeds each, parallelised.
Run from project root: python experiments/run_ga.py [--profile baseline|realistic]
Do NOT run this from a notebook — multiprocessing + DEAP global state = problems.
"""

import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import get_context
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.ga import run_ga

ALPHA = 0.5
N_SEEDS = 50
GA_PARAMS = {"n_gen": 300, "pop_size": 100, "cx_prob": 0.9, "mut_prob": 0.2}


def run_one(args):
    cfg, seed, profile = args
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed, profile=profile)
    result = run_ga(inst, **GA_PARAMS, alpha=ALPHA, seed=seed)
    return cfg["label"], {
        "seed":               seed,
        "composite":          result["best_fitness"],
        "weighted_tardiness": result["weighted_tardiness"],
        "setup_cost":         result["setup_cost"],
        "makespan":           result["makespan"],
    }


def run(profile="baseline"):
    tasks = [(cfg, seed, profile) for cfg in _CFG_LIST for seed in range(N_SEEDS)]
    results = {cfg["label"]: [] for cfg in _CFG_LIST}

    if _SMOKE:
        for label, data in map(run_one, tasks):
            results[label].append(data)
            print(f"  Done [{profile}]: {label} seed={data['seed']}")
    else:
        with get_context("spawn").Pool() as pool:
            for label, data in pool.imap_unordered(run_one, tasks):
                results[label].append(data)
                print(f"  Done [{profile}]: {label} seed={data['seed']}")

    os.makedirs("results/raw", exist_ok=True)
    path = f"results/raw/ga_{profile}.json"
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
        GA_PARAMS = {**GA_PARAMS, "n_gen": 10}
        print("[SMOKE] Overriding GA params")
    run(profile=args.profile)
