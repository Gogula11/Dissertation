"""
Run SPT and nearest-neighbour baselines across all instance configs.
Saves to results/raw/baselines.json
Run from project root: python experiments/run_baselines.py [--smoke] [--small]
"""

import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance, INSTANCE_CONFIGS, INSTANCE_CONFIGS_SMALL
from src.heuristics import spt, nearest_neighbour_greedy
from src.evaluator import evaluate, estimate_scales

ALPHA = 0.5
N_SEEDS = 50


def run():
    results = {cfg["label"]: {"spt": [], "nn_greedy": []} for cfg in _CFG_LIST}

    for cfg in _CFG_LIST:
        print(f"Running baselines: {cfg['label']}")
        for seed in range(N_SEEDS):
            inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
            f1s, f2s = estimate_scales(inst)
            for name, fn in [("spt", spt), ("nn_greedy", nearest_neighbour_greedy)]:
                sigma = fn(inst)
                ev = evaluate(sigma, inst, alpha=ALPHA, f1_scale=f1s, f2_scale=f2s)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (3 seeds)")
    parser.add_argument("--small", action="store_true", help="Only configs with n <= 50")
    args = parser.parse_args()
    _CFG_LIST = INSTANCE_CONFIGS_SMALL if args.small else INSTANCE_CONFIGS
    if args.smoke:
        N_SEEDS = 3
        _CFG_LIST = [c for c in _CFG_LIST if c["label"] == "tiny_1m"]
        print("[SMOKE] Overriding baselines params")
    run()
