"""
Sensitivity analysis: rerun GA + Hybrid on medium configs with α∈{0.3, 0.5, 0.7}, parallelised.
Run from project root: python experiments/run_sensitivity.py [--profile baseline|realistic]
Requires trained PPO model at models/ppo_hyperheuristic_{profile}.zip
"""

import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import get_context
from stable_baselines3 import PPO
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.ga import run_ga
from src.drl_agent import run_hybrid

ALPHAS = [0.3, 0.5, 0.7]
N_SEEDS = 30
CONFIGS = INSTANCE_CONFIGS
TOTAL_GENS = 100


def run_one(args):
    cfg, seed, alpha, profile = args
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed, profile=profile)
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


_worker_model = None


def _init_worker(model_path):
    global _worker_model
    _worker_model = PPO.load(model_path, device="cpu")


def run(profile="baseline"):
    if _SMOKE:
        _run_sequential(profile)
        return
    model_path = f"models/ppo_hyperheuristic_{profile}"

    tasks = [(cfg, seed, alpha, profile) for cfg in CONFIGS for seed in range(N_SEEDS) for alpha in ALPHAS]
    results = {}

    with get_context("spawn").Pool(initializer=_init_worker, initargs=(model_path,)) as pool:
        for entry in pool.imap_unordered(run_one, tasks):
            label = entry.pop("cfg_label")
            results.setdefault(label, []).append(entry)
            print(f"  [{profile}] {label} seed={entry['seed']} α={entry['alpha']}  "
                  f"GA={entry['ga_composite']:.3f}  Hybrid={entry['hybrid_composite']:.3f}")

    os.makedirs("results/raw", exist_ok=True)
    path = f"results/raw/sensitivity_{profile}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


def _run_sequential(profile):
    _init_worker(f"models/ppo_hyperheuristic_{profile}")
    os.makedirs("results/raw", exist_ok=True)
    results = {}
    for cfg in CONFIGS:
        for seed in range(N_SEEDS):
            for alpha in ALPHAS:
                inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed, profile=profile)
                ga_result = run_ga(inst, alpha=alpha, seed=seed, n_gen=TOTAL_GENS)
                hybrid_result = run_hybrid(inst, _worker_model, seed=seed, total_gens=TOTAL_GENS, alpha=alpha)
                entry = {
                    "seed": seed,
                    "alpha": alpha,
                    "ga_composite": ga_result["best_fitness"],
                    "ga_weighted_tardiness": ga_result["weighted_tardiness"],
                    "ga_setup_cost": ga_result["setup_cost"],
                    "hybrid_composite": hybrid_result["composite"],
                    "hybrid_weighted_tardiness": hybrid_result["weighted_tardiness"],
                    "hybrid_setup_cost": hybrid_result["setup_cost"],
                }
                results.setdefault(cfg["label"], []).append(entry)
                print(f"  [{profile}] {cfg['label']} seed={seed} α={alpha}  "
                      f"GA={entry['ga_composite']:.3f}  Hybrid={entry['hybrid_composite']:.3f}")
    path = f"results/raw/sensitivity_{profile}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="baseline", choices=["baseline", "realistic"])
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (tiny config, 3 seeds, n_gen=5)")
    args = parser.parse_args()
    _SMOKE = args.smoke
    CONFIGS = [c for c in INSTANCE_CONFIGS if c["label"] == "tiny_2m"] if _SMOKE else INSTANCE_CONFIGS
    if _SMOKE:
        N_SEEDS = 3
        TOTAL_GENS = 5
        print("[SMOKE] Overriding sensitivity params")
    run(profile=args.profile)
