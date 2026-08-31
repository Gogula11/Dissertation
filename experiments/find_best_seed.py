"""
Find the best training seed for PPO hyper-heuristic.
Trains 20 models (seeds 0-19), evaluates each on all configs, picks the best.
Run from project root: python experiments/find_best_seed.py [--smoke] [--seeds N]
"""

import json, sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from src.instance_generator import generate_instance, INSTANCE_CONFIGS, INSTANCE_CONFIGS_SMALL
from src.drl_agent import train_ppo, run_hybrid

ALPHA = 0.7
EVAL_SEEDS = 50
TOTAL_GENS = 300
N_TRAIN_SEEDS = 20
TOTAL_TIMESTEPS = 200_000

_worker_model = None


def _init_worker(model_path):
    global _worker_model
    _worker_model = PPO.load(model_path, device="cpu")


def eval_one(args):
    cfg, seed = args
    inst = generate_instance(jobs_per_machine=cfg["jobs_per_machine"], m=cfg["m"], seed=seed)
    result = run_hybrid(inst, _worker_model, seed=seed, total_gens=TOTAL_GENS, alpha=ALPHA)
    return cfg["label"], {
        "seed":               seed,
        "composite":          result["composite"],
        "weighted_tardiness": result["weighted_tardiness"],
        "setup_cost":         result["setup_cost"],
        "total_setup_time":   result["total_setup_time"],
        "makespan":           result["makespan"],
    }


def train_and_eval(train_seed, instance_pool, cfg_list, smoke=False):
    """Train one model, evaluate on all configs."""
    save_path = f"models/seed_search/ppo_seed_{train_seed}"
    os.makedirs("models/seed_search", exist_ok=True)

    n_envs = 1 if smoke else min(8, os.cpu_count() - 2)
    timesteps = 1_000 if smoke else TOTAL_TIMESTEPS

    t0 = time.time()
    train_ppo(
        instance_pool,
        total_timesteps=timesteps,
        save_path=save_path,
        verbose=0,
        pop_size=25,
        n_envs=n_envs,
        seed=train_seed,
    )
    train_time = time.time() - t0

    # Evaluate
    eval_tasks = [(cfg, s) for cfg in cfg_list for s in range(EVAL_SEEDS)]
    results = {cfg["label"]: [] for cfg in cfg_list}

    model = PPO.load(save_path, device="cpu")
    for cfg, seed in eval_tasks:
        inst = generate_instance(jobs_per_machine=cfg["jobs_per_machine"], m=cfg["m"], seed=seed)
        result = run_hybrid(inst, model, seed=seed, total_gens=TOTAL_GENS, alpha=ALPHA)
        results[cfg["label"]].append({
            "seed": seed,
            "composite": result["composite"],
        })

    # Compute mean composite per config and overall
    config_means = {}
    for label, entries in results.items():
        config_means[label] = sum(e["composite"] for e in entries) / len(entries)
    overall_mean = sum(config_means.values()) / len(config_means)

    return overall_mean, config_means, train_time


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (1k steps, 3 eval seeds)")
    parser.add_argument("--small", action="store_true", help="Only configs with n <= 50")
    parser.add_argument("--seeds", type=int, default=N_TRAIN_SEEDS, help="Number of training seeds to search")
    args = parser.parse_args()

    cfg_list = INSTANCE_CONFIGS_SMALL if args.small else INSTANCE_CONFIGS
    n_seeds = args.seeds

    instance_pool = [
        generate_instance(jobs_per_machine=cfg["jobs_per_machine"], m=cfg["m"], seed=s)
        for cfg in INSTANCE_CONFIGS
        for s in range(10)
    ]

    print(f"Searching {n_seeds} training seeds (0-{n_seeds-1})")
    print(f"Evaluating on {len(cfg_list)} configs x {EVAL_SEEDS} eval seeds")
    print(f"{'='*70}")

    all_results = []
    for train_seed in range(n_seeds):
        t0 = time.time()
        overall_mean, config_means, train_time = train_and_eval(
            train_seed, instance_pool, cfg_list, smoke=args.smoke
        )
        elapsed = time.time() - t0
        all_results.append({
            "train_seed": train_seed,
            "overall_mean": overall_mean,
            "config_means": config_means,
            "train_time": train_time,
            "total_time": elapsed,
        })
        print(f"Seed {train_seed:2d}: mean_composite={overall_mean:.4f}  "
              f"train={train_time:.0f}s  total={elapsed:.0f}s")

    # Sort by overall mean (lower is better)
    all_results.sort(key=lambda x: x["overall_mean"])

    print(f"\n{'='*70}")
    print("RANKING (lower composite = better)")
    print(f"{'='*70}")
    for i, r in enumerate(all_results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  #{i+1}  seed={r['train_seed']:2d}  "
              f"mean_composite={r['overall_mean']:.4f}{marker}")

    best = all_results[0]
    print(f"\nBest training seed: {best['train_seed']}")
    print(f"Mean composite: {best['overall_mean']:.4f}")
    print(f"\nPer-config breakdown:")
    for label, mean in sorted(best["config_means"].items()):
        print(f"  {label:10s}: {mean:.4f}")

    # Save results
    with open("results/raw/seed_search.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/raw/seed_search.json")


if __name__ == "__main__":
    run()
