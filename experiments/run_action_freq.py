"""
Action-frequency analysis of the trained PPO policy on multi-machine
configurations. Loads the trained model, runs the hybrid on each config
for N seeds, and reports per-episode-third operator selection rates.
Run from project root: python experiments/run_action_freq.py [--smoke]
"""

import json, sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import Counter
from stable_baselines3 import PPO
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.drl_agent import run_hybrid

N_SEEDS = 10
TOTAL_GENS = 300
ALPHA = 0.7
ACTION_NAMES = ["swap", "inversion", "insertion"]
DEFAULT_CONFIGS = ["j63_m3", "j95_m5"]


def thirds(actions):
    n = len(actions)
    cuts = [actions[: n // 3], actions[n // 3: 2 * n // 3], actions[2 * n // 3:]]
    return [
        {ACTION_NAMES[a]: round(c / len(part) * 100, 1)
         for a, c in Counter(part).items()}
        for part in cuts if part
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = range(2) if args.smoke else range(N_SEEDS)

    model = PPO.load("models/ppo_hyperheuristic.zip", device="cpu")
    results = {}
    for label in DEFAULT_CONFIGS:
        cfg = next(c for c in INSTANCE_CONFIGS if c["label"] == label)
        all_actions = []
        for seed in seeds:
            inst = generate_instance(
                jobs_per_machine=cfg["jobs_per_machine"], m=cfg["m"], seed=seed)
            res = run_hybrid(inst, model, seed=seed, total_gens=TOTAL_GENS,
                             alpha=ALPHA, collect_actions=True)
            all_actions.extend(res["actions"])
        results[label] = {"n_runs": len(seeds), "thirds": thirds(all_actions)}
        print(f"{label}: {json.dumps(results[label]['thirds'])}")

    out = "results/raw/action_freq_multimachine.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
