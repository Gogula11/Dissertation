"""
Train PPO on a pool of instances.
Run from project root: python experiments/train_ppo.py [--smoke] [--small]
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance, INSTANCE_CONFIGS, INSTANCE_CONFIGS_SMALL
from src.drl_agent import train_ppo

TOTAL_TIMESTEPS = 100_000
TRAIN_CONFIGS = INSTANCE_CONFIGS
TRAIN_SEEDS = list(range(10))


def run():
    save_path = "models/ppo_hyperheuristic"
    instance_pool = [
        generate_instance(n=cfg["n"], m=cfg["m"], seed=s)
        for cfg in TRAIN_CONFIGS
        for s in TRAIN_SEEDS
    ]
    print(f"Training on {len(instance_pool)} instances "
          f"({len(TRAIN_CONFIGS)} configs x {len(TRAIN_SEEDS)} seeds)")
    print(f"Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Output: {save_path}.zip")

    train_ppo(
        instance_pool,
        total_timesteps=TOTAL_TIMESTEPS,
        save_path=save_path,
        verbose=1,
        pop_size=25,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (tiny config, 2 seeds, 1k steps)")
    parser.add_argument("--small", action="store_true", help="Only configs with n <= 50")
    args = parser.parse_args()
    TRAIN_CONFIGS = INSTANCE_CONFIGS_SMALL if args.small else INSTANCE_CONFIGS
    if args.smoke:
        TOTAL_TIMESTEPS = 1_000
        TRAIN_CONFIGS = [c for c in TRAIN_CONFIGS if c["label"] == "tiny_1m"]
        TRAIN_SEEDS = list(range(2))
        print("[SMOKE] Overriding training params")
    run()