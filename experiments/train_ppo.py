"""
Train PPO on a pool of instances (small_2m, seeds 0-4).
Run from project root: python experiments/train_ppo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.drl_agent import train_ppo

TOTAL_TIMESTEPS = 100_000
SAVE_PATH = "models/ppo_hyperheuristic"
TRAIN_SEEDS = list(range(10))


def run():
    instance_pool = [
        generate_instance(n=cfg["n"], m=cfg["m"], seed=s)
        for cfg in INSTANCE_CONFIGS
        for s in TRAIN_SEEDS
    ]
    print(f"Training on {len(instance_pool)} instances "
          f"({len(INSTANCE_CONFIGS)} configs x {len(TRAIN_SEEDS)} seeds)")
    print(f"Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Output: {SAVE_PATH}.zip")

    train_ppo(
        instance_pool,
        total_timesteps=TOTAL_TIMESTEPS,
        save_path=SAVE_PATH,
        verbose=1,
        pop_size=50,
    )


if __name__ == "__main__":
    run()
