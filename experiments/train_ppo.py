"""
Train PPO on a pool of instances (small_2m, seeds 0-4).
Run from project root: python experiments/train_ppo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance
from src.drl_agent import train_ppo

TOTAL_TIMESTEPS = 100_000
SAVE_PATH = "models/ppo_hyperheuristic"


def run():
    train_seeds = list(range(5))
    instance_pool = [
        generate_instance(n=10, m=2, seed=s) for s in train_seeds
    ]
    print(f"Training on {len(instance_pool)} instances (seeds {train_seeds})")
    print(f"Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Output: {SAVE_PATH}.zip")

    train_ppo(
        instance_pool,
        total_timesteps=TOTAL_TIMESTEPS,
        save_path=SAVE_PATH,
        verbose=1,
    )


if __name__ == "__main__":
    run()
