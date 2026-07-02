"""
Train PPO on a pool of instances.
Run from project root: python experiments/train_ppo.py [--profile baseline|realistic]
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.drl_agent import train_ppo

TOTAL_TIMESTEPS = 100_000
TRAIN_CONFIGS = INSTANCE_CONFIGS
TRAIN_SEEDS = list(range(10))


def run(profile="baseline"):
    save_path = f"models/ppo_hyperheuristic_{profile}"
    instance_pool = [
        generate_instance(n=cfg["n"], m=cfg["m"], seed=s, profile=profile)
        for cfg in TRAIN_CONFIGS
        for s in TRAIN_SEEDS
    ]
    print(f"Training [{profile}] on {len(instance_pool)} instances "
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
    parser.add_argument("--profile", default="baseline", choices=["baseline", "realistic"])
    args = parser.parse_args()
    run(profile=args.profile)
