"""
PPO training and inference for the GA hyper-heuristic.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ga_env import GAHyperHeuristicEnv


def make_env_fn(instance_pool, total_gens=200, step_gens=10, pop_size=100, alpha=0.5):
    def _init():
        if isinstance(instance_pool, dict):
            inst = instance_pool
            pool = [instance_pool]
        else:
            inst = instance_pool[0]
            pool = instance_pool
        return GAHyperHeuristicEnv(
            inst, total_gens=total_gens,
            step_gens=step_gens, pop_size=pop_size, alpha=alpha,
            instance_pool=pool,
        )
    return _init


def train_ppo(
    instance_pool: list,
    total_timesteps: int = 100_000,
    save_path: str = "models/ppo_hyperheuristic",
    verbose: int = 1,
) -> PPO:
    """
    Train a PPO agent on a pool of GA environments.
    Each episode randomly samples an instance from instance_pool.
    Saves the model to save_path.zip.
    """
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env_fn(instance_pool)])

    log_dir = "logs/ppo_tensorboard"
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log=log_dir,
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved to {save_path}.zip")
    return model


def run_hybrid(
    instance: dict,
    model: PPO,
    seed: int = None,
    total_gens: int = 200,
    step_gens: int = 10,
    pop_size: int = 100,
    alpha: float = 0.5,
) -> dict:
    """
    Run the GA with the trained PPO hyper-heuristic.
    Returns same result dict format as run_ga().
    """
    import random
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = GAHyperHeuristicEnv(
        instance, total_gens=total_gens,
        step_gens=step_gens, pop_size=pop_size, alpha=alpha
    )
    obs, _ = env.reset(seed=seed)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    return env.get_best_result()
