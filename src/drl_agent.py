"""
PPO training and inference for the GA hyper-heuristic.
"""

import os
import gc
import random
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.ga_env import GAHyperHeuristicEnv


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = tqdm(total=total_timesteps, desc="PPO training")

    def _on_step(self):
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.close()


def make_env_fn(instance_pool, total_gens=200, step_gens=10, pop_size=100, alpha=0.5, warmup_ratio=0.2, base_seed=42):
    max_steps = total_gens // step_gens
    warmup_steps = int(max_steps * warmup_ratio)
    def _init():
        env = GAHyperHeuristicEnv(
            instance_pool[0], total_gens=total_gens,
            step_gens=step_gens, pop_size=pop_size, alpha=alpha,
            instance_pool=instance_pool, warmup_steps=warmup_steps,
        )
        env.reset(seed=base_seed)
        return env
    return _init


def train_ppo(
    instance_pool: list,
    total_timesteps: int = 20_000,
    save_path: str = "models/ppo_hyperheuristic",
    verbose: int = 1,
    pop_size: int = 25,
    total_gens: int = 100,
    step_gens: int = 10,
    n_envs: int = 16,
    seed: int = None,
) -> PPO:
    """
    Train a PPO agent on a pool of GA environments.
    Each episode randomly samples an instance from instance_pool.
    Saves the model to save_path.zip.
    """
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    # Seed everything for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    n_envs = min(n_envs, multiprocessing.cpu_count() - 2)
    print(f"Using {n_envs} parallel environments")
    env_fns = [
        make_env_fn(
            instance_pool, pop_size=pop_size, total_gens=total_gens,
            step_gens=step_gens, base_seed=seed + i if seed is not None else 42,
        )
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method='spawn')

    log_dir = "logs/ppo_tensorboard"
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        tensorboard_log=log_dir,
        device="cpu",
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, callback=TqdmCallback(total_timesteps))
    model.save(save_path)
    print(f"Saved to {save_path}.zip")
    vec_env.close()
    # gc: n_envs workers linger, explicit cleanup prevents CPU spike
    del vec_env
    gc.collect()
    return model


def run_hybrid(
    instance: dict,
    model: PPO,
    seed: int = None,
    total_gens: int = 200,
    step_gens: int = 10,
    pop_size: int = 100,
    alpha: float = 0.5,
    collect_actions: bool = False,
) -> dict:
    """
    Run the GA with the trained PPO hyper-heuristic.
    Returns same result dict format as run_ga().
    If collect_actions is True, the result dict additionally contains
    "actions": the list of operator indices chosen at each step.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = GAHyperHeuristicEnv(
        instance, total_gens=total_gens,
        step_gens=step_gens, pop_size=pop_size, alpha=alpha
    )
    obs, _ = env.reset(seed=seed)

    done = False
    actions = [] if collect_actions else None
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if collect_actions:
            actions.append(int(action))
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    result = env.get_best_result()
    if collect_actions:
        result["actions"] = actions
    return result
