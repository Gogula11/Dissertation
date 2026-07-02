import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance
from src.ga_env import GAHyperHeuristicEnv, _population_diversity


def test_env_creation():
    inst = generate_instance(n=10, m=2, seed=0)
    env = GAHyperHeuristicEnv(inst, total_gens=10, step_gens=5, pop_size=5)
    assert env.observation_space.shape == (8,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 3
    env.close()


def test_reset_returns_valid_obs():
    inst = generate_instance(n=10, m=2, seed=0)
    env = GAHyperHeuristicEnv(inst, total_gens=10, step_gens=5, pop_size=5)
    obs, info = env.reset(seed=42)
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert np.all((obs >= 0.0) & (obs <= 1.0 + 1e-6))
    assert isinstance(info, dict)
    env.close()


def test_step_action_0():
    inst = generate_instance(n=10, m=2, seed=0)
    env = GAHyperHeuristicEnv(inst, total_gens=10, step_gens=5, pop_size=5)
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    env.close()


def test_step_action_1():
    inst = generate_instance(n=10, m=2, seed=0)
    env = GAHyperHeuristicEnv(inst, total_gens=10, step_gens=5, pop_size=5)
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    env.close()


def test_step_action_2():
    inst = generate_instance(n=10, m=2, seed=0)
    env = GAHyperHeuristicEnv(inst, total_gens=10, step_gens=5, pop_size=5)
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(2)
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    env.close()


def test_episode_truncated_on_step_limit():
    inst = generate_instance(n=10, m=2, seed=0)
    env = GAHyperHeuristicEnv(inst, total_gens=10, step_gens=5, pop_size=5)
    env.reset(seed=42)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
    assert not terminated
    assert truncated
    env.close()


def test_get_best_result_returns_result_dict():
    inst = generate_instance(n=10, m=2, seed=0)
    env = GAHyperHeuristicEnv(inst, total_gens=10, step_gens=5, pop_size=5)
    env.reset(seed=42)
    for _ in range(2):
        env.step(0)
    result = env.get_best_result()
    assert isinstance(result, dict)
    assert "composite" in result
    assert "weighted_tardiness" in result
    assert "setup_cost" in result
    assert "makespan" in result
    assert "completion_times" in result
    env.close()


def test_population_diversity_edge_cases():
    assert _population_diversity([]) == 0.0
    assert _population_diversity([[1, 2, 3]]) == 0.0
    assert _population_diversity([[0, 1, 2], [0, 1, 2]]) == 0.0
    div = _population_diversity([[0, 1, 2], [2, 1, 0]])
    assert div > 0.0
    assert div <= 1.0


if __name__ == "__main__":
    test_env_creation()
    test_reset_returns_valid_obs()
    test_step_action_0()
    test_step_action_1()
    test_step_action_2()
    test_episode_truncated_on_step_limit()
    test_get_best_result_returns_result_dict()
    test_population_diversity_edge_cases()
    print("All ga_env tests passed.")
