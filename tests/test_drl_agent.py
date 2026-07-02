import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock
from src.instance_generator import generate_instance
from src.drl_agent import make_env_fn, run_hybrid


def test_make_env_fn():
    inst = generate_instance(n=10, m=2, seed=0)
    fn = make_env_fn(inst, total_gens=10, step_gens=5, pop_size=5)
    env = fn()
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 3
    env.close()


def test_run_hybrid_returns_result():
    inst = generate_instance(n=10, m=2, seed=0)
    mock_model = MagicMock()
    mock_model.predict.return_value = (0, None)

    result = run_hybrid(
        inst, mock_model, seed=42,
        total_gens=10, step_gens=5, pop_size=5,
    )
    assert isinstance(result, dict)
    assert "composite" in result
    assert "weighted_tardiness" in result
    assert "setup_cost" in result
    assert "makespan" in result


if __name__ == "__main__":
    test_make_env_fn()
    test_run_hybrid_returns_result()
    print("All drl_agent tests passed.")
