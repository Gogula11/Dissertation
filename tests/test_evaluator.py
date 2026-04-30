"""
Tests for the evaluator. Run with: python -m pytest tests/ -v
These are the minimum tests — write more as you find edge cases.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance
from src.evaluator import evaluate, compute_completion_times, validate_sigma


def test_all_jobs_scheduled():
    inst = generate_instance(n=10, m=2, seed=0)
    sigma = [[0,1,2,3,4], [5,6,7,8,9]]
    result = evaluate(sigma, inst)
    assert len(result["completion_times"]) == 10
    assert all(result["completion_times"] > 0)


def test_completion_time_monotone_single_machine():
    """On a single machine, completion times must be strictly increasing."""
    inst = generate_instance(n=5, m=1, seed=42)
    sigma = [[0,1,2,3,4]]
    C = compute_completion_times(sigma, inst)
    for i in range(1, len(C)):
        assert C[i] > C[i-1]


def test_zero_setup_zero_cost():
    inst = generate_instance(n=4, m=1, seed=7)
    inst["setup_cost"][:] = 0.0
    inst["setup_time"][:] = 0.0
    sigma = [[0,1,2,3]]
    result = evaluate(sigma, inst)
    assert result["setup_cost"] == 0.0


def test_invalid_sigma_caught():
    inst = generate_instance(n=4, m=2, seed=0)
    bad_sigma = [[0,1,2,3], [0]]  # job 0 appears twice
    try:
        evaluate(bad_sigma, inst)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_tardiness_nonnegative():
    inst = generate_instance(n=10, m=2, seed=5)
    sigma = [[0,1,2,3,4], [5,6,7,8,9]]
    result = evaluate(sigma, inst)
    assert all(result["tardiness_per_job"] >= 0)


def test_same_seed_same_instance():
    inst_a = generate_instance(n=20, m=2, seed=99)
    inst_b = generate_instance(n=20, m=2, seed=99)
    np.testing.assert_array_equal(inst_a["proc_times"], inst_b["proc_times"])
    np.testing.assert_array_equal(inst_a["setup_cost"], inst_b["setup_cost"])


if __name__ == "__main__":
    test_all_jobs_scheduled()
    test_completion_time_monotone_single_machine()
    test_zero_setup_zero_cost()
    test_invalid_sigma_caught()
    test_tardiness_nonnegative()
    test_same_seed_same_instance()
    print("All tests passed.")
