"""
Tests for the evaluator. Run with: python -m pytest tests/ -v
These are the minimum tests — write more as you find edge cases.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance
from src.evaluator import evaluate, compute_completion_times, validate_sigma, estimate_scales


def test_all_jobs_scheduled():
    inst = generate_instance(n=10, m=2, seed=0)
    f1s, f2s = estimate_scales(inst)
    sigma = [[0,1,2,3,4], [5,6,7,8,9]]
    result = evaluate(sigma, inst, f1_scale=f1s, f2_scale=f2s)
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
    f1s, f2s = estimate_scales(inst)
    inst["setup_cost"][:] = 0.0
    inst["setup_time"][:] = 0.0
    sigma = [[0,1,2,3]]
    result = evaluate(sigma, inst, f1_scale=f1s, f2_scale=f2s)
    assert result["setup_cost"] == 0.0


def test_invalid_sigma_caught():
    inst = generate_instance(n=4, m=2, seed=0)
    f1s, f2s = estimate_scales(inst)
    bad_sigma = [[0,1,2,3], [0]]  # job 0 appears twice
    try:
        evaluate(bad_sigma, inst, f1_scale=f1s, f2_scale=f2s)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_tardiness_nonnegative():
    inst = generate_instance(n=10, m=2, seed=5)
    f1s, f2s = estimate_scales(inst)
    sigma = [[0,1,2,3,4], [5,6,7,8,9]]
    result = evaluate(sigma, inst, f1_scale=f1s, f2_scale=f2s)
    assert all(result["tardiness_per_job"] >= 0)


def test_empty_machine_sequence():
    inst = generate_instance(n=3, m=3, seed=0)
    sigma = [[0, 1, 2], [], []]
    C = compute_completion_times(sigma, inst)
    assert len(C) == 3
    assert C[0] > 0
    assert C[1] > 0
    assert C[2] > 0


def test_same_seed_same_instance():
    inst_a = generate_instance(n=20, m=2, seed=99)
    inst_b = generate_instance(n=20, m=2, seed=99)
    np.testing.assert_array_equal(inst_a["proc_times"], inst_b["proc_times"])
    np.testing.assert_array_equal(inst_a["setup_cost"], inst_b["setup_cost"])



