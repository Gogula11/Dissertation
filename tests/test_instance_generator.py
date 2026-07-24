import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance


def test_setup_time_scale_default():
    inst = generate_instance(n=10, m=2, seed=42)
    avg_setup = inst["setup_time"].mean()
    avg_proc = inst["proc_times"].mean()
    ratio = avg_setup / avg_proc
    assert 0.10 < ratio < 0.16, f"setup/proc ratio {ratio:.3f} outside [0.10, 0.16]"


def test_setup_time_zero_diagonal():
    inst = generate_instance(n=10, m=2, seed=0)
    assert np.allclose(np.diag(inst["setup_time"]), 0.0)


def test_weekly_capacity_sanity():
    inst = generate_instance(n=10, m=2, seed=7)
    total_work = inst["proc_times"].sum()
    weekly_capacity = inst["m"] * 168.0
    assert total_work < weekly_capacity * 1.2
    assert total_work > weekly_capacity * 0.3


def test_due_dates_within_week():
    inst = generate_instance(n=20, m=3, seed=42)
    assert inst["due_dates"].min() >= 0
    assert inst["due_dates"].max() <= 200


def test_weights_all_one():
    inst = generate_instance(n=10, m=2, seed=0)
    assert np.allclose(inst["weights"], 1.0)


def test_release_all_zero():
    inst = generate_instance(n=10, m=2, seed=0)
    assert inst["release"].max() == 0.0


def test_instance_has_required_keys():
    inst = generate_instance(n=10, m=2, seed=0)
    required = ["n", "m", "proc_times", "due_dates", "weights", "release",
                "setup_cost", "setup_time", "colour_ids", "colour_darkness"]
    for key in required:
        assert key in inst, f"Missing key: {key}"


def test_same_seed_same_instance():
    inst_a = generate_instance(n=20, m=2, seed=99)
    inst_b = generate_instance(n=20, m=2, seed=99)
    np.testing.assert_array_equal(inst_a["proc_times"], inst_b["proc_times"])
    np.testing.assert_array_equal(inst_a["setup_cost"], inst_b["setup_cost"])