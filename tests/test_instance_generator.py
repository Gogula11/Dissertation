import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance


def test_setup_time_scale_default_backward_compat():
    """Default setup_time_scale=0.1 matches explicit 0.1"""
    inst_default = generate_instance(n=10, m=2, seed=42)
    inst_explicit = generate_instance(n=10, m=2, seed=42, setup_time_scale=0.1)
    assert np.allclose(inst_default["setup_time"], inst_explicit["setup_time"])


def test_setup_time_scale_doubled():
    inst_default = generate_instance(n=10, m=2, seed=0)
    inst_scaled = generate_instance(n=10, m=2, seed=0, setup_time_scale=0.2)
    assert np.allclose(inst_scaled["setup_time"], inst_default["setup_time"] * 2.0, atol=1e-4)


def test_setup_time_scale_zero():
    inst = generate_instance(n=10, m=2, seed=0, setup_time_scale=0.0)
    assert inst["setup_time"].max() == 0.0


def test_realistic_profile_uses_continuous_colours_and_segments():
    inst = generate_instance(n=20, m=3, seed=7, profile="realistic")
    assert set(inst["chemistries"]).issubset({"direct", "reactive", "vat"})
    assert len(inst["chemistries"]) == 20
    assert inst["colour_darkness"].min() >= 1.0
    assert inst["colour_darkness"].max() <= 10.0
    assert inst["setup_cost"].shape == (20, 20)
    assert inst["setup_time"].shape == (20, 20)
    assert not np.allclose(inst["weights"], 1.0)


def test_sample_colours_uniform_dist():
    from src.instance_generator import _sample_colours
    rng = np.random.default_rng(0)
    ids, darkness, chems = _sample_colours(10, 5, "uniform", 0.0, rng)
    assert len(ids) == 10
    assert len(darkness) == 10
    assert len(chems) == 10


def test_baseline_profile_categorical():
    inst = generate_instance(n=10, m=2, seed=0, profile="baseline")
    assert all(c == "" for c in inst["chemistries"])
    assert np.allclose(inst["weights"], 1.0)
    assert inst["colour_ids"].max() <= 6


def test_categorical_with_proc_colour_corr():
    inst = generate_instance(n=10, m=2, seed=0, profile="baseline", proc_colour_corr=0.5)
    assert all(c == "" for c in inst["chemistries"])
    assert not np.allclose(inst["proc_times"], inst["proc_times"][0])  # varied by darkness
