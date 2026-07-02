import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance, validate_instance


def test_validate_valid_instance_default():
    inst = generate_instance(n=10, m=3, seed=42)
    errors = validate_instance(inst)
    assert errors == [], f"Expected no errors, got: {errors}"


def test_validate_valid_instance_profile_enhanced():
    inst = generate_instance(n=10, m=3, seed=42, profile="enhanced")
    errors = validate_instance(inst)
    assert errors == [], f"Enhanced profile errors: {errors}"


def test_validate_valid_instance_profile_realistic():
    inst = generate_instance(n=20, m=2, seed=7, profile="realistic")
    errors = validate_instance(inst)
    assert errors == [], f"Realistic profile errors: {errors}"


def test_validate_valid_instance_all_sizes():
    sizes = [(5, 2), (10, 2), (30, 3), (50, 3), (100, 5)]
    for n, m in sizes:
        for profile in ("baseline", "enhanced", "realistic"):
            inst = generate_instance(n=n, m=m, seed=0, profile=profile)
            errs = validate_instance(inst)
            assert errs == [], f"{profile} ({n}x{m}): {errs}"


def test_validate_catches_negative_proc_times():
    inst = generate_instance(n=10, m=2, seed=0)
    inst["proc_times"][0] = -1.0
    errors = validate_instance(inst)
    assert "proc_times" in str(errors)


def test_validate_catches_shape_mismatch():
    inst = generate_instance(n=10, m=2, seed=0)
    inst["weights"] = np.array([1.0, 2.0])
    errors = validate_instance(inst)
    assert any("shape" in e.lower() for e in errors)


def test_validate_catches_negative_setup_cost():
    inst = generate_instance(n=10, m=2, seed=0)
    inst["setup_cost"][0, 1] = -5.0
    errors = validate_instance(inst)
    assert any("negative" in e for e in errors)


def test_validate_catches_nonzero_diagonal():
    inst = generate_instance(n=10, m=2, seed=0)
    inst["setup_time"][3, 3] = 1.0
    errors = validate_instance(inst)
    assert any("diagonal" in e for e in errors)


def test_validate_catches_bad_chemistries():
    inst = generate_instance(n=10, m=2, seed=0, profile="realistic")
    inst["chemistries"][2] = "invalid_chem"
    errors = validate_instance(inst)
    assert any("chemistries" in e for e in errors)


def test_validate_catches_negative_colour_ids():
    inst = generate_instance(n=10, m=2, seed=0)
    inst["colour_ids"][0] = -1
    errors = validate_instance(inst)
    assert any("negative" in e for e in errors)


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
