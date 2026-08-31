import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance, PROC_TIME_RANGE, N_DYE_CATEGORIES


def test_setup_time_scale():
    inst = generate_instance(jobs_per_machine=10, m=2, seed=42)
    st = inst["setup_time"]
    off_diag = st[~np.eye(st.shape[0], dtype=bool)]
    mean_setup = off_diag.mean()
    # norm_setup has mean 1, setup_time = norm_setup * SETUP_TIME_MEAN (1.5h)
    assert 1.0 < mean_setup < 2.0, f"mean setup {mean_setup:.3f}h outside [1.0, 2.0]"


def test_setup_time_zero_diagonal():
    inst = generate_instance(jobs_per_machine=10, m=2, seed=0)
    assert np.allclose(np.diag(inst["setup_time"]), 0.0)


def test_weekly_capacity_sanity():
    inst = generate_instance(jobs_per_machine=10, m=2, seed=7)
    total_work = inst["proc_times"].sum()
    weekly_capacity = inst["m"] * 168.0
    assert total_work <= weekly_capacity
    assert total_work > 0


def test_due_dates_within_week():
    inst = generate_instance(jobs_per_machine=15, m=3, seed=42)
    assert inst["due_dates"].min() >= 0
    assert inst["due_dates"].max() <= 200


def test_weights_all_one():
    inst = generate_instance(jobs_per_machine=10, m=2, seed=0)
    assert np.allclose(inst["weights"], 1.0)


def test_release_all_zero():
    inst = generate_instance(jobs_per_machine=10, m=2, seed=0)
    assert inst["release"].max() == 0.0


def test_instance_has_required_keys():
    inst = generate_instance(jobs_per_machine=10, m=2, seed=0)
    required = ["n", "m", "proc_times", "due_dates", "weights", "release",
                "setup_cost", "setup_time", "colour_ids", "colour_darkness",
                "dye_category"]
    for key in required:
        assert key in inst, f"Missing key: {key}"


def test_same_seed_same_instance():
    inst_a = generate_instance(jobs_per_machine=15, m=2, seed=99)
    inst_b = generate_instance(jobs_per_machine=15, m=2, seed=99)
    np.testing.assert_array_equal(inst_a["proc_times"], inst_b["proc_times"])
    np.testing.assert_array_equal(inst_a["setup_cost"], inst_b["setup_cost"])


def test_category_count():
    inst = generate_instance(jobs_per_machine=10, m=3, seed=0)
    cats = np.unique(inst["dye_category"])
    assert len(cats) <= 4
    assert cats[0] >= 0


def test_proc_times_per_category():
    inst = generate_instance(jobs_per_machine=10, m=3, seed=42)
    for cat_id in range(N_DYE_CATEGORIES):
        mask = inst["dye_category"] == cat_id
        if mask.any():
            lo, hi = PROC_TIME_RANGE[cat_id]
            assert np.all(inst["proc_times"][mask] >= lo), f"cat {cat_id} below {lo}"
            assert np.all(inst["proc_times"][mask] <= hi), f"cat {cat_id} above {hi}"