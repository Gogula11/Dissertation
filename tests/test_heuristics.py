import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance
from src.heuristics import spt, nearest_neighbour_greedy


def test_spt_returns_all_jobs():
    inst = generate_instance(n=10, m=2, seed=0)
    sigma = spt(inst)
    all_jobs = sorted([j for seq in sigma for j in seq])
    assert all_jobs == list(range(10)), f"Missing jobs: {set(range(10)) - set(all_jobs)}"


def test_spt_correct_machine_count():
    inst = generate_instance(n=10, m=3, seed=0)
    sigma = spt(inst)
    assert len(sigma) == 3


def test_spt_even_distribution():
    inst = generate_instance(n=10, m=2, seed=0)
    sigma = spt(inst)
    sizes = [len(seq) for seq in sigma]
    assert max(sizes) - min(sizes) <= 1


def test_nn_greedy_returns_all_jobs():
    inst = generate_instance(n=20, m=2, seed=0)
    sigma = nearest_neighbour_greedy(inst)
    all_jobs = sorted([j for seq in sigma for j in seq])
    assert all_jobs == list(range(20))


def test_nn_greedy_correct_machine_count():
    inst = generate_instance(n=10, m=3, seed=0)
    sigma = nearest_neighbour_greedy(inst)
    assert len(sigma) == 3


def test_nn_greedy_no_empty_machines():
    inst = generate_instance(n=15, m=3, seed=0)
    sigma = nearest_neighbour_greedy(inst)
    assert all(len(seq) > 0 for seq in sigma), "All machines should have at least one job"


def test_heuristics_deterministic():
    inst = generate_instance(n=10, m=2, seed=42)
    sigma_a = spt(inst)
    sigma_b = spt(inst)
    assert sigma_a == sigma_b

    sigma_c = nearest_neighbour_greedy(inst)
    sigma_d = nearest_neighbour_greedy(inst)
    assert sigma_c == sigma_d



