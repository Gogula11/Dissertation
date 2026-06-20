import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance
from src.ga import run_ga, decode_chromosome, mutInsertion


def test_ga_runs_and_returns_dict():
    inst = generate_instance(n=10, m=2, seed=0)
    result = run_ga(inst, n_gen=5, pop_size=10, seed=42)
    assert isinstance(result, dict)
    assert "best_fitness" in result
    assert "weighted_tardiness" in result
    assert "setup_cost" in result
    assert "makespan" in result
    assert "best_sigma" in result


def test_ga_sigma_valid():
    inst = generate_instance(n=10, m=2, seed=0)
    result = run_ga(inst, n_gen=5, pop_size=10, seed=42)
    sigma = result["best_sigma"]
    all_jobs = sorted([j for seq in sigma for j in seq])
    assert all_jobs == list(range(10))


def test_ga_swap_mutation():
    inst = generate_instance(n=10, m=2, seed=0)
    result = run_ga(inst, n_gen=5, pop_size=10, seed=42, mutation_strategy="swap")
    assert result["best_fitness"] > 0


def test_ga_inversion_mutation():
    inst = generate_instance(n=10, m=2, seed=0)
    result = run_ga(inst, n_gen=5, pop_size=10, seed=42, mutation_strategy="inversion")
    assert result["best_fitness"] > 0


def test_ga_insertion_mutation():
    inst = generate_instance(n=10, m=2, seed=0)
    result = run_ga(inst, n_gen=5, pop_size=10, seed=42, mutation_strategy="insertion")
    assert result["best_fitness"] > 0


def test_mut_insertion_preserves_jobs():
    ind = list(range(10))
    mutated, = mutInsertion(ind[:], indpb=1.0)
    assert sorted(mutated) == list(range(10))
    assert mutated != ind  # very likely different with indpb=1.0


def test_ga_different_seeds_different_results():
    inst = generate_instance(n=10, m=2, seed=0)
    r1 = run_ga(inst, n_gen=10, pop_size=20, seed=1)
    r2 = run_ga(inst, n_gen=10, pop_size=20, seed=2)
    assert r1["best_fitness"] != r2["best_fitness"]


def test_ga_aggressive_swap_mutation():
    inst = generate_instance(n=10, m=2, seed=0)
    result = run_ga(inst, n_gen=5, pop_size=10, seed=42, mutation_strategy="aggressive_swap")
    assert result["best_fitness"] > 0


def test_decode_chromosome_preserves_jobs():
    ind = list(range(10))
    sigma = decode_chromosome(ind, m=2)
    flat = [j for seq in sigma for j in seq]
    assert flat == list(range(10))


if __name__ == "__main__":
    test_ga_runs_and_returns_dict()
    test_ga_sigma_valid()
    test_ga_swap_mutation()
    test_ga_inversion_mutation()
    test_ga_aggressive_swap_mutation()
    test_ga_different_seeds_different_results()
    test_decode_chromosome_preserves_jobs()
    print("All GA tests passed.")
