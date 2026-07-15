"""
Pure evaluator for PMSP-SDSC solutions.

A solution sigma is a list of m lists:
  sigma[k] = ordered list of job indices assigned to machine k

Example for n=6, m=2:
  sigma = [[0, 3, 5], [1, 2, 4]]
  Machine 0: jobs 0 → 3 → 5
  Machine 1: jobs 1 → 2 → 4
"""

import numpy as np
from typing import List
from src.heuristics import nearest_neighbour_greedy


def validate_sigma(sigma: List[List[int]], n: int) -> None:
    """Raise ValueError if any job is missing or duplicated."""
    all_jobs = sorted([j for seq in sigma for j in seq])
    if all_jobs != list(range(n)):
        raise ValueError(f"Invalid sigma: expected jobs 0..{n-1}, got {all_jobs}")


def compute_completion_times(sigma: List[List[int]], instance: dict) -> np.ndarray:
    """
    Compute completion time C_i for every job.
    Returns array of shape (n,).
    """
    n = instance["n"]
    proc = instance["proc_times"]
    setup_t = instance["setup_time"]
    release = instance["release"]

    C = np.zeros(n, dtype=np.float32)

    for machine_seq in sigma:
        if not machine_seq:
            continue
        t = 0.0
        for idx, job in enumerate(machine_seq):
            t = max(t, release[job])
            if idx > 0:
                prev_job = machine_seq[idx - 1]
                t += setup_t[prev_job][job]
            t += proc[job]
            C[job] = t

    return C


def compute_tardiness(C: np.ndarray, instance: dict) -> np.ndarray:
    """T_i = max(0, C_i - d_i) for each job."""
    return np.maximum(0.0, C - instance["due_dates"])


def compute_weighted_tardiness(T: np.ndarray, instance: dict) -> float:
    """f1 = sum(w_i * T_i)"""
    return float(np.dot(instance["weights"], T))


def compute_setup_cost(sigma: List[List[int]], instance: dict) -> float:
    """f2 = total sequence-dependent setup cost across all machines."""
    S = instance["setup_cost"]
    total = 0.0
    for machine_seq in sigma:
        for idx in range(1, len(machine_seq)):
            total += S[machine_seq[idx - 1]][machine_seq[idx]]
    return float(total)


def compute_makespan(C: np.ndarray) -> float:
    """C_max = max(C_i)"""
    return float(C.max())


def _estimate_scales_schedule(instance: dict, order: list) -> tuple:
    """Evaluate a job-ordering-based schedule (round-robin to machines)."""
    m = instance["m"]
    sigma = [[] for _ in range(m)]
    for idx, job in enumerate(order):
        sigma[idx % m].append(int(job))
    C = compute_completion_times(sigma, instance)
    T = compute_tardiness(C, instance)
    f1 = compute_weighted_tardiness(T, instance)
    f2 = compute_setup_cost(sigma, instance)
    return f1, f2


def estimate_scales(instance: dict, rng=None) -> tuple:
    """Estimate normalisation scales from heuristic and random schedules.

    Runs SPT, NN-Greedy, and a random schedule; takes max of all three
    × 1.5 margin. Ensures both f1 and f2 scalars have similar magnitude
    so the composite objective trades them fairly.

    Returns (f1_scale, f2_scale).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = instance["n"]
    proc = instance["proc_times"]

    # SPT
    f1_spt, f2_spt = _estimate_scales_schedule(instance, list(np.argsort(proc)))

    # NN-Greedy
    nn_sigma = nearest_neighbour_greedy(instance)
    C_nn = compute_completion_times(nn_sigma, instance)
    T_nn = compute_tardiness(C_nn, instance)
    f1_nn = compute_weighted_tardiness(T_nn, instance)
    f2_nn = compute_setup_cost(nn_sigma, instance)

    # Random permutation (guarantees non-zero tardiness on most instances)
    rand_order = list(range(n))
    rng.shuffle(rand_order)
    f1_rand, f2_rand = _estimate_scales_schedule(instance, rand_order)

    f1_scale = max(max(f1_spt, f1_nn, f1_rand) * 1.5, 1.0)
    f2_scale = max(max(f2_spt, f2_nn, f2_rand) * 1.5, 1.0)
    return f1_scale, f2_scale


def evaluate(sigma: List[List[int]], instance: dict, alpha: float = 0.5, *,
             f1_scale: float, f2_scale: float) -> dict:
    """
    Full evaluation of a solution.

    Args:
        sigma:    list of m machine sequences (job index lists)
        instance: problem instance dict
        alpha:    objective weighting (0=setup cost only, 1=tardiness only)
        f1_scale: normalisation scale for weighted tardiness (use estimate_scales())
        f2_scale: normalisation scale for setup cost (use estimate_scales())

    Returns:
        dict with: weighted_tardiness, setup_cost, composite, makespan,
                   tardiness_per_job, completion_times
    """
    validate_sigma(sigma, instance["n"])

    C = compute_completion_times(sigma, instance)
    T = compute_tardiness(C, instance)
    f1 = compute_weighted_tardiness(T, instance)
    f2 = compute_setup_cost(sigma, instance)
    f1_norm = f1 / f1_scale
    f2_norm = f2 / f2_scale
    F = alpha * f1_norm + (1 - alpha) * f2_norm

    return {
        "weighted_tardiness": f1,
        "setup_cost": f2,
        "composite": F,
        "makespan": compute_makespan(C),
        "tardiness_per_job": T,
        "completion_times": C,
        "f1_norm": f1_norm,
        "f2_norm": f2_norm,
    }
