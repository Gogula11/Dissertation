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


def estimate_scales(instance: dict) -> tuple:
    """Estimate normalisation scales from instance data.

    Returns (f1_scale, f2_scale) where:
      f1_scale: upper bound on weighted tardiness
      f2_scale: upper bound on setup cost
    """
    n = instance["n"]
    max_C = float(instance["proc_times"].sum() + instance["setup_time"].max() * n)
    f1_scale = float(np.dot(instance["weights"], np.maximum(0, max_C - instance["due_dates"])))
    f2_scale = float(np.max(instance["setup_cost"]) * max(n - 1, 1))
    return max(f1_scale, 1.0), max(f2_scale, 1.0)


def evaluate(sigma: List[List[int]], instance: dict, alpha: float = 0.5,
             f1_scale: float = None, f2_scale: float = None) -> dict:
    """
    Full evaluation of a solution.

    Args:
        sigma:    list of m machine sequences (job index lists)
        instance: problem instance dict
        alpha:    objective weighting (0=setup cost only, 1=tardiness only)
        f1_scale: normalisation scale for weighted tardiness (auto if None)
        f2_scale: normalisation scale for setup cost (auto if None)

    Returns:
        dict with: weighted_tardiness, setup_cost, composite, makespan,
                   tardiness_per_job, completion_times
    """
    validate_sigma(sigma, instance["n"])

    C = compute_completion_times(sigma, instance)
    T = compute_tardiness(C, instance)
    f1 = compute_weighted_tardiness(T, instance)
    f2 = compute_setup_cost(sigma, instance)
    if f1_scale is None or f2_scale is None:
        f1_scale, f2_scale = estimate_scales(instance)
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
