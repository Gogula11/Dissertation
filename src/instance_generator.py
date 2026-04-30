"""
Synthetic instance generator for PMSP-SDSC.

A 'problem instance' is a dict with:
  n          : int — number of jobs
  m          : int — number of machines
  proc_times : np.ndarray shape (n,) — processing time for each job
  due_dates  : np.ndarray shape (n,) — due date for each job
  weights    : np.ndarray shape (n,) — priority weights (1.0 for unweighted baseline)
  release    : np.ndarray shape (n,) — release times (0 for baseline)
  setup_cost : np.ndarray shape (n, n) — asymmetric transition cost matrix S
  setup_time : np.ndarray shape (n, n) — asymmetric transition time matrix
  colour_ids : np.ndarray shape (n,) — integer colour class for each job
"""

import numpy as np
from typing import Optional


# Colour darkness ranking — higher value = darker colour
COLOUR_DARKNESS = {
    0: 1,   # white
    1: 2,   # yellow
    2: 3,   # light blue
    3: 4,   # green
    4: 5,   # red
    5: 6,   # navy
    6: 7,   # black
}
N_COLOURS = len(COLOUR_DARKNESS)


def _build_cost_matrix(colour_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Asymmetric n×n cost matrix S.
    S[i][j] = cost of transitioning from job i's colour to job j's colour.
    Dark-to-light transitions are more expensive than light-to-dark.
    Diagonal is zero (same job repeated = no transition cost).
    """
    n = len(colour_ids)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            di = COLOUR_DARKNESS[colour_ids[i]]
            dj = COLOUR_DARKNESS[colour_ids[j]]
            darkness_diff = di - dj  # positive = going from dark to light (expensive)
            base_cost = max(0, darkness_diff) * 10.0
            noise = rng.uniform(0.0, 2.0)
            S[i][j] = base_cost + noise
    return S


def generate_instance(
    n: int,
    m: int,
    seed: Optional[int] = None,
    tightness: float = 1.5,
) -> dict:
    """
    Generate one synthetic PMSP-SDSC instance.

    Args:
        n:         number of jobs
        m:         number of machines
        seed:      random seed for reproducibility
        tightness: due-date tightness factor (lower = tighter deadlines)
    Returns:
        instance dict
    """
    rng = np.random.default_rng(seed)

    proc_times = rng.integers(5, 31, size=n).astype(np.float32)
    colour_ids = rng.integers(0, N_COLOURS, size=n)

    setup_cost = _build_cost_matrix(colour_ids, rng)
    setup_time = (setup_cost / 10.0) * rng.uniform(0.8, 1.2, size=(n, n))
    np.fill_diagonal(setup_time, 0.0)

    total_proc = proc_times.sum()
    avg_load_per_machine = total_proc / m
    due_dates = (proc_times / proc_times.sum()) * avg_load_per_machine * tightness * n
    due_dates += rng.uniform(0, avg_load_per_machine * 0.2, size=n)
    due_dates = due_dates.astype(np.float32)

    return {
        "n": n,
        "m": m,
        "proc_times": proc_times,
        "due_dates": due_dates,
        "weights": np.ones(n, dtype=np.float32),
        "release": np.zeros(n, dtype=np.float32),
        "setup_cost": setup_cost,
        "setup_time": setup_time,
        "colour_ids": colour_ids,
    }


# Pre-defined experiment configurations
INSTANCE_CONFIGS = [
    {"n": 10, "m": 2, "label": "small_2m"},
    {"n": 10, "m": 3, "label": "small_3m"},
    {"n": 20, "m": 2, "label": "medium_2m"},
    {"n": 20, "m": 3, "label": "medium_3m"},
    {"n": 50, "m": 2, "label": "large_2m"},
    {"n": 50, "m": 3, "label": "large_3m"},
]