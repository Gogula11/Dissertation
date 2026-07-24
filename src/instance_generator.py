"""
Synthetic instance generator for PMSP-SDSC.

A 'problem instance' is a dict with:
  n               : int — number of jobs
  m               : int — number of machines
  proc_times      : np.ndarray shape (n,) — processing time (hours) for each job
  due_dates       : np.ndarray shape (n,) — due date (hours from time zero) for each job
  weights         : np.ndarray shape (n,) — priority weights (1.0 for all)
  release         : np.ndarray shape (n,) — release times (0 for all)
  setup_cost      : np.ndarray shape (n, n) — asymmetric transition cost matrix S
  setup_time      : np.ndarray shape (n, n) — asymmetric transition time (hours)
  colour_ids      : np.ndarray shape (n,) — integer colour family index for each job
  colour_darkness : np.ndarray shape (n,) — darkness value [1, 7] for each job

Justification:
  - Machines run 168 hrs/week (24/7 continuous textile operation).
  - Avg processing time = (m * 168) / n, so total work fills exactly one week.
  - Due dates are spread proportionally within the week following SPT order.
  - Setup time averages 1/8 of processing time (vat cleaning ~1-2 hrs vs dye cycle ~8-16 hrs).
  - Setup cost is asymmetric: dark-to-light transitions are expensive (deep vat cleaning).
"""

import numpy as np


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

COLOUR_HEX = {
    0: "#FFFACD", 1: "#FFD700", 2: "#87CEEB",
    3: "#228B22", 4: "#DC143C", 5: "#000080", 6: "#1C1C1C",
}


def _build_cost_matrix(darkness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Asymmetric n×n cost matrix S.
    Dark-to-light transitions are expensive; light-to-dark is cheap.
    Diagonal is zero (same job = no transition cost).
    """
    n = len(darkness)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = darkness[i] - darkness[j]
            if diff > 0:
                darkness_term = diff * 10.0
            else:
                darkness_term = abs(diff) * 3.0
            noise = rng.uniform(0.0, 2.0)
            S[i][j] = darkness_term + noise
    return S


def generate_instance(
    n: int,
    m: int,
    seed: int = None,
    weekly_hours: float = 168.0,
) -> dict:
    """
    Generate one synthetic PMSP-SDSC instance grounded in weekly capacity.

    Args:
        n:             number of jobs
        m:             number of machines
        seed:          random seed for reproducibility
        weekly_hours:  operating hours per machine per week (default 168)
    """
    rng = np.random.default_rng(seed)

    # Processing time — total work fills one week across all machines
    avg_proc = (m * weekly_hours) / n
    proc_times = rng.uniform(avg_proc * 0.3, avg_proc * 1.7, size=n).astype(np.float32)

    # Colour assignment (uniform across 7 levels)
    colour_ids = rng.integers(0, N_COLOURS, size=n)
    colour_darkness = np.array([COLOUR_DARKNESS[cid] for cid in colour_ids], dtype=np.float32)

    # Setup cost — darkness asymmetry only
    setup_cost = _build_cost_matrix(colour_darkness, rng)

    # Setup time — average = 1/8 of avg proc time, proportional to setup cost
    norm_setup = setup_cost / max(setup_cost.mean(), 1e-8)
    setup_time = norm_setup * proc_times.mean() / 8.0
    np.fill_diagonal(setup_time, 0.0)

    # Due dates — proportional within the week, following SPT order
    order = np.argsort(proc_times)
    cum = np.cumsum(proc_times[order])
    due_dates = np.empty(n, dtype=np.float32)
    due_dates[order] = (cum / cum[-1]) * weekly_hours
    due_dates += rng.uniform(0, weekly_hours / n, size=n).astype(np.float32)

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
        "colour_darkness": colour_darkness,
    }


INSTANCE_CONFIGS = [
    {"n": 5,   "m": 1,  "label": "tiny_1m"},
    {"n": 10,  "m": 1,  "label": "small_1m"},
    {"n": 20,  "m": 1,  "label": "medium_1m"},
    {"n": 50,  "m": 1,  "label": "large_1m"},
    {"n": 100, "m": 1,  "label": "xlarge_1m"},
    {"n": 20,  "m": 3,  "label": "medium_3m"},
    {"n": 50,  "m": 5,  "label": "large_5m"},
    {"n": 500, "m": 10, "label": "xlarge_10m"},
]

INSTANCE_CONFIGS_SMALL = [c for c in INSTANCE_CONFIGS if c["n"] <= 50]