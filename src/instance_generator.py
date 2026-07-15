"""
Synthetic instance generator for PMSP-SDSC.

A 'problem instance' is a dict with:
  n               : int — number of jobs
  m               : int — number of machines
  proc_times      : np.ndarray shape (n,) — processing time for each job
  due_dates       : np.ndarray shape (n,) — due date for each job
  weights         : np.ndarray shape (n,) — priority weights (1.0 for unweighted baseline)
  release         : np.ndarray shape (n,) — release times (0 for baseline)
  setup_cost      : np.ndarray shape (n, n) — asymmetric transition cost matrix S
  setup_time      : np.ndarray shape (n, n) — asymmetric transition time matrix
  colour_ids      : np.ndarray shape (n,) — integer colour family index for each job
  colour_darkness : np.ndarray shape (n,) — continuous darkness value [1, 10]
  chemistries     : list[str] — dye chemistry class per job
"""

import numpy as np
from typing import Optional


# Colour darkness ranking — higher value = darker colour (legacy, categorical mode)
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

# Colour families — continuous model
COLOUR_FAMILIES = {
    "white":    {"base": 1.0, "shade_std": 0.3,  "chemistry": "direct"},
    "pink":     {"base": 1.6, "shade_std": 0.4,  "chemistry": "reactive"},
    "yellow":   {"base": 2.0, "shade_std": 0.5,  "chemistry": "reactive"},
    "orange":   {"base": 3.0, "shade_std": 0.5,  "chemistry": "reactive"},
    "green":    {"base": 4.0, "shade_std": 0.6,  "chemistry": "reactive"},
    "blue":     {"base": 4.0, "shade_std": 0.7,  "chemistry": "reactive"},
    "brown":    {"base": 4.5, "shade_std": 0.5,  "chemistry": "reactive"},
    "red":      {"base": 5.0, "shade_std": 0.8,  "chemistry": "reactive"},
    "purple":   {"base": 5.5, "shade_std": 0.5,  "chemistry": "reactive"},
    "navy":     {"base": 6.0, "shade_std": 0.4,  "chemistry": "vat"},
    "grey":     {"base": 6.5, "shade_std": 0.4,  "chemistry": "direct"},
    "black":    {"base": 7.0, "shade_std": 0.2,  "chemistry": "vat"},
}
COLOUR_FAMILIES_KEYS = list(COLOUR_FAMILIES.keys())

# Chemistry speed factors for proc-colour correlation
CHEM_FACTOR = {"direct": 1.0, "reactive": 1.3, "vat": 1.5}

# Chemistry incompatibility cost — cross-type transitions cost extra
CHEM_COMPAT = {
    ("direct", "direct"):    0.0,
    ("direct", "reactive"):  0.3,
    ("direct", "vat"):       0.5,
    ("reactive", "direct"):  0.3,
    ("reactive", "reactive"):0.1,
    ("reactive", "vat"):     0.6,
    ("vat", "direct"):       0.5,
    ("vat", "reactive"):     0.6,
    ("vat", "vat"):          0.1,
}


def _build_cost_matrix(
    darkness: np.ndarray,
    rng: np.random.Generator,
    asymmetry_strength: float = 1.0,
    chemistries: Optional[list[str]] = None,
    chemistry_penalty: bool = False,
) -> np.ndarray:
    """
    Asymmetric n×n cost matrix S.

    Three-component model:
      1. Darkness penalty — nonlinear, asymmetric (dark→light expensive)
      2. Chemistry incompatibility — cross-type transitions add cost
      3. Gamma noise — rare expensive cleanings

    Diagonal is zero (same job = no transition cost).
    """
    n = len(darkness)
    S = np.zeros((n, n), dtype=np.float32)
    has_chem = chemistries is not None and all(c != "" for c in chemistries)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = darkness[i] - darkness[j]
            if diff > 0:  # dark → light (expensive cleaning)
                darkness_term = max(0, diff) ** 1.5
            else:  # light → dark (small cleaning base)
                darkness_term = abs(diff) ** 0.5 * 0.3
            term = asymmetry_strength * darkness_term

            if chemistry_penalty and has_chem:
                term += CHEM_COMPAT.get((chemistries[i], chemistries[j]), 0.0) * 10.0

            noise = rng.gamma(shape=2, scale=1.0)
            S[i][j] = term + noise
    return S


def _sample_colours(
    n: int,
    n_colours: int,
    colour_dist: str,
    colour_clustering: float,
    rng: np.random.Generator,
):
    """Generate colour IDs, continuous darkness, and chemistries."""
    family_keys = COLOUR_FAMILIES_KEYS[:n_colours]

    if colour_dist == "skewed":
        # linearly decreasing probs: lighter families more common
        probs = np.linspace(1.0, 0.3, n_colours)
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_colours) / n_colours

    colour_ids = np.empty(n, dtype=np.intp)
    colour_darkness = np.empty(n, dtype=np.float32)
    chemistries = []

    for i in range(n):
        if i > 0 and rng.uniform() < colour_clustering:
            colour_ids[i] = colour_ids[i - 1]
        else:
            colour_ids[i] = int(rng.choice(n_colours, p=probs))

        info = COLOUR_FAMILIES[family_keys[colour_ids[i]]]
        darkness = info["base"] + rng.normal(0, info["shade_std"])
        colour_darkness[i] = np.clip(darkness, 1.0, 10.0)
        chemistries.append(info["chemistry"])

    return colour_ids, colour_darkness, chemistries


# Profile presets — set sane defaults for all params at once
PROFILES = {
    "baseline": {
        "colour_model": "categorical",
        "n_colours": 7,
        "colour_dist": "uniform",
        "colour_clustering": 0.0,
        "asymmetry_strength": 1.0,
        "chemistry_penalty": False,
        "proc_colour_corr": 0.0,
        "customer_segments": False,
    },
    "realistic": {
        "colour_model": "continuous",
        "n_colours": 12,
        "colour_dist": "skewed",
        "colour_clustering": 0.3,
        "asymmetry_strength": 2.0,
        "chemistry_penalty": True,
        "proc_colour_corr": 0.8,
        "customer_segments": True,
    },
}


def generate_instance(
    n: int,
    m: int,
    seed: Optional[int] = None,
    tightness: float = 1.5,
    profile: Optional[str] = None,
    proc_colour_corr: Optional[float] = None,
    setup_time_scale: float = 0.1,
) -> dict:
    """
    Generate one synthetic PMSP-SDSC instance.
    """
    p = dict(PROFILES["baseline"])
    if profile is not None and profile in PROFILES:
        p.update(PROFILES[profile])
    if proc_colour_corr is not None:
        p["proc_colour_corr"] = proc_colour_corr

    customer_segments = p["customer_segments"]
    colour_model = p["colour_model"]
    n_colours = p["n_colours"]
    colour_dist = p["colour_dist"]
    colour_clustering = p["colour_clustering"]
    proc_colour_corr = p["proc_colour_corr"]
    asymmetry_strength = p["asymmetry_strength"]
    chemistry_penalty = p["chemistry_penalty"]
    rng = np.random.default_rng(seed)

    if colour_model == "continuous":
        colour_ids, colour_darkness, chemistries = _sample_colours(
            n, n_colours, colour_dist, colour_clustering, rng,
        )
        darkness_vals = colour_darkness
    else:
        colour_ids = rng.integers(0, N_COLOURS, size=n)
        colour_darkness = np.array([COLOUR_DARKNESS[cid] for cid in colour_ids], dtype=np.float32)
        chemistries = [""] * n
        darkness_vals = colour_darkness

    base_proc = rng.integers(5, 31, size=n).astype(np.float32)
    if proc_colour_corr > 0:
        darkness_factor = 1.0 + colour_darkness * 0.1 * proc_colour_corr
        if colour_model == "continuous":
            chem_factor = np.array([CHEM_FACTOR[c] for c in chemistries], dtype=np.float32)
        else:
            chem_factor = 1.0
        proc_times = base_proc * darkness_factor * chem_factor
    else:
        proc_times = base_proc

    setup_cost = _build_cost_matrix(
        darkness_vals, rng,
        asymmetry_strength=asymmetry_strength,
        chemistries=chemistries,
        chemistry_penalty=chemistry_penalty,
    )
    setup_time = setup_cost * setup_time_scale * rng.uniform(0.8, 1.2, size=(n, n))
    np.fill_diagonal(setup_time, 0.0)

    if customer_segments:
        seg_weights = np.array([3.0, 1.0, 1.0/3.0], dtype=np.float32)
        seg_tightness = np.array([tightness/2.25, tightness, tightness*2.25])
        seg_idx = rng.choice(3, size=n, p=[0.2, 0.5, 0.3])
        weights = seg_weights[seg_idx]
        per_job_tightness = seg_tightness[seg_idx]
    else:
        weights = np.ones(n, dtype=np.float32)
        per_job_tightness = np.full(n, tightness)

    total_proc = proc_times.sum()
    avg_load_per_machine = total_proc / m
    due_dates = (proc_times / proc_times.sum()) * avg_load_per_machine * per_job_tightness * n
    due_dates += rng.uniform(0, avg_load_per_machine * 0.2, size=n)
    due_dates = due_dates.astype(np.float32)

    return {
        "n": n,
        "m": m,
        "proc_times": proc_times,
        "due_dates": due_dates,
        "weights": weights,
        "release": np.zeros(n, dtype=np.float32),
        "setup_cost": setup_cost,
        "setup_time": setup_time,
        "colour_ids": colour_ids,
        "colour_darkness": colour_darkness,
        "chemistries": chemistries,
    }


# Pre-defined experiment configurations
INSTANCE_CONFIGS = [
    {"n": 5,   "m": 2,  "label": "tiny_2m"},
    {"n": 10,  "m": 2,  "label": "small_2m"},
    {"n": 10,  "m": 3,  "label": "small_3m"},
    {"n": 20,  "m": 2,  "label": "medium_2m"},
    {"n": 20,  "m": 3,  "label": "medium_3m"},
    {"n": 30,  "m": 3,  "label": "medium_30_3m"},
    {"n": 50,  "m": 2,  "label": "large_2m"},
    {"n": 50,  "m": 3,  "label": "large_3m"},
    {"n": 50,  "m": 5,  "label": "large_5m"},
    {"n": 100, "m": 5,  "label": "xlarge_5m"},
    {"n": 100, "m": 10, "label": "xlarge_10m"},
]