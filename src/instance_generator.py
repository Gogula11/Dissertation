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
    ("reactive", "reactive"):0.1,
    ("reactive", "vat"):     0.6,
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
    customer_segments: Optional[bool] = None,
    segment_mix: tuple = (0.2, 0.5, 0.3),
    segment_weight_spread: float = 3.0,
    segment_tightness_spread: float = 2.25,
    colour_model: Optional[str] = None,
    n_colours: Optional[int] = None,
    colour_dist: Optional[str] = None,
    colour_clustering: Optional[float] = None,
    proc_colour_corr: Optional[float] = None,
    asymmetry_strength: Optional[float] = None,
    chemistry_penalty: Optional[bool] = None,
    setup_time_scale: float = 0.1,
) -> dict:
    """
    Generate one synthetic PMSP-SDSC instance.

    Args:
        n:         number of jobs
        m:         number of machines
        seed:      random seed for reproducibility
        tightness: due-date tightness factor (lower = tighter deadlines)
        profile:   "baseline" or "realistic". Overrides individual colour/customer params.
        customer_segments: enable premium/standard/economy segments
        segment_mix: (premium, standard, economy) proportions
        segment_weight_spread: ratio of highest to lowest segment weight
        segment_tightness_spread: ratio of loosest to tightest segment tightness
        colour_model: "categorical" (legacy 7-class) or "continuous" (COLOUR_FAMILIES)
        n_colours: number of colour families to use (1-12)
        colour_dist: "uniform" or "skewed" (some families dominate)
        colour_clustering: prob [0,1] consecutive jobs share same colour
        proc_colour_corr: strength [0,1] of colour→proc time link (0=none)
        asymmetry_strength: scales dark→light cost penalty
        chemistry_penalty: enable chemistry-based cross-type cost
        setup_time_scale: scales setup_time relative to setup_cost (default 0.1 matches legacy)

        Explicit params override profile defaults when both are set.
    Returns:
        instance dict
    """
    # Resolve profile defaults, then apply explicit overrides
    p = dict(PROFILES["baseline"])  # start with baseline defaults
    if profile is not None and profile in PROFILES:
        p.update(PROFILES[profile])
    for param_name in ("customer_segments", "colour_model", "n_colours",
                       "colour_dist", "colour_clustering", "proc_colour_corr",
                       "asymmetry_strength", "chemistry_penalty"):
        val = locals()[param_name]
        if val is not None:
            p[param_name] = val

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
        seg_weights = np.array([segment_weight_spread, 1.0, 1.0 / segment_weight_spread])
        seg_tightness = np.array([
            tightness / segment_tightness_spread,
            tightness,
            tightness * segment_tightness_spread,
        ])
        seg_idx = rng.choice(3, size=n, p=segment_mix)
        weights = seg_weights[seg_idx].astype(np.float32)
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


def validate_instance(inst: dict) -> list[str]:
    """Validate an instance dict. Returns list of error messages (empty = valid)."""
    errors = []
    n = inst["n"]
    m = inst["m"]

    for key in ("proc_times", "due_dates", "weights", "release"):
        arr = inst[key]
        if not isinstance(arr, np.ndarray):
            errors.append(f"{key}: not ndarray (got {type(arr)})")
            continue
        if arr.shape != (n,):
            errors.append(f"{key}: shape {arr.shape}, expected ({n},)")
        if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{key}: dtype {arr.dtype} not numeric")
        if np.any(arr < 0):
            errors.append(f"{key}: contains negative values")
    if np.any(inst["due_dates"] <= 0):
        errors.append("due_dates: all must be positive")

    for key in ("setup_cost", "setup_time"):
        arr = inst[key]
        if not isinstance(arr, np.ndarray):
            errors.append(f"{key}: not ndarray")
            continue
        if arr.shape != (n, n):
            errors.append(f"{key}: shape {arr.shape}, expected ({n},{n})")
        if np.any(arr < 0):
            errors.append(f"{key}: contains negative values")
        if not np.allclose(np.diag(arr), 0.0):
            errors.append(f"{key}: diagonal not all zero")

    cids = inst["colour_ids"]
    if not isinstance(cids, np.ndarray) or cids.shape != (n,):
        errors.append(f"colour_ids: shape mismatch")
    elif cids.min() < 0:
        errors.append("colour_ids: contains negative indices")
    else:
        unique_cids = np.unique(cids)
        if unique_cids.max() >= 100:
            errors.append(f"colour_ids: max value {unique_cids.max()} unreasonably large")

    darkness = inst["colour_darkness"]
    if not isinstance(darkness, np.ndarray) or darkness.shape != (n,):
        errors.append(f"colour_darkness: shape mismatch")
    if np.any(darkness < 0):
        errors.append("colour_darkness: contains negative values")

    chemistries = inst["chemistries"]
    if not isinstance(chemistries, list) or len(chemistries) != n:
        errors.append(f"chemistries: expected list of length {n}")
    valid_chems = ["direct", "reactive", "vat"]
    if any(c not in valid_chems and c != "" for c in chemistries):
        errors.append(f"chemistries: invalid entries (valid: {valid_chems} or empty string)")

    return errors


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
]