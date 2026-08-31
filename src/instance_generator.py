"""
Synthetic instance generator for PMSP-SDSC.

A 'problem instance' is a dict with:
  n               : int — number of jobs
  m               : int — number of machines
  proc_times      : np.ndarray shape (n,) — processing time (hours) per job, drawn per dye category
  due_dates       : np.ndarray shape (n,) — due date (hours from time zero) for each job
  weights         : np.ndarray shape (n,) — priority weights (1.0 for all)
  release         : np.ndarray shape (n,) — release times (0 for all)
  setup_cost      : np.ndarray shape (n, n) — asymmetric transition cost matrix S
  setup_time      : np.ndarray shape (n, n) — asymmetric transition time (hours)
  colour_ids      : np.ndarray shape (n,) — integer colour index into COLOUR_NAMES
  colour_darkness : np.ndarray shape (n,) — darkness value [1, 7] for each job
  dye_category    : np.ndarray shape (n,) — dye category index [0, 3]

Justification:
  - Processing times drawn per dye category: reactive/vat (cotton) U(6, 10),
    disperse (polyester) U(4, 6), acid (nylon/wool) U(5, 8) — reflects
    full dye-cycle time per fibre type (LinkedIn mill breakdown; Cotton Inc.).
  - Colour and dye category are sampled independently — a colour (e.g., navy)
    can appear on any fibre type, reflecting real dyehouse flexibility.
  - 10 configurations from 3 to 60 jobs per batch (literature: ~15 jobs/machine
    per week, Uttamapinant).
  - Due dates set at 1.2x SPT completion times (20% slack, realistic buffer).
  - Setup time = norm_setup × SETUP_TIME_MEAN (1.5h), decoupled from proc time.
    Cleaning averages 1-2 hours (Wang et al. 2026: quick 1h, thorough 2h).
  - Setup cost: same-category transitions use darkness asymmetry; cross-category
    transitions incur a flat penalty.
  - 20 colours across 4 dye categories (reactive, disperse, vat, acid).
"""

import numpy as np

PROC_TIME_RANGE = {
    0: (6.0, 10.0),   # reactive → cotton: full dye cycle (load→bleach→dye→wash→unload)
    1: (4.0, 6.0),    # disperse → polyester: shorter HT/HP cycle
    2: (6.0, 10.0),   # vat → cotton: similar to reactive
    3: (5.0, 8.0),    # acid → nylon/wool: moderate cycle
}
WEEKLY_HOURS = 168.0
CROSS_CATEGORY_PENALTY = 50.0
SETUP_TIME_MEAN = 1.5

DYE_CATEGORY_NAMES = ["reactive", "disperse", "vat", "acid"]

N_DYE_CATEGORIES = len(DYE_CATEGORY_NAMES)

# Global colour pool — colours are independent of dye categories
GLOBAL_COLOUR_NAMES = {
    0: "white", 1: "cream", 2: "yellow", 3: "navy", 4: "royal blue",
    5: "light blue", 6: "sky", 7: "red", 8: "black",
    9: "green", 10: "olive", 11: "teal",
    12: "pink", 13: "orange", 14: "purple", 15: "brown",
    16: "magenta", 17: "burgundy", 18: "beige", 19: "rust",
}

GLOBAL_COLOUR_DARKNESS = {
    0: 1, 1: 2, 2: 3, 3: 6, 4: 7,
    5: 2, 6: 3, 7: 5, 8: 7,
    9: 3, 10: 4, 11: 5,
    12: 1, 13: 2, 14: 3, 15: 4,
    16: 5, 17: 6, 18: 7, 19: 4,
}

N_COLOURS = len(GLOBAL_COLOUR_NAMES)

_hex_palette = [
    "#FFFFFF", "#FFF5E6", "#FFD700", "#000080", "#4169E1",
    "#87CEEB", "#4682B4", "#DC143C", "#1C1C1C",
    "#228B22", "#556B2F", "#008080",
    "#FF69B4", "#FF8C00", "#800080", "#A0522D", "#FF00FF",
    "#800020", "#DEB887", "#CD853F",
]
GLOBAL_COLOUR_HEX = {i: _hex_palette[i] if i < len(_hex_palette) else "#999999" for i in range(N_COLOURS)}


def _build_cost_matrix(
    colour_darkness: np.ndarray,
    dye_category: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(colour_darkness)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dye_category[i] == dye_category[j]:
                diff = colour_darkness[i] - colour_darkness[j]
                if diff > 0:
                    darkness_term = diff * 10.0
                else:
                    darkness_term = abs(diff) * 3.0
            else:
                darkness_term = CROSS_CATEGORY_PENALTY
            noise = rng.uniform(0.0, 2.0)
            S[i][j] = darkness_term + noise
    return S


def generate_instance(
    jobs_per_machine: int = 10,
    m: int = 1,
    seed: int = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n = jobs_per_machine * m

    dye_category = rng.integers(0, N_DYE_CATEGORIES, size=n)
    colour_ids = rng.integers(0, N_COLOURS, size=n)
    colour_darkness = np.array([GLOBAL_COLOUR_DARKNESS[cid] for cid in colour_ids], dtype=np.float32)

    proc_times = np.zeros(n, dtype=np.float32)
    for cat_id in range(N_DYE_CATEGORIES):
        mask = dye_category == cat_id
        if mask.any():
            lo, hi = PROC_TIME_RANGE[cat_id]
            proc_times[mask] = rng.uniform(lo, hi, size=mask.sum()).astype(np.float32)

    setup_cost = _build_cost_matrix(colour_darkness, dye_category, rng)

    norm_setup = setup_cost / max(setup_cost.mean(), 1e-8)
    setup_time = norm_setup * SETUP_TIME_MEAN
    np.fill_diagonal(setup_time, 0.0)

    # Due dates: SPT completion time * 1.2 (20% slack)
    spt_order = np.argsort(proc_times)
    C_spt = np.zeros(n, dtype=np.float32)
    for machine_idx in range(m):
        t = 0.0
        for job_idx in range(machine_idx, n, m):
            job = spt_order[job_idx]
            t += proc_times[job]
            C_spt[job] = t
    due_dates = (C_spt * 1.2).astype(np.float32)

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
        "dye_category": dye_category,
    }


INSTANCE_CONFIGS = [
    # Low utilisation (~15-20%)
    {"jobs_per_machine": 3,  "m": 1,  "label": "j3_m1"},   # 15% — 25h of 168h
    {"jobs_per_machine": 4,  "m": 3,  "label": "j12_m3"},  # 20% — 33h of 168h per machine
    # Moderate utilisation (~59%)
    {"jobs_per_machine": 12, "m": 1,  "label": "j12_m1"},  # 59% — 100h of 168h
    {"jobs_per_machine": 12, "m": 3,  "label": "j36_m3"},  # 59% — 100h of 168h per machine
    {"jobs_per_machine": 12, "m": 5,  "label": "j60_m5"},  # 59% — 100h of 168h per machine
    # High utilisation (~94%)
    {"jobs_per_machine": 19, "m": 1,  "label": "j19_m1"},  # 94% — 158h of 168h
    {"jobs_per_machine": 19, "m": 3,  "label": "j57_m3"},  # 94% — 158h of 168h per machine
    {"jobs_per_machine": 19, "m": 5,  "label": "j95_m5"},  # 94% — 158h of 168h per machine
    # Overloaded (104%)
    {"jobs_per_machine": 21, "m": 1,  "label": "j21_m1"},  # 104% — 174h of 168h
    {"jobs_per_machine": 21, "m": 3,  "label": "j63_m3"},  # 104% — 174h of 168h per machine
]

INSTANCE_CONFIGS_SMALL = [c for c in INSTANCE_CONFIGS if c["jobs_per_machine"] * c["m"] <= 60]