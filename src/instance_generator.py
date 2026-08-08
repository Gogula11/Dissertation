"""
Synthetic instance generator for PMSP-SDSC.

A 'problem instance' is a dict with:
  n               : int — number of jobs
  m               : int — number of machines
  proc_times      : np.ndarray shape (n,) — processing time (hours) for each job (all 8.0)
  due_dates       : np.ndarray shape (n,) — due date (hours from time zero) for each job
  weights         : np.ndarray shape (n,) — priority weights (1.0 for all)
  release         : np.ndarray shape (n,) — release times (0 for all)
  setup_cost      : np.ndarray shape (n, n) — asymmetric transition cost matrix S
  setup_time      : np.ndarray shape (n, n) — asymmetric transition time (hours)
  colour_ids      : np.ndarray shape (n,) — integer colour index into COLOUR_NAMES
  colour_darkness : np.ndarray shape (n,) — darkness value [1, 7] for each job
  dye_category    : np.ndarray shape (n,) — dye category index [0, 3]

Justification:
  - Each job takes exactly 8 hours (1 working day).
  - n = jobs_per_machine * m, so total work fits within one week (168 hrs).
  - Due dates based on SPT heuristic completion times.
  - Setup time averages 1/8 of processing time (vat cleaning 1 hr vs dye 8 hrs).
  - Setup cost: same-category transitions use darkness asymmetry; cross-category
    transitions incur a flat penalty.
  - 20 colours across 4 dye categories (reactive, disperse, vat, acid).
"""

import numpy as np

PROC_TIME = 8.0
WEEKLY_HOURS = 168.0
JOBS_PER_MACHINE = 21
CROSS_CATEGORY_PENALTY = 50.0

DYE_CATEGORIES = {
    0: {
        "name": "reactive",
        "colours": ["white", "cream", "yellow", "navy", "royal blue"],
        "darkness": [1, 2, 3, 6, 7],
    },
    1: {
        "name": "disperse",
        "colours": ["light blue", "sky", "red", "black"],
        "darkness": [2, 3, 5, 7],
    },
    2: {
        "name": "vat",
        "colours": ["green", "olive", "teal"],
        "darkness": [3, 4, 5],
    },
    3: {
        "name": "acid",
        "colours": ["pink", "orange", "purple", "brown", "magenta", "burgundy", "beige", "rust"],
        "darkness": [1, 2, 3, 4, 5, 6, 7, 4],
    },
}

N_DYE_CATEGORIES = len(DYE_CATEGORIES)

GLOBAL_COLOUR_NAMES = {}
GLOBAL_COLOUR_HEX = {}
GLOBAL_COLOUR_DARKNESS = {}
GLOBAL_COLOUR_CATEGORIES = {}
_cid = 0
for cat_id, cat in DYE_CATEGORIES.items():
    for name, d in zip(cat["colours"], cat["darkness"]):
        GLOBAL_COLOUR_NAMES[_cid] = name
        GLOBAL_COLOUR_DARKNESS[_cid] = d
        GLOBAL_COLOUR_CATEGORIES[_cid] = cat_id
        _cid += 1

N_COLOURS = len(GLOBAL_COLOUR_NAMES)

_hex_palette = [
    "#FFFFFF", "#FFF5E6", "#FFD700", "#000080", "#4169E1",  # reactive: white, cream, yellow, navy, royal blue
    "#87CEEB", "#4682B4", "#DC143C", "#1C1C1C",             # disperse: light blue, sky, red, black
    "#228B22", "#556B2F", "#008080",                          # vat: green, olive, teal
    "#FF69B4", "#FF8C00", "#800080", "#A0522D", "#FF00FF",    # acid: pink, orange, purple, brown, magenta
    "#800020", "#DEB887", "#CD853F",                          # acid: burgundy, beige, rust
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

    proc_times = np.full(n, PROC_TIME, dtype=np.float32)

    colour_ids = rng.integers(0, N_COLOURS, size=n)
    colour_darkness = np.array([GLOBAL_COLOUR_DARKNESS[cid] for cid in colour_ids], dtype=np.float32)
    dye_category = np.array([GLOBAL_COLOUR_CATEGORIES[cid] for cid in colour_ids], dtype=np.int32)

    setup_cost = _build_cost_matrix(colour_darkness, dye_category, rng)

    norm_setup = setup_cost / max(setup_cost.mean(), 1e-8)
    setup_time = norm_setup * PROC_TIME / 8.0
    np.fill_diagonal(setup_time, 0.0)

    due_dates = np.full(n, WEEKLY_HOURS, dtype=np.float32)

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
    # Easy (underloaded)
    {"jobs_per_machine": 10, "m": 1,  "label": "j10_m1"},
    {"jobs_per_machine": 7,  "m": 3,  "label": "j7_m3"},
    {"jobs_per_machine": 10, "m": 5,  "label": "j10_m5"},
    # Tight (near capacity)
    {"jobs_per_machine": 20, "m": 1,  "label": "j20_m1"},
    {"jobs_per_machine": 14, "m": 3,  "label": "j14_m3"},
    {"jobs_per_machine": 20, "m": 5,  "label": "j20_m5"},
    # Constrained (at capacity)
    {"jobs_per_machine": 18, "m": 1,  "label": "j18_m1"},
    {"jobs_per_machine": 21, "m": 3,  "label": "j21_m3"},
    {"jobs_per_machine": 18, "m": 5,  "label": "j18_m5"},
    # Stress test (overloaded, for limit-behaviour analysis)
    {"jobs_per_machine": 30, "m": 5,  "label": "j30_m5"},
]

INSTANCE_CONFIGS_SMALL = [c for c in INSTANCE_CONFIGS if c["jobs_per_machine"] * c["m"] <= 50]