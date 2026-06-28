# Instance Generator — Improvement Plan

Current source: `src/instance_generator.py`
Evaluator: `src/evaluator.py`
Chapter reference: `notes/chapter_03_system_design.md`

---

## Current Weaknesses

| Component | Current | Problem |
|-----------|---------|---------|
| Colour model | 7 discrete ordinal classes (white→black) | Real colours aren't linearly orderable; no within-class variance |
| Cost matrix | `max(0, d_i - d_j) × 10 + uniform(0,2)` | No chemical basis, no machine dependence, no family effects |
| Processing times | Uniform(5,31), independent of colour | Dark dyes take longer to fix/set in reality |
| Setup time | `setup_cost / 10 × uniform(0.8, 1.2)` | Tied 1:1 to cost — they should be independent |
| Due dates | Proportional to proc time only | No order priority, no customer segments, no seasonality |
| Machine model | All machines identical, unrestricted | Real mills have machine-colour compatibility |
| Release times | All zero | No material arrival / order release realism |
| Configs | Only n={10,20,50}, m={2,3} | Too few; missing large industrial scale |

---

## Final Parameter Specification

Everything the generator should expose. Organized by category.

### A. Structural Parameters (instance shape)

| Param | Values | Default | Why |
|-------|--------|---------|-----|
| `n` | 5, 10, 20, 30, 50, 100, 200 | — | Job count sweep |
| `m` | 2, 3, 5, 10 | — | Machine count sweep |
| `seed` | int | — | Reproducibility |

### B. Colour & Setup Parameters

| Param | Values | Default | Why |
|-------|--------|---------|-----|
| `colour_model` | `"categorical"` / `"continuous"` | `"categorical"` | Categorical = 7 discrete classes (baseline). Continuous = per-job float darkness + shade variance (Phase 1) |
| `n_colours` | 3, 5, 7, 10 | 7 | Number of colour families (3 = light/medium/dark; 7 = full) |
| `colour_dist` | `"uniform"` / `"skewed"` | `"uniform"` | Skewed = some colours dominate (e.g., 40% white). Tests robustness to distribution shift |
| `colour_clustering` | float [0, 1] | 0 | Probability that consecutive jobs share a colour. 0 = independent, 1 = bursty arrivals |
| `asymmetry_strength` | float [0, 3] | 1.0 | Scales dark→light cost. 0 = symmetric, 3 = extreme asymmetry |
| `chemistry_penalty` | bool | False | Add chemistry-based cost between families (Phase 2) |
| `setup_cost_scale` | float (0, ∞) | 10.0 | Global multiplier on all setup costs. Controls setup vs. tardiness trade-off |
| `setup_time_scale` | float (0, ∞) | 0.1 | Separate multiplier for setup time. Decouples time from cost (Phase 4) |
| `min_cleaning_cost` | float ≥ 0 | 0.5 | Minimum cost even for same-colour transitions. No free lunch |

### C. Processing Time Parameters

| Param | Values | Default | Why |
|-------|--------|---------|-----|
| `proc_dist` | `"uniform"` / `"exponential"` / `"bimodal"` | `"uniform"` | Bimodal = some jobs are "long runs", others are "short batches" |
| `proc_min` | int | 5 | Min processing time |
| `proc_max` | int | 31 | Max processing time |
| `proc_colour_corr` | float [0, 1] | 0 | Strength of colour → processing time link. 0 = none, 1 = dark jobs take up to 2× longer (Phase 3) |
| `proc_outlier_prob` | float [0, 0.1] | 0 | Probability a job is a "long outlier" (5-10× normal). Creates bottleneck jobs |

### D. Due-Date & Customer Parameters

| Param | Values | Default | Why |
|-------|--------|---------|-----|
| `due_date_type` | `"distinct"` / `"common"` / `"window"` | `"distinct"` | Distinct = per-job due dates (current). Common = all same. Window = [earliest, latest] interval |
| `tightness` | float [0.5, 3.0] | 1.5 | Due-date tightness factor. Lower = tighter deadlines |
| `tightness_sweep` | list | `[0.8, 1.0, 1.5, 2.0]` | Sweep as separate experiment dimension |
| `customer_segments` | bool | False | Enable premium/standard/economy (Phase 5) |
| `segment_mix` | (float, float, float) | (0.2, 0.5, 0.3) | % premium / standard / economy |
| `segment_weight_spread` | float ≥ 1 | 3.0 | Ratio of highest to lowest segment weight |
| `segment_tightness_spread` | float ≥ 1 | 2.25 | Ratio of loosest to tightest segment tightness |

### E. Machine Parameters

| Param | Values | Default | Why |
|-------|--------|---------|-----|
| `machine_type` | `"identical"` / `"uniform"` / `"unrelated"` | `"identical"` | Identical = all machines same (current). Uniform = speed factor. Unrelated = per-job-per-machine times |
| `machine_speed_spread` | float ≥ 1 | 1.5 | Ratio of fastest to slowest machine speed (uniform type only) |
| `machine_eligibility` | bool | False | Enable machine-colour restrictions (Phase 6) |
| `machine_dedication` | `"none"` / `"chemistry"` / `"darkness"` / `"both"` | `"none"` | How machines are restricted |

### F. Release & Arrival Parameters

| Param | Values | Default | Why |
|-------|--------|---------|-----|
| `release_fraction` | float [0, 1] | 0 | Fraction of jobs with non-zero release time |
| `release_spread` | float ≥ 0 | 0.5 × makespan | How spread out releases are. 0 = all at same time |
| `release_pattern` | `"uniform"` / `"batched"` | `"uniform"` | Batched = cohorts arrive together (e.g., weekly orders) |

### G. Objective Parameters

| Param | Values | Default | Why |
|-------|--------|---------|-----|
| `objective` | `"composite"` / `"tardiness"` / `"setup_cost"` / `"makespan"` | `"composite"` | Which objective to minimise |
| `alpha` | float [0, 1] | 0.5 | Composite trade-off weight |
| `alpha_sweep` | list | `[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` | Finer grid for sensitivity analysis |
| `earliness_penalty` | bool | False | Penalise finishing *before* due date (JIT) |

### H. Profile Metaparameter

Controls all above in presets. The main knob experimenters touch.

| Param | Values | Default |
|-------|--------|---------|
| `profile` | `"baseline"` / `"enhanced"` / `"realistic"` | `"baseline"` |

---

## Defaults Per Profile

| Param | Baseline | Enhanced | Realistic |
|-------|----------|----------|-----------|
| colour_model | categorical | continuous | continuous |
| n_colours | 7 | 7 | 10 |
| colour_dist | uniform | skewed | skewed |
| colour_clustering | 0 | 0.2 | 0.3 |
| asymmetry_strength | 1.0 | 1.5 | 2.0 |
| chemistry_penalty | false | true | true |
| proc_dist | uniform | uniform | exponential |
| proc_colour_corr | 0 | 0.5 | 0.8 |
| due_date_type | distinct | distinct | window |
| tightness | 1.5 | 1.5 | 1.2 |
| customer_segments | false | true | true |
| machine_type | identical | uniform | unrelated |
| machine_eligibility | false | false | true |
| release_fraction | 0 | 0.2 | 0.4 |
| objective | composite | composite | composite |
| alpha_sweep | [0.3, 0.5, 0.7] | [0.1, 0.5, 0.9] | [0.0, 0.3, 0.5, 0.7, 1.0] |

---

## Experiment Design With These Parameters

| Experiment | Sweep these | Hold fixed |
|------------|-------------|------------|
| Main comparison (4 methods) | profile ∈ {baseline, enhanced} | seed, alpha=0.5, tightness=1.5 |
| Tightness sensitivity | tightness ∈ [0.8, 1.0, 1.5, 2.0] | profile=baseline, alpha=0.5 |
| Alpha sensitivity | alpha ∈ [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] | profile=baseline, tightness=1.5 |
| Colour model effect | colour_model, n_colours | profile=baseline, alpha=0.5 |
| Customer segment effect | segment_mix, segment_weight_spread | profile=enhanced, alpha=0.5 |
| Machine heterogeneity | machine_type, machine_speed_spread | profile=enhanced, alpha=0.5 |

Total new code params: ~25. ~15 controlled by `profile` metaparameter. Experimenter sees **6 knobs**: `(n, m, seed, profile, alpha, tightness)`.

---

## Phase 1: Colour Model — Continuous + Sub-classes

**Target:** `COLOUR_DARKNESS` dict → `COLOUR_FAMILIES` dict

```python
COLOUR_FAMILIES = {
    "white":    {"base": 1,  "shade_std": 0.3,  "chemistry": "direct"},
    "yellow":   {"base": 2,  "shade_std": 0.5,  "chemistry": "reactive"},
    "blue":     {"base": 4,  "shade_std": 0.7,  "chemistry": "reactive"},
    "green":    {"base": 4,  "shade_std": 0.6,  "chemistry": "reactive"},
    "red":      {"base": 5,  "shade_std": 0.8,  "chemistry": "reactive"},
    "navy":     {"base": 6,  "shade_std": 0.4,  "chemistry": "vat"},
    "black":    {"base": 7,  "shade_std": 0.2,  "chemistry": "vat"},
}
```

Each job gets:
- `colour_family` (categorical: one of 7)
- `colour_darkness` (continuous float: `base + normal(0, shade_std)`, clipped [1,10])
- `chemistry` (string: direct/reactive/vat)

**Why:** Continuous darkness captures shade variation within "red" (scarlet vs burgundy). Chemistry category enables cross-type cost penalties (reactive→vat more expensive than reactive→reactive even at same darkness).

**Mapping to parameters:** Controlled by `colour_model` (categorical→uses COLOUR_DARKNESS as before; continuous→uses COLOUR_FAMILIES), `n_colours`, `colour_dist`, `colour_clustering`.

---

## Phase 2: Cost Matrix — Three-Component Model

**Target:** `_build_cost_matrix()`

```
S[i][j] =  asymmetry_strength * darkness_penalty +
           chemistry_penalty_bool * chemistry_penalty  +
           noise_term
```

### Component 1 — Darkness penalty (nonlinear asymmetry)

```python
diff = d_i - d_j
# dark->light: quadratically expensive
darkness_penalty = max(0, diff) ** 1.5 * base_rate
# light->dark: small base (cleaning never free)
if diff < 0:
    darkness_penalty = abs(diff) ** 0.5 * base_rate * 0.3
```

### Component 2 — Chemistry incompatibility

```python
CHEM_COMPAT = {
    ("direct", "direct"):    0.0,
    ("direct", "reactive"):  0.3,
    ("direct", "vat"):       0.5,
    ("reactive", "reactive"):0.1,
    ("reactive", "vat"):     0.6,
    ("vat", "vat"):          0.1,
}
chemistry_penalty = CHEM_COMPAT[chem_i, chem_j] * max_cost
```

### Component 3 — Stochastic noise

```python
noise = gamma(shape=2, scale=1.0)  # right-skewed: rare expensive cleanings
```

**Global scaling:** `setup_cost_scale` multiplies the final cost matrix.

---

## Phase 3: Processing Times — Colour-Correlated

**Target:** `proc_times`

```python
base_proc = rng.uniform(proc_min, proc_max, size=n)
if proc_colour_corr > 0:
    darkness_factor = 1.0 + colour_darkness * 0.1 * proc_colour_corr
    chem_factor = {"direct": 1.0, "reactive": 1.3, "vat": 1.5}
    proc_times = base_proc * darkness_factor * [chem_factor[c] for c in chemistries]
```

**Why:** Creates trade-off — grouping dark jobs saves setup cost but increases tardiness risk. The hybrid's adaptive agent can learn this balance.

**Mapping to parameters:** `proc_colour_corr` (0 = off, 1 = full effect), `proc_dist`, `proc_outlier_prob`.

---

## Phase 4: Decouple Setup Time from Setup Cost

**Target:** Remove `setup_time = setup_cost / 10`

```python
# setup_cost: economic (dye waste, water, effluent)
setup_cost = _build_cost_matrix(...)

# setup_time: duration (drain, rinse, fill, heat)
# Correlated but not deterministic
base_setup_time = (setup_cost / setup_cost_scale) * rng.lognormal(0, 0.2) * setup_time_scale
setup_time = np.maximum(base_setup_time, min_cleaning_cost)
```

**Why:** Some transitions cost more chemicals but take similar time. Decoupling lets `alpha` in the composite objective create genuinely different optimal schedules.

**Mapping to parameters:** `setup_cost_scale`, `setup_time_scale`, `min_cleaning_cost`.

---

## Phase 5: Due Dates — Multi-Segment Customer Model

**Target:** `due_dates`, `weights`

```python
segments = ["premium", "standard", "economy"]
seg_weights = {"premium": segment_weight_spread, "standard": 1.0, "economy": 1.0 / segment_weight_spread}
seg_tightness = {"premium": tightness / segment_tightness_spread, "standard": tightness, "economy": tightness * segment_tightness_spread}

for i in range(n):
    seg = rng.choice(segments, p=segment_mix)
    weights[i] = seg_weights[seg]
    tightness_i = seg_tightness[seg]
    due_dates[i] = (proc_times[i] / total_proc) * avg_load * tightness_i * n + noise
```

**Mapping to parameters:** `customer_segments` (bool), `segment_mix`, `segment_weight_spread`, `segment_tightness_spread`, `tightness`.

---

## Phase 6: Machine Model — Heterogeneous + Eligibility

**Target:** New `machines` field in instance dict. Replaces scalar `m`.

```python
machines = [
    {"id": 0, "chemistries": ["direct", "reactive"], "max_darkness": 5,  "speed": 1.0},
    {"id": 1, "chemistries": ["reactive", "vat"],    "min_darkness": 3,  "speed": 1.2},
    {"id": 2, "chemistries": ["vat"],                 "max_darkness": 10, "speed": 0.8},
]
```

**Mapping to parameters:** `machine_type` (identical→no speed variance, uniform→speed spread only, unrelated→per-cell times), `machine_speed_spread`, `machine_eligibility`, `machine_dedication`.

**Requires updates to:**
- `validate_sigma()` in `evaluator.py` — check machine eligibility for each job
- `heuristics.py` — SPT must skip ineligible machines
- GA permutation encoding stays unchanged (assignment happens at decode step)

---

## Phase 7: Richer Instance Configurations

**Target:** `INSTANCE_CONFIGS`

```python
INSTANCE_CONFIGS = [
    {"n": 5,  "m": 2, "label": "tiny_2m"},        # exact MIP verification
    {"n": 10, "m": 2, "label": "small_2m"},
    {"n": 10, "m": 3, "label": "small_3m"},
    {"n": 20, "m": 2, "label": "medium_2m"},
    {"n": 20, "m": 3, "label": "medium_3m"},
    {"n": 30, "m": 3, "label": "medium_30_3m"},
    {"n": 50, "m": 2, "label": "large_2m"},
    {"n": 50, "m": 3, "label": "large_3m"},
    {"n": 50, "m": 5, "label": "large_5m"},
    {"n": 100, "m": 5,  "label": "xlarge_5m"},
    {"n": 100, "m": 10, "label": "xlarge_10m"},
    {"n": 200, "m": 10, "label": "xxlarge_10m"},
]
```

---

## Phase 8: Configuration Profiles

Metaparameter that sets sane defaults for all params at once.

```python
PROFILES = {
    "baseline": {
        "colour_model": "categorical",
        "n_colours": 7,
        "colour_dist": "uniform",
        "colour_clustering": 0,
        "asymmetry_strength": 1.0,
        "chemistry_penalty": False,
        "setup_cost_scale": 10.0,
        "setup_time_scale": 0.1,
        "min_cleaning_cost": 0.0,
        "proc_dist": "uniform",
        "proc_min": 5,
        "proc_max": 31,
        "proc_colour_corr": 0,
        "proc_outlier_prob": 0,
        "due_date_type": "distinct",
        "tightness": 1.5,
        "customer_segments": False,
        "machine_type": "identical",
        "machine_eligibility": False,
        "release_fraction": 0,
        "objective": "composite",
        "earliness_penalty": False,
    },
    "enhanced": {
        "colour_model": "continuous",
        "n_colours": 7,
        "colour_dist": "skewed",
        "colour_clustering": 0.2,
        "asymmetry_strength": 1.5,
        "chemistry_penalty": True,
        "setup_cost_scale": 10.0,
        "setup_time_scale": 0.1,
        "min_cleaning_cost": 0.5,
        "proc_dist": "uniform",
        "proc_min": 5,
        "proc_max": 31,
        "proc_colour_corr": 0.5,
        "proc_outlier_prob": 0.02,
        "due_date_type": "distinct",
        "tightness": 1.5,
        "customer_segments": True,
        "machine_type": "uniform",
        "machine_eligibility": True,
        "release_fraction": 0.2,
        "objective": "composite",
        "earliness_penalty": False,
    },
    "realistic": {
        "colour_model": "continuous",
        "n_colours": 10,
        "colour_dist": "skewed",
        "colour_clustering": 0.3,
        "asymmetry_strength": 2.0,
        "chemistry_penalty": True,
        "setup_cost_scale": 10.0,
        "setup_time_scale": 0.1,
        "min_cleaning_cost": 0.5,
        "proc_dist": "exponential",
        "proc_min": 5,
        "proc_max": 31,
        "proc_colour_corr": 0.8,
        "proc_outlier_prob": 0.05,
        "due_date_type": "window",
        "tightness": 1.2,
        "customer_segments": True,
        "machine_type": "unrelated",
        "machine_eligibility": True,
        "release_fraction": 0.4,
        "objective": "composite",
        "earliness_penalty": True,
    },
}
```

---

## Phase 9: Validation Harness

```python
def validate_instance(inst: dict) -> list[str]:
    """Return list of issues, empty = valid."""
    issues = []
    # Cost matrix asymmetric?
    # Every job assignable to >=1 machine?
    # Due dates achievable under non-idle schedule?
    # Normalisation scales comparable (f1_norm, f2_norm in similar range)?
    # Reproducibility: same seed -> same instance?
    return issues
```

---

## Impact on Rest of Codebase

| File | Changes Required |
|------|-----------------|
| `evaluator.py` | `validate_sigma()` — check machine eligibility. `estimate_scales()` — handle heterogeneous speed. Add `due_date_type="window"` logic. |
| `heuristics.py` | SPT — skip ineligible machines, consider release times. NN-Greedy — respect machine restrictions. |
| `ga.py` | None (instance consumed as-is; fitness unchanged). |
| `ga_env.py` | None (observation space unchanged). |
| `drl_agent.py` | May need more timesteps for richer instances. |
| `experiments/*.py` | Add `--profile` flag. 12 configs × 2 profiles × 30 seeds × 4 methods = 2,880 runs. |

---

## Recommended Implementation Order

| # | Phase | Effort | Value | Rationale |
|---|-------|--------|-------|-----------|
| 1 | Phase 7 (more configs) | Small | High | Better experimental coverage immediately |
| 2 | Phase 5 (due date segments) | Small | High | Customer priority realism with minimal code |
| 3 | Phase 1 (continuous colour) | Medium | Medium | Better colour model |
| 4 | Phase 3 (proc-colour correlation) | Medium | High | Creates trade-off dynamics → hybrid advantage |
| 5 | Phase 2 (chemical cost matrix) | Large | High | Core generator contribution |
| 6 | Phase 8+9 (profiles + validation) | Medium | High | Wraps everything, enables clean sensitivity analysis |
| 7 | Phase 4 (decouple time/cost) | Medium | Medium | Nuance, but lower priority |
| 8 | Phase 6 (heterogeneous machines) | Large | Medium | Biggest code change, affects many files |

---

## Key Metric

After implementation, run hybrid vs GA on both baseline and enhanced profiles. If hybrid's advantage grows under enhanced (colour-correlated proc times + customer segments + chemistry penalties + release times), that is publishable evidence of the method's robustness to realistic problem complexity.