# Configuration Changes: Before `81922a6` → Current (`037da0c`)

Only files that actually changed: `src/instance_generator.py`, `src/evaluator.py`,
and all 5 experiment scripts. `ga_env.py`, `drl_agent.py`, and `ga.py` are
**unchanged** between these two commits.

---

## 1. Instance Generator (`src/instance_generator.py`) — major rewrite

### Removed entirely

| Item | Description |
|------|-------------|
| `COLOUR_FAMILIES` | 12 colour families (white→black) with base darkness, shade_std, chemistry type |
| `COLOUR_FAMILIES_KEYS` | List of family names |
| `CHEM_FACTOR` | `{direct: 1.0, reactive: 1.3, vat: 1.5}` — speed factors per chemistry |
| `CHEM_COMPAT` | 6 chemistry incompatibility pairs (direct/reactive/vat) |
| `_sample_colours()` | Continuous darkness + chemistry generation |
| `PROFILES` | `baseline` and `realistic` presets |
| `chemistries` key | Removed from output instance dict |

### `_build_cost_matrix()` simplified

| Aspect | Old | New |
|--------|-----|-----|
| Darkness penalty (dark→light) | `diff ** 1.5` (nonlinear) | `diff * 10.0` (linear) |
| Darkness penalty (light→dark) | `abs(diff) ** 0.5 * 0.3` | `abs(diff) * 3.0` |
| `asymmetry_strength` param | Yes (default 1.0) | Removed |
| Chemistry penalty | Optional CHEM_COMPAT add-on | Removed |
| Noise | `gamma(shape=2, scale=1.0)` | `uniform(0.0, 2.0)` |

### `generate_instance()` signature

| Parameter | Old | New |
|-----------|-----|-----|
| `n` | ✓ | ✓ |
| `m` | ✓ | ✓ |
| `seed` | `Optional[int]` | `int` |
| `tightness` | `1.5` | **removed** |
| `profile` | `None` (baseline/realistic) | **removed** |
| `proc_colour_corr` | `None` | **removed** |
| `setup_time_scale` | `0.1` | **removed** |
| `weekly_hours` | — | `168.0` (new) |

### Instance generation logic

| Step | Old | New |
|------|-----|-----|
| **Processing time** | `rng.integers(5, 31)` fixed range | `rng.uniform(avg_proc * 0.3, avg_proc * 1.7)` where `avg_proc = (m * 168) / n` |
| **Proc-colour correlation** | `base_proc * darkness_factor * chem_factor` if enabled | None — independent of colour |
| **Colour assignment** | Dual mode: categorical (7) or continuous (12 families + chemistry) | Categorical only (7 levels) |
| **Setup cost** | Darkness + optional chemistry penalty, nonlinear | Darkness only, linear asymmetry |
| **Setup time** | `setup_cost * 0.1 * uniform(0.8, 1.2)` | `normalised_cost * avg_proc / 8` |
| **Due dates** | `tightness`-based formula | Proportional within 168-hr week, SPT order |
| **Weights** | Customer segments (VIP/standard/economy) if enabled | Always `np.ones(n)` |
| **Output dict keys** | Includes `chemistries` | Drops `chemistries` |

### Instance configs

| Label | Old | New |
|-------|-----|-----|
| tiny_2m | n=5, m=2 | **removed** |
| small_2m | n=10, m=2 | **removed** |
| small_3m | n=10, m=3 | **removed** |
| medium_2m | n=20, m=2 | **removed** |
| n20_m3 | n=20, m=3 | kept |
| medium_30_3m | n=30, m=3 | **removed** |
| large_2m | n=50, m=2 | **removed** |
| large_3m | n=50, m=3 | **removed** |
| n50_m5 | n=50, m=5 | kept |
| xn50_m5 | n=100, m=5 | **removed** |
| n500_m10 | n=100, m=10 | **removed** |
| n5_m1 | — | n=5, m=1 (new) |
| n10_m1 | — | n=10, m=1 (new) |
| n20_m1 | — | n=20, m=1 (new) |
| n50_m1 | — | n=50, m=1 (new) |
| xn50_m1 | — | n=100, m=1 (new) |
| n500_m10 (new) | — | n=500, m=10 (new) |
| `INSTANCE_CONFIGS_SMALL` | — | `[c for c in INSTANCE_CONFIGS if c["n"] <= 50]` (new) |

**Total: 11 → 8 configs. Shifted from multi-machine (m=2,3,5,10) to single-machine (m=1) base.**

---

## 2. Evaluator (`src/evaluator.py`) — minor

### `estimate_scales()`

| Aspect | Old | New |
|--------|-----|-----|
| `rng` param | `rng=None` (creates new if None) | Removed — hardcoded `np.random.default_rng(0)` |

### New function: `extract_schedule()`

Returns list of dicts per job: `{job, machine, start, end, setup_time, proc_time, colour_id, tardiness}`. Not present in old version.

---

## 3. `ga_env.py`, `drl_agent.py`, `ga.py` — **no changes**

PPO hyperparameters, obs/action spaces, GA toolbox, and all other model parameters are identical between old and new.

---

## 4. Experiment Scripts — profile removal + CLI changes

### All 5 scripts: common changes

| Aspect | Old | New |
|--------|-----|-----|
| `--profile` arg | `baseline` or `realistic` | **removed** |
| `--small` arg | — | **added** (configs with n ≤ 50) |
| `generate_instance()` call | `generate_instance(..., profile=profile)` | `generate_instance(...)` |
| Output paths | `{profile}`-suffixed (e.g. `ga_baseline.json`) | Plain (`ga.json`) |
| Smoke test label | `tiny_2m` | `n5_m1` |

### Per-script specifics

| Script | Param | Old | New |
|--------|-------|-----|-----|
| `train_ppo.py` | save_path | `ppo_hyperheuristic_{profile}` | `ppo_hyperheuristic` |
| `run_ga.py` | (all params same) | — | — |
| `run_hybrid.py` | model_path | `ppo_hyperheuristic_{profile}` | `ppo_hyperheuristic` |
| `run_sensitivity.py` | model_path | `ppo_hyperheuristic_{profile}` | `ppo_hyperheuristic` |
| `run_baselines.py` | (no model) | — | — |

---

## 5. Summary of design changes

1. **Simplified instance model** — dropped profiles, chemistry model, continuous colour model, customer segments
2. **Grounded in reality** — processing times fill a 168-hr week; setup time = proc/8; due dates proportional to SPT order
3. **Linear cost asymmetry** — replaced nonlinear `diff^1.5` with linear `diff × 10`; light→dark now has non-zero cost (`|diff| × 3`)
4. **Single-machine base** — 5 configs at m=1 for clean scaling analysis, plus 3 multi-machine
5. **Deterministic normalisation** — `estimate_scales()` now uses fixed seed=0
6. **Simplified CLI** — `--profile` replaced by `--small`; output paths no longer profile-suffixed
