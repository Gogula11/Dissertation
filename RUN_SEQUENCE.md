# Run Sequence

Activate env first: `source /home/dopedino/miniforge3/bin/activate dissertation`

---

## 0. Pre-Run Checklist

```bash
# Verify clean state
ls results/raw/          # should be empty
ls models/               # should be empty
ls figures/              # should be empty
ls logs/                 # should not exist

# Verify tests
python -m pytest tests/ -q
```
Expected: **45 passed**.

---

## 1. Notebooks 01→04 — Development

Open and **run all cells** in order:

| Notebook | What it does |
|----------|-------------|
| `01_exploration.ipynb` | Explore instance data distributions (12 configs, 12 colour families) |
| `02_heuristic_baselines.ipynb` | Test SPT + NN-Greedy heuristics |
| `03_ga_tuning.ipynb` | Tune GA hyperparameters (single config) |
| `04_drl_training.ipynb` | Train PPO + action frequency analysis |

Outputs: `figures/01_*.png`, `02_*.png`, `03_*.png`, `04_*.png`

Note: these notebooks generate instances with `generate_instance()` default params (= `profile="baseline"`).

---

## 2. Profiled Experiments — Run in terminal

Three profiles defined in `src/instance_generator.py`:

| Profile | Colours | Segments | Chemistry | Colour–proc corr | Asymmetry |
|---------|---------|----------|-----------|-----------------|-----------|
| `baseline` | 7 (categorical) | off | none | off | off |
| `realistic` | 12 (continuous) | on | full | full | full |

Run each profile in sequence:

```bash
# === BASELINE PROFILE ===
python experiments/run_baselines.py --profile baseline
python experiments/train_ppo.py --profile baseline
python experiments/run_ga.py --profile baseline
python experiments/run_hybrid.py --profile baseline
python experiments/run_sensitivity.py --profile baseline

# === REALISTIC PROFILE ===
python experiments/run_baselines.py --profile realistic
python experiments/train_ppo.py --profile realistic
python experiments/run_ga.py --profile realistic
python experiments/run_hybrid.py --profile realistic
python experiments/run_sensitivity.py --profile realistic
```

Outputs by profile:
- `results/raw/baselines_{profile}.json`
- `results/raw/ga_{profile}.json`
- `results/raw/hybrid_{profile}.json`
- `results/raw/sensitivity_{profile}.json`
- `models/ppo_hyperheuristic_{profile}.zip`
- `logs/training_{profile}.log`
- `logs/ppo_tensorboard_{profile}/`

**Order matters:** `train_ppo` must finish before `run_hybrid` (hybrid needs the PPO model). Baselines can run in parallel with PPO training.

**Estimated times** (per profile, 12 configs × 30 seeds):

| Step | Time | Notes |
|------|------|-------|
| `run_baselines` | ~2 min | SPT + NN only, deterministic |
| `train_ppo` | ~10 min | 100k timesteps |
| `run_ga` | ~1–2 hrs | 200 gen × 100 pop (parallelised) |
| `run_hybrid` | ~1–2 hrs | Reuses GA population + PPO inference |
| `run_sensitivity` | ~5 min | medium_2m + medium_3m only, 10 seeds |

**Total: ~3–6 hrs per profile, ~6–12 hrs both.** Can parallelise across profiles on separate machines.

---

## 3. Notebooks 05→06 — Analysis

After all experiments complete, open and **run all cells**:

| Notebook | What it does |
|----------|-------------|
| `05_final_evaluation.ipynb` | Box plots, Wilcoxon tests, α sensitivity, export LaTeX tables |
| `06_visualisations.ipynb` | Gantt chart (SPT vs Hybrid), convergence plot (GA vs Hybrid) |

Before running 05, set `PROFILE = "baseline"` or `"realistic"` at the top of the loading cell to select which results to analyse.

Outputs (namespace per profile, run once for each):

- `figures/05_boxplots_composite_{PROFILE}.png`
- `figures/05_sensitivity_alpha_{PROFILE}.png`
- `results_summary_{PROFILE}.csv`, `results_summary_{PROFILE}.tex`, `results_full_{PROFILE}.tex`
- `figures/06_*.png` (not profile-dependent)

---

## Key Changes Since Original

- `generate_instance()` now has a `profile` param (baseline/realistic)
- Colour families expanded from 7 → 12 (white→black)
- Customer segments (premium/standard/economy) with weight + tightness spread
- Chemistry-based setup cost (`direct`/`reactive`/`vat` with cross-type penalties)
- Processing-time colour correlation (darker + vat = slower)
- `estimate_scales()` in `evaluator.py` auto-normalises composite via inline SPT+NN+random heuristics
- Normalisation fix: `estimate_scales()` replaces hardcoded `/10.0` formula
- Backward compatible: no profile = baseline = old behaviour
- All 45 tests pass
