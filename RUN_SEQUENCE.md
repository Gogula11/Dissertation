# Run Sequence

Activate env first: `source /home/dopedino/miniforge3/bin/activate dissertation`



---

## 1. Verify Tests

```bash
python -m pytest tests/ -v
```

Expected: 21 passed.

---

## 2. Notebooks 01→04 — Development

Open and **run all cells** in order:

| Notebook | What it does |
|----------|-------------|
| `01_exploration.ipynb` | Explore instance data distributions |
| `02_heuristic_baselines.ipynb` | Test SPT + NN-Greedy heuristics |
| `03_ga_tuning.ipynb` | Tune GA hyperparameters |
| `04_drl_training.ipynb` | Train PPO + action frequency analysis |

Outputs: `figures/01_*.png`, `02_*.png`, `03_*.png`, `04_*.png`

---

## 3. Experiments — Run in terminal

```bash
python experiments/train_ppo.py        # ~30-60 min → models/ppo_hyperheuristic.zip
python experiments/run_baselines.py    # ~30 min    → results/raw/baselines.json
python experiments/run_ga.py           # ~1-1.5 hrs → results/raw/ga.json     (was 4-6 hrs; n_gen 200→50)
python experiments/run_hybrid.py       # ~20-40 min → results/raw/hybrid.json (was 2-3 hrs; n_gen 200→50 + parallelised)
python experiments/run_sensitivity.py  # ~5-10 min  → results/raw/sensitivity.json (was 30 min; n_gen 200→50 + parallelised)
```

---

## 4. Notebooks 05→06 — Analysis

Open and **run all cells**:

| Notebook | What it does |
|----------|-------------|
| `05_final_evaluation.ipynb` | Box plots, Wilcoxon tests, α sensitivity, export LaTeX tables |
| `06_visualisations.ipynb` | Gantt chart (SPT vs Hybrid), convergence plot (GA vs Hybrid) |

Outputs: `figures/05_*.png`, `06_*.png`, `results_summary.csv`, `results_summary.tex`
