# WORKFLOW — Step by Step

**You are here:** 22 Apr 2026. Only instance_generator.py done. 17 weeks to deadline.

**Timeline:**
- **Apr 22–28 (Week 1):** Notebooks 01 & 02 — explore & test baselines
- **Apr 28–May 4 (Week 2):** Run baselines & start GA tuning
- **May 5–25 (Week 3–5):** GA tuning & experiments
- **May 26–Jun 15 (Week 6–8):** DRL training & hybrid
- **Jun 16–Jul 6 (Week 9–11):** All 720 runs done, stats
- **Jul 7–Aug 25 (Week 12–17):** Figures, dissertation, submit

---

## PHASE 0: Exploration (Apr 22–Apr 28) — 1 WEEK

### Step 0a: Create notebook 01 — Explore instances
Create file `notebooks/01_exploration.ipynb`:

```python
from src.instance_generator import generate_instance, INSTANCE_CONFIGS

# Generate and visualize one instance from each config
for cfg in INSTANCE_CONFIGS:
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=0)
    print(f"\n{cfg['label']}:")
    print(f"  n={inst['n']}, m={inst['m']}")
    print(f"  proc_times: min={inst['proc_times'].min()}, max={inst['proc_times'].max()}, mean={inst['proc_times'].mean():.1f}")
    print(f"  due_dates: min={inst['due_dates'].min():.1f}, max={inst['due_dates'].max():.1f}")
    print(f"  setup_cost: min={inst['setup_cost'].min():.1f}, max={inst['setup_cost'].max():.1f}")
```

**Purpose:** Understand the data. Check ranges make sense.

---

### Step 0b: Create notebook 02 — Test heuristics
Create file `notebooks/02_heuristics_baselines.ipynb`:

```python
from src.instance_generator import generate_instance
from src.heuristics import spt, nearest_neighbour_greedy
from src.evaluator import evaluate

inst = generate_instance(n=20, m=2, seed=0)

# Test SPT
spt_sigma = spt(inst)
spt_result = evaluate(spt_sigma, inst, alpha=0.5)
print(f"SPT: composite={spt_result['composite']:.2f}, tardiness={spt_result['weighted_tardiness']:.2f}, setup_cost={spt_result['setup_cost']:.2f}")

# Test NN
nn_sigma = nearest_neighbour_greedy(inst)
nn_result = evaluate(nn_sigma, inst, alpha=0.5)
print(f"NN:  composite={nn_result['composite']:.2f}, tardiness={nn_result['weighted_tardiness']:.2f}, setup_cost={nn_result['setup_cost']:.2f}")
```

**Purpose:** Verify heuristics work. Understand baseline quality.

---

## PHASE 1: Baselines (Apr 28–May 4) — 1 WEEK

### Step 1: Run baseline experiments
```bash
source activate dissertation
python experiments/run_baselines.py
```
**What happens:** Creates `results/raw/baselines.json` with SPT and NN greedy results on all 6 instance configs, 30 seeds each.

**When done:** Check file exists and has data:
```bash
head results/raw/baselines.json
```

---

## PHASE 2: GA Tuning (May 5–May 25) — 3 WEEKS

### Step 2: Create notebook 03 for GA tuning
Create file `notebooks/03_ga_tuning.ipynb` with this structure:
1. Import: `from src.instance_generator import generate_instance; from src.ga import run_ga`
2. Generate one instance: `inst = generate_instance(n=10, m=2, seed=0)`
3. Test GA with different hyperparams:
   ```python
   params_grid = [
       {"pop_size": 50, "n_gen": 100, "cx_prob": 0.7, "mut_prob": 0.1},
       {"pop_size": 50, "n_gen": 100, "cx_prob": 0.7, "mut_prob": 0.2},
       {"pop_size": 50, "n_gen": 100, "cx_prob": 0.7, "mut_prob": 0.3},
       # ... test 27 combinations total (3 pop_size × 3 n_gen × 3 cx_prob × 3 mut_prob)
   ]
   for params in params_grid:
       result = run_ga(inst, **params, seed=0)
       print(f"{params} → {result['best_fitness']:.2f}")
   ```
4. Pick best params and update `experiments/run_ga.py` line 14:
   ```python
   GA_PARAMS = {"n_gen": XXX, "pop_size": XXX, "cx_prob": XXX, "mut_prob": XXX}
   ```

**When done:** Update GA_PARAMS with your best hyperparams.

---

### Step 3: Run GA experiments
```bash
python experiments/run_ga.py
```
**What happens:** Creates `results/raw/ga.json` with 30 seeds × 6 configs = 180 runs.

**When done:** Check file exists:
```bash
head results/raw/ga.json
```

---

## PHASE 3: DRL Training (May 26–Jun 15) — 3 WEEKS

### Step 4: Create notebook 04 for DRL training
Create file `notebooks/04_drl_training.ipynb` with:

```python
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.drl_agent import train_ppo

# Train on 5 different seeds of (n=20, m=2) to avoid overfitting
instances = [generate_instance(n=20, m=2, seed=s) for s in range(5)]

# Train PPO on pool
model = train_ppo(
    instances[0],  # train on first instance
    total_timesteps=50_000,
    save_path="models/ppo_hyperheuristic",
    verbose=1
)

print("Model saved to models/ppo_hyperheuristic.zip")
```

**When done:** File `models/ppo_hyperheuristic.zip` exists.

---

### Step 5: Run hybrid experiments
```bash
python experiments/run_hybrid.py
```
**What happens:** Creates `results/raw/hybrid.json` with 30 seeds × 6 configs = 180 runs using trained PPO.

**When done:** Check file exists:
```bash
head results/raw/hybrid.json
```

---

## PHASE 4: Results & Stats (Jun 16–Jul 6) — 3 WEEKS

### Step 6: Create notebook 05 for stats
Create file `notebooks/05_final_evaluation.ipynb` with:

```python
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Load all results
with open("results/raw/baselines.json") as f: baselines = json.load(f)
with open("results/raw/ga.json") as f: ga = json.load(f)
with open("results/raw/hybrid.json") as f: hybrid = json.load(f)

# For each instance config, compute mean ± std for each algorithm
# Then run Wilcoxon signed-rank test: hybrid vs GA, hybrid vs NN, etc.
# Save table to results/figures/comparison_table.csv
```

**When done:** CSV table with all results and p-values.

---

## PHASE 5: Figures & Writing (Jul 7–Aug 25) — 8 WEEKS

### Step 7: Create figures
Create `notebooks/06_visualisations.ipynb` with:
- Gantt charts (one good schedule, one bad)
- Convergence plots (GA vs Hybrid over generations)
- Box plots (SPT vs NN vs GA vs Hybrid per config)
- Table: mean ± std and p-values

Save all to `results/figures/`.

---

### Step 8: Write dissertation
Create `dissertation.tex` (or Google Doc) with 7 chapters:

1. **Introduction** — Why machine scheduling matters, why RL+GA is novel
2. **Literature Review** — GA, DRL, hyper-heuristics, scheduling gap
3. **Problem Formalisation** — PMSP-SDSC, cost matrix, normalisation
4. **Methods** — Instance generator, evaluator, GA, DRL env, PPO
5. **Experiments** — 720 runs, Wilcoxon, sensitivity to alpha
6. **Results** — Tables, figures, statistical significance
7. **Conclusion** — What worked, limitations, future work

**Target:** 15,000–20,000 words.

---

## CHECKLIST

- [ ] **Apr 22–25:** Notebooks 01 & 02 — explore data, test heuristics
- [ ] **Apr 28:** Run `run_baselines.py` → `results/raw/baselines.json`
- [ ] **May 1:** Tune GA hyperparams → update `GA_PARAMS`
- [ ] **May 4:** Run `run_ga.py` → `results/raw/ga.json`
- [ ] **May 25:** Train PPO → `models/ppo_hyperheuristic.zip`
- [ ] **Jun 1:** Run `run_hybrid.py` → `results/raw/hybrid.json`
- [ ] **Jun 10:** Stats notebook → `results/figures/comparison_table.csv`
- [ ] **Jul 6:** All figures in `results/figures/`
- [ ] **Aug 1:** Dissertation draft done
- [ ] **Aug 25:** Final polishing, submit

---

## COMMANDS CHEAT SHEET

```bash
# Activate env
source activate dissertation

# Run baselines (30 min)
python experiments/run_baselines.py

# Run GA (4-6 hours, parallelised)
python experiments/run_ga.py

# Run hybrid (2-3 hours)
python experiments/run_hybrid.py

# Run tests
python tests/test_evaluator.py

# Quick GA test
python -c "from src.instance_generator import generate_instance; from src.ga import run_ga; inst = generate_instance(10, 2, seed=0); result = run_ga(inst, n_gen=5, pop_size=10); print(f'Best: {result[\"best_fitness\"]:.2f}')"
```

---

## IF SOMETHING BREAKS

- Evaluator test fails? → Check `compute_completion_times()` logic
- GA crashes? → Restart Python (DEAP global state issue)
- Env check fails? → Check `ga_env.py` observation bounds
- Hybrid hangs? → Model not trained yet (Step 4)

