# Code Architecture — How Everything Fits

## Module Dependency Graph

```
instance_generator.py  (standalone — generates problem instances)
       │
       ▼
  evaluator.py         (scores any solution on any instance)
       │
       ├──> heuristics.py    (SPT, NN-Greedy — simple baselines)
       │
       ├──> ga.py            (GA with DEAP, uses evaluator for fitness)
       │         │
       │         ▼
       │    ga_env.py        (Gymnasium env: wraps GA for RL training)
       │         │
       │         ▼
       │    drl_agent.py     (PPO training + hybrid inference)
       │
       └──> visualisation.py (Gantt charts, box plots, etc.)
```

**Dependency rule:** arrows go down. A module only imports from modules above it. No circular imports.

---

## Module-by-Module

### `src/instance_generator.py` (274 lines)
**What it does:** Creates synthetic scheduling problem instances.

**Key function:**
```python
generate_instance(n, m, seed, profile="baseline") → dict
```
Returns dict with: `proc_times`, `due_dates`, `setup_cost`, `setup_time`, `colour_class`, `darkness`, `chemistry`, etc.

Two profiles controlled by `profile` param:
- **baseline** — 7 colour classes, simple cost asymmetry
- **realistic** — 12 continuous colour families, chemistry penalties, customer segments, colour-processing time correlation

Also exports `INSTANCE_CONFIGS` — 11 standard configs from `tiny_2m` (5 jobs, 2 machines) to `xlarge_10m` (100 jobs, 10 machines).

**Design note:** Asymmetric setup cost matrix is built from three components: max(0,) lightness penalty (dark→light expensive), colour channel diff, chemistry mismatch penalty. Plus noise. This models real textile dyeing.

---

### `src/evaluator.py` (159 lines)
**What it does:** Pure function. Scores any schedule on any instance. No side effects, no randomness, deterministic.

**Key functions:**
```python
validate_sigma(sigma, n) → None  # raises if invalid
compute_completion_times(sigma, instance) → array  # C_j per job
evaluate(sigma, instance, alpha=0.5, f1_scale=None, f2_scale=None) → dict
```
Returns dict: `composite`, `weighted_tardiness`, `setup_cost`, `makespan`.

**Critical: normalisation.** `estimate_scales()` runs SPT + NN-Greedy + random schedule to find typical f1/f2 magnitudes, then uses 1.5× max as scale. Without this, tardiness (can be 500+) dwarfs setup cost (50-200) in the composite score, making setup cost invisible.

**Composite formula:**
```
composite = alpha * (f1 / f1_scale) + (1 - alpha) * (f2 / f2_scale)
```
Alpha = 0.5 by default (equal weight tardiness and setup cost).

---

### `src/heuristics.py` (56 lines)
**What it does:** Two simple baseline schedulers. Fast, deterministic, no parameters.

**Functions:**
```python
spt(instance) → sigma  # sort jobs by processing time, round-robin to machines
nearest_neighbour_greedy(instance) → sigma  # assign each job to machine minimizing setup cost
```

Both return `sigma` = list of lists: `[[machine_0_jobs], [machine_1_jobs], ...]`.

**Why they matter:** SPT = simplest possible scheduler (ignores setup costs entirely). NN-Greedy = accounts for setup cost but only makes locally optimal decisions. Both serve as lower-bound baselines — if your hybrid can't beat these, your contribution is worthless.

---

### `src/ga.py` (151 lines)
**What it does:** Genetic Algorithm using DEAP framework for the scheduling problem.

**Key functions:**
```python
decode_chromosome(chromosome, m) → sigma  # flat permutation → per-machine lists
build_toolbox(n, instance, alpha) → toolbox  # DEAP toolbox with registered operators
run_ga(n, m, instance, pop_size=100, n_gen=300, ...) → {
    "best_sigma", "best_fitness", "logbook", "history"
}
```

**Three mutation operators** (the PPO agent chooses which one to apply):
- **swap** (indpb=0.05) — swap random pairs. Conservative fine-tuning.
- **inversion** (indpb=0.05) — reverse a random segment. Moderate disruption.
- **insertion** (indpb=0.15) — remove and reinsert elements. High exploration.

**DEAP quirk:** DEAP uses global `creator` registry. `hasattr` guards prevent "FitnessMin already exists" errors in notebooks. Multiprocessing uses `get_context("spawn")` not `fork` to avoid DEAP state issues across processes.

---

### `src/ga_env.py` (198 lines)
**What it does:** Gymnasium environment wrapping the GA for RL training. The bridge between your GA and the PPO agent.

**Class:** `GAHyperHeuristicEnv(gym.Env)`

**Observation space** (8D, all normalized [0,1]):
| Feature | What it measures |
|---------|-----------------|
| best_norm | Current best / initial best (↓ as GA improves) |
| mean_norm | Population mean / initial best (convergence indicator) |
| diversity | Mean pairwise Hamming distance in population |
| stagnation | Steps since last improvement / max steps |
| n_norm | Number of jobs / 100 (problem scale context) |
| m_norm | Number of machines / 10 |
| cost_mean_norm | Avg off-diagonal setup cost / max |
| darkness_mean_norm | Avg colour darkness / 10 |

**Action space** — Discrete(3):
- 0 = swap mutation (conservative)
- 1 = inversion mutation (moderate)
- 2 = insertion mutation (exploratory)

**Reward:** relative improvement in best fitness each step. Plateau penalty of -0.01 to discourage idling.

**Episode = one complete GA run** (300 generations). Each step = `step_gens` (default 10) GA generations with the mutation selected by the agent.

---

### `src/drl_agent.py` (95 lines)
**What it does:** PPO training (Stable-Baselines3) + hybrid inference.

**Functions:**
```python
train_ppo(profile, total_timesteps=100000, ...) → model
```
- Creates vectorised env with instance pool (110 instances: 11 configs × 10 seeds)
- Trains PPO with MlpPolicy
- Saves model to `models/ppo_hyperheuristic_{profile}.zip`
- Training env uses reduced GA params (pop=25, gens=100) — makes each episode harder for GA alone, giving the agent room to improve

**PPO hyperparameters:** lr=3e-4, n_steps=2048, batch=64, n_epochs=10, gamma=0.99, ent_coef=0.05

**Hybrid inference:**
```python
run_hybrid(n, m, instance, model, ...) → {"best_sigma", "best_fitness", ...}
```
Loads trained model, runs GA with agent selecting mutations deterministically.

---

### `src/visualisation.py` (37 lines)
**What it does:** Helper for Gantt charts. Most visualisations are in notebooks.

**Key function:**
```python
plot_gantt(sigma, instance, title, ax, alpha_eval) → ax
```

---

## Experiment Pipeline (Run Order)

```
1. train_ppo.py --profile X     → trains PPO model
2. run_baselines.py --profile X → SPT + NN-Greedy results (~2 min)
3. run_ga.py --profile X        → standalone GA results (~1-2 hrs)
4. run_hybrid.py --profile X    → GA+PPO results (~1-2 hrs)
5. run_sensitivity.py --profile X → alpha={0.3,0.5,0.7} comparison (~5 min)
```

Steps 2-3 can run in parallel with step 1. Step 4 needs step 1 done.

Each script runs 12 configs × 50 seeds = 600 runs per profile. Results saved to `results/raw/{algorithm}_{profile}.json`.

**Notebooks (development + analysis):**
| Notebook | Purpose |
|----------|---------|
| `01_exploration.ipynb` | Explore instance data, visualize cost matrix |
| `02_heuristic_baselines.ipynb` | Test SPT + NN, view schedules |
| `03_ga_tuning.ipynb` | Tune GA hyperparams, convergence curves |
| `04_drl_training.ipynb` | Train PPO, action frequency analysis |
| `05_final_evaluation.ipynb` | Box plots, Wilcoxon, alpha sensitivity, export tables |
| `06_visualisations.ipynb` | Gantt charts, convergence plots |

---

## Key Design Decisions in One Line Each

| Decision | Why |
|----------|-----|
| GA + PPO (not pure DRL) | GA handles the discrete optimization; PPO only controls mutation selection — simpler learning problem |
| PPO over DQN/SAC | More stable for continuous obs + discrete actions; less hyperparameter tuning |
| 8D observation space | Captures GA state (convergence, diversity) + problem context (scale, cost structure) |
| 3 mutation actions | swap (fine-tune), inversion (medium), insertion (explore) — covers the exploration-exploitation spectrum |
| Normalized composite objective | Prevents tardiness from dominating setup cost; both objectives contribute |
| Two profiles (baseline/realistic) | Isolates the effect of problem complexity; baseline validates the method, realistic tests real-world applicability |
| Giant-tour encoding | Simple, any permutation maps to a valid schedule via equal-ish split across machines |
| Instance pool training | Prevents overfitting to a single instance; forces generalisable policy |
| Wilcoxon signed-rank | Non-parametric, paired by seed — correct for comparing algorithms on same instances |
| Profile-specific models | Realistic profile has different dynamics; a single model wouldn't handle both well |
