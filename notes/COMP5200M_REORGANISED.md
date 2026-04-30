# COMP5200M — Project Guide (Reorganised by Task Type)
## An Intelligent Hybrid Approach to Machine Scheduling using RL and Genetic Algorithms

> **How to use this document:** This is your end-to-end project guide. Every section tells you **exactly what to build** (specific file names, function signatures), **why it matters**, **how to verify it works**, and **what pitfall to avoid**. For reference code, see `COMP5200M_Scaffolding.md`.

---

## CURRENT STATE (22 April 2026)

You are in **Week 22** of a 30-week MSc programme, with submission on **31 August 2026** (17 weeks remaining).

**Completed:** only `src/instance_generator.py`. ✓

**Behind schedule:** Milestone M1 (literature review, evaluator, heuristics, tests) is partially unstarted. You have 2 weeks to catch up.

**Recovery schedule (strict):**
- **Apr 22 – May 4 (2 weeks):** evaluator + heuristics + all tests (catch up M1)
- **May 5 – May 25 (3 weeks):** GA implementation, tuning, notebook 03 (M2)
- **May 26 – Jun 15 (3 weeks):** DRL env, agent, training (M3)
- **Jun 16 – Jul 6 (3 weeks):** 720 experiment runs, stats (M4)
- **Jul 7 – Jul 31 (4 weeks):** visualisations, dissertation draft (M5)
- **Aug 1 – Aug 25 (4 weeks):** revision, polish, submit (M6)

If you do not complete the evaluator by **5 May**, the entire project timeline collapses.

---

## Table of Contents

### PART 1 — PROJECT SETUP
1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)

### PART 2 — UNDERSTAND THE PROBLEM
3. [Problem Formalisation & Instance Generator](#3-problem-formalisation--instance-generator)
4. [Literature Review](#4-literature-review)

### PART 3 — BUILD THE SYSTEM
5. [Instance Generator (Code)](#5-instance-generator-code)
6. [The Evaluator](#6-the-evaluator)
7. [Baseline Heuristics](#7-baseline-heuristics)
8. [Genetic Algorithm](#8-genetic-algorithm)
9. [DRL Hyper-Heuristic Agent](#9-drl-hyper-heuristic-agent)

### PART 4 — RUN & ANALYSE
10. [Experiments & Statistical Analysis](#10-experiments--statistical-analysis)
11. [Visualisations](#11-visualisations)

### PART 5 — WRITE & SUBMIT
12. [Dissertation Writing](#12-dissertation-writing)
13. [Submission Checklist](#13-submission-checklist)

---

# PART 1 — PROJECT SETUP

Do these once. Skip if already done.

## 1. Environment Setup

### 1.1 Why a Separate Conda Environment

Do not reuse `ds-env`. This project has conflicting dependency chains — PyTorch (for SB3), DEAP, and Gymnasium all need to coexist at specific versions. A fresh environment prevents you from breaking your ML coursework setup and gives you a clean `requirements.txt` to submit alongside your code.

### 1.2 Creating the Environment

Use Python 3.11. Stable-Baselines3 requires 3.10+ and PyTorch is most stable on 3.11 as of now. Create and activate the environment with mamba, then install in this order:

1. Core scientific stack via mamba: numpy, scipy, matplotlib, seaborn, pandas
2. PyTorch via pip (let pip resolve the CUDA version automatically — your RTX 3070 Ti will be detected)
3. Gymnasium and Stable-Baselines3 with extras via pip
4. DEAP via pip
5. Plotly and kaleido via pip (for figure export)
6. pingouin via pip (cleaner Wilcoxon test interface than raw scipy)
7. JupyterLab and ipywidgets via mamba
8. pytest, black, ruff via pip

Immediately after installing everything, freeze it: `pip freeze > requirements.txt`. Do this before writing a single line of code so your environment is reproducible from day one.

### 1.3 Registering the Kernel

Register the environment with Jupyter so VSCodium can see it. Use `python -m ipykernel install --user` with a display name you'll recognise. Select this kernel for all notebooks in the project.

### 1.4 Git Setup

Initialise a git repo in your project root immediately. You want version history from the very first file. Key things to put in `.gitignore` from the start: `__pycache__`, `.ipynb_checkpoints`, your `results/raw/` folder (raw experiment data files will be large), and any trained model `.zip` files except your final best model. Commit after every working module — not after every file, but after every unit that does something testable.

---

## 2. Project Structure

### 2.1 The Core Principle

Keep a hard separation between three kinds of code:

- **`src/`** — pure logic modules with no side effects. These are importable, testable, and have no print statements or file I/O. Everything in here should work whether called from a notebook, a script, or a test.
- **`experiments/`** — CLI scripts that wire `src/` modules together, run the 30-rep experiments, and write results to disk. These are meant to be run from a terminal, not imported.
- **`notebooks/`** — exploration, development, and final visualisation. Notebooks are for looking at things, not for running production experiments.

### 2.2 Directory Layout

```
msc-scheduling/
│
├── src/
│   ├── __init__.py
│   ├── instance_generator.py
│   ├── evaluator.py
│   ├── heuristics.py
│   ├── ga.py
│   ├── ga_env.py
│   └── drl_agent.py
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_heuristic_baselines.ipynb
│   ├── 03_ga_development.ipynb
│   ├── 04_drl_training.ipynb
│   └── 05_final_evaluation.ipynb
│
├── experiments/
│   ├── run_baselines.py
│   ├── run_ga.py
│   └── run_hybrid.py
│
├── results/
│   ├── raw/
│   └── figures/
│
├── tests/
│   ├── test_evaluator.py
│   ├── test_heuristics.py
│   └── test_ga.py
│
├── models/
├── logs/
├── requirements.txt
└── README.md
```

### 2.3 Which Notebooks Do What

- `01_exploration.ipynb` — generate instances, visualise the cost matrix as a heatmap, check colour-transition asymmetry, plot distributions of processing times and due dates
- `02_heuristic_baselines.ipynb` — run SPT and nearest-neighbour on a few instances by hand, eyeball results, plot example Gantt charts
- `03_ga_development.ipynb` — tune GA hyperparameters, plot convergence curves, verify the GA is actually improving
- `04_drl_training.ipynb` — train the PPO agent, monitor reward curves via TensorBoard, verify agent learning
- `05_final_evaluation.ipynb` — load all results from `results/raw/`, produce comparison tables, box plots, Gantt charts, and Wilcoxon test outputs

---

# PART 2 — UNDERSTAND THE PROBLEM

Read these sections to understand what you're building and why it's novel. Then read the academic literature.

## 3. Problem Formalisation & Instance Generator

### 3.1 Why You Write This First

The instance generator is the foundation everything else sits on. The GA needs instances to optimise. The heuristics need instances to evaluate. The evaluator needs instances to score solutions. If your instance generator has a bug — say, due dates that are always achievable with zero tardiness regardless of scheduling — your entire experimental comparison is meaningless. Write it first, understand it completely, test it thoroughly.

### 3.2 What an Instance Represents

An instance is a single snapshot of the scheduling problem: a specific set of jobs with specific processing times, due dates, release times, weights, and a specific transition cost matrix. In your experiments, each seed generates a unique instance. Two runs with the same seed must produce identical instances — this is what makes your 30-run comparisons paired (same instances, different algorithm behaviour).

### 3.3 The Transition Cost Matrix

This is the most domain-specific part of your model and the one your examiner will scrutinise most. The asymmetry must be intentional and explainable. The rule is: transitioning from a darker colour to a lighter colour costs more waste than the reverse, because residual dark dye contaminates the lighter batch.

Concretely, define a darkness ranking for your colour palette (e.g. white=1, yellow=2, light blue=3, green=4, red=5, navy=6, black=7). The cost of transitioning from colour A to colour B should scale with max(0, darkness(A) − darkness(B)). Same colour should cost zero. Add a small noise term so the matrix is not degenerate (identical costs create trivial structure that makes the problem easier than reality).

The setup time matrix can follow the same structure at a different scale — setup time and setup cost are distinct things in your formal model.

### 3.4 Due Date Tightness

Due dates need to be calibrated carefully. If they are too loose, every algorithm achieves zero tardiness and the only thing you're measuring is setup cost. If they are too tight, every algorithm has massive tardiness and again differentiation is hard. A tightness factor that scales due dates relative to average machine load is standard practice. Test your due date generation by running your heuristics on a few instances and checking that tardiness is nonzero but not catastrophically large.

### 3.5 Testing the Instance Generator

At minimum, verify:
- Processing times are positive and within your defined range
- The cost matrix diagonal is exactly zero (no cost transitioning to the same job)
- The matrix is genuinely asymmetric: S[i][j] ≠ S[j][i] for most pairs
- Due dates are positive
- The same seed always produces the same instance

---

## 4. Literature Review

### 4.1 What You Need to Cover

Your SP cites four papers. The dissertation literature review needs roughly 15–25 sources, woven into a coherent narrative. Topics to cover, in order:

**Parallel machine scheduling (PMSP).** Start broad. Establish that PMSP is NP-hard in the general case and explain what makes your variant — asymmetric, sequence-dependent setup costs across parallel machines — particularly difficult.

**Sequence-dependent setup costs.** This is what makes your problem non-trivial. Explain the asymmetry and why standard heuristics like SPT treat all transitions as equivalent.

**Classical heuristics.** Cover SPT, NEH (Nawaz-Enscore-Ham), and nearest-neighbour greedy. Explain their time complexity and why they are used as baselines — fast and deterministic — not because they are good.

**Genetic algorithms for scheduling.** Focus on chromosome representations for scheduling problems (permutation-based). Order crossover (OX) and its preservation of relative job ordering. Mutation operators. The tension between exploration and exploitation.

**Deep Reinforcement Learning for combinatorial optimisation.** Cover the general paradigm shift from hand-crafted to learned policies. PPO specifically — why it is more stable than DQN for continuous or structured action spaces. Briefly cover prior work applying DRL to scheduling (there is growing literature from 2019 onwards).

**Hyper-heuristics.** Define the concept formally: a heuristic that selects or generates other heuristics. Frame your DRL agent as a hyper-heuristic operating at the meta-level of the GA, not at the level of individual schedules. This framing is what makes your contribution distinct from just "RL for scheduling."

**The gap.** End the literature review with a paragraph that explicitly states: prior work has combined GA and RL in scheduling, but not specifically as a PPO hyper-heuristic controlling mutation operator selection in an asymmetric parallel machine scheduling problem. This gap is what your dissertation fills.

### 4.2 Where to Search

Use Google Scholar, IEEE Xplore, and ScienceDirect. For the most recent DRL-for-scheduling work (2022–2025), arXiv cs.AI and cs.LG will have preprints before journal publication. Zotero is the right tool for managing references — install the browser connector, attach PDFs, and write a 2–3 sentence note on each paper explaining what it contributes to your work specifically.

### 4.3 Key Papers to Find Beyond Your SP

Beyond what you already cited, look specifically for:

- The NEH heuristic original paper (Nawaz, Enscore, Ham, 1983) — foundational baseline
- Pinedo's "Scheduling: Theory, Algorithms, and Systems" textbook — the standard reference your examiner will expect to see cited for problem formalisation
- Recent (2020–2024) papers on attention-based or transformer models for scheduling — even if you're not using them, acknowledging this direction and explaining why you chose the GA-DRL hybrid instead shows breadth
- Burke et al. on hyper-heuristics (2013, JORS) — the standard definition paper for the field

### 4.4 What Not to Do

Do not write the literature review as a series of "Paper X found Y. Paper Z found W." paragraphs. Your examiner will notice. Write it as a story: here is the problem class, here is what people tried, here is what worked, here is what the limitations were, here is what nobody has done yet. Every paper you cite should be serving that narrative.

---

# PART 3 — BUILD THE SYSTEM

Code in this order. All code must pass tests before moving to the next section.

## 5. Instance Generator (Code)

**File:** `src/instance_generator.py` — Already complete. ✓

This module generates synthetic PMSP-SDSC instances. Every module after this depends on it.

**Functions:**
- `generate_instance(n, m, seed=None, tightness=1.5)` → dict with all instance data
- `INSTANCE_CONFIGS` — list of 6 standard problem sizes: (n=10,m=2), (n=10,m=3), (n=20,m=2), (n=20,m=3), (n=50,m=2), (n=50,m=3)

**Verify:** Run `python -c "from src.instance_generator import generate_instance; inst = generate_instance(10, 2, seed=0); print(inst.keys())"`

---

## 6. The Evaluator

**Purpose:** The evaluator scores solutions. Any bug here breaks all 720 experiments. Build this first, test it thoroughly, move on.

### 6.1 What You Will Build

**File:** `src/evaluator.py`

**Functions to implement:**

1. **`validate_sigma(sigma: list, n: int) → bool`**
   - Check that sigma contains every job index 0..n-1 exactly once, distributed across its machines.
   - Raise `ValueError` if invalid.
   - Called by evaluate() before any computation.

2. **`compute_completion_times(sigma: list, instance: dict) → dict`**
   - Input: sigma (list of m machine sequences), instance (from instance_generator).
   - For each job on each machine, compute its completion time = max(previous job completion + setup time, release time) + processing time.
   - Setup time for the first job on a machine is 0.
   - Return dict: `{"completion_times": array, "last_completion": dict}`.

3. **`compute_tardiness(sigma: list, instance: dict) → float`**
   - Compute total weighted tardiness: Σ weight_i × max(0, C_i − d_i) for all jobs i.
   - Return single float.

4. **`compute_setup_cost(sigma: list, instance: dict) → float`**
   - Sum all setup costs S[job_i][job_j] for transitions from job_i to job_j on each machine.
   - Return single float.

5. **`evaluate(sigma: list, instance: dict, alpha: float = 0.5) → dict`**
   - Main entry point.
   - Call `validate_sigma(sigma, instance['n'])` first — fail hard if invalid.
   - Compute completion times, tardiness f1, setup cost f2.
   - **Normalise both objectives before combining:**
     - f1_norm = f1 / mean(f1 over all prior runs on this instance size) [or use a default of 100.0 if no prior runs]
     - f2_norm = f2 / mean(f2 over all prior runs on this instance size) [or use a default of 50.0 if no prior runs]
     - **Important:** Without normalisation, one objective dominates regardless of alpha. Document this normalisation approach in a comment.
   - composite = α × f1_norm + (1−α) × f2_norm
   - Compute makespan C_max (max completion time).
   - Return dict:
     ```python
     {
       "composite": composite,
       "weighted_tardiness": f1,
       "setup_cost": f2,
       "makespan": C_max,
       "f1_normalised": f1_norm,
       "f2_normalised": f2_norm
     }
     ```

### 6.2 Design Rules

- **No randomness, no state, no side effects.** Call evaluate() a million times with the same sigma and instance and get the same result every time.
- **Fail fast.** If sigma is invalid, raise immediately. Do not compute anything.
- **Test against hand-computed examples** before moving on.

### 6.3 Testing the Evaluator

Create a test file `tests/test_evaluator.py`.

**Minimal test:** n=3, m=1, seed=42.
- Generate instance: `instance = generate_instance(3, 1, seed=42)`
- Create a test solution sigma: say, sigma = [[0, 1, 2]] (all jobs on machine 0 in order)
- Compute by hand:
  - proc_times from the instance (e.g., [15, 8, 22])
  - setup_times from the instance
  - C_0 = 0 + 0 + 15 = 15 (first job: no setup)
  - C_1 = 15 + setup_time[0][1] + 8 = …
  - C_2 = C_1 + setup_time[1][2] + 22 = …
  - tardiness = Σ weight_i × max(0, C_i − d_i)
- Call `evaluate(sigma, instance, alpha=0.5)` and verify the weighted_tardiness and setup_cost match your hand calculations exactly.

**Additional test:** Verify validate_sigma rejects [0, 2] on n=3 (missing job 1).

**Verification command:**
```bash
python -m pytest tests/test_evaluator.py -v
```

All tests must pass before moving to Section 7.

### 6.4 Key Pitfall

**Pitfall:** Assuming setup cost and tardiness are on the same scale. They are not. On n=50 instances, tardiness can be 500+, while setup cost is typically 50–200. Combining them without normalisation means setup cost is invisible in the composite score. The code comment `# Normalise by instance mean` is not a suggestion — it is critical. Document how you compute the normalisation constant (e.g., empirical mean from baselines) in Chapter 3 of your dissertation.

---

## 7. Baseline Heuristics

**Purpose:** Baselines are essential. Examiners expect you to compare against something simple.

### 7.1 What You Will Build

**File:** `src/heuristics.py`

**Functions to implement:**

1. **`spt(instance: dict) → list`** (Shortest Processing Time) 
   - Sort all jobs by processing time ascending. 
   - Assign jobs round-robin to machines. 
   - Call `validate_sigma()` before returning.

2. **`nearest_neighbour_greedy(instance: dict) → list`** (NN Greedy) 
   - For each job in order: find the machine where the transition cost from the last job is lowest (0 if machine is empty). 
   - Assign to that machine. 
   - Call `validate_sigma()` before returning.

### 7.2 Testing Heuristics

Create `tests/test_heuristics.py`. Test both functions on n=4, m=2, seed=42. Verify all jobs appear exactly once across machines.

Verification command:
```bash
python -m pytest tests/test_heuristics.py -v
```

### 7.3 Key Pitfall

**Pitfall:** Assuming all jobs must be processed in order 0, 1, 2, .... SPT reorders jobs by processing time, then assigns round-robin. Nearest-neighbour assigns jobs in order 0, 1, 2, ... but decides which machine each goes to by minimising transition cost. These are fundamentally different algorithms.

---

## 8. Genetic Algorithm

**Purpose:** GA is your baseline solver. It must work before the DRL agent can control it.

### 8.1 What You Will Build

**File:** `src/ga.py`

**Functions to implement:**

1. **`decode_chromosome(individual: list, m: int) → list`** 
   - Split flat permutation into m contiguous segments. 
   - If n % m != 0, distribute remainder: final machines get one extra job each.

2. **`make_fitness_fn(instance: dict, alpha: float = 0.5) → callable`** 
   - Return a fitness function that takes a chromosome, decodes it, evaluates it, returns composite fitness. 
   - Factory for DEAP toolbox.

3. **`build_toolbox(instance: dict, alpha: float = 0.5) → deap.base.Toolbox`** 
   - Register individual, population, evaluate, mate (cxOrdered), three mutation operators (swap, inversion, aggressive_swap), and selection (selTournament, tournsize=3). 
   - Guard against DEAP global state with hasattr check.

4. **`run_ga(instance, pop_size=100, n_gen=200, cx_prob=0.8, mut_prob=0.2, mutation_strategy="swap", alpha=0.5, seed=None, verbose=False) → dict`** 
   - Build toolbox, initialise population, run generational GA, return (best_individual, best_fitness_per_gen, final_population).

### 8.2 Hyperparameter Grid

Tune on n=10, m=2 only, instance seed=42.

| Pop Size | Generations | Cx Prob | Mut Prob | Mutation Ops |
|---|---|---|---|---|
| 50 | 100 | 0.7 | 0.1 | [swap] |
| 100 | 200 | 0.8 | 0.2 | [swap, inversion] |
| 200 | 500 | 0.9 | 0.3 | [swap, inversion, aggressive] |

Run each config 10 times with seeds 0–9. Pick configuration with lowest mean fitness. Lock these parameters for all subsequent experiments.

### 8.3 What Good Convergence Looks Like

- Gen 1–40: steep drop.
- Gen 40–end: flatter.
- No sudden spikes.

If flat from gen 1: check fitness function is returning different values. If never stops dropping: increase num_gens.

### 8.4 DEAP Global State: Multiprocessing Fix

Use `multiprocessing.get_context("spawn")` instead of default fork when running in parallel. Add to your experiment scripts.

### 8.5 Key Pitfall

**Pitfall:** Restarting kernel mid-session and re-running build_toolbox(). DEAP's global state persists. Always restart kernel fully before running GA code again. Terminal scripts are not affected.

---

## 9. DRL Hyper-Heuristic Agent

**Purpose:** The DRL agent is the novel contribution. It controls which GA mutation operator to use, learning a meta-policy.

### 9.1 What You Will Build

- **`src/ga_env.py`** — Gymnasium environment wrapping the GA.
- **`src/drl_agent.py`** — PPO training and inference.

### 9.2 The Gymnasium Environment (ga_env.py)

**Class:** `GAHyperHeuristicEnv(gym.Env)`

Constructor: `def __init__(self, instance, total_gens=200, step_gens=10, seed=None)`

Methods:
1. **`reset()`** — Initialise fresh GA population. Compute initial state observation. Return (observation, info).
2. **`step(action: int)`** — Run GA for step_gens generations with chosen mutation operator (0=swap, 1=inversion, 2=aggressive_swap). Compute new observation. Compute reward: (best_fitness_before − best_fitness_after) / best_fitness_before. Set done=True if generations_elapsed >= total_gens. Return (observation, reward, terminated, truncated, info).
3. **`close()`** — Clean up.

### 9.3 State Space (4D Observation)

1. **best_fitness_norm:** current best / initial best. Range: [0.0, 1.0].
2. **mean_fitness_norm:** current population mean / initial best.
3. **diversity:** mean pairwise Hamming distance among sample of min(20, pop_size) chromosomes, divided by n. Range: [0.0, 1.0].
4. **stagnation_norm:** (generations since best improved) / total_gens. Range: [0.0, 1.0].

Return `np.array([best_fitness_norm, mean_fitness_norm, diversity, stagnation_norm], dtype=np.float32)`.

### 9.4 Action Space

Discrete(3):
- Action 0: swap mutation (indpb=0.05)
- Action 1: inversion mutation (indpb=0.05)
- Action 2: aggressive swap mutation (indpb=0.20)

### 9.5 Training Details (drl_agent.py)

**`train_ppo(instance, num_timesteps=50000, seed=None) → PPO`:**
1. Create env.
2. Run check_env() — fix all warnings before proceeding.
3. Create PPO agent: MlpPolicy, n_steps=256, ent_coef=0.01, learning_rate=3e-4.
4. Train and monitor via TensorBoard.
5. Save to `models/ppo_n20m2`.
6. Return model.

**`run_hybrid(instance, model) → dict`:** Create env, reset, loop step with deterministic predictions until done.

### 9.6 Training Instance Pool

Do not train on a single instance — the agent will overfit. Generate 5 instances with (n=20, m=2) and seeds 0–4. During training, each reset() picks a random instance from the pool.

### 9.7 Reward Sparsity Fallback

If reward is 0 for more than 50 consecutive steps: reduce step_gens to ceil(step_gens / 2), floor at 1.

### 9.8 Training Duration

- Minimum: 50,000 timesteps.
- Stopping criterion: if mean episode reward plateaus (no improvement for 10 consecutive log intervals), stop early.
- If reward still rising at 50,000 ts, continue to 100,000 ts.

### 9.9 Post-Training Validation

After training, run agent for 10 episodes on test instances (seeds 100–109). Log action chosen at each step. Plot action frequency histograms (early, middle, late thirds of episode). Expected: inversion early (diversity), swap late (fine-tuning). If agent picks one action always → learning failed.

### 9.10 Key Pitfall

**Pitfall:** Training on a single instance. Use an instance pool. Document this in your dissertation.

---

# PART 4 — RUN & ANALYSE

## 10. Experiments & Statistical Analysis

**Purpose:** Run 720 experiments to validate your contribution.

### 10.1 Experiment Design

| Algorithm | Seeds | Instances | Runs |
|---|---|---|---|
| SPT | 0–29 | 6 configs | 180 |
| NN Greedy | 0–29 | 6 configs | 180 |
| GA | 0–29 | 6 configs | 180 |
| Hybrid | 0–29 | 6 configs | 180 |
| **Total** | | | **720** |

Instance configs: (n=10, m=2), (n=10, m=3), (n=20, m=2), (n=20, m=3), (n=50, m=2), (n=50, m=3).

### 10.2 Pre-Experiment Benchmark (REQUIRED)

Before committing to 30 seeds, estimate runtime by running each algorithm on all 6 configs once. Do not skip this.

### 10.3 Result File Format

Directory: `results/raw/`

Filename pattern: `{algorithm}_{config_label}_{seed}.json` (e.g. `spt_n20m2_seed0042.json`)

Contents: algorithm, instance_label, seed, composite_fitness, weighted_tardiness, setup_cost, makespan, runtime_seconds.

### 10.4 Running Experiments (Terminal Scripts)

Do not run from notebooks. Three scripts in `experiments/`: `run_baselines.py`, `run_ga.py`, `run_hybrid.py`. Use `multiprocessing.get_context("spawn")` in all.

Run from terminal:
```bash
python experiments/run_baselines.py  # ~15 min
python experiments/run_ga.py         # ~2–3 hours
python experiments/run_hybrid.py     # ~2–3 hours
```

### 10.5 Statistical Testing

Wilcoxon signed-rank test (non-parametric, paired) using pingouin. For each of 6 instance sizes × 3 algorithm pairs = 18 tests. Significance threshold: p < 0.05. Do not hide non-significant results.

### 10.6 Sensitivity Analysis (α Variation)

Re-run GA and Hybrid on medium instances (n=20, m=2 and n=20, m=3) with α = 0.3, 0.5, 0.7. Use 10 seeds per (config, alpha) pair. Discuss whether the hybrid's advantage persists across all α.

---

## 11. Visualisations

### 11.1 Gantt Charts

Each machine as a row, each job as a horizontal bar. Colour-code by the job's colour class. Show setup times as hatched grey blocks between jobs. Always show side-by-side comparison: SPT (worst baseline) on left, hybrid on right, same instance. Export as PDF, not PNG.

### 11.2 Convergence Curves

Plot best fitness vs generation for plain GA and hybrid. Show that hybrid converges faster or to a lower value. Show convergence for at least three instance sizes (small, medium, large) in a grid of subplots.

### 11.3 Box Plots

One subplot per instance configuration, four boxes per subplot (SPT, NN, GA, Hybrid). 30 runs per box. The hybrid's box should be systematically lower.

### 11.4 Action Frequency Analysis

Record action chosen at each step across multiple runs. Plot bar chart showing action selection frequency over episode (early, middle, late thirds).

### 11.5 Cost Matrix Heatmap

Plot the transition cost matrix S as a heatmap for a small instance. The asymmetry should be visually obvious — upper triangle (dark→light) substantially more colourful than lower triangle.

---

# PART 5 — WRITE & SUBMIT

## 12. Dissertation Writing

### 12.1 Chapter Structure and Target Lengths

| Chapter | Content | Target pages |
|---|---|---|
| 1 — Introduction | Motivation, research question, contributions, outline | 4–6 |
| 2 — Literature Review | As per Section 4 of this guide | 12–16 |
| 3 — Problem Formulation | Formal PMSP-SDSC model, notation table, instance generator description | 5–7 |
| 4 — System Design & Implementation | Architecture, GA design, GA-env design, PPO design | 15–20 |
| 5 — Experimental Evaluation | Setup, results, statistical tests, visualisations, analysis | 12–18 |
| 6 — Discussion | What worked, limitations, threats to validity, future work | 5–8 |
| 7 — Conclusion | Restate contributions, summarise findings, closing | 2–3 |
| References | 15–25 sources | — |

Leeds MSc dissertations are typically 15,000–20,000 words. Check the COMP5200M guidelines on Minerva for the exact requirement.

### 12.2 LaTeX Setup

Use LaTeX. Use the `report` class with 12pt font, 25mm margins, and packages: `amsmath, amssymb, graphicx, booktabs, hyperref, natbib` and `algorithm, algpseudocode` (NOT algorithm2e). Use Zotero to export a `.bib` file.

### 12.3 Writing Chapter 3 (Problem Formulation) — 5–7 Pages

Must include:
1. Formal problem definition with notation table.
2. Instance generator explanation: colour classes, darkness ranking, cost matrix asymmetry rule.
3. Objective function: F = α·f1_norm + (1−α)·f2_norm with explicit normalisation explanation.
4. Solution representation.

Exact subsections:
- 3.1 Parallel Machine Scheduling with Sequence-Dependent Setup Costs
- 3.2 Problem Formalisation (notation table)
- 3.3 Instance Generation & Asymmetric Cost Structure
- 3.4 Composite Objective & Normalisation

### 12.4 Writing Chapter 4 (System Design & Implementation) — 15–20 Pages

Key requirements:
- Pseudocode, not code. Use `algpseudocode`. Code snippets in appendices only.
- Two parameter tables: GA parameters (with tuning justification) and PPO parameters (with reasoning).
- Justify every design choice.
- System architecture diagram showing the full pipeline.

Exact subsections:
- 4.1 Evaluator & Objective Function
- 4.2 Baseline Heuristics (SPT & NN Greedy)
- 4.3 Genetic Algorithm Design
- 4.4 Gymnasium Environment for Hyper-Heuristic
- 4.5 PPO Agent Training & Inference
- 4.6 Parameter Tables & Justification

### 12.5 Writing Chapter 5 (Experimental Evaluation) — 12–18 Pages

Structure:
1. 5.1 Experiment Setup — 2 pages.
2. 5.2 Main Results — 4 pages. Lead with headline result in the first sentence.
3. 5.3 Statistical Analysis — 2 pages. State p-values explicitly.
4. 5.4 Sensitivity Analysis — 2 pages.
5. 5.5 Action Frequency Analysis — 2 pages.

First sentence rule: Lead with your main result, not the methodology. Be precise with numbers: always include the metric, baseline, and p-value.

### 12.6 Writing Chapter 6 (Discussion) — 5–8 Pages

Key sections:
- 6.1 Interpretation of Results — explain the mechanism of the hybrid's behaviour via action-frequency analysis.
- 6.2 Non-Significant Results — explain, don't hide.
- 6.3 Threats to Validity — synthetic instances, same distribution for train/eval, 30-run sample size, no truly unseen instances.
- 6.4 Limitations — parallel machine scheduling only, training cost not counted, single-size training.
- 6.5 Future Work — curriculum learning, real data, stronger baselines, extended objectives.

### 12.7 The Abstract

Write last, ~250 words. Must cover in order:
1. Problem: parallel machine scheduling with sequence-dependent setup costs
2. Why hard: NP-hard, large combinatorial search space
3. Approach: hybrid GA-DRL system where PPO hyper-heuristic controls mutation operator selection
4. Results: specific numbers (X% improvement, p-value, instance sizes where significant)
5. Implication: hyper-heuristics can effectively meta-control metaheuristics for complex scheduling

### 12.8 Writing Rules

- Numbers: never "performed better" — always include the metric, baseline, and p-value.
- Figures: every figure needs a self-contained caption.
- Citations: every claim needs a citation or your experimental evidence.
- Consistency: pick one name for your system and use it throughout.
- Proofreading: print the draft and proofread on paper.

---

## 13. Submission Checklist

### Code Quality

- [ ] All `src/` modules have docstrings
- [ ] `validate_sigma()` is called inside `evaluate()`
- [ ] All random seeds are explicitly set and logged in result files
- [ ] `requirements.txt` is up to date
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] All notebooks run cleanly from top to bottom after a full kernel restart

### Experiments

- [ ] 30 runs × 6 instance configs × 4 algorithms completed (720 total)
- [ ] All raw results saved to `results/raw/` as JSON or CSV
- [ ] Wilcoxon tests computed for all algorithm pairs on all instance sizes
- [ ] All figures exported as PDF to `results/figures/`
- [ ] Sensitivity analysis (varying alpha) completed on at least medium instances

### Dissertation

- [ ] Word count within Leeds School of Computer Science guidelines
- [ ] All algorithms presented as formal pseudocode in main text
- [ ] Parameter table present for GA hyperparameters
- [ ] Parameter table present for PPO hyperparameters
- [ ] Every figure referenced in the text and has a self-contained caption
- [ ] References formatted consistently
- [ ] Ethics form included (Appendix A)
- [ ] Full results tables in Appendix B if main text tables are abbreviated
- [ ] PDF compiles cleanly with no missing references or figures
- [ ] Spell-checked and proofread
- [ ] Submitted via Minerva by 31 August 2026

### Submission Files

- `GOGULA26-Dissertation.pdf` — main submission via Minerva
- `GOGULA26-Code.zip` — all `src/`, `notebooks/`, `experiments/`, `requirements.txt`, and a `README.md`. Do not include `results/raw/` or trainer model files; include only the final trained PPO model.

### README.md Structure

The README must have exact sections for Setup, Reproducing Results (run_baselines.py, run_ga.py, PPO training command, run_hybrid.py), Visualisations & Analysis, and Key Files.

---

## Quick Reference: Timeline vs This Guide

| Milestone | Period | Sections |
|---|---|---|
| M1: Literature review + environment model | Mar–Apr (Weeks 20–22) | §3, §4, §5, §6 |
| M2: GA baseline | May–Jun (Weeks 25–28) | §7 |
| M3: DRL agent + integration | Jun–Jul (Weeks 29–S2) | §8 |
| M4: Evaluation + write-up | Jul–Aug (Weeks S3–S7) | §9, §10, §11, §12 |

---

**This reorganised guide flows: Setup → Understand → Code → Run → Write.**

Use `WORKFLOW.md` for day-to-day step-by-step commands.
Use this guide when you need detail on any section.
