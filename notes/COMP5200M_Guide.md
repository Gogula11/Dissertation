# COMP5200M — Project Guide
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

**PART 1 — PROJECT SETUP**
1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)

**PART 2 — UNDERSTAND BEFORE YOU CODE**
3. [Problem Formalisation & Instance Generator](#3-problem-formalisation--instance-generator)
4. [Literature Review](#4-literature-review)

**PART 3 — BUILD THE SYSTEM**
5. [The Evaluator](#5-the-evaluator)
6. [Baseline Heuristics](#6-baseline-heuristics)
7. [Genetic Algorithm](#7-genetic-algorithm)
8. [DRL Hyper-Heuristic Agent](#8-drl-hyper-heuristic-agent)

**PART 4 — RUN & ANALYSE**
9. [Experiments & Statistical Analysis](#9-experiments--statistical-analysis)
10. [Visualisations](#10-visualisations)

**PART 5 — WRITE & SUBMIT**
11. [Dissertation Writing](#11-dissertation-writing)
12. [Submission Checklist](#12-submission-checklist)

---

# PART 1 — PROJECT SETUP

Do this once at the start. Skip if already done.

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

Create all directories and empty `__init__.py` files before writing any logic. Having the skeleton in place forces modular thinking.

### 2.3 Which Notebooks Do What

- `01_exploration.ipynb` — generate instances, visualise the cost matrix as a heatmap, check that your colour-transition asymmetry looks sensible, plot distributions of processing times and due dates
- `02_heuristic_baselines.ipynb` — run SPT and nearest-neighbour on a few instances by hand, eyeball results, plot example Gantt charts
- `03_ga_development.ipynb` — tune GA hyperparameters, plot convergence curves, verify the GA is actually improving over generations and not plateauing immediately
- `04_drl_training.ipynb` — train the PPO agent, monitor reward curves via TensorBoard, verify the agent is learning a non-trivial policy
- `05_final_evaluation.ipynb` — load all results from `results/raw/`, produce the comparison tables, box plots, Gantt charts, and Wilcoxon test outputs that go directly into your dissertation

---

# PART 2 — UNDERSTAND BEFORE YOU CODE

Read these sections to understand the problem you're solving and the existing literature. This is the foundation for all coding work.

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

Your SP cites four papers. The dissertation literature review needs roughly 15–25 sources, and they need to be woven into a coherent narrative, not listed as summaries. The topics to cover, in the order they should appear in Chapter 2:

**Parallel machine scheduling (PMSP).** Start broad. Establish that PMSP is NP-hard in the general case and explain what makes your variant — asymmetric, sequence-dependent setup costs across parallel machines — particularly difficult. The combinatorial explosion of the search space should be quantified.

**Sequence-dependent setup costs.** This is what makes your problem non-trivial compared to classical PMSP. Explain the asymmetry and why standard heuristics like SPT treat all transitions as equivalent (which is why they underperform).

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

Write these modules in order. Each section explains exactly what to build, test it thoroughly, then move on. Dependencies: evaluator → heuristics → GA → GA environment → DRL agent.

## 5. The Evaluator

**Purpose:** The evaluator scores solutions. Any bug here breaks all 720 experiments. Build this first, test it thoroughly, move on.

### 5.1 What You Will Build

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

### 5.2 Design Rules

- **No randomness, no state, no side effects.** Call evaluate() a million times with the same sigma and instance and get the same result every time.
- **Fail fast.** If sigma is invalid, raise immediately. Do not compute anything.
- **Test against hand-computed examples** before moving on.

### 5.3 Testing the Evaluator

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

All tests must pass before moving to Section 6.

### 5.4 Key Pitfall

**Pitfall:** Assuming setup cost and tardiness are on the same scale. They are not. On n=50 instances, tardiness can be 500+, while setup cost is typically 50–200. Combining them without normalisation means setup cost is invisible in the composite score. The code comment `# Normalise by instance mean` is not a suggestion — it is critical. Document how you compute the normalisation constant (e.g., empirical mean from baselines) in Chapter 3 of your dissertation.

---

## 6. Baseline Heuristics

**Purpose:** Baselines are essential. Examiners expect you to compare against something simple — if your hybrid doesn't beat the baselines, it is not a contribution. Build these heuristics now.

### 6.1 What You Will Build

**File:** `src/heuristics.py`

**Functions to implement:**

1. **`spt(instance: dict) → list`** (Shortest Processing Time)
   - Sort all jobs (0..n-1) by processing time ascending.
   - Assign jobs round-robin: job 0 → machine 0, job 1 → machine 1, ..., job m → machine 0, etc.
   - Return sigma (list of m machine sequences).
   - Example: n=5, m=2, sorted jobs [2, 0, 4, 1, 3] → sigma = [[2, 4, 1], [0, 1, 3]].
   - **Call `validate_sigma(sigma, instance['n'])` before returning.**

2. **`nearest_neighbour_greedy(instance: dict) → list`** (NN Greedy)
   - Maintain an empty schedule with m machines.
   - For each job in order (0, 1, 2, ...):
     - Find the machine where the transition cost from the last job on that machine is lowest.
     - If a machine is empty, transition cost is 0.
     - Assign this job to that machine.
   - Return sigma.
   - **Call `validate_sigma(sigma, instance['n'])` before returning.**

### 6.2 Testing Heuristics

Create `tests/test_heuristics.py`.

**Test for spt():**
- Generate instance with n=4, m=2, seed=42.
- Call `spt(instance)`.
- Verify all 4 jobs appear exactly once across machines.
- Verify the returned sigma is a valid list of lists.

**Test for nearest_neighbour_greedy():**
- Same instance, call `nn_greedy(instance)`.
- Verify all 4 jobs appear exactly once.
- Spot-check: the first job (index 0) should go to machine 0 (no prior jobs, cost 0).

**Verification command:**
```bash
python -m pytest tests/test_heuristics.py -v
```

### 6.3 Key Pitfall

**Pitfall:** Assuming all jobs must be processed in order 0, 1, 2, .... SPT reorders jobs by processing time, then assigns that reordered sequence round-robin. Nearest-neighbour assigns jobs in order 0, 1, 2, ... but decides which machine each goes to by minimising transition cost. These are fundamentally different algorithms — do not confuse them.

---

## 7. Genetic Algorithm

**Purpose:** GA is your baseline solver. It must work before the DRL agent can control it. Build, tune, lock down parameters.

### 7.1 What You Will Build

**File:** `src/ga.py`

**Functions to implement:**

1. **`decode_chromosome(chromosome: list, m: int) → list`**
   - Input: a permutation of n job indices, number of machines m.
   - Split into m contiguous segments: sigma[0] = chromosome[0 : n//m], sigma[1] = chromosome[n//m : 2*(n//m)], ...
   - If n % m != 0, distribute remainder: final machines get one extra job each.
   - Return sigma (list of m machine sequences).
   - Example: n=5, m=2, chromosome=[2,0,4,1,3] → sigma=[[2,0,4], [1,3]].

2. **`make_fitness_fn(instance: dict, alpha: float = 0.5) → callable`**
   - Return a fitness function that takes a chromosome, decodes it, evaluates it, returns composite fitness.
   - This factory exists because DEAP needs a callable to pass to the toolbox.
   - The returned function should call your evaluator.

3. **`build_toolbox(n: int, instance: dict, alpha: float = 0.5) → deap.base.Toolbox`**
   - Create and register:
     - Attribute (individual job indices): `toolbox.attr_job = tools.initRepeatContainer(range, n, container=list)`
     - Individual (permutation): `toolbox.individual = tools.initPermutation(...)`
     - Population: `toolbox.population`
     - Evaluation: `toolbox.register("evaluate", fitness_fn)`
     - Crossover: `toolbox.register("mate", tools.cxOrdered)`
     - Three mutation operators:
       ```python
       def swap_mutation(individual, indpb=0.05):
           for i in range(len(individual)):
               if random.random() < indpb:
                   j = random.randint(0, len(individual)-1)
                   individual[i], individual[j] = individual[j], individual[i]
           return individual,
       
       def inversion_mutation(individual, indpb=0.05):
           for _ in range(int(len(individual) * indpb)):
               i, j = sorted(random.sample(range(len(individual)), 2))
               individual[i:j+1] = individual[i:j+1][::-1]
           return individual,
       
       def aggressive_swap_mutation(individual, indpb=0.20):
           # Same as swap_mutation but with higher indpb
       ```
     - Register: `toolbox.register("mutate_swap", swap_mutation)`, etc.
     - Selection: `toolbox.register("select", tools.selTournament, tournsize=3)`
   - **Guard against DEAP global state:** Before any creator.create() or tools call, check `hasattr(creator, "FitnessMin")`. Only create if not already exists.
   - Return the toolbox.

4. **`run_ga(n: int, m: int, instance: dict, pop_size: int = 100, num_gens: int = 200, cx_prob: float = 0.8, mut_prob: float = 0.2, mutation_ops: list = ["swap"], seed: Optional[int] = None) → tuple`**
   - Build toolbox.
   - Initialise population of size pop_size.
   - Run generational GA:
     - For each generation:
       - Select parents via tournament.
       - Apply crossover with probability cx_prob.
       - Apply mutation (one of the mutation_ops chosen at random) with probability mut_prob.
       - Evaluate all individuals.
       - Replace population with best+offspring (generational replacement).
     - Track best fitness each generation.
   - Return (best_individual, best_fitness_per_gen, final_population).

### 7.2 Hyperparameter Grid

**Tune on n=10, m=2 ONLY.** Use instance seed=42 for consistency.

| Pop Size | Generations | Cx Prob | Mut Prob | Mutation Ops | Notes |
|---|---|---|---|---|---|
| 50 | 100 | 0.7 | 0.1 | [swap] | Baseline: small & fast |
| 100 | 200 | 0.8 | 0.2 | [swap, inversion] | Balanced exploration+exploitation |
| 200 | 500 | 0.9 | 0.3 | [swap, inversion, aggressive] | Large: more time to search |

**Tuning procedure (in notebook 03_ga_development.ipynb):**
1. For each row in the grid:
   - Run GA 10 times with seeds 0–9.
   - Record mean composite fitness and std.
   - Plot convergence curves (best fitness vs generation).
2. Pick the configuration with lowest mean fitness.
3. Lock these parameters in all subsequent experiments (baselines and hybrid).
4. Document the choice in your dissertation Chapter 4.

**Verification command:**
```bash
python -c "from src.ga import run_ga; best, hist, pop = run_ga(10, 2, instance, seed=42); print(f'Best: {best.fitness.values[0]:.2f}')"
```

### 7.3 What Good Convergence Looks Like

- Gen 1–40: steep drop (population finds improving solutions quickly).
- Gen 40–end: flatter (population exploits near best solution).
- No sudden spikes (means your crossover is broken, creating invalids).

If flat from gen 1: check that fitness function is returning different values (not constant).  
If never stops dropping: increase num_gens.

### 7.4 DEAP Global State: Multiprocessing Fix

**Critical for experiments:** When running `run_ga()` in parallel across multiple processes (via `multiprocessing.Pool`), DEAP's global `creator` object may cause problems in fork-mode on Linux.

**Solution:** Use `multiprocessing.get_context("spawn")` instead of default fork. Add this to your run_ga.py experiment script.

### 7.5 Key Pitfall

**Pitfall:** Restarting your kernel mid-session and re-running the build_toolbox() cell. DEAP's global state persists and causes "FitnessMin already exists" warnings. The hasattr guard prevents crashes but can cause silent inconsistencies. **Always restart the kernel fully before running GA code again.** In your terminal scripts (experiments/), each process starts fresh so this is not a problem.

---

## 8. DRL Hyper-Heuristic Agent

**Purpose:** The DRL agent is the novel contribution. It controls which GA mutation operator to use, learning a meta-policy. Train and validate it thoroughly before running final experiments.

### 8.1 What You Will Build

**File:** `src/ga_env.py` — Gymnasium environment wrapping the GA.

**File:** `src/drl_agent.py` — PPO training and inference.

### 8.2 The Gymnasium Environment (ga_env.py)

**Class:** `GAHyperHeuristicEnv(gym.Env)`

**Constructor:**
```python
def __init__(self, instance: dict, total_gens: int = 200, step_gens: int = 10, seed: Optional[int] = None):
```
- instance: one (n, m) instance dict.
- total_gens: total GA generations per episode.
- step_gens: number of GA gens per step (agent decision frequency).
- seed: random seed.

**Methods:**

1. **`reset() → tuple`**
   - Initialise fresh GA population.
   - Compute initial state observation (see 8.3).
   - Return (observation, info).

2. **`step(action: int) → tuple`**
   - Run GA for step_gens generations with the chosen mutation operator (action ∈ {0=swap, 1=inversion, 2=aggressive_swap}).
   - Compute new observation.
   - Compute reward: (best_fitness_before − best_fitness_after) / best_fitness_before.
     - If best_fitness_before is 0, reward = 0.
   - Set done=True if generations_elapsed >= total_gens.
   - Return (observation, reward, terminated, truncated, info).

3. **`close()`**
   - Clean up (if needed).

### 8.3 State Space (4D Observation)

Compute and normalise all four features:

1. **best_fitness_norm:** current best solution fitness / initial best solution fitness.
   - Range: [0.0, 1.0] (fitness improves over time, so this decreases).
   
2. **mean_fitness_norm:** current population mean fitness / initial best fitness.
   - Range: depends on diversity.
   
3. **diversity:** mean pairwise Hamming distance among a sample of min(20, pop_size) chromosomes, divided by n.
   - Hamming distance: count positions where two chromosomes differ.
   - Range: [0.0, 1.0].
   
4. **stagnation_norm:** (generations since best improved) / total_gens.
   - Range: [0.0, 1.0].

Return observation as `np.array([best_fitness_norm, mean_fitness_norm, diversity, stagnation_norm], dtype=np.float32)`.

### 8.4 Action Space

Discrete(3):
- Action 0: swap mutation (indpb=0.05)
- Action 1: inversion mutation (indpb=0.05)
- Action 2: aggressive swap mutation (indpb=0.20)

### 8.5 Training Details (drl_agent.py)

**Function:** `train_ppo(instance: dict, num_timesteps: int = 50000, seed: Optional[int] = None) → PPO`

1. Create env: `env = GAHyperHeuristicEnv(instance, total_gens=200, step_gens=10, seed=seed)`
2. **Run check_env():**
   ```python
   from stable_baselines3.common.env_checker import check_env
   check_env(env)
   ```
   Fix any warnings before proceeding.
3. Create PPO agent:
   ```python
   model = PPO(
       "MlpPolicy", env,
       n_steps=256,
       ent_coef=0.01,
       learning_rate=3e-4,
       verbose=1
   )
   ```
4. Train: `model.learn(total_timesteps=num_timesteps, log_interval=10)`
5. Monitor TensorBoard: `tensorboard --logdir=logs/`
6. Save: `model.save("models/ppo_n20m2")`
7. Return model.

**Function:** `run_hybrid(instance: dict, model: PPO) → dict`

1. Create env with same config.
2. Observation, info = env.reset().
3. For each step:
   - Action, _ = model.predict(observation, deterministic=True).
   - Observation, reward, done, _, info = env.step(action).
4. Return final best solution and fitness.

### 8.6 Training Instance Pool

**Critical:** Do not train on a single instance — the agent will overfit.

- Generate 5 instances with (n=20, m=2) and seeds 0, 1, 2, 3, 4.
- During training, each reset() picks a random instance from the pool.
- Wrap env with `gym.wrappers.RecordEpisodeStatistics` to track episode rewards.

### 8.7 Reward Sparsity Fallback

If reward is 0 for more than 50 consecutive steps during training:
- Reduce step_gens to ceil(step_gens / 2), floor at 1.
- This gives the agent more frequent decision points and a denser reward signal.

Log this adjustment in your code comments.

### 8.8 Training Duration

- **Minimum:** 50,000 timesteps (≈ 2,500 episodes at 20 steps each).
- **Stopping criterion:** If mean episode reward plateaus (no improvement for 10 consecutive log intervals), stop early.
- Monitor via TensorBoard. If reward is still rising at 50,000 ts, continue to 100,000 ts.

### 8.9 Post-Training Validation

After training, run the agent for 10 episodes on test instances (n=20, m=2, seeds 100–109 — different from training):
- Log the action chosen at each step.
- Plot action frequency histograms (early, middle, late thirds of episode).
- Expected pattern: inversion early (diversity), swap late (fine-tuning).
- If agent picks one action always → learning failed, debug reward signal.

### 8.10 Key Pitfall

**Pitfall:** Training on a single instance. The agent will learn to exploit that specific instance structure and fail to generalise. Training on an instance pool is harder to set up but essential for a valid contribution. Document this approach in your dissertation.

---

# PART 4 — RUN & ANALYSE

Run your experiments, collect results, and produce tables and figures for your dissertation.

## 9. Experiments & Statistical Analysis

**Purpose:** Run 720 experiments to validate your contribution. This is the core of your dissertation. Be meticulous and patient.

### 9.1 Experiment Design

| Algorithm | Seeds | Instances | Runs | Notes |
|---|---|---|---|---|
| SPT | 0–29 | 6 configs | 180 | Baseline (fast, seconds each) |
| NN Greedy | 0–29 | 6 configs | 180 | Baseline (medium speed) |
| GA | 0–29 | 6 configs | 180 | Plain GA, locked params from tuning |
| Hybrid | 0–29 | 6 configs | 180 | GA + trained PPO agent |
| **Total** | | | **720** | |

Same seed → same instance for all algorithms. Paired design.

**Instance configs (all 6):**
- (n=10, m=2)
- (n=10, m=3)
- (n=20, m=2)
- (n=20, m=3)
- (n=50, m=2)
- (n=50, m=3)

### 9.2 Pre-Experiment Benchmark (REQUIRED)

Before committing to 30 seeds, estimate runtime:
1. Run SPT on all 6 configs once (seed=42): measure time.
2. Run NN greedy on all 6 configs once: measure time.
3. Run GA on n=50, m=2 and n=50, m=3 once: measure time.
4. Extrapolate: 30 seeds × (SPT time + NN time + GA time) = total runtime estimate.

Example: if one full cycle is 2 minutes, 30 cycles ≈ 1 hour. Hybrid is similar to GA.

**Do not skip this.** Running 720 experiments without knowing the duration is asking for a late-night disaster.

### 9.3 Result File Format

**Directory:** `results/raw/`

**Filename pattern:** `{algorithm}_{config_label}_{seed}.json`

Example: `spt_n20m2_seed0042.json`

**File contents (JSON):**
```json
{
  "algorithm": "SPT",
  "instance_label": "n20m2",
  "seed": 42,
  "composite_fitness": 234.5,
  "weighted_tardiness": 145.3,
  "setup_cost": 89.2,
  "makespan": 478.9,
  "runtime_seconds": 0.023
}
```

**Why every individual run?** You need the raw data for Wilcoxon tests and box plots. Aggregated means alone are useless.

### 9.4 Running Experiments (Terminal Scripts)

**Do not run from notebooks.** Write three scripts in `experiments/`:

1. **`run_baselines.py`**
   ```python
   from multiprocessing import Pool, get_context
   import json
   from src.instance_generator import generate_instance
   from src.heuristics import spt, nn_greedy
   from src.evaluator import evaluate
   
   def run_spt(seed, config):
       instance = generate_instance(config['n'], config['m'], seed=seed)
       sigma = spt(instance)
       result = evaluate(sigma, instance)
       return {
           "algorithm": "SPT",
           "instance_label": config['label'],
           "seed": seed,
           "composite_fitness": result['composite'],
           "weighted_tardiness": result['weighted_tardiness'],
           "setup_cost": result['setup_cost'],
           "makespan": result['makespan'],
       }
   
   # Similar for NN greedy
   
   # Main: use Pool.imap_unordered to run in parallel
   with get_context("spawn").Pool(8) as pool:
       for result in pool.imap_unordered(...):
           save_result(result)
   ```

2. **`run_ga.py`**
   ```python
   # Similar structure, uses your GA code
   # Critical: use get_context("spawn") not fork
   ```

3. **`run_hybrid.py`**
   ```python
   # Same as run_ga.py but calls the trained PPO agent in step()
   # Load model with PPO.load("models/ppo_n20m2")
   ```

**Run from terminal:**
```bash
python experiments/run_baselines.py  # ~15 min
python experiments/run_ga.py         # ~2–3 hours
python experiments/run_hybrid.py     # ~2–3 hours
```

### 9.5 Statistical Testing

**Test:** Wilcoxon signed-rank test (non-parametric, paired).

For each instance size (6 total) and each algorithm pair (3 pairs: hybrid vs SPT, hybrid vs NN, hybrid vs GA):
```python
import pingouin as pg

# Load results for this config
hybrid_scores = [...]  # 30 runs
baseline_scores = [...]  # 30 runs

stat, p_value = pg.wilcoxon(hybrid_scores, baseline_scores, alternative="less")
# "less": test if hybrid < baseline (lower is better)
```

**Significance threshold:** p < 0.05.

**Reporting:**
- p < 0.001: "statistically significant (p < 0.001)"
- 0.001 ≤ p < 0.01: "statistically significant (p < 0.01)"
- 0.01 ≤ p < 0.05: "statistically significant (p < 0.05)"
- p ≥ 0.05: "not statistically significant (p = X.XXX)"

Do not hide non-significant results. They are informative.

### 9.6 Sensitivity Analysis (α Variation)

After main results, re-run GA and Hybrid on medium instances (n=20, m=2 and n=20, m=3) with three alpha values:
- α = 0.3 (weight tardiness less)
- α = 0.5 (balanced, main results)
- α = 0.7 (weight tardiness more)

Re-run 10 seeds per (config, alpha) pair. This is fast (only 40 extra runs). Show a table:

| Config | Algorithm | α=0.3 | α=0.5 | α=0.7 |
|---|---|---|---|---|
| n20m2 | GA | ... | ... | ... |
| n20m2 | Hybrid | ... | ... | ... |

Discuss: does the hybrid's advantage persist across all α? Or is it sensitive to the weighting?

---

## 10. Visualisations

### 10.1 Gantt Charts

A Gantt chart shows the schedule visually: each machine is a row, each job is a horizontal bar, its position is its start time, its width is its processing time. Colour-code the bars by the job's colour class. Show setup times as hatched grey blocks between jobs.

Always show a side-by-side comparison: the worst-performing baseline (SPT) on the left, your hybrid on the right, same instance. The examiner should be able to see visually that your hybrid groups similar colours together while SPT scatters them randomly. This is your most intuitive result.

Export figures as PDF (vector graphics), not PNG. PDF figures scale perfectly in LaTeX. Use `plt.savefig("name.pdf", dpi=150, bbox_inches="tight")`.

### 10.2 Convergence Curves

Plot best fitness vs generation for the plain GA and (conceptually) for the hybrid. For the hybrid, you can plot best fitness at each checkpoint (every `step_gens` generations). Show that the hybrid converges faster or to a lower value than the plain GA. Include mean and best on the same plot with different line styles.

Show convergence for at least three instance sizes (small, medium, large) in a grid of subplots. The convergence behaviour will be qualitatively different across sizes — on small instances, both may converge quickly; on large instances, the hybrid's advantage should be more visible.

### 10.3 Box Plots

For your main comparison, box plots communicate more than a table. One subplot per instance configuration, four boxes per subplot (SPT, NN, GA, Hybrid). The box shows the interquartile range over 30 runs, the whiskers show the range, and outliers are dots. A good result will show the hybrid's box systematically lower than the others.

### 10.4 Action Frequency Analysis

For the trained PPO agent, run it on representative instances and record which action it chose at each step across multiple runs. Plot a bar chart showing action selection frequency over the course of an episode (early, middle, late thirds). This is your qualitative evidence that the agent has learned a meaningful policy.

### 10.5 Cost Matrix Heatmap

In your exploration notebook (and possibly Chapter 3 of your dissertation), plot the transition cost matrix S as a heatmap for a small instance. The asymmetry should be visually obvious — the upper triangle (dark→light) should be substantially more colourful than the lower triangle (light→dark).

---

# PART 5 — WRITE & SUBMIT

Write your dissertation, create your final code archive, and submit by the deadline.

## 11. Dissertation Writing

### 11.1 Chapter Structure and Target Lengths

| Chapter | Content | Target pages |
|---|---|---|
| 1 — Introduction | Motivation, research question, contributions, outline | 4–6 |
| 2 — Literature Review | As per Section 3 of this guide | 12–16 |
| 3 — Problem Formulation | Formal PMSP-SDSC model, notation table, instance generator description | 5–7 |
| 4 — System Design & Implementation | Architecture, GA design, GA-env design, PPO design | 15–20 |
| 5 — Experimental Evaluation | Setup, results, statistical tests, visualisations, analysis | 12–18 |
| 6 — Discussion | What worked, limitations, threats to validity, future work | 5–8 |
| 7 — Conclusion | Restate contributions, summarise findings, closing | 2–3 |
| References | 15–25 sources | — |

Leeds MSc dissertations are typically 15,000–20,000 words. Check the School of Computer Science guidelines on Minerva for the exact requirement for COMP5200M.

### 11.2 LaTeX Setup

Use LaTeX. Ask your supervisor whether Leeds provides an official dissertation template — many programmes do. If not, use the `report` class with 12pt font, 25mm margins, and these packages:
```latex
\usepackage{amsmath, amssymb, graphicx, booktabs, hyperref, natbib}
\usepackage{algorithm, algpseudocode}  % For pseudocode (NOT algorithm2e)
```

For your bibliography: use Zotero to export a `.bib` file. Check every entry for formatting errors before including.

### 11.3 Writing Chapter 3 (Problem Formulation) — 5–7 Pages

**Must include:**
1. Formal problem definition: PMSP-SDSC as a minimisation problem with notation table.
2. Instance generator explanation: colour classes, darkness ranking, cost matrix asymmetry rule.
3. Objective function: F = α·f1_norm + (1−α)·f2_norm. **Explicitly explain the normalisation approach** — this addresses a key ambiguity.
4. Solution representation: sigma as list of m machine sequences.

**Exact subsections:**
- 3.1 Parallel Machine Scheduling with Sequence-Dependent Setup Costs
- 3.2 Problem Formalisation (notation table)
- 3.3 Instance Generation & Asymmetric Cost Structure
- 3.4 Composite Objective & Normalisation

### 11.4 Writing Chapter 4 (System Design & Implementation) — 15–20 Pages

This is the chapter your technical examiner will scrutinise most. Key requirements:

**Pseudocode, not code.** Present all algorithms as formal pseudocode using `algpseudocode` package (NOT algorithm2e). Your GA's main loop, fitness evaluation, Gymnasium env's step function, and PPO training should all appear as formal algorithms. Code snippets in appendices only.

**Every parameter in a table.** Create two parameter tables:
- **GA parameters:** pop_size, num_gens, cx_prob, mut_prob, tournament_size, with tuning justification
- **PPO parameters:** learning_rate, n_steps, ent_coef, total_timesteps, with reasoning

**Justify every design choice.** Not "we used OX crossover" but "OX crossover was selected because it preserves job orderings, which encode colour transition information critical to the domain."

**System architecture diagram.** One diagram showing: Instance Generator → Instance Dict → Evaluator → GA (with 3 mutation ops) → GA Env → PPO → selected action → back to GA. Reference this throughout.

**Exact subsections:**
- 4.1 Evaluator & Objective Function
- 4.2 Baseline Heuristics (SPT & NN Greedy) — brief, use pseudocode
- 4.3 Genetic Algorithm Design (chromosome, operators, DEAP setup) — pseudocode for main loop
- 4.4 Gymnasium Environment for Hyper-Heuristic (observation space, action space, reward)
- 4.5 PPO Agent Training & Inference
- 4.6 Parameter Tables & Justification

### 11.5 Writing Chapter 5 (Experimental Evaluation) — 12–18 Pages

**Structure:**
1. **5.1 Experiment Setup** — 2 pages. Repeat instance configs, seeds, runtimes, computational environment.
2. **5.2 Main Results** — 4 pages. Lead with your headline result in the first sentence, then tables, Gantt charts, convergence plots.
3. **5.3 Statistical Analysis** — 2 pages. Wilcoxon test results for all pairs and instance sizes. State p-values explicitly.
4. **5.4 Sensitivity Analysis** — 2 pages. Results for α = 0.3, 0.5, 0.7. Discuss robustness.
5. **5.5 Action Frequency Analysis** — 2 pages. PPO agent behaviour: which actions when? Does it show learned policy?

**First sentence rule:** Lead with your main result, not the methodology. Bad: "We ran 720 experiments across 6 instance sizes." Good: "The hybrid GA-DRL system achieves 23% lower composite fitness than SPT (p < 0.001) and 18% lower than plain GA (p = 0.003) on all instance sizes except n=10 where all algorithms perform similarly."

**Be precise with numbers.** Not "The hybrid performed better" but "The hybrid achieved mean composite fitness 47.3 ± 8.1 on (n=20, m=2), a 23.4% reduction relative to SPT (82.1 ± 12.4, p < 0.001, Wilcoxon signed-rank test)."

### 11.6 Writing Chapter 6 (Discussion) — 5–8 Pages

This chapter separates a good dissertation from a great one. Key sections:

**6.1 Interpretation of Results** — Why did the hybrid win/lose? Explain the mechanism: the PPO agent learned to select inversion mutations early when diversity is high (exploring the search space) and switch to swap mutations late when the population converges (fine-tuning). **Show this with your action-frequency analysis.**

**6.2 Non-Significant Results** — If the hybrid doesn't beat plain GA on n=10, don't hide it. Explain: small instances have such limited search spaces that even plain GA converges near-optimally in <10 generations, leaving little for the hyper-heuristic to improve. This is an informative finding, not a failure.

**6.3 Threats to Validity** — Acknowledge:
- Instances are synthetic, not from real textile dyeing
- PPO training and evaluation use the same instance distribution
- 30 runs is standard for GA literature but a relatively small sample
- The agent was not tested on truly unseen instance distributions

**6.4 Limitations** — Discuss scope:
- Only tested with parallel machine scheduling; generalisability to other combinatorial optimisation problems unknown
- Computational cost of training PPO not included in experimental times
- Single-size training (n=20, m=2) may not transfer well to very large instances

**6.5 Future Work** — Suggest:
- Curriculum learning: train agent on small instances, gradually scale up
- Real data: dyehouse scheduling problem from actual textile mills
- Stronger baselines: branch-and-bound for small instances, simulated annealing
- Extended objectives: add energy consumption, worker scheduling constraints

### 11.7 The Abstract

Write this last (≈250 words). Must cover, in order:
1. Problem: parallel machine scheduling with sequence-dependent setup costs
2. Why hard: NP-hard, large combinatorial search space
3. Approach: hybrid GA-DRL system where PPO hyper-heuristic controls GA mutation operator selection
4. Results: specific numbers (X% improvement, p-value, instance sizes where significant)
5. Implication: hyper-heuristics can effectively meta-control metaheuristics for complex scheduling

Example opening: "Machine scheduling with sequence-dependent setup costs is an NP-hard combinatorial optimisation problem with applications in textile manufacturing and other domains. We propose a hybrid approach where a Proximal Policy Optimisation agent learns to dynamically select genetic algorithm mutation operators as a hyper-heuristic meta-controller..."

### 11.8 Writing Rules

- **Numbers:** Never "the hybrid performed better." Always include the metric, baseline, and p-value.
- **Figures:** Every figure needs a self-contained caption. Someone should understand it without reading surrounding text.
- **Citations:** Every claim needs either a citation or your experimental evidence. No unsupported assertions.
- **Consistency:** Pick one name for your system ("the hybrid GA-DRL system" or "the proposed hyper-heuristic") and use it throughout. Don't switch between "the agent," "the PPO," and "the hybrid."
- **Proofreading:** Print the draft and proofread on paper. You catch different errors than on screen.

---

## 12. Submission Checklist

### Code Quality

- [ ] All `src/` modules have docstrings explaining their purpose and the format of inputs/outputs
- [ ] `validate_sigma()` is called inside `evaluate()` — invalid solutions are caught immediately
- [ ] All random seeds are explicitly set and logged in result files
- [ ] `requirements.txt` is up to date
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] All notebooks run cleanly from top to bottom after a full kernel restart (no hidden state)

### Experiments

- [ ] 30 runs × 6 instance configs × 4 algorithms completed (720 total)
- [ ] All raw results saved to `results/raw/` as JSON or CSV with seed and instance label recorded
- [ ] Wilcoxon tests computed and recorded for all algorithm pairs on all instance sizes
- [ ] All figures exported as PDF to `results/figures/`
- [ ] Sensitivity analysis (varying alpha) completed on at least medium instances

### Dissertation

- [ ] Word count within Leeds School of Computer Science guidelines
- [ ] All algorithms presented as formal pseudocode (not code blocks) in the main text
- [ ] Parameter table present for GA hyperparameters
- [ ] Parameter table present for PPO hyperparameters
- [ ] Every figure referenced in the text and has a self-contained caption
- [ ] References formatted consistently — check every entry
- [ ] Ethics form included (Appendix A — already done in your SP)
- [ ] Full results tables in Appendix B if main text tables are abbreviated
- [ ] PDF compiles cleanly with no missing references or figures
- [ ] Spell-checked and proofread
- [ ] Submitted via Minerva by 31 August 2026

### Submission Files

- `GOGULA26-Dissertation.pdf` — main submission via Minerva
- `GOGULA26-Code.zip` — all `src/`, `notebooks/`, `experiments/`, `requirements.txt`, and a `README.md`. Do not include `results/raw/` (too large) or trainer model files; include only the final trained PPO model.

### README.md Structure

Your README must have these exact sections so examiners can reproduce experiments:

```markdown
# Hybrid GA-DRL for Machine Scheduling

## Setup

1. Create conda environment: `conda env create -f environment.yml`
2. Activate: `conda activate scheduling`
3. Install: `pip install -r requirements.txt`

## Reproducing Results

### Baselines (SPT, NN Greedy)
```bash
python experiments/run_baselines.py
```
Outputs: 360 JSON files to `results/raw/` (n×6 configs × 2 algorithms × 30 seeds).

### Genetic Algorithm
```bash
python experiments/run_ga.py
```
Outputs: 180 JSON files (6 configs × 30 seeds).

### Hybrid GA-DRL (requires PPO model pre-training)
First train the PPO agent:
```bash
python -c "from src.drl_agent import train_ppo; from src.instance_generator import generate_instance; train_ppo(generate_instance(20, 2, seed=0), num_timesteps=50000)"
```
Then run experiments:
```bash
python experiments/run_hybrid.py
```
Outputs: 180 JSON files.

## Visualisations & Analysis

See `notebooks/05_final_evaluation.ipynb` for:
- Gantt chart comparisons
- Convergence curves
- Box plots
- Wilcoxon test results
- Sensitivity analysis (α = 0.3, 0.5, 0.7)

## Key Files

- `src/instance_generator.py` — synthetic instance generation
- `src/evaluator.py` — solution evaluation (composite fitness)
- `src/heuristics.py` — SPT and NN greedy baselines
- `src/ga.py` — genetic algorithm
- `src/ga_env.py` — Gymnasium environment (GA state observation, mutation action, reward)
- `src/drl_agent.py` — PPO training and inference
- `experiments/run_*.py` — terminal-based experiment runners (not notebooks)

## Results

Final results will be in `results/raw/` as JSON. Load and analyse with `notebooks/05_final_evaluation.ipynb`.
```

**Key:** This README is not for showing off — it is for reproducibility. Examiners will try to run your code. Make sure these three commands work.

---

## Quick Reference: Timeline vs This Guide

| Milestone | Period | Sections |
|---|---|---|
| M1: Literature review + environment model | Mar–Apr (Weeks 20–22) | §3, §4, §5, §6 |
| M2: GA baseline | May–Jun (Weeks 25–28) | §7 |
| M3: DRL agent + integration | Jun–Jul (Weeks 29–S2) | §8 |
| M4: Evaluation + write-up | Jul–Aug (Weeks S3–S7) | §9, §10, §11, §12 |
