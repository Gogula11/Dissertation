# COMP5200M — Project Guide
## An Intelligent Hybrid Approach to Machine Scheduling using RL and Genetic Algorithms

> **How to use this document:** This is your end-to-end project guide covering everything from environment setup to dissertation submission. It tells you *what* to do, *why*, and *what to watch out for* at every stage — but does not contain code. For scaffolding code to reference while building each module, see the companion document `COMP5200M_Scaffolding.md`.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)
3. [Literature Review](#3-literature-review)
4. [Problem Formalisation & Instance Generator](#4-problem-formalisation--instance-generator)
5. [The Evaluator](#5-the-evaluator)
6. [Baseline Heuristics](#6-baseline-heuristics)
7. [Genetic Algorithm](#7-genetic-algorithm)
8. [DRL Hyper-Heuristic Agent](#8-drl-hyper-heuristic-agent)
9. [Experiments & Statistical Analysis](#9-experiments--statistical-analysis)
10. [Visualisations](#10-visualisations)
11. [Dissertation Writing](#11-dissertation-writing)
12. [Submission Checklist](#12-submission-checklist)

---

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

## 3. Literature Review

### 3.1 What You Need to Cover

Your SP cites four papers. The dissertation literature review needs roughly 15–25 sources, and they need to be woven into a coherent narrative, not listed as summaries. The topics to cover, in the order they should appear in Chapter 2:

**Parallel machine scheduling (PMSP).** Start broad. Establish that PMSP is NP-hard in the general case and explain what makes your variant — asymmetric, sequence-dependent setup costs across parallel machines — particularly difficult. The combinatorial explosion of the search space should be quantified.

**Sequence-dependent setup costs.** This is what makes your problem non-trivial compared to classical PMSP. Explain the asymmetry and why standard heuristics like SPT treat all transitions as equivalent (which is why they underperform).

**Classical heuristics.** Cover SPT, NEH (Nawaz-Enscore-Ham), and nearest-neighbour greedy. Explain their time complexity and why they are used as baselines — fast and deterministic — not because they are good.

**Genetic algorithms for scheduling.** Focus on chromosome representations for scheduling problems (permutation-based). Order crossover (OX) and its preservation of relative job ordering. Mutation operators. The tension between exploration and exploitation.

**Deep Reinforcement Learning for combinatorial optimisation.** Cover the general paradigm shift from hand-crafted to learned policies. PPO specifically — why it is more stable than DQN for continuous or structured action spaces. Briefly cover prior work applying DRL to scheduling (there is growing literature from 2019 onwards).

**Hyper-heuristics.** Define the concept formally: a heuristic that selects or generates other heuristics. Frame your DRL agent as a hyper-heuristic operating at the meta-level of the GA, not at the level of individual schedules. This framing is what makes your contribution distinct from just "RL for scheduling."

**The gap.** End the literature review with a paragraph that explicitly states: prior work has combined GA and RL in scheduling, but not specifically as a PPO hyper-heuristic controlling mutation operator selection in an asymmetric parallel machine scheduling problem. This gap is what your dissertation fills.

### 3.2 Where to Search

Use Google Scholar, IEEE Xplore, and ScienceDirect. For the most recent DRL-for-scheduling work (2022–2025), arXiv cs.AI and cs.LG will have preprints before journal publication. Zotero is the right tool for managing references — install the browser connector, attach PDFs, and write a 2–3 sentence note on each paper explaining what it contributes to your work specifically.

### 3.3 Key Papers to Find Beyond Your SP

Beyond what you already cited, look specifically for:

- The NEH heuristic original paper (Nawaz, Enscore, Ham, 1983) — foundational baseline
- Pinedo's "Scheduling: Theory, Algorithms, and Systems" textbook — the standard reference your examiner will expect to see cited for problem formalisation
- Recent (2020–2024) papers on attention-based or transformer models for scheduling — even if you're not using them, acknowledging this direction and explaining why you chose the GA-DRL hybrid instead shows breadth
- Burke et al. on hyper-heuristics (2013, JORS) — the standard definition paper for the field

### 3.4 What Not to Do

Do not write the literature review as a series of "Paper X found Y. Paper Z found W." paragraphs. Your examiner will notice. Write it as a story: here is the problem class, here is what people tried, here is what worked, here is what the limitations were, here is what nobody has done yet. Every paper you cite should be serving that narrative.

---

## 4. Problem Formalisation & Instance Generator

### 4.1 Why You Write This First

The instance generator is the foundation everything else sits on. The GA needs instances to optimise. The heuristics need instances to evaluate. The evaluator needs instances to score solutions. If your instance generator has a bug — say, due dates that are always achievable with zero tardiness regardless of scheduling — your entire experimental comparison is meaningless. Write it first, understand it completely, test it thoroughly.

### 4.2 What an Instance Represents

An instance is a single snapshot of the scheduling problem: a specific set of jobs with specific processing times, due dates, release times, weights, and a specific transition cost matrix. In your experiments, each seed generates a unique instance. Two runs with the same seed must produce identical instances — this is what makes your 30-run comparisons paired (same instances, different algorithm behaviour).

### 4.3 The Transition Cost Matrix

This is the most domain-specific part of your model and the one your examiner will scrutinise most. The asymmetry must be intentional and explainable. The rule is: transitioning from a darker colour to a lighter colour costs more waste than the reverse, because residual dark dye contaminates the lighter batch.

Concretely, define a darkness ranking for your colour palette (e.g. white=1, yellow=2, light blue=3, green=4, red=5, navy=6, black=7). The cost of transitioning from colour A to colour B should scale with max(0, darkness(A) − darkness(B)). Same colour should cost zero. Add a small noise term so the matrix is not degenerate (identical costs create trivial structure that makes the problem easier than reality).

The setup time matrix can follow the same structure at a different scale — setup time and setup cost are distinct things in your formal model.

### 4.4 Due Date Tightness

Due dates need to be calibrated carefully. If they are too loose, every algorithm achieves zero tardiness and the only thing you're measuring is setup cost. If they are too tight, every algorithm has massive tardiness and again differentiation is hard. A tightness factor that scales due dates relative to average machine load is standard practice. Test your due date generation by running your heuristics on a few instances and checking that tardiness is nonzero but not catastrophically large.

### 4.5 Testing the Instance Generator

At minimum, verify:
- Processing times are positive and within your defined range
- The cost matrix diagonal is exactly zero (no cost transitioning to the same job)
- The matrix is genuinely asymmetric: S[i][j] ≠ S[j][i] for most pairs
- Due dates are positive
- The same seed always produces the same instance

---

## 5. The Evaluator

### 5.1 Design Principle

The evaluator must be pure — given a solution and an instance, it returns numbers. No randomness, no state, no side effects. This is the most important design decision in the whole project. If your evaluator has any stochastic behaviour, your entire experiment is broken and you will not be able to diagnose it.

### 5.2 Solution Representation

A solution sigma is a list of m lists, where sigma[k] is the ordered sequence of job indices assigned to machine k. This is sometimes called the "partition-permutation" representation. Every job index must appear in exactly one machine sequence — not zero, not two. Write a validation function that checks this and call it inside your evaluator. If you have a bug in your GA's crossover operator that occasionally produces invalid chromosomes, this check will catch it immediately.

### 5.3 Computing Completion Times

The completion time of a job depends on: when the previous job on the same machine finished, the setup time between the previous job and this job, the current job's processing time, and whether the job's release time has passed. The completion time formula is recursive within each machine sequence. Write this carefully and test it against hand-computed examples.

### 5.4 The Composite Objective

Your fitness function is F(Σ) = α·f1 + (1−α)·f2 where f1 is total weighted tardiness and f2 is total setup cost. Alpha is a tunable parameter. In your main experiments, use α=0.5 as the default. In your sensitivity analysis, vary it and show how the algorithm rankings change (or don't). This is a good dissertation discussion point — does the hybrid agent's advantage persist regardless of how much you weight tardiness vs setup cost?

### 5.5 What Else to Track

Beyond the composite score, always compute and store: makespan (C_max), total weighted tardiness alone (f1), total setup cost alone (f2), and tardiness per job. You want these separately because your dissertation results section will need to show that the improvement in composite score is not entirely driven by one objective at the expense of the other.

### 5.6 Testing the Evaluator

Test against hand-computed examples. Take n=3, m=1, simple processing times, simple setup times with no release times. Compute the completion times and tardiness by hand with a calculator. Run your evaluator and verify it matches. This is the only way to be confident the evaluator is correct. If it has a bug, every experiment you run is wrong.

---

## 6. Baseline Heuristics

### 6.1 Why Baselines Matter

Your contribution is only meaningful relative to what already exists. SPT is the simplest possible baseline — it completely ignores the structure of your problem (the asymmetric transition costs). Nearest-neighbour greedy uses some of that structure. A plain GA uses a lot more. Your hybrid should beat all three. If it doesn't beat all three consistently, that is also a valid finding — but you need to understand why.

### 6.2 SPT

Shortest Processing Time: sort all jobs by processing time ascending, assign round-robin across machines. Time complexity O(n log n). The key thing to understand about SPT is that it minimises mean completion time on a single machine — a classical result. On parallel machines with sequence-dependent setup costs, it has no theoretical guarantee. It is your weakest baseline and your hybrid should beat it comfortably.

### 6.3 Nearest-Neighbour Greedy

This heuristic explicitly uses the transition cost matrix. Starting from an empty schedule, it always assigns the next job to the machine where the transition cost from the last job is lowest. It is greedy — it makes locally optimal choices without lookahead — so it can get trapped in poor global solutions. Time complexity O(n² × m) roughly. This is your stronger baseline and the more interesting comparison.

### 6.4 Validation

After every heuristic, call your solution validation function. Heuristics are simple enough that bugs usually produce obviously invalid solutions (e.g. a job scheduled twice, or not at all). Catch this early.

---

## 7. Genetic Algorithm

### 7.1 Chromosome Representation

You are using the "giant tour" representation: a flat permutation of all n job indices, which is then split into m contiguous segments (one per machine). This is the standard approach for parallel machine scheduling GAs because it handles the assignment problem implicitly — you never need to decide which job goes to which machine separately from the ordering.

The split can be a fixed equal partition or you can evolve the split points as well. Start with a fixed equal split — it is simpler and your SP describes it this way. If you have time and want a stronger contribution, try evolving the split points and see if it helps.

### 7.2 Crossover

Order Crossover (OX) is the right operator for permutation chromosomes. It preserves the relative order of jobs between parents, which is important because the sequencing structure carries problem-relevant information. Do not use single-point or two-point crossover on permutations — they produce invalid chromosomes (duplicate jobs) without repair.

DEAP has OX implemented as `tools.cxOrdered`. Use it.

### 7.3 Mutation

Implement and register at least two mutation operators — you need them for the DRL agent to switch between:

- **Swap mutation:** pick two positions in the chromosome at random and swap the job indices. Low disruption, good for fine-tuning near a good solution.
- **Inversion mutation:** pick a random subsequence and reverse it. Higher disruption, good for escaping local optima.
- **Aggressive swap:** swap mutation but with a higher per-gene probability. A third action for the DRL agent to choose when it needs to explore hard.

### 7.4 Selection

Tournament selection is standard and works well for scheduling GAs. Tournament size 3 is a reasonable default. Larger tournaments increase selection pressure (converges faster but risks premature convergence). Smaller tournaments are more random (slower convergence but more diversity).

### 7.5 The DEAP Global State Problem

DEAP uses module-level global state for the `creator` object. If you run `creator.create("FitnessMin", ...)` twice in the same Python session it throws a warning and potentially inconsistent behaviour. This is a known DEAP gotcha. Guard against it by checking `hasattr(creator, "FitnessMin")` before registering. In notebooks, be especially careful — if you restart the kernel and re-run cells partially, you can get into weird states. When in doubt, restart the kernel fully.

### 7.6 Hyperparameter Tuning

You must tune the GA before running your 30-rep experiments. The parameters to tune are: population size (try 50, 100, 200), number of generations (try 100, 200, 500), crossover probability (try 0.7, 0.8, 0.9), and mutation probability (try 0.1, 0.2, 0.3). Use a small instance (n=10, m=2) for tuning — fast to iterate. Run 10 seeds per configuration and pick the one with the lowest mean composite fitness. Lock those parameters in and use them for all subsequent experiments.

Do your tuning in notebook `03_ga_development.ipynb`. Plot convergence curves for each configuration so you can see not just the final fitness but how fast the GA converges and whether it plateaus early.

### 7.7 What Good Convergence Looks Like

A healthy convergence curve drops steeply in the first 20–30% of generations, then flattens as the population exploits the best solutions found. If your curve is completely flat from generation 1, your fitness function or mutation rate is wrong. If it never stops dropping, you probably need more generations. If it drops then shoots back up, your crossover operator is producing invalid solutions that are being penalised.

### 7.8 Multiprocessing

Your 30-run experiments must be parallelised with Python's `multiprocessing` module. **Do not run this from inside a notebook.** Multiprocessing in notebooks on Linux uses fork-mode and interacts badly with DEAP's global state. Write the experiment runner as a standalone `.py` script in `experiments/` and run it from your terminal. The `Pool.imap_unordered` pattern is clean for this — it lets you print progress as results come in rather than waiting for the whole batch.

---

## 8. DRL Hyper-Heuristic Agent

### 8.1 The Key Architectural Decision

The PPO agent does not schedule jobs directly. It controls the GA. At every decision point (every `k` generations), the agent observes the current state of the GA's population and selects a mutation operator for the next `k` generations. This is the hyper-heuristic framing.

This is a subtle but important distinction. A lot of DRL-for-scheduling work has the agent construct schedules directly (sequence one job at a time). Your approach is different: the agent is a meta-controller. This distinction needs to be clearly explained in your dissertation and is part of what makes your contribution novel relative to simpler hybrids.

### 8.2 State Space Design

The state vector is a 4-dimensional observation summarising the current GA population:

- **Best fitness (normalised):** The best solution's composite score, divided by the initial best fitness. Normalising prevents the agent from having to deal with very different scales across instance sizes.
- **Mean fitness (normalised):** The population average, same normalisation. The gap between best and mean tells the agent something about population diversity.
- **Population diversity:** Measure as mean pairwise Hamming distance between a sample of chromosomes, divided by n. High diversity = lots of exploration happening. Low diversity = population has converged.
- **Stagnation counter (normalised):** How many generations since the best fitness improved, divided by total generations. This tells the agent whether the GA is stuck.

Why these four? They capture exactly the information you'd want as a human watching the GA: is it converging? Is it stuck? Is the population diverse enough? Four features is also small enough for a simple MLP policy (which is what SB3's default `MlpPolicy` uses) to learn from quickly.

### 8.3 Action Space

Three discrete actions: swap mutation (conservative), inversion mutation (moderate disruption), aggressive swap mutation (high disruption). Three actions is enough to demonstrate that the agent learns a non-trivial policy. More actions would require more training.

### 8.4 Reward Design

The reward at each step is the relative improvement in best fitness: (fitness_before − fitness_after) / fitness_before. Positive when the GA improved, zero or negative when it didn't. This relative formulation is important — it makes the reward scale-independent across different instance sizes, which means you can train the agent on medium instances and (ideally) have it generalise to other sizes.

Reward design is the hardest part of any RL problem. If your agent is not learning, the first thing to check is whether the reward signal is too sparse (the GA rarely improves in a single step) or too noisy (the improvement varies so much between runs that the signal is uninformative). You may need to adjust `step_gens` (the number of GA generations between PPO decisions) to get a clean signal.

### 8.5 The Gymnasium Environment

Your GA env inherits from `gym.Env` and implements `reset()`, `step()`, and `close()`. The `reset()` method initialises a fresh GA population and returns the initial state. The `step()` method runs `step_gens` generations of the GA with the chosen mutation operator and returns the new state, reward, done flag, and info dict.

**Run `check_env()` on your environment before training PPO.** SB3's env checker will catch common mistakes: wrong observation shapes, out-of-range observations, incorrect done signal handling. Fix every warning it raises — warnings often become errors during training.

### 8.6 PPO Hyperparameters

SB3's defaults are reasonable starting points. The parameters most worth adjusting for your problem:

- `n_steps`: how many environment steps to collect before each PPO update. Your episodes are short (total_gens / step_gens steps each), so keep this modest.
- `ent_coef`: entropy coefficient, controls exploration. Start at 0.01. If the agent collapses to always selecting one action, increase it.
- `learning_rate`: 3e-4 is the standard default. Don't change this unless training is clearly unstable.

### 8.7 What Instance to Train On

Train on medium instances (n=20, m=2). Small instances are too simple — the agent won't learn anything generalising. Large instances make each episode very slow. Train on medium, evaluate on all sizes. If the agent generalises to small and large, that is a strong result worth highlighting in your dissertation.

### 8.8 How Long to Train

With `step_gens=10` and `total_gens=200`, each episode is 20 steps. To get 2,500 training episodes, you need 50,000 timesteps. Start there. Monitor the mean episode reward in TensorBoard — if it is still climbing at 50,000 timesteps, go to 100,000. If it plateaued early and is flat, training is done.

### 8.9 Diagnosing a Non-Learning Agent

If the reward curve is flat or random throughout training:

1. Check that your reward is actually varying — print a few reward values from manual environment interaction
2. Check population diversity is not immediately collapsing to zero (if it is, your GA is converging in 10 generations and there's nothing for the agent to control)
3. Check your observation normalisation — if observations are all near 1.0 or all near 0.0, the agent has no signal to learn from
4. Try increasing `ent_coef` to force more exploration
5. Try increasing `step_gens` so each action has a longer effect and a cleaner signal

### 8.10 Analysing What the Agent Learned

After training, run the agent on test instances and log which action it selects at each step. Plot the action frequency over the course of an episode. A well-trained agent should show a pattern: perhaps preferring inversion mutation early (high diversity, escape basins) and switching to conservative swap mutation later (fine-tuning near a good solution). If the agent just picks one action always, it has not learned a useful policy.

This action-frequency analysis belongs in your dissertation results section. It is qualitative evidence that the agent is doing something meaningful, not just accidentally performing well.

---

## 9. Experiments & Statistical Analysis

### 9.1 The 30-Run Protocol

Every algorithm on every instance configuration must be run 30 independent times with different random seeds (seeds 0 through 29). The same seed must produce the same instance for all algorithms — this is what makes comparisons paired. Paired comparisons are more statistically powerful than unpaired ones because instance-level difficulty cancels out.

### 9.2 Instance Configurations

Run all six configurations from your SP: n=10/m=2, n=10/m=3, n=20/m=2, n=20/m=3, n=50/m=2, n=50/m=3. That is 30 seeds × 6 configs × 4 algorithms = 720 experiment runs total. The GA and hybrid runs on n=50 will be the slowest — estimate your runtime before committing to 30 reps so you're not surprised.

### 9.3 Saving Results

Save every run as structured data (JSON or CSV) to `results/raw/`. Each record should contain: instance label, seed, algorithm name, composite fitness, weighted tardiness, setup cost, makespan. Do not save only the mean — save every individual run. You need the individual run data for the Wilcoxon test and for box plots.

### 9.4 Running Order

Run baselines first (fast, seconds per instance), then plain GA (minutes per instance), then hybrid (similar to GA). Do not run the hybrid until the PPO model is trained and saved. Load the saved model in `run_hybrid.py` rather than retraining it.

### 9.5 Statistical Tests

The Wilcoxon signed-rank test is the right test here. It is non-parametric (you cannot assume your fitness values are normally distributed), and it works on paired data (same instance, different algorithm). Run it with the alternative hypothesis "less" — you are testing whether the hybrid's fitness is significantly lower (better) than each baseline.

The threshold for statistical significance is p < 0.05. If p < 0.01, say so — it is a stronger result. If p > 0.05 on some instance size, do not hide it — discuss why the hybrid does not significantly outperform there (probably small instances where the problem is easy enough that all algorithms find good solutions).

Report results as mean ± standard deviation in your main table. p-values belong in a separate significance table.

### 9.6 Sensitivity Analysis

After your main results, vary alpha (try 0.3, 0.5, 0.7) and re-run on at least the medium instances. Show whether the algorithm rankings are robust to the weighting between tardiness and setup cost. This is a relatively quick additional experiment that substantially strengthens your dissertation.

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

Use LaTeX. Ask your supervisor whether Leeds provides an official dissertation template — many programmes do. If not, use the `report` class with 12pt font, 25mm margins, and the standard packages: `amsmath`, `amssymb`, `algorithm`, `algpseudocode`, `graphicx`, `booktabs`, `hyperref`, `natbib`.

For your bibliography, use Zotero to export a `.bib` file. Check every entry for formatting errors — Zotero exports are often imperfect for conference papers and technical reports.

### 11.3 Writing Chapter 4 (Implementation) Well

This is the chapter your technical examiner will scrutinise most. Key requirements:

**Pseudocode, not code.** Present all algorithms as formal pseudocode using the `algorithm2e` or `algorithmicx` LaTeX package. Your GA's main loop, the fitness function, the Gymnasium env's step function, and the PPO training procedure should all appear as pseudocode. Code blocks in appendices are fine, but the main text needs formal algorithmic notation.

**Every parameter in a table.** Every hyperparameter you chose — GA population size, crossover probability, mutation probability, PPO learning rate, n_steps, ent_coef, step_gens, total_gens, number of colours, processing time range — must appear in a parameter table with the value you used and a brief justification or citation.

**Justify your design decisions.** Don't just say "we used OX crossover." Say "OX crossover was chosen because it preserves the relative ordering of jobs, which carries scheduling-relevant information about colour transition sequences, unlike uniform crossover which would destroy this structure." Every non-obvious design decision needs a sentence of justification.

**System architecture diagram.** A single diagram showing how the modules interact — instance → evaluator → GA → GA-env → PPO → back to GA — is worth a page of text. Draw it clearly and reference it throughout Chapter 4.

### 11.4 Writing Chapter 5 (Results) Well

Lead with your main result. Don't bury it. Your first paragraph should state the key finding directly: the hybrid achieves X% lower composite fitness than SPT, Y% lower than nearest-neighbour, and Z% lower than the plain GA, with statistical significance on all instance sizes above n=10.

Then walk through the evidence. Start with the summary table (all algorithms × all instance sizes, mean±std). Then convergence analysis. Then Gantt charts. Then statistical significance table. Then sensitivity analysis. End with a paragraph summarising what the results mean for your research question.

**Be precise with numbers.** "The hybrid performed better" is not acceptable. "The hybrid achieved a mean composite fitness of 47.3 ± 8.1 on medium instances (n=20, m=2), representing a 23.4% reduction relative to SPT (82.1 ± 12.4, p < 0.001, Wilcoxon signed-rank test)" is what your examiner wants.

### 11.5 Writing Chapter 6 (Discussion) Well

This chapter is often underdeveloped in MSc dissertations. Key things to cover:

**What worked and why.** Explain the mechanism behind your hybrid's success, not just that it succeeded. The DRL agent learns to apply high-disruption operators (inversion) when the population has converged and diversity is low, and to switch to conservative operators (swap) when the population is diverse and the GA is making progress. Show this from your action-frequency analysis.

**What didn't work as expected.** If the hybrid doesn't significantly beat the plain GA on small instances, explain why: small instances have such a small search space that even a plain GA converges to near-optimal quickly, leaving little for the hyper-heuristic to improve. This is not a failure — it is an informative result.

**Threats to validity.** Acknowledge that your instances are synthetic, not from a real dyeing facility. Acknowledge that your PPO agent was trained and evaluated on instances from the same distribution. Acknowledge that 30 runs, while standard in the GA literature, is a relatively small sample.

**Future work.** At minimum: training the agent on multiple instance sizes simultaneously (curriculum learning), testing on real industrial data, comparing against more sophisticated baselines (branch-and-bound for small instances), extending to other objective functions.

### 11.6 The Abstract

Write this last. It should be roughly 250 words and cover: what problem you address, why it is hard, what your approach is, what your key results are, and what the implications are. No citations in the abstract.

### 11.7 General Writing Rules

- Never write "the results show X is better." Write "the hybrid achieves X% lower fitness than [baseline] (p < 0.05)."
- Every figure needs a self-contained caption — someone should understand what the figure shows without reading the surrounding text.
- Every claim needs either a citation or your own experimental evidence. No unsupported assertions.
- Refer to your system consistently. Pick a name ("the hybrid GA-DRL system" or "the proposed hyper-heuristic") and use it throughout.
- Proofread on paper, not on screen. You catch different errors.

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
- `GOGULA26-Code.zip` — all `src/`, `notebooks/`, `experiments/`, `requirements.txt`, and a `README.md` explaining how to reproduce the experiments. Do not include `results/raw/` (too large) or model `.zip` files except the final trained PPO model.

---

## Quick Reference: Timeline vs This Guide

| Milestone | Period | Sections |
|---|---|---|
| M1: Literature review + environment model | Mar–Apr (Weeks 20–22) | §3, §4, §5, §6 |
| M2: GA baseline | May–Jun (Weeks 25–28) | §7 |
| M3: DRL agent + integration | Jun–Jul (Weeks 29–S2) | §8 |
| M4: Evaluation + write-up | Jul–Aug (Weeks S3–S7) | §9, §10, §11, §12 |
