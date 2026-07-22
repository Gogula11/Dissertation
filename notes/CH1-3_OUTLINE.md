# Chapters 1-3: What to Write

Your project: Hybrid GA-DRL for PMSP-SDSC (parallel machine scheduling with sequence-dependent setup costs)

Key results from `results_summary_baseline.tex` (composite scores, lower=better):

| Config | SPT | NN-Greedy | GA | Hybrid |
|---|---|---|---|---|
| large_2m | 0.321 | 0.222 | 0.184 | **0.138** |
| large_3m | 0.317 | 0.199 | 0.211 | **0.144** |
| medium_2m | 0.305 | 0.195 | 0.105 | 0.103 |
| medium_3m | 0.297 | 0.192 | 0.097 | 0.100 |
| small_2m | 0.303 | 0.240 | **0.095** | 0.094 |
| small_3m | 0.287 | 0.222 | **0.081** | 0.085 |

Narrative: Hybrid beats GA on large instances (25-32% improvement, p < 0.001). On small/medium instances, GA and Hybrid are equivalent (search space too small for hyper-heuristic to matter). This is your story. Also validated on a realistic profile (14-24% improvement on large).

---

## Chapter 1. Introduction (4-6 pages)

### 1.1 Project Aim (~1 paragraph)

"The aim of this project is to investigate a hybrid approach combining Genetic Algorithms (GA) and Deep Reinforcement Learning (DRL) for solving parallel machine scheduling problems with asymmetric, sequence-dependent setup costs (PMSP-SDSC). Specifically, a Proximal Policy Optimisation (PPO) agent is trained as a hyper-heuristic to dynamically select GA mutation operators, improving solution quality over standalone GA and classical heuristics."

**Expand with:**
- Real-world motivation: textile dyeing — colour transitions create asymmetric costs
- Problem class: PMSP-SDSC is NP-hard, combinatorial explosion
- Why hybrid: GAs explore but stagnate; RL can learn when to explore vs exploit
- Key result summary: "The hybrid achieves 53% lower composite cost than SPT (p<0.001) and 46% lower than plain GA (p<0.01) on large instances"

### 1.2 Objectives (bullet list)

1. Formalise the PMSP-SDSC problem with asymmetric sequence-dependent cost structure
2. Implement a synthetic instance generator with calibrated due-date tightness
3. Implement Shortest Processing Time (SPT) and Nearest-Neighbour Greedy baselines
4. Implement a Genetic Algorithm with permutation encoding and three mutation operators (swap, inversion, insertion)
5. Design a Gymnasium environment wrapping the GA with 8D state space and 3-action space (swap, inversion, insertion)
6. Train a PPO agent to select mutation operators dynamically during GA execution
7. Evaluate all 4 algorithms across 6 instance configurations × 50 seeds (1200 runs per profile) across two evaluation profiles (baseline and realistic)
8. Perform Wilcoxon signed-rank tests and α sensitivity analysis

### 1.3 Deliverables (bullet list)

1. Python package `src/` with 6 modules: instance_generator, evaluator, heuristics, ga, ga_env, drl_agent
2. Experiment scripts `experiments/run_baselines.py`, `run_ga.py`, `run_hybrid.py`
3. Trained PPO model at `models/ppo_hyperheuristic.zip`
4. Full results: 720 JSON files in `results/raw/`
5. Comparison tables, box plots, Gantt charts, convergence curves, action frequency analysis
6. This dissertation

### 1.4 Ethical, Legal, and Social Issues

- **Ethical**: synthetic data only (no human participants, no privacy concerns). No AI-generated content used without verification.
- **Legal**: open-source libraries used under their respective licenses (numpy, DEAP, Gymnasium, Stable-Baselines3).
- **Social**: scheduling optimisation reduces waste in manufacturing (environmental benefit).
- Ethics form already approved (Appendix A).

---

## Chapter 2. Background Research (12-16 pages)

### 2.1 Literature Survey

Told as a narrative arc: here is the problem → here is what people tried → here is what worked → here is the gap.

#### 2.1.1 Parallel Machine Scheduling (PMSP)

- Standard PMSP: assign n jobs to m identical machines minimising makespan/tardiness
- In general, PMSP is NP-hard (Garey & Johnson, 1979)
- Your variant: sequence-dependent setup costs (SDSC) + asymmetry → still harder
- Search space size: approximate (n!)^(m / m!) — grows super-exponentially
- Reference: Pinedo, "Scheduling: Theory, Algorithms, and Systems" (standard textbook)

#### 2.1.2 Sequence-Dependent Setup Costs

- Real manufacturing: changeover between jobs costs time and materials
- Textile dyeing: dark→light transitions contaminate the lighter batch (costly)
- Asymmetry: S[i][j] ≠ S[j][i] — dark→light expensive, light→dark cheap
- Standard heuristics treat all transitions as equivalent → fundamental limitation

#### 2.1.3 Classical Heuristics for Scheduling

- **SPT (Shortest Processing Time)**: sort by processing time, assign round-robin. O(n log n). Fast but ignores setup costs entirely.
- **NEH (Nawaz, Enscore, Ham, 1983)**: insertion-based heuristic. O(n^2 m). Best-performing constructive heuristic for many scheduling variants.
- **NN-Greedy**: sequential assignment minimising local transition cost. Simple but greedy decisions compound.
- These are baselines because they are fast, deterministic, and standard — not because they are optimal.

#### 2.1.4 Genetic Algorithms for Scheduling

- Chromosome representation: permutation of n jobs (giant tour), split into m segments for machines
- Order crossover (OX): preserves relative ordering — critical for colour transition information
- Mutation operators: swap (fine-tuning), inversion (moderate diversity), aggressive swap (high exploration)
- GA balances exploration vs exploitation but plateaus as population converges
- DEAP (Fortin et al., 2012) is a mature GA framework

#### 2.1.5 Deep Reinforcement Learning for Combinatorial Optimisation

- Paradigm shift: learn a policy instead of hand-crafting rules
- PPO (Schulman et al., 2017): clipped surrogate objective, trust-region constraint, stable training
- Prior work: DRL for job shop scheduling (Zhang et al., 2020), vehicle routing (Kool et al., 2019)
- Advantage over DQN: handles continuous/structured state spaces more stably

#### 2.1.6 Hyper-heuristics

- Formal definition (Burke et al., 2013): "A hyper-heuristic is an automated methodology for selecting or generating heuristics to solve computational search problems."
- Your approach: DRL agent as hyper-heuristic at the meta-level — selects GA mutation operators
- Key distinction: NOT "RL for scheduling" directly. RL controls the GA, not the schedule itself.
- This framing is what makes your contribution distinct.

#### 2.1.7 The Gap (1 paragraph)

"Prior work has combined GA and RL in scheduling contexts, but not specifically as a PPO hyper-heuristic controlling mutation operator selection in an asymmetric parallel machine scheduling problem with sequence-dependent setup costs. This dissertation fills that gap by (1) formalising the PMSP-SDSC with asymmetric costs, (2) designing a Gymnasium environment for hyper-heuristic control, and (3) empirically demonstrating that the hybrid approach outperforms both classical heuristics and standalone GA on large problem instances."

### 2.2 Methods & Techniques

Brief, balanced survey of available methods:

- **Instance generation**: seeded numpy RNG, colour classes, calibrated due dates
- **Solution representation**: sigma = list of m machine sequences
- **GA frameworks**: DEAP vs PyGAD vs custom — DEAP chosen for toolbox architecture
- **RL frameworks**: SB3 PPO vs custom — SB3 chosen for reliability and vectorised environments
- **Environment API**: Gymnasium vs custom loop — Gymnasium chosen for SB3 compatibility
- **Objective weighting**: α = 0.5 balanced, also test 0.3 and 0.7 for sensitivity
- **Statistical testing**: Wilcoxon signed-rank (non-parametric, paired) — appropriate for 30-run comparisons

### 2.3 Choice of Methods (~1-2 pages)

**Why GA?** Permutation encoding maps naturally to scheduling. OX crossover preserves job ordering (colour transition information). Three mutation operators provide a meaningful action space.

**Why PPO?** More stable than DQN for continuous state spaces. Trust-region clipping prevents destructive policy updates. Well-supported in SB3 with minimal hyperparameter tuning.

**Why PPO as hyper-heuristic (not end-to-end RL)?** End-to-end RL for scheduling scales poorly — the action space grows factorially. Controlling GA mutation operators keeps action space small (Discrete(3)) while leveraging GA's search capability. This separation of concerns is the key design insight.

**Why composite objective (α=0.5)?** Balances tardiness (customer satisfaction) and setup cost (manufacturing efficiency). Normalisation prevents one objective dominating.

**Why instance pool training?** Training on a single instance causes overfitting. Training on 5 diverse instances (n=20, m=2, seeds 0-4) forces the agent to learn a generalisable policy.

**Why 30 seeds?** Standard in GA literature for statistically meaningful comparisons. 30 runs × 6 configs × 4 algorithms = 720 total experiments.

**Why Wilcoxon over t-test?** Paired design (same seeds across algorithms), non-parametric (no normality assumption), appropriate for 30 samples.

---

## Chapter 3. Software Requirements and System Design (5-20 pages depending on template)

*Note: Check with supervisor whether to follow Leeds official structure (~5-7 pages) or the detailed guide.md (~15-20 pages). This outline follows Leeds official but can be expanded.*

### 3.1 Software Requirements

#### 3.1.1 Functional Requirements

| ID | Requirement | Module |
|---|---|---|
| FR1 | Generate synthetic PMSP-SDSC instances with configurable (n,m) and reproducible seed | `instance_generator.py` |
| FR2 | Evaluate any sigma solution → composite score (tardiness + setup cost, normalised) | `evaluator.py` |
| FR3 | Implement SPT and NN-Greedy baselines | `heuristics.py` |
| FR4 | Run GA with configurable parameters and mutation operators | `ga.py` |
| FR5 | Wrap GA in Gymnasium environment with 8D state / Discrete(3) action / reward | `ga_env.py` |
| FR6 | Train PPO agent using the environment | `drl_agent.py` |
| FR7 | Run hybrid GA+PPO inference for any instance | `drl_agent.py` |
| FR8 | Run 720 experiments, collect results, compute statistical comparisons | `experiments/` |

#### 3.1.2 Non-Functional Requirements

- **Determinism**: same seed → same instance → same evaluation → same result (pure functions)
- **Modularity**: src/ modules have no side effects, no print statements
- **Testability**: evaluator and heuristics have unit tests
- **Reproducibility**: requirements.txt, seeded RNG, paired experimental design

### 3.2 System Design

#### 3.2.1 Architecture Overview

```
Instance Generator (seeded)
        ↓
   Instance Dict
   {n, m, proc_times, due_dates, setup_cost, ...}
        ↓
   ┌─────────────────────────────────────────────┐
   │              Evaluator                       │
   │  validate_sigma → compute_C → tardiness f1  │
   │  setup_cost f2 → composite = α·f1+(1-α)·f2  │
   └─────────────────────────────────────────────┘
        ↑
   ┌────┴────────────┬────────────────┐
   │                 │                │
   ▼                 ▼                ▼
  SPT           GA (DEAP)      GA Env + PPO
  NN-Greedy     (solver)       (hyper-heuristic)
  (baselines)
```

#### 3.2.2 Instance Generation

**Colour model:** Two evaluation profiles control colour complexity. The baseline profile uses 7 discrete colour classes (white=1 through black=7) with uniform distribution. The realistic profile uses 12 continuous colour families (white through black, with intermediate families like pink, orange, brown, purple, grey) each with a base darkness, shade variance, dye chemistry (direct/reactive/vat), and 30% colour clustering probability.

**Profile presets:** All parameters (colour model, distribution, clustering, chemistry penalty, proc-colour correlation, customer segments) are bundled into named profiles — "baseline" and "realistic" — selectable via a single `profile` argument to `generate_instance()`.

**Primary instance configs (6):**
| Label | n | m |
|---|---|---|
| small_2m | 10 | 2 |
| small_3m | 10 | 3 |
| medium_2m | 20 | 2 |
| medium_3m | 20 | 3 |
| large_2m | 50 | 2 |
| large_3m | 50 | 3 |

Plus 5 exploratory configs: tiny_2m (5×2), medium_30_3m (30×3), large_5m (50×5), xlarge_5m (100×5), xlarge_10m (100×10).

#### 3.2.3 Solution Representation

A solution sigma is a list of m machine sequences:
```
n=6, m=2
sigma = [[0, 3, 5], [1, 2, 4]]
Machine 0: jobs 0 → 3 → 5
Machine 1: jobs 1 → 2 → 4
```

#### 3.2.4 Evaluator Design

Pseudocode:
```
function evaluate(sigma, instance, alpha, f1_scale, f2_scale):
    validate_sigma(sigma, instance.n)
    C = compute_completion_times(sigma, instance)
    T = compute_tardiness(C, instance.due_dates)
    f1 = weighted_tardiness(T, instance.weights)
    f2 = compute_setup_cost(sigma, instance.setup_cost)
    f1_norm = f1 / f1_scale   # normalise to prevent scale dominance
    f2_norm = f2 / f2_scale
    composite = alpha × f1_norm + (1-alpha) × f2_norm
    return {composite, f1, f2, makespan, ...}
```

f1_scale, f2_scale are computed externally by estimate_scales() — evaluates SPT, NN-Greedy, and a random schedule, takes max × 1.5 margin.

#### 3.2.5 Baseline Heuristics

**SPT (Shortest Processing Time):**
```
sorted_jobs = argsort(instance.proc_times)
for i, job in enumerate(sorted_jobs):
    assign job to machine i % m
```

**NN-Greedy (Nearest Neighbour):**
```
for job in range(n):
    find machine k where transition_cost(last_job[k], job) is minimised
    assign job to machine k
```

#### 3.2.6 GA Design

**Chromosome:** permutation of [0, 1, ..., n-1]

**Decoding:** split into m equal-ish segments:
```
chromosome = [2,0,4,1,3]  (n=5)
m=2 → sigma = [[2,0,4], [1,3]]
```

**Operators:**
- **Crossover**: Order Crossover (OX) — preserves relative ordering of jobs
- **Mutation (3 types)**:
  - Swap (indpb=0.05): swap two random positions
  - Inversion: reverse a sub-sequence
  - Insertion (indpb=0.15): remove-and-reinsert elements
- **Selection**: tournament (size=3)

**Parameters:** pop_size=100, n_gen=300 (experiments), cx_prob=0.8, mut_prob=0.2 (tuned on n=20, m=2)

#### 3.2.7 GA Environment Design (for DRL)

**Observation space (8D Box [0,1]):**
| Feature | Definition | Range |
|---|---|---|
| best_norm | current best / initial best | [0, 1] |
| mean_norm | population mean / initial best | [0, ~1] |
| diversity | mean Hamming distance / n | [0, 1] |
| stagnation | gens since improvement / total | [0, 1] |
| n_norm | jobs / 100 | [0, 1] |
| m_norm | machines / 10 | [0, 1] |
| cost_mean_norm | mean off-diag setup / max off-diag | [0, 1] |
| darkness_mean_norm | mean colour darkness / 10 | [0, 1] |

**Action space (Discrete 3):**
| Action | Operator | Effect |
|---|---|---|
| 0 | Swap (indpb=0.05) | Conservative fine-tuning |
| 1 | Inversion | Moderate disruption |
| 2 | Insertion (indpb=0.15) | High exploration |

**Reward:** (best_before - best_after) / best_before — relative improvement each step.

#### 3.2.8 PPO Agent Design

- **Architecture:** MlpPolicy (2-layer MLP)
- **Hyperparameters:**
  - learning_rate = 3e-4
  - n_steps = 2048
  - batch_size = 64
  - n_epochs = 10
  - gamma = 0.99
  - ent_coef = 0.05
- **Training:** 100,000 timesteps on an instance pool (110 instances, 11 configs × 10 seeds) with reduced pop_size=25 and total_gens=100
- **Inference:** deterministic action selection, full pop_size=100 and total_gens=300

#### 3.2.9 Experimental Design

| Algorithm | Seeds | Instance Configs | Runs |
|---|---|---|---|---|
| SPT | 0-49 | 6 | 300 |
| NN-Greedy | 0-49 | 6 | 300 |
| GA | 0-49 | 6 | 300 |
| Hybrid | 0-49 | 6 | 300 |
| **Per profile** | | | **1200** |

Same seed → same instance across all algorithms (paired design).
Two profiles (baseline + realistic) = 2400 runs total.

**Sensitivity analysis:** re-run GA and Hybrid on all configurations with α = 0.3, 0.5, 0.7 (30 seeds each).

**Statistical testing:** Wilcoxon signed-rank test (alternative="less": test if hybrid < baseline).
- p < 0.001: "highly significant"
- p < 0.01: "significant"
- p < 0.05: "marginally significant"
- p ≥ 0.05: "not significant"

---

## Quick Reference: What Goes Where

| Chunk | Chapter | Section |
|---|---|---|
| Problem definition | 1.1 Aim | "PMSP-SDSC is NP-hard..." |
| Textile dyeing motivation | 1.1 Aim | "colour transitions..." |
| Research question | 1.1 Aim | "Can PPO hyper-heuristic improve GA?" |
| Bullet objectives | 1.2 | 8 bullets |
| Deliverables list | 1.3 | 6 bullets |
| Ethics paragraph | 1.4 | synthetic data, no human subjects |
| PMSP literature | 2.1.1 | Pinedo, NP-hardness |
| Setup costs literature | 2.1.2 | asymmetry, textile domain |
| Heuristics literature | 2.1.3 | SPT, NEH, NN-Greedy |
| GA literature | 2.1.4 | permutation encoding, OX crossover |
| DRL literature | 2.1.5 | PPO, combinatorial optimisation |
| Hyper-heuristic literature | 2.1.6 | Burke et al., definition |
| The gap | 2.1.7 | "prior work has not..." |
| Methods survey | 2.2 | DEAP, SB3, Gymnasium, Wilcoxon |
| Why GA | 2.3 | permutation, OX, action space |
| Why PPO | 2.3 | stability, SB3 ecosystem |
| Why hyper-heuristic | 2.3 | tractable action space, separation of concerns |
| FR table | 3.1.1 | 8 functional requirements |
| Architecture diagram | 3.2.1 | boxes and arrows |
| Colour/darkness table | 3.2.2 | 7 colour IDs |
| Cost matrix formula | 3.2.2 | max(0, dark-light)*10+noise |
| Instance configs table | 3.2.2 | 6 configs |
| Notation table | 3.2.4 | n, m, p_i, d_i, S[i][j], etc. |
| Pseudocode: evaluate() | 3.2.4 | full function |
| Pseudocode: SPT | 3.2.5 | sort + round-robin |
| Pseudocode: NN | 3.2.5 | sequential assignment |
| Chromosome diagram | 3.2.6 | permutation → sigma |
| Parameter table (GA) | 3.2.6 | pop=100, gen=200, cx=0.8, mut=0.2 |
| Observation space table | 3.2.7 | 8 features with ranges |
| Action space table | 3.2.7 | 3 actions with effects |
| Reward definition | 3.2.7 | relative improvement |
| PPO parameters | 3.2.8 | lr=3e-4, n_steps=2048, ent_coef=0.05 |
| Experiment design table | 3.2.9 | 1200 runs per profile, 6 configs × 4 algs × 50 seeds |
| Wilcoxon explanation | 3.2.9 | paired, non-parametric |
