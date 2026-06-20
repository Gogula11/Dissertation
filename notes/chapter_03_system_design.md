# Chapter 3. Software Requirements and System Design

## 3.1 Software Requirements

### 3.1.1 Functional Requirements

The following functional requirements specify the capabilities that the software system must provide to fulfil the project objectives.

| ID | Requirement | Module |
|----|-------------|--------|
| FR1 | Generate synthetic PMSP-SDSC instances with configurable number of jobs n, number of machines m, and reproducible random seed | `instance_generator.py` |
| FR2 | Evaluate any candidate solution sigma against a problem instance, computing weighted tardiness, setup cost, normalised composite score, makespan, and per-job completion times | `evaluator.py` |
| FR3 | Implement Shortest Processing Time and Nearest-Neighbour Greedy baseline heuristics that return a complete solution for any valid instance | `heuristics.py` |
| FR4 | Run a Genetic Algorithm with configurable population size, number of generations, crossover probability, mutation probability, and mutation operator selection, returning the best solution found | `ga.py` |
| FR5 | Wrap the GA execution loop in a Gymnasium environment exposing a 4-dimensional continuous observation space, a 3-action discrete space, and a reward signal based on relative improvement in best fitness | `ga_env.py` |
| FR6 | Train a Proximal Policy Optimisation agent using the Gymnasium environment, supporting configurable training timesteps, instance pool, and policy hyperparameters | `drl_agent.py` |
| FR7 | Run hybrid GA+PPO inference for any problem instance, returning the best solution found under the agent's mutation operator selection policy | `drl_agent.py` |
| FR8 | Execute 720 experimental runs across four algorithms and six instance configurations, collecting results for statistical analysis | `experiments/` |

### 3.1.2 Non-Functional Requirements

**Determinism.** For a given random seed, the same instance generation, the same heuristic execution, and the same GA run must produce identical results. This is achieved through seeded pseudo-random number generators (NumPy and Python random) and pure functions in the evaluator and heuristic modules. Determinism is essential for reproducibility and for the paired experimental design.

**Modularity.** Each module in the `src/` package has a single, well-defined responsibility. Modules communicate through standard Python data structures (dicts for instances, lists of lists for solutions). No module produces side effects such as file I/O or print statements except when explicitly invoked as a standalone script. This design enables individual testing and independent reuse.

**Testability.** Core computational modules (evaluator and heuristics) have dedicated unit tests that verify correctness on small, manually verified examples. The GA and environment modules have tests that verify structural properties (observation shape, action space, episode termination) and functional correctness (solution validity, mutation operator effects).

**Reproducibility.** All experimental results can be reproduced by running the experiment scripts in sequence with the provided random seeds. The `requirements.txt` file specifies all dependencies with pinned versions. The paired experimental design ensures that differences between algorithms are attributable to algorithm performance rather than instance variation.

## 3.2 System Design

### 3.2.1 Architecture Overview

The software system is organised into six core modules in the `src/` package, five experiment scripts in `experiments/`, and a suite of tests in `tests/`. The data flow between components is as follows:

```
Instance Generator (seeded)
        |
        v
   Instance Dict
   {n, m, proc_times, due_dates, setup_cost, setup_time, weights, release, colour_ids}
        |
        v
   +-----------------------------------------------------+
   |                   Evaluator                          |
   |  validate_sigma -> compute_C -> tardiness f1         |
   |  setup_cost f2 -> composite = alpha*f1 + (1-alpha)*f2  |
   +-----------------------------------------------------+
        ^
        |
   +----+----------------+-----------------+
   |    |                |                 |
   v    v                v                 v
  SPT  NN-Greedy    GA (DEAP)        GA Env + PPO
  (heuristics.py)   (ga.py)          (ga_env.py + drl_agent.py)
```

The instance generator produces a dict containing all problem data for a given (n, m, seed) triple. The evaluator provides a pure function for computing the objective values of any solution. The baseline heuristics, GA solver, and hybrid environment each consume instances and produce solutions, using the evaluator to measure quality.

### 3.2.2 Instance Generation

Instances are generated using NumPy's seeded random number generator, ensuring reproducibility across runs.

**Colour classes.** Each job is assigned one of seven colour classes, each with a darkness ranking that determines the asymmetry of setup costs:

| Colour ID | Colour | Darkness |
|-----------|--------|----------|
| 0 | White | 1 |
| 1 | Yellow | 2 |
| 2 | Light Blue | 3 |
| 3 | Green | 4 |
| 4 | Red | 5 |
| 5 | Navy | 6 |
| 6 | Black | 7 |

**Cost matrix asymmetry.** The asymmetric setup cost matrix S is constructed from colour classes using the following rule:

```
S[i][j] = max(0, darkness[i] - darkness[j]) * 10 + noise
```

where noise is drawn uniformly from [0, 2]. Dark-to-light transitions (e.g., Black to White) incur high costs, while light-to-dark transitions incur near-zero costs. The diagonal (S[i][i]) is zero, representing no transition cost when the same colour follows itself.

**Instance configurations.** Six standard configurations are used throughout the experiments, spanning small, medium, and large problem sizes:

| Label | n | m |
|-------|---|---|
| small_2m | 10 | 2 |
| small_3m | 10 | 3 |
| medium_2m | 20 | 2 |
| medium_3m | 20 | 3 |
| large_2m | 50 | 2 |
| large_3m | 50 | 3 |

**Due date calibration.** Due dates are calibrated relative to processing times using a tightness parameter. The total processing workload is distributed across machines, and due dates are set proportional to each job's share of the total processing time, scaled by the tightness factor. A small uniform random perturbation is added to introduce variation.

### 3.2.3 Solution Representation

A solution sigma is a list of m lists, where sigma[k] is the ordered sequence of job indices assigned to machine k. Each job appears exactly once across all machine sequences:

```
n = 6, m = 2
sigma = [[0, 3, 5], [1, 2, 4]]
Machine 0: jobs 0 -> 3 -> 5
Machine 1: jobs 1 -> 2 -> 4
```

The GA uses the giant-tour encoding internally: a chromosome is a flat permutation of all n job indices. Decoding splits this permutation into m equal-ish segments, where the first n mod m machines receive one additional job each to account for indivisible remainder. This representation ensures that any permutation corresponds to a valid solution, simplifying crossover and mutation operations.

### 3.2.4 Evaluator Design

The evaluator is a pure function that computes the full set of scheduling performance metrics for a given solution-instance pair.

**Algorithm (pseudocode):**

```
function evaluate(sigma, instance, alpha):
    validate_sigma(sigma, instance.n)
    C = compute_completion_times(sigma, instance)
    T = compute_tardiness(C, instance.due_dates)
    f1 = compute_weighted_tardiness(T, instance.weights)
    f2 = compute_setup_cost(sigma, instance.setup_cost)
    (f1_scale, f2_scale) = estimate_scales(instance)
    f1_norm = f1 / f1_scale
    f2_norm = f2 / f2_scale
    composite = alpha * f1_norm + (1 - alpha) * f2_norm
    return {composite, f1, f2, makespan, completion_times, tardiness_per_job}
```

**Normalisation.** The two objective components operate on different scales: on large instances, weighted tardiness can exceed setup cost by an order of magnitude. Without normalisation, the composite objective would be dominated by tardiness, and the GA would effectively ignore setup cost. The `estimate_scales` function computes upper bounds for each component — the maximum possible weighted tardiness (assuming all jobs are delayed to the worst-case completion time) and the maximum possible setup cost (assuming the most expensive transition for every consecutive job pair). These scale estimates ensure that both components contribute meaningfully to the composite score.

**Completion time computation.** For each machine, completion times are computed sequentially with setup times inserted between consecutive jobs and release times applied at each job's start:

```
t = max(t, release[job])
t += setup_time[prev_job][job]  (if not first job)
t += proc_time[job]
C[job] = t
```

### 3.2.5 Baseline Heuristics

**SPT (Shortest Processing Time).** Jobs are sorted by ascending processing time and assigned round-robin to machines:

```
sorted_jobs = argsort(instance.proc_times)
for i, job in enumerate(sorted_jobs):
    assign job to machine i % m
```

SPT ignores setup costs and due dates entirely. Its O(n log n) time complexity makes it the fastest baseline.

**NN-Greedy (Nearest Neighbour).** Jobs are assigned one at a time to the machine with the lowest current load, selecting the unscheduled job with the minimum setup cost from that machine's last job:

```
while unscheduled_jobs is not empty:
    k = argmin(machine_loads)
    if machine k has no jobs:
        job = min(unscheduled_jobs, by=proc_time)
    else:
        last = last_job_on_machine[k]
        job = min(unscheduled_jobs, by=setup_cost[last][job])
    assign job to machine k
```

NN-Greedy accounts for setup costs but makes locally optimal decisions that may lead to globally poor solutions.

### 3.2.6 GA Design

**Chromosome encoding.** A chromosome is a permutation of job indices [0, 1, ..., n-1].

**Decoding.** The permutation is split into m segments with approximately equal length:

```
chromosome = [2, 0, 4, 1, 3]  (n = 5)
m = 2  ->  sigma = [[2, 0, 4], [1, 3]]
```

**Genetic operators.**

| Operator | Type | Behaviour |
|----------|------|-----------|
| Crossover | Order Crossover (OX) | Preserves relative job ordering from one parent, fills remaining positions from the other |
| Mutation (swap) | Shuffle Indexes (indpb = 0.05) | Swaps pairs of positions with 5% probability per gene |
| Mutation (inversion) | Inversion | Reverses a random sub-sequence |
| Mutation (aggressive swap) | Shuffle Indexes (indpb = 0.20) | Swaps pairs of positions with 20% probability per gene |
| Selection | Tournament (size = 3) | Selects the best individual among 3 randomly sampled candidates |

**Parameters.** The GA parameters were tuned via grid search on a medium instance (n = 20, m = 2):

| Parameter | Value |
|-----------|-------|
| Population size | 100 |
| Number of generations | 200 |
| Crossover probability | 0.9 |
| Mutation probability | 0.2 |

**Elitism.** The best individual from each generation is preserved via DEAP's Hall-of-Fame mechanism with size 1, ensuring monotonic non-degradation of best fitness across generations.

### 3.2.7 GA Environment Design

The Gymnasium environment wraps the GA execution loop, providing an RL-compatible interface for the PPO agent. An episode corresponds to one complete GA run, and each step corresponds to `step_gens` generations of the GA with a fixed mutation operator chosen by the agent.

**Observation space** (Box(4,) with range [0, 1]):

| Feature | Definition | Purpose |
|---------|------------|---------|
| best_norm | Current best fitness / initial best fitness | Measures progress from the starting point |
| mean_norm | Population mean fitness / initial best fitness | Measures population convergence |
| diversity | Mean pairwise Hamming distance across population | Measures remaining exploration potential |
| stagnation | Consecutive steps without improvement / max steps | Detects plateaus requiring disruption |

**Action space** (Discrete(3)):

| Action | Operator | Effect |
|--------|----------|--------|
| 0 | Swap mutation (indpb = 0.05) | Conservative fine-tuning of existing solutions |
| 1 | Inversion mutation | Moderate disruption through sub-sequence reversal |
| 2 | Aggressive swap mutation (indpb = 0.20) | High exploration through extensive shuffling |

**Reward.** The reward at each step is the relative improvement in best fitness over that step:

```
reward = (best_before - best_after) / max(best_before, 1e-6)
```

A plateau penalty of -0.01 replaces zero reward when no improvement occurs, preventing the agent from learning idle behaviour that neither helps nor hurts.

**Episode termination.** An episode terminates when the step counter reaches the maximum number of steps (total_gens / step_gens). This is signalled as a time-limit truncation rather than a terminal state, allowing the value function to bootstrap correctly during PPO training.

**Stochastic instance sampling.** During training, each call to `reset()` randomly samples an instance from the training pool. This forces the agent to learn a generalisable policy rather than memorising an instance-specific mutation schedule.

### 3.2.8 PPO Agent Design

The PPO agent is implemented using Stable-Baselines3 with an MLP policy architecture.

**Policy architecture.** The policy network is a multi-layer perceptron with two hidden layers, taking the 4-dimensional observation as input and outputting action probabilities and state values.

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Steps per update (n_steps) | 512 |
| Batch size | 64 |
| Epochs per update (n_epochs) | 10 |
| Discount factor (gamma) | 0.99 |
| Entropy coefficient | 0.01 |

**Training.** The agent is trained for 100,000 timesteps on a diversified instance pool of 60 instances (6 configurations x 10 seeds each). During training, instances are sampled uniformly at each episode. TensorBoard logging records reward trends, policy entropy, and value function loss.

**Inference.** For evaluation, the trained agent operates in deterministic mode: at each step, it selects the action with the highest probability. This ensures reproducible results and represents the learned policy's best estimate of the optimal mutation operator at each decision point.

### 3.2.9 Experimental Design

The experimental design encompasses four algorithms across six instance configurations with 30 random seeds each, yielding 720 individual experimental runs.

| Algorithm | Seeds | Instance Configurations | Runs |
|-----------|-------|-------------------------|------|
| SPT | 0-29 | 6 | 180 |
| NN-Greedy | 0-29 | 6 | 180 |
| GA | 0-29 | 6 | 180 |
| Hybrid (GA+PPO) | 0-29 | 6 | 180 |
| **Total** | | | **720** |

**Paired design.** The same set of 30 random seeds is used for all algorithms on each configuration. This means that for a given seed and configuration, all four algorithms solve the same problem instance. The paired design is essential for the Wilcoxon signed-rank test, which operates on paired differences.

**Sensitivity analysis.** In addition to the primary experiments with alpha = 0.5, a sensitivity analysis is conducted on the medium configurations (medium_2m and medium_3m) with alpha = 0.3 and alpha = 0.7, using 10 seeds each. This tests whether the relative performance of the algorithms is robust to the choice of objective weighting.

**Statistical testing.** Performance differences are assessed using the Wilcoxon signed-rank test with the alternative hypothesis that the hybrid approach yields a lower composite score than the comparator algorithm (one-tailed test). Significance levels are interpreted as follows:

| p-value | Interpretation |
|---------|---------------|
| p < 0.001 | Highly significant |
| p < 0.01 | Significant |
| p < 0.05 | Marginally significant |
| p >= 0.05 | Not significant |