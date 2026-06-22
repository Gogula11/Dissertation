# Chapter 1. Introduction

## 1.1 Project Aim

The aim of this project is to investigate a hybrid approach combining Genetic Algorithms (GA) and Deep Reinforcement Learning (DRL) for solving parallel machine scheduling problems with asymmetric, sequence-dependent setup costs (PMSP-SDSC). Specifically, a Proximal Policy Optimisation (PPO) agent is trained as a hyper-heuristic to dynamically select GA mutation operators, improving solution quality over standalone GA and classical heuristics.

The parallel machine scheduling problem concerns the assignment of a set of jobs to a set of identical machines operating in parallel, where each job must be processed by exactly one machine. This problem class is fundamental to manufacturing systems and is known to be NP-hard (Garey & Johnson, 1979). The variant considered in this project, PMSP-SDSC, introduces sequence-dependent setup costs: the cost of transitioning between jobs on a machine depends on the specific pair of jobs and is asymmetric. This structure is motivated by real-world manufacturing domains such as textile dyeing, where colour transitions between batches incur material and time costs. A dark-to-light colour transition may contaminate the lighter batch, making it significantly more expensive than a light-to-dark transition. This asymmetry renders classical scheduling heuristics inadequate, as they treat all inter-job transitions as equivalent.

The combinatorial explosion inherent in PMSP-SDSC — the search space grows as (n!)^(m / m!) — means that exact optimisation methods are intractable for problem instances of practical size. Metaheuristic approaches such as Genetic Algorithms are widely used for such problems, as they can explore large search spaces effectively. However, GAs typically employ fixed mutation operators with static parameters, leading to a well-documented limitation: the balance between exploration and exploitation degrades as the population converges, often resulting in premature stagnation on local optima.

Deep Reinforcement Learning offers a complementary capability: an agent can learn a policy that adapts its behaviour based on the current state of the search process. By framing mutation operator selection as a sequential decision problem, a DRL agent can dynamically choose whether to apply conservative fine-tuning or aggressive exploration depending on the population's convergence status. This separation of concerns — the GA explores the solution space while the DRL agent controls the search strategy — is the central design insight of this project.

The hybrid approach developed in this project achieves a 53% lower composite cost than the Shortest Processing Time heuristic (p < 0.001) and a 46% lower composite cost than a standalone GA (p < 0.01) on large problem instances. These results demonstrate that DRL-guided mutation selection is an effective strategy for improving GA performance on challenging scheduling problems.

## 1.2 Objectives

The following objectives were defined to achieve the overall aim of the project:

1. Formalise the PMSP-SDSC problem with asymmetric sequence-dependent cost structure, including mathematical definitions of weighted tardiness, setup cost, and the composite objective function with normalisation.

2. Implement a synthetic instance generator that produces reproducible PMSP-SDSC instances with colour-based cost matrices, calibrated due-date tightness, and six configurable sizes (small, medium, large, each with two- and three-machine variants).

3. Implement Shortest Processing Time (SPT) and Nearest-Neighbour Greedy heuristic baselines to serve as lower-bound comparators for the GA and hybrid approaches.

4. Implement a Genetic Algorithm using the DEAP framework with permutation chromosome encoding, order crossover, three mutation operators (swap, inversion, aggressive swap), and configurable population size, generation count, and crossover and mutation probabilities.

5. Design a Gymnasium environment that wraps the GA execution loop and exposes a 4-dimensional continuous observation space (normalised best fitness, normalised mean fitness, population diversity, and stagnation count) and a 3-action discrete space (swap, inversion, aggressive swap mutation), with a reward signal based on relative improvement in best fitness.

6. Train a Proximal Policy Optimisation (PPO) agent within the Gymnasium environment to learn a mutation selection policy, using an instance pool of 60 diverse training instances to promote generalisation.

7. Evaluate all four algorithms — SPT, NN-Greedy, GA, and Hybrid (GA+PPO) — across six instance configurations with 30 random seeds each, totalling 720 individual experimental runs.

8. Perform statistical analysis using Wilcoxon signed-rank tests to assess the significance of performance differences, and conduct a sensitivity analysis across three values of the objective weighting parameter alpha.

## 1.3 Deliverables

The following deliverables were produced as part of this project:

1. A Python software package (`src/`) comprising six modules: `instance_generator.py`, `evaluator.py`, `heuristics.py`, `ga.py`, `ga_env.py`, and `drl_agent.py`.

2. Experiment scripts (`experiments/`) for running baselines, GA, PPO training, hybrid inference, and sensitivity analysis: `run_baselines.py`, `run_ga.py`, `train_ppo.py`, `run_hybrid.py`, and `run_sensitivity.py`.

3. A trained PPO model saved at `models/ppo_hyperheuristic.zip`.

4. Full experimental results comprising 720 individual runs across four algorithms and six instance configurations, stored in JSON format under `results/raw/`.

5. Visualisation and analysis outputs: comparison tables, box plots, Gantt charts, convergence curves, and action frequency distributions, alongside LaTeX-formatted summary tables.

6. This dissertation.

## 1.4 Ethical, Legal, and Social Issues

This project involves no human participants, no personally identifiable data, and no privacy concerns. All data used in the experiments is synthetically generated using seeded pseudo-random number generators, ensuring reproducibility without recourse to real-world data. The project therefore raises no ethical concerns regarding data collection, consent, or privacy.

The software developed in this project uses only open-source libraries, each used under its respective license: NumPy (BSD-3-Clause), DEAP (LGPL-3.0-or-later), Gymnasium (MIT), and Stable-Baselines3 (MIT). No proprietary software was used in the development or execution of the experiments.

From a social perspective, scheduling optimisation has clear environmental and economic benefits. Improved scheduling in manufacturing reduces material waste from colour changeovers, decreases energy consumption through more efficient machine utilisation, and lowers production costs. These benefits are particularly relevant in energy-intensive industries such as textile dyeing and chemical processing, where sequence-dependent setup costs represent a substantial fraction of total production cost.

The ethics approval form for this project has been submitted and approved (see Appendix A).

## 1.5 Structure of the Dissertation

The remainder of this dissertation is organised as follows. Chapter 2 presents a literature survey covering parallel machine scheduling, asymmetric setup costs, classical heuristics, genetic algorithms, deep reinforcement learning, and hyper-heuristics, followed by a justification of the methods chosen for this project. Chapter 3 describes the software requirements and system design, including the architecture of all six core modules, the GA environment, and the experimental design. Chapter 4 presents the implementation details and experimental results, including performance comparisons, statistical analysis, and visualisations. Chapter 5 evaluates and discusses the findings, including limitations and threats to validity. Chapter 6 concludes the dissertation with a summary of contributions, key findings, directions for future work, and a personal reflection.