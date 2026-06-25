
**Chapter 1. Introduction**

**1.1 Project Aim**

The aim of this project is to investigate a hybrid approach combining Genetic Algorithms (GA) and Deep Reinforcement Learning (DRL) for solving parallel machine scheduling problems with asymmetric, sequence-dependent setup costs (PMSP-SDSC). Specifically, the primary goal is to evaluate the performance of a Proximal Policy Optimisation (PPO) agent trained as a hyper-heuristic to dynamically select GA mutation operators based on real-time population convergence metrics.

The parallel machine scheduling problem deals with a set of jobs where in, a set of identical machines are operating in parallel, where each job must be processed by exactly one machine. This problem is fundamental to manufacturing systems and is known to be NP-hard (Garey & Johnson, 1979). The variant considered in this project, PMSP-SDSC, introduces sequence-dependent setup costs. That is, the cost of transitioning between jobs on a machine depends on the specific pair of jobs and is asymmetric. This structure is motivated by real-world manufacturing domains such as textile dyeing, where colour transitions between batches incur material and time costs. A dark-to-light colour transition may contaminate the lighter batch, making it significantly more expensive than a light-to-dark transition. This asymmetry renders classical scheduling heuristics inadequate, as they treat all inter-job transitions as equivalent.

The combinatorial explosion inherent in PMSP-SDSC renders exact optimisation methods intractable for instances of practical size where the search space grows factorially as (n!)^(m / m!). To navigate this vast search spaces effectively, meta-heuristic approaches such as Genetic Algorithms are widely used for such problems, as they can explore large search spaces effectively. However, GAs typically employ fixed mutation operators with static parameters, leading to a well-documented limitation: the balance between exploration and exploitation degrades as the population converges, often resulting in premature stagnation on local optima.

Deep Reinforcement Learning offers a complementary capability, as an agent can learn a policy that adapts its behaviour based on the current state of the search process. By framing mutation operator selection as a sequential decision problem, a DRL agent can dynamically choose whether to apply conservative fine-tuning or aggressive exploration depending on the convergence status of the population. This separation of concerns, where the GA explores the solution space while the DRL agent controls the search strategy, forms the central design insight of this project.

The hybrid approach developed in this project achieves a 53% lower composite cost than the Shortest Processing Time heuristic (p < 0.001) and a 46% lower composite cost than a standalone GA (p < 0.01) on large problem instances. These results demonstrate that DRL-guided mutation selection is an effective strategy for improving GA performance on challenging scheduling problems.

**1.2 Objectives**

The following objectives were defined to achieve the overall aim of the project:

1. Formalise the PMSP-SDSC problem with asymmetric sequence-dependent cost structure, including mathematical definitions of weighted tardiness, setup cost, and the composite objective function with normalisation.
2. Implement a synthetic instance generator that produces reproducible PMSP-SDSC instances with colour-based cost matrices, calibrated due-date tightness, and six configurable sizes (small, medium, large, each with two- and three-machine variants).
3. Implement Shortest Processing Time (SPT) and Nearest-Neighbour Greedy heuristic baselines to serve as lower-bound comparators for the GA and hybrid approaches.
4. Implement a Genetic Algorithm using the DEAP framework with permutation chromosome encoding, order crossover, three mutation operators (swap, inversion, aggressive swap), and configurable population size, generation count, and crossover and mutation probabilities.
5. Design a Gymnasium environment that wraps the GA execution loop and exposes a 4-dimensional continuous observation space (normalised best fitness, normalised mean fitness, population diversity, and stagnation count) and a 3-action discrete space (swap, inversion, aggressive swap mutation), with a reward signal based on relative improvement in best fitness.
6. Train a Proximal Policy Optimisation (PPO) agent within the Gymnasium environment to learn a mutation selection policy, using an instance pool of 60 diverse training instances to promote generalisation.
7. Evaluate all four algorithms — SPT, NN-Greedy, GA, and Hybrid (GA+PPO) — across six instance configurations with 30 random seeds each, totalling 720 individual experimental runs.
8. Perform statistical analysis using Wilcoxon signed-rank tests to assess the significance of performance differences, and conduct a sensitivity analysis across three values of the objective weighting parameter alpha.

### **1.3 Deliverables**

The following deliverables were produced as part of this project:

1. **A hybrid optimisation and machine learning software framework.** Built using contemporary Python libraries and frameworks including DEAP, Gymnasium, and Stable-Baselines3, designed to solve parallel machine scheduling problems with asymmetric, sequence-dependent setup costs.
2. **A structured software repository.** Contains the full source code of the system, organised into modules for synthetic instance generation, schedule evaluation, classical baselines, and reinforcement learning agent training.
3. **Experimental documentation and results.** Provides a comprehensive archive of raw evaluation data stored in JSON format across 720 distinct experimental runs, alongside a suite of visualisations including Gantt charts, box plots, and convergence curves that document system performance.
4. **This MSc project report.** A comprehensive dissertation detailing the theoretical background, software design, implementation methodology, and statistical evaluation of the hybrid architecture.

**1.4 Ethical, Legal, and Social Issues**

This project involves no human participants, no personally identifiable data, and no privacy concerns. All data used in the experiments is synthetically generated using seeded pseudo-random number generators, ensuring reproducibility without recourse to real-world data. The project therefore raises no ethical concerns regarding data collection, consent, or privacy.

The software developed in this project uses only open-source libraries, each used under its respective license: NumPy (BSD-3-Clause), DEAP (LGPL-3.0-or-later), Gymnasium (MIT), and Stable-Baselines3 (MIT). No proprietary software was used in the development or execution of the experiments.

From a social perspective, scheduling optimisation has clear environmental and economic benefits. Improved scheduling in manufacturing reduces material waste from colour changeovers, decreases energy consumption through more efficient machine utilisation, and lowers production costs. These benefits are particularly relevant in energy-intensive industries such as textile dyeing and chemical processing, where sequence-dependent setup costs represent a substantial fraction of total production cost.

### **1.5 Structure of the Dissertation**

The remainder of this dissertation is organised as follows:

* **Chapter 2: Background Research** – Presents a comprehensive literature survey covering parallel machine scheduling, asymmetric setup costs, classical heuristics, genetic algorithms, deep reinforcement learning, and hyper-heuristics, concluding with a theoretical justification for the methods selected in this project.
* **Chapter 3: Software Requirements and System Design** – Outlines the functional and non-functional requirements of the system and details the structural design of the six core software modules, the custom Gymnasium environment, and the paired experimental framework.
* **Chapter 4: Implementation and Results** – Details the concrete software implementation and presents the quantitative experimental results, providing comprehensive performance comparisons, statistical significance analyses, and data visualisations.
* **Chapter 5: Evaluation and Discussion** – Critically evaluates the project findings, assesses the strengths and operational limitations of the hybrid hyper-heuristic, and discusses potential threats to validity.
* **Chapter 6: Conclusion** – Concludes the dissertation with a summary of major research contributions, key findings, directions for future algorithmic work, and a personal reflection on the project.
