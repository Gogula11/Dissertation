# Chapter 2. Background Research

## 2.1 Literature Survey

This section presents a survey of prior research relevant to the project, organised as a narrative progression: the problem domain is introduced, followed by the solution methods that have been applied to it, and finally the gap in the existing literature that this project addresses.

### 2.1.1 Parallel Machine Scheduling

The Parallel Machine Scheduling Problem (PMSP) is a well-studied combinatorial optimisation problem in which a set of n jobs must be assigned to m identical machines operating in parallel. Each job has a processing time and must be processed by exactly one machine. The objective typically involves minimising a function of completion times, such as makespan (the maximum completion time across all machines) or total weighted tardiness (the sum of penalties for jobs completed after their due dates, weighted by job priority).

PMSP is known to be NP-hard in the general case (Garey & Johnson, 1979). Even the single-machine variant with weighted tardiness minimisation is NP-hard (Lawler, 1977). Introduction of additional constraints and cost structures — such as sequence-dependent setup costs, release times, and job weights — increases the problem complexity further. Pinedo (2016) provides a comprehensive treatment of scheduling theory, classifying PMSP variants by objective function, machine environment, and processing constraints.

The search space for PMSP-SDSC is the set of all ways to partition n jobs among m machines and order them within each machine. An approximate bound is O((n!)^m / m!), reflecting the factorial growth in job permutations on each machine, divided by the symmetry of identical machines. For a medium-sized instance (n = 20, m = 2), the search space exceeds 10^36 points, precluding exhaustive enumeration or exact methods.

### 2.1.2 Sequence-Dependent Setup Costs

In real-world manufacturing, transitioning between different jobs on the same machine incurs a setup cost — time, material, or both. This cost depends on the specific pair of jobs being sequenced consecutively, giving rise to a sequence-dependent setup cost matrix. When this matrix is asymmetric — meaning the cost of transitioning from job i to job j differs from the cost of going from job j to job i — the problem becomes substantially harder to solve heuristically.

The motivating domain for this project is textile dyeing, where batches of fabric are dyed in sequence. Each batch has a target colour, and the dyeing machine must be cleaned between batches. The cleaning cost depends on the colour transition: moving from a dark colour to a light colour requires extensive cleaning to prevent contamination of the lighter batch, whereas moving from light to dark requires minimal cleaning. This creates a natural asymmetry in the setup cost matrix.

Standard scheduling heuristics, such as Shortest Processing Time, treat all inter-job transitions as having zero or uniform cost. This assumption is violated in the presence of sequence-dependent setup costs, making these heuristics unsuitable for domains such as textile dyeing. Explicit modelling of the asymmetric cost structure is necessary for effective scheduling in such environments.

#### Problem Formalisation

The PMSP-SDSC problem addressed in this project is formally defined as follows. Let J = {0, 1, ..., n-1} be a set of n jobs and M = {0, 1, ..., m-1} be a set of m identical parallel machines. Each job j has a processing time p_j, a due date d_j, a weight w_j (priority), and a release time r_j. Transitioning from job i to job j on the same machine incurs a sequence-dependent setup time s_ij and setup cost c_ij, where c_ij is generally not equal to c_ji.

| Symbol | Definition |
|--------|-----------|
| n | Number of jobs |
| m | Number of identical parallel machines |
| p_j | Processing time of job j |
| d_j | Due date of job j |
| w_j | Priority weight of job j |
| r_j | Release time of job j (earliest start) |
| c_ij | Setup cost of transitioning from job i to job j |
| s_ij | Setup time of transitioning from job i to job j |
| C_j | Completion time of job j |
| T_j | Tardiness of job j: max(0, C_j - d_j) |
| sigma | Solution: list of m machine sequences |

A solution sigma is a partition of the n jobs into m sequences, one per machine, where sigma[k] is the ordered list of jobs assigned to machine k. The completion time C_j for each job is computed by traversing its machine's sequence, accumulating processing times, setup times (between consecutive jobs), and respecting release times.

The weighted tardiness objective is:

f1 = sum over j of w_j * T_j = sum over j of w_j * max(0, C_j - d_j)

The total setup cost objective is:

f2 = sum over k of sum over consecutive pairs (i, j) on machine k of c_ij

These two objectives are normalised to a common scale and combined into a single composite objective:

F = alpha * (f1 / f1_scale) + (1 - alpha) * (f2 / f2_scale)

where f1_scale and f2_scale are normalisation constants estimated for the given instance, and alpha in [0, 1] controls the trade-off between tardiness and setup cost minimisation. The normalisation constants are computed by evaluating SPT, NN-Greedy, and a random schedule on the instance, and setting each scale to 1.5 times the maximum observed value of the corresponding objective. This heuristic sampling approach ensures both objectives contribute equitably to the composite score, preventing the larger-magnitude objective from dominating.

### 2.1.3 Classical Heuristics for Scheduling

Classical heuristics for scheduling problems are valued for their simplicity, speed, and deterministic behaviour. They serve as baselines against which more sophisticated methods can be compared.

The Shortest Processing Time (SPT) heuristic sorts jobs in ascending order of processing time and assigns them round-robin to machines. Its time complexity is O(n log n), dominated by the sorting step. SPT is optimal for minimising mean completion time on a single machine but makes no attempt to minimise setup costs or tardiness, as it ignores due dates and setup matrices entirely.

The NEH heuristic (Nawaz, Enscore, & Ham, 1983) is an insertion-based constructive method originally developed for the flow shop scheduling problem. It builds a schedule by repeatedly inserting jobs into the partial sequence at the position that minimises the partial objective. NEH is widely regarded as one of the best-performing constructive heuristics for permutation scheduling problems, with a time complexity of O(n^2 m). However, its performance on asymmetric SDSC variants has not been extensively characterised.

The Nearest-Neighbour Greedy heuristic is a sequential assignment method that builds machine schedules one job at a time. At each step, it selects the machine with the lowest current load and assigns the unscheduled job that minimises the transition cost from that machine's last job. The greedy approach makes locally optimal decisions that may compound into poor global solutions.

Additional heuristics relevant to the PMSP domain include the Earliest Due Date (EDD) rule, which prioritises jobs with the earliest due dates and is optimal for minimising maximum lateness on a single machine, and the Weighted Shortest Processing Time (WSPT) rule, which orders jobs by the ratio of weight to processing time. These heuristics are not included as baselines in this project because they target different objective functions and their performance on the composite PMSP-SDSC objective is expected to be inferior to SPT and NN-Greedy.

These heuristics are included in this project as baselines because they are fast, deterministic, and standard in the scheduling literature. They are not expected to produce optimal solutions for PMSP-SDSC, but they provide a reference point for evaluating the more complex GA and hybrid approaches.

### 2.1.4 Genetic Algorithms for Scheduling

Genetic Algorithms are a class of evolutionary optimisation methods inspired by natural selection (Holland, 1975). In the scheduling domain, GAs operate on a population of candidate solutions, iteratively applying selection, crossover, and mutation to produce improved generations.

A common encoding for scheduling problems is the permutation representation, where a chromosome is a permutation of job indices representing the order in which jobs are processed. For the multi-machine case, the giant-tour representation encodes a single permutation of all n jobs, which is then split into m segments corresponding to the machines. This representation has the advantage of naturally handling the job assignment and sequencing decisions simultaneously.

The theoretical foundation of GAs rests on the schema theorem (Holland, 1975) and the building block hypothesis (Goldberg, 1989). A schema is a pattern of gene values with fixed and wildcard positions; short, low-order schemata with above-average fitness are called building blocks. The schema theorem shows that building blocks receive exponentially increasing trials in successive generations under selection, crossover, and mutation. In the permutation encoding context, the relevant schemata are relative orderings of subsets of jobs, and Order Crossover is designed to preserve these order-based building blocks. This theoretical grounding explains why GA frameworks with permutation encoding and crossover operators that preserve relative ordering are effective for scheduling problems, where the ordering of jobs within a machine's sequence is critical to solution quality.

Order Crossover (OX) is a crossover operator designed for permutation representations. It preserves the relative ordering of jobs from one parent while incorporating job positions from the other, maintaining the feasibility of the offspring permutation. This property is important for PMSP-SDSC because the relative ordering of jobs encodes colour transition information that affects setup costs.

Mutation operators for permutation encodings include swap mutation (exchanging two random positions), inversion mutation (reversing a sub-sequence), and insertion mutation (removing one element and inserting it elsewhere). Swap mutation with a low per-gene probability (indpb = 0.05) makes fine-grained changes suitable for conservative fine-tuning. Inversion mutation reverses a sub-sequence, producing moderate reordering. Insertion mutation with a higher per-gene probability (indpb = 0.15) removes and reinserts multiple elements, generating high disruption for exploration. The choice of mutation operator controls the balance between exploration and exploitation in the GA.

The DEAP framework (Fortin et al., 2012) provides a mature, modular implementation of GA components, including built-in support for permutation operators, Hall-of-Fame elitism, and population statistics. It was chosen for this project due to its flexibility and the ease with which custom operators can be integrated.

A known limitation of GAs is that as the population converges, the diversity of candidate solutions decreases, and the algorithm may stagnate on local optima. Fixed mutation operators with static parameters cannot adapt to this changing state, motivating the need for adaptive control strategies.

### 2.1.5 Deep Reinforcement Learning for Combinatorial Optimisation

Deep Reinforcement Learning combines the sequential decision-making framework of reinforcement learning with the function approximation capabilities of deep neural networks. In the context of combinatorial optimisation, DRL offers an alternative to hand-crafted heuristics and metaheuristics: instead of designing a fixed algorithm, a neural network learns a policy that can adapt to the specific problem instance and search state.

Proximal Policy Optimisation (Schulman et al., 2017) is a policy gradient method that has become a standard choice for DRL due to its stability and ease of use. PPO employs a clipped surrogate objective that constrains the policy update at each step, preventing destructive large updates that can occur in vanilla policy gradient methods. This trust-region property makes PPO robust to hyperparameter choices and suitable for a wide range of environments.

Prior work has applied DRL to scheduling problems. Zhang et al. (2020) used DRL for dynamic job shop scheduling, where the agent selects dispatching rules based on the current shop floor state. Kool, van Hoof, and Welling (2019) applied attention-based pointer networks to the travelling salesman and vehicle routing problems, demonstrating that neural network architectures designed for sequence-to-sequence problems can learn effective heuristics for combinatorial optimisation. These approaches show that DRL can capture complex patterns in scheduling data that are difficult to encode in hand-crafted rules. However, they address the optimisation problem directly, requiring the agent to learn the entire search process from scratch, which demands large action spaces and extensive training.

Compared to Deep Q-Networks (Mnih et al., 2015), PPO offers several advantages for the scheduling domain. DQN uses a value-based approach, learning the Q-function and deriving a policy from it. While effective for discrete action spaces, DQN can be unstable when the Q-function is approximated by a neural network, requiring techniques such as experience replay and target networks. PPO belongs to the policy gradient family, which optimises the policy directly through gradient ascent on expected return. The key innovation in PPO is the clipped surrogate objective:

L_CLIP(theta) = E_t[min(r_t(theta) * A_t, clip(r_t(theta), 1-epsilon, 1+epsilon) * A_t)]

where r_t(theta) is the probability ratio between the new and old policies, A_t is the advantage estimate, and epsilon is a clipping hyperparameter (typically 0.2). This clipping mechanism constrains the policy update to a trust region, preventing destructive large updates that can occur in vanilla policy gradient methods. The result is stable, monotonic improvement that is robust to hyperparameter choices. The PPO implementation in this project uses the stable hyperparameters recommended by Stable-Baselines3: a learning rate of 3e-4, n_steps of 2048, batch size of 64, 10 epochs per update, a discount factor of 0.99, and an entropy coefficient of 0.05 to encourage exploration during training.

The advantages of PPO for the hyper-heuristic scheduling context are threefold. First, the clipped objective ensures stable training across episodes with varying difficulty (different instances from the training pool). Second, the policy gradient formulation naturally handles stochastic policies, allowing exploration during training. Third, PPO has been extensively validated across a diverse set of environments in the Gymnasium benchmark suite and is well-supported in production RL frameworks.

### 2.1.6 Hyper-heuristics

Burke et al. (2013) formally define a hyper-heuristic as "an automated methodology for selecting or generating heuristics to solve computational search problems." The key distinction from a standard metaheuristic is the separation of two levels: the domain level, where problem-specific heuristics operate, and the meta level, where the hyper-heuristic controls which heuristic to apply and when.

Hyper-heuristics are categorised into heuristic selection (choosing among existing heuristics) and heuristic generation (creating new heuristics from components). This project falls into the heuristic selection category: the DRL agent selects among three GA mutation operators at each decision step during the GA execution.

The hyper-heuristic framing is what distinguishes this project from direct RL-for-scheduling approaches. The DRL agent does not learn to construct schedules directly — that would require an action space proportional to the number of jobs, which grows factorially. Instead, the agent controls the GA's mutation operator, keeping the action space small (Discrete(3)) while leveraging the GA's existing search capability. This separation of concerns allows each component to operate at its appropriate level of abstraction: the GA explores the scheduling solution space efficiently, while the DRL agent modulates the exploration strategy based on the current search state.

### 2.1.7 The Gap

Prior work has applied GAs to PMSP with setup costs, and prior work has applied DRL to scheduling problems. However, no existing study has combined these specifically as a PPO-driven hyper-heuristic for mutation operator selection in the context of asymmetric, parallel machine scheduling with sequence-dependent setup costs. The closest related work involves DRL for operator selection in continuous optimisation (Sharma & Sutton, 2021) and GAs with adaptive mutation rates (Thierens, 2002), but neither addresses the asymmetric SDSC domain. Furthermore, this work evaluates the hybrid approach across two distinct instance profiles — a simple categorical colour model and a more complex continuous colour model with chemistry constraints and customer segments — to assess the robustness of the hyper-heuristic under varying problem difficulty.

This dissertation fills that gap by: (1) formalising the PMSP-SDSC problem with an asymmetric cost structure motivated by real-world manufacturing; (2) designing a Gymnasium environment that enables a PPO agent to control GA mutation at a hyper-heuristic level; and (3) empirically demonstrating that the hybrid approach significantly outperforms both classical heuristics and standalone GA on large problem instances across two distinct evaluation profiles.

## 2.2 Methods and Techniques

This section provides a brief, balanced survey of the methods and techniques available for solving the PMSP-SDSC problem, covering instance generation, solution representation, GA frameworks, RL frameworks, environment design, objective weighting, and statistical testing.

**Instance generation.** Synthetic instance generation is necessary because there is no standard benchmark set for PMSP-SDSC with asymmetric costs. The instance generator must produce reproducible instances with controlled properties: processing times drawn from a uniform distribution, due dates calibrated by a tightness parameter, and asymmetric setup costs structured around colour classes. This project defines two evaluation profiles for instance generation: a baseline profile using 7 discrete colour classes with uniform distribution and linear cost asymmetry, and a realistic profile using 12 continuous colour families with skewed distribution, chemistry compatibility penalties, colour clustering, and customer priority tiers. The dual-profile design tests whether the hybrid approach's performance generalises beyond the simplest cost model.

**Solution representation.** The standard representation for permutation-based scheduling is a list of m machine sequences (sigma), where each machine has an ordered list of jobs. The giant-tour encoding flattens this into a single permutation of n jobs, which is then split into m segments. Alternative encodings include binary matrices (job-machine assignment) and priority-rule-based representations, but the permutation encoding is the most natural for genetic operators designed for sequencing problems.

**GA frameworks.** The available Python frameworks for implementing GAs include DEAP (Fortin et al., 2012), PyGAD (Gad, 2021), and custom implementations. DEAP was selected for its mature support of permutation operators, its Hall-of-Fame elitism mechanism, and its toolbox-based architecture that allows runtime modification of registered operators — a capability essential for the hyper-heuristic environment.

**RL frameworks.** Stable-Baselines3 (Raffin et al., 2021) provides reliable, well-tested implementations of modern DRL algorithms including PPO. Alternatives include TensorFlow Agents and custom PyTorch implementations, but SB3 offers the best balance of documentation, community support, and integration with the Gymnasium environment API.

**Environment API.** The Gymnasium interface (formerly OpenAI Gym) provides a standardised API for RL environments with `reset()` and `step()` methods. This interface ensures compatibility with SB3's training loop and allows the use of built-in utilities such as `DummyVecEnv` for parallelisation and `check_env` for validation.

**Objective weighting.** The composite objective combines weighted tardiness and setup cost using a convex combination controlled by alpha. When alpha = 0, the objective is pure setup cost minimisation; when alpha = 1, it is pure tardiness minimisation. Normalisation is essential because the two objectives operate on different scales — on large instances, tardiness can dominate setup cost by an order of magnitude.

**Statistical testing.** The Wilcoxon signed-rank test (Wilcoxon, 1945) is a non-parametric paired difference test. It is appropriate for this project because the experimental design is paired (the same seeds are used across algorithms), the normality assumption of the paired t-test cannot be guaranteed for scheduling objective values, and 30 paired samples are sufficient for the test to achieve reasonable statistical power.

## 2.3 Choice of Methods

This section justifies the specific design decisions made for this project, providing the rationale for each choice in light of the alternatives surveyed in Section 2.2.

**Why GA?** The Genetic Algorithm was chosen as the base solver because permutation encoding maps naturally to the scheduling domain. Order Crossover preserves relative job ordering, which is critical for maintaining colour transition information. The three mutation operators (swap, inversion, aggressive swap) define a meaningful action space for the hyper-heuristic, providing distinct exploration strategies ranging from conservative fine-tuning to high disruption. The DEAP framework supports runtime operator registration, enabling the hyper-heuristic environment to swap mutation operators on the fly.

**Why PPO?** Proximal Policy Optimisation was chosen over alternative DRL algorithms for three reasons. First, its clipped surrogate objective provides stable training without the need for extensive hyperparameter tuning, which is important given the computational cost of training. Second, PPO has been extensively validated on a wide range of continuous-state environments, including those with structured observations similar to our 4-dimensional state space. Third, SB3's PPO implementation is production-quality, well-documented, and includes utilities for TensorBoard logging and model checkpointing.

**Why PPO as hyper-heuristic (not end-to-end RL)?** This is the most important design decision in the project. End-to-end RL for scheduling — where the agent directly outputs job assignments — would require an action space proportional to the number of jobs, which is impractical for instances of realistic size. By instead using the DRL agent as a hyper-heuristic that controls the GA's mutation operator, the action space is reduced to three discrete actions regardless of problem size. This design leverages the GA's existing search capability while adding adaptive control at the meta level. The separation of concerns — the GA searches the solution space; the DRL agent controls the search strategy — is the key insight that makes the hybrid approach tractable.

**Why composite objective (alpha = 0.5)?** The composite objective with alpha = 0.5 was chosen as the primary evaluation metric because it balances two competing objectives: weighted tardiness (representing customer satisfaction) and setup cost (representing manufacturing efficiency). The sensitivity analysis at alpha = 0.3 and 0.7 tests whether the conclusions are robust to the choice of weighting. Normalisation of both objectives to a common scale prevents either one from dominating the composite score due to differences in magnitude.

**Why instance pool training?** Training on a single instance would cause the PPO agent to overfit to that instance's specific characteristics. By training on an instance pool of 110 diverse instances (11 configurations × 10 seeds each), the agent must learn a generalisable policy that performs well across different problem sizes and structures. During training, each episode randomly samples one instance from the pool, exposing the agent to varied search dynamics. The training uses a reduced population size of 25 (compared to 100 for evaluation), intentionally making each episode harder for the GA to improve on its own and encouraging the agent to learn effective mutation selection rather than relying on brute-force search.

**Why 50 seeds?** Fifty random seeds per configuration is a robust choice for statistical comparison. This sample size provides good statistical power for the Wilcoxon signed-rank test while keeping the total experimental workload manageable (50 seeds × 6 configs × 4 algorithms = 1200 runs per profile). The paired design — using the same 50 seeds for all four algorithms — ensures that differences in solution quality can be attributed to algorithm performance rather than random variation in instance difficulty across seeds.

**Why Wilcoxon over t-test?** The Wilcoxon signed-rank test was chosen over the paired t-test for three reasons. First, Wilcoxon is non-parametric and does not require the assumption of normally distributed differences, which is unlikely to hold for scheduling objective values. Second, the test is based on ranks rather than absolute values, making it robust to outliers — important when some instances may produce anomalously poor schedules. Third, Wilcoxon is appropriate for the paired experimental design used in this project, where the same set of 30 random seeds is applied to all algorithms on each configuration.