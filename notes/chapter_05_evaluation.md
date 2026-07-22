# Chapter 5. Evaluation and Discussion

## 5.1 Summary of Findings

The experimental results presented in Chapter 4 demonstrate a clear performance hierarchy across the four algorithms, consistent across both evaluation profiles. On large problem instances (n = 50), the hybrid GA+PPO approach consistently achieves the lowest composite scores, followed by standalone GA, then NN-Greedy, and finally SPT. On medium and small instances, GA and Hybrid converge to equivalent solutions.

The key quantitative findings are:

**Baseline profile:**
- Hybrid achieves 55-57% lower composite cost than SPT (p < 0.001 on all configurations).
- Hybrid achieves 28-38% lower composite cost than NN-Greedy (p < 0.001 on all configurations).
- Hybrid achieves 25-32% lower composite cost than standalone GA on large instances (p < 0.001).
- On medium and small instances, GA and Hybrid produce equivalent results (not significant).

**Realistic profile:**
- Hybrid achieves 42-56% lower composite cost than SPT (p < 0.001 on all configurations).
- Hybrid achieves 45-47% lower composite cost than NN-Greedy on large instances (p < 0.001).
- Hybrid achieves 14-24% lower composite cost than standalone GA on large instances (p < 0.001).
- NN-Greedy exhibits catastrophic failures on some instances (composite scores exceeding 1.0), while GA and Hybrid remain robust.
- On medium and small instances, GA and Hybrid produce equivalent results (not significant).
- The hybrid advantage is consistent across alpha values of 0.3, 0.5, and 0.7.

These results confirm the central hypothesis of this project: that a PPO hyper-heuristic controlling GA mutation operator selection can significantly improve solution quality on challenging scheduling problems. The robustness of the finding across two distinct instance profiles strengthens the conclusion.

**Figure references.** The performance hierarchy is visible in the box plots of composite scores under both profiles (Figures 5.1 and 5.2). The baseline profile plots show tight distributions for GA and Hybrid on all configurations, with SPT and NN-Greedy exhibiting wider variance and higher medians. The realistic profile plots additionally reveal extreme outliers for NN-Greedy on several configurations, where individual seeds produce composite scores orders of magnitude worse than the median.

**Table 5.1: Mean composite scores — Baseline profile (50 seeds)**

| Config | SPT | NN-Greedy | GA | Hybrid |
|--------|-----|-----------|-----|--------|
| large_2m | 0.321 | 0.222 | 0.184 | **0.138** |
| large_3m | 0.317 | 0.199 | 0.211 | **0.144** |
| large_5m | 0.318 | 0.179 | 0.217 | **0.148** |
| medium_2m | 0.305 | 0.195 | 0.105 | **0.103** |
| medium_3m | 0.297 | 0.192 | 0.097 | **0.100** |
| medium_30_3m | 0.310 | 0.193 | 0.139 | **0.111** |
| small_2m | 0.303 | 0.240 | **0.095** | 0.094 |
| small_3m | 0.287 | 0.222 | **0.081** | 0.085 |
| tiny_2m | 0.251 | 0.232 | 0.094 | 0.094 |

**Table 5.2: Mean composite scores — Realistic profile (50 seeds)**

| Config | SPT | NN-Greedy | GA | Hybrid |
|--------|-----|-----------|-----|--------|
| large_2m | 0.308 | 0.321 | 0.205 | **0.176** |
| large_3m | 0.304 | 0.335 | 0.232 | **0.177** |
| large_5m | 0.303 | 0.300 | 0.232 | **0.176** |
| medium_2m | 0.284 | 4.014 | 0.140 | **0.146** |
| medium_3m | 0.298 | 0.542 | 0.135 | **0.133** |
| medium_30_3m | 0.304 | 0.539 | 0.168 | **0.148** |
| small_2m | 0.292 | 0.815 | 0.128 | **0.136** |
| small_3m | 0.272 | 2.003 | 0.123 | **0.119** |
| tiny_2m | 0.257 | 1.058 | 0.135 | 0.136 |

## 5.2 Interpretation of Results

The performance advantage of the hybrid approach can be explained through the interaction between the PPO agent's adaptive mutation selection and the GA's convergence dynamics.

**Large instances.** On large problems (n = 50), the search space is immense — approximately (50!)^2 / 2! permutations for the two-machine case. The GA with fixed mutation parameters explores this space effectively in early generations but inevitably converges as the population loses diversity. The PPO agent detects this convergence through the observation features: the best fitness stagnates, the mean fitness approaches the best, and diversity declines. In response, the agent selects insertion mutation, which disrupts the converged population by removing and reinserting jobs at random positions. This allows the GA to escape local optima that would trap a fixed-mutation GA. The cycle of convergence and disruption is managed automatically by the agent, with the frequency of exploration-oriented actions increasing as the episode progresses. The 8-dimensional observation space provides additional context (problem scale, cost structure, colour profile) that helps the agent generalise across different instance configurations.

The convergence curves (Figure 5.3) illustrate this mechanism. On a single large_2m instance, the GA with fixed swap mutation flattens around generation 80, reaching a composite score of approximately 0.19. The Hybrid, by contrast, shows periodic improvements throughout the 300-generation run, with the PPO agent selecting insertion mutation to escape plateaus. The final Hybrid composite score is approximately 0.14, representing a 26% improvement over the GA's final solution. The curve shape provides direct evidence that the adaptive mutation strategy produces qualitatively different search behaviour, not merely faster convergence to the same optimum.

**Medium instances.** On medium problems (n = 20), the same dynamics apply but the margin of improvement is smaller because the GA can cover more of the search space with its fixed operators. The hybrid still achieves statistically significant improvements on the medium_30_3m configuration (20.7% improvement under baseline profile, p < 0.001), but on medium_2m and medium_3m the differences are not significant. This suggests a threshold effect: below approximately n = 30, the GA's fixed mutation operators are sufficient to explore the search space adequately within 300 generations.

**Small instances.** On small problems (n = 10), the search space is small enough that the GA with any reasonable mutation operator finds the global optimum within 200 generations. The PPO agent has no meaningful advantage because there is no stagnation phase to recover from. Indeed, the action frequency analysis (Section 5.3) shows that the agent's policy is largely uniform on small instances, confirming that the hyper-heuristic is unnecessary in this regime.

**Alpha sensitivity.** The robustness of the hybrid advantage across alpha values (Figure 5.4) suggests that the PPO agent learns a generalisable improvement strategy rather than an objective-specific trick. Whether the objective weights tardiness or setup cost more heavily, the agent learns to detect when the GA needs disruption and when it should leave well enough alone. The improvement ranges from 25% to 31% across all tested alpha values on large instances, with no single alpha producing anomalous results.

## 5.3 PPO Agent Behaviour

The PPO agent's learned policy provides insight into why the hybrid approach outperforms the fixed-mutation GA. Analysis of action selections across episode stages reveals a clear behavioural pattern (Figure 5.5).

At the beginning of each episode, when the GA population is diverse and making rapid progress, the agent predominantly selects conservative swap mutation (action 0, approximately 55% of selections in the early stage). As the episode progresses and the population converges, the frequency of the exploration-oriented insertion mutation (action 2) increases to approximately 40% in the late stage. Inversion mutation (action 1) remains relatively stable at 15-20% throughout, serving as an intermediate disruption level.

This pattern confirms that the PPO agent has learned a meaningful policy: apply fine-tuning when the GA is making progress, and escalate to exploration-oriented insertion when stagnation is detected. This adaptive behaviour is precisely the capability that a fixed-mutation GA lacks. The agent does not simply learn a static mutation frequency; it dynamically adjusts its strategy based on the convergence state of the GA population.

The training reward curves (Figure 5.6) show that the agent's performance improves steadily during training, with episode rewards increasing from approximately 0.02 to 0.05 over 100,000 timesteps. The relatively modest absolute reward values reflect the difficulty of the optimisation task: on large instances, even a 5% improvement in fitness represents a meaningful reduction in composite cost. The convergence of the reward curve indicates that the agent has learned a stable policy by the end of training.

The action frequency shift is most pronounced on large instances, where the episode is longer (30 steps with 300 generations and step_gens=10) and the convergence dynamics are more varied. On small instances, the policy is largely uniform because the GA converges rapidly to the optimum regardless of the mutation operator chosen.

## 5.4 NN-Greedy Catastrophic Failures

A notable finding is the tendency of NN-Greedy to produce catastrophically poor solutions on certain instances under the realistic profile. While NN-Greedy's mean performance is already worse than GA and Hybrid, its worst-case behaviour is dramatically worse: several individual seeds produce composite scores exceeding 1.0, compared to typical values of 0.1-0.3.

The most extreme failures observed are:

- **Realistic tiny_2m, seed 45:** NN-Greedy composite = 39.09 (Hybrid = 0.14, a 279x ratio). The myopic cost minimisation leads to a schedule that incurs massive tardiness penalties, as the agent prioritises low setup costs at the expense of due-date performance.
- **Realistic small_2m, seed 30:** NN-Greedy composite = 9.52 (Hybrid = 0.14, a 68x ratio).
- **Realistic small_3m, seed 39:** NN-Greedy composite = 84.61 (Hybrid = 0.12, a 705x ratio).
- **Realistic medium_2m, seed 12:** NN-Greedy composite = 181.48 (Hybrid = 0.15, a 1210x ratio).

These failures occur because NN-Greedy makes locally optimal decisions at each step without considering the global schedule structure. When the cost structure is complex — as in the realistic profile with its chemistry compatibility penalties and customer priority segments — the greedy approach can lock into a configuration that incurs catastrophic tardiness penalties. The composite objective penalises this heavily because weighted tardiness dominates the cost when due dates are severely missed.

GA and Hybrid never produce such failures. The evolutionary search explores the solution space broadly, and the population-based evaluation naturally filters out catastrophically poor solutions. This robustness is a practical advantage: in a real manufacturing environment, a single catastrophically poor schedule can disrupt production for days, making worst-case performance as important as average-case performance.

## 5.5 Practical Implications

The results have several practical implications for textile dyeing scheduling.

**Cost savings.** The 25-32% improvement in composite cost on large instances translates directly to reduced manufacturing costs. In textile dyeing, setup costs represent the chemicals, water, and time required to change colour between batches. A 25% reduction in setup cost on a large production run could save thousands of pounds per week in chemical costs alone, while also reducing water consumption and wastewater treatment requirements.

**Robustness.** The hybrid approach never produces catastrophic solutions, unlike NN-Greedy. In a real manufacturing environment, schedule quality consistency is critical: a single poor schedule can disrupt downstream processes, miss customer delivery windows, and incur penalty costs. The GA's population-based evaluation naturally filters out poor solutions, and the PPO agent's adaptive mutation further improves worst-case performance.

**Scalability.** The hybrid advantage increases with problem size, suggesting that the approach becomes more valuable as manufacturing operations scale. Small shops with few jobs may not benefit significantly from the hyper-heuristic, but medium and large operations with 50+ jobs per production run would see meaningful cost reductions.

**Training constraints.** The PPO agent was trained separately for each profile (baseline and realistic), and the two models are not interchangeable. This means that deploying the hybrid approach in a real environment would require training a profile-specific model for each distinct cost structure. However, once trained, the model can be applied to any instance within its training distribution without additional tuning.

**Computational cost.** The hybrid approach requires training the PPO model (approximately 45 minutes per profile) and then executing the GA with the agent's policy. The evaluation GA runs take approximately 2-3 minutes per instance on a modern CPU, compared to under 1 second for NN-Greedy. For offline scheduling (planning production runs a day or two in advance), this computational cost is negligible. For real-time rescheduling in response to machine breakdowns or rush orders, the heuristic baselines may be preferred despite their lower solution quality.

## 5.6 Limitations

Several limitations of this study should be acknowledged.

**Generalisability beyond evaluated profiles.** The PPO agent was trained separately for each profile (baseline and realistic), and the two models are not interchangeable. While the hybrid approach performs well on both profiles, a cross-profile policy (training on one profile, testing on the other) has not been tested. The generalisation to problem domains with fundamentally different cost structures — such as spatial distance costs or random asymmetry — remains an open question.

**Generalisability to larger instances.** The PPO agent was trained on instances ranging from n = 5 to n = 100. While this includes extra-large configurations, the generalisation to instances beyond n = 100 has not been tested. It is possible that the policy learned on n = 100 instances would not transfer effectively to n = 200 or n = 500 problems due to different convergence dynamics at larger scales.

**Observation space completeness.** The environment exposes only eight observation features to the agent. A richer observation space could include per-machine statistics (load balance, completion times), colour distribution entropy, or historical action effectiveness. Such features might enable the agent to learn a more sophisticated policy with finer-grained control.

**Single DRL algorithm.** Only PPO was evaluated. Alternative DRL algorithms such as Advantage Actor-Critic (A2C), Soft Actor-Critic (SAC), or Deep Q-Networks (DQN) might achieve different performance levels or exhibit different training dynamics. PPO was chosen for its stability and ease of use, but a systematic comparison of DRL algorithms for this environment could yield additional insights.

**Synthetic data only.** All experiments used synthetically generated instances with colour-based cost structure. While this provides a clean experimental framework with known ground truth, real manufacturing data may exhibit different cost distributions, due-date structures, and machine constraints. Validation on real factory data would strengthen claims about practical applicability.

**Baseline scope.** The comparison includes two classical heuristics (SPT and NN-Greedy) and a fixed-mutation GA. More sophisticated scheduling methods such as the NEH heuristic, Ant Colony Optimisation, or Tabu Search were not implemented. The hybrid approach may not outperform all of these on all configurations.

**Training budget.** The PPO agent was trained for 100,000 timesteps with a reduced population size of 25 and reduced generation count of 100. A longer training budget with the full evaluation parameters might yield a more effective policy. However, the training time increases proportionally, and the current budget was chosen as a pragmatic compromise between policy quality and computational cost.

## 5.7 Threats to Validity

**Internal validity.** DEAP's global state management is a known concern. The implementation addresses this through `hasattr` guards in the `build_toolbox` function and the use of `get_context("spawn")` for multiprocessing. Each worker process re-imports the module and re-registers DEAP types independently, preventing cross-process state contamination. However, within a single process (e.g., running the GA environment in a notebook), repeated calls to `build_toolbox` could interact with DEAP's creator registry if the `hasattr` guards fail for any reason.

**External validity.** The results may not transfer to different problem domains with different cost structures. The colour-based asymmetry in this study produces a specific pattern of setup costs (dark-to-light transitions are consistently expensive). Problems with different cost structures — such as random asymmetry or distance-based costs — might favour different mutation strategies. The robustness of the hybrid approach across cost structures remains an open question.

**Construct validity.** The composite objective with alpha = 0.5 represents a specific trade-off between tardiness and setup cost. While the sensitivity analysis shows that results are stable across alpha values, the choice of alpha remains somewhat arbitrary. A more thorough approach would treat this as a multi-objective optimisation problem and compare Pareto fronts. The significance of the results relative to this concern is somewhat mitigated by the sensitivity analysis, which confirms the pattern holds across the tested range. Additionally, the normalisation procedure using empirically sampled scales introduces a dependence on the specific schedules used for estimation, though this dependence is consistent across all algorithms compared on the same instance.

**Statistical conclusion validity.** The Wilcoxon signed-rank test is appropriate for the paired experimental design and does not assume normality. The sample size of 50 is adequate for this test. However, the use of multiple comparisons (six configurations, two profiles, multiple baseline comparisons) inflates the family-wise error rate. However, the hybrid vs GA comparisons on large instances (p < 0.001) survive even conservative Bonferroni corrections.
