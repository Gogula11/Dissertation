# Chapter 5. Evaluation and Discussion

## 5.1 Summary of Findings

The experimental results presented in Chapter 4 demonstrate a clear performance hierarchy across the four algorithms. On large problem instances (n ≥ 50), the hybrid GA+PPO approach consistently achieves the lowest composite scores, followed by standalone GA, then NN-Greedy, and finally SPT. On medium and small instances, GA and Hybrid converge to equivalent solutions.

The key quantitative findings are:

- Hybrid achieves 49-62% lower composite cost than SPT on large single-machine instances (p < 0.001).
- Hybrid achieves 47-62% lower composite cost than NN-Greedy on large single-machine instances (p < 0.001).
- Hybrid achieves 7-42% lower composite cost than standalone GA on large instances (p < 0.001), with the advantage increasing as the search space grows.
- On medium and small instances (n ≤ 20), GA and Hybrid produce equivalent results (improvement ≤ 5%, mostly not significant).
- The hybrid advantage is consistent across alpha values of 0.3, 0.5, and 0.7.

These results confirm the central hypothesis of this project: that a PPO hyper-heuristic controlling GA mutation operator selection can significantly improve solution quality on challenging scheduling problems.

![Figure 5.1: Box plots of composite scores showing the distribution across 50 seeds for each algorithm and instance configuration (lower is better).](../figures/05_boxplots_composite.png)

**Table 5.1: Mean composite scores (50 seeds)**

| Config | SPT | NN-Greedy | GA | Hybrid |
|--------|-----|-----------|-----|--------|
| n50_m1 | 0.530 | 0.522 | 0.349 | **0.201** |
| xn50_m1 | 0.548 | 0.531 | 0.421 | **0.278** |
| n50_m5 | 0.529 | 0.415 | 0.288 | **0.238** |
| n500_m10 | 0.562 | 0.527 | 0.527 | **0.491** |
| n20_m1 | 0.528 | 0.521 | 0.226 | **0.214** |
| n20_m3 | 0.526 | 0.357 | 0.181 | **0.181** |
| n10_m1 | 0.526 | 0.523 | 0.231 | **0.223** |
| n5_m1 | 0.524 | 0.523 | 0.252 | **0.247** |

## 5.2 Interpretation of Results

The performance advantage of the hybrid approach can be explained through the interaction between the PPO agent's adaptive mutation selection and the GA's convergence dynamics.

**Large instances.** On large problems (n ≥ 50), the search space is immense. The GA with fixed mutation parameters explores this space effectively in early generations but inevitably converges as the population loses diversity. The PPO agent detects this convergence through the observation features: the best fitness stagnates, the mean fitness approaches the best, and diversity declines. In response, the agent selects insertion mutation, which disrupts the converged population by removing and reinserting jobs at random positions. This allows the GA to escape local optima that would trap a fixed-mutation GA. The cycle of convergence and disruption is managed automatically by the agent, with the frequency of exploration-oriented actions increasing as the episode progresses.

The hybrid advantage is most pronounced on single-machine large instances (n50_m1: 42.2%, xn50_m1: 34.1%), where the search space is largest relative to the GA's ability to explore it. On multi-machine instances (n50_m5: 17.5%, n500_m10: 6.7%), the advantage is smaller but still significant, suggesting that the multi-machine structure provides additional exploration through the population-based search.

**Medium and small instances.** On medium problems (n = 20) and small problems (n = 10), the same dynamics apply but the margin of improvement is smaller because the GA can cover more of the search space with its fixed operators. On n20_m3, the difference is not statistically significant. This suggests a threshold effect: below approximately n = 30, the GA's fixed mutation operators are sufficient to explore the search space adequately within 300 generations.

**Alpha sensitivity.** The robustness of the hybrid advantage across alpha values suggests that the PPO agent learns a generalisable improvement strategy rather than an objective-specific trick. Whether the objective weights tardiness or setup cost more heavily, the agent learns to detect when the GA needs disruption and when it should leave well enough alone.

![Figure 5.5: Scatter plot of weighted tardiness versus setup cost for GA and Hybrid on n50_m1 (50 seeds each, lower is better — closer to origin). Hybrid solutions cluster toward the origin, indicating simultaneous improvement in both objectives rather than trading one off against the other.](../figures/05_scatter_tardiness_setup.png)

![Figure 5.2: Sensitivity analysis across alpha values of 0.3, 0.5, and 0.7 (lower is better). The hybrid advantage is consistent across all three weightings.](../figures/05_sensitivity_alpha.png)

## 5.3 PPO Agent Behaviour

The PPO agent's learned policy provides insight into why the hybrid approach outperforms the fixed-mutation GA. Analysis of action selections across episode stages reveals a clear behavioural pattern.

![Figure 5.3: PPO action frequency across episode stages. The agent never selects swap mutation, using insertion exclusively in early stages before shifting toward inversion in later stages.](../figures/04_action_frequency_thirds.png)

The agent never selects swap mutation (action 0) at any stage. Instead, it divides its selections between insertion mutation (action 2) and inversion mutation (action 1), shifting the balance as the episode progresses. In the early stage, the agent selects insertion mutation exclusively (100%). Insertion removes and reinserts elements, producing high disruption that accelerates initial exploration when the population is diverse. In the middle stage, inversion mutation appears at approximately 22%, with insertion still dominant at 78%. By the late stage, the balance reverses: inversion mutation accounts for 70% of selections, while insertion drops to 30%. Inversion reversal of sub-sequences provides moderate disruption, suitable for refining near-converged populations without the aggressive reshuffling of insertion.

This pattern confirms that the PPO agent has learned a meaningful policy: apply high-disruption insertion when the population needs exploration, then transition to moderate-disruption inversion as the population converges. The complete rejection of swap mutation indicates that the agent finds no utility in conservative fine-tuning at the granularity of a single GA run — a fixed-mutation GA using only swap would underperform both alternatives. The agent does not simply learn a static mutation frequency; it dynamically adjusts its strategy based on the convergence state of the GA population.

The training reward curves show that the agent's performance improves steadily during training, with episode rewards increasing from approximately 0.02 to 0.05 over 100,000 timesteps. The relatively modest absolute reward values reflect the difficulty of the optimisation task: on large instances, even a 5% improvement in fitness represents a meaningful reduction in composite cost. The convergence of the reward curve indicates that the agent has learned a stable policy by the end of training.

![Figure 5.4: PPO training reward curves (higher is better). The agent's performance improves steadily over 100,000 timesteps, converging to a stable policy.](../figures/04_ppo_curves.png)

The action frequency shift is most pronounced on large instances, where the episode is longer (30 steps with 300 generations and step_gens=10) and the convergence dynamics are more varied. On small instances, the policy is more uniform because the GA converges rapidly to the optimum regardless of the mutation operator chosen.

## 5.4 NN-Greedy Catastrophic Failures

A notable finding is the tendency of NN-Greedy to produce catastrophically poor solutions on certain instances. While NN-Greedy's mean performance is already worse than GA and Hybrid, its worst-case behaviour is dramatically worse. These failures occur because NN-Greedy makes locally optimal decisions at each step without considering the global schedule structure. The greedy approach can lock into a configuration that incurs catastrophic tardiness penalties, as the agent prioritises low setup costs at the expense of due-date performance.

GA and Hybrid never produce such failures. The evolutionary search explores the solution space broadly, and the population-based evaluation naturally filters out catastrophically poor solutions. This robustness is a practical advantage: in a real manufacturing environment, a single catastrophically poor schedule can disrupt production for days, making worst-case performance as important as average-case performance.

![Figure 5.6: Mean and worst-case composite scores for NN-Greedy, GA, and Hybrid across four large instance configurations (lower is better). NN-Greedy's worst-case performance is dramatically worse than its mean, while GA and Hybrid maintain consistent quality even in their worst runs.](../figures/05_nn_greedy_failures.png)

## 5.5 Practical Implications

The results have several practical implications for textile dyeing scheduling.

**Cost savings.** The 34-42% improvement in composite cost on large single-machine instances translates directly to reduced manufacturing costs. In textile dyeing, setup costs represent the chemicals, water, and time required to change colour between batches. A 35% reduction in setup cost on a large production run could save thousands of pounds per week in chemical costs alone, while also reducing water consumption and wastewater treatment requirements.

**Robustness.** The hybrid approach never produces catastrophic solutions, unlike NN-Greedy. In a real manufacturing environment, schedule quality consistency is critical: a single poor schedule can disrupt downstream processes, miss customer delivery windows, and incur penalty costs. The GA's population-based evaluation naturally filters out poor solutions, and the PPO agent's adaptive mutation further improves worst-case performance.

**Scalability.** The hybrid advantage increases with problem size, suggesting that the approach becomes more valuable as manufacturing operations scale. Small shops with few jobs may not benefit significantly from the hyper-heuristic, but medium and large operations with 50+ jobs per production run would see meaningful cost reductions.

**Computational cost.** The hybrid approach requires training the PPO model (approximately 45 minutes) and then executing the GA with the agent's policy. The evaluation GA runs take approximately 2-3 minutes per instance on a modern CPU, compared to under 1 second for NN-Greedy. For offline scheduling (planning production runs a day or two in advance), this computational cost is negligible. For real-time rescheduling in response to machine breakdowns or rush orders, the heuristic baselines may be preferred despite their lower solution quality.

## 5.6 Limitations

Several limitations of this study should be acknowledged.

**Generalisability to different cost structures.** The instance generator uses a specific colour-based cost structure motivated by textile dyeing. The generalisation to problem domains with fundamentally different cost structures — such as spatial distance costs or random asymmetry — remains an open question.

**Generalisability to larger instances.** The PPO agent was trained on instances ranging from n = 10 to n = 500. While this includes extra-large configurations, the generalisation to instances beyond n = 500 has not been tested. It is possible that the policy learned on n = 500 instances would not transfer effectively to n = 1000 or larger problems due to different convergence dynamics at larger scales.

**Observation space completeness.** The environment exposes only eight observation features to the agent. A richer observation space could include per-machine statistics (load balance, completion times), colour distribution entropy, or historical action effectiveness. Such features might enable the agent to learn a more sophisticated policy with finer-grained control.

**Single DRL algorithm.** Only PPO was evaluated. Alternative DRL algorithms such as Advantage Actor-Critic (A2C), Soft Actor-Critic (SAC), or Deep Q-Networks (DQN) might achieve different performance levels or exhibit different training dynamics. PPO was chosen for its stability and ease of use, but a systematic comparison of DRL algorithms for this environment could yield additional insights.

**Synthetic data only.** All experiments used synthetically generated instances with colour-based cost structure. While this provides a clean experimental framework with known ground truth, real manufacturing data may exhibit different cost distributions, due-date structures, and machine constraints. Validation on real factory data would strengthen claims about practical applicability.

**Baseline scope.** The comparison includes two classical heuristics (SPT and NN-Greedy) and a fixed-mutation GA. More sophisticated scheduling methods such as the NEH heuristic, Ant Colony Optimisation, or Tabu Search were not implemented. The hybrid approach may not outperform all of these on all configurations.

**Training budget.** The PPO agent was trained for 100,000 timesteps with a reduced population size of 25 and reduced generation count of 100. A longer training budget with the full evaluation parameters might yield a more effective policy. However, the training time increases proportionally, and the current budget was chosen as a pragmatic compromise between policy quality and computational cost.

## 5.7 Threats to Validity

**Internal validity.** DEAP's global state management is a known concern. The implementation addresses this through `hasattr` guards in the `build_toolbox` function and the use of `get_context("spawn")` for multiprocessing. Each worker process re-imports the module and re-registers DEAP types independently, preventing cross-process state contamination. However, within a single process (e.g., running the GA environment in a notebook), repeated calls to `build_toolbox` could interact with DEAP's creator registry if the `hasattr` guards fail for any reason.

**External validity.** The results may not transfer to different problem domains with different cost structures. The colour-based asymmetry in this study produces a specific pattern of setup costs (dark-to-light transitions are consistently expensive). Problems with different cost structures — such as random asymmetry or distance-based costs — might favour different mutation strategies. The robustness of the hybrid approach across cost structures remains an open question.

**Construct validity.** The composite objective with alpha = 0.5 represents a specific trade-off between tardiness and setup cost. While the sensitivity analysis shows that results are stable across alpha values, the choice of alpha remains somewhat arbitrary. A more thorough approach would treat this as a multi-objective optimisation problem and compare Pareto fronts. The significance of the results relative to this concern is somewhat mitigated by the sensitivity analysis, which confirms the pattern holds across the tested range. Additionally, the normalisation procedure using empirically sampled scales introduces a dependence on the specific schedules used for estimation, though this dependence is consistent across all algorithms compared on the same instance.

**Statistical conclusion validity.** The Wilcoxon signed-rank test is appropriate for the paired experimental design and does not assume normality. The sample size of 50 is adequate for this test. However, the use of multiple comparisons (eight configurations, multiple baseline comparisons) inflates the family-wise error rate. However, the hybrid vs GA comparisons on large instances (p < 0.001) survive even conservative Bonferroni corrections.
