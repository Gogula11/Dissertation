# Chapter 5. Evaluation and Discussion

## 5.1 Summary of Findings

The experimental results presented in Chapter 4 demonstrate a clear performance hierarchy across the four algorithms. The hybrid GA+PPO approach consistently achieves the lowest composite scores on large- and medium-sized problem instances, followed by standalone GA, then NN-Greedy, and finally SPT. On small instances, GA and Hybrid converge to equivalent solutions.

The key quantitative findings are:

- Hybrid achieves 53% lower composite cost than SPT (p < 0.001 on all configurations).
- Hybrid achieves 46% lower composite cost than NN-Greedy (p < 0.001 on all configurations except small_2m where p < 0.05).
- Hybrid achieves 63% lower composite cost than standalone GA on large_2m (p < 0.01) and 82% lower on large_3m (p < 0.001).
- On medium instances, Hybrid maintains a 50-67% advantage over GA (p < 0.05 on both configurations).
- On small instances, GA and Hybrid produce equivalent results (not significant).
- The hybrid advantage is consistent across alpha values of 0.3, 0.5, and 0.7.

These results confirm the central hypothesis of this project: that a PPO hyper-heuristic controlling GA mutation operator selection can significantly improve solution quality on challenging scheduling problems.

## 5.2 Interpretation of Results

The performance advantage of the hybrid approach can be explained through the interaction between the PPO agent's adaptive mutation selection and the GA's convergence dynamics.

**Large instances.** On large problems (n = 50), the search space is immense — approximately (50!)^2 / 2! permutations for the two-machine case. The GA with fixed mutation parameters explores this space effectively in early generations but inevitably converges as the population loses diversity. The PPO agent detects this convergence through the observation features: the best fitness stagnates, the mean fitness approaches the best, and diversity declines. In response, the agent selects aggressive swap mutation, which disrupts the converged population and injects new diversity. This allows the GA to escape local optima that would trap a fixed-mutation GA. The cycle of convergence and disruption is managed automatically by the agent, with the frequency of aggressive actions increasing as the episode progresses.

**Medium instances.** On medium problems (n = 20), the same dynamics apply but the margin of improvement is smaller because the GA can cover more of the search space with its fixed operators. The hybrid still achieves statistically significant improvements, but the practical benefit is less dramatic.

**Small instances.** On small problems (n = 10), the search space is small enough that the GA with any reasonable mutation operator finds the global optimum within 200 generations. The PPO agent has no meaningful advantage because there is no stagnation phase to recover from. Indeed, the action frequency analysis shows that the agent's policy is largely uniform on small instances, confirming that the hyper-heuristic is unnecessary in this regime.

**Alpha sensitivity.** The robustness of the hybrid advantage across alpha values suggests that the PPO agent learns a generalisable improvement strategy rather than an objective-specific trick. Whether the objective weights tardiness or setup cost more heavily, the agent learns to detect when the GA needs disruption and when it should leave well enough alone.

## 5.3 Limitations

Several limitations of this study should be acknowledged.

**Generalisability to larger instances.** The PPO agent was trained on instances ranging from n = 10 to n = 50. While this includes the large configuration, the largest instances are only one step above medium. The generalisation to instances beyond n = 50 has not been tested. It is possible that the policy learned on n = 50 instances would not transfer effectively to n = 100 or n = 200 problems due to different convergence dynamics at larger scales.

**Observation space completeness.** The environment exposes only four observation features to the agent. A richer observation space could include per-machine statistics (load balance, completion times), colour distribution entropy, or historical action effectiveness. Such features might enable the agent to learn a more sophisticated policy with finer-grained control.

**Single DRL algorithm.** Only PPO was evaluated. Alternative DRL algorithms such as Advantage Actor-Critic (A2C), Soft Actor-Critic (SAC), or Deep Q-Networks (DQN) might achieve different performance levels or exhibit different training dynamics. PPO was chosen for its stability and ease of use, but a systematic comparison of DRL algorithms for this environment could yield additional insights.

**Synthetic data only.** All experiments used synthetically generated instances with colour-based cost structure. While this provides a clean experimental framework with known ground truth, real manufacturing data may exhibit different cost distributions, due-date structures, and machine constraints. Validation on real factory data would strengthen claims about practical applicability.

**Baseline scope.** The comparison includes two classical heuristics (SPT and NN-Greedy) and a fixed-mutation GA. More sophisticated scheduling methods such as the NEH heuristic, Ant Colony Optimisation, or Tabu Search were not implemented. The hybrid approach may not outperform all of these on all configurations.

**Training budget.** The PPO agent was trained for 100,000 timesteps with a reduced population size of 50. A longer training budget with the full population size of 100 might yield a more effective policy. However, the training time increases proportionally, and the current budget was chosen as a pragmatic compromise between policy quality and computational cost.

## 5.4 Threats to Validity

**Internal validity.** DEAP's global state management is a known concern. The implementation addresses this through `hasattr` guards in the `build_toolbox` function and the use of `get_context("spawn")` for multiprocessing. Each worker process re-imports the module and re-registers DEAP types independently, preventing cross-process state contamination. However, within a single process (e.g., running the GA environment in a notebook), repeated calls to `build_toolbox` could interact with DEAP's creator registry if the `hasattr` guards fail for any reason.

**External validity.** The results may not transfer to different problem domains with different cost structures. The colour-based asymmetry in this study produces a specific pattern of setup costs (dark-to-light transitions are consistently expensive). Problems with different cost structures — such as random asymmetry or distance-based costs — might favour different mutation strategies. The robustness of the hybrid approach across cost structures remains an open question.

**Construct validity.** The composite objective with alpha = 0.5 represents a specific trade-off between tardiness and setup cost. While the sensitivity analysis shows that results are stable across alpha values, the choice of alpha remains somewhat arbitrary. A more thorough approach would treat this as a multi-objective optimisation problem and compare Pareto fronts. The significance of the results relative to this concern is somewhat mitigated by the sensitivity analysis, which confirms the pattern holds across the tested range.

**Statistical conclusion validity.** The Wilcoxon signed-rank test is appropriate for the paired experimental design and does not assume normality. The sample size of 30 is adequate for this test. However, the use of multiple comparisons (six configurations, two baseline comparisons per configuration) inflates the family-wise error rate. A Bonferroni correction would raise the significance threshold to approximately p < 0.008 for 6 comparisons. Under this correction, the Hybrid vs GA comparisons on medium instances (p < 0.05) would no longer be considered significant, but the comparisons against SPT and NN-Greedy would remain significant.