# Project Audit — 2026-04-28

## Done ✓
- `src/` all 6 modules. `experiments/` all 3 scripts. `notebooks/` 01-05. 720 runs (baselines+ga+hybrid). Figures (11 PNG). Stats tables. PPO model. Wilcoxon tests.
- Timeline: way ahead. Notes plan Jun-Aug for evaluation, done Apr 28.

## Critical Issues 🔴

### 1. Evaluator missing normalisation (notes §6.1, §6.4 — "critical")
- Code: `F = α·f1 + (1−α)·f2` raw
- f1 (tardiness) range 0-500+, f2 (setup) 50-200 → f1 dominates → α meaningless
- Notebook 02 self-flags this. Fix in `src/evaluator.py:77`.

### 2. Hybrid LOSES to GA (results_summary.tex)
| config | GA | Hybrid |
|---|---|---|
| large_2m | 87.5 | 130.9 ❌ |
| large_3m | 78.3 | 129.1 ❌ |
| medium_2m | 15.1 | 16.0 ❌ |
| medium_3m | 6.6 | 7.2 ❌ |

Cause: PPO trained on **single instance** (`notebooks/04` cell 7). Notes §9.6 + §9.10 explicitly warn: instance pool required, single = overfit.

### 3. GA missing aggressive_swap mutation (notes §8.1)
- `run_ga()` handles only `"swap"|"inversion"`. `ga_env` has all 3 but env+ga inconsistent.
- `src/ga.py:103`

### 4. ga_env.step duplicates eaSimple loop with hardcoded mut_prob=0.2
- `src/ga_env.py:144`
- Code comment admits "verify it matches your GA's logic" — it doesn't fully.

## Medium Issues 🟡

### 5. Tests incomplete
Only `test_evaluator.py`. Notes need `test_heuristics.py`, `test_ga.py`.

### 6. Action frequency analysis shallow
Notes §9.9 want early/middle/late thirds histograms. Notebook 04 has 1 bar chart total.

### 7. No instance pool in env
Single instance per `reset()`. Notes §9.6.

### 8. Reward sparsity fallback missing
Notes §9.7.

### 9. Stagnation bookkeeping
`src/ga_env.py:158` — increments by `step_gens`, compares to `total_gens`, denominator inconsistent with step counter.

### 10. PPO TensorBoard logging off
`tensorboard_log=None`, `logs/` empty.

### 11. Sensitivity analysis post-hoc only
Notebook 05 cell 12 re-normalises stored runs. Notes §10.6 want **re-run** GA+Hybrid on medium configs with α ∈ {0.3, 0.5, 0.7}, 10 seeds each.

## Missing 🔵

### 12. No Gantt SPT-vs-Hybrid same-instance comparison
Notes §11.1.

### 13. No dissertation .tex draft
No chapter files present.

### 14. Literature review status unknown

## Recommended Next Steps (priority order)

1. Fix evaluator normalisation (1 hour) → invalidates current results, must rerun
2. Add `aggressive_swap` to `run_ga` + unify env/ga mutation interface
3. Add instance pool training to `ga_env`
4. Retrain PPO on pool, 100k timesteps, TensorBoard on
5. Rerun all 720 with normalised evaluator
6. Add missing tests (`test_heuristics.py`, `test_ga.py`)
7. Start dissertation Ch3 + Ch4 (have all code + results)

## Bottom Line

Code complete, results exist, but **PPO undertrained + evaluator unnormalised** = headline result currently shows hybrid worse than GA. Both fixable in 1-2 days. Then focus shift to writing.
