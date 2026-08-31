# Hybrid GA+DRL for Parallel Machine Scheduling with Sequence-Dependent Setup Costs

A hybrid optimisation framework combining Genetic Algorithms (GA) and Deep Reinforcement Learning (DRL) to solve the Parallel Machine Scheduling Problem with asymmetric, Sequence-Dependent Setup Costs (PMSP-SDSC). A Proximal Policy Optimisation (PPO) agent is trained as a hyper-heuristic to dynamically select the GA's mutation operator based on real-time population convergence metrics.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
  instance_generator.py   # Synthetic PMSP-SDSC instance generation
  evaluator.py            # Schedule evaluation (tardiness, setup cost, composite)
  heuristics.py           # SPT and NN-Greedy baseline heuristics
  ga.py                   # Genetic Algorithm (DEAP-based)
  ga_env.py               # Custom Gymnasium environment wrapping the GA
  drl_agent.py            # PPO agent training and inference (Stable-Baselines3)
  visualisation.py        # Gantt charts, box plots, convergence curves

experiments/
  run_baselines.py        # Execute SPT and NN-Greedy baselines
  run_ga.py               # Execute standalone GA across configurations
  run_hybrid.py           # Execute hybrid GA+PPO across configurations
  train_ppo.py            # Train the PPO agent
  run_sensitivity.py      # Sensitivity analysis across alpha values
  run_action_freq.py      # Action-frequency analysis of trained policy
  find_best_seed.py       # Identify representative seeds for visualisation
  validate_bruteforce.py  # Brute-force validation on tiny instances

tests/
  test_evaluator.py       # Unit tests for schedule evaluation
  test_heuristics.py      # Unit tests for baseline heuristics
  test_instance_generator.py  # Unit tests for instance generation
  test_ga.py              # Structural tests for GA
  test_ga_env.py          # Structural tests for Gymnasium environment
  test_drl_agent.py       # Structural tests for PPO agent
  test_visualisation.py   # Unit tests for visualisation
```

## Running Experiments

```bash
# Run all baselines
python experiments/run_baselines.py

# Run standalone GA
python experiments/run_ga.py

# Train PPO agent
python experiments/train_ppo.py

# Run hybrid GA+PPO
python experiments/run_hybrid.py

# Run sensitivity analysis
python experiments/run_sensitivity.py
```

## Licence

This project is open-source under the BSD 3-Clause Licence. See `LICENSE` for details.

| Library | Version | Licence | Links |
|---------|---------|---------|-------|
| NumPy | 2.5.0 | BSD-3-Clause | [numpy.org](https://numpy.org) · [Licence](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| DEAP | 1.4.4 | LGPL-3.0-or-later | [github.com/DEAP/deap](https://github.com/DEAP/deap) · [Licence](https://github.com/DEAP/deap/blob/master/LICENSE.txt) |
| Gymnasium | 1.2.3 | MIT | [gymnasium.farama.org](https://gymnasium.farama.org) · [Licence](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE) |
| Stable-Baselines3 | 2.8.0 | MIT | [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io) · [Licence](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) |

Additional dependencies: PyTorch, Matplotlib, TensorBoard, pytest.
