# COMP5200M — Scaffolding Code Reference
## For use alongside `COMP5200M_Guide.md`

> **How to use this document:** This is a code reference, not working software. Every module here is a structural starting point — read it, understand every line, then write your own version. Do not copy-paste blindly. Where you see a design decision (a specific data structure, a specific formula, a specific operator choice) refer to the guide for the reasoning behind it.
>
> **Reliability by module:**
> - `instance_generator.py` — high confidence, straightforward numpy
> - `evaluator.py` — high confidence, pure functions, easy to verify against hand calculations
> - `heuristics.py` — high confidence, simple logic
> - `ga.py` — medium confidence, DEAP global state requires care
> - `ga_env.py` — lower confidence, the step() loop needs checking against SB3's check_env()
> - `drl_agent.py` — medium confidence, SB3 API is stable but PPO hyperparameters need tuning

---

## `src/instance_generator.py`

```python
"""
Synthetic instance generator for PMSP-SDSC.

A 'problem instance' is a dict with:
  n          : int — number of jobs
  m          : int — number of machines
  proc_times : np.ndarray shape (n,) — processing time for each job
  due_dates  : np.ndarray shape (n,) — due date for each job
  weights    : np.ndarray shape (n,) — priority weights (1.0 for unweighted baseline)
  release    : np.ndarray shape (n,) — release times (0 for baseline)
  setup_cost : np.ndarray shape (n, n) — asymmetric transition cost matrix S
  setup_time : np.ndarray shape (n, n) — asymmetric transition time matrix
  colour_ids : np.ndarray shape (n,) — integer colour class for each job
"""

import numpy as np
from typing import Optional


# Colour darkness ranking — higher value = darker colour
COLOUR_DARKNESS = {
    0: 1,   # white
    1: 2,   # yellow
    2: 3,   # light blue
    3: 4,   # green
    4: 5,   # red
    5: 6,   # navy
    6: 7,   # black
}
N_COLOURS = len(COLOUR_DARKNESS)


def _build_cost_matrix(colour_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Asymmetric n×n cost matrix S.
    S[i][j] = cost of transitioning from job i's colour to job j's colour.
    Dark-to-light transitions are more expensive than light-to-dark.
    Diagonal is zero (same job repeated = no transition cost).
    """
    n = len(colour_ids)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            di = COLOUR_DARKNESS[colour_ids[i]]
            dj = COLOUR_DARKNESS[colour_ids[j]]
            darkness_diff = di - dj  # positive = going from dark to light (expensive)
            base_cost = max(0, darkness_diff) * 10.0
            noise = rng.uniform(0.0, 2.0)
            S[i][j] = base_cost + noise
    return S


def generate_instance(
    n: int,
    m: int,
    seed: Optional[int] = None,
    tightness: float = 1.5,
) -> dict:
    """
    Generate one synthetic PMSP-SDSC instance.

    Args:
        n:         number of jobs
        m:         number of machines
        seed:      random seed for reproducibility
        tightness: due-date tightness factor (lower = tighter deadlines)
    Returns:
        instance dict
    """
    rng = np.random.default_rng(seed)

    proc_times = rng.integers(5, 31, size=n).astype(np.float32)
    colour_ids = rng.integers(0, N_COLOURS, size=n)

    setup_cost = _build_cost_matrix(colour_ids, rng)
    setup_time = (setup_cost / 10.0) * rng.uniform(0.8, 1.2, size=(n, n))
    np.fill_diagonal(setup_time, 0.0)

    total_proc = proc_times.sum()
    avg_load_per_machine = total_proc / m
    due_dates = (proc_times / proc_times.sum()) * avg_load_per_machine * tightness * n
    due_dates += rng.uniform(0, avg_load_per_machine * 0.2, size=n)
    due_dates = due_dates.astype(np.float32)

    return {
        "n": n,
        "m": m,
        "proc_times": proc_times,
        "due_dates": due_dates,
        "weights": np.ones(n, dtype=np.float32),
        "release": np.zeros(n, dtype=np.float32),
        "setup_cost": setup_cost,
        "setup_time": setup_time,
        "colour_ids": colour_ids,
    }


# Pre-defined experiment configurations
INSTANCE_CONFIGS = [
    {"n": 10, "m": 2, "label": "small_2m"},
    {"n": 10, "m": 3, "label": "small_3m"},
    {"n": 20, "m": 2, "label": "medium_2m"},
    {"n": 20, "m": 3, "label": "medium_3m"},
    {"n": 50, "m": 2, "label": "large_2m"},
    {"n": 50, "m": 3, "label": "large_3m"},
]
```

---

## `src/evaluator.py`

```python
"""
Pure evaluator for PMSP-SDSC solutions.

A solution sigma is a list of m lists:
  sigma[k] = ordered list of job indices assigned to machine k

Example for n=6, m=2:
  sigma = [[0, 3, 5], [1, 2, 4]]
  Machine 0: jobs 0 → 3 → 5
  Machine 1: jobs 1 → 2 → 4
"""

import numpy as np
from typing import List


def validate_sigma(sigma: List[List[int]], n: int) -> None:
    """Raise ValueError if any job is missing or duplicated."""
    all_jobs = sorted([j for seq in sigma for j in seq])
    if all_jobs != list(range(n)):
        raise ValueError(f"Invalid sigma: expected jobs 0..{n-1}, got {all_jobs}")


def compute_completion_times(sigma: List[List[int]], instance: dict) -> np.ndarray:
    """
    Compute completion time C_i for every job.
    Returns array of shape (n,).
    """
    n = instance["n"]
    proc = instance["proc_times"]
    setup_t = instance["setup_time"]
    release = instance["release"]

    C = np.zeros(n, dtype=np.float32)

    for machine_seq in sigma:
        if not machine_seq:
            continue
        t = 0.0
        for idx, job in enumerate(machine_seq):
            t = max(t, release[job])
            if idx > 0:
                prev_job = machine_seq[idx - 1]
                t += setup_t[prev_job][job]
            t += proc[job]
            C[job] = t

    return C


def compute_tardiness(C: np.ndarray, instance: dict) -> np.ndarray:
    """T_i = max(0, C_i - d_i) for each job."""
    return np.maximum(0.0, C - instance["due_dates"])


def compute_weighted_tardiness(T: np.ndarray, instance: dict) -> float:
    """f1 = sum(w_i * T_i)"""
    return float(np.dot(instance["weights"], T))


def compute_setup_cost(sigma: List[List[int]], instance: dict) -> float:
    """f2 = total sequence-dependent setup cost across all machines."""
    S = instance["setup_cost"]
    total = 0.0
    for machine_seq in sigma:
        for idx in range(1, len(machine_seq)):
            total += S[machine_seq[idx - 1]][machine_seq[idx]]
    return float(total)


def compute_makespan(C: np.ndarray) -> float:
    """C_max = max(C_i)"""
    return float(C.max())


def evaluate(sigma: List[List[int]], instance: dict, alpha: float = 0.5) -> dict:
    """
    Full evaluation of a solution.

    Args:
        sigma:    list of m machine sequences (job index lists)
        instance: problem instance dict
        alpha:    objective weighting (0=setup cost only, 1=tardiness only)

    Returns:
        dict with: weighted_tardiness, setup_cost, composite, makespan,
                   tardiness_per_job, completion_times
    """
    validate_sigma(sigma, instance["n"])

    C = compute_completion_times(sigma, instance)
    T = compute_tardiness(C, instance)
    f1 = compute_weighted_tardiness(T, instance)
    f2 = compute_setup_cost(sigma, instance)
    F = alpha * f1 + (1 - alpha) * f2

    return {
        "weighted_tardiness": f1,
        "setup_cost": f2,
        "composite": F,
        "makespan": compute_makespan(C),
        "tardiness_per_job": T,
        "completion_times": C,
    }
```

---

## `src/heuristics.py`

```python
"""
Baseline scheduling heuristics for PMSP-SDSC.
All heuristics take an instance dict and return a sigma (list of m lists).
"""

import numpy as np
from typing import List


def spt(instance: dict) -> List[List[int]]:
    """
    Shortest Processing Time (SPT).
    Sort jobs ascending by processing time, assign round-robin to machines.
    Ignores all setup costs.
    """
    n, m = instance["n"], instance["m"]
    order = np.argsort(instance["proc_times"])
    sigma = [[] for _ in range(m)]
    for i, job in enumerate(order):
        sigma[i % m].append(int(job))
    return sigma


def nearest_neighbour_greedy(instance: dict) -> List[List[int]]:
    """
    Nearest-Neighbour Greedy heuristic.
    Assigns the next job to the machine with lowest current load,
    selecting the unscheduled job with the lowest setup cost from
    that machine's last job.
    """
    n, m = instance["n"], instance["m"]
    S = instance["setup_cost"]
    proc = instance["proc_times"]

    unscheduled = set(range(n))
    sigma = [[] for _ in range(m)]
    machine_time = np.zeros(m, dtype=np.float32)
    machine_last = [None] * m

    while unscheduled:
        k = int(np.argmin(machine_time))

        if machine_last[k] is None:
            job = min(unscheduled, key=lambda j: proc[j])
        else:
            last = machine_last[k]
            job = min(unscheduled, key=lambda j: S[last][j])

        sigma[k].append(job)
        unscheduled.remove(job)
        if machine_last[k] is not None:
            machine_time[k] += instance["setup_time"][machine_last[k]][job]
        machine_time[k] += proc[job]
        machine_last[k] = job

    return sigma
```

---

## `src/ga.py`

```python
"""
DEAP-based Genetic Algorithm for PMSP-SDSC.

Chromosome encoding: flat permutation of n job indices (giant tour).
Split into m equal-length segments to assign jobs to machines.

DEAP global state warning: creator.create() registers globally.
The hasattr guards below prevent re-registration errors in notebooks,
but always restart the kernel if you change fitness weights.
"""

import random
import numpy as np
from typing import List, Optional, Callable
from deap import base, creator, tools, algorithms

from src.evaluator import evaluate


def decode_chromosome(individual: list, m: int) -> List[List[int]]:
    """
    Split flat permutation into m machine sequences (equal-ish split).
    Machine k gets jobs individual[start:start+size].
    """
    n = len(individual)
    sigma = []
    per_machine = n // m
    remainder = n % m
    start = 0
    for k in range(m):
        size = per_machine + (1 if k < remainder else 0)
        sigma.append(individual[start:start + size])
        start += size
    return sigma


def make_fitness_fn(instance: dict, alpha: float) -> Callable:
    """Returns a DEAP-compatible fitness function (must return a tuple)."""
    def fitness(individual):
        sigma = decode_chromosome(list(individual), instance["m"])
        result = evaluate(sigma, instance, alpha=alpha)
        return (result["composite"],)
    return fitness


def build_toolbox(instance: dict, alpha: float = 0.5) -> base.Toolbox:
    """
    Construct and return a DEAP Toolbox for this instance.

    Registers:
      individual, population, evaluate, mate, mutate_swap,
      mutate_inversion, mutate (default=swap), select
    """
    n = instance["n"]

    # Guard against DEAP global re-registration
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", make_fitness_fn(instance, alpha))
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate_swap",      tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("mutate_inversion", tools.mutInversion)
    toolbox.register("mutate",           tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_ga(
    instance: dict,
    n_gen: int = 200,
    pop_size: int = 100,
    cx_prob: float = 0.8,
    mut_prob: float = 0.2,
    alpha: float = 0.5,
    seed: Optional[int] = None,
    verbose: bool = False,
    mutation_strategy: str = "swap",
) -> dict:
    """
    Run the plain GA (no DRL) on a given instance.

    Args:
        mutation_strategy: "swap" | "inversion"

    Returns:
        dict with: best_fitness, weighted_tardiness, setup_cost, makespan,
                   best_sigma, logbook, hof
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    toolbox = build_toolbox(instance, alpha)

    if mutation_strategy == "inversion":
        toolbox.register("mutate", tools.mutInversion)
    else:
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min",  np.min)
    stats.register("mean", np.mean)
    stats.register("std",  np.std)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    best_sigma = decode_chromosome(list(hof[0]), instance["m"])
    best_result = evaluate(best_sigma, instance, alpha)

    return {
        "best_fitness": best_result["composite"],
        "weighted_tardiness": best_result["weighted_tardiness"],
        "setup_cost": best_result["setup_cost"],
        "makespan": best_result["makespan"],
        "best_sigma": best_sigma,
        "logbook": logbook,
        "hof": hof,
    }
```

---

## `src/ga_env.py`

```python
"""
Gymnasium environment wrapping the GA for PPO control.

Each episode = one full GA run.
Each step    = step_gens generations of the GA with the chosen mutation operator.
Action       = which mutation operator to use for the next step_gens generations.
Observation  = 4D vector summarising current GA population state.
Reward       = relative improvement in best fitness over the step.

IMPORTANT: Run check_env(env) from stable_baselines3.common.env_checker
before training. Fix all warnings before proceeding.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from src.ga import build_toolbox, decode_chromosome
from src.evaluator import evaluate
from deap import tools


def _population_diversity(pop, sample_size: int = 20) -> float:
    """
    Mean pairwise normalised Hamming distance between a sample of individuals.
    Sampled to keep computation fast on large populations.
    """
    n_ind = min(len(pop), sample_size)
    if n_ind < 2:
        return 0.0
    n_genes = len(pop[0])
    total, count = 0.0, 0
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            diffs = sum(1 for a, b in zip(pop[i], pop[j]) if a != b)
            total += diffs / n_genes
            count += 1
    return total / count if count > 0 else 0.0


class GAHyperHeuristicEnv(gym.Env):
    """
    Gymnasium env where PPO controls the GA's mutation operator selection.

    observation_space: Box(4,) — [best_norm, mean_norm, diversity, stagnation_norm]
    action_space:      Discrete(3)
      0 = swap mutation (conservative)
      1 = inversion mutation (moderate)
      2 = aggressive swap (high disruption)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: dict,
        total_gens: int = 200,
        step_gens: int = 10,
        pop_size: int = 100,
        cx_prob: float = 0.8,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.instance = instance
        self.total_gens = total_gens
        self.step_gens = step_gens
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.alpha = alpha
        self.max_steps = total_gens // step_gens

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Initialised in reset()
        self.toolbox = None
        self.pop = None
        self.hof = None
        self.current_step = 0
        self.best_at_start = None
        self.stagnation_count = 0
        self.last_best = None

    def _obs(self) -> np.ndarray:
        fitnesses = [ind.fitness.values[0] for ind in self.pop]
        best = float(min(fitnesses))
        mean = float(np.mean(fitnesses))
        denom = max(self.best_at_start, 1e-6)
        div = _population_diversity(self.pop)
        stag = min(self.stagnation_count / max(self.total_gens, 1), 1.0)
        return np.array([best / denom, mean / denom, div, stag], dtype=np.float32)

    def _apply_action(self, action: int):
        if action == 0:
            self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        elif action == 1:
            self.toolbox.register("mutate", tools.mutInversion)
        else:
            self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.20)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.toolbox = build_toolbox(self.instance, self.alpha)
        self.pop = self.toolbox.population(n=self.pop_size)
        self.hof = tools.HallOfFame(1)

        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        self.hof.update(self.pop)
        self.best_at_start = float(self.hof[0].fitness.values[0])
        self.last_best = self.best_at_start
        self.stagnation_count = 0
        self.current_step = 0

        return self._obs(), {}

    def step(self, action: int):
        self._apply_action(int(action))
        best_before = float(self.hof[0].fitness.values[0])

        # Run step_gens generations manually
        # NOTE: this mirrors eaSimple — verify it matches your GA's logic
        for _ in range(self.step_gens):
            offspring = self.toolbox.select(self.pop, len(self.pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            self.pop[:] = offspring
            self.hof.update(self.pop)

        best_after = float(self.hof[0].fitness.values[0])

        reward = (best_before - best_after) / max(best_before, 1e-6)

        if best_after < self.last_best - 1e-6:
            self.stagnation_count = 0
            self.last_best = best_after
        else:
            self.stagnation_count += self.step_gens

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        return self._obs(), float(reward), terminated, False, {}

    def get_best_result(self) -> dict:
        """Call after episode ends. Returns same format as evaluator.evaluate()."""
        sigma = decode_chromosome(list(self.hof[0]), self.instance["m"])
        return evaluate(sigma, self.instance, self.alpha)
```

---

## `src/drl_agent.py`

```python
"""
PPO training and inference for the GA hyper-heuristic.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ga_env import GAHyperHeuristicEnv


def make_env_fn(instance, total_gens=200, step_gens=10, pop_size=100, alpha=0.5):
    def _init():
        return GAHyperHeuristicEnv(
            instance, total_gens=total_gens,
            step_gens=step_gens, pop_size=pop_size, alpha=alpha
        )
    return _init


def train_ppo(
    instance: dict,
    total_timesteps: int = 50_000,
    save_path: str = "models/ppo_hyperheuristic",
    verbose: int = 1,
) -> PPO:
    """
    Train a PPO agent on the GA environment for the given instance.
    Saves the model to save_path.zip.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    vec_env = DummyVecEnv([make_env_fn(instance)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log="logs/ppo_training/",
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved to {save_path}.zip")
    return model


def run_hybrid(
    instance: dict,
    model: PPO,
    seed: int = None,
    total_gens: int = 200,
    step_gens: int = 10,
    pop_size: int = 100,
    alpha: float = 0.5,
) -> dict:
    """
    Run the GA with the trained PPO hyper-heuristic.
    Returns same result dict format as run_ga().
    """
    import random
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = GAHyperHeuristicEnv(
        instance, total_gens=total_gens,
        step_gens=step_gens, pop_size=pop_size, alpha=alpha
    )
    obs, _ = env.reset(seed=seed)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    return env.get_best_result()
```

---

## `tests/test_evaluator.py`

```python
"""
Tests for the evaluator. Run with: python -m pytest tests/ -v
These are the minimum tests — write more as you find edge cases.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.instance_generator import generate_instance
from src.evaluator import evaluate, compute_completion_times, validate_sigma


def test_all_jobs_scheduled():
    inst = generate_instance(n=10, m=2, seed=0)
    sigma = [[0,1,2,3,4], [5,6,7,8,9]]
    result = evaluate(sigma, inst)
    assert len(result["completion_times"]) == 10
    assert all(result["completion_times"] > 0)


def test_completion_time_monotone_single_machine():
    """On a single machine, completion times must be strictly increasing."""
    inst = generate_instance(n=5, m=1, seed=42)
    sigma = [[0,1,2,3,4]]
    C = compute_completion_times(sigma, inst)
    for i in range(1, len(C)):
        assert C[i] > C[i-1]


def test_zero_setup_zero_cost():
    inst = generate_instance(n=4, m=1, seed=7)
    inst["setup_cost"][:] = 0.0
    inst["setup_time"][:] = 0.0
    sigma = [[0,1,2,3]]
    result = evaluate(sigma, inst)
    assert result["setup_cost"] == 0.0


def test_invalid_sigma_caught():
    inst = generate_instance(n=4, m=2, seed=0)
    bad_sigma = [[0,1,2,3], [0]]  # job 0 appears twice
    try:
        evaluate(bad_sigma, inst)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_tardiness_nonnegative():
    inst = generate_instance(n=10, m=2, seed=5)
    sigma = [[0,1,2,3,4], [5,6,7,8,9]]
    result = evaluate(sigma, inst)
    assert all(result["tardiness_per_job"] >= 0)


def test_same_seed_same_instance():
    inst_a = generate_instance(n=20, m=2, seed=99)
    inst_b = generate_instance(n=20, m=2, seed=99)
    np.testing.assert_array_equal(inst_a["proc_times"], inst_b["proc_times"])
    np.testing.assert_array_equal(inst_a["setup_cost"], inst_b["setup_cost"])


if __name__ == "__main__":
    test_all_jobs_scheduled()
    test_completion_time_monotone_single_machine()
    test_zero_setup_zero_cost()
    test_invalid_sigma_caught()
    test_tardiness_nonnegative()
    test_same_seed_same_instance()
    print("All tests passed.")
```

---

## `experiments/run_baselines.py`

```python
"""
Run SPT and nearest-neighbour baselines across all instance configs.
Saves to results/raw/baselines.json
Run from project root: python experiments/run_baselines.py
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.heuristics import spt, nearest_neighbour_greedy
from src.evaluator import evaluate

ALPHA = 0.5
N_SEEDS = 30


def run():
    results = {cfg["label"]: {"spt": [], "nn_greedy": []} for cfg in INSTANCE_CONFIGS}

    for cfg in INSTANCE_CONFIGS:
        print(f"Running baselines: {cfg['label']}")
        for seed in range(N_SEEDS):
            inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
            for name, fn in [("spt", spt), ("nn_greedy", nearest_neighbour_greedy)]:
                sigma = fn(inst)
                ev = evaluate(sigma, inst, alpha=ALPHA)
                results[cfg["label"]][name].append({
                    "seed": seed,
                    "composite":          ev["composite"],
                    "weighted_tardiness": ev["weighted_tardiness"],
                    "setup_cost":         ev["setup_cost"],
                    "makespan":           ev["makespan"],
                })

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/raw/baselines.json")


if __name__ == "__main__":
    run()
```

---

## `experiments/run_ga.py`

```python
"""
Run plain GA for all instance configs, 30 seeds each, parallelised.
Run from project root: python experiments/run_ga.py
Do NOT run this from a notebook — multiprocessing + DEAP global state = problems.
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import Pool
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.ga import run_ga

ALPHA = 0.5
N_SEEDS = 30
# Lock these down after tuning in notebook 03
GA_PARAMS = {"n_gen": 200, "pop_size": 100, "cx_prob": 0.8, "mut_prob": 0.2}


def run_one(args):
    cfg, seed = args
    inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
    result = run_ga(inst, **GA_PARAMS, alpha=ALPHA, seed=seed)
    return cfg["label"], {
        "seed":               seed,
        "composite":          result["best_fitness"],
        "weighted_tardiness": result["weighted_tardiness"],
        "setup_cost":         result["setup_cost"],
        "makespan":           result["makespan"],
    }


def run():
    tasks = [(cfg, seed) for cfg in INSTANCE_CONFIGS for seed in range(N_SEEDS)]
    results = {cfg["label"]: [] for cfg in INSTANCE_CONFIGS}

    with Pool() as pool:
        for label, data in pool.imap_unordered(run_one, tasks):
            results[label].append(data)
            print(f"  Done: {label} seed={data['seed']}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/ga.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/raw/ga.json")


if __name__ == "__main__":
    run()
```

---

## `experiments/run_hybrid.py`

```python
"""
Run hybrid GA+DRL for all instance configs, 30 seeds each.
Requires a trained PPO model at models/ppo_hyperheuristic.zip
Run from project root: python experiments/run_hybrid.py
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from src.instance_generator import generate_instance, INSTANCE_CONFIGS
from src.drl_agent import run_hybrid

ALPHA = 0.5
N_SEEDS = 30
MODEL_PATH = "models/ppo_hyperheuristic"


def run():
    model = PPO.load(MODEL_PATH)
    results = {cfg["label"]: [] for cfg in INSTANCE_CONFIGS}

    for cfg in INSTANCE_CONFIGS:
        print(f"Running hybrid: {cfg['label']}")
        for seed in range(N_SEEDS):
            inst = generate_instance(n=cfg["n"], m=cfg["m"], seed=seed)
            result = run_hybrid(inst, model, seed=seed, total_gens=200, alpha=ALPHA)
            results[cfg["label"]].append({
                "seed":               seed,
                "composite":          result["composite"],
                "weighted_tardiness": result["weighted_tardiness"],
                "setup_cost":         result["setup_cost"],
                "makespan":           result["makespan"],
            })
            print(f"  seed={seed} composite={result['composite']:.2f}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/hybrid.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/raw/hybrid.json")


if __name__ == "__main__":
    run()
```

---

## Statistical Analysis Snippet (for `notebooks/05_final_evaluation.ipynb`)

```python
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

with open("results/raw/baselines.json") as f: baselines = json.load(f)
with open("results/raw/ga.json")        as f: ga_raw    = json.load(f)
with open("results/raw/hybrid.json")    as f: hyb_raw   = json.load(f)

LABELS = ["small_2m", "small_3m", "medium_2m", "medium_3m", "large_2m", "large_3m"]

def vals(data, label, alg=None):
    if alg:
        return np.array([r["composite"] for r in data[label][alg]])
    return np.array([r["composite"] for r in data[label]])

rows = []
for label in LABELS:
    spt_v = vals(baselines, label, "spt")
    nn_v  = vals(baselines, label, "nn_greedy")
    ga_v  = vals(ga_raw,    label)
    hyb_v = vals(hyb_raw,   label)

    def fmt(v): return f"{v.mean():.1f}±{v.std():.1f}"
    def pval(a, b):
        if np.all(a == b): return "1.0000"
        _, p = wilcoxon(a, b, alternative="less")
        return f"{p:.4f}"

    rows.append({
        "Instance":        label,
        "SPT":             fmt(spt_v),
        "NN Greedy":       fmt(nn_v),
        "GA":              fmt(ga_v),
        "Hybrid":          fmt(hyb_v),
        "p vs SPT":        pval(hyb_v, spt_v),
        "p vs NN":         pval(hyb_v, nn_v),
        "p vs GA":         pval(hyb_v, ga_v),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results/figures/comparison_table.csv", index=False)
```

---

## Gantt Chart Snippet (for notebooks)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.evaluator import compute_completion_times

COLOUR_MAP = {
    0: "#FFFACD", 1: "#FFD700", 2: "#87CEEB",
    3: "#228B22", 4: "#DC143C", 5: "#000080", 6: "#1C1C1C",
}

def plot_gantt(sigma, instance, title="Schedule", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))

    proc     = instance["proc_times"]
    setup_t  = instance["setup_time"]
    colours  = instance["colour_ids"]
    m        = instance["m"]

    for k, seq in enumerate(sigma):
        t = 0.0
        for idx, job in enumerate(seq):
            if idx > 0:
                st = float(setup_t[seq[idx-1]][job])
                ax.barh(k, st, left=t, height=0.35,
                        color="lightgrey", edgecolor="black", hatch="//", linewidth=0.5)
                t += st
            colour = COLOUR_MAP.get(int(colours[job]), "#cccccc")
            ax.barh(k, float(proc[job]), left=t, height=0.6,
                    color=colour, edgecolor="black", linewidth=0.5)
            ax.text(t + proc[job]/2, k, str(job),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            t += float(proc[job])

    ax.set_yticks(range(m))
    ax.set_yticklabels([f"Machine {k}" for k in range(m)])
    ax.set_xlabel("Time units")
    ax.set_title(title)
    patches = [mpatches.Patch(color=c, label=f"Colour {i}") for i, c in COLOUR_MAP.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=4)
    return ax
```

---

## Env Checker Snippet (run before training PPO)

```python
from stable_baselines3.common.env_checker import check_env
from src.instance_generator import generate_instance
from src.ga_env import GAHyperHeuristicEnv

inst = generate_instance(n=10, m=2, seed=0)
env = GAHyperHeuristicEnv(inst, total_gens=50, step_gens=5, pop_size=20)
check_env(env, warn=True)
print("Environment check passed.")
```

Fix every warning before training. Warnings become silent errors during PPO training.
