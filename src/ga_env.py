"""
Gymnasium environment wrapping the GA for PPO control.

Each episode = one full GA run.
Each step    = step_gens generations of the GA with the chosen mutation operator.
Action       = which mutation operator to use for the next step_gens generations.
Observation  = 4D vector summarising current GA population state.
Reward       = relative improvement in best fitness over the step.

Reward rationale:
  reward = (best_before - best_after) / max(best_before, 1e-6)
  This is the fractional improvement in best fitness over step_gens generations.
  Positive when GA improves, zero when plateaued, negative if best worsens
  (possible due to elitism not being strict in the current step).
  A small penalty (-0.01) replaces zero reward to discourage idle actions
  that neither help nor hurt — this prevents the PPO from learning to stall.

IMPORTANT: Run check_env(env) from stable_baselines3.common.env_checker
before training. Fix all warnings before proceeding.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from src.ga import build_toolbox, decode_chromosome, mutInsertion
from src.evaluator import evaluate, estimate_scales
from deap import tools


def _population_diversity(pop, sample_size: int = 20) -> float:
    n_ind = min(len(pop), sample_size)
    if n_ind < 2:
        return 0.0
    arr = np.array([list(ind) for ind in pop[:n_ind]])
    i_idx, j_idx = np.triu_indices(n_ind, k=1)
    return float((arr[i_idx] != arr[j_idx]).mean())


class GAHyperHeuristicEnv(gym.Env):
    """
    Gymnasium env where PPO controls the GA's mutation operator selection.

    observation_space: Box(4,) — [best_norm, mean_norm, diversity, stagnation_norm]
    action_space:      Discrete(3)
      0 = swap mutation (conservative, indpb=0.05)
      1 = inversion mutation (segment reversal)
      2 = insertion mutation (remove-and-reinsert, indpb=0.15)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: dict,
        total_gens: int = 200,
        step_gens: int = 10,
        pop_size: int = 100,
        cx_prob: float = 0.8,
        mut_prob: float = 0.2,
        alpha: float = 0.5,
        instance_pool: list = None,
    ):
        super().__init__()
        self.instance = instance
        self.instance_pool = instance_pool if instance_pool is not None else [instance]
        self.total_gens = total_gens
        self.step_gens = step_gens
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
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
        stag = min(self.stagnation_count / max(self.max_steps, 1), 1.0)
        best_norm = np.clip(best / denom, 0.0, 1.0)
        mean_norm = np.clip(mean / denom, 0.0, 1.0)

        return np.array([
            best_norm, mean_norm, div, stag,
        ], dtype=np.float32)

    def _apply_action(self, action: int):
        if action == 0:
            self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        elif action == 1:
            self.toolbox.register("mutate", tools.mutInversion)
        else:
            self.toolbox.register("mutate", mutInsertion, indpb=0.15)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Sample random instance from pool
        self.instance = self.instance_pool[
            int(random.randint(0, len(self.instance_pool) - 1))
        ]
        self._f1_scale, self._f2_scale = estimate_scales(self.instance)
        self.toolbox = build_toolbox(self.instance, self.alpha, self._f1_scale, self._f2_scale)
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

        for _ in range(self.step_gens):
            offspring = self.toolbox.select(self.pop, len(self.pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mut_prob:
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
        if abs(reward) < 1e-8:
            reward = -0.01

        if best_after < self.last_best - 1e-6:
            self.stagnation_count = 0
            self.last_best = best_after
        else:
            self.stagnation_count += 1

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._obs(), float(reward), False, truncated, {}

    def get_best_result(self) -> dict:
        """Call after episode ends. Returns same format as evaluator.evaluate()."""
        sigma = decode_chromosome(list(self.hof[0]), self.instance["m"])
        return evaluate(sigma, self.instance, self.alpha,
                        f1_scale=self._f1_scale, f2_scale=self._f2_scale)
