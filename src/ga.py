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


def mutInsertion(individual: list, indpb: float = 0.05) -> tuple:
    """Insertion mutation: remove a random element, insert at random position."""
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            idx = random.randrange(size)
            elem = individual.pop(idx)
            ins = random.randrange(size)
            individual.insert(ins, elem)
    return (individual,)


def mutAggressiveSwap(individual: list, indpb: float = 0.20) -> tuple:
    """Aggressive swap mutation: higher disruption via mutShuffleIndexes."""
    return tools.mutShuffleIndexes(individual, indpb=indpb)


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
    toolbox.register("mutate_swap",           tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("mutate_inversion",      tools.mutInversion)
    toolbox.register("mutate_insertion",      mutInsertion, indpb=0.15)
    toolbox.register("mutate_aggressive_swap", mutAggressiveSwap, indpb=0.20)
    toolbox.register("mutate",                tools.mutShuffleIndexes, indpb=0.05)
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
        mutation_strategy: "swap" | "inversion" | "insertion"

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
    elif mutation_strategy == "insertion":
        toolbox.register("mutate", mutInsertion, indpb=0.15)
    elif mutation_strategy == "aggressive_swap":
        toolbox.register("mutate", mutAggressiveSwap, indpb=0.20)
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
