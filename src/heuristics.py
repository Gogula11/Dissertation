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
