"""Exhaustive ground-truth validation: enumerate ALL schedules for tiny
instances (n<=8, m=2) and compare the GA result against the known optimum.

Usage: python experiments/validate_bruteforce.py
"""
import itertools
from src.instance_generator import generate_instance
from src.evaluator import evaluate, estimate_scales
from src.ga import run_ga

def brute_force_optimum(inst, alpha=0.7):
    f1s, f2s = estimate_scales(inst)
    n = inst["n"]; jobs = list(range(n)); best = float("inf")
    for mask in range(1 << n):
        A = [j for j in jobs if mask >> j & 1]
        for pa in itertools.permutations(A):
            rest = [j for j in jobs if j not in pa]
            for pb in itertools.permutations(rest):
                c = evaluate([list(pa), list(pb)], inst,
                             alpha=alpha, f1_scale=f1s, f2_scale=f2s)["composite"]
                if c < best:
                    best = c
    return best

def main():
    results = {}
    for n in (6, 8):
        inst = generate_instance(jobs_per_machine=n // 2, m=2, seed=7)
        opt = brute_force_optimum(inst)
        ga = run_ga(inst, n_gen=40, pop_size=60, alpha=0.7, seed=7)["best_fitness"]
        gap = (ga - opt) / opt * 100
        results[f"n{n}_m2"] = {"optimal": round(opt, 4), "ga": round(ga, 4),
                               "gap_pct": round(gap, 2)}
        print(f"n={n}: optimal={opt:.4f}  GA={ga:.4f}  gap={gap:.2f}%")
    return results

if __name__ == "__main__":
    main()
