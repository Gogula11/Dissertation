"""
Generate the problem + system flow chart for the dissertation.
Outputs: figures/flow_chart_problem_system.svg (vector)
         figures/flow_chart_problem_system.png (raster)
Call from project root: python scripts/generate_flow_chart.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis("off")

# ─── helpers ───
def box(x, y, w, h, lines, color="#E8F0FE", ec="black", lw=2.5, fs=13, ha="center"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3",
        facecolor=color, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(rect)
    line_height = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        yy = y + h - line_height * (i + 0.5)
        weight = "bold" if i == 0 else "normal"
        size = fs + 3 if i == 0 else fs
        ax.text(x + w / 2, yy, line, ha=ha, va="center", fontsize=size, fontweight=weight)

def arrow(x1, y1, x2, y2, lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color="#333333"))

# ─── TIER 1: Textile Dyeing Domain ───
box(1.5, 13.0, 19, 2.0,
    ["TEXTILE DYEING DOMAIN",
     "Grey fabric → colour selection → dye chemistry (direct / reactive / vat)",
     "Dark→light batch transition = expensive cleaning    Light→dark = minimal cleaning"],
    color="#FFF3E0", ec="#E65100", fs=14)

arrow(11, 13.0, 11, 11.5)

# ─── TIER 2: PMSP-SDSC Problem Formalisation ───
box(1.5, 8.2, 19, 3.3,
    ["PMSP-SDSC PROBLEM FORMALISATION",
     "n jobs × m identical parallel machines",
     "Asymmetric setup cost c_ij  (colour darkness diff + chemistry penalty + gamma noise)",
     "Objective:  F = α · f1/f1_scale  +  (1−α) · f2/f2_scale",
     "f1 = weighted tardiness          f2 = total setup cost",
     "Normalisation: heuristic sampling (R=1000 random schedules)"],
    color="#E3F2FD", ec="#1565C0", fs=13)

# ─── branching arrows ───
arrow(11, 8.2, 11, 7.2)
ax.plot([4.5, 11, 17.5], [7.2, 7.2, 7.2], color="#333333", lw=2.5)
for xp in [4.5, 11, 17.5]:
    arrow(xp, 7.2, xp, 6.2)

# ─── TIER 3a: Classical Heuristics ───
box(0.5, 2.5, 7, 3.7,
    ["CLASSICAL HEURISTICS",
     "SPT (Shortest Processing Time)",
     "  → sort by proc_time, round-robin",
     "NN-Greedy (Nearest-Neighbour)",
     "  → minimise local setup cost at each step",
     "Baseline: fast, no learning"],
    color="#F3E5F5", ec="#7B1FA2", fs=13)

# ─── TIER 3b: Genetic Algorithm ───
box(8.0, 2.5, 6, 3.7,
    ["GENETIC ALGORITHM  (DEAP)",
     "Population = 100     Generations = 300",
     "Encoding: permutation (giant-tour)",
     "Crossover: Order Crossover (OX)",
     "Mutation: swap / inversion / insertion",
     "Fixed operator rates per generation"],
    color="#E8F5E9", ec="#2E7D32", fs=13)

# ─── TIER 3c: GA + PPO Hybrid ───
box(14.5, 2.5, 7, 3.7,
    ["GA + PPO HYPER-HEURISTIC",
     "Env wraps GA: obs(8D) → action(3)",
     "PPO policy selects mutation operator",
     "  based on convergence state",
     "Training: 100k steps, 110 instances",
     "Adaptive operator selection"],
    color="#FFEBEE", ec="#C62828", fs=13)

# ─── arrows to results ───
arrow(4.5, 2.5, 4.5, 1.2)
arrow(11, 2.5, 11, 1.2)
arrow(17.5, 2.5, 17.5, 1.2)

# ─── TIER 4: Output ───
box(3, 0.3, 16, 1.0,
    ["EVALUATION  —  composite scores  /  Wilcoxon p-values  /  box plots  /  Gantt charts"],
    color="#F9FBE7", ec="#F57F17", fs=14)

# ─── legend ───
legend_items = [
    mpatches.Patch(color="#FFF3E0", label="Problem domain"),
    mpatches.Patch(color="#E3F2FD", label="Problem formalisation"),
    mpatches.Patch(color="#F3E5F5", label="Baseline methods"),
    mpatches.Patch(color="#E8F5E9", label="Optimisation method"),
    mpatches.Patch(color="#FFEBEE", label="Learning-enhanced method"),
    mpatches.Patch(color="#F9FBE7", label="Output / results"),
]
ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(11, -0.03),
          ncol=6, fontsize=11, framealpha=0.9, edgecolor="gray")

# ─── title ───
ax.text(11, 15.3, "Hybrid GA+DRL for PMSP-SDSC — Problem & System Flow",
        ha="center", va="center", fontsize=22, fontweight="bold")

plt.savefig("figures/flow_chart_problem_system.svg", bbox_inches="tight", dpi=150)
plt.savefig("figures/flow_chart_problem_system.png", bbox_inches="tight", dpi=150)
print("Saved: figures/flow_chart_problem_system.svg + .png")
