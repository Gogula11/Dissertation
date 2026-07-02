import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLOUR_MAP = {
    0: "#FFFACD", 1: "#FFD700", 2: "#87CEEB",
    3: "#228B22", 4: "#DC143C", 5: "#000080", 6: "#1C1C1C",
}

def plot_gantt(sigma, instance, title="Schedule", ax=None, alpha_eval=None):
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
