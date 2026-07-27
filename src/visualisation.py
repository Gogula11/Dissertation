import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.instance_generator import COLOUR_HEX

def plot_gantt(sigma, instance, title="Schedule", ax=None, alpha_eval=None, add_legend=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

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
            colour = COLOUR_HEX.get(int(colours[job]), "#cccccc")
            ax.barh(k, float(proc[job]), left=t, height=0.6,
                    color=colour, edgecolor="black", linewidth=0.5)
            ax.text(t + proc[job]/2, k, str(job),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            t += float(proc[job])

    ax.set_yticks(range(m))
    ax.set_yticklabels([f"Machine {k}" for k in range(m)])
    ax.set_xlabel("Time (hours)")
    ax.set_title(title)
    
    # Highlight 168-hour (1 week) mark
    ax.axvline(x=168, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='1 week (168h)')
    
    if add_legend:
        patches = [mpatches.Patch(color=c, label=f"Colour {i}") for i, c in COLOUR_HEX.items()]
        patches.append(mpatches.Patch(facecolor="lightgrey", edgecolor="black", hatch="//", label="Setup time"))
        patches.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='1 week (168h)'))
        fig = ax.get_figure()
        fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=11, ncol=4,
                   frameon=True, fancybox=True, shadow=True)
    return ax
