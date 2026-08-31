import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.instance_generator import GLOBAL_COLOUR_HEX, GLOBAL_COLOUR_NAMES, WEEKLY_HOURS, DYE_CATEGORY_NAMES, PROC_TIME_RANGE

CATEGORY_EDGE_COLOURS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

def plot_gantt(sigma, instance, title="Schedule", ax=None, alpha_eval=None, add_legend=True, machine=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    proc     = instance["proc_times"]
    setup_t  = instance["setup_time"]
    colours  = instance["colour_ids"]
    categories = instance.get("dye_category", np.zeros(len(proc), dtype=np.int32))
    m        = instance["m"]

    seqs = [sigma[machine]] if machine is not None else sigma
    y_offset = 0 if machine is None else machine
    max_t = 0.0
    for k, seq in enumerate(seqs):
        t = 0.0
        for idx, job in enumerate(seq):
            if idx > 0:
                st = float(setup_t[seq[idx-1]][job])
                ax.barh(k + y_offset, st, left=t, height=0.35,
                        color="lightgrey", edgecolor="black", hatch="//", linewidth=0.5)
                t += st
            colour = GLOBAL_COLOUR_HEX.get(int(colours[job]), "#cccccc")
            cat = int(categories[job])
            edge = CATEGORY_EDGE_COLOURS[cat] if 0 <= cat < len(CATEGORY_EDGE_COLOURS) else 'black'
            ax.barh(k + y_offset, float(proc[job]), left=t, height=0.6,
                    color=colour, edgecolor=edge, linewidth=2)
            ax.text(t + proc[job]/2, k + y_offset, str(job),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            t += float(proc[job])
        max_t = max(max_t, t)

    if machine is None:
        ax.set_yticks(range(m))
        ax.set_yticklabels([f"Machine {k}" for k in range(m)])
    else:
        ax.set_yticks([machine])
        ax.set_yticklabels([f"Machine {machine}"])
    ax.set_xlabel("Time (hours)")
    ax.set_title(title)

    ax.axvline(x=WEEKLY_HOURS, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='1 week (168h)')
    ax.set_xlim(left=-2, right=max(max_t * 1.1, WEEKLY_HOURS * 1.05))

    if add_legend:
        patches = []
        for cid in range(len(GLOBAL_COLOUR_NAMES)):
            if cid in GLOBAL_COLOUR_HEX:
                patches.append(mpatches.Patch(color=GLOBAL_COLOUR_HEX[cid], label=GLOBAL_COLOUR_NAMES[cid]))
        patches.append(mpatches.Patch(facecolor="lightgrey", edgecolor="black", hatch="//", label="Setup time"))
        patches.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='1 week (168h)'))
        for cat_id, name in enumerate(DYE_CATEGORY_NAMES):
            lo, hi = PROC_TIME_RANGE[cat_id]
            patches.append(mpatches.Patch(facecolor='none', edgecolor=CATEGORY_EDGE_COLOURS[cat_id],
                                          linewidth=2, label=f"{name} ({lo}-{hi}h)"))
        fig = ax.get_figure()
        fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.05), fontsize=18, ncol=4,
                   frameon=True, fancybox=True, shadow=True)
    return ax


def plot_gantt_legend(save_path=None):
    patches = []
    for cid in range(len(GLOBAL_COLOUR_NAMES)):
        if cid in GLOBAL_COLOUR_HEX:
            patches.append(mpatches.Patch(color=GLOBAL_COLOUR_HEX[cid], label=GLOBAL_COLOUR_NAMES[cid]))
    patches.append(mpatches.Patch(facecolor="lightgrey", edgecolor="black", hatch="//", label="Setup time"))
    patches.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='1 week (168h)'))
    for cat_id, name in enumerate(DYE_CATEGORY_NAMES):
        lo, hi = PROC_TIME_RANGE[cat_id]
        patches.append(mpatches.Patch(facecolor='none', edgecolor=CATEGORY_EDGE_COLOURS[cat_id],
                                      linewidth=2, label=f"{name} ({lo}-{hi}h)"))
    fig, ax = plt.subplots(figsize=(24, 2.5))
    ax.axis('off')
    leg = fig.legend(handles=patches, loc="center", ncol=5, fontsize=18,
                     frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', bbox_extra_artists=[leg])
        print(f'Saved: {save_path}')
    return fig