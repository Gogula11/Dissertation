import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.instance_generator import GLOBAL_COLOUR_HEX, GLOBAL_COLOUR_NAMES, GLOBAL_COLOUR_CATEGORIES, DYE_CATEGORIES

def plot_gantt(sigma, instance, title="Schedule", ax=None, alpha_eval=None, add_legend=True, machine=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    proc     = instance["proc_times"]
    setup_t  = instance["setup_time"]
    colours  = instance["colour_ids"]
    categories = instance.get("dye_category", None)
    m        = instance["m"]

    seqs = [sigma[machine]] if machine is not None else sigma
    y_offset = 0 if machine is None else machine
    for k, seq in enumerate(seqs):
        t = 0.0
        for idx, job in enumerate(seq):
            if idx > 0:
                st = float(setup_t[seq[idx-1]][job])
                ax.barh(k + y_offset, st, left=t, height=0.35,
                        color="lightgrey", edgecolor="black", hatch="//", linewidth=0.5)
                t += st
            colour = GLOBAL_COLOUR_HEX.get(int(colours[job]), "#cccccc")
            ax.barh(k + y_offset, float(proc[job]), left=t, height=0.6,
                    color=colour, edgecolor="black", linewidth=0.5)
            ax.text(t + proc[job]/2, k + y_offset, str(job),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            t += float(proc[job])

    if machine is None:
        ax.set_yticks(range(m))
        ax.set_yticklabels([f"Machine {k}" for k in range(m)])
    else:
        ax.set_yticks([machine])
        ax.set_yticklabels([f"Machine {machine}"])
    ax.set_xlabel("Time (hours)")
    ax.set_title(title)

    ax.axvline(x=168, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='1 week (168h)')

    if add_legend:
        patches = []
        for cat_id in sorted(DYE_CATEGORIES.keys()):
            cat_name = DYE_CATEGORIES[cat_id]["name"]
            cat_colours = DYE_CATEGORIES[cat_id]["colours"]
            cid_start = sum(len(DYE_CATEGORIES[c]["colours"]) for c in range(cat_id))
            for offset, cname in enumerate(cat_colours):
                cid = cid_start + offset
                if cid in GLOBAL_COLOUR_HEX:
                    patches.append(mpatches.Patch(color=GLOBAL_COLOUR_HEX[cid], label=f"{cat_name}: {cname}"))
        patches.append(mpatches.Patch(facecolor="lightgrey", edgecolor="black", hatch="//", label="Setup time"))
        patches.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='1 week (168h)'))
        fig = ax.get_figure()
        fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=9, ncol=5,
                   frameon=True, fancybox=True, shadow=True)
    return ax