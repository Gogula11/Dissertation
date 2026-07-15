import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use("Agg")

from src.instance_generator import generate_instance
from src.heuristics import spt
from src.visualisation import plot_gantt


def test_plot_gantt_returns_ax():
    inst = generate_instance(n=5, m=2, seed=0)
    sigma = spt(inst)
    ax = plot_gantt(sigma, inst, title="Test")
    assert ax is not None
    assert len(ax.patches) > 0
