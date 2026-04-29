"""
plot_style.py – Shared Matplotlib Style
========================================

Centralises all Matplotlib style settings so that metric_explorer.py and
testfisico.py always produce visually consistent figures without duplicating
the configuration block.

Usage
-----
    from plot_style import setup_plot_style
    setup_plot_style()

Call this once at module import time (after `import matplotlib.pyplot as plt`).
"""

import matplotlib.pyplot as plt


def setup_plot_style():
    """
    Apply the project-wide Matplotlib style:

    - Base style: 'seaborn-v0_8-whitegrid' (falls back to defaults if
      the style name is unavailable in older Matplotlib versions).
    - Large, readable fonts with thick axis lines for print/PDF quality.
    - High-DPI defaults for both screen rendering and saved files.
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass  # fallback to Matplotlib defaults

    plt.rcParams.update({
        'font.size':          13,
        'axes.titlesize':     14,
        'axes.labelsize':     13,
        'axes.linewidth':     1.6,
        'lines.linewidth':    2.2,
        'xtick.major.width':  1.4,
        'ytick.major.width':  1.4,
        'legend.fontsize':    11,
        'figure.dpi':         120,
        'savefig.dpi':        200,
        'savefig.bbox':       'tight',
    })
