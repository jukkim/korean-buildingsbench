"""
Generate paper figures for Applied Energy submission.
fig2_comparison.png/pdf  — zero-shot performance bar chart
fig4_revin_asymmetry.png/pdf — RevIN scale-dependent effect

All values from results/RESULTS_REGISTRY.md (2026-04-29 confirmed).
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
})

C_OURS    = '#1565C0'   # Korean-700 ON (blue)
C_SOTA    = '#C62828'   # BB-900K baseline (red)
C_ORANGE  = '#E65100'   # BB 900K+RevIN
C_GRAY    = '#616161'   # Baselines/OFF
C_GREEN   = '#2E7D32'   # improvement arrow
C_RED_ARR = '#C62828'   # degradation arrow


# ─────────────────────────────────────────────────────────────
# Fig. 2  —  Zero-shot performance comparison
# ─────────────────────────────────────────────────────────────
def fig2_comparison():
    models = [
        'Korean-700\n(RevIN ON, aug)',
        'BB-900K baseline\n(900K, no RevIN)',
        'BB 900K\n(RevIN ON)',
        'BB-700+aug\n(RevIN ON)',
        'Korean-700\n(RevIN OFF)',
        'BB-700\n(RevIN ON,\nno aug)',
        'BB-700\n(RevIN OFF)',
        'Persistence\nEnsemble',
    ]
    values = [13.11, 13.27, 13.89, 14.26, 14.72, 15.28, 16.44, 16.68]
    errors = [0.17,  0,     0,     0,     0.28,  0,     0,     0    ]
    colors = [C_OURS, C_SOTA, C_ORANGE, '#EF9A9A', '#90CAF9', '#FFCC80', C_GRAY, '#BDBDBD']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')

    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, width=0.65,
                  edgecolor='white', linewidth=1.2, zorder=3)

    # Error bars
    for i, (v, e) in enumerate(zip(values, errors)):
        if e > 0:
            ax.errorbar(i, v, yerr=e, fmt='none', ecolor='black',
                        capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)

    # SOTA dashed line
    ax.axhline(y=13.27, color=C_SOTA, linestyle='--', linewidth=1.5,
               alpha=0.7, zorder=2, label='BB-900K baseline (13.27%)')

    # Value labels above bars
    for i, (v, e) in enumerate(zip(values, errors)):
        if e > 0:
            label = f'{v:.2f}±{e:.2f}%'
            y_pos = v + e + 0.15
        else:
            label = f'{v:.2f}%'
            y_pos = v + 0.15
        ax.text(i, y_pos, label, ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=colors[i])

    # Best-seed annotation for Korean-700
    ax.annotate('best seed\n12.93%', xy=(0, 12.93), xytext=(0.65, 12.45),
                fontsize=9.5, color=C_OURS, style='italic',
                arrowprops=dict(arrowstyle='->', color=C_OURS, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('NRMSE (%)', fontsize=13, fontweight='bold')
    ax.set_ylim(12.0, 18.5)
    ax.set_title('Zero-Shot Commercial Load Forecasting (955 buildings)', fontsize=14,
                 fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.8)

    fig.tight_layout()
    out = DOCS / "fig2_comparison"
    fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f'{out}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out}.png / .pdf")


# ─────────────────────────────────────────────────────────────
# Fig. 4  —  RevIN asymmetric effect across dataset scales
# ─────────────────────────────────────────────────────────────
def fig4_revin_asymmetry():
    # (label, OFF value, ON value)  — values from RESULTS_REGISTRY
    groups = [
        ('Korean-700\n(n=50)',      14.72, 13.11),   # 5-seed means
        ('BB-700\n(n=700, no aug)', 16.44, 15.28),   # seed 42
        ('BB 900K\n(full corpus)',  13.27, 13.89),   # reproduced SOTA
    ]
    labels   = [g[0] for g in groups]
    off_vals = [g[1] for g in groups]
    on_vals  = [g[2] for g in groups]
    deltas   = [on - off for off, on in zip(off_vals, on_vals)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('white')

    x = np.arange(len(groups))
    w = 0.32

    bars_off = ax.bar(x - w/2, off_vals, w, label='RevIN OFF',
                      color='#ECEFF1', edgecolor='#546E7A', linewidth=1.5, zorder=3)
    bars_on  = ax.bar(x + w/2, on_vals,  w, label='RevIN ON',
                      color='#1565C0', edgecolor='white', linewidth=1.2, zorder=3)

    # Value labels
    for bar, v in zip(bars_off, off_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.12,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=11, color='#37474F')
    for bar, v in zip(bars_on, on_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.12,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=11,
                color=C_OURS, fontweight='bold')

    # Delta arrows and annotations
    arrow_kw = dict(arrowstyle='->', lw=2.5, connectionstyle='arc3,rad=0')
    for i, (off, on, delta) in enumerate(zip(off_vals, on_vals, deltas)):
        y_top = max(off, on) + 0.75
        color = C_GREEN if delta < 0 else C_RED_ARR
        sign  = '▼' if delta < 0 else '▲'
        ax.annotate('', xy=(x[i] + w/2, on + 0.08),
                    xytext=(x[i] - w/2, off + 0.08),
                    arrowprops=dict(color=color, **arrow_kw))
        label = f'{sign} {abs(delta):.2f} pp'
        ax.text(x[i], y_top, label, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=color)

    # Reference lines
    ax.axhline(y=13.27, color=C_SOTA, linestyle='--', linewidth=1.5,
               alpha=0.6, zorder=2, label='BB-900K baseline (13.27%)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('NRMSE (%)', fontsize=13, fontweight='bold')
    ax.set_ylim(11.0, 20.5)
    ax.set_title("RevIN's Asymmetric Effect Across Dataset Scales", fontsize=14,
                 fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(facecolor='#ECEFF1', edgecolor='#546E7A', label='RevIN OFF'),
        mpatches.Patch(facecolor='#1565C0', label='RevIN ON'),
        plt.Line2D([0], [0], color=C_SOTA, linestyle='--', label='BB-900K baseline (13.27%)'),
        mpatches.Patch(facecolor=C_GREEN, label='RevIN helps (▼ NRMSE)'),
        mpatches.Patch(facecolor=C_RED_ARR, label='RevIN hurts (▲ NRMSE)'),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc='upper right', framealpha=0.85)

    # Note for Korean-700
    ax.text(0.01, 0.02, 'Korean-700: five-seed means.  BB-700: seed 42 only.',
            transform=ax.transAxes, fontsize=9.5, color='#777', style='italic')

    fig.tight_layout()
    out = DOCS / "fig4_revin_asymmetry"
    fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f'{out}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out}.png / .pdf")


if __name__ == '__main__':
    print("Generating paper figures...")
    fig2_comparison()
    fig4_revin_asymmetry()
    print("Done.")
