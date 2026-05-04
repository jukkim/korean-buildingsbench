"""Generate Fig. 3: N-Scaling curve for Applied Energy paper."""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
n_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80])
buildings = n_vals * 14
nrmse = np.array([
    14.72, 14.07, 13.47, 13.45, 13.24,
    13.25, 13.25, 13.18, 13.18, 13.18,
    13.23, 13.11, 13.22, 12.93, 13.20, 13.20, 13.15
])

# Evenly-spaced x positions (categorical-like) to prevent crowding
x_pos = np.arange(len(buildings))

# BB-900K baseline reference
sota_m = 13.27

# Plateau band: min/max of n=5..80
plateau_mask = n_vals >= 5
plateau_min = nrmse[plateau_mask].min()
plateau_max = nrmse[plateau_mask].max()

# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Plateau band (light blue shading, spanning full x range)
ax.axhspan(plateau_min, plateau_max, color='#B3D9FF', alpha=0.35, zorder=0,
           label='Plateau band (n=5-80)')

# BB-900K baseline horizontal line
ax.axhline(y=sota_m, color='red', linestyle='--', linewidth=2.0, zorder=2,
           label=f'BB-900K baseline ({sota_m}%)')

# Main data line
ax.plot(x_pos, nrmse, '-o', color='#1f77b4', linewidth=2.0,
        markersize=8, markerfacecolor='#1f77b4', markeredgecolor='#1f77b4',
        zorder=3)

# ── Annotations ───────────────────────────────────────────────────────
# Helper: map building count to x_pos
bldg_to_x = dict(zip(buildings, x_pos))

# n=1: 14 bldgs
ax.annotate('n=1\n14 bldgs',
            xy=(bldg_to_x[14], 14.72),
            xytext=(bldg_to_x[14] + 0.8, 15.0),
            fontsize=11, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            color='#333333')

# n=5: 13.24% with 5-seed info (UPDATED annotation)
ax.annotate('n=5: 13.24% (seed 42)\n5-seed mean: 13.28 $\\pm$ 0.12%',
            xy=(bldg_to_x[70], 13.24),
            xytext=(bldg_to_x[70] + 1.5, 13.58),
            fontsize=10.5, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            color='#333333')

# n=50: 12.93%
ax.annotate('n=50: 12.93%',
            xy=(bldg_to_x[700], 12.93),
            xytext=(bldg_to_x[700] + 0.6, 12.60),
            fontsize=11, ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            color='#333333')

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xlabel('Number of Training Buildings', fontsize=13, labelpad=8)
ax.set_ylabel('NRMSE (%)', fontsize=13, labelpad=8)
ax.set_title('N-Scaling: Performance vs. Training Set Size', fontsize=15, pad=12)

# X-axis: evenly spaced with building count labels
ax.set_xticks(x_pos)
ax.set_xticklabels(buildings, fontsize=10)

# Y-axis range and formatting
ax.set_ylim(12.4, 15.15)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax.tick_params(axis='y', labelsize=11)

# Grid
ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
ax.grid(True, axis='x', alpha=0.15, linewidth=0.5)

# Slight margin on x
ax.set_xlim(-0.5, len(buildings) - 0.5)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='#cccccc')

# Tight layout
fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────
out_base = r'C:\Users\User\Desktop\myjob\8.simulation\Korean_BB\docs\fig3_nscaling_new'
fig.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out_base}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
plt.close(fig)
