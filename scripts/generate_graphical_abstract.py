"""
Graphical Abstract for Applied Energy submission.
Korean_BB: Operational Diversity Over Scale
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 0,
    'figure.facecolor': 'white',
})

fig = plt.figure(figsize=(16, 9), dpi=300, facecolor='white')

# ══════════════════════════════════════════════════════════
# Layout: 3 rows
#   Row 1 (top):     Title banner
#   Row 2 (middle):  Pipeline flow (left to right)
#   Row 3 (bottom):  Key result comparison chart
# ══════════════════════════════════════════════════════════

# ── Colors ──
C_BLUE    = '#2196F3'
C_BLUE_D  = '#1565C0'
C_GREEN   = '#4CAF50'
C_GREEN_D = '#2E7D32'
C_ORANGE  = '#FF9800'
C_RED     = '#E53935'
C_PURPLE  = '#7E57C2'
C_GRAY    = '#9E9E9E'
C_BG_LIGHT = '#F5F7FA'
C_SOTA_RED = '#E53935'

# ── Row 1: Title ──
ax_title = fig.add_axes([0.02, 0.88, 0.96, 0.10])
ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
ax_title.axis('off')

ax_title.text(0.5, 0.65, 'Operational Diversity Over Scale',
              fontsize=22, fontweight='bold', ha='center', va='center',
              color='#1a1a2e')
ax_title.text(0.5, 0.18,
              '700 Parametric Simulations Outperform 900,000 Buildings in Zero-Shot Load Forecasting',
              fontsize=13, ha='center', va='center', color='#555555', style='italic')

# ── Row 2: Pipeline ──
ax_pipe = fig.add_axes([0.03, 0.44, 0.94, 0.42])
ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 3)
ax_pipe.axis('off')

# Pipeline boxes
boxes = [
    (0.2,  '14 Building\nArchetypes',        '#E3F2FD', C_BLUE_D),
    (2.0,  '12D LHS\nSampling',              '#E8F5E9', C_GREEN_D),
    (3.8,  'EnergyPlus\nSimulation',          '#FFF3E0', '#E65100'),
    (5.6,  'Box-Cox +\nRevIN',               '#F3E5F5', '#6A1B9A'),
    (7.4,  'Transformer\nTraining',           '#FCE4EC', C_RED),
    (9.0,  'Zero-Shot\nForecast',             '#E0F7FA', '#00695C'),
]

box_w, box_h = 1.5, 1.8
for x, label, bg, border in boxes:
    rect = FancyBboxPatch((x, 0.6), box_w, box_h,
                          boxstyle="round,pad=0.12",
                          facecolor=bg, edgecolor=border, linewidth=2)
    ax_pipe.add_patch(rect)
    ax_pipe.text(x + box_w/2, 1.5, label,
                 fontsize=11, fontweight='bold', ha='center', va='center',
                 color=border)

# Sub-labels (details under each box)
sub_labels = [
    (0.2,  'office, hotel, hospital,\nschool, retail, ...'),
    (2.0,  'op_start, baseload,\nscale_mult, ...'),
    (3.8,  '700 buildings\n(50 per archetype)'),
    (5.6,  'per-instance\nnormalization'),
    (7.4,  '18,000 steps\nsub-epoch'),
    (9.0,  '168h → 24h\nno lat/lon'),
]
for x, sub in sub_labels:
    ax_pipe.text(x + box_w/2, 0.85, sub,
                 fontsize=8.5, ha='center', va='center',
                 color='#666666', style='italic')

# Arrows between boxes
arrow_style = dict(arrowstyle='->', color='#888888', lw=2,
                   connectionstyle='arc3,rad=0')
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + box_w
    x2 = boxes[i+1][0]
    ax_pipe.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                     arrowprops=arrow_style)

# ── Row 3: Result comparison (split into left chart + right key findings) ──

# Left: bar chart
ax_bar = fig.add_axes([0.06, 0.04, 0.42, 0.38])

models = ['Korean-700\n(ours)', 'BB SOTA-M\n(900K)', 'BB 900K\n+RevIN',
          'Korean-700\nRevIN OFF', 'Persistence\nEnsemble']
values = [13.11, 13.27, 13.89, 14.72, 16.68]
colors = [C_BLUE, C_SOTA_RED, C_ORANGE, '#90CAF9', C_GRAY]
errors = [0.17, 0, 0, 0.28, 0]

bars = ax_bar.bar(range(len(models)), values, color=colors, width=0.65,
                  edgecolor='white', linewidth=1.5, zorder=3)

# Error bars for multi-seed results
for i, (v, e) in enumerate(zip(values, errors)):
    if e > 0:
        ax_bar.errorbar(i, v, yerr=e, fmt='none', ecolor='black',
                        capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)

# SOTA line
ax_bar.axhline(y=13.27, color=C_SOTA_RED, linestyle='--', linewidth=1.5,
               alpha=0.7, zorder=2)

# Value labels
for i, (v, e) in enumerate(zip(values, errors)):
    label = f'{v}%'
    if e > 0:
        label = f'{v}±{e}%'
    y_pos = v + e + 0.15 if e > 0 else v + 0.15
    ax_bar.text(i, y_pos, label, ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=colors[i])

ax_bar.set_xticks(range(len(models)))
ax_bar.set_xticklabels(models, fontsize=9)
ax_bar.set_ylabel('CVRMSE (%)', fontsize=11, fontweight='bold')
ax_bar.set_ylim(12.0, 17.5)
ax_bar.set_title('Zero-Shot Commercial Load Forecasting', fontsize=12,
                 fontweight='bold', pad=8)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_linewidth(1)
ax_bar.spines['bottom'].set_linewidth(1)
ax_bar.yaxis.set_tick_params(width=1)
ax_bar.xaxis.set_tick_params(width=1)
ax_bar.set_axisbelow(True)
ax_bar.grid(axis='y', alpha=0.3, linewidth=0.5)

# Right: Key findings panel
ax_find = fig.add_axes([0.55, 0.04, 0.42, 0.38])
ax_find.set_xlim(0, 10); ax_find.set_ylim(0, 10)
ax_find.axis('off')

# Background
rect_bg = FancyBboxPatch((0.2, 0.3), 9.6, 9.4,
                         boxstyle="round,pad=0.3",
                         facecolor='#F8F9FA', edgecolor='#DEE2E6',
                         linewidth=1.5)
ax_find.add_patch(rect_bg)

ax_find.text(5, 9.2, 'Key Findings', fontsize=13, fontweight='bold',
             ha='center', va='center', color='#1a1a2e')

findings = [
    ('1', '0.08% data beats SOTA',
     '700 buildings > 900,000 buildings', C_BLUE_D),
    ('2', 'Data design > normalization',
     'Design effect (2.2pp) > RevIN effect (1.61pp)', C_GREEN_D),
    ('3', 'RevIN: scale-dependent',
     'Helps small data (−1.61pp), hurts large (+0.62pp)', C_PURPLE),
    ('4', 'Rapid saturation at n=5',
     '70 buildings already match SOTA (13.28%±0.12%)', C_ORANGE),
    ('5', 'Cross-climate zero-shot',
     'Korean climate → US buildings, no lat/lon needed', '#00695C'),
]

for i, (num, title, desc, color) in enumerate(findings):
    y = 7.8 - i * 1.6
    # Number circle
    circle = plt.Circle((1.0, y), 0.38, facecolor=color, edgecolor='white',
                         linewidth=2, zorder=5)
    ax_find.add_patch(circle)
    ax_find.text(1.0, y, num, fontsize=11, fontweight='bold',
                 ha='center', va='center', color='white', zorder=6)
    # Title + description
    ax_find.text(1.8, y + 0.22, title, fontsize=10.5, fontweight='bold',
                 va='center', color=color)
    ax_find.text(1.8, y - 0.28, desc, fontsize=8.5,
                 va='center', color='#555555')

# ── Save ──
out_path = 'docs/graphical_abstract'
fig.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig(f'{out_path}.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved: {out_path}.png + .pdf")
print(f"  PNG size: {16*300}x{9*300} = 4800x2700 px")
