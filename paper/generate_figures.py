import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

# ===== Figure 1: Pipeline Diagram =====
fig, ax = plt.subplots(1, 1, figsize=(7, 2.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis('off')

boxes = [
    (0.3, 0.7, '14 DOE-Derived\nArchetypes'),
    (1.8, 0.7, '12D LHS\nSampling'),
    (3.3, 0.7, 'EnergyPlus\n(700 sims)'),
    (4.8, 0.7, 'RevIN +\nAugmentation'),
    (6.3, 0.7, 'Transformer-M\n(15.8M params)'),
    (8.0, 0.7, 'Zero-Shot\nInference'),
]

colors = ['#E8F4FD', '#D4EDDA', '#FFF3CD', '#F8D7DA', '#E2D5F1', '#D1ECF1']

for i, (x, y, label) in enumerate(boxes):
    box = FancyBboxPatch((x, y), 1.2, 0.9, boxstyle='round,pad=0.05',
                         facecolor=colors[i], edgecolor='#333333', linewidth=0.8)
    ax.add_patch(box)
    ax.text(x + 0.6, y + 0.45, label, ha='center', va='center', fontsize=7.5, fontweight='bold')

for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 1.2
    x2 = boxes[i+1][0]
    y_mid = 1.15
    ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

sublabels = [
    'Seoul, Busan,\nDaegu, Gangneung, Jeju',
    'n=50/archetype\n(Table 1)',
    '8,760 h/building\nhourly kWh',
    'Window jitter\nNoise, Scaling',
    'Gaussian NLL\n18K steps',
    '955 real\nbuildings'
]
for i, (x, y, _) in enumerate(boxes):
    ax.text(x + 0.6, y - 0.15, sublabels[i], ha='center', va='top', fontsize=5.5, color='#555')

ax.set_title('Fig. 1. End-to-end pipeline from simulation design to zero-shot inference.',
             fontsize=9, loc='left', pad=5)
plt.tight_layout()
plt.savefig('figures/fig1_pipeline.pdf', bbox_inches='tight')
plt.savefig('figures/fig1_pipeline.png', bbox_inches='tight', dpi=300)
plt.close()
print('Fig 1 done')


# ===== Figure 2: Main Results Bar Chart =====
fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

models = ['Persist.', 'BB-700', 'BB-700\n(aug)', 'BB+RevIN\n(900K)', 'BB-900K', 'US-700\n(ours)', 'K-700\n(ours)']
nrmse = [16.68, 15.28, 14.26, 13.89, 13.27, 13.64, 13.11]
errors = [0, 0, 0, 0, 0, 0.65, 0.17]
colors2 = ['#adb5bd', '#6c757d', '#6c757d', '#495057', '#343a40', '#2196F3', '#1565C0']

bars = ax.bar(range(len(models)), nrmse, color=colors2, edgecolor='white', linewidth=0.5, width=0.7)
ax.errorbar(range(len(models)), nrmse, yerr=errors, fmt='none', ecolor='black', capsize=3, linewidth=0.8)

ax.axhline(y=13.27, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='BB-900K reference (13.27%)')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=7.5)
ax.set_ylabel('NRMSE (%)')
ax.set_ylim(12, 17.5)
ax.legend(loc='upper left', fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, (bar, v) in enumerate(zip(bars, nrmse)):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.15 + errors[i], f'{v:.2f}',
            ha='center', va='bottom', fontsize=6.5)

ax.set_title('Fig. 2. Zero-shot NRMSE on 955 real commercial buildings.', fontsize=9, loc='left')
plt.tight_layout()
plt.savefig('figures/fig2_main_results.pdf', bbox_inches='tight')
plt.savefig('figures/fig2_main_results.png', bbox_inches='tight', dpi=300)
plt.close()
print('Fig 2 done')


# ===== Figure 3: N-Scaling Curve =====
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

n_vals = [1, 3, 5, 10, 20, 50, 70, 80]
nrmse_n = [14.72, 13.47, 13.28, 13.18, 13.23, 12.93, 13.20, 13.15]

ax.plot(n_vals, nrmse_n, 'o-', color='#1565C0', linewidth=1.5, markersize=5, label='K-LHS (this work)')
ax.axhline(y=13.27, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='BB-900K (13.27%)')
ax.axhspan(12.93 - 0.17, 13.27, alpha=0.08, color='blue', label='K-700 5-seed range')

ax.annotate('Plateau region\n(n >= 5)', xy=(20, 13.22), fontsize=7, color='#555',
            ha='center', va='bottom')

ax.set_xlabel('Samples per archetype (n)')
ax.set_ylabel('NRMSE (%)')
ax.set_ylim(12.5, 15.2)
ax.set_xscale('log')
ax.set_xticks(n_vals)
ax.set_xticklabels([str(n) for n in n_vals])
ax.legend(loc='upper right', fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Fig. 3. N-scaling: NRMSE vs. samples per archetype.', fontsize=9, loc='left')
plt.tight_layout()
plt.savefig('figures/fig3_nscaling.pdf', bbox_inches='tight')
plt.savefig('figures/fig3_nscaling.png', bbox_inches='tight', dpi=300)
plt.close()
print('Fig 3 done')


# ===== Figure 4: RevIN Regime-Dependent Effect =====
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

categories = ['K-700\n(LHS, 700)', 'BB-700\n(stock, 700)', 'BB-900K\n(stock, 900K)']
revin_on = [13.11, 15.28, 13.89]
revin_off = [14.72, 15.28, 13.27]

x = np.arange(len(categories))
width = 0.32

bars1 = ax.bar(x - width/2, revin_on, width, label='RevIN ON', color='#1565C0', edgecolor='white')
bars2 = ax.bar(x + width/2, revin_off, width, label='RevIN OFF', color='#FF8F00', edgecolor='white')

deltas = [revin_on[i] - revin_off[i] for i in range(len(categories))]
for i, d in enumerate(deltas):
    sign = '+' if d > 0 else ''
    color = '#2E7D32' if d < 0 else '#C62828'
    y_pos = max(revin_on[i], revin_off[i]) + 0.15
    ax.text(x[i], y_pos, f'Delta = {sign}{d:.2f}', ha='center', va='bottom', fontsize=7,
            color=color, fontweight='bold')

ax.set_ylabel('NRMSE (%)')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylim(12, 16.5)
ax.legend(loc='upper right', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Fig. 4. RevIN effect: helps LHS-designed, hurts stock-model.', fontsize=9, loc='left')
plt.tight_layout()
plt.savefig('figures/fig4_revin_effect.pdf', bbox_inches='tight')
plt.savefig('figures/fig4_revin_effect.png', bbox_inches='tight', dpi=300)
plt.close()
print('Fig 4 done')

print('\nAll figures saved to figures/')
