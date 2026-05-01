# -*- coding: utf-8 -*-
"""
Generate scaling law charts for the paper:
1. N-scaling: Commercial CVRMSE vs n (log scale), compared to BB's power-law
2. n × steps heatmap
3. M vs L vs BB comparison
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(r"C:\Users\User\Desktop\myjob\8.simulation\Korean_BB\docs")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# M model best at each n (best steps per n)
# ============================================================
M_best = {
    10: (140, 14.18, 8000),
    20: (280, 13.04, 18000),
    30: (420, 13.00, 18000),
    40: (560, 12.99, 18000),
    50: (700, 12.87, 16000),
    60: (840, 12.98, 18000),
    70: (980, 12.90, 18000),
    80: (1120, 13.05, 18000),
    90: (1260, 13.08, 18000),
    100: (1400, 12.98, 16000),
    250: (3500, 13.54, 8000),
    500: (7000, 13.46, 9800),
    1000: (14000, 12.98, 15500),
}

# ============================================================
# L model best at each n
# ============================================================
L_best = {
    250: (3500, 13.46, 18000),
    500: (7000, 13.25, 18000),
    1000: (14000, 13.34, 18000),
    2000: (28000, 13.28, 18000),
    4000: (56000, 13.11, 18000),
    6000: (84000, 13.34, 27000),
    8000: (112000, 13.84, 36000),
    10000: (140000, 13.98, 45000),
}

# BB SOTA
BB_M = 13.28  # Transformer-M
BB_L = 13.31  # Transformer-L
BB_persistence = 16.68
BB_DATA_SIZE = 900000


# ============================================================
# Figure 1: N-scaling (log-log, M model + L model + BB)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

m_buildings = np.array([v[0] for v in M_best.values()])
m_cvrmse = np.array([v[1] for v in M_best.values()])
l_buildings = np.array([v[0] for v in L_best.values()])
l_cvrmse = np.array([v[1] for v in L_best.values()])

ax.plot(m_buildings, m_cvrmse, 'o-', color="#1f77b4", label="Ours M (3+3 layers)", markersize=8)
ax.plot(l_buildings, l_cvrmse, 's-', color="#ff7f0e", label="Ours L (12+12 layers)", markersize=8)

# BB reference horizontal lines
ax.axhline(BB_M, color="#2ca02c", linestyle="--", alpha=0.7, linewidth=2,
           label=f"BB Transformer-M SOTA ({BB_M}%)")
ax.axhline(BB_L, color="#d62728", linestyle="--", alpha=0.5, linewidth=2,
           label=f"BB Transformer-L SOTA ({BB_L}%)")
ax.axhline(BB_persistence, color="gray", linestyle=":", alpha=0.5,
           label=f"BB Persistence Ensemble ({BB_persistence}%)")

# BB 900K point
ax.plot([BB_DATA_SIZE], [BB_M], marker="*", color="#2ca02c", markersize=20,
        label=f"BB uses 900K buildings", zorder=5)

# Annotate key points
ax.annotate(f"n=50: {M_best[50][1]}%\n(700 bldgs)",
            xy=(700, 12.87), xytext=(300, 12.4),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10, fontweight="bold")
ax.annotate(f"n=4k: {L_best[4000][1]}%\n(56K bldgs)",
            xy=(56000, 13.11), xytext=(200000, 13.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10)

ax.set_xscale("log")
ax.set_xlabel("Number of training buildings (log scale)", fontsize=12)
ax.set_ylabel("Commercial CVRMSE (%)", fontsize=12)
ax.set_title("Data Efficiency: Ours vs BuildingsBench SOTA\n"
             "Small + smart beats big + simple", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_ylim(12.0, 17.5)

plt.tight_layout()
plt.savefig(OUT / "fig1_n_scaling.png", dpi=150)
plt.savefig(OUT / "fig1_n_scaling.pdf")
print(f"Saved: {OUT / 'fig1_n_scaling.png'}")


# ============================================================
# Figure 2: M model n × steps heatmap
# ============================================================
M_full = {
    # n: {steps: cvrmse}
    10: {2000: 15.74, 4000: 14.34, 8000: 14.18},
    20: {18000: 13.04},
    30: {18000: 13.00},
    40: {18000: 12.99},
    50: {4000: 14.31, 8000: 13.56, 13000: 13.00, 16000: 12.87,
         18000: 12.87, 20000: 12.88, 25000: 12.92, 30000: 12.94, 40000: 12.96},
    60: {18000: 12.98},
    70: {18000: 12.90},
    80: {18000: 13.05},
    90: {18000: 13.08},
    100: {8000: 13.57, 10000: 13.41, 13000: 13.16, 16000: 12.98,
          20000: 13.07, 25000: 13.09},
    250: {6500: 13.78, 6800: 13.78, 7100: 13.61, 7400: 13.67, 7700: 13.55,
          8000: 13.54, 8300: 13.58, 8600: 13.55, 8900: 13.57, 9200: 13.56,
          9500: 13.50, 9800: 13.50},
    500: {6500: 13.74, 6800: 13.70, 7100: 13.68, 7400: 13.64, 7700: 13.56,
          8000: 13.51, 8300: 13.60, 8600: 13.51, 8900: 13.53, 9200: 13.49,
          9500: 13.50, 9800: 13.46},
    1000: {6500: 13.72, 9200: 13.42, 13000: 13.25, 15500: 12.98, 16000: 13.00,
           17000: 13.02, 18000: 12.87, 19000: 13.08, 20000: 12.99, 25000: 13.17,
           30000: 13.19, 35000: 13.16},
    2000: {6500: 13.73, 9200: 13.49, 18000: None},
    4000: {8300: 13.61, 8600: 13.58, 8900: 13.60},
}

# Extract all n and steps for heatmap
ns = sorted(M_full.keys())
all_steps = sorted({s for n in ns for s in M_full[n].keys()})

Z = np.full((len(ns), len(all_steps)), np.nan)
for i, n in enumerate(ns):
    for j, s in enumerate(all_steps):
        if s in M_full[n] and M_full[n][s] is not None:
            Z[i, j] = M_full[n][s]

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(Z, aspect="auto", cmap="RdYlGn_r", vmin=12.8, vmax=14.5)
ax.set_xticks(range(len(all_steps)))
ax.set_xticklabels([f"{s//1000}K" if s >= 1000 else str(s) for s in all_steps], rotation=45)
ax.set_yticks(range(len(ns)))
ax.set_yticklabels([f"n={n}" for n in ns])
ax.set_xlabel("Training steps", fontsize=12)
ax.set_ylabel("Buildings per archetype (n)", fontsize=12)
ax.set_title("M Model: n × steps Landscape (Commercial CVRMSE %)\n"
             "Dark green = better, red = worse", fontsize=13, fontweight="bold")

# Annotate cells
for i in range(len(ns)):
    for j in range(len(all_steps)):
        v = Z[i, j]
        if not np.isnan(v):
            color = "white" if v > 13.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

plt.colorbar(im, ax=ax, label="CVRMSE (%)")
plt.tight_layout()
plt.savefig(OUT / "fig2_nxsteps_heatmap.png", dpi=150)
plt.savefig(OUT / "fig2_nxsteps_heatmap.pdf")
print(f"Saved: {OUT / 'fig2_nxsteps_heatmap.png'}")


# ============================================================
# Figure 3: M vs L optimal n (differences in optimal data size)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(m_buildings, m_cvrmse, 'o-', color="#1f77b4",
        label="M model (15M params)", markersize=10, linewidth=2)
ax.plot(l_buildings, l_cvrmse, 's-', color="#ff7f0e",
        label="L model (160M params)", markersize=10, linewidth=2)

ax.axhline(BB_M, color="#2ca02c", linestyle="--", linewidth=2, alpha=0.7,
           label=f"BB SOTA-M ({BB_M}%)")
ax.axhline(BB_L, color="#d62728", linestyle="--", linewidth=2, alpha=0.5,
           label=f"BB SOTA-L ({BB_L}%)")

# Highlight best points
best_m_n = min(M_best, key=lambda k: M_best[k][1])
ax.scatter([M_best[best_m_n][0]], [M_best[best_m_n][1]], s=300,
           facecolor="none", edgecolor="red", linewidth=2, zorder=5)
ax.annotate(f"M best: n={best_m_n}\n{M_best[best_m_n][1]}%",
            xy=(M_best[best_m_n][0], M_best[best_m_n][1]),
            xytext=(200, 12.3), fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"))

best_l_n = min(L_best, key=lambda k: L_best[k][1])
ax.scatter([L_best[best_l_n][0]], [L_best[best_l_n][1]], s=300,
           facecolor="none", edgecolor="red", linewidth=2, zorder=5)
ax.annotate(f"L best: n={best_l_n}\n{L_best[best_l_n][1]}%",
            xy=(L_best[best_l_n][0], L_best[best_l_n][1]),
            xytext=(3000, 13.7), fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"))

ax.set_xscale("log")
ax.set_xlabel("Number of training buildings (log scale)", fontsize=12)
ax.set_ylabel("Commercial CVRMSE (%)", fontsize=12)
ax.set_title("Optimal Training Set Size Scales with Model Capacity\n"
             "Small model: fewer buildings; Large model: more buildings",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(12.5, 14.2)

plt.tight_layout()
plt.savefig(OUT / "fig3_M_vs_L.png", dpi=150)
plt.savefig(OUT / "fig3_M_vs_L.pdf")
print(f"Saved: {OUT / 'fig3_M_vs_L.png'}")


# ============================================================
# Summary
# ============================================================
print("\n=== Summary ===")
print(f"M best: n={best_m_n} ({M_best[best_m_n][0]} bldgs), CVRMSE={M_best[best_m_n][1]}%")
print(f"L best: n={best_l_n} ({L_best[best_l_n][0]} bldgs), CVRMSE={L_best[best_l_n][1]}%")
print(f"BB SOTA-M: {BB_M}% (uses {BB_DATA_SIZE:,} buildings)")
print(f"Data efficiency: {BB_DATA_SIZE // M_best[best_m_n][0]}× fewer buildings for M model")
