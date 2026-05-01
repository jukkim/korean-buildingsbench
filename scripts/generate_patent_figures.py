"""
Generate patent figures (도 1~5) with large, readable fonts for A4 print.
Output: docs/patent_fig1_system.png ~ patent_fig5_comparison.png
Run: python scripts/generate_patent_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
DPI = 200
FONT = "DejaVu Sans"


# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})


def save(fig, name):
    path = DOCS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}  ({path.stat().st_size//1024} KB)")


def draw_box(ax, x, y, w, h, label, sub="", facecolor="#d0e8f8", edgecolor="#333",
             fontsize=15, subfontsize=12):
    """Draw a rounded box with label and optional subtitle."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.03",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y + (h*0.12 if sub else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold", wrap=True,
            multialignment="center")
    if sub:
        ax.text(x, y - h*0.22, sub,
                ha="center", va="center", fontsize=subfontsize, color="#555",
                fontstyle="italic", multialignment="center")


def arrow(ax, x0, y0, x1, y1, color="#333"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))


# ── Fig. 1: System Block Diagram ──────────────────────────────────────────────
def fig1_system():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.set_title("Fig. 1  Building Energy Consumption Prediction System (100)",
                 fontsize=22, fontweight="bold", pad=18)

    # Row 1: boxes
    bw, bh = 3.8, 2.0
    row1_y = 8.0
    boxes_r1 = [
        (2.5,  row1_y, "Parameter\nGeneration Unit",  "(110)\n12D LHS Sampling",    "#cce5ff"),
        (7.5,  row1_y, "Simulation\nExecution Unit",  "(120)\nEnergyPlus Annual",   "#ffd6cc"),
        (12.5, row1_y, "Data\nPreprocessing Unit",    "(130)\nBox-Cox + Windowing", "#d4edda"),
    ]
    for x, y, lbl, sub, col in boxes_r1:
        draw_box(ax, x, y, bw, bh, lbl, sub, facecolor=col, fontsize=16, subfontsize=13)

    # Row 2 boxes
    row2_y = 3.8
    boxes_r2 = [
        (4.5,  row2_y, "Model\nTraining Unit",   "(140)\nRevIN + Transformer",    "#fff3cd"),
        (10.5, row2_y, "Energy\nPrediction Unit", "(150)\nZero-shot Inference",   "#e0d5f5"),
    ]
    for x, y, lbl, sub, col in boxes_r2:
        draw_box(ax, x, y, bw, bh, lbl, sub, facecolor=col, fontsize=16, subfontsize=13)

    # Row 1 arrows
    arrow(ax, 4.4,  row1_y, 5.6,  row1_y)
    arrow(ax, 9.4,  row1_y, 10.6, row1_y)
    # Row1→Row2: (130) down-left to (140), (130) down to (150)
    arrow(ax, 12.5, row1_y - bh/2, 10.5, row2_y + bh/2)
    # (140)→(150)
    arrow(ax, 6.4,  row2_y, 8.6,  row2_y)

    # Input: building archetypes
    draw_box(ax, 2.5, 10.2, 3.0, 0.9, "Building Archetypes\n(14 types)",
             facecolor="#f8f9fa", edgecolor="#666", fontsize=13)
    arrow(ax, 2.5, 9.75, 2.5, row1_y + bh/2)

    # Output: predicted energy
    draw_box(ax, 14.5, row2_y, 2.6, 0.9, "Predicted Energy\n(24h ahead)",
             facecolor="#d4edda", edgecolor="#28a745", fontsize=13)
    arrow(ax, 12.4, row2_y, 13.2, row2_y)

    # Input to (150): target building
    draw_box(ax, 10.5, 1.5, 3.4, 0.9, "Target Building History (168h)",
             facecolor="#f8f9fa", edgecolor="#666", fontsize=13)
    arrow(ax, 10.5, 1.95, 10.5, row2_y - bh/2)

    # No lat/lon note
    ax.text(8, 0.5, "No latitude/longitude required",
            ha="center", va="center", fontsize=17, color="#c00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#c00", lw=1.5))

    save(fig, "patent_fig1_system.png")


# ── Fig. 2: LHS Parameter Generation ─────────────────────────────────────────
def fig2_lhs():
    fig = plt.figure(figsize=(16, 12))

    # Left: flowchart (65% width)
    ax_flow = fig.add_axes([0.02, 0.02, 0.60, 0.90])
    ax_flow.set_xlim(0, 6)
    ax_flow.set_ylim(0, 12)
    ax_flow.axis("off")
    ax_flow.set_title("Fig. 2  Parameter Generation by 12D LHS",
                       fontsize=20, fontweight="bold", x=0.8, y=1.01)

    steps = [
        (3, 10.5, "Define Building Archetypes",    "14 types: office, retail,\nschool, hotel, hospital ...", "#cce5ff"),
        (3,  7.8, "Define 12D Parameter Space",   "op_start, baseload_pct,\nnight_equip_frac ...",          "#ffd6cc"),
        (3,  5.1, "Latin Hypercube Sampling",      "N=50 samples/archetype\nspace-filling design",           "#d4edda"),
        (3,  2.4, "Modify EnergyPlus IDF Files",  "envelope + HVAC +\nschedules per parameter set",        "#fff3cd"),
        (3,  0.1, "Output: 700 IDF Files",         "14 archetypes × 50 × 5 cities",                         "#f3e5f5"),
    ]
    bw, bh = 5.0, 1.7
    for x, y, lbl, sub, col in steps:
        draw_box(ax_flow, x, y, bw, bh, lbl, sub, facecolor=col, fontsize=16, subfontsize=13)

    for i in range(len(steps) - 1):
        y_top = steps[i][1] - bh/2
        y_bot = steps[i+1][1] + bh/2
        arrow(ax_flow, 3, y_top, 3, y_bot)

    # Right: scatter plot
    ax_sc = fig.add_axes([0.65, 0.38, 0.32, 0.45])
    np.random.seed(0)
    n = 50
    # LHS sampling simulation
    x_lhs = (np.random.permutation(n) + np.random.uniform(0, 1, n)) / n
    y_lhs = (np.random.permutation(n) + np.random.uniform(0, 1, n)) / n
    ax_sc.scatter(x_lhs * 12, y_lhs * 0.9 + 0.05, c="#1f77b4", s=60, alpha=0.8)
    ax_sc.set_xlabel("op_start (h)", fontsize=14)
    ax_sc.set_ylabel("baseload_pct", fontsize=14)
    ax_sc.set_title("2D Projection of LHS\n(50 samples)", fontsize=15, fontweight="bold")
    ax_sc.tick_params(labelsize=13)
    ax_sc.set_xlim(-0.5, 12.5)
    ax_sc.set_ylim(0, 1.0)

    # Key properties text
    ax_txt = fig.add_axes([0.65, 0.05, 0.32, 0.30])
    ax_txt.axis("off")
    props = "Key Properties:\n\n• Space-filling coverage\n• Each dimension sampled\n  evenly\n• No clustering or gaps\n• 700 buildings cover\n  12D space"
    ax_txt.text(0.1, 0.95, props, va="top", fontsize=14, linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ccc"))

    save(fig, "patent_fig2_lhs.png")


# ── Fig. 3: RevIN + Transformer Model ────────────────────────────────────────
def fig3_revin():
    fig, axes = plt.subplots(2, 1, figsize=(18, 13))
    fig.suptitle("Fig. 3  Model Training Process with Reversible Instance Normalization",
                 fontsize=21, fontweight="bold", y=0.99)

    labels = [
        ["Training Phase", "#2196F3"],
        ["Inference Phase (Zero-Shot)", "#E91E63"],
    ]

    # Training row boxes
    train_boxes = [
        ("Simulation Data\n700 buildings\n8760h each",         "#cce5ff"),
        ("Sliding Window\n168h context\n+ 24h target",        "#ffd6cc"),
        ("RevIN Normalize\nx_norm =\n(x−μ)/σ",               "#d4edda"),
        ("Transformer\nEncoder-Decoder\n15.8M params",        "#fff3cd"),
        ("Gaussian\nNLL Loss",                                  "#f3e5f5"),
    ]
    # Inference row boxes
    infer_boxes = [
        ("Target Building\nHistory (168h)\nNO location",       "#d4edda"),
        ("Compute\nμ_ctx, σ_ctx\nfrom 168h",                  "#cce5ff"),
        ("RevIN Normalize\n(x−μ_ctx)/σ_ctx",                  "#d4edda"),
        ("Trained\nTransformer\n(frozen)",                     "#fff3cd"),
        ("RevIN Denorm\nŷ·σ_ctx + μ_ctx\n→ Predicted kWh",    "#f3e5f5"),
    ]

    for ax_idx, (ax, (phase_label, phase_color), boxes) in enumerate(
        zip(axes, labels, [train_boxes, infer_boxes])
    ):
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 5)
        ax.axis("off")

        # Phase label
        ax.text(0.3, 4.5, phase_label, fontsize=19, fontweight="bold",
                color=phase_color)

        # Draw boxes
        xs = [2, 5, 8, 11, 14]
        bw, bh = 2.6, 3.0
        for i, (x, (lbl, col)) in enumerate(zip(xs, boxes)):
            draw_box(ax, x, 2.3, bw, bh, lbl, facecolor=col,
                     fontsize=14, subfontsize=12)
            if i < len(xs) - 1:
                arrow(ax, x + bw/2, 2.3, xs[i+1] - bw/2, 2.3)

        # Final output
        if ax_idx == 1:
            draw_box(ax, 17.0, 2.3, 2.4, 1.4, "Predicted\n24h Energy\n(kWh)",
                     facecolor="#c8e6c9", edgecolor="#2e7d32", fontsize=13)
            arrow(ax, 15.3, 2.3, 15.7, 2.3)

    # Note at bottom
    axes[1].text(9, 0.3,
        "RevIN removes building-specific scale  →  model learns only temporal patterns  →  no location info needed",
        ha="center", fontsize=14, color="#555", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "patent_fig3_revin.png")


# ── Fig. 4: N-Scaling Curve ───────────────────────────────────────────────────
def fig4_nscaling():
    # Data from Table 4
    n_vals  = [1,   3,    5,    10,   20,   50,   70,   80]
    n_bldgs = [14,  42,   70,   140,  280,  700,  980,  1120]
    nrmse   = [14.72, 13.47, 13.28, 13.18, 13.23, 12.93, 13.20, 13.15]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title("Fig. 4  Prediction Performance vs. Number of Training Buildings (N-Scaling)",
                 fontsize=20, fontweight="bold", pad=16)

    ax.plot(n_bldgs, nrmse, "o-", color="#1565C0", lw=2.5, ms=10,
            label="Korean Parametric (RevIN ON)", zorder=5)

    sota = 13.27
    ax.axhline(sota, color="red", ls="--", lw=2,
               label=f"BB SOTA-M (900K buildings): {sota}%")
    ax.axhline(15.28, color="orange", ls=":", lw=2,
               label="BB-700 (RevIN ON): 15.28%")

    # Saturation shading
    ax.axvspan(140, 1200, alpha=0.08, color="green", label="Saturation zone (n ≥ 10)")

    # Annotation: n=5 matching SOTA
    ax.annotate(f"n=5 (70 bldgs): 13.28%\n≈ SOTA within 0.01 pp",
                xy=(70, 13.28), xytext=(200, 13.65),
                fontsize=14, ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888"))

    # Annotation: best at n=50
    ax.annotate("n=50 (700 bldgs)\nbest: 12.93%",
                xy=(700, 12.93), xytext=(480, 12.60),
                fontsize=14, ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.35", fc="#e3f2fd", ec="#1565C0"))

    # Secondary x-axis for n values
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(n_bldgs)
    ax2.set_xticklabels([f"n={n}" for n in n_vals], fontsize=13)
    ax2.set_xlabel("n (samples per archetype)", fontsize=16)

    ax.set_xlabel("Number of Training Buildings", fontsize=17)
    ax.set_ylabel("NRMSE (%)", fontsize=17)
    ax.set_ylim(12.2, 15.8)
    ax.legend(loc="upper right", fontsize=14, framealpha=0.9)
    ax.grid(axis="y", alpha=0.4)
    ax.tick_params(labelsize=15)

    save(fig, "patent_fig4_nscaling.png")


# ── Fig. 5: Performance Comparison Bar Chart ──────────────────────────────────
def fig5_comparison():
    models = [
        "BB SOTA-M\n(900K, no RevIN)",
        "Korean-700\n(RevIN ON)",
        "Korean-700\n(RevIN OFF)",
        "BB-700\n(aug-matched,\nRevIN ON)",
        "BB-700\n(RevIN ON,\nno aug)",
        "BB-700\n(RevIN OFF)",
        "Persistence\nEnsemble",
    ]
    nrmse = [13.27, 13.11, 14.72, 14.26, 15.28, 16.44, 16.68]
    colors = ["#e53935", "#1565C0", "#42A5F5", "#EF6C00", "#FFA726", "#FFD54F", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title("Fig. 5  Performance Comparison: Proposed Method vs. Prior Art",
                 fontsize=21, fontweight="bold", pad=16)

    bars = ax.bar(range(len(models)), nrmse, color=colors, edgecolor="#333", linewidth=1.2,
                  width=0.65)

    # SOTA reference line
    sota = 13.27
    ax.axhline(sota, color="red", ls="--", lw=2, alpha=0.8,
               label=f"BB SOTA-M baseline: {sota}%")

    # Value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, nrmse)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.06,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold")

    # Training size inside bars
    sizes = ["900K", "700", "700", "700", "700", "700", "—"]
    for i, (bar, sz) in enumerate(zip(bars, sizes)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_y() + bar.get_height()/2,
                f"({sz})", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")

    # Data design effect bracket
    x_k = 1   # Korean-700 RevIN ON
    x_b = 3   # BB-700 aug-matched
    y_bracket = 14.55
    ax.annotate("", xy=(x_k + 0.05, y_bracket), xytext=(x_b - 0.05, y_bracket),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.8))
    ax.text((x_k + x_b) / 2, y_bracket + 0.12,
            "Data design effect\n1.15 pp",
            ha="center", va="bottom", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff9c4", ec="#888"))

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=13)
    ax.set_ylabel("NRMSE (%)", fontsize=17)
    ax.set_ylim(11.5, 18.0)
    ax.legend(fontsize=14, loc="upper left")
    ax.grid(axis="y", alpha=0.35)
    ax.tick_params(axis="y", labelsize=15)

    save(fig, "patent_fig5_comparison.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating patent figures...")
    fig1_system()
    fig2_lhs()
    fig3_revin()
    fig4_nscaling()
    fig5_comparison()
    print("Done.")
