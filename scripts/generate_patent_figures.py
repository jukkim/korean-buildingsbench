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
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
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


# ── Fig. 2: LHS Parameter Generation  (portrait, A4-friendly) ────────────────
def fig2_lhs():
    # Portrait: 10" wide × 16" tall → fits A4 with minimal scale-down
    fig = plt.figure(figsize=(10, 16))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.975, "Fig. 2  Parameter Generation by 12D LHS",
             ha="center", va="top", fontsize=22, fontweight="bold")

    # ── Flowchart (top 60%) ───────────────────────────────────────────
    ax = fig.add_axes([0.08, 0.38, 0.84, 0.58])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    steps = [
        (5, 9.0, "Define Building Archetypes",  "14 types: office, retail, school,\nhotel, hospital, warehouse ...", "#cce5ff"),
        (5, 7.0, "Define 12D Parameter Space",  "op_start, baseload_pct, ramp_hours,\nnight_equip_frac, process_load_frac ...", "#ffd6cc"),
        (5, 5.0, "Latin Hypercube Sampling",     "N=50 samples per archetype\n→ space-filling, no clustering",      "#d4edda"),
        (5, 3.0, "Modify EnergyPlus IDF Files", "Adjust schedules, equipment density,\noccupancy per parameter set", "#fff3cd"),
        (5, 1.0, "Output: 700 IDF Files",        "14 archetypes × 50 samples × 5 cities",                           "#f3e5f5"),
    ]
    bw, bh = 8.5, 1.55
    for x, y, lbl, sub, col in steps:
        draw_box(ax, x, y, bw, bh, lbl, sub, facecolor=col, fontsize=18, subfontsize=15)
    for i in range(len(steps) - 1):
        arrow(ax, 5, steps[i][1] - bh/2, 5, steps[i+1][1] + bh/2)

    # ── Bottom panel: scatter + key properties side by side ───────────
    ax_sc = fig.add_axes([0.08, 0.13, 0.44, 0.22])
    np.random.seed(0)
    n = 50
    xl = (np.random.permutation(n) + np.random.uniform(0, 1, n)) / n
    yl = (np.random.permutation(n) + np.random.uniform(0, 1, n)) / n
    ax_sc.scatter(xl * 12, yl, c="#1f77b4", s=60, alpha=0.8)
    ax_sc.set_xlabel("op_start (h)", fontsize=15)
    ax_sc.set_ylabel("baseload_pct", fontsize=15)
    ax_sc.set_title("2D LHS Projection (50 samples)", fontsize=15, fontweight="bold")
    ax_sc.tick_params(labelsize=13)

    ax_kp = fig.add_axes([0.56, 0.10, 0.40, 0.26])
    ax_kp.axis("off")
    ax_kp.text(0.05, 0.95,
        "Key Properties:\n\n"
        "•  Space-filling coverage\n"
        "•  Each axis sampled evenly\n"
        "•  No clustering or gaps\n"
        "•  700 buildings cover 12D space",
        va="top", fontsize=15, linespacing=1.8,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bbb", lw=1.5))

    save(fig, "patent_fig2_lhs.png")


# ── Fig. 3: RevIN + Transformer Model (portrait, A4-friendly) ────────────────
def fig3_revin():
    # Portrait 10" × 16" — 3 boxes per row, 2 rows per phase → compact & readable
    fig = plt.figure(figsize=(10, 18))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.975,
             "Fig. 3  RevIN + Transformer: Training & Inference",
             ha="center", va="top", fontsize=21, fontweight="bold")

    # ── Phase definitions ─────────────────────────────────────────────
    # Training: 5 steps → row1(3) + row2(2)
    # Inference: 5 steps + output → row1(3) + row2(3)
    train_r1 = [
        ("Simulation Data\n700 buildings\n8760 h each",   "#cce5ff"),
        ("Sliding Window\n168h context\n+ 24h target",    "#ffd6cc"),
        ("RevIN Normalize\nx_norm =\n(x − μ) / σ",       "#d4edda"),
    ]
    train_r2 = [
        ("Transformer\nEncoder-Decoder\n15.8M params",    "#fff3cd"),
        ("Gaussian NLL Loss\n→ minimize\nper-batch",      "#f3e5f5"),
    ]
    infer_r1 = [
        ("Target Building\nHistory (168h)\nNO location",  "#d4edda"),
        ("Compute\nμ_ctx, σ_ctx\nfrom 168h",             "#cce5ff"),
        ("RevIN Normalize\n(x − μ_ctx)\n/ σ_ctx",        "#d4edda"),
    ]
    infer_r2 = [
        ("Trained\nTransformer\n(frozen weights)",        "#fff3cd"),
        ("RevIN Denorm\nŷ · σ_ctx + μ_ctx\n→ kWh",       "#f3e5f5"),
        ("Predicted\n24h Energy\n(kWh)",                  "#c8e6c9"),
    ]

    def draw_phase(fig, y_top, phase_label, phase_color, row1, row2, output_col=None):
        H = 0.22   # height fraction per phase block
        ax = fig.add_axes([0.04, y_top - H, 0.92, H])
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.text(0.1, 9.5, phase_label, fontsize=18, fontweight="bold", color=phase_color)

        # Row 1
        xs1 = [1.4, 5.0, 8.6]
        for xi, (lbl, col) in zip(xs1, row1):
            draw_box(ax, xi, 7.5, 2.8, 3.5, lbl, facecolor=col, fontsize=15)
        for i in range(len(xs1)-1):
            arrow(ax, xs1[i]+1.4, 7.5, xs1[i+1]-1.4, 7.5)

        # Arrow from row1-end down to row2-start
        arrow(ax, xs1[-1], 7.5-1.75, xs1[-1], 4.5+1.75)

        # Row 2
        xs2 = [8.6, 5.0] if len(row2) == 2 else [8.6, 5.0, 1.4]
        for xi, (lbl, col) in zip(xs2, row2):
            fc = "#c8e6c9" if (output_col and xi == xs2[-1]) else col
            ec = "#2e7d32" if (output_col and xi == xs2[-1]) else "#333"
            draw_box(ax, xi, 2.8, 2.8, 3.5, lbl, facecolor=fc, edgecolor=ec, fontsize=15)
        for i in range(len(xs2)-1):
            # arrows go right-to-left on row 2
            arrow(ax, xs2[i]-1.4, 2.8, xs2[i+1]+1.4, 2.8)

    draw_phase(fig, 0.94, "① Training Phase",              "#1565C0", train_r1, train_r2)
    draw_phase(fig, 0.68, "② Inference Phase (Zero-Shot)", "#C62828", infer_r1, infer_r2, output_col=True)

    fig.text(0.5, 0.04,
             "RevIN removes per-building magnitude  →  model learns temporal shape only"
             "  →  no location metadata needed",
             ha="center", fontsize=14, color="#444", style="italic")

    save(fig, "patent_fig3_revin.png")


# ── Fig. 4: N-Scaling Curve ───────────────────────────────────────────────────
def fig4_nscaling():
    n_vals  = [1,     3,     5,     10,    20,    50,    70,    80]
    n_bldgs = [14,    42,    70,    140,   280,   700,   980,   1120]
    nrmse   = [14.72, 13.47, 13.28, 13.18, 13.23, 12.93, 13.20, 13.15]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title(
        "Fig. 4  Prediction Performance vs. Number of Training Buildings",
        fontsize=22, fontweight="bold", pad=20)

    ax.plot(n_bldgs, nrmse, "o-", color="#1565C0", lw=3, ms=12,
            label="Korean Parametric (RevIN ON)", zorder=5)

    sota = 13.27
    ax.axhline(sota, color="red",    ls="--", lw=2.5,
               label=f"BB SOTA-M (900K buildings): {sota}%")
    ax.axhline(15.28, color="darkorange", ls=":",  lw=2.5,
               label="BB-700 (RevIN ON, no aug): 15.28%")
    ax.axvspan(140, 1200, alpha=0.08, color="green")
    ax.text(0.20, 0.91, "Saturation zone  (n ≥ 10)",
            transform=ax.transAxes,
            fontsize=16, color="#2e7d32", fontweight="bold", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    # Annotation: n=5
    ax.annotate("n=5  (70 bldgs)\n13.28% ≈ SOTA",
                xy=(70, 13.28), xytext=(220, 13.75),
                fontsize=16, ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", lw=1.5))

    # Annotation: n=50 best
    ax.annotate("n=50  (700 bldgs)\nbest: 12.93%",
                xy=(700, 12.93), xytext=(480, 12.50),
                fontsize=16, ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2),
                bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#1565C0", lw=1.5))

    # Secondary x-axis: only label non-crowded points
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Skip n=1,3 (too close to n=5 at the left edge) — label only key breakpoints
    key_n   = [5,   10,  20,  50,  70,  80]
    key_x   = [70, 140, 280, 700, 980, 1120]
    ax2.set_xticks(key_x)
    ax2.set_xticklabels([f"n={n}" for n in key_n], fontsize=17)
    ax2.set_xlabel("n (samples per archetype)", fontsize=19, labelpad=10)

    ax.set_xlabel("Number of Training Buildings", fontsize=19)
    ax.set_ylabel("NRMSE (%)", fontsize=19)
    ax.set_ylim(12.1, 16.2)
    ax.legend(loc="upper right", fontsize=16, framealpha=0.95)
    ax.grid(axis="y", alpha=0.4)
    ax.tick_params(labelsize=17)

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
