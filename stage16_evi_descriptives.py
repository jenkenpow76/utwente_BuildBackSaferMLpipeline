"""
stage11_evi_descriptives.py
======================
Stage 2 | EVI — Descriptives and Frequencies

Input  : imputed_EVI_scores.csv  (written by Stage 0c)
Outputs: evi_distribution.png          — histogram of EVI_norm_1_5
         evi_indicator_frequencies.png — stacked bar chart per indicator

Description
-----------
Mirrors stage06_fvi_descriptives.py exactly. Visualises the distribution of
the composite EVI score and the frequency of vulnerability levels
(Low/Medium/High) for each of the 12 CIMDEN indicators.

Reference
---------
Villagran De León, J.C. (2004). Vulnerability: A Conceptual and
    Methodological Review. UNU-EHS Source No. 4. United Nations University.
"""

# ─── Pipeline note ───────────────────────────────────────────────────────────
# When run via run_evi_pipeline.py, the working directory is OUTPUT_DIR.
# INPUT_CSV = str(Path.cwd() / "imputed_EVI_scores.csv") resolves to OUTPUT_DIR/imputed_EVI_scores.csv.
# ─────────────────────────────────────────────────────────────────────────────

"""
EVI CIMDEN — Descriptives and Frequencies
==========================================

Input  : imputed_EVI_scores.csv   (output of stage05_mice_imputation.py)
Outputs: evi_distribution.png
         evi_indicator_frequencies.png

Reference:
    Villagran De León, J.C. (2004). Vulnerability: A Conceptual and
    Methodological Review. UNU-EHS Source No. 4. United Nations
    University, Bonn. Section 3.4.2, pp. 43-44.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Import analytical constants from pipeline_config ─────────────────────────
import importlib.util as _ilu, pathlib as _pl
_cfg_path = _pl.Path(__file__).resolve().parent / "pipeline_config.py"
_cfg_spec  = _ilu.spec_from_file_location("pipeline_config", str(_cfg_path))
_cfg_mod   = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)
_PIPELINE_CFG = {k: getattr(_cfg_mod, k) for k in [
    "SURVEYOR_COL","COMMUNITY_COL","FVI_SCORE_COL","EVI_SCORE_COL",
    "BNDRY_LM","BNDRY_MH","STEPWISE_PIN","STEPWISE_POUT",
    "MORANS_THRESHOLD_KM","FVI_WEIGHT","EVI_WEIGHT","RANDOM_STATE",
]}


matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV = str(Path.cwd() / "imputed_EVI_scores.csv")   # update path if running from a different directory
OUT_HIST  = str(Path.cwd() / "evi_distribution.png")
OUT_BARS  = str(Path.cwd() / "evi_indicator_frequencies.png")

# ── Colour palette — matches reference images ─────────────────────────────────
C_LOW    = "#4DB6AC"   # teal        — Low vulnerability
C_MED    = "#FFB74D"   # amber       — Medium vulnerability
C_HIGH   = "#E57373"   # coral-red   — High vulnerability
C_BAR    = "#37474F"   # dark slate  — histogram bars
C_MEAN   = "#E53935"   # red dashed  — mean line
C_MEDIAN = "#7B1FA2"   # purple -.   — median line

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig', low_memory=False)
print(f"Loaded {len(df):,} rows from '{INPUT_CSV}'")

# ── Force numeric types on all FVI output columns ────────────────────────────
# pd.read_csv can infer mixed types if the CSV contains blank strings or
# residual text. pd.to_numeric(..., errors="coerce") converts any non-numeric
# value to NaN rather than raising a TypeError in the histogram step.
fvi_numeric_cols = [
    "EVI_norm_1_5", "EVI_CLASS",
    "EVI_D1_01",   "EVI_D1_02",  "EVI_D2_01",
    "EVI_D2_02",  "EVI_D3_01",     "EVI_D3_02",
    "EVI_D3_03", "EVI_D3_04",    "EVI_D4_01",
    "EVI_D4_02",  "EVI_D5_01", "EVI_D5_04",
]
for col in fvi_numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Convert FVI_INDEX_0_100 to 1–5 scale ─────────────────────────────────────
# Linear mapping: 0 → 1.0,  100 → 5.0
fvi_15  = df["EVI_norm_1_5"].dropna()
n_total = len(fvi_15)

# Descriptives on 1–5 scale
mean_v = fvi_15.mean()
med_v  = fvi_15.median()
sd_v   = fvi_15.std()
min_v  = fvi_15.min()
max_v  = fvi_15.max()
p25_v  = fvi_15.quantile(0.25)
p75_v  = fvi_15.quantile(0.75)

# Class boundaries on 1–5 scale
# Low   0–33.33  on 0-100  →  1.000–2.333
# Med  33.33–66.67          →  2.333–3.667
# High 66.67–100            →  3.667–5.000
BNDRY_LM = _PIPELINE_CFG["BNDRY_LM"]
BNDRY_MH = _PIPELINE_CFG["BNDRY_MH"]

# =============================================================================
# FIGURE 1 — Histogram
# =============================================================================
fig1, ax = plt.subplots(figsize=(13, 7))
fig1.patch.set_facecolor("white")
ax.set_facecolor("white")

# Coloured zone backgrounds
ax.axvspan(1.0,       BNDRY_LM, alpha=0.18, color=C_LOW,  zorder=0)
ax.axvspan(BNDRY_LM, BNDRY_MH, alpha=0.18, color=C_MED,  zorder=0)
ax.axvspan(BNDRY_MH,  5.0,     alpha=0.18, color=C_HIGH, zorder=0)

# Histogram with 0.25-wide bins
bins = np.arange(1.0, 5.26, 0.25)
ax.hist(fvi_15, bins=bins, color=C_BAR, edgecolor="white",
        linewidth=0.7, zorder=2)

# Mean and median vertical lines
ax.axvline(mean_v, color=C_MEAN,   linestyle="--", linewidth=2.0, zorder=4,
           label=f"Mean = {mean_v:.2f}")
ax.axvline(med_v,  color=C_MEDIAN, linestyle="-.", linewidth=2.0, zorder=4,
           label=f"Median = {med_v:.2f}")

# Zone boundary dotted lines
ax.axvline(BNDRY_LM, color=C_LOW,  linestyle=":", linewidth=1.5,
           alpha=0.85, zorder=1)
ax.axvline(BNDRY_MH, color=C_HIGH, linestyle=":", linewidth=1.5,
           alpha=0.85, zorder=1)

# Zone italic labels — draw canvas first so y-limits are known
fig1.canvas.draw()
_, yhi = ax.get_ylim()

for lbl, xc, col in [
    ("Low",    (1.0       + BNDRY_LM) / 2, C_LOW),
    ("Medium", (BNDRY_LM + BNDRY_MH) / 2, "#8D6E00"),
    ("High",   (BNDRY_MH + 5.0)      / 2, C_HIGH),
]:
    ax.text(xc, yhi * 0.975, lbl,
            ha="center", va="top",
            fontstyle="italic", fontsize=12.5,
            color=col, zorder=5)

# Stats box — top-right, monospaced font
stats_lines = (
    f"Mean:    {mean_v:>5.2f}\n"
    f"Median: {med_v:>5.2f}\n"
    f"SD:        {sd_v:>5.2f}\n"
    f"Min:       {min_v:>5.2f}\n"
    f"Max:     {max_v:>5.2f}\n"
    f"P25:      {p25_v:>5.2f}\n"
    f"P75:      {p75_v:>5.2f}"
)
ax.text(0.978, 0.975, stats_lines,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10.0, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#BBBBBB", linewidth=0.8, alpha=0.93),
        zorder=6)

# Legend — top-left
legend_handles = [
    mpatches.Patch(facecolor=C_LOW,  alpha=0.45, edgecolor="none",
                   label="Low zone  (1–2.33)"),
    mpatches.Patch(facecolor=C_MED,  alpha=0.45, edgecolor="none",
                   label="Medium zone (2.33–3.67)"),
    mpatches.Patch(facecolor=C_HIGH, alpha=0.45, edgecolor="none",
                   label="High zone  (3.67–5)"),
    Line2D([0], [0], color=C_MEAN,   linestyle="--", linewidth=2.0,
           label=f"Mean = {mean_v:.2f}"),
    Line2D([0], [0], color=C_MEDIAN, linestyle="-.", linewidth=2.0,
           label=f"Median = {med_v:.2f}"),
]
ax.legend(handles=legend_handles, loc="upper left",
          fontsize=9.5, framealpha=0.93,
          edgecolor="#BBBBBB", frameon=True)

# Axis labels and formatting
ax.set_xlim(1.0, 5.0)
ax.set_xlabel("Composite EVI (1–5 scale)", fontsize=13, labelpad=8)
ax.set_ylabel("Number of houses",          fontsize=13, labelpad=8)
ax.set_title(
    f"Distribution of Composite Earthquake Vulnerability Index\n"
    f"by House (n = {n_total:,})",
    fontsize=14, fontweight="bold", pad=14,
)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
ax.tick_params(axis="both", labelsize=11)

plt.tight_layout(pad=1.5)
fig1.savefig(OUT_HIST, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved: {OUT_HIST}")

# =============================================================================
# FIGURE 2 — Stacked horizontal bar chart
# =============================================================================

# Indicators — column : display label
# Listed in display order (bottom-to-top = FVI1 last in list → plotted at y=0)
indicators = [
    ("EVI_D1_01",   "EVI D1-01  Adjacency"),
    ("EVI_D1_02",  "EVI D1-02  Slope/Edge"),
    ("EVI_D2_01", "EVI D2-01  Plan shape"),
    ("EVI_D2_02",  "EVI D2-02  LW ratio"),
    ("EVI_D3_01",     "EVI D3-01  H/t ratio"),
    ("EVI_D3_02",     "EVI D3-02  Openings %"),
    ("EVI_D3_03", "EVI D3-03  Opening dist"),
    ("EVI_D3_04",    "EVI D3-04  Wall thick"),
    ("EVI_D4_01",    "EVI D4-01  Roof shape"),
    ("EVI_D4_02",  "EVI D4-02  Roof mass"),
    ("EVI_D5_01",  "EVI D5-01  Horiz. bands"),
    ("EVI_D5_04",  "EVI D5-04  Roof fasten"),
]

# Build percentage rows — reversed list puts FVI11 at top, FVI1 at bottom
rows = []
for col, label in indicators:
    s   = df[col].dropna()
    n_i = len(s)
    rows.append({
        "label": label,
        "n":     n_i,
        "low":   (s == 1).sum() / n_i * 100 if n_i else 0.0,
        "med":   (s == 3).sum() / n_i * 100 if n_i else 0.0,
        "high":  (s == 5).sum() / n_i * 100 if n_i else 0.0,
    })

labels   = [r["label"] for r in rows]
ns       = [r["n"]     for r in rows]
pct_low  = np.array([r["low"]  for r in rows])
pct_med  = np.array([r["med"]  for r in rows])
pct_high = np.array([r["high"] for r in rows])

fig2, ax2 = plt.subplots(figsize=(13, 8))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

y      = np.arange(len(labels))
height = 0.55

# Stacked bars: Low | Medium | High
b_low  = ax2.barh(y, pct_low,
                  height=height, color=C_LOW,  label="Low (1)",    zorder=2)
b_med  = ax2.barh(y, pct_med,  left=pct_low,
                  height=height, color=C_MED,  label="Medium (3)", zorder=2)
b_high = ax2.barh(y, pct_high, left=pct_low + pct_med,
                  height=height, color=C_HIGH, label="High (5)",   zorder=2)

# White bold percentage labels centred in each segment (shown when >= 8 pp)
def add_pct_labels(bars, lefts, values):
    """Centre white bold percentage text in each bar segment if wide enough."""
    for bar, left, val in zip(bars, lefts, values):
        if val >= 8.0:
            cx = left + val / 2.0
            ax2.text(cx, bar.get_y() + bar.get_height() / 2.0,
                     f"{val:.0f}%",
                     ha="center", va="center",
                     fontsize=10.0, fontweight="bold",
                     color="white", zorder=4)

add_pct_labels(b_low,  np.zeros(len(rows)),  pct_low)
add_pct_labels(b_med,  pct_low,              pct_med)
add_pct_labels(b_high, pct_low + pct_med,    pct_high)

# n= labels to the right of each bar
for i, n_val in enumerate(ns):
    ax2.text(101.8, i, f"n={n_val:,}",
             va="center", ha="left", fontsize=9.5, color="#555555")

# Axes formatting
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=10.5)
ax2.set_xlabel("Percentage of houses (%)", fontsize=12, labelpad=8)
ax2.set_xlim(0, 116)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax2.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
ax2.set_title(
    "Earthquake Vulnerability Indicator Score Frequencies\n"
    "(% of non-missing cases)",
    fontsize=14, fontweight="bold", pad=12,
)
ax2.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
ax2.tick_params(axis="both", labelsize=10.5)

# Legend — bottom-right, matches reference image position
legend_patches = [
    mpatches.Patch(color=C_LOW,  label="Low (1)"),
    mpatches.Patch(color=C_MED,  label="Medium (3)"),
    mpatches.Patch(color=C_HIGH, label="High (5)"),
]
ax2.legend(handles=legend_patches,
           title="Vulnerability class", title_fontsize=9.5,
           fontsize=9.5, loc="lower right",
           framealpha=0.93, edgecolor="#BBBBBB", frameon=True)

plt.tight_layout(pad=1.5)
fig2.savefig(OUT_BARS, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {OUT_BARS}")

# =============================================================================
# CONSOLE — Descriptive statistics and frequency tables
# =============================================================================

print("\n" + "=" * 58)
print("COMPOSITE EVI — DESCRIPTIVE STATISTICS  (1–5 scale)")
print("=" * 58)
for lbl, val in [
    ("n",      f"{n_total:,}"),
    ("Mean",   f"{mean_v:.3f}"),
    ("Median", f"{med_v:.3f}"),
    ("SD",     f"{sd_v:.3f}"),
    ("Min",    f"{min_v:.3f}"),
    ("Max",    f"{max_v:.3f}"),
    ("P25",    f"{p25_v:.3f}"),
    ("P75",    f"{p75_v:.3f}"),
]:
    print(f"  {lbl:<10}: {val}")

print("\n" + "=" * 58)
print("EVI CLASS FREQUENCIES")
print("=" * 58)
class_map   = {1: "Low", 2: "Medium", 3: "High"}
class_freqs = df["EVI_CLASS"].value_counts().sort_index()
for k, cnt in class_freqs.items():
    pct = cnt / len(df) * 100
    print(f"  {int(k)}  {class_map[int(k)]:<8}:  {cnt:>5,}  ({pct:.1f}%)")

print("\n" + "=" * 58)
print("INDICATOR LEVEL FREQUENCIES  (% of non-missing cases)")
print("=" * 58)
level_map = {1.0: "Low", 3.0: "Medium", 5.0: "High"}
for col, label in indicators:
    s   = df[col].dropna()
    n_i = len(s)
    print(f"\n  {label}  (n={n_i:,})")
    for lvl in [1.0, 3.0, 5.0]:
        cnt = (s == lvl).sum()
        pct = cnt / n_i * 100 if n_i > 0 else 0.0
        print(f"    {level_map[lvl]:<8}:  {cnt:>5,}  ({pct:.1f}%)")
