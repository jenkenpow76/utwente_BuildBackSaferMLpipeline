"""
missing_data_analysis.py
========================
Missing Data Diagnostics for EVI Indicator Columns
====================================================

PURPOSE
-------
This script tests whether missing data in the EVI (Earthquake Vulnerability
Index) indicators are Missing Completely At Random (MCAR), Missing At Random
(MAR), or Missing Not At Random (MNAR).

Understanding WHY data are missing is critical before deciding how to handle
them. Using pairwise deletion (as in stage02_evi_scoring.py) is only defensible
if missingness is random. If it is not, the composite EVI scores may be
systematically biased.

MISSING DATA MECHANISMS — DEFINITIONS
--------------------------------------
  MCAR  Missing Completely At Random
        The probability of a value being missing has NO relationship to any
        variable — observed or unobserved. Example: a surveyor accidentally
        skipped a question. Pairwise deletion is valid under MCAR.

  MAR   Missing At Random
        Missingness is related to OBSERVED variables, but not to the missing
        value itself. Example: stone buildings are harder to measure, so
        wall thickness is missing more often for stone — but once you know
        the material, missingness is fully explained. Multiple imputation
        is the preferred remedy under MAR.

  MNAR  Missing Not At Random
        Missingness is related to the UNOBSERVED (missing) value itself.
        Example: a collapsed wall has no measurable thickness — the damage
        IS the reason for missingness. MNAR cannot be proven from data alone;
        it requires domain reasoning and sensitivity analysis.

APPROACH
--------
  Step 1  Descriptive — visualise missingness patterns
  Step 2  Little's MCAR test (formal statistical test)
  Step 3  Logistic regression — test whether observed variables
          predict missingness (MAR signal)
  Step 4  MNAR assessment — domain reasoning + sensitivity analysis

DEPENDENCIES
------------
  pip install pandas numpy matplotlib seaborn scipy scikit-learn pyampute

REFERENCES
----------
Little, R.J.A. (1988). A test of missing completely at random for
  multivariate data with missing values. Journal of the American
  Statistical Association, 83(404), 1198-1202.
  https://doi.org/10.2307/2290157

Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
  epidemiological and clinical research. BMJ, 338, b2393.
  https://doi.org/10.1136/bmj.b2393

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
  CRC Press. https://stefvanbuuren.name/fimd/
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# pyampute provides Little's MCAR test
# Install: pip install pyampute
try:
    from pyampute.exploration.mcar_statistical_tests import MCARTest
    PYAMPUTE_AVAILABLE = True
except ImportError:
    PYAMPUTE_AVAILABLE = False
    print(
        "WARNING: pyampute not installed. Little's MCAR test will be skipped.\n"
        "Install with: pip install pyampute"
    )

# =============================================================================
# FILE PATHS  — update if running from a different directory
# =============================================================================

# Input: use the scored EVI output from stage02_evi_scoring.py so we have both
# the raw indicators AND the pre-computed helper columns (NumFloors, etc.)
INPUT_CSV = str(
    Path(__file__).resolve().parent / "EVI_CIMDEN_scores.csv"
)

# Output directory for saved figures
OUTPUT_DIR = Path(__file__).resolve().parent / "missing_data_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)

# infer_objects(copy=False) opts into the future pandas behaviour and
# suppresses the FutureWarning about silent downcasting after replace().
# See: https://pandas.pydata.org/docs/whatsnew/v2.1.0.html
df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)

print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns")

# EVI indicator columns — the 12 scored indicators
EVI_INDICATORS = [
    "EVI_D1_01", "EVI_D1_02",
    "EVI_D2_01", "EVI_D2_02",
    "EVI_D3_01", "EVI_D3_02", "EVI_D3_03", "EVI_D3_04",
    "EVI_D4_01", "EVI_D4_02",
    "EVI_D5_01", "EVI_D5_04",
]

# Human-readable labels for plots
INDICATOR_LABELS = {
    "EVI_D1_01": "D1.01 Adjacency",
    "EVI_D1_02": "D1.02 Slope/Edge",
    "EVI_D2_01": "D2.01 Plan Shape",
    "EVI_D2_02": "D2.02 L/W Ratio",
    "EVI_D3_01": "D3.01 H/t Ratio",
    "EVI_D3_02": "D3.02 Openings %",
    "EVI_D3_03": "D3.03 Opening Dist",
    "EVI_D3_04": "D3.04 Wall Thick",
    "EVI_D4_01": "D4.01 Roof Shape",
    "EVI_D4_02": "D4.02 Roof Mass",
    "EVI_D5_01": "D5.01 H. Bands",
    "EVI_D5_04": "D5.04 Roof Fasten",
}

# Observed (auxiliary) variables used in MAR logistic regression.
# These must be present in the scored CSV.
AUXILIARY_VARS = ["NumFloors", "LW_ratio", "Ht_ratio"]


# =============================================================================
# STEP 1 — DESCRIPTIVE: VISUALISE MISSINGNESS PATTERNS
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS SHOWS
# ---------------
# Two complementary views of missingness:
#   (a) Bar chart — % missing per indicator. High missingness on specific
#       indicators may suggest structural survey problems or MNAR patterns.
#   (b) Heatmap — co-occurrence of missingness. If certain indicators are
#       always missing together, they likely share a common cause (e.g., a
#       survey question not asked, or a building type not applicable).
#
# HOW TO INTERPRET
# ----------------
#   - Uniform low missingness across all indicators → consistent with MCAR
#   - Specific indicators with very high missingness → investigate cause
#   - Strong co-occurrence patterns in heatmap → MAR or MNAR more likely
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 1 — DESCRIPTIVE MISSINGNESS ANALYSIS")
print("=" * 65)

df_ind = df[EVI_INDICATORS].copy()
df_ind.rename(columns=INDICATOR_LABELS, inplace=True)

# --- 1a: Per-indicator missingness bar chart --------------------------------

miss_pct = df_ind.isna().mean() * 100
miss_pct = miss_pct.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(miss_pct.index, miss_pct.values, color="#4C72B0", edgecolor="white")
ax.set_xlabel("% Missing", fontsize=11)
ax.set_title(
    "Missing Data Rate per EVI Indicator\n"
    "Nepal Lumbini Survey (n=2,993 buildings) | Values >20% warrant investigation",
    fontsize=11,
    pad=12,
)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.axvline(20, color="crimson", linestyle="--", linewidth=1, label="20% threshold")
ax.legend(fontsize=9)

# Annotate bars with exact percentage
for bar, pct in zip(bars, miss_pct.values):
    ax.text(
        pct + 0.3, bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%", va="center", fontsize=8
    )

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_missingness_per_indicator.png", dpi=150)
plt.close()
print("Saved: fig1_missingness_per_indicator.png")
print(miss_pct.to_string())

# --- 1b: Co-occurrence heatmap -----------------------------------------------
# Shows the proportion of rows where BOTH indicators are missing simultaneously.
# A high value (dark cell) means two indicators tend to be missing together.
# Both sides must be float before the dot product — bool.T.dot(bool)
# produces object dtype in pandas, which matplotlib cannot render.

miss_matrix = df_ind.isna().astype(float)
co_occur = miss_matrix.T.dot(miss_matrix) / len(df_ind)  # proportion

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    co_occur,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    linewidths=0.5,
    ax=ax,
    vmin=0,
    vmax=1,
)
ax.set_title(
    "Co-occurrence of Missingness — EVI Indicators\n"
    "Cell value = proportion of buildings where both indicators are simultaneously missing",
    fontsize=11,
    pad=12,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_missingness_cooccurrence.png", dpi=150)
plt.close()
print("Saved: fig2_missingness_cooccurrence.png")


# =============================================================================
# STEP 2 — LITTLE'S MCAR TEST
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS TESTS
# ---------------
# Little's (1988) chi-square test is the standard formal test for MCAR.
# It compares the observed pattern of missingness to what would be expected
# if data were truly missing at random.
#
# H0 : Data are MCAR
# H1 : Data are NOT MCAR (MAR or MNAR)
#
# HOW TO INTERPRET
# ----------------
#   p > 0.05 → Fail to reject H0. Data are consistent with MCAR.
#              Pairwise deletion in EVI composite is defensible.
#
#   p ≤ 0.05 → Reject H0. Data are NOT MCAR. Proceed to Step 3 to
#              test for MAR. Consider multiple imputation.
#
# LIMITATION
# ----------
# A significant result distinguishes MCAR from (MAR + MNAR) but does NOT
# tell you which of MAR or MNAR applies. That requires Step 3 and domain
# reasoning (Step 4).
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2 — LITTLE'S MCAR TEST")
print("=" * 65)

if PYAMPUTE_AVAILABLE:
    # pyampute expects a plain numpy array or DataFrame with numeric columns
    df_mcar = df[EVI_INDICATORS].copy()

    mcar_test = MCARTest(method="little")
    try:
        p_val = mcar_test.little_mcar_test(df_mcar)
        print(f"\nLittle's MCAR test p-value : {p_val:.4f}")

        if p_val > 0.05:
            print(
                "RESULT : Fail to reject H0 (p > 0.05).\n"
                "         Data are consistent with MCAR.\n"
                "         Pairwise deletion in the EVI composite is defensible."
            )
        else:
            print(
                "RESULT : Reject H0 (p ≤ 0.05).\n"
                "         Data are NOT MCAR. Missingness is systematic.\n"
                "         Proceed to Step 3 to test for MAR.\n"
                "         Consider multiple imputation (e.g., mice in R or\n"
                "         IterativeImputer in scikit-learn)."
            )
    except Exception as e:
        print(f"Little's test raised an error: {e}")
        print("This can occur if all rows have complete data for some indicators.")
else:
    print("pyampute not available — Little's MCAR test skipped.")
    print("Install with: pip install pyampute")


# =============================================================================
# STEP 3 — MAR TEST: LOGISTIC REGRESSION ON MISSINGNESS
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS TESTS
# ---------------
# For each EVI indicator, we create a binary outcome:
#   1 = this indicator is MISSING for a given building
#   0 = this indicator is OBSERVED
#
# We then fit a logistic regression using observed auxiliary variables
# (NumFloors, LW_ratio, Ht_ratio) as predictors.
#
# If observed variables significantly predict missingness, that is evidence
# of MAR — missingness is systematic but explainable by what we CAN see.
#
# HOW TO INTERPRET
# ----------------
#   AUC close to 0.5 → predictors have no power → consistent with MCAR
#   AUC > 0.65       → predictors explain missingness → MAR likely
#   AUC > 0.80       → strong MAR signal — multiple imputation strongly advised
#
# NOTE: A high AUC does NOT prove MAR over MNAR. It shows that observed
# variables predict missingness. MNAR requires domain reasoning (Step 4).
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 3 — MAR TEST: LOGISTIC REGRESSION ON MISSINGNESS")
print("=" * 65)
print(
    "\nFor each indicator, a logistic regression predicts whether the value\n"
    "is missing using observed auxiliary variables: NumFloors, LW_ratio, Ht_ratio.\n"
    "AUC (Area Under the ROC Curve) measures prediction power:\n"
    "  0.50 = random (supports MCAR)  |  >0.65 = MAR signal  |  >0.80 = strong MAR\n"
)

# Store results for plotting
mar_results = []

for col in EVI_INDICATORS:
    label = INDICATOR_LABELS[col]

    # Create missingness flag for this indicator
    miss_flag = df[col].isna().astype(int)

    # Build modelling dataframe using auxiliary variables + missingness flag
    model_df = df[AUXILIARY_VARS + [col]].copy()
    model_df["missing"] = miss_flag

    # Drop rows where any AUXILIARY variable is missing
    # (we can only use rows where predictors are fully observed)
    model_df = model_df.dropna(subset=AUXILIARY_VARS)

    n_total   = len(model_df)
    n_missing = model_df["missing"].sum()
    n_obs     = n_total - n_missing

    # Skip if fewer than 10 missing cases — logistic regression unreliable
    if n_missing < 10 or n_obs < 10:
        print(f"  {label:<22}: SKIP — insufficient missing cases (n_missing={n_missing})")
        mar_results.append({
            "indicator": label,
            "n_missing": n_missing,
            "pct_missing": n_missing / len(df) * 100,
            "auc": np.nan,
            "signal": "Insufficient data",
        })
        continue

    X = model_df[AUXILIARY_VARS].values
    y = model_df["missing"].values

    # Standardise features — logistic regression is sensitive to scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression with L2 regularisation (default)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_scaled, y)

    # Predict probabilities for AUC calculation
    y_prob = lr.predict_proba(X_scaled)[:, 1]

    # AUC: 0.5 = no better than chance; 1.0 = perfect prediction
    auc = roc_auc_score(y, y_prob)

    # Interpret AUC
    if auc < 0.55:
        signal = "MCAR consistent"
    elif auc < 0.65:
        signal = "Weak MAR signal"
    elif auc < 0.80:
        signal = "Moderate MAR signal"
    else:
        signal = "Strong MAR signal"

    print(
        f"  {label:<22}: AUC={auc:.3f}  n_miss={n_missing:4d} "
        f"({n_missing / len(df) * 100:.1f}%)  → {signal}"
    )

    mar_results.append({
        "indicator": label,
        "n_missing": n_missing,
        "pct_missing": n_missing / len(df) * 100,
        "auc": auc,
        "signal": signal,
    })

# --- Plot AUC results --------------------------------------------------------

mar_df = pd.DataFrame(mar_results).dropna(subset=["auc"])

if not mar_df.empty:
    mar_df = mar_df.sort_values("auc", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Colour bars by signal strength
    colours = []
    for auc in mar_df["auc"]:
        if auc < 0.55:
            colours.append("#2ca02c")    # green  = MCAR consistent
        elif auc < 0.65:
            colours.append("#ff7f0e")   # orange = weak MAR
        elif auc < 0.80:
            colours.append("#d62728")   # red    = moderate MAR
        else:
            colours.append("#8c1c1c")   # dark red = strong MAR

    bars = ax.barh(
        mar_df["indicator"], mar_df["auc"],
        color=colours, edgecolor="white"
    )

    # Reference lines
    ax.axvline(0.50, color="grey",   linestyle=":",  linewidth=1.2, label="0.50 chance level")
    ax.axvline(0.65, color="#ff7f0e", linestyle="--", linewidth=1.2, label="0.65 weak MAR threshold")
    ax.axvline(0.80, color="#d62728", linestyle="--", linewidth=1.2, label="0.80 strong MAR threshold")

    ax.set_xlabel("AUC (ROC)", fontsize=11)
    ax.set_xlim(0.4, 1.0)
    ax.set_title(
        "MAR Test: Logistic Regression AUC per EVI Indicator\n"
        "(AUC > 0.65 suggests observed variables explain why data are missing — MAR likely)",
        fontsize=11,
        pad=12,
    )
    ax.legend(fontsize=9, loc="lower right")

    # Annotate bars
    for bar, auc in zip(bars, mar_df["auc"]):
        ax.text(
            auc + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{auc:.3f}", va="center", fontsize=8
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_mar_auc.png", dpi=150)
    plt.close()
    print("\nSaved: fig3_mar_auc.png")


# =============================================================================
# STEP 4 — MNAR ASSESSMENT: DOMAIN REASONING + SENSITIVITY ANALYSIS
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# WHY YOU CANNOT TEST MNAR STATISTICALLY
# ----------------------------------------
# MNAR means the probability of a value being missing depends on the value
# ITSELF — which you never observe. No statistical test can confirm MNAR
# from the data alone (van Buuren, 2018, §1.2).
#
# WHAT YOU CAN DO
# ---------------
# 1. Apply domain reasoning: which indicators would be unmeasurable precisely
#    because of high vulnerability? (e.g., collapsed walls → no thickness)
# 2. Run a sensitivity analysis: impute missing values at the HIGH end (score=5)
#    and LOW end (score=1) and compare composite EVI distributions.
#    If results change substantially, MNAR is a serious concern.
#
# HOW TO INTERPRET THE SENSITIVITY PLOT
# --------------------------------------
# The plot shows three EVI_composite distributions:
#   - Pairwise mean (current approach): ignores missing values
#   - MNAR-high imputation (score=5):   assumes missing = most vulnerable
#   - MNAR-low imputation  (score=1):   assumes missing = least vulnerable
#
# If the three distributions diverge significantly, your composite scores
# are sensitive to the missing data assumption. This should be reported as
# a limitation and motivates multiple imputation.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 4 — MNAR SENSITIVITY ANALYSIS")
print("=" * 65)
print(
    "\nIndicators most likely to be MNAR in earthquake vulnerability surveys:\n"
    "  EVI_D3_01 (H/t Ratio)    — wall may be collapsed/inaccessible\n"
    "  EVI_D3_04 (Wall Thick)   — same; unmeasurable if wall is damaged\n"
    "  EVI_D3_03 (Opening Dist) — opening may not exist if wall is destroyed\n"
    "  EVI_D5_01 (H. Bands)     — 'can't see' option already coded as score=5\n"
    "\nSensitivity approach: impute missing indicators at score=5 (worst case)\n"
    "and score=1 (best case), then compare composite distributions.\n"
)

# --- MNAR sensitivity: impute at HIGH end (score=5) -------------------------

df_high = df[EVI_INDICATORS].fillna(5)   # worst-case assumption
df_low  = df[EVI_INDICATORS].fillna(1)   # best-case assumption
df_pair = df[EVI_INDICATORS]             # current pairwise mean

composite_pair = df_pair.mean(axis=1)    # pairwise (ignores NaN)
composite_high = df_high.mean(axis=1)    # MNAR worst case
composite_low  = df_low.mean(axis=1)     # MNAR best case

# Summary statistics for each scenario
print(
    f"\n{'Scenario':<30} {'Mean':>6} {'Median':>8} {'SD':>6} "
    f"{'Min':>6} {'Max':>6}"
)
print("-" * 64)
for label, series in [
    ("Pairwise mean (current)", composite_pair),
    ("MNAR-high imputation (=5)", composite_high),
    ("MNAR-low imputation (=1)",  composite_low),
]:
    print(
        f"  {label:<28} {series.mean():>6.3f} {series.median():>8.3f} "
        f"{series.std():>6.3f} {series.min():>6.3f} {series.max():>6.3f}"
    )

# Classify each scenario and show class shifts
def classify_series(s: pd.Series) -> pd.Series:
    """Map composite score (1-5) to class: 1=Low, 2=Medium, 3=High."""
    norm = (s - 1) / 4.0
    return pd.cut(
        norm,
        bins=[-np.inf, 1/3, 2/3, np.inf],
        labels=[1, 2, 3]
    ).astype(float)


class_pair = classify_series(composite_pair)
class_high = classify_series(composite_high)
class_low  = classify_series(composite_low)

print("\nClass distribution under each scenario:")
class_map = {1: "Low", 2: "Medium", 3: "High"}
for k in [1, 2, 3]:
    n_pair = (class_pair == k).sum()
    n_high = (class_high == k).sum()
    n_low  = (class_low  == k).sum()
    print(
        f"  {class_map[k]:<8}: "
        f"Pairwise={n_pair:4d} ({n_pair/len(df)*100:.1f}%)  "
        f"MNAR-high={n_high:4d} ({n_high/len(df)*100:.1f}%)  "
        f"MNAR-low={n_low:4d} ({n_low/len(df)*100:.1f}%)"
    )

# --- Sensitivity plot --------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

bins = np.linspace(1, 5, 30)

ax.hist(
    composite_pair.dropna(), bins=bins, alpha=0.6,
    label="Pairwise mean (current)", color="#4C72B0", edgecolor="white"
)
ax.hist(
    composite_high, bins=bins, alpha=0.5,
    label="MNAR-high imputed (score=5)", color="#d62728", edgecolor="white"
)
ax.hist(
    composite_low, bins=bins, alpha=0.5,
    label="MNAR-low imputed (score=1)", color="#2ca02c", edgecolor="white"
)

ax.set_xlabel("EVI Composite Score (1–5)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(
    "MNAR Sensitivity Analysis: EVI Composite Score Under Different Imputation Assumptions\n"
    "Observed vs worst-case and best-case bounds\n"
    "Large divergence between curves indicates high sensitivity to MNAR assumptions",
    fontsize=11,
    pad=12,
)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_mnar_sensitivity.png", dpi=150)
plt.close()
print("\nSaved: fig4_mnar_sensitivity.png")


# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY — MISSING DATA DIAGNOSIS")
print("=" * 65)
print(
    "\nIndicator            | % Miss | AUC   | Likely Mechanism"
    "\n" + "-" * 65
)

for entry in mar_results:
    auc_str = f"{entry['auc']:.3f}" if not np.isnan(entry["auc"]) else "  N/A"
    print(
        f"  {entry['indicator']:<22}| "
        f"{entry['pct_missing']:5.1f}% | "
        f"{auc_str} | "
        f"{entry['signal']}"
    )

print(
    "\nNEXT STEPS"
    "\n----------"
    "\n  MCAR confirmed (p>0.05, AUC~0.5) :"
    "\n    Pairwise deletion is defensible. Report EVI_n_valid as a"
    "\n    data quality flag."
    "\n"
    "\n  MAR confirmed (AUC>0.65) :"
    "\n    Use Multiple Imputation. In Python: sklearn.impute.IterativeImputer"
    "\n    (MICE-equivalent). In R: mice package."
    "\n    Reference: van Buuren (2018) https://stefvanbuuren.name/fimd/"
    "\n"
    "\n  MNAR suspected (domain reasoning) :"
    "\n    Report as a limitation. Conduct sensitivity analysis (Step 4)."
    "\n    Consider pattern-mixture models if MNAR is severe."
    "\n    Reference: Little & Rubin (2002) Statistical Analysis with"
    "\n    Missing Data. https://doi.org/10.1002/9781119013563"
)

print(f"\nAll figures saved to: {OUTPUT_DIR}")