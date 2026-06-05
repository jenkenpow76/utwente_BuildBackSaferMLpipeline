"""
fvi_missing_data_diagnostic.py
================================
Missing Data Mechanism Diagnostic — FVI Indicators
Nepal Lumbini Survey Dataset

Description
-----------
This script diagnoses the missing data mechanism for the 10 active FVI
indicators used in CIMDEN scoring. It follows a three-step protocol:

    Step 1 — Visualise the missing data pattern
    Step 2 — Test for MCAR using Little's test
    Step 3 — Distinguish MAR from MNAR using logistic regression

The goal is to determine whether missingness is:
    MCAR : Missing Completely At Random  — safe to impute
    MAR  : Missing At Random             — imputation works well
    MNAR : Missing Not At Random         — imputation may introduce bias;
                                           must be reported as a limitation

References
----------
Little, R.J.A. (1988). A test of missing completely at random for multivariate
    data with missing values. Journal of the American Statistical Association,
    83(404), 1198–1202. https://doi.org/10.1080/01621459.1988.10478722

Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
    epidemiological and clinical research. BMJ, 338, b2393.
    https://doi.org/10.1136/bmj.b2393

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
    CRC Press. https://stefvanbuuren.name/fimd/

Dependencies
------------
    pip install pandas numpy matplotlib seaborn scikit-learn pyampute
"""

import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")  # Suppress convergence warnings for diagnostics

# =============================================================================
# FILE PATHS
# =============================================================================

INPUT_CSV   = Path(__file__).resolve().parent / "20263003_Nepal_Lumbini_data.csv"
OUTPUT_DIR  = Path(__file__).resolve().parent / "missing_data_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# LOAD RAW DATA
# =============================================================================

df_raw = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
print(f"Loaded {len(df_raw):,} rows x {len(df_raw.columns):,} columns")
# ── Normalise ward name inconsistency ─────────────────────────────────────────
# GD005_VDC_name contains 'tulsipur-ward_14' (hyphen) in the raw survey CSV.
# Normalise to underscore here to match ward_information.csv and stage01/02 output.
_WARD_COL = "GD005_VDC_name"
if _WARD_COL in df_raw.columns:
    _before = df_raw[_WARD_COL].astype(str).eq("tulsipur-ward_14").sum()
    df_raw[_WARD_COL] = (
        df_raw[_WARD_COL].astype(str)
        .str.replace("tulsipur-ward_14", "tulsipur_ward_14", regex=False)
        .where(df_raw[_WARD_COL].notna(), other=np.nan)
    )
    if _before:
        print(f"  Normalised {_before} rows: 'tulsipur-ward_14' → 'tulsipur_ward_14'")


# =============================================================================
# RECONSTRUCT FVI INDICATOR LEVELS FROM RAW SURVEY COLUMNS
# =============================================================================
# The FVI level columns (e.g. FVI2_RoofMat_LVL) are computed by stage01_fvi_scoring.py.
# We reconstruct them here so this script is self-contained and can be run
# on the raw CSV directly, without requiring the pipeline output file.
# Missing values (NaN) in the level columns reflect cases where the survey
# response was absent, ambiguous, or coded as 'i_cannot_see' / 'other'.
# These NaNs are exactly what we are diagnosing.
# =============================================================================

def to_num(series):
    """Coerce a Series to numeric, converting blanks and spaces to NaN.

    infer_objects(copy=False) suppresses the FutureWarning about silent
    downcasting after replace(), introduced in pandas 2.1.
    See: https://pandas.pydata.org/docs/whatsnew/v2.1.0.html
    """
    return pd.to_numeric(
        series.replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )

def flag(col):
    """Return boolean Series: True where binary dummy column equals 1."""
    return to_num(df_raw[col]) == 1


# ── FVI2 : Roof Material ──────────────────────────────────────────────────────
lvl2 = pd.Series(np.nan, index=df_raw.index)
lvl2[flag("SAR04_What_roofing_material_is_used_concrete")              |
     flag("SAR04_What_roofing_material_is_used_cgi_sheet")             |
     flag("SAR04_What_roofing_material_is_used_stone_roofing_tiles")   |
     flag("SAR04_What_roofing_material_is_used_clay_roofing_tiles")    |
     flag("SAR04_What_roofing_material_is_used_concrete_roofing_tiles")] = 1
lvl2[flag("SAR04_What_roofing_material_is_used_thatched_mud")               |
     flag("SAR04_What_roofing_material_is_used_thatched__covered_with_mud") |
     flag("SAR04_What_roofing_material_is_used_other")               |
     flag("SAR04_What_roofing_material_is_used_i_cannot_see")   |
     flag("SAR04_What_roofing_material_is_used_wood")]                       = 5

# ── FVI3 : Roof Connections ───────────────────────────────────────────────────
lvl3 = pd.Series(np.nan, index=df_raw.index)
lvl3[flag("SARA06_Which_material_is_used_to_fastnails") |
     flag("SARA06_Which_material_is_used_to_fastbeams_top_roofing") |
     flag("SARA06_Which_material_is_used_to_fastbolts")] = 1
lvl3[flag("SARA06_Which_material_is_used_to_fastrope")]  = 3
lvl3[flag("SARA06_Which_material_is_used_to_fastno_connection")  |
     flag("SARA06_Which_material_is_used_to_fastdowels")       |
     flag("SARA06_Which_material_is_used_to_fastother")       |

     flag("SARA06_Which_material_is_used_to_faststone_laying_o") |
     flag("SARA06_Which_material_is_used_to_fasti_cannot_see")]  = 5

# ── FVI4 : Wall Materials ─────────────────────────────────────────────────────
lvl4 = pd.Series(np.nan, index=df_raw.index)
lvl4[flag("SAW01_What_material_is_used_for_the_concrete")]            = 1
lvl4[flag("SAW01_What_material_is_used_for_the_stone")]               = 3
lvl4[flag("SAW01_What_material_is_used_for_the_bamboo")              |
     flag("SAW01_What_material_is_used_for_the_compressed_earth")    |
     flag("SAW01_What_material_is_used_for_the_comp_earth_nonstabl") |
     flag("SAW01_What_material_is_used_for_the_mud")                 |
     flag("SAW01_What_material_is_used_for_the_mud_straw")           |
     flag("SAW01_What_material_is_used_for_the_cgi_sheet")           |
     flag("SAW01_What_material_is_used_for_the_tarpaulin")           |
     flag("SAW01_What_material_is_used_for_the_wood")               |
     flag("SAW01_What_material_is_used_for_the_the_wall_is_plastered")   |
     flag("SAW01_What_material_is_used_for_the_other")    |
     flag("SAW01_What_material_is_used_for_the_i_cannot_see")]       = 5

# ── FVI5 : Elevated Ground / Stilts ──────────────────────────────────────────
raw5 = df_raw["SAF02_Is_the_house_build_on_sti"].replace(" ", np.nan).infer_objects(copy=False)
lvl5 = pd.Series(np.nan, index=df_raw.index)
lvl5[raw5.isin(["Yes", "yes", "yes_it_is_build_on_plinth"])] = 1
lvl5[raw5.isin(["No",  "no",  "Don't_know", "Don't_Know"])] = 5

# ── FVI6 : Geometry of Building ───────────────────────────────────────────────
lvl6 = pd.Series(np.nan, index=df_raw.index)
lvl6[flag("SAL05_rectangle") | flag("SAL05_square")] = 1
lvl6[flag("SAL05_l_shape")   | flag("SAL05_t_shape") |
     flag("SAL05_u_shape")   | flag("SAL05_no_usual_shape") |
     flag("SAL05_other")]    = 5

# ── FVI7 : Roof Overhang ──────────────────────────────────────────────────────
raw7 = df_raw["SAR02_How_much_is_th_the_roof_ADD_IMAGE"].replace(" ", np.nan).infer_objects(copy=False)
lvl7 = pd.Series(np.nan, index=df_raw.index)
lvl7[raw7 == "500_600_mm"] = 1
lvl7[raw7 == "600_900_mm"] = 3
lvl7[raw7.isin(["less_than_500_mm", "more_than_900_mm", "none",
                "overhang_size_differs_around_the_house",
                "i_cannot_see", "other"])] = 5

# ── FVI8 : Presence of Apron ──────────────────────────────────────────────────
raw8 = df_raw["SAF03_Does_the_house_on_ADD_IMAGES"].replace(" ", np.nan).infer_objects(copy=False)
lvl8 = pd.Series(np.nan, index=df_raw.index)
lvl8[raw8 == "yes"]                           = 1
lvl8[raw8 == "yes__around_part_of_the_house"] = 3
lvl8[raw8.isin(["no", "i_don_t_know"])]       = 5

# ── FVI9 : Water Drainage on Site ────────────────────────────────────────────
n_drain = (
    flag("SAWA01_Is_there_water_the_site_ADD_IMAGESyes__rain_gutters").astype(int) +
    flag("SAWA01_Is_there_water_the_site_yes__drainage_pipes").astype(int) +
    flag("SAWA01_Is_there_water_the_site_ADD_IMAGESyes__area_slopes_away_f").astype(int)
)
no_drain  = flag("SAWA01_Is_there_water_the_site_no")
unk_drain = flag("SAWA01_Is_there_water_the_site_unknown")
lvl9 = pd.Series(np.nan, index=df_raw.index)
lvl9[n_drain > 0]          = 1
lvl9[no_drain | unk_drain] = 5

# ── FVI10 : Number of Floors (v27.3 revised scheme) ──────────────────────────
# Scoring: 1F → High(5), 2–3F → Low(1), 4F+ → High(5) but unobservable.
# The SAL04 instrument records only three binary floor columns; nf_count
# therefore tops out at 3. See stage01_fvi_scoring.py for full rationale.
nf_count = (
    flag("SAL04_What_parts_of_the_building_are1st_floor").astype(int) +
    flag("SAL04_What_parts_of_the_building_are2nd_floor").astype(int) +
    flag("SAL04_What_parts_of_the_building_are3rd_floor").astype(int)
)
lvl10 = pd.Series(np.nan, index=df_raw.index)
lvl10[nf_count == 1]         = 5  # 1 floor  → High vulnerability
lvl10[nf_count.isin([2, 3])] = 1  # 2–3 floors → Low vulnerability

# ── FVI11 : Width-to-Length Ratio ─────────────────────────────────────────────
length  = to_num(df_raw["SAM01_What_is_the_length_of_the_core_house"])
width   = to_num(df_raw["SAM02_What_is_the_width_of_the_core_house"])
valid_w = width.notna() & (width > 0)
lw      = pd.Series(np.nan, index=df_raw.index)
lw[valid_w] = length[valid_w] / width[valid_w]
lvl11 = pd.Series(np.nan, index=df_raw.index)
lvl11[lw <  3] = 1
lvl11[lw >= 3] = 5

# ── Assemble FVI indicator DataFrame ─────────────────────────────────────────
df_fvi = pd.DataFrame({
    "FVI2_RoofMat":   lvl2,
    "FVI3_RoofConn":  lvl3,
    "FVI4_WallMat":   lvl4,
    "FVI5_Elev":      lvl5,
    "FVI6_Geom":      lvl6,
    "FVI7_Overhang":  lvl7,
    "FVI8_Apron":     lvl8,
    "FVI9_Drain":     lvl9,
    "FVI10_Floors":   lvl10,
    "FVI11_LWratio":  lvl11,
})

print(f"\nFVI indicator matrix shape: {df_fvi.shape}")
print(f"Total cases: {len(df_fvi):,}")


# =============================================================================
# STEP 1 — VISUALISE THE MISSING DATA PATTERN
# =============================================================================
#
# Before any formal testing, inspect the structure of missingness visually.
#
# Two outputs are produced:
#   (a) Missing rate bar chart — shows the % missing per indicator.
#       High rates (>20%) may affect imputation reliability.
#   (b) Missing data heatmap — each column is an indicator, each row is a
#       building. Blue cells are missing. Look for:
#         - Random scatter  → consistent with MCAR
#         - Horizontal bands → whole indicators missing (MAR or MNAR)
#         - Vertical bands  → whole buildings missing (MAR or MNAR)
#
# Reference: van Buuren (2018), Chapter 2. https://stefvanbuuren.name/fimd/
# =============================================================================

print("\n" + "=" * 65)
print("STEP 1 — MISSING DATA PATTERN VISUALISATION")
print("=" * 65)

# ── (a) Missing rate per indicator ────────────────────────────────────────────
missing_pct = df_fvi.isna().mean() * 100

print("\nMissing data rate per FVI indicator (%):")
print(missing_pct.round(2).sort_values(ascending=False).to_string())

fig1, ax1 = plt.subplots(figsize=(10, 5))
missing_pct.sort_values(ascending=False).plot(
    kind="bar", ax=ax1, color="steelblue", edgecolor="white"
)
ax1.set_title(
    "Step 1a — Missing Data Rate per FVI Indicator\n"
    "Nepal Lumbini Survey (n=2,993)",
    fontsize=13
)
ax1.set_xlabel("FVI Indicator", fontsize=11)
ax1.set_ylabel("Missing (%)", fontsize=11)
ax1.axhline(20, color="orange", linestyle="--", linewidth=1,
            label="20% threshold")
ax1.axhline(40, color="red",    linestyle="--", linewidth=1,
            label="40% threshold (imputation reliability limit)")
ax1.legend(fontsize=9)
ax1.tick_params(axis="x", rotation=45)
plt.tight_layout()
plot1_path = OUTPUT_DIR / "step1a_missing_rate_per_indicator.png"
fig1.savefig(plot1_path, dpi=150)
print(f"\nPlot saved: {plot1_path}")
plt.close(fig1)

# ── (b) Missing data heatmap ──────────────────────────────────────────────────
# Binary matrix: 1 = missing (blue), 0 = observed (white).
# A random scatter pattern is consistent with MCAR.
miss_matrix = df_fvi.isna().astype(int)

fig2, ax2 = plt.subplots(figsize=(12, 4))
sns.heatmap(
    miss_matrix.T,          # Transpose: indicators as rows, buildings as columns
    cbar=False,
    cmap="Blues",
    ax=ax2,
    xticklabels=False       # Too many buildings (2,993) to label individually
)
ax2.set_title(
    "Step 1b — Missing Data Pattern Heatmap\n"
    "Blue = Missing | Each column = one building (n=2,993)",
    fontsize=13
)
ax2.set_ylabel("FVI Indicator", fontsize=11)
ax2.set_xlabel("Buildings (ordered by survey row)", fontsize=11)
plt.tight_layout()
plot2_path = OUTPUT_DIR / "step1b_missing_data_heatmap.png"
fig2.savefig(plot2_path, dpi=150)
print(f"Plot saved: {plot2_path}")
plt.close(fig2)


# =============================================================================
# STEP 2 — TEST FOR MCAR USING LITTLE'S TEST
# =============================================================================
#
# Little's MCAR test assesses whether missingness is completely random.
#
# H0 (null hypothesis)  : Data is MCAR
# H1 (alt. hypothesis)  : Data is MAR or MNAR
#
# A p-value < 0.05 rejects MCAR. This does NOT tell you whether data is MAR
# or MNAR — only that it is not purely random. You need Step 3 for that.
#
# Implementation uses pyampute. Install with:  pip install pyampute
#
# Reference: Little, R.J.A. (1988). JASA, 83(404), 1198-1202.
#            https://doi.org/10.1080/01621459.1988.10478722
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2 — LITTLE'S MCAR TEST")
print("=" * 65)
print(
    "\nH0: Data is Missing Completely At Random (MCAR)"
    "\nH1: Data is MAR or MNAR"
    "\nSignificance threshold: p < 0.05 → reject MCAR\n"
)

try:
    from pyampute.exploration.mcar_statistical_tests import MCARTest

    # Little's test requires at least partial overlap between cases —
    # drop rows where ALL indicators are missing (fully unscored buildings).
    df_fvi_test = df_fvi.dropna(how="all")
    print(f"Rows used in Little's test (at least one indicator scored): "
          f"{len(df_fvi_test):,}")

    mcar_test = MCARTest(method="little")
    p_value   = mcar_test.little_mcar_test(df_fvi_test)

    print(f"\nLittle's MCAR test p-value: {p_value:.4f}")

    if p_value > 0.05:
        print(
            "\nResult: FAIL TO REJECT H0"
            "\nInterpretation: Data is consistent with MCAR."
            "\nMissingness appears random — imputation is appropriate."
        )
    else:
        print(
            "\nResult: REJECT H0 (p < 0.05)"
            "\nInterpretation: Data is NOT missing completely at random."
            "\nProceed to Step 3 to distinguish MAR from MNAR."
        )

except ImportError:
    print(
        "pyampute is not installed. Install it with:"
        "\n    pip install pyampute"
        "\nSkipping Little's test. Proceeding to Step 3."
    )
    p_value = None

except Exception as e:
    print(f"Little's test could not be completed: {e}")
    print("This can occur when too few cases have complete data across all indicators.")
    print("Proceeding to Step 3.")
    p_value = None


# =============================================================================
# STEP 3 — DISTINGUISH MAR FROM MNAR USING LOGISTIC REGRESSION
# =============================================================================
#
# For each FVI indicator, we:
#   1. Create a binary missingness indicator (1 = missing, 0 = observed).
#   2. Use all OTHER observed FVI indicators as predictors.
#   3. Fit a logistic regression.
#   4. Interpret the result:
#
#       High accuracy + large coefficients
#           → Missingness is predicted by other observed variables
#           → Consistent with MAR
#           → Imputation using other indicators is justified
#
#       Low accuracy + small coefficients
#           → Other observed variables cannot explain missingness
#           → MNAR is plausible if domain knowledge supports it
#           → Flag this indicator; report as a limitation
#
# NOTE: This test cannot prove MNAR. MNAR requires domain knowledge —
# discuss findings with your Nepal and ITC experts.
#
# Reference: Sterne et al. (2009). BMJ, 338, b2393.
#            https://doi.org/10.1136/bmj.b2393
# =============================================================================

print("\n" + "=" * 65)
print("STEP 3 — MAR vs MNAR DIAGNOSTIC (LOGISTIC REGRESSION)")
print("=" * 65)
print(
    "\nFor each indicator, logistic regression tests whether missingness"
    "\ncan be predicted from the other observed FVI indicators."
    "\n"
    "\nInterpretation guide:"
    "\n  Accuracy >> 50% + large |coef|  → MAR likely"
    "\n  Accuracy ~  50%  + small |coef|  → MNAR possible; consult experts"
    "\n"
    "\nNote: Predictors are mean-filled solely for this diagnostic."
    "\nThis is NOT the final imputation step.\n"
)

fvi_cols     = df_fvi.columns.tolist()
mar_results  = {}   # Store summary for final report

for target in fvi_cols:

    # ── Missingness indicator for the target variable ─────────────────────────
    y = df_fvi[target].isna().astype(int)

    # Skip if the indicator has no missing values — nothing to test
    n_missing = y.sum()
    if n_missing == 0:
        print(f"\n{target}: No missing values — skipping.")
        mar_results[target] = {
            "n_missing": 0,
            "pct_missing": 0.0,
            "accuracy": None,
            "interpretation": "No missing values"
        }
        continue

    # Skip if the indicator is entirely missing — cannot test
    if n_missing == len(y):
        print(f"\n{target}: Entirely missing — cannot test.")
        mar_results[target] = {
            "n_missing": n_missing,
            "pct_missing": 100.0,
            "accuracy": None,
            "interpretation": "Entirely missing — cannot assess"
        }
        continue

    # ── Predictors: all OTHER FVI indicators, mean-filled for this test ───────
    predictor_cols = [c for c in fvi_cols if c != target]
    X = df_fvi[predictor_cols].copy()
    X = X.fillna(X.mean())   # Temporary fill — diagnostic only

    # ── Standardise predictors ────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Fit logistic regression ───────────────────────────────────────────────
    # max_iter=1000 to allow convergence; class_weight='balanced' accounts
    # for imbalance between missing and observed cases.
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )
    lr.fit(X_scaled, y)
    accuracy = lr.score(X_scaled, y)

    # ── Coefficient summary ───────────────────────────────────────────────────
    coef_df = pd.DataFrame({
        "Predictor":   predictor_cols,
        "Coefficient": lr.coef_[0]
    }).sort_values("Coefficient", key=abs, ascending=False)

    # ── Interpretation ────────────────────────────────────────────────────────
    pct_missing = n_missing / len(y) * 100
    if accuracy >= 0.65:
        interpretation = "MAR likely — missingness predicted by other indicators"
    elif accuracy >= 0.55:
        interpretation = "Uncertain — weak MAR signal; consult domain experts"
    else:
        interpretation = "MNAR possible — consult domain experts"

    mar_results[target] = {
        "n_missing":      n_missing,
        "pct_missing":    round(pct_missing, 2),
        "accuracy":       round(accuracy, 3),
        "interpretation": interpretation
    }

    # ── Print per-indicator results ───────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"Target variable : {target}")
    print(f"Missing cases   : {n_missing:,} / {len(y):,} ({pct_missing:.1f}%)")
    print(f"Model accuracy  : {accuracy:.3f}  (baseline = "
          f"{max(y.mean(), 1 - y.mean()):.3f})")
    print(f"Interpretation  : {interpretation}")
    print(f"\nTop predictors of missingness (|coef| ranked):")
    print(coef_df.head(5).to_string(index=False))


# =============================================================================
# STEP 3 — VISUALISATION: LOGISTIC REGRESSION ACCURACY SUMMARY
# =============================================================================
# Bar chart comparing model accuracy to the no-information baseline for each
# indicator. Bars clearly above their baseline suggest MAR.
# =============================================================================

# Collect results for indicators with a valid accuracy score
plot_data = {
    k: v for k, v in mar_results.items()
    if v["accuracy"] is not None
}

if plot_data:
    labels       = list(plot_data.keys())
    accuracies   = [plot_data[k]["accuracy"] for k in labels]
    pct_missing  = [plot_data[k]["pct_missing"] for k in labels]

    fig3, ax3 = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(labels))
    width = 0.5

    bars = ax3.bar(x, accuracies, width, color="steelblue",
                   edgecolor="white", label="Model accuracy")

    # Add a 0.5 reference line (random chance baseline)
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=1,
                label="Chance baseline (0.50)")
    ax3.axhline(0.65, color="orange", linestyle="--", linewidth=1,
                label="MAR signal threshold (0.65)")

    # Annotate bars with % missing
    for bar, pct in zip(bars, pct_missing):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{pct:.1f}%\nmissing",
            ha="center", va="bottom", fontsize=7.5, color="dimgray"
        )

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha="right")
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Logistic Regression Accuracy", fontsize=11)
    ax3.set_title(
        "Step 3 — MAR vs MNAR Diagnostic\n"
        "Accuracy of predicting missingness from other FVI indicators",
        fontsize=13
    )
    ax3.legend(fontsize=9)
    plt.tight_layout()
    plot3_path = OUTPUT_DIR / "step3_mar_mnar_diagnostic.png"
    fig3.savefig(plot3_path, dpi=150)
    print(f"\n\nPlot saved: {plot3_path}")
    plt.close(fig3)

# =============================================================================
# FVI MISSING DATA DIAGNOSTIC — STEP 4: MNAR SENSITIVITY ANALYSIS
# =============================================================================
# Purpose:
#   Assess how sensitive the FVI composite score is to worst-case and
#   best-case assumptions about MNAR-plausible indicators. Missing values
#   for flagged indicators are imputed at score=5 (maximum vulnerability,
#   MNAR-high) and score=1 (minimum vulnerability, MNAR-low), and the
#   resulting composite distributions are compared against the current
#   pairwise-mean estimate.
#
# Methodology:
#   Pattern-mixture model approach following Little & Rubin (2002).
#   Reference: https://doi.org/10.1002/9781119013563
#
# MNAR-plausible indicators (domain reasoning):
#   FVI6_Geom      — geometry unmeasurable for heavily damaged/irregular buildings
#   FVI8_Apron     — apron absent for building types where feature does not exist
#   FVI9_Drain     — drainage absent in specific elevation bands or clusters
#   FVI11_LWratio  — footprint unmeasurable if irregular or inaccessible
#   FVI3_RoofConn  — 44.5% missing; even under MAR, imputation uncertainty
#                    warrants a sensitivity bound at this rate
#
# Inputs (expected in scope when appended to fvi_missing_data_diagnostic.py):
#   fvi_matrix   : pd.DataFrame, shape (n_buildings, n_fvi_indicators)
#                  Column names must include the FVI indicator columns listed above.
#   output_dir   : pathlib.Path, directory where figures are saved.
#
# Outputs:
#   Console table  — mean, median, SD, min, max per scenario
#   Console table  — class distribution (Low / Medium / High) per scenario
#   Figure         — overlaid histogram of composite FVI under three scenarios
#                    saved to output_dir / "step4_fvi_mnar_sensitivity.png"
# =============================================================================
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
print()
print("=" * 65)
print("STEP 4 — FVI MNAR SENSITIVITY ANALYSIS")
print("=" * 65)
 
# ---------------------------------------------------------------------------
# 4a. Define MNAR-plausible indicators and scoring bounds
# ---------------------------------------------------------------------------
 
# Indicators flagged as MNAR-plausible in Step 3 (accuracy ≤ 0.50,
# i.e. missingness not predictable from observed FVI variables) plus
# FVI3_RoofConn included for its extreme missing rate (44.5%).
MNAR_INDICATORS = [
    "FVI3_RoofConn",    # 44.5% missing — imputation reliability limit exceeded
    "FVI6_Geom",        # 3.0% missing  — MNAR possible (accuracy=0.472)
    "FVI8_Apron",       # 9.4% missing  — MNAR possible (accuracy=0.530)
    "FVI9_Drain",       # 3.4% missing  — MNAR possible (accuracy=0.441)
    "FVI11_LWratio",    # 4.3% missing  — MNAR possible (accuracy=0.455)
]
 
# FVI scores are on a 1–5 ordinal scale (1=lowest vulnerability, 5=highest).
SCORE_MIN = 1   # Best-case imputation (MNAR-low scenario)
SCORE_MAX = 5   # Worst-case imputation (MNAR-high scenario)
 
# Confirm all MNAR indicators are present in the matrix.
missing_cols = [c for c in MNAR_INDICATORS if c not in fvi_matrix.columns]
if missing_cols:
    raise ValueError(
        f"MNAR indicators not found in fvi_matrix: {missing_cols}\n"
        f"Available columns: {list(fvi_matrix.columns)}"
    )
 
# Report which indicators are being analysed and their observed missing rates.
print()
print("Indicators included in MNAR sensitivity analysis:")
for col in MNAR_INDICATORS:
    n_miss = fvi_matrix[col].isna().sum()
    pct = n_miss / len(fvi_matrix) * 100
    print(f"  {col:<20} n_missing={n_miss:>4}  ({pct:.1f}%)")
 
# ---------------------------------------------------------------------------
# 4b. Build three composite FVI score vectors
# ---------------------------------------------------------------------------
# Strategy:
#   Scenario 1 (current)   — pairwise row mean across all observed FVI scores.
#   Scenario 2 (MNAR-high) — MNAR indicators filled with SCORE_MAX=5, then
#                            pairwise row mean (non-MNAR indicators unchanged).
#   Scenario 3 (MNAR-low)  — MNAR indicators filled with SCORE_MIN=1, then
#                            pairwise row mean.
#
# Row mean is computed with min_count=1 so that rows with ALL values missing
# remain NaN rather than silently returning 0. This preserves any
# fully-missing buildings as NaN in all three scenarios.
 
# Scenario 1: current pairwise mean (no imputation).
composite_current = fvi_matrix.mean(axis=1, skipna=True)
 
# Scenario 2: worst-case MNAR — fill flagged indicators with 5.
fvi_mnar_high = fvi_matrix.copy()
for col in MNAR_INDICATORS:
    fvi_mnar_high[col] = fvi_mnar_high[col].fillna(SCORE_MAX)
composite_mnar_high = fvi_mnar_high.mean(axis=1, skipna=True)
 
# Scenario 3: best-case MNAR — fill flagged indicators with 1.
fvi_mnar_low = fvi_matrix.copy()
for col in MNAR_INDICATORS:
    fvi_mnar_low[col] = fvi_mnar_low[col].fillna(SCORE_MIN)
composite_mnar_low = fvi_mnar_low.mean(axis=1, skipna=True)
 
# ---------------------------------------------------------------------------
# 4c. Descriptive statistics per scenario
# ---------------------------------------------------------------------------
 
def _describe(series: pd.Series, label: str) -> dict:
    """Return a dict of summary statistics for a composite score series."""
    return {
        "label":  label,
        "mean":   series.mean(),
        "median": series.median(),
        "sd":     series.std(),
        "min":    series.min(),
        "max":    series.max(),
    }
 
scenarios = [
    _describe(composite_current,   "Pairwise mean (current)"),
    _describe(composite_mnar_high, "MNAR-high imputation (=5)"),
    _describe(composite_mnar_low,  "MNAR-low  imputation (=1)"),
]
 
print()
print(
    f"{'Scenario':<35} {'Mean':>6} {'Median':>8} {'SD':>6} "
    f"{'Min':>6} {'Max':>6}"
)
print("-" * 68)
for s in scenarios:
    print(
        f"  {s['label']:<33} {s['mean']:>6.3f} {s['median']:>8.3f} "
        f"{s['sd']:>6.3f} {s['min']:>6.3f} {s['max']:>6.3f}"
    )
 
# ---------------------------------------------------------------------------
# 4d. Vulnerability class distribution per scenario
# ---------------------------------------------------------------------------
# Classification thresholds mirror those used in the EVI pipeline:
#   Low    : composite score < 2.33  (< (1+2+2+1)/3 = lower third of 1–5 range)
#   Medium : 2.33 ≤ score < 3.67
#   High   : score ≥ 3.67
#
# Adjust these thresholds if your pipeline uses different cut-points.
 
LOW_THRESH  = 7 / 3   # ≈ 2.333
HIGH_THRESH = 11 / 3  # ≈ 3.667
 
def _class_counts(series: pd.Series) -> dict:
    """Return counts and percentages for Low / Medium / High classes."""
    n = len(series.dropna())
    low    = (series < LOW_THRESH).sum()
    high   = (series >= HIGH_THRESH).sum()
    medium = n - low - high
    return {
        "low":    low,
        "medium": medium,
        "high":   high,
        "n":      n,
        "low_pct":    low    / n * 100,
        "medium_pct": medium / n * 100,
        "high_pct":   high   / n * 100,
    }
 
counts = {
    "current":   _class_counts(composite_current),
    "mnar_high": _class_counts(composite_mnar_high),
    "mnar_low":  _class_counts(composite_mnar_low),
}
 
print()
print("Class distribution under each scenario:")
for cls in ("low", "medium", "high"):
    label = cls.capitalize()
    c = counts["current"]
    h = counts["mnar_high"]
    l = counts["mnar_low"]
    print(
        f"  {label:<8}: "
        f"Pairwise={c[cls]:>4} ({c[cls+'_pct']:.1f}%)  "
        f"MNAR-high={h[cls]:>4} ({h[cls+'_pct']:.1f}%)  "
        f"MNAR-low={l[cls]:>4} ({l[cls+'_pct']:.1f}%)"
    )
 
# ---------------------------------------------------------------------------
# 4e. Visualisation — overlaid histogram
# ---------------------------------------------------------------------------
# Design mirrors EVI Figure 4 for visual consistency across the thesis.
# Three semi-transparent histograms are overlaid so divergence is visible.
 
fig, ax = plt.subplots(figsize=(10, 5))
 
BINS = np.arange(1.0, 5.25, 0.2)   # 0.2-wide bins spanning the 1–5 score range
 
# Blue  = current pairwise mean
ax.hist(
    composite_current.dropna(),
    bins=BINS, alpha=0.55, color="#5B8DB8", label="Pairwise mean (current)",
    edgecolor="white", linewidth=0.4,
)
# Red/pink = worst-case MNAR (score=5)
ax.hist(
    composite_mnar_high.dropna(),
    bins=BINS, alpha=0.50, color="#D4786A", label="MNAR-high imputed (score=5)",
    edgecolor="white", linewidth=0.4,
)
# Green = best-case MNAR (score=1)
ax.hist(
    composite_mnar_low.dropna(),
    bins=BINS, alpha=0.45, color="#7AB87A", label="MNAR-low  imputed (score=1)",
    edgecolor="white", linewidth=0.4,
)
 
# Vertical reference lines for class thresholds.
ax.axvline(LOW_THRESH,  color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
ax.axvline(HIGH_THRESH, color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(LOW_THRESH  + 0.04, ax.get_ylim()[1] * 0.97,
        "Low | Medium", fontsize=8, va="top", color="#555")
ax.text(HIGH_THRESH + 0.04, ax.get_ylim()[1] * 0.97,
        "Medium | High", fontsize=8, va="top", color="#555")
 
ax.set_xlabel("FVI Composite Score (1–5)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(
    "Step 4 — MNAR Sensitivity: FVI Composite Under Different\n"
    "Missing Data Assumptions\n"
    "(large divergence between curves = high MNAR sensitivity)",
    fontsize=11,
)
ax.legend(fontsize=9, framealpha=0.85)
ax.set_xlim(1.0, 5.2)
fig.tight_layout()
 
# Save figure to the shared output directory used by earlier steps.
fig4_fvi_path = output_dir / "step4_fvi_mnar_sensitivity.png"
fig.savefig(fig4_fvi_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print()
print(f"Saved: {fig4_fvi_path}")
# =============================================================================
# FINAL SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 65)
print("DIAGNOSTIC SUMMARY — MISSING DATA MECHANISM")
print("=" * 65)
print(f"\n{'Indicator':<20} {'Missing n':>10} {'Missing %':>10} "
      f"{'Accuracy':>10}  Interpretation")
print("─" * 90)

for k, v in mar_results.items():
    acc_str = f"{v['accuracy']:.3f}" if v["accuracy"] is not None else "  N/A "
    print(
        f"{k:<20} {v['n_missing']:>10,} {v['pct_missing']:>9.1f}% "
        f"{acc_str:>10}  {v['interpretation']}"
    )

print("\n" + "=" * 65)
print("RECOMMENDED NEXT STEPS")
print("=" * 65)
print("""
  1. Review the heatmap (Step 1b) for structural patterns in missingness.
     Random scatter supports MCAR; banding suggests MAR or MNAR.

  2. If Little's test (Step 2) rejects MCAR (p < 0.05), proceed with
     Multiple Imputation by Chained Equations (MICE).

  3. For indicators flagged as 'MNAR possible' in Step 3, consult
     your Nepal and ITC domain experts. Ask specifically:
       - Were certain building features inaccessible because of damage?
       - Were any questions systematically skipped for building types?
     Document any MNAR findings as limitations in your methodology.

  4. Regardless of mechanism, report missing data rates per indicator
     in your methods section (FAIR principle — transparency).

References
----------
  Little (1988)  : https://doi.org/10.1080/01621459.1988.10478722
  Sterne (2009)  : https://doi.org/10.1136/bmj.b2393
  van Buuren (2018): https://stefvanbuuren.name/fimd/
""")
