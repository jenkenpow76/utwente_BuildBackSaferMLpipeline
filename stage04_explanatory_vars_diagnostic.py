"""
explanatory_vars_diagnostic.py
================================
Missing Data Diagnostic — MAO and Socioeconomic Explanatory Variables
Nepal Lumbini Survey Dataset

PURPOSE
-------
This script diagnoses missing data in the MAO (Motivation, Ability,
Opportunity) and socioeconomic predictor variables that are used in
Stages 3–6 of the pipeline (bivariate screening, stepwise regression,
mixed-effects modelling).

It is DISTINCT from missing_data_diagnostic.py, which covers the EVI and
FVI structural indicator columns. The explanatory variables are collected
through structured interviews (not physical building inspection), so the
sources and patterns of missingness differ.

WHY THIS IS A SEPARATE SCRIPT
-------------------------------
The EVI/FVI diagnostic asks: "Why are physical building measurements
missing?" The explanatory variable diagnostic asks: "Why are interview
responses missing, and does it bias the n=1,235 analysis sample?"

These are different questions with different methodological consequences:
  - Physical measurement missingness → MCAR/MAR/MNAR at the building level
  - Interview response missingness   → systematic non-response, question
    skip patterns, enumerator interview style, or translation issues

KEY OBSERVATIONS FROM EXPLORATORY ANALYSIS
-------------------------------------------
  Acc_Perc_Neg_expression : 87.4% overall missing
    enumerator_3 (samjhana): 99.5%   enumerator_11 (prahbat): 20.3%
    → Near-total missingness for most enumerators; likely a question
      added/removed mid-fieldwork or systematically skipped

  App_Perc_Pos_expression : 52.2% overall missing
    enumerator_3 (samjhana): 90.4%   enumerator_11 (prahbat): 17.7%
    → Very strong enumerator effect (72.7pp range)

  AB_Financial_capacity_Neg : 19.6% overall missing
    enumerator_9 (adarsha): 61.5%   enumerator_3 (samjhana): 0.0%
    → Extreme enumerator-level variation

  HC003 (age), SE01_edu, SE04 (occupation) : < 1% missing
    → Near-complete; no imputation needed

FIVE DIAGNOSTIC STEPS (mirrors missing_data_diagnostic.py)
------------------------------------------------------------
  Step 1 — Visualise missingness patterns
  Step 2 — Analysis sample representativeness test
  Step 3 — MAR test via logistic regression (cross-predictor, AUC)
  Step 4 — Surveyor / community / temporal clustering
  Step 5 — Final decision table (impute / delete / flag as limitation)

NOTE ON STEP 2
--------------
Step 2 here replaces Little's MCAR test (which is for outcome indicators).
For explanatory variables, the relevant question is not MCAR vs MAR per se,
but whether the n=1,235 complete-case analysis sample (after joint pairwise
deletion across all predictors) is representative of the full n=2,993.

This is tested by comparing retained vs dropped cases on key background
variables (age, household size, community, education). A significant
difference means pairwise deletion introduces selection bias into the
regression estimates.

OUTPUTS — 7 figures
--------------------
  fig_exp01_missing_rate.png          Per-predictor missing rate bar chart
  fig_exp02_row_heatmap.png           Row-level missingness heatmap
  fig_exp03_cooccurrence.png          Co-occurrence heatmap
  fig_exp04_mar_auc.png               MAR logistic regression AUC
  fig_exp05_surveyor_clustering.png   Missing % by enumerator
  fig_exp06_community_clustering.png  Missing % by VDC/ward
  fig_exp07_temporal_clustering.png   Daily missing rate over time
  fig_exp08_sample_comparison.png     Retained vs dropped comparison

DEPENDENCIES
------------
    pip install pandas numpy matplotlib seaborn scipy scikit-learn

REFERENCES
----------
Little, R.J.A. & Rubin, D.B. (2002). Statistical Analysis with Missing
  Data (2nd ed.). Wiley. https://doi.org/10.1002/9781119013563

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
  CRC Press. https://stefvanbuuren.name/fimd/

Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
  epidemiological and clinical research. BMJ, 338, b2393.
  https://doi.org/10.1136/bmj.b2393

Groves, R.M. (2006). Nonresponse rates and nonresponse bias in
  household surveys. Public Opinion Quarterly, 70(5), 646-675.
  https://doi.org/10.1093/poq/nfl033

West, B.T. & Olson, K. (2010). How much of interviewer variance is
  really nonresponse error variance? Public Opinion Quarterly, 74(5),
  1004-1026. https://doi.org/10.1093/poq/nfq061
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# FILE PATHS
# =============================================================================

INPUT_CSV  = Path(__file__).resolve().parent / "20263003_Nepal_Lumbini_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "explanatory_vars_diagnostic_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# METADATA COLUMNS
# =============================================================================

SURVEYOR_COL  = "GD001_Name_of_the_surveyor"
COMMUNITY_COL = "GD005_VDC_name"
# START_COL: try "start" first (standard ODK column), then known alternatives.
# If none are found, timestamp-dependent steps are skipped gracefully.
_START_CANDIDATES = ["start", "starttime", "start_time",
                     "@_submission_time", "submission_time", "SubmissionDate"]
START_COL = next((c for c in _START_CANDIDATES if c in
                  pd.read_csv(INPUT_CSV, nrows=0,
                              encoding="utf-8-sig").columns), None)

# =============================================================================
# PREDICTOR GROUPS
# =============================================================================
# All candidate predictors used in bivariate screening (Stage 3) and
# regression (Stages 4 and 6). Grouped for interpretability.
#
# Source: pipeline_config.py — FVI_CANDIDATES and EVI_CANDIDATES
# =============================================================================

# Individual Model predictors — the final retained set after stepwise
# NOTE: These local lists are SNAPSHOTS from a previous pipeline run, used
# only to colour-code the predictor missingness diagnostic plot. They are
# NOT the source of truth for the final Individual Model — that role belongs
# to pipeline_config.FVI_IM_PREDICTORS and EVI_IM_PREDICTORS, which are
# derived at runtime from the stage13/18 stepwise output CSVs.
# After a fresh stepwise re-run, update these lists by hand or leave them
# stale — the consequence is only that the diagnostic plot's IM-marker
# colour may not match the latest stepwise output.
# Used to define the "analysis sample" in Stage 4 / Stage 6
FVI_IM_PREDICTORS = [
    "AB_Location_Pos",
    "App_Perc_Pos_expression",
    "SE04_day_labour",
    "SE01_edu_no_education",
    "SE01_edu_high_school",
    "SE01_edu_can_read_write",
    "SE01_edu_university",
    "AB_Financial_capacity_Neg",
    "DE12_shelter_score",
    "OP_Training_Pos",
]

EVI_IM_PREDICTORS = [
    "SE01_edu_university",
    "AB_Time_Neg",
    "AB_Selfefficacy_Neg",
    "PP01_rebuild_repair_house_001totally_damaged",
    "AB_Location_Pos",
    "AB_Financial_capacity_Neg",
    "AB_Location_Neg",
]

# Full candidate pool — all variables screened in Stage 3
ALL_CANDIDATES = {
    # Motivation
    "Uti_Perc_Pos_expression":       "Utility perception (+)",
    "Uti_Perc_Neg_expression":       "Utility perception (−)",
    "App_Perc_Pos_expression":       "Applicability perc. (+)",
    "App_Perc_Neg_expression":       "Applicability perc. (−)",
    "Acc_Perc_Pos_expression":       "Acceptability perc. (+)",
    "Acc_Perc_Neg_expression":       "Acceptability perc. (−)",
    # Ability
    "AB_Selfefficacy_Pos":           "Self-efficacy (+)",
    "AB_Selfefficacy_Neg":           "Self-efficacy (−)",
    "AB_Physical_capacity_Pos":      "Physical capacity (+)",
    "AB_Physical_capacity_Neg":      "Physical capacity (−)",
    "AB_Financial_capacity_Pos":     "Financial capacity (+)",
    "AB_Financial_capacity_Neg":     "Financial capacity (−)",
    "AB_Location_Pos":               "Location suitability (+)",
    "AB_Location_Neg":               "Location suitability (−)",
    "AB_Time_Pos":                   "Time available (+)",
    "AB_Time_Neg":                   "Time available (−)",
    # Opportunity
    "OP_Training_Pos":               "Training access (+)",
    "OP_Training_Neg":               "Training access (−)",
    "OP_Manpower_Pos":               "Manpower (+)",
    "OP_Manpower_Neg":               "Manpower (−)",
    "OP_Materials_Pos":              "Materials (+)",
    "OP_Materials_Neg":              "Materials (−)",
    "OP_Location_Pos":               "Location externally (+)",
    "OP_Location_Neg":               "Location externally (−)",
    "OP_Funding_Pos":                "External funding (+)",
    "OP_Funding_Neg":                "External funding (−)",
    # Experience / preparedness
    "DE01_earthquake":               "EQ: major hazard",
    "DE02_earthquake":               "EQ: minor hazard",
    "DE12_shelter_score":            "Shelter/evacuation score",
    "PP01_rebuild_repair_house_001totally_damaged":   "EQ rebuild — totally damaged",
    "PP01_rebuild_repair_house_002partially_damaged": "EQ rebuild — partially damaged",
    "PP02_rebuild_repair_house_001yes__minor_damage": "Flood rebuild — minor damage",
    "PP02_rebuild_repair_house_002no":                "Flood rebuild — no damage",
    # Socioeconomic
    "HC003_Respondent_s_age":        "Age",
    "HC004_amount_people_household": "Household size",
    "SE01_edu_no_education":         "Edu: none",
    "SE01_edu_can_read_write":       "Edu: read/write",
    "SE01_edu_elementary_school":    "Edu: elementary",
    "SE01_edu_high_school":          "Edu: high school",
    "SE01_edu_university":           "Edu: university",
    "SE04_agriculture":              "Occ: agriculture",
    "SE04_day_labour":               "Occ: day labour",
    "SE04_construction_worker":      "Occ: construction",
    "SE04_education":                "Occ: education",
    "SE04_remittances":              "Income: remittances",
    "SE04_business":                 "Occ: business",
    "SE04_government_officer":       "Occ: government",
}

# Background variables for the representativeness test (Step 2)
# These should be near-complete and not part of the regression model
BACKGROUND_VARS = [
    "HC003_Respondent_s_age",
    "HC004_amount_people_household",
    "SE01_edu_no_education",
    "SE01_edu_high_school",
    "SE04_day_labour",
]

# Data quality caps — mirrors pipeline_config.py
CAP_COLS = {
    "AB_Physical_capacity_Pos": 100,
    "OP_Training_Neg": 100,
}

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)

# Replace blank/whitespace-only strings with NaN.
# infer_objects(copy=False) opts into the future pandas behaviour and
# suppresses the FutureWarning about silent downcasting after replace().
# See: https://pandas.pydata.org/docs/whatsnew/v2.1.0.html
df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)

# Parse timestamps
# ── Normalise ward name inconsistency ─────────────────────────────────────────
# GD005_VDC_name contains 'tulsipur-ward_14' (hyphen) in the raw survey CSV.
# Normalise to underscore here to match ward_information.csv and stage01/02 output.
_WARD_COL = "GD005_VDC_name"
if _WARD_COL in df.columns:
    _before = df[_WARD_COL].astype(str).eq("tulsipur-ward_14").sum()
    df[_WARD_COL] = (
        df[_WARD_COL].astype(str)
        .str.replace("tulsipur-ward_14", "tulsipur_ward_14", regex=False)
        .where(df[_WARD_COL].notna(), other=np.nan)
    )
    if _before:
        print(f"  Normalised {_before} rows: 'tulsipur-ward_14' → 'tulsipur_ward_14'")

# Parse timestamps — skip gracefully if the start column is absent.
if START_COL and START_COL in df.columns:
    df["_start_dt"]    = pd.to_datetime(df[START_COL], utc=True, errors="coerce")
    df["_survey_date"] = df["_start_dt"].dt.date
    df["_survey_week"] = df["_start_dt"].dt.isocalendar().week.astype("Int64")
else:
    print(f"  WARNING: start-time column not found (tried {_START_CANDIDATES}). ")
    print("  Temporal clustering analysis will be skipped.")
    df["_start_dt"]    = pd.NaT
    df["_survey_date"] = pd.NaT
    df["_survey_week"] = pd.array([pd.NA] * len(df), dtype="Int64")
df["_surveyor"]    = df[SURVEYOR_COL].fillna("unknown")
df["_community"]   = df[COMMUNITY_COL].fillna("unknown")

# Coerce all candidate columns to numeric; apply data quality caps
for col in ALL_CANDIDATES:
    if col in df.columns:
        s = pd.to_numeric(
            df[col].replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
            errors="coerce"
        )
        if col in CAP_COLS:
            n_cap = (s > CAP_COLS[col]).sum()
            if n_cap > 0:
                print(f"  Data quality: {col} — {n_cap} values > {CAP_COLS[col]} capped to NaN")
            s = s.where(s <= CAP_COLS[col])
        df[col] = s

print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns")
print(f"Surveyors  : {df['_surveyor'].nunique()} unique")
print(f"Communities: {df['_community'].nunique()} unique")
if df["_start_dt"].notna().any():
    print(f"Date range : {df['_start_dt'].min().date()} to {df['_start_dt'].max().date()}")
else:
    print("Date range : unknown (start column not found)")

# Build the explanatory variable DataFrame — only columns present in the data
avail_cols = {k: v for k, v in ALL_CANDIDATES.items() if k in df.columns}
df_exp = df[list(avail_cols.keys())].copy()
print(f"\nExplanatory variable matrix: {df_exp.shape[0]:,} rows x {df_exp.shape[1]} predictors")


# =============================================================================
# STEP 1 — VISUALISE MISSINGNESS PATTERNS
# =============================================================================

def step1_visualise() -> None:
    """
    Step 1 — Visualise missing data patterns across all candidate predictors.

    Three figures:
      (a) Bar chart sorted by % missing. Colour-coded: green (<5%), orange
          (5–20%), red (>20%). Threshold lines at 5%, 20%, and 50%.
          Predictors above 50% are effectively unusable without imputation.

      (b) Row-level heatmap (respondents × predictors).
          Structured vertical bands = a whole MAO domain was skipped
          (e.g., negative perception questions skipped by some enumerators).

      (c) Co-occurrence heatmap. In the MAO context, high co-occurrence
          within a domain (e.g., all Ability-negative items missing together)
          indicates an interview section was systematically skipped.
    """
    n = len(df_exp)

    # (a) Missing rate bar chart
    miss_pct    = df_exp.isna().mean() * 100
    miss_sorted = miss_pct.sort_values(ascending=False)
    labels      = [avail_cols.get(c, c) for c in miss_sorted.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    colours = ["#d62728" if p > 50 else "#ff7f0e" if p > 20
               else "#98df8a" if p > 5 else "#2ca02c"
               for p in miss_sorted.values]
    bars = ax.bar(range(len(miss_sorted)), miss_sorted.values,
                  color=colours, edgecolor="white")
    ax.axhline(5,  color="#2ca02c", linestyle=":",  linewidth=1.0, label="5%  low concern")
    ax.axhline(20, color="#ff7f0e", linestyle="--", linewidth=1.2, label="20% imputation advised")
    ax.axhline(50, color="#d62728", linestyle="--", linewidth=1.5, label="50% critical")
    ax.set_xticks(range(len(miss_sorted)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("% Missing", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(
        f"Missing Data Rate per Explanatory Variable\n"
        f"n={n:,} respondents | Green <5% missing | Orange 5–20% | Red >50%",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9)
    for bar, pct in zip(bars, miss_sorted.values):
        if pct > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=6.5, rotation=90)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_exp01_missing_rate.png", dpi=150)
    plt.close()
    print("  Saved: fig_exp01_missing_rate.png")

    # Print summary grouped by missingness severity
    print("\n  Missing rate summary:")
    for threshold, label in [(0, "< 5% (complete)"), (5, "5–20% (moderate)"),
                              (20, "20–50% (high)"), (50, "> 50% (critical)")]:
        upper = {0: 5, 5: 20, 20: 50, 50: 101}[threshold]
        cols  = [c for c, p in miss_pct.items() if threshold <= p < upper]
        if cols:
            print(f"\n    {label}:")
            for c in cols:
                print(f"      {avail_cols.get(c, c):<35}: {miss_pct[c]:.1f}%")

    # (b) Row-level heatmap — sorted by domain for interpretability
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        df_exp[miss_sorted.index].isna().astype(int).T,
        cbar=False, cmap="Blues", ax=ax, xticklabels=False,
        yticklabels=[avail_cols.get(c, c) for c in miss_sorted.index]
    )
    ax.set_title(
        "Row-Level Missingness Pattern — Explanatory Variables\n"
        "Blue = Missing | Vertical bands indicate a MAO domain was systematically skipped\n"
        "Horizontal bands indicate an enumerator consistently skipped a section",
        fontsize=11, pad=10
    )
    ax.set_xlabel(f"Respondents (n={n:,}, ordered by survey row)", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_exp02_row_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: fig_exp02_row_heatmap.png")

    # (c) Co-occurrence heatmap
    # Both sides must be float before the dot product — bool.T.dot(int)
    # produces object dtype in pandas, which matplotlib cannot render.
    miss_bool = df_exp.isna().astype(float)
    co_occur  = miss_bool.T.dot(miss_bool) / n
    co_labels = [avail_cols.get(c, c) for c in co_occur.index]

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        co_occur.values, annot=False, cmap="Blues",
        linewidths=0.2, ax=ax, vmin=0,
        vmax=max(float(co_occur.values.max()), 0.01),
        xticklabels=co_labels, yticklabels=co_labels
    )
    ax.set_title(
        "Co-occurrence of Missingness — Explanatory Variables\n"
        "Cell value = proportion of respondents where both predictors are simultaneously missing\n"
        "Dark blocks indicate entire interview sections were skipped together",
        fontsize=11, pad=10
    )
    ax.tick_params(axis="both", labelsize=7)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_exp03_cooccurrence.png", dpi=150)
    plt.close()
    print("  Saved: fig_exp03_cooccurrence.png")


# =============================================================================
# STEP 2 — ANALYSIS SAMPLE REPRESENTATIVENESS TEST
# =============================================================================

def step2_sample_representativeness() -> None:
    """
    Step 2 — Is the complete-case analysis sample representative?

    The regression stages (4 and 6) use pairwise deletion across all
    predictors simultaneously. The n=1,235 complete cases are those
    where EVERY predictor in the Individual Model has a valid value.

    This step tests whether the retained cases differ systematically
    from the dropped cases on key background variables. If they do,
    pairwise deletion introduces selection bias into the regression
    estimates — the coefficients apply to a non-representative subset.

    Test:
      - Define 'complete case' as having valid data on all FVI IM predictors
      - Compare retained (complete) vs dropped cases on background variables
      - Mann-Whitney U test for continuous variables (non-parametric)
      - Chi-square test for binary/categorical variables
      - Report effect sizes (rank-biserial r for U test)

    INTERPRETATION
    --------------
    No significant differences (p > 0.05 across all background variables):
      → Pairwise deletion is defensible. Dropped cases are similar
        to retained cases. Report n=1,235 with this justification.

    Significant differences on multiple background variables:
      → Selection bias is present. Consider MICE on predictor columns,
        or at minimum report this as a limitation with the comparison
        statistics as evidence.
    """
    print("\n  Defining complete-case sample from FVI Individual Model predictors...")

    # Identify which IM predictors are present
    im_avail = [p for p in FVI_IM_PREDICTORS if p in df.columns]
    missing_im = [p for p in FVI_IM_PREDICTORS if p not in df.columns]
    if missing_im:
        print(f"  WARNING: IM predictors not found in data: {missing_im}")

    # FVI outcome needed too
    outcome_col = "FVI_norm_1_5"
    analysis_cols = im_avail + ([outcome_col] if outcome_col in df.columns else [])

    # Complete case flag: 1 = retained in regression, 0 = dropped
    complete_mask = df[analysis_cols].notna().all(axis=1)
    n_retained = complete_mask.sum()
    n_dropped  = (~complete_mask).sum()
    print(f"  FVI IM analysis sample : n={n_retained:,} retained, "
          f"n={n_dropped:,} dropped (pairwise deletion)")

    results = []

    for col in BACKGROUND_VARS:
        if col not in df.columns:
            continue

        retained = df.loc[complete_mask,  col].dropna()
        dropped  = df.loc[~complete_mask, col].dropna()

        if len(retained) < 10 or len(dropped) < 10:
            continue

        # Choose test based on variable type
        n_unique = df[col].dropna().nunique()
        if n_unique <= 3:
            # Binary or ordinal — chi-square on 2×2 or 2×k table
            ct = pd.crosstab(complete_mask, df[col])
            try:
                chi2, p, dof, _ = stats.chi2_contingency(ct)
                test_name = "Chi-square"
                stat      = chi2
                effect    = np.sqrt(chi2 / (chi2 + len(df)))  # Cramér's V approx
            except Exception:
                continue
        else:
            # Continuous — Mann-Whitney U (non-parametric; no normality assumption)
            u_stat, p = stats.mannwhitneyu(retained, dropped, alternative="two-sided")
            test_name = "Mann-Whitney U"
            stat      = u_stat
            # Rank-biserial correlation as effect size
            n1, n2    = len(retained), len(dropped)
            effect    = 1 - (2 * u_stat) / (n1 * n2)

        label = avail_cols.get(col, col)
        sig   = ("*** p<0.001" if p < 0.001 else "** p<0.01"
                 if p < 0.01 else "* p<0.05" if p < 0.05 else "ns")
        results.append({
            "variable": label, "test": test_name, "stat": stat,
            "p": p, "effect": abs(effect), "sig": sig,
            "mean_retained": retained.mean(), "mean_dropped": dropped.mean()
        })

    if results:
        print(f"\n  Comparison: retained (n={n_retained:,}) vs dropped (n={n_dropped:,})")
        print(f"\n  {'Variable':<35} {'Test':<17} {'Stat':>8} {'p':>8} "
              f"{'Effect':>8}  {'Retained mean':>14}  {'Dropped mean':>12}  Sig")
        print("  " + "-" * 115)
        for r in results:
            print(
                f"  {r['variable']:<35} {r['test']:<17} "
                f"{r['stat']:>8.1f} {r['p']:>8.4f} "
                f"{r['effect']:>8.3f}  {r['mean_retained']:>14.3f}  "
                f"{r['mean_dropped']:>12.3f}  {r['sig']}"
            )

        # Overall recommendation
        n_sig = sum(1 for r in results if r["p"] < 0.05)
        print(f"\n  {n_sig}/{len(results)} background variables show "
              f"significant differences (p < 0.05).")
        if n_sig == 0:
            print(
                "  RESULT: Complete cases are statistically similar to dropped cases.\n"
                "  → Pairwise deletion is defensible.\n"
                "  → Report n=1,235 with this representativeness check as justification."
            )
        elif n_sig <= 2:
            print(
                "  RESULT: Minor differences detected on a subset of variables.\n"
                "  → Pairwise deletion is cautiously acceptable.\n"
                "  → Report these differences as a limitation."
            )
        else:
            print(
                "  RESULT: Multiple significant differences detected.\n"
                "  → Pairwise deletion introduces selection bias.\n"
                "  → Consider MICE on predictor columns before regression.\n"
                "  → At minimum, report these differences explicitly."
            )

        # Plot: mean comparison for continuous variables
        plot_r = [r for r in results if r["test"] == "Mann-Whitney U"]
        if plot_r:
            vars_    = [r["variable"] for r in plot_r]
            m_ret    = [r["mean_retained"] for r in plot_r]
            m_drop   = [r["mean_dropped"]  for r in plot_r]
            sig_flag = [r["sig"] != "ns"   for r in plot_r]

            x      = np.arange(len(vars_))
            width  = 0.35
            fig, ax = plt.subplots(figsize=(10, 5))
            b1 = ax.bar(x - width/2, m_ret,  width, label="Retained (complete cases)",
                        color="#4C72B0", edgecolor="white")
            b2 = ax.bar(x + width/2, m_drop, width, label="Dropped (pairwise deletion)",
                        color="#dd8452", edgecolor="white")

            # Mark significant differences with asterisks
            for i, (sig, mr, md) in enumerate(zip(sig_flag, m_ret, m_drop)):
                if sig:
                    y_top = max(mr, md) * 1.05
                    ax.text(i, y_top, "*", ha="center", fontsize=14, color="red")

            ax.set_xticks(x)
            ax.set_xticklabels(vars_, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Mean value", fontsize=11)
            ax.set_title(
                "Complete Cases vs Dropped Cases: Background Variable Comparison\n"
                f"Retained n={n_retained:,} vs Dropped n={n_dropped:,} "
                "(FVI Individual Model, pairwise deletion)\n"
                "* = significant difference (Mann-Whitney U, p < 0.05)",
                fontsize=11, pad=10
            )
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "fig_exp08_sample_comparison.png", dpi=150)
            plt.close()
            print("  Saved: fig_exp08_sample_comparison.png")


# =============================================================================
# STEP 3 — MAR TEST: LOGISTIC REGRESSION (CROSS-PREDICTOR, AUC)
# =============================================================================

def step3_mar_test() -> list[dict]:
    """
    Step 3 — MAR Test via Logistic Regression (cross-predictor).

    For each candidate predictor, logistic regression tests whether its
    missingness can be predicted from all other available predictors.

    Only predictors with > 10 missing AND > 10 observed cases are tested.
    Predictors with < 1% missing are skipped — effectively complete.

    Design: same as missing_data_diagnostic.py
      - class_weight='balanced'
      - AUC as evaluation metric
      - Cross-predictor: all other candidates as predictors

    NOTE: High AUC for MAO variables (e.g., Acc_Perc_Neg_expression)
    likely reflects that entire interview SECTIONS were skipped by
    specific enumerators — the missingness pattern is so structured
    that it is almost perfectly predictable from which other sections
    were skipped. This is MAR with enumerator as the key driver.
    """
    cols    = df_exp.columns.tolist()
    results = []

    for target in cols:
        n_missing = df_exp[target].isna().sum()
        n_obs     = df_exp[target].notna().sum()
        pct_miss  = n_missing / len(df_exp) * 100

        # Skip near-complete predictors
        if pct_miss < 1.0:
            results.append({
                "predictor": avail_cols.get(target, target), "col": target,
                "n_missing": n_missing, "pct_missing": pct_miss,
                "auc": np.nan, "signal": "Near-complete (<1% missing)"
            })
            continue
        if n_missing < 10 or n_obs < 10:
            results.append({
                "predictor": avail_cols.get(target, target), "col": target,
                "n_missing": n_missing, "pct_missing": pct_miss,
                "auc": np.nan, "signal": "Insufficient cases"
            })
            continue

        other_cols = [c for c in cols if c != target]
        X = df_exp[other_cols].fillna(df_exp[other_cols].mean())
        y = df_exp[target].isna().astype(int)

        try:
            X_scaled = StandardScaler().fit_transform(X)
            lr = LogisticRegression(class_weight="balanced",
                                    max_iter=1000, random_state=42)
            lr.fit(X_scaled, y)
            auc = roc_auc_score(y, lr.predict_proba(X_scaled)[:, 1])
        except Exception as e:
            results.append({
                "predictor": avail_cols.get(target, target), "col": target,
                "n_missing": n_missing, "pct_missing": pct_miss,
                "auc": np.nan, "signal": f"Model error: {e}"
            })
            continue

        if auc < 0.55:   signal = "MCAR consistent"
        elif auc < 0.65: signal = "Weak MAR signal"
        elif auc < 0.80: signal = "Moderate MAR signal"
        else:            signal = "Strong MAR signal"

        results.append({
            "predictor": avail_cols.get(target, target), "col": target,
            "n_missing": n_missing, "pct_missing": pct_miss,
            "auc": auc, "signal": signal
        })

    # AUC bar chart — only testable predictors
    plot_df = pd.DataFrame([r for r in results if not np.isnan(r.get("auc", np.nan))])
    if not plot_df.empty:
        plot_df = plot_df.sort_values("auc", ascending=True)
        cmap    = {"MCAR consistent": "#2ca02c", "Weak MAR signal": "#ff7f0e",
                   "Moderate MAR signal": "#d62728", "Strong MAR signal": "#8c1c1c"}
        colours = [cmap.get(s, "#aec7e8") for s in plot_df["signal"]]

        fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.35)))
        bars = ax.barh(plot_df["predictor"], plot_df["auc"],
                       color=colours, edgecolor="white")
        ax.axvline(0.50, color="grey",    linestyle=":",  linewidth=1.2,
                   label="0.50 — chance level")
        ax.axvline(0.65, color="#ff7f0e", linestyle="--", linewidth=1.2,
                   label="0.65 — weak MAR threshold")
        ax.axvline(0.80, color="#d62728", linestyle="--", linewidth=1.2,
                   label="0.80 — strong MAR threshold")
        ax.set_xlabel("AUC (ROC)", fontsize=11)
        ax.set_xlim(0.4, 1.0)
        ax.set_title(
            "MAR Test: Logistic Regression AUC per Explanatory Variable\n"
            "Predictors = all other observed candidate variables\n"
            "High AUC for MAO negatives suggests interview sections were skipped together",
            fontsize=11, pad=10
        )
        ax.legend(fontsize=9, loc="lower right")
        for bar, row in zip(bars, plot_df.itertuples()):
            ax.text(row.auc + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{row.auc:.3f}  ({row.pct_missing:.0f}% miss)",
                    va="center", fontsize=7.5)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig_exp04_mar_auc.png", dpi=150)
        plt.close()
        print("  Saved: fig_exp04_mar_auc.png")

    return results


# =============================================================================
# STEP 4 — CLUSTERING: SURVEYOR / COMMUNITY / TIME
# =============================================================================

def step4_clustering() -> None:
    """
    Step 4 — Missingness Clustering by Surveyor, Community, and Time.

    This is the most important step for explanatory variable missingness.
    The exploratory analysis revealed extreme enumerator-level variation:
      - App_Perc_Pos_expression: 17.7% (enumerator_11) vs 90.4% (enumerator_3)
      - AB_Financial_capacity_Neg: 0.0% (enumerator_3) vs 61.5% (enumerator_9)

    This pattern strongly indicates that:
      1. Specific interview sections were skipped by specific enumerators
      2. Enumerator identity is a strong MAR predictor
      3. Surveyor should be included as a predictor variable in MICE

    NOTE ON COMMUNITY VS SURVEYOR CONFOUNDING
    -------------------------------------------
    The cross-tabulation of surveyors × communities shows that most
    communities were covered by a single enumerator. This means:
      - Community-level missingness partly reflects enumerator effects
      - The two effects cannot be cleanly separated without additional data
      - Both should be included as MICE predictors to hedge against this

    TEMPORAL PATTERN
    -----------------
    If certain enumerators worked in specific time windows, temporal
    clustering will mirror surveyor clustering. Interpret Fig EXP07
    alongside Fig EXP05 — if the temporal pattern tracks with known
    enumerator start/end dates, the effect is enumerator-driven.
    """

    df_w = df_exp.copy()
    df_w["_surveyor"]    = df["_surveyor"].values
    df_w["_community"]   = df["_community"].values
    df_w["_survey_date"] = df["_survey_date"].values
    ind_cols = df_exp.columns.tolist()

    # Focus on predictors with > 5% missing for the clustering plots
    # (near-complete variables add noise without information)
    miss_pct   = df_exp.isna().mean() * 100
    plot_cols  = [c for c in ind_cols if miss_pct[c] >= 5.0]
    plot_labels = [avail_cols.get(c, c) for c in plot_cols]

    if not plot_cols:
        print("  No predictors with >=5% missing — clustering plots skipped.")
        return

    # ── 5a: Surveyor clustering heatmap ──────────────────────────────────────

    surv_miss = (
        df_w.groupby("_surveyor")[plot_cols]
        .apply(lambda g: g.isna().mean() * 100)
    )
    surv_miss.columns = plot_labels
    surv_miss.index   = (
        surv_miss.index
        .str.replace(r"enumerator_\d+___", "", regex=True)
        .str.replace("_", " ")
        .str.title()
    )

    fig, ax = plt.subplots(
        figsize=(max(12, len(plot_cols) * 0.7), max(5, len(surv_miss) * 0.55))
    )
    sns.heatmap(
        surv_miss, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.4, ax=ax, vmin=0, vmax=100,
        annot_kws={"size": 8}
    )
    ax.set_title(
        "Explanatory Variable Missingness Rate by Surveyor\n"
        "Large variation between enumerators suggests interview style drives missingness\n"
        "Dark rows indicate an enumerator systematically skipped an entire MAO domain",
        fontsize=11, pad=10
    )
    ax.set_xlabel("Predictor", fontsize=10)
    ax.set_ylabel("Enumerator", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_exp05_surveyor_clustering.png", dpi=150)
    plt.close()
    print("  Saved: fig_exp05_surveyor_clustering.png")

    # Kruskal-Wallis — surveyor effect per predictor
    print("\n  Kruskal-Wallis — predictor missingness by enumerator:")
    for col, label in zip(plot_cols, plot_labels):
        groups = [
            df_w.loc[df_w["_surveyor"] == sv, col].isna().astype(float).values
            for sv in df_w["_surveyor"].unique()
            if (df_w["_surveyor"] == sv).sum() >= 10
        ]
        if len(groups) >= 2:
            try:
                h, p = stats.kruskal(*groups)
                sig = ("*** p<0.001" if p < 0.001 else "** p<0.01"
                       if p < 0.01 else "* p<0.05" if p < 0.05 else "ns")
                print(f"    {label:<35}: H={h:.2f}  p={p:.4f}  {sig}")
            except Exception:
                pass

    # ── 5b: Community clustering heatmap ─────────────────────────────────────

    comm_miss = (
        df_w.groupby("_community")[plot_cols]
        .apply(lambda g: g.isna().mean() * 100)
    )
    comm_miss.columns = plot_labels
    comm_miss.index   = comm_miss.index.str.replace("_", " ").str.title()

    fig, ax = plt.subplots(
        figsize=(max(12, len(plot_cols) * 0.7), max(6, len(comm_miss) * 0.5))
    )
    sns.heatmap(
        comm_miss, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.4, ax=ax, vmin=0, vmax=100,
        annot_kws={"size": 8}
    )
    ax.set_title(
        "Explanatory Variable Missingness Rate by Community (VDC/Ward)\n"
        "Community and enumerator effects are partly confounded\n"
        "(most communities covered by a single enumerator)",
        fontsize=11, pad=10
    )
    ax.set_xlabel("Predictor", fontsize=10)
    ax.set_ylabel("Community (VDC/Ward)", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_exp06_community_clustering.png", dpi=150)
    plt.close()
    print("  Saved: fig_exp06_community_clustering.png")

    # ── 5c: Temporal clustering — daily missing rate ──────────────────────────

    df_w["_overall_miss"] = df_w[plot_cols].isna().mean(axis=1)
    daily = (
        df_w.groupby("_survey_date")["_overall_miss"]
        .agg(["mean", "count"])
        .reset_index()
    )
    daily.columns     = ["date", "miss_rate", "n_surveys"]
    daily["miss_pct"] = daily["miss_rate"] * 100
    daily["date"]     = pd.to_datetime(daily["date"])
    daily["roll"]     = daily["miss_pct"].rolling(7, min_periods=3).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax2.bar(daily["date"], daily["n_surveys"], alpha=0.2,
            color="#aec7e8", label="Surveys per day")
    ax2.set_ylabel("Surveys per day", fontsize=9, color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.plot(daily["date"], daily["miss_pct"], "o-",
             color="#d62728", alpha=0.5, markersize=4,
             label="Daily missing % (predictors ≥5% missing)")
    ax1.plot(daily["date"], daily["roll"],
             color="#8c1c1c", linewidth=2.5, label="7-day rolling mean")
    ax1.set_xlabel("Survey date", fontsize=11)
    ax1.set_ylabel("Average Missing Rate (%)", fontsize=11)
    ax1.set_title(
        "Explanatory Variable Missing Rate Over Survey Period\n"
        "If pattern tracks enumerator deployment dates, the mechanism is surveyor-driven\n"
        "A uniform pattern across time indicates the mechanism is not time-dependent",
        fontsize=11, pad=10
    )
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_exp07_temporal_clustering.png", dpi=150)
    plt.close()
    print("  Saved: fig_exp07_temporal_clustering.png")

    d_clean  = daily.dropna(subset=["miss_pct"])
    rho, p_r = stats.spearmanr(range(len(d_clean)), d_clean["miss_pct"])
    sig      = "significant" if p_r < 0.05 else "not significant"
    print(f"\n  Temporal trend (Spearman ρ): ρ={rho:.3f}  p={p_r:.4f}  → {sig}")


# =============================================================================
# STEP 5 — DECISION TABLE
# =============================================================================

def step5_decision_table(mar_results: list[dict]) -> None:
    """
    Step 5 — Final Decision Table.

    Combines missing rate, AUC (MAR signal), and clustering findings
    into a per-predictor decision: impute, delete, or flag as limitation.

    Decision rules:
      pct_missing < 1%                   → Complete. No action needed.
      pct_missing 1–5%, AUC < 0.65       → Low concern. Pairwise deletion OK.
      pct_missing 1–20%, AUC ≥ 0.65      → MAR. Consider MICE.
      pct_missing > 20%, any AUC          → High concern. MICE advised.
      pct_missing > 50%                   → Critical. Exclude from analysis
                                            OR impute with strong justification.
    """
    print(f"\n  {'Predictor':<35} {'% Miss':>7} {'AUC':>7}  Decision")
    print("  " + "-" * 75)
    for r in sorted(mar_results, key=lambda x: x["pct_missing"], reverse=True):
        pct = r["pct_missing"]
        auc = r.get("auc", np.nan)
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "  N/A "

        if pct < 1.0:
            decision = "✓ Complete — no action"
        elif pct > 50:
            decision = "⚠ Critical — exclude or impute with strong justification"
        elif pct > 20:
            decision = "! High — MICE imputation advised"
        elif not np.isnan(auc) and auc >= 0.65:
            decision = "~ MAR — consider MICE"
        else:
            decision = "- Low concern — pairwise deletion defensible"

        print(f"  {r['predictor']:<35} {pct:>6.1f}%  {auc_str:>7}  {decision}")


# =============================================================================
# EXECUTE ALL STEPS
# =============================================================================

print("\n" + "=" * 65)
print("EXPLANATORY VARIABLES MISSING DATA DIAGNOSTIC")
print("=" * 65)

print("\n--- Step 1: Visualise Missingness Patterns ---")
step1_visualise()

print("\n--- Step 2: Analysis Sample Representativeness ---")
step2_sample_representativeness()

print("\n--- Step 3: MAR Test (Logistic Regression, AUC) ---")
mar_results = step3_mar_test()

print("\n--- Step 4: Clustering (Surveyor / Community / Time) ---")
step4_clustering()

print("\n--- Step 5: Decision Table ---")
step5_decision_table(mar_results)

print(f"""
{'=' * 65}
MICE PREDICTOR RECOMMENDATION FOR EXPLANATORY VARIABLES
{'=' * 65}

  If MICE imputation is warranted (pct_missing > 20%, AUC > 0.65):

  Required predictors in the MICE model:
    1. All other candidate predictors (cross-predictor imputation)
    2. Surveyor ID (categorical) — confirmed strong driver from Fig EXP05
    3. Community (VDC/ward, categorical) — geographic context
    4. Survey week (numeric) — if temporal trend is significant (Fig EXP07)
    5. FVI_norm_1_5 / EVI_norm_1_5 — include the outcome variable in MICE
       (van Buuren, 2018 §6.3: the outcome must be in the imputation model)

  CRITICAL NOTE on Acc_Perc_Neg_expression (87.4% missing):
    This variable has near-total missingness for most enumerators.
    It should be EXCLUDED from the analysis entirely, not imputed.
    87% missing exceeds any reliable imputation threshold. Document
    this exclusion in your methods section.

  FAIR REPORTING REQUIREMENT:
    Report n_retained and n_dropped from Step 2.
    Report comparison statistics (Mann-Whitney U / chi-square).
    Report whether differences were significant.
    State whether pairwise deletion was used or MICE applied.

REFERENCES
----------
  Little & Rubin (2002): https://doi.org/10.1002/9781119013563
  van Buuren (2018)     : https://stefvanbuuren.name/fimd/
  Sterne (2009)         : https://doi.org/10.1136/bmj.b2393
  Groves (2006)         : https://doi.org/10.1093/poq/nfl033
  West & Olson (2010)   : https://doi.org/10.1093/poq/nfq061
""")

print(f"All figures saved to: {OUTPUT_DIR}")
