"""
stage05b_mao_imputation.py
===========================
Stage 05b | Pre-Processing — MICE Imputation of MAO Composite Scores
Nepal Lumbini Survey Dataset

PURPOSE
-------
Imputes missing values in MAO (Motivation, Ability, Opportunity) and
selected socioeconomic predictor variables using Multiple Imputation by
Chained Equations (MICE). Produces a single merged analysis-ready CSV
that all regression stages (08, 10, 13, 16) read from.

WHY COMPOSITE-LEVEL IMPUTATION (NOT RAW ITEMS)
-----------------------------------------------
The MAO composites are proportional scores computed as:
    composite = SUM(relevant items) / SUM(ALL items answered in block) × 100

The denominator is a variable-sized pool of shared binary dummy columns
(PP08, PP10, PP11, KN03, AT04A, AT06A, AT06B, AW01, etc.) that differs
per composite. Imputing the raw binary items and recomputing would require
simultaneously imputing numerator and denominator items from a pool of
150+ columns, with the same denominator columns shared across composites.
This would produce inconsistent denominators across composites for the
same respondent — a violation of the proportional score structure.

Imputing at the composite level is therefore methodologically correct.
Composites behave as continuous 0–100 proportions in all downstream
regressions. MICE on continuous bounded outcomes is well-established.
Reference: van Buuren (2018, §4.4) — imputing derived variables with
variable denominators.

DATA QUALITY CORRECTIONS
-------------------------
Two columns have data-entry errors (values >> 100 despite 0–100 scale):
    AB_Physical_capacity_Pos : 2 values up to 1e11 → capped to NaN
    OP_Training_Neg          : 225 values up to 1e11 → capped to NaN
These are capped before imputation to prevent corruption of MICE model.

MISSINGNESS CLASSIFICATION (from explanatory_vars_diagnostic.py)
-----------------------------------------------------------------
IMPUTE (MAR, enumerator-driven, 15–30% missing):
    Uti_Perc_Neg_expression      17.7%  — enumerator block skip
    App_Perc_Neg_expression      28.7%  — enumerator block skip
    AB_Selfefficacy_Neg          17.6%  — enumerator block skip
    AB_Physical_capacity_Pos     19.3%  — enumerator block skip
    AB_Physical_capacity_Neg     19.4%  — enumerator block skip
    AB_Financial_capacity_Pos    19.0%  — enumerator block skip
    AB_Financial_capacity_Neg    19.6%  — enumerator block skip
    AB_Location_Pos              19.0%  — enumerator block skip
    AB_Time_Pos                  19.7%  — enumerator block skip
    AB_Time_Neg                  19.3%  — enumerator block skip
    OP_Materials_Pos             19.7%  — enumerator block skip
    OP_Materials_Neg             19.7%  — enumerator block skip
    OP_Location_Pos              19.1%  — enumerator block skip
    OP_Location_Neg              19.1%  — enumerator block skip
    DE12_shelter_score           16.6%  — enumerator block skip

EXCLUDE (>50% missing or structurally absent):
    Acc_Perc_Neg_expression      87.4%  — near-total; exclude from analysis
    App_Perc_Pos_expression      52.2%  — structural; enumerator-9 specific
    AB_Location_Neg              61.2%  — structural; question block absent
    OP_Manpower_Pos/Neg          92.5%  — structural; question not universal
    OP_Funding_Pos               72.1%  — structural
    OP_Funding_Neg               85.8%  — structural

COMPLETE (<5% missing — no imputation needed, included as predictors):
    Uti_Perc_Pos_expression, Acc_Perc_Pos_expression,
    AB_Selfefficacy_Pos, OP_Training_Pos, OP_Training_Neg,
    socioeconomic dummies, DE01/DE02_earthquake, DE02_flood,
    PP01_rebuild_repair_house_001totally_damaged

MICE MODEL DESIGN
-----------------
Each imputed column is predicted by:
  1. All other imputed and complete MAO/socioeconomic columns
     (cross-predictor MICE — standard chained equations)
  2. FVI_norm_1_5  — outcome must be in imputation model to prevent
     attenuation of regression coefficients (van Buuren 2018, §6.3)
  3. EVI_norm_1_5  — same rationale
  4. Surveyor ID   — confirmed strong MAR predictor (Step 5 of
     missing_data_diagnostic.py; Kruskal-Wallis p<0.001 for most vars)
  5. Community     — geographic context (partly confounded with surveyor)

INPUTS
------
    outputs/imputed_FVI_scores.csv     FVI composites (from Stage 05)
    outputs/imputed_EVI_scores.csv     EVI composites (from Stage 05)
    20263003_Nepal_Lumbini_data.csv    Raw survey (MAO columns)

OUTPUTS (all in outputs/)
------
    analysis_dataset.csv               Merged analysis-ready dataset.
                                       All 2,993 rows. Contains imputed
                                       FVI/EVI scores + imputed MAO columns
                                       + original complete columns.
    mao_imputation_log.csv             Per-column: n_missing before/after,
                                       imputation method, mean shift.
    mao_imputation_report.txt          Human-readable audit trail.

NOTE ON SINGLE IMPUTATION vs MULTIPLE IMPUTATION POOLING
---------------------------------------------------------
This script uses sklearn IterativeImputer (MICE with m=1 chain).
Single imputation produces unbiased point estimates but underestimates
standard errors in downstream regression. For final thesis submission,
multiple imputation (m=5 to m=10) with Rubin's (1987) pooling rules
is the methodologically complete approach. Single imputation is used
here for pipeline integration; report as a limitation and cite
van Buuren (2018) and Rubin (1987).

REFERENCES
----------
Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
  Wiley. https://doi.org/10.1002/9780470316696

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
  CRC Press. https://stefvanbuuren.name/fimd/
  §4.4 (derived variables), §6.3 (outcome in imputation model)

Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
  epidemiological and clinical research. BMJ, 338, b2393.
  https://doi.org/10.1136/bmj.b2393

West, B.T. & Olson, K. (2010). How much of interviewer variance is
  really nonresponse error variance? Public Opinion Quarterly, 74(5),
  1004-1026. https://doi.org/10.1093/poq/nfq061
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

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


warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

HERE       = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FVI_CSV    = OUTPUT_DIR / "imputed_FVI_scores.csv"
EVI_CSV    = OUTPUT_DIR / "imputed_EVI_scores.csv"
SURVEY_CSV = HERE / "20263003_Nepal_Lumbini_data.csv"

OUT_ANALYSIS = OUTPUT_DIR / "analysis_dataset.csv"
OUT_LOG      = OUTPUT_DIR / "mao_imputation_log.csv"
OUT_REPORT   = OUTPUT_DIR / "mao_imputation_report.txt"

SURVEYOR_COL  = _PIPELINE_CFG["SURVEYOR_COL"]
COMMUNITY_COL = _PIPELINE_CFG["COMMUNITY_COL"]

# =============================================================================
# COLUMN GROUPS
# =============================================================================

# Columns to impute — continuous 0-100 proportion scores,
# 15-30% missing, confirmed MAR from enumerator effects.
IMPUTE_COLS = [
    "Uti_Perc_Neg_expression",
    "App_Perc_Neg_expression",
    "AB_Selfefficacy_Neg",
    "AB_Physical_capacity_Pos",    # requires outlier cap first
    "AB_Physical_capacity_Neg",
    "AB_Financial_capacity_Pos",
    "AB_Financial_capacity_Neg",
    "AB_Location_Pos",
    "AB_Time_Pos",
    "AB_Time_Neg",
    "OP_Materials_Pos",
    "OP_Materials_Neg",
    "OP_Location_Pos",
    "OP_Location_Neg",
    "DE12_shelter_score",
]

# Columns that are complete or near-complete (<5% missing).
# Included as predictors in MICE but not imputed themselves.
COMPLETE_COLS = [
    "Uti_Perc_Pos_expression",
    "Acc_Perc_Pos_expression",
    "AB_Selfefficacy_Pos",
    "OP_Training_Pos",
    "OP_Training_Neg",              # requires outlier cap first
    "HC003_Respondent_s_age",
    "HC004_amount_people_household",
    "SE01_edu_no_education",
    "SE01_edu_can_read_write",
    "SE01_edu_elementary_school",
    "SE01_edu_high_school",
    "SE01_edu_university",
    "SE04_agriculture",
    "SE04_day_labour",
    "SE04_construction_worker",
    "SE04_education",
    "SE04_remittances",
    "SE04_business",
    "SE04_government_officer",
    "PP01_rebuild_repair_house_001totally_damaged",
    "PP01_rebuild_repair_house_001no",              # new EVI predictor: no rebuild (survived)
    "PP12_own_land",                                # new EVI predictor: land ownership
    "FA01_If_you_would_have_more_money_winvest_in_current_house",  # new EVI predictor
    "RT02_household_reconstruction_orien",          # new EVI predictor: orientation attended
    "DE01_earthquake",
    "DE02_earthquake",
    "DE02_flood",
]

# Columns excluded from analysis — too many missing or structural absence.
# Listed here for documentation; not loaded or imputed.
EXCLUDED_COLS = [
    "Acc_Perc_Neg_expression",   # 87.4% missing
    "App_Perc_Pos_expression",   # 52.2% missing
    "AB_Location_Neg",           # 61.2% missing
    "OP_Manpower_Pos",           # 92.5% missing
    "OP_Manpower_Neg",           # 92.5% missing
    "OP_Funding_Pos",            # 72.1% missing
    "OP_Funding_Neg",            # 85.8% missing
]

# Columns with known data-entry errors (values >> 100) to cap before MICE.
# These columns are on a 0-100 proportion scale; values above 100 are errors.
CAP_COLS = {
    "AB_Physical_capacity_Pos": 100.0,
    "OP_Training_Neg":          100.0,
}

# MICE random seed for reproducibility (FAIR principle).
RANDOM_STATE = _PIPELINE_CFG["RANDOM_STATE"]

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 65)
print("STAGE 05b — MAO COMPOSITE IMPUTATION")
print("=" * 65)

# Load scored composite files (output of Stage 05)
for path, label in [(FVI_CSV, "FVI"), (EVI_CSV, "EVI")]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} imputed scores not found: {path}\n"
            f"Run Stage 05 (mice_imputation.py) first."
        )

df_fvi = pd.read_csv(FVI_CSV, encoding="utf-8-sig", low_memory=False)
df_evi = pd.read_csv(EVI_CSV, encoding="utf-8-sig", low_memory=False)

print(f"\nLoaded FVI scores : {len(df_fvi):,} rows")
print(f"Loaded EVI scores : {len(df_evi):,} rows")

if len(df_fvi) != len(df_evi):
    raise ValueError(
        f"Row count mismatch: FVI={len(df_fvi)}, EVI={len(df_evi)}. "
        "Both must derive from the same survey file."
    )

# Load raw survey for MAO columns and metadata
df_raw = pd.read_csv(SURVEY_CSV, encoding="utf-8-sig", low_memory=False)
df_raw = df_raw.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
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

print(f"Loaded raw survey : {len(df_raw):,} rows x {len(df_raw.columns):,} cols")

if len(df_raw) != len(df_fvi):
    raise ValueError(
        f"Row count mismatch between raw survey ({len(df_raw)}) "
        f"and scored files ({len(df_fvi)}). All files must be from "
        "the same survey dataset in the same row order."
    )

# =============================================================================
# STEP 1 — DATA QUALITY CORRECTIONS
# =============================================================================
# Cap outlier values before any analysis. These columns have known data-entry
# errors (values up to 1e11 on a 0-100 scale). Cap to NaN so MICE treats
# them as missing rather than corrupting the imputation model.

print("\n" + "=" * 65)
print("STEP 1 — DATA QUALITY CORRECTIONS")
print("=" * 65)

for col, cap in CAP_COLS.items():
    if col in df_raw.columns:
        s = pd.to_numeric(df_raw[col], errors="coerce")
        n_over = (s > cap).sum()
        if n_over > 0:
            df_raw.loc[s > cap, col] = np.nan
            print(f"  {col}: {n_over} values > {cap} → set to NaN")
        else:
            print(f"  {col}: no values > {cap} (no correction needed)")

# =============================================================================
# STEP 2 — ASSEMBLE FEATURE MATRIX FOR MICE
# =============================================================================
# The MICE model for each imputed column uses:
#   - All other imputed/complete MAO columns (cross-predictor)
#   - FVI_norm_1_5 and EVI_norm_1_5 (outcome variables MUST be in model)
#   - Surveyor encoded as integer (strong MAR predictor)
#   - Community encoded as integer (geographic auxiliary)

print("\n" + "=" * 65)
print("STEP 2 — ASSEMBLING MICE FEATURE MATRIX")
print("=" * 65)

# Collect all MAO and socioeconomic columns that exist in the raw survey
all_mao_cols = IMPUTE_COLS + COMPLETE_COLS
existing_cols = [c for c in all_mao_cols if c in df_raw.columns]
missing_from_survey = [c for c in all_mao_cols if c not in df_raw.columns]

if missing_from_survey:
    print(f"\n  WARNING: {len(missing_from_survey)} columns not found in survey:")
    for c in missing_from_survey:
        print(f"    {c}")

print(f"\n  MAO columns found in survey   : {len(existing_cols)}")
print(f"  Columns to impute             : {len([c for c in IMPUTE_COLS if c in df_raw.columns])}")
print(f"  Complete columns (predictors) : {len([c for c in COMPLETE_COLS if c in df_raw.columns])}")
print(f"  Excluded columns              : {len(EXCLUDED_COLS)}")

# Coerce all MAO columns to numeric
df_mao = pd.DataFrame(index=df_raw.index)
for col in existing_cols:
    df_mao[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Add outcome variables as auxiliary predictors
df_mao["FVI_norm_1_5"] = pd.to_numeric(
    df_fvi["FVI_norm_1_5"] if "FVI_norm_1_5" in df_fvi.columns else np.nan,
    errors="coerce"
)
df_mao["EVI_norm_1_5"] = pd.to_numeric(
    df_evi["EVI_norm_1_5"] if "EVI_norm_1_5" in df_evi.columns else np.nan,
    errors="coerce"
)

# Encode surveyor as integer auxiliary predictor
le_surv = LabelEncoder()
surv_raw = df_raw[SURVEYOR_COL].fillna("unknown") if SURVEYOR_COL in df_raw.columns else pd.Series(["unknown"] * len(df_raw))
df_mao["_surveyor_enc"] = le_surv.fit_transform(surv_raw)

# Encode community as integer auxiliary predictor
le_comm = LabelEncoder()
comm_raw = df_raw[COMMUNITY_COL].fillna("unknown") if COMMUNITY_COL in df_raw.columns else pd.Series(["unknown"] * len(df_raw))
df_mao["_community_enc"] = le_comm.fit_transform(comm_raw)

# Report pre-imputation missingness for columns to impute
print(f"\n  Pre-imputation missingness for imputed columns:")
print(f"  {'Column':<45} {'n_missing':>10} {'%_missing':>10}")
print("  " + "-" * 67)
pre_missing = {}
for col in IMPUTE_COLS:
    if col in df_mao.columns:
        n = df_mao[col].isna().sum()
        pct = n / len(df_mao) * 100
        pre_missing[col] = n
        print(f"  {col:<45} {n:>10,} {pct:>9.1f}%")

# =============================================================================
# STEP 3 — FIT MICE IMPUTER
# =============================================================================
# IterativeImputer implements MICE (BayesianRidge estimator by default).
# max_iter=20 gives sufficient convergence for 15-30% missingness.
# initial_strategy='mean' initialises missing values before iteration.

print("\n" + "=" * 65)
print("STEP 3 — RUNNING MICE IMPUTATION")
print("=" * 65)
print(f"\n  Fitting IterativeImputer on {df_mao.shape[1]} columns x {len(df_mao):,} rows...")

imputer = IterativeImputer(
    max_iter=20,
    random_state=RANDOM_STATE,
    initial_strategy="mean",
    min_value=0.0,       # proportion scores cannot be negative
    max_value=100.0,     # proportion scores cannot exceed 100
    verbose=0,
)

imputed_array = imputer.fit_transform(df_mao.values)
df_imputed = pd.DataFrame(imputed_array, columns=df_mao.columns, index=df_mao.index)

print("  MICE imputation complete.")

# =============================================================================
# STEP 4 — POST-IMPUTATION VALIDATION
# =============================================================================

print("\n" + "=" * 65)
print("STEP 4 — POST-IMPUTATION VALIDATION")
print("=" * 65)

log_rows = []
print(f"\n  {'Column':<45} {'Before':>8} {'After':>8} {'Imputed':>8} {'Mean_shift':>12}")
print("  " + "-" * 83)

for col in IMPUTE_COLS:
    if col not in df_mao.columns:
        continue

    original = df_mao[col]
    imputed  = df_imputed[col]

    n_before = original.isna().sum()
    n_after  = imputed.isna().sum()
    n_imp    = n_before - n_after

    mean_orig = original.mean()
    mean_imp  = imputed.mean()
    mean_shift = mean_imp - mean_orig

    # Bounds check
    n_below = (imputed < 0).sum()
    n_above = (imputed > 100).sum()
    bounds_ok = (n_below == 0) and (n_above == 0)

    print(
        f"  {col:<45} {n_before:>8,} {n_after:>8,} {n_imp:>8,} "
        f"{mean_shift:>+11.3f}{'  [BOUNDS OK]' if bounds_ok else '  [BOUNDS VIOLATION]'}"
    )

    log_rows.append({
        "column":       col,
        "n_missing_before": n_before,
        "n_missing_after":  n_after,
        "n_imputed":    n_imp,
        "pct_missing_before": round(n_before / len(df_mao) * 100, 2),
        "mean_original": round(mean_orig, 4),
        "mean_imputed":  round(mean_imp, 4),
        "mean_shift":    round(mean_shift, 4),
        "bounds_ok":     bounds_ok,
        "method":        "MICE (IterativeImputer, BayesianRidge, max_iter=20)",
    })

# Save imputation log
log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUT_LOG, index=False)
print(f"\n  Imputation log saved: {OUT_LOG.name}")

# =============================================================================
# STEP 5 — BUILD ANALYSIS DATASET
# =============================================================================
# Merge imputed MAO columns into the FVI scored file (which already has
# all 2,993 rows with complete composite scores). The analysis dataset
# is the single file read by all regression stages.

print("\n" + "=" * 65)
print("STEP 5 — BUILDING ANALYSIS DATASET")
print("=" * 65)

# Start from the FVI imputed file as the base (has FVI_norm_1_5 + all
# survey metadata rows intact)
df_out = df_fvi.copy()

# Add EVI composite columns
evi_cols_to_add = [c for c in ["EVI_norm_1_5", "EVI_norm_0_1", "EVI_CLASS",
                                "EVI_composite", "EVI_n_valid"]
                   if c in df_evi.columns]
for col in evi_cols_to_add:
    df_out[col] = df_evi[col].values

# Add imputed MAO columns — use imputed values where imputation occurred,
# original values otherwise (complete columns are also in df_imputed
# but unchanged by MICE since they had no missing values to fill)
for col in existing_cols:
    if col in df_imputed.columns:
        df_out[col] = df_imputed[col].values
    else:
        df_out[col] = df_raw[col].values

# Add imputation flag columns for the imputed variables
for col in IMPUTE_COLS:
    if col in df_mao.columns:
        flag_col = f"{col}_imputed"
        df_out[flag_col] = df_mao[col].isna().values  # True = was missing, now imputed

# Add surveyor and community from raw survey (for downstream use)
if SURVEYOR_COL in df_raw.columns:
    df_out[SURVEYOR_COL] = df_raw[SURVEYOR_COL].values
if COMMUNITY_COL in df_raw.columns:
    df_out[COMMUNITY_COL] = df_raw[COMMUNITY_COL].values

# Drop internal encoder columns
df_out.drop(columns=["_surveyor_enc", "_community_enc"], errors="ignore", inplace=True)

df_out.to_csv(OUT_ANALYSIS, index=False)
print(f"\n  Analysis dataset saved : {OUT_ANALYSIS.name}")
print(f"  Rows                   : {len(df_out):,}")
print(f"  Columns                : {len(df_out.columns):,}")

# Report which columns are now in the analysis dataset
imputed_present = [c for c in IMPUTE_COLS if c in df_out.columns]
complete_present = [c for c in COMPLETE_COLS if c in df_out.columns]
print(f"\n  Imputed MAO columns    : {len(imputed_present)}")
print(f"  Complete MAO columns   : {len(complete_present)}")
print(f"  Excluded columns       : {len(EXCLUDED_COLS)} (not in dataset by design)")

# =============================================================================
# STEP 6 — WRITE AUDIT REPORT
# =============================================================================

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("MAO COMPOSITE IMPUTATION — AUDIT REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Dataset        : Nepal Lumbini Survey (n={len(df_raw):,})\n")
    f.write(f"Method         : MICE — sklearn IterativeImputer (BayesianRidge)\n")
    f.write(f"Random seed    : {RANDOM_STATE}\n")
    f.write(f"max_iter       : 20\n")
    f.write(f"Bounds         : [0, 100] (proportion score scale)\n\n")

    f.write("MICE AUXILIARY PREDICTORS\n")
    f.write("-" * 40 + "\n")
    f.write("  All other MAO/socioeconomic columns (cross-predictor)\n")
    f.write("  FVI_norm_1_5  — outcome included per van Buuren (2018, §6.3)\n")
    f.write("  EVI_norm_1_5  — outcome included per van Buuren (2018, §6.3)\n")
    f.write("  Surveyor ID   — MAR predictor (enumerator block skipping)\n")
    f.write("  Community     — geographic auxiliary\n\n")

    f.write("IMPUTED COLUMNS\n")
    f.write("-" * 40 + "\n")
    for row in log_rows:
        f.write(
            f"  {row['column']:<45}  "
            f"{row['n_missing_before']:>4} missing → "
            f"{row['n_imputed']:>4} imputed  "
            f"mean shift: {row['mean_shift']:>+.3f}\n"
        )

    f.write("\nEXCLUDED COLUMNS (not imputed — too many missing or structural)\n")
    f.write("-" * 40 + "\n")
    reasons = {
        "Acc_Perc_Neg_expression":  "87.4% missing — exclude from analysis entirely",
        "App_Perc_Pos_expression":  "52.2% missing — above imputation reliability limit",
        "AB_Location_Neg":          "61.2% missing — structural question block absence",
        "OP_Manpower_Pos":          "92.5% missing — question not universally asked",
        "OP_Manpower_Neg":          "92.5% missing — question not universally asked",
        "OP_Funding_Pos":           "72.1% missing — structural",
        "OP_Funding_Neg":           "85.8% missing — structural",
    }
    for col, reason in reasons.items():
        f.write(f"  {col:<45}  {reason}\n")

    f.write("\nDATA QUALITY CORRECTIONS APPLIED\n")
    f.write("-" * 40 + "\n")
    f.write("  AB_Physical_capacity_Pos: values > 100 set to NaN before MICE\n")
    f.write("  OP_Training_Neg:          values > 100 set to NaN before MICE\n")

    f.write("\nLIMITATIONS\n")
    f.write("-" * 40 + "\n")
    f.write(
        "  Single imputation (m=1) is used for pipeline integration.\n"
        "  This produces unbiased point estimates but underestimates\n"
        "  standard errors in downstream regression. For final thesis\n"
        "  submission, multiple imputation (m=5 to m=10) with Rubin's\n"
        "  (1987) pooling rules is the methodologically complete approach.\n"
        "  Report as limitation and cite van Buuren (2018) and Rubin (1987).\n"
    )

    f.write("\nREFERENCES\n")
    f.write("-" * 40 + "\n")
    f.write("  Rubin (1987)      : https://doi.org/10.1002/9780470316696\n")
    f.write("  van Buuren (2018) : https://stefvanbuuren.name/fimd/\n")
    f.write("  Sterne (2009)     : https://doi.org/10.1136/bmj.b2393\n")
    f.write("  West & Olson (2010): https://doi.org/10.1093/poq/nfq061\n")

print(f"\n  Audit report saved : {OUT_REPORT.name}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("STAGE 05b COMPLETE")
print("=" * 65)
print(f"""
  analysis_dataset.csv is now the single input for all regression stages.
  Update stages 08, 10, 13, 16 to read from analysis_dataset.csv
  instead of imputed_FVI_scores.csv / imputed_EVI_scores.csv.

  Excluded columns — do NOT add back to regression models:
    {chr(10).join(f"    {c}" for c in EXCLUDED_COLS)}

  Imputation flag columns added (True = value was imputed):
    {chr(10).join(f"    {c}_imputed" for c in IMPUTE_COLS if c in df_mao.columns)}

  Outputs:
    {OUT_ANALYSIS.name:<35} — use as input for Stages 08-16
    {OUT_LOG.name:<35} — imputation counts and mean shifts
    {OUT_REPORT.name:<35} — full audit trail

REFERENCES
----------
  Rubin (1987)       : https://doi.org/10.1002/9780470316696
  van Buuren (2018)  : https://stefvanbuuren.name/fimd/
  Sterne (2009)      : https://doi.org/10.1136/bmj.b2393
  West & Olson (2010): https://doi.org/10.1093/poq/nfq061
""")
