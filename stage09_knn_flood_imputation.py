"""
stage05c_knn_flood_imputation.py
=================================
Stage 05c | Pre-Processing — KNN Imputation of Flood Experience Scores
Nepal Lumbini Survey Dataset (n=2,993)

PURPOSE
-------
Imputes missing values in flood experience perception scores for households
that reported flood exposure (DE01_flood = 1). Imputation is restricted to
flood-exposed households because the scores are perception measures — they
only exist where an experience exists. Non-exposed households (DE01_flood = 0)
are left as NaN and handled via pairwise deletion at the regression stage.

Imputation uses K-Nearest Neighbours (sklearn.impute.KNNImputer). Neighbours
are defined on variables that are substantively related to flood experience
— other perception scores, vulnerability indices, and household
characteristics — not on structural typology, which carries no information
about flood history.

MISSINGNESS CLASSIFICATION (from explanatory_vars_diagnostic.py)
-----------------------------------------------------------------
Among flood-exposed households (n=2,824):

    IMPUTED HERE (MAR, enumerator-driven, 15–19% missing):
        DE03_flood_frequency_score    15.0%  missing
        DE04_flood_depth_score        18.9%  missing
        DE06_flood_future_score        1.5%  missing
        DE08_worry_score               0.3%  missing
        DE09_worry_future_score        1.4%  missing
        DE13_feel_safe_score           1.3%  missing

    EXCLUDED — exceeds 50% threshold (van Buuren 2018, §9.1.4):
        DE05_negative_impact_score          43.9%  missing
        DE07_negative_impact_future_score   43.9%  missing

    NOT IMPUTED — complete for all flood-exposed households:
        DE12_shelter_score    0.0%  missing

WHY KNN RATHER THAN MICE OR TYPOLOGY-BASED IMPUTATION
------------------------------------------------------
MICE (Stage 05) is appropriate for structural indicators whose missingness
is driven by enumerator skip patterns on building features. Typology-based
imputation (Stage 04) is appropriate for MNAR structural indicators where
the building's construction type is a valid donor class.

Neither is appropriate here because:
  1. Flood experience scores are perception measures, not structural
     measurements. A building's typology cluster (wall/roof material,
     floor count) carries no information about how deeply or frequently
     that building floods.
  2. KNN defines neighbours using variables directly related to flood
     experience: other perception scores, vulnerability indices, and
     household demographics. These are substantively meaningful donors.

KNN is a well-established approach for perception-scale data where
observations cluster in related-response space. It is non-parametric,
handles ordinal and continuous mixed inputs, and does not require the
distributional assumptions of MICE's BayesianRidge predictor.

Reference: Troyanskaya et al. (2001). Missing value estimation methods
  for DNA microarrays. Bioinformatics, 17(6), 520–525.
  https://doi.org/10.1093/bioinformatics/17.6.520

KNN PREDICTOR VARIABLES
-----------------------
Predictors used to define neighbours (all near-complete in flood-exposed
sub-sample):

    DE12_shelter_score        0.0%  missing — flood preparedness (complete)
    DE08_worry_score          0.3%  missing — current flood worry
    DE09_worry_future_score   1.4%  missing — future flood worry
    DE13_feel_safe_score      1.3%  missing — general flood safety
    DE06_flood_future_score   1.5%  missing — future flood expectation
    FVI_norm_1_5              0.0%  missing — structural flood vulnerability
    EVI_norm_1_5              0.0%  missing — structural earthquake vulnerability
    HC003_Respondent_s_age    0.0%  missing — respondent age
    HC004_amount_people_household 0.0% missing — household size

These predictors are included in the KNN feature matrix alongside the
target columns so that the imputer finds neighbours that are similar in
both the variables being imputed and the complete contextual predictors.

K SELECTION
-----------
k=5 is used as the default, consistent with the convention for survey
data (Troyanskaya et al. 2001; Acuna & Rodriguez 2004). A sensitivity
check over k = {3, 5, 7, 10} is written to the audit log so that the
choice can be evaluated in the thesis appendix.

PIPELINE POSITION
-----------------
    Stage 05   -> imputation/imputed_EVI_scores.csv, imputed_FVI_scores.csv
    Stage 05b  -> analysis_dataset.csv  (MAO composites merged)
    Stage 05c  -> THIS SCRIPT
               -> analysis_dataset.csv  [updated in place]
               -> imputation/knn_flood_imputation_log.csv
               -> imputation/knn_flood_imputation_report.txt
    Stage 06a  -> variable missingness screening (reads updated analysis_dataset.csv)

INPUTS
------
    outputs/analysis_dataset.csv    — produced by Stage 05b; updated in place

OUTPUTS
-------
    outputs/analysis_dataset.csv              Updated in place with imputed scores
    outputs/imputation/
        knn_flood_imputation_log.csv          Per-column imputation counts and rates
        knn_flood_imputation_report.txt       Full audit trail for methods section

FAIR RESEARCH PRINCIPLES
-------------------------
    Findable      : Output files use consistent knn_flood_ prefix
    Accessible    : All decisions logged to knn_flood_imputation_report.txt
    Interoperable : CSV outputs; no proprietary formats
    Reusable      : Fixed random seed (RANDOM_STATE=42); fully documented

REFERENCES
----------
Acuna, E. & Rodriguez, C. (2004). The treatment of missing values and its
  effect on classifier accuracy. In D. Banks et al. (eds.), Classification,
  Clustering, and Data Mining Applications. Springer.
  https://doi.org/10.1007/978-3-642-17103-1_60

Troyanskaya, O. et al. (2001). Missing value estimation methods for DNA
  microarrays. Bioinformatics, 17(6), 520–525.
  https://doi.org/10.1093/bioinformatics/17.6.520

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
  CRC Press. https://stefvanbuuren.name/fimd/

DEPENDENCIES
------------
    pip install pandas numpy scikit-learn
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Working directory is set to outputs/ by run_pipeline.py
HERE = Path.cwd()

# Random seed — matches pipeline-wide RANDOM_STATE
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# KNN neighbour count
K_NEIGHBOURS = 5

# Flood exposure indicator — only households with value 1 are imputed
FLOOD_INDICATOR = "DE01_flood"

# Target columns to impute (flood-exposed households only)
# DE05 and DE07 are excluded: 43.9% missing exceeds the 50% threshold
# (van Buuren 2018, §9.1.4). DE12 is complete (0% missing) — no imputation needed.
FLOOD_TARGET_COLS = [
    "DE03_flood_frequency_score",
    "DE04_flood_depth_score",
    "DE06_flood_future_score",
    "DE08_worry_score",
    "DE09_worry_future_score",
    "DE13_feel_safe_score",
]

# Predictor columns used to define KNN neighbours.
# Selected because they are (a) substantively related to flood experience
# and (b) near-complete in the flood-exposed sub-sample.
KNN_PREDICTORS = [
    "DE12_shelter_score",           # flood preparedness — complete
    "DE08_worry_score",             # current flood worry
    "DE09_worry_future_score",      # future flood worry
    "DE13_feel_safe_score",         # general flood safety perception
    "DE06_flood_future_score",      # future flood expectation
    "FVI_norm_1_5",                 # structural flood vulnerability index
    "EVI_norm_1_5",                 # structural earthquake vulnerability index
    "HC003_Respondent_s_age",       # respondent age
    "HC004_amount_people_household", # household size
]

# Sensitivity check: evaluate these k values in the audit log
K_SENSITIVITY = [3, 5, 7, 10]

# Output paths
IMPUTE_DIR = HERE / "imputation"
IMPUTE_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH    = IMPUTE_DIR / "knn_flood_imputation_log.csv"
REPORT_PATH = IMPUTE_DIR / "knn_flood_imputation_report.txt"
DATASET_PATH = HERE / "analysis_dataset.csv"

# =============================================================================
# HELPERS
# =============================================================================

def _section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def _to_numeric_col(series: pd.Series) -> pd.Series:
    """Coerce a column to numeric, converting non-parseable values to NaN."""
    return pd.to_numeric(series, errors="coerce")

# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    """Execute Stage 05c: KNN imputation of flood experience scores."""

    report_lines = []

    def log(line: str = "") -> None:
        """Write to console and collect for report file."""
        print(line)
        report_lines.append(line)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — LOAD ANALYSIS DATASET
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 1 — LOAD ANALYSIS DATASET")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"analysis_dataset.csv not found at {DATASET_PATH}. "
            "Ensure Stage 05b has completed successfully."
        )

    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", low_memory=False)
    log(f"Loaded analysis_dataset.csv: {len(df):,} rows, {df.shape[1]} columns")

    # Coerce all target and predictor columns to numeric
    for col in FLOOD_TARGET_COLS + KNN_PREDICTORS + [FLOOD_INDICATOR]:
        if col in df.columns:
            df[col] = _to_numeric_col(df[col])
        else:
            log(f"  WARNING: column '{col}' not found in dataset — skipping")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — IDENTIFY FLOOD-EXPOSED SUB-SAMPLE
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 2 — IDENTIFY FLOOD-EXPOSED SUB-SAMPLE")

    flood_mask = df[FLOOD_INDICATOR] == 1
    n_flood    = flood_mask.sum()
    n_no_flood = (~flood_mask).sum()

    log(f"Flood-exposed households (DE01_flood=1): n={n_flood:,}")
    log(f"Non-exposed households  (DE01_flood=0): n={n_no_flood:,}")
    log(f"Total: {len(df):,}")
    log()
    log("Non-exposed households are NOT imputed. Flood experience scores")
    log("are perception measures that only exist where flood exposure exists.")
    log("Their NaN values remain and are handled by pairwise deletion at")
    log("the regression stage, consistent with Stages 08 and 13.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — PRE-IMPUTATION MISSINGNESS REPORT
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 3 — PRE-IMPUTATION MISSINGNESS (FLOOD-EXPOSED ONLY)")

    flood_df = df.loc[flood_mask].copy()

    log(f"{'Column':<40}  {'n_missing':>9}  {'pct_missing':>11}  {'action'}")
    log("-" * 75)

    # Track which target columns actually need imputation
    cols_to_impute = []
    for col in FLOOD_TARGET_COLS:
        if col not in flood_df.columns:
            log(f"  {col:<38}  {'N/A':>9}  {'N/A':>11}  COLUMN ABSENT — skipped")
            continue
        n_miss = flood_df[col].isna().sum()
        pct    = n_miss / n_flood * 100
        if n_miss == 0:
            action = "COMPLETE — no imputation needed"
        else:
            action = f"IMPUTE (k={K_NEIGHBOURS})"
            cols_to_impute.append(col)
        log(f"  {col:<38}  {n_miss:>9,}  {pct:>10.1f}%  {action}")

    log()
    log("Excluded (>50% missing — van Buuren 2018, §9.1.4):")
    for col in ["DE05_negative_impact_score", "DE07_negative_impact_future_score"]:
        if col in flood_df.columns:
            n_miss = flood_df[col].isna().sum()
            pct    = n_miss / n_flood * 100
            log(f"  {col:<38}  {n_miss:>9,}  {pct:>10.1f}%  EXCLUDED")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — BUILD KNN FEATURE MATRIX (FLOOD-EXPOSED ONLY)
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 4 — BUILD KNN FEATURE MATRIX")

    # Use columns that are actually present in the dataset
    target_present    = [c for c in cols_to_impute if c in flood_df.columns]
    predictor_present = [c for c in KNN_PREDICTORS if c in flood_df.columns]

    log(f"Target columns to impute:   {len(target_present)}")
    for c in target_present:
        log(f"    {c}")
    log(f"\nPredictor columns for KNN:  {len(predictor_present)}")
    for c in predictor_present:
        n_miss = flood_df[c].isna().sum()
        log(f"    {c}  ({n_miss} missing)")

    # Feature matrix: targets + predictors combined
    # KNNImputer imputes all NaN values in the matrix simultaneously,
    # using the non-missing features to compute Euclidean distances.
    knn_cols   = target_present + [c for c in predictor_present if c not in target_present]
    knn_matrix = flood_df[knn_cols].copy()

    log(f"\nKNN feature matrix shape: {knn_matrix.shape}")
    log(f"Total NaN in matrix before imputation: {knn_matrix.isna().sum().sum():,}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — SCALE FEATURES FOR EUCLIDEAN DISTANCE
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 5 — STANDARDISE FEATURES")

    # KNNImputer uses Euclidean distance. Variables on different scales
    # (e.g. age 18–80 vs perception scores 0–5) will dominate distance
    # calculations unless standardised. We scale, impute, then inverse-scale.
    scaler         = StandardScaler()
    knn_matrix_arr = knn_matrix.values.astype(float)

    # Fit scaler on observed values only (column-wise mean/std ignoring NaN)
    # by temporarily filling NaN with column means for the fit, then restoring.
    col_means = np.nanmean(knn_matrix_arr, axis=0)
    col_stds  = np.nanstd(knn_matrix_arr, axis=0)
    col_stds  = np.where(col_stds == 0, 1.0, col_stds)  # prevent divide-by-zero

    knn_matrix_scaled = (knn_matrix_arr - col_means) / col_stds
    # Re-apply NaN mask (standardisation fills NaN with 0 implicitly via broadcasting)
    knn_matrix_scaled[np.isnan(knn_matrix_arr)] = np.nan

    log("Features standardised to zero mean, unit variance.")
    log("NaN positions preserved after scaling.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — KNN IMPUTATION (k=5)
    # ─────────────────────────────────────────────────────────────────────────
    _section(f"STEP 6 — KNN IMPUTATION (k={K_NEIGHBOURS})")

    imputer        = KNNImputer(n_neighbors=K_NEIGHBOURS, weights="distance")
    imputed_scaled = imputer.fit_transform(knn_matrix_scaled)

    # Inverse-scale back to original units
    imputed_arr = (imputed_scaled * col_stds) + col_means

    # Rebuild DataFrame with original index
    imputed_df = pd.DataFrame(
        imputed_arr,
        index=flood_df.index,
        columns=knn_cols,
    )

    log(f"KNNImputer fitted and applied on {len(flood_df):,} flood-exposed rows.")
    log(f"NaN remaining after imputation: {np.isnan(imputed_arr).sum()}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — WRITE IMPUTED VALUES BACK TO MAIN DATAFRAME
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 7 — WRITE IMPUTED VALUES TO ANALYSIS DATASET")

    log_rows = []

    for col in target_present:
        original_series  = df.loc[flood_mask, col].copy()
        imputed_series   = imputed_df[col]

        # Only update cells that were originally NaN
        was_missing_mask = original_series.isna()
        n_imputed        = was_missing_mask.sum()

        if n_imputed > 0:
            # Write imputed values back to main df (flood-exposed rows only)
            df.loc[flood_mask & df[col].isna(), col] = (
                imputed_series[was_missing_mask].values
            )
            # Add boolean flag column for audit transparency
            flag_col = col + "_knn_imputed"
            df[flag_col] = False
            df.loc[flood_mask & was_missing_mask, flag_col] = True

        n_still_missing = df.loc[flood_mask, col].isna().sum()
        pct_imputed     = n_imputed / n_flood * 100

        log(f"  {col}: {n_imputed:,} values imputed "
            f"({pct_imputed:.1f}% of flood-exposed); "
            f"{n_still_missing} remaining NaN")

        log_rows.append({
            "column":          col,
            "n_flood_exposed": n_flood,
            "n_imputed":       n_imputed,
            "pct_imputed":     round(pct_imputed, 2),
            "k_neighbours":    K_NEIGHBOURS,
            "n_still_missing": n_still_missing,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — K SENSITIVITY CHECK
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 8 — K SENSITIVITY CHECK")

    log(f"{'k':>4}  {'mean_imputed_DE03':>18}  {'mean_imputed_DE04':>18}")
    log("-" * 46)

    # Check DE03 and DE04 as representative targets (highest missingness)
    sens_targets = [c for c in ["DE03_flood_frequency_score", "DE04_flood_depth_score"]
                    if c in target_present]

    for k in K_SENSITIVITY:
        sens_imputer = KNNImputer(n_neighbors=k, weights="distance")
        sens_imputed = sens_imputer.fit_transform(knn_matrix_scaled)
        sens_arr     = (sens_imputed * col_stds) + col_means
        sens_df      = pd.DataFrame(sens_arr, index=flood_df.index, columns=knn_cols)

        means = []
        for col in sens_targets:
            orig_missing = flood_df[col].isna()
            means.append(f"{sens_df.loc[orig_missing, col].mean():.4f}")

        row = f"  {k:>2}  " + "  ".join(f"{m:>18}" for m in means)
        log(row)

    log(f"\nSelected k={K_NEIGHBOURS} (default for survey perception data;")
    log("Troyanskaya et al. 2001; Acuna & Rodriguez 2004).")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9 — SAVE OUTPUTS
    # ─────────────────────────────────────────────────────────────────────────
    _section("STEP 9 — SAVE OUTPUTS")

    # Update analysis_dataset.csv in place
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")
    log(f"Updated analysis_dataset.csv: {len(df):,} rows, {df.shape[1]} columns")
    log(f"  Path: {DATASET_PATH}")

    # Imputation log CSV
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    log(f"\nImputation log: {LOG_PATH}")

    # Full audit report
    report_text = "\n".join(report_lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    log(f"Audit report:   {REPORT_PATH}")

    _section("STAGE 05c COMPLETE")
    log(f"KNN imputation applied to {len(target_present)} flood experience columns.")
    log(f"Flood-exposed sub-sample: n={n_flood:,}  |  k={K_NEIGHBOURS}")
    log("analysis_dataset.csv updated in place.")
    log("Non-exposed households (DE01_flood=0) remain NaN — pairwise deletion")
    log("at regression stage (Stages 08, 13) handles these correctly.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
