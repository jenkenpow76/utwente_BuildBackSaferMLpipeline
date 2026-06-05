"""
stage08b_ward_covariates.py
===========================
Stage 08b | PRE | Ward-Level Covariate Integration

Purpose
-------
Reads ward_information.csv (one row per ward, 18 wards), computes two
ward-level composite scores, and merges them into analysis_dataset.csv
produced by Stage 08.  The resulting enriched dataset is written back to
analysis_dataset.csv so all downstream stages (09 onwards) pick up the
new columns transparently.

The two scores are Level-2 (ward-level) predictors for the mixed-effects
models in Stages 15 (FVI) and 21 (EVI).  They are constant within a ward
and vary only between wards — which is precisely what Level-2 predictors
must be in a random-intercept model (Raudenbush & Bryk, 2002).

Scores
------
transport_access_score  (0 – 1, higher = better access)
    Count of 11 binary ward-level transport indicators, normalised to [0,1].

    Mode indicators (8):
        accessable_by_foot, accessable_by_car, accessable_by_bus,
        accessible_by_motorbike, accessible_by_bicycle,
        accessible_by_boat, accessible_by_riksja, accessible_by_tractor

    Road infrastructure indicators (3):
        land_roads, paved_roads, other_roads

    count_bridges is recorded as an integer (not binary) and is excluded
    from the normalised score to preserve interval comparability across items.
    It is retained in analysis_dataset.csv as ward_bridge_count for reference.

assistance_intensity_score  (0 – 1, higher = more assistance)
    Count of 10 binary ward-level housing assistance indicators,
    normalised to [0,1].

    Assistance type indicators (6):
        assistance_demonstrationhouse, assistance_carpenter/masontraining,
        assistance_doortodoor, assistance_communityorientation,
        assistance_materials, assistance_financial

    Programme source indicators (3):
        housing_assistance_janta, housing_assistance_surakshitawas,
        housing_assistance_other

    Information indicator (1):
        assistance_info_received

    housing_assistance_none is the logical inverse and is excluded
    (its information is already captured by a zero on all other items).
    housing_assistance_NDRRMA and housing_assistance_CRS are always 0
    in this dataset and are excluded to avoid deflating scores.

Alignment handling
------------------
Two alignment issues between analysis_dataset.csv (GD005_VDC_name) and
ward_information.csv (name) are resolved before merging:

1. tulsipur-ward_14 (hyphen) in analysis data vs
   tulsipur_ward_14 (underscore) in ward file.
   Resolution: normalise both to underscore before matching.

2. baijanath_ward_07 appears in analysis_dataset (n=1 household) but
   has no row in ward_information.
   Resolution: that household receives NaN for both scores.
   The single-household ward is already flagged in the mixed-effects
   caterpillar plot as having very wide confidence intervals.  NaN
   propagation to Stage 15/21 is handled by pairwise deletion.

Inputs
------
    outputs/analysis_dataset.csv     — produced by Stage 08
    ward_information.csv             — ward-level data file

Outputs
-------
    outputs/analysis_dataset.csv     — updated in place (two new columns added)
    outputs/ward_covariates.csv      — standalone ward-level score table
                                       (one row per ward, for audit and thesis
                                       appendix)

References
----------
Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical Linear Models:
    Applications and Data Analysis Methods (2nd ed.). Sage.

Snijders, T. A. B., & Bosker, R. J. (2012). Multilevel Analysis: An
    Introduction to Basic and Advanced Multilevel Modelling (2nd ed.). Sage.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

HERE       = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WARD_CSV     = HERE / "ward_information.csv"
ANALYSIS_CSV = OUTPUT_DIR / "analysis_dataset.csv"
OUT_WARD     = OUTPUT_DIR / "ward_covariates.csv"

# Column in analysis_dataset.csv that holds ward names
COMMUNITY_COL = "GD005_VDC_name"

# =============================================================================
# SCORE DEFINITIONS
# =============================================================================

# 11 binary transport indicators summed and normalised to [0, 1]
TRANSPORT_COLS = [
    "accessable_by_foot",
    "accessable_by_car",
    "accessable_by_bus",
    "accessible_by_motorbike",
    "accessible_by_bicycle",
    "accessible_by_boat",
    "accessible_by_riksja",
    "accessible_by_tractor",
    "land_roads",
    "paved_roads",
    "other_roads",
]

# 10 binary assistance indicators summed and normalised to [0, 1]
# Excludes: housing_assistance_none (logical inverse)
#           housing_assistance_NDRRMA, housing_assistance_CRS (always 0)
ASSISTANCE_COLS = [
    "assistance_demonstrationhouse",
    "assistance_carpenter/masontraining",
    "assistance_doortodoor",
    "assistance_communityorientation",
    "assistance_materials",
    "assistance_financial",
    "housing_assistance_janta",
    "housing_assistance_surakshitawas",
    "housing_assistance_other",
    "assistance_info_received",
]

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=" * 65)
print("STAGE 08b — WARD-LEVEL COVARIATE INTEGRATION")
print("=" * 65)

if not WARD_CSV.exists():
    raise FileNotFoundError(
        f"ward_information.csv not found at {WARD_CSV}. "
        "Place it in the same directory as run_pipeline.py."
    )
if not ANALYSIS_CSV.exists():
    raise FileNotFoundError(
        f"analysis_dataset.csv not found at {ANALYSIS_CSV}. "
        "Run Stage 08 (MAO Imputation) before Stage 08b."
    )

ward_df = pd.read_csv(WARD_CSV, encoding="utf-8-sig")
print(f"\nLoaded ward_information.csv : {len(ward_df)} wards x {len(ward_df.columns)} columns")

analysis_df = pd.read_csv(ANALYSIS_CSV, encoding="utf-8-sig", low_memory=False)
print(f"Loaded analysis_dataset.csv : {len(analysis_df):,} rows x {len(analysis_df.columns)} columns")

# =============================================================================
# 2. NORMALISE WARD NAMES FOR MATCHING
# =============================================================================
# Both name columns are lower-cased and hyphens normalised to underscores.
# This resolves the tulsipur-ward_14 / tulsipur_ward_14 discrepancy.

def normalise_ward_name(s):
    """Lower-case, strip whitespace, replace hyphens with underscores."""
    if pd.isna(s):
        return np.nan
    return str(s).strip().lower().replace("-", "_")

ward_df["_ward_key"]      = ward_df["name"].apply(normalise_ward_name)
analysis_df["_ward_key"]  = analysis_df[COMMUNITY_COL].apply(normalise_ward_name)

# Report alignment before merge
ward_keys     = set(ward_df["_ward_key"].dropna())
analysis_keys = set(analysis_df["_ward_key"].dropna())

unmatched_analysis = analysis_keys - ward_keys
unmatched_ward     = ward_keys - analysis_keys

print("\n--- Ward name alignment ---")
print(f"  Wards in ward_information     : {len(ward_keys)}")
print(f"  Unique wards in analysis data : {len(analysis_keys)}")
if unmatched_analysis:
    print(f"  In analysis but NOT in ward file (will receive NaN):")
    for w in sorted(unmatched_analysis):
        n = (analysis_df["_ward_key"] == w).sum()
        print(f"    '{w}'  (n={n} households)")
if unmatched_ward:
    print(f"  In ward file but NOT in analysis data (unused rows):")
    for w in sorted(unmatched_ward):
        print(f"    '{w}'")
if not unmatched_analysis and not unmatched_ward:
    print("  Perfect match — all wards aligned.")

# =============================================================================
# 3. COMPUTE WARD-LEVEL COMPOSITE SCORES
# =============================================================================

print("\n--- Computing composite scores ---")

# Coerce indicator columns to numeric (handles any stray strings)
for col in TRANSPORT_COLS + ASSISTANCE_COLS:
    ward_df[col] = pd.to_numeric(ward_df[col], errors="coerce")

# transport_access_score: proportion of 11 binary items that equal 1
ward_df["transport_access_score"] = (
    ward_df[TRANSPORT_COLS].sum(axis=1) / len(TRANSPORT_COLS)
)

# assistance_intensity_score: proportion of 10 binary items that equal 1
ward_df["assistance_intensity_score"] = (
    ward_df[ASSISTANCE_COLS].sum(axis=1) / len(ASSISTANCE_COLS)
)

# Also carry forward count_bridges as a raw integer (not normalised)
ward_df["ward_bridge_count"] = pd.to_numeric(ward_df["count_bridges"], errors="coerce")

# Build the merge table: key + three new columns
ward_merge = ward_df[["_ward_key",
                       "transport_access_score",
                       "assistance_intensity_score",
                       "ward_bridge_count"]].copy()

print(f"\n  transport_access_score:")
print(f"    min={ward_df['transport_access_score'].min():.3f}  "
      f"max={ward_df['transport_access_score'].max():.3f}  "
      f"mean={ward_df['transport_access_score'].mean():.3f}")
print(f"  assistance_intensity_score:")
print(f"    min={ward_df['assistance_intensity_score'].min():.3f}  "
      f"max={ward_df['assistance_intensity_score'].max():.3f}  "
      f"mean={ward_df['assistance_intensity_score'].mean():.3f}")

# =============================================================================
# 4. MERGE INTO ANALYSIS DATASET
# =============================================================================

# Drop any pre-existing versions of these columns (idempotent re-runs)
for col in ["transport_access_score", "assistance_intensity_score", "ward_bridge_count"]:
    if col in analysis_df.columns:
        analysis_df.drop(columns=[col], inplace=True)

analysis_df = analysis_df.merge(ward_merge, on="_ward_key", how="left")
analysis_df.drop(columns=["_ward_key"], inplace=True)

# Validate merge
n_transport_null = analysis_df["transport_access_score"].isna().sum()
n_assist_null    = analysis_df["assistance_intensity_score"].isna().sum()

print(f"\n--- Merge results ---")
print(f"  Rows in analysis_dataset    : {len(analysis_df):,}")
print(f"  transport_access_score NaN  : {n_transport_null} "
      f"({'expected — unmatched ward' if n_transport_null else 'none'})")
print(f"  assistance_intensity NaN    : {n_assist_null} "
      f"({'expected — unmatched ward' if n_assist_null else 'none'})")

# =============================================================================
# 5. CONSTRUCT PP01 / PP02 3-LEVEL COLLAPSED CATEGORICALS
# =============================================================================
# PP01 (earthquake rebuild history) and PP02 (flood rebuild history) are the
# raw categorical fields from the survey. Their original response categories
# include "no", "totally_damaged", "partially_damaged", "yes__minor_damage",
# "no__but_my_old_house_was_damag", "no__but_previous_house_in_this",
# "I_don't_know", "other", "rebuild_one_time", "rebuild_multiple_times",
# plus blank entries (the survey form's missing-value token is a single space).
#
# For the v49 EVI/FVI candidate pool we collapse these into 3 levels:
#   damaged      = household reports any damage or rebuild
#   undamaged    = household reports house survived without damage
#   unknown_other = household reports unknown / other / blank
#
# Reference category = "undamaged". Two binary dummies are created per index
# (the third level is implied by both dummies being 0). This avoids the dummy-
# variable trap and yields a tractable predictor count for the stepwise model.
#
# Note on inclusion (Section 2.6.1 of the thesis): we verified against
# Syntax_Motivation_02_01_25__4_.sps that PP01 and PP02 do not appear as
# inputs to the Motivation MAO composite scores (Utility, Applicability,
# Acceptability). Ability and Opportunity composite syntax files were not
# available for verification; this is acknowledged as a residual methodological
# uncertainty in Section 5.4 (Limitations) of the thesis.

DAMAGED_TOKENS = (
    "totally_damaged", "partially_damaged",
    "yes__minor_damage",
    "rebuild_one_time", "rebuild_multiple_times",
    "no__but_my_old_house_was_damag",
)

UNDAMAGED_TOKENS = (
    "no__but_previous_house_in_this",  # the previous house was damaged, not this one
)

UNKNOWN_OTHER_TOKENS = (
    "i_don't_know", "i_dont_know", "other",
)


def collapse_3lvl(value):
    """
    Collapse a raw PP01 / PP02 categorical response to one of three levels:
    'damaged', 'undamaged', or 'unknown_other'.

    Multi-valued responses (e.g., 'totally_damaged partially_damaged' or
    'partially_damaged no') resolve as follows: any token indicating
    damage classifies the row as 'damaged'. Otherwise an explicit 'no'
    or 'no__but_previous_house_in_this' classifies as 'undamaged'. All
    remaining cases (blank, 'I_don't_know', 'other', or unmatched tokens)
    classify as 'unknown_other'.
    """
    if value is None:
        return "unknown_other"
    s = str(value).strip().lower()
    if s == "" or s == "nan":
        return "unknown_other"
    # Damage tokens take priority — any damage signal classifies as damaged.
    for tok in DAMAGED_TOKENS:
        if tok in s:
            return "damaged"
    # Then unknown_other tokens.
    for tok in UNKNOWN_OTHER_TOKENS:
        if tok in s:
            return "unknown_other"
    # Then explicit undamaged token (the survivor-no-rebuild case).
    for tok in UNDAMAGED_TOKENS:
        if tok in s:
            return "undamaged"
    # Bare "no" — household reports the house was not rebuilt and is undamaged.
    # Test exact equality and split-by-space to catch combinations like "no other"
    # (already handled by the loops above), or pure "no".
    tokens = s.split()
    if tokens == ["no"]:
        return "undamaged"
    if "no" in tokens:
        return "undamaged"
    # Catch-all — should be rare. Treat unmatched values as unknown_other.
    return "unknown_other"


def make_3lvl_dummies(df, parent_col, prefix):
    """
    Apply collapse_3lvl to a parent column and create two binary dummies.
    The dummies are named '<prefix>_damaged' and '<prefix>_unknown_other'.
    The reference category 'undamaged' is encoded by both dummies = 0.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to modify in place.
    parent_col : str
        Name of the source categorical column (e.g.,
        'PP01_rebuild_repair_house_001').
    prefix : str
        Prefix for the new dummy columns (e.g.,
        'PP01_eq_rebuild_3lvl').
    """
    if parent_col not in df.columns:
        print(f"  WARNING: parent column {parent_col!r} not found — "
              f"skipping {prefix} construction.")
        return
    collapsed = df[parent_col].apply(collapse_3lvl)
    df[f"{prefix}_damaged"]       = (collapsed == "damaged").astype(int)
    df[f"{prefix}_unknown_other"] = (collapsed == "unknown_other").astype(int)
    # Report the distribution for audit traceability.
    counts = collapsed.value_counts()
    print(f"\n  {prefix} 3-level distribution (from {parent_col}):")
    for level in ("damaged", "undamaged", "unknown_other"):
        n = counts.get(level, 0)
        pct = 100.0 * n / len(df)
        print(f"    {level:14s}  n={n:>5}  ({pct:5.1f}%)")


print("\n--- Constructing PP01 (earthquake rebuild) 3-level categorical ---")
make_3lvl_dummies(analysis_df,
                  parent_col="PP01_rebuild_repair_house_001",
                  prefix="PP01_eq_rebuild_3lvl")

print("\n--- Constructing PP02 (flood rebuild) 3-level categorical ---")
make_3lvl_dummies(analysis_df,
                  parent_col="PP02_rebuild_repair_house_001_001",
                  prefix="PP02_flood_rebuild_3lvl")

# =============================================================================
# 6. SAVE OUTPUTS
# =============================================================================

# Overwrite analysis_dataset.csv with the enriched version
analysis_df.to_csv(ANALYSIS_CSV, index=False)
print(f"\n  Updated analysis_dataset.csv : {len(analysis_df):,} rows, "
      f"{len(analysis_df.columns)} columns")

# Write standalone ward-level score table for audit and thesis appendix
ward_export = ward_df[["name", "district", "municipality", "ward", "population",
                        "transport_access_score", "assistance_intensity_score",
                        "ward_bridge_count"]].copy()
ward_export.to_csv(OUT_WARD, index=False)
print(f"  Ward covariate table saved   : {OUT_WARD.name}")

# =============================================================================
# 6. VALIDATION SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("WARD COVARIATE SCORES — VALIDATION SUMMARY")
print("=" * 65)

# Per-ward score table
ward_summary = ward_df[["name", "transport_access_score",
                         "assistance_intensity_score"]].copy()
ward_summary = ward_summary.sort_values("transport_access_score", ascending=False)
print(f"\n{'Ward':<30} {'Transport':>10} {'Assistance':>12}")
print("-" * 55)
for _, row in ward_summary.iterrows():
    print(f"  {row['name']:<28} {row['transport_access_score']:>10.3f} "
          f"{row['assistance_intensity_score']:>12.3f}")

print(f"\nColumn summary in analysis_dataset.csv:")
for col in ["transport_access_score", "assistance_intensity_score"]:
    s = analysis_df[col]
    print(f"  {col}:")
    print(f"    non-null={s.notna().sum():,}  "
          f"null={s.isna().sum()}  "
          f"min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}")

print("\nNote: Both scores are ward-level constants (Level-2 predictors).")
print("Every household in the same ward receives the same score value.")
print("They are intended for use as fixed-effect Level-2 covariates in")
print("the mixed-effects models (Stages 15 and 21).")
