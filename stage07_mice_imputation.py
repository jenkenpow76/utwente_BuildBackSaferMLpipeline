"""
stage05_mice_imputation.py
==================
Stage 0 | Pre-Processing — MICE Imputation of MAR Indicator Columns
Nepal Lumbini Survey Dataset

PURPOSE
-------
This script imputes missing values in EVI and FVI indicator columns that
were diagnosed as Missing At Random (MAR) by missing_data_diagnostic.py.
It runs AFTER Stages 1 and 7 (which produce scored indicator columns) and
BEFORE all analysis stages (2–6, 8–13).

All outputs are saved to outputs/imputation/ to keep them separate from
scoring outputs and analysis outputs.

MISSING DATA DECISION BASIS
----------------------------
Decisions from missing_data_diagnostic.py (run once, outside pipeline):

    GROUP 1 — MCAR consistent (AUC ~ 0.50): pairwise deletion retained
        EVI: D1_01 Adjacency, D2_01 Plan Shape, D3_02 Openings %

    GROUP 2 — MAR confirmed (AUC > 0.65): MICE imputation applied
        EVI: D5_04 Roof Fasten (AUC=0.960, 47.2% missing)
             D3_01 H/t Ratio   (AUC=0.784, 35.2% missing)
             D3_04 Wall Thick  (AUC=0.784, 35.1% missing)
             D4_02 Roof Mass   (AUC=0.737, 18.9% missing)
             D4_01 Roof Shape  (AUC=0.655, 10.9% missing)
        FVI: FVI3 Roof Conn    (AUC=0.811, 44.5% missing)
             FVI2 Roof Mat     (AUC=0.706, 18.6% missing)
             FVI11 L/W Ratio   (AUC=0.670,  4.3% missing)
             FVI4 Wall Mat     (AUC=0.653,  5.3% missing)

    GROUP 3 — MNAR also plausible (domain reasoning): treated as MAR
        EVI D3_01 H/t Ratio, D3_04 Wall Thick — wall may be collapsed
        or inaccessible. MNAR sensitivity analysis in thesis appendix
        (Fig 5EVI) quantifies the impact. Imputation applied but results
        under worst-case MNAR (score=5) are provided as robustness check.

SURVEYOR CLUSTERING NOTE
-------------------------
missing_data_diagnostic.py Step 5 confirmed strong enumerator effects:
    EVI D5_04 Roof Fasten: 31–58% range across enumerators
    EVI D3_04 Wall Thick : 0.5–15% range (enumerator_8: 15.2%)
    FVI3 Roof Conn       : uniform 31–58% (physical access issue)

Surveyor identity (GD001_Name_of_the_surveyor) is included as an
auxiliary variable in the MICE imputation model to account for
enumerator-driven missingness, consistent with MAR assumptions.

WHAT IS IMPUTED
---------------
MICE operates on SCORED indicator columns (1/3/5 ordinal scale), not
on raw survey dummy columns. This is methodologically correct because:
  1. Scored columns are the direct inputs to composite indices
  2. The 1/3/5 scale preserves ordinal structure
  3. Imputing raw dummies would require re-scoring

After imputation, continuous predictions are rounded to nearest valid
score {1, 3, 5} to preserve the ordinal measurement scale.

OUTPUT FILES (all saved to outputs/imputation/)
------------------------------------------------
    imputed_EVI_scores.csv      EVI scored columns with imputed values
    imputed_FVI_scores.csv      FVI scored columns with imputed values
    mice_imputation_log.csv     Per-column imputation counts (for methods)
    mice_mnar_sensitivity.csv   Class distribution under MNAR scenarios

PIPELINE INTEGRATION
--------------------
This script runs as Stage 0 in run_pipeline.py:
    Stage 1  -> stage01_fvi_scoring.py   -> outputs/FVI_CIMDEN_scores.csv
    Stage 7  -> stage02_evi_scoring.py  -> outputs/EVI_CIMDEN_scores.csv
    Stage 0  -> stage05_mice_imputation.py -> outputs/imputation/imputed_*.csv
    Stages 2–6, 8–13 -> read from outputs/imputation/imputed_*.csv

DEPENDENCIES
------------
    pip install pandas numpy scikit-learn

REFERENCES
----------
van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
  Imputation by Chained Equations in R. Journal of Statistical
  Software, 45(3), 1–67. https://doi.org/10.18637/jss.v045.i03

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
  CRC Press. https://stefvanbuuren.name/fimd/

Azur, M.J. et al. (2011). Multiple imputation by chained equations:
  what is it and how does it work? International Journal of Methods
  in Psychiatric Research, 20(1), 40–49.
  https://doi.org/10.1002/mpr.329
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# =============================================================================
# FILE PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent

# Inputs: MNAR-imputed CSVs from Stage 0b (stage04_mnar_imputation.py)
EVI_SCORES_CSV = HERE / "outputs" / "imputation" / "EVI_imputed.csv"
FVI_SCORES_CSV = HERE / "outputs" / "imputation" / "FVI_imputed.csv"

# Raw survey — needed to extract surveyor as auxiliary variable
RAW_SURVEY_CSV = HERE / "20263003_Nepal_Lumbini_data.csv"

# Output subdirectory — all imputation outputs go here
IMPUTE_DIR = HERE / "outputs" / "imputation"
IMPUTE_DIR.mkdir(parents=True, exist_ok=True)

# Output files
EVI_IMPUTED_CSV = IMPUTE_DIR / "imputed_EVI_scores.csv"
FVI_IMPUTED_CSV = IMPUTE_DIR / "imputed_FVI_scores.csv"
LOG_CSV         = IMPUTE_DIR / "mice_imputation_log.csv"
MNAR_CSV        = IMPUTE_DIR / "mice_mnar_sensitivity.csv"

# =============================================================================
# COLUMN CONFIGURATION
# =============================================================================
# MAR-confirmed columns: from missing_data_diagnostic.py Step 3 (AUC > 0.65)
# Columns not listed here retain pairwise deletion (MCAR or low missingness)

EVI_MAR_COLS = [
    "EVI_D5_04",   # Roof Fasten  — AUC=0.960  47.2% missing  Strong MAR
    "EVI_D3_01",   # H/t Ratio    — AUC=0.784  35.2% missing  Moderate MAR + MNAR possible
    "EVI_D3_04",   # Wall Thick   — AUC=0.784  35.1% missing  Moderate MAR + MNAR possible
    "EVI_D4_02",   # Roof Mass    — AUC=0.737  18.9% missing  Moderate MAR
    "EVI_D4_01",   # Roof Shape   — AUC=0.655  10.9% missing  Moderate MAR
]

FVI_MAR_COLS = [
    "FVI3_RoofConn_LVL",   # Roof Conn  — AUC=0.811  44.5% missing  Strong MAR
    "FVI2_RoofMat_LVL",    # Roof Mat   — AUC=0.706  18.6% missing  Moderate MAR
    "FVI11_LWratio_LVL",   # L/W Ratio  — AUC=0.670   4.3% missing  Moderate MAR
    "FVI4_WallMat_LVL",    # Wall Mat   — AUC=0.653   5.3% missing  Moderate MAR
]

# Surveyor metadata column in raw survey
SURVEYOR_COL = "GD001_Name_of_the_surveyor"

# Valid ordinal scores — imputed values are snapped to nearest valid score
VALID_SCORES = np.array([1.0, 3.0, 5.0])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def round_to_valid(series: pd.Series) -> pd.Series:
    """
    Round each imputed continuous value to the nearest valid ordinal score.

    MICE produces continuous predictions. The indicator scale is ordinal
    (1=Low, 3=Medium, 5=High), so we snap each prediction to the closest
    valid score. This preserves the measurement scale used in CIMDEN scoring.

    Parameters
    ----------
    series : pd.Series of imputed continuous values.

    Returns
    -------
    pd.Series with values in {1, 3, 5}.
    """
    def snap(v):
        if pd.isna(v):
            return np.nan
        return VALID_SCORES[np.argmin(np.abs(VALID_SCORES - v))]
    return series.apply(snap)


def encode_surveyor(df_scores: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add surveyor as an encoded numeric auxiliary variable for MICE.

    Surveyor identity is a confirmed MAR predictor (Step 5 of
    missing_data_diagnostic.py). Including it in the imputation model
    means MICE uses information about who collected the data to inform
    what values are likely to be missing.

    Parameters
    ----------
    df_scores : Scored indicator DataFrame (index aligned with raw_df).
    raw_df    : Raw survey DataFrame containing SURVEYOR_COL.

    Returns
    -------
    df_scores with '_surveyor_encoded' column added.
    """
    if SURVEYOR_COL not in raw_df.columns:
        print(f"  WARNING: '{SURVEYOR_COL}' not found — surveyor not added to MICE")
        return df_scores

    surveyor_raw = raw_df[SURVEYOR_COL].fillna("unknown")
    le = LabelEncoder()
    df_scores = df_scores.copy()
    df_scores["_surveyor_encoded"] = le.fit_transform(surveyor_raw)
    print(
        f"  Surveyor auxiliary variable added: "
        f"{surveyor_raw.nunique()} unique enumerators encoded"
    )
    return df_scores


def run_mice(
    df: pd.DataFrame,
    mar_cols: list,
    index_label: str,
    indicator_cols: list
) -> tuple:
    """
    Run MICE imputation on a scored indicator DataFrame.

    All indicator columns PLUS the surveyor auxiliary variable are passed
    to the imputer together. Cross-indicator relationships and enumerator
    identity jointly inform each imputed value.

    Only MAR-confirmed columns are written back. MCAR-consistent columns
    are included as predictors but restored to their original values after
    imputation — they retain pairwise deletion in the composite.

    Parameters
    ----------
    df             : DataFrame with all indicator columns + surveyor aux.
    mar_cols       : List of column names to impute and write back.
    index_label    : 'EVI' or 'FVI' — used in printed output.
    indicator_cols : All indicator columns (excluding aux variables).

    Returns
    -------
    df_imputed : Full DataFrame with MAR columns imputed.
    log_df     : DataFrame recording imputation counts per column.
    """
    print(f"\n  Running MICE for {index_label}...")
    print(f"  Input shape: {df.shape}")
    print(f"  Imputing {len(mar_cols)} MAR columns using cross-indicator "
          f"relationships + surveyor identity")

    # Columns available as predictors (indicators + surveyor aux)
    all_pred_cols = [c for c in df.columns]

    # Count missing before imputation
    missing_before = df[mar_cols].isna().sum()

    # ── Fit and transform ────────────────────────────────────────────────────
    # BayesianRidge: recommended for MICE on continuous/ordinal data.
    # Handles multicollinearity better than OLS and provides calibrated
    # uncertainty (van Buuren, 2018, §4.5).
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,             # standard MICE cycle count
        random_state=42,         # reproducibility
        initial_strategy="mean", # starting point for iterative process
        verbose=0
    )

    arr_imputed = imputer.fit_transform(df[all_pred_cols])
    df_imputed  = pd.DataFrame(arr_imputed, columns=all_pred_cols, index=df.index)

    # Round imputed values to nearest valid ordinal score {1, 3, 5}
    for col in mar_cols:
        if col in df_imputed.columns:
            df_imputed[col] = round_to_valid(df_imputed[col])

    # Restore non-MAR indicator columns to their original values
    non_mar_ind = [c for c in indicator_cols if c not in mar_cols]
    for col in non_mar_ind:
        df_imputed[col] = df[col]

    # Remove auxiliary variable from output — it was only needed for imputation
    df_imputed = df_imputed.drop(
        columns=[c for c in df_imputed.columns if c.startswith("_")],
        errors="ignore"
    )

    # ── Build log ────────────────────────────────────────────────────────────
    missing_after = df_imputed[mar_cols].isna().sum()
    log_rows = []
    for col in mar_cols:
        n_miss  = int(missing_before[col])
        n_imp   = int(missing_before[col] - missing_after[col])
        pct     = n_miss / len(df) * 100
        print(
            f"    {col:<30}: "
            f"{n_miss:4d} missing ({pct:.1f}%) -> {n_imp} imputed"
        )
        log_rows.append({
            "index":             index_label,
            "column":            col,
            "n_missing_before":  n_miss,
            "pct_missing":       round(pct, 2),
            "n_imputed":         n_imp,
            "n_missing_after":   int(missing_after[col]),
        })

    return df_imputed, pd.DataFrame(log_rows)


def recalculate_composite(
    df: pd.DataFrame,
    indicator_cols: list,
    prefix: str
) -> pd.DataFrame:
    """
    Recalculate composite score, normalised indices, and class after imputation.

    Mirrors the composite logic in stage01_fvi_scoring.py (weighted sum, FVI) and
    stage02_evi_scoring.py (equal-weight mean, EVI) exactly. All downstream stages
    receive consistent column names and scales.

    FVI uses weighted sum normalisation; EVI uses equal-weight mean.
    Both are normalised to 0–1 then rescaled to 1–5.

    Parameters
    ----------
    df             : DataFrame with imputed indicator level columns.
    indicator_cols : List of scored indicator column names.
    prefix         : 'FVI' or 'EVI'.

    Returns
    -------
    df with updated composite, normalised, and class columns.
    """
    def classify(v):
        """Return 1=Low, 2=Medium, 3=High, or NaN if missing."""
        if pd.isna(v):
            return np.nan
        if v <= 1 / 3:
            return 1
        elif v <= 2 / 3:
            return 2
        return 3

    if prefix == "EVI":
        # EVI: equal-weight pairwise mean of 12 indicators (scale 1–5)
        df["EVI_composite"] = df[indicator_cols].mean(axis=1)
        df["EVI_n_valid"]   = df[indicator_cols].notna().sum(axis=1)

        valid = df["EVI_composite"].notna()
        df["EVI_norm_0_1"] = np.nan
        df.loc[valid, "EVI_norm_0_1"] = (df.loc[valid, "EVI_composite"] - 1) / 4.0
        df["EVI_norm_1_5"] = np.nan
        df.loc[valid, "EVI_norm_1_5"] = 1 + df.loc[valid, "EVI_norm_0_1"] * 4
        df["EVI_CLASS"] = df["EVI_norm_0_1"].apply(classify)

    elif prefix == "FVI":
        # FVI: weighted sum using original per-indicator weights
        # Weights: FVI2=8, FVI3=9, FVI4=9, FVI5=8, FVI6=9,
        #          FVI7=9, FVI8=9, FVI9=8, FVI10=8, FVI11=8
        ws_cols = [
            "FVI2_RoofMat_WS", "FVI3_RoofConn_WS", "FVI4_WallMat_WS",
            "FVI5_Elev_WS",    "FVI6_Geom_WS",      "FVI7_Overhang_WS",
            "FVI8_Apron_WS",   "FVI9_Drain_WS",     "FVI10_Floors_WS",
            "FVI11_LWratio_WS"
        ]
        lvl_ws_map = {
            "FVI2_RoofMat_LVL":   ("FVI2_RoofMat_WS",   8),
            "FVI3_RoofConn_LVL":  ("FVI3_RoofConn_WS",  9),
            "FVI4_WallMat_LVL":   ("FVI4_WallMat_WS",   9),
            "FVI5_Elev_LVL":      ("FVI5_Elev_WS",      8),
            "FVI6_Geom_LVL":      ("FVI6_Geom_WS",      9),
            "FVI7_Overhang_LVL":  ("FVI7_Overhang_WS",  9),
            "FVI8_Apron_LVL":     ("FVI8_Apron_WS",     9),
            "FVI9_Drain_LVL":     ("FVI9_Drain_WS",     8),
            "FVI10_Floors_LVL":   ("FVI10_Floors_WS",   8),
            "FVI11_LWratio_LVL":  ("FVI11_LWratio_WS",  8),
        }
        weights     = [8, 9, 9, 8, 9, 9, 9, 8, 8, 8]
        weights_min = [w * 1 for w in weights]
        weights_max = [w * 5 for w in weights]

        # Recalculate weighted scores using imputed level columns
        for lvl_col, (ws_col, w) in lvl_ws_map.items():
            if lvl_col in df.columns:
                df[ws_col] = np.where(df[lvl_col].notna(), w * df[lvl_col], np.nan)

        df["FVI_SCORE"] = df[ws_cols].sum(axis=1, skipna=True, min_count=1)

        df["FVI_MIN_AVAIL"] = sum(
            weights_min[i] * df[ws_cols[i]].notna().astype(int)
            for i in range(len(ws_cols))
        )
        df["FVI_MAX_AVAIL"] = sum(
            weights_max[i] * df[ws_cols[i]].notna().astype(int)
            for i in range(len(ws_cols))
        )

        valid = (
            df["FVI_SCORE"].notna()
            & (df["FVI_MAX_AVAIL"] > df["FVI_MIN_AVAIL"])
        )
        df["FVI_norm_0_1"] = np.nan
        df.loc[valid, "FVI_norm_0_1"] = (
            (df.loc[valid, "FVI_SCORE"] - df.loc[valid, "FVI_MIN_AVAIL"])
            / (df.loc[valid, "FVI_MAX_AVAIL"] - df.loc[valid, "FVI_MIN_AVAIL"])
        )
        df["FVI_norm_1_5"] = np.nan
        df.loc[valid, "FVI_norm_1_5"] = 1 + df.loc[valid, "FVI_norm_0_1"] * 4
        df["FVI_CLASS"] = df["FVI_norm_0_1"].apply(classify)

    return df


def mnar_sensitivity(
    df_ind: pd.DataFrame,
    indicator_cols: list,
    index_label: str
) -> dict:
    """
    Compute class distribution under pairwise, MNAR-high, and MNAR-low.

    Used to populate the MNAR sensitivity CSV for thesis appendix reporting.
    Shows how composite class allocations change under worst-case MNAR
    assumptions compared to pairwise deletion (pre-imputation).

    Parameters
    ----------
    df_ind         : Original (pre-imputation) indicator DataFrame.
    indicator_cols : List of indicator column names.
    index_label    : 'EVI' or 'FVI'.

    Returns
    -------
    dict of scenario -> class distribution counts.
    """
    composite_pair = df_ind[indicator_cols].mean(axis=1)
    composite_high = df_ind[indicator_cols].fillna(5).mean(axis=1)
    composite_low  = df_ind[indicator_cols].fillna(1).mean(axis=1)

    def classify_series(s):
        norm = (s - 1) / 4.0
        return pd.cut(norm, bins=[-np.inf, 1/3, 2/3, np.inf],
                      labels=[1, 2, 3]).astype(float)

    n = len(df_ind)
    rows = []
    for label, s in [
        ("pairwise_mean",   composite_pair),
        ("mnar_high_imp5",  composite_high),
        ("mnar_low_imp1",   composite_low),
    ]:
        c = classify_series(s)
        rows.append({
            "index":          index_label,
            "scenario":       label,
            "mean_composite": round(s.mean(), 3),
            "n_low":          int((c == 1).sum()),
            "pct_low":        round((c == 1).sum() / n * 100, 1),
            "n_medium":       int((c == 2).sum()),
            "pct_medium":     round((c == 2).sum() / n * 100, 1),
            "n_high":         int((c == 3).sum()),
            "pct_high":       round((c == 3).sum() / n * 100, 1),
        })
    return rows


# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    """
    Execute MICE imputation for EVI and FVI indicator columns.

    Steps
    -----
    1. Verify inputs exist (scored CSVs from Stages 1 and 7).
    2. Load raw survey to extract surveyor auxiliary variable.
    3. Run MNAR sensitivity on pre-imputation data (for appendix reporting).
    4. Run MICE for EVI and FVI using MAR-confirmed columns + surveyor aux.
    5. Recalculate composite scores with imputed values.
    6. Write all outputs to outputs/imputation/.
    """
    print("=" * 65)
    print("STAGE 0 — MICE IMPUTATION (Pre-Processing)")
    print("=" * 65)
    print(
        "\nDecision basis: missing_data_diagnostic.py"
        "\n  MCAR rejected (Little's p=0.000 FVI; singular matrix EVI)"
        "\n  MAR confirmed for AUC > 0.65 indicators (Step 3)"
        "\n  Enumerator clustering confirmed (Step 5) — surveyor included"
        "\n  in MICE as auxiliary variable"
        "\n\nMethod: MICE via sklearn.impute.IterativeImputer (BayesianRidge)"
        "\nReference: van Buuren & Groothuis-Oudshoorn (2011)"
        "\n  https://doi.org/10.18637/jss.v045.i03"
    )

    # ── Verify inputs ────────────────────────────────────────────────────────
    for path in [EVI_SCORES_CSV, FVI_SCORES_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required scored CSV not found: {path}\n"
                "Run Stage 1 (stage01_fvi_scoring.py) and Stage 7 "
                "(stage02_evi_scoring.py) before Stage 0."
            )

    # ── Load data ─────────────────────────────────────────────────────────────
    df_evi = pd.read_csv(EVI_SCORES_CSV, encoding="utf-8-sig", low_memory=False)
    df_fvi = pd.read_csv(FVI_SCORES_CSV, encoding="utf-8-sig", low_memory=False)

    raw_df = None
    if RAW_SURVEY_CSV.exists():
        raw_df = pd.read_csv(RAW_SURVEY_CSV, encoding="utf-8-sig", low_memory=False)
        # ── Normalise ward name inconsistency ─────────────────────────────────
        # GD005_VDC_name contains 'tulsipur-ward_14' (hyphen) in the raw CSV.
        # Normalise to underscore to match ward_information.csv and stage01/02.
        _WARD_COL = "GD005_VDC_name"
        if _WARD_COL in raw_df.columns:
            _before = raw_df[_WARD_COL].astype(str).eq("tulsipur-ward_14").sum()
            raw_df[_WARD_COL] = (
                raw_df[_WARD_COL].astype(str)
                .str.replace("tulsipur-ward_14", "tulsipur_ward_14", regex=False)
                .where(raw_df[_WARD_COL].notna(), other=np.nan)
            )
            if _before:
                print(f"  Normalised {_before} rows: "
                      "'tulsipur-ward_14' → 'tulsipur_ward_14'")
        print(f"\nRaw survey loaded: {len(raw_df):,} rows (for surveyor aux)")
    else:
        print(f"\nWARNING: Raw survey not found at {RAW_SURVEY_CSV}")
        print("Imputation will proceed without surveyor auxiliary variable.")

    print(f"EVI scores loaded: {df_evi.shape[0]:,} rows x {df_evi.shape[1]} cols")
    print(f"FVI scores loaded: {df_fvi.shape[0]:,} rows x {df_fvi.shape[1]} cols")

    # ── Identify indicator columns ────────────────────────────────────────────
    evi_indicator_cols = [c for c in df_evi.columns
                          if c.startswith("EVI_D") and not c.startswith("EVI_D1")
                          or c in ["EVI_D1_01", "EVI_D1_02"]]
    # Use the 12 standard EVI indicator columns
    evi_indicator_cols = [
        "EVI_D1_01", "EVI_D1_02",
        "EVI_D2_01", "EVI_D2_02",
        "EVI_D3_01", "EVI_D3_02", "EVI_D3_03", "EVI_D3_04",
        "EVI_D4_01", "EVI_D4_02",
        "EVI_D5_01", "EVI_D5_04",
    ]
    evi_indicator_cols = [c for c in evi_indicator_cols if c in df_evi.columns]

    fvi_indicator_cols = [
        "FVI2_RoofMat_LVL", "FVI3_RoofConn_LVL", "FVI4_WallMat_LVL",
        "FVI5_Elev_LVL",    "FVI6_Geom_LVL",      "FVI7_Overhang_LVL",
        "FVI8_Apron_LVL",   "FVI9_Drain_LVL",     "FVI10_Floors_LVL",
        "FVI11_LWratio_LVL"
    ]
    fvi_indicator_cols = [c for c in fvi_indicator_cols if c in df_fvi.columns]

    # Validate MAR columns are present
    evi_mar_avail = [c for c in EVI_MAR_COLS if c in df_evi.columns]
    fvi_mar_avail = [c for c in FVI_MAR_COLS if c in df_fvi.columns]

    missing_evi = [c for c in EVI_MAR_COLS if c not in df_evi.columns]
    missing_fvi = [c for c in FVI_MAR_COLS if c not in df_fvi.columns]
    if missing_evi:
        print(f"\nWARNING: EVI MAR columns not found: {missing_evi}")
    if missing_fvi:
        print(f"\nWARNING: FVI MAR columns not found: {missing_fvi}")

    # ── MNAR sensitivity (pre-imputation, for appendix) ───────────────────────
    print("\n--- MNAR Sensitivity (pre-imputation, for thesis appendix) ---")
    mnar_rows = []
    mnar_rows += mnar_sensitivity(df_evi, evi_indicator_cols, "EVI")
    mnar_rows += mnar_sensitivity(df_fvi, fvi_indicator_cols, "FVI")

    mnar_df = pd.DataFrame(mnar_rows)
    mnar_df.to_csv(MNAR_CSV, index=False)
    print(f"  Saved MNAR sensitivity table: {MNAR_CSV.name}")

    for lbl in ["EVI", "FVI"]:
        rows = [r for r in mnar_rows if r["index"] == lbl]
        print(f"\n  {lbl} — class distribution under MNAR scenarios:")
        print(f"  {'Scenario':<22} {'Mean':>6}  {'Low%':>6}  {'Med%':>6}  {'High%':>6}")
        print("  " + "-" * 52)
        for r in rows:
            print(
                f"  {r['scenario']:<22} {r['mean_composite']:>6.3f}  "
                f"{r['pct_low']:>5.1f}%  {r['pct_medium']:>5.1f}%  "
                f"{r['pct_high']:>5.1f}%"
            )

    # ── Build imputation DataFrames ────────────────────────────────────────────
    # Extract only indicator columns for MICE input
    df_evi_ind = df_evi[evi_indicator_cols].copy()
    df_fvi_ind = df_fvi[fvi_indicator_cols].copy()

    # Add surveyor as auxiliary variable (confirmed MAR predictor from Step 5)
    if raw_df is not None:
        df_evi_ind = encode_surveyor(df_evi_ind, raw_df)
        df_fvi_ind = encode_surveyor(df_fvi_ind, raw_df)

    # ── Run MICE ───────────────────────────────────────────────────────────────
    print("\n--- EVI MICE Imputation ---")
    df_evi_imp, log_evi = run_mice(
        df_evi_ind, evi_mar_avail, "EVI", evi_indicator_cols
    )

    print("\n--- FVI MICE Imputation ---")
    df_fvi_imp, log_fvi = run_mice(
        df_fvi_ind, fvi_mar_avail, "FVI", fvi_indicator_cols
    )

    # ── Write imputed values back into full DataFrames ────────────────────────
    for col in evi_indicator_cols:
        if col in df_evi_imp.columns:
            df_evi[col] = df_evi_imp[col]

    for col in fvi_indicator_cols:
        if col in df_fvi_imp.columns:
            df_fvi[col] = df_fvi_imp[col]

    # ── Recalculate composites ─────────────────────────────────────────────────
    print("\n--- Recalculating Composite Scores ---")
    df_evi = recalculate_composite(df_evi, evi_indicator_cols, "EVI")
    df_fvi = recalculate_composite(df_fvi, fvi_indicator_cols, "FVI")

    # Print composite summary
    for lbl, df_, norm_col in [
        ("EVI", df_evi, "EVI_norm_0_1"),
        ("FVI", df_fvi, "FVI_norm_0_1"),
    ]:
        if norm_col in df_.columns:
            s = df_[norm_col]
            print(
                f"  {lbl} norm_0_1 after imputation: "
                f"mean={s.mean():.3f}  sd={s.std():.3f}  "
                f"n_valid={s.notna().sum():,}"
            )

    # ── Class distributions after imputation ──────────────────────────────────
    print("\n--- Class Distribution After Imputation ---")
    class_map = {1.0: "Low", 2.0: "Medium", 3.0: "High"}
    for lbl, df_, class_col in [
        ("EVI", df_evi, "EVI_CLASS"),
        ("FVI", df_fvi, "FVI_CLASS"),
    ]:
        if class_col not in df_.columns:
            continue
        n = len(df_)
        print(f"\n  {lbl}:")
        for k, name in class_map.items():
            count = (df_[class_col] == k).sum()
            print(f"    {name:<8}: {count:4,} ({count/n*100:.1f}%)")

    # ── Save outputs ───────────────────────────────────────────────────────────
    df_evi.to_csv(EVI_IMPUTED_CSV, index=False)
    df_fvi.to_csv(FVI_IMPUTED_CSV, index=False)

    log_all = pd.concat([log_evi, log_fvi], ignore_index=True)
    log_all.to_csv(LOG_CSV, index=False)

    print(f"\n{'=' * 65}")
    print("STAGE 0 COMPLETE")
    print(f"{'=' * 65}")
    print(f"  {EVI_IMPUTED_CSV.name:<40} -> {EVI_IMPUTED_CSV.parent.name}/")
    print(f"  {FVI_IMPUTED_CSV.name:<40} -> {FVI_IMPUTED_CSV.parent.name}/")
    print(f"  {LOG_CSV.name:<40} -> {LOG_CSV.parent.name}/")
    print(f"  {MNAR_CSV.name:<40} -> {MNAR_CSV.parent.name}/")
    print(
        "\n  Stages 2–6 should read  : outputs/imputation/imputed_FVI_scores.csv"
        "\n  Stages 8–13 should read : outputs/imputation/imputed_EVI_scores.csv"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================
# run() is called at module level (not only inside __main__) so that
# run_pipeline.py's _run_stage() executes it correctly via importlib's
# exec_module(). The __main__ guard is retained for direct command-line use.

run()

if __name__ == "__main__":
    pass  # run() already called above at module level
