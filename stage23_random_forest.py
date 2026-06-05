"""
Stage 18 | RF | Random Forest Supplementary Validation
=======================================================

Purpose
-------
Train Random Forest regression models for FVI and EVI as a supplementary
robustness check alongside the primary mixed-effects linear models.

Random forest is used here for two specific purposes:
    1. Out-of-sample predictive R² comparison with the linear models, to
       quantify how much explanatory power the linearity assumption costs.
    2. Permutation importance ranking of all candidate variables, to
       triangulate variable selection results from the stepwise procedure.

This stage does NOT replace the primary analysis. The mixed-effects linear
models (Stages 10 and 16) remain the primary results because:
    - The research question is explanatory, not predictive.
    - The MAO framework requires interpretable coefficients and directions.
    - Mixed-effects models partition community-level variance (ICC) which
      random forest cannot reproduce.
    - The lineage from Hendriks & Stokmans (2020) and Saputra et al. (2026)
      uses linear regression throughout.

Random forest is reported as a supplementary check in the thesis appendix or
robustness section, framed as: "To assess variable selection stability, a
random forest was estimated as a comparison model."

Inputs
------
    analysis_dataset.csv         — produced by Stage 05b
    fvi_domains_filtered.csv     — filtered FVI candidates from Stage 06a
    evi_domains_filtered.csv     — filtered EVI candidates from Stage 06a
    fvi_im_predictors.csv        — stepwise-selected FVI predictors (Stage 08)
    evi_im_predictors.csv        — stepwise-selected EVI predictors (Stage 13)

Outputs (written to cwd = outputs/random_forest/)
-------------------------------------------------
    rf_fvi_metrics.csv           — OOS R², RMSE, MAE vs linear model
    rf_evi_metrics.csv           — same for EVI
    rf_fvi_importance.csv        — permutation importance for all FVI candidates
    rf_evi_importance.csv        — permutation importance for all EVI candidates
    rf_fvi_importance.png        — top-20 importance bar chart with stepwise flags
    rf_evi_importance.png        — same for EVI
    rf_fvi_partial_dependence.png — partial dependence for top-5 RF predictors
    rf_evi_partial_dependence.png — same for EVI
    rf_predicted_vs_actual.png   — scatter: RF predictions vs actual index values
    rf_comparison_summary.txt    — human-readable summary for thesis appendix

Methodology
-----------
    Algorithm : RandomForestRegressor (scikit-learn)
    Tuning    : 5-fold cross-validated grid search over n_estimators and
                max_features; best parameters selected by negative MSE.
    Evaluation: Stratified 70/30 train-test split (stratified on vulnerability
                class to preserve class distribution in both sets).
    Importance: Permutation importance on held-out test set (not MDI/impurity,
                which is biased toward high-cardinality features).
    Reference : Breiman, L. (2001). Random forests. Machine Learning, 45(1),
                5-32. https://doi.org/10.1023/A:1010933404324
                Strobl, C. et al. (2007). Bias in random forest variable
                importance measures. BMC Bioinformatics, 8, 25.
                https://doi.org/10.1186/1471-2105-8-25

References
----------
    Breiman (2001) : https://doi.org/10.1023/A:1010933404324
    Strobl et al. (2007) : https://doi.org/10.1186/1471-2105-8-25
    scikit-learn RandomForestRegressor :
        https://scikit-learn.org/stable/modules/generated/
        sklearn.ensemble.RandomForestRegressor.html
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Import analytical constants from pipeline_config ─────────────────────────
import importlib.util as _ilu
_cfg_path = Path(__file__).resolve().parent / "pipeline_config.py"
_cfg_spec  = _ilu.spec_from_file_location("pipeline_config", str(_cfg_path))
_cfg_mod   = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)

FVI_CANDIDATES = _cfg_mod.FVI_CANDIDATES
EVI_CANDIDATES = _cfg_mod.EVI_CANDIDATES
FVI_PRED_LABELS = _cfg_mod.FVI_PRED_LABELS
EVI_PRED_LABELS = _cfg_mod.EVI_PRED_LABELS
CAP_COLS       = _cfg_mod.CAP_COLS

_PIPELINE_CFG = {k: getattr(_cfg_mod, k) for k in [
    "COMMUNITY_COL", "FVI_SCORE_COL", "EVI_SCORE_COL",
    "BNDRY_LM", "BNDRY_MH", "RANDOM_STATE",
]}

RANDOM_STATE = _PIPELINE_CFG["RANDOM_STATE"]

# =============================================================================
# CONSTANTS
# =============================================================================

# Train/test split ratio
TRAIN_SIZE = 0.70   # 70 % training, 30 % test

# Cross-validation folds for hyperparameter tuning
CV_FOLDS = 5

# Hyperparameter grid — kept small for speed; expand for production
PARAM_GRID = {
    "rf__n_estimators": [200, 500],
    "rf__max_features": ["sqrt", 0.33],
    "rf__min_samples_leaf": [5, 10],
}

# Number of permutation importance repeats (more = more stable estimates)
N_PERMUTATION_REPEATS = 20

# Number of top predictors to show in importance chart
TOP_N_CHART = 20

# Variables dropped from the RF candidate pool before fitting due to
# confirmed perfect collinearity (r=1.000 after median imputation).
# DE07 receives an identical median value to DE05 after imputation,
# making them perfectly collinear. DE05 is retained as the more direct
# measure of past flood impact.
PERFECT_COLLINEAR_DROPS = [
    "DE07_negative_impact_future_score",
]

# Output directory
# ── Resolve input directory ──────────────────────────────────────────────────
# Stage 18 can be invoked in two ways:
#   (a) Via run_pipeline.py  → cwd is already outputs/, files are in cwd
#   (b) Standalone from the pipeline root → files are in cwd/outputs/
#   (c) From any other location → falls back to the script's parent/outputs/
#
# _find_input_dir() checks all three locations in order so the script works
# correctly regardless of how it is invoked.

def _find_input_dir():
    """Return the directory containing analysis_dataset.csv."""
    candidates = [
        Path.cwd(),                                      # (a) run_pipeline.py
        Path.cwd() / "outputs",                          # (b) pipeline root
        Path(__file__).resolve().parent / "outputs",     # (c) absolute fallback
    ]
    for d in candidates:
        if (d / "analysis_dataset.csv").exists():
            return d
    return Path.cwd()   # return cwd so the error message shows a useful path


INPUT_DIR = _find_input_dir()
OUT_DIR   = INPUT_DIR / "random_forest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=" * 70)
print("STAGE 18 — RANDOM FOREST SUPPLEMENTARY VALIDATION")
print("=" * 70)
print()

analysis_path = INPUT_DIR / "analysis_dataset.csv"
if not analysis_path.exists():
    sys.exit("ERROR: analysis_dataset.csv not found. Run Stages 01-05b first.")

df = pd.read_csv(analysis_path, encoding="utf-8-sig", low_memory=False)
df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
n_total = len(df)

print(f"Loaded analysis_dataset.csv: {n_total:,} rows")

# Apply data quality caps (mirrors stepwise stages)
for col in CAP_COLS:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s.where(s <= 100)

def clean_col(series):
    """Cast to numeric, treating blanks as NaN."""
    return pd.to_numeric(series.replace(" ", np.nan).infer_objects(copy=False), errors="coerce")


# =============================================================================
# 1b. IMPUTATION AUDIT
# =============================================================================
# Before training, report the imputation status and residual missingness of
# every candidate variable in the analysis dataset. This confirms:
#   (a) which variables were imputed by Stage 05b MICE (complete in dataset)
#   (b) which variables are natively complete (0% missing)
#   (c) which variables have residual missingness and at what rate
#   (d) whether listwise deletion (dropna) will reduce the RF sample
#
# This audit is for transparency and reproducibility. The random forest uses
# complete cases only (via dropna in prepare_XY); if the complete-case count
# equals n_total, no rows are lost to missingness.
#
# Reference: van Buuren (2018, §9.1.4) — variables requiring >50% imputation
# are excluded upstream by Stage 06a before reaching this stage.

# Variables imputed by Stage 05b MICE — complete in analysis_dataset.csv
# (pre-imputation missingness shown in parentheses for reference)
IMPUTED_BY_05B = {
    "Uti_Perc_Neg_expression",      # 17.7% missing before imputation
    "App_Perc_Neg_expression",      # 28.7%
    "AB_Selfefficacy_Neg",          # 17.6%
    "AB_Physical_capacity_Pos",     # 19.4%
    "AB_Physical_capacity_Neg",     # 19.4%
    "AB_Financial_capacity_Pos",    # 19.0%
    "AB_Financial_capacity_Neg",    # 19.6%
    "AB_Location_Pos",              # 19.0%
    "AB_Time_Pos",                  # 19.7%
    "AB_Time_Neg",                  # 19.3%
    "OP_Materials_Pos",             # 19.7%
    "OP_Materials_Neg",             # 19.7%
    "OP_Location_Pos",              # 19.1%
    "OP_Location_Neg",              # 19.1%
    "DE12_shelter_score",           # 16.6%
}

def _run_imputation_audit(candidates_config, index_name):
    """
    Print an imputation status table for all candidate variables.

    For each candidate: reports whether it was imputed by Stage 05b,
    the number of valid (non-missing) values, and the residual missingness
    rate in analysis_dataset.csv. Variables missing in the config but absent
    from the dataset are flagged separately.

    Parameters
    ----------
    candidates_config : dict — {domain: [(col, label, test), ...]}
    index_name        : str  — "FVI" or "EVI" (for display)
    """
    all_candidates = [
        col for domain in candidates_config.values()
        for col, _, _ in domain
    ]

    print()
    print("=" * 72)
    print(f"{index_name} — IMPUTATION AUDIT (analysis_dataset.csv)")
    print("=" * 72)
    print(f"  {'Variable':<52} {'Imputed by':<14} {'n_valid':>8} {'%_miss':>7}")
    print("  " + "-" * 86)

    n_imputed   = 0
    n_complete  = 0
    n_partial   = 0
    n_absent    = 0
    total_rows_lost = 0

    for col in all_candidates:
        if col not in df.columns:
            print(f"  {col:<52} {'ABSENT':14} {'—':>8} {'—':>7}  "
                  f"[not in dataset]")
            n_absent += 1
            continue

        series   = pd.to_numeric(df[col], errors="coerce")
        n_valid  = int(series.notna().sum())
        n_miss   = n_total - n_valid
        pct_miss = n_miss / n_total * 100

        if col in IMPUTED_BY_05B:
            source = "Stage 05b"
            n_imputed += 1
        elif pct_miss == 0.0:
            source = "Complete"
            n_complete += 1
        else:
            source = "Partial"
            n_partial += 1

        flag = ""
        if pct_miss > 0 and col not in IMPUTED_BY_05B:
            flag = f"  [{pct_miss:.1f}% missing — handled by dropna]"

        print(f"  {col:<52} {source:<14} {n_valid:>8,} {pct_miss:>6.1f}%"
              f"{flag}")

    # Complete-case count across all candidates
    present_cols = [c for c in all_candidates if c in df.columns]
    sub = df[present_cols].copy()
    for col in present_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    complete_cases = int(sub.dropna().shape[0])
    rows_lost = n_total - complete_cases

    print()
    print(f"  Summary:")
    print(f"    Total candidates         : {len(all_candidates)}")
    print(f"    Imputed by Stage 05b     : {n_imputed}  "
          f"(complete in analysis_dataset.csv)")
    print(f"    Natively complete (0%)   : {n_complete}")
    print(f"    Partially missing        : {n_partial}  "
          f"(handled by complete-case dropna)")
    print(f"    Absent from dataset      : {n_absent}")
    print(f"    Complete cases (dropna)  : {complete_cases:,} / {n_total:,}  "
          f"({'no rows lost' if rows_lost == 0 else f'{rows_lost} rows lost to listwise deletion'})")
    if rows_lost == 0:
        print(f"    VERDICT: Listwise deletion retains the full sample "
              f"(n={n_total:,}). No imputation needed before RF.")
    else:
        print(f"    VERDICT: Without imputation, {rows_lost} rows would be lost "
              f"to listwise deletion (complete cases = {complete_cases:,}/{n_total:,}).")
        print(f"    Median imputation will be applied to partially missing "
              f"variables immediately after this audit (see Section 1c below).")
        print(f"    This restores the full sample before RF training.")
        # Flag DE07 if present — it will be dropped after imputation
        if index_name == "FVI" and any(
            "DE07_negative_impact_future_score" in str(c)
            for c in candidates_config.values()
            for c, _, _ in (c if isinstance(c, list) else [])
        ):
            print(f"    NOTE: DE07_negative_impact_future_score is listed above "
                  f"as Partial (45.5% missing). After median imputation it "
                  f"becomes r=1.000 with DE05 and will be dropped from the RF "
                  f"candidate pool before training (see PERFECT_COLLINEAR_DROPS).")
    print()


print()
print("Running imputation audit before model training...")
_run_imputation_audit(FVI_CANDIDATES, "FVI")
_run_imputation_audit(EVI_CANDIDATES, "EVI")
# =============================================================================
# 1c. MEDIAN IMPUTATION FOR STRUCTURALLY MISSING FVI CANDIDATES
# =============================================================================
# The flood experience variables (DE03, DE04, DE05, DE07, DE08, DE06, DE09,
# DE13) are missing for households that reported no flood exposure. This is
# MNAR by design: the question did not apply, so the value is structurally
# absent — not a data quality problem.
#
# Without imputation, listwise deletion in prepare_XY() reduces the FVI
# complete-case sample from n=2,993 to ~1,240 (a 58% loss), making the FVI
# random forest unrepresentative of the full survey population and preventing
# meaningful comparison with the linear model fitted on n=2,989.
#
# Median imputation is the correct approach here because:
#   (a) The missingness is MNAR: absent because the household had no flood
#       exposure, so the missing households are plausibly at the low end of
#       flood impact scores. The observed median is a conservative default.
#   (b) MICE would be wrong: it assumes MAR (missingness independent of the
#       unobserved value), but here missingness IS caused by the value
#       (no flood = no depth/impact to report).
#   (c) This imputation is for the RF supplementary stage ONLY. The primary
#       linear models (Stages 07-16) use the original analysis_dataset.csv
#       and are completely unaffected.
#
# Reference: van Buuren (2018, §2.2) — simple imputation is appropriate for
# MNAR data where the direction of missingness is known and theoretically
# grounded.

print("\nApplying median imputation for structurally missing candidates...")
df_rf = df.copy()
_all_rf_candidates = list(set(
    [col for domain in FVI_CANDIDATES.values() for col, _, _ in domain] +
    [col for domain in EVI_CANDIDATES.values() for col, _, _ in domain]
))
_median_imputed = []
for _col in _all_rf_candidates:
    if _col in df_rf.columns:
        _s = pd.to_numeric(df_rf[_col], errors="coerce")
        _n_miss = int(_s.isna().sum())
        if _n_miss > 0:
            _med = _s.median()
            df_rf[_col] = _s.fillna(_med)
            _median_imputed.append((_col, _n_miss, _med))

if _median_imputed:
    print(f"  {len(_median_imputed)} column(s) median-imputed for RF only:")
    for _col, _n, _m in _median_imputed:
        print(f"    {_col:<52} n_imputed={_n:,}  median={_m:.3f}")
    print("  Primary linear models (Stages 07-16) are unaffected.")
else:
    print("  No median imputation needed — all candidates complete.")




# =============================================================================
# 2. LOAD CANDIDATE VARIABLE LISTS
# =============================================================================
# Use Stage 06a filtered lists where available (removes >50% missing vars).
# Falls back to full config candidate pools if not present.

def load_candidates(index, candidates_config):
    """
    Load the candidate variable list for one index.

    If the filtered CSV from Stage 06a exists, use it to rebuild the
    candidate dict (removes variables with >50% missing). Otherwise
    use the full config candidate pool.

    Parameters
    ----------
    index             : str — "fvi" or "evi"
    candidates_config : dict — {domain: [(col, label, test), ...]}

    Returns
    -------
    list of str — column names for this index's candidate pool
    """
    # Stage 06a writes two filtered CSVs:
    #   fvi_candidates_filtered.csv — full candidate pool minus >50% missing vars
    #   fvi_domains_filtered.csv    — post-bivariate domain pool (for Stages 07/08)
    # Random forest must use the CANDIDATE-level list so it evaluates all
    # variables that passed the missingness threshold, not just those that
    # survived stepwise selection. Using the domain-level list would pre-filter
    # by the linear pipeline before the forest has a chance to evaluate them.
    cand_csv   = INPUT_DIR / f"{index.lower()}_candidates_filtered.csv"
    domain_csv = INPUT_DIR / f"{index.lower()}_domains_filtered.csv"

    if cand_csv.exists():
        filtered_vars = set(pd.read_csv(cand_csv)["variable"].tolist())
        cols = [col for domain in candidates_config.values()
                for col, _, _ in domain if col in filtered_vars]
        print(f"  {index.upper()}: loaded candidate-level filtered list "
              f"({len(cols)} variables from stage06a — pre-bivariate pool)")
    elif domain_csv.exists():
        # Fallback: domain-level list (smaller — bivariate-filtered)
        filtered_vars = set(pd.read_csv(domain_csv)["variable"].tolist())
        cols = [col for domain in candidates_config.values()
                for col, _, _ in domain if col in filtered_vars]
        print(f"  {index.upper()}: WARNING — using domain-level filtered list "
              f"({len(cols)} variables). Re-run Stage 06a to generate "
              f"{index.lower()}_candidates_filtered.csv for full RF evaluation.")
    else:
        cols = [col for domain in candidates_config.values()
                for col, _, _ in domain]
        print(f"  {index.upper()}: using full candidate pool ({len(cols)} variables) "
              f"— Stage 06a output not found")
    return cols


print("\nLoading candidate variable lists...")
fvi_candidates = load_candidates("fvi", FVI_CANDIDATES)
evi_candidates = load_candidates("evi", EVI_CANDIDATES)

# Drop perfectly collinear variables before training
_n_fvi_before = len(fvi_candidates)
_n_evi_before = len(evi_candidates)
fvi_candidates = [c for c in fvi_candidates if c not in PERFECT_COLLINEAR_DROPS]
evi_candidates = [c for c in evi_candidates if c not in PERFECT_COLLINEAR_DROPS]
_dropped = _n_fvi_before - len(fvi_candidates)
if _dropped:
    print(f"  Dropped {_dropped} perfectly collinear variable(s) from RF pool:")
    for v in PERFECT_COLLINEAR_DROPS:
        print(f"    {v}  (r=1.000 with DE05 after median imputation)")


def load_stepwise_predictors(index):
    """
    Load the stepwise-selected predictors for one index.

    Reads from the CSV written by Stage 08 (FVI) or Stage 13 (EVI).
    Falls back to an empty list (no flagging) if the file is not found.

    Parameters
    ----------
    index : str — "fvi" or "evi"

    Returns
    -------
    list of str — stepwise-selected predictor column names
    """
    csv_path = INPUT_DIR / f"{index}_im_predictors.csv"
    if csv_path.exists():
        preds = pd.read_csv(csv_path)["predictor"].tolist()
        print(f"  {index.upper()}: loaded {len(preds)} stepwise predictors "
              f"from {csv_path.name}")
        return preds
    print(f"  {index.upper()}: {csv_path.name} not found — "
          f"stepwise flags will be absent from importance chart")
    return []


print("Loading stepwise predictor lists...")
fvi_stepwise = load_stepwise_predictors("fvi")
evi_stepwise = load_stepwise_predictors("evi")


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def prepare_XY(df, candidate_cols, outcome_col):
    """
    Build a complete-case feature matrix and outcome vector.

    Converts all columns to numeric, drops rows with any missing value
    across features or outcome, and returns X and y aligned.

    Parameters
    ----------
    df            : pd.DataFrame — full analysis dataset
    candidate_cols: list of str — predictor column names
    outcome_col   : str — name of the outcome column

    Returns
    -------
    X : pd.DataFrame — feature matrix (complete cases only)
    y : pd.Series    — outcome vector (aligned with X)
    retained_cols : list of str — columns actually present in X
    """
    present = [c for c in candidate_cols if c in df.columns]
    missing_cols = [c for c in candidate_cols if c not in df.columns]
    if missing_cols:
        print(f"    Warning: {len(missing_cols)} candidate columns absent "
              f"from dataset — excluded.")

    sub = df[[outcome_col] + present].copy()
    for col in present:
        sub[col] = clean_col(sub[col])
    sub[outcome_col] = clean_col(sub[outcome_col])

    sub = sub.dropna()
    X = sub[present]
    y = sub[outcome_col]
    return X, y, present


def vulnerability_class(score, bndry_lm, bndry_mh):
    """Map a continuous score to its 1/2/3 vulnerability class."""
    if score < bndry_lm:
        return 1
    elif score < bndry_mh:
        return 2
    return 3


def train_rf(X_train, y_train):
    """
    Fit a RandomForestRegressor with cross-validated hyperparameter tuning.

    Uses a scikit-learn Pipeline (StandardScaler -> RandomForestRegressor)
    with 5-fold CV grid search. StandardScaler is included for compatibility
    but does not affect tree-based models; it simplifies extension to other
    algorithms.

    Parameters
    ----------
    X_train : pd.DataFrame — training features
    y_train : pd.Series    — training outcome

    Returns
    -------
    best_estimator : fitted Pipeline with tuned hyperparameters
    best_params    : dict of best hyperparameter values
    cv_results     : pd.DataFrame of full cross-validation results
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            oob_score=True,
        ))
    ])

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, pd.DataFrame(gs.cv_results_)


def evaluate(model, X_test, y_test):
    """
    Compute out-of-sample prediction metrics.

    Parameters
    ----------
    model  : fitted sklearn estimator
    X_test : pd.DataFrame — held-out features
    y_test : pd.Series    — held-out outcome

    Returns
    -------
    dict with oos_r2, oos_rmse, oos_mae, and predicted values
    """
    y_pred = model.predict(X_test)
    return {
        "oos_r2":   r2_score(y_test, y_pred),
        "oos_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "oos_mae":  mean_absolute_error(y_test, y_pred),
        "y_pred":   y_pred,
    }


def compute_permutation_importance(model, X_test, y_test,
                                   candidate_cols, label_dict,
                                   stepwise_preds):
    """
    Compute permutation importance on the held-out test set.

    Permutation importance (Breiman 2001; Strobl et al. 2007) measures how
    much model performance drops when a feature's values are randomly
    shuffled. Unlike mean decrease in impurity (MDI), it is unbiased toward
    high-cardinality features and is evaluated on unseen data.

    Parameters
    ----------
    model          : fitted sklearn Pipeline
    X_test         : pd.DataFrame — held-out features
    y_test         : pd.Series    — held-out outcome
    candidate_cols : list of str — feature column names in X_test
    label_dict     : dict — {column: human_readable_label}
    stepwise_preds : list of str — columns selected by stepwise regression

    Returns
    -------
    pd.DataFrame sorted by mean importance (descending)
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "variable":        candidate_cols,
        "label":           [label_dict.get(c, c) for c in candidate_cols],
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
        "in_stepwise":     [c in stepwise_preds for c in candidate_cols],
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp_df["rank_rf"] = imp_df.index + 1
    return imp_df


# =============================================================================
# 4. MAIN ANALYSIS LOOP — FVI and EVI
# =============================================================================

results_summary = []
BNDRY_LM = _PIPELINE_CFG["BNDRY_LM"]
BNDRY_MH = _PIPELINE_CFG["BNDRY_MH"]

def _read_linear_r2(index_lower):
    """
    Read the Individual Model adjusted R² and sample size from the stepwise
    results CSV written by Stage 08 (FVI) or Stage 13 (EVI).

    Falls back to hardcoded values from previous runs if the CSV is not found,
    with a warning so the user knows the comparison may be stale.

    Parameters
    ----------
    index_lower : str — "fvi" or "evi"

    Returns
    -------
    (adj_r2, n) : (float, int)
    """
    # Fallback values — updated after each full pipeline run
    _fallbacks = {"fvi": (0.216, 2989), "evi": (0.097, 2989)}
    csv_path = INPUT_DIR / f"{index_lower}_stepwise_stage2_results.csv"
    if csv_path.exists():
        try:
            _df = pd.read_csv(csv_path)
            # Stage 2 CSV has one row per predictor; model-level stats are
            # repeated on every row — take first non-null value.
            adj_r2 = float(_df["adj_r2"].dropna().iloc[0])
            n      = int(_df["n"].dropna().iloc[0])
            print(f"  {index_lower.upper()}: read linear adj. R²={adj_r2:.3f} "
                  f"(n={n:,}) from {csv_path.name}")
            return adj_r2, n
        except Exception as e:
            print(f"  {index_lower.upper()}: could not parse {csv_path.name} "
                  f"({e}) — using fallback values")
    else:
        print(f"  {index_lower.upper()}: {csv_path.name} not found — "
              f"using fallback linear R²={_fallbacks[index_lower][0]:.3f}")
    return _fallbacks[index_lower]


print("\nReading linear model R² from stepwise outputs...")
fvi_linear_r2, fvi_linear_n = _read_linear_r2("fvi")
evi_linear_r2, evi_linear_n = _read_linear_r2("evi")

index_configs = [
    {
        "index":      "FVI",
        "outcome":    _PIPELINE_CFG["FVI_SCORE_COL"],
        "candidates": fvi_candidates,
        "stepwise":   fvi_stepwise,
        "labels":     FVI_PRED_LABELS,
        "linear_r2":  fvi_linear_r2,
        "linear_n":   fvi_linear_n,
    },
    {
        "index":      "EVI",
        "outcome":    _PIPELINE_CFG["EVI_SCORE_COL"],
        "candidates": evi_candidates,
        "stepwise":   evi_stepwise,
        "labels":     EVI_PRED_LABELS,
        "linear_r2":  evi_linear_r2,
        "linear_n":   evi_linear_n,
    },
]

fig_scatter, axes_scatter = plt.subplots(1, 2, figsize=(13, 6))
fig_scatter.suptitle(
    "Random Forest: Predicted vs Actual Vulnerability Scores",
    fontsize=13, fontweight="bold"
)

for idx_cfg, ax_scatter in zip(index_configs, axes_scatter):
    index      = idx_cfg["index"]
    outcome    = idx_cfg["outcome"]
    candidates = idx_cfg["candidates"]
    stepwise   = idx_cfg["stepwise"]
    labels     = idx_cfg["labels"]
    linear_r2  = idx_cfg["linear_r2"]
    linear_n   = idx_cfg["linear_n"]

    print()
    print("=" * 70)
    print(f"{index} — RANDOM FOREST ANALYSIS")
    print("=" * 70)

    # ── 4a. Prepare data ──────────────────────────────────────────────────────
    X, y, retained = prepare_XY(df_rf, candidates, outcome)
    n_complete = len(X)
    print(f"  Complete cases: {n_complete:,} "
          f"({n_complete/n_total*100:.1f}% of {n_total:,} total)")
    print(f"  Features:       {len(retained)}")

    # Stratified split on vulnerability class
    classes = y.apply(lambda s: vulnerability_class(s, BNDRY_LM, BNDRY_MH))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=classes,
    )
    print(f"  Train n: {len(X_train):,}   Test n: {len(X_test):,}")

    # ── 4b. Hyperparameter tuning and model fitting ───────────────────────────
    print(f"\n  Fitting Random Forest with {CV_FOLDS}-fold CV grid search...")
    best_model, best_params, cv_results = train_rf(X_train, y_train)
    rf_estimator = best_model.named_steps["rf"]

    print(f"  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  OOB R² (training):  {rf_estimator.oob_score_:.3f}")

    # ── 4c. Out-of-sample evaluation ─────────────────────────────────────────
    metrics = evaluate(best_model, X_test, y_test)
    print(f"\n  Out-of-sample performance (n_test={len(X_test):,}):")
    print(f"    OOS R²   : {metrics['oos_r2']:.3f}")
    print(f"    OOS RMSE : {metrics['oos_rmse']:.4f}")
    print(f"    OOS MAE  : {metrics['oos_mae']:.4f}")
    print(f"\n  Linear model (Stage 08/13) adjusted R²: {linear_r2:.3f} "
          f"(n={linear_n:,}, full sample)")
    r2_gain = metrics["oos_r2"] - linear_r2
    print(f"  RF OOS R² gain over linear: {r2_gain:+.3f}")

    # Save metrics CSV
    metrics_df = pd.DataFrame([{
        "index":           index,
        "rf_oos_r2":       metrics["oos_r2"],
        "rf_oob_r2":       rf_estimator.oob_score_,
        "rf_oos_rmse":     metrics["oos_rmse"],
        "rf_oos_mae":      metrics["oos_mae"],
        "linear_adj_r2":   linear_r2,
        "linear_n":        linear_n,
        "rf_n_train":      len(X_train),
        "rf_n_test":       len(X_test),
        "rf_n_features":   len(retained),
        "rf_n_estimators": best_params["rf__n_estimators"],
        "rf_max_features": best_params["rf__max_features"],
        "rf_min_samples_leaf": best_params["rf__min_samples_leaf"],
        "cv_folds":        CV_FOLDS,
        "train_size":      TRAIN_SIZE,
        "random_state":    RANDOM_STATE,
        "r2_gain_rf_vs_linear": r2_gain,
    }])
    metrics_path = OUT_DIR / f"rf_{index.lower()}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics saved: {metrics_path.name}")

    # ── 4d. Permutation importance ────────────────────────────────────────────
    print(f"\n  Computing permutation importance "
          f"({N_PERMUTATION_REPEATS} repeats on test set)...")
    imp_df = compute_permutation_importance(
        best_model, X_test, y_test,
        retained, labels, stepwise
    )
    imp_path = OUT_DIR / f"rf_{index.lower()}_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  Importance saved: {imp_path.name}")

    # Agreement with stepwise: how many of the top N RF vars are in stepwise?
    n_stepwise = len(stepwise)
    if n_stepwise > 0:
        top_n = min(n_stepwise, len(imp_df))
        top_rf_vars = set(imp_df.head(top_n)["variable"].tolist())
        stepwise_set = set(stepwise)
        overlap = len(top_rf_vars & stepwise_set)
        print(f"\n  Variable selection agreement:")
        print(f"    Stepwise selected: {n_stepwise} predictors")
        print(f"    Top-{top_n} RF variables in stepwise selection: "
              f"{overlap}/{top_n} ({overlap/top_n*100:.0f}%)")

        in_rf_not_stepwise = top_rf_vars - stepwise_set
        in_stepwise_not_rf = stepwise_set - top_rf_vars
        if in_rf_not_stepwise:
            print(f"    In top-{top_n} RF but NOT in stepwise: "
                  f"{sorted(in_rf_not_stepwise)}")
        if in_stepwise_not_rf:
            print(f"    In stepwise but NOT in top-{top_n} RF: "
                  f"{sorted(in_stepwise_not_rf)}")

    # ── 4e. Importance chart ──────────────────────────────────────────────────
    plot_df = imp_df.head(TOP_N_CHART).copy()
    plot_df["short_label"] = plot_df["label"].apply(
        lambda x: x[:45] + "..." if len(x) > 48 else x
    )

    colours = ["#1F78B4" if sw else "#A8D1E7"
               for sw in plot_df["in_stepwise"]]

    fig_imp, ax_imp = plt.subplots(figsize=(11, 8))
    bars = ax_imp.barh(
        range(len(plot_df)),
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        color=colours,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "#444444"},
    )
    ax_imp.set_yticks(range(len(plot_df)))
    ax_imp.set_yticklabels(plot_df["short_label"], fontsize=9)
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Permutation Importance\n(mean decrease in R², test set)",
                      fontsize=10)
    ax_imp.set_title(
        f"{index} — Random Forest Permutation Importance (top {TOP_N_CHART})\n"
        f"OOS R²={metrics['oos_r2']:.3f}  |  "
        f"Linear model adj. R²={linear_r2:.3f}  |  "
        f"n_test={len(X_test):,}",
        fontsize=11, fontweight="bold", pad=12,
    )

    # Legend: stepwise flag
    legend_handles = [
        mpatches.Patch(color="#1F78B4",
                       label="Also selected by stepwise regression"),
        mpatches.Patch(color="#A8D1E7",
                       label="RF top-20 only (not in stepwise)"),
    ]
    ax_imp.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax_imp.axvline(0, color="black", linewidth=0.7)

    fig_imp.tight_layout()
    imp_fig_path = OUT_DIR / f"rf_{index.lower()}_importance.png"
    fig_imp.savefig(imp_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig_imp)
    print(f"  Importance chart saved: {imp_fig_path.name}")

    # ── 4f. Partial dependence plots — top 5 RF predictors ───────────────────
    # Partial dependence (PD) shows the marginal effect of each predictor
    # averaged over the distribution of all other features. For a random
    # forest this reveals non-linear relationships that linear regression
    # cannot capture (Friedman 2001, Ann. Stat. 29(5): 1189-1232).
    top5_cols = imp_df.head(5)["variable"].tolist()
    top5_indices = [retained.index(c) for c in top5_cols if c in retained]
    top5_labels  = [labels.get(c, c) for c in top5_cols if c in retained]

    if top5_indices:
        try:
            fig_pd, axes_pd = plt.subplots(1, len(top5_indices),
                                           figsize=(4 * len(top5_indices), 4))
            if len(top5_indices) == 1:
                axes_pd = [axes_pd]

            display = PartialDependenceDisplay.from_estimator(
                best_model,
                X_train,
                features=top5_indices,
                feature_names=retained,
                ax=axes_pd,
                line_kw={"color": "#1F78B4", "linewidth": 2},
            )
            for ax_pd, lbl in zip(axes_pd, top5_labels):
                ax_pd.set_title(lbl[:40], fontsize=9, fontweight="bold")
                ax_pd.set_xlabel(ax_pd.get_xlabel(), fontsize=8)
                ax_pd.set_ylabel("Partial dependence", fontsize=8)

            fig_pd.suptitle(
                f"{index} — Partial Dependence: Top 5 RF Predictors\n"
                "(marginal effect averaged over all other features; "
                "non-linearity indicates where linear assumption may cost R²)",
                fontsize=10, fontweight="bold",
            )
            fig_pd.tight_layout()
            pd_path = OUT_DIR / f"rf_{index.lower()}_partial_dependence.png"
            fig_pd.savefig(pd_path, dpi=200, bbox_inches="tight")
            plt.close(fig_pd)
            print(f"  Partial dependence saved: {pd_path.name}")
        except Exception as e:
            print(f"  Partial dependence plot skipped: {e}")

    # ── 4g. Predicted vs actual scatter ──────────────────────────────────────
    y_pred = metrics["y_pred"]
    ax_scatter.scatter(y_test, y_pred, alpha=0.25, s=8,
                       color="#1F78B4", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    ax_scatter.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax_scatter.set_xlabel(f"Actual {index}_norm_1_5", fontsize=10)
    ax_scatter.set_ylabel(f"Predicted {index}_norm_1_5", fontsize=10)
    ax_scatter.set_title(
        f"{index}  OOS R²={metrics['oos_r2']:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax_scatter.legend(fontsize=9)

    # Collect summary for text report
    results_summary.append({
        "index":      index,
        "metrics":    metrics,
        "linear_r2":  linear_r2,
        "linear_n":   linear_n,
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "n_features": len(retained),
        "best_params": best_params,
        "oob_r2":     rf_estimator.oob_score_,
        "imp_df":     imp_df,
        "stepwise":   stepwise,
    })

# Save predicted vs actual figure
fig_scatter.tight_layout()
scatter_path = OUT_DIR / "rf_predicted_vs_actual.png"
fig_scatter.savefig(scatter_path, dpi=200, bbox_inches="tight")
plt.close(fig_scatter)
print(f"\nPredicted vs actual scatter saved: {scatter_path.name}")


# =============================================================================
# 5. WRITE PLAIN-TEXT SUMMARY REPORT
# =============================================================================

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("STAGE 18 — RANDOM FOREST SUPPLEMENTARY VALIDATION REPORT")
summary_lines.append("=" * 70)
summary_lines.append("")
summary_lines.append("PURPOSE")
summary_lines.append("-" * 40)
summary_lines.append(
    "This report summarises random forest (RF) models estimated as a "
    "supplementary robustness check alongside the primary mixed-effects "
    "linear models (Stages 10 and 16). RF results are reported in the "
    "thesis appendix or robustness section and do not replace the primary "
    "interpretable analysis."
)
summary_lines.append("")
summary_lines.append("METHODS")
summary_lines.append("-" * 40)
summary_lines.append(f"Algorithm      : RandomForestRegressor (scikit-learn)")
summary_lines.append(f"Train/test     : {int(TRAIN_SIZE*100)}/{int((1-TRAIN_SIZE)*100)} split "
                     f"(stratified on vulnerability class)")
summary_lines.append(f"Tuning         : {CV_FOLDS}-fold CV grid search "
                     f"(n_estimators, max_features, min_samples_leaf)")
summary_lines.append(f"Importance     : Permutation importance on held-out test set "
                     f"({N_PERMUTATION_REPEATS} repeats)")
summary_lines.append(f"Random seed    : {RANDOM_STATE}")
summary_lines.append("")
summary_lines.append("References:")
summary_lines.append("  Breiman (2001). Random forests. Machine Learning, 45(1), 5-32.")
summary_lines.append("    https://doi.org/10.1023/A:1010933404324")
summary_lines.append("  Strobl et al. (2007). Bias in random forest variable importance.")
summary_lines.append("    BMC Bioinformatics, 8, 25.")
summary_lines.append("    https://doi.org/10.1186/1471-2105-8-25")
summary_lines.append("")

for rs in results_summary:
    index   = rs["index"]
    metrics = rs["metrics"]
    imp_df  = rs["imp_df"]
    stepwise = rs["stepwise"]

    summary_lines.append("=" * 70)
    summary_lines.append(f"{index} RESULTS")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("Model fit")
    summary_lines.append("-" * 40)
    summary_lines.append(f"  RF OOS R²                 : {metrics['oos_r2']:.3f}")
    summary_lines.append(f"  RF OOB R² (train)         : {rs['oob_r2']:.3f}")
    summary_lines.append(f"  RF OOS RMSE               : {metrics['oos_rmse']:.4f}")
    summary_lines.append(f"  RF OOS MAE                : {metrics['oos_mae']:.4f}")
    summary_lines.append(f"  Linear model adj. R²      : {rs['linear_r2']:.3f} "
                         f"(n={rs['linear_n']:,}, full sample)")
    r2_gain = metrics["oos_r2"] - rs["linear_r2"]
    summary_lines.append(f"  R² gain (RF vs linear)    : {r2_gain:+.3f}")
    summary_lines.append(f"  N train / N test          : "
                         f"{rs['n_train']:,} / {rs['n_test']:,}")
    summary_lines.append(f"  N candidate features      : {rs['n_features']}")
    summary_lines.append(f"  Best n_estimators         : "
                         f"{rs['best_params']['rf__n_estimators']}")
    summary_lines.append(f"  Best max_features         : "
                         f"{rs['best_params']['rf__max_features']}")
    summary_lines.append(f"  Best min_samples_leaf     : "
                         f"{rs['best_params']['rf__min_samples_leaf']}")
    summary_lines.append("")
    summary_lines.append("Top 15 predictors by permutation importance")
    summary_lines.append("-" * 40)
    summary_lines.append(
        f"  {'Rank':<5} {'Variable':<52} {'Importance':>12} "
        f"{'Stepwise':>10}"
    )
    summary_lines.append("  " + "-" * 82)
    for _, row in imp_df.head(15).iterrows():
        sw_flag = "YES" if row["in_stepwise"] else "-"
        summary_lines.append(
            f"  {int(row['rank_rf']):<5} {row['variable']:<52} "
            f"{row['importance_mean']:>12.5f}  {sw_flag:>8}"
        )

    if stepwise:
        n_top = len(stepwise)
        top_rf_vars = set(imp_df.head(n_top)["variable"].tolist())
        overlap = len(top_rf_vars & set(stepwise))
        summary_lines.append("")
        summary_lines.append("Variable selection agreement")
        summary_lines.append("-" * 40)
        summary_lines.append(f"  Stepwise predictors     : {len(stepwise)}")
        summary_lines.append(
            f"  In top-{n_top} RF and stepwise : "
            f"{overlap}/{n_top} ({overlap/n_top*100:.0f}%)"
        )
        only_rf   = top_rf_vars - set(stepwise)
        only_step = set(stepwise) - top_rf_vars
        if only_rf:
            summary_lines.append(f"  Top-{n_top} RF only (not stepwise) :")
            for v in sorted(only_rf):
                summary_lines.append(f"    {v}")
        if only_step:
            summary_lines.append(f"  Stepwise only (not top-{n_top} RF) :")
            for v in sorted(only_step):
                summary_lines.append(f"    {v}")
    summary_lines.append("")

summary_lines.append("=" * 70)
summary_lines.append("INTERPRETATION GUIDANCE")
summary_lines.append("=" * 70)
summary_lines.append("")
summary_lines.append(
    "1. R² comparison: The RF OOS R² is estimated on a 30% held-out test set "
    "and is directly comparable to a cross-validated R². The linear model "
    "adjusted R² is estimated on the full sample; compare with caution. "
    "A large RF gain (>0.05) indicates non-linearities or interactions that "
    "the linear model misses. A small gain confirms that linearity is a "
    "reasonable assumption for this dataset."
)
summary_lines.append("")
summary_lines.append(
    "2. Importance agreement: High overlap between the top-N RF predictors "
    "and the stepwise-selected predictors provides convergent validity for "
    "the variable selection procedure. Predictors that rank highly in RF "
    "but were dropped by stepwise may have non-linear effects or were "
    "suppressed by collinearity in the linear framework."
)
summary_lines.append("")
summary_lines.append(
    "3. Partial dependence plots: Non-linear partial dependence curves for "
    "the top predictors indicate where the linear model imposes the strongest "
    "approximation. Threshold effects (e.g. a kink in the education curve) "
    "suggest that the linear coefficient is averaging over heterogeneous "
    "effects at different score levels."
)
summary_lines.append("")
summary_lines.append(
    "4. Primary analysis: These RF results supplement but do not replace the "
    "mixed-effects linear models in Stages 10 and 16, which provide "
    "interpretable coefficients, directional effects, significance tests, "
    "community-level random intercepts, and ICC estimates required for the "
    "MAO framework interpretation."
)

report_path = OUT_DIR / "rf_comparison_summary.txt"
report_path.write_text("\n".join(summary_lines), encoding="utf-8")

# =============================================================================
# 6. CONSOLE SUMMARY
# =============================================================================

print()
print("=" * 70)
print("STAGE 18 COMPLETE")
print("=" * 70)
for rs in results_summary:
    index   = rs["index"]
    metrics = rs["metrics"]
    r2_gain = metrics["oos_r2"] - rs["linear_r2"]
    print(f"  {index}:")
    print(f"    RF OOS R²    = {metrics['oos_r2']:.3f}  "
          f"(linear adj. R² = {rs['linear_r2']:.3f}, "
          f"gain = {r2_gain:+.3f})")
    print(f"    OOB R²       = {rs['oob_r2']:.3f}")
    imp_df = rs["imp_df"]
    top3 = ", ".join(imp_df.head(3)["variable"].tolist())
    print(f"    Top-3 by RF importance: {top3}")

print()
print(f"All outputs saved to: {OUT_DIR}")
print(f"  rf_fvi_metrics.csv            — model performance comparison")
print(f"  rf_evi_metrics.csv            — model performance comparison")
print(f"  rf_fvi_importance.csv         — permutation importance, all vars")
print(f"  rf_evi_importance.csv         — permutation importance, all vars")
print(f"  rf_fvi_importance.png         — importance chart with stepwise flags")
print(f"  rf_evi_importance.png         — importance chart with stepwise flags")
print(f"  rf_fvi_partial_dependence.png — partial dependence, top 5 predictors")
print(f"  rf_evi_partial_dependence.png — partial dependence, top 5 predictors")
print(f"  rf_predicted_vs_actual.png    — predicted vs actual scatter")
print(f"  rf_comparison_summary.txt     — full text report for thesis appendix")
