"""
Stage 18b | RF | Random Forest Validation & Calibration
========================================================

Purpose
-------
Validation and calibration checks for the random forest models estimated
in Stage 18. These checks are necessary before reporting RF results in a
thesis because they establish whether the OOS R² is stable, whether the
importance rankings are reliable, and whether the forest is well-calibrated
across the vulnerability distribution.

This stage reads the analysis_dataset.csv and candidate lists directly —
it does NOT read Stage 18 model objects, as scikit-learn Pipelines are not
persisted to disk by Stage 18. All models are re-fitted here with the same
hyperparameters found optimal in Stage 18, so results are exactly reproducible.

Checks performed
----------------
    1. Nested cross-validation
       Evaluates the full modelling procedure (split + tune + predict) across
       10 folds, reporting mean ± SD of OOS R². This confirms whether the
       single-split R² reported in Stage 18 is stable across different data
       partitions, addressing the risk that RANDOM_STATE=42 produced a
       particularly lucky or unlucky split.
       Reference: Cawley & Talbot (2010). JMLR, 11, 2079-2107.
                  https://www.jmlr.org/papers/v11/cawley10a.html

    2. Learning curves
       Trains the model on progressively larger subsets of training data
       (10%–100% in 10 steps) and plots training vs cross-validated R².
       Confirms whether n=2,993 is sufficient for stable forest performance.
       A plateau in the validation curve before 100% of training data confirms
       sample size adequacy.
       Reference: Figueroa et al. (2012). BMC Med Inform Decis Mak, 12, 8.
                  https://doi.org/10.1186/1472-6947-12-8

    3. Hyperparameter sensitivity
       Reports OOS R² across all grid combinations evaluated in Stage 18's
       tuning. Quantifies how much the result depends on hyperparameter choice.
       A narrow range confirms robustness to tuning decisions.

    4. Community residual analysis
       Computes RF residuals per community and plots them as a caterpillar
       chart alongside the mixed-effects random intercepts from Stages 10/16.
       If RF residuals show systematic community-level patterns comparable to
       the mixed-effects random intercepts, this confirms that community-level
       clustering is genuine and not captured by individual-level predictors —
       directly justifying the mixed-effects model as the primary analysis.

    5. Calibration
       Bins predicted scores into deciles and plots mean actual vs mean
       predicted per bin (reliability diagram). Quantifies tail under-prediction
       (regression-toward-the-mean bias common in random forests). Important
       for interpreting the forest as a vulnerability assessment tool.

    6. Permutation importance stability
       Re-runs permutation importance with 50 repeats on the test set and
       computes 95% confidence intervals per variable. Flags pairs of adjacent-
       ranked variables whose intervals overlap — these should be reported as
       "similarly important" rather than assigned strict ordinal ranks.
       Reference: Strobl et al. (2007). BMC Bioinformatics, 8, 25.
                  https://doi.org/10.1186/1471-2105-8-25

    7. MDI vs permutation importance comparison
       Computes mean decrease in impurity (MDI) alongside permutation
       importance and plots a rank-comparison scatter. Divergence for specific
       variables — especially continuous predictors ranking much higher under
       MDI — flags the Strobl et al. (2007) bias and confirms that permutation
       importance is the correct measure to report.

    8. Predictor correlation heatmap
       Plots the Pearson correlation matrix for all candidate predictors.
       Variables with |r| > 0.6 are flagged as potentially causing unstable
       importance estimates (Strobl et al. 2008).
       Reference: Strobl et al. (2008). BMC Bioinformatics, 9, 307.
                  https://doi.org/10.1186/1471-2105-9-307

Inputs
------
    analysis_dataset.csv          — produced by Stage 05b
    fvi_domains_filtered.csv      — filtered FVI candidates (Stage 06a)
    evi_domains_filtered.csv      — filtered EVI candidates (Stage 06a)
    fvi_im_predictors.csv         — stepwise-selected FVI predictors (Stage 08)
    evi_im_predictors.csv         — stepwise-selected EVI predictors (Stage 13)
    fvi_random_intercepts.csv     — community random intercepts (Stage 10)
    evi_random_intercepts.csv     — community random intercepts (Stage 16)

Outputs (written to outputs/random_forest/validation/)
-------------------------------------------------------
    rfval_nested_cv.csv                 — nested CV R² per fold + mean/SD
    rfval_nested_cv.png                 — violin + strip plot of fold R²
    rfval_learning_curves.png           — training vs validation R² vs n
    rfval_hyperparameter_sensitivity.csv — R² across all grid combinations
    rfval_hyperparameter_sensitivity.png — dot plot of grid R²
    rfval_community_residuals.png       — RF residuals vs ME intercepts
    rfval_community_residuals.csv       — data underlying the above
    rfval_calibration.png               — reliability diagram (decile bins)
    rfval_calibration.csv               — calibration bin data
    rfval_importance_stability.png      — 95% CI chart for top-20 variables
    rfval_importance_stability.csv      — all variables with CI bounds
    rfval_mdi_vs_permutation.png        — rank comparison scatter
    rfval_mdi_vs_permutation.csv        — MDI and permutation ranks
    rfval_predictor_correlation.png     — heatmap with high-r flags
    rfval_validation_summary.txt        — plain-text report for thesis appendix

References
----------
    Breiman (2001)       : https://doi.org/10.1023/A:1010933404324
    Strobl et al. (2007) : https://doi.org/10.1186/1471-2105-8-25
    Strobl et al. (2008) : https://doi.org/10.1186/1471-2105-9-307
    Cawley & Talbot (2010): https://www.jmlr.org/papers/v11/cawley10a.html
    Figueroa et al. (2012): https://doi.org/10.1186/1472-6947-12-8
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
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV,
    cross_val_score, learning_curve,
)
from sklearn.inspection import permutation_importance
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

FVI_CANDIDATES  = _cfg_mod.FVI_CANDIDATES
EVI_CANDIDATES  = _cfg_mod.EVI_CANDIDATES
FVI_PRED_LABELS = _cfg_mod.FVI_PRED_LABELS
EVI_PRED_LABELS = _cfg_mod.EVI_PRED_LABELS
CAP_COLS        = _cfg_mod.CAP_COLS

_PIPELINE_CFG = {k: getattr(_cfg_mod, k) for k in [
    "COMMUNITY_COL", "FVI_SCORE_COL", "EVI_SCORE_COL",
    "BNDRY_LM", "BNDRY_MH", "RANDOM_STATE",
]}

RANDOM_STATE = _PIPELINE_CFG["RANDOM_STATE"]
BNDRY_LM     = _PIPELINE_CFG["BNDRY_LM"]
BNDRY_MH     = _PIPELINE_CFG["BNDRY_MH"]

# =============================================================================
# CONSTANTS
# =============================================================================

# Same split and tuning parameters as Stage 18 — must match for comparability
TRAIN_SIZE = 0.70
CV_FOLDS   = 5

# Best hyperparameters from Stage 18 grid search (used to re-fit efficiently)
# These match the best_params found in Stage 18 for both indices.
BEST_PARAMS = {
    "rf__n_estimators":      500,
    "rf__max_features":      0.33,
    "rf__min_samples_leaf":  5,
}

# Full grid — same as Stage 18 — for hyperparameter sensitivity check
PARAM_GRID = {
    "rf__n_estimators":     [200, 500],
    "rf__max_features":     ["sqrt", 0.33],
    "rf__min_samples_leaf": [5, 10],
}

# Nested CV: outer folds for stable R² estimation
NESTED_CV_FOLDS = 10

# Permutation repeats for stability check (more than Stage 18)
N_STABILITY_REPEATS = 50

# Learning curve training sizes (fractions of training set)
LEARNING_CURVE_SIZES = np.linspace(0.10, 1.0, 10)

# Calibration bins
N_CALIBRATION_BINS = 10

# Correlation threshold for flagging collinear predictors
CORR_THRESHOLD = 0.60

# VIF warning threshold — predictors above this are flagged in the
# collinearity report (O'Brien 2007; Hair et al. 2019)
VIF_THRESHOLD = 5.0

# Condition number threshold above which severe multicollinearity is
# indicated (Belsley, Kuh & Welsch 1980)
CONDITION_NUM_THRESHOLD = 30.0

# Variables dropped from the RF candidate pool before fitting due to
# confirmed perfect collinearity (r=1.000 after median imputation).
# DE07_negative_impact_future_score receives identical median value to
# DE05_negative_impact_score — they are effectively duplicates in the
# imputed dataset. DE05 is retained as the more direct impact measure.
PERFECT_COLLINEAR_DROPS = [
    "DE07_negative_impact_future_score",
]

# =============================================================================
# PATH RESOLUTION — mirrors Stage 18 exactly
# =============================================================================

def _find_input_dir():
    """Return the directory containing analysis_dataset.csv."""
    candidates = [
        Path.cwd(),
        Path.cwd() / "outputs",
        Path(__file__).resolve().parent / "outputs",
    ]
    for d in candidates:
        if (d / "analysis_dataset.csv").exists():
            return d
    return Path.cwd()


INPUT_DIR = _find_input_dir()
OUT_DIR   = INPUT_DIR / "random_forest" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STARTUP
# =============================================================================

print("=" * 70)
print("STAGE 18b — RANDOM FOREST VALIDATION & CALIBRATION")
print("=" * 70)
print()

analysis_path = INPUT_DIR / "analysis_dataset.csv"
if not analysis_path.exists():
    sys.exit(
        "ERROR: analysis_dataset.csv not found.\n"
        "Run Stages 01-05b before Stage 18b."
    )

df = pd.read_csv(analysis_path, encoding="utf-8-sig", low_memory=False)
df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
n_total = len(df)
print(f"Loaded analysis_dataset.csv: {n_total:,} rows")

# Apply data quality caps (mirrors Stage 18)
for col in CAP_COLS:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s.where(s <= 100)


def clean_col(series):
    """Cast to numeric, replacing blanks with NaN."""
    return pd.to_numeric(series.replace(" ", np.nan).infer_objects(copy=False), errors="coerce")


# =============================================================================
# DATA LOADING HELPERS — mirrors Stage 18
# =============================================================================

def load_candidates(index, candidates_config):
    """
    Load the candidate variable list for one index.

    Reads from fvi/evi_candidates_filtered.csv — the full pre-bivariate
    candidate pool minus any variables removed by Stage 06a for >50% missing.
    Falls back to the domain-level filtered list (smaller, post-bivariate)
    with a warning, or to the full config pool if neither file exists.
    """
    # Stage 06a writes candidate-level filtered list for RF use
    cand_csv   = INPUT_DIR / f"{index.lower()}_candidates_filtered.csv"
    # Domain-level filtered list (for Stages 07/08 — post-bivariate, smaller)
    domain_csv = INPUT_DIR / f"{index.lower()}_domains_filtered.csv"

    if cand_csv.exists():
        filtered_vars = set(pd.read_csv(cand_csv)["variable"].tolist())
        cols = [col for domain in candidates_config.values()
                for col, _, _ in domain if col in filtered_vars]
        print(f"  {index.upper()}: {len(cols)} candidates "
              f"(stage06a candidate-level filtered — pre-bivariate pool)")
    elif domain_csv.exists():
        filtered_vars = set(pd.read_csv(domain_csv)["variable"].tolist())
        cols = [col for domain in candidates_config.values()
                for col, _, _ in domain if col in filtered_vars]
        print(f"  {index.upper()}: WARNING — using domain-level filtered list "
              f"({len(cols)} variables). Re-run Stage 06a to generate "
              f"{index.lower()}_candidates_filtered.csv for full RF evaluation.")
    else:
        cols = [col for domain in candidates_config.values()
                for col, _, _ in domain]
        print(f"  {index.upper()}: {len(cols)} candidates (full pool, no filter found)")
    return cols


def load_stepwise_predictors(index):
    """Load stepwise-selected predictors for one index."""
    csv_path = INPUT_DIR / f"{index}_im_predictors.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)["predictor"].tolist()
    return []


def load_community_intercepts(index):
    """
    Load mixed-effects random intercepts from Stage 10 (FVI) or Stage 16 (EVI).

    Returns a DataFrame with columns [community, u_j] or None if not found.
    """
    csv_path = INPUT_DIR / f"{index.lower()}_random_intercepts.csv"
    if csv_path.exists():
        ri = pd.read_csv(csv_path)
        # Normalise column names — Stage 10/16 write 'community' and 'u_j'
        ri.columns = ri.columns.str.strip()
        if "community" in ri.columns and "u_j" in ri.columns:
            return ri[["community", "u_j"]].copy()
    return None


def prepare_XY(df, candidate_cols, outcome_col):
    """Build complete-case feature matrix and outcome vector."""
    present = [c for c in candidate_cols if c in df.columns]
    sub = df[[outcome_col] + present].copy()
    for col in present:
        sub[col] = clean_col(sub[col])
    sub[outcome_col] = clean_col(sub[outcome_col])
    sub = sub.dropna()
    return sub[present], sub[outcome_col], present


def vulnerability_class(score):
    """Map continuous score to vulnerability class (1/2/3)."""
    if score < BNDRY_LM:
        return 1
    elif score < BNDRY_MH:
        return 2
    return 3


def make_pipeline(n_estimators=None, max_features=None, min_samples_leaf=None):
    """Build a StandardScaler + RandomForestRegressor pipeline."""
    ne  = n_estimators     or BEST_PARAMS["rf__n_estimators"]
    mf  = max_features     or BEST_PARAMS["rf__max_features"]
    msl = min_samples_leaf or BEST_PARAMS["rf__min_samples_leaf"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=ne,
            max_features=mf,
            min_samples_leaf=msl,
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1,
        ))
    ])


# =============================================================================
# LOAD ALL DATA NEEDED
# =============================================================================

print("\nLoading candidate and predictor lists...")
fvi_candidates = load_candidates("fvi", FVI_CANDIDATES)
evi_candidates = load_candidates("evi", EVI_CANDIDATES)
fvi_stepwise   = load_stepwise_predictors("fvi")
evi_stepwise   = load_stepwise_predictors("evi")


# =============================================================================
# MEDIAN IMPUTATION FOR STRUCTURALLY MISSING FVI CANDIDATES
# =============================================================================
# Without this block, the FVI complete-case sample drops from n=2,993 to
# ~1,240 (58% loss) due to flood experience variables that are structurally
# missing for households with no flood exposure (MNAR by design).
# Median imputation is applied here for the RF stage only. The primary linear
# models (Stages 07-16) use the original analysis_dataset.csv unchanged.
# Reference: van Buuren (2018, §2.2).

print("\nApplying median imputation for structurally missing candidates...")
df_rf = df.copy()
_all_rf_candidates_18b = list(set(
    [col for domain in FVI_CANDIDATES.values() for col, _, _ in domain] +
    [col for domain in EVI_CANDIDATES.values() for col, _, _ in domain]
))
_median_imputed_18b = []
for _col in _all_rf_candidates_18b:
    if _col in df_rf.columns:
        _s = pd.to_numeric(df_rf[_col], errors="coerce")
        _n_miss = int(_s.isna().sum())
        if _n_miss > 0:
            _med = _s.median()
            df_rf[_col] = _s.fillna(_med)
            _median_imputed_18b.append((_col, _n_miss, _med))

if _median_imputed_18b:
    print(f"  {len(_median_imputed_18b)} column(s) median-imputed for RF only:")
    for _col, _n, _m in _median_imputed_18b:
        print(f"    {_col:<52} n_imputed={_n:,}  median={_m:.3f}")
    print("  Primary linear models (Stages 07-16) are unaffected.")
else:
    print("  No median imputation needed — all candidates complete.")

# Drop perfectly collinear variables from RF candidate pools
# These are removed AFTER median imputation (which caused the r=1.000)
# and BEFORE feature matrix construction, ensuring clean inputs to both
# training and validation.
_n_before_fvi = len(fvi_candidates)
_n_before_evi = len(evi_candidates)
fvi_candidates = [c for c in fvi_candidates if c not in PERFECT_COLLINEAR_DROPS]
evi_candidates = [c for c in evi_candidates if c not in PERFECT_COLLINEAR_DROPS]
_dropped_fvi = _n_before_fvi - len(fvi_candidates)
_dropped_evi = _n_before_evi - len(evi_candidates)
if _dropped_fvi or _dropped_evi:
    print(f"\nDropped perfectly collinear variables from RF candidate pool:")
    for v in PERFECT_COLLINEAR_DROPS:
        print(f"  {v}  (r=1.000 after median imputation — duplicate of DE05)")
    print(f"  FVI candidates: {_n_before_fvi} → {len(fvi_candidates)}")
    print(f"  EVI candidates: {_n_before_evi} → {len(evi_candidates)}")

print("\nPreparing feature matrices...")
X_fvi, y_fvi, fvi_cols = prepare_XY(
    df_rf, fvi_candidates, _PIPELINE_CFG["FVI_SCORE_COL"])
X_evi, y_evi, evi_cols = prepare_XY(
    df_rf, evi_candidates, _PIPELINE_CFG["EVI_SCORE_COL"])

# Community column — needed for residual analysis
comm_col = _PIPELINE_CFG["COMMUNITY_COL"]

# Stratified train/test split — identical split to Stage 18
fvi_classes = y_fvi.apply(vulnerability_class)
evi_classes = y_evi.apply(vulnerability_class)

X_fvi_tr, X_fvi_te, y_fvi_tr, y_fvi_te = train_test_split(
    X_fvi, y_fvi, train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE, stratify=fvi_classes)

X_evi_tr, X_evi_te, y_evi_tr, y_evi_te = train_test_split(
    X_evi, y_evi, train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE, stratify=evi_classes)

print(f"  FVI: n_train={len(X_fvi_tr):,}  n_test={len(X_fvi_te):,}  "
      f"features={len(fvi_cols)}")
print(f"  EVI: n_train={len(X_evi_tr):,}  n_test={len(X_evi_te):,}  "
      f"features={len(evi_cols)}")

# Fit best models (needed for checks 4–7)
print("\nFitting best models (Stage 18 parameters)...")
model_fvi = make_pipeline()
model_fvi.fit(X_fvi_tr, y_fvi_tr)
print(f"  FVI model fitted  (OOB R²={model_fvi['rf'].oob_score_:.3f})")

model_evi = make_pipeline()
model_evi.fit(X_evi_tr, y_evi_tr)
print(f"  EVI model fitted  (OOB R²={model_evi['rf'].oob_score_:.3f})")

# Reference OOS R² from Stage 18 (for annotation)
FVI_OOS_R2_S18 = r2_score(y_fvi_te, model_fvi.predict(X_fvi_te))
EVI_OOS_R2_S18 = r2_score(y_evi_te, model_evi.predict(X_evi_te))

# Collect all results for the summary report
summary = {}

# =============================================================================
# CHECK 1 — NESTED CROSS-VALIDATION
# =============================================================================
# Outer loop: 10-fold split. Inner loop: 5-fold grid search for tuning.
# Each outer fold trains and evaluates completely independently.
# This is the correct way to get an unbiased estimate of generalisation error
# for a tuned model (Cawley & Talbot 2010).

print()
print("=" * 70)
print("CHECK 1 — NESTED CROSS-VALIDATION")
print(f"  Outer folds: {NESTED_CV_FOLDS}   Inner folds (tuning): {CV_FOLDS}")
print("=" * 70)

nested_results = {}

for index, X, y in [("FVI", X_fvi, y_fvi), ("EVI", X_evi, y_evi)]:
    print(f"\n  {index}: running {NESTED_CV_FOLDS}-fold nested CV...")

    outer_cv = KFold(n_splits=NESTED_CV_FOLDS, shuffle=True,
                     random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True,
                     random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            random_state=RANDOM_STATE, oob_score=False, n_jobs=-1))
    ])

    gs = GridSearchCV(pipe, PARAM_GRID, cv=inner_cv,
                      scoring="r2", n_jobs=-1)

    fold_r2 = []
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        gs.fit(X_tr, y_tr)
        y_pred = gs.best_estimator_.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        fold_r2.append(r2)
        print(f"    Fold {fold_idx+1:2d}: R²={r2:.3f}  "
              f"best_params={gs.best_params_}")

    fold_r2 = np.array(fold_r2)
    mean_r2 = fold_r2.mean()
    std_r2  = fold_r2.std()
    ci95_lo = mean_r2 - 1.96 * std_r2
    ci95_hi = mean_r2 + 1.96 * std_r2

    nested_results[index] = {
        "fold_r2": fold_r2,
        "mean":    mean_r2,
        "std":     std_r2,
        "ci95_lo": ci95_lo,
        "ci95_hi": ci95_hi,
    }

    print(f"\n  {index} nested CV: mean R²={mean_r2:.3f} ± {std_r2:.3f}  "
          f"(95% CI: [{ci95_lo:.3f}, {ci95_hi:.3f}])")
    print(f"  Stage 18 single-split R²: "
          f"{FVI_OOS_R2_S18 if index=='FVI' else EVI_OOS_R2_S18:.3f}")

summary["nested_cv"] = nested_results

# Save CSV
ncv_rows = []
for index, res in nested_results.items():
    for i, r2 in enumerate(res["fold_r2"]):
        ncv_rows.append({"index": index, "fold": i+1, "oos_r2": r2})
    ncv_rows.append({"index": index, "fold": "mean",
                     "oos_r2": res["mean"]})
    ncv_rows.append({"index": index, "fold": "std",
                     "oos_r2": res["std"]})
pd.DataFrame(ncv_rows).to_csv(OUT_DIR / "rfval_nested_cv.csv", index=False)

# Plot — violin + individual fold dots
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.suptitle(
    f"Nested {NESTED_CV_FOLDS}-Fold Cross-Validation\n"
    "Stability of OOS R² across data partitions",
    fontsize=12, fontweight="bold"
)
colours = {"FVI": "#1F78B4", "EVI": "#E31A1C"}

for ax, (index, res) in zip(axes, nested_results.items()):
    s18_r2 = FVI_OOS_R2_S18 if index == "FVI" else EVI_OOS_R2_S18
    folds = res["fold_r2"]

    parts = ax.violinplot(folds, positions=[0], showmedians=True,
                          showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(colours[index])
        pc.set_alpha(0.4)
    parts["cmedians"].set_color(colours[index])
    parts["cbars"].set_color(colours[index])
    parts["cmaxes"].set_color(colours[index])
    parts["cmins"].set_color(colours[index])

    # Individual fold dots with jitter
    jitter = np.random.default_rng(RANDOM_STATE).uniform(
        -0.06, 0.06, len(folds))
    ax.scatter(jitter, folds, color=colours[index],
               edgecolors="white", zorder=3, s=40, linewidths=0.5)

    # Stage 18 single-split reference
    ax.axhline(s18_r2, color="black", linestyle="--",
               linewidth=1.2, label=f"Stage 18 single split ({s18_r2:.3f})")

    # Mean ± SD band
    ax.axhline(res["mean"], color=colours[index], linewidth=1.5,
               linestyle="-", label=f"Nested CV mean ({res['mean']:.3f})")
    ax.axhspan(res["ci95_lo"], res["ci95_hi"],
               alpha=0.15, color=colours[index], label="95% CI")

    ax.set_title(f"{index}", fontsize=11, fontweight="bold")
    ax.set_ylabel("OOS R²", fontsize=10)
    ax.set_xticks([])
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.4, 0.4)

fig.tight_layout()
fig.savefig(OUT_DIR / "rfval_nested_cv.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: rfval_nested_cv.csv / .png")


# =============================================================================
# CHECK 2 — LEARNING CURVES
# =============================================================================
# Trains on progressively larger fractions of training data.
# Plateau in validation curve before 100% confirms sample size adequacy.

print()
print("=" * 70)
print("CHECK 2 — LEARNING CURVES")
print(f"  Training sizes: {[f'{s:.0%}' for s in LEARNING_CURVE_SIZES]}")
print("=" * 70)

lc_results = {}
for index, X_tr, y_tr, X_te, y_te in [
    ("FVI", X_fvi_tr, y_fvi_tr, X_fvi_te, y_fvi_te),
    ("EVI", X_evi_tr, y_evi_tr, X_evi_te, y_evi_te),
]:
    print(f"\n  {index}: computing learning curve...")

    pipe = make_pipeline()
    cv   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipe, X_tr, y_tr,
        train_sizes=LEARNING_CURVE_SIZES,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    lc_results[index] = {
        "sizes":       train_sizes_abs,
        "train_mean":  train_scores.mean(axis=1),
        "train_std":   train_scores.std(axis=1),
        "val_mean":    val_scores.mean(axis=1),
        "val_std":     val_scores.std(axis=1),
    }

    final_val = lc_results[index]["val_mean"][-1]
    print(f"    Final validation R² at 100% training data: {final_val:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Learning Curves — Sample Size Adequacy\n"
    "Plateau in validation curve before 100% confirms n=2,993 is sufficient",
    fontsize=12, fontweight="bold"
)

for ax, (index, lc) in zip(axes, lc_results.items()):
    colour = colours[index]
    sizes  = lc["sizes"]

    ax.fill_between(sizes,
                    lc["train_mean"] - lc["train_std"],
                    lc["train_mean"] + lc["train_std"],
                    alpha=0.15, color="grey")
    ax.fill_between(sizes,
                    lc["val_mean"] - lc["val_std"],
                    lc["val_mean"] + lc["val_std"],
                    alpha=0.20, color=colour)

    ax.plot(sizes, lc["train_mean"], "o-", color="grey",
            linewidth=1.5, markersize=5, label="Training R²")
    ax.plot(sizes, lc["val_mean"], "s-", color=colour,
            linewidth=2, markersize=5, label="CV Validation R²")

    ax.set_xlabel("Training set size (n)", fontsize=10)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "rfval_learning_curves.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
print("\n  Saved: rfval_learning_curves.png")


# =============================================================================
# CHECK 3 — HYPERPARAMETER SENSITIVITY
# =============================================================================
# Runs a full grid search and reports R² for every combination.

print()
print("=" * 70)
print("CHECK 3 — HYPERPARAMETER SENSITIVITY")
print("=" * 70)

hp_results = []
for index, X_tr, y_tr, X_te, y_te in [
    ("FVI", X_fvi_tr, y_fvi_tr, X_fvi_te, y_fvi_te),
    ("EVI", X_evi_tr, y_evi_tr, X_evi_te, y_evi_te),
]:
    print(f"\n  {index}: fitting all {len(PARAM_GRID['rf__n_estimators']) * len(PARAM_GRID['rf__max_features']) * len(PARAM_GRID['rf__min_samples_leaf'])} grid combinations...")

    cv   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            random_state=RANDOM_STATE, oob_score=False, n_jobs=-1))
    ])
    gs = GridSearchCV(pipe, PARAM_GRID, cv=cv,
                      scoring="r2", n_jobs=-1, return_train_score=True)
    gs.fit(X_tr, y_tr)

    for i, params in enumerate(gs.cv_results_["params"]):
        cv_r2  = gs.cv_results_["mean_test_score"][i]
        cv_std = gs.cv_results_["std_test_score"][i]
        combo  = (f"n={params['rf__n_estimators']} "
                  f"mf={params['rf__max_features']} "
                  f"msl={params['rf__min_samples_leaf']}")
        hp_results.append({
            "index": index, "combo": combo,
            "cv_r2": cv_r2, "cv_std": cv_std,
            "n_estimators": params["rf__n_estimators"],
            "max_features":  params["rf__max_features"],
            "min_samples_leaf": params["rf__min_samples_leaf"],
        })
        print(f"    {combo:35s}  CV R²={cv_r2:.3f} ± {cv_std:.3f}")

hp_df = pd.DataFrame(hp_results)
hp_df.to_csv(OUT_DIR / "rfval_hyperparameter_sensitivity.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Hyperparameter Sensitivity\n"
    "OOS CV R² across all grid combinations — narrow range = robust to tuning",
    fontsize=12, fontweight="bold"
)

for ax, index in zip(axes, ["FVI", "EVI"]):
    sub = hp_df[hp_df["index"] == index].reset_index(drop=True)
    colour = colours[index]

    ax.errorbar(range(len(sub)), sub["cv_r2"], yerr=sub["cv_std"],
                fmt="o", color=colour, capsize=4,
                markersize=7, linewidth=1.2, elinewidth=0.8)

    r2_range = sub["cv_r2"].max() - sub["cv_r2"].min()
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["combo"], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("CV R² (mean ± SD across folds)", fontsize=10)
    ax.set_title(
        f"{index}  (range: {r2_range:.4f})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(sub["cv_r2"].max(), color=colour, linestyle="--",
               linewidth=0.8, alpha=0.6, label="Best")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "rfval_hyperparameter_sensitivity.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: rfval_hyperparameter_sensitivity.csv / .png")

summary["hp_sensitivity"] = {
    idx: {"range": hp_df[hp_df["index"] == idx]["cv_r2"].max()
                   - hp_df[hp_df["index"] == idx]["cv_r2"].min(),
          "min": hp_df[hp_df["index"] == idx]["cv_r2"].min(),
          "max": hp_df[hp_df["index"] == idx]["cv_r2"].max()}
    for idx in ["FVI", "EVI"]
}


# =============================================================================
# CHECK 4 — COMMUNITY RESIDUAL ANALYSIS
# =============================================================================
# RF residuals aggregated by community, plotted against mixed-effects
# random intercepts. Agreement confirms community clustering is genuine.

print()
print("=" * 70)
print("CHECK 4 — COMMUNITY RESIDUAL ANALYSIS")
print("=" * 70)

comm_residual_data = {}

for index, model, X_full, y_full, cols in [
    ("FVI", model_fvi, X_fvi, y_fvi, fvi_cols),
    ("EVI", model_evi, X_evi, y_evi, evi_cols),
]:
    # Get community labels aligned with X_full
    community_series = df.loc[X_full.index, comm_col].values
    y_pred_full = model.predict(X_full)
    residuals   = y_full.values - y_pred_full

    res_df = pd.DataFrame({
        "community": community_series,
        "residual":  residuals,
    })
    comm_res = res_df.groupby("community")["residual"].agg(
        ["mean", "std", "count"]).reset_index()
    comm_res.columns = ["community", "rf_mean_resid", "rf_std_resid", "n"]
    comm_res["rf_se"] = comm_res["rf_std_resid"] / np.sqrt(comm_res["n"])

    # Load mixed-effects random intercepts
    ri = load_community_intercepts(index)
    if ri is not None:
        # Standardise community name format for merging
        comm_res["community_clean"] = (comm_res["community"]
                                       .str.strip().str.lower())
        ri["community_clean"] = ri["community"].str.strip().str.lower()
        merged = comm_res.merge(ri[["community_clean", "u_j"]],
                                on="community_clean", how="left")
        comm_residual_data[index] = merged
        print(f"  {index}: {len(merged)} communities  "
              f"(ME intercepts merged: "
              f"{merged['u_j'].notna().sum()} matched)")
    else:
        comm_residual_data[index] = comm_res
        comm_residual_data[index]["u_j"] = np.nan
        print(f"  {index}: {len(comm_res)} communities  "
              f"(ME intercepts not found — community plot only)")

# Save
comm_out = pd.concat([
    v.assign(index=k) for k, v in comm_residual_data.items()
], ignore_index=True)
comm_out.to_csv(OUT_DIR / "rfval_community_residuals.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    "Community-Level RF Residuals vs Mixed-Effects Random Intercepts\n"
    "Agreement confirms community clustering is genuine and not captured "
    "by individual predictors",
    fontsize=11, fontweight="bold"
)

for ax, (index, crd) in zip(axes, comm_residual_data.items()):
    colour = colours[index]
    crd_sorted = crd.sort_values("rf_mean_resid").reset_index(drop=True)

    # RF residuals as caterpillar
    ax.errorbar(
        crd_sorted["rf_mean_resid"],
        range(len(crd_sorted)),
        xerr=1.96 * crd_sorted["rf_se"],
        fmt="o", color=colour, capsize=3, markersize=5,
        linewidth=0.8, elinewidth=0.6,
        label="RF mean residual ± 95% CI",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Mixed-effects intercepts if available
    if crd_sorted["u_j"].notna().any():
        ax.scatter(
            crd_sorted["u_j"],
            range(len(crd_sorted)),
            marker="D", color="#FF7F00", s=30, zorder=4,
            label="ME random intercept (u_j)",
        )

    ax.set_yticks(range(len(crd_sorted)))
    ax.set_yticklabels(crd_sorted["community"], fontsize=7.5)
    ax.set_xlabel("Mean residual (actual − predicted)", fontsize=10)
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, axis="x")

fig.tight_layout()
fig.savefig(OUT_DIR / "rfval_community_residuals.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved: rfval_community_residuals.csv / .png")


# =============================================================================
# CHECK 5 — CALIBRATION
# =============================================================================
# Bins predicted scores into deciles, plots mean actual vs mean predicted.
# Under-prediction at tails is the classic random forest bias.

print()
print("=" * 70)
print("CHECK 5 — CALIBRATION")
print("=" * 70)

cal_results = {}

for index, model, X_te, y_te in [
    ("FVI", model_fvi, X_fvi_te, y_fvi_te),
    ("EVI", model_evi, X_evi_te, y_evi_te),
]:
    y_pred = model.predict(X_te)
    cal_df  = pd.DataFrame({"actual": y_te.values, "predicted": y_pred})

    # Bin by predicted score decile
    cal_df["decile"] = pd.qcut(cal_df["predicted"], q=N_CALIBRATION_BINS,
                                labels=False, duplicates="drop")
    cal_bin = cal_df.groupby("decile").agg(
        mean_actual=("actual", "mean"),
        mean_predicted=("predicted", "mean"),
        n=("actual", "count"),
    ).reset_index()

    # Mean absolute calibration error
    mace = np.abs(cal_bin["mean_actual"] - cal_bin["mean_predicted"]).mean()
    cal_results[index] = {"bins": cal_bin, "mace": mace}
    print(f"  {index}: mean absolute calibration error = {mace:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Calibration — Predicted vs Actual by Decile\n"
    "Points below the diagonal = under-prediction (regression toward mean); "
    "above = over-prediction",
    fontsize=11, fontweight="bold"
)

for ax, (index, res) in zip(axes, cal_results.items()):
    colour = colours[index]
    bins = res["bins"]

    lims = [min(bins["mean_predicted"].min(), bins["mean_actual"].min()) - 0.05,
            max(bins["mean_predicted"].max(), bins["mean_actual"].max()) + 0.05]
    ax.plot(lims, lims, "--", color="grey", linewidth=1, label="Perfect calibration")

    sc = ax.scatter(bins["mean_predicted"], bins["mean_actual"],
                    s=bins["n"] / 3, c=colour, edgecolors="white",
                    linewidths=0.5, zorder=3, alpha=0.85,
                    label="Decile bin (size ∝ n)")

    # Annotate each bin
    for _, row in bins.iterrows():
        ax.annotate(f'n={int(row["n"])}',
                    xy=(row["mean_predicted"], row["mean_actual"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=6.5)

    ax.set_xlabel("Mean predicted score (decile)", fontsize=10)
    ax.set_ylabel("Mean actual score (decile)", fontsize=10)
    ax.set_title(
        f"{index}  MACE={res['mace']:.4f}",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

fig.tight_layout()
fig.savefig(OUT_DIR / "rfval_calibration.png", dpi=200, bbox_inches="tight")
plt.close(fig)

cal_out = pd.concat([
    res["bins"].assign(index=idx, mace=res["mace"])
    for idx, res in cal_results.items()
], ignore_index=True)
cal_out.to_csv(OUT_DIR / "rfval_calibration.csv", index=False)
print("  Saved: rfval_calibration.csv / .png")


# =============================================================================
# CHECK 6 — PERMUTATION IMPORTANCE STABILITY (50 repeats, 95% CI)
# =============================================================================

print()
print("=" * 70)
print(f"CHECK 6 — PERMUTATION IMPORTANCE STABILITY ({N_STABILITY_REPEATS} repeats)")
print("=" * 70)

stability_results = {}

for index, model, X_te, y_te, cols, labels, stepwise in [
    ("FVI", model_fvi, X_fvi_te, y_fvi_te, fvi_cols,
     FVI_PRED_LABELS, fvi_stepwise),
    ("EVI", model_evi, X_evi_te, y_evi_te, evi_cols,
     EVI_PRED_LABELS, evi_stepwise),
]:
    print(f"\n  {index}: computing {N_STABILITY_REPEATS}-repeat "
          f"permutation importance...")

    result = permutation_importance(
        model, X_te, y_te,
        n_repeats=N_STABILITY_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "variable":    cols,
        "label":       [labels.get(c, c) for c in cols],
        "mean":        result.importances_mean,
        "std":         result.importances_std,
        "ci95_lo":     result.importances_mean - 1.96 * result.importances_std,
        "ci95_hi":     result.importances_mean + 1.96 * result.importances_std,
        "in_stepwise": [c in stepwise for c in cols],
    }).sort_values("mean", ascending=False).reset_index(drop=True)
    imp_df["rank"] = imp_df.index + 1

    # Flag overlapping adjacent pairs (unstable rank order)
    imp_df["overlaps_next"] = False
    for i in range(len(imp_df) - 1):
        if imp_df.loc[i, "ci95_lo"] < imp_df.loc[i+1, "ci95_hi"]:
            imp_df.loc[i,   "overlaps_next"] = True

    stability_results[index] = imp_df
    n_overlap = imp_df["overlaps_next"].sum()
    print(f"    Adjacent pairs with overlapping 95% CIs: {n_overlap} "
          f"(should be reported as similarly important)")

    imp_df.to_csv(
        OUT_DIR / f"rfval_importance_stability_{index.lower()}.csv",
        index=False)

# Plot — top 20 per index with CI bars and overlap flags
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle(
    f"Permutation Importance Stability ({N_STABILITY_REPEATS} repeats)\n"
    "95% CI — overlapping intervals indicate similarly important variables",
    fontsize=12, fontweight="bold"
)

for ax, (index, imp_df) in zip(axes, stability_results.items()):
    colour = colours[index]
    plot_df = imp_df.head(20).copy()
    plot_df["short"] = plot_df["label"].apply(
        lambda x: x[:42] + "..." if len(x) > 45 else x)

    bar_colours = []
    for i, row in plot_df.iterrows():
        if row["in_stepwise"]:
            bar_colours.append("#1F78B4")
        else:
            bar_colours.append("#A8D1E7")

    ax.barh(range(len(plot_df)), plot_df["mean"],
            xerr=1.96 * plot_df["std"],
            color=bar_colours, edgecolor="white",
            capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333333"})

    # Hatching for overlapping pairs
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row["overlaps_next"] and i < len(plot_df) - 1:
            ax.barh(i, row["mean"], color="none",
                    edgecolor="red", linewidth=1.5, hatch="//")

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (mean ± 95% CI)", fontsize=10)
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.7)

    legend_handles = [
        mpatches.Patch(color="#1F78B4", label="In stepwise selection"),
        mpatches.Patch(color="#A8D1E7", label="RF only"),
        mpatches.Patch(facecolor="none", edgecolor="red",
                       hatch="//", label="Overlaps adjacent rank"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(OUT_DIR / "rfval_importance_stability.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: rfval_importance_stability_fvi/evi.csv / .png")


# =============================================================================
# CHECK 7 — MDI vs PERMUTATION IMPORTANCE
# =============================================================================
# Compares MDI (in-bag, biased) with permutation (test-set, unbiased).
# Divergence flags the Strobl et al. (2007) cardinality bias.

print()
print("=" * 70)
print("CHECK 7 — MDI vs PERMUTATION IMPORTANCE COMPARISON")
print("=" * 70)

mdi_results = {}

for index, model, cols, labels, perm_df in [
    ("FVI", model_fvi, fvi_cols, FVI_PRED_LABELS,
     stability_results["FVI"]),
    ("EVI", model_evi, evi_cols, EVI_PRED_LABELS,
     stability_results["EVI"]),
]:
    # MDI from the fitted forest
    mdi_raw = model["rf"].feature_importances_
    mdi_df  = pd.DataFrame({
        "variable": cols,
        "label":    [labels.get(c, c) for c in cols],
        "mdi":      mdi_raw,
    }).sort_values("mdi", ascending=False).reset_index(drop=True)
    mdi_df["rank_mdi"] = mdi_df.index + 1

    # Merge with permutation ranks
    merged = mdi_df.merge(
        perm_df[["variable", "mean", "rank"]].rename(
            columns={"mean": "perm_imp", "rank": "rank_perm"}),
        on="variable"
    )
    merged["rank_diff"] = (merged["rank_mdi"] - merged["rank_perm"]).abs()
    merged = merged.sort_values("rank_perm").reset_index(drop=True)

    mdi_results[index] = merged
    merged.to_csv(OUT_DIR / f"rfval_mdi_vs_permutation_{index.lower()}.csv",
                  index=False)

    # Flag largest divergences
    top_div = merged.nlargest(5, "rank_diff")[
        ["variable", "rank_mdi", "rank_perm", "rank_diff"]]
    print(f"\n  {index} — top rank divergences (MDI vs permutation):")
    for _, row in top_div.iterrows():
        print(f"    {row['variable']:<52} "
              f"MDI rank={int(row['rank_mdi']):2d}  "
              f"Perm rank={int(row['rank_perm']):2d}  "
              f"diff={int(row['rank_diff'])}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(
    "MDI vs Permutation Importance — Rank Comparison\n"
    "Variables above the diagonal are overranked by MDI (Strobl et al. 2007 bias); "
    "colour = |rank difference|",
    fontsize=11, fontweight="bold"
)

for ax, (index, merged) in zip(axes, mdi_results.items()):
    sc = ax.scatter(
        merged["rank_perm"], merged["rank_mdi"],
        c=merged["rank_diff"], cmap="YlOrRd",
        s=45, edgecolors="white", linewidths=0.5, zorder=3,
    )
    max_rank = max(merged["rank_mdi"].max(), merged["rank_perm"].max()) + 1
    ax.plot([1, max_rank], [1, max_rank], "--", color="grey",
            linewidth=1, label="Perfect agreement")

    # Annotate top divergers
    for _, row in merged.nlargest(4, "rank_diff").iterrows():
        short = labels.get(row["variable"], row["variable"])[:22]
        ax.annotate(short, xy=(row["rank_perm"], row["rank_mdi"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color="#333333")

    plt.colorbar(sc, ax=ax, label="|Rank difference|")
    ax.set_xlabel("Permutation importance rank", fontsize=10)
    ax.set_ylabel("MDI rank", fontsize=10)
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "rfval_mdi_vs_permutation.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: rfval_mdi_vs_permutation_fvi/evi.csv / .png")


# =============================================================================
# CHECK 8 — COLLINEARITY ANALYSIS
# =============================================================================
# A comprehensive collinearity assessment replacing the previous pairwise
# correlation heatmap. Covers:
#   8a. VIF and tolerance per predictor
#   8b. Condition number and condition indices (Belsley et al. 1980)
#   8c. Eigenvalue decomposition of the correlation matrix
#   8d. Variance decomposition proportions (identifies which variables
#       share variance on each near-zero eigenvalue)
#   8e. Correlation dendrogram with hierarchical cluster identification
#   8f. Cluster-based importance aggregation (Strobl et al. 2008)
#
# References:
#   Belsley, D.A., Kuh, E., & Welsch, R.E. (1980). Regression Diagnostics.
#     Wiley. https://doi.org/10.1002/0471725153
#   O'Brien, R.M. (2007). Quality & Quantity, 41(5), 673-690.
#     https://doi.org/10.1007/s11135-006-9018-6
#   Strobl et al. (2008). BMC Bioinformatics, 9, 307.
#     https://doi.org/10.1186/1471-2105-9-307
#   Hair et al. (2019). Multivariate Data Analysis (8th ed.). Cengage.

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

print()
print("=" * 70)
print("CHECK 8 — COLLINEARITY ANALYSIS")
print("=" * 70)

collinearity_summary = {}

# Importance CSVs are written by Stage 18 into the random_forest/ directory,
# not the validation/ subdirectory. Resolve from INPUT_DIR to avoid path errors.
_RF_DIR = INPUT_DIR / "random_forest"

for index, X_df, cols, labels, imp_csv in [
    ("FVI", X_fvi, fvi_cols, FVI_PRED_LABELS,
     _RF_DIR / "rf_fvi_importance.csv"),
    ("EVI", X_evi, evi_cols, EVI_PRED_LABELS,
     _RF_DIR / "rf_evi_importance.csv"),
]:
    print(f"\n  {index}:")
    n_vars = len(cols)
    X_arr  = X_df.values.astype(float)

    # ── 8a. VIF and tolerance ─────────────────────────────────────────────
    # VIF_j = 1 / (1 - R²_j) where R²_j is the R² from regressing predictor
    # j on all other predictors. Computed via the diagonal of the inverse
    # of the correlation matrix (numerically equivalent and avoids n regressions).
    corr_mat = X_df.corr(method="pearson").values
    # Add small ridge to handle near-singular matrices
    ridge = 1e-8 * np.eye(n_vars)
    try:
        corr_inv = np.linalg.inv(corr_mat + ridge)
    except np.linalg.LinAlgError:
        corr_inv = np.linalg.pinv(corr_mat)

    vif_vals = np.diag(corr_inv)
    tol_vals = 1.0 / vif_vals

    vif_df = pd.DataFrame({
        "variable":  cols,
        "vif":       np.round(vif_vals, 3),
        "tolerance": np.round(tol_vals, 3),
        "vif_flag":  vif_vals > VIF_THRESHOLD,
    })
    vif_df.to_csv(OUT_DIR / f"rfval_vif_{index.lower()}.csv", index=False)

    n_flagged_vif = vif_df["vif_flag"].sum()
    print(f"  8a. VIF and tolerance ({n_flagged_vif} variables with VIF > {VIF_THRESHOLD}):")
    for _, row in vif_df.sort_values("vif", ascending=False).head(8).iterrows():
        flag = " [HIGH]" if row["vif_flag"] else ""
        print(f"      {row['variable']:<52} VIF={row['vif']:6.2f}  "
              f"Tol={row['tolerance']:.3f}{flag}")

    # ── 8b. Condition number ──────────────────────────────────────────────
    # Condition number = sqrt(λ_max / λ_min) of the correlation matrix.
    # Values > 30 indicate serious multicollinearity (Belsley et al. 1980).
    eigvals = np.linalg.eigvalsh(corr_mat)
    eigvals = np.maximum(eigvals, 1e-12)        # numerical floor
    eigvals_desc = np.sort(eigvals)[::-1]       # largest first
    cond_num = np.sqrt(eigvals_desc[0] / eigvals_desc[-1])
    cond_indices = np.sqrt(eigvals_desc[0] / eigvals_desc)

    print(f"\n  8b. Condition number: {cond_num:.1f}  "
          f"({'SEVERE' if cond_num > CONDITION_NUM_THRESHOLD else 'ACCEPTABLE'} "
          f"threshold={CONDITION_NUM_THRESHOLD})")

    # ── 8c. Eigenvalue decomposition ─────────────────────────────────────
    # Number of eigenvalues < 0.10 indicates effective dimensionality loss.
    # If k predictors share one dimension (near-zero eigenvalue), the model
    # cannot reliably distinguish their separate contributions.
    n_small_eigvals = (eigvals_desc < 0.10).sum()
    print(f"\n  8c. Eigenvalue decomposition ({n_vars} predictors):")
    print(f"      Eigenvalues: {np.round(eigvals_desc[:8], 3).tolist()}"
          f"{'...' if n_vars > 8 else ''}")
    print(f"      Eigenvalues < 0.10: {n_small_eigvals} "
          f"(effective dimensions lost to collinearity)")

    eigval_df = pd.DataFrame({
        "rank":       range(1, n_vars + 1),
        "eigenvalue": np.round(eigvals_desc, 4),
        "cond_index": np.round(cond_indices, 2),
        "pct_variance": np.round(100 * eigvals_desc / eigvals_desc.sum(), 2),
    })
    eigval_df.to_csv(OUT_DIR / f"rfval_eigenvalues_{index.lower()}.csv", index=False)

    # ── 8d. Variance decomposition proportions ────────────────────────────
    # For each small eigenvalue (condition index > 30), identify which
    # variables have high variance decomposition proportions — these are
    # the specific variables causing that collinearity dependency.
    # Method: Belsley et al. (1980) §3.3.
    eigvecs = np.linalg.eigh(corr_mat)[1][:, ::-1]   # columns = eigenvectors
    # Variance decomposition: φ_jk = (v_jk² / λ_k) / Σ_k(v_jk² / λ_k)
    phi = (eigvecs ** 2) / eigvals_desc[np.newaxis, :]
    vdp = phi / phi.sum(axis=1, keepdims=True)        # row-normalise

    severe_dims = np.where(cond_indices > CONDITION_NUM_THRESHOLD)[0]
    vdp_rows = []
    if len(severe_dims) > 0:
        print(f"\n  8d. Variance decomposition proportions "
              f"(condition index > {CONDITION_NUM_THRESHOLD}):")
        for dim in severe_dims:
            ci_val = cond_indices[dim]
            # Variables with proportion > 0.50 on this dimension are implicated
            implicated = [(cols[j], round(vdp[j, dim], 3))
                          for j in range(n_vars) if vdp[j, dim] > 0.50]
            if implicated:
                print(f"      Dimension {dim+1} (CI={ci_val:.1f}): "
                      f"{', '.join(f'{v}={p:.2f}' for v, p in implicated)}")
            for j in range(n_vars):
                vdp_rows.append({
                    "variable": cols[j],
                    "dimension": dim + 1,
                    "cond_index": round(ci_val, 2),
                    "vdp": round(vdp[j, dim], 4),
                    "implicated": vdp[j, dim] > 0.50,
                })
    else:
        print(f"\n  8d. Variance decomposition: no dimensions with CI > "
              f"{CONDITION_NUM_THRESHOLD} — multicollinearity not severe.")

    if vdp_rows:
        pd.DataFrame(vdp_rows).to_csv(
            OUT_DIR / f"rfval_vdp_{index.lower()}.csv", index=False)

    # ── 8e. Correlation dendrogram and cluster identification ─────────────
    # Hierarchical clustering of predictors by correlation distance
    # (distance = 1 - |r|). Clusters identified at distance threshold 0.40
    # (corresponding to |r| > 0.60). This makes the collinear structure
    # visible as a tree rather than a list of pairs.
    corr_full = X_df.corr(method="pearson")
    dist_mat  = 1 - np.abs(corr_full.values)
    np.fill_diagonal(dist_mat, 0)
    dist_condensed = squareform(np.clip(dist_mat, 0, None))
    linkage_mat = linkage(dist_condensed, method="ward")

    CLUSTER_DIST = 1 - CORR_THRESHOLD   # 0.40 corresponds to |r| > 0.60
    cluster_labels = fcluster(linkage_mat, t=CLUSTER_DIST, criterion="distance")
    n_clusters = cluster_labels.max()

    # Build cluster membership dict
    clusters = {}
    for j, cl in enumerate(cluster_labels):
        clusters.setdefault(cl, []).append(cols[j])

    multi_member_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"\n  8e. Predictor clusters (|r| > {CORR_THRESHOLD} threshold):")
    print(f"      {n_clusters} clusters identified, "
          f"{len(multi_member_clusters)} with 2+ members:")
    for cl_id, members in sorted(multi_member_clusters.items(),
                                  key=lambda x: -len(x[1])):
        print(f"      Cluster {cl_id}: {', '.join(members)}")

    # Dendrogram plot
    short_labels = [labels.get(c, c)[:24] for c in cols]
    fig_dend, ax_dend = plt.subplots(figsize=(14, max(6, n_vars * 0.35)))
    dendrogram(
        linkage_mat,
        labels=short_labels,
        orientation="right",
        color_threshold=CLUSTER_DIST,
        ax=ax_dend,
        leaf_font_size=8,
    )
    ax_dend.axvline(CLUSTER_DIST, color="red", linestyle="--",
                    linewidth=1.2, label=f"|r|={CORR_THRESHOLD} threshold")
    ax_dend.set_xlabel("Distance (1 − |r|)", fontsize=10)
    ax_dend.set_title(
        f"{index} \u2014 Predictor Collinearity Dendrogram\n"
        f"Variables left of red line form collinear clusters "
        f"(|r| > {CORR_THRESHOLD})\n"
        f"{len(multi_member_clusters)} cluster(s) with 2+ members",
        fontsize=10, fontweight="bold"
    )
    ax_dend.legend(fontsize=9)
    fig_dend.tight_layout()
    fig_dend.savefig(OUT_DIR / f"rfval_collinearity_dendrogram_{index.lower()}.png",
                     dpi=180, bbox_inches="tight")
    plt.close(fig_dend)
    print(f"  Saved: rfval_collinearity_dendrogram_{index.lower()}.png")

    # ── 8f. Cluster-based importance aggregation ──────────────────────────
    # For collinear clusters the RF distributes importance across all cluster
    # members — no single member appears as important as the cluster truly is.
    # Strobl et al. (2008) recommend summing importance within clusters.
    # Load permutation importance from Stage 18 output.
    cluster_imp_rows = []
    if imp_csv.exists():
        imp_df = pd.read_csv(imp_csv)
        # imp_df has columns: variable, importance_mean, importance_std
        imp_col = "importance_mean" if "importance_mean" in imp_df.columns                   else imp_df.columns[1]
        imp_dict = dict(zip(imp_df["variable"], imp_df[imp_col]))

        print(f"\n  8f. Cluster-based importance aggregation (Strobl et al. 2008):")
        for cl_id, members in sorted(clusters.items()):
            cluster_total = sum(imp_dict.get(m, 0.0) for m in members)
            if len(members) > 1:
                print(f"      Cluster {cl_id} (n={len(members)})  "
                      f"total importance = {cluster_total:.4f}")
                for m in members:
                    ind_imp = imp_dict.get(m, np.nan)
                    print(f"        {m:<52} individual = {ind_imp:.4f}")
            for m in members:
                cluster_imp_rows.append({
                    "index":           index,
                    "cluster_id":      cl_id,
                    "variable":        m,
                    "n_cluster":       len(members),
                    "individual_imp":  imp_dict.get(m, np.nan),
                    "cluster_total":   cluster_total,
                    "pct_of_cluster":  (imp_dict.get(m, 0) / cluster_total * 100
                                        if cluster_total > 0 else np.nan),
                })
    else:
        print(f"\n  8f. Cluster importance: {imp_csv.name} not found — "
              f"run Stage 18 before Stage 18b.")

    if cluster_imp_rows:
        pd.DataFrame(cluster_imp_rows).to_csv(
            OUT_DIR / f"rfval_cluster_importance_{index.lower()}.csv", index=False)
        print(f"  Saved: rfval_cluster_importance_{index.lower()}.csv")

    # Store summary for report
    collinearity_summary[index] = {
        "vif_df":        vif_df,
        "cond_num":      cond_num,
        "n_small_eig":   n_small_eigvals,
        "n_clusters":    n_clusters,
        "n_multiclusters": len(multi_member_clusters),
        "severe_dims":   len(severe_dims),
        "n_flagged_vif": int(n_flagged_vif),
    }


# =============================================================================
# WRITE PLAIN-TEXT VALIDATION SUMMARY
# =============================================================================

lines = []
lines.append("=" * 70)
lines.append("STAGE 18b — RANDOM FOREST VALIDATION & CALIBRATION REPORT")
lines.append("=" * 70)
lines.append("")
lines.append("PURPOSE")
lines.append("-" * 40)
lines.append(
    "This report documents eight validation checks performed on the random "
    "forest models from Stage 18. These checks establish: (1) stability of "
    "the OOS R² across data partitions, (2) sample size adequacy, "
    "(3) robustness to hyperparameter choice, (4) whether community-level "
    "clustering is captured, (5) calibration accuracy, (6) reliability of "
    "importance rankings, (7) presence of MDI bias, and (8) collinearity "
    "among predictors."
)
lines.append("")
lines.append("CONFIGURATION")
lines.append("-" * 40)
lines.append(f"  Random seed        : {RANDOM_STATE}")
lines.append(f"  Train/test split   : {int(TRAIN_SIZE*100)}/{int((1-TRAIN_SIZE)*100)}")
lines.append(f"  Nested CV folds    : {NESTED_CV_FOLDS} outer / {CV_FOLDS} inner")
lines.append(f"  Stability repeats  : {N_STABILITY_REPEATS}")
lines.append(f"  Calibration bins   : {N_CALIBRATION_BINS}")
lines.append(f"  Corr. threshold    : |r| > {CORR_THRESHOLD}")
lines.append("")

for index in ["FVI", "EVI"]:
    s18_r2 = FVI_OOS_R2_S18 if index == "FVI" else EVI_OOS_R2_S18
    ncv    = nested_results[index]

    lines.append("=" * 70)
    lines.append(f"{index} VALIDATION RESULTS")
    lines.append("=" * 70)
    lines.append("")

    lines.append("CHECK 1 — Nested cross-validation")
    lines.append("-" * 40)
    lines.append(f"  Stage 18 single-split OOS R²  : {s18_r2:.3f}")
    lines.append(f"  Nested {NESTED_CV_FOLDS}-fold CV mean R²    : {ncv['mean']:.3f}")
    lines.append(f"  Nested CV SD                  : {ncv['std']:.3f}")
    lines.append(f"  95% CI                        : "
                 f"[{ncv['ci95_lo']:.3f}, {ncv['ci95_hi']:.3f}]")
    within_ci = ncv["ci95_lo"] <= s18_r2 <= ncv["ci95_hi"]
    lines.append(f"  Stage 18 R² within nested CI  : {'YES' if within_ci else 'NO'}")
    lines.append("")

    hp = summary["hp_sensitivity"][index]
    lines.append("CHECK 3 — Hyperparameter sensitivity")
    lines.append("-" * 40)
    lines.append(f"  CV R² range across grid        : {hp['min']:.3f} – {hp['max']:.3f}")
    lines.append(f"  Range width                    : {hp['range']:.4f}")
    lines.append(f"  Verdict: {'ROBUST (range < 0.02)' if hp['range'] < 0.02 else 'SENSITIVE (range >= 0.02)'}")
    lines.append("")

    cal = cal_results[index]
    lines.append("CHECK 5 — Calibration")
    lines.append("-" * 40)
    lines.append(f"  Mean absolute calibration error: {cal['mace']:.4f}")
    lines.append(f"  Interpretation: Forest predictions are on average "
                 f"{cal['mace']:.4f} score units away from actual")
    lines.append(f"  decile means. Values < 0.05 indicate good calibration.")
    lines.append("")

    stab = stability_results[index]
    n_overlap = int(stab["overlaps_next"].sum())
    lines.append("CHECK 6 — Importance stability")
    lines.append("-" * 40)
    lines.append(f"  Repeats                        : {N_STABILITY_REPEATS}")
    lines.append(f"  Adjacent pairs with overlapping")
    lines.append(f"  95% CIs (unstable rank order)  : {n_overlap}")
    if n_overlap > 0:
        overlap_pairs = stab[stab["overlaps_next"]]["variable"].tolist()
        lines.append(f"  Variables with unstable ranks  : {overlap_pairs}")
    lines.append("")

    mdi_merged = mdi_results[index]
    top_div = mdi_merged.nlargest(3, "rank_diff")
    lines.append("CHECK 7 — MDI vs permutation bias")
    lines.append("-" * 40)
    lines.append(f"  Mean absolute rank difference  : "
                 f"{mdi_merged['rank_diff'].mean():.1f} positions")
    lines.append(f"  Top-3 most divergent variables:")
    for _, row in top_div.iterrows():
        lines.append(f"    {row['variable']:<52} "
                     f"MDI={int(row['rank_mdi']):2d}  "
                     f"Perm={int(row['rank_perm']):2d}  "
                     f"diff={int(row['rank_diff'])}")
    lines.append("")

    coll = collinearity_summary.get(index, {})
    lines.append("CHECK 8 — Collinearity analysis")
    lines.append("-" * 40)
    lines.append(f"  Variables dropped (r=1.000)    : {PERFECT_COLLINEAR_DROPS}")
    lines.append(f"  Variables with VIF > {VIF_THRESHOLD}        : "
                 f"{coll.get('n_flagged_vif', 'N/A')}")
    lines.append(f"  Condition number               : "
                 f"{coll.get('cond_num', float('nan')):.1f}  "
                 f"({'SEVERE' if coll.get('cond_num', 0) > CONDITION_NUM_THRESHOLD else 'ACCEPTABLE'})")
    lines.append(f"  Eigenvalues < 0.10             : "
                 f"{coll.get('n_small_eig', 'N/A')} "
                 f"(dimensions lost to collinearity)")
    lines.append(f"  Predictor clusters (|r|>{CORR_THRESHOLD})   : "
                 f"{coll.get('n_multiclusters', 'N/A')} clusters with 2+ members")
    lines.append(f"  Severe VDP dimensions (CI>30)  : "
                 f"{coll.get('severe_dims', 'N/A')}")
    lines.append("")
    lines.append("  NOTE: Importance scores for collinear cluster members should")
    lines.append("  be interpreted as cluster-level importance, not individual")
    lines.append("  importance. See rfval_cluster_importance_*.csv for aggregated")
    lines.append("  cluster totals (Strobl et al. 2008).")
    lines.append("")

lines.append("=" * 70)
lines.append("REFERENCES")
lines.append("=" * 70)
lines.append("  Breiman (2001). Machine Learning, 45(1), 5-32.")
lines.append("    https://doi.org/10.1023/A:1010933404324")
lines.append("  Strobl et al. (2007). BMC Bioinformatics, 8, 25.")
lines.append("    https://doi.org/10.1186/1471-2105-8-25")
lines.append("  Strobl et al. (2008). BMC Bioinformatics, 9, 307.")
lines.append("    https://doi.org/10.1186/1471-2105-9-307")
lines.append("  Cawley & Talbot (2010). JMLR, 11, 2079-2107.")
lines.append("    https://www.jmlr.org/papers/v11/cawley10a.html")
lines.append("  Figueroa et al. (2012). BMC Med Inform Decis Mak, 12, 8.")
lines.append("    https://doi.org/10.1186/1472-6947-12-8")

report_path = OUT_DIR / "rfval_validation_summary.txt"
report_path.write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

print()
print("=" * 70)
print("STAGE 18b COMPLETE")
print("=" * 70)
for index in ["FVI", "EVI"]:
    s18_r2 = FVI_OOS_R2_S18 if index == "FVI" else EVI_OOS_R2_S18
    ncv    = nested_results[index]
    print(f"  {index}:")
    print(f"    Stage 18 OOS R²        : {s18_r2:.3f}")
    print(f"    Nested CV R²           : {ncv['mean']:.3f} ± {ncv['std']:.3f}  "
          f"(95% CI [{ncv['ci95_lo']:.3f}, {ncv['ci95_hi']:.3f}])")
    hp = summary["hp_sensitivity"][index]
    print(f"    HP sensitivity range   : {hp['range']:.4f}")
    print(f"    Calibration MACE       : {cal_results[index]['mace']:.4f}")
    n_ol = int(stability_results[index]["overlaps_next"].sum())
    print(f"    Unstable rank pairs    : {n_ol}")
    coll = collinearity_summary.get(index, {})
    print(f"    VIF > {VIF_THRESHOLD} count         : {coll.get('n_flagged_vif', 'N/A')}")
    print(f"    Condition number       : {coll.get('cond_num', float('nan')):.1f}")
    print(f"    Collinear clusters     : {coll.get('n_multiclusters', 'N/A')}")
print()
print(f"All outputs saved to: {OUT_DIR}")
print()
print("  Outputs:")
print("    rfval_nested_cv.csv / .png")
print("    rfval_learning_curves.png")
print("    rfval_hyperparameter_sensitivity.csv / .png")
print("    rfval_community_residuals.csv / .png")
print("    rfval_calibration.csv / .png")
print("    rfval_importance_stability_fvi/evi.csv / .png")
print("    rfval_mdi_vs_permutation_fvi/evi.csv / .png")
print("    rfval_vif_fvi/evi.csv")
print("    rfval_eigenvalues_fvi/evi.csv")
print("    rfval_vdp_fvi/evi.csv  (if severe dimensions found)")
print("    rfval_collinearity_dendrogram_fvi/evi.png")
print("    rfval_cluster_importance_fvi/evi.csv")
print("    rfval_validation_summary.txt")
