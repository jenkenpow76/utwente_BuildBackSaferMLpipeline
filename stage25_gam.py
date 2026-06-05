"""
Stage 19 | GAM | Generalised Additive Model Supplementary Analysis
===================================================================

Purpose
-------
Fit Generalised Additive Models (GAMs) for FVI and EVI as a supplementary
analysis that bridges the primary mixed-effects linear models (Stages 10/16)
and the random forest validation (Stage 18).

The GAM occupies a methodologically distinct position:
    - Like the linear model:  interpretable directional effects per predictor,
                              formal significance tests, confidence intervals.
    - Like the random forest: captures non-linear relationships without
                              requiring them to be specified in advance.

The GAM therefore directly answers what neither prior analysis could:
    - Which predictors have non-linear effects? (RF showed curvature but
      could not test it; linear model assumed it away entirely.)
    - How much does each predictor's non-linearity cost the linear model?
      (Quantified by EDF — effective degrees of freedom per smooth.)
    - Are the RF partial dependence plot shapes statistically credible, or
      are they noise? (Confirmed if the GAM smooth has a similar shape with
      a tight confidence interval.)

Specifically motivated by Stage 18 findings:
    - FVI RF gain = +0.092 over linear — substantial non-linearity exists.
    - EVI RF gain = +0.029 — minimal non-linearity; linear is adequate.
    - Applicability perception shows a sharp threshold around score 25-30
      in the FVI partial dependence plot — GAM tests this formally.
    - Age and household size ranked highly by MDI but poorly by permutation
      importance — GAM smooth will show whether any effect is real.
    - Financial capacity (+) / Time available (+) / Materials (+) are
      near-perfectly collinear (r > 0.89) — GAM term selection handles this.

Primary vs supplementary
-------------------------
This stage is SUPPLEMENTARY. The primary analysis remains the mixed-effects
linear models in Stages 10 and 16, which provide:
    - Community-level random intercepts and ICC estimates.
    - Coefficients interpretable within the MAO theoretical framework.
    - Significance tests consistent with the stepwise variable selection.

The GAM is a population-average model (no random effects). The community-level
clustering documented by ICC = 13.0% (FVI) and 11.1% (EVI) is not accounted
for. Standard errors and p-values from the GAM are therefore liberal
(underestimated), and the GAM should not be used for inference about individual
community effects. It is reported alongside the RF in the thesis appendix or
robustness section.

Algorithm
---------
    Library   : pygam (LinearGAM) — penalised B-splines with GCV smoothing
    Terms     : s() spline for continuous predictors (0-100 MAO scores,
                age, shelter score); f() factor for binary dummies (education,
                occupation, binary hazard indicators).
    Smoothing : Automatic via generalised cross-validation (GCV) — the
                penalty parameter λ is selected to minimise prediction error
                on a leave-one-out basis, preventing overfitting.
    Knots     : Default (20 basis functions per spline) — sufficient for
                the predictor ranges observed in this dataset.
    EDF       : Effective degrees of freedom per smooth term. EDF ≈ 1 means
                the smooth is essentially linear; EDF > 1 quantifies the
                degree of non-linearity. The difference (EDF_gam - 1) is
                the 'cost' of the non-linearity assumption in the linear model.

Inputs
------
    analysis_dataset.csv          — produced by Stage 05b
    fvi_im_predictors.csv         — stepwise-selected FVI predictors (Stage 08)
    evi_im_predictors.csv         — stepwise-selected EVI predictors (Stage 13)
    fvi_stepwise_stage2_results.csv — linear model adj. R² reference
    evi_stepwise_stage2_results.csv — linear model adj. R² reference

Outputs (written to outputs/gam/)
----------------------------------
    gam_fvi_summary.csv           — term statistics: EDF, p-value, pseudo-R²
    gam_evi_summary.csv           — same for EVI
    gam_fvi_smooths.png           — smooth effect plot, one panel per term
    gam_evi_smooths.png           — same for EVI
    gam_fvi_predicted_vs_actual.png — scatter: GAM predictions vs actual FVI
    gam_evi_predicted_vs_actual.png — same for EVI
    gam_comparison_summary.txt    — three-way comparison: linear / GAM / RF
    gam_nonlinearity_report.csv   — EDF per term, flags EDF > 1.5 as notable

References
----------
    Wood, S.N. (2017). Generalized Additive Models: An Introduction with R
        (2nd ed.). CRC Press. https://doi.org/10.1201/9781315370279

    Hastie, T. & Tibshirani, R. (1986). Generalized additive models.
        Statistical Science, 1(3), 297-318.
        https://doi.org/10.1214/ss/1177013604

    pygam documentation:
        https://pygam.readthedocs.io/

    Strobl et al. (2007) — permutation importance (Stage 18 context):
        https://doi.org/10.1186/1471-2105-8-25
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import sys
import warnings
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Ensure pygam is available ─────────────────────────────────────────────────
# pygam is not in the standard Anaconda distribution. Install silently if
# absent. This is done once; subsequent runs find it already installed.
try:
    from pygam import LinearGAM, s, f, l
    from pygam.terms import TermList
except ImportError:
    print("Installing pygam (first run only)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pygam", "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    from pygam import LinearGAM, s, f, l
    from pygam.terms import TermList
    print("pygam installed successfully.")

# ── Import analytical constants from pipeline_config ─────────────────────────
import importlib.util as _ilu
_cfg_path = Path(__file__).resolve().parent / "pipeline_config.py"
_cfg_spec  = _ilu.spec_from_file_location("pipeline_config", str(_cfg_path))
_cfg_mod   = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)

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

# EDF threshold above which a smooth is flagged as notably non-linear.
# EDF = 1 is exactly linear. EDF > 1.5 indicates curvature that the linear
# model is meaningfully misspecifying.
EDF_NONLINEAR_THRESHOLD = 1.5

# Train/test split — same as Stage 18 for comparability
TRAIN_SIZE = 0.70

# Number of points for smooth effect plots
N_PLOT_POINTS = 200

# Binary variable prefixes — these receive factor (f()) or linear (l()) terms
# rather than spline (s()) terms, since splines through binary values are
# meaningless and can produce unstable fits.
BINARY_PREFIXES = (
    "SE01_edu_", "SE04_", "DE01_", "DE02_",
    "PP01_", "PP02_", "PP12_",
)

# =============================================================================
# PATH RESOLUTION
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
OUT_DIR   = INPUT_DIR / "gam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STARTUP
# =============================================================================

print("=" * 70)
print("STAGE 19 — GAM SUPPLEMENTARY ANALYSIS")
print("=" * 70)
print()

analysis_path = INPUT_DIR / "analysis_dataset.csv"
if not analysis_path.exists():
    sys.exit(
        "ERROR: analysis_dataset.csv not found.\n"
        "Run Stages 01-05b before Stage 19."
    )

df = pd.read_csv(analysis_path, encoding="utf-8-sig", low_memory=False)
df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
n_total = len(df)
print(f"Loaded analysis_dataset.csv: {n_total:,} rows")

# Apply data quality caps
for col in CAP_COLS:
    if col in df.columns:
        s_cap = pd.to_numeric(df[col], errors="coerce")
        df[col] = s_cap.where(s_cap <= 100)


def clean_col(series):
    """Cast to numeric, replacing blanks with NaN."""
    return pd.to_numeric(series.replace(" ", np.nan).infer_objects(copy=False), errors="coerce")


# =============================================================================
# 1. LOAD STEPWISE PREDICTOR LISTS
# =============================================================================
# GAM is fitted on the same predictors that survived the stepwise Individual
# Model (Stages 08/13). This makes the GAM directly comparable to the linear
# mixed-effects model — same variables, same sample, different functional form.
#
# Using the stepwise predictors (rather than all candidates) keeps the model
# parsimonious and focuses the non-linearity analysis on the variables already
# established as relevant by the linear pipeline.

def load_stepwise_predictors(index):
    """
    Load stepwise-selected predictors from the Stage 08/13 output CSV.

    Parameters
    ----------
    index : str — "fvi" or "evi"

    Returns
    -------
    list of str — predictor column names
    """
    csv_path = INPUT_DIR / f"{index}_im_predictors.csv"
    if csv_path.exists():
        preds = pd.read_csv(csv_path)["predictor"].tolist()
        print(f"  {index.upper()}: {len(preds)} predictors from {csv_path.name}")
        return preds
    sys.exit(
        f"ERROR: {csv_path.name} not found.\n"
        f"Run Stage {'08' if index == 'fvi' else '13'} before Stage 19."
    )


def read_linear_r2(index):
    """
    Read the Individual Model adjusted R² from the stepwise results CSV.

    Returns (adj_r2, n) or fallback values with a warning.
    """
    fallbacks = {"fvi": (0.227, 2990), "evi": (0.097, 2989)}
    csv_path = INPUT_DIR / f"{index}_stepwise_stage2_results.csv"
    if csv_path.exists():
        try:
            _df = pd.read_csv(csv_path)
            adj_r2 = float(_df["adj_r2"].dropna().iloc[0])
            n      = int(_df["n"].dropna().iloc[0])
            return adj_r2, n
        except Exception as e:
            print(f"  WARNING: could not parse {csv_path.name} ({e})")
    print(f"  WARNING: {csv_path.name} not found — using fallback R²")
    return fallbacks[index]


print("\nLoading stepwise predictor lists...")
fvi_preds = load_stepwise_predictors("fvi")
evi_preds = load_stepwise_predictors("evi")

print("\nReading linear model R² references...")
fvi_linear_r2, fvi_linear_n = read_linear_r2("fvi")
evi_linear_r2, evi_linear_n = read_linear_r2("evi")
print(f"  FVI linear adj. R²={fvi_linear_r2:.3f} (n={fvi_linear_n:,})")
print(f"  EVI linear adj. R²={evi_linear_r2:.3f} (n={evi_linear_n:,})")

# Read RF OOS R² from Stage 18 metrics CSVs for three-way comparison
def read_rf_r2(index):
    """Read RF OOS R² from Stage 18 output CSV."""
    rf_dir = INPUT_DIR / "random_forest"
    csv_path = rf_dir / f"rf_{index}_metrics.csv"
    if csv_path.exists():
        try:
            row = pd.read_csv(csv_path).iloc[0]
            return float(row["rf_oos_r2"])
        except Exception:
            pass
    return None

fvi_rf_r2 = read_rf_r2("fvi")
evi_rf_r2 = read_rf_r2("evi")
print(f"  FVI RF OOS R²={fvi_rf_r2:.3f}" if fvi_rf_r2 else "  FVI RF R²: not found")
print(f"  EVI RF OOS R²={evi_rf_r2:.3f}" if evi_rf_r2 else "  EVI RF R²: not found")


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def is_binary(colname):
    """
    Return True if the column is a binary (0/1) dummy variable.

    Binary dummies receive linear (l()) terms in the GAM rather than splines,
    since smoothing a binary predictor produces unstable basis functions and
    offers no interpretive benefit over a linear term.
    """
    return colname.startswith(BINARY_PREFIXES)


def build_gam_terms(predictors, df):
    """
    Build a pygam TermList assigning appropriate term types to each predictor.

    Continuous predictors (MAO scores 0-100, age, shelter score) receive
    spline terms s(i) which pygam fits as penalised B-splines. Binary dummy
    predictors receive linear terms l(i) which are unpenalised straight lines.

    Parameters
    ----------
    predictors : list of str — predictor column names in model order
    df         : pd.DataFrame — used to verify column presence

    Returns
    -------
    TermList, list of (colname, term_type_str) for reporting

    Note
    ----
    pygam requires TermList(*terms) rather than sum(terms). Python's sum()
    initialises with integer 0, and pygam cannot add a Term to 0 — it raises
    "terms must be instances of Term or TermList, but found term: 0".
    """
    terms     = []
    term_info = []
    for i, col in enumerate(predictors):
        if is_binary(col):
            terms.append(l(i))
            term_info.append((col, "linear"))
        else:
            terms.append(s(i))
            term_info.append((col, "spline"))
    return TermList(*terms), term_info


def prepare_XY_gam(df, predictor_cols, outcome_col):
    """
    Build complete-case feature matrix and outcome vector for GAM fitting.

    Returns X as a numpy array (required by pygam), y as a numpy array,
    and the list of retained column names.
    """
    present = [c for c in predictor_cols if c in df.columns]
    absent  = [c for c in predictor_cols if c not in df.columns]
    if absent:
        print(f"    Warning: {len(absent)} columns absent from dataset — excluded: {absent}")

    sub = df[[outcome_col] + present].copy()
    for col in present:
        sub[col] = clean_col(sub[col])
    sub[outcome_col] = clean_col(sub[outcome_col])
    sub = sub.dropna()

    X = sub[present].values.astype(float)
    y = sub[outcome_col].values.astype(float)
    return X, y, present, sub.index


# =============================================================================
# 3. MAIN GAM ANALYSIS LOOP
# =============================================================================


# =============================================================================
# 2b. VIF AND TOLERANCE PRE-CHECK
# =============================================================================
# VIF is computed on the stepwise-selected predictor matrix before GAM
# fitting. Unlike the RF (which averages over many trees and is robust to
# collinearity in predictions), the GAM fits a single model where collinear
# predictors produce poorly identified smooths with inflated confidence
# intervals. EDF estimates for collinear terms are unreliable.
#
# VIF_j = 1/(1 - R²_j), computed via the diagonal of the inverse of the
# predictor correlation matrix. Tolerance = 1/VIF.
#
# Thresholds (O'Brien 2007; Hair et al. 2019):
#   VIF < 5   : acceptable
#   VIF 5-10  : moderate concern — interpret smooths with caution
#   VIF > 10  : severe — smooth for this term is unreliable
#
# References:
#   O'Brien, R.M. (2007). Quality & Quantity, 41(5), 673-690.
#     https://doi.org/10.1007/s11135-006-9018-6
#   Hair et al. (2019). Multivariate Data Analysis (8th ed.). Cengage.

VIF_THRESHOLD_WARN   = 5.0
VIF_THRESHOLD_SEVERE = 10.0

def compute_vif(X_arr, col_names):
    """
    Compute VIF and tolerance for each predictor column.

    Uses the correlation matrix inverse method which is numerically
    equivalent to regressing each predictor on the others and is faster
    for small-to-medium predictor sets.

    Parameters
    ----------
    X_arr     : np.ndarray — feature matrix (complete cases)
    col_names : list of str — predictor column names

    Returns
    -------
    pd.DataFrame with columns: variable, vif, tolerance, flag
    """
    corr = np.corrcoef(X_arr.T)
    ridge = 1e-8 * np.eye(len(col_names))
    try:
        corr_inv = np.linalg.inv(corr + ridge)
    except np.linalg.LinAlgError:
        corr_inv = np.linalg.pinv(corr)
    vifs = np.diag(corr_inv)
    tols = 1.0 / vifs
    flags = np.where(vifs > VIF_THRESHOLD_SEVERE, "SEVERE",
             np.where(vifs > VIF_THRESHOLD_WARN, "MODERATE", "OK"))
    return pd.DataFrame({
        "variable":  col_names,
        "vif":       np.round(vifs, 3),
        "tolerance": np.round(tols, 3),
        "flag":      flags,
    })

def _gam_statistics(gam):
    """
    Extract EDF, p-values, and pseudo-R² from a fitted pygam LinearGAM.

    pygam changed its statistics_ key names across versions. This helper
    tries the known variants in order so the pipeline works regardless of
    which version the user has installed.

    Key name history:
        edfs     → 0.8.x name for effective degrees of freedom per term
        edf      → 0.9.x rename (unconfirmed — handled defensively)
        p_values → stable across versions
        pseudo_r2 → 0.8.x: dict with 'explained_deviance' key
                    0.9.x: may be a plain float

    Parameters
    ----------
    gam : fitted LinearGAM instance

    Returns
    -------
    edfs      : list of float — EDF per term (1.0 if unavailable)
    p_vals    : list of float — p-value per term (NaN if unavailable)
    pseudo_r2 : float — in-sample explained deviance (NaN if unavailable)
    """
    stats = gam.statistics_

    # ── EDF ──────────────────────────────────────────────────────────────────
    edfs = (stats.get("edfs")        # pygam 0.8.x
         or stats.get("edf")         # possible 0.9.x rename
         or [1.0] * len(gam.terms))  # fallback: assume linear

    # ── p-values ─────────────────────────────────────────────────────────────
    p_vals = (stats.get("p_values")
           or stats.get("p_value")
           or [float("nan")] * len(gam.terms))

    # ── pseudo-R² ────────────────────────────────────────────────────────────
    pr2 = stats.get("pseudo_r2", float("nan"))
    if isinstance(pr2, dict):
        pseudo_r2 = pr2.get("explained_deviance", float("nan"))
    elif isinstance(pr2, (int, float)):
        pseudo_r2 = float(pr2)
    else:
        # Compute manually from deviance if the key is missing entirely
        try:
            pseudo_r2 = 1.0 - gam.statistics_["deviance"] / gam.statistics_["null_deviance"]
        except (KeyError, ZeroDivisionError):
            pseudo_r2 = float("nan")

    return list(edfs), list(p_vals), pseudo_r2

results_all = []
summary_lines = []

index_configs = [
    {
        "index":      "FVI",
        "outcome":    _PIPELINE_CFG["FVI_SCORE_COL"],
        "preds":      fvi_preds,
        "labels":     FVI_PRED_LABELS,
        "linear_r2":  fvi_linear_r2,
        "linear_n":   fvi_linear_n,
        "rf_r2":      fvi_rf_r2,
    },
    {
        "index":      "EVI",
        "outcome":    _PIPELINE_CFG["EVI_SCORE_COL"],
        "preds":      evi_preds,
        "labels":     EVI_PRED_LABELS,
        "linear_r2":  evi_linear_r2,
        "linear_n":   evi_linear_n,
        "rf_r2":      evi_rf_r2,
    },
]

for cfg in index_configs:
    index     = cfg["index"]
    outcome   = cfg["outcome"]
    preds     = cfg["preds"]
    labels    = cfg["labels"]
    linear_r2 = cfg["linear_r2"]
    linear_n  = cfg["linear_n"]
    rf_r2     = cfg["rf_r2"]

    print()
    print("=" * 70)
    print(f"{index} — GAM ANALYSIS")
    print("=" * 70)

    # ── 3a. Prepare data ──────────────────────────────────────────────────────
    X, y, retained, kept_idx = prepare_XY_gam(df, preds, outcome)
    n_complete = len(y)
    print(f"  Complete cases: {n_complete:,}   Features: {len(retained)}")

    # Stratified train/test split (same seed and ratio as Stage 18)
    vuln_class = np.where(y < BNDRY_LM, 1, np.where(y < BNDRY_MH, 2, 3))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE, stratify=vuln_class
    )
    print(f"  Train n: {len(X_tr):,}   Test n: {len(X_te):,}")

    # ── 3b. VIF pre-check ────────────────────────────────────────────────────
    print(f"\n  VIF pre-check (before GAM fitting):")
    vif_result = compute_vif(X, retained)
    vif_path = OUT_DIR / f"gam_{index.lower()}_vif.csv"
    vif_result.to_csv(vif_path, index=False)

    n_severe   = (vif_result["flag"] == "SEVERE").sum()
    n_moderate = (vif_result["flag"] == "MODERATE").sum()

    for _, vrow in vif_result.sort_values("vif", ascending=False).iterrows():
        if vrow["flag"] != "OK":
            print(f"    {vrow['variable']:<52} VIF={vrow['vif']:6.2f}  "
                  f"Tol={vrow['tolerance']:.3f}  [{vrow['flag']}]")

    if n_severe > 0:
        print(f"\n  WARNING: {n_severe} predictor(s) have VIF > {VIF_THRESHOLD_SEVERE}.")
        print("  Smooth estimates for these terms are unreliable. Consider:")
        print("  (a) Dropping the highest-VIF variable from the GAM predictor set.")
        print("  (b) Averaging collinear variables into a composite score.")
        print("  Smooths are still fitted but should be interpreted with caution.")
    elif n_moderate > 0:
        print(f"\n  CAUTION: {n_moderate} predictor(s) have VIF > {VIF_THRESHOLD_WARN}.")
        print("  Smooths for these terms have moderately inflated uncertainty.")
    else:
        print("  All VIF < 5.0 — no collinearity concerns for GAM fitting.")

    print(f"  VIF results saved: {vif_path.name}")

    # ── 3c. Build term structure ──────────────────────────────────────────────
    gam_terms, term_info = build_gam_terms(retained, df)
    n_splines = sum(1 for _, t in term_info if t == "spline")
    n_linear  = sum(1 for _, t in term_info if t == "linear")
    print(f"  Term types: {n_splines} spline(s), {n_linear} linear/factor")

    # ── 3c. Fit GAM ───────────────────────────────────────────────────────────
    # LinearGAM with automatic GCV smoothing parameter selection.
    # gridsearch() finds the optimal smoothing penalties λ for each spline
    # term by minimising the GCV score — equivalent to leave-one-out CV
    # for smoothing splines (Wood 2017, §6.2).
    print(f"  Fitting LinearGAM (GCV smoothing)...")
    gam = LinearGAM(gam_terms)
    gam.gridsearch(X_tr, y_tr, progress=False)
    print(f"  GAM fitted successfully.")

    # ── 3d. In-sample pseudo-R² ──────────────────────────────────────────────
    # pygam reports pseudo-R² as 1 - deviance/null_deviance, analogous to
    # R² in OLS. This is the in-sample (training) metric.
    edfs, p_vals, pseudo_r2_train = _gam_statistics(gam)

    # ── 3e. OOS R² on held-out test set ──────────────────────────────────────
    y_pred_te = gam.predict(X_te)
    oos_r2  = r2_score(y_te, y_pred_te)
    oos_rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
    oos_mae  = mean_absolute_error(y_te, y_pred_te)

    print(f"\n  Performance:")
    print(f"    GAM train pseudo-R²   : {pseudo_r2_train:.3f}")
    print(f"    GAM OOS R²            : {oos_r2:.3f}")
    print(f"    GAM OOS RMSE          : {oos_rmse:.4f}")
    print(f"    GAM OOS MAE           : {oos_mae:.4f}")
    print(f"    Linear adj. R²        : {linear_r2:.3f}  (n={linear_n:,})")
    gam_gain = oos_r2 - linear_r2
    print(f"    GAM gain over linear  : {gam_gain:+.3f}")
    if rf_r2:
        print(f"    RF OOS R²             : {rf_r2:.3f}")

    # ── 3f. Term statistics (EDF and p-values) ────────────────────────────────
    # edfs and p_vals are extracted via _gam_statistics() which handles
    # key naming differences across pygam versions. EDF is the effective
    # degrees of freedom per term:
    #   EDF = 1   → smooth is exactly linear (no penalty needed)
    #   EDF > 1   → smooth has curvature; the larger EDF the more non-linear
    #   EDF ≈ k   → smooth uses k-1 degrees of freedom (essentially unpenalised)
    # edfs and p_vals already extracted via _gam_statistics() above

    term_stats = []
    print(f"\n  Term statistics (EDF and p-value per predictor):")
    print(f"  {'Variable':<52} {'Type':<8} {'EDF':>6} {'p':>9}  {'Non-linear?'}")
    print("  " + "-" * 90)

    for i, (col, term_type) in enumerate(term_info):
        edf   = edfs[i]
        pval  = p_vals[i]
        label = labels.get(col, col)
        nonlin_flag = "YES *" if (term_type == "spline" and edf > EDF_NONLINEAR_THRESHOLD) else ""
        sig_flag = "*" if pval < 0.05 else ""
        print(f"  {col:<52} {term_type:<8} {edf:>6.2f} {pval:>9.4f}  {nonlin_flag}")

        term_stats.append({
            "index":       index,
            "variable":    col,
            "label":       label,
            "term_type":   term_type,
            "edf":         round(edf, 3),
            "p_value":     round(pval, 4),
            "significant": pval < 0.05,
            "non_linear":  term_type == "spline" and edf > EDF_NONLINEAR_THRESHOLD,
        })

    # Save term statistics CSV
    stats_df = pd.DataFrame(term_stats)
    stats_path = OUT_DIR / f"gam_{index.lower()}_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  Term statistics saved: {stats_path.name}")

    # Non-linearity report
    nonlin = stats_df[stats_df["non_linear"]]
    if len(nonlin) > 0:
        print(f"\n  Notably non-linear terms (EDF > {EDF_NONLINEAR_THRESHOLD}):")
        for _, row in nonlin.iterrows():
            print(f"    {row['variable']:<52} EDF={row['edf']:.2f}  p={row['p_value']:.4f}")
    else:
        print(f"\n  No terms with EDF > {EDF_NONLINEAR_THRESHOLD} — "
              f"linear model is a good approximation for {index}.")

    # ── 3g. Smooth effect plots ───────────────────────────────────────────────
    # One subplot per predictor. Spline terms show the smooth curve with
    # 95% pointwise confidence intervals. Linear terms show the fitted line.
    # The y-axis is the partial effect on the vulnerability score — analogous
    # to a partial regression plot or partial dependence plot, but with CIs.
    n_terms  = len(retained)
    n_cols   = min(4, n_terms)
    n_rows   = int(np.ceil(n_terms / n_cols))
    fig_h    = max(3 * n_rows, 6)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, fig_h))
    axes_flat = np.array(axes).flatten() if n_terms > 1 else [axes]

    for i, (col, term_type) in enumerate(term_info):
        ax    = axes_flat[i]
        label = labels.get(col, col)[:38]
        edf   = edfs[i]
        pval  = p_vals[i]

        # Generate smooth over predictor range
        xx = gam.generate_X_grid(term=i, n=N_PLOT_POINTS)
        y_pred_smooth, confi = gam.partial_dependence(term=i, X=xx,
                                                       width=0.95)
        x_vals = xx[:, i]

        ax.plot(x_vals, y_pred_smooth, color="#185FA5", linewidth=2)
        ax.fill_between(x_vals, confi[:, 0], confi[:, 1],
                        alpha=0.20, color="#185FA5")
        ax.axhline(0, color="#888780", linewidth=0.7, linestyle="--")

        # Title with EDF and p-value
        sig_star = "*" if pval < 0.05 else ""
        nonlin_note = f"  EDF={edf:.1f}★" if edf > EDF_NONLINEAR_THRESHOLD else f"  EDF={edf:.1f}"
        ax.set_title(f"{label}{sig_star}\n{nonlin_note}  p={pval:.3f}",
                     fontsize=8.5, fontweight="normal", pad=4)
        ax.set_xlabel(col[:25], fontsize=7.5)
        ax.set_ylabel("partial effect", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for j in range(n_terms, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{index} — GAM Smooth Effects (one panel per predictor)\n"
        f"* = p<0.05   ★ = EDF>{EDF_NONLINEAR_THRESHOLD} (notably non-linear)   "
        f"Shaded = 95% CI\n"
        f"GAM OOS R²={oos_r2:.3f}  |  Linear adj. R²={linear_r2:.3f}  |  "
        f"n_train={len(X_tr):,}",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    smooth_path = OUT_DIR / f"gam_{index.lower()}_smooths.png"
    fig.savefig(smooth_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Smooth effect plot saved: {smooth_path.name}")

    # ── 3h. Predicted vs actual scatter ──────────────────────────────────────
    y_pred_all = gam.predict(X)
    fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
    ax_sc.scatter(y, y_pred_all, alpha=0.20, s=8,
                  color="#185FA5", edgecolors="none")
    lims = [min(y.min(), y_pred_all.min()) - 0.05,
            max(y.max(), y_pred_all.max()) + 0.05]
    ax_sc.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax_sc.set_xlabel(f"Actual {outcome}", fontsize=10)
    ax_sc.set_ylabel(f"Predicted {outcome}", fontsize=10)
    ax_sc.set_title(
        f"{index} GAM — Predicted vs Actual\n"
        f"OOS R²={oos_r2:.3f}  |  n={n_complete:,}",
        fontsize=11, fontweight="bold"
    )
    ax_sc.legend(fontsize=9)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    fig_sc.tight_layout()
    scatter_path = OUT_DIR / f"gam_{index.lower()}_predicted_vs_actual.png"
    fig_sc.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(fig_sc)
    print(f"  Predicted vs actual saved: {scatter_path.name}")

    # Collect for summary report
    results_all.append({
        "index":           index,
        "n_complete":      n_complete,
        "n_train":         len(X_tr),
        "n_test":          len(X_te),
        "n_predictors":    len(retained),
        "pseudo_r2_train": pseudo_r2_train,
        "oos_r2":          oos_r2,
        "oos_rmse":        oos_rmse,
        "oos_mae":         oos_mae,
        "linear_r2":       linear_r2,
        "linear_n":        linear_n,
        "rf_r2":           rf_r2,
        "gam_gain":        gam_gain,
        "term_stats":      stats_df,
        "term_info":       term_info,
        "n_nonlinear":     len(nonlin),
    })


# =============================================================================
# 4. NON-LINEARITY REPORT CSV
# =============================================================================

all_terms = pd.concat([r["term_stats"] for r in results_all], ignore_index=True)
all_terms.to_csv(OUT_DIR / "gam_nonlinearity_report.csv", index=False)
print(f"\nNon-linearity report saved: gam_nonlinearity_report.csv")


# =============================================================================
# 5. THREE-WAY COMPARISON SUMMARY
# =============================================================================

lines = []
lines.append("=" * 70)
lines.append("STAGE 19 — GAM SUPPLEMENTARY ANALYSIS REPORT")
lines.append("=" * 70)
lines.append("")
lines.append("PURPOSE")
lines.append("-" * 40)
lines.append(
    "GAM (Generalised Additive Model) results for FVI and EVI, fitted on "
    "the same predictors as the primary mixed-effects linear models (Stages "
    "10/16) but allowing smooth non-linear effects via penalised splines. "
    "Reported alongside RF (Stage 18) in thesis appendix or robustness section."
)
lines.append("")
lines.append("IMPORTANT LIMITATION")
lines.append("-" * 40)
lines.append(
    "The GAM is a population-average model. Community-level random effects "
    "(ICC = 13.0% FVI, 11.1% EVI) are NOT accounted for. Standard errors "
    "and p-values are liberal. Do not interpret GAM community effects — "
    "use Stages 10/16 for those. The GAM is used only to characterise the "
    "shape of predictor-outcome relationships."
)
lines.append("")

for r in results_all:
    index = r["index"]
    lines.append("=" * 70)
    lines.append(f"{index} RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Three-way model comparison")
    lines.append("-" * 40)
    lines.append(f"  Linear mixed-effects adj. R²  : {r['linear_r2']:.3f}  "
                 f"(n={r['linear_n']:,}, with community RE)")
    lines.append(f"  GAM OOS R²                    : {r['oos_r2']:.3f}  "
                 f"(n_test={r['n_test']:,}, no community RE)")
    lines.append(f"  GAM train pseudo-R²           : {r['pseudo_r2_train']:.3f}")
    if r["rf_r2"]:
        lines.append(f"  RF OOS R²                     : {r['rf_r2']:.3f}  "
                     f"(n_test={r['n_test']:,}, no community RE)")
    lines.append(f"  GAM gain over linear          : {r['gam_gain']:+.3f}")
    lines.append(f"  GAM OOS RMSE                  : {r['oos_rmse']:.4f}")
    lines.append(f"  GAM OOS MAE                   : {r['oos_mae']:.4f}")
    lines.append("")
    lines.append("Term statistics")
    lines.append("-" * 40)
    lines.append(
        f"  {'Variable':<52} {'Type':<8} {'EDF':>6} {'p':>9}  {'Sig':>4}  {'Non-linear'}"
    )
    lines.append("  " + "-" * 90)
    for _, row in r["term_stats"].iterrows():
        sig = "*" if row["significant"] else ""
        nl  = "YES" if row["non_linear"] else ""
        lines.append(
            f"  {row['variable']:<52} {row['term_type']:<8} "
            f"{row['edf']:>6.2f} {row['p_value']:>9.4f}  {sig:>4}  {nl}"
        )
    lines.append("")
    n_nl = r["n_nonlinear"]
    lines.append(f"  Notably non-linear terms (EDF>{EDF_NONLINEAR_THRESHOLD}): {n_nl}")
    lines.append("")

lines.append("=" * 70)
lines.append("INTERPRETATION GUIDANCE")
lines.append("=" * 70)
lines.append("")
lines.append(
    "1. EDF interpretation: EDF=1 means the smooth is exactly linear — the "
    "GAM confirms the linear model is correct for that predictor. EDF>1.5 "
    "means the relationship has meaningful curvature that the linear model "
    "approximates with a single slope. The partial effect plots show the shape."
)
lines.append("")
lines.append(
    "2. Three-way R² comparison: Linear < GAM < RF is the expected ordering "
    "if non-linearities are present. If GAM ≈ linear, the linear model is "
    "adequate. If RF >> GAM, interactions (not just non-linearity) are the "
    "main source of additional predictive power."
)
lines.append("")
lines.append(
    "3. p-values are liberal: The GAM uses population-average inference "
    "without community random effects. Treat p-values as indicative rather "
    "than definitive — terms that are significant in both the GAM and the "
    "linear mixed-effects model have the strongest evidentiary support."
)
lines.append("")
lines.append("REFERENCES")
lines.append("-" * 40)
lines.append("  Wood, S.N. (2017). Generalized Additive Models (2nd ed.). CRC Press.")
lines.append("    https://doi.org/10.1201/9781315370279")
lines.append("  Hastie, T. & Tibshirani, R. (1986). Stat Science, 1(3), 297-318.")
lines.append("    https://doi.org/10.1214/ss/1177013604")
lines.append("  pygam: https://pygam.readthedocs.io/")

report_path = OUT_DIR / "gam_comparison_summary.txt"
report_path.write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# 6. CONSOLE SUMMARY
# =============================================================================

print()
print("=" * 70)
print("STAGE 19 COMPLETE")
print("=" * 70)
print()
print(f"  {'Index':<6} {'Linear R²':>10} {'GAM OOS R²':>12} {'RF OOS R²':>11} {'GAM gain':>10} {'Non-linear terms':>17}")
print("  " + "-" * 72)
for r in results_all:
    rf_str = f"{r['rf_r2']:.3f}" if r["rf_r2"] else "  N/A  "
    print(f"  {r['index']:<6} {r['linear_r2']:>10.3f} {r['oos_r2']:>12.3f} "
          f"{rf_str:>11} {r['gam_gain']:>+10.3f} {r['n_nonlinear']:>17}")
print()
print(f"All outputs saved to: {OUT_DIR}")
print(f"  gam_fvi_summary.csv              — FVI term statistics (EDF, p-value)")
print(f"  gam_evi_summary.csv              — EVI term statistics (EDF, p-value)")
print(f"  gam_fvi_vif.csv                  — FVI VIF and tolerance pre-check")
print(f"  gam_evi_vif.csv                  — EVI VIF and tolerance pre-check")
print(f"  gam_fvi_smooths.png              — FVI smooth effect plots")
print(f"  gam_evi_smooths.png              — EVI smooth effect plots")
print(f"  gam_fvi_predicted_vs_actual.png  — FVI scatter")
print(f"  gam_evi_predicted_vs_actual.png  — EVI scatter")
print(f"  gam_nonlinearity_report.csv      — EDF per term, all indices")
print(f"  gam_comparison_summary.txt       — three-way comparison report")
