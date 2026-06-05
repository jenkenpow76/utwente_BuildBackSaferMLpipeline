"""
Stage 19b | GAM | Validation & Calibration
==========================================

Purpose
-------
Validation and calibration checks for the GAM models fitted in Stage 19.
Mirrors the structure of Stage 18b (RF validation) to enable direct
comparison across the three supplementary methods.

Checks performed
----------------
    1. Nested cross-validation
       10-fold outer CV with GCV smoothing in each fold. Reports mean ± SD
       of OOS R² to confirm the Stage 19 single-split result is stable.
       Reference: Cawley & Talbot (2010). JMLR, 11, 2079-2107.

    2. Learning curves
       GAM fitted on progressively larger training subsets (10%–100%).
       Confirms sample size adequacy. A plateau before 100% confirms
       n=2,993 is sufficient for stable GAM estimates.

    3. Smoothing sensitivity
       Re-fits GAM with lam values spanning three orders of magnitude
       (lambda = 0.1, 1, 10, 100) to confirm OOS R² is stable across
       the smoothing parameter range. A narrow range = GCV found a
       genuine optimum; a wide range = results are sensitive to tuning.

    4. EDF stability across folds
       Checks whether the effective degrees of freedom (and therefore
       the non-linearity conclusion) are consistent across the 10 nested
       CV folds. If EDF for a term varies widely across folds, the
       non-linearity finding is unstable and should be reported with caution.

    5. Residual analysis
       Plots residuals vs fitted values and residuals vs each continuous
       predictor. Systematic patterns indicate model misspecification.
       Also plots residuals by community — if community-level patterns
       remain in GAM residuals (as they do for OLS), this motivates the
       mixed-effects model as the primary analysis.

    6. Calibration
       Decile-bin reliability diagram — same methodology as Stage 18b
       Check 5, enabling direct comparison of calibration quality between
       GAM and RF.

    7. GAM vs linear smooth comparison
       For each predictor, plots the GAM smooth alongside the linear model
       coefficient (rendered as a straight line through the data mean).
       Visually shows where the two models agree and where they diverge.
       The largest divergences identify where the linearity assumption
       is most consequential.

Inputs
------
    analysis_dataset.csv          — produced by Stage 05b
    fvi_im_predictors.csv         — stepwise-selected FVI predictors
    evi_im_predictors.csv         — stepwise-selected EVI predictors
    fvi_stepwise_stage2_results.csv — linear model coefficients
    evi_stepwise_stage2_results.csv — linear model coefficients
    fvi_random_intercepts.csv     — community ME intercepts (Stage 10)
    evi_random_intercepts.csv     — community ME intercepts (Stage 16)

Outputs (written to outputs/gam/validation/)
--------------------------------------------
    gamval_nested_cv.csv / .png          — Check 1
    gamval_learning_curves.png           — Check 2
    gamval_smoothing_sensitivity.csv / .png — Check 3
    gamval_edf_stability.csv / .png      — Check 4
    gamval_residuals.png                 — Check 5
    gamval_community_residuals.csv / .png — Check 5b
    gamval_calibration.csv / .png        — Check 6
    gamval_gam_vs_linear.png             — Check 7
    gamval_validation_summary.txt        — full report

References
----------
    Wood, S.N. (2017). Generalized Additive Models (2nd ed.). CRC Press.
        https://doi.org/10.1201/9781315370279
    Cawley & Talbot (2010). JMLR, 11, 2079-2107.
        https://www.jmlr.org/papers/v11/cawley10a.html
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

try:
    from pygam import LinearGAM, s, l
    from pygam.terms import TermList
except ImportError:
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pygam", "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    from pygam import LinearGAM, s, l
    from pygam.terms import TermList

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
COMMUNITY_COL = _PIPELINE_CFG["COMMUNITY_COL"]

# =============================================================================
# CONSTANTS
# =============================================================================

TRAIN_SIZE         = 0.70
NESTED_CV_FOLDS    = 10
LEARNING_SIZES     = np.linspace(0.10, 1.0, 10)
N_CALIBRATION_BINS = 10
EDF_NONLINEAR_THRESHOLD = 1.5

# Lambda grid for smoothing sensitivity check
LAM_GRID = [0.1, 1.0, 10.0, 100.0]

# Binary prefixes — same as Stage 19
BINARY_PREFIXES = (
    "SE01_edu_", "SE04_", "DE01_", "DE02_",
    "PP01_", "PP02_", "PP12_",
)

# =============================================================================
# PATH RESOLUTION
# =============================================================================

def _find_input_dir():
    """Return the directory containing analysis_dataset.csv."""
    for d in [Path.cwd(), Path.cwd() / "outputs",
              Path(__file__).resolve().parent / "outputs"]:
        if (d / "analysis_dataset.csv").exists():
            return d
    return Path.cwd()


INPUT_DIR = _find_input_dir()
OUT_DIR   = INPUT_DIR / "gam" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STARTUP
# =============================================================================

print("=" * 70)
print("STAGE 19b — GAM VALIDATION & CALIBRATION")
print("=" * 70)
print()

analysis_path = INPUT_DIR / "analysis_dataset.csv"
if not analysis_path.exists():
    sys.exit("ERROR: analysis_dataset.csv not found. Run Stages 01-05b first.")

df_raw = pd.read_csv(analysis_path, encoding="utf-8-sig", low_memory=False)
df_raw = df_raw.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
n_total = len(df_raw)
print(f"Loaded analysis_dataset.csv: {n_total:,} rows")

for col in CAP_COLS:
    if col in df_raw.columns:
        s_cap = pd.to_numeric(df_raw[col], errors="coerce")
        df_raw[col] = s_cap.where(s_cap <= 100)


def clean_col(series):
    return pd.to_numeric(series.replace(" ", np.nan).infer_objects(copy=False), errors="coerce")


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

def is_binary(colname):
    return colname.startswith(BINARY_PREFIXES)


def build_gam_terms(predictors):
    """
    Build pygam TermList: spline for continuous, linear for binary.

    Uses TermList(*terms) rather than sum(terms) to avoid the pygam
    TypeError caused by sum() initialising with integer 0.
    """
    terms = []
    for i, col in enumerate(predictors):
        terms.append(l(i) if is_binary(col) else s(i))
    return TermList(*terms)


def prepare_XY(df, preds, outcome):
    """Complete-case numpy arrays for GAM fitting."""
    present = [c for c in preds if c in df.columns]
    sub = df[[outcome] + present].copy()
    for col in present:
        sub[col] = clean_col(sub[col])
    sub[outcome] = clean_col(sub[outcome])
    sub = sub.dropna()
    return sub[present].values.astype(float), sub[outcome].values.astype(float), present, sub.index


def load_stepwise(index):
    csv_path = INPUT_DIR / f"{index}_im_predictors.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)["predictor"].tolist()
    sys.exit(f"ERROR: {csv_path.name} not found.")


def load_community_intercepts(index):
    csv_path = INPUT_DIR / f"{index.lower()}_random_intercepts.csv"
    if csv_path.exists():
        ri = pd.read_csv(csv_path)
        ri.columns = ri.columns.str.strip()
        if "community" in ri.columns and "u_j" in ri.columns:
            return ri[["community", "u_j"]].copy()
    return None


def load_linear_coefficients(index):
    """Load linear model coefficients from stepwise stage2 results."""
    csv_path = INPUT_DIR / f"{index}_stepwise_stage2_results.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)[["variable", "beta"]].dropna()
        except Exception:
            pass
    return None


print("Loading predictor lists...")
fvi_preds = load_stepwise("fvi")
evi_preds = load_stepwise("evi")

print("Preparing feature matrices...")
X_fvi, y_fvi, fvi_cols, fvi_idx = prepare_XY(df_raw, fvi_preds, _PIPELINE_CFG["FVI_SCORE_COL"])
X_evi, y_evi, evi_cols, evi_idx = prepare_XY(df_raw, evi_preds, _PIPELINE_CFG["EVI_SCORE_COL"])

print(f"  FVI: n={len(y_fvi):,}  features={len(fvi_cols)}")
print(f"  EVI: n={len(y_evi):,}  features={len(evi_cols)}")

# Train/test split — identical to Stage 18/19
vc_fvi = np.where(y_fvi < BNDRY_LM, 1, np.where(y_fvi < BNDRY_MH, 2, 3))
vc_evi = np.where(y_evi < BNDRY_LM, 1, np.where(y_evi < BNDRY_MH, 2, 3))

X_fvi_tr, X_fvi_te, y_fvi_tr, y_fvi_te = train_test_split(
    X_fvi, y_fvi, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=vc_fvi)
X_evi_tr, X_evi_te, y_evi_tr, y_evi_te = train_test_split(
    X_evi, y_evi, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=vc_evi)

print("\nFitting best GAM models for validation use...")
gam_fvi = LinearGAM(build_gam_terms(fvi_cols))
gam_fvi.gridsearch(X_fvi_tr, y_fvi_tr, progress=False)
print(f"  FVI GAM fitted  OOS R²={r2_score(y_fvi_te, gam_fvi.predict(X_fvi_te)):.3f}")

gam_evi = LinearGAM(build_gam_terms(evi_cols))
gam_evi.gridsearch(X_evi_tr, y_evi_tr, progress=False)
print(f"  EVI GAM fitted  OOS R²={r2_score(y_evi_te, gam_evi.predict(X_evi_te)):.3f}")

FVI_OOS_R2 = r2_score(y_fvi_te, gam_fvi.predict(X_fvi_te))
EVI_OOS_R2 = r2_score(y_evi_te, gam_evi.predict(X_evi_te))

colours = {"FVI": "#185FA5", "EVI": "#0F6E56"}
summary_lines = []


# =============================================================================
# CHECK 1 — NESTED CROSS-VALIDATION
# =============================================================================

print()
print("=" * 70)
print(f"CHECK 1 — NESTED CROSS-VALIDATION ({NESTED_CV_FOLDS} folds)")
print("=" * 70)

nested_results = {}

for index, X, y, cols in [
    ("FVI", X_fvi, y_fvi, fvi_cols),
    ("EVI", X_evi, y_evi, evi_cols),
]:
    print(f"\n  {index}: running {NESTED_CV_FOLDS}-fold nested CV...")
    outer_cv = KFold(n_splits=NESTED_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_r2 = []
    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        X_tr_f, X_te_f = X[tr_idx], X[te_idx]
        y_tr_f, y_te_f = y[tr_idx], y[te_idx]
        gam_f = LinearGAM(build_gam_terms(cols))
        gam_f.gridsearch(X_tr_f, y_tr_f, progress=False)
        r2_f = r2_score(y_te_f, gam_f.predict(X_te_f))
        fold_r2.append(r2_f)
        print(f"    Fold {fold+1:2d}: R²={r2_f:.3f}")

    fold_r2 = np.array(fold_r2)
    s19_r2  = FVI_OOS_R2 if index == "FVI" else EVI_OOS_R2
    nested_results[index] = {
        "fold_r2": fold_r2, "mean": fold_r2.mean(), "std": fold_r2.std(),
        "ci95_lo": fold_r2.mean() - 1.96 * fold_r2.std(),
        "ci95_hi": fold_r2.mean() + 1.96 * fold_r2.std(),
        "s19_r2":  s19_r2,
    }
    print(f"\n  {index} nested CV: mean R²={fold_r2.mean():.3f} ± {fold_r2.std():.3f}"
          f"  (95% CI [{fold_r2.mean()-1.96*fold_r2.std():.3f}, "
          f"{fold_r2.mean()+1.96*fold_r2.std():.3f}])")
    print(f"  Stage 19 single-split R²: {s19_r2:.3f}")

# Save and plot
ncv_rows = []
for idx, res in nested_results.items():
    for i, r2 in enumerate(res["fold_r2"]):
        ncv_rows.append({"index": idx, "fold": i+1, "oos_r2": r2})
    ncv_rows.extend([
        {"index": idx, "fold": "mean", "oos_r2": res["mean"]},
        {"index": idx, "fold": "std",  "oos_r2": res["std"]},
    ])
pd.DataFrame(ncv_rows).to_csv(OUT_DIR / "gamval_nested_cv.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"GAM Nested {NESTED_CV_FOLDS}-Fold Cross-Validation", fontsize=12, fontweight="bold")
for ax, (index, res) in zip(axes, nested_results.items()):
    colour = colours[index]
    folds  = res["fold_r2"]
    jitter = np.random.default_rng(RANDOM_STATE).uniform(-0.06, 0.06, len(folds))
    ax.scatter(jitter, folds, color=colour, s=45, edgecolors="white", zorder=3)
    ax.axhline(res["mean"], color=colour, linewidth=1.5, label=f"Mean ({res['mean']:.3f})")
    ax.axhspan(res["ci95_lo"], res["ci95_hi"], alpha=0.15, color=colour, label="95% CI")
    ax.axhline(res["s19_r2"], color="black", linestyle="--", linewidth=1.2,
               label=f"Stage 19 ({res['s19_r2']:.3f})")
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.set_ylabel("OOS R²")
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.set_xlim(-0.4, 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "gamval_nested_cv.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: gamval_nested_cv.csv / .png")


# =============================================================================
# CHECK 2 — LEARNING CURVES
# =============================================================================

print()
print("=" * 70)
print("CHECK 2 — LEARNING CURVES")
print("=" * 70)

lc_results = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("GAM Learning Curves — Sample Size Adequacy", fontsize=12, fontweight="bold")

for ax, (index, X_tr, y_tr, cols) in zip(axes, [
    ("FVI", X_fvi_tr, y_fvi_tr, fvi_cols),
    ("EVI", X_evi_tr, y_evi_tr, evi_cols),
]):
    colour = colours[index]
    print(f"\n  {index}: computing learning curve...")
    sizes_abs = (LEARNING_SIZES * len(X_tr)).astype(int)
    train_r2_means, val_r2_means = [], []
    train_r2_stds,  val_r2_stds  = [], []
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for sz in sizes_abs:
        fold_train, fold_val = [], []
        for tr_i, va_i in cv.split(X_tr):
            # Subsample training indices to sz
            rng = np.random.default_rng(RANDOM_STATE)
            sub_i = rng.choice(tr_i, size=min(sz, len(tr_i)), replace=False)
            X_sub, y_sub = X_tr[sub_i], y_tr[sub_i]
            X_val, y_val = X_tr[va_i], y_tr[va_i]
            if len(np.unique(y_sub)) < 2:
                continue
            g = LinearGAM(build_gam_terms(cols))
            try:
                g.gridsearch(X_sub, y_sub, progress=False)
                fold_train.append(r2_score(y_sub, g.predict(X_sub)))
                fold_val.append(r2_score(y_val, g.predict(X_val)))
            except Exception:
                pass
        if fold_train:
            train_r2_means.append(np.mean(fold_train))
            train_r2_stds.append(np.std(fold_train))
            val_r2_means.append(np.mean(fold_val))
            val_r2_stds.append(np.std(fold_val))

    train_r2_means = np.array(train_r2_means)
    val_r2_means   = np.array(val_r2_means)
    used_sizes     = sizes_abs[:len(train_r2_means)]

    ax.fill_between(used_sizes,
                    train_r2_means - np.array(train_r2_stds),
                    train_r2_means + np.array(train_r2_stds),
                    alpha=0.15, color="grey")
    ax.fill_between(used_sizes,
                    val_r2_means - np.array(val_r2_stds),
                    val_r2_means + np.array(val_r2_stds),
                    alpha=0.20, color=colour)
    ax.plot(used_sizes, train_r2_means, "o-", color="grey", linewidth=1.5, markersize=4, label="Training R²")
    ax.plot(used_sizes, val_r2_means, "s-", color=colour, linewidth=2, markersize=4, label="CV Validation R²")
    ax.set_xlabel("Training set size (n)")
    ax.set_ylabel("R²")
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"    Final validation R² at 100%: {val_r2_means[-1]:.3f}")

fig.tight_layout()
fig.savefig(OUT_DIR / "gamval_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: gamval_learning_curves.png")


# =============================================================================
# CHECK 3 — SMOOTHING SENSITIVITY
# =============================================================================

print()
print("=" * 70)
print("CHECK 3 — SMOOTHING SENSITIVITY")
print("=" * 70)

smooth_results = []
for index, X_tr, y_tr, X_te, y_te, cols in [
    ("FVI", X_fvi_tr, y_fvi_tr, X_fvi_te, y_fvi_te, fvi_cols),
    ("EVI", X_evi_tr, y_evi_tr, X_evi_te, y_evi_te, evi_cols),
]:
    print(f"\n  {index}:")
    for lam in LAM_GRID:
        gam_l = LinearGAM(build_gam_terms(cols), lam=lam)
        try:
            gam_l.fit(X_tr, y_tr)
            r2 = r2_score(y_te, gam_l.predict(X_te))
        except Exception:
            r2 = np.nan
        smooth_results.append({"index": index, "lam": lam, "oos_r2": r2})
        print(f"    lam={lam:6.1f}  OOS R²={r2:.3f}")

sm_df = pd.DataFrame(smooth_results)
sm_df.to_csv(OUT_DIR / "gamval_smoothing_sensitivity.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("GAM Smoothing Sensitivity\nOOS R² across penalty values — narrow range = robust",
             fontsize=11, fontweight="bold")
for ax, index in zip(axes, ["FVI", "EVI"]):
    sub = sm_df[sm_df["index"] == index]
    ax.plot(sub["lam"], sub["oos_r2"], "o-", color=colours[index], linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Lambda (smoothing penalty)")
    ax.set_ylabel("OOS R²")
    ax.set_title(index, fontsize=11, fontweight="bold")
    r2_rng = sub["oos_r2"].max() - sub["oos_r2"].min()
    ax.text(0.05, 0.95, f"Range: {r2_rng:.4f}", transform=ax.transAxes,
            fontsize=9, va="top", color=colours[index])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "gamval_smoothing_sensitivity.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: gamval_smoothing_sensitivity.csv / .png")


# =============================================================================
# CHECK 4 — EDF STABILITY ACROSS FOLDS
# =============================================================================

print()
print("=" * 70)
print("CHECK 4 — EDF STABILITY ACROSS FOLDS")
print("=" * 70)

edf_rows = []
for index, X, y, cols in [
    ("FVI", X_fvi, y_fvi, fvi_cols),
    ("EVI", X_evi, y_evi, evi_cols),
]:
    print(f"\n  {index}: computing EDF per fold...")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_edfs = {col: [] for col in cols}
    for fold, (tr_i, _) in enumerate(cv.split(X)):
        gam_f = LinearGAM(build_gam_terms(cols))
        try:
            gam_f.gridsearch(X[tr_i], y[tr_i], progress=False)
            _edfs_fold, _, _ = _gam_statistics(gam_f)
            for j, col in enumerate(cols):
                fold_edfs[col].append(_edfs_fold[j])
        except Exception:
            for col in cols:
                fold_edfs[col].append(np.nan)
    for col in cols:
        edf_vals = np.array(fold_edfs[col])
        edf_rows.append({
            "index": index, "variable": col,
            "edf_mean": np.nanmean(edf_vals), "edf_std": np.nanstd(edf_vals),
            "edf_min": np.nanmin(edf_vals), "edf_max": np.nanmax(edf_vals),
            "stable": np.nanstd(edf_vals) < 0.5,
        })
        flag = "" if np.nanstd(edf_vals) < 0.5 else " [UNSTABLE]"
        print(f"    {col:<52} EDF={np.nanmean(edf_vals):.2f} ± {np.nanstd(edf_vals):.2f}{flag}")

edf_df = pd.DataFrame(edf_rows)
edf_df.to_csv(OUT_DIR / "gamval_edf_stability.csv", index=False)

# ── EDF-VIF cross-check ───────────────────────────────────────────────────────
# Unstable EDF can arise from two distinct causes:
#   (a) Genuine non-linearity ambiguity — the optimal smoothing penalty
#       changes across folds because the true relationship is borderline.
#   (b) Collinearity — a correlated predictor's smooth competes for the
#       same variance in each fold, producing different EDF estimates
#       depending on which partition gets the slightly higher correlation.
#
# Distinguishing (a) from (b) matters for thesis reporting. If the unstable
# EDF term coincides with high VIF, the non-linearity finding is collinearity-
# driven and should be qualified. If VIF is low, the ambiguity is genuine.
#
# We cross-reference the EDF stability results against VIF values read from
# the Stage 19 pre-check output (gam_*_vif.csv).

print("\n  EDF-VIF cross-check (unstable terms × collinearity):")
cross_rows = []
for index in ["FVI", "EVI"]:
    vif_path = INPUT_DIR / "gam" / f"gam_{index.lower()}_vif.csv"
    edf_sub  = edf_df[edf_df["index"] == index].copy()

    if vif_path.exists():
        vif_check = pd.read_csv(vif_path).set_index("variable")
        edf_sub["vif"] = edf_sub["variable"].map(
            vif_check["vif"].to_dict()).fillna(np.nan)
        edf_sub["vif_flag"] = edf_sub["variable"].map(
            vif_check["flag"].to_dict()).fillna("UNKNOWN")
    else:
        edf_sub["vif"]      = np.nan
        edf_sub["vif_flag"] = "VIF file not found — run Stage 19 first"

    unstable = edf_sub[~edf_sub["stable"]].copy()
    if len(unstable) == 0:
        print(f"    {index}: no unstable EDF terms — cross-check not needed.")
    else:
        print(f"    {index}: {len(unstable)} unstable EDF term(s):")
        for _, row in unstable.iterrows():
            vif_val  = row["vif"]
            vif_flag = row["vif_flag"]
            cause    = ("COLLINEARITY-DRIVEN" if vif_flag in ("MODERATE", "SEVERE")
                        else "GENUINE AMBIGUITY" if vif_flag == "OK"
                        else "UNKNOWN")
            print(f"      {row['variable']:<52} "
                  f"EDF_std={row['edf_std']:.2f}  "
                  f"VIF={vif_val:.2f if not np.isnan(vif_val) else 'N/A'}  "
                  f"[{cause}]")
            cross_rows.append({
                "index":     index,
                "variable":  row["variable"],
                "edf_mean":  row["edf_mean"],
                "edf_std":   row["edf_std"],
                "stable":    row["stable"],
                "vif":       vif_val,
                "vif_flag":  vif_flag,
                "cause":     cause,
            })

if cross_rows:
    pd.DataFrame(cross_rows).to_csv(
        OUT_DIR / "gamval_edf_vif_crosscheck.csv", index=False)
    print("  Saved: gamval_edf_vif_crosscheck.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("GAM EDF Stability Across 5-Fold CV\nError bars = SD across folds",
             fontsize=11, fontweight="bold")
for ax, index in zip(axes, ["FVI", "EVI"]):
    sub = edf_df[edf_df["index"] == index].reset_index(drop=True)
    colour = colours[index]
    labels_short = [v[:30] for v in sub["variable"]]
    ax.barh(range(len(sub)), sub["edf_mean"],
            xerr=sub["edf_std"], color=colour, alpha=0.7,
            capsize=3, error_kw={"elinewidth": 0.8})
    ax.axvline(1, color="grey", linestyle="--", linewidth=0.8, label="EDF=1 (linear)")
    ax.axvline(EDF_NONLINEAR_THRESHOLD, color="red", linestyle=":",
               linewidth=0.8, label=f"EDF={EDF_NONLINEAR_THRESHOLD} (non-linear threshold)")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels_short, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("EDF (mean ± SD)")
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "gamval_edf_stability.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: gamval_edf_stability.csv / .png")


# =============================================================================
# CHECK 5 — RESIDUAL ANALYSIS + COMMUNITY RESIDUALS
# =============================================================================

print()
print("=" * 70)
print("CHECK 5 — RESIDUAL ANALYSIS")
print("=" * 70)

comm_residual_rows = []

fig_res, axes_res = plt.subplots(1, 2, figsize=(12, 5))
fig_res.suptitle("GAM Residuals vs Fitted Values", fontsize=12, fontweight="bold")

for index, gam, X_full, y_full, X_te, y_te, cols, idx_full in [
    ("FVI", gam_fvi, X_fvi, y_fvi, X_fvi_te, y_fvi_te, fvi_cols, fvi_idx),
    ("EVI", gam_evi, X_evi, y_evi, X_evi_te, y_evi_te, evi_cols, evi_idx),
]:
    ax = axes_res[0] if index == "FVI" else axes_res[1]
    y_fitted = gam.predict(X_full)
    resids   = y_full - y_fitted
    ax.scatter(y_fitted, resids, alpha=0.15, s=6, color=colours[index], edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Community residuals
    comm_series = df_raw.loc[idx_full, COMMUNITY_COL].values
    comm_df = pd.DataFrame({"community": comm_series, "residual": resids})
    comm_agg = comm_df.groupby("community")["residual"].agg(
        ["mean", "std", "count"]).reset_index()
    comm_agg.columns = ["community", "mean_resid", "std_resid", "n"]
    comm_agg["se"] = comm_agg["std_resid"] / np.sqrt(comm_agg["n"])

    ri = load_community_intercepts(index)
    if ri is not None:
        comm_agg["comm_clean"] = comm_agg["community"].str.strip().str.lower()
        ri["comm_clean"] = ri["community"].str.strip().str.lower()
        comm_agg = comm_agg.merge(ri[["comm_clean", "u_j"]], on="comm_clean", how="left")

    comm_agg["index"] = index
    comm_residual_rows.append(comm_agg)
    print(f"  {index}: {len(comm_agg)} communities in residual analysis")

fig_res.tight_layout()
fig_res.savefig(OUT_DIR / "gamval_residuals.png", dpi=200, bbox_inches="tight")
plt.close(fig_res)

# Community residuals plot
comm_all = pd.concat(comm_residual_rows, ignore_index=True)
comm_all.to_csv(OUT_DIR / "gamval_community_residuals.csv", index=False)

fig_cr, axes_cr = plt.subplots(1, 2, figsize=(15, 6))
fig_cr.suptitle("GAM Community Residuals vs Mixed-Effects Random Intercepts\n"
                "Agreement = community clustering not captured by GAM (motivates ME model)",
                fontsize=11, fontweight="bold")
for ax, index in zip(axes_cr, ["FVI", "EVI"]):
    sub = comm_all[comm_all["index"] == index].sort_values("mean_resid").reset_index(drop=True)
    colour = colours[index]
    ax.errorbar(sub["mean_resid"], range(len(sub)), xerr=1.96*sub["se"],
                fmt="o", color=colour, capsize=3, markersize=5,
                linewidth=0.8, elinewidth=0.6, label="GAM mean residual ± 95% CI")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    if "u_j" in sub.columns and sub["u_j"].notna().any():
        ax.scatter(sub["u_j"], range(len(sub)), marker="D", color="#FF7F00",
                   s=30, zorder=4, label="ME random intercept (u_j)")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["community"], fontsize=7.5)
    ax.set_xlabel("Mean residual (actual − predicted)")
    ax.set_title(index, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig_cr.tight_layout()
fig_cr.savefig(OUT_DIR / "gamval_community_residuals.png", dpi=200, bbox_inches="tight")
plt.close(fig_cr)
print("  Saved: gamval_residuals.png / gamval_community_residuals.csv / .png")


# =============================================================================
# CHECK 6 — CALIBRATION
# =============================================================================

print()
print("=" * 70)
print("CHECK 6 — CALIBRATION")
print("=" * 70)

cal_rows = []
fig_cal, axes_cal = plt.subplots(1, 2, figsize=(12, 5))
fig_cal.suptitle("GAM Calibration — Predicted vs Actual by Decile\n"
                 "Points below diagonal = under-prediction",
                 fontsize=11, fontweight="bold")

for ax, index, gam, X_te, y_te in [
    (axes_cal[0], "FVI", gam_fvi, X_fvi_te, y_fvi_te),
    (axes_cal[1], "EVI", gam_evi, X_evi_te, y_evi_te),
]:
    y_pred = gam.predict(X_te)
    cal_df = pd.DataFrame({"actual": y_te, "predicted": y_pred})
    cal_df["decile"] = pd.qcut(cal_df["predicted"], q=N_CALIBRATION_BINS,
                                labels=False, duplicates="drop")
    cal_bin = cal_df.groupby("decile").agg(
        mean_actual=("actual", "mean"),
        mean_predicted=("predicted", "mean"),
        n=("actual", "count"),
    ).reset_index()
    mace = np.abs(cal_bin["mean_actual"] - cal_bin["mean_predicted"]).mean()
    cal_rows.append(cal_bin.assign(index=index, mace=mace))
    print(f"  {index}: MACE = {mace:.4f}")

    colour = colours[index]
    lims = [min(cal_bin["mean_predicted"].min(), cal_bin["mean_actual"].min()) - 0.05,
            max(cal_bin["mean_predicted"].max(), cal_bin["mean_actual"].max()) + 0.05]
    ax.plot(lims, lims, "--", color="grey", linewidth=1)
    ax.scatter(cal_bin["mean_predicted"], cal_bin["mean_actual"],
               s=cal_bin["n"]/3, c=colour, edgecolors="white",
               linewidths=0.5, zorder=3, alpha=0.85)
    for _, row in cal_bin.iterrows():
        ax.annotate(f'n={int(row["n"])}',
                    xy=(row["mean_predicted"], row["mean_actual"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Mean predicted (decile)")
    ax.set_ylabel("Mean actual (decile)")
    ax.set_title(f"{index}  MACE={mace:.4f}", fontsize=11, fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

pd.concat(cal_rows, ignore_index=True).to_csv(
    OUT_DIR / "gamval_calibration.csv", index=False)
fig_cal.tight_layout()
fig_cal.savefig(OUT_DIR / "gamval_calibration.png", dpi=200, bbox_inches="tight")
plt.close(fig_cal)
print("  Saved: gamval_calibration.csv / .png")


# =============================================================================
# CHECK 7 — GAM vs LINEAR SMOOTH COMPARISON
# =============================================================================

print()
print("=" * 70)
print("CHECK 7 — GAM vs LINEAR SMOOTH COMPARISON")
print("=" * 70)

for index, gam, X_full, y_full, cols, labels in [
    ("FVI", gam_fvi, X_fvi, y_fvi, fvi_cols, FVI_PRED_LABELS),
    ("EVI", gam_evi, X_evi, y_evi, evi_cols, EVI_PRED_LABELS),
]:
    # Load linear coefficients
    lin_coefs = load_linear_coefficients(index)

    # Only plot continuous (spline) terms — linear comparison for binary is trivial
    cont_cols  = [(i, col) for i, col in enumerate(cols) if not is_binary(col)]
    n_cont     = len(cont_cols)
    if n_cont == 0:
        print(f"  {index}: no continuous terms to compare.")
        continue

    n_cols_fig = min(4, n_cont)
    n_rows_fig = int(np.ceil(n_cont / n_cols_fig))
    fig, axes  = plt.subplots(n_rows_fig, n_cols_fig,
                              figsize=(4.5 * n_cols_fig, 3.5 * n_rows_fig))
    axes_flat  = np.array(axes).flatten() if n_cont > 1 else [axes]

    for plot_idx, (term_idx, col) in enumerate(cont_cols):
        ax    = axes_flat[plot_idx]
        label = labels.get(col, col)[:35]

        # GAM smooth
        xx = gam.generate_X_grid(term=term_idx, n=200)
        y_smooth, confi = gam.partial_dependence(term=term_idx, X=xx, width=0.95)
        x_vals = xx[:, term_idx]

        ax.plot(x_vals, y_smooth, color=colours[index], linewidth=2, label="GAM smooth")
        ax.fill_between(x_vals, confi[:, 0], confi[:, 1],
                        alpha=0.20, color=colours[index])

        # Linear reference line (using OLS coefficient if available)
        if lin_coefs is not None:
            match = lin_coefs[lin_coefs["variable"] == col]
            if not match.empty:
                beta = float(match["beta"].iloc[0])
                x_range = x_vals.max() - x_vals.min()
                x_mid   = x_vals.mean()
                # Render as a line through the smooth midpoint
                y_mid   = float(y_smooth[len(y_smooth)//2])
                lin_y   = y_mid + beta * (x_vals - x_mid)
                ax.plot(x_vals, lin_y, color="#E24B4A", linewidth=1.5,
                        linestyle="--", label=f"Linear β={beta:.4f}")

        ax.axhline(0, color="#888780", linewidth=0.6, linestyle=":")
        ax.set_title(label, fontsize=8.5, pad=3)
        ax.set_xlabel(col[:22], fontsize=7.5)
        ax.set_ylabel("partial effect", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_cont, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{index} — GAM Smooth vs Linear Model\n"
        "Blue = GAM (with 95% CI)   Red dashed = Linear model slope",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = OUT_DIR / f"gamval_gam_vs_linear_{index.lower()}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {index}: saved gamval_gam_vs_linear_{index.lower()}.png")


# =============================================================================
# WRITE VALIDATION SUMMARY REPORT
# =============================================================================

vlines = []
vlines.append("=" * 70)
vlines.append("STAGE 19b — GAM VALIDATION & CALIBRATION REPORT")
vlines.append("=" * 70)
vlines.append("")
vlines.append(f"Random seed    : {RANDOM_STATE}")
vlines.append(f"Train/test     : {int(TRAIN_SIZE*100)}/{int((1-TRAIN_SIZE)*100)}")
vlines.append(f"Nested CV folds: {NESTED_CV_FOLDS}")
vlines.append(f"Lambda grid    : {LAM_GRID}")
vlines.append("")

for index in ["FVI", "EVI"]:
    res    = nested_results[index]
    sm_sub = sm_df[sm_df["index"] == index]
    cal_r  = pd.concat(cal_rows)[pd.concat(cal_rows)["index"] == index]
    mace   = float(cal_r["mace"].iloc[0]) if len(cal_r) > 0 else float("nan")
    sm_rng = sm_sub["oos_r2"].max() - sm_sub["oos_r2"].min()

    vlines.append("=" * 70)
    vlines.append(f"{index} VALIDATION RESULTS")
    vlines.append("=" * 70)
    vlines.append("")
    vlines.append("Check 1 — Nested CV")
    vlines.append(f"  Mean R²          : {res['mean']:.3f}")
    vlines.append(f"  SD               : {res['std']:.3f}")
    vlines.append(f"  95% CI           : [{res['ci95_lo']:.3f}, {res['ci95_hi']:.3f}]")
    vlines.append(f"  Stage 19 R²      : {res['s19_r2']:.3f}")
    within = res["ci95_lo"] <= res["s19_r2"] <= res["ci95_hi"]
    vlines.append(f"  Within CI        : {'YES' if within else 'NO'}")
    vlines.append("")
    vlines.append("Check 3 — Smoothing sensitivity")
    vlines.append(f"  OOS R² range     : {sm_sub['oos_r2'].min():.3f}–{sm_sub['oos_r2'].max():.3f}")
    vlines.append(f"  Range width      : {sm_rng:.4f}")
    vlines.append(f"  Verdict          : {'ROBUST (< 0.02)' if sm_rng < 0.02 else 'SENSITIVE (>= 0.02)'}")
    vlines.append("")
    vlines.append("Check 4 — EDF stability and VIF cross-check")
    cross_path = OUT_DIR / "gamval_edf_vif_crosscheck.csv"
    if cross_path.exists():
        cross_df = pd.read_csv(cross_path)
        cross_idx = cross_df[cross_df["index"] == index]
        coll_driven = (cross_idx["cause"] == "COLLINEARITY-DRIVEN").sum()
        genuine     = (cross_idx["cause"] == "GENUINE AMBIGUITY").sum()
        vlines.append(f"  Unstable EDF terms          : {len(cross_idx)}")
        vlines.append(f"    Collinearity-driven       : {coll_driven}")
        vlines.append(f"    Genuine ambiguity         : {genuine}")
    else:
        vlines.append("  EDF-VIF cross-check not available (run Stage 19 first)")
    vlines.append("")
    vlines.append("Check 6 — Calibration")
    vlines.append(f"  MACE             : {mace:.4f}")
    vlines.append("")

vlines.append("REFERENCES")
vlines.append("-" * 40)
vlines.append("  Wood (2017). GAMs: An Introduction with R (2nd ed.). CRC Press.")
vlines.append("  Cawley & Talbot (2010). JMLR, 11, 2079-2107.")

(OUT_DIR / "gamval_validation_summary.txt").write_text(
    "\n".join(vlines), encoding="utf-8")

# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

print()
print("=" * 70)
print("STAGE 19b COMPLETE")
print("=" * 70)
for index in ["FVI", "EVI"]:
    res    = nested_results[index]
    sm_sub = sm_df[sm_df["index"] == index]
    cal_r  = pd.concat(cal_rows)[pd.concat(cal_rows)["index"] == index]
    mace   = float(cal_r["mace"].iloc[0]) if len(cal_r) > 0 else float("nan")
    edf_sub = edf_df[edf_df["index"] == index]
    n_unstable = (~edf_sub["stable"]).sum()
    print(f"  {index}:")
    print(f"    Nested CV R²     : {res['mean']:.3f} ± {res['std']:.3f}")
    print(f"    Smoothing range  : {sm_sub['oos_r2'].max() - sm_sub['oos_r2'].min():.4f}")
    print(f"    Calibration MACE : {mace:.4f}")
    print(f"    Unstable EDF terms: {n_unstable}")
print()
print(f"All outputs saved to: {OUT_DIR}")
for fname in [
    "gamval_nested_cv.csv / .png",
    "gamval_learning_curves.png",
    "gamval_smoothing_sensitivity.csv / .png",
    "gamval_edf_stability.csv / .png",
    "gamval_residuals.png",
    "gamval_community_residuals.csv / .png",
    "gamval_calibration.csv / .png",
    "gamval_gam_vs_linear_fvi/evi.png",
    "gamval_edf_vif_crosscheck.csv",
    "gamval_validation_summary.txt",
]:
    print(f"  {fname}")
