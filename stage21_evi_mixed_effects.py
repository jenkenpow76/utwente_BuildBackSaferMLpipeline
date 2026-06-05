"""
stage16_evi_mixed_effects.py
=======================
Stage 6 | EVI — Mixed-Effects (Multilevel) Model

Input  : analysis_dataset.csv  (from Stage 05b)
Outputs: evi_mixed_effects_results.csv, evi_mixed_effects_variance.csv,
         evi_mixed_effects_comparison.csv, evi_mixed_effects_plot.png,
         evi_random_intercepts.png

Description
-----------
Mirrors stage10_fvi_mixed_effects.py exactly. Fits a random-intercept
multilevel model with households nested within communities (VDC/ward).

This is the FINAL VALIDATED EVI score. The column EVI_norm_1_5 in
imputed_EVI_scores.csv is the input for composite index construction
(FVI + EVI = CVI).

Reference
---------
Raudenbush, S.W. & Bryk, A.S. (2002). Hierarchical Linear Models:
    Applications and Data Analysis Methods (2nd ed.). SAGE Publications.
"""

# ─── Pipeline note ────────────────────────────────────────────────────────────
# When run via run_evi_pipeline.py, the working directory is OUTPUT_DIR.
# ─────────────────────────────────────────────────────────────────────────────

"""
Mixed-Effects (Multilevel) Model: FVI as Dependent Variable
============================================================
Random Intercept Model: households nested within communities (VDC/ward).

Model specification:
  FVI_ij = (β0 + u0j) + β1*X1_ij + ... + βk*Xk_ij + e_ij

  Where:
    FVI_ij  = FVI score for household i in community j
    β0      = grand intercept (fixed)
    u0j     = random intercept for community j ~ N(0, τ²)
    β1..βk  = fixed-effect slopes (same predictors as IM Stage 2)
    e_ij    = household-level residual ~ N(0, σ²)

Updated to align with 4_2__fvi_stepwise_updated.ipynb:
  - FILE_PATH: imputed_EVI_scores.csv (contains EVI_norm_1_5 + all predictors)
  - PREDICTORS: 10-variable Individual Model from updated Stage 2
  - Coverage filter removed: imputed_EVI_scores.csv already filters by
    FVI_weight_available >= 70% during scoring

Estimation: Restricted Maximum Likelihood (REML) via EM algorithm.

Comparison metric: Intraclass Correlation Coefficient (ICC)
  ICC = τ² / (τ² + σ²)
  Proportion of total variance explained by community-level differences.
  ICC > 0.05 justifies multilevel modelling (Hox et al., 2018).

References:
  Hox, J.J., Moerbeek, M., & van de Schoot, R. (2018). Multilevel
  Analysis: Techniques and Applications (3rd ed.). Routledge.
  https://doi.org/10.4324/9781315650982

  Snijders, T.A.B. & Bosker, R.J. (2012). Multilevel Analysis (2nd ed.).
  SAGE Publications.

  Laird, N.M. & Ware, J.H. (1982). Random-effects models for longitudinal
  data. Biometrics, 38(4), 963-974. https://doi.org/10.2307/2529876

Outputs:
  evi_mixed_effects_results.csv     — fixed effects table
  evi_mixed_effects_variance.csv    — variance components and ICC
  evi_mixed_effects_comparison.csv  — OLS vs mixed-effects comparison
  evi_mixed_effects_plot.png        — coefficient comparison plot
  evi_random_intercepts.png         — caterpillar plot of random intercepts
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import warnings


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

from pipeline_config import (
    EVI_IM_PREDICTORS,
    EVI_PRED_LABELS,
    CAP_COLS,
    WARD_TRANSPORT_COL,
    WARD_ASSISTANCE_COL,
    assert_no_drift,
)

warnings.filterwarnings("ignore")

# Verify pipeline integrity invariants before doing any work.
# This stage is an EVI-side consumer (reads evi_im_predictors.csv).
# We use consumer_evi mode which verifies the EVI invariants only.
assert_no_drift("consumer_evi")

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
# imputed_EVI_scores.csv is the scored dataset produced by evi_cimden.py.
# It contains EVI_norm_1_5, all MAO/socioeconomic predictors, GPS, and
# community identifiers. The raw survey file does NOT contain EVI_norm_1_5.

FILE_PATH = str(Path.cwd() / "analysis_dataset.csv")
DEPENDENT = "EVI_norm_1_5"

df = pd.read_csv(FILE_PATH, encoding='utf-8-sig', low_memory=False)

def clean_col(series):
    """Replace blank-string placeholders with NaN, cast to float."""
    return pd.to_numeric(
        series.replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )

df[DEPENDENT]   = clean_col(df[DEPENDENT])
df["community"] = df[_PIPELINE_CFG["COMMUNITY_COL"]].replace(" ", np.nan).infer_objects(copy=False)

# ── Load Individual Model predictor list from Stage 13 output ────────────────
# Stage 13 writes evi_im_predictors.csv after its stepwise selection completes.
# We read from that file so the mixed-effects model always uses the exact
# predictors that survived the current run — not a stale hardcoded list.
# Falls back to EVI_IM_PREDICTORS from pipeline_config.py if the file does
# not exist (e.g. when running Stage 16 standalone).
_im_csv = Path.cwd() / "evi_im_predictors.csv"
if _im_csv.exists():
    import pandas as _pd_im
    _LOADED_PREDICTORS = _pd_im.read_csv(_im_csv)["predictor"].tolist()
    print(f"Loaded {len(_LOADED_PREDICTORS)} predictors from evi_im_predictors.csv")
else:
    _LOADED_PREDICTORS = EVI_IM_PREDICTORS
    print("evi_im_predictors.csv not found — using EVI_IM_PREDICTORS from config")

# Individual Model predictors — loaded from Stage 13 output (evi_im_predictors.csv).
# Falls back to EVI_IM_PREDICTORS from pipeline_config.py if run standalone.
PREDICTORS = _LOADED_PREDICTORS

# Ward-level Level-2 covariates (added by Stage 08b).
# These are constant within each ward — they vary only between wards.
# They enter the fixed-effect design matrix alongside the household-level
# Individual Model predictors, allowing the model to partial out ward-level
# structural differences before estimating the random intercept variance.
# Reference: Raudenbush & Bryk (2002), Chapter 5.
WARD_LEVEL_COVARIATES = []
for _wc in [WARD_TRANSPORT_COL, WARD_ASSISTANCE_COL]:
    if _wc in df.columns and df[_wc].notna().any():
        df[_wc] = pd.to_numeric(df[_wc], errors="coerce")
        WARD_LEVEL_COVARIATES.append(_wc)
    else:
        print(f"  WARNING: ward covariate '{_wc}' not found or all-NaN — "
              "run Stage 08b (ward_covariates) before this stage.")

ALL_PREDICTORS = PREDICTORS + WARD_LEVEL_COVARIATES
if WARD_LEVEL_COVARIATES:
    print(f"  Ward-level covariates added: {WARD_LEVEL_COVARIATES}")

# Human-readable labels for plots and tables
PRED_LABELS = EVI_PRED_LABELS

# Apply data quality caps (mirrors stepwise notebook Section 2)
for _cap_col in CAP_COLS:
    if _cap_col in df.columns:
        _s = pd.to_numeric(df[_cap_col], errors="coerce")
        df[_cap_col] = _s.where(_s <= 100)

for col in PREDICTORS:
    if col in df.columns:
        df[col] = clean_col(df[col])
    else:
        print(f"  WARNING: '{col}' not found in dataset.")

# Ward covariates are already numeric from Stage 08b;
# run clean_col defensively to catch any stray blank strings.
for col in WARD_LEVEL_COVARIATES:
    if col in df.columns:
        df[col] = clean_col(df[col])

# Complete cases across DV, community, and all predictors
cols_needed = [DEPENDENT, "community"] + ALL_PREDICTORS
data = df[cols_needed].dropna().copy()
print(f"Complete cases: {len(data)}")
print(f"Communities: {data['community'].nunique()}")

# Remove outliers at ±3 SD of OLS residuals (consistent with Saputra protocol)
X_ols = np.column_stack(
    [np.ones(len(data))] + [data[p].values for p in ALL_PREDICTORS]
)
y_ols     = data[DEPENDENT].values
c_ols, _, _, _ = np.linalg.lstsq(X_ols, y_ols, rcond=None)
resid_ols = y_ols - X_ols @ c_ols
mse_ols   = np.sum(resid_ols**2) / (len(y_ols) - len(ALL_PREDICTORS) - 1)
z_ols     = resid_ols / np.sqrt(mse_ols)
data      = data[np.abs(z_ols) <= 3].copy().reset_index(drop=True)

print(f"Analysis sample after outlier removal: n = {len(data)}")
print(f"Community sizes:\n{data.groupby('community').size().sort_values(ascending=False).to_string()}")

# Encode community as integer index for EM algorithm
communities  = data["community"].unique()
comm_to_idx  = {c: i for i, c in enumerate(communities)}
data["comm_i"] = data["community"].map(comm_to_idx)
J = len(communities)   # number of communities

y  = data[DEPENDENT].values.astype(float)
X  = np.column_stack(
    [np.ones(len(data))] + [data[p].values.astype(float) for p in ALL_PREDICTORS]
)
gj = data["comm_i"].values   # community group index per observation
n  = len(y)
k  = X.shape[1]              # fixed-effect parameters: intercept + IM predictors + ward covariates

# =============================================================================
# 2. REML-EM ALGORITHM FOR RANDOM INTERCEPT MODEL
# =============================================================================
# Estimates fixed effects β, random intercept variance τ², and residual
# variance σ² via the EM algorithm.
#
# Reference: Laird & Ware (1982), Biometrics, 38(4), 963-974.
#            https://doi.org/10.2307/2529876

def reml_em_random_intercept(y, X, group, J, max_iter=200, tol=1e-6):
    """
    REML via EM algorithm for a random intercept model.

    Parameters
    ----------
    y        : (n,) array   — dependent variable
    X        : (n, k) array — fixed-effect design matrix (includes intercept)
    group    : (n,) int     — community index per observation
    J        : int          — number of communities
    max_iter : int          — maximum EM iterations
    tol      : float        — convergence tolerance on log-likelihood change

    Returns
    -------
    beta      : (k,) fixed-effect coefficients
    tau2      : float — random intercept variance
    sigma2    : float — residual variance
    u         : (J,) BLUP random intercept estimates per community
    se_beta   : (k,) standard errors of fixed effects
    converged : bool
    """
    # Initialise from OLS
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid_init    = y - X @ beta
    sigma2 = np.var(resid_init) * 0.9
    tau2   = np.var(resid_init) * 0.1

    ll_prev = -np.inf

    for iteration in range(max_iter):

        # E-STEP: compute BLUP random intercepts
        # Posterior mean: u_j = tau2 * sum(r_ij) / (n_j*tau2 + sigma2)
        r  = y - X @ beta
        u  = np.zeros(J)
        nj = np.zeros(J)

        for j in range(J):
            mask_j = group == j
            nj[j]  = mask_j.sum()
            u[j]   = (tau2 * r[mask_j].sum()) / (nj[j] * tau2 + sigma2)

        # M-STEP: update fixed effects on u-adjusted response
        y_adj = y - u[group]
        beta, _, _, _ = np.linalg.lstsq(X, y_adj, rcond=None)

        # Update variance components
        r_new = y - X @ beta - u[group]

        # σ²: expected residual variance (includes uncertainty in u)
        sigma2_new = (
            np.sum(r_new**2) +
            sigma2 * np.sum(nj * tau2 / (nj * tau2 + sigma2))
        ) / n

        # τ²: expected random intercept variance
        tau2_new = (
            np.sum(u**2) +
            np.sum(sigma2 * tau2 / (nj * tau2 + sigma2))
        ) / J

        sigma2 = max(sigma2_new, 1e-8)
        tau2   = max(tau2_new,   1e-8)

        # Approximate log-likelihood for convergence check
        ll = -0.5 * (n * np.log(2 * np.pi * sigma2) +
                     np.sum(r_new**2) / sigma2)

        if abs(ll - ll_prev) < tol:
            print(f"  EM converged at iteration {iteration + 1}")
            break
        ll_prev = ll

    converged = iteration < max_iter - 1

    # Standard errors via V^-1 sandwich formula
    # V_ij = sigma2 + tau2 (same community), sigma2 (different community)
    W = np.zeros((n, n))
    for j in range(J):
        mask_j = np.where(group == j)[0]
        nj_    = len(mask_j)
        a = 1.0 / sigma2
        b = -tau2 / (sigma2 * (sigma2 + nj_ * tau2))
        block = a * np.eye(nj_) + b * np.ones((nj_, nj_))
        for ii, i_ in enumerate(mask_j):
            for jj, j_ in enumerate(mask_j):
                W[i_, j_] = block[ii, jj]

    XtWX_inv = np.linalg.pinv(X.T @ W @ X)
    se_beta  = np.sqrt(np.diag(XtWX_inv))

    return beta, tau2, sigma2, u, se_beta, converged


print("\nFitting Random Intercept Model via REML-EM...")
beta, tau2, sigma2, u_j, se_beta, converged = reml_em_random_intercept(
    y, X, gj, J
)

# =============================================================================
# 3. FIXED EFFECTS TABLE
# =============================================================================

# Degrees of freedom: n - k (fixed params) - J (random intercepts)
df_resid    = n - k - J
t_vals      = beta / se_beta
p_vals      = 2 * stats.t.sf(np.abs(t_vals), df=df_resid)
t_crit      = stats.t.ppf(0.975, df=df_resid)
ci_lo       = beta - t_crit * se_beta
ci_hi       = beta + t_crit * se_beta
param_names = ["Intercept"] + ALL_PREDICTORS

print(f"\nFixed Effects (n={n}, J={J} communities):")
print(f"  {'Parameter':<45} {'β':>8} {'SE':>8} {'t':>8} {'p':>8}")
print("  " + "-" * 78)
for i, name in enumerate(param_names):
    label = PRED_LABELS.get(name, name)
    sig   = "*" if p_vals[i] < 0.05 else ""
    print(f"  {label:<45} {beta[i]:>8.4f} {se_beta[i]:>8.4f} "
          f"{t_vals[i]:>8.3f} {p_vals[i]:>8.4f} {sig}")

# =============================================================================
# 4. VARIANCE COMPONENTS AND ICC
# =============================================================================

total_var = tau2 + sigma2
icc       = tau2 / total_var

print(f"\nVariance Components:")
print(f"  Random intercept variance (τ²): {tau2:.4f}")
print(f"  Residual variance        (σ²): {sigma2:.4f}")
print(f"  Total variance               : {total_var:.4f}")
print(f"  ICC = τ²/(τ²+σ²)             : {icc:.4f} ({icc*100:.1f}%)")
print(f"\n  Interpretation (Hox et al. 2018):")
if icc < 0.05:
    print(f"  ICC < 0.05 — community clustering explains <5% of variance.")
    print(f"  OLS is justified; mixed-effects adds minimal benefit.")
elif icc < 0.10:
    print(f"  ICC 0.05-0.10 — modest community clustering.")
    print(f"  Mixed-effects is prudent but OLS is not severely biased.")
else:
    print(f"  ICC > 0.10 — substantial community clustering.")
    print(f"  Mixed-effects model strongly justified over OLS.")

# =============================================================================
# 5. SAVE RESULTS
# =============================================================================

# Fixed effects
fe_rows = []
for i, name in enumerate(param_names):
    fe_rows.append({
        "parameter": name,
        "label":     PRED_LABELS.get(name, name),
        "beta":      beta[i],
        "se":        se_beta[i],
        "t":         t_vals[i],
        "p":         p_vals[i],
        "ci_lo":     ci_lo[i],
        "ci_hi":     ci_hi[i],
        "sig_05":    p_vals[i] < 0.05,
    })
fe_df = pd.DataFrame(fe_rows)
fe_df.to_csv("evi_mixed_effects_results.csv", index=False)

# Variance components
vc_df = pd.DataFrame([{
    "tau2_random_intercept": tau2,
    "sigma2_residual":       sigma2,
    "total_variance":        total_var,
    "ICC":                   icc,
    "n_households":          n,
    "J_communities":         J,
    "converged":             converged,
}])
vc_df.to_csv("evi_mixed_effects_variance.csv", index=False)
print("\nResults saved: evi_mixed_effects_results.csv, evi_mixed_effects_variance.csv")

# =============================================================================
# 6. OLS vs MIXED-EFFECTS COMPARISON
# =============================================================================
# Refit OLS on the same cleaned sample for a fair side-by-side comparison.

beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
y_hat_ols  = X @ beta_ols
resid_comp = y - y_hat_ols
mse_comp   = np.sum(resid_comp**2) / (n - k)
XtX_inv    = np.linalg.pinv(X.T @ X)
se_ols     = np.sqrt(np.diag(mse_comp * XtX_inv))
t_ols      = beta_ols / se_ols
p_ols      = 2 * stats.t.sf(np.abs(t_ols), df=n - k)

comp_rows = []
for i, name in enumerate(param_names):
    comp_rows.append({
        "parameter":   name,
        "label":       PRED_LABELS.get(name, name),
        "beta_OLS":    beta_ols[i],
        "se_OLS":      se_ols[i],
        "p_OLS":       p_ols[i],
        "beta_ME":     beta[i],
        "se_ME":       se_beta[i],
        "p_ME":        p_vals[i],
        "beta_change": beta[i] - beta_ols[i],
        "pct_change":  (beta[i] - beta_ols[i]) / (abs(beta_ols[i]) + 1e-10) * 100,
    })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv("evi_mixed_effects_comparison.csv", index=False)
print("Saved: evi_mixed_effects_comparison.csv")

# =============================================================================
# 7. FIGURE 1 — COEFFICIENT COMPARISON PLOT (OLS vs Mixed-Effects)
# =============================================================================

plot_df = comp_df[comp_df["parameter"] != "Intercept"].copy()
plot_df = plot_df.sort_values("beta_ME")
labels  = plot_df["label"].values
y_pos   = np.arange(len(labels))

fig1, ax1 = plt.subplots(figsize=(10, 7))

# Mixed-effects bars (primary model)
ax1.barh(y_pos + 0.18, plot_df["beta_ME"].values,
         height=0.32, color="#2196A6", alpha=0.85,
         label="Mixed-effects (random intercept)", zorder=3)

# OLS bars (comparison)
ax1.barh(y_pos - 0.18, plot_df["beta_OLS"].values,
         height=0.32, color="#F4A71D", alpha=0.7,
         label="OLS (Individual Model)", zorder=3)

ax1.axvline(0, color="#888888", linewidth=1.0, zorder=2)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel(
    "β coefficient  (positive = higher EVI = more vulnerable)",
    fontsize=10,
)
ax1.set_title(
    f"Fixed Effect Coefficients — EVI: OLS vs Mixed-Effects Model\n"
    f"ICC = {icc:.3f} ({icc*100:.1f}% of variance at community level), "
    f"n = {n}, J = {J} communities",
    fontsize=11, fontweight="bold", pad=12,
)
ax1.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#cccccc")
ax1.set_axisbelow(True)

plt.tight_layout()
plt.savefig("evi_mixed_effects_plot.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nFigure 1 saved: evi_mixed_effects_plot.png")

# =============================================================================
# 8. FIGURE 2 — CATERPILLAR PLOT OF RANDOM INTERCEPTS
# =============================================================================
# Each point is one community's random intercept (u_j) ± 95% CI.
# Faded points: CI crosses zero — not significantly different from average.
# Red = above average vulnerability, Blue = below average vulnerability.

nj_vec    = np.array([np.sum(gj == j) for j in range(J)])
se_u      = np.sqrt(tau2 * sigma2 / (nj_vec * tau2 + sigma2))
t_crit2   = stats.t.ppf(0.975, df=n - k - J)

order        = np.argsort(u_j)
u_sorted     = u_j[order]
se_sorted    = se_u[order]
nj_sorted    = nj_vec[order]
comm_sorted  = [communities[i] for i in order]
labels_sorted = [
    c.replace("_", " ").replace("ward", "Ward").title()
    + f" (n={int(nj_sorted[i])})"
    for i, c in enumerate(comm_sorted)
]

fig2, ax2 = plt.subplots(figsize=(9, 8))

for i, (uj, sej, lab) in enumerate(zip(u_sorted, se_sorted, labels_sorted)):
    ci_lo_u = uj - t_crit2 * sej
    ci_hi_u = uj + t_crit2 * sej
    sig     = (ci_lo_u > 0) or (ci_hi_u < 0)   # CI does not cross zero
    color   = "#D94F3D" if uj > 0 else "#2196A6"
    alpha   = 1.0 if sig else 0.40

    ax2.errorbar(
        x=uj, y=i,
        xerr=[[uj - ci_lo_u], [ci_hi_u - uj]],
        fmt="o", color=color, ecolor=color,
        elinewidth=1.5, capsize=4, markersize=6,
        alpha=alpha, zorder=3,
    )

ax2.axvline(0, color="#888888", linewidth=1.2, linestyle="--", zorder=1)

ax2.set_yticks(range(J))
ax2.set_yticklabels(labels_sorted, fontsize=8)
ax2.set_xlabel(
    "Random Intercept (u_j)  ±95% CI\n"
    "Red = above average vulnerability  |  Blue = below average  |  "
    "Faded = CI crosses zero",
    fontsize=9,
)
ax2.set_title(
    "EVI — Caterpillar Plot: Community-level Random Intercepts\n"
    f"τ² = {tau2:.4f},  σ² = {sigma2:.4f},  ICC = {icc:.3f}",
    fontsize=11, fontweight="bold", pad=12,
)
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#cccccc")
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig("evi_random_intercepts.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 2 saved: evi_random_intercepts.png")

# =============================================================================
# 9. SAVE RANDOM INTERCEPTS TO CSV
# =============================================================================
# Exports the full per-community random intercept table:
#   community       : VDC/ward name
#   n               : number of households in that community
#   u_j             : random intercept (BLUP estimate)
#   se_u            : standard error of the BLUP
#   ci_lo / ci_hi   : 95% confidence interval
#   sig             : True if CI does not cross zero
#   direction       : "above average" or "below average" vulnerability
#
# u_j > 0  means the community has higher vulnerability than the grand mean
#           after controlling for all fixed-effect predictors.
# u_j < 0  means lower vulnerability than the grand mean.
#
# Reference:
#   Raudenbush & Bryk (2002). Hierarchical Linear Models (2nd ed.). SAGE.

ri_rows = []
for idx in order:           # already sorted low → high by caterpillar plot
    uj      = u_j[idx]
    sej     = se_u[idx]
    ci_lo_v = uj - t_crit2 * sej
    ci_hi_v = uj + t_crit2 * sej
    ri_rows.append({
        "community":    communities[idx],
        "n":            int(nj_vec[idx]),
        "u_j":          round(float(uj),  6),
        "se_u":         round(float(sej), 6),
        "ci_lo_95":     round(float(ci_lo_v), 6),
        "ci_hi_95":     round(float(ci_hi_v), 6),
        "sig":          bool((ci_lo_v > 0) or (ci_hi_v < 0)),
        "direction":    "above average" if uj > 0 else "below average",
    })

ri_df = pd.DataFrame(ri_rows)
ri_df.to_csv("evi_random_intercepts.csv", index=False)
print("Saved: evi_random_intercepts.csv")
print(ri_df.to_string(index=False))

print("\nMixed-effects model complete.")
