"""
stage08_fvi_stepwise.py
==================
Stage 4 | Stepwise OLS Regression

Input  : analysis_dataset.csv  (from Stage 05b)
Outputs: fvi_stepwise_stage1_results.csv   — domain stepwise results
         fvi_stepwise_stage2_results.csv   — Individual Model (IM)
         fvi_stepwise_stage3_results.csv   — Composite Model (CM)
         fvi_stepwise_diagnostics.png      — residual diagnostic plots
         fvi_stepwise_coefficients.png     — coefficient comparison plot

Description
-----------
Four-stage stepwise OLS regression following Saputra, Schwarz & Hendriks
(2026). Stage 1 runs domain-level stepwise regressions (PIN=0.05, POUT=0.10);
Stage 2 pools significant variables into a joint Individual Model; Stage 3
tests composite MAO net scores (Composite Model); Stage 4 runs diagnostics.
Outliers are removed at ±3 SD of standardised residuals.

Reference
---------
Saputra, A., Schwarz, J., & Hendriks, E. (2026). Identifying factors
    influencing housing safety in post-earthquake reconstruction by
    households in Nepal. IJDRR, 133, 105913.
    https://doi.org/10.1016/j.ijdrr.2025.105913
"""

# ─── Pipeline note ────────────────────────────────────────────────────────────
# When run via run_fvi_pipeline.py, the working directory is OUTPUT_DIR.
# FILE_PATH = str(Path.cwd() / "analysis_dataset.csv") resolves to OUTPUT_DIR/analysis_dataset.csv.
# ─────────────────────────────────────────────────────────────────────────────

"""
Stepwise Linear Regression: FVI_norm_1_5 as Dependent Variable
===============================================================
Adapted from the methodology of Saputra, Schwarz & Hendriks (2026):
  "Identifying factors influencing housing safety in post-earthquake
   reconstruction by households in Nepal"
  Int. J. Disaster Risk Reduction 133 (2026) 105913
  https://doi.org/10.1016/j.ijdrr.2025.105913

Protocol:
  Stage 1: Stepwise OLS per domain (motivation, ability, opportunity,
           flood experience, socioeconomic) — PIN=0.05, POUT=0.10
  Stage 2: Joint regression using significant variables from Stage 1
           (Individual Model, IM)
  Stage 3: Composite variable model (Composite Model, CM)
  Stage 4: Diagnostics — VIF, Durbin-Watson, residual plots,
           outlier detection at ±3 SD

Missing data: pairwise deletion
Outliers:     removed at ±3 SD of standardised residuals (as per paper)

Outputs:
  fvi_stepwise_stage1_results.csv   — domain-level stepwise results
  fvi_stepwise_stage2_results.csv   — joint Individual Model results
  fvi_stepwise_stage3_results.csv   — Composite Model results
  fvi_stepwise_diagnostics.png      — residual plots
  fvi_stepwise_coefficients.png     — coefficient plot for both models
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    FVI_DOMAINS,
    FVI_ALL_DOMAIN_VARS,
    CAP_COLS,
    assert_no_drift,
)

warnings.filterwarnings("ignore")

# Verify pipeline integrity invariants before doing any work.
# This stage PRODUCES fvi_im_predictors.csv, so we use producer_fvi mode
# which skips the FVI_IM_PREDICTORS check (any stale file is about to be
# replaced by this stage's output).
assert_no_drift("producer_fvi")

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

FILE_PATH = str(Path.cwd() / "analysis_dataset.csv")
df = pd.read_csv(FILE_PATH, encoding='utf-8-sig', low_memory=False)

# Clean dependent variable
df["FVI_norm_1_5"] = pd.to_numeric(df["FVI_norm_1_5"], errors="coerce")

# =============================================================================
# 2. CLEAN PREDICTOR COLUMNS TO NUMERIC
# =============================================================================

def clean_col(series):
    """Replace blank-string placeholders with NaN, cast to float."""
    return pd.to_numeric(
        series.replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )

# Apply to all candidate columns
# ── Updated from bivariate_screening_significant.csv (32 variables, p<0.05) ──
candidate_cols = FVI_ALL_DOMAIN_VARS


# Cap known data-entry outliers before cleaning
# AB_Physical_capacity_Pos and OP_Training_Neg had values of 1e11 (entry errors)
for _cap_col in CAP_COLS:
    if _cap_col in df.columns:
        _s = pd.to_numeric(df[_cap_col], errors="coerce")
        df[_cap_col] = _s.where(_s <= 100)  # values > 100 become NaN

for col in candidate_cols:
    if col in df.columns:
        df[col] = clean_col(df[col])

# =============================================================================
# 3. DEFINE DOMAIN GROUPS
# =============================================================================
# Each domain lists only the variables that passed bivariate screening (p<0.05).
# Stepwise regression within each domain selects the final subset.

DOMAINS = FVI_DOMAINS

DEPENDENT = "FVI_norm_1_5"
PIN  = _PIPELINE_CFG["STEPWISE_PIN"]   # p-value threshold to ENTER variable
POUT = _PIPELINE_CFG["STEPWISE_POUT"]  # p-value threshold to REMOVE variable

# =============================================================================
# 4. OLS HELPER FUNCTIONS
# =============================================================================

def ols_fit(y, X_df):
    """
    Fit OLS using numpy. Returns dict with coefficients, standard errors,
    t-statistics, p-values, 95% CIs, R², adjusted R², and VIF.

    Parameters
    ----------
    y    : np.ndarray  — dependent variable (n,)
    X_df : pd.DataFrame — predictors (without intercept column)

    Returns
    -------
    dict with 'params', 'se', 't', 'p', 'ci_lo', 'ci_hi',
              'r2', 'adj_r2', 'n', 'vif', 'dw'
    """
    n    = len(y)
    k    = X_df.shape[1]           # number of predictors (excluding intercept)
    X    = np.column_stack([np.ones(n), X_df.values.astype(float)])

    # OLS solution
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat  = X @ coeffs
    resid  = y - y_hat
    df_res = n - k - 1             # residual degrees of freedom

    # Mean squared error
    mse    = np.sum(resid**2) / df_res

    # Variance-covariance matrix of coefficients: mse * (X'X)^-1
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_b   = mse * XtX_inv
    se      = np.sqrt(np.diag(var_b))

    # t-statistics and p-values (two-tailed)
    t_vals = coeffs / se
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=df_res)

    # 95% confidence intervals
    t_crit = stats.t.ppf(0.975, df=df_res)
    ci_lo  = coeffs - t_crit * se
    ci_hi  = coeffs + t_crit * se

    # R² and adjusted R²
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum(resid**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_res if df_res > 0 else np.nan

    # Variance Inflation Factor (VIF) for each predictor
    # VIF_j = 1 / (1 - R²_j) where R²_j is from regressing predictor j
    # on all other predictors
    vif = {}
    cols = list(X_df.columns)
    for j, col in enumerate(cols):
        X_other = X_df.drop(columns=col).values.astype(float)
        x_j     = X_df[col].values.astype(float)
        if X_other.shape[1] == 0:
            vif[col] = 1.0
            continue
        X_o = np.column_stack([np.ones(n), X_other])
        c_o, _, _, _ = np.linalg.lstsq(X_o, x_j, rcond=None)
        xj_hat = X_o @ c_o
        ss_tot_j = np.sum((x_j - x_j.mean())**2)
        ss_res_j = np.sum((x_j - xj_hat)**2)
        r2_j     = 1 - ss_res_j / ss_tot_j if ss_tot_j > 0 else 0.0
        vif[col] = 1 / (1 - r2_j) if r2_j < 1 else np.inf

    # Durbin-Watson statistic (checks autocorrelation of residuals)
    dw = np.sum(np.diff(resid)**2) / np.sum(resid**2)

    return {
        "intercept": coeffs[0],
        "params":  dict(zip(cols, coeffs[1:])),
        "se":      dict(zip(cols, se[1:])),
        "t":       dict(zip(cols, t_vals[1:])),
        "p":       dict(zip(cols, p_vals[1:])),
        "ci_lo":   dict(zip(cols, ci_lo[1:])),
        "ci_hi":   dict(zip(cols, ci_hi[1:])),
        "r2":      r2,
        "adj_r2":  adj_r2,
        "n":       n,
        "vif":     vif,
        "dw":      dw,
        "resid":   resid,
        "y_hat":   y_hat,
    }


def stepwise_ols(y_series, X_df_full, pin=0.05, pout=0.10):
    """
    Forward stepwise OLS: mirrors SPSS /METHOD=STEPWISE with
    /CRITERIA=PIN(.05) POUT(.10).

    Algorithm:
      1. Start with no predictors in model.
      2. At each step, test adding each candidate variable. Add the one
         with lowest p-value IF p < PIN.
      3. After adding, check all variables in model: remove any with p > POUT.
      4. Repeat until no variable can be added or removed.

    Parameters
    ----------
    y_series   : pd.Series  — dependent variable
    X_df_full  : pd.DataFrame — all candidate predictors
    pin        : float — p-value threshold to enter (default 0.05)
    pout       : float — p-value threshold to remove (default 0.10)

    Returns
    -------
    selected   : list of selected variable names
    final_fit  : dict from ols_fit() for the final model
    steps      : list of step records for audit trail
    """
    # Pairwise complete cases across all variables
    combined = pd.concat([y_series, X_df_full], axis=1).dropna()
    y   = combined.iloc[:, 0].values
    X_a = combined.iloc[:, 1:]

    selected = []    # variables currently in the model
    steps    = []    # audit log of each step

    for iteration in range(100):   # max 100 steps to prevent infinite loop
        # --- FORWARD STEP: try adding each candidate ---
        candidates = [c for c in X_a.columns if c not in selected]
        best_col, best_p = None, 1.0

        for col in candidates:
            trial_cols = selected + [col]
            fit = ols_fit(y, X_a[trial_cols])
            p_col = fit["p"][col]
            if p_col < best_p:
                best_p   = p_col
                best_col = col

        added = False
        if best_col is not None and best_p < pin:
            selected.append(best_col)
            steps.append({"step": iteration + 1, "action": "ADD",
                          "variable": best_col, "p": best_p})
            added = True

        # --- BACKWARD STEP: remove any variable with p > POUT ---
        removed = True
        while removed and len(selected) > 0:
            removed = False
            if len(selected) == 0:
                break
            fit = ols_fit(y, X_a[selected])
            # Find variable with highest p-value
            p_in_model = {v: fit["p"][v] for v in selected}
            worst_col  = max(p_in_model, key=p_in_model.get)
            if p_in_model[worst_col] > pout:
                selected.remove(worst_col)
                steps.append({"step": iteration + 1, "action": "REMOVE",
                              "variable": worst_col,
                              "p": p_in_model[worst_col]})
                removed = True

        # Stop if nothing was added or removed this iteration
        if not added and not removed:
            break

    # Final model fit
    if len(selected) > 0:
        combined2 = pd.concat([y_series, X_a[selected]], axis=1).dropna()
        y_f  = combined2.iloc[:, 0].values
        X_f  = combined2.iloc[:, 1:]
        final_fit = ols_fit(y_f, X_f)
    else:
        final_fit = None

    return selected, final_fit, steps


# =============================================================================
# 5. REMOVE OUTLIERS AT ±3 SD OF STANDARDISED RESIDUALS
# =============================================================================

def remove_outliers_3sd(y_series, X_df, selected_cols):
    """
    Fit model, compute standardised residuals, remove cases with
    |z_resid| > 3. Returns cleaned y and X. Mirrors SPSS /CASEWISE OUTLIERS(3).
    """
    combined = pd.concat([y_series, X_df[selected_cols]], axis=1).dropna()
    y_c = combined.iloc[:, 0].values
    X_c = combined.iloc[:, 1:]
    fit = ols_fit(y_c, X_c)

    # Standardised residuals: resid / sqrt(MSE)
    mse      = np.sum(fit["resid"]**2) / (len(y_c) - len(selected_cols) - 1)
    z_resid  = fit["resid"] / np.sqrt(mse)
    mask     = np.abs(z_resid) <= 3   # keep cases within ±3 SD
    n_out    = (~mask).sum()

    y_clean = pd.Series(y_c[mask], name=y_series.name)
    X_clean = pd.DataFrame(X_c.values[mask], columns=selected_cols)

    return y_clean, X_clean, n_out


# =============================================================================
# 6. STAGE 1 — DOMAIN-LEVEL STEPWISE REGRESSIONS
# =============================================================================

print("=" * 60)
print("STAGE 1: Domain-level stepwise regressions")
print("=" * 60)

stage1_results  = []   # summary rows for CSV output
stage1_selected = {}   # significant variables per domain

for domain, cols in DOMAINS.items():
    # Keep only columns that exist in the dataset
    avail = [c for c in cols if c in df.columns]
    if not avail:
        print(f"\n  {domain}: no columns available — skipping")
        continue

    selected, fit, steps = stepwise_ols(df[DEPENDENT], df[avail], PIN, POUT)

    print(f"\n  {domain}: {len(selected)} variable(s) selected")
    for col in selected:
        print(f"    {col}: β={fit['params'][col]:.4f}, "
              f"p={fit['p'][col]:.4f}, VIF={fit['vif'][col]:.2f}")

    stage1_selected[domain] = selected

    # Record results
    if fit:
        for col in selected:
            stage1_results.append({
                "domain":  domain,
                "variable": col,
                "beta":    fit["params"][col],
                "se":      fit["se"][col],
                "t":       fit["t"][col],
                "p":       fit["p"][col],
                "ci_lo":   fit["ci_lo"][col],
                "ci_hi":   fit["ci_hi"][col],
                "vif":     fit["vif"][col],
                "model_r2":     fit["r2"],
                "model_adj_r2": fit["adj_r2"],
                "model_n":      fit["n"],
            })

stage1_df = pd.DataFrame(stage1_results)
stage1_df.to_csv("fvi_stepwise_stage1_results.csv", index=False)
print("\nStage 1 results saved.")

# =============================================================================
# 7. STAGE 2 — JOINT REGRESSION (INDIVIDUAL MODEL)
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 2: Joint regression — Individual Model (IM)")
print("=" * 60)

# Pool all Stage 1 significant variables
joint_pool = []
for domain, sel in stage1_selected.items():
    joint_pool.extend(sel)

joint_pool = list(dict.fromkeys(joint_pool))   # deduplicate, preserve order
print(f"\n  Variables entering joint model: {len(joint_pool)}")
print(f"  {joint_pool}")

if joint_pool:
    # First pass: stepwise on pooled variables
    sel_joint, fit_joint, _ = stepwise_ols(
        df[DEPENDENT], df[joint_pool], PIN, POUT)

    print(f"\n  Variables retained after joint stepwise: {len(sel_joint)}")

    if sel_joint and fit_joint:
        # Outlier removal at ±3 SD (as per paper's /CASEWISE OUTLIERS(3))
        y_clean, X_clean, n_out = remove_outliers_3sd(
            df[DEPENDENT], df[sel_joint], sel_joint)
        print(f"  Outliers removed (|z|>3): {n_out}")

        # Refit on cleaned data
        sel_final, fit_final, _ = stepwise_ols(
            y_clean, X_clean, PIN, POUT)

        print(f"\n  INDIVIDUAL MODEL — Final results (n={fit_final['n']}):")
        print(f"  Adjusted R²: {fit_final['adj_r2']:.3f}")
        print(f"  Durbin-Watson: {fit_final['dw']:.3f}")
        print(f"  {'Variable':<40} {'β':>8} {'SE':>8} {'t':>8} {'p':>8} {'VIF':>6}")
        print("  " + "-" * 80)
        for col in sel_final:
            print(f"  {col:<40} "
                  f"{fit_final['params'][col]:>8.4f} "
                  f"{fit_final['se'][col]:>8.4f} "
                  f"{fit_final['t'][col]:>8.3f} "
                  f"{fit_final['p'][col]:>8.4f} "
                  f"{fit_final['vif'][col]:>6.2f}")

        # Save Stage 2 results
        s2_rows = []
        for col in sel_final:
            s2_rows.append({
                "model":   "Individual Model (IM)",
                "variable": col,
                "beta":    fit_final["params"][col],
                "se":      fit_final["se"][col],
                "t":       fit_final["t"][col],
                "p":       fit_final["p"][col],
                "ci_lo":   fit_final["ci_lo"][col],
                "ci_hi":   fit_final["ci_hi"][col],
                "vif":     fit_final["vif"][col],
                "adj_r2":  fit_final["adj_r2"],
                "n":       fit_final["n"],
                "dw":      fit_final["dw"],
            })
        pd.DataFrame(s2_rows).to_csv(
            "fvi_stepwise_stage2_results.csv", index=False)
        print("\nStage 2 results saved.")

        # ── Write retained predictor list for Stage 10 ────────────────────
        # Stage 10 (mixed-effects) reads this file instead of the hardcoded
        # FVI_IM_PREDICTORS list in pipeline_config.py. This ensures the
        # mixed-effects model always uses exactly the predictors that survived
        # the current run's stepwise elimination — not a stale manual list.
        pd.DataFrame({"predictor": sel_final}).to_csv(
            "fvi_im_predictors.csv", index=False)
        print("Predictor list saved: fvi_im_predictors.csv "
              f"({len(sel_final)} variables → Stage 10)")

# =============================================================================
# 8. STAGE 3 — COMPOSITE MODEL
# =============================================================================
# Composite variables: net MAO scores (Positive - Negative proportion)
# and composite flood experience index.
# Mirrors the paper's approach of creating summary indices per construct.

print("\n" + "=" * 60)
print("STAGE 3: Composite Model (CM)")
print("=" * 60)

# --- Create composite variables ---

# MAO net scores: Positive proportion minus Negative proportion
# A positive net score = more positive than negative perception
df["MAO_Utility_net"] = clean_col(df["Uti_Perc_Pos_expression"])
df["MAO_Applicability_net"]  = (clean_col(df["App_Perc_Pos_expression"]) -
                                clean_col(df["App_Perc_Neg_expression"]))
df["MAO_Selfefficacy_net"]   =  clean_col(df["AB_Selfefficacy_Pos"])
df["MAO_Financial_barrier"]  =  (clean_col(df["AB_Financial_capacity_Pos"]) -
                                clean_col(df["AB_Financial_capacity_Neg"]))
df["MAO_Physical_capacity_net"] = clean_col(df["AB_Physical_capacity_Neg"])
df["MAO_Time_net"] = clean_col(df["AB_Time_Pos"])
df["MAO_Location_net"]       =  clean_col(df["AB_Location_Pos"])
df["MAO_Training_net"]       =  (clean_col(df["OP_Training_Pos"]) - 
                                  clean_col(df["OP_Training_Neg"]))
df["MAO_Materials_barrier"]  =  (clean_col(df["OP_Materials_Pos"])- 
                                 clean_col(df["OP_Materials_Neg"]))
df["MAO_Manpower_net"] = clean_col(df["OP_Manpower_Pos"])

# Flood experience composite: sum of experience scores
# (frequency, depth, past damage, worry — all on numeric scales)
df["Flood_Exp_composite"] = (
    pd.to_numeric(df["DE02_flood"], errors="coerce").fillna(0) +
    #pd.to_numeric(df["DE04_flood_depth_score"],     errors="coerce").fillna(0) +
    #pd.to_numeric(df["PP02_damage_score"],          errors="coerce").fillna(0)
    pd.to_numeric(df["DE12_shelter_score"], errors="coerce").fillna(0)
)

# Risk perception composite: worry + feel safe (reverse coded) + future expectation
#df["Risk_Perception_composite"] = (
#    pd.to_numeric(df["DE08_worry_score"],       errors="coerce").fillna(0) +
#    pd.to_numeric(df["DE13_feel_safe_score"],   errors="coerce").fillna(0) +
#    pd.to_numeric(df["DE06_flood_future_score"],errors="coerce").fillna(0)
#)

# ── Updated socioeconomic controls: use screened variables only ──────────────
COMPOSITE_VARS = [
    "MAO_Utility_net",
    "MAO_Applicability_net",
    "MAO_Selfefficacy_net",
    "MAO_Physical_capacity_net",
    "MAO_Financial_barrier",
    "MAO_Location_net",
    "MAO_Training_net",
    "MAO_Time_net",
    "MAO_Manpower_net",
    "MAO_Materials_barrier",
    "Flood_Exp_composite",
    #"Risk_Perception_composite",
    # Socioeconomic controls — screened variables only (10 from bivariate screening)
    "HC003_Respondent_s_age",
    "SE01_edu_no_education",
    "SE01_edu_can_read_write",
    "SE01_edu_high_school",
    "SE01_edu_university",
    "SE04_agriculture",
    "SE04_business",
    "SE04_day_labour",
    "SE04_government_officer",
    "SE04_remittances",
]

# Run composite model stepwise
sel_cm, fit_cm, _ = stepwise_ols(
    df[DEPENDENT], df[COMPOSITE_VARS], PIN, POUT)

if sel_cm and fit_cm:
    # Outlier removal
    y_cm, X_cm, n_out_cm = remove_outliers_3sd(
        df[DEPENDENT], df[sel_cm], sel_cm)
    print(f"  Outliers removed: {n_out_cm}")

    sel_cm_f, fit_cm_f, _ = stepwise_ols(y_cm, X_cm, PIN, POUT)

    print(f"\n  COMPOSITE MODEL — Final results (n={fit_cm_f['n']}):")
    print(f"  Adjusted R²: {fit_cm_f['adj_r2']:.3f}")
    print(f"  Durbin-Watson: {fit_cm_f['dw']:.3f}")
    print(f"  {'Variable':<40} {'β':>8} {'SE':>8} {'t':>8} {'p':>8} {'VIF':>6}")
    print("  " + "-" * 80)
    for col in sel_cm_f:
        print(f"  {col:<40} "
              f"{fit_cm_f['params'][col]:>8.4f} "
              f"{fit_cm_f['se'][col]:>8.4f} "
              f"{fit_cm_f['t'][col]:>8.3f} "
              f"{fit_cm_f['p'][col]:>8.4f} "
              f"{fit_cm_f['vif'][col]:>6.2f}")

    s3_rows = []
    for col in sel_cm_f:
        s3_rows.append({
            "model":    "Composite Model (CM)",
            "variable":  col,
            "beta":     fit_cm_f["params"][col],
            "se":       fit_cm_f["se"][col],
            "t":        fit_cm_f["t"][col],
            "p":        fit_cm_f["p"][col],
            "ci_lo":    fit_cm_f["ci_lo"][col],
            "ci_hi":    fit_cm_f["ci_hi"][col],
            "vif":      fit_cm_f["vif"][col],
            "adj_r2":   fit_cm_f["adj_r2"],
            "n":        fit_cm_f["n"],
            "dw":       fit_cm_f["dw"],
        })
    pd.DataFrame(s3_rows).to_csv(
        "fvi_stepwise_stage3_results.csv", index=False)
    print("\nStage 3 results saved.")

# =============================================================================
# 9. STAGE 4 — DIAGNOSTICS PLOTS
# =============================================================================
# Mirrors SPSS: /SCATTERPLOT(*ZRESID,*ZPRED)
#               /RESIDUALS DURBIN HISTOGRAM(ZRESID) NORMPROB(ZRESID)

print("\n" + "=" * 60)
print("STAGE 4: Diagnostics")
print("=" * 60)

def make_diagnostic_plots(fit, title, axes):
    """
    Generate residual diagnostic plots for a fitted model.
    ax[0] = standardised residuals vs predicted (scatterplot)
    ax[1] = histogram of standardised residuals
    ax[2] = normal probability plot (P-P plot)
    """
    resid  = fit["resid"]
    y_hat  = fit["y_hat"]
    n      = len(resid)
    k      = len(fit["params"])
    mse    = np.sum(resid**2) / (n - k - 1)
    z_res  = resid / np.sqrt(mse)   # standardised residuals

    # Plot 1: z-residuals vs predicted values
    axes[0].scatter(y_hat, z_res, alpha=0.25, s=12, color="#2196A6")
    axes[0].axhline(0, color="#888", linewidth=1)
    axes[0].axhline(3,  color="#D94F3D", linewidth=0.8, linestyle="--")
    axes[0].axhline(-3, color="#D94F3D", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Predicted FVI", fontsize=9)
    axes[0].set_ylabel("Standardised Residual", fontsize=9)
    axes[0].set_title(f"{title}\nResiduals vs Predicted", fontsize=9)
    axes[0].spines[["top","right"]].set_visible(False)

    # Plot 2: histogram of standardised residuals
    axes[1].hist(z_res, bins=30, color="#8E44AD", alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("Standardised Residual", fontsize=9)
    axes[1].set_ylabel("Frequency", fontsize=9)
    axes[1].set_title("Histogram of Residuals", fontsize=9)
    axes[1].spines[["top","right"]].set_visible(False)

    # Plot 3: Normal P-P plot
    z_sorted = np.sort(z_res)
    # Theoretical normal quantiles
    p_vals = (np.arange(1, n+1) - 0.5) / n
    norm_q = stats.norm.ppf(p_vals)
    axes[2].plot(norm_q, z_sorted, "o", markersize=3, alpha=0.4,
                 color="#27AE60")
    # Reference line
    mn, mx = norm_q.min(), norm_q.max()
    axes[2].plot([mn, mx], [mn, mx], "r--", linewidth=1)
    axes[2].set_xlabel("Expected Normal", fontsize=9)
    axes[2].set_ylabel("Observed Residual", fontsize=9)
    axes[2].set_title("Normal P-P Plot", fontsize=9)
    axes[2].spines[["top","right"]].set_visible(False)


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Regression Diagnostics: FVI Stepwise Models\n"
             "(Replicating Saputra et al. 2026 protocol)",
             fontsize=12, fontweight="bold")

# Individual model diagnostics
if 'fit_final' in locals() and fit_final:
    make_diagnostic_plots(fit_final, "Individual Model (IM)", axes[0])

# Composite model diagnostics
# Draw composite model diagnostics only if stepwise selected at least one predictor.
# fit_cm_f is only assigned inside the 'if sel_cm and fit_cm:' block above;
# locals() correctly returns False when no predictors survived selection.
if 'fit_cm_f' in locals() and fit_cm_f:
    make_diagnostic_plots(fit_cm_f, "Composite Model (CM)", axes[1])
else:
    # No composite predictors survived stepwise selection — annotate the
    # bottom row so the blank subplots are self-explanatory in the output.
    for ax in axes[1]:
        ax.text(0.5, 0.5,
                "Composite Model: no predictors\nsurvived stepwise selection",
                ha='center', va='center', fontsize=10,
                color='#888888', transform=ax.transAxes)
        ax.set_axis_off()

plt.tight_layout()
plt.savefig("fvi_stepwise_diagnostics.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("  Diagnostic plots saved.")
print("\nAll outputs saved to ")