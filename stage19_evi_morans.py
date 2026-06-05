"""
stage14_evi_morans.py
==================
Stage 5 | EVI — Spatial Autocorrelation (Moran's I)

Input  : imputed_EVI_scores.csv  (from Stage 0c)
Outputs: evi_morans_sensitivity.csv, evi_morans_correlogram.png,
         evi_residual_map.png, evi_residual_by_community.png,
         evi_local_morans_results.csv, evi_local_morans_map.png

Description
-----------
Mirrors stage09_fvi_morans.py exactly. Tests whether residuals from the
EVI Individual Model exhibit spatial autocorrelation using global
Moran's I and LISA.

Reference
---------
Anselin, L. (1995). Local indicators of spatial association — LISA.
    Geographical Analysis, 27(2), 93–115.
    https://doi.org/10.1111/j.1538-4632.1995.tb00338.x
"""

# ─── Pipeline note ────────────────────────────────────────────────────────────
# When run via run_evi_pipeline.py, the working directory is OUTPUT_DIR.
# ─────────────────────────────────────────────────────────────────────────────

"""
Autocorrelation Diagnosis: Spatial and Community-level Clustering
=================================================================
Tests whether residuals from the EVI stepwise regression are 
autocorrelated due to spatial proximity or community clustering.

Model used: Individual Model (IM) — Stage 2 of 4_2__fvi_stepwise_updated.ipynb
Rationale:  The IM is the primary reported model. The Composite Model is a robustness 
check. Moran's I is run on the IM because:
              (1) IM variables (AB_Location_Pos, DE12_shelter_score, DE13_feel_safe_
              score) are spatially-grounded — residual 
              clustering after controlling for these is the meaningful test.
              (2) The IM is the model whose independence assumption is reported.
            
            See: Saputra, Schwarz & Hendriks (2026), IJDRR 133, 105913.
                 https://doi.org/10.1016/j.ijdrr.2025.105913

NOTE on sample size: IM n=1,222 (down from 2,363 in the prior run).
  The drop is driven by two variables with high missingness:
    APP_Perc_Pos_expression_CAT_33PROCENT : 52% missing
    DE12_shelter_score                    : 17% missing
  Listwise deletion across all 10 predictors compounds this.
  GPS coverage within the IM complete cases is 97.6%, so the spatial
  sample is not geographically biased despite the smaller n.

Four diagnostic outputs:
  1. GPS coverage report by VDC (console + text file)
  2. Spatial correlogram — Moran's I across 7 distance thresholds
  3. Residual map by GPS coordinates
  4. Boxplot of residuals by community

References:
  Aksha, S.K., Juran, L., Resler, L.M., & Zhang, Y. (2019). An analysis 
  of social vulnerability to natural hazards in Nepal using a modified 
  social vulnerability index. International Journal of Disaster Risk 
  Science, 10, 103–116. https://doi.org/10.1007/s13753-018-0192-7


Outputs:
  evi_gps_coverage.txt          - GPS coverage by VDC
  evi_morans_correlogram.png    - Moran's I across distance thresholds
  evi_morans_sensitivity.csv    - Moran's I table (all thresholds)
  evi_morans_i_results.txt      - Moran's I final report
  evi_residual_map.png          - GPS map of residuals
  evi_residual_by_community.png - boxplot by VDC/ward
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

FILE_PATH = str(Path.cwd() / "imputed_EVI_scores.csv")   # scored dataset: contains EVI_norm_1_5,
                                       # all predictors, GPS, and community
DEPENDENT = "EVI_norm_1_5"

# Individual Model predictors from Stage 2 of 4_2__fvi_stepwise_updated.ipynb
# (updated 32-variable run; final n=1,222 after listwise deletion)
# Key missingness drivers:
#   APP_Perc_Pos_expression_CAT_33PROCENT : 52% missing
#   DE12_shelter_score                    : 17% missing
from pipeline_config import (
    EVI_IM_PREDICTORS,
    CAP_COLS,
)

# ── Load Individual Model predictor list from Stage 18 output ────────────────
# Stage 18 writes evi_im_predictors.csv after stepwise selection completes.
# Reading from that file ensures Moran's I uses the exact predictors that
# survived the current run, not a stale hardcoded list in pipeline_config.
# Falls back to EVI_IM_PREDICTORS from pipeline_config if the file is absent.
_im_csv_evi = Path.cwd() / "evi_im_predictors.csv"
if _im_csv_evi.exists():
    import pandas as _pd_im
    _LOADED_IM_PREDICTORS = _pd_im.read_csv(_im_csv_evi)["predictor"].tolist()
    print(f"Loaded {len(_LOADED_IM_PREDICTORS)} predictors from evi_im_predictors.csv")
else:
    _LOADED_IM_PREDICTORS = EVI_IM_PREDICTORS
    print("evi_im_predictors.csv not found — using EVI_IM_PREDICTORS from pipeline_config")

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

IM_PREDICTORS = _LOADED_IM_PREDICTORS

# Distance thresholds for sensitivity analysis (km)
# Span: hamlet (0.25 km) to district (10 km)
THRESHOLDS = [0.25, 0.50, 1.00, 2.00, 3.00, 5.00, 10.00]

# Reporting threshold: 1 km selected as village-to-village scale,
# most interpretable for rural Nepal. Result is robust across all thresholds.
FINAL_THRESHOLD_KM = _PIPELINE_CFG["MORANS_THRESHOLD_KM"]

# =============================================================================
# 2. LOAD DATA AND CLEAN COLUMNS
# =============================================================================

df = pd.read_csv(FILE_PATH, encoding='utf-8-sig', low_memory=False)

def clean_col(series):
    """Replace blank-string placeholders with NaN, cast to float."""
    return pd.to_numeric(
        series.replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )

# Dependent variable
df[DEPENDENT] = clean_col(df[DEPENDENT])

# GPS coordinates
df["lat"] = clean_col(df["@_GPS_coordinates_of_the_house_latitude"])
df["lon"] = clean_col(df["@_GPS_coordinates_of_the_house_longitude"])

# Community identifier
df["community"] = df[_PIPELINE_CFG["COMMUNITY_COL"]].replace(" ", np.nan).infer_objects(copy=False)

# Apply data quality caps (mirrors stepwise notebook Section 2)
# AB_Physical_capacity_Pos and OP_Training_Neg had data-entry errors up to 1e11
for _cap_col in CAP_COLS:
    if _cap_col in df.columns:
        _s = pd.to_numeric(df[_cap_col], errors="coerce")
        df[_cap_col] = _s.where(_s <= 100)   # values > 100 become NaN

# Clean all predictor columns
missing_cols = []
for col in IM_PREDICTORS:
    if col in df.columns:
        df[col] = clean_col(df[col])
    else:
        print(f"  WARNING: '{col}' not found in dataset.")
        missing_cols.append(col)

PREDICTORS = [p for p in IM_PREDICTORS if p not in missing_cols]

# =============================================================================
# 3. GPS COVERAGE DIAGNOSTIC
# =============================================================================
# Verify GPS coverage is not concentrated in a subset of VDCs before running
# Moran's I. Uneven coverage could bias the spatial test toward areas with
# denser sampling.

print("=" * 60)
print("GPS COVERAGE DIAGNOSTIC")
print("=" * 60)

n_total = len(df)
n_gps   = df["lat"].notna().sum()
pct_gps = n_gps / n_total * 100
print(f"\nOverall GPS coverage: {n_gps}/{n_total} = {pct_gps:.1f}%")

# Coverage within IM complete cases
im_cols  = [DEPENDENT] + PREDICTORS
complete = df[im_cols].dropna()
comp_gps = df[im_cols + ["lat", "lon"]].dropna()
pct_comp = len(comp_gps) / len(complete) * 100
print(f"IM complete cases:    {len(complete)}")
print(f"  with GPS:           {len(comp_gps)} ({pct_comp:.1f}%)")

# Coverage by VDC
gps_by_vdc = (
    df.groupby("community")
    .apply(lambda g: pd.Series({
        "n_total": len(g),
        "n_gps":   g["lat"].notna().sum(),
        "pct_gps": g["lat"].notna().mean() * 100,
    }))
    .sort_values("n_total", ascending=False)
)

print(f"\nGPS coverage by VDC (n={len(gps_by_vdc)} communities):")
print(f"  {'VDC':<35} {'n':>6}  {'n_GPS':>6}  {'%GPS':>6}")
print("  " + "-" * 58)
for vdc, row in gps_by_vdc.iterrows():
    flag = "  <- LOW" if row["pct_gps"] < 80 else ""
    print(f"  {str(vdc):<35} {int(row['n_total']):>6}  "
          f"{int(row['n_gps']):>6}  {row['pct_gps']:>5.1f}%{flag}")

# Save coverage report
with open("evi_gps_coverage.txt", "w") as f:
    f.write("GPS Coverage Report - Flood Vulnerability Index\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Overall: {n_gps}/{n_total} = {pct_gps:.1f}%\n")
    f.write(f"IM complete cases with GPS: {len(comp_gps)}/{len(complete)} "
            f"= {pct_comp:.1f}%\n\n")
    f.write(f"{'VDC':<35} {'n':>6}  {'n_GPS':>6}  {'%GPS':>6}\n")
    f.write("-" * 58 + "\n")
    for vdc, row in gps_by_vdc.iterrows():
        f.write(f"{str(vdc):<35} {int(row['n_total']):>6}  "
                f"{int(row['n_gps']):>6}  {row['pct_gps']:>5.1f}%\n")

print("\nGPS coverage report saved: evi_gps_coverage.txt")

# =============================================================================
# 4. FIT MODEL AND EXTRACT RESIDUALS
# =============================================================================

def ols_fit_residuals(df, dep, predictors):
    """
    Fit OLS on complete cases. Returns DataFrame with residuals,
    standardised residuals, predicted values, lat, lon, and community.

    Outliers at +/-3 SD removed, consistent with Saputra et al. (2026).
    """
    cols_needed = [dep] + predictors
    data = df[cols_needed + ["lat", "lon", "community"]].dropna(
        subset=cols_needed)

    y = data[dep].values
    X = np.column_stack(
        [np.ones(len(y))] + [data[p].values for p in predictors]
    )

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    resid = y - y_hat

    # Standardised residuals: resid / sqrt(MSE)
    k     = len(predictors)
    mse   = np.sum(resid**2) / (len(y) - k - 1)
    z_res = resid / np.sqrt(mse)

    # Remove outliers at +/-3 SD (consistent with stepwise notebook)
    mask  = np.abs(z_res) <= 3
    data  = data[mask].copy()
    resid = resid[mask]
    z_res = z_res[mask]
    y_hat = y_hat[mask]

    data["resid"]   = resid
    data["z_resid"] = z_res
    data["y_hat"]   = y_hat

    return data


print("\n" + "=" * 60)
print("FITTING INDIVIDUAL MODEL (IM)")
print("=" * 60)

result_df = ols_fit_residuals(df, DEPENDENT, PREDICTORS)
print(f"n after outlier removal: {len(result_df)}")
print(f"Residual mean: {result_df['resid'].mean():.4f}  (should be ~0)")
print(f"Residual SD:   {result_df['resid'].std():.4f}")

# =============================================================================
# 5. MORAN'S I - SENSITIVITY ANALYSIS ACROSS DISTANCE THRESHOLDS
# =============================================================================
# Moran's I measures whether nearby observations have similar residuals.
#   I > 0  =  positive spatial autocorrelation (clustering of similar values)
#   I ~= 0 =  random spatial pattern
#   I < 0  =  negative autocorrelation (dispersed / checkerboard pattern)
#
# A spatial correlogram plots I across distance bands. The threshold where I
# first drops to non-significance marks the practical neighbourhood scale.
# A flat correlogram (I stable across thresholds) suggests community-level
# clustering rather than pure geographic proximity.
#
# References:
#   Moran (1950), Biometrika, 37(1/2), 17-23. https://doi.org/10.2307/2332142
#   Cliff & Ord (1981). Spatial Processes. Pion, London.
#   Legendre & Fortin (1989), Vegetation, 80(2), 107-138.
#   https://doi.org/10.1007/BF00048036

# --- 5a. Build distance matrix once (full GPS sample) ----------------------
# Build once, reuse across all thresholds to avoid redundant computation.

spatial_df = result_df.dropna(subset=["lat", "lon"]).copy()
print(f"\nHouseholds with GPS for Moran's I: {len(spatial_df)}")

r_all   = spatial_df["resid"].values
la_all  = np.radians(spatial_df["lat"].values)
lo_all  = np.radians(spatial_df["lon"].values)
n_all   = len(r_all)
R_earth = 6371.0

print("Building distance matrix (full GPS sample)...", flush=True)
dist_all = np.zeros((n_all, n_all))
for i in range(n_all):
    dlat        = la_all - la_all[i]
    dlon        = lo_all - lo_all[i]
    a           = (np.sin(dlat / 2)**2 +
                   np.cos(la_all[i]) * np.cos(la_all) * np.sin(dlon / 2)**2)
    dist_all[i] = 2 * R_earth * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
print("Done.")


# --- 5b. Moran's I function ------------------------------------------------

def morans_i(r, dist, max_dist_km):
    """
    Compute Moran's I using a pre-built Haversine distance matrix.

    Uses inverse-distance weights (w_ij = 1/d_ij) within max_dist_km.
    Variance estimated under the randomisation assumption (Cliff & Ord 1981).

    Parameters
    ----------
    r           : np.ndarray -- regression residuals (n,)
    dist        : np.ndarray -- symmetric Haversine distance matrix (n x n, km)
    max_dist_km : float      -- neighbourhood radius threshold (km)

    Returns
    -------
    I, E_I, z_score, p_val : floats
    """
    n = len(r)

    # Inverse-distance weights within threshold; diagonal stays zero
    with np.errstate(divide="ignore"):
        W = np.where((dist > 0) & (dist <= max_dist_km), 1.0 / dist, 0.0)

    S0 = W.sum()
    if S0 == 0:
        print(f"  WARNING: No neighbour pairs within {max_dist_km} km.")
        return np.nan, np.nan, np.nan, np.nan

    # Moran's I: I = (n / S0) * (z'Wz / z'z)
    z_r = r - r.mean()
    I   = (n / S0) * (z_r @ W @ z_r) / (z_r @ z_r)

    # Expected value: E[I] = -1/(n-1)
    E_I = -1.0 / (n - 1)

    # Variance under randomisation (Cliff & Ord 1981)
    S1  = 0.5 * np.sum((W + W.T)**2)
    S2  = np.sum((W.sum(axis=1) + W.sum(axis=0))**2)
    m2  = np.sum(z_r**2) / n
    m4  = np.sum(z_r**4) / n
    k   = m4 / (m2**2)

    var_I = (
        (n * ((n**2 - 3*n + 3)*S1 - n*S2 + 3*S0**2)
         - k * ((n**2 - n)*S1 - 2*n*S2 + 6*S0**2))
        / ((n-1) * (n-2) * (n-3) * S0**2)
        - E_I**2
    )

    z_score = (I - E_I) / np.sqrt(max(var_I, 1e-10))
    p_val   = 2 * stats.norm.sf(np.abs(z_score))

    return I, E_I, z_score, p_val


# --- 5c. Sensitivity sweep -------------------------------------------------

print(f"\nMoran's I sensitivity analysis (n={n_all}, full GPS sample):")
print(f"  {'Threshold':>10}  {'I':>8}  {'E[I]':>8}  {'z':>7}  {'p':>8}  sig")
print("  " + "-" * 58)

sensitivity_rows = []
for t in THRESHOLDS:
    I_t, E_I_t, z_t, p_t = morans_i(r_all, dist_all, t)
    sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
    print(f"  {t:>9.2f} km  {I_t:>8.4f}  {E_I_t:>8.4f}  "
          f"{z_t:>7.3f}  {p_t:>8.4f}  {sig}")
    sensitivity_rows.append({
        "threshold_km": t, "I": I_t, "E_I": E_I_t,
        "z": z_t, "p": p_t, "sig": sig,
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df.to_csv("evi_morans_sensitivity.csv", index=False)
print("\nSensitivity results saved: evi_morans_sensitivity.csv")


# --- 5d. Select and report final threshold ---------------------------------

row_final     = sensitivity_df[sensitivity_df["threshold_km"] == FINAL_THRESHOLD_KM].iloc[0]
I_final       = row_final["I"]
E_I_final     = row_final["E_I"]
z_moran_final = row_final["z"]
p_moran_final = row_final["p"]

print(f"\nReporting threshold: {FINAL_THRESHOLD_KM} km "
      f"(village-to-village scale; robust across all thresholds)")
print(f"  Moran's I :  {I_final:.4f}")
print(f"  Expected I:  {E_I_final:.4f}  (= -1/(n-1) under null)")
print(f"  Z-score   :  {z_moran_final:.3f}")
print(f"  p-value   :  {p_moran_final:.4f}")

if p_moran_final < 0.05:
    print("  SIGNIFICANT spatial autocorrelation (p < 0.05).")
    print("  Residuals cluster spatially. Recommendation: mixed-effects model.")
    # Flag borderline results for transparent reporting
    if p_moran_final > 0.04:
        print(f"  NOTE: p={p_moran_final:.4f} is marginal (close to alpha=0.05).")
        # Check how many thresholds were significant
        n_sig = (sensitivity_df["p"] < 0.05).sum()
        n_total_thresh = len(sensitivity_df)
        print(f"  Significant at {n_sig}/{n_total_thresh} distance thresholds tested.")
        if n_sig < n_total_thresh:
            n_ns = n_total_thresh - n_sig
            ns_thresholds = sensitivity_df[sensitivity_df["p"] >= 0.05][
                "threshold_km"].tolist()
            print(f"  Not significant at: {ns_thresholds} km threshold(s).")
            print("  Interpret spatial clustering with caution — "
                  "significance is threshold-dependent.")
else:
    print("  No significant spatial autocorrelation (p >= 0.05).")


# --- 5e. Save text report --------------------------------------------------

with open("evi_morans_i_results.txt", "w") as f:
    f.write("Moran's I - Spatial Autocorrelation of OLS Residuals\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: Individual Model (IM) - Stage 2\n")
    f.write(f"Dependent variable: {DEPENDENT}\n")
    f.write(f"Predictors (n={len(PREDICTORS)}): {', '.join(PREDICTORS)}\n\n")
    f.write(f"GPS sample for Moran's I: n = {n_all}\n")
    f.write(f"GPS coverage in IM complete cases: {pct_comp:.1f}%\n\n")
    f.write("Method: inverse-distance weights (w_ij = 1/d_ij)\n\n")
    f.write("Sensitivity analysis across distance thresholds:\n")
    f.write(f"  {'Threshold':>10}  {'I':>8}  {'E[I]':>8}  {'z':>7}  {'p':>8}  sig\n")
    f.write("  " + "-" * 58 + "\n")
    for _, row_ in sensitivity_df.iterrows():
        f.write(f"  {row_['threshold_km']:>9.2f} km  {row_['I']:>8.4f}  "
                f"{row_['E_I']:>8.4f}  {row_['z']:>7.3f}  "
                f"{row_['p']:>8.4f}  {row_['sig']}\n")
    f.write(f"\nReporting threshold: {FINAL_THRESHOLD_KM} km\n")
    f.write(f"  Moran's I :  {I_final:.4f}\n")
    f.write(f"  Expected I:  {E_I_final:.4f}\n")
    f.write(f"  Z-score   :  {z_moran_final:.3f}\n")
    f.write(f"  p-value   :  {p_moran_final:.4f}\n\n")
    f.write("References:\n")
    f.write("  Moran (1950), Biometrika, 37(1/2), 17-23.\n")
    f.write("  https://doi.org/10.2307/2332142\n")
    f.write("  Cliff & Ord (1981). Spatial Processes. Pion, London.\n")
    f.write("  Legendre & Fortin (1989), Vegetation, 80(2), 107-138.\n")
    f.write("  https://doi.org/10.1007/BF00048036\n")

# =============================================================================
# 6. FIGURE 1 - SPATIAL CORRELOGRAM
# =============================================================================
# A flat correlogram (I stable across all thresholds) indicates community-level
# clustering rather than pure geographic proximity. This supports a
# mixed-effects model with VDC/ward random intercepts over spatial regression.

fig0, ax0 = plt.subplots(figsize=(8, 4))

xs    = sensitivity_df["threshold_km"].values
Is    = sensitivity_df["I"].values
E_I_v = sensitivity_df["E_I"].iloc[0]
zs    = sensitivity_df["z"].values

# 95% CI: SE = (I - E[I]) / z
se_v  = np.where(zs != 0, np.abs(Is - E_I_v) / np.abs(zs), np.nan)
ci_lo = Is - 1.96 * se_v
ci_hi = Is + 1.96 * se_v

ax0.fill_between(xs, ci_lo, ci_hi, alpha=0.15, color="#2196A6", label="95% CI")
ax0.plot(xs, Is, "o-", color="#2196A6", linewidth=2, markersize=7,
         label="Moran's I")
ax0.axhline(E_I_v, color="#888", linewidth=1, linestyle="--",
            label=f"E[I] = {E_I_v:.4f} (null)")
ax0.axhline(0, color="#ccc", linewidth=0.8)
ax0.axvline(FINAL_THRESHOLD_KM, color="#8E44AD", linewidth=1.5,
            linestyle="--", alpha=0.8,
            label=f"Reporting threshold ({FINAL_THRESHOLD_KM} km)")

# Annotate significance stars
for _, row_ in sensitivity_df.iterrows():
    if row_["sig"] != "ns":
        ax0.annotate(
            row_["sig"],
            xy=(row_["threshold_km"], row_["I"]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=8, color="#2196A6",
        )

ax0.set_xlabel("Distance threshold (km)", fontsize=10)
ax0.set_ylabel("Moran's I", fontsize=10)
ax0.set_title(
    "Spatial Correlogram — Moran's I at Multiple Distance Thresholds (EVI)\n"
    "Individual Model Residuals, Lumbini Province, Nepal",
    fontsize=10, fontweight="bold", pad=8,
)
ax0.legend(fontsize=8, framealpha=0.8)
ax0.spines[["top", "right"]].set_visible(False)
ax0.set_xscale("log")
ax0.set_xticks(THRESHOLDS)
ax0.set_xticklabels([str(t) for t in THRESHOLDS], fontsize=8)

plt.tight_layout()
plt.savefig("evi_morans_correlogram.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nFigure 1 saved: evi_morans_correlogram.png")

# =============================================================================
# 7. FIGURE 2 - RESIDUAL MAP BY GPS COORDINATES
# =============================================================================
# Colour encodes residual sign and magnitude.
#   Red  = model over-predicts FVI (actual vulnerability lower than predicted)
#   Blue = model under-predicts FVI (actual vulnerability higher than predicted)
#
# Community centroids are labelled with name and n.
# Moran's I (1 km threshold) is reported in the title for context.

fig1, ax1 = plt.subplots(figsize=(13, 8))

# Symmetric diverging scale centred on zero.
# 95th percentile of absolute residuals sets the colour range so extreme
# outliers do not compress the scale for the majority of households.
vmax = np.percentile(np.abs(spatial_df["resid"].values), 95)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

sc = ax1.scatter(
    spatial_df["lon"],
    spatial_df["lat"],
    c=spatial_df["resid"],
    cmap="RdBu_r",
    norm=norm,
    s=14,
    alpha=0.65,
    linewidths=0,
    zorder=3,
)

cbar = plt.colorbar(sc, ax=ax1, shrink=0.65, pad=0.02, aspect=25)
cbar.set_label(
    "OLS Residual (FVI 1-5 scale)\n"
    "Red = over-predicted  |  Blue = under-predicted",
    fontsize=9,
)
cbar.ax.tick_params(labelsize=8)

# Community centroid labels with n shown
for comm, grp in spatial_df.groupby("community"):
    if pd.isna(comm) or str(comm).strip() == "":
        continue
    cx    = grp["lon"].mean()
    cy    = grp["lat"].mean()
    n_c   = len(grp)
    label = str(comm).replace("_", " ").title()
    ax1.text(
        cx, cy,
        f"{label}\n(n={n_c})",
        fontsize=6.5,
        ha="center",
        va="bottom",
        color="#111111",
        alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55),
        zorder=4,
    )

ax1.set_xlabel("Longitude", fontsize=10)
ax1.set_ylabel("Latitude",  fontsize=10)
ax1.set_title(
    "Spatial Distribution of OLS Residuals — EVI Individual Model (IM)\n"
    f"Lumbini Province, Nepal  |  "
    f"Moran's I = {I_final:.3f},  p = {p_moran_final:.4f}  "
    f"({'significant clustering' if p_moran_final < 0.05 else 'no significant clustering'})",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
ax1.tick_params(labelsize=8)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_facecolor("#eeeeee")  # fallback: shown only if basemap tiles fail
ax1.grid(True, color="white", linewidth=0.5, alpha=0.6, zorder=1)

plt.tight_layout()
plt.savefig("evi_residual_map.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nFigure 2 saved: evi_residual_map.png")

# =============================================================================
# 8. FIGURE 3 - RESIDUALS BY COMMUNITY (annotated boxplot + median dot plot)
# =============================================================================
# Upper panel: boxplot sorted by median residual.
#   Colour encodes direction of systematic bias:
#     Red  = community over-predicted (model overestimates vulnerability)
#     Blue = community under-predicted (model underestimates vulnerability)
#     Grey = centred on zero
#   Each box is annotated with n.
#
# Lower panel: Cleveland dot plot of median residuals.
#   Easier to read exact values than from the boxplot alone.

import matplotlib.patches as mpatches

MIN_COMM_N = 10   # minimum observations per community for inclusion

# Restrict to communities with sufficient observations
comm_counts  = result_df.groupby("community")["resid"].count()
keep_comms   = comm_counts[comm_counts >= MIN_COMM_N].index
plot_df      = result_df[result_df["community"].isin(keep_comms)].copy()
n_excluded   = result_df["community"].nunique() - len(keep_comms)

# Compute per-community stats and sort by median
comm_stats = plot_df.groupby("community")["resid"].agg(
    median="median",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
    n="count",
).sort_values("median")
ordered = comm_stats.index.tolist()

def box_colour(med, threshold=0.03):
    """Return fill colour based on median residual direction."""
    if med > threshold:
        return "#F4A58A"    # red red — over-predicted
    elif med < -threshold:
        return "#92C5DE"    # blue blue — under-predicted
    else:
        return "#E8E8E8"    # grey — centred

fig2, (ax2, ax3) = plt.subplots(
    2, 1,
    figsize=(14, 9),
    gridspec_kw={"height_ratios": [3, 1]},
)

# --- Upper panel: boxplot ---------------------------------------------------

box_data = [
    plot_df.loc[plot_df["community"] == c, "resid"].values
    for c in ordered
]

bp = ax2.boxplot(
    box_data,
    positions=range(len(ordered)),
    widths=0.6,
    patch_artist=True,
    medianprops=dict(color="#333333", linewidth=2.0),
    flierprops=dict(
        marker="o", markersize=3, alpha=0.35,
        markerfacecolor="#999999", markeredgewidth=0,
    ),
    whiskerprops=dict(linewidth=1.0, color="#555555"),
    capprops=dict(linewidth=1.0, color="#555555"),
    boxprops=dict(linewidth=0.8),
)

for patch, comm in zip(bp["boxes"], ordered):
    patch.set_facecolor(box_colour(comm_stats.loc[comm, "median"]))

ax2.axhline(0, color="#444444", linewidth=1.2, linestyle="--", zorder=1, alpha=0.7)

# Annotate each box with n
for i, comm in enumerate(ordered):
    med = comm_stats.loc[comm, "median"]
    n_c = int(comm_stats.loc[comm, "n"])
    ax2.text(
        i, med + 0.03,
        f"n={n_c}",
        ha="center", va="bottom",
        fontsize=6.5, color="#333333",
    )

clean_labels = [
    str(c).replace("_", " ").replace("ward", "Ward").title()
    for c in ordered
]
ax2.set_xticks(range(len(ordered)))
ax2.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=8)
ax2.set_ylabel("OLS Residual (EVI scale 1–5)", fontsize=10)
ax2.set_title(
    "EVI — OLS Residuals by Community (VDC/Ward), Individual Model (IM)\n"
    f"n = {len(plot_df)} households across {len(ordered)} communities  "
    f"({n_excluded} communities excluded: n < {MIN_COMM_N})",
    fontsize=11, fontweight="bold", pad=10,
)
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
ax2.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(facecolor="#F4A58A", edgecolor="#888",
                   label="Over-predicted (median > 0)"),
    mpatches.Patch(facecolor="#92C5DE", edgecolor="#888",
                   label="Under-predicted (median < 0)"),
    mpatches.Patch(facecolor="#E8E8E8", edgecolor="#888",
                   label="Centred (median ~= 0)"),
]
ax2.legend(handles=legend_patches, fontsize=8, loc="upper left",
           framealpha=0.85, edgecolor="#cccccc")

# --- Lower panel: median dot plot ------------------------------------------

medians = [comm_stats.loc[c, "median"] for c in ordered]
colours = [box_colour(m) for m in medians]

ax3.scatter(
    range(len(ordered)), medians,
    c=colours, s=60, zorder=3,
    edgecolors="#555555", linewidths=0.6,
)
ax3.axhline(0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)

for i, (med, comm) in enumerate(zip(medians, ordered)):
    ax3.text(
        i, med + (0.015 if med >= 0 else -0.025),
        f"{med:+.2f}",
        ha="center",
        va="bottom" if med >= 0 else "top",
        fontsize=6.5, color="#333333",
    )

ax3.set_xticks(range(len(ordered)))
ax3.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=8)
ax3.set_ylabel("Median\nresidual", fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)
ax3.yaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
ax3.set_axisbelow(True)

plt.tight_layout(h_pad=0.5)
plt.savefig("evi_residual_by_community.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 3 saved: evi_residual_by_community.png")
print("\nDiagnosis complete.")



"""
Autocorrelation Diagnosis: Spatial and Community-level Clustering
=================================================================
Tests whether residuals from the EVI stepwise regression are
autocorrelated due to spatial proximity or community clustering.

Model used: Individual Model (IM) — Stage 2 of 4_2__fvi_stepwise_updated.ipynb
Rationale:  The IM is the primary reported model. The Composite Model is a
            robustness check. Moran's I is run on the IM because:
              (1) IM variables (AB_Location_Pos, DE12_shelter_score,
                  DE13_feel_safe_score) are spatially-grounded — residual
                  clustering after controlling for these is the meaningful test.
              (2) The IM is the model whose independence assumption is reported.
            See: Saputra, Schwarz & Hendriks (2026), IJDRR 133, 105913.
                 https://doi.org/10.1016/j.ijdrr.2025.105913

NOTE on sample size: IM n=1,222 (down from 2,363 in the prior run).
  The drop is driven by two variables with high missingness:
    APP_Perc_Pos_expression_CAT_33PROCENT : 52% missing
    DE12_shelter_score                    : 17% missing
  Listwise deletion across all 10 predictors compounds this.
  GPS coverage within the IM complete cases is 97.6%, so the spatial
  sample is not geographically biased despite the smaller n.

Four diagnostic outputs:
  1. GPS coverage report by VDC (console + text file)
  2. Spatial correlogram — Moran's I across 7 distance thresholds
  3. Residual map by GPS coordinates
  4. Boxplot of residuals by community

References:
  Moran, P.A.P. (1950). Notes on continuous stochastic phenomena.
  Biometrika, 37(1/2), 17-23. https://doi.org/10.2307/2332142

  Cliff, A.D. & Ord, J.K. (1981). Spatial Processes: Models &
  Applications. Pion, London.

  Legendre, P. & Fortin, M.J. (1989). Spatial pattern and ecological
  analysis. Vegetation, 80(2), 107-138.
  https://doi.org/10.1007/BF00048036

Outputs:
  evi_gps_coverage.txt          - GPS coverage by VDC
  evi_morans_correlogram.png    - Moran's I across distance thresholds
  evi_morans_sensitivity.csv    - Moran's I table (all thresholds)
  evi_morans_i_results.txt      - Moran's I final report
  evi_residual_map.png          - GPS map of residuals
  evi_residual_by_community.png - boxplot by VDC/ward
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

# OSM basemap — optional; pipeline never crashes if unavailable
try:
    from osm_tiles import add_basemap as _osm_add_basemap
    _OSM_AVAILABLE = True
except ImportError:
    _OSM_AVAILABLE = False
    def _osm_add_basemap(ax, **kw):  # noqa: F811
        """No-op when osm_tiles.py is not found."""
        return False


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

FILE_PATH = str(Path.cwd() / "imputed_EVI_scores.csv")   # scored dataset: contains EVI_norm_1_5,
                                       # all predictors, GPS, and community
DEPENDENT = "EVI_norm_1_5"

# Individual Model predictors from Stage 2 of 4_2__fvi_stepwise_updated.ipynb
# (updated 32-variable run; final n=1,222 after listwise deletion)
# Key missingness drivers:
#   APP_Perc_Pos_expression_CAT_33PROCENT : 52% missing
#   DE12_shelter_score                    : 17% missing
IM_PREDICTORS = _LOADED_IM_PREDICTORS

# Distance thresholds for sensitivity analysis (km)
# Span: hamlet (0.25 km) to district (10 km)
THRESHOLDS = [0.25, 0.50, 1.00, 2.00, 3.00, 5.00, 10.00]

# Reporting threshold: 1 km selected as village-to-village scale,
# most interpretable for rural Nepal. Result is robust across all thresholds.
FINAL_THRESHOLD_KM = _PIPELINE_CFG["MORANS_THRESHOLD_KM"]

# =============================================================================
# 2. LOAD DATA AND CLEAN COLUMNS
# =============================================================================

df = pd.read_csv(FILE_PATH, encoding='utf-8-sig', low_memory=False)

def clean_col(series):
    """Replace blank-string placeholders with NaN, cast to float."""
    return pd.to_numeric(
        series.replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )

# Dependent variable
df[DEPENDENT] = clean_col(df[DEPENDENT])

# GPS coordinates
df["lat"] = clean_col(df["@_GPS_coordinates_of_the_house_latitude"])
df["lon"] = clean_col(df["@_GPS_coordinates_of_the_house_longitude"])

# Community identifier
df["community"] = df[_PIPELINE_CFG["COMMUNITY_COL"]].replace(" ", np.nan).infer_objects(copy=False)

# Apply data quality caps (mirrors stepwise notebook Section 2)
# AB_Physical_capacity_Pos and OP_Training_Neg had data-entry errors up to 1e11
for _cap_col in CAP_COLS:
    if _cap_col in df.columns:
        _s = pd.to_numeric(df[_cap_col], errors="coerce")
        df[_cap_col] = _s.where(_s <= 100)   # values > 100 become NaN

# Clean all predictor columns
missing_cols = []
for col in IM_PREDICTORS:
    if col in df.columns:
        df[col] = clean_col(df[col])
    else:
        print(f"  WARNING: '{col}' not found in dataset.")
        missing_cols.append(col)

PREDICTORS = [p for p in IM_PREDICTORS if p not in missing_cols]

# =============================================================================
# 3. GPS COVERAGE DIAGNOSTIC
# =============================================================================
# Verify GPS coverage is not concentrated in a subset of VDCs before running
# Moran's I. Uneven coverage could bias the spatial test toward areas with
# denser sampling.

print("=" * 60)
print("GPS COVERAGE DIAGNOSTIC")
print("=" * 60)

n_total = len(df)
n_gps   = df["lat"].notna().sum()
pct_gps = n_gps / n_total * 100
print(f"\nOverall GPS coverage: {n_gps}/{n_total} = {pct_gps:.1f}%")

# Coverage within IM complete cases
im_cols  = [DEPENDENT] + PREDICTORS
complete = df[im_cols].dropna()
comp_gps = df[im_cols + ["lat", "lon"]].dropna()
pct_comp = len(comp_gps) / len(complete) * 100
print(f"IM complete cases:    {len(complete)}")
print(f"  with GPS:           {len(comp_gps)} ({pct_comp:.1f}%)")

# Coverage by VDC
gps_by_vdc = (
    df.groupby("community")
    .apply(lambda g: pd.Series({
        "n_total": len(g),
        "n_gps":   g["lat"].notna().sum(),
        "pct_gps": g["lat"].notna().mean() * 100,
    }))
    .sort_values("n_total", ascending=False)
)

print(f"\nGPS coverage by VDC (n={len(gps_by_vdc)} communities):")
print(f"  {'VDC':<35} {'n':>6}  {'n_GPS':>6}  {'%GPS':>6}")
print("  " + "-" * 58)
for vdc, row in gps_by_vdc.iterrows():
    flag = "  <- LOW" if row["pct_gps"] < 80 else ""
    print(f"  {str(vdc):<35} {int(row['n_total']):>6}  "
          f"{int(row['n_gps']):>6}  {row['pct_gps']:>5.1f}%{flag}")

# Save coverage report
with open("evi_gps_coverage.txt", "w") as f:
    f.write("GPS Coverage Report - Flood Vulnerability Index\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Overall: {n_gps}/{n_total} = {pct_gps:.1f}%\n")
    f.write(f"IM complete cases with GPS: {len(comp_gps)}/{len(complete)} "
            f"= {pct_comp:.1f}%\n\n")
    f.write(f"{'VDC':<35} {'n':>6}  {'n_GPS':>6}  {'%GPS':>6}\n")
    f.write("-" * 58 + "\n")
    for vdc, row in gps_by_vdc.iterrows():
        f.write(f"{str(vdc):<35} {int(row['n_total']):>6}  "
                f"{int(row['n_gps']):>6}  {row['pct_gps']:>5.1f}%\n")

print("\nGPS coverage report saved: evi_gps_coverage.txt")

# =============================================================================
# 4. FIT MODEL AND EXTRACT RESIDUALS
# =============================================================================

def ols_fit_residuals(df, dep, predictors):
    """
    Fit OLS on complete cases. Returns DataFrame with residuals,
    standardised residuals, predicted values, lat, lon, and community.

    Outliers at +/-3 SD removed, consistent with Saputra et al. (2026).
    """
    cols_needed = [dep] + predictors
    data = df[cols_needed + ["lat", "lon", "community"]].dropna(
        subset=cols_needed)

    y = data[dep].values
    X = np.column_stack(
        [np.ones(len(y))] + [data[p].values for p in predictors]
    )

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    resid = y - y_hat

    # Standardised residuals: resid / sqrt(MSE)
    k     = len(predictors)
    mse   = np.sum(resid**2) / (len(y) - k - 1)
    z_res = resid / np.sqrt(mse)

    # Remove outliers at +/-3 SD (consistent with stepwise notebook)
    mask  = np.abs(z_res) <= 3
    data  = data[mask].copy()
    resid = resid[mask]
    z_res = z_res[mask]
    y_hat = y_hat[mask]

    data["resid"]   = resid
    data["z_resid"] = z_res
    data["y_hat"]   = y_hat

    return data


print("\n" + "=" * 60)
print("FITTING INDIVIDUAL MODEL (IM)")
print("=" * 60)

result_df = ols_fit_residuals(df, DEPENDENT, PREDICTORS)
print(f"n after outlier removal: {len(result_df)}")
print(f"Residual mean: {result_df['resid'].mean():.4f}  (should be ~0)")
print(f"Residual SD:   {result_df['resid'].std():.4f}")

# =============================================================================
# 5. MORAN'S I - SENSITIVITY ANALYSIS ACROSS DISTANCE THRESHOLDS
# =============================================================================
# Moran's I measures whether nearby observations have similar residuals.
#   I > 0  =  positive spatial autocorrelation (clustering of similar values)
#   I ~= 0 =  random spatial pattern
#   I < 0  =  negative autocorrelation (dispersed / checkerboard pattern)
#
# A spatial correlogram plots I across distance bands. The threshold where I
# first drops to non-significance marks the practical neighbourhood scale.
# A flat correlogram (I stable across thresholds) suggests community-level
# clustering rather than pure geographic proximity.
#
# References:
#   Moran (1950), Biometrika, 37(1/2), 17-23. https://doi.org/10.2307/2332142
#   Cliff & Ord (1981). Spatial Processes. Pion, London.
#   Legendre & Fortin (1989), Vegetation, 80(2), 107-138.
#   https://doi.org/10.1007/BF00048036

# --- 5a. Build distance matrix once (full GPS sample) ----------------------
# Build once, reuse across all thresholds to avoid redundant computation.

spatial_df = result_df.dropna(subset=["lat", "lon"]).copy()
print(f"\nHouseholds with GPS for Moran's I: {len(spatial_df)}")

r_all   = spatial_df["resid"].values
la_all  = np.radians(spatial_df["lat"].values)
lo_all  = np.radians(spatial_df["lon"].values)
n_all   = len(r_all)
R_earth = 6371.0

print("Building distance matrix (full GPS sample)...", flush=True)
dist_all = np.zeros((n_all, n_all))
for i in range(n_all):
    dlat        = la_all - la_all[i]
    dlon        = lo_all - lo_all[i]
    a           = (np.sin(dlat / 2)**2 +
                   np.cos(la_all[i]) * np.cos(la_all) * np.sin(dlon / 2)**2)
    dist_all[i] = 2 * R_earth * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
print("Done.")


# --- 5b. Moran's I function ------------------------------------------------

def morans_i(r, dist, max_dist_km):
    """
    Compute Moran's I using a pre-built Haversine distance matrix.

    Uses inverse-distance weights (w_ij = 1/d_ij) within max_dist_km.
    Variance estimated under the randomisation assumption (Cliff & Ord 1981).

    Parameters
    ----------
    r           : np.ndarray -- regression residuals (n,)
    dist        : np.ndarray -- symmetric Haversine distance matrix (n x n, km)
    max_dist_km : float      -- neighbourhood radius threshold (km)

    Returns
    -------
    I, E_I, z_score, p_val : floats
    """
    n = len(r)

    # Inverse-distance weights within threshold; diagonal stays zero
    with np.errstate(divide="ignore"):
        W = np.where((dist > 0) & (dist <= max_dist_km), 1.0 / dist, 0.0)

    S0 = W.sum()
    if S0 == 0:
        print(f"  WARNING: No neighbour pairs within {max_dist_km} km.")
        return np.nan, np.nan, np.nan, np.nan

    # Moran's I: I = (n / S0) * (z'Wz / z'z)
    z_r = r - r.mean()
    I   = (n / S0) * (z_r @ W @ z_r) / (z_r @ z_r)

    # Expected value: E[I] = -1/(n-1)
    E_I = -1.0 / (n - 1)

    # Variance under randomisation (Cliff & Ord 1981)
    S1  = 0.5 * np.sum((W + W.T)**2)
    S2  = np.sum((W.sum(axis=1) + W.sum(axis=0))**2)
    m2  = np.sum(z_r**2) / n
    m4  = np.sum(z_r**4) / n
    k   = m4 / (m2**2)

    var_I = (
        (n * ((n**2 - 3*n + 3)*S1 - n*S2 + 3*S0**2)
         - k * ((n**2 - n)*S1 - 2*n*S2 + 6*S0**2))
        / ((n-1) * (n-2) * (n-3) * S0**2)
        - E_I**2
    )

    z_score = (I - E_I) / np.sqrt(max(var_I, 1e-10))
    p_val   = 2 * stats.norm.sf(np.abs(z_score))

    return I, E_I, z_score, p_val


# --- 5c. Sensitivity sweep -------------------------------------------------

print(f"\nMoran's I sensitivity analysis (n={n_all}, full GPS sample):")
print(f"  {'Threshold':>10}  {'I':>8}  {'E[I]':>8}  {'z':>7}  {'p':>8}  sig")
print("  " + "-" * 58)

sensitivity_rows = []
for t in THRESHOLDS:
    I_t, E_I_t, z_t, p_t = morans_i(r_all, dist_all, t)
    sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
    print(f"  {t:>9.2f} km  {I_t:>8.4f}  {E_I_t:>8.4f}  "
          f"{z_t:>7.3f}  {p_t:>8.4f}  {sig}")
    sensitivity_rows.append({
        "threshold_km": t, "I": I_t, "E_I": E_I_t,
        "z": z_t, "p": p_t, "sig": sig,
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df.to_csv("evi_morans_sensitivity.csv", index=False)
print("\nSensitivity results saved: evi_morans_sensitivity.csv")


# --- 5d. Select and report final threshold ---------------------------------

row_final     = sensitivity_df[sensitivity_df["threshold_km"] == FINAL_THRESHOLD_KM].iloc[0]
I_final       = row_final["I"]
E_I_final     = row_final["E_I"]
z_moran_final = row_final["z"]
p_moran_final = row_final["p"]

print(f"\nReporting threshold: {FINAL_THRESHOLD_KM} km "
      f"(village-to-village scale; robust across all thresholds)")
print(f"  Moran's I :  {I_final:.4f}")
print(f"  Expected I:  {E_I_final:.4f}  (= -1/(n-1) under null)")
print(f"  Z-score   :  {z_moran_final:.3f}")
print(f"  p-value   :  {p_moran_final:.4f}")

if p_moran_final < 0.05:
    print("  SIGNIFICANT spatial autocorrelation (p < 0.05).")
    print("  Residuals cluster spatially. Recommendation: mixed-effects model.")
    # Flag borderline results for transparent reporting
    if p_moran_final > 0.04:
        print(f"  NOTE: p={p_moran_final:.4f} is marginal (close to alpha=0.05).")
        # Check how many thresholds were significant
        n_sig = (sensitivity_df["p"] < 0.05).sum()
        n_total_thresh = len(sensitivity_df)
        print(f"  Significant at {n_sig}/{n_total_thresh} distance thresholds tested.")
        if n_sig < n_total_thresh:
            n_ns = n_total_thresh - n_sig
            ns_thresholds = sensitivity_df[sensitivity_df["p"] >= 0.05][
                "threshold_km"].tolist()
            print(f"  Not significant at: {ns_thresholds} km threshold(s).")
            print("  Interpret spatial clustering with caution — "
                  "significance is threshold-dependent.")
else:
    print("  No significant spatial autocorrelation (p >= 0.05).")


# --- 5e. Save text report --------------------------------------------------

with open("evi_morans_i_results.txt", "w") as f:
    f.write("Moran's I - Spatial Autocorrelation of OLS Residuals\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: Individual Model (IM) - Stage 2\n")
    f.write(f"Dependent variable: {DEPENDENT}\n")
    f.write(f"Predictors (n={len(PREDICTORS)}): {', '.join(PREDICTORS)}\n\n")
    f.write(f"GPS sample for Moran's I: n = {n_all}\n")
    f.write(f"GPS coverage in IM complete cases: {pct_comp:.1f}%\n\n")
    f.write("Method: inverse-distance weights (w_ij = 1/d_ij)\n\n")
    f.write("Sensitivity analysis across distance thresholds:\n")
    f.write(f"  {'Threshold':>10}  {'I':>8}  {'E[I]':>8}  {'z':>7}  {'p':>8}  sig\n")
    f.write("  " + "-" * 58 + "\n")
    for _, row_ in sensitivity_df.iterrows():
        f.write(f"  {row_['threshold_km']:>9.2f} km  {row_['I']:>8.4f}  "
                f"{row_['E_I']:>8.4f}  {row_['z']:>7.3f}  "
                f"{row_['p']:>8.4f}  {row_['sig']}\n")
    f.write(f"\nReporting threshold: {FINAL_THRESHOLD_KM} km\n")
    f.write(f"  Moran's I :  {I_final:.4f}\n")
    f.write(f"  Expected I:  {E_I_final:.4f}\n")
    f.write(f"  Z-score   :  {z_moran_final:.3f}\n")
    f.write(f"  p-value   :  {p_moran_final:.4f}\n\n")
    f.write("References:\n")
    f.write("  Moran (1950), Biometrika, 37(1/2), 17-23.\n")
    f.write("  https://doi.org/10.2307/2332142\n")
    f.write("  Cliff & Ord (1981). Spatial Processes. Pion, London.\n")
    f.write("  Legendre & Fortin (1989), Vegetation, 80(2), 107-138.\n")
    f.write("  https://doi.org/10.1007/BF00048036\n")

# =============================================================================
# 6. FIGURE 1 - SPATIAL CORRELOGRAM
# =============================================================================
# A flat correlogram (I stable across all thresholds) indicates community-level
# clustering rather than pure geographic proximity. This supports a
# mixed-effects model with VDC/ward random intercepts over spatial regression.

fig0, ax0 = plt.subplots(figsize=(8, 4))

xs    = sensitivity_df["threshold_km"].values
Is    = sensitivity_df["I"].values
E_I_v = sensitivity_df["E_I"].iloc[0]
zs    = sensitivity_df["z"].values

# 95% CI: SE = (I - E[I]) / z
se_v  = np.where(zs != 0, np.abs(Is - E_I_v) / np.abs(zs), np.nan)
ci_lo = Is - 1.96 * se_v
ci_hi = Is + 1.96 * se_v

ax0.fill_between(xs, ci_lo, ci_hi, alpha=0.15, color="#2196A6", label="95% CI")
ax0.plot(xs, Is, "o-", color="#2196A6", linewidth=2, markersize=7,
         label="Moran's I")
ax0.axhline(E_I_v, color="#888", linewidth=1, linestyle="--",
            label=f"E[I] = {E_I_v:.4f} (null)")
ax0.axhline(0, color="#ccc", linewidth=0.8)
ax0.axvline(FINAL_THRESHOLD_KM, color="#8E44AD", linewidth=1.5,
            linestyle="--", alpha=0.8,
            label=f"Reporting threshold ({FINAL_THRESHOLD_KM} km)")

# Annotate significance stars
for _, row_ in sensitivity_df.iterrows():
    if row_["sig"] != "ns":
        ax0.annotate(
            row_["sig"],
            xy=(row_["threshold_km"], row_["I"]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=8, color="#2196A6",
        )

ax0.set_xlabel("Distance threshold (km)", fontsize=10)
ax0.set_ylabel("Moran's I", fontsize=10)
ax0.set_title(
    "Spatial Correlogram — Moran's I at Multiple Distance Thresholds (EVI)\n"
    "Individual Model Residuals, Lumbini Province, Nepal",
    fontsize=10, fontweight="bold", pad=8,
)
ax0.legend(fontsize=8, framealpha=0.8)
ax0.spines[["top", "right"]].set_visible(False)
ax0.set_xscale("log")
ax0.set_xticks(THRESHOLDS)
ax0.set_xticklabels([str(t) for t in THRESHOLDS], fontsize=8)

plt.tight_layout()
plt.savefig("evi_morans_correlogram.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nFigure 1 saved: evi_morans_correlogram.png")

# =============================================================================
# 7. FIGURE 2 - RESIDUAL MAP BY GPS COORDINATES
# =============================================================================
# Colour encodes residual sign and magnitude.
#   Red  = model over-predicts FVI (actual vulnerability lower than predicted)
#   Blue = model under-predicts FVI (actual vulnerability higher than predicted)
#
# Community centroids are labelled with name and n.
# Moran's I (1 km threshold) is reported in the title for context.

fig1, ax1 = plt.subplots(figsize=(13, 8))

# Set axes extent BEFORE fetching OSM tiles.
_lon_pad_f2 = (spatial_df["lon"].max() - spatial_df["lon"].min()) * 0.04
_lat_pad_f2 = (spatial_df["lat"].max() - spatial_df["lat"].min()) * 0.04
ax1.set_xlim(spatial_df["lon"].min() - _lon_pad_f2,
             spatial_df["lon"].max() + _lon_pad_f2)
ax1.set_ylim(spatial_df["lat"].min() - _lat_pad_f2,
             spatial_df["lat"].max() + _lat_pad_f2)

# CartoDB Positron: minimal grey — rivers not relevant for seismic maps.
# Rendered at zorder=0 so all scatter artists appear on top.
_osm_add_basemap(
    ax1,
    zoom="auto",
    provider="carto_voyager",
    alpha=0.50,
    cache_dir=Path.cwd() / "osm_tile_cache",
)

# Symmetric diverging scale centred on zero.
vmax = np.percentile(np.abs(spatial_df["resid"].values), 95)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

sc = ax1.scatter(
    spatial_df["lon"],
    spatial_df["lat"],
    c=spatial_df["resid"],
    cmap="RdBu_r",
    norm=norm,
    s=14,
    alpha=0.65,
    linewidths=0,
    zorder=3,
)

cbar = plt.colorbar(sc, ax=ax1, shrink=0.65, pad=0.02, aspect=25)
cbar.set_label(
    "OLS Residual (FVI 1-5 scale)\n"
    "Red = over-predicted  |  Blue = under-predicted",
    fontsize=9,
)
cbar.ax.tick_params(labelsize=8)

# Community centroid labels with n shown
for comm, grp in spatial_df.groupby("community"):
    if pd.isna(comm) or str(comm).strip() == "":
        continue
    cx    = grp["lon"].mean()
    cy    = grp["lat"].mean()
    n_c   = len(grp)
    label = str(comm).replace("_", " ").title()
    ax1.text(
        cx, cy,
        f"{label}\n(n={n_c})",
        fontsize=6.5,
        ha="center",
        va="bottom",
        color="#111111",
        alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55),
        zorder=4,
    )

ax1.set_xlabel("Longitude", fontsize=10)
ax1.set_ylabel("Latitude",  fontsize=10)
ax1.set_title(
    "Spatial Distribution of OLS Residuals — EVI Individual Model (IM)\n"
    f"Lumbini Province, Nepal  |  "
    f"Moran's I = {I_final:.3f},  p = {p_moran_final:.4f}  "
    f"({'significant clustering' if p_moran_final < 0.05 else 'no significant clustering'})",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
ax1.tick_params(labelsize=8)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_facecolor("#eeeeee")  # fallback: shown only if basemap tiles fail
ax1.grid(True, color="white", linewidth=0.5, alpha=0.6, zorder=1)

plt.tight_layout()
plt.savefig("evi_residual_map.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nFigure 2 saved: evi_residual_map.png")

# =============================================================================
# 8. FIGURE 3 - RESIDUALS BY COMMUNITY (annotated boxplot + median dot plot)
# =============================================================================
# Upper panel: boxplot sorted by median residual.
#   Colour encodes direction of systematic bias:
#     Red  = community over-predicted (model overestimates vulnerability)
#     Blue = community under-predicted (model underestimates vulnerability)
#     Grey = centred on zero
#   Each box is annotated with n.
#
# Lower panel: Cleveland dot plot of median residuals.
#   Easier to read exact values than from the boxplot alone.

import matplotlib.patches as mpatches




MIN_COMM_N = 10   # minimum observations per community for inclusion

# Restrict to communities with sufficient observations
comm_counts  = result_df.groupby("community")["resid"].count()
keep_comms   = comm_counts[comm_counts >= MIN_COMM_N].index
plot_df      = result_df[result_df["community"].isin(keep_comms)].copy()
n_excluded   = result_df["community"].nunique() - len(keep_comms)

# Compute per-community stats and sort by median
comm_stats = plot_df.groupby("community")["resid"].agg(
    median="median",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
    n="count",
).sort_values("median")
ordered = comm_stats.index.tolist()

def box_colour(med, threshold=0.03):
    """Return fill colour based on median residual direction."""
    if med > threshold:
        return "#F4A58A"    # muted red — over-predicted
    elif med < -threshold:
        return "#92C5DE"    # muted blue — under-predicted
    else:
        return "#E8E8E8"    # grey — centred

fig2, (ax2, ax3) = plt.subplots(
    2, 1,
    figsize=(14, 9),
    gridspec_kw={"height_ratios": [3, 1]},
)

# --- Upper panel: boxplot ---------------------------------------------------

box_data = [
    plot_df.loc[plot_df["community"] == c, "resid"].values
    for c in ordered
]

bp = ax2.boxplot(
    box_data,
    positions=range(len(ordered)),
    widths=0.6,
    patch_artist=True,
    medianprops=dict(color="#333333", linewidth=2.0),
    flierprops=dict(
        marker="o", markersize=3, alpha=0.35,
        markerfacecolor="#999999", markeredgewidth=0,
    ),
    whiskerprops=dict(linewidth=1.0, color="#555555"),
    capprops=dict(linewidth=1.0, color="#555555"),
    boxprops=dict(linewidth=0.8),
)

for patch, comm in zip(bp["boxes"], ordered):
    patch.set_facecolor(box_colour(comm_stats.loc[comm, "median"]))

ax2.axhline(0, color="#444444", linewidth=1.2, linestyle="--", zorder=1, alpha=0.7)

# Annotate each box with n
for i, comm in enumerate(ordered):
    med = comm_stats.loc[comm, "median"]
    n_c = int(comm_stats.loc[comm, "n"])
    ax2.text(
        i, med + 0.03,
        f"n={n_c}",
        ha="center", va="bottom",
        fontsize=6.5, color="#333333",
    )

clean_labels = [
    str(c).replace("_", " ").replace("ward", "Ward").title()
    for c in ordered
]
ax2.set_xticks(range(len(ordered)))
ax2.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=8)
ax2.set_ylabel("OLS Residual (EVI scale 1–5)", fontsize=10)
ax2.set_title(
    "EVI — OLS Residuals by Community (VDC/Ward), Individual Model (IM)\n"
    f"n = {len(plot_df)} households across {len(ordered)} communities  "
    f"({n_excluded} communities excluded: n < {MIN_COMM_N})",
    fontsize=11, fontweight="bold", pad=10,
)
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
ax2.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(facecolor="#F4A58A", edgecolor="#888",
                   label="Over-predicted (median > 0)"),
    mpatches.Patch(facecolor="#92C5DE", edgecolor="#888",
                   label="Under-predicted (median < 0)"),
    mpatches.Patch(facecolor="#E8E8E8", edgecolor="#888",
                   label="Centred (median ~= 0)"),
]
ax2.legend(handles=legend_patches, fontsize=8, loc="upper left",
           framealpha=0.85, edgecolor="#cccccc")

# --- Lower panel: median dot plot ------------------------------------------

medians = [comm_stats.loc[c, "median"] for c in ordered]
colours = [box_colour(m) for m in medians]

ax3.scatter(
    range(len(ordered)), medians,
    c=colours, s=60, zorder=3,
    edgecolors="#555555", linewidths=0.6,
)
ax3.axhline(0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)

for i, (med, comm) in enumerate(zip(medians, ordered)):
    ax3.text(
        i, med + (0.015 if med >= 0 else -0.025),
        f"{med:+.2f}",
        ha="center",
        va="bottom" if med >= 0 else "top",
        fontsize=6.5, color="#333333",
    )

ax3.set_xticks(range(len(ordered)))
ax3.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=8)
ax3.set_ylabel("Median\nresidual", fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)
ax3.yaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
ax3.set_axisbelow(True)

plt.tight_layout(h_pad=0.5)
plt.savefig("evi_residual_by_community.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 3 saved: evi_residual_by_community.png")

# =============================================================================
# 9. LOCAL MORAN'S I (LISA)
# =============================================================================
# Local Indicators of Spatial Association (Anselin 1995) decompose the global
# Moran's I into a per-observation statistic. Each household receives its own
# I_i value and a cluster type classification.
#
# Cluster types (based on standardised residual z_i and spatial lag Wz_i):
#   High-High (HH): z_i > 0, Wz_i > 0 — high residual surrounded by high
#   Low-Low   (LL): z_i < 0, Wz_i < 0 — low residual surrounded by low
#   High-Low  (HL): z_i > 0, Wz_i < 0 — high residual surrounded by low
#   Low-High  (LH): z_i < 0, Wz_i > 0 — low residual surrounded by high
#   Not significant: local I_i not significant at p < 0.05
#
# Statistical significance assessed via permutation test (999 permutations).
# This is the standard approach: under the null, residuals are randomly
# reassigned across locations and the empirical distribution of I_i is built.
#
# Distance threshold: FINAL_THRESHOLD_KM (1 km) — same as global test,
# ensures results are comparable and consistently reported.
#
# References:
#   Anselin, L. (1995). Local indicators of spatial association — LISA.
#   Geographical Analysis, 27(2), 93-115.
#   https://doi.org/10.1111/j.1538-4632.1995.tb00338.x
#
#   Aksha et al. (2019) used local Moran's I to classify HH/LL/HL/LH
#   clusters of social vulnerability across Nepal VDCs. Same approach applied
#   here to EVI regression residuals.
#   https://doi.org/10.1007/s13753-018-0192-7
#
# Outputs:
#   evi_local_morans_results.csv   — per-household I_i, z_i, p_i, cluster type
#   evi_local_morans_map.png       — LISA cluster map
#   evi_local_morans_summary.txt   — cluster frequency table

N_PERMUTATIONS = 999   # standard for permutation significance testing
LISA_THRESHOLD = FINAL_THRESHOLD_KM   # 1 km — consistent with global test
LISA_ALPHA     = 0.05  # significance level for cluster classification

print("\n" + "=" * 60)
print("LOCAL MORAN'S I (LISA)")
print("=" * 60)
print(f"Distance threshold : {LISA_THRESHOLD} km")
print(f"Permutations       : {N_PERMUTATIONS}")
print(f"Significance level : p < {LISA_ALPHA}")

# --- 9a. Build weight matrix for LISA threshold ----------------------------
# Reuse dist_all (already computed for global test).
# Same inverse-distance weights, same threshold.

with np.errstate(divide="ignore"):
    W_lisa = np.where(
        (dist_all > 0) & (dist_all <= LISA_THRESHOLD),
        1.0 / dist_all,
        0.0
    )

# Row-standardise: each row sums to 1 so the spatial lag is a weighted mean.
# This is the standard form for LISA (Anselin 1995).
row_sums = W_lisa.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1   # avoid divide-by-zero for isolated observations
W_row = W_lisa / row_sums

# --- 9b. Compute local I_i for each observation ----------------------------

r_lisa  = spatial_df["resid"].values
z_lisa  = r_lisa - r_lisa.mean()          # mean-centred residuals
m2      = np.mean(z_lisa**2)              # variance (denominator)

# Local I_i = z_i * (Wz)_i / m2
# (Wz)_i is the row-standardised spatial lag of z at location i
Wz      = W_row @ z_lisa                  # spatial lag vector
I_local = (z_lisa * Wz) / m2             # local Moran's I for each observation

# --- 9c. Permutation test for significance ---------------------------------
# Under the null: residuals are spatially random.
# Procedure: shuffle residuals 999 times, recompute I_i each time,
# build the empirical null distribution for each location.
# p_i = proportion of permuted I_i >= observed I_i (two-tailed).

print("\nRunning permutation test...", flush=True)
rng        = np.random.default_rng(seed=42)   # reproducible
perm_counts = np.zeros(len(r_lisa))            # counts |I_perm| >= |I_obs|

for _ in range(N_PERMUTATIONS):
    r_perm   = rng.permutation(z_lisa)         # shuffle residuals
    Wz_perm  = W_row @ r_perm                  # spatial lag of shuffled
    I_perm   = (r_perm * Wz_perm) / m2         # local I under null
    # Two-tailed: count permutations where |I_perm| >= |I_obs|
    perm_counts += (np.abs(I_perm) >= np.abs(I_local)).astype(int)

p_local = perm_counts / N_PERMUTATIONS         # empirical p-value per location
print("Done.")

# --- 9d. Classify cluster types --------------------------------------------
# Classification only assigned where p_i < LISA_ALPHA.
# Quadrant determined by sign of z_i (own value) and sign of Wz_i (spatial lag).

cluster_type = np.full(len(r_lisa), "Not significant", dtype=object)
sig_mask     = p_local < LISA_ALPHA

hh = sig_mask & (z_lisa > 0) & (Wz > 0)
ll = sig_mask & (z_lisa < 0) & (Wz < 0)
hl = sig_mask & (z_lisa > 0) & (Wz < 0)
lh = sig_mask & (z_lisa < 0) & (Wz > 0)

cluster_type[hh] = "High-High"
cluster_type[ll] = "Low-Low"
cluster_type[hl] = "High-Low"
cluster_type[lh] = "Low-High"

# --- 9e. Assemble results dataframe ----------------------------------------

lisa_df = spatial_df[["lat", "lon", "community", "resid", "z_resid"]].copy()
lisa_df["I_local"]      = I_local
lisa_df["Wz_lag"]       = Wz           # spatial lag of centred residual
lisa_df["p_local"]      = p_local
lisa_df["cluster_type"] = cluster_type

lisa_df.to_csv("evi_local_morans_results.csv", index=False)
print(f"\nResults saved: evi_local_morans_results.csv ({len(lisa_df)} observations)")

# --- 9f. Console summary ---------------------------------------------------

print("\nLISA cluster type frequencies:")
print(f"  {'Cluster type':<20}  {'n':>5}  {'%':>7}")
print("  " + "-" * 35)
cluster_order = ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]
n_sig_total = sig_mask.sum()
for ct in cluster_order:
    n_ct  = (cluster_type == ct).sum()
    pct_ct = n_ct / len(cluster_type) * 100
    print(f"  {ct:<20}  {n_ct:>5}  {pct_ct:>6.1f}%")
print(f"\n  Total significant: {n_sig_total} ({n_sig_total/len(cluster_type)*100:.1f}%)")

# --- 9g. Save text summary -------------------------------------------------

with open("evi_local_morans_summary.txt", "w") as f:
    f.write("Local Moran's I (LISA) - Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model            : Individual Model (IM) - Stage 2\n")
    f.write(f"Dependent        : {DEPENDENT}\n")
    f.write(f"Distance threshold: {LISA_THRESHOLD} km\n")
    f.write(f"Permutations     : {N_PERMUTATIONS}\n")
    f.write(f"Significance     : p < {LISA_ALPHA}\n")
    f.write(f"n (GPS sample)   : {len(lisa_df)}\n\n")
    f.write("Cluster type frequencies:\n")
    f.write(f"  {'Cluster type':<20}  {'n':>5}  {'%':>7}\n")
    f.write("  " + "-" * 35 + "\n")
    for ct in cluster_order:
        n_ct  = (cluster_type == ct).sum()
        pct_ct = n_ct / len(cluster_type) * 100
        f.write(f"  {ct:<20}  {n_ct:>5}  {pct_ct:>6.1f}%\n")
    f.write(f"\n  Total significant: {n_sig_total} ({n_sig_total/len(cluster_type)*100:.1f}%)\n\n")
    f.write("References:\n")
    f.write("  Anselin, L. (1995). Local indicators of spatial association — LISA.\n")
    f.write("  Geographical Analysis, 27(2), 93-115.\n")
    f.write("  https://doi.org/10.1111/j.1538-4632.1995.tb00338.x\n")
    f.write("  Aksha et al. (2019). Int J Disaster Risk Sci, 10, 103-116.\n")
    f.write("  https://doi.org/10.1007/s13753-018-0192-7\n")

print("Summary saved: evi_local_morans_summary.txt")

# =============================================================================
# 10. FIGURE 4 - LISA CLUSTER MAP
# =============================================================================
# Colour scheme follows Anselin (1995) convention:
#   Red      = High-High  (hotspot: high residual cluster)
#   Blue     = Low-Low    (coldspot: low residual cluster)
#   Pink     = High-Low   (spatial outlier: high surrounded by low)
#   Light blue = Low-High (spatial outlier: low surrounded by high)
#   Grey     = Not significant
#
# HH clusters identify areas where the model systematically under-predicts
# vulnerability (actual > predicted). These are communities worth examining
# for unmeasured risk factors not captured in the IM predictors.
# LL clusters identify areas where vulnerability is lower than predicted.

CLUSTER_COLOURS = {
    "High-High":       "#D7191C",   # red    — hotspot
    "Low-Low":         "#2C7BB6",   # blue   — coldspot
    "High-Low":        "#F4A58A",   # salmon — outlier (high)
    "Low-High":        "#ABD9E9",   # light blue — outlier (low)
    "Not significant": "#CCCCCC",   # grey
}
CLUSTER_SIZES = {
    "High-High": 30, "Low-Low": 30,
    "High-Low":  22, "Low-High": 22,
    "Not significant": 8,
}

fig4, ax4 = plt.subplots(figsize=(13, 8))

# Set axes extent BEFORE fetching OSM tiles.
_lon_pad_f4 = (lisa_df["lon"].max() - lisa_df["lon"].min()) * 0.04
_lat_pad_f4 = (lisa_df["lat"].max() - lisa_df["lat"].min()) * 0.04
ax4.set_xlim(lisa_df["lon"].min() - _lon_pad_f4,
             lisa_df["lon"].max() + _lon_pad_f4)
ax4.set_ylim(lisa_df["lat"].min() - _lat_pad_f4,
             lisa_df["lat"].max() + _lat_pad_f4)

# CartoDB Positron: minimal grey — appropriate for seismic LISA map.
_osm_add_basemap(
    ax4,
    zoom="auto",
    provider="carto_voyager",
    alpha=0.50,
    cache_dir=Path.cwd() / "osm_tile_cache",
)

# Plot non-significant first (background), then clusters on top
for ct in ["Not significant", "Low-High", "High-Low", "Low-Low", "High-High"]:
    mask_ct = lisa_df["cluster_type"] == ct
    ax4.scatter(
        lisa_df.loc[mask_ct, "lon"],
        lisa_df.loc[mask_ct, "lat"],
        c=CLUSTER_COLOURS[ct],
        s=CLUSTER_SIZES[ct],
        alpha=0.75 if ct != "Not significant" else 0.35,
        linewidths=0.3,
        edgecolors="#555555" if ct != "Not significant" else "none",
        label=f"{ct} (n={(mask_ct).sum()})",
        zorder=3 if ct != "Not significant" else 2,
    )

# Community centroid labels
for comm, grp in lisa_df.groupby("community"):
    if pd.isna(comm) or str(comm).strip() == "":
        continue
    cx    = grp["lon"].mean()
    cy    = grp["lat"].mean()
    label = str(comm).replace("_", " ").title()
    ax4.text(
        cx, cy, label,
        fontsize=6.5, ha="center", va="bottom",
        color="#222222", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.5),
        zorder=5,
    )

ax4.set_xlabel("Longitude", fontsize=10)
ax4.set_ylabel("Latitude",  fontsize=10)
ax4.set_title(
    "LISA Cluster Map — Local Moran's I of OLS Residuals (EVI)\n"
    f"Individual Model (IM), Lumbini Province, Nepal  |  "
    f"Threshold = {LISA_THRESHOLD} km, p < {LISA_ALPHA}, "
    f"{N_PERMUTATIONS} permutations",
    fontsize=11, fontweight="bold", pad=12,
)
ax4.legend(
    title="Cluster type", title_fontsize=9,
    fontsize=8.5, loc="lower left",
    framealpha=0.9, edgecolor="#bbbbbb",
)
ax4.tick_params(labelsize=8)
ax4.spines[["top", "right"]].set_visible(False)
ax4.set_facecolor("#eeeeee")  # fallback: shown only if basemap tiles fail
ax4.grid(True, color="white", linewidth=0.5, alpha=0.6, zorder=1)

plt.tight_layout()
plt.savefig("evi_local_morans_map.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 4 saved: evi_local_morans_map.png")
print("\nDiagnosis complete.")
