"""
stage04_mnar_imputation.py
============================
MNAR Imputation via Building Typology Cluster Modal Scores
Nepal Lumbini Survey Dataset (n=2,993)

PURPOSE
-------
Imputes missing values for MNAR-flagged EVI and FVI indicators using the
building typology clusters identified by stage03_typology_clusters.py.

For each MNAR indicator, each missing building is assigned the modal
vulnerability score of its typology cluster (computed from observed donors
within that cluster). This is a hot-deck imputation by donor class, which
is more principled than blanket worst/best-case sensitivity bounds because
the donor class is empirically grounded in the building's own structural
characteristics.

PIPELINE POSITION
-----------------
  1. stage02_evi_scoring.py          → EVI_CIMDEN_scores.csv
  2. stage01_fvi_scoring.py           → FVI_CIMDEN_scores.csv
  3. missing_data_diagnostic.py → Diagnoses MCAR/MAR/MNAR
  4. stage03_typology_clusters.py → typology_clusters.csv (TYPOLOGY_CLUSTER)
  5. THIS SCRIPT                → *_imputed.csv  ← produces imputed outputs
  6. MICE imputation            → handles remaining MAR indicators
  7. Regression pipeline        → Stages 3-6

INPUTS
------
  EVI_CIMDEN_scores.csv         — output of stage02_evi_scoring.py
  FVI_CIMDEN_scores.csv         — output of stage01_fvi_scoring.py
  typology_clusters.csv         — output of stage03_typology_clusters.py
                                  (must contain TYPOLOGY_CLUSTER column)

OUTPUTS
-------
  EVI_imputed.csv               — EVI scores with MNAR indicators imputed
  FVI_imputed.csv               — FVI scores with MNAR indicators imputed,
                                  FVI_SCORE / norms / class recomputed
  fig_mnar_imputation_evi.png   — Before/after missingness + score distribution
  fig_mnar_imputation_fvi.png   — Before/after missingness + score distribution
  mnar_imputation_report.txt    — Full audit trail for methods section

MNAR INDICATORS IMPUTED
-----------------------
  EVI (from missing_data_diagnostic.py Step 3/4):
    EVI_D3_01  H/t Ratio      — wall inaccessible if damaged/collapsed
    EVI_D3_04  Wall Thickness — same; unmeasurable if wall is destroyed
    EVI_D5_04  Roof Fastening — 44.5% missing; roof detail inaccessible

  FVI (from missing_data_diagnostic.py Step 3):
    FVI3_RoofConn_LVL  Roof connections — 44.5% missing; MAR confirmed
                       but 44.5% exceeds imputation reliability threshold;
                       typology modal is more reliable than MICE at this rate
    FVI6_Geom_LVL      Geometry         — MNAR possible (accuracy=0.472)
    FVI8_Apron_LVL     Apron            — MNAR possible (accuracy=0.530)
    FVI9_Drain_LVL     Drainage         — MNAR possible (accuracy=0.441)
    FVI11_LWratio_LVL  L/W ratio        — MNAR possible (accuracy=0.455)

IMPUTATION LOGIC
----------------
For each MNAR indicator:
  1. For each cluster c, compute modal score from observed (non-missing)
     buildings in that cluster.
  2. For buildings missing the indicator, assign the modal score of their
     cluster.
  3. If a cluster has fewer than MIN_DONORS observed values for that
     indicator, fall back to the global modal score.
  4. Flag each imputed value with a companion boolean column
     (e.g. EVI_D3_01_imputed = True).

FVI RECOMPUTATION
-----------------
After imputing FVI _LVL columns, the weighted score (_WS) and composite
(FVI_SCORE, FVI_MIN_AVAIL, FVI_MAX_AVAIL, FVI_norm_0_1, FVI_norm_1_5,
FVI_CLASS) are fully recomputed to maintain internal consistency.
Mirrors stage01_fvi_scoring.py exactly.

TRANSPARENCY AND REPRODUCIBILITY (FAIR PRINCIPLES)
---------------------------------------------------
All imputation decisions are logged to mnar_imputation_report.txt,
including: which clusters triggered the global fallback, how many values
were imputed per indicator per cluster, and the modal score used for each.
This allows any reader to reproduce or audit the imputation.

REFERENCES
----------
Andridge, R.R. & Little, R.J.A. (2010). A Review of Hot Deck Imputation
  for Survey Non-Response. International Statistical Review, 78(1), 40-64.
  https://doi.org/10.1111/j.1751-5823.2010.00103.x

Little, R.J.A. & Rubin, D.B. (2002). Statistical Analysis with Missing
  Data (2nd ed.). Wiley. https://doi.org/10.1002/9781119013563

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
  CRC Press. https://stefvanbuuren.name/fimd/

DEPENDENCIES
------------
    pip install pandas numpy matplotlib seaborn scipy
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# FILE PATHS
# =============================================================================

HERE       = Path(__file__).resolve().parent
BASE_DIR   = HERE
EVI_CSV    = HERE / "outputs" / "EVI_CIMDEN_scores.csv"
FVI_CSV    = HERE / "outputs" / "FVI_CIMDEN_scores.csv"
CLUST_CSV  = HERE / "outputs" / "typology_clusters.csv"
OUTPUT_DIR = HERE / "outputs" / "imputation" / "mnar_typology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum number of observed donors in a cluster before falling back to
# the global modal score. Clusters with fewer observed values than this
# threshold are considered too small to provide a reliable modal estimate.
MIN_DONORS = 10

# Cluster 5 (n=17) was flagged as anomalously small in the cluster analysis.
# It is listed here explicitly so it always uses the global fallback
# regardless of MIN_DONORS.
SMALL_CLUSTERS = {5}

# FVI indicator weights — mirrors stage01_fvi_scoring.py exactly.
# Order: FVI2, FVI3, FVI4, FVI5, FVI6, FVI7, FVI8, FVI9, FVI10, FVI11
FVI_WEIGHTS = {
    "FVI2_RoofMat_LVL":  8,
    "FVI3_RoofConn_LVL": 9,
    "FVI4_WallMat_LVL":  9,
    "FVI5_Elev_LVL":     8,
    "FVI6_Geom_LVL":     9,
    "FVI7_Overhang_LVL": 9,
    "FVI8_Apron_LVL":    9,
    "FVI9_Drain_LVL":    8,
    "FVI10_Floors_LVL":  8,
    "FVI11_LWratio_LVL": 8,
}

# MNAR indicators to impute — EVI
EVI_MNAR = [
    "EVI_D5_04",   # Roof Fastening — 44.5% missing, domain MNAR reasoning
    "EVI_D3_01",   # H/t Ratio     — wall inaccessible if damaged
    "EVI_D3_04",   # Wall Thickness — same
]

# MNAR indicators to impute — FVI (_LVL columns)
FVI_MNAR = [
    "FVI3_RoofConn_LVL",   # 44.5% missing — MAR confirmed, above reliability limit
    "FVI6_Geom_LVL",       # MNAR possible (accuracy=0.472)
    "FVI8_Apron_LVL",      # MNAR possible (accuracy=0.530)
    "FVI9_Drain_LVL",      # MNAR possible (accuracy=0.441)
    "FVI11_LWratio_LVL",   # MNAR possible (accuracy=0.455)
]

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 65)
print("MNAR TYPOLOGY IMPUTATION")
print("=" * 65)

df_evi   = pd.read_csv(EVI_CSV,   encoding="utf-8-sig", low_memory=False)
df_fvi   = pd.read_csv(FVI_CSV,   encoding="utf-8-sig", low_memory=False)
df_clust = pd.read_csv(CLUST_CSV, encoding="utf-8-sig", low_memory=False)

# Verify cluster column is present
if "TYPOLOGY_CLUSTER" not in df_clust.columns:
    raise ValueError(
        "TYPOLOGY_CLUSTER column not found in typology_clusters.csv.\n"
        "Run stage03_typology_clusters.py first."
    )

print(f"\nLoaded EVI scores  : {df_evi.shape[0]:,} rows x {df_evi.shape[1]:,} cols")
print(f"Loaded FVI scores  : {df_fvi.shape[0]:,} rows x {df_fvi.shape[1]:,} cols")
print(f"Loaded clusters    : {df_clust.shape[0]:,} rows, "
      f"{df_clust['TYPOLOGY_CLUSTER'].nunique()} clusters")

# Attach cluster labels to EVI and FVI using row index (same CSV, same order)
if len(df_evi) != len(df_clust) or len(df_fvi) != len(df_clust):
    raise ValueError(
        f"Row count mismatch: EVI={len(df_evi)}, FVI={len(df_fvi)}, "
        f"clusters={len(df_clust)}.\n"
        "All three CSVs must derive from the same raw survey file in the "
        "same row order."
    )

df_evi["TYPOLOGY_CLUSTER"] = df_clust["TYPOLOGY_CLUSTER"].values
df_fvi["TYPOLOGY_CLUSTER"] = df_clust["TYPOLOGY_CLUSTER"].values

cluster_ids = sorted(df_clust["TYPOLOGY_CLUSTER"].unique())
print(f"Cluster IDs        : {cluster_ids}")

# =============================================================================
# CORE IMPUTATION FUNCTION
# =============================================================================

def impute_mnar_by_cluster(
    df: pd.DataFrame,
    indicator: str,
    cluster_col: str = "TYPOLOGY_CLUSTER",
    min_donors: int  = MIN_DONORS,
    small_clusters: set = SMALL_CLUSTERS,
    report_lines: list = None,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Hot-deck imputation of a MNAR indicator using cluster modal scores.

    For each cluster, computes the mode of the indicator from observed
    (non-missing) rows. Missing rows in that cluster are assigned the
    cluster mode. Clusters with fewer than min_donors observed values,
    or in small_clusters, fall back to the global mode.

    Parameters
    ----------
    df            : DataFrame containing the indicator and cluster columns.
    indicator     : Column name of the MNAR indicator to impute.
    cluster_col   : Column name containing cluster labels.
    min_donors    : Minimum observed values in a cluster to use cluster mode.
    small_clusters: Set of cluster IDs that always use the global fallback.
    report_lines  : List to append audit lines to (for report file).

    Returns
    -------
    imputed_series  : pd.Series — indicator values after imputation.
    flag_series     : pd.Series — boolean True where a value was imputed.
    cluster_modals  : dict      — {cluster_id: modal_score_used}.
    """
    if indicator not in df.columns:
        raise KeyError(
            f"Indicator '{indicator}' not found in DataFrame.\n"
            f"Available columns: {[c for c in df.columns if 'EVI' in c or 'FVI' in c]}"
        )

    series       = df[indicator].copy()
    clusters     = df[cluster_col]
    missing_mask = series.isna()
    n_missing    = missing_mask.sum()

    if n_missing == 0:
        if report_lines is not None:
            report_lines.append(f"  {indicator}: no missing values — skipped")
        return series, pd.Series(False, index=df.index), {}

    # Global modal score — fallback for small/sparse clusters
    observed_global = series.dropna()
    global_mode     = observed_global.mode()
    if len(global_mode) == 0:
        raise ValueError(f"No observed values for {indicator} — cannot impute.")
    global_mode = global_mode.iloc[0]

    imputed_series = series.copy()
    flag_series    = pd.Series(False, index=df.index)
    cluster_modals = {}
    fallback_clusters = []

    for c in sorted(clusters.unique()):
        # Rows in this cluster that are missing the indicator
        cluster_missing = missing_mask & (clusters == c)
        n_to_impute     = cluster_missing.sum()
        if n_to_impute == 0:
            continue

        # Observed values in this cluster
        cluster_observed = series[(clusters == c) & ~missing_mask]
        n_donors         = len(cluster_observed)

        # Decide: cluster mode or global fallback
        use_fallback = (c in small_clusters) or (n_donors < min_donors)

        if use_fallback:
            modal_score = global_mode
            fallback_clusters.append(c)
            reason = (
                f"SMALL_CLUSTERS override"
                if c in small_clusters
                else f"insufficient donors (n={n_donors} < {min_donors})"
            )
        else:
            cluster_mode_result = cluster_observed.mode()
            modal_score = (
                cluster_mode_result.iloc[0]
                if len(cluster_mode_result) > 0
                else global_mode
            )
            reason = f"cluster modal (n_donors={n_donors})"

        cluster_modals[c] = modal_score
        imputed_series.loc[cluster_missing] = modal_score
        flag_series.loc[cluster_missing]    = True

        line = (
            f"    Cluster {c}: imputed {n_to_impute:>4} values → "
            f"score={modal_score:.0f}  [{reason}]"
        )
        print(line)
        if report_lines is not None:
            report_lines.append(line)

    n_imputed = flag_series.sum()
    summary = (
        f"  {indicator}: {n_missing} missing → {n_imputed} imputed  "
        f"| global_mode={global_mode:.0f}"
        + (f"  | fallback clusters: {fallback_clusters}" if fallback_clusters else "")
    )
    print(summary)
    if report_lines is not None:
        report_lines.append(summary)

    return imputed_series, flag_series, cluster_modals


# =============================================================================
# STEP 1 — IMPUTE EVI MNAR INDICATORS
# =============================================================================

print("\n" + "=" * 65)
print("STEP 1 — EVI MNAR IMPUTATION")
print("=" * 65)

evi_report  = ["EVI MNAR IMPUTATION REPORT", "=" * 50]
evi_before  = {}   # Store pre-imputation missing counts for reporting
evi_after   = {}

for indicator in EVI_MNAR:
    n_before = df_evi[indicator].isna().sum()
    evi_before[indicator] = n_before
    pct = n_before / len(df_evi) * 100
    print(f"\n--- {indicator} ({n_before:,} missing, {pct:.1f}%) ---")
    evi_report.append(f"\n{indicator}  ({n_before} missing, {pct:.1f}%)")

    imputed, flag, modals = impute_mnar_by_cluster(
        df_evi, indicator, report_lines=evi_report
    )
    df_evi[indicator]                  = imputed
    df_evi[f"{indicator}_imputed"]     = flag
    evi_after[indicator]               = df_evi[indicator].isna().sum()

# Recompute EVI composite after imputation
EVI_INDICATORS = [
    "EVI_D1_01", "EVI_D1_02",
    "EVI_D2_01", "EVI_D2_02",
    "EVI_D3_01", "EVI_D3_02", "EVI_D3_03", "EVI_D3_04",
    "EVI_D4_01", "EVI_D4_02",
    "EVI_D5_01", "EVI_D5_04",
]

# Verify all EVI indicator columns are present
missing_evi_cols = [c for c in EVI_INDICATORS if c not in df_evi.columns]
if missing_evi_cols:
    raise ValueError(
        f"EVI indicator columns missing from EVI_CIMDEN_scores.csv: "
        f"{missing_evi_cols}\nCheck that stage02_evi_scoring.py was run first."
    )

df_evi["EVI_composite"] = df_evi[EVI_INDICATORS].mean(axis=1)
df_evi["EVI_n_valid"]   = df_evi[EVI_INDICATORS].notna().sum(axis=1)

valid_evi = df_evi["EVI_composite"].notna()
df_evi["EVI_norm_0_1"] = np.nan
df_evi.loc[valid_evi, "EVI_norm_0_1"] = (
    (df_evi.loc[valid_evi, "EVI_composite"] - 1) / 4.0
)
df_evi["EVI_norm_1_5"] = np.nan
df_evi.loc[valid_evi, "EVI_norm_1_5"] = (
    1 + df_evi.loc[valid_evi, "EVI_norm_0_1"] * 4
)

def classify_evi(v):
    """Return 1=Low, 2=Medium, 3=High, or NaN."""
    if pd.isna(v): return np.nan
    if v <= 1/3:   return 1
    elif v <= 2/3: return 2
    else:          return 3

df_evi["EVI_CLASS"] = df_evi["EVI_norm_0_1"].apply(classify_evi)

print("\nEVI composite recomputed after imputation.")
print(f"  EVI_norm_0_1 — mean: {df_evi['EVI_norm_0_1'].mean():.4f}  "
      f"median: {df_evi['EVI_norm_0_1'].median():.4f}")
print(f"  EVI_CLASS distribution:")
for k, label in {1: "Low", 2: "Medium", 3: "High"}.items():
    n = (df_evi["EVI_CLASS"] == k).sum()
    print(f"    {label:8s}: {n:,} ({n/len(df_evi)*100:.1f}%)")


# =============================================================================
# STEP 2 — IMPUTE FVI MNAR INDICATORS
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2 — FVI MNAR IMPUTATION")
print("=" * 65)

fvi_report = ["FVI MNAR IMPUTATION REPORT", "=" * 50]
fvi_before = {}
fvi_after  = {}

for indicator in FVI_MNAR:
    n_before = df_fvi[indicator].isna().sum()
    fvi_before[indicator] = n_before
    pct = n_before / len(df_fvi) * 100
    print(f"\n--- {indicator} ({n_before:,} missing, {pct:.1f}%) ---")
    fvi_report.append(f"\n{indicator}  ({n_before} missing, {pct:.1f}%)")

    imputed, flag, modals = impute_mnar_by_cluster(
        df_fvi, indicator, report_lines=fvi_report
    )
    df_fvi[indicator]              = imputed
    df_fvi[f"{indicator}_imputed"] = flag
    fvi_after[indicator]           = df_fvi[indicator].isna().sum()

# ── Recompute FVI weighted scores (_WS) for all indicators ───────────────────
# Mirrors stage01_fvi_scoring.py exactly. Must recompute ALL _WS columns because
# FVI_SCORE is the sum of all weighted scores, not just the imputed ones.
print("\nRecomputing FVI weighted scores and composite...")

for lvl_col, weight in FVI_WEIGHTS.items():
    ws_col = lvl_col.replace("_LVL", "_WS")
    if lvl_col not in df_fvi.columns:
        print(f"  WARNING: {lvl_col} not found in FVI CSV — skipping _WS recompute")
        continue
    df_fvi[ws_col] = np.where(
        df_fvi[lvl_col].notna(),
        weight * df_fvi[lvl_col],
        np.nan
    )

ws_cols = [col.replace("_LVL", "_WS") for col in FVI_WEIGHTS]
ws_cols = [c for c in ws_cols if c in df_fvi.columns]

weights_list     = [FVI_WEIGHTS[c.replace("_WS", "_LVL")] for c in ws_cols]
weights_min_list = [w * 1 for w in weights_list]
weights_max_list = [w * 5 for w in weights_list]

df_fvi["FVI_SCORE"] = df_fvi[ws_cols].sum(axis=1, skipna=True, min_count=1)

df_fvi["FVI_MIN_AVAIL"] = sum(
    weights_min_list[i] * df_fvi[ws_cols[i]].notna().astype(int)
    for i in range(len(ws_cols))
)
df_fvi["FVI_MAX_AVAIL"] = sum(
    weights_max_list[i] * df_fvi[ws_cols[i]].notna().astype(int)
    for i in range(len(ws_cols))
)

valid_fvi = (
    df_fvi["FVI_SCORE"].notna()
    & (df_fvi["FVI_MAX_AVAIL"] > df_fvi["FVI_MIN_AVAIL"])
)
df_fvi["FVI_norm_0_1"] = np.nan
df_fvi.loc[valid_fvi, "FVI_norm_0_1"] = (
    (df_fvi.loc[valid_fvi, "FVI_SCORE"] - df_fvi.loc[valid_fvi, "FVI_MIN_AVAIL"])
    / (df_fvi.loc[valid_fvi, "FVI_MAX_AVAIL"] - df_fvi.loc[valid_fvi, "FVI_MIN_AVAIL"])
)
df_fvi["FVI_norm_1_5"] = np.nan
df_fvi.loc[valid_fvi, "FVI_norm_1_5"] = (
    1 + df_fvi.loc[valid_fvi, "FVI_norm_0_1"] * 4
)

def classify_fvi(v):
    """Return 1=Low, 2=Medium, 3=High, or NaN."""
    if pd.isna(v): return np.nan
    if v <= 1/3:   return 1
    elif v <= 2/3: return 2
    else:          return 3

df_fvi["FVI_CLASS"] = df_fvi["FVI_norm_0_1"].apply(classify_fvi)

print("FVI composite recomputed after imputation.")
print(f"  FVI_norm_0_1 — mean: {df_fvi['FVI_norm_0_1'].mean():.4f}  "
      f"median: {df_fvi['FVI_norm_0_1'].median():.4f}")
print(f"  FVI_CLASS distribution:")
for k, label in {1: "Low", 2: "Medium", 3: "High"}.items():
    n = (df_fvi["FVI_CLASS"] == k).sum()
    print(f"    {label:8s}: {n:,} ({n/len(df_fvi)*100:.1f}%)")


# =============================================================================
# STEP 3 — VISUALISATIONS
# =============================================================================

def _plot_imputation_result(
    df_before_vals: dict,
    df_after_vals:  dict,
    norm_col:       str,
    class_col:      str,
    df:             pd.DataFrame,
    index_label:    str,
    flag_cols:      list,
    out_path:       Path,
) -> None:
    """
    Two-panel figure per index:
      Left:  Missing count before vs after per MNAR indicator (bar chart).
      Right: Composite score distribution split by imputed vs observed rows.
    """
    indicators = list(df_before_vals.keys())
    n_ind      = len(indicators)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: missing count before/after ────────────────────────────────────
    x       = np.arange(n_ind)
    width   = 0.35
    before  = [df_before_vals[i] for i in indicators]
    after   = [df_after_vals[i]  for i in indicators]
    short   = [i.replace("EVI_", "").replace("_LVL", "").replace("_", " ")
               for i in indicators]

    b1 = ax1.bar(x - width/2, before, width,
                 color="#d62728", alpha=0.8, label="Before imputation")
    b2 = ax1.bar(x + width/2, after,  width,
                 color="#2ca02c", alpha=0.8, label="After imputation")

    for bar, val in zip(b1, before):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     str(val), ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, after):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(val), ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Number of missing values", fontsize=10)
    ax1.set_title(f"{index_label} — Missing values before and after\n"
                  "typology-based MNAR imputation", fontsize=10)
    ax1.legend(fontsize=9)

    # ── Right: composite score distribution ──────────────────────────────────
    # Mark rows where ANY indicator was imputed
    any_imputed = pd.Series(False, index=df.index)
    for fc in flag_cols:
        if fc in df.columns:
            any_imputed = any_imputed | df[fc]

    obs_scores = df.loc[~any_imputed, norm_col].dropna()
    imp_scores = df.loc[any_imputed,  norm_col].dropna()

    bins = np.linspace(0, 1, 25)
    ax2.hist(obs_scores, bins=bins, alpha=0.6, color="#4C72B0",
             label=f"Fully observed (n={len(obs_scores):,})",
             edgecolor="white", linewidth=0.4)
    ax2.hist(imp_scores, bins=bins, alpha=0.6, color="#ff7f0e",
             label=f"Contains imputed value (n={len(imp_scores):,})",
             edgecolor="white", linewidth=0.4)

    ax2.set_xlabel(f"{index_label} norm_0_1 (composite score)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title(f"{index_label} composite score distribution\n"
                  "Observed vs rows with imputed MNAR value", fontsize=10)
    ax2.legend(fontsize=9)

    # Kolmogorov-Smirnov test: are the two distributions significantly different?
    if len(obs_scores) > 0 and len(imp_scores) > 0:
        ks_stat, ks_p = stats.ks_2samp(obs_scores, imp_scores)
        ax2.text(0.02, 0.97,
                 f"KS test: D={ks_stat:.3f}, p={ks_p:.4f}\n"
                 + ("Distributions differ significantly"
                    if ks_p < 0.05
                    else "Distributions not significantly different"),
                 transform=ax2.transAxes, va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


print("\n" + "=" * 65)
print("STEP 3 — VISUALISATIONS")
print("=" * 65)

evi_flag_cols = [f"{ind}_imputed" for ind in EVI_MNAR]
fvi_flag_cols = [f"{ind}_imputed" for ind in FVI_MNAR]

_plot_imputation_result(
    evi_before, evi_after,
    norm_col    = "EVI_norm_0_1",
    class_col   = "EVI_CLASS",
    df          = df_evi,
    index_label = "EVI",
    flag_cols   = evi_flag_cols,
    out_path    = OUTPUT_DIR / "fig_mnar_imputation_evi.png",
)

_plot_imputation_result(
    fvi_before, fvi_after,
    norm_col    = "FVI_norm_0_1",
    class_col   = "FVI_CLASS",
    df          = df_fvi,
    index_label = "FVI",
    flag_cols   = fvi_flag_cols,
    out_path    = OUTPUT_DIR / "fig_mnar_imputation_fvi.png",
)

# =============================================================================
# STEP 4 — SAVE OUTPUT CSVs
# =============================================================================

print("\n" + "=" * 65)
print("STEP 4 — SAVING OUTPUTS")
print("=" * 65)

evi_out = OUTPUT_DIR.parent / "EVI_imputed.csv"
fvi_out = OUTPUT_DIR.parent / "FVI_imputed.csv"

# Drop the working TYPOLOGY_CLUSTER column from outputs so downstream
# scripts receive the same column structure as the original scoring outputs,
# with only the new *_imputed flag columns added.
df_evi.drop(columns=["TYPOLOGY_CLUSTER"], errors="ignore").to_csv(
    evi_out, index=False
)
df_fvi.drop(columns=["TYPOLOGY_CLUSTER"], errors="ignore").to_csv(
    fvi_out, index=False
)

print(f"  EVI_imputed.csv saved  ({len(df_evi):,} rows)")
print(f"  FVI_imputed.csv saved  ({len(df_fvi):,} rows)")
print(f"\n  New columns added to EVI_imputed.csv:")
for col in evi_flag_cols:
    n_true = df_evi[col].sum() if col in df_evi.columns else 0
    print(f"    {col}: {n_true:,} True")
print(f"\n  New columns added to FVI_imputed.csv:")
for col in fvi_flag_cols:
    n_true = df_fvi[col].sum() if col in df_fvi.columns else 0
    print(f"    {col}: {n_true:,} True")

# =============================================================================
# STEP 5 — WRITE AUDIT REPORT
# =============================================================================
# The report documents every imputation decision for methods transparency.
# It is written to a plain text file so it can be included as a
# supplementary file or appendix to the thesis.

report_path = OUTPUT_DIR / "mnar_imputation_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("MNAR TYPOLOGY IMPUTATION — AUDIT REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Dataset        : Nepal Lumbini Survey (n={len(df_evi):,})\n")
    f.write(f"Cluster file   : {CLUST_CSV.name}\n")
    f.write(f"MIN_DONORS     : {MIN_DONORS}\n")
    f.write(f"SMALL_CLUSTERS : {SMALL_CLUSTERS} (always use global fallback)\n\n")

    f.write("METHOD\n------\n")
    f.write(
        "Hot-deck imputation by donor class (Andridge & Little, 2010).\n"
        "For each MNAR indicator, the modal vulnerability score is computed\n"
        "from observed buildings within the same typology cluster. Missing\n"
        "buildings are assigned their cluster's modal score. Clusters with\n"
        "fewer than MIN_DONORS observed values use the global modal score.\n\n"
    )

    f.write("\n".join(evi_report))
    f.write("\n\n")
    f.write("\n".join(fvi_report))
    f.write("\n\n")

    f.write("POST-IMPUTATION COMPOSITE SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write(f"EVI_norm_0_1: mean={df_evi['EVI_norm_0_1'].mean():.4f}  "
            f"median={df_evi['EVI_norm_0_1'].median():.4f}  "
            f"sd={df_evi['EVI_norm_0_1'].std():.4f}\n")
    f.write(f"FVI_norm_0_1: mean={df_fvi['FVI_norm_0_1'].mean():.4f}  "
            f"median={df_fvi['FVI_norm_0_1'].median():.4f}  "
            f"sd={df_fvi['FVI_norm_0_1'].std():.4f}\n\n")

    f.write("EVI_CLASS distribution (post-imputation):\n")
    for k, label in {1: "Low", 2: "Medium", 3: "High"}.items():
        n = (df_evi["EVI_CLASS"] == k).sum()
        f.write(f"  {label:8s}: {n:,} ({n/len(df_evi)*100:.1f}%)\n")
    f.write("\nFVI_CLASS distribution (post-imputation):\n")
    for k, label in {1: "Low", 2: "Medium", 3: "High"}.items():
        n = (df_fvi["FVI_CLASS"] == k).sum()
        f.write(f"  {label:8s}: {n:,} ({n/len(df_fvi)*100:.1f}%)\n")

    f.write("\nREFERENCES\n----------\n")
    f.write("Andridge & Little (2010): https://doi.org/10.1111/j.1751-5823.2010.00103.x\n")
    f.write("Little & Rubin (2002)   : https://doi.org/10.1002/9781119013563\n")
    f.write("van Buuren (2018)       : https://stefvanbuuren.name/fimd/\n")

print(f"\n  Audit report saved: {report_path.name}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print("\nEVI — imputation summary:")
for ind in EVI_MNAR:
    b  = evi_before[ind]
    a  = evi_after[ind]
    fc = f"{ind}_imputed"
    n_imp = df_evi[fc].sum() if fc in df_evi.columns else 0
    print(f"  {ind:<16}: {b:>4} missing → {a:>4} remaining  "
          f"({n_imp:>4} imputed)")

print("\nFVI — imputation summary:")
for ind in FVI_MNAR:
    b  = fvi_before[ind]
    a  = fvi_after[ind]
    fc = f"{ind}_imputed"
    n_imp = df_fvi[fc].sum() if fc in df_fvi.columns else 0
    print(f"  {ind:<24}: {b:>4} missing → {a:>4} remaining  "
          f"({n_imp:>4} imputed)")

print(f"""
NEXT STEP — MICE IMPUTATION
----------------------------
EVI_imputed.csv and FVI_imputed.csv now have all MNAR indicators filled.
Remaining NaN values in these files are MAR indicators, suitable for
Multiple Imputation by Chained Equations (MICE).

  Python: sklearn.impute.IterativeImputer
  R:      mice package (van Buuren, 2018)

Include the *_imputed flag columns as predictors in the MICE model so
that the imputation method (typology vs observed) is accounted for.
Also include TYPOLOGY_CLUSTER, surveyor ID, and survey week as auxiliary
predictors in the MICE model (identified as MAR drivers in Step 5 of
missing_data_diagnostic.py).

All outputs saved to: {OUTPUT_DIR}
""")
