"""
stage03_typology_clusters.py
=============================
Building Typology Identification via K-Means Clustering
Nepal Lumbini Survey Dataset (n=2,993)

PURPOSE
-------
Identifies data-driven building typology classes from raw structural
variables. The resulting cluster labels are used as donor classes for
hot-deck imputation of MNAR-flagged indicators in the EVI and FVI pipelines.

This script is distinct from and must run BEFORE:
  - stage04_mnar_imputation.py  (uses cluster labels to impute)
  - MICE imputation               (handles remaining MAR indicators)

WHY CLUSTER-BASED RATHER THAN RULE-BASED TYPOLOGIES
----------------------------------------------------
Manually defined typologies (e.g. "stone + 2 floors + gable = Hill type")
embed researcher assumptions. Data-driven clustering lets the building
stock define its own natural groupings, which is more defensible for MNAR
imputation because the donor class is empirically grounded rather than
assumed. The clusters are then validated against established Nepal
vernacular typologies from the literature (external face validity).

CLUSTERING VARIABLES
--------------------
Eight raw structural variables — chosen because they:
  (a) are fully or nearly fully observed (no MNAR contamination in input)
  (b) structurally characterise the building typology
  (c) correspond directly to typology-defining dimensions in the Nepal
      vernacular architecture literature (Rijal, 2018; Kandel et al., 2024)

  Variable          Survey source    Encoding
  ----------------  ---------------  --------------------------------
  Wall material     SAW01            One-hot (6 groups, see below)
  Roof material     SAR04            One-hot (4 groups)
  Roof shape        SAR01            One-hot (3 groups)
  Plan shape        SAL05            Binary (regular vs irregular)
  Number of floors  NumFloors        Numeric (1, 2, 3)
  L/W ratio         SAM01/02         Numeric (continuous)
  Elevated/stilts   SAF02            Binary (elevated=1 vs not=0)
  Adjacency         SAL02A           Binary (detached=1 vs attached=0)

EXTERNAL VALIDATION REFERENCE TYPOLOGIES
-----------------------------------------
From Rijal (2018, Chapter 6) and Kandel et al. (2024), Nepal's vernacular
building stock organises into recognisable material-floor combinations:

  Terai (subtropical, ~150 m):
    Wall: mud / timber / bamboo  |  Roof: thatch or GI  |  Floors: 1
    Elevated on plinth  |  Semi-open veranda  |  Scattered settlement

  Hill-temperate (Dhading, Kaski, ~1500-1700 m):
    Wall: stone  |  Roof: slate or CGI  |  Floors: 2-3
    Thick walls (0.45 m)  |  South-facing  |  Veranda + balcony

  Hill-urban (Bhaktapur, ~1350 m):
    Wall: brick  |  Roof: tile  |  Floors: 4-5
    Compact courtyard  |  Dense row settlement

  Cool-mountain (Solukhumbu, ~2600 m):
    Wall: stone  |  Roof: stone slate  |  Floors: 2-3
    Compact attached  |  Small openings

Lumbini Province spans the Terai-to-Hill transition, so clusters
dominated by Terai and Hill types are expected. If a cluster does not
correspond to any of the above literature types, it may represent a
transitional or modernised typology (CGI/concrete mixed construction).

MNAR INDICATORS TO BE IMPUTED DOWNSTREAM
-----------------------------------------
After cluster assignment, the following MNAR-flagged indicators are
imputed in stage04_mnar_imputation.py using each building's cluster
modal score:
  EVI: EVI_D5_04 (Roof Fastening), EVI_D3_01 (H/t Ratio),
       EVI_D3_04 (Wall Thickness)
  FVI: FVI3_RoofConn, FVI6_Geom, FVI8_Apron, FVI9_Drain, FVI11_LWratio

OUTPUTS
-------
  typology_clusters.csv            — Original rows + TYPOLOGY_CLUSTER column
  fig01_elbow_silhouette.png       — Elbow + silhouette plot for k selection
  fig02_cluster_profile_heatmap.png — Modal variable per cluster
  fig03_cluster_size_bar.png       — Building count per cluster
  fig04_pca_scatter.png            — PCA projection coloured by cluster
  fig05_validation_comparison.png  — Cluster profiles vs literature types
  typology_cluster_profiles.csv    — Summary table for thesis appendix

DEPENDENCIES
------------
    pip install pandas numpy matplotlib seaborn scikit-learn

REFERENCES
----------
Rijal, H.B. (2018). Nepal: Traditional Houses. In T. Kubota et al. (eds.),
  Sustainable Houses and Living in the Hot-Humid Climates of Asia.
  Springer. https://doi.org/10.1007/978-981-10-8465-2_6

Kandel, S. et al. (2024). Vernacular Architecture in Nepal: A Review on
  Planning and Building Materials. Nepal Engineers' Association Gandaki
  Province Technical Journal, 4.

Hartigan, J.A. & Wong, M.A. (1979). Algorithm AS 136: A K-Means Clustering
  Algorithm. Journal of the Royal Statistical Society Series C, 28(1),
  100-108. https://doi.org/10.2307/2346830

Rousseeuw, P.J. (1987). Silhouettes: A graphical aid to the interpretation
  and validation of cluster analysis. Journal of Computational and Applied
  Mathematics, 20, 53-65. https://doi.org/10.1016/0377-0427(87)90125-7
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# FILE PATHS
# =============================================================================

HERE       = Path(__file__).resolve().parent
INPUT_CSV  = HERE / "20263003_Nepal_Lumbini_data.csv"
OUTPUT_DIR = HERE / "outputs" / "typology_clusters"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

# K range to evaluate. Based on the literature (Terai / Hill-temperate /
# Hill-urban / Cool-mountain / Transitional-modern), a range of 3-9 is
# appropriate. The elbow and silhouette plots will guide final selection.
K_MIN = 3
K_MAX = 9

# Minimum cluster size. Clusters smaller than this will be flagged —
# very small clusters indicate either outliers or over-segmentation.
MIN_CLUSTER_SIZE = 50

# Random seed for reproducibility (FAIR principle).
RANDOM_STATE = 42

# =============================================================================
# LOAD RAW DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
# ── Normalise ward name inconsistency ─────────────────────────────────────────
# GD005_VDC_name contains 'tulsipur-ward_14' (hyphen) in the raw survey CSV.
# Normalise to underscore here to match ward_information.csv and stage01/02 output.
_WARD_COL = "GD005_VDC_name"
if _WARD_COL in df.columns:
    _before = df[_WARD_COL].astype(str).eq("tulsipur-ward_14").sum()
    df[_WARD_COL] = (
        df[_WARD_COL].astype(str)
        .str.replace("tulsipur-ward_14", "tulsipur_ward_14", regex=False)
        .where(df[_WARD_COL].notna(), other=np.nan)
    )
    if _before:
        print(f"  Normalised {_before} rows: 'tulsipur-ward_14' → 'tulsipur_ward_14'")

df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_num(col: str) -> pd.Series:
    """Coerce column to numeric; blanks and spaces become NaN."""
    return pd.to_numeric(
        df[col].replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )

def flag(col: str) -> pd.Series:
    """Return boolean Series: True where binary dummy column equals 1.
    Safely returns False for absent columns."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return to_num(col) == 1

# =============================================================================
# STEP 1 — CONSTRUCT CLUSTERING FEATURE MATRIX
# =============================================================================
# All features are derived from raw survey columns. Features are computed
# here independently from stage02_evi_scoring.py and stage01_fvi_scoring.py to keep
# this script self-contained (FAIR: reproducibility).
#
# Feature design follows the typology dimensions identified in:
#   Rijal (2018): wall material, floors, roof material, spatial organisation
#   Kandel et al. (2024): wall material, roof material, floors, building form
# =============================================================================

print("\n" + "=" * 65)
print("STEP 1 — CONSTRUCTING FEATURE MATRIX")
print("=" * 65)

features = pd.DataFrame(index=df.index)

# ── Wall material (one-hot, 6 groups) ────────────────────────────────────────
# Groups reflect structural performance distinctions from IS 1905 / NBC 203
# and the vernacular typology literature.
# Group 1: Concrete / RC (modern, strong)
# Group 2: Stone masonry (traditional hill/mountain type)
# Group 3: Brick (traditional urban hill type — Bhaktapur)
# Group 4: Mud / compressed earth / mud-straw (traditional Terai type)
# Group 5: Bamboo / timber (light, traditional Terai/transitional)
# Group 6: CGI sheet / tarpaulin / other (non-structural / modern transitional)
features["wall_concrete"] = flag("SAW01_What_material_is_used_for_the_concrete").astype(float)
features["wall_stone"]    = flag("SAW01_What_material_is_used_for_the_stone").astype(float)
features["wall_brick"]    = (
    flag("SAW01_What_material_is_used_for_the_compressed_earth")
    | flag("SAW01_What_material_is_used_for_the_comp_earth_nonstabl")
).astype(float)
features["wall_mud"]      = (
    flag("SAW01_What_material_is_used_for_the_mud")
    | flag("SAW01_What_material_is_used_for_the_mud_straw")
).astype(float)
features["wall_bamboo"]   = (
    flag("SAW01_What_material_is_used_for_the_bamboo")
    | flag("SAW01_What_material_is_used_for_the_wood")
).astype(float)
features["wall_other"]    = (
    flag("SAW01_What_material_is_used_for_the_cgi_sheet")
    | flag("SAW01_What_material_is_used_for_the_tarpaulin")
    | flag("SAW01_What_material_is_used_for_the_other")
    | flag("SAW01_What_material_is_used_for_the_i_cannot_see")
    | flag("SAW01_What_material_is_used_for_the_the_wall_is_plastered")
).astype(float)

# ── Roof material (one-hot, 4 groups) ────────────────────────────────────────
# Concrete slab / flat roof  → heavy, modern
# Tile (stone, clay, concrete) → traditional hill
# CGI sheet / corrugated iron → modern transitional (most common in rural Nepal)
# Thatch / mud / wood         → traditional Terai / mountain
features["roof_concrete"]  = flag("SAR04_What_roofing_material_is_used_concrete").astype(float)
features["roof_tile"]      = (
    flag("SAR04_What_roofing_material_is_used_stone_roofing_tiles")
    | flag("SAR04_What_roofing_material_is_used_clay_roofing_tiles")
    | flag("SAR04_What_roofing_material_is_used_concrete_roofing_tiles")
).astype(float)
features["roof_cgi"]       = flag("SAR04_What_roofing_material_is_used_cgi_sheet").astype(float)
features["roof_organic"]   = (
    flag("SAR04_What_roofing_material_is_used_thatched_mud")
    | flag("SAR04_What_roofing_material_is_used_thatched__covered_with_mud")
    | flag("SAR04_What_roofing_material_is_used_wood")
).astype(float)

# ── Roof shape (one-hot, 3 groups) ───────────────────────────────────────────
# Pitched symmetric (gable/hip/bonnet/pyramid) → traditional sloped
# Shed / asymmetric                             → transitional
# Flat                                          → modern / mountain dry zone
features["roof_pitched"]   = (
    flag("SAR01_What_is_the_type_of_the_roofgable_roof")
    | flag("SAR01_What_is_the_type_of_the_roofhip_roof")
    | flag("SAR01_What_is_the_type_of_the_roofbonnet_roof")
    | flag("SAR01_What_is_the_type_of_the_roofpyramid_hip_roof")
).astype(float)
features["roof_shed"]      = flag("SAR01_What_is_the_type_of_the_roofshed_roof").astype(float)
features["roof_flat"]      = flag("SAR01_What_is_the_type_of_the_roofflat_roof").astype(float)

# ── Plan shape (binary) ───────────────────────────────────────────────────────
# Regular (rectangle/square)=1, Irregular (L/T/U/other)=0
features["plan_regular"]   = (
    flag("SAL05_rectangle") | flag("SAL05_square")
).astype(float)

# ── Number of floors (numeric 1–3+) ──────────────────────────────────────────
features["num_floors"]     = (
    flag("SAL04_What_parts_of_the_building_are1st_floor").astype(int)
    + flag("SAL04_What_parts_of_the_building_are2nd_floor").astype(int)
    + flag("SAL04_What_parts_of_the_building_are3rd_floor").astype(int)
).clip(lower=1).astype(float)   # clip: at least 1 floor

# ── L/W ratio (numeric) ───────────────────────────────────────────────────────
_w = to_num("SAM02_What_is_the_width_of_the_core_house")
_l = to_num("SAM01_What_is_the_length_of_the_core_house")
features["lw_ratio"]       = np.where(_w.notna() & (_w > 0), _l / _w, np.nan)
# Winsorise at 99th percentile to reduce influence of extreme outliers.
_lw_p99 = features["lw_ratio"].quantile(0.99)
features["lw_ratio"]       = features["lw_ratio"].clip(upper=_lw_p99)

# ── Elevated / stilts (binary) ────────────────────────────────────────────────
# Elevated or on plinth=1 (Terai type), on ground=0
_raw_stilt = df["SAF02_Is_the_house_build_on_sti"].replace(" ", np.nan).infer_objects(copy=False)
features["elevated"]       = _raw_stilt.isin(
    ["Yes", "yes", "yes_it_is_build_on_plinth"]
).astype(float)

# ── Adjacency (binary) ───────────────────────────────────────────────────────
# Fully detached=1 (scattered Terai/rural), attached/within 3m=0 (urban/hill)
features["detached"]       = flag(
    "SAL02A_How_is_the_location_of_the_buifully_detached"
).astype(float)

# ── Report feature matrix ─────────────────────────────────────────────────────
print(f"\nFeature matrix shape: {features.shape}")
print(f"Features: {list(features.columns)}")
missing_pct = features.isna().mean() * 100
print("\nMissing % per feature (must be low — these are clustering inputs):")
print(missing_pct.round(1).to_string())

# =============================================================================
# STEP 2 — HANDLE MISSING VALUES IN FEATURES
# =============================================================================
# For one-hot encoded binary features, NaN means the relevant dummy was
# absent — treat as 0 (not that material/type). For numeric features
# (num_floors, lw_ratio), fill with the column median.
#
# This fill is for clustering ONLY and is not propagated to the survey data.
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2 — HANDLING MISSING VALUES IN FEATURES")
print("=" * 65)

features_filled = features.copy()

# Binary / one-hot features: NaN → 0
binary_cols = [c for c in features.columns if c not in ("num_floors", "lw_ratio")]
features_filled[binary_cols] = features_filled[binary_cols].fillna(0)

# Numeric features: NaN → median
for col in ("num_floors", "lw_ratio"):
    median_val = features_filled[col].median()
    n_filled   = features_filled[col].isna().sum()
    features_filled[col] = features_filled[col].fillna(median_val)
    print(f"  {col:<15}: filled {n_filled:,} NaN with median={median_val:.2f}")

print(f"\n  Feature matrix after fill: {features_filled.shape}, "
      f"NaN remaining: {features_filled.isna().sum().sum()}")

# =============================================================================
# STEP 3 — STANDARDISE FEATURES
# =============================================================================
# All features are scaled to zero mean and unit variance before k-means so
# that numeric features (num_floors, lw_ratio) do not dominate the distance
# metric relative to the binary one-hot features.
#
# Reference: Hartigan & Wong (1979). K-Means uses Euclidean distance, which
# is sensitive to scale differences between features.
# =============================================================================

scaler         = StandardScaler()
features_scaled = scaler.fit_transform(features_filled)

# =============================================================================
# STEP 4 — FIND OPTIMAL K: ELBOW + SILHOUETTE
# =============================================================================
# Two complementary criteria:
#
# Elbow (inertia): within-cluster sum of squared distances. The "elbow"
#   is the k at which the marginal gain in reducing inertia diminishes.
#   This is a visual heuristic — look for the kink in the curve.
#
# Silhouette score: measures how similar each point is to its own cluster
#   compared to the nearest other cluster. Range [-1, 1]; higher is better.
#   The k that maximises silhouette is the most compact and well-separated
#   clustering.
#
# Both criteria should be considered together. If they agree, choose that k.
# If they disagree, prefer the silhouette score (more rigorous) but consider
# whether the elbow k produces more interpretable clusters.
#
# Reference: Rousseeuw (1987). Silhouettes.
# =============================================================================

print("\n" + "=" * 65)
print("STEP 4 — ELBOW AND SILHOUETTE ANALYSIS")
print("=" * 65)

k_values    = range(K_MIN, K_MAX + 1)
inertias    = []
silhouettes = []

for k in k_values:
    km   = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    lbls = km.fit_predict(features_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(features_scaled, lbls, sample_size=min(2000, len(df)),
                           random_state=RANDOM_STATE)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:>10,.1f}  silhouette={sil:.4f}")

# Elbow: compute second derivative of inertia to find the kink.
inertia_arr = np.array(inertias)
d1          = np.diff(inertia_arr)
d2          = np.diff(d1)
elbow_k     = list(k_values)[np.argmin(d2) + 1]   # offset for two diffs
sil_k       = list(k_values)[np.argmax(silhouettes)]
print(f"\n  Elbow heuristic suggests  k = {elbow_k}")
print(f"  Silhouette maximum at     k = {sil_k}")

# ── Plot elbow + silhouette ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(list(k_values), inertias, "o-", color="#4C72B0", linewidth=2)
ax1.axvline(elbow_k, color="#d62728", linestyle="--", linewidth=1.2,
            label=f"Elbow k={elbow_k}")
ax1.set_xlabel("Number of clusters (k)", fontsize=11)
ax1.set_ylabel("Within-cluster sum of squares (inertia)", fontsize=11)
ax1.set_title("Within-Cluster Inertia by Number of Clusters (k)\n"
              "Kink in the curve indicates the optimal k", fontsize=11)
ax1.legend(fontsize=9)
ax1.set_xticks(list(k_values))

ax2.plot(list(k_values), silhouettes, "s-", color="#2ca02c", linewidth=2)
ax2.axvline(sil_k, color="#d62728", linestyle="--", linewidth=1.2,
            label=f"Max silhouette k={sil_k}")
ax2.set_xlabel("Number of clusters (k)", fontsize=11)
ax2.set_ylabel("Average silhouette score", fontsize=11)
ax2.set_title("Average Silhouette Score by Number of Clusters (k)\n"
              "Higher score = more compact and better-separated clusters", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_xticks(list(k_values))

plt.suptitle("Optimal Number of Clusters: Elbow and Silhouette Criteria\n"
             "Nepal Lumbini Building Stock (n=2,993 buildings)", fontsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig01_elbow_silhouette.png", dpi=150)
plt.close()
print(f"\n  Saved: fig01_elbow_silhouette.png")

# =============================================================================
# STEP 5 — FIT FINAL CLUSTERING
# =============================================================================
# Choose k based on the elbow/silhouette results. The default below uses
# the silhouette-optimal k, which is the more rigorous criterion.
# Override K_FINAL manually if domain knowledge or the elbow suggests
# a different value provides more interpretable typology classes.
#
# n_init=50: runs the algorithm 50 times with different centroid seeds and
# keeps the best result. This reduces sensitivity to initialisation.
# =============================================================================

# K_FINAL: set to the silhouette-optimal k by default.
# If elbow and silhouette disagree, inspect the cluster profiles in Step 6
# and override this value before committing to the final imputation.
K_FINAL = sil_k
print("\n" + "=" * 65)
print(f"STEP 5 — FITTING FINAL CLUSTERING (k={K_FINAL})")
print("=" * 65)
print(f"  NOTE: Override K_FINAL in the script if domain inspection")
print(f"  suggests a different k produces more interpretable typologies.")

km_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE,
                  n_init=50, max_iter=1000)
cluster_labels = km_final.fit_predict(features_scaled)

# Attach to main dataframe
df["TYPOLOGY_CLUSTER"] = cluster_labels

# Report cluster sizes
print(f"\n  Cluster sizes (k={K_FINAL}):")
for c, n in sorted(
    zip(*np.unique(cluster_labels, return_counts=True)),
    key=lambda x: -x[1]
):
    flag_str = " *** SMALL — check for outliers" if n < MIN_CLUSTER_SIZE else ""
    print(f"    Cluster {c}: n={n:>5,} ({n/len(df)*100:.1f}%){flag_str}")

# =============================================================================
# STEP 6 — PROFILE CLUSTERS
# =============================================================================
# For each cluster, compute:
#   - Modal value of each one-hot feature (most common category)
#   - Mean value of numeric features (num_floors, lw_ratio)
#   - Proportion of buildings with each binary flag (elevated, detached)
#
# The profile is the basis for two things:
#   1. Assigning a typology label (validated against literature)
#   2. Computing the modal vulnerability score for MNAR imputation
# =============================================================================

print("\n" + "=" * 65)
print("STEP 6 — CLUSTER PROFILES")
print("=" * 65)

# Compute mean of each feature per cluster (for one-hot features,
# the mean = proportion of buildings with that category = 1)
profile = features_filled.copy()
profile["CLUSTER"] = cluster_labels
cluster_means = profile.groupby("CLUSTER").mean().round(3)

print("\nCluster profile (mean per feature; "
      "one-hot features = proportion with value=1):")
print(cluster_means.to_string())

# ── Assign typology labels based on dominant material per cluster ─────────────
# Wall: pick the one-hot wall feature with highest mean → dominant material
# Roof: pick the roof feature with highest mean
# Floors: round mean num_floors

wall_cols  = ["wall_concrete", "wall_stone", "wall_brick",
              "wall_mud",      "wall_bamboo", "wall_other"]
roof_cols  = ["roof_concrete", "roof_tile", "roof_cgi", "roof_organic"]
shape_cols = ["roof_pitched",  "roof_shed",  "roof_flat"]

WALL_LABELS  = {
    "wall_concrete": "Concrete",
    "wall_stone":    "Stone",
    "wall_brick":    "Earth/Brick",
    "wall_mud":      "Mud",
    "wall_bamboo":   "Bamboo/Timber",
    "wall_other":    "Other/Mixed"
}
ROOF_MAT_LABELS = {
    "roof_concrete": "Concrete slab",
    "roof_tile":     "Tile",
    "roof_cgi":      "CGI sheet",
    "roof_organic":  "Thatch/Organic"
}
ROOF_SHAPE_LABELS = {
    "roof_pitched":  "Pitched",
    "roof_shed":     "Shed",
    "roof_flat":     "Flat"
}

typology_labels = {}
profile_rows    = []

for c in range(K_FINAL):
    row   = cluster_means.loc[c]
    wall  = WALL_LABELS[row[wall_cols].idxmax()]
    rmat  = ROOF_MAT_LABELS[row[roof_cols].idxmax()]
    rshp  = ROOF_SHAPE_LABELS[row[shape_cols].idxmax()]
    flrs  = round(row["num_floors"])
    lw    = round(row["lw_ratio"], 2)
    elev  = "Elevated" if row["elevated"] > 0.5 else "On-ground"
    dtch  = "Detached" if row["detached"] > 0.5 else "Attached"
    n     = (cluster_labels == c).sum()

    # Five-dimension label: wall · roof material · roof shape · floors · adjacency
    # Using all five avoids collisions when wall+shape+floors alone are identical
    # across clusters (e.g. two Stone/Flat/1F clusters differing only in roof
    # material or building placement).
    label = f"C{c}: {wall} · {rmat} · {rshp} · {flrs}F · {dtch}"
    typology_labels[c] = label
    profile_rows.append({
        "Cluster":        c,
        "n":              n,
        "pct":            round(n / len(df) * 100, 1),
        "Dominant wall":  wall,
        "Dominant roof":  rmat,
        "Roof shape":     rshp,
        "Avg floors":     round(row["num_floors"], 2),
        "Avg L/W":        lw,
        "Elevated (%)":   round(row["elevated"] * 100, 1),
        "Detached (%)":   round(row["detached"] * 100, 1),
        "Auto label":     label
    })
    print(f"\n  Cluster {c} (n={n:,}, {n/len(df)*100:.1f}%):")
    print(f"    Wall: {wall}  |  Roof mat: {rmat}  |  Roof shape: {rshp}")
    print(f"    Avg floors: {row['num_floors']:.2f}  |  "
          f"Avg L/W: {lw}  |  Elevated: {row['elevated']*100:.0f}%  |  "
          f"Detached: {row['detached']*100:.0f}%")
    print(f"    Auto label: {label}")

# Save profile table
profile_df = pd.DataFrame(profile_rows)
profile_df.to_csv(OUTPUT_DIR / "typology_cluster_profiles.csv", index=False)
print(f"\n  Saved: typology_cluster_profiles.csv")

# =============================================================================
# STEP 7 — VISUALISATIONS
# =============================================================================

# ── Fig 2: Cluster profile heatmap ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, max(4, K_FINAL * 0.7)))
sns.heatmap(
    cluster_means,
    annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.4, ax=ax,
    vmin=0, vmax=1,
    annot_kws={"size": 8},
    yticklabels=[typology_labels[c] for c in range(K_FINAL)]
)
ax.set_title(
    "Building Typology Cluster Profiles — Mean Feature Values\n"
    "One-hot encoded features shown as proportion; numeric features as normalised mean\n"
    "Yellow = low value   |   Red = high value",
    fontsize=11, pad=10
)
ax.set_xlabel("Feature", fontsize=10)
ax.set_ylabel("Cluster", fontsize=10)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig02_cluster_profile_heatmap.png", dpi=150)
plt.close()
print("  Saved: fig02_cluster_profile_heatmap.png")

# ── Fig 3: Cluster size bar chart ─────────────────────────────────────────────
sizes      = [(cluster_labels == c).sum() for c in range(K_FINAL)]
fig, ax    = plt.subplots(figsize=(max(8, K_FINAL * 1.5), 5))
colours    = plt.cm.Set2(np.linspace(0, 1, K_FINAL))
bars       = ax.bar(
    [typology_labels[c] for c in range(K_FINAL)],
    sizes, color=colours, edgecolor="white"
)
for bar, n in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"n={n:,}\n({n/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Number of buildings", fontsize=11)
ax.set_title(
    f"Number of Buildings per Typology Cluster (n={len(df):,} total, k={K_FINAL} clusters)\n"
    "Bar height shows count; percentage of total shown above each bar",
    fontsize=11, pad=10
)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig03_cluster_size_bar.png", dpi=150)
plt.close()
print("  Saved: fig03_cluster_size_bar.png")

# ── Fig 4: PCA scatter ────────────────────────────────────────────────────────
# Reduce to 2 dimensions for visual inspection. PCA is not used for
# clustering — it is only used for this diagnostic visualisation.
pca        = PCA(n_components=2, random_state=RANDOM_STATE)
coords     = pca.fit_transform(features_scaled)
var1, var2 = pca.explained_variance_ratio_ * 100

fig, ax    = plt.subplots(figsize=(9, 7))
colours    = plt.cm.Set1(np.linspace(0, 0.9, K_FINAL))
for c in range(K_FINAL):
    mask = cluster_labels == c
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=[colours[c]], label=typology_labels[c],
        alpha=0.4, s=12, edgecolors="none"
    )
ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
ax.set_title(
    "PCA Projection of Building Feature Space — Cluster Assignments\n"
    "Two principal components shown for visualisation only; clustering used all features",
    fontsize=11, pad=10
)
ax.legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig04_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig04_pca_scatter.png")

# ── Fig 5: External validation comparison ─────────────────────────────────────
# Compare cluster profiles against the four reference typologies from
# Rijal (2018) and Kandel et al. (2024). For each reference typology,
# encode the expected material/form combinations as binary vectors on the
# same feature set, then display as a side-by-side heatmap.

# Reference typologies encoded as binary feature vectors
# (1 = dominant/expected, 0 = not dominant/expected for this typology)
ref_typologies = {
    "Terai traditional\n(Rijal 2018: Banke)": {
        "wall_concrete": 0, "wall_stone": 0, "wall_brick": 0,
        "wall_mud": 1, "wall_bamboo": 1, "wall_other": 0,
        "roof_concrete": 0, "roof_tile": 0,
        "roof_cgi": 1, "roof_organic": 1,
        "roof_pitched": 1, "roof_shed": 0, "roof_flat": 0,
        "plan_regular": 1, "num_floors": 1.0, "lw_ratio": 1.5,
        "elevated": 1, "detached": 1
    },
    "Hill temperate\n(Rijal 2018: Dhading/Kaski)": {
        "wall_concrete": 0, "wall_stone": 1, "wall_brick": 0,
        "wall_mud": 0, "wall_bamboo": 0, "wall_other": 0,
        "roof_concrete": 0, "roof_tile": 0,
        "roof_cgi": 1, "roof_organic": 0,
        "roof_pitched": 1, "roof_shed": 0, "roof_flat": 0,
        "plan_regular": 1, "num_floors": 2.0, "lw_ratio": 1.5,
        "elevated": 0, "detached": 0
    },
    "Hill urban\n(Rijal 2018: Bhaktapur)": {
        "wall_concrete": 0, "wall_stone": 0, "wall_brick": 1,
        "wall_mud": 0, "wall_bamboo": 0, "wall_other": 0,
        "roof_concrete": 0, "roof_tile": 1,
        "roof_cgi": 0, "roof_organic": 0,
        "roof_pitched": 1, "roof_shed": 0, "roof_flat": 0,
        "plan_regular": 1, "num_floors": 4.0, "lw_ratio": 1.2,
        "elevated": 0, "detached": 0
    },
    "Cool mountain\n(Rijal 2018: Solukhumbu)": {
        "wall_concrete": 0, "wall_stone": 1, "wall_brick": 0,
        "wall_mud": 0, "wall_bamboo": 0, "wall_other": 0,
        "roof_concrete": 0, "roof_tile": 1,
        "roof_cgi": 0, "roof_organic": 0,
        "roof_pitched": 1, "roof_shed": 0, "roof_flat": 0,
        "plan_regular": 1, "num_floors": 2.5, "lw_ratio": 1.3,
        "elevated": 0, "detached": 0
    }
}

ref_df = pd.DataFrame(ref_typologies).T[features_filled.columns]

# Normalise numeric columns to 0-1 for visual comparison
_norm_cols = ["num_floors", "lw_ratio"]
for col in _norm_cols:
    col_max = max(ref_df[col].max(), cluster_means[col].max())
    if col_max > 0:
        ref_df[col]          = ref_df[col]          / col_max
        cluster_means[col]   = cluster_means[col]   / col_max

# Stack reference and cluster profiles
combined = pd.concat([
    cluster_means.rename(index=typology_labels),
    ref_df
])

fig, ax = plt.subplots(figsize=(16, max(6, len(combined) * 0.7)))
sns.heatmap(
    combined, annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.4, ax=ax, vmin=0, vmax=1,
    annot_kws={"size": 7}
)

# Draw a horizontal line separating data clusters from reference typologies
ax.axhline(K_FINAL, color="black", linewidth=2)
ax.set_title(
    "Data-Driven Cluster Profiles vs Literature Reference Typologies\n"
    "Top rows: identified clusters  |  Bottom rows: Rijal (2018) reference typologies\n"
    "Strong similarity between a cluster and a reference row indicates face validity",
    fontsize=11, pad=10
)
ax.set_xlabel("Feature (normalised)", fontsize=10)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig05_validation_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig05_validation_comparison.png")

# =============================================================================
# STEP 8 — SAVE OUTPUT CSV
# =============================================================================
# The TYPOLOGY_CLUSTER column is the primary output consumed by
# stage04_mnar_imputation.py. All original survey columns are preserved.
# =============================================================================

output_csv = HERE / "outputs" / "typology_clusters.csv"
df.to_csv(output_csv, index=False)
print(f"\n  Saved: typology_clusters.csv  "
      f"({len(df):,} rows, TYPOLOGY_CLUSTER column added)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("CLUSTER ANALYSIS SUMMARY")
print("=" * 65)
print(f"\n  Optimal k (silhouette): {sil_k}")
print(f"  Elbow heuristic k     : {elbow_k}")
print(f"  K used for final fit  : {K_FINAL}")
print(f"\n  Cluster profiles saved to: typology_cluster_profiles.csv")
print(f"  Figures saved to         : {OUTPUT_DIR}")

print(f"""
INTERPRETATION GUIDE
--------------------
  1. Inspect fig02_cluster_profile_heatmap.png.
     Assign a human-readable typology label to each cluster based on its
     dominant wall material, roof type, and floor count.

  2. Inspect fig05_validation_comparison.png.
     Each data cluster should correspond reasonably to one of the four
     reference typologies from Rijal (2018) and Kandel et al. (2024).
     If a cluster has no clear literature match, it likely represents a
     transitional/modernised building type (e.g. concrete-frame + CGI).
     This is still a valid donor class for imputation.

  3. If two clusters have nearly identical profiles (heatmap rows look
     the same), consider reducing K_FINAL by 1.

  4. If a cluster is flagged as SMALL (n < {MIN_CLUSTER_SIZE}), it may represent
     outliers. Consider merging it with the nearest cluster.

  5. Document the final typology labels and their literature correspondence
     in your methods section (FAIR principle — transparency).

  6. Pass typology_clusters.csv to stage04_mnar_imputation.py, which
     reads the TYPOLOGY_CLUSTER column and imputes MNAR indicators using
     each cluster's modal vulnerability score.

REFERENCES
----------
  Rijal (2018)        : https://doi.org/10.1007/978-981-10-8465-2_6
  Kandel et al.(2024) : Nepal Engineers' Association Gandaki Province
                        Technical Journal, Vol 4
  Hartigan & Wong (1979): https://doi.org/10.2307/2346830
  Rousseeuw (1987)    : https://doi.org/10.1016/0377-0427(87)90125-7
""")

print(f"All outputs saved to: {OUTPUT_DIR}")
