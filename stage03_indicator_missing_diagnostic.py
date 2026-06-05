"""
missing_data_diagnostic.py
===========================
Unified Missing Data Mechanism Diagnostic — EVI and FVI Indicators
Nepal Lumbini Survey Dataset

PURPOSE
-------
This script diagnoses whether missing data in the EVI and FVI indicator
columns are MCAR, MAR, or MNAR — and identifies whether missingness
clusters by surveyor, community, or time period.

This distinction matters because:
  - Surveyor clustering → enumerator behaviour drives missingness (MAR)
  - Community clustering → building type or physical access drives missingness
  - Temporal clustering → survey fatigue or protocol drift
  All three produce MAR signals but require different reporting and have
  different implications for which variables to include in MICE imputation.

KEY FINDINGS FROM EXPLORATORY ANALYSIS (Nepal Lumbini, n=2,993)
-----------------------------------------------------------------
  EVI D5_04 Roof Fasten: 47.2% overall missing
    Enumerator range: 31% (enumerator_7) to 58% (enumerator_2)
    → Strong enumerator effect; not purely building inaccessibility

  EVI D3_04 Wall Thick: ~35% overall missing
    enumerator_8: 15.2% missing vs 0.5–1.5% for most others
    → Individual enumerator behaviour, not building type alone

  FVI3 Roof Conn: 44.5% overall missing
    All enumerators show elevated missingness (31–58%)
    → More uniform pattern; physical inaccessibility across all areas

FIVE DIAGNOSTIC STEPS
----------------------
  Step 1 — Visualise missingness patterns (bar chart + heatmaps)
  Step 2 — Little's MCAR test (formal statistical test)
  Step 3 — MAR test via logistic regression (cross-indicator, AUC)
  Step 4 — MNAR sensitivity analysis (worst/best-case imputation)
  Step 5 — Clustering analysis (surveyor / community / time)

OUTPUTS — 8 figures per index (16 total)
-----------------------------------------
  fig1[evi/fvi]_missing_rate.png          Per-indicator missing rate
  fig2[evi/fvi]_row_heatmap.png           Row-level missingness pattern
  fig3[evi/fvi]_cooccurrence.png          Co-occurrence heatmap
  fig4[evi/fvi]_mar_auc.png               MAR logistic regression AUC
  fig5[evi/fvi]_mnar_sensitivity.png      MNAR sensitivity distributions
  fig6[evi/fvi]_surveyor_clustering.png   Missingness % by enumerator
  fig7[evi/fvi]_community_clustering.png  Missingness % by VDC/ward
  fig8[evi/fvi]_temporal_clustering.png   Daily missing rate over time

DEPENDENCIES
------------
    pip install pandas numpy matplotlib seaborn scipy scikit-learn pyampute

REFERENCES
----------
Little, R.J.A. (1988). A test of missing completely at random for
  multivariate data with missing values. JASA, 83(404), 1198-1202.
  https://doi.org/10.2307/2290157

Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
  epidemiological and clinical research. BMJ, 338, b2393.
  https://doi.org/10.1136/bmj.b2393

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
  CRC Press. https://stefvanbuuren.name/fimd/

Groves, R.M. (2006). Nonresponse rates and nonresponse bias in
  household surveys. Public Opinion Quarterly, 70(5), 646-675.
  https://doi.org/10.1093/poq/nfl033

West, B.T. & Olson, K. (2010). How much of interviewer variance is
  really nonresponse error variance? Public Opinion Quarterly, 74(5),
  1004-1026. https://doi.org/10.1093/poq/nfq061
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from pyampute.exploration.mcar_statistical_tests import MCARTest
    PYAMPUTE_AVAILABLE = True
except ImportError:
    PYAMPUTE_AVAILABLE = False
    print(
        "WARNING: pyampute not installed — Little's MCAR test will be skipped.\n"
        "Install with: pip install pyampute\n"
    )

# =============================================================================
# FILE PATHS
# =============================================================================

INPUT_CSV  = Path(__file__).resolve().parent / "20263003_Nepal_Lumbini_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "missing_data_diagnostic_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# METADATA COLUMNS
# =============================================================================

SURVEYOR_COL  = "GD001_Name_of_the_surveyor"
COMMUNITY_COL = "GD005_VDC_name"
# START_COL: try "start" first (standard ODK column), then known alternatives.
# If none are found, timestamp-dependent steps are skipped gracefully.
_START_CANDIDATES = ["start", "starttime", "start_time",
                     "@_submission_time", "submission_time", "SubmissionDate"]
START_COL = next((c for c in _START_CANDIDATES if c in
                  pd.read_csv(INPUT_CSV, nrows=0,
                              encoding="utf-8-sig").columns), None)

# =============================================================================
# LOAD RAW DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)

# Replace blank/whitespace-only strings with NaN.
# infer_objects(copy=False) opts into the future pandas behaviour and
# suppresses the FutureWarning about silent downcasting after replace().
# See: https://pandas.pydata.org/docs/whatsnew/v2.1.0.html
df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)

# Parse timestamps and attach metadata
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

# Parse timestamps — skip gracefully if the start column is absent.
if START_COL and START_COL in df.columns:
    df["_start_dt"]    = pd.to_datetime(df[START_COL], utc=True, errors="coerce")
    df["_survey_date"] = df["_start_dt"].dt.date
    df["_survey_week"] = df["_start_dt"].dt.isocalendar().week.astype("Int64")
else:
    print(f"  WARNING: start-time column not found (tried {_START_CANDIDATES}). ")
    print("  Temporal clustering analysis will be skipped.")
    df["_start_dt"]    = pd.NaT
    df["_survey_date"] = pd.NaT
    df["_survey_week"] = pd.array([pd.NA] * len(df), dtype="Int64")
df["_surveyor"]    = df[SURVEYOR_COL].fillna("unknown")
df["_community"]   = df[COMMUNITY_COL].fillna("unknown")

print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns")
print(f"Surveyors  : {df['_surveyor'].nunique()} unique "
      f"({df['_surveyor'].value_counts().to_dict()})")
print(f"Communities: {df['_community'].nunique()} unique")
if df["_start_dt"].notna().any():
    print(f"Date range : {df['_start_dt'].min().date()} "
          f"to {df['_start_dt'].max().date()}")
else:
    print("Date range : unknown (start column not found)")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_num(series: pd.Series) -> pd.Series:
    """Coerce to numeric; blanks and spaces become NaN.

    The chained .replace() calls are followed by infer_objects(copy=False)
    to suppress the FutureWarning about silent downcasting introduced in
    pandas 2.1. pd.to_numeric(..., errors='coerce') then handles the rest.
    """
    return pd.to_numeric(
        series.replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )


def flag(col: str) -> pd.Series:
    """Return boolean Series: True where binary dummy column equals 1."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return to_num(df[col]) == 1


# =============================================================================
# RECONSTRUCT EVI INDICATORS FROM RAW SURVEY
# =============================================================================
# Mirrors stage02_evi_scoring.py. NaN = survey response absent or inapplicable.
# =============================================================================

df["NumFloors"] = (
    flag("SAL04_What_parts_of_the_building_are1st_floor").astype(int)
    + flag("SAL04_What_parts_of_the_building_are2nd_floor").astype(int)
    + flag("SAL04_What_parts_of_the_building_are3rd_floor").astype(int)
)
_w = to_num(df["SAM02_What_is_the_width_of_the_core_house"])
_l = to_num(df["SAM01_What_is_the_length_of_the_core_house"])
df["LW_ratio"] = np.where(_w.notna() & (_w > 0), _l / _w, np.nan)

_h = to_num(df["SAM03_What_is_the_heigth_of_the_core_house"])
_t = to_num(df["SAM07_What_is_the_thicknes_the_core_house_wall"])
df["Ht_ratio"] = np.where(_t.notna() & (_t > 0), _h / _t, np.nan)

# Domain 1 — Site & Placement
evi_d1_01 = pd.Series(np.nan, index=df.index)
evi_d1_01[flag("SAL02A_How_is_the_location_of_the_buidirectly_against_other_buil")] = 5
evi_d1_01[flag("SAL02A_How_is_the_location_of_the_buiwithing_3_m_of_another_bu")]   = 3
evi_d1_01[flag("SAL02A_How_is_the_location_of_the_buifully_detached")]               = 1

evi_d1_02 = pd.Series(np.nan, index=df.index)
evi_d1_02[
    flag("SAL02_How_is_the_building_placed_on_no_slope_or_th")
    & flag("SAL02_How_is_the_building_placed_on_no_edge_or_the")
] = 1
evi_d1_02[
    flag("SAL02_How_is_the_building_placed_on_within_3_meter")
    | flag("SAL02_How_is_the_building_placed_on_within_3_meter_1")
    | flag("SAL02_How_is_the_building_placed_on_a_retaining_wa")
] = 5

# Domain 2 — Plan Geometry
evi_d2_01 = pd.Series(np.nan, index=df.index)
evi_d2_01[flag("SAL05_rectangle") | flag("SAL05_square")] = 1
evi_d2_01[
    flag("SAL05_l_shape") | flag("SAL05_t_shape")
    | flag("SAL05_u_shape") | flag("SAL05_no_usual_shape")
] = 5

_lw = pd.to_numeric(df["LW_ratio"], errors="coerce")
evi_d2_02 = pd.Series(np.nan, index=df.index)
evi_d2_02[_lw.le(3.0) & _lw.notna()] = 1
evi_d2_02[_lw.gt(3.0)]               = 5

# Domain 3 — Wall System & Openings
_ht = pd.to_numeric(df["Ht_ratio"], errors="coerce")
_is_stone    = flag("SAW01_What_material_is_used_for_the_stone")
_is_concrete = flag("SAW01_What_material_is_used_for_the_concrete")

evi_d3_01 = pd.Series(np.nan, index=df.index)
evi_d3_01[_is_stone    & _ht.le(8)]  = 1
evi_d3_01[_is_stone    & _ht.gt(8)]  = 5
evi_d3_01[_is_concrete & _ht.le(12)] = 1
evi_d3_01[_is_concrete & _ht.gt(12)] = 5

evi_d3_02 = pd.Series(np.nan, index=df.index)
evi_d3_02[flag("SAM06_What_is_the_total_pe_most_penetrated_wallless_than_50")] = 3
evi_d3_02[flag("SAM06_What_is_the_total_pe_most_penetrated_wallaround_50")]    = 3
evi_d3_02[flag("SAM06_What_is_the_total_pe_most_penetrated_wallmore_than_50")] = 5

df["min_opening_dist"] = df[[
    "SAM04_What_is_the_length_b_and_the_wall_opening",
    "SAM05_What_is_the_length_b_ween_2_wall_openings"
]].apply(pd.to_numeric, errors="coerce").min(axis=1)
_md = pd.to_numeric(df["min_opening_dist"], errors="coerce")
evi_d3_03 = pd.Series(np.nan, index=df.index)
evi_d3_03[_md.ge(0.9)]               = 1
evi_d3_03[_md.ge(0.6) & _md.lt(0.9)] = 3
evi_d3_03[_md.lt(0.6) & _md.notna()] = 5

_tw = pd.to_numeric(df["SAM07_What_is_the_thicknes_the_core_house_wall"], errors="coerce")
evi_d3_04 = pd.Series(np.nan, index=df.index)
evi_d3_04[_is_stone    & _tw.ge(0.35)] = 1
evi_d3_04[_is_stone    & _tw.lt(0.35)] = 5
evi_d3_04[_is_concrete & _tw.ge(0.23)] = 1
evi_d3_04[_is_concrete & _tw.lt(0.23)] = 5

# Domain 4 — Roof System
evi_d4_01 = pd.Series(np.nan, index=df.index)
evi_d4_01[
    flag("SAR01_What_is_the_type_of_the_roofgable_roof")
    | flag("SAR01_What_is_the_type_of_the_roofhip_roof")
    | flag("SAR01_What_is_the_type_of_the_roofbonnet_roof")
    | flag("SAR01_What_is_the_type_of_the_roofpyramid_hip_roof")
] = 1
evi_d4_01[
    flag("SAR01_What_is_the_type_of_the_roofflat_roof")
    | flag("SAR01_What_is_the_type_of_the_roofshed_roof")
    | flag("SAR01_What_is_the_type_of_the_roofother")
] = 5

evi_d4_02 = pd.Series(np.nan, index=df.index)
evi_d4_02[flag("SAR04_What_roofing_material_is_used_cgi_sheet")] = 1
evi_d4_02[
    flag("SAR04_What_roofing_material_is_used_stone_roofing_tiles")
    | flag("SAR04_What_roofing_material_is_used_clay_roofing_tiles")
    | flag("SAR04_What_roofing_material_is_used_concrete_roofing_tiles")
    | flag("SAR04_What_roofing_material_is_used_thatched_mud")
] = 3
evi_d4_02[flag("SAR04_What_roofing_material_is_used_concrete")] = 5

# Domain 5 — Bands & Connectivity
_band_scores = {
    "SAB01_What_is_the_material_of_the_baconcrete":      1,
    "SAB01_What_is_the_material_of_the_bawood":          3,
    "SAB01_What_is_the_material_of_the_babamboo":        3,
    "SAB01_What_is_the_material_of_the_bano_bands_used": 5,
    "SAB01_What_is_the_material_of_the_baother":         5,
    "SAB01_What_is_the_material_of_the_bai_can_t_see":   5,
}
_band_cols = []
for _col, _score in _band_scores.items():
    if _col in df.columns:
        _s = pd.to_numeric(df[_col], errors="coerce")
        _band_cols.append(_s.eq(1).map({True: _score, False: np.nan}))
evi_d5_01 = (
    pd.concat(_band_cols, axis=1).max(axis=1)
    if _band_cols else pd.Series(np.nan, index=df.index)
)

evi_d5_04 = pd.Series(np.nan, index=df.index)
evi_d5_04[
    flag("SARA06_Which_material_is_used_to_fastnails")
    | flag("SARA06_Which_material_is_used_to_fastbolts")
    | flag("SARA06_Which_material_is_used_to_fastdowels")
] = 1
evi_d5_04[
    flag("SARA06_Which_material_is_used_to_fastrope")
    | flag("SARA06_Which_material_is_used_to_fastbeams_top_roofing")
] = 3
evi_d5_04[flag("SARA06_Which_material_is_used_to_fastno_connection")] = 5

df_evi = pd.DataFrame({
    "EVI_D1_01 Adjacency":    evi_d1_01,
    "EVI_D1_02 Slope/Edge":   evi_d1_02,
    "EVI_D2_01 Plan Shape":   evi_d2_01,
    "EVI_D2_02 L/W Ratio":    evi_d2_02,
    "EVI_D3_01 H/t Ratio":    evi_d3_01,
    "EVI_D3_02 Openings %":   evi_d3_02,
    "EVI_D3_03 Opening Dist": evi_d3_03,
    "EVI_D3_04 Wall Thick":   evi_d3_04,
    "EVI_D4_01 Roof Shape":   evi_d4_01,
    "EVI_D4_02 Roof Mass":    evi_d4_02,
    "EVI_D5_01 H.Bands":      evi_d5_01,
    "EVI_D5_04 Roof Fasten":  evi_d5_04,
})

# =============================================================================
# RECONSTRUCT FVI INDICATORS FROM RAW SURVEY
# =============================================================================
# Mirrors stage01_fvi_scoring.py. NaN = absent or inapplicable response.
# =============================================================================

fvi_2 = pd.Series(np.nan, index=df.index)
fvi_2[
    flag("SAR04_What_roofing_material_is_used_concrete")
    | flag("SAR04_What_roofing_material_is_used_cgi_sheet")
    | flag("SAR04_What_roofing_material_is_used_stone_roofing_tiles")
    | flag("SAR04_What_roofing_material_is_used_clay_roofing_tiles")
    | flag("SAR04_What_roofing_material_is_used_concrete_roofing_tiles")
] = 1
fvi_2[
    flag("SAR04_What_roofing_material_is_used_thatched_mud")
    | flag("SAR04_What_roofing_material_is_used_thatched__covered_with_mud")
    | flag("SAR04_What_roofing_material_is_used_other")
    | flag("SAR04_What_roofing_material_is_used_i_cannot_see")
    | flag("SAR04_What_roofing_material_is_used_wood")
] = 5

fvi_3 = pd.Series(np.nan, index=df.index)
fvi_3[
    flag("SARA06_Which_material_is_used_to_fastnails")
    | flag("SARA06_Which_material_is_used_to_fastbeams_top_roofing")
    | flag("SARA06_Which_material_is_used_to_fastbolts")
] = 1
fvi_3[flag("SARA06_Which_material_is_used_to_fastrope")] = 3
fvi_3[
    flag("SARA06_Which_material_is_used_to_fastno_connection")
    | flag("SARA06_Which_material_is_used_to_fastdowels")
    | flag("SARA06_Which_material_is_used_to_fastother")
    | flag("SARA06_Which_material_is_used_to_faststone_laying_o")
    | flag("SARA06_Which_material_is_used_to_fasti_cannot_see")
] = 5

fvi_4 = pd.Series(np.nan, index=df.index)
fvi_4[flag("SAW01_What_material_is_used_for_the_concrete")] = 1
fvi_4[flag("SAW01_What_material_is_used_for_the_stone")]    = 3
fvi_4[
    flag("SAW01_What_material_is_used_for_the_bamboo")
    | flag("SAW01_What_material_is_used_for_the_compressed_earth")
    | flag("SAW01_What_material_is_used_for_the_comp_earth_nonstabl")
    | flag("SAW01_What_material_is_used_for_the_mud")
    | flag("SAW01_What_material_is_used_for_the_mud_straw")
    | flag("SAW01_What_material_is_used_for_the_cgi_sheet")
    | flag("SAW01_What_material_is_used_for_the_tarpaulin")
    | flag("SAW01_What_material_is_used_for_the_wood")
    | flag("SAW01_What_material_is_used_for_the_the_wall_is_plastered")
    | flag("SAW01_What_material_is_used_for_the_other")
    | flag("SAW01_What_material_is_used_for_the_i_cannot_see")
] = 5

_raw5 = df["SAF02_Is_the_house_build_on_sti"].replace(" ", np.nan).infer_objects(copy=False)
fvi_5 = pd.Series(np.nan, index=df.index)
fvi_5[_raw5.isin(["Yes", "yes", "yes_it_is_build_on_plinth"])] = 1
fvi_5[_raw5.isin(["No",  "no",  "Don't_know", "Don't_Know"])] = 5

fvi_6 = pd.Series(np.nan, index=df.index)
fvi_6[flag("SAL05_rectangle") | flag("SAL05_square")] = 1
fvi_6[
    flag("SAL05_l_shape") | flag("SAL05_t_shape")
    | flag("SAL05_u_shape") | flag("SAL05_no_usual_shape")
    | flag("SAL05_other")
] = 5

_raw7 = df["SAR02_How_much_is_th_the_roof_ADD_IMAGE"].replace(" ", np.nan).infer_objects(copy=False)
fvi_7 = pd.Series(np.nan, index=df.index)
fvi_7[_raw7 == "500_600_mm"] = 1
fvi_7[_raw7 == "600_900_mm"] = 3
fvi_7[_raw7.isin([
    "less_than_500_mm", "more_than_900_mm", "none",
    "overhang_size_differs_around_the_house", "i_cannot_see", "other"
])] = 5

_raw8 = df["SAF03_Does_the_house_on_ADD_IMAGES"].replace(" ", np.nan).infer_objects(copy=False)
fvi_8 = pd.Series(np.nan, index=df.index)
fvi_8[_raw8 == "yes"]                           = 1
fvi_8[_raw8 == "yes__around_part_of_the_house"] = 3
fvi_8[_raw8.isin(["no", "i_don_t_know"])]       = 5

_n_drain = (
    flag("SAWA01_Is_there_water_the_site_ADD_IMAGESyes__rain_gutters").astype(int)
    + flag("SAWA01_Is_there_water_the_site_yes__drainage_pipes").astype(int)
    + flag("SAWA01_Is_there_water_the_site_ADD_IMAGESyes__area_slopes_away_f").astype(int)
)
fvi_9 = pd.Series(np.nan, index=df.index)
fvi_9[_n_drain > 0] = 1
fvi_9[
    flag("SAWA01_Is_there_water_the_site_no")
    | flag("SAWA01_Is_there_water_the_site_unknown")
] = 5

_nf = (
    flag("SAL04_What_parts_of_the_building_are1st_floor").astype(int)
    + flag("SAL04_What_parts_of_the_building_are2nd_floor").astype(int)
    + flag("SAL04_What_parts_of_the_building_are3rd_floor").astype(int)
) * 3
fvi_10 = pd.Series(np.nan, index=df.index)
fvi_10[_nf <= 1]               = 1
fvi_10[(_nf > 1) & (_nf <= 3)] = 3
fvi_10[_nf >= 4]               = 5

_lw2 = pd.Series(np.nan, index=df.index)
_w2  = to_num(df["SAM02_What_is_the_width_of_the_core_house"])
_l2  = to_num(df["SAM01_What_is_the_length_of_the_core_house"])
_lw2[_w2.notna() & (_w2 > 0)] = _l2 / _w2
fvi_11 = pd.Series(np.nan, index=df.index)
fvi_11[_lw2 < 3]  = 1
fvi_11[_lw2 >= 3] = 5

df_fvi = pd.DataFrame({
    "FVI2 Roof Mat":   fvi_2,
    "FVI3 Roof Conn":  fvi_3,
    "FVI4 Wall Mat":   fvi_4,
    "FVI5 Elevation":  fvi_5,
    "FVI6 Geometry":   fvi_6,
    "FVI7 Overhang":   fvi_7,
    "FVI8 Apron":      fvi_8,
    "FVI9 Drainage":   fvi_9,
    "FVI10 Floors":    fvi_10,
    "FVI11 L/W Ratio": fvi_11,
})

print(f"\nEVI indicator matrix: {df_evi.shape[0]:,} rows x {df_evi.shape[1]} indicators")
print(f"FVI indicator matrix: {df_fvi.shape[0]:,} rows x {df_fvi.shape[1]} indicators")


# =============================================================================
# STEP 1 — VISUALISE MISSINGNESS PATTERNS
# =============================================================================

def step1_visualise(df_ind: pd.DataFrame, index_label: str, out_dir: Path) -> None:
    """
    Step 1 — Visualise missing data patterns.

    Three figures:
      (a) Bar chart: % missing per indicator with 20% and 40% thresholds.
          Colour: blue (<20%), orange (20–40%), red (>40%).

      (b) Row-level heatmap (buildings × indicators): blue = missing.
          Random scatter → MCAR. Structured bands → MAR or MNAR.

      (c) Co-occurrence heatmap: proportion of buildings where BOTH
          indicators are simultaneously missing. Dark clusters = shared
          causes (enumerator behaviour, survey section skip, physical access).
    """
    n = len(df_ind)

    # (a) Missing rate bar chart
    miss_pct    = df_ind.isna().mean() * 100
    miss_sorted = miss_pct.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#d62728" if p > 40 else "#ff7f0e" if p > 20 else "#4C72B0"
               for p in miss_sorted.values]
    bars = ax.bar(miss_sorted.index, miss_sorted.values,
                  color=colours, edgecolor="white")
    ax.axhline(20, color="#ff7f0e", linestyle="--", linewidth=1.2,
               label="20% caution threshold")
    ax.axhline(40, color="#d62728", linestyle="--", linewidth=1.2,
               label="40% imputation reliability limit")
    ax.set_ylabel("% Missing", fontsize=11)
    ax.set_title(
        f"Missing Data Rate per {index_label} Indicator\n"
        f"n={n:,} buildings | Red > 40% missing | Orange > 20% missing",
        fontsize=11, pad=10
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    for bar, pct in zip(bars, miss_sorted.values):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig1{index_label.lower()}_missing_rate.png", dpi=150)
    plt.close()
    print(f"  Saved: fig1{index_label.lower()}_missing_rate.png")

    # (b) Row-level heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        df_ind.isna().astype(int).T,
        cbar=False, cmap="Blues", ax=ax, xticklabels=False
    )
    ax.set_title(
        f"Row-Level Missingness Pattern — {index_label} Indicators\n"
        "Blue = Missing | Each column = one building\n"
        "Random scatter indicates MCAR; structured bands indicate MAR or MNAR",
        fontsize=11, pad=10
    )
    ax.set_ylabel("Indicator", fontsize=10)
    ax.set_xlabel(f"Buildings (n={n:,}, ordered by survey row)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig2{index_label.lower()}_row_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: fig2{index_label.lower()}_row_heatmap.png")

    # (c) Co-occurrence heatmap — correctly computed on isna() booleans.
    # Both sides must be cast to float (not int) before the dot product.
    # Using bool.T.dot(int) produces an object-dtype result in pandas when
    # the DataFrame has mixed-type columns, which matplotlib cannot render.
    # Casting both sides to float64 guarantees a numeric result regardless
    # of the source column dtypes.
    miss_bool = df_ind.isna().astype(float)
    co_occur  = miss_bool.T.dot(miss_bool) / n
    vmax      = max(float(co_occur.values.max()), 0.01)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(co_occur, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, vmin=0, vmax=vmax)
    ax.set_title(
        f"Co-occurrence of Missingness — {index_label} Indicators\n"
        "Cell value = proportion of buildings where both indicators are simultaneously missing\n"
        "Dark clusters indicate shared survey section, enumerator effect, or physical access barriers",
        fontsize=11, pad=10
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"fig3{index_label.lower()}_cooccurrence.png", dpi=150)
    plt.close()
    print(f"  Saved: fig3{index_label.lower()}_cooccurrence.png")


# =============================================================================
# STEP 2 — LITTLE'S MCAR TEST
# =============================================================================

def step2_little_mcar(df_ind: pd.DataFrame, index_label: str) -> None:
    """
    Step 2 — Little's MCAR Test.

    H0: Data are MCAR.
    p > 0.05 → fail to reject MCAR (pairwise deletion defensible).
    p ≤ 0.05 → reject MCAR (systematic missingness; proceed to Steps 3 & 5).

    A singular matrix error means too few complete cases exist to form the
    covariance matrix — this implies extensive systematic missingness and
    should be treated as implicit rejection of MCAR.
    """
    print(f"\n  H0: {index_label} data are MCAR")
    if not PYAMPUTE_AVAILABLE:
        print("  pyampute not installed — Little's test skipped.")
        return

    df_test = df_ind.dropna(how="all")
    print(f"  Rows with at least one indicator scored: {len(df_test):,}")
    try:
        p_val = MCARTest(method="little").little_mcar_test(df_test)
        print(f"  Little's MCAR test p-value: {p_val:.4f}")
        if p_val > 0.05:
            print(
                "  RESULT: Fail to reject H0 (p > 0.05)\n"
                "  → Data consistent with MCAR. Pairwise deletion defensible.\n"
                "  → Still run Step 5 to confirm no surveyor/temporal pattern."
            )
        else:
            print(
                "  RESULT: Reject H0 (p ≤ 0.05)\n"
                "  → Data are NOT MCAR. Missingness is systematic.\n"
                "  → Proceed to Steps 3 and 5 to identify the source."
            )
    except Exception as exc:
        print(f"  Little's test error: {exc}")
        print(
            "  Singular matrix — insufficient complete cases.\n"
            "  Treat as implicit MCAR rejection. See Steps 3 and 5."
        )


# =============================================================================
# STEP 3 — MAR TEST: LOGISTIC REGRESSION (CROSS-INDICATOR, AUC)
# =============================================================================

def step3_mar_test(
    df_ind: pd.DataFrame,
    index_label: str,
    out_dir: Path
) -> list[dict]:
    """
    Step 3 — MAR Test via Logistic Regression.

    For each indicator, a logistic regression predicts whether it is missing
    using all OTHER indicators in the same index as predictors (cross-indicator
    approach). This directly tests the MAR question: does knowing the rest of
    the survey explain why this indicator is missing?

    Design choices:
      - class_weight='balanced': corrects for imbalanced missing/observed ratio
      - AUC (not accuracy): correct under class imbalance; a model predicting
        'never missing' gets high accuracy but AUC = 0.50

    AUC interpretation:
      ~0.50       → MCAR consistent
      0.55–0.65   → Weak MAR signal
      0.65–0.80   → Moderate MAR signal → consider MICE
      > 0.80      → Strong MAR signal → MICE strongly advised
    """
    cols    = df_ind.columns.tolist()
    results = []

    for target in cols:
        n_missing = df_ind[target].isna().sum()
        n_obs     = df_ind[target].notna().sum()
        pct_miss  = n_missing / len(df_ind) * 100

        if n_missing == 0:
            results.append({"index": index_label, "indicator": target,
                             "n_missing": 0, "pct_missing": 0.0,
                             "auc": np.nan, "signal": "No missing values"})
            continue
        if n_obs == 0:
            results.append({"index": index_label, "indicator": target,
                             "n_missing": n_missing, "pct_missing": 100.0,
                             "auc": np.nan, "signal": "Entirely missing"})
            continue
        if n_missing < 10 or n_obs < 10:
            results.append({"index": index_label, "indicator": target,
                             "n_missing": n_missing, "pct_missing": pct_miss,
                             "auc": np.nan, "signal": "Insufficient cases"})
            continue

        other_cols = [c for c in cols if c != target]
        X = df_ind[other_cols].fillna(df_ind[other_cols].mean())
        y = df_ind[target].isna().astype(int)

        X_scaled = StandardScaler().fit_transform(X)
        lr = LogisticRegression(class_weight="balanced",
                                max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        auc = roc_auc_score(y, lr.predict_proba(X_scaled)[:, 1])

        if auc < 0.55:   signal = "MCAR consistent"
        elif auc < 0.65: signal = "Weak MAR signal"
        elif auc < 0.80: signal = "Moderate MAR signal"
        else:            signal = "Strong MAR signal"

        results.append({"index": index_label, "indicator": target,
                         "n_missing": n_missing, "pct_missing": pct_miss,
                         "auc": auc, "signal": signal})

    # AUC bar chart
    plot_df = pd.DataFrame([r for r in results if not np.isnan(r["auc"])])
    if not plot_df.empty:
        plot_df  = plot_df.sort_values("auc", ascending=True)
        cmap     = {"MCAR consistent": "#2ca02c", "Weak MAR signal": "#ff7f0e",
                    "Moderate MAR signal": "#d62728", "Strong MAR signal": "#8c1c1c"}
        colours  = [cmap.get(s, "#aec7e8") for s in plot_df["signal"]]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(plot_df["indicator"], plot_df["auc"],
                       color=colours, edgecolor="white")
        ax.axvline(0.50, color="grey",    linestyle=":",  linewidth=1.2,
                   label="0.50 — chance level")
        ax.axvline(0.65, color="#ff7f0e", linestyle="--", linewidth=1.2,
                   label="0.65 — weak MAR threshold")
        ax.axvline(0.80, color="#d62728", linestyle="--", linewidth=1.2,
                   label="0.80 — strong MAR threshold")
        ax.set_xlabel("AUC (ROC)", fontsize=11)
        ax.set_xlim(0.4, 1.0)
        ax.set_title(
            f"MAR Test: Logistic Regression AUC per {index_label} Indicator\n"
            "Predictors = all other indicators in the same index\n"
            "AUC > 0.65 indicates missingness is explained by observed data (MAR likely)",
            fontsize=11, pad=10
        )
        ax.legend(fontsize=9, loc="lower right")
        for bar, row in zip(bars, plot_df.itertuples()):
            ax.text(row.auc + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{row.auc:.3f}  ({row.pct_missing:.1f}% miss)",
                    va="center", fontsize=7.5)
        plt.tight_layout()
        fig.savefig(out_dir / f"fig4{index_label.lower()}_mar_auc.png", dpi=150)
        plt.close()
        print(f"  Saved: fig4{index_label.lower()}_mar_auc.png")

    return results


# =============================================================================
# STEP 4 — MNAR SENSITIVITY ANALYSIS
# =============================================================================

def step4_mnar_sensitivity(
    df_ind: pd.DataFrame,
    index_label: str,
    out_dir: Path
) -> None:
    """
    Step 4 — MNAR Sensitivity Analysis.

    Imputes missing values at score=5 (worst case — most vulnerable) and
    score=1 (best case — least vulnerable), then compares composite score
    distributions and class allocations against pairwise mean (current).

    Large divergence between the three distributions = composite scores are
    sensitive to the missing data assumption → report as limitation.

    The class shift table (Low/Medium/High %) is the key number to report
    in your methods section.
    """
    composite_pair = df_ind.mean(axis=1)
    composite_high = df_ind.fillna(5).mean(axis=1)
    composite_low  = df_ind.fillna(1).mean(axis=1)

    print(
        f"\n  {'Scenario':<30} {'Mean':>6} {'Median':>8} "
        f"{'SD':>6} {'Min':>6} {'Max':>6}"
    )
    print("  " + "-" * 60)
    for label, s in [
        ("Pairwise mean (current)", composite_pair),
        ("MNAR-high imputed (=5)",  composite_high),
        ("MNAR-low  imputed (=1)",  composite_low),
    ]:
        print(f"  {label:<30} {s.mean():>6.3f} {s.median():>8.3f} "
              f"{s.std():>6.3f} {s.min():>6.3f} {s.max():>6.3f}")

    def classify(s):
        norm = (s - 1) / 4.0
        return pd.cut(norm, bins=[-np.inf, 1/3, 2/3, np.inf],
                      labels=[1, 2, 3]).astype(float)

    c_p, c_h, c_l = classify(composite_pair), classify(composite_high), classify(composite_low)
    n = len(df_ind)
    class_map = {1: "Low", 2: "Medium", 3: "High"}
    print(f"\n  Class distribution under each scenario (n={n:,}):")
    print(f"  {'Class':<10} {'Pairwise':>12} {'MNAR-high':>12} {'MNAR-low':>12}")
    print("  " + "-" * 50)
    for k in [1, 2, 3]:
        np_ = (c_p == k).sum()
        nh  = (c_h == k).sum()
        nl  = (c_l == k).sum()
        print(f"  {class_map[k]:<10} {np_:>6} ({np_/n*100:.1f}%)  "
              f"{nh:>6} ({nh/n*100:.1f}%)  {nl:>6} ({nl/n*100:.1f}%)")

    bins = np.linspace(1, 5, 30)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(composite_pair.dropna(), bins=bins, alpha=0.6,
            label="Pairwise mean (current)", color="#4C72B0", edgecolor="white")
    ax.hist(composite_high, bins=bins, alpha=0.5,
            label="MNAR-high imputed (score=5)", color="#d62728", edgecolor="white")
    ax.hist(composite_low,  bins=bins, alpha=0.5,
            label="MNAR-low imputed  (score=1)", color="#2ca02c", edgecolor="white")
    ax.set_xlabel(f"{index_label} Composite Score (1–5)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"MNAR Sensitivity Analysis: {index_label} Composite Score Under Different Imputation Assumptions\n"
        "Observed vs worst-case and best-case bounds\n"
        "Large divergence between curves indicates high sensitivity to MNAR assumptions",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig5{index_label.lower()}_mnar_sensitivity.png", dpi=150)
    plt.close()
    print(f"\n  Saved: fig5{index_label.lower()}_mnar_sensitivity.png")


# =============================================================================
# STEP 5 — CLUSTERING ANALYSIS: SURVEYOR / COMMUNITY / TIME
# =============================================================================

def step5_clustering(
    df_ind: pd.DataFrame,
    index_label: str,
    out_dir: Path
) -> None:
    """
    Step 5 — Missingness Clustering by Surveyor, Community, and Time.

    WHY THIS STEP MATTERS
    ----------------------
    Steps 1–4 tell you HOW MUCH is missing and whether it is random (MCAR)
    or systematic (MAR/MNAR). Step 5 tells you WHERE the missingness comes
    from — which source variable explains the pattern.

    This matters for two practical reasons:
      1. It determines which variables to include as predictors in MICE.
         If surveyor drives missingness, surveyor identity must be a
         predictor in the imputation model (van Buuren, 2018, §6.3).
      2. It determines what to report in your limitations section. Surveyor-
         driven missingness is a data collection quality issue, distinct
         from physical inaccessibility (MNAR) or random skip (MCAR).

    SUB-ANALYSES
    ------------
    5a) Surveyor clustering heatmap (Fig 6)
        % missing per indicator per enumerator.
        Large variation between rows = enumerator behaviour is a key driver.
        Uniform rows = physical building characteristics drive missingness.

        Kruskal-Wallis test: tests whether missingness rates differ
        significantly across enumerators for each indicator.
        Significant result → surveyor is a MAR predictor.

    5b) Community clustering heatmap (Fig 7)
        % missing per indicator per VDC/ward.
        Note: Community and surveyor effects are CONFOUNDED because
        enumerators were assigned geographically. Interpret alongside Fig 6.

    5c) Temporal clustering — daily missing rate (Fig 8)
        Overall missing rate averaged across indicators, plotted daily
        with a 7-day rolling mean and survey volume (secondary axis).

        Rising trend   → survey fatigue or protocol drift
        Step changes   → possible change in field team or procedure
        No trend       → time is not a primary driver

        Spearman ρ quantifies the temporal trend statistically.
        If significant, include survey date as a MICE predictor.

    REFERENCES
    ----------
    Groves, R.M. (2006). Nonresponse rates and nonresponse bias in
      household surveys. Public Opinion Quarterly, 70(5), 646-675.
      https://doi.org/10.1093/poq/nfl033

    West, B.T. & Olson, K. (2010). How much of interviewer variance is
      really nonresponse error variance? Public Opinion Quarterly, 74(5),
      1004-1026. https://doi.org/10.1093/poq/nfq061
    """

    # Attach survey metadata to indicator DataFrame
    df_w = df_ind.copy()
    df_w["_surveyor"]    = df["_surveyor"].values
    df_w["_community"]   = df["_community"].values
    df_w["_survey_date"] = df["_survey_date"].values
    ind_cols = df_ind.columns.tolist()

    # ── 5a: Surveyor clustering heatmap ──────────────────────────────────────

    surv_miss = (
        df_w.groupby("_surveyor")[ind_cols]
        .apply(lambda g: g.isna().mean() * 100)
    )
    # Shorten enumerator labels for readability
    surv_miss.index = (
        surv_miss.index
        .str.replace(r"enumerator_\d+___", "", regex=True)
        .str.replace("_", " ")
        .str.title()
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(ind_cols) * 0.85), max(5, len(surv_miss) * 0.55))
    )
    sns.heatmap(
        surv_miss, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.4, ax=ax, vmin=0, vmax=100,
        annot_kws={"size": 8}
    )
    ax.set_title(
        f"{index_label} Indicator Missingness Rate by Surveyor\n"
        "Large variation between enumerators suggests surveyor behaviour drives missingness\n"
        "Uniform pattern suggests building characteristics or physical access drive missingness",
        fontsize=11, pad=10
    )
    ax.set_xlabel("Indicator", fontsize=10)
    ax.set_ylabel("Enumerator", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig6{index_label.lower()}_surveyor_clustering.png", dpi=150)
    plt.close()
    print(f"  Saved: fig6{index_label.lower()}_surveyor_clustering.png")

    # Summarise enumerator range per indicator
    print(f"\n  {index_label} — Missingness range across enumerators per indicator:")
    for col in ind_cols:
        if col in surv_miss.columns:
            vals = surv_miss[col].dropna()
            print(
                f"    {col:<30}: min={vals.min():.0f}%  "
                f"max={vals.max():.0f}%  "
                f"range={vals.max()-vals.min():.0f}pp"
            )

    # Kruskal-Wallis: is missingness significantly different across enumerators?
    print(f"\n  Kruskal-Wallis — {index_label} missingness by enumerator:")
    for col in ind_cols:
        groups = [
            df_w.loc[df_w["_surveyor"] == sv, col].isna().astype(float).values
            for sv in df_w["_surveyor"].unique()
            if (df_w["_surveyor"] == sv).sum() >= 10
        ]
        if len(groups) >= 2:
            try:
                h, p = stats.kruskal(*groups)
                sig = ("*** p<0.001" if p < 0.001 else "** p<0.01"
                       if p < 0.01 else "* p<0.05" if p < 0.05 else "ns")
                print(f"    {col:<30}: H={h:.2f}  p={p:.4f}  {sig}")
            except Exception:
                pass

    # ── 5b: Community clustering heatmap ─────────────────────────────────────

    comm_miss = (
        df_w.groupby("_community")[ind_cols]
        .apply(lambda g: g.isna().mean() * 100)
    )
    comm_miss.index = (
        comm_miss.index.str.replace("_", " ").str.title()
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(ind_cols) * 0.85), max(6, len(comm_miss) * 0.5))
    )
    sns.heatmap(
        comm_miss, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.4, ax=ax, vmin=0, vmax=100,
        annot_kws={"size": 8}
    )
    ax.set_title(
        f"{index_label} Indicator Missingness Rate by Community (VDC/Ward)\n"
        "Note: community and enumerator effects are partly confounded\n"
        "(each community was primarily covered by 1–2 enumerators)",
        fontsize=11, pad=10
    )
    ax.set_xlabel("Indicator", fontsize=10)
    ax.set_ylabel("Community (VDC/Ward)", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig7{index_label.lower()}_community_clustering.png", dpi=150)
    plt.close()
    print(f"\n  Saved: fig7{index_label.lower()}_community_clustering.png")

    # ── 5c: Temporal clustering — daily missing rate ──────────────────────────

    df_w["_overall_miss"] = df_w[ind_cols].isna().mean(axis=1)
    daily = (
        df_w.groupby("_survey_date")["_overall_miss"]
        .agg(["mean", "count"])
        .reset_index()
    )
    daily.columns      = ["date", "miss_rate", "n_surveys"]
    daily["miss_pct"]  = daily["miss_rate"] * 100
    daily["date"]      = pd.to_datetime(daily["date"])
    daily["roll_mean"] = daily["miss_pct"].rolling(7, min_periods=3).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax2.bar(daily["date"], daily["n_surveys"], alpha=0.2,
            color="#aec7e8", label="Surveys per day")
    ax2.set_ylabel("Surveys per day", fontsize=9, color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    ax1.plot(daily["date"], daily["miss_pct"], "o-",
             color="#d62728", alpha=0.5, markersize=4, label="Daily missing %")
    ax1.plot(daily["date"], daily["roll_mean"],
             color="#8c1c1c", linewidth=2.5, label="7-day rolling mean")

    ax1.set_xlabel("Survey date", fontsize=11)
    ax1.set_ylabel(f"{index_label} Average Missing Rate (%)", fontsize=11)
    ax1.set_title(
        f"{index_label} Indicator Missing Rate Over Survey Period\n"
        "Rising trend may indicate survey fatigue or protocol drift\n"
        "Step changes may indicate a change in field team or data collection procedure",
        fontsize=11, pad=10
    )
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / f"fig8{index_label.lower()}_temporal_clustering.png", dpi=150)
    plt.close()
    print(f"  Saved: fig8{index_label.lower()}_temporal_clustering.png")

    # Spearman ρ — temporal trend in missingness rate
    d_clean  = daily.dropna(subset=["miss_pct"])
    rho, p_r = stats.spearmanr(range(len(d_clean)), d_clean["miss_pct"])
    sig      = "significant" if p_r < 0.05 else "not significant"
    print(f"\n  Temporal trend (Spearman ρ): ρ={rho:.3f}  p={p_r:.4f}  → {sig}")
    if abs(rho) > 0.3 and p_r < 0.05:
        direction = "increases" if rho > 0 else "decreases"
        cause     = "Survey fatigue or protocol drift" if rho > 0 else "Enumerator experience effect"
        print(
            f"  INTERPRETATION: Missing rate {direction} over the field period.\n"
            f"  {cause} is a plausible driver.\n"
            "  Include survey date as a predictor variable in MICE imputation."
        )
    else:
        print(
            "  INTERPRETATION: No significant temporal trend.\n"
            "  Time-based protocol drift is not a primary driver."
        )


# =============================================================================
# EXECUTE — EVI
# =============================================================================

print("\n" + "=" * 65)
print("EVI MISSING DATA DIAGNOSTIC")
print("=" * 65)

print("\n--- Step 1: Visualise Missingness Patterns ---")
step1_visualise(df_evi, "EVI", OUTPUT_DIR)

print("\n--- Step 2: Little's MCAR Test ---")
step2_little_mcar(df_evi, "EVI")

print("\n--- Step 3: MAR Test (Logistic Regression, AUC) ---")
evi_results = step3_mar_test(df_evi, "EVI", OUTPUT_DIR)

print("\n--- Step 4: MNAR Sensitivity Analysis ---")
step4_mnar_sensitivity(df_evi, "EVI", OUTPUT_DIR)

print("\n--- Step 5: Clustering Analysis ---")
step5_clustering(df_evi, "EVI", OUTPUT_DIR)

# =============================================================================
# EXECUTE — FVI
# =============================================================================

print("\n" + "=" * 65)
print("FVI MISSING DATA DIAGNOSTIC")
print("=" * 65)

print("\n--- Step 1: Visualise Missingness Patterns ---")
step1_visualise(df_fvi, "FVI", OUTPUT_DIR)

print("\n--- Step 2: Little's MCAR Test ---")
step2_little_mcar(df_fvi, "FVI")

print("\n--- Step 3: MAR Test (Logistic Regression, AUC) ---")
fvi_results = step3_mar_test(df_fvi, "FVI", OUTPUT_DIR)

print("\n--- Step 4: MNAR Sensitivity Analysis ---")
step4_mnar_sensitivity(df_fvi, "FVI", OUTPUT_DIR)

print("\n--- Step 5: Clustering Analysis ---")
step5_clustering(df_fvi, "FVI", OUTPUT_DIR)

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY — EVI and FVI Missing Data Diagnosis")
print("=" * 65)

all_results = evi_results + fvi_results
print(
    f"\n  {'Index':<5} {'Indicator':<26} {'n miss':>7} "
    f"{'% miss':>8} {'AUC':>7}  Signal"
)
print("  " + "-" * 75)
for r in all_results:
    auc_str = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "  N/A "
    print(
        f"  {r['index']:<5} {r['indicator']:<26} "
        f"{r['n_missing']:>7,} {r['pct_missing']:>7.1f}%  "
        f"{auc_str:>7}  {r['signal']}"
    )

print(f"""
{'=' * 65}
CLUSTERING INTERPRETATION GUIDE
{'=' * 65}

  Surveyor clustering (Fig 6EVI/FVI):
    Large variation between enumerators → enumerator behaviour is a
    key MAR predictor. Include surveyor ID in MICE imputation model.
    Uniform patterns → physical access or building type drives missingness.

  Community clustering (Fig 7EVI/FVI):
    Variation across VDC/wards reflects both building stock and
    enumerator assignment (confounded). Interpret alongside Fig 6.

  Temporal clustering (Fig 8EVI/FVI):
    Significant rising Spearman ρ → survey fatigue. Include survey
    date or survey week as a predictor in MICE.

  MICE PREDICTOR RECOMMENDATION (based on Step 5):
    Baseline     : other indicators in the same index (cross-indicator)
    If surveyor significant  : add surveyor ID as categorical predictor
    If temporal significant  : add survey_week as numeric predictor
    Always include           : community (VDC/ward) as categorical predictor

  FAIR REPORTING:
    Report per-indicator missing rate, the diagnosed mechanism,
    clustering source, and remedy applied (pairwise deletion or MICE).

REFERENCES
----------
  Little (1988)      : https://doi.org/10.2307/2290157
  Sterne (2009)      : https://doi.org/10.1136/bmj.b2393
  van Buuren (2018)  : https://stefvanbuuren.name/fimd/
  Groves (2006)      : https://doi.org/10.1093/poq/nfl033
  West & Olson (2010): https://doi.org/10.1093/poq/nfq061
""")

print(f"All figures saved to: {OUTPUT_DIR}")
