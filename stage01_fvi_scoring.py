"""
stage01_fvi_scoring.py
================
Stage 1 | CIMDEN Scoring

Input  : RAW_SURVEY_CSV (set via PIPELINE_CONFIG in run_fvi_pipeline.py)
         e.g. "20263003_Nepal_Lumbini_data.csv"
Output : FVI_CIMDEN_scores.csv

Description
-----------
Applies the CIMDEN vulnerability scoring method to the raw survey data.
Each of 10 structural indicators is classified into Low (1), Medium (3),
or High (5) vulnerability levels, then multiplied by its researcher-assigned
weight. The composite FVI score is normalised to 0-1 (FVI_norm_0_1) and
rescaled to 1-5 (FVI_norm_1_5) for regression compatibility.

NOTE: Geographical Position (FVI1) was removed from the scoring at the
recommendation of Nepal and ITC domain experts.

This is the PRIMARY OUTPUT of the pipeline. All subsequent stages read
FVI_CIMDEN_scores.csv as their input.

Reference
---------
Villagran De León, J.C. (2004). Vulnerability: A Conceptual and
    Methodological Review. UNU-EHS Source No. 4. United Nations University.
"""

# ─── Pipeline note ────────────────────────────────────────────────────────────
# When run via run_fvi_pipeline.py, the working directory is set to the
# directory containing the raw survey CSV. INPUT_CSV and OUTPUT_CSV below
# are therefore resolved relative to that directory.
# ─────────────────────────────────────────────────────────────────────────────


"""
Flood Vulnerability Index (FVI) – CIMDEN Approach
===================================================
Python translation of: FloodVulnerabilityIndex_Score_CIMDEN_FINAL.sps

Reference:
    Villagran De León, J.C. (2004). Vulnerability: A Conceptual and
    Methodological Review. UNU-EHS Source No. 4. United Nations
    University, Bonn. Section 3.4.2, pp. 43-44.

Method:
    Each indicator is classified into three vulnerability levels:
        Low    = 1
        Medium = 3
        High   = 5
    Each indicator carries an explicit weight (researcher-assigned).
    Composite score is a weighted linear sum:
        FVI_SCORE = SUM(weight_j * level_j)  for all scored indicators
    Normalised to 0-1 using only weights of scored indicators:
        FVI_norm_0_1 = (FVI_SCORE - FVI_MIN_AVAIL)
                       / (FVI_MAX_AVAIL - FVI_MIN_AVAIL)
    Classification uses three equal bands of the 0-1 range:
        Low    : 0.000 – 0.333
        Medium : 0.333 – 0.667
        High   : 0.667 – 1.000

Indicators and weights (10 active indicators; sum of weights = 85):
    NOTE: Geographical Position (original FVI1, weight=8) was removed
    at the recommendation of Nepal and ITC domain experts.

    2.  Roof material              weight = 8   [Low/High only]
    3.  Roof connections           weight = 9
    4.  Wall building materials    weight = 9
    5.  Elevated ground (stilts)   weight = 8   [Low/High only]
    6.  Geometry of building       weight = 9   [Low/High only]
    7.  Roof overhang              weight = 9
    8.  Presence of an apron       weight = 9
    9.  Water drainage on site     weight = 8   [Low/High only]
    10. Number of storeys/floors   weight = 8
    11. Building width-to-length   weight = 8   [Low/High only]
        Sum of weights = 85

    Binary indicators (only Low=1 and High=5 observed in data, no Medium=3):
        FVI2  Roof material
        FVI5  Elevated ground / stilts
        FVI6  Geometry of building
        FVI9  Water drainage on site
        FVI11 Building width-to-length ratio

    Three-level indicators (Low=1, Medium=3, High=5):
        FVI3  Roof connections
        FVI4  Wall building materials
        FVI7  Roof overhang
        FVI8  Presence of an apron
        FVI10 Number of floors (binary: 1F → High, 2–3F → Low; 4F+ unobservable)

All-indicator bounds (when all 10 active indicators are scored):
    Minimum (all Low  = 1): 85
    Maximum (all High = 5): 425
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ── File paths ────────────────────────────────────────────────────────────────

INPUT_CSV  = str(Path(__file__).resolve().parent / "20263003_Nepal_Lumbini_data.csv")
OUTPUT_CSV = str(Path(__file__).resolve().parent / "FVI_CIMDEN_scores.csv")

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig', low_memory=False)
print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns from '{INPUT_CSV}'")

# ── Normalise ward name inconsistency ─────────────────────────────────────────
# GD005_VDC_name contains 'tulsipur-ward_14' (hyphen) in the raw survey CSV.
# The ward_information.csv reference file uses 'tulsipur_ward_14' (underscore).
# Normalise to underscore here at source so all downstream files are consistent:
# FVI_CIMDEN_scores.csv → analysis_dataset.csv → mixed-effects stages 15 & 21.
_WARD_COL = "GD005_VDC_name"
if _WARD_COL in df.columns:
    _before = (df[_WARD_COL] == "tulsipur-ward_14").sum()
    df[_WARD_COL] = df[_WARD_COL].str.replace(
        "tulsipur-ward_14", "tulsipur_ward_14", regex=False
    )
    if _before:
        print(f"  Normalised {_before} rows: 'tulsipur-ward_14' → 'tulsipur_ward_14'")

# ── Helper: coerce a column to numeric, returning NaN for blanks/text ─────────

def to_num(col):
    """Return a numeric Series, coercing non-numeric values to NaN."""
    return pd.to_numeric(
        df[col].replace(" ", np.nan).replace("", np.nan).infer_objects(copy=False),
        errors="coerce"
    )

def flag(col):
    """Return True where a binary dummy column equals 1, else False."""
    return to_num(col) == 1

# =============================================================================
# NOTE: INDICATOR 1 – GEOGRAPHICAL POSITION (weight = 8) REMOVED
# =============================================================================
# FVI1 (Geographical Position) was excluded from this scoring run at the
# recommendation of Nepal and ITC domain experts. The SAL01 sub-columns are
# preserved in the raw data but are not used in the FVI calculation.
# Original weight contribution: 8 (out of former total 93).
# Revised total weight sum: 85 (10 indicators).
# =============================================================================

# =============================================================================
# INDICATOR 2 – ROOF MATERIAL  (weight = 8)   *** BINARY ***
# =============================================================================
# Source: SAR04 sub-columns
#
# Low  (1): Concrete OR Tiles (stone/clay/concrete) OR CGI sheet
# High (5): Thatched mud OR Thatched/mud-covered OR Wood
# NaN     : i_cannot_see / other / missing
#
# No Medium class — binary in this dataset.
# High overwrites Low — worst-case wins.
# =============================================================================

lvl2 = pd.Series(np.nan, index=df.index, name="FVI2_RoofMat_LVL")

lvl2[flag("SAR04_What_roofing_material_is_used_concrete")         |
     flag("SAR04_What_roofing_material_is_used_cgi_sheet")        |
     flag("SAR04_What_roofing_material_is_used_stone_roofing_tiles") |
     flag("SAR04_What_roofing_material_is_used_clay_roofing_tiles")  |
     flag("SAR04_What_roofing_material_is_used_concrete_roofing_tiles")] = 1

lvl2[flag("SAR04_What_roofing_material_is_used_thatched_mud")              |
     flag("SAR04_What_roofing_material_is_used_thatched__covered_with_mud") |
     flag("SAR04_What_roofing_material_is_used_wood")       |
     flag("SAR04_What_roofing_material_is_used_other")      |
     flag("SAR04_What_roofing_material_is_used_i_cannot_see")]                      = 5

df["FVI2_RoofMat_LVL"] = lvl2
df["FVI2_RoofMat_WS"]  = np.where(lvl2.notna(), 8 * lvl2, np.nan)

# =============================================================================
# INDICATOR 3 – ROOF CONNECTIONS / FASTENING  (weight = 9)   [THREE LEVELS]
# =============================================================================
# Source: SARA06 sub-columns
#
# Low  (1): Nails AND/OR bolts
# Med  (3): Rope
# High (5): No connection / stone laying / i_cannot_see
#
# Priority: High > Medium > Low (worst-case wins).
# =============================================================================

lvl3 = pd.Series(np.nan, index=df.index, name="FVI3_RoofConn_LVL")

lvl3[flag("SARA06_Which_material_is_used_to_fastnails") |
     flag("SARA06_Which_material_is_used_to_fastbolts")  |
     flag("SARA06_Which_material_is_used_to_fastbeams_top_roofing")] = 1

lvl3[flag("SARA06_Which_material_is_used_to_fastrope")  |
     flag("SARA06_Which_material_is_used_to_fastdowels")] = 3

lvl3[flag("SARA06_Which_material_is_used_to_fastno_connection") |
     flag("SARA06_Which_material_is_used_to_faststone_laying_o") |
     flag("SARA06_Which_material_is_used_to_fasti_cannot_see")  |
     flag("SARA06_Which_material_is_used_to_fastoption_6")  |
     flag("SARA06_Which_material_is_used_to_fastoption_7")  |
     flag("SARA06_Which_material_is_used_to_fastother")]  = 5

df["FVI3_RoofConn_LVL"] = lvl3
df["FVI3_RoofConn_WS"]  = np.where(lvl3.notna(), 9 * lvl3, np.nan)

# =============================================================================
# INDICATOR 4 – WALL BUILDING MATERIALS  (weight = 9)   [THREE LEVELS]
# =============================================================================
# Source: SAW01 sub-columns
#
# Low  (1): Concrete
# Med  (3): Stone
# High (5): Bamboo / compressed earth / mud / mud+straw / CGI sheet /
#           tarpaulin / wood / i_cannot_see
#
# Priority: High > Medium > Low (worst-case wins).
# =============================================================================

lvl4 = pd.Series(np.nan, index=df.index, name="FVI4_WallMat_LVL")

lvl4[flag("SAW01_What_material_is_used_for_the_concrete")] = 1

lvl4[flag("SAW01_What_material_is_used_for_the_stone")] = 3

lvl4[flag("SAW01_What_material_is_used_for_the_bamboo")           |
     flag("SAW01_What_material_is_used_for_the_compressed_earth") |
     flag("SAW01_What_material_is_used_for_the_comp_earth_nonstabl") |
     flag("SAW01_What_material_is_used_for_the_mud")              |
     flag("SAW01_What_material_is_used_for_the_mud_straw")        |
     flag("SAW01_What_material_is_used_for_the_cgi_sheet")        |
     flag("SAW01_What_material_is_used_for_the_tarpaulin")        |
     flag("SAW01_What_material_is_used_for_the_wood")             |
     flag("SAW01_What_material_is_used_for_the_i_cannot_see")]    = 5

df["FVI4_WallMat_LVL"] = lvl4
df["FVI4_WallMat_WS"]  = np.where(lvl4.notna(), 9 * lvl4, np.nan)

# =============================================================================
# INDICATOR 5 – ELEVATED GROUND / STILTS  (weight = 8)   *** BINARY ***
# =============================================================================
# Source: SAF02_Is_the_house_build_on_sti  (string column)
#
# Low  (1): Yes / yes / yes_it_is_build_on_plinth
# High (5): No / no / Don't_know / Don't_Know
# NaN     : any other value / missing
#
# No Medium class — binary in this dataset.
# =============================================================================

raw5 = df["SAF02_Is_the_house_build_on_sti"].replace(" ", np.nan).infer_objects(copy=False)

lvl5 = pd.Series(np.nan, index=df.index, name="FVI5_Elev_LVL")
lvl5[raw5.isin(["Yes", "yes", "yes_it_is_build_on_plinth"])] = 1
lvl5[raw5.isin(["No",  "no",  "Don't_know", "Don't_Know"])] = 5

df["FVI5_Elev_LVL"] = lvl5
df["FVI5_Elev_WS"]  = np.where(lvl5.notna(), 8 * lvl5, np.nan)

# =============================================================================
# INDICATOR 6 – GEOMETRY OF BUILDING  (weight = 9)   *** BINARY ***
# =============================================================================
# Source: SAL05 sub-columns
#
# Low  (1): Rectangle OR Square
# High (5): L-shape / T-shape / U-shape / no_usual_shape / other
# NaN     : missing
#
# No Medium class — binary in this dataset.
# High overwrites Low — worst-case wins.
# =============================================================================

lvl6 = pd.Series(np.nan, index=df.index, name="FVI6_Geom_LVL")

lvl6[flag("SAL05_rectangle") | flag("SAL05_square")] = 1

lvl6[flag("SAL05_l_shape")        |
     flag("SAL05_t_shape")        |
     flag("SAL05_u_shape")        |
     flag("SAL05_no_usual_shape") |
     flag("SAL05_other")]         = 5

df["FVI6_Geom_LVL"] = lvl6
df["FVI6_Geom_WS"]  = np.where(lvl6.notna(), 9 * lvl6, np.nan)

# =============================================================================
# INDICATOR 7 – ROOF OVERHANG  (weight = 9)   [THREE LEVELS]
# =============================================================================
# Source: SAR02_How_much_is_th_the_roof_ADD_IMAGE  (string column)
#
# Low  (1): 500_600_mm
# Med  (3): 600_900_mm
# High (5): less_than_500_mm / more_than_900_mm / none /
#           overhang_size_differs_around_the_house / i_cannot_see / other
# NaN     : missing
# =============================================================================

raw7 = df["SAR02_How_much_is_th_the_roof_ADD_IMAGE"].replace(" ", np.nan).infer_objects(copy=False)

lvl7 = pd.Series(np.nan, index=df.index, name="FVI7_Overhang_LVL")
lvl7[raw7 == "500_600_mm"] = 1
lvl7[raw7 == "600_900_mm"] = 3
lvl7[raw7.isin(["less_than_500_mm",
                "more_than_900_mm",
                "none",
                "overhang_size_differs_around_the_house",
                "i_cannot_see",
                "other"])] = 5

df["FVI7_Overhang_LVL"] = lvl7
df["FVI7_Overhang_WS"]  = np.where(lvl7.notna(), 9 * lvl7, np.nan)

# =============================================================================
# INDICATOR 8 – PRESENCE OF AN APRON  (weight = 9)   [THREE LEVELS]
# =============================================================================
# Source: SAF03_Does_the_house_on_ADD_IMAGES  (string column)
#
# Low  (1): yes (all around)
# Med  (3): yes__around_part_of_the_house
# High (5): no / i_don_t_know
# NaN     : missing
# =============================================================================

raw8 = df["SAF03_Does_the_house_on_ADD_IMAGES"].replace(" ", np.nan).infer_objects(copy=False)

lvl8 = pd.Series(np.nan, index=df.index, name="FVI8_Apron_LVL")
lvl8[raw8 == "yes"]                           = 1
lvl8[raw8 == "yes__around_part_of_the_house"] = 3
lvl8[raw8.isin(["no", "i_don_t_know"])]       = 5

df["FVI8_Apron_LVL"] = lvl8
df["FVI8_Apron_WS"]  = np.where(lvl8.notna(), 9 * lvl8, np.nan)

# =============================================================================
# INDICATOR 9 – WATER DRAINAGE ON SITE  (weight = 8)   *** BINARY ***
# =============================================================================
# Source: SAWA01 sub-columns
#
# Low  (1): Any drainage present (rain gutters / drainage pipes / slopes away)
# High (5): No drainage (no OR unknown)
# NaN     : only 'other' flagged / missing
#
# No Medium class — binary in this dataset.
# Worst-case rule: no/unknown overwrites drainage present.
# =============================================================================

n_drain = (
    flag("SAWA01_Is_there_water_the_site_ADD_IMAGESyes__rain_gutters").astype(int) +
    flag("SAWA01_Is_there_water_the_site_yes__drainage_pipes").astype(int) +
    flag("SAWA01_Is_there_water_the_site_ADD_IMAGESyes__area_slopes_away_f").astype(int)
)
no_drain  = flag("SAWA01_Is_there_water_the_site_no")
unk_drain = flag("SAWA01_Is_there_water_the_site_unknown")

lvl9 = pd.Series(np.nan, index=df.index, name="FVI9_Drain_LVL")
lvl9[n_drain > 0]          = 1   # any drainage → Low
lvl9[no_drain | unk_drain] = 5   # no/unknown → High (overwrites Low)

df["FVI9_Drain_LVL"] = lvl9
df["FVI9_Drain_WS"]  = np.where(lvl9.notna(), 8 * lvl9, np.nan)

# =============================================================================
# INDICATOR 10 – NUMBER OF STOREYS / FLOORS  (weight = 8)   *** BINARY ***
# =============================================================================
# Scoring rationale (updated v27.3):
#
#   High (5) — 1 floor:
#       Single-storey buildings cannot shelter occupants above flood level.
#       The entire structure is exposed to inundation and destruction forces.
#
#   Low  (1) — 2–3 floors:
#       Upper floors provide vertical evacuation capacity. The additional
#       structural mass and wall continuity improve resistance to flood
#       lateral loads compared to a single-storey building.
#
#   High (5) — 4+ floors:
#       Tall multi-storey buildings face elevated foundation risk under
#       flood-induced soil saturation and scour. The increased dead load
#       magnifies settlement and overturning susceptibility.
#
# SURVEY INSTRUMENT LIMITATION:
#   SAL04 records only three binary sub-columns (1st, 2nd, 3rd floor).
#   The maximum floor count detectable from this dataset is 3. Buildings
#   with 4 or more floors cannot be distinguished from 3-floor buildings.
#   The 4F+ High-vulnerability class is documented here for methodological
#   completeness and replication fidelity; it cannot be scored without an
#   additional survey field capturing absolute storey count.
#
#   Consequence: FVI10 is BINARY in this dataset (levels 1 and 5 only).
#   Medium (3) is never produced because the 2–3 floor range maps to Low (1)
#   and 4+ floors are not observable.
#
# nf_count: sum of binary presence flags (1st + 2nd + 3rd floor columns).
#   nf_count = 1  → 1 floor  → High (5)
#   nf_count = 2  → 2 floors → Low  (1)
#   nf_count = 3  → 3 floors → Low  (1)
#   nf_count = 0  → unresolved (no floors ticked) → NaN
# =============================================================================

# Count floors from binary presence columns (each = 1 if floor exists, else 0)
nf_count = (
    flag("SAL04_What_parts_of_the_building_are1st_floor").astype(int) +
    flag("SAL04_What_parts_of_the_building_are2nd_floor").astype(int) +
    flag("SAL04_What_parts_of_the_building_are3rd_floor").astype(int)
)

lvl10 = pd.Series(np.nan, index=df.index, name="FVI10_Floors_LVL")
lvl10[nf_count == 1]               = 5  # 1 floor  → High: full inundation exposure
lvl10[nf_count.isin([2, 3])]       = 1  # 2–3 floors → Low: vertical evacuation capacity
# nf_count >= 4 is not observable in this dataset (see limitation note above).
# nf_count == 0 remains NaN (no floor columns ticked — missing or inapplicable).

df["FVI10_Floors_LVL"] = lvl10
df["FVI10_Floors_WS"]  = np.where(lvl10.notna(), 8 * lvl10, np.nan)

# =============================================================================
# INDICATOR 11 – BUILDING WIDTH-TO-LENGTH RATIO  (weight = 8)   *** BINARY ***
# =============================================================================
# Pre-compute LW_ratio = SAM01 length / SAM02 width.
#
# Low  (1): LW_ratio < 3  (length < 3 × width — compact footprint)
# High (5): LW_ratio >= 3 (length >= 3 × width — elongated, less stable)
# NaN     : missing or width = 0
#
# No Medium class — binary in this dataset.
# =============================================================================

length = to_num("SAM01_What_is_the_length_of_the_core_house")
width  = to_num("SAM02_What_is_the_width_of_the_core_house")

lw = pd.Series(np.nan, index=df.index)
valid_w      = width.notna() & (width > 0)
lw[valid_w]  = length[valid_w] / width[valid_w]

lvl11 = pd.Series(np.nan, index=df.index, name="FVI11_LWratio_LVL")
lvl11[lw <  3] = 1
lvl11[lw >= 3] = 5

df["FVI11_LWratio_LVL"] = lvl11
df["FVI11_LWratio_WS"]  = np.where(lvl11.notna(), 8 * lvl11, np.nan)

# =============================================================================
# COMPOSITE FVI SCORE
# =============================================================================
# pandas .sum(skipna=True, min_count=1) mirrors SPSS SUM():
# returns NaN only if ALL inputs are NaN, otherwise sums available values.
#
# FVI1 (Geographical Position) is excluded per expert recommendation.
# Active indicators: FVI2–FVI11 (10 indicators; sum of weights = 85).
# =============================================================================

ws_cols = [
    "FVI2_RoofMat_WS",  "FVI3_RoofConn_WS",
    "FVI4_WallMat_WS",  "FVI5_Elev_WS",     "FVI6_Geom_WS",
    "FVI7_Overhang_WS", "FVI8_Apron_WS",    "FVI9_Drain_WS",
    "FVI10_Floors_WS",  "FVI11_LWratio_WS"
]

# Weights aligned to ws_cols order (FVI2 through FVI11; FVI1 removed)
weights     = [8,  9,  9,  8,  9,  9,  9,  8,  8,  8]
weights_min = [w * 1 for w in weights]   # Low  score (level = 1)
weights_max = [w * 5 for w in weights]   # High score (level = 5)

df["FVI_SCORE"] = df[ws_cols].sum(axis=1, skipna=True, min_count=1)

# FVI_MIN_AVAIL: sum of (weight × 1) for each indicator that was scored
df["FVI_MIN_AVAIL"] = sum(
    weights_min[i] * df[ws_cols[i]].notna().astype(int)
    for i in range(10)
)

# FVI_MAX_AVAIL: sum of (weight × 5) for each indicator that was scored
df["FVI_MAX_AVAIL"] = sum(
    weights_max[i] * df[ws_cols[i]].notna().astype(int)
    for i in range(10)
)

# =============================================================================
# NORMALISATION  — 0 to 1
# =============================================================================
# FVI_norm_0_1 = (FVI_SCORE - FVI_MIN_AVAIL) / (FVI_MAX_AVAIL - FVI_MIN_AVAIL)
#
# FVI_norm_1_5 retained for regression compatibility:
#   FVI_norm_1_5 = 1 + FVI_norm_0_1 * 4
#   Maps 0→1 and 1→5, preserving the 1-5 scale used in stepwise regressions.
# =============================================================================

valid = df["FVI_SCORE"].notna() & (df["FVI_MAX_AVAIL"] > df["FVI_MIN_AVAIL"])

df["FVI_norm_0_1"] = np.nan
df.loc[valid, "FVI_norm_0_1"] = (
    (df.loc[valid, "FVI_SCORE"] - df.loc[valid, "FVI_MIN_AVAIL"])
    / (df.loc[valid, "FVI_MAX_AVAIL"] - df.loc[valid, "FVI_MIN_AVAIL"])
)

# Rescale to 1-5 for regression models (maintains backward compatibility)
df["FVI_norm_1_5"] = np.nan
df.loc[valid, "FVI_norm_1_5"] = 1 + df.loc[valid, "FVI_norm_0_1"] * 4

# =============================================================================
# CLASSIFICATION  — three equal bands of the 0-1 range
# =============================================================================
# Mirrors previous SPSS thresholds, now expressed on 0-1 scale:
#   Low    : 0.000 – 0.333
#   Medium : 0.333 – 0.667
#   High   : 0.667 – 1.000
# =============================================================================

def classify(v):
    """Return 1=Low, 2=Medium, 3=High, or NaN if missing."""
    if pd.isna(v):
        return np.nan
    if v <= 1/3:
        return 1
    elif v <= 2/3:
        return 2
    else:
        return 3

df["FVI_CLASS"] = df["FVI_norm_0_1"].apply(classify)

# =============================================================================
# BINARY INDICATOR REPORT
# =============================================================================
# Binary indicators have only two vulnerability levels in this dataset
# (Low=1 and High=5, no Medium=3). This is a property of the data, not
# the scoring rules — the classification scheme allows for Medium=3 but
# the survey responses do not produce it for these five indicators.
#
# NOTE: FVI1 (Geographical Position) is excluded from this report
# per expert recommendation.
# =============================================================================

print("\n" + "=" * 65)
print("BINARY INDICATOR REPORT")
print("=" * 65)
print("An indicator is binary if only Low (1) and High (5) are observed")
print("in the data — no Medium (3) responses were recorded.")
print()

lvl_info = [
    ("FVI2_RoofMat_LVL",  "FVI2  Roof material",           "binary"),
    ("FVI3_RoofConn_LVL", "FVI3  Roof connections",        "three-level"),
    ("FVI4_WallMat_LVL",  "FVI4  Wall building materials", "three-level"),
    ("FVI5_Elev_LVL",     "FVI5  Elevated ground/stilts",  "binary"),
    ("FVI6_Geom_LVL",     "FVI6  Geometry of building",    "binary"),
    ("FVI7_Overhang_LVL", "FVI7  Roof overhang",           "three-level"),
    ("FVI8_Apron_LVL",    "FVI8  Presence of apron",       "three-level"),
    ("FVI9_Drain_LVL",    "FVI9  Water drainage on site",  "binary"),
    ("FVI10_Floors_LVL",  "FVI10 Number of floors",        "binary"),    # v27.3: 1F=High, 2-3F=Low
    ("FVI11_LWratio_LVL", "FVI11 Width-to-length ratio",   "binary"),
]

print(f"  {'Indicator':<32} {'Observed levels':<20} {'Type'}")
print("  " + "-" * 62)
for col, name, expected_type in lvl_info:
    levels  = sorted(df[col].dropna().unique().tolist())
    is_bin  = set(levels) == {1.0, 5.0}
    type_str = "*** BINARY ***" if is_bin else "three-level"
    flag_mismatch = " [CHECK]" if (is_bin != (expected_type == "binary")) else ""
    print(f"  {name:<32} {str([int(l) for l in levels]):<20} {type_str}{flag_mismatch}")

print()
binary_names = [name for col, name, _ in lvl_info
                if set(sorted(df[col].dropna().unique())) == {1.0, 5.0}]
print(f"  Binary indicators ({len(binary_names)}/10):")
for b in binary_names:
    print(f"    - {b.strip()}")

# =============================================================================
# VALIDATION SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("FVI CIMDEN — VALIDATION SUMMARY")
print("=" * 65)
print(f"Total rows              : {len(df):,}")
valid_n = df["FVI_norm_0_1"].notna().sum()
print(f"Valid FVI_norm_0_1      : {valid_n:,} ({valid_n/len(df)*100:.1f}%)")
print(f"\nFVI_norm_0_1 descriptives:")
print(df["FVI_norm_0_1"].describe().round(4).to_string())

print(f"\nFVI_norm_1_5 descriptives (regression scale):")
print(df["FVI_norm_1_5"].describe().round(4).to_string())

print(f"\nFVI_CLASS distribution:")
class_map = {1: "Low", 2: "Medium", 3: "High"}
vc = df["FVI_CLASS"].value_counts().sort_index()
for k, n_c in vc.items():
    print(f"  {int(k)} {class_map[int(k)]:8s}: {n_c:,} ({n_c/len(df)*100:.1f}%)")

print(f"\nIndicator coverage and level distribution:")
# FVI1 excluded; report covers FVI2–FVI11 (10 active indicators)
lvl_series = [lvl2,  lvl3,  lvl4,  lvl5,  lvl6,
              lvl7,  lvl8,  lvl9,  lvl10, lvl11]
ind_names  = [
    "RoofMat", "RoofConn", "WallMat", "Elev",    "Geom",
    "Overhang","Apron",    "Drain",   "Floors",  "LWratio"
]
# FVI indicator numbers skip 1 (removed); start enumeration at 2
for i, (nm, lvl) in enumerate(zip(ind_names, lvl_series), 2):
    n_scored = lvl.notna().sum()
    vc2      = {int(k): v for k, v in lvl.value_counts().sort_index().items()}
    pct      = n_scored / len(df) * 100
    print(f"  FVI{i:2d} {nm:10s}: n={n_scored:4,} ({pct:4.1f}%)  {vc2}")

# =============================================================================
# SAVE OUTPUT
# =============================================================================

new_cols = ws_cols + [
    "FVI_SCORE", "FVI_MIN_AVAIL", "FVI_MAX_AVAIL",
    "FVI_norm_0_1", "FVI_norm_1_5", "FVI_CLASS"
]
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nOutput saved to: {OUTPUT_CSV}")
print(f"Columns added  : {', '.join(new_cols)}")