"""
stage02_evi_scoring.py
=================
Stage 1 | Earthquake Vulnerability Index (EVI) — CIMDEN-style Scoring

Input  : 20263003_Nepal_Lumbini_data.csv  (raw survey)
Output : EVI_CIMDEN_scores.csv

Method
------
Each of 12 structural indicators across 5 domains is classified into
Low (1), Medium (3), or High (5) vulnerability. Equal weighting is used:
the composite score is the pairwise mean of all scored indicators.

    EVI_composite = mean(indicator_1 ... indicator_12)  [pairwise, scale 1–5]

Normalisation mirrors FVI exactly:
    EVI_norm_0_1 = (EVI_composite - 1) / (5 - 1)     [0–1]
    EVI_norm_1_5 = 1 + EVI_norm_0_1 * 4               [1–5, for regression]

Equal weighting rationale
--------------------------
Indicators are drawn from different structural domains (site, geometry,
walls, roof, connectivity). In the absence of empirical justification for
differential weights, equal weighting is the methodologically sound
default (Villagran De León, 2004; European Commission, 2018).

Domains and indicators
-----------------------
  Domain 1 — Site & Placement  (2 indicators)
    EVI_D1_01 : Adjacency / free-standing
    EVI_D1_02 : Edge / slope hazard within 3 m

  Domain 2 — Plan Geometry  (2 indicators)
    EVI_D2_01 : Plan shape regularity
    EVI_D2_02 : Length-to-width ratio

  Domain 3 — Wall System & Openings  (4 indicators)
    EVI_D3_01 : Height-to-thickness ratio (material-specific)
    EVI_D3_02 : Percentage openings in most-penetrated wall
    EVI_D3_03 : Minimum opening distance (corner & between)
    EVI_D3_04 : Wall thickness (material-specific)

  Domain 4 — Roof System  (2 indicators)
    EVI_D4_01 : Roof shape symmetry
    EVI_D4_02 : Roof mass class

  Domain 5 — Bands & Connectivity  (2 indicators)
    EVI_D5_01 : Horizontal bands per floor
    EVI_D5_04 : Roof fastening type

Output CSV columns added
------------------------
    EVI_D1_01 ... EVI_D5_04     (indicator level per observation)
    EVI_D1 ... EVI_D5           (domain means, pairwise)
    EVI_composite               (overall mean, 1–5)
    EVI_n_valid                 (count of non-missing indicators)
    EVI_norm_0_1                (normalised index, 0–1)   ← PRIMARY OUTPUT
    EVI_norm_1_5                (rescaled to 1–5 for regression)
    EVI_CLASS                   (1=Low, 2=Medium, 3=High)

EVI_D3_01 — Height-to-thickness (H:T) ratio scoring
-----------------------------------------------------
Thirteen wall materials are recognised (SAW01). Each material is mapped to
material-specific H:T breakpoints and scored on a three-tier scale:

  Material                  Low (1)   Medium (3)  High (5)   Standard
  ------------------------  --------  ----------  ---------  ---------
  Stone masonry             ≤ 1:8     1:8–1:10    > 1:10     IS 1905 / NBC 203
  Concrete (RC / block)     ≤ 1:12    1:12–1:16   > 1:16     IS 1905 Table 4
  Compressed earth (stab.)  ≤ 1:8     1:8–1:10    > 1:10     IS 13827 §7
  Bamboo                    ≤ 1:10    1:10–1:14   > 1:14     NBC 109 proxy
  Wood (timber-framed)      ≤ 1:10    1:10–1:14   > 1:14     NBC 109 proxy
  Comp. earth (non-stab.)   ≤ 1:6     1:6–1:8     > 1:8      IS 13827 §5
  Mud                       ≤ 1:6     1:6–1:8     > 1:8      IS 13827 §5
  Mud-straw                 ≤ 1:6     1:6–1:8     > 1:8      IS 13827 §5
  CGI sheet                 —         —            Always 5   Non-structural
  Tarpaulin                 —         —            Always 5   Non-structural
  Plastered wall            —         —            Always 5   Substrate-dependent
  Other                     —         —            Always 5   Unknown material
  I cannot see              —         —            Always 5   Not observable

References
----------
Bureau of Indian Standards (BIS). (1993). IS 13827: Improving Earthquake
    Resistance of Earthen Buildings — Guidelines. New Delhi: BIS.

Bureau of Indian Standards (BIS). (2018). IS 1905: Code of Practice for
    Structural Use of Unreinforced Masonry (4th rev.). New Delhi: BIS.

Government of Nepal (GoN). (1994). Nepal National Building Code NBC 109:
    Masonry: Unreinforced. Kathmandu: Ministry of Physical Planning.

Government of Nepal (GoN). (1994). Nepal National Building Code NBC 203:
    Guidelines for Earthquake Resistant Building Construction: Low Strength
    Masonry. Kathmandu: Ministry of Physical Planning.

Villagran De León, J.C. (2004). Vulnerability: A Conceptual and
    Methodological Review. UNU-EHS Source No. 4. United Nations University.

D'Ayala, D. et al. (2020). Flood vulnerability and risk assessment of
    urban traditional buildings. NHESS, 20, 2221–2241.
    https://doi.org/10.5194/nhess-20-2221-2020

European Commission (2018). Step 6: Weighting — 10-Step Guide to
    Composite Indicators.
    https://knowledge4policy.ec.europa.eu/composite-indicators/
    10-step-guide/step-6-weighting_en
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# FILE PATHS  — update if running from a different directory
# =============================================================================

INPUT_CSV  = str(Path(__file__).resolve().parent / "20263003_Nepal_Lumbini_data.csv")   # raw survey (same as FVI)
OUTPUT_CSV = str(Path(__file__).resolve().parent / "EVI_CIMDEN_scores.csv")

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig', low_memory=False)

# Replace blank/whitespace-only strings with NaN, then coerce all columns
# that can be numeric. This mirrors the FVI pipeline's data loading step and
# handles SPSS CSV exports where missing values are exported as spaces.
#
# infer_objects(copy=False) opts into the future pandas behaviour that
# suppresses the FutureWarning about silent downcasting after replace().
# See: https://pandas.pydata.org/docs/whatsnew/v2.1.0.html
df = df.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)
for col in df.columns:
    converted = pd.to_numeric(df[col], errors='coerce')
    if converted.notna().any() or df[col].isna().all():
        df[col] = converted

# ── Normalise ward name inconsistency ─────────────────────────────────────────
# GD005_VDC_name contains 'tulsipur-ward_14' (hyphen) in the raw survey CSV.
# Normalise to underscore here to match ward_information.csv and stage01 output.
_WARD_COL = "GD005_VDC_name"
if _WARD_COL in df.columns:
    _before = (df[_WARD_COL].astype(str) == "tulsipur-ward_14").sum()
    df[_WARD_COL] = df[_WARD_COL].astype(str).str.replace(
        "tulsipur-ward_14", "tulsipur_ward_14", regex=False
    ).where(df[_WARD_COL].notna(), other=np.nan)
    if _before:
        print(f"  Normalised {_before} rows: 'tulsipur-ward_14' → 'tulsipur_ward_14'")

print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns from '{INPUT_CSV}'")


def eq1(col_name: str) -> pd.Series:
    """Return boolean Series: True where column equals 1.
    Safely handles NaN (returns False for NaN).
    """
    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' not found in dataset.")
    return df[col_name].eq(1)


# =============================================================================
# PRE-COMPUTATIONS
# =============================================================================

# Number of floors (sum of floor-presence flags)
df['NumFloors'] = (
    df['SAL04_What_parts_of_the_building_are1st_floor'].eq(1).astype(int)
    + df['SAL04_What_parts_of_the_building_are2nd_floor'].eq(1).astype(int)
    + df['SAL04_What_parts_of_the_building_are3rd_floor'].eq(1).astype(int)
)

# Length-to-width ratio (used in Domain 2)
width  = df['SAM02_What_is_the_width_of_the_core_house']
length = df['SAM01_What_is_the_length_of_the_core_house']
df['LW_ratio'] = np.where(
    width.notna() & (width > 0),
    length / width,
    np.nan
)

# Height-to-thickness ratio (used in Domain 3)
height    = df['SAM03_What_is_the_heigth_of_the_core_house']
thickness = df['SAM07_What_is_the_thicknes_the_core_house_wall']
df['Ht_ratio'] = np.where(
    thickness.notna() & (thickness > 0),
    height / thickness,
    np.nan
)

# =============================================================================
# DOMAIN 1 — SITE & PLACEMENT
# =============================================================================

# EVI_D1_01: Free-standing / adjacency
# Sequential assignment — last match wins (best overwrites worst).
df['EVI_D1_01'] = np.nan
df.loc[eq1('SAL02A_How_is_the_location_of_the_buidirectly_against_other_buil'),
       'EVI_D1_01'] = 5
df.loc[eq1('SAL02A_How_is_the_location_of_the_buiwithing_3_m_of_another_bu'),
       'EVI_D1_01'] = 3
df.loc[eq1('SAL02A_How_is_the_location_of_the_buifully_detached'),
       'EVI_D1_01'] = 1

# EVI_D1_02: Edge / slope hazard within 3 m
df['EVI_D1_02'] = np.nan
safe = (
    eq1('SAL02_How_is_the_building_placed_on_no_slope_or_th')
    & eq1('SAL02_How_is_the_building_placed_on_no_edge_or_the')
)
df.loc[safe, 'EVI_D1_02'] = 1
at_risk = (
    eq1('SAL02_How_is_the_building_placed_on_within_3_meter')
    | eq1('SAL02_How_is_the_building_placed_on_within_3_meter_1')
    | eq1('SAL02_How_is_the_building_placed_on_a_retaining_wa')
)
df.loc[at_risk, 'EVI_D1_02'] = 5

# Domain 1 pairwise mean
df['EVI_D1'] = df[['EVI_D1_01', 'EVI_D1_02']].mean(axis=1)

# =============================================================================
# DOMAIN 2 — PLAN GEOMETRY
# =============================================================================

# EVI_D2_01: Plan shape regularity
df['EVI_D2_01'] = np.nan
regular  = eq1('SAL05_rectangle') | eq1('SAL05_square')
irregular = (
    eq1('SAL05_l_shape') | eq1('SAL05_t_shape')
    | eq1('SAL05_u_shape') | eq1('SAL05_no_usual_shape')
)
df.loc[regular,   'EVI_D2_01'] = 1
df.loc[irregular, 'EVI_D2_01'] = 5

# EVI_D2_02: Length-to-width ratio
df['EVI_D2_02'] = np.nan
lw = pd.to_numeric(df['LW_ratio'], errors='coerce')
df.loc[lw.le(3.0) & lw.notna(), 'EVI_D2_02'] = 1
df.loc[lw.gt(3.0),              'EVI_D2_02'] = 5

# Domain 2 pairwise mean
df['EVI_D2'] = df[['EVI_D2_01', 'EVI_D2_02']].mean(axis=1)

# =============================================================================
# DOMAIN 3 — WALL SYSTEM & OPENINGS
# =============================================================================

# EVI_D3_01: Height-to-thickness ratio (material-specific thresholds)
# -----------------------------------------------------------------------
# Scoring logic: three vulnerability tiers (Low=1, Medium=3, High=5).
#
# Each material has two H:T breakpoints (low_thresh, high_thresh):
#   H:T <= low_thresh  → score 1 (Low    vulnerability)
#   H:T <= high_thresh → score 3 (Medium vulnerability)
#   H:T >  high_thresh → score 5 (High   vulnerability)
#
# Materials with NO structural wall capacity (CGI sheet, tarpaulin) receive
# a fixed score of 5 regardless of measured H:T.
#
# Materials whose H:T risk is entirely determined by the underlying substrate
# (plastered walls) or cannot be assessed (other, i_cannot_see) are also
# assigned a fixed score of 5 (worst-case / unknown).
#
# Thresholds are derived from:
#   • IS 1905 (BIS, 2018) — unreinforced masonry
#   • IS 13827 (BIS, 1993) — earthen buildings in seismic zones
#   • Nepal NBC 109 / 203 (GoN, 1994) — masonry and timber-framed construction
#
# References
# ----------
# Bureau of Indian Standards (BIS). (1993). IS 13827: Improving Earthquake
#   Resistance of Earthen Buildings — Guidelines. New Delhi: BIS.
# Bureau of Indian Standards (BIS). (2018). IS 1905: Code of Practice for
#   Structural Use of Unreinforced Masonry (4th rev.). New Delhi: BIS.
# Government of Nepal (GoN). (1994). Nepal National Building Code NBC 109:
#   Masonry: Unreinforced. Kathmandu: Ministry of Physical Planning.
# Government of Nepal (GoN). (1994). Nepal National Building Code NBC 203:
#   Guidelines for Earthquake Resistant Building Construction: Low Strength
#   Masonry. Kathmandu: Ministry of Physical Planning.
#
# Lookup table format
# -------------------
# Keys   : SAW01 column suffix (string after 'SAW01_What_material_is_used_for_the_')
# Values : (low_thresh, high_thresh) — numeric H:T breakpoints, OR
#           None                     — fixed score of 5 (non-structural / unknown)
#
# Concrete (RC) walls — IS 1905 Table 4: slenderness ≤ 12 (Low), 12–16 (Medium), >16 (High)
# Stone masonry       — IS 1905 / NBC 203: ≤ 8 (Low), 8–10 (Medium), > 10 (High)
# Compressed earth (stabilised)
#                     — IS 13827 §7: ≤ 8 (Low), 8–10 (Medium), > 10 (High)
# Bamboo              — treated analogously to timber; NBC 109 proxy ≤ 10 (Low),
#                       10–14 (Medium), > 14 (High)
# Wood (timber-framed)— NBC 109: ≤ 10 (Low), 10–14 (Medium), > 14 (High)
# Non-stabilised earth, mud, mud-straw
#                     — IS 13827 §5: stricter limit; ≤ 6 (Low), 6–8 (Medium), > 8 (High)
# CGI sheet, tarpaulin— no load-bearing wall capacity → fixed score 5
# Plastered wall      — plaster does not alter substrate H:T → fixed score 5
# Other, i_cannot_see — material unknown → worst-case fixed score 5

_HT_THRESHOLDS = {
    # column suffix                  : (low_thresh, high_thresh)  or  None → fixed 5
    'stone'               : (8,  10),   # IS 1905 / NBC 203
    'concrete'            : (12, 16),   # IS 1905 Table 4 (RC / concrete block)
    'compressed_earth'    : (8,  10),   # IS 13827 §7 (stabilised earthen)
    'bamboo'              : (10, 14),   # NBC 109 proxy (similar to timber)
    'wood'                : (10, 14),   # NBC 109 timber-framed proxy
    'comp_earth_nonstabl' : (6,   8),   # IS 13827 §5 (non-stabilised; stricter)
    'mud'                 : (6,   8),   # IS 13827 §5
    'mud_straw'           : (6,   8),   # IS 13827 §5 (organic content reduces cohesion)
    'cgi_sheet'           : None,       # non-structural — fixed score 5
    'tarpaulin'           : None,       # non-structural — fixed score 5
    'the_wall_is_plastered': None,      # substrate-dependent — worst-case 5
    'other'               : None,       # unknown material  — worst-case 5
    'i_cannot_see'        : None,       # not observable    — worst-case 5
}

# Column prefix shared by all SAW01 wall-material flags
_SAW01_PREFIX = 'SAW01_What_material_is_used_for_the_'


def eq1_safe(col_name: str) -> pd.Series:
    """Return boolean Series (True where col == 1).

    Unlike eq1(), this silently returns an all-False Series when the column
    is absent from the dataset, so optional/new survey columns do not raise.
    """
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col_name].eq(1)


df['EVI_D3_01'] = np.nan
ht = pd.to_numeric(df['Ht_ratio'], errors='coerce')

for _suffix, _thresholds in _HT_THRESHOLDS.items():
    _col    = _SAW01_PREFIX + _suffix
    _is_mat = eq1_safe(_col)   # True where this material was recorded

    if _thresholds is None:
        # Non-structural or unknown material: assign worst-case score directly.
        df.loc[_is_mat, 'EVI_D3_01'] = 5
    else:
        _lo, _hi = _thresholds
        # Low vulnerability  : H:T <= low threshold
        df.loc[_is_mat & ht.le(_lo),                    'EVI_D3_01'] = 1
        # Medium vulnerability: low threshold < H:T <= high threshold
        df.loc[_is_mat & ht.gt(_lo) & ht.le(_hi),       'EVI_D3_01'] = 3
        # High vulnerability  : H:T > high threshold
        df.loc[_is_mat & ht.gt(_hi) & ht.notna(),        'EVI_D3_01'] = 5

# EVI_D3_02: Percentage openings in most-penetrated wall
df['EVI_D3_02'] = np.nan
df.loc[eq1('SAM06_What_is_the_total_pe_most_penetrated_wallless_than_50'),
       'EVI_D3_02'] = 3
df.loc[eq1('SAM06_What_is_the_total_pe_most_penetrated_wallaround_50'),
       'EVI_D3_02'] = 3
df.loc[eq1('SAM06_What_is_the_total_pe_most_penetrated_wallmore_than_50'),
       'EVI_D3_02'] = 5

# EVI_D3_03: Minimum opening distance (corner & between openings)
df['min_opening_dist'] = df[[
    'SAM04_What_is_the_length_b_and_the_wall_opening',
    'SAM05_What_is_the_length_b_ween_2_wall_openings'
]].min(axis=1)
df['EVI_D3_03'] = np.nan
md = pd.to_numeric(df['min_opening_dist'], errors='coerce')
df.loc[md.ge(0.9),               'EVI_D3_03'] = 1
df.loc[md.ge(0.6) & md.lt(0.9), 'EVI_D3_03'] = 3
df.loc[md.lt(0.6) & md.notna(), 'EVI_D3_03'] = 5

# EVI_D3_04: Wall thickness (material-specific minimum requirements)
# -----------------------------------------------------------------------
# Each structural material has a code-minimum wall thickness. Walls meeting
# or exceeding that minimum are scored Low (1); walls below it are scored
# High (5). Non-structural materials (CGI, tarpaulin) and unknowns receive
# a fixed score of 5.
#
# Minimum thickness thresholds
# ----------------------------
# Material                 Min (m)  Standard
# ----------------------   -------  ---------------------------------
# Stone masonry            0.35     NBC 203 §6.3 / IS 1905 cl. 4.6
# Concrete (RC / block)    0.23     IS 1905 cl. 4.6 (one-brick equiv.)
# Compressed earth (stab.) 0.30     IS 13827 §7.2
# Comp. earth (non-stab.)  0.40     IS 13827 §5.3 (thicker due to lower strength)
# Mud                      0.40     IS 13827 §5.3
# Mud-straw                0.40     IS 13827 §5.3
# Bamboo                   0.10     NBC 109 proxy (panel/frame thickness)
# Wood (timber-framed)     0.10     NBC 109 proxy (structural panel thickness)
# CGI sheet                —        Non-structural → fixed 5
# Tarpaulin                —        Non-structural → fixed 5
# Plastered wall           —        Substrate-dependent → fixed 5
# Other                    —        Unknown → fixed 5
# I cannot see             —        Not observable → fixed 5
#
# References
# ----------
# BIS (1993). IS 13827. BIS (2018). IS 1905.
# GoN (1994). NBC 109 / NBC 203.

_THICKNESS_THRESHOLDS = {
    # column suffix                  : min_thickness (m)  or  None → fixed 5
    'stone'                : 0.35,
    'concrete'             : 0.23,
    'compressed_earth'     : 0.30,
    'comp_earth_nonstabl'  : 0.40,
    'mud'                  : 0.40,
    'mud_straw'            : 0.40,
    'bamboo'               : 0.10,
    'wood'                 : 0.10,
    'cgi_sheet'            : None,
    'tarpaulin'            : None,
    'the_wall_is_plastered': None,
    'other'                : None,
    'i_cannot_see'         : None,
}

df['EVI_D3_04'] = np.nan
tw = pd.to_numeric(df['SAM07_What_is_the_thicknes_the_core_house_wall'],
                   errors='coerce')

for _suffix, _min_t in _THICKNESS_THRESHOLDS.items():
    _col    = _SAW01_PREFIX + _suffix
    _is_mat = eq1_safe(_col)

    if _min_t is None:
        # Non-structural or unknown: worst-case score.
        df.loc[_is_mat, 'EVI_D3_04'] = 5
    else:
        # Meets minimum thickness → Low; below minimum → High.
        df.loc[_is_mat & tw.ge(_min_t),              'EVI_D3_04'] = 1
        df.loc[_is_mat & tw.lt(_min_t) & tw.notna(), 'EVI_D3_04'] = 5

# Domain 3 pairwise mean
df['EVI_D3'] = df[['EVI_D3_01', 'EVI_D3_02',
                    'EVI_D3_03', 'EVI_D3_04']].mean(axis=1)

# =============================================================================
# DOMAIN 4 — ROOF SYSTEM
# =============================================================================

# EVI_D4_01: Roof shape symmetry
df['EVI_D4_01'] = np.nan
symmetric = (
    eq1('SAR01_What_is_the_type_of_the_roofgable_roof')
    | eq1('SAR01_What_is_the_type_of_the_roofhip_roof')
    | eq1('SAR01_What_is_the_type_of_the_roofbonnet_roof')
    | eq1('SAR01_What_is_the_type_of_the_roofpyramid_hip_roof')
)
non_symmetric = (
    eq1('SAR01_What_is_the_type_of_the_roofflat_roof')
    | eq1('SAR01_What_is_the_type_of_the_roofshed_roof')
    | eq1('SAR01_What_is_the_type_of_the_roofother')
    | eq1('SAR01_What_is_the_type_of_the_roofi_cannot_see')
)
df.loc[symmetric,     'EVI_D4_01'] = 1
df.loc[non_symmetric, 'EVI_D4_01'] = 5

# EVI_D4_02: Roof mass class/material
# Light = Low vulnerability; moderate = Medium; heavy concrete = High
df['EVI_D4_02'] = np.nan
df.loc[eq1('SAR04_What_roofing_material_is_used_cgi_sheet'),
       'EVI_D4_02'] = 1
moderate_roof = (
    eq1('SAR04_What_roofing_material_is_used_stone_roofing_tiles')
    | eq1('SAR04_What_roofing_material_is_used_clay_roofing_tiles')
    | eq1('SAR04_What_roofing_material_is_used_concrete_roofing_tiles')
    | eq1('SAR04_What_roofing_material_is_used_thatched_mud')
    | eq1('SAR04_What_roofing_material_is_used_thatched__covered_with_mud')
    | eq1('SAR04_What_roofing_material_is_used_wood')
)
df.loc[moderate_roof, 'EVI_D4_02'] = 3
df.loc[eq1('SAR04_What_roofing_material_is_used_concrete')
    | eq1('SAR04_What_roofing_material_is_used_other')
    | eq1('SAR04_What_roofing_material_is_used_i_cannot_see'),
       'EVI_D4_02'] = 5

# Domain 4 pairwise mean
df['EVI_D4'] = df[['EVI_D4_01', 'EVI_D4_02']].mean(axis=1)

# =============================================================================
# DOMAIN 5 — BANDS & CONNECTIVITY
# =============================================================================

# EVI_D5_01: Horizontal bands — scored by band material quality
# Classification based on SAB01 material response columns:
#   Score 5 (High vulnerability)  : no_bands_used = 1, other = 1, or can't_see = 1
#   Score 3 (Medium vulnerability): wood = 1 or bamboo = 1
#   Score 1 (Low vulnerability)   : concrete = 1
# Worst-case priority rule: if multiple materials are ticked,
# the highest vulnerability score takes precedence.

# Map each material column to its vulnerability score
_band_material_scores = {
    'SAB01_What_is_the_material_of_the_baconcrete':    1,   # reinforced concrete — best
    'SAB01_What_is_the_material_of_the_bawood':        3,   # timber — intermediate
    'SAB01_What_is_the_material_of_the_babamboo':      3,   # bamboo — intermediate
    'SAB01_What_is_the_material_of_the_bano_bands_used': 5, # no bands — most vulnerable
    'SAB01_What_is_the_material_of_the_baother':       5,   # unknown other — most vulnerable
    'SAB01_What_is_the_material_of_the_bai_can_t_see': 5,   # not visible — most vulnerable
}

# For each row, take the maximum (worst-case) score across all ticked materials.
# Rows where none of the columns are ticked receive NaN (missing).
_band_score_cols = []
for _col, _score in _band_material_scores.items():
    if _col in df.columns:
        _series = pd.to_numeric(df[_col], errors='coerce')
        _band_score_cols.append(
            _series.eq(1).map({True: _score, False: np.nan})
        )

if _band_score_cols:
    df['EVI_D5_01'] = pd.concat(_band_score_cols, axis=1).max(axis=1)
else:
    df['EVI_D5_01'] = np.nan

# EVI_D5_04: Roof fastening type
df['EVI_D5_04'] = np.nan
rf_low = (
    eq1('SARA06_Which_material_is_used_to_fastnails')
    | eq1('SARA06_Which_material_is_used_to_fastbolts')
    | eq1('SARA06_Which_material_is_used_to_fastbeams_top_roofing')
)
rf_med = (
    eq1('SARA06_Which_material_is_used_to_fastrope')
    | eq1('SARA06_Which_material_is_used_to_fastdowels')
)
df.loc[rf_low, 'EVI_D5_04'] = 1
df.loc[rf_med, 'EVI_D5_04'] = 3
df.loc[eq1('SARA06_Which_material_is_used_to_fastno_connection')
       | eq1('SARA06_Which_material_is_used_to_fastother')
       | eq1('SARA06_Which_material_is_used_to_fasti_cannot_see')
       | eq1('SARA06_Which_material_is_used_to_faststone_laying_o')
       | eq1('SARA06_Which_material_is_used_to_fastoption_6')
       | eq1('SARA06_Which_material_is_used_to_fastoption_7'),
       'EVI_D5_04'] = 5

# Domain 5 pairwise mean
df['EVI_D5'] = df[['EVI_D5_01', 'EVI_D5_04']].mean(axis=1)

# =============================================================================
# COMPOSITE EVI SCORE  (equal-weight pairwise mean)
# =============================================================================

EVI_INDICATORS = [
    'EVI_D1_01', 'EVI_D1_02',
    'EVI_D2_01', 'EVI_D2_02',
    'EVI_D3_01', 'EVI_D3_02', 'EVI_D3_03', 'EVI_D3_04',
    'EVI_D4_01', 'EVI_D4_02',
    'EVI_D5_01', 'EVI_D5_04',
]

# Pairwise mean — skips NaN, requires at least 1 valid indicator
df['EVI_composite'] = df[EVI_INDICATORS].mean(axis=1)
df['EVI_n_valid']   = df[EVI_INDICATORS].notna().sum(axis=1)

# =============================================================================
# NORMALISATION  — mirrors FVI exactly
# =============================================================================
# Because equal weights are used and all indicators share the same 1–5 scale,
# the composite is already in [1, 5]. Normalisation is therefore:
#
#   EVI_norm_0_1 = (EVI_composite - 1) / (5 - 1)
#   EVI_norm_1_5 = EVI_composite   (already on 1–5 scale)
#
# However, to maintain strict equivalence with the FVI pipeline (which derives
# norm_0_1 and norm_1_5 from the weighted composite), we apply the same
# min–max logic so the output columns are structurally identical.

valid = df['EVI_composite'].notna()

df['EVI_norm_0_1'] = np.nan
df.loc[valid, 'EVI_norm_0_1'] = (df.loc[valid, 'EVI_composite'] - 1) / 4.0

df['EVI_norm_1_5'] = np.nan
df.loc[valid, 'EVI_norm_1_5'] = 1 + df.loc[valid, 'EVI_norm_0_1'] * 4

# =============================================================================
# CLASSIFICATION  — three equal bands of the 0–1 range (mirrors FVI)
# =============================================================================
#   Low    : 0.000 – 0.333
#   Medium : 0.333 – 0.667
#   High   : 0.667 – 1.000


def classify(v):
    """Return 1=Low, 2=Medium, 3=High, or NaN if missing."""
    if pd.isna(v):
        return np.nan
    if v <= 1 / 3:
        return 1
    elif v <= 2 / 3:
        return 2
    else:
        return 3


df['EVI_CLASS'] = df['EVI_norm_0_1'].apply(classify)

# =============================================================================
# VALIDATION SUMMARY  — mirrors FVI_1 output format
# =============================================================================

print('\n' + '=' * 65)
print('EVI CIMDEN — VALIDATION SUMMARY')
print('=' * 65)
print(f'Total rows              : {len(df):,}')
valid_n = df['EVI_norm_0_1'].notna().sum()
print(f'Valid EVI_norm_0_1      : {valid_n:,} ({valid_n / len(df) * 100:.1f}%)')

print(f'\nEVI_norm_0_1 descriptives:')
print(df['EVI_norm_0_1'].describe().round(4).to_string())

print(f'\nEVI_norm_1_5 descriptives (regression scale):')
print(df['EVI_norm_1_5'].describe().round(4).to_string())

print(f'\nEVI_CLASS distribution:')
class_map = {1: 'Low', 2: 'Medium', 3: 'High'}
for k, n_c in df['EVI_CLASS'].value_counts().sort_index().items():
    print(f'  {int(k)} {class_map[int(k)]:8s}: {n_c:,} ({n_c / len(df) * 100:.1f}%)')

print(f'\nIndicator coverage and level distribution:')
ind_names = [
    'D1_01 Adjacency',    'D1_02 Slope/Edge',
    'D2_01 PlanShape',    'D2_02 LW_ratio',
    'D3_01 Ht_ratio',     'D3_02 Openings%',
    'D3_03 OpeningDist',  'D3_04 WallThick',
    'D4_01 RoofShape',    'D4_02 RoofMass',
    'D5_01 HorizBands',   'D5_04 RoofFasten',
]
for nm, col in zip(ind_names, EVI_INDICATORS):
    s        = df[col].dropna()
    n_scored = len(s)
    pct      = n_scored / len(df) * 100
    vc       = {int(k): v for k, v in s.value_counts().sort_index().items()}
    print(f'  {nm:<22}: n={n_scored:4,} ({pct:4.1f}%)  {vc}')

# =============================================================================
# SAVE OUTPUT
# =============================================================================

new_cols = EVI_INDICATORS + [
    'EVI_D1', 'EVI_D2', 'EVI_D3', 'EVI_D4', 'EVI_D5',
    'EVI_composite', 'EVI_n_valid',
    'EVI_norm_0_1', 'EVI_norm_1_5', 'EVI_CLASS',
]
df.to_csv(OUTPUT_CSV, index=False)
print(f'\nOutput saved to : {OUTPUT_CSV}')
print(f'Columns added   : {", ".join(new_cols)}')