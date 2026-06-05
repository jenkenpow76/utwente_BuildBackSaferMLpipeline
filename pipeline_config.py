"""
pipeline_config.py
==================
Single source of truth for all analytical parameters used across the
vulnerability index pipeline (Stages 01-18b).

Purpose
-------
Every stage script imports its variable lists, column names, thresholds,
and analytical constants from here instead of declaring them independently.
This ensures:

    - Adding or removing a candidate predictor in one place automatically
      propagates to missingness screening (Stage 06a), bivariate screening
      (Stages 07, 12), stepwise regression (Stages 08, 13), Moran's I
      (Stages 09, 14), mixed-effects modelling (Stages 10, 16), and random
      forest validation (Stages 18, 18b).
    - Human-readable labels used in plots and tables are defined once and
      shared across all stages.
    - Analytical thresholds (significance levels, class boundaries, weights)
      are changed in one place only.
    - The pipeline is fully reproducible: RANDOM_STATE=42 is the single
      seed used by all stages that involve randomness.

How to update variable lists
-----------------------------
1. Edit FVI_CANDIDATES or EVI_CANDIDATES to add or remove a candidate.
   Follow the (col_name, human_label, test_type) tuple format.

2. Update FVI_PRED_LABELS or EVI_PRED_LABELS with a human-readable label
   for the new variable (used in all plots and printed tables).

3. Run the pipeline from Stage 06a onwards. Stages 07, 08, 12, 13, 18, and
   18b will automatically receive the updated candidate lists.

4. After the pipeline runs, copy the new FVI_IM_PREDICTORS and
   EVI_IM_PREDICTORS values from the Stage 08/13 output (fvi_im_predictors.csv,
   evi_im_predictors.csv) back into this file to keep the fallback lists
   current. The pipeline uses the CSV chain automatically when running end-to-end;
   the config lists are only used when running a stage standalone.

Variable list hierarchy
------------------------
CANDIDATES  (44 FVI / 41 EVI variables)
    Full candidate pool for bivariate screening (Stage 07/12).
    Format per entry: (col_name, human_label, test_type)
    test_type = 'pearson'  for continuous 0-100 percentage scores
              = 'spearman' for ordinal, binary, or Likert items
    Consumed by Stage 06a (missingness screening) and Stage 18 (RF).
    Variables excluded from these lists had >50% missing in the raw survey
    and were dropped before pipeline development:
      App_Perc_Pos_expression (52.2%), Acc_Perc_Neg_expression (87.4%),
      AB_Location_Neg (61.2%), OP_Manpower_Pos/Neg (92.5%),
      OP_Funding_Pos (72.1%), OP_Funding_Neg (85.8%).

DOMAINS  (post-bivariate pool, grouped by MAO domain)
    Variables that passed bivariate screening (p < 0.05), grouped by domain
    for Stage 1 (within-domain) stepwise regression in Stages 08/13.
    Consumed by Stages 08, 09, 10, 13, 14, 16.
    Stage 06a writes fvi_domains_filtered.csv / evi_domains_filtered.csv
    from these lists (after removing any variables flagged >50% missing),
    which Stages 07/12 read at startup.

IM_PREDICTORS  (fallback predictor list)
    Variables retained in the final Individual Model after Stage 2 joint
    stepwise elimination (Stages 08/13). Used as a fallback by Stages 09/10
    and 14/16 when running standalone (without the CSV chain from Stage 08/13).
    When running via run_pipeline.py, these stages automatically read from
    fvi_im_predictors.csv / evi_im_predictors.csv (Stage 08/13 outputs).
    Update these lists after each full pipeline run to keep them current.

PRED_LABELS  (human-readable labels)
    Maps column name -> human-readable string for all plots, tables, and
    importance charts. Shared base labels (_SHARED_LABELS) cover MAO and
    socioeconomic variables; index-specific labels are added per-index.

CAP_COLS  (data-quality caps)
    Columns with known data-entry errors (values recorded as 1e11).
    Capped at 100 before any analysis in all stages. Applied in Stages
    07, 08, 10, 12, 13, 16, 18, 18b.

Survey column identifiers
--------------------------
    COMMUNITY_COL  : column identifying the VDC/ward community
    SURVEYOR_COL   : column identifying the enumerator
    FVI_SCORE_COL  : outcome column for FVI regressions (FVI_norm_1_5)
    EVI_SCORE_COL  : outcome column for EVI regressions (EVI_norm_1_5)
    CVI_SCORE_COL  : output column written by Stage 17

Analytical thresholds
----------------------
    BNDRY_LM       : Low/Medium class boundary = 1 + (1/3)*4 = 2.333
    BNDRY_MH       : Medium/High class boundary = 1 + (2/3)*4 = 3.667
    STEPWISE_PIN   : p-value to enter stepwise regression = 0.05
    STEPWISE_POUT  : p-value to leave stepwise regression = 0.10
    MORANS_THRESHOLD_KM : reporting distance for Moran's I = 1.0 km
                     (village-to-village scale; robust across 0.25-10 km)
    FVI_WEIGHT     : FVI weight in CVI = 0.5
    EVI_WEIGHT     : EVI weight in CVI = 0.5
    RANDOM_STATE   : global random seed = 42

Imputation context
-------------------
The MAO composite scores in FVI_CANDIDATES and EVI_CANDIDATES fall into
three groups with respect to imputation:

    Already imputed by Stage 05b MICE (complete in analysis_dataset.csv):
      Uti_Perc_Neg_expression, App_Perc_Neg_expression,
      AB_Selfefficacy_Neg, AB_Physical_capacity_Pos/Neg,
      AB_Financial_capacity_Pos/Neg, AB_Location_Pos,
      AB_Time_Pos/Neg, OP_Materials_Pos/Neg,
      OP_Location_Pos/Neg, DE12_shelter_score.
      Pre-imputation missingness: 15-30% (enumerator-driven MAR pattern).

    Complete in analysis_dataset.csv (no imputation needed):
      Uti_Perc_Pos_expression, AB_Selfefficacy_Pos, OP_Training_Pos/Neg,
      Acc_Perc_Pos_expression, and all socioeconomic / binary variables.

    Structurally missing (not imputed by design):
      Flood experience variables (DE03, DE04, DE05, DE07, etc.) are absent
      for households that reported no flood exposure. This is MNAR by
      design. These variables are included in analyses via complete-case
      handling; their missingness does not reduce the usable sample below
      n=2,993 because the missingness pattern overlaps perfectly with
      DE01_flood / DE02_flood indicators that are complete.

Stage 18 (random forest) prints an imputation audit table at startup
confirming the imputation status and remaining missingness rate of every
candidate variable before the forest is trained.

References
----------
Hendriks, E. & Stokmans, M. (2020). Drivers and barriers of adoption of
    hazard-resistant construction knowledge in Nepal. IJDRR, 51, 101778.
    https://doi.org/10.1016/j.ijdrr.2020.101778

Saputra, A., Schwarz, J., & Hendriks, E. (2026). Identifying factors
    influencing housing safety in post-earthquake reconstruction in Nepal.
    IJDRR, 133, 105913. https://doi.org/10.1016/j.ijdrr.2025.105913

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
    CRC Press. §9.1.4 (50% missingness threshold).
    https://stefvanbuuren.name/fimd/
"""

# =============================================================================
# DATA QUALITY CAPS
# =============================================================================
# These columns contain known data-entry errors (values up to 1e11).
# Cap at 100 before ANY analysis — applies to both FVI and EVI pipelines.
# Applied in: fvi_03, fvi_04, fvi_05, fvi_06, evi_03, evi_04, evi_05, evi_06.

CAP_COLS = [
    "AB_Physical_capacity_Pos",  # Physical capacity — Positive (entry errors up to 1e11)
    "OP_Training_Neg",           # Training access — Negative  (entry errors up to 1e11)
]

# =============================================================================
# SHARED HUMAN-READABLE LABELS
# =============================================================================
# Labels that are identical across FVI and EVI (MAO + socioeconomic variables).
# Index-specific labels are defined separately below in FVI_PRED_LABELS and
# EVI_PRED_LABELS, which extend this shared base.

_SHARED_LABELS = {
    # Motivation
    "Uti_Perc_Pos_expression":       "Utility perception (+)",
    "Uti_Perc_Neg_expression":       "Utility perception (−)",
    "App_Perc_Neg_expression":       "Applicability perception (−)",
    "Acc_Perc_Pos_expression":       "Acceptability perception (+)",
    # Ability
    "AB_Selfefficacy_Pos":           "Self-efficacy (+)",
    "AB_Selfefficacy_Neg":           "Self-efficacy (−)",
    "AB_Physical_capacity_Pos":      "Physical capacity (+)",
    "AB_Physical_capacity_Neg":      "Physical capacity (−)",
    "AB_Financial_capacity_Pos":     "Financial capacity (+)",
    "AB_Financial_capacity_Neg":     "Financial capacity (barrier)",
    "AB_Location_Pos":               "Location suitability (+)",
    "AB_Time_Pos":                   "Time available (+)",
    "AB_Time_Neg":                   "Time available (−)",
    # Opportunity
    "OP_Training_Pos":               "Training access (+)",
    "OP_Training_Neg":               "Training access (−)",
    "OP_Materials_Pos":              "Materials (+)",
    "OP_Materials_Neg":              "Materials (barrier)",
    "OP_Location_Pos":               "Location externally (+)",
    "OP_Location_Neg":               "Location externally (−)",
    # Socioeconomic — demographics
    "HC003_Respondent_s_age":        "Respondent age",
    "HC004_amount_people_household": "Household size",
    # Socioeconomic — education
    "SE01_edu_no_education":         "Education: none",
    "SE01_edu_can_read_write":       "Education: can read and write",
    "SE01_edu_elementary_school":    "Education: elementary",
    "SE01_edu_high_school":          "Education: high school",
    "SE01_edu_university":           "Education: university",
    # Socioeconomic — occupation
    "SE04_agriculture":              "Occupation: agriculture",
    "SE04_day_labour":               "Occupation: day labour",
    "SE04_construction_worker":      "Occupation: construction worker",
    "SE04_education":                "Occupation: education",
    "SE04_remittances":              "Income: remittances",
    "SE04_business":                 "Occupation: business",
    "SE04_government_officer":       "Occupation: government officer",
}

# =============================================================================
# FVI — FLOOD VULNERABILITY INDEX
# =============================================================================

# -----------------------------------------------------------------------------
# FVI_CANDIDATES — full candidate pool for bivariate screening (Stage 3)
# -----------------------------------------------------------------------------
# Format: (column_name, human_label, test_type)
# test_type = 'pearson'  for continuous 0–100 raw scores
# test_type = 'spearman' for ordinal CAT scores, binary dummies, Likert scores

FVI_CANDIDATES = {

    "1_Motivation": [
        ("Uti_Perc_Pos_expression",  "Utility perception — Positive",        "pearson"),
        ("Uti_Perc_Neg_expression",  "Utility perception — Negative",         "pearson"),
        ("App_Perc_Neg_expression",  "Applicability perception — Negative",   "pearson"),
        ("Acc_Perc_Pos_expression",  "Acceptability perception — Positive",   "pearson"),
    ],

    "2_Ability": [
        ("AB_Selfefficacy_Pos",        "Self-efficacy — Positive",         "pearson"),
        ("AB_Selfefficacy_Neg",        "Self-efficacy — Negative",         "pearson"),
        ("AB_Physical_capacity_Pos",   "Physical capacity — Positive",     "pearson"),
        ("AB_Physical_capacity_Neg",   "Physical capacity — Negative",     "pearson"),
        ("AB_Financial_capacity_Pos",  "Financial capacity — Positive",    "pearson"),
        ("AB_Financial_capacity_Neg",  "Financial capacity — Negative",    "pearson"),
        ("AB_Location_Pos",            "Location suitability — Positive",  "pearson"),
        ("AB_Time_Pos",                "Time available — Positive",        "pearson"),
        ("AB_Time_Neg",                "Time available — Negative",        "pearson"),
    ],

    "3_Opportunity": [
        ("OP_Training_Pos",  "Training access — Positive",      "pearson"),
        ("OP_Training_Neg",  "Training access — Negative",      "pearson"),
        ("OP_Materials_Pos", "Materials — Positive",            "pearson"),
        ("OP_Materials_Neg", "Materials — Negative",            "pearson"),
        ("OP_Location_Pos",  "Location externally — Positive",  "pearson"),
        ("OP_Location_Neg",  "Location externally — Negative",  "pearson"),
    ],

    "4_Flood_Experience": [
        ("DE03_flood_frequency_score",       "Flood frequency score",           "spearman"),
        ("DE04_flood_depth_score",           "Flood depth score",               "spearman"),
        ("DE08_worry_score",                 "Current flood worry score",       "spearman"),
        ("DE06_flood_future_score",          "Future flood expectation",        "spearman"),
        ("DE09_worry_future_score",          "Future flood worry",              "spearman"),
        ("DE12_shelter_score",               "Shelter/evacuation plan score",   "spearman"),
        ("DE13_feel_safe_score",             "Feel safe score",                 "spearman"),
        ("DE05_negative_impact_score",       "Past negative impact score",      "spearman"),
        ("DE07_negative_impact_future_score","Future negative impact score",    "spearman"),
        ("DE01_flood",                       "Flood: first-mentioned major hazard",    "spearman"),
        ("DE02_flood",                       "Flood: second-mentioned major hazard",   "spearman"),
        # PP02 (flood rebuild history) — 3-level collapsed categorical.
        # Constructed by stage08b_ward_covariates.py from the parent field
        # PP02_rebuild_repair_house_001_001. Reference category = "undamaged".
        # The SPSS Motivation MAO syntax (Hendriks 2025) does NOT use PP02 as
        # an input to Utility, Applicability, or Acceptability composites, so
        # there is no by-construction collinearity with the Motivation MAO
        # composites. Ability and Opportunity syntax not available; this is
        # acknowledged as a residual methodological uncertainty in Section
        # 5.4 (Limitations) of the thesis.
        ("PP02_flood_rebuild_3lvl_damaged",
         "Flood rebuild history: damaged (vs undamaged)",      "spearman"),
        ("PP02_flood_rebuild_3lvl_unknown_other",
         "Flood rebuild history: unknown/other (vs undamaged)", "spearman"),
    ],

    "5_Socioeconomic": [
        ("HC003_Respondent_s_age",        "Respondent age",                "pearson"),
        ("HC004_amount_people_household", "Household size",                "pearson"),
        ("SE01_edu_no_education",         "Education: none",               "spearman"),
        ("SE01_edu_can_read_write",       "Education: can read/write",     "spearman"),
        ("SE01_edu_elementary_school",    "Education: elementary",         "spearman"),
        ("SE01_edu_high_school",          "Education: high school",        "spearman"),
        ("SE01_edu_university",           "Education: university",         "spearman"),
        ("SE04_agriculture",              "Occupation: agriculture",       "spearman"),
        ("SE04_day_labour",               "Occupation: day labour",        "spearman"),
        ("SE04_construction_worker",      "Occupation: construction worker","spearman"),
        ("SE04_education",                "Occupation: education",         "spearman"),
        ("SE04_remittances",              "Income: remittances",           "spearman"),
        ("SE04_business",                 "Occupation: business",          "spearman"),
        ("SE04_government_officer",       "Occupation: government",        "spearman"),
    ],
}

# -----------------------------------------------------------------------------
# FVI_DOMAINS, FVI_ALL_DOMAIN_VARS, FVI_IM_PREDICTORS
#
# These names are now DERIVED at runtime from pipeline output files, not
# hand-coded. The derivation rules are:
#
#   FVI_DOMAINS         = variables in FVI_CANDIDATES that passed the
#                         bivariate screening (p < 0.05), grouped by domain.
#                         Source: outputs/bivariate_screening_significant.csv
#                         (produced by stage12_fvi_bivariate.py).
#
#   FVI_ALL_DOMAIN_VARS = flat list of every variable in FVI_DOMAINS.
#
#   FVI_IM_PREDICTORS   = the variables retained by the Stage 13 joint
#                         stepwise regression (the final Individual Model).
#                         Source: outputs/fvi_stepwise_stage2_results.csv
#                         (produced by stage13_fvi_stepwise.py).
#
# See `_pipeline_lazy_loader.py` for the implementation. The lazy-loading
# pattern (PEP 562 module __getattr__) means downstream stages can still
# write `from pipeline_config import FVI_DOMAINS` and it will be resolved
# from the CSV at first access.
#
# WHY THIS CHANGE WAS MADE
# ------------------------
# v46 had these as hand-coded constants. The Stage 12 (bivariate) output
# and the EVI_DOMAINS / EVI_IM_PREDICTORS constants drifted out of sync
# over time: variables that did not pass bivariate were nonetheless
# included in EVI_DOMAINS, and variables in EVI_IM_PREDICTORS bypassed
# the stepwise candidate pool altogether. The architectural fix in v47
# makes drift structurally impossible: every list is derived from the
# upstream CSV at access time.
#
# See `assert_no_drift()` below for the runtime invariant.
# -----------------------------------------------------------------------------

# Path constants for the lazy loader.
# Anchored to the directory containing pipeline_config.py rather than to
# the current working directory, because run_pipeline.py chdirs into
# outputs/ before launching each analysis stage. Using an absolute path
# means the lazy loader works regardless of where the stage is launched
# from (run_pipeline.py orchestration, manual invocation from the
# project root, or manual invocation from within outputs/).
import os
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_OUTPUTS_DIR = _HERE / "outputs"

_BIVARIATE_FVI_CSV   = str(_OUTPUTS_DIR / "bivariate_screening_significant.csv")
_BIVARIATE_EVI_CSV   = str(_OUTPUTS_DIR / "evi_bivariate_screening_significant.csv")
_STEPWISE_FVI_IM_CSV = str(_OUTPUTS_DIR / "fvi_im_predictors.csv")
_STEPWISE_EVI_IM_CSV = str(_OUTPUTS_DIR / "evi_im_predictors.csv")


def _load_domains_from_bivariate(candidates_dict, sig_csv_path):
    """
    Load the post-bivariate domains dict from the stage12/17 significant CSV.

    Parameters
    ----------
    candidates_dict : dict
        Either FVI_CANDIDATES or EVI_CANDIDATES — the source of truth for
        which variables belong to which domain.
    sig_csv_path : str
        Path to the stage12/17 output CSV containing the variables that
        passed bivariate screening at p < 0.05.

    Returns
    -------
    dict
        {domain_name: [list of column names]} for variables that passed
        bivariate screening. Domain names are stripped of the leading
        number-underscore prefix that appears in *_CANDIDATES (e.g.,
        '1_Motivation' -> 'Motivation').

    Raises
    ------
    FileNotFoundError
        If the bivariate output CSV does not exist (i.e., Stage 12/17
        has not been run yet). The error message identifies which stage
        must run first.
    """
    import pandas as pd
    if not os.path.exists(sig_csv_path):
        raise FileNotFoundError(
            f"Cannot derive *_DOMAINS — bivariate output not found at "
            f"{sig_csv_path}. Run stage12_fvi_bivariate.py (FVI) or "
            f"stage17_evi_bivariate.py (EVI) first."
        )
    sig_df = pd.read_csv(sig_csv_path)
    sig_cols = set(sig_df["Column"])
    domains = {}
    for domain_key, predictor_list in candidates_dict.items():
        # Strip the leading "N_" prefix used in *_CANDIDATES.
        domain_name = domain_key.split("_", 1)[1] if "_" in domain_key else domain_key
        passing_vars = [col for col, _, _ in predictor_list if col in sig_cols]
        if passing_vars:
            domains[domain_name] = passing_vars
    return domains


def _load_im_predictors(im_csv_path):
    """
    Load the final Individual Model variable list from the stepwise output.

    Parameters
    ----------
    im_csv_path : str
        Path to the stage13/18 Stage 2 stepwise output CSV.

    Returns
    -------
    list
        Column names of the variables retained in the final Individual
        Model (i.e., the predictors that survived both Stage 1 domain
        selection and Stage 2 joint backward elimination).

    Raises
    ------
    FileNotFoundError
        If the stepwise output CSV does not exist.
    """
    import pandas as pd
    if not os.path.exists(im_csv_path):
        raise FileNotFoundError(
            f"Cannot derive *_IM_PREDICTORS — stepwise output not found "
            f"at {im_csv_path}. Run stage13_fvi_stepwise.py (FVI) or "
            f"stage18_evi_stepwise.py (EVI) first."
        )
    im_df = pd.read_csv(im_csv_path)
    # The canonical *_im_predictors.csv files use the column name "predictor".
    # (Note: the bivariate output CSV uses capitalised "Column" — they differ
    # historically across pipeline versions.)
    return list(im_df["predictor"])


def assert_no_drift(stage_role="consumer"):
    """
    Runtime invariant check for pipeline integrity.

    Two invariants are checked:
      1. DOMAINS invariant: every variable in *_DOMAINS appears in
         *_CANDIDATES AND passed bivariate screening at p < 0.05.
      2. IM_PREDICTORS invariant: every variable in *_IM_PREDICTORS
         is also in *_DOMAINS.

    Stage role
    ----------
    The check that applies depends on what the calling stage does.

    "producer_fvi"  : Stage 13 (FVI stepwise). About to overwrite
                      fvi_im_predictors.csv. Skips the IM_PREDICTORS
                      check on BOTH sides because the FVI side is about
                      to be replaced and the EVI side may not yet
                      reflect the new candidate pool (Stage 18 runs
                      later).

    "producer_evi"  : Stage 18 (EVI stepwise). Symmetric to producer_fvi.

    "consumer_fvi"  : Stages 14, 15 (any FVI-side downstream stage).
                      Verifies the FVI invariants only. Skips EVI
                      checks because the EVI side may not yet be
                      refreshed (Stage 18 runs later in the pipeline).

    "consumer_evi"  : Stages 19, 20, 21 (any EVI-side downstream stage).
                      Symmetric to consumer_fvi.

    "consumer"      : Stages 22, 23, 24, 25, 26, 27 (cross-method or
                      composite stages). Verifies both invariants on
                      both sides. The strongest check.

    Parameters
    ----------
    stage_role : str
        One of "consumer" (default, both sides), "consumer_fvi",
        "consumer_evi", "producer_fvi", "producer_evi".

    Raises
    ------
    AssertionError
        If any drift is detected, with a descriptive message identifying
        which invariant fails on which side and which variable is at
        fault.
    """
    # Validate the stage_role argument so a typo fails loud.
    valid_roles = {"consumer", "consumer_fvi", "consumer_evi",
                   "producer_fvi", "producer_evi"}
    assert stage_role in valid_roles, (
        f"assert_no_drift: unknown stage_role {stage_role!r}. "
        f"Must be one of {valid_roles}."
    )

    # Decide which side(s) to check and whether to check IM_PREDICTORS.
    # The "side" controls which DOMAINS we verify; "check_im" controls
    # whether we also verify the IM_PREDICTORS invariant.
    fvi_side_active = stage_role in {"consumer", "consumer_fvi",
                                      "producer_fvi", "producer_evi"}
    evi_side_active = stage_role in {"consumer", "consumer_evi",
                                      "producer_fvi", "producer_evi"}

    # In producer modes, the IM_PREDICTORS check is skipped entirely:
    # the producer-side file is about to be overwritten, and the other
    # side may not yet reflect the new candidate pool.
    is_producer = stage_role in {"producer_fvi", "producer_evi"}
    check_im = not is_producer

    # FVI side
    if fvi_side_active:
        try:
            fvi_dom = _load_domains_from_bivariate(FVI_CANDIDATES, _BIVARIATE_FVI_CSV)
            fvi_dom_flat = {v for vlist in fvi_dom.values() for v in vlist}
            if check_im and os.path.exists(_STEPWISE_FVI_IM_CSV):
                fvi_im = _load_im_predictors(_STEPWISE_FVI_IM_CSV)
                drift = set(fvi_im) - fvi_dom_flat
                assert not drift, (
                    f"DRIFT: FVI_IM_PREDICTORS contains variables not in "
                    f"FVI_DOMAINS: {drift}. Re-run stage13 with the current "
                    f"candidate pool."
                )
        except FileNotFoundError as e:
            print(f"  assert_no_drift: FVI check skipped ({e})")

    # EVI side
    if evi_side_active:
        try:
            evi_dom = _load_domains_from_bivariate(EVI_CANDIDATES, _BIVARIATE_EVI_CSV)
            evi_dom_flat = {v for vlist in evi_dom.values() for v in vlist}
            if check_im and os.path.exists(_STEPWISE_EVI_IM_CSV):
                evi_im = _load_im_predictors(_STEPWISE_EVI_IM_CSV)
                drift = set(evi_im) - evi_dom_flat
                assert not drift, (
                    f"DRIFT: EVI_IM_PREDICTORS contains variables not in "
                    f"EVI_DOMAINS: {drift}. Re-run stage18 with the current "
                    f"candidate pool."
                )
        except FileNotFoundError as e:
            print(f"  assert_no_drift: EVI check skipped ({e})")

    # Describe what we just verified so the log is self-explanatory.
    if stage_role == "producer_fvi":
        print("  assert_no_drift: PASSED (producer_fvi mode — "
              "IM_PREDICTORS checks skipped because this stage will "
              "overwrite fvi_im_predictors.csv).")
    elif stage_role == "producer_evi":
        print("  assert_no_drift: PASSED (producer_evi mode — "
              "IM_PREDICTORS checks skipped because this stage will "
              "overwrite evi_im_predictors.csv).")
    elif stage_role == "consumer_fvi":
        print("  assert_no_drift: PASSED (consumer_fvi mode — "
              "FVI invariants verified; EVI side not checked).")
    elif stage_role == "consumer_evi":
        print("  assert_no_drift: PASSED (consumer_evi mode — "
              "EVI invariants verified; FVI side not checked).")
    else:
        print("  assert_no_drift: PASSED — no drift detected (both sides).")


def __getattr__(name):
    """
    PEP 562 module-level __getattr__ for lazy-loading the derived constants.

    This is invoked automatically by Python when downstream code writes
    `from pipeline_config import FVI_DOMAINS` (or any of the derived names).
    The CSV is read at first access, not at module import.
    """
    if name == "FVI_DOMAINS":
        return _load_domains_from_bivariate(FVI_CANDIDATES, _BIVARIATE_FVI_CSV)
    if name == "EVI_DOMAINS":
        return _load_domains_from_bivariate(EVI_CANDIDATES, _BIVARIATE_EVI_CSV)
    if name == "FVI_ALL_DOMAIN_VARS":
        return [v for vlist in
                _load_domains_from_bivariate(FVI_CANDIDATES, _BIVARIATE_FVI_CSV).values()
                for v in vlist]
    if name == "EVI_ALL_DOMAIN_VARS":
        return [v for vlist in
                _load_domains_from_bivariate(EVI_CANDIDATES, _BIVARIATE_EVI_CSV).values()
                for v in vlist]
    if name == "FVI_IM_PREDICTORS":
        return _load_im_predictors(_STEPWISE_FVI_IM_CSV)
    if name == "EVI_IM_PREDICTORS":
        return _load_im_predictors(_STEPWISE_EVI_IM_CSV)
    raise AttributeError(f"module 'pipeline_config' has no attribute {name!r}")


# -----------------------------------------------------------------------------
# FVI_PRED_LABELS — human-readable labels for FVI plots and tables
# -----------------------------------------------------------------------------
# Built by extending the shared labels with FVI-specific flood variables.

FVI_PRED_LABELS = {
    **_SHARED_LABELS,
    # Flood experience variables
    "DE01_flood":                       "Flood: first-mentioned major hazard",
    "DE02_flood":                       "Flood: second-mentioned major hazard",
    "DE03_flood_frequency_score":       "Flood frequency score",
    "DE04_flood_depth_score":           "Flood depth score",
    "DE05_negative_impact_score":       "Past flood impact score",
    "DE06_flood_future_score":          "Future flood expectation",
    "DE07_negative_impact_future_score":"Future flood impact score",
    "DE08_worry_score":                 "Current flood worry score",
    "DE09_worry_future_score":          "Future flood worry",
    "DE12_shelter_score":               "Shelter/evacuation plan score",
    "DE13_feel_safe_score":             "Feel safe from flooding",
    # PP02 (flood rebuild history) 3-level dummies
    "PP02_flood_rebuild_3lvl_damaged":       "Flood rebuild: damaged",
    "PP02_flood_rebuild_3lvl_unknown_other": "Flood rebuild: unknown/other",
}

# =============================================================================
# EVI — EARTHQUAKE VULNERABILITY INDEX
# =============================================================================

# -----------------------------------------------------------------------------
# EVI_CANDIDATES — full candidate pool for bivariate screening (Stage 9)
# -----------------------------------------------------------------------------
# Motivation, Ability, Opportunity, and Socioeconomic domains are identical
# to FVI. The hazard-experience domain is replaced with EQ-specific variables.

EVI_CANDIDATES = {

    "1_Motivation": [
        ("Uti_Perc_Pos_expression",  "Utility perception — Positive",        "pearson"),
        ("Uti_Perc_Neg_expression",  "Utility perception — Negative",         "pearson"),
        ("App_Perc_Neg_expression",  "Applicability perception — Negative",   "pearson"),
        ("Acc_Perc_Pos_expression",  "Acceptability perception — Positive",   "pearson"),
    ],

    "2_Ability": [
        ("AB_Selfefficacy_Pos",        "Self-efficacy — Positive",         "pearson"),
        ("AB_Selfefficacy_Neg",        "Self-efficacy — Negative",         "pearson"),
        ("AB_Physical_capacity_Pos",   "Physical capacity — Positive",     "pearson"),
        ("AB_Physical_capacity_Neg",   "Physical capacity — Negative",     "pearson"),
        ("AB_Financial_capacity_Pos",  "Financial capacity — Positive",    "pearson"),
        ("AB_Financial_capacity_Neg",  "Financial capacity — Negative",    "pearson"),
        ("AB_Location_Pos",            "Location suitability — Positive",  "pearson"),
        ("AB_Time_Pos",                "Time available — Positive",        "pearson"),
        ("AB_Time_Neg",                "Time available — Negative",        "pearson"),
    ],

    "3_Opportunity": [
        ("OP_Training_Pos",  "Training access — Positive",      "pearson"),
        ("OP_Training_Neg",  "Training access — Negative",      "pearson"),
        ("OP_Materials_Pos", "Materials — Positive",            "pearson"),
        ("OP_Materials_Neg", "Materials — Negative",            "pearson"),
        ("OP_Location_Pos",  "Location externally — Positive",  "pearson"),
        ("OP_Location_Neg",  "Location externally — Negative",  "pearson"),
    ],

    # Earthquake experience domain for EVI.
    # PP12 (land ownership) is NOT a candidate here because the Motivation MAO
    # syntax (Hendriks 2025) uses all 11 PP12 dummies in the Utility composite
    # denominator. Adding PP12 standalone would introduce by-construction
    # collinearity with the Motivation MAO composites.
    # PP01 (earthquake rebuild history) IS included as a 3-level collapsed
    # categorical, verified against Syntax_Motivation_02_01_25__4_.sps to not
    # appear in any Motivation MAO composite. The Ability and Opportunity
    # composite syntax files were not available for verification and this is
    # acknowledged as a residual methodological uncertainty in Section 5.4
    # (Limitations) of the thesis.
    "4_Earthquake_Experience": [
        ("DE01_earthquake",
         "Earthquake: first-mentioned major hazard",  "spearman"),
        ("DE02_earthquake",
         "Earthquake: second-mentioned major hazard", "spearman"),
        # PP01 (earthquake rebuild history) — 3-level collapsed categorical.
        # Constructed by stage08b_ward_covariates.py from the parent field
        # PP01_rebuild_repair_house_001. Reference category = "undamaged".
        ("PP01_eq_rebuild_3lvl_damaged",
         "EQ rebuild history: damaged (vs undamaged)",      "spearman"),
        ("PP01_eq_rebuild_3lvl_unknown_other",
         "EQ rebuild history: unknown/other (vs undamaged)", "spearman"),
    ],

    "5_Socioeconomic": [
        ("HC003_Respondent_s_age",        "Respondent age",                "pearson"),
        ("HC004_amount_people_household", "Household size",                "pearson"),
        ("SE01_edu_no_education",         "Education: none",               "spearman"),
        ("SE01_edu_can_read_write",       "Education: can read/write",     "spearman"),
        ("SE01_edu_elementary_school",    "Education: elementary",         "spearman"),
        ("SE01_edu_high_school",          "Education: high school",        "spearman"),
        ("SE01_edu_university",           "Education: university",         "spearman"),
        ("SE04_agriculture",              "Occupation: agriculture",       "spearman"),
        ("SE04_day_labour",               "Occupation: day labour",        "spearman"),
        ("SE04_construction_worker",      "Occupation: construction worker","spearman"),
        ("SE04_education",                "Occupation: education",         "spearman"),
        ("SE04_remittances",              "Income: remittances",           "spearman"),
        ("SE04_business",                 "Occupation: business",          "spearman"),
        ("SE04_government_officer",           "Occupation: government",          "spearman"),
    ],
}

# -----------------------------------------------------------------------------
# EVI_DOMAINS, EVI_ALL_DOMAIN_VARS, EVI_IM_PREDICTORS
# Same architecture as the FVI counterparts above: derived at runtime from
# pipeline output files via the module-level __getattr__ defined earlier.
# See the FVI comment block (above) for full documentation.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# EVI_PRED_LABELS — human-readable labels for EVI plots and tables
# -----------------------------------------------------------------------------

EVI_PRED_LABELS = {
    **_SHARED_LABELS,
    # Earthquake experience variables
    "DE01_earthquake":                               "Earthquake: first-mentioned major hazard",
    "DE02_earthquake":                               "Earthquake: second-mentioned major hazard",
    # PP01 (earthquake rebuild history) 3-level dummies
    "PP01_eq_rebuild_3lvl_damaged":                  "EQ rebuild: damaged",
    "PP01_eq_rebuild_3lvl_unknown_other":            "EQ rebuild: unknown/other",
}

# =============================================================================
# PIPELINE-WIDE ANALYTICAL CONSTANTS
# =============================================================================
# All analytical parameters that affect results are defined once here.
# Every stage script imports what it needs — nothing is hardcoded in scripts.

# ── Survey metadata column names ──────────────────────────────────────────────
SURVEYOR_COL  = "GD001_Name_of_the_surveyor"  # enumerator identity
COMMUNITY_COL = "GD005_VDC_name"              # VDC/ward community identifier

# Ward-level covariate columns added by Stage 08b.
# Both are Level-2 predictors (constant within ward) for Stages 15 and 21.
WARD_TRANSPORT_COL  = "transport_access_score"    # proportion of 11 access items
WARD_ASSISTANCE_COL = "assistance_intensity_score" # proportion of 10 assistance items

# ── Vulnerability index column names ─────────────────────────────────────────
FVI_SCORE_COL  = "FVI_norm_1_5"   # outcome column for FVI regressions
EVI_SCORE_COL  = "EVI_norm_1_5"   # outcome column for EVI regressions
CVI_SCORE_COL  = "CVI_norm_1_5"   # composite index column

# ── Vulnerability class boundaries (1–5 scale) ───────────────────────────────
# Low = [1.000, BNDRY_LM),  Medium = [BNDRY_LM, BNDRY_MH),  High = [BNDRY_MH, 5.000]
# Based on equal-width tertile split of the 1–5 range.
BNDRY_LM = 1.0 + (1.0 / 3.0) * 4.0   # 2.333 — Low/Medium boundary
BNDRY_MH = 1.0 + (2.0 / 3.0) * 4.0   # 3.667 — Medium/High boundary

# ── Stepwise regression p-value thresholds ───────────────────────────────────
# Applied consistently in Stages 07/08 (FVI) and 12/13 (EVI).
# PIN:  variable enters the model if p < PIN  (forward step)
# POUT: variable is removed if p > POUT after entry (backward step)
# Reference: Hendriks & Stokmans (2020); Saputra et al. (2026)
STEPWISE_PIN  = 0.05   # entry threshold
STEPWISE_POUT = 0.10   # removal threshold

# ── Moran's I and LISA distance threshold (km) ───────────────────────────────
# Reporting threshold for spatial autocorrelation tests.
# 1.0 km represents the village-to-village scale in Nepal's Terai settlement
# pattern. Sensitivity analysis across 0.25–10 km is always reported.
MORANS_THRESHOLD_KM = 1.0

# ── CVI component weights ─────────────────────────────────────────────────────
# Equal weighting following CIMDEN methodology (Villagrán De León 2006).
# Sensitivity to variance-normalised weights is tested in Stage 17.
FVI_WEIGHT = 0.5
EVI_WEIGHT = 0.5

# ── Random seed for reproducibility (FAIR principle) ─────────────────────────
RANDOM_STATE = 42
