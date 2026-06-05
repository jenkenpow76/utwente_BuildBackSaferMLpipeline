"""
stage27_cross_method_comparison.py
==================================

Stage 20 | Cross-Method Predictor Agreement (Four-Method)
==========================================================

Purpose
-------
This stage produces the cross-method agreement table that operationalises the
contribution claim made in Chapter 1: "cross-model agreement identifies the
most defensible targets for reconstruction-assistance design, while cross-model
divergence identifies factors where qualitative follow-up is needed before
programmatic action."

For each of the two single-hazard indices (FVI, EVI), this stage joins the
predictor-level results of four analytical methods used elsewhere in the
pipeline:

    1. Ordinary Least Squares (OLS)         — read from the OLS/ME comparison
                                              files (fvi_mixed_effects_comparison.csv,
                                              evi_mixed_effects_comparison.csv).
                                              These files contain the final OLS
                                              regression on the post-stepwise
                                              predictor set, NOT the stepwise
                                              selection results themselves. The
                                              OLS used here is estimated on
                                              exactly the same predictor list
                                              and sample as the mixed-effects
                                              model, which makes the OLS vs ME
                                              contrast a pure test of clustering
                                              correction (same predictors, same
                                              data, different standard-error
                                              treatment).
    2. Linear mixed-effects regression (ME) — Stage 15 for FVI, Stage 21 for EVI.
                                              Estimated on the post-stepwise
                                              predictor set.
    3. Random Forest (RF)                   — Stage 23 permutation importance.
                                              RF is the one method that uses the
                                              FULL candidate pool, not just the
                                              stepwise-selected list, because RF
                                              ranks all candidates by importance.
                                              Predictors in the RF table that
                                              are absent from the ME/OLS/GAM
                                              tables are flagged with
                                              me_present=False, ols_present=False,
                                              gam_present=False in the output.
    4. Generalised Additive Model (GAM)     — Stage 25 term summaries. Estimated
                                              on the post-stepwise predictor set.

The output for each index is a single table with one row per predictor. Each
column captures one piece of information from one method. Two agreement scores
per predictor count how many of the four methods flag the predictor as
important, on a strict and a lenient definition.

This is a *post-pipeline synthesis* stage. It introduces no new statistical
estimation. It only joins existing per-predictor results so that interpretation
in Chapter 4 has a single consolidated artefact to reference.

The 2x2 design — clustering × functional form
----------------------------------------------
The four methods sit at the corners of a 2x2 design:

                              Clustering correction
                            Yes (ME)        No  (OLS, RF, GAM)
                          +-------------+-----------------------+
    Functional   Linear   | ME          | OLS                   |
    form         Non-     | (none in    | RF (tree-based)       |
                 linear   |  this       | GAM (spline-based)    |
                          |  pipeline)  |                       |
                          +-------------+-----------------------+

Each pairwise contrast isolates one analytical assumption:

    OLS vs ME       — does clustering correction matter for this predictor?
                      If OLS flags it but ME does not, the OLS finding was
                      a clustering artefact (standard errors too small).
    OLS vs GAM      — does linearity matter? If the GAM smooth detects
                      a non-linear effect that OLS averages over, the linear
                      coefficient is partly misleading.
    OLS vs RF       — do interactions matter? RF detects interactions
                      automatically; OLS does not.
    ME vs RF        — does the predictor survive both honest clustering
                      correction AND tests of interaction-rich effects?
    ME vs GAM       — does the predictor survive both honest clustering
                      correction AND relaxation of linearity?
    RF vs GAM       — is the non-linearity isolated to this predictor
                      (GAM) or interaction-driven (RF)?

Agreement counts 0-4 instead of 0-3 give a richer interpretive framework:

    4-of-4 strict  — robust to clustering, linearity, AND interaction
                     assumptions; strongest possible target for assistance
                     design.
    3-of-4         — the pattern of which method disagrees identifies
                     which assumption matters.
    2-of-4         — typically (OLS+ME) or (RF+GAM) together; tells you
                     whether the effect is linear-only or non-linear.
    1-of-4         — sensitive to one specific assumption; weak finding.
    0-of-4         — no evidence of association under any method.

Inputs (read from the standard pipeline output locations)
----------------------------------------------------------
    fvi_mixed_effects_comparison.csv       — OLS + ME side by side, FVI
    evi_mixed_effects_comparison.csv       — OLS + ME side by side, EVI
    fvi_mixed_effects_results.csv          — Stage 15 ME-only fixed-effects
    evi_mixed_effects_results.csv          — Stage 21 ME-only fixed-effects
    random_forest/rf_fvi_importance.csv    — Stage 23 permutation importance, FVI
    random_forest/rf_evi_importance.csv    — Stage 23 permutation importance, EVI
    gam/gam_fvi_summary.csv                — Stage 25 GAM term statistics, FVI
    gam/gam_evi_summary.csv                — Stage 25 GAM term statistics, EVI

Outputs (written to outputs/cross_method/)
-------------------------------------------
    cross_method_fvi.csv                   — predictor-by-method agreement, FVI
    cross_method_evi.csv                   — predictor-by-method agreement, EVI
    cross_method_summary.txt               — counts of agreement levels, both indices

Methodology
-----------
For each predictor and each method, a boolean flag is computed:

    ols_significant                = (p_OLS < 0.05)
    me_significant                 = (p_ME  < 0.05)
    rf_top_decile                  = rank_rf <= ceil(n_predictors / 10)
                                     AND importance >= 0.005
    gam_significant                = significant == True

A predictor's *strict agreement* is the count (0-4) of methods that flag it
under the rules above. The strict criterion uses the published thresholds
in each method's standard reporting:

    - p < 0.05 is the conventional significance threshold for regression
      (Fisher, 1925; commonly applied in social-science research).
    - Top-decile importance is a conservative threshold for Random Forest
      permutation importance (Strobl et al., 2007, BMC Bioinformatics, 8, 25;
      https://doi.org/10.1186/1471-2105-8-25).
    - p < 0.05 on the GAM term test is the standard significance criterion
      for additive models (Wood, 2017, Generalized Additive Models, 2e).

A predictor's *lenient agreement* relaxes the RF threshold to top-third (top
33% of predictors by permutation importance), to allow detection of factors
that are consistently meaningful but not in the top decile of any single
method.

The output is *descriptive*: it summarises what each method already reported
about each predictor. Interpretation of which predictors should be targeted
for assistance design is the work of Chapter 4 and Chapter 5.

References
----------
    Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007).
        Bias in random forest variable importance measures: Illustrations,
        sources and a solution. BMC Bioinformatics, 8, 25.
        https://doi.org/10.1186/1471-2105-8-25

    Wood, S. N. (2017). Generalized Additive Models: An Introduction with R
        (2nd ed.). Chapman & Hall/CRC.
        https://doi.org/10.1201/9781315370279

    Fisher, R. A. (1925). Statistical Methods for Research Workers.
        Oliver and Boyd.

    Snijders, T. A. B., & Bosker, R. J. (2012). Multilevel Analysis: An
        Introduction to Basic and Advanced Multilevel Modeling (2nd ed.).
        SAGE Publications.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import sys
import math
import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

# Significance threshold for the mixed-effects p-value and the GAM term p-value.
# 0.05 is the conventional threshold in social-science reporting.
ALPHA = 0.05

# Random Forest importance thresholds.
# Strict: top decile (top 10% by permutation importance rank).
# Lenient: top third (top 33% by permutation importance rank).
RF_STRICT_QUANTILE = 0.10
RF_LENIENT_QUANTILE = 0.33

# Predictors with permutation importance below this absolute value are not
# considered, even if their rank is in the top decile. This guards against
# the case where all permutation importances are near zero (the model has
# weak signal) and the top decile is therefore not meaningful.
RF_MIN_ABS_IMPORTANCE = 0.005


# =============================================================================
# PATHS
# =============================================================================

def _find_outputs_dir() -> Path:
    """
    Return the directory containing the pipeline outputs.

    Stage 27 can be invoked in three ways, exactly like stages 23/25/26:
      (a) Via run_pipeline.py — cwd is already outputs/, where the required
          input files (e.g. fvi_mixed_effects_results.csv) live directly in
          cwd. This is the path run_pipeline.py takes because it sets
          working_dir=OUTPUT_DIR before executing each stage script.
      (b) Standalone from the pipeline_v46 root — cwd is pipeline_v46/, so
          the outputs live in cwd/outputs/.
      (c) From any other location — fall back to script_dir/outputs/.

    The function looks for fvi_mixed_effects_results.csv as the marker file.
    That file is one of stage 27's required inputs and is reliably present
    in the outputs directory once stages 15 has run. Using a marker file
    rather than a marker directory matches the pattern used by stages 23,
    25, and 26, which makes the path resolution robust across invocation
    modes.
    """
    marker = "fvi_mixed_effects_results.csv"
    candidates = [
        Path.cwd(),                                     # (a) run_pipeline.py
        Path.cwd() / "outputs",                          # (b) pipeline root
        Path(__file__).resolve().parent / "outputs",     # (c) absolute fallback
    ]
    for d in candidates:
        if (d / marker).exists():
            return d
    # Fall through to a clear error so the user sees which paths were tried.
    tried = "\n  ".join(str(c) for c in candidates)
    sys.exit(
        f"ERROR: could not find pipeline outputs directory.\n"
        f"Looked for '{marker}' in:\n  {tried}\n"
        f"Run this stage from the pipeline_v46 folder, from its parent "
        f"directory, or via run_pipeline.py."
    )


OUTPUTS_DIR = _find_outputs_dir()
OUT_DIR = OUTPUTS_DIR / "cross_method"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INPUT FILE LOCATIONS
# =============================================================================

# One dict per single-hazard index, mapping a method label to its CSV file.
# Centralising the file locations here makes it easy to update if the pipeline
# output structure changes.
#
# The 'ols_me_comparison' file is produced by Stages 15 and 21. It contains
# OLS and mixed-effects estimates side by side for the same predictor set,
# which makes the apples-to-apples comparison straightforward.
INPUTS = {
    "FVI": {
        "ols_me_comparison": OUTPUTS_DIR / "fvi_mixed_effects_comparison.csv",
        "mixed_effects":     OUTPUTS_DIR / "fvi_mixed_effects_results.csv",
        "rf_importance":     OUTPUTS_DIR / "random_forest" / "rf_fvi_importance.csv",
        "gam_summary":       OUTPUTS_DIR / "gam" / "gam_fvi_summary.csv",
    },
    "EVI": {
        "ols_me_comparison": OUTPUTS_DIR / "evi_mixed_effects_comparison.csv",
        "mixed_effects":     OUTPUTS_DIR / "evi_mixed_effects_results.csv",
        "rf_importance":     OUTPUTS_DIR / "random_forest" / "rf_evi_importance.csv",
        "gam_summary":       OUTPUTS_DIR / "gam" / "gam_evi_summary.csv",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_csv(path: Path) -> pd.DataFrame:
    """
    Read a pipeline CSV with the same encoding used elsewhere in pipeline_v46.

    The pipeline writes UTF-8 files with a BOM marker. Reading with
    'utf-8-sig' strips the BOM cleanly. The low_memory option is False to
    avoid mixed-dtype warnings on wide files.
    """
    if not path.exists():
        sys.exit(f"ERROR: required input file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def prepare_ols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-predictor OLS significance from the OLS/ME comparison table.

    The comparison file (fvi_mixed_effects_comparison.csv,
    evi_mixed_effects_comparison.csv) is produced by Stages 15 and 21. It
    contains both OLS and mixed-effects estimates for the same predictor
    set, which guarantees that the OLS coefficients in this table are
    estimated on exactly the predictor selection used by the mixed-effects
    model. This is the apples-to-apples comparison.

    The output columns mirror the mixed-effects columns so the agreement
    table reads consistently across the two linear methods:

        variable                 — predictor name (joins to RF/GAM 'variable')
        label                    — human-readable label
        ols_beta                 — OLS coefficient
        ols_p                    — OLS p-value (uncorrected for clustering)
        ols_significant          — bool, True if ols_p < ALPHA
        ols_sign                 — '+', '-', or '0' depending on coefficient sign

    The 'Intercept' row is dropped because it is not a predictor.

    Note on the clustering interpretation: OLS p-values do not correct for
    community-level clustering. A predictor flagged by OLS but not by ME
    indicates that the OLS finding is a clustering artefact. The agreement
    table preserves both so this contrast is visible.
    """
    df = df.copy()
    df = df[df["parameter"] != "Intercept"].copy()
    df = df.rename(columns={"parameter": "variable"})
    df["ols_beta"] = pd.to_numeric(df["beta_OLS"], errors="coerce")
    df["ols_p"] = pd.to_numeric(df["p_OLS"], errors="coerce")
    df["ols_significant"] = df["ols_p"] < ALPHA
    df["ols_sign"] = np.where(df["ols_beta"] > 0, "+",
                     np.where(df["ols_beta"] < 0, "-", "0"))
    # Rename label so the coalesce step downstream can use it as a fallback
    # when ME does not have a label (parallel structure to rf_label, gam_label).
    return df[["variable", "label", "ols_beta", "ols_p",
               "ols_significant", "ols_sign"]].rename(
        columns={"label": "ols_label"}
    )


def prepare_mixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-predictor significance from a mixed-effects fixed-effects table.

    The output has columns:
        variable                 — predictor name (joins to RF/GAM 'variable')
        label                    — human-readable label
        me_beta                  — fixed-effect coefficient
        me_p                     — p-value
        me_significant           — bool, True if me_p < ALPHA
        me_sign                  — '+', '-', or '0' depending on coefficient sign

    The 'Intercept' row is dropped because it is not a predictor.
    """
    df = df.copy()
    # Drop the intercept row — it is not a predictor and has no comparison.
    df = df[df["parameter"] != "Intercept"].copy()
    # Rename the predictor column for the join with RF and GAM tables.
    df = df.rename(columns={"parameter": "variable"})
    df["me_beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["me_p"] = pd.to_numeric(df["p"], errors="coerce")
    df["me_significant"] = df["me_p"] < ALPHA
    df["me_sign"] = np.where(df["me_beta"] > 0, "+",
                    np.where(df["me_beta"] < 0, "-", "0"))
    return df[["variable", "label", "me_beta", "me_p",
               "me_significant", "me_sign"]]


def prepare_rf(df: pd.DataFrame, n_predictors: int) -> pd.DataFrame:
    """
    Compute strict and lenient importance flags from a Random Forest
    permutation importance table.

    The strict flag is True if:
        - rank_rf is within the top RF_STRICT_QUANTILE of all ranks, AND
        - importance_mean is at least RF_MIN_ABS_IMPORTANCE.

    The lenient flag relaxes the rank threshold to RF_LENIENT_QUANTILE.

    The absolute-importance gate guards against the degenerate case where the
    model has weak overall signal, in which case the top decile contains
    predictors whose importance values are not meaningfully different from
    zero.
    """
    df = df.copy()
    df["rf_importance"] = pd.to_numeric(df["importance_mean"], errors="coerce")
    df["rf_rank"] = pd.to_numeric(df["rank_rf"], errors="coerce")

    # Number of predictors in the top decile (rounded up so small predictor
    # sets retain at least one slot in the decile).
    n_top_strict = max(1, math.ceil(n_predictors * RF_STRICT_QUANTILE))
    n_top_lenient = max(1, math.ceil(n_predictors * RF_LENIENT_QUANTILE))

    df["rf_top_decile"] = (
        (df["rf_rank"] <= n_top_strict)
        & (df["rf_importance"] >= RF_MIN_ABS_IMPORTANCE)
    )
    df["rf_top_third"] = (
        (df["rf_rank"] <= n_top_lenient)
        & (df["rf_importance"] >= RF_MIN_ABS_IMPORTANCE)
    )

    return df[["variable", "label", "rf_importance", "rf_rank",
               "rf_top_decile", "rf_top_third"]].rename(
        columns={"label": "rf_label"}
    )


def prepare_gam(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-predictor significance and non-linearity flags from the GAM
    term summary table.

    The output columns capture both whether a term is statistically detected
    (significant in the GAM test) and whether it shows non-linear effects
    (non_linear flag from Stage 25, set when EDF > 1.5).
    """
    df = df.copy()
    df["gam_edf"] = pd.to_numeric(df["edf"], errors="coerce")
    df["gam_p"] = pd.to_numeric(df["p_value"], errors="coerce")
    # The 'significant' column in stage25 output is already a boolean.
    df["gam_significant"] = df["significant"].astype(bool)
    df["gam_non_linear"] = df["non_linear"].astype(bool)
    return df[["variable", "label", "gam_edf", "gam_p",
               "gam_significant", "gam_non_linear"]].rename(
        columns={"label": "gam_label"}
    )


def build_agreement_table(index_name: str) -> pd.DataFrame:
    """
    Build the cross-method agreement table for a single index (FVI or EVI).

    The procedure is:
        1. Load and shape each of the four method outputs for this index.
        2. Outer-join on 'variable' so no predictor is lost.
        3. Compute strict and lenient agreement counts (0-4).
        4. Sort by strict agreement (descending), then by RF importance.
    """
    src = INPUTS[index_name]

    ols = prepare_ols(load_csv(src["ols_me_comparison"]))
    me = prepare_mixed_effects(load_csv(src["mixed_effects"]))
    rf_raw = load_csv(src["rf_importance"])
    rf = prepare_rf(rf_raw, n_predictors=len(rf_raw))
    gam = prepare_gam(load_csv(src["gam_summary"]))

    # Outer join keeps a predictor in the output even if one method did not
    # estimate it. Some predictors are mixed-effects only (categorical
    # baselines) and others are RF only (RF runs on all candidates, not just
    # the stepwise-selected ones). The output therefore reflects the union
    # of variables seen by any method.
    #
    # We start with OLS+ME (which share the stepwise predictor set), then
    # outer-join RF and GAM. This preserves all predictors in any of the
    # four methods.
    merged = ols.merge(me, on="variable", how="outer")
    merged = merged.merge(rf, on="variable", how="outer")
    merged = merged.merge(gam, on="variable", how="outer")

    # Coalesce the label across all four sources. Mixed-effects 'label' is
    # preferred when present because the ME table uses the published
    # human-readable labels from Stage 15/21. OLS, RF, and GAM labels are
    # used to backfill predictors that those methods saw but ME did not.
    # OLS comes first in the fallback chain because OLS comes from the
    # same comparison table as ME and uses identical labels.
    merged["label"] = (
        merged["label"]
        .fillna(merged["ols_label"])
        .fillna(merged["rf_label"])
        .fillna(merged["gam_label"])
    )
    # Final fallback: if no method had a label (should not happen, but
    # safe to defend against), use the variable name itself so the report
    # is never blank.
    merged["label"] = merged["label"].fillna(merged["variable"])
    merged = merged.drop(columns=["ols_label", "rf_label", "gam_label"])

    # Fill the boolean and presence flags. Missing means the method did
    # not estimate this predictor, which is informationally distinct from
    # 'estimated but not significant'. The 'present' columns make this
    # distinction visible.
    merged["ols_present"] = merged["ols_p"].notna()
    merged["me_present"] = merged["me_p"].notna()
    merged["rf_present"] = merged["rf_importance"].notna()
    merged["gam_present"] = merged["gam_p"].notna()

    for col in ("ols_significant", "me_significant",
                "rf_top_decile", "rf_top_third",
                "gam_significant", "gam_non_linear"):
        merged[col] = merged[col].fillna(False).astype(bool)

    # Agreement counts: how many of the four methods flag this predictor.
    # Strict uses RF top decile; lenient uses RF top third.
    # The maximum agreement count is now 4 (OLS, ME, RF, GAM).
    merged["agreement_strict"] = (
        merged["ols_significant"].astype(int)
        + merged["me_significant"].astype(int)
        + merged["rf_top_decile"].astype(int)
        + merged["gam_significant"].astype(int)
    )
    merged["agreement_lenient"] = (
        merged["ols_significant"].astype(int)
        + merged["me_significant"].astype(int)
        + merged["rf_top_third"].astype(int)
        + merged["gam_significant"].astype(int)
    )

    # Order columns for readability. Methods are grouped together so a
    # reader scanning the CSV can see each method's full result in one
    # contiguous block.
    out = merged[[
        "variable", "label",
        "ols_present", "ols_beta", "ols_p", "ols_sign", "ols_significant",
        "me_present", "me_beta", "me_p", "me_sign", "me_significant",
        "rf_present", "rf_importance", "rf_rank",
        "rf_top_decile", "rf_top_third",
        "gam_present", "gam_edf", "gam_p",
        "gam_significant", "gam_non_linear",
        "agreement_strict", "agreement_lenient",
    ]].copy()

    # Sort so that predictors flagged by all four methods come first, then
    # within agreement level, sort by RF importance descending so the most
    # influential predictor surfaces at the top of each agreement tier.
    out = out.sort_values(
        ["agreement_strict", "rf_importance"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    return out


def _format_predictor_line(row: pd.Series, rf_flag_col: str) -> str:
    """
    Render one predictor as a single readable line for the summary report.

    The line shows:
      - the coefficient sign from the mixed-effects model (+, -, 0, or ?).
        If ME did not estimate this predictor, the OLS sign is used as
        fallback. If neither estimated it, '?' is shown.
      - the human-readable label
      - which methods flagged the predictor, in a fixed flag list
        ('OLS', 'ME', 'RF', 'GAM' joined by '+')
      - the RF rank if the predictor was scored by RF
      - the mixed-effects beta if the predictor was estimated by ME
        (or the OLS beta as fallback)

    `rf_flag_col` selects whether the strict (rf_top_decile) or lenient
    (rf_top_third) RF threshold is being reported.
    """
    # Prefer the ME sign; fall back to OLS sign if ME did not estimate this
    # predictor (some predictors are RF-only because they were not stepwise
    # selected, but OLS does not estimate them either, so this fallback is
    # mainly for completeness).
    if row["me_present"]:
        sign = row["me_sign"]
    elif row["ols_present"]:
        sign = row["ols_sign"]
    else:
        sign = "?"

    # Build the list of methods that flagged this predictor under the
    # currently active threshold set. Fixed order OLS, ME, RF, GAM mirrors
    # the 2x2 design described in the module docstring (clustering on the
    # vertical axis, functional form on the horizontal).
    flagged_by = []
    if row["ols_significant"]:
        flagged_by.append("OLS")
    if row["me_significant"]:
        flagged_by.append("ME")
    if row[rf_flag_col]:
        flagged_by.append("RF")
    if row["gam_significant"]:
        flagged_by.append("GAM")
    flags = "+".join(flagged_by) if flagged_by else "(none)"

    # RF rank — only show if RF estimated the predictor.
    rank_str = (f"RF rank {int(row['rf_rank'])}"
                if pd.notna(row["rf_rank"]) else "RF rank —")
    # Coefficient — prefer ME beta; fall back to OLS beta.
    if pd.notna(row["me_beta"]):
        beta_str = f"ME beta {row['me_beta']:+.3f}"
    elif pd.notna(row["ols_beta"]):
        beta_str = f"OLS beta {row['ols_beta']:+.3f}"
    else:
        beta_str = "beta —"

    label = str(row["label"])
    return f"    [{sign}] {label}  [{flags}]  ({rank_str}, {beta_str})"


def _list_predictors_at_level(table: pd.DataFrame, agreement_col: str,
                              rf_flag_col: str, level: int) -> list:
    """
    Return a list of formatted lines for every predictor at the given
    agreement level under the given threshold set (strict or lenient).

    Predictors are sorted by mixed-effects beta magnitude descending so the
    most influential predictor appears first within each level. Ties are
    broken by RF importance descending. If ME did not estimate the
    predictor, the OLS beta is used as the sort fallback.
    """
    sub = table[table[agreement_col] == level].copy()
    if sub.empty:
        return []

    # Sort by absolute ME beta magnitude (descending), falling back to OLS
    # beta where ME is missing. Tie-breaker is RF importance.
    sub["me_beta_abs"] = sub["me_beta"].abs()
    sub["ols_beta_abs"] = sub["ols_beta"].abs()
    sub["sort_beta"] = sub["me_beta_abs"].fillna(sub["ols_beta_abs"]).fillna(0)
    sub["rf_importance_sort"] = sub["rf_importance"].fillna(0)
    sub = sub.sort_values(
        ["sort_beta", "rf_importance_sort"],
        ascending=[False, False],
    )
    return [_format_predictor_line(row, rf_flag_col)
            for _, row in sub.iterrows()]


def write_summary(tables: dict, path: Path) -> None:
    """
    Write a plain-text summary of the agreement counts for both indices.

    For each index, the summary lists predictor names at every agreement
    level (4-of-4, 3-of-4, 2-of-4, 1-of-4) under both strict and lenient
    thresholds. Predictors at the 0-of-4 level are reported as a count only,
    because listing every predictor that no method flagged would add many
    lines without analytical use.

    The summary mirrors the style of gam_comparison_summary.txt so it slots
    naturally into the pipeline's existing reporting outputs.

    Format key for the [OLS+ME+RF+GAM] flag column:
        OLS = significant in ordinary least squares (p < ALPHA, naive on
              clustering; standard errors too small)
        ME  = significant in mixed-effects regression (p < ALPHA;
              clustering-corrected standard errors)
        RF  = within the active Random Forest importance threshold
              (top decile in strict view, top third in lenient view)
        GAM = significant in the GAM term test (p < ALPHA)
    """
    lines = []
    lines.append("CROSS-METHOD AGREEMENT SUMMARY (FOUR METHODS)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "Counts the number of predictors flagged by each combination of the "
        "four analytical methods (OLS, mixed-effects regression, Random "
        "Forest, Generalised Additive Model). Each predictor is listed "
        "below its agreement level."
    )
    lines.append("")
    lines.append("The four methods sit at the corners of a 2x2 design:")
    lines.append("    OLS  : linear, no clustering correction")
    lines.append("    ME   : linear, clustering-corrected standard errors")
    lines.append("    RF   : non-linear with interactions, no clustering correction")
    lines.append("    GAM  : non-linear additive, no clustering correction")
    lines.append("Pairwise contrasts isolate one analytical assumption each.")
    lines.append("See the module docstring for the full interpretation guide.")
    lines.append("")
    lines.append(f"Significance threshold (OLS, ME, GAM): p < {ALPHA}")
    lines.append(
        f"Random Forest strict threshold: top "
        f"{int(RF_STRICT_QUANTILE * 100)}% by permutation importance, "
        f"importance >= {RF_MIN_ABS_IMPORTANCE}"
    )
    lines.append(
        f"Random Forest lenient threshold: top "
        f"{int(RF_LENIENT_QUANTILE * 100)}% by permutation importance, "
        f"importance >= {RF_MIN_ABS_IMPORTANCE}"
    )
    lines.append("")
    lines.append(
        "Format per predictor: [sign] Label  [methods flagging]  "
        "(RF rank, beta)"
    )
    lines.append(
        "  Sign is the direction of the mixed-effects coefficient "
        "(or OLS if ME did not estimate this predictor). "
        "Positive means the predictor is associated with higher "
        "vulnerability. Negative means lower vulnerability."
    )
    lines.append("")

    # Iterate over both indices (FVI, EVI) and, within each, over both
    # threshold definitions (strict, lenient). Each view lists predictors at
    # agreement levels 4, 3, 2, 1.
    for index_name, table in tables.items():
        lines.append("=" * 70)
        lines.append(f"INDEX: {index_name}")
        lines.append("=" * 70)
        lines.append(f"Predictors covered: {len(table)}")
        lines.append("")

        for view_label, agreement_col, rf_flag_col in (
            ("STRICT (RF top decile)",  "agreement_strict",  "rf_top_decile"),
            ("LENIENT (RF top third)",  "agreement_lenient", "rf_top_third"),
        ):
            lines.append("-" * 70)
            lines.append(f"View: {view_label}")
            lines.append("-" * 70)

            for level, level_label in (
                (4, "Flagged by all 4 methods"),
                (3, "Flagged by 3 of 4 methods"),
                (2, "Flagged by 2 of 4 methods"),
                (1, "Flagged by 1 of 4 methods"),
            ):
                n_at_level = int((table[agreement_col] == level).sum())
                lines.append("")
                lines.append(f"  {level_label}: {n_at_level} predictors")
                if n_at_level > 0:
                    lines.extend(
                        _list_predictors_at_level(
                            table, agreement_col, rf_flag_col, level
                        )
                    )

            # 0-of-4 group: count only, no names. Listing every predictor that
            # no method flagged would add many lines without analytical use.
            n_zero = int((table[agreement_col] == 0).sum())
            lines.append("")
            lines.append(
                f"  Flagged by 0 of 4 methods: {n_zero} predictors "
                f"(not listed; see CSV)"
            )
            lines.append("")

    lines.append("=" * 70)
    lines.append(
        "See cross_method_fvi.csv and cross_method_evi.csv for the full "
        "predictor-by-method tables, including all per-method statistics."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

print("=" * 70)
print("STAGE 27 — CROSS-METHOD PREDICTOR AGREEMENT")
print("=" * 70)
print()

tables = {}
for index_name in ("FVI", "EVI"):
    print(f"Building agreement table for {index_name}...")
    table = build_agreement_table(index_name)
    out_path = OUT_DIR / f"cross_method_{index_name.lower()}.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    tables[index_name] = table
    print(f"  Wrote {out_path}  ({len(table)} predictors)")

summary_path = OUT_DIR / "cross_method_summary.txt"
write_summary(tables, summary_path)
print()
print(f"Summary written to {summary_path}")
print()

print("=" * 70)
print("STAGE 27 COMPLETE")
print("=" * 70)
for index_name, table in tables.items():
    strict_4 = int((table["agreement_strict"] == 4).sum())
    lenient_4 = int((table["agreement_lenient"] == 4).sum())
    print(
        f"  {index_name}: {strict_4} predictors flagged by all 4 methods (strict), "
        f"{lenient_4} (lenient)"
    )
print()
print(f"All outputs saved to: {OUT_DIR}")
print(f"  cross_method_fvi.csv      — predictor-by-method table for FVI")
print(f"  cross_method_evi.csv      — predictor-by-method table for EVI")
print(f"  cross_method_summary.txt  — readable summary, both indices")
