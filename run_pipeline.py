"""
run_pipeline.py
===============
Master Pipeline Runner — FVI / EVI / CVI
Nepal Lumbini Survey | Vulnerability Index Pipeline
Master's Thesis | University of Twente

PURPOSE
-------
Single entry point for the full vulnerability index analysis. Running this
one script produces all three vulnerability indices and supplementary random
forest validation from the raw survey data (n=2,993 buildings).

USAGE
-----
    python run_pipeline.py

The survey data file (20263003_Nepal_Lumbini_data.csv) must be in the same
directory as this script. All outputs are written to outputs/. A full
timestamped log is written to pipeline.log.

PIPELINE MAP (28 stages)
-------------------------
    Stage 01   FVI | CIMDEN Scoring                        -> outputs/FVI_CIMDEN_scores.csv
    Stage 02   EVI | CIMDEN Scoring                        -> outputs/EVI_CIMDEN_scores.csv
    ------------------------------------------------------------------
    Stage 03   DX  | Indicator Missing Data Diagnostic     -> missing_data_diagnostic_plots/
    Stage 04   DX  | Explanatory Variable Diagnostic       -> explanatory_vars_diagnostic_plots/
    ------------------------------------------------------------------
    Stage 05   PRE | Typology Cluster Analysis             -> outputs/typology_clusters/
    Stage 06   PRE | MNAR Typology Imputation              -> outputs/imputation/
    Stage 07   PRE | MICE Imputation                       -> outputs/imputation/
    Stage 08   PRE | MAO Composite Imputation              -> outputs/analysis_dataset.csv
    Stage 09   PRE | KNN Flood Imputation                  -> outputs/analysis_dataset.csv (updated)
    Stage 10   PRE | Variable Missingness Screening        -> outputs/fvi/evi_*_filtered.csv
    ------------------------------------------------------------------
    Stage 11   FVI | Descriptives & Frequencies            -> outputs/
    Stage 12   FVI | Bivariate Screening                   -> outputs/
    Stage 13   FVI | Stepwise Regression                   -> outputs/fvi_im_predictors.csv
    Stage 14   FVI | Moran's I Autocorrelation             -> outputs/
    Stage 15   FVI | Mixed-Effects Model                   -> outputs/fvi_random_intercepts.csv
    ------------------------------------------------------------------
    Stage 16   EVI | Descriptives & Frequencies            -> outputs/
    Stage 17   EVI | Bivariate Screening                   -> outputs/
    Stage 18   EVI | Stepwise Regression                   -> outputs/evi_im_predictors.csv
    Stage 19   EVI | Moran's I Autocorrelation             -> outputs/
    Stage 20   LISA| Cluster Maps (FVI + EVI)              -> outputs/
    Stage 21   EVI | Mixed-Effects Model                   -> outputs/evi_random_intercepts.csv
    ------------------------------------------------------------------
    Stage 22   CVI | Composite Score (FVI + EVI)           -> outputs/CVI_scores.csv
    ------------------------------------------------------------------
    Stage 23   RF  | Random Forest Supplementary Validation -> outputs/random_forest/
    Stage 24   RF  | RF Validation & Calibration           -> outputs/random_forest/validation/
    ------------------------------------------------------------------
    Stage 25   GAM | Generalised Additive Model            -> outputs/gam/
    Stage 26   GAM | GAM Validation & Calibration          -> outputs/gam/validation/
    ------------------------------------------------------------------
    Stage 27   SYN | Cross-Method Predictor Agreement      -> outputs/cross_method/
                     Four-method comparison: OLS, mixed-effects, RF, GAM.

DATA FLOW — KEY HANDOFFS
-------------------------
    Stage 01/02   ->  FVI_CIMDEN_scores.csv, EVI_CIMDEN_scores.csv
    Stage 03      ->  missing_data_diagnostic_plots/   [diagnostic only — no data modified]
    Stage 04      ->  explanatory_vars_diagnostic_plots/ [diagnostic only — no data modified]
    Stage 05      ->  typology_clusters.csv
    Stage 06      ->  imputation/EVI_imputed.csv, FVI_imputed.csv
    Stage 07      ->  imputation/imputed_EVI_scores.csv, imputed_FVI_scores.csv
    Stage 08      ->  analysis_dataset.csv        [MAO composites merged]
    Stage 08b    ->  analysis_dataset.csv (updated: transport_access_score,
                      assistance_intensity_score added as ward-level columns)
                      ward_covariates.csv (audit table)
    Stage 09      ->  analysis_dataset.csv        [updated in place — flood scores imputed]
    Stage 10      ->  fvi_domains_filtered.csv    [for Stages 12, 13]
                      evi_domains_filtered.csv    [for Stages 17, 18]
                      fvi_candidates_filtered.csv [for Stage 23 RF]
                      evi_candidates_filtered.csv [for Stage 23 RF]
    Stage 13      ->  fvi_im_predictors.csv            [for Stages 15, 25, 26]
                      fvi_stepwise_stage2_results.csv [for Stages 23, 25, 26 — linear R²]
    Stage 18      ->  evi_im_predictors.csv            [for Stages 21, 25, 26]
                      evi_stepwise_stage2_results.csv [for Stages 23, 25, 26 — linear R²]
    Stage 15      ->  fvi_random_intercepts.csv        [for Stages 22, 24, 26]
                      fvi_mixed_effects_results.csv    [for Stage 27 — ME fixed effects]
                      fvi_mixed_effects_comparison.csv [for Stage 27 — OLS estimated on
                                                        the post-stepwise predictor set,
                                                        paired with ME for clustering
                                                        contrast]
    Stage 21      ->  evi_random_intercepts.csv        [for Stages 22, 24, 26]
                      evi_mixed_effects_results.csv    [for Stage 27 — ME fixed effects]
                      evi_mixed_effects_comparison.csv [for Stage 27 — OLS estimated on
                                                        the post-stepwise predictor set,
                                                        paired with ME for clustering
                                                        contrast]
    Stage 23      ->  random_forest/rf_fvi_metrics.csv  [for Stage 25 — RF R² reference]
                      random_forest/rf_evi_metrics.csv  [for Stage 25 — RF R² reference]
                      random_forest/rf_fvi_importance.csv [for Stage 24 Check 8f]
                      random_forest/rf_evi_importance.csv [for Stage 24 Check 8f]
    Stage 25      ->  gam/gam_fvi_vif.csv          [for Stage 26 — EDF-VIF cross-check]
                      gam/gam_evi_vif.csv          [for Stage 26 — EDF-VIF cross-check]
                      gam/gam_fvi_summary.csv      [for Stage 26 — term statistics]
                                                   [also for Stage 27 — agreement table]
                      gam/gam_evi_summary.csv      [for Stage 26 — term statistics]
                                                   [also for Stage 27 — agreement table]
    Stage 27      ->  cross_method/cross_method_fvi.csv  [per-predictor agreement, FVI]
                      cross_method/cross_method_evi.csv  [per-predictor agreement, EVI]
                      cross_method/cross_method_summary.txt [readable summary]
                      [joins per-predictor results from four methods:
                         OLS  — ordinary least squares (from Stage 15/21
                                OLS/ME comparison files; estimated on the
                                post-stepwise predictor set; no clustering
                                correction)
                         ME   — linear mixed-effects regression (Stages 15, 21;
                                random community intercept for clustering)
                         RF   — Random Forest (Stage 23; non-linear with
                                interactions)
                         GAM  — Generalised Additive Model (Stage 25;
                                non-linear additive)
                       Agreement counts run from 0 to 4.]

    DEPENDENCY NOTES
    ----------------
    Stage 25 reads Stage 23 RF metrics optionally. If Stage 23 has not run,
    the three-way comparison (linear / GAM / RF) shows RF R² as N/A. Stage 25
    can run standalone after Stages 13/18 for development purposes.

    Stage 26 has a required dependency on Stage 25. The EDF-VIF cross-check
    reads gam_fvi_vif.csv and gam_evi_vif.csv produced by Stage 25. If these
    files are absent Stage 26 skips the cross-check with a printed warning
    rather than raising an error. Running Stage 26 before Stage 25 will
    produce incomplete output without a pipeline failure.

    Stage 27 has required dependencies on Stages 15, 21, 23, and 25. It joins
    their per-predictor outputs into a single cross-method agreement table
    for each index. The stage performs no new statistical estimation; it
    only synthesises existing pipeline outputs. The four methods compared
    are OLS, mixed-effects regression, Random Forest, and GAM, arranged as
    a 2x2 design across clustering correction (yes for mixed-effects, no
    for the other three) and functional form (linear for OLS and ME,
    non-linear for RF and GAM). The pairwise contrasts in the agreement
    table reveal which assumption is driving each finding. Missing any
    input file causes Stage 27 to exit with a clear error message
    identifying the missing file. The stage must therefore run after the
    four predecessor stages have completed successfully.

DESIGN: WHY SCORING PRECEDES PRE-PROCESSING
--------------------------------------------
Stages 03-05 operate on SCORED indicator columns (1/3/5 vulnerability ordinal
scale), not raw survey dummy columns. Stages 01-02 must run first to produce
those scored columns. Imputing in scored space is methodologically correct:
the 1/3/5 scale is the direct input to composite indices, and imputing raw
binary dummies then re-scoring would introduce noise and be unauditable.

MISSING DATA STRATEGY
----------------------
Imputation strategy informed by diagnostic Stages 03 and 04:

    Stage 03  — Indicator missing data diagnostic (run as part of pipeline)
        Diagnoses MCAR/MAR/MNAR in EVI and FVI indicator columns.
        Produces Spearman rho for temporal trend. If significant,
        add survey_week as auxiliary predictor in Stage 07.

    Stage 04  — Explanatory variable diagnostic (run as part of pipeline)
        Diagnoses missingness in MAO and socioeconomic candidate variables.
        Produces Spearman rho for temporal trend. If significant,
        add survey_week as auxiliary predictor in Stage 08.
        Also tests whether the analysis sample is representative of n=2,993.

    Stage 06  MNAR imputation  — structural indicators absent because the
              assessor could not access the building feature (modal score by
              typology cluster; Andridge & Little 2010).

    Stage 07  MICE imputation  — remaining MAR structural indicators imputed
              via IterativeImputer with surveyor ID as auxiliary predictor
              (van Buuren & Groothuis-Oudshoorn 2011).

    Stage 08  MAO imputation   — 15 MAO composite scores with 15-30% missing
              (enumerator-driven MAR pattern) imputed via composite-level MICE.
              Composite-level imputation is correct because composites are
              proportional scores; imputing raw binary items and recomputing
              would produce inconsistent denominators (van Buuren 2018, §4.4).
              Single imputation (m=1) is a limitation; report as such.

    Stage 09  KNN flood imputation — flood experience perception scores
              (DE03, DE04, DE06, DE08, DE09, DE13) imputed for flood-exposed
              households only (DE01_flood=1, n=2,824) using K-Nearest Neighbours
              (k=5). Neighbours defined on related perception scores, vulnerability
              indices, and household demographics. Non-exposed households remain
              NaN and are handled by pairwise deletion at the regression stage.
              DE05 and DE07 excluded (43.9% missing; exceeds 50% threshold).
              Reference: Troyanskaya et al. (2001).

    Stage 10  Missingness screening — removes any candidate variable with
              >50% missing before bivariate screening (van Buuren 2018, §9.1.4).
              Result: FVI 0 removed; EVI 3 removed at 100% missing.

IMPUTATION AND THE RANDOM FOREST
----------------------------------
Stage 23 receives analysis_dataset.csv, which already contains Stage 05b
MICE-imputed values for the 15 MAO variables. All other candidates are
complete or structurally missing (flood experience variables absent for
households with no flood exposure) in patterns that do not reduce the
complete-case sample below n=2,993.

Stage 23 reports an imputation audit table at startup confirming which
variables are imputed, which are complete, and remaining missingness rates.

Stage 23 reads fvi_candidates_filtered.csv / evi_candidates_filtered.csv
(the full pre-bivariate candidate pool, minus >50% missing variables) so
that its importance rankings are independent of the linear pipeline's
variable selection. This is distinct from fvi_domains_filtered.csv (the
smaller post-bivariate pool used by stepwise stages).

PRIMARY vs SUPPLEMENTARY ANALYSIS
-----------------------------------
    PRIMARY:       Stages 15, 21 — mixed-effects linear models providing
                   interpretable coefficients, significance tests, community
                   random intercepts, and ICC. These answer the research question
                   within the MAO theoretical framework.

    SUPPLEMENTARY: Stages 23, 24 — random forest for convergent validity of
                   variable selection and non-linearity detection. Reported in
                   the thesis appendix. RF cannot replace the primary analysis:
                   it provides no directional effects, no significance tests,
                   no ICC, and no community random intercepts.

CONFIGURATION
--------------
pipeline_config.py is the single source of truth for all analytical parameters.
Edit that file only — no other files need changing when updating variable lists.
See pipeline_config.py for full documentation of all parameters.

FAIR RESEARCH PRINCIPLES
-------------------------
    Findable      : Consistent prefixes (fvi_, evi_, cvi_, rf_); subdirectory structure
    Accessible    : Single outputs/ tree; all paths logged to pipeline.log
    Interoperable : CSV for all tabular data; PNG for all figures
    Reusable      : Self-contained; fully documented; fixed random seed (RANDOM_STATE=42)

REFERENCES
----------
Anselin, L. (1995). Local indicators of spatial association — LISA.
    Geographical Analysis, 27(2), 93-115.
    https://doi.org/10.1111/j.1538-4632.1995.tb00338.x

Andridge, R.R. & Little, R.J.A. (2010). A Review of Hot Deck Imputation.
    International Statistical Review, 78(1), 40-64.
    https://doi.org/10.1111/j.1751-5823.2010.00103.x

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    https://doi.org/10.1023/A:1010933404324

Hendriks, E. & Stokmans, M. (2020). Drivers and barriers of adoption of
    hazard-resistant construction knowledge. IJDRR, 51, 101778.
    https://doi.org/10.1016/j.ijdrr.2020.101778

Hox, J.J., Moerbeek, M., & Van de Schoot, R. (2018). Multilevel Analysis:
    Techniques and Applications (3rd ed.). Routledge.

Saputra, A., Schwarz, J., & Hendriks, E. (2026). Identifying factors
    influencing housing safety in post-earthquake reconstruction in Nepal.
    IJDRR, 133, 105913. https://doi.org/10.1016/j.ijdrr.2025.105913

Strobl, C. et al. (2007). Bias in random forest variable importance measures.
    BMC Bioinformatics, 8, 25. https://doi.org/10.1186/1471-2105-8-25

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
    CRC Press. https://stefvanbuuren.name/fimd/

van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
    Imputation by Chained Equations in R. Journal of Statistical Software,
    45(3). https://doi.org/10.18637/jss.v045.i03

Villagran De Leon, J.C. (2004). Vulnerability: A Conceptual and Methodological
    Review. UNU-EHS Source No. 4. United Nations University.
"""

import importlib.util
import io
import logging
import os
import shutil
import sys
import time
from pathlib import Path


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

def _load_config_exports(config_path: Path) -> dict:
    """
    Load pipeline_config.py and return all public names as a dict.

    Parameters
    ----------
    config_path : Path — absolute path to pipeline_config.py.

    Returns
    -------
    dict mapping name -> value for every name not starting with '_'.
    """
    spec   = importlib.util.spec_from_file_location("pipeline_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


# =============================================================================
# PATHS
# =============================================================================

HERE       = Path(__file__).resolve().parent
DATA_FILE  = HERE / "20263003_Nepal_Lumbini_data.csv"
OUTPUT_DIR = HERE / "outputs"
LOG_FILE   = HERE / "pipeline.log"

# Imputation subdirectory — created by stage07_mice_imputation.py
IMPUTE_DIR = OUTPUT_DIR / "imputation"

_CONFIG_EXPORTS: dict = {}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        ),
        logging.FileHandler(str(LOG_FILE), mode="w"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# STAGE RUNNER
# =============================================================================

class _Tee:
    """
    Write to two streams simultaneously — terminal stdout and a stage log file.

    Used by _run_stage to capture all print() output from stage scripts into
    a per-stage text file in outputs/stage_logs/ without suppressing the
    terminal display.
    """

    def __init__(self, primary, secondary):
        self._primary   = primary    # original sys.stdout (terminal)
        self._secondary = secondary  # open file handle for stage log

    def write(self, data):
        self._primary.write(data)
        self._secondary.write(data)

    def flush(self):
        self._primary.flush()
        self._secondary.flush()

    def isatty(self):
        return self._primary.isatty()

    # Proxy all other attribute access to the primary stream so that
    # libraries that inspect sys.stdout (e.g. tqdm, rich) work correctly.
    def __getattr__(self, name):
        return getattr(self._primary, name)


def _run_stage(stage_label: str, script_path: Path, working_dir: Path) -> None:
    """
    Load and execute a single pipeline stage script.

    The working directory is temporarily changed to working_dir before
    execution and restored afterwards. This ensures relative file paths
    inside stage scripts resolve against the correct directory.

    pipeline_config.py is loaded once; its exported names are injected
    into each stage module namespace before execution.

    All print() output from the stage is captured to a per-stage text file
    in outputs/stage_logs/ via a Tee stream, in addition to being displayed
    in the terminal. This makes the full output available for review after
    the pipeline completes without relying on the terminal scroll buffer.

    Parameters
    ----------
    stage_label  : Human-readable label used in log output.
    script_path  : Absolute path to the stage .py script.
    working_dir  : Directory to set as cwd during execution.
    """
    import sys as _sys

    # Build a safe filename from the stage label
    # e.g. "Stage 06a | PRE | Variable Missingness Screening"
    #   -> "stage_10_pre_variable_missingness_screening.txt"
    # (safe name derived from the stage label string)
    _safe = (stage_label.lower()
             .replace(" | ", "_")
             .replace(" ", "_")
             .replace("/", "-"))
    _log_dir = OUTPUT_DIR / "stage_logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _log_dir / f"{_safe}.txt"

    original_dir    = os.getcwd()
    original_stdout = _sys.stdout

    with open(_log_path, "w", encoding="utf-8", errors="replace") as _fh:
        # Write a header so the file is self-contained
        import datetime as _dt
        _fh.write(f"{'=' * 65}\n")
        _fh.write(f"  {stage_label}\n")
        _fh.write(f"  Script  : {script_path.name}\n")
        _fh.write(f"  Started : "
                  f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _fh.write(f"{'=' * 65}\n\n")

        _sys.stdout = _Tee(original_stdout, _fh)
        try:
            os.chdir(str(working_dir))
            spec   = importlib.util.spec_from_file_location(
                         stage_label, str(script_path))
            module = importlib.util.module_from_spec(spec)
            for name, value in _CONFIG_EXPORTS.items():
                setattr(module, name, value)
            spec.loader.exec_module(module)
        finally:
            _sys.stdout = original_stdout
            os.chdir(original_dir)


# =============================================================================
# HELPERS
# =============================================================================

def _copy_to_output(filename: str, src_dir: Path, dest_dir: Path) -> None:
    """
    Copy a file from src_dir to dest_dir if it is not already there.

    Used to move scored CSVs from HERE (where scoring scripts run) into
    OUTPUT_DIR (where all analysis stages read from).

    Parameters
    ----------
    filename : Name of the file to copy.
    src_dir  : Source directory (typically HERE).
    dest_dir : Destination directory (OUTPUT_DIR).
    """
    src  = src_dir  / filename
    dest = dest_dir / filename
    if src.exists() and src.resolve() != dest.resolve():
        shutil.copy2(str(src), str(dest))
        log.info(f"  Copied {filename} -> {dest_dir.name}/")
    elif not src.exists():
        log.warning(f"  Expected output not found: {src}")


def _confirm_imputation_outputs() -> bool:
    """
    Confirm that Stage 0 (MICE) produced its expected output files.

    Returns True if all files are present; False otherwise with warnings.
    """
    expected = [
        IMPUTE_DIR / "imputed_EVI_scores.csv",
        IMPUTE_DIR / "imputed_FVI_scores.csv",
        IMPUTE_DIR / "mice_imputation_log.csv",
        IMPUTE_DIR / "mice_mnar_sensitivity.csv",
    ]
    all_ok = True
    for f in expected:
        if f.exists():
            log.info(f"  Confirmed: {f.name}")
        else:
            log.warning(f"  Missing imputation output: {f}")
            all_ok = False
    return all_ok


# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    """
    Execute the full vulnerability index pipeline.

    Execution order
    ---------------
    1. Stage 01  — FVI CIMDEN scoring   (reads raw CSV, writes scored CSV)
    2. Stage 02  — EVI CIMDEN scoring   (reads raw CSV, writes scored CSV)
    3. Stage 03  — Indicator diagnostic (reads raw CSV, writes diagnostic plots)
    4. Stage 04  — Explanatory diagnostic (reads raw CSV, writes diagnostic plots)
    5. Stage 05  — Typology clusters    (reads raw CSV, writes typology_clusters.csv)
    6. Stage 06  — MNAR imputation      (reads scored + clusters, writes EVI/FVI_imputed)
    7. Stage 07  — MICE imputation      (reads MNAR-imputed CSVs, writes imputed CSVs)
    8. Stage 08  — MAO imputation       (imputes MAO composites, writes analysis_dataset.csv)
    9. Stage 09  — KNN flood imputation (imputes flood scores in analysis_dataset.csv)
    10. Stage 10 — Missingness screening (filters candidate variable lists)
    11. Stages 06–10 — FVI analysis     (read from analysis_dataset.csv)
    12. Stages 11–16 — EVI analysis     (read from analysis_dataset.csv)
    13. Stage 17 — CVI composite        (reads fvi/evi_random_intercepts.csv)
    14. Stages 18–19b — RF / GAM validation (supplementary)

    Exit codes
    ----------
    0 : All stages completed successfully.
    1 : A stage raised an exception; details written to pipeline.log.
    """

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not DATA_FILE.exists():
        log.error(f"Survey data not found: {DATA_FILE}")
        log.error("Ensure 20263003_Nepal_Lumbini_data.csv is in the same "
                  "directory as run_pipeline.py.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = HERE / "pipeline_config.py"
    if not config_path.exists():
        log.error(f"pipeline_config.py not found in: {HERE}")
        sys.exit(1)

    global _CONFIG_EXPORTS
    _CONFIG_EXPORTS = _load_config_exports(config_path)
    log.info(f"  Loaded pipeline_config.py ({len(_CONFIG_EXPORTS)} exported names)")

    # ── Stage definitions ──────────────────────────────────────────────────────
    # Format: (stage_label, script_path, working_directory)
    #
    # Scoring stages (1, 7) run in HERE so their INPUT_CSV (which uses
    # Path(__file__).resolve().parent) resolves to the raw data file.
    # All other stages run in OUTPUT_DIR where their Path.cwd() calls resolve.
    # Stage 0 (MICE) runs in HERE so it can find outputs/ as a subdirectory.

    stages = [
        # ── Stage 01: FVI CIMDEN scoring ──────────────────────────────────────
        # Reads  : 20263003_Nepal_Lumbini_data.csv
        # Writes : FVI_CIMDEN_scores.csv (copied to outputs/)
        ("Stage 01 | FVI | CIMDEN Scoring",
         HERE / "stage01_fvi_scoring.py",   HERE),

        # ── Stage 02: EVI CIMDEN scoring ──────────────────────────────────────
        # Reads  : 20263003_Nepal_Lumbini_data.csv
        # Writes : EVI_CIMDEN_scores.csv (copied to outputs/)
        ("Stage 02 | EVI | CIMDEN Scoring",
         HERE / "stage02_evi_scoring.py",   HERE),

        # ── Stage 03: Indicator missing data diagnostic ────────────────────────
        # Reads  : 20263003_Nepal_Lumbini_data.csv
        # Writes : missing_data_diagnostic_plots/
        #            fig1[evi/fvi]_missing_rate.png
        #            fig2[evi/fvi]_row_heatmap.png
        #            fig3[evi/fvi]_cooccurrence.png
        #            fig4[evi/fvi]_mar_auc.png
        #            fig5[evi/fvi]_mnar_sensitivity.png
        #            fig6[evi/fvi]_surveyor_clustering.png
        #            fig7[evi/fvi]_community_clustering.png
        #            fig8[evi/fvi]_temporal_clustering.png
        #
        # Five diagnostic steps on EVI and FVI indicator columns:
        #   Step 1 — Missingness rates and heatmaps
        #   Step 2 — Little's MCAR test
        #   Step 3 — MAR logistic regression AUC per indicator
        #   Step 4 — MNAR sensitivity (worst/best-case bounds)
        #   Step 5 — Clustering by surveyor, community, and survey date
        #             Spearman rho quantifies temporal trend.
        #             If significant, add survey_week to Stage 07 MICE models.
        # NOTE: This stage does not modify any data. It is diagnostic only.
        ("Stage 03 | DX | Indicator Missing Data Diagnostic",
         HERE / "stage03_indicator_missing_diagnostic.py", HERE),

        # ── Stage 04: Explanatory variable diagnostic ─────────────────────────
        # Reads  : 20263003_Nepal_Lumbini_data.csv
        # Writes : explanatory_vars_diagnostic_plots/
        #            fig_exp01_missing_rate.png
        #            fig_exp02_row_heatmap.png
        #            fig_exp03_cooccurrence.png
        #            fig_exp04_mar_auc.png
        #            fig_exp05_surveyor_clustering.png
        #            fig_exp06_community_clustering.png
        #            fig_exp07_temporal_clustering.png
        #            fig_exp08_sample_comparison.png
        #
        # Five diagnostic steps on MAO and socioeconomic candidate variables:
        #   Step 1 — Missingness rates and heatmaps
        #   Step 2 — Analysis sample representativeness (retained vs dropped)
        #   Step 3 — MAR logistic regression AUC per variable
        #   Step 4 — Clustering by surveyor, community, and survey date
        #             Spearman rho quantifies temporal trend.
        #   Step 5 — Final decision table (impute / delete / flag)
        # NOTE: This stage does not modify any data. It is diagnostic only.
        ("Stage 04 | DX | Explanatory Variable Diagnostic",
         HERE / "stage04_explanatory_vars_diagnostic.py", HERE),

        # ── Stage 05: Typology cluster analysis ───────────────────────────────
        # Reads  : 20263003_Nepal_Lumbini_data.csv
        # Writes : outputs/typology_clusters.csv
        #          outputs/typology_clusters/fig01-05_*.png
        #          outputs/typology_clusters/typology_cluster_profiles.csv
        ("Stage 05 | PRE | Typology Cluster Analysis",
         HERE / "stage05_typology_clusters.py", HERE),

        # ── Stage 06: MNAR typology imputation ────────────────────────────────
        # Reads  : outputs/EVI_CIMDEN_scores.csv
        #          outputs/FVI_CIMDEN_scores.csv
        #          outputs/typology_clusters.csv
        # Writes : outputs/imputation/EVI_imputed.csv
        #          outputs/imputation/FVI_imputed.csv
        #          outputs/imputation/mnar_typology/fig_mnar_*.png
        #          outputs/imputation/mnar_typology/mnar_imputation_report.txt
        ("Stage 06 | PRE | MNAR Typology Imputation",
         HERE / "stage06_mnar_imputation.py", HERE),

        # ── Stage 07: MICE imputation ─────────────────────────────────────────
        # Reads  : outputs/imputation/EVI_imputed.csv
        #          outputs/imputation/FVI_imputed.csv
        # Writes : outputs/imputation/imputed_EVI_scores.csv
        #          outputs/imputation/imputed_FVI_scores.csv
        #          outputs/imputation/mice_imputation_log.csv
        #          outputs/imputation/mice_mnar_sensitivity.csv
        ("Stage 07 | PRE | MICE Imputation",
         HERE / "stage07_mice_imputation.py", HERE),

        # ── Stage 08: MAO composite imputation ───────────────────────────────
        # Reads  : outputs/imputed_FVI_scores.csv
        #          outputs/imputed_EVI_scores.csv
        #          20263003_Nepal_Lumbini_data.csv  (MAO columns)
        # Writes : outputs/analysis_dataset.csv  (MAO composites merged)
        #          outputs/mao_imputation_log.csv
        #          outputs/mao_imputation_report.txt
        ("Stage 08 | PRE | MAO Composite Imputation",
         HERE / "stage08_mao_imputation.py", HERE),

        # ── Stage 08b: Ward-level covariate integration ─────────────────────
        # Reads  : ward_information.csv
        #          outputs/analysis_dataset.csv  (produced by Stage 08)
        # Writes : outputs/analysis_dataset.csv  (updated in place: +2 columns)
        #          outputs/ward_covariates.csv   (audit table, one row per ward)
        #
        # Computes two ward-level Level-2 predictors for Stages 15 and 21:
        #   transport_access_score    — proportion of 11 binary access items
        #   assistance_intensity_score — proportion of 10 binary assistance items
        # Both scores are ward constants (vary only between wards, not within).
        ("Stage 08b | PRE | Ward-Level Covariate Integration",
         HERE / "stage08b_ward_covariates.py", OUTPUT_DIR),

        # ── Stage 09: KNN flood experience imputation ─────────────────────────
        # Reads  : outputs/analysis_dataset.csv  (produced by Stage 08)
        # Writes : outputs/analysis_dataset.csv  (updated in place)
        #          outputs/imputation/knn_flood_imputation_log.csv
        #          outputs/imputation/knn_flood_imputation_report.txt
        #
        # Imputes flood experience perception scores (DE03, DE04, DE06, DE08,
        # DE09, DE13) for flood-exposed households only (DE01_flood=1).
        # Uses KNNImputer (k=5) with neighbours defined on related perception
        # scores, vulnerability indices, and household demographics.
        # Non-exposed households (DE01_flood=0) are left as NaN and handled
        # by pairwise deletion at regression stages (13, 18).
        # DE05 and DE07 excluded (43.9% missing; van Buuren 2018, §9.1.4).
        ("Stage 09 | PRE | KNN Flood Imputation",
         HERE / "stage09_knn_flood_imputation.py", OUTPUT_DIR),

        # ── Stage 10: Explanatory variable missingness screening ──────────────
        # Reads  : outputs/analysis_dataset.csv
        #          pipeline_config.py  (FVI_CANDIDATES, EVI_CANDIDATES,
        #                               FVI_DOMAINS,    EVI_DOMAINS)
        # Writes : outputs/variable_screening_report.csv
        #          outputs/variable_screening_summary.txt
        #          outputs/fvi_domains_filtered.csv  (→ Stage 12, 13)
        #          outputs/evi_domains_filtered.csv  (→ Stage 17, 18)
        #
        # Removes any candidate variable whose proportion of missing values
        # exceeds 50%, per van Buuren (2018, §9.1.4). The filtered domain
        # lists are consumed by Stages 12 and 17 (bivariate screening) so
        # that excluded variables never enter the regression pipeline.
        ("Stage 10 | PRE | Variable Missingness Screening",
         HERE / "stage10_variable_screening.py", OUTPUT_DIR),

        # ── Stage 11–15: FVI analysis ─────────────────────────────────────────
        ("Stage 11 | FVI | Descriptives & Frequencies",
         HERE / "stage11_fvi_descriptives.py",  OUTPUT_DIR),

        ("Stage 12 | FVI | Bivariate Screening",
         HERE / "stage12_fvi_bivariate.py",     OUTPUT_DIR),

        ("Stage 13 | FVI | Stepwise Regression",
         HERE / "stage13_fvi_stepwise.py",      OUTPUT_DIR),

        ("Stage 14 | FVI | Moran's I Autocorrelation",
         HERE / "stage14_fvi_morans.py",        OUTPUT_DIR),

        ("Stage 15 | FVI | Mixed-Effects Model",
         HERE / "stage15_fvi_mixed_effects.py", OUTPUT_DIR),

        # ── Stage 16–21: EVI analysis ─────────────────────────────────────────
        ("Stage 16 | EVI | Descriptives & Frequencies",
         HERE / "stage16_evi_descriptives.py",  OUTPUT_DIR),

        ("Stage 17 | EVI | Bivariate Screening",
         HERE / "stage17_evi_bivariate.py",     OUTPUT_DIR),

        ("Stage 18 | EVI | Stepwise Regression",
         HERE / "stage18_evi_stepwise.py",      OUTPUT_DIR),

        ("Stage 19 | EVI | Moran's I Autocorrelation",
         HERE / "stage19_evi_morans.py",        OUTPUT_DIR),

        ("Stage 20 | LISA | Cluster Maps (FVI + EVI)",
         HERE / "stage20_lisa_cluster_maps.py", OUTPUT_DIR),

        ("Stage 21 | EVI | Mixed-Effects Model",
         HERE / "stage21_evi_mixed_effects.py", OUTPUT_DIR),

        # ── Stage 22: CVI composite ───────────────────────────────────────────
        ("Stage 22 | CVI | Composite Score (FVI + EVI)",
         HERE / "stage22_composite_cvi.py",     OUTPUT_DIR),

        # ── Stage 23: Random Forest supplementary validation ──────────────────
        # Reads  : outputs/analysis_dataset.csv
        #          outputs/fvi_domains_filtered.csv  (Stage 10)
        #          outputs/evi_domains_filtered.csv  (Stage 10)
        #          outputs/fvi_im_predictors.csv     (Stage 13)
        #          outputs/evi_im_predictors.csv     (Stage 18)
        # Writes : outputs/random_forest/rf_*.csv / *.png / *.txt
        #
        # PURPOSE: Supplementary robustness check only. RF provides:
        #   (1) Out-of-sample R² comparison with linear models.
        #   (2) Permutation importance to triangulate stepwise selection.
        # RF does NOT replace Stages 15/21 (primary mixed-effects analysis).
        ("Stage 23 | RF | Random Forest Supplementary Validation",
         HERE / "stage23_random_forest.py",     OUTPUT_DIR),

        # ── Stage 24: RF validation & calibration ─────────────────────────────
        # Reads  : outputs/analysis_dataset.csv
        #          outputs/fvi_domains_filtered.csv / evi_domains_filtered.csv
        #          outputs/fvi_im_predictors.csv / evi_im_predictors.csv
        #          outputs/fvi_random_intercepts.csv / evi_random_intercepts.csv
        # Writes : outputs/random_forest/validation/rfval_*.csv / *.png / *.txt
        #
        # Eight validation checks:
        #   1. Nested cross-validation  — stable OOS R² across partitions
        #   2. Learning curves          — sample size adequacy
        #   3. Hyperparameter sensitivity — robustness to tuning choices
        #   4. Community residuals      — confirms clustering RF cannot capture
        #   5. Calibration              — tail under-prediction quantified
        #   6. Importance stability     — 95% CI, flags unstable rank pairs
        #   7. MDI vs permutation       — Strobl et al. (2007) bias check
        #   8. Predictor correlation    — flags collinear pairs (|r|>0.60)
        ("Stage 24 | RF | Validation & Calibration",
         HERE / "stage24_rf_validation.py",    OUTPUT_DIR),

        # ── Stage 25: GAM supplementary analysis ──────────────────────────────
        # Reads  : outputs/analysis_dataset.csv
        #          outputs/fvi_im_predictors.csv / evi_im_predictors.csv
        #          outputs/fvi_stepwise_stage2_results.csv / evi_stepwise_stage2_results.csv
        #          outputs/random_forest/rf_fvi_metrics.csv / rf_evi_metrics.csv (optional)
        # Writes : outputs/gam/gam_fvi_summary.csv
        #          outputs/gam/gam_evi_summary.csv
        #          outputs/gam/gam_fvi_smooths.png
        #          outputs/gam/gam_evi_smooths.png
        #          outputs/gam/gam_fvi_predicted_vs_actual.png
        #          outputs/gam/gam_evi_predicted_vs_actual.png
        #          outputs/gam/gam_nonlinearity_report.csv
        #          outputs/gam/gam_comparison_summary.txt
        #
        # Purpose: Bridges the linear and RF analyses by fitting penalised
        # splines (LinearGAM via pygam) on the stepwise-selected predictors.
        # Formally tests which predictors have non-linear effects (EDF > 1.5)
        # and by how much — directly motivated by the FVI RF gain of +0.092.
        ("Stage 25 | GAM | Generalised Additive Model",
         HERE / "stage25_gam.py",               OUTPUT_DIR),

        # ── Stage 26: GAM validation & calibration ────────────────────────────
        # Reads  : outputs/analysis_dataset.csv
        #          outputs/fvi_im_predictors.csv / evi_im_predictors.csv
        #          outputs/fvi_random_intercepts.csv / evi_random_intercepts.csv
        #          outputs/fvi_stepwise_stage2_results.csv / evi_stepwise_stage2_results.csv
        # Writes : outputs/gam/validation/gamval_*.csv / *.png / *.txt
        #
        # Seven validation checks:
        #   1. Nested 10-fold CV        — OOS R² stability across partitions
        #   2. Learning curves          — sample size adequacy
        #   3. Smoothing sensitivity    — robustness to lambda choices
        #   4. EDF stability            — consistency of non-linearity conclusions
        #   5. Residual analysis        — misspecification and community patterns
        #   6. Calibration              — decile-bin MACE (mirrors Stage 24)
        #   7. GAM vs linear comparison — smooth overlaid on linear slope
        ("Stage 26 | GAM | Validation & Calibration",
         HERE / "stage26_gam_validation.py",   OUTPUT_DIR),

        # ── Stage 27: Cross-method predictor agreement ────────────────────────
        # Reads  : outputs/fvi_mixed_effects_comparison.csv  (OLS + ME side-by-side)
        #          outputs/evi_mixed_effects_comparison.csv  (OLS + ME side-by-side)
        #          outputs/fvi_mixed_effects_results.csv
        #          outputs/evi_mixed_effects_results.csv
        #          outputs/random_forest/rf_fvi_importance.csv
        #          outputs/random_forest/rf_evi_importance.csv
        #          outputs/gam/gam_fvi_summary.csv
        #          outputs/gam/gam_evi_summary.csv
        # Writes : outputs/cross_method/cross_method_fvi.csv
        #          outputs/cross_method/cross_method_evi.csv
        #          outputs/cross_method/cross_method_summary.txt
        #
        # Synthesis stage. Joins the per-predictor outputs of four analytical
        # methods (OLS, mixed-effects regression, Random Forest, GAM) for
        # each single-hazard index (FVI, EVI) into a single agreement table.
        # The four methods sit at the corners of a 2x2 design:
        #     OLS  : linear, no clustering correction
        #     ME   : linear, clustering-corrected
        #     RF   : non-linear with interactions, no clustering correction
        #     GAM  : non-linear additive, no clustering correction
        # Pairwise contrasts reveal which assumption drives each finding.
        # For each predictor and index, records whether each method flagged
        # the predictor as important under published criteria:
        #   - OLS, mixed-effects, GAM: p < 0.05 (Fisher 1925)
        #   - Random Forest: top decile (strict) or top third (lenient) of
        #                    permutation importance (Strobl et al. 2007,
        #                    https://doi.org/10.1186/1471-2105-8-25)
        # The 'agreement_strict' and 'agreement_lenient' columns count the
        # number of methods (0-4) flagging each predictor under each criterion.
        # No new statistical estimation is performed.
        ("Stage 27 | SYN | Cross-Method Predictor Agreement",
         HERE / "stage27_cross_method_comparison.py", OUTPUT_DIR),
    ]

    # Verify all stage scripts exist before starting
    missing_scripts = [s[1] for s in stages if not s[1].exists()]
    if missing_scripts:
        for m in missing_scripts:
            log.error(f"Stage script not found: {m}")
        sys.exit(1)

    # ── Banner ────────────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("VULNERABILITY INDEX PIPELINE")
    log.info("=" * 65)
    log.info(f"Data file  : {DATA_FILE}")
    log.info(f"Output dir : {OUTPUT_DIR}")
    log.info(f"Log file   : {LOG_FILE}")
    log.info(f"Stages     : {len(stages)}")
    log.info("=" * 65)

    pipeline_start = time.time()
    completed      = 0
    failed_stage   = None

    # ── Execute stages ────────────────────────────────────────────────────────
    for stage_label, script_path, working_dir in stages:

        log.info("\n" + "-" * 65)
        log.info(f"  {stage_label}")
        log.info(f"  Script : {script_path.name}")
        log.info("-" * 65)

        stage_start = time.time()

        try:
            _run_stage(stage_label, script_path, working_dir)

        except Exception as exc:
            log.error(f"  [FAILED]: {exc}", exc_info=True)
            failed_stage = stage_label
            break

        elapsed = time.time() - stage_start
        log.info(f"  [OK]  Completed in {elapsed:.1f}s")
        completed += 1

        # ── Post-stage actions ─────────────────────────────────────────────
        # After FVI scoring: copy scored CSV and raw survey to OUTPUT_DIR
        if stage_label.startswith("Stage 01"):
            _copy_to_output("FVI_CIMDEN_scores.csv", HERE, OUTPUT_DIR)
            _copy_to_output("20263003_Nepal_Lumbini_data.csv", HERE, OUTPUT_DIR)

        # After EVI scoring: copy scored CSV to OUTPUT_DIR.
        # Guard with "Stage 02 |" so this does not fire for 02a or 02b.
        if stage_label.startswith("Stage 02 |"):
            _copy_to_output("EVI_CIMDEN_scores.csv", HERE, OUTPUT_DIR)

        # After indicator diagnostic: confirm plot directory was produced
        if stage_label.startswith("Stage 03 |"):
            dx_dir = HERE / "missing_data_diagnostic_plots"
            if dx_dir.exists():
                n_plots = len(list(dx_dir.glob("*.png")))
                log.info(f"  Confirmed: missing_data_diagnostic_plots/ "
                         f"({n_plots} figures)")
            else:
                log.warning("  missing_data_diagnostic_plots/ not found — "
                            "check stage03_indicator_missing_diagnostic.py")

        # After explanatory variable diagnostic: confirm plot directory
        if stage_label.startswith("Stage 04 |"):
            dx_dir = HERE / "explanatory_vars_diagnostic_plots"
            if dx_dir.exists():
                n_plots = len(list(dx_dir.glob("*.png")))
                log.info(f"  Confirmed: explanatory_vars_diagnostic_plots/ "
                         f"({n_plots} figures)")
            else:
                log.warning("  explanatory_vars_diagnostic_plots/ not found — "
                            "check stage04_explanatory_vars_diagnostic.py")

        # After MAO imputation: confirm analysis_dataset.csv was produced
        if stage_label.startswith("Stage 08 |"):
            ad = OUTPUT_DIR / "analysis_dataset.csv"
            if ad.exists():
                log.info(f"  Confirmed: analysis_dataset.csv ({ad.stat().st_size:,} bytes)")
            else:
                log.error("  analysis_dataset.csv not found — check Stage 08")

        # After KNN flood imputation: confirm log and updated dataset
        if stage_label.startswith("Stage 09 |"):
            knn_log = IMPUTE_DIR / "knn_flood_imputation_log.csv"
            if knn_log.exists():
                log.info(f"  Confirmed: imputation/knn_flood_imputation_log.csv "
                         f"({knn_log.stat().st_size:,} bytes)")
            else:
                log.error("  knn_flood_imputation_log.csv not found — check Stage 09")
            ad = OUTPUT_DIR / "analysis_dataset.csv"
            log.info(f"  analysis_dataset.csv size after KNN update: "
                     f"{ad.stat().st_size:,} bytes")

        # After typology clustering: confirm typology_clusters.csv was written
        if stage_label.startswith("Stage 05 |"):
            tc = OUTPUT_DIR / "typology_clusters.csv"
            if tc.exists():
                log.info(f"  Confirmed: typology_clusters.csv ({tc.stat().st_size:,} bytes)")
            else:
                log.error("  typology_clusters.csv not found in outputs/ — check Stage 05")

        # After MNAR imputation: confirm EVI_imputed.csv and FVI_imputed.csv
        if stage_label.startswith("Stage 06 |"):
            for fname in ("EVI_imputed.csv", "FVI_imputed.csv"):
                fp = IMPUTE_DIR / fname
                if fp.exists():
                    log.info(f"  Confirmed: imputation/{fname} ({fp.stat().st_size:,} bytes)")
                else:
                    log.error(f"  {fname} not found in outputs/imputation/ — check Stage 06")

        # After MICE imputation: confirm all four output files exist.
        # MICE imputation check — Stage 07
        if stage_label.startswith("Stage 07 |"):
            ok = _confirm_imputation_outputs()
            if not ok:
                log.error("  Stage 07 did not produce all expected outputs.")
                log.error("  Check outputs/imputation/ and pipeline.log.")
                sys.exit(1)

            # Also copy imputed CSVs to OUTPUT_DIR root so analysis stages
            # can find them with Path.cwd() / "imputed_*.csv"
            _copy_to_output("imputed_FVI_scores.csv", IMPUTE_DIR, OUTPUT_DIR)
            _copy_to_output("imputed_EVI_scores.csv", IMPUTE_DIR, OUTPUT_DIR)

        # After cross-method comparison: confirm the agreement files were
        # produced. Stage 27 is the final stage; this check helps the user
        # see at a glance that the synthesis output is on disk.
        if stage_label.startswith("Stage 27 |"):
            cm_dir = OUTPUT_DIR / "cross_method"
            for fname in ("cross_method_fvi.csv",
                          "cross_method_evi.csv",
                          "cross_method_summary.txt"):
                fp = cm_dir / fname
                if fp.exists():
                    log.info(f"  Confirmed: cross_method/{fname} "
                             f"({fp.stat().st_size:,} bytes)")
                else:
                    log.warning(f"  cross_method/{fname} not found — "
                                f"check stage27_cross_method_comparison.py")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start

    log.info(f"\n{'=' * 65}")
    if failed_stage is None:
        log.info(
            f"  PIPELINE COMPLETE  —  {completed}/{len(stages)} stages  "
            f"({total_elapsed:.1f}s total)"
        )
        log.info(f"\n  Key outputs in: {OUTPUT_DIR}")
        log.info(f"    FVI_CIMDEN_scores.csv        Raw FVI scores (pre-imputation)")
        log.info(f"    EVI_CIMDEN_scores.csv        Raw EVI scores (pre-imputation)")
        log.info(f"    imputation/imputed_FVI*.csv  FVI after MICE imputation")
        log.info(f"    imputation/imputed_EVI*.csv  EVI after MICE imputation")
        log.info(f"    imputation/mice_*.csv        Imputation log + MNAR table")
        log.info(f"    analysis_dataset.csv         Merged dataset for regressions")
        log.info(f"    fvi_im_predictors.csv        FVI Individual Model predictors (-> Stage 15)")
        log.info(f"    evi_im_predictors.csv        EVI Individual Model predictors (-> Stage 21)")
        log.info(f"    CVI_scores.csv               Composite index (1-5)")
        log.info(f"    cross_method/                Cross-method predictor agreement")
        log.info(f"                                 (OLS, ME, RF, GAM; 2x2 design)")
        log.info(f"    stage_logs/                  Per-stage console output (one .txt per stage)")
        log.info(f"\n  Full log: {LOG_FILE}")
    else:
        log.error(
            f"  PIPELINE FAILED at {failed_stage}  "
            f"({completed}/{len(stages)} stages completed)"
        )
        log.error(f"  See {LOG_FILE} for the full traceback.")
        sys.exit(1)

    log.info("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
