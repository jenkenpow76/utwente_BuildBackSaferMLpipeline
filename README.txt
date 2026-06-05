================================================================================
                              PIPELINE v47 NOTE
================================================================================

This is pipeline v47 (3 June 2026), an update from v46 that:
  - Fixes an architectural integrity bug (drift between candidate / domain /
    IM_PREDICTOR variable lists)
  - Removes 6 EVI candidate variables now known to be embedded inside the
    MAO composite scores (FA01, RT02, PP01/PP02 dummies)
  - Fixes the Villagrán De León citation year (2006, not 2004)
  - Adds runtime drift assertion at the top of stages 13, 15, 18, 21

For the full list of changes and rationale:
  → See CHANGELOG_v47.md

For the exact sequence of stages to re-run after pulling v47:
  → See RUN_ORDER.md

The v46 originals of all modified files are preserved in _v46_originals/.

================================================================================

vulnerability_index_pipeline
=============================
FVI * EVI * CVI Analysis Pipeline — Nepal Lumbini Dataset (n=2,993)
Master's Thesis | University of Twente

QUICK START
-----------
    1. Place 20263003_Nepal_Lumbini_data.csv in the same folder as run_pipeline.py
    2. python run_pipeline.py

All 28 stages run automatically in the correct order. Outputs are written to
outputs/. A full log is written to pipeline.log.

STAGES (in execution order)
----------------------------
    Stage 01   FVI | CIMDEN Scoring                         stage01_fvi_scoring.py
    Stage 02   EVI | CIMDEN Scoring                         stage02_evi_scoring.py
    Stage 03   DX  | Indicator Missing Data Diagnostic      stage03_indicator_missing_diagnostic.py
    Stage 04   DX  | Explanatory Variable Diagnostic        stage04_explanatory_vars_diagnostic.py
    Stage 05   PRE | Typology Cluster Analysis              stage05_typology_clusters.py
    Stage 06   PRE | MNAR Typology Imputation               stage06_mnar_imputation.py
    Stage 07   PRE | MICE Imputation                        stage07_mice_imputation.py
    Stage 08   PRE | MAO Composite Imputation               stage08_mao_imputation.py
    Stage 08b  PRE | Ward-Level Covariate Integration       stage08b_ward_covariates.py
    Stage 09   PRE | KNN Flood Experience Imputation        stage09_knn_flood_imputation.py
    Stage 10   PRE | Variable Missingness Screening         stage10_variable_screening.py
    Stage 11   FVI | Descriptives & Frequencies             stage11_fvi_descriptives.py
    Stage 12   FVI | Bivariate Screening                    stage12_fvi_bivariate.py
    Stage 13   FVI | Stepwise Regression                    stage13_fvi_stepwise.py
    Stage 14   FVI | Moran's I Autocorrelation              stage14_fvi_morans.py
    Stage 15   FVI | Mixed-Effects Model                    stage15_fvi_mixed_effects.py
    Stage 16   EVI | Descriptives & Frequencies             stage16_evi_descriptives.py
    Stage 17   EVI | Bivariate Screening                    stage17_evi_bivariate.py
    Stage 18   EVI | Stepwise Regression                    stage18_evi_stepwise.py
    Stage 19   EVI | Moran's I Autocorrelation              stage19_evi_morans.py
    Stage 20   LISA | Cluster Maps (FVI + EVI)              stage20_lisa_cluster_maps.py
    Stage 21   EVI | Mixed-Effects Model                    stage21_evi_mixed_effects.py
    Stage 22   CVI | Composite Score (FVI + EVI)            stage22_composite_cvi.py
    Stage 23   RF  | Random Forest Supplementary Validation stage23_random_forest.py
    Stage 24   RF  | RF Validation & Calibration            stage24_rf_validation.py
    Stage 25   GAM | Generalised Additive Model             stage25_gam.py
    Stage 26   GAM | GAM Validation & Calibration           stage26_gam_validation.py
    Stage 27   SYN | Cross-Method Predictor Agreement       stage27_cross_method_comparison.py
                     [four-method comparison: OLS, mixed-effects, RF, GAM]

PIPELINE DESIGN: WHY SCORING PRECEDES PRE-PROCESSING
------------------------------------------------------
Stages 05-07 (typology clustering, MNAR imputation, MICE) operate on SCORED
indicator columns (1/3/5 vulnerability ordinal scale), not on raw survey dummy
columns. Stages 01 and 02 must therefore run first. Stages 03 and 04 are diagnostic. Imputing in scored space is
methodologically correct: the 1/3/5 scale is the direct input to composite
indices, and imputing raw binary dummies then re-scoring would introduce
additional noise and be difficult to audit.

PIPELINE DESIGN: DATA FLOW
---------------------------
The pipeline reads from and writes to a single outputs/ directory. Key handoffs:

    stage01/02 write  ->  FVI_CIMDEN_scores.csv, EVI_CIMDEN_scores.csv
    stage03    writes ->  missing_data_diagnostic_plots/  [diagnostic only]
    stage04    writes ->  explanatory_vars_diagnostic_plots/  [diagnostic only]
    stage02b   writes ->  explanatory_vars_diagnostic_plots/  [diagnostic only]
    stage03    writes ->  typology_clusters.csv
    stage04    writes ->  imputation/EVI_imputed.csv, FVI_imputed.csv
    stage05    writes ->  imputation/imputed_EVI_scores.csv, imputed_FVI_scores.csv
    stage05b   writes ->  analysis_dataset.csv  (MAO composites merged)
    stage05c   writes ->  analysis_dataset.csv  [updated in place — flood scores imputed]
                          imputation/knn_flood_imputation_log.csv
                          imputation/knn_flood_imputation_report.txt
    stage06a   writes ->  fvi_domains_filtered.csv     (for Stages 07, 08)
                          evi_domains_filtered.csv     (for Stages 12, 13)
                          fvi_candidates_filtered.csv  (for Stage 18)
                          evi_candidates_filtered.csv  (for Stage 18)
    stage08    writes ->  fvi_im_predictors.csv  (for Stage 10)
    stage13    writes ->  evi_im_predictors.csv  (for Stage 16)
    stage10    writes ->  fvi_random_intercepts.csv  (for Stage 17, 18b)
    stage16    writes ->  evi_random_intercepts.csv  (for Stage 17, 18b)
    stage17    writes ->  CVI_scores.csv
    stage18    writes ->  random_forest/rf_*.csv / *.png / *.txt
    stage18b   writes ->  random_forest/validation/rfval_*.csv / *.png / *.txt
    stage27    writes ->  cross_method/cross_method_fvi.csv
                          cross_method/cross_method_evi.csv
                          cross_method/cross_method_summary.txt
                          (synthesis stage; joins per-predictor results from
                          four methods:
                            OLS  — ordinary least squares (no clustering correction)
                            ME   — mixed-effects regression (clustering-corrected)
                            RF   — Random Forest (non-linear with interactions)
                            GAM  — Generalised Additive Model (non-linear additive)
                          arranged in a 2x2 design across clustering correction
                          and functional form)

MISSING DATA IMPUTATION APPROACH
---------------------------------
Imputation strategy informed by diagnostic Stages 02a and 02b:

    Stage 02a — Indicator missing data diagnostic
        Diagnoses MCAR/MAR/MNAR in EVI and FVI indicator columns.
        Produces Spearman rho for temporal trend (Fig 8). If significant,
        add survey_week as auxiliary predictor in Stage 05.
        Reference: Little (1988); van Buuren (2018).

    Stage 02b — Explanatory variable diagnostic
        Diagnoses missingness in MAO and socioeconomic candidate variables.
        Produces Spearman rho for temporal trend (Fig EXP07). If significant,
        add survey_week as auxiliary predictor in Stage 05b.
        Tests whether the analysis sample is representative of n=2,993.
        Reference: van Buuren (2018); Groves (2006).

    Stage 03 — Building typology clustering
        K-means (k=8) on raw structural features (wall material, roof material,
        roof shape, floor count, plan shape, L/W ratio, elevation, adjacency).
        Identifies eight building typology classes used as donor groups.
        Reference: Rijal (2018); Kandel et al. (2024); Hartigan & Wong (1979).

    Stage 04 — MNAR typology imputation
        Structural vulnerability indicators flagged as MNAR (missing not at
        random — absent because assessor could not access the relevant building
        feature) are imputed using the modal vulnerability score of each
        building's typology cluster (hot-deck imputation by donor class).
        EVI: D3_01 (H/t ratio), D3_04 (wall thickness), D5_04 (roof fastening).
        FVI: RoofConn, Geom, Apron, Drain, LWratio.
        Clusters with n < 10 donors fall back to the global modal score.
        Reference: Andridge & Little (2010).

    Stage 05 — MICE imputation (structural indicators)
        Remaining MAR-confirmed missing structural indicator values imputed via
        sklearn.impute.IterativeImputer (BayesianRidge). Surveyor ID included
        as auxiliary predictor (enumerator clustering confirmed in Step 5 of
        missing_data_diagnostic.py).
        Reference: van Buuren & Groothuis-Oudshoorn (2011).

    Stage 05b — MAO composite imputation
        Missing values in MAO (Motivation, Ability, Opportunity) composite
        scores imputed at the composite level using MICE. Composite-level
        imputation is correct because composites are proportional scores
        (sum of relevant items / sum of all items × 100); imputing raw binary
        items and recomputing would produce inconsistent denominators.

        Variables imputed (15-30% missing, MAR enumerator-driven pattern):
          Uti_Perc_Neg_expression    (17.7% missing before imputation)
          App_Perc_Neg_expression    (28.7%)
          AB_Selfefficacy_Neg        (17.6%)
          AB_Physical_capacity_Pos   (19.4%)
          AB_Physical_capacity_Neg   (19.4%)
          AB_Financial_capacity_Pos  (19.0%)
          AB_Financial_capacity_Neg  (19.6%)
          AB_Location_Pos            (19.0%)
          AB_Time_Pos                (19.7%)
          AB_Time_Neg                (19.3%)
          OP_Materials_Pos           (19.7%)
          OP_Materials_Neg           (19.7%)
          OP_Location_Pos            (19.1%)
          OP_Location_Neg            (19.1%)
          DE12_shelter_score         (16.6%)

        Variables NOT imputed (complete in analysis_dataset.csv):
          Uti_Perc_Pos_expression, AB_Selfefficacy_Pos, OP_Training_Pos/Neg,
          Acc_Perc_Pos_expression, and all socioeconomic / binary variables.

        Variables excluded by design (>50% missing — van Buuren 2018, §9.1.4):
          App_Perc_Pos_expression (52.2%), Acc_Perc_Neg_expression (87.4%),
          AB_Location_Neg (61.2%), OP_Manpower_Pos/Neg (92.5%),
          OP_Funding_Pos (72.1%), OP_Funding_Neg (85.8%).
          These were excluded before pipeline development and are absent from
          FVI_CANDIDATES / EVI_CANDIDATES in pipeline_config.py.

        Limitation: single imputation (m=1) underestimates standard errors.
        Multiple imputation (m=5-10) with Rubin (1987) pooling is the complete
        approach; treat as a limitation in the thesis.
        Reference: van Buuren (2018, §4.4, §6.3); Rubin (1987).

    Stage 05c — KNN flood experience imputation
        Flood experience perception scores imputed for flood-exposed households
        only (DE01_flood=1, n=2,824) using K-Nearest Neighbours (k=5).

        Imputed variables (MAR, enumerator-driven missingness):
          DE03_flood_frequency_score    (15.0% missing among flood-exposed)
          DE04_flood_depth_score        (18.9%)
          DE06_flood_future_score        (1.5%)
          DE08_worry_score               (0.3%)
          DE09_worry_future_score        (1.4%)
          DE13_feel_safe_score           (1.3%)

        Excluded variables (>50% threshold; van Buuren 2018, §9.1.4):
          DE05_negative_impact_score          (43.9% missing)
          DE07_negative_impact_future_score   (43.9% missing)

        KNN predictors (near-complete, substantively related to flood exposure):
          DE12_shelter_score, DE08_worry_score, DE09_worry_future_score,
          DE13_feel_safe_score, DE06_flood_future_score,
          FVI_norm_1_5, EVI_norm_1_5,
          HC003_Respondent_s_age, HC004_amount_people_household.

        Rationale for KNN over typology-based imputation: building typology
        clusters are defined on structural variables (wall/roof material, floor
        count) which carry no information about flood history. KNN defines
        neighbours using variables directly related to flood experience,
        producing substantively meaningful donors.

        Non-exposed households (DE01_flood=0, n=169) are not imputed. Flood
        experience scores are perception measures that only exist where flood
        exposure exists. Remaining NaN values in non-exposed households are
        handled by pairwise deletion at regression stages (08, 13).

        Sensitivity check over k = {3, 5, 7, 10} is written to
        imputation/knn_flood_imputation_report.txt.
        Reference: Troyanskaya et al. (2001); Acuna & Rodriguez (2004).

    Stage 06a — Variable missingness screening
        Checks all FVI and EVI candidate variables against analysis_dataset.csv.
        Removes any variable with >50% missing (van Buuren 2018, §9.1.4).
        Result: FVI 0 removed (44 retained). EVI 3 removed at 100% missing
        (PP01_rebuild_repair_house_002partially_damaged,
         PP02_rebuild_repair_house_001yes__minor_damage,
         PP02_rebuild_repair_house_002no).
        Writes four filtered variable lists:
          fvi_domains_filtered.csv     — for bivariate/stepwise stages (07, 08)
          evi_domains_filtered.csv     — for bivariate/stepwise stages (12, 13)
          fvi_candidates_filtered.csv  — full pre-bivariate pool for RF (Stage 18)
          evi_candidates_filtered.csv  — full pre-bivariate pool for RF (Stage 18)

IMPUTATION AND THE RANDOM FOREST (Stages 18, 18b)
--------------------------------------------------
The random forest receives analysis_dataset.csv, which already contains
Stage 05b MICE-imputed values for the 15 MAO variables. Variables not
imputed by Stage 05b (socioeconomic dummies, binary hazard indicators,
flood experience scales) are complete or nearly complete in the dataset,
so listwise deletion in the RF (via dropna()) retains all 2,993 rows.

Stage 18 reports an imputation audit table confirming which variables are
imputed, which are complete, and the missingness rate of any remaining
partial variables. This ensures the RF analysis is fully transparent.

Stage 18 uses fvi_candidates_filtered.csv / evi_candidates_filtered.csv
(the full pre-bivariate candidate pool, minus >50% missing variables) rather
than the domain-filtered lists. This gives the forest access to all 44 FVI
and 38 EVI candidate variables so that its importance ranking is independent
of the linear pipeline's variable selection.

PRIMARY vs SUPPLEMENTARY ANALYSIS
-----------------------------------
    PRIMARY:       Stages 10, 16 — mixed-effects linear models with
                   interpretable coefficients, community random intercepts,
                   and ICC estimates. These answer the research question.

    SUPPLEMENTARY: Stages 18, 18b — random forest for convergent validity
                   of variable selection and non-linearity detection.
                   Results reported in thesis appendix or robustness section.
                   RF cannot replace the primary analysis because it provides
                   no directional effects, no significance tests, no ICC,
                   and no community-level random intercepts.

OUTPUT DIRECTORY STRUCTURE
---------------------------
    missing_data_diagnostic_plots/             Stage 02a — indicator diagnostics
        fig1[evi/fvi]_missing_rate.png
        fig2[evi/fvi]_row_heatmap.png
        fig3[evi/fvi]_cooccurrence.png
        fig4[evi/fvi]_mar_auc.png
        fig5[evi/fvi]_mnar_sensitivity.png
        fig6[evi/fvi]_surveyor_clustering.png
        fig7[evi/fvi]_community_clustering.png
        fig8[evi/fvi]_temporal_clustering.png  (Spearman rho temporal trend)
    explanatory_vars_diagnostic_plots/         Stage 02b — explanatory diagnostics
        fig_exp01_missing_rate.png
        fig_exp02_row_heatmap.png
        fig_exp03_cooccurrence.png
        fig_exp04_mar_auc.png
        fig_exp05_surveyor_clustering.png
        fig_exp06_community_clustering.png
        fig_exp07_temporal_clustering.png      (Spearman rho temporal trend)
        fig_exp08_sample_comparison.png
    outputs/
        FVI_CIMDEN_scores.csv                  Raw FVI indicators (pre-imputation)
        EVI_CIMDEN_scores.csv                  Raw EVI indicators (pre-imputation)
        typology_clusters.csv                  Building typology assignments (k=8)
        typology_clusters/
            fig01_elbow_silhouette.png
            fig02_cluster_profile_heatmap.png
            fig03_cluster_size_bar.png
            fig04_pca_scatter.png
            fig05_validation_comparison.png
            typology_cluster_profiles.csv
        20263003_Nepal_Lumbini_data.csv        Raw survey (copied for analysis)
        imputation/
            EVI_imputed.csv                    EVI after MNAR imputation (Stage 04)
            FVI_imputed.csv                    FVI after MNAR imputation (Stage 04)
            imputed_EVI_scores.csv             EVI after MICE imputation (Stage 05)
            imputed_FVI_scores.csv             FVI after MICE imputation (Stage 05)
            mice_imputation_log.csv            Per-column imputation counts
            mice_mnar_sensitivity.csv          MNAR scenario comparison
            knn_flood_imputation_log.csv       Per-column KNN imputation counts (Stage 05c)
            knn_flood_imputation_report.txt    KNN audit trail incl. k sensitivity (Stage 05c)
            mnar_typology/
                fig_mnar_imputation_evi.png
                fig_mnar_imputation_fvi.png
                mnar_imputation_report.txt
        analysis_dataset.csv                   Merged dataset — input for all regressions
        mao_imputation_log.csv                 MAO MICE imputation counts (Stage 05b)
        mao_imputation_report.txt              MAO imputation audit trail (Stage 05b)
        variable_screening_report.csv          Missingness report per candidate (Stage 06a)
        variable_screening_summary.txt         Plain-text screening summary (Stage 06a)
        fvi_domains_filtered.csv               FVI domain vars for Stages 07/08
        evi_domains_filtered.csv               EVI domain vars for Stages 12/13
        fvi_candidates_filtered.csv            FVI candidates for Stage 18 RF
        evi_candidates_filtered.csv            EVI candidates for Stage 18 RF
        fvi_im_predictors.csv                  FVI stepwise predictors -> Stage 10
        evi_im_predictors.csv                  EVI stepwise predictors -> Stage 16
        fvi_distribution.png                   Stage 06
        fvi_indicator_frequencies.png          Stage 06
        bivariate_screening_results.csv        Stage 07
        bivariate_screening_significant.csv    Stage 07
        fvi_stepwise_*.csv / *.png             Stage 08
        fvi_morans_*.csv / *.png               Stage 09
        fvi_local_morans_results.csv           Stage 09
        fvi_local_morans_summary.txt           Stage 09
        fvi_mixed_effects_results.csv          Stage 10
        fvi_mixed_effects_variance.csv         Stage 10
        fvi_random_intercepts.csv / .png       Stage 10
        evi_distribution.png                   Stage 11
        evi_indicator_frequencies.png          Stage 11
        evi_bivariate_screening_*.csv          Stage 12
        evi_stepwise_*.csv / *.png             Stage 13
        evi_morans_*.csv / *.png               Stage 14
        evi_local_morans_results.csv           Stage 14
        evi_local_morans_summary.txt           Stage 14
        lisa_*.png                             Stage 15
        lisa_summary.csv                       Stage 15
        evi_mixed_effects_results.csv          Stage 16
        evi_mixed_effects_variance.csv         Stage 16
        evi_random_intercepts.csv / .png       Stage 16
        CVI_scores.csv                         Stage 17
        cvi_descriptives.csv                   Stage 17
        cvi_random_intercepts.csv / .png       Stage 17
        cvi_sensitivity_variance_normalisation.csv  Stage 17
        osm_tile_cache/                        Cached map tiles (Stages 09, 14, 15)
        random_forest/
            rf_fvi_metrics.csv                 Stage 18 — RF vs linear R² comparison
            rf_evi_metrics.csv                 Stage 18
            rf_fvi_importance.csv              Stage 18 — permutation importance
            rf_evi_importance.csv              Stage 18
            rf_fvi_importance.png              Stage 18 — importance chart
            rf_evi_importance.png              Stage 18
            rf_fvi_partial_dependence.png      Stage 18 — top-5 PD plots
            rf_evi_partial_dependence.png      Stage 18
            rf_predicted_vs_actual.png         Stage 18
            rf_comparison_summary.txt          Stage 18 — thesis appendix report
            validation/
                rfval_nested_cv.csv / .png     Stage 18b — Check 1
                rfval_learning_curves.png      Stage 18b — Check 2
                rfval_hyperparameter_sensitivity.csv / .png  Stage 18b — Check 3
                rfval_community_residuals.csv / .png         Stage 18b — Check 4
                rfval_calibration.csv / .png   Stage 18b — Check 5
                rfval_importance_stability_fvi/evi.csv / .png  Stage 18b — Check 6
                rfval_mdi_vs_permutation_fvi/evi.csv / .png    Stage 18b — Check 7
                rfval_predictor_correlation_fvi/evi.png         Stage 18b — Check 8
                rfval_validation_summary.txt   Stage 18b — full validation report
        gam/
            gam_fvi_summary.csv                Stage 25 — GAM term statistics, FVI
            gam_evi_summary.csv                Stage 25 — GAM term statistics, EVI
            gam_fvi_smooths.png                Stage 25 — smooth effect plots, FVI
            gam_evi_smooths.png                Stage 25 — smooth effect plots, EVI
            gam_fvi_predicted_vs_actual.png    Stage 25
            gam_evi_predicted_vs_actual.png    Stage 25
            gam_nonlinearity_report.csv        Stage 25 — EDF flags, both indices
            gam_comparison_summary.txt         Stage 25 — three-way comparison
            validation/                        Stage 26 — seven validation checks
        cross_method/
            cross_method_fvi.csv               Stage 27 — per-predictor agreement, FVI
                                                  Four methods compared:
                                                    OLS  — ordinary least squares
                                                    ME   — linear mixed-effects regression
                                                    RF   — Random Forest
                                                    GAM  — Generalised Additive Model
                                                  Columns include per-method significance
                                                  flags and agreement counts (0-4) under
                                                  strict and lenient RF thresholds.
            cross_method_evi.csv               Stage 27 — per-predictor agreement, EVI
                                                  Same four-method structure as FVI.
            cross_method_summary.txt           Stage 27 — readable agreement summary
                                                  Lists predictors at each agreement level
                                                  (4-of-4, 3-of-4, 2-of-4, 1-of-4) for
                                                  both strict and lenient RF thresholds.

KEY OUTPUT COLUMNS
------------------
    FVI_norm_1_5     Flood vulnerability index (1-5 scale)
                       Low < 2.333 | Medium 2.333-3.667 | High > 3.667
    EVI_norm_1_5     Earthquake vulnerability index (1-5 scale)
                       Same class boundaries as FVI
    CVI_norm_1_5     Composite vulnerability index (1-5 scale)
                       CVI = 0.5 x FVI + 0.5 x EVI (equal weights)

RANDOM INTERCEPTS (per community/VDC)
--------------------------------------
Written by Stages 10 (FVI), 16 (EVI), and 17 (CVI).

    community    VDC/ward name
    n            number of households in community
    u_j          random intercept BLUP estimate (deviation from grand mean)
    se_u         standard error of the BLUP
    ci_lo_95     lower 95% confidence interval
    ci_hi_95     upper 95% confidence interval
    sig          True if 95% CI does not cross zero
    direction    "above average" or "below average" vulnerability

PIPELINE CONFIGURATION
-----------------------
pipeline_config.py is the single source of truth for all analytical parameters:
    - FVI_CANDIDATES / EVI_CANDIDATES : full candidate variable pools
    - FVI_DOMAINS / EVI_DOMAINS       : post-bivariate domain groups
    - FVI_IM_PREDICTORS / EVI_IM_PREDICTORS : fallback predictor lists
    - FVI_PRED_LABELS / EVI_PRED_LABELS : human-readable labels for plots
    - CAP_COLS     : columns with known data-entry outliers (capped at 100)
    - COMMUNITY_COL, FVI_SCORE_COL, EVI_SCORE_COL : column name constants
    - BNDRY_LM, BNDRY_MH : vulnerability class boundaries (2.333, 3.667)
    - STEPWISE_PIN, STEPWISE_POUT : entry/exit thresholds (0.05, 0.10)
    - MORANS_THRESHOLD_KM : reporting threshold for Moran's I (1.0 km)
    - FVI_WEIGHT, EVI_WEIGHT : CVI component weights (0.5, 0.5)
    - RANDOM_STATE : random seed for reproducibility (42)

To add or remove a candidate variable: edit pipeline_config.py only.
No other files need editing. Re-run the pipeline from Stage 06a onwards.

STANDALONE SCRIPTS (run outside pipeline)
------------------------------------------
These diagnostic scripts run automatically as Stages 02a and 02b within
the pipeline. They can also be run independently on their own:

    missing_data_diagnostic.py       -> missing_data_diagnostic_plots/
    explanatory_vars_diagnostic.py   -> explanatory_vars_diagnostic_plots/

The following scripts remain standalone (not part of the pipeline) and
should be run separately if needed:

    fvi_missing_data_diagnostic.py   -> missing_data_plots/
    evi_missing_data_diagnostic.py   -> missing_data_plots/

REQUIREMENTS
------------
    Python 3.9+
    pip install pandas numpy scipy matplotlib statsmodels libpysal esda scikit-learn

FAIR RESEARCH PRINCIPLES
-------------------------
    Findable      : Consistent prefixes (fvi_, evi_, cvi_); subdirectory structure
    Accessible    : Single outputs/ tree; all file paths logged to pipeline.log
    Interoperable : CSV for all tabular data; PNG for all figures
    Reusable      : Self-contained pipeline; fully documented; random seed fixed (42)

REFERENCES
----------
Acuna, E. & Rodriguez, C. (2004). The treatment of missing values and its
    effect on classifier accuracy. In D. Banks et al. (eds.), Classification,
    Clustering, and Data Mining Applications. Springer.
    https://doi.org/10.1007/978-3-642-17103-1_60

Andridge, R.R. & Little, R.J.A. (2010). A Review of Hot Deck Imputation
    for Survey Non-Response. International Statistical Review, 78(1), 40-64.
    https://doi.org/10.1111/j.1751-5823.2010.00103.x

Anselin, L. (1995). Local indicators of spatial association — LISA.
    Geographical Analysis, 27(2), 93-115.
    https://doi.org/10.1111/j.1538-4632.1995.tb00338.x

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    https://doi.org/10.1023/A:1010933404324

Cawley, G.C. & Talbot, N.L.C. (2010). On over-fitting in model selection and
    subsequent selection bias in performance evaluation. JMLR, 11, 2079-2107.
    https://www.jmlr.org/papers/v11/cawley10a.html

Hartigan, J.A. & Wong, M.A. (1979). Algorithm AS 136: A K-means clustering
    algorithm. Journal of the Royal Statistical Society C, 28(1), 100-108.
    https://doi.org/10.2307/2346830

Hendriks, E. & Stokmans, M. (2020). Drivers and barriers of adoption of
    hazard-resistant construction knowledge. IJDRR, 51, 101778.
    https://doi.org/10.1016/j.ijdrr.2020.101778

Hox, J.J., Moerbeek, M., & Van de Schoot, R. (2018). Multilevel Analysis:
    Techniques and Applications (3rd ed.). Routledge.

Kandel, S. et al. (2024). Vernacular Architecture in Nepal.
    Nepal Engineers' Association Gandaki Province Technical Journal, 4.

Little, R.J.A. (1988). A test of missing completely at random.
    JASA, 83(404), 1198-1202.
    https://doi.org/10.1080/01621459.1988.10478722

Rijal, H.B. (2018). Nepal: Traditional Houses. In T. Kubota et al. (eds.),
    Sustainable Houses and Living in the Hot-Humid Climates of Asia.
    Springer. https://doi.org/10.1007/978-981-10-8465-2_6

Rousseeuw, P.J. (1987). Silhouettes: A graphical aid. JCAM, 20, 53-65.
    https://doi.org/10.1016/0377-0427(87)90125-7

Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
    Wiley. https://doi.org/10.1002/9780470316696

Saputra, A., Schwarz, J., & Hendriks, E. (2026). Identifying factors
    influencing housing safety in post-earthquake reconstruction in Nepal.
    IJDRR, 133, 105913. https://doi.org/10.1016/j.ijdrr.2025.105913

Strobl, C. et al. (2007). Bias in random forest variable importance measures.
    BMC Bioinformatics, 8, 25. https://doi.org/10.1186/1471-2105-8-25

Strobl, C. et al. (2008). Conditional variable importance for random forests.
    BMC Bioinformatics, 9, 307. https://doi.org/10.1186/1471-2105-9-307

Troyanskaya, O. et al. (2001). Missing value estimation methods for DNA
    microarrays. Bioinformatics, 17(6), 520–525.
    https://doi.org/10.1093/bioinformatics/17.6.520

van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
    CRC Press. https://stefvanbuuren.name/fimd/

van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
    Imputation by Chained Equations in R. Journal of Statistical Software,
    45(3). https://doi.org/10.18637/jss.v045.i03

Villagran De Leon, J.C. (2004). Vulnerability: A Conceptual and Methodological
    Review. UNU-EHS Source No. 4. United Nations University.


VERSION HISTORY
---------------
v46-cm (2026-05-31)
    NEW       Stage 27 added — Cross-Method Predictor Agreement
              (stage27_cross_method_comparison.py). A synthesis stage that
              joins the per-predictor outputs of four analytical methods
              for each single-hazard index (FVI, EVI) into a single
              agreement table:
                - Ordinary Least Squares (OLS) — from the Stage 15/21
                  OLS/ME comparison files
                - Linear mixed-effects regression (ME) — Stages 15, 21
                - Random Forest (RF) — Stage 23
                - Generalised Additive Model (GAM) — Stage 25
              The four methods sit at the corners of a 2x2 design across
              clustering correction (yes for ME; no for OLS, RF, GAM) and
              functional form (linear for OLS, ME; non-linear for RF, GAM).
              Pairwise contrasts isolate one analytical assumption each:
              OLS vs ME exposes clustering artefacts, OLS vs GAM exposes
              non-linearity, OLS vs RF exposes interaction effects.
              No new statistical estimation is performed. For each
              predictor the stage records which methods flagged it under
              published criteria (p < 0.05 for OLS, ME, GAM; top-decile
              strict / top-third lenient for RF permutation importance,
              with a minimum absolute-importance gate of 0.005), then
              counts agreement at strict and lenient thresholds on a
              0-to-4 scale.
              References:
                Fisher, R. A. (1925). Statistical Methods for Research
                  Workers. Oliver and Boyd.
                Strobl, C., et al. (2007). Bias in random forest variable
                  importance measures. BMC Bioinformatics, 8, 25.
                  https://doi.org/10.1186/1471-2105-8-25
                Wood, S. N. (2017). Generalized Additive Models: An
                  Introduction with R (2nd ed.). Chapman & Hall/CRC.
                  https://doi.org/10.1201/9781315370279
                Snijders, T. A. B., & Bosker, R. J. (2012). Multilevel
                  Analysis (2nd ed.). SAGE Publications.
              Outputs:
                outputs/cross_method/cross_method_fvi.csv
                outputs/cross_method/cross_method_evi.csv
                outputs/cross_method/cross_method_summary.txt
              README.txt and run_pipeline.py docstrings updated to reflect
              the new stage. Total pipeline stage count: 28.

v27 (2026-04-06)
    BUG FIX   missing_data_diagnostic.py (Stage 02a) — crashed with
              "Image data of dtype object cannot be converted to float"
              in the co-occurrence heatmap. Root cause: bool.T.dot(int)
              produces object dtype in pandas. Fixed by casting both sides
              to float64 before the dot product in step1_visualise():
                  miss_bool = df_ind.isna().astype(float)
                  co_occur  = miss_bool.T.dot(miss_bool) / n
              vmax also cast to float() defensively.

    BUG FIX   explanatory_vars_diagnostic.py (Stage 02b) — identical
              dtype object crash in the explanatory variable co-occurrence
              heatmap. Same fix applied.

    BUG FIX   evi_missing_data_diagnostic.py — identical dtype object
              crash. Same fix applied.

    WARNING   FutureWarning "Downcasting behavior in replace is
              deprecated" eliminated across all 19 affected scripts by
              chaining .infer_objects(copy=False) after every .replace()
              call on numeric data. This opts into the future pandas 2.x
              behaviour now rather than waiting for it to become an error.
              Files fixed:
                missing_data_diagnostic.py
                explanatory_vars_diagnostic.py
                evi_missing_data_diagnostic.py
                fvi_missing_data_diagnostic.py
                stage01_fvi_scoring.py
                stage02_evi_scoring.py
                stage03_typology_clusters.py
                stage05b_mao_imputation.py
                stage06a_variable_screening.py
                stage08_fvi_stepwise.py
                stage09_fvi_morans.py
                stage10_fvi_mixed_effects.py
                stage13_evi_stepwise.py
                stage14_evi_morans.py
                stage16_evi_mixed_effects.py
                stage18_random_forest.py
                stage18b_rf_validation.py
                stage19_gam.py
                stage19b_gam_validation.py

v26 (2026-04-06)
    Initial versioned release.

v27.1 (2026-04-06) — Clarity & Naming Refactor
    RENAME    All diagnostic scripts renamed to fit the sequential stage scheme:
                missing_data_diagnostic.py       -> stage03_indicator_missing_diagnostic.py
                explanatory_vars_diagnostic.py   -> stage04_explanatory_vars_diagnostic.py
                evi_missing_data_diagnostic.py   -> stage03a_evi_indicator_diagnostic.py
                fvi_missing_data_diagnostic.py   -> stage03b_fvi_indicator_diagnostic.py
              All subsequent stages renumbered 05–26 (previously 03–19b).

    NUMBERING Stages now run 01–26 with no sub-labels (previously 02a, 02b,
              05b, 05c, 06a, 18b, 19b). See STAGES table above.

    TYPOLOGY  Cluster auto-labels now carry five structural dimensions:
                Wall material · Roof material · Roof shape · Floors · Adjacency
              Example: "C0: Stone · CGI sheet · Flat · 1F · Attached"
              vs       "C4: Stone · Tile · Flat · 1F · Detached"
              This eliminates label collisions when wall+shape+floors match
              across clusters.

    TITLES    All chart and figure titles updated:
              • "Fig N —" and "Figure N —" prefixes removed from all plots.
              • Titles rewritten to describe chart content directly and
                unambiguously (axis context, what patterns mean, thresholds).

    DOCS      run_pipeline.py docstring and README stage tables updated to
              reflect all new stage numbers and filenames.

v27.2 (2026-04-06) — Composite Model Diagnostic Plot Fix
    BUG FIX   stage13_fvi_stepwise.py, stage18_evi_stepwise.py
              The bottom row of the 2×3 diagnostic plot (Composite Model)
              was always blank. Two bugs:
              1. Guard used `'fit_cm_f' in dir()` — dir() lists names in
                 the current scope but does not distinguish between assigned
                 and unassigned names inside conditional blocks. When no
                 composite predictors survived stepwise selection, fit_cm_f
                 was never assigned, dir() returned False, and the row was
                 skipped. Fixed by using `locals()` instead.
              2. When the composite model genuinely produces no predictors,
                 the three bottom subplots now display an informative
                 annotation ("Composite Model: no predictors survived
                 stepwise selection") instead of blank axes with default
                 0–1 tick scales.
              Same fix applied to both FVI (stage13) and EVI (stage18).

v27.3 (2026-04-06) — FVI10 Floor Scoring Revision + Environment Files
    SCORING   stage01_fvi_scoring.py, stage03b_fvi_indicator_diagnostic.py
              FVI10 (Number of Storeys) scoring scheme revised:
                Old: 1F → Low(1), 2F → Medium(3), 3F+ → High(5)
                New: 1F → High(5), 2–3F → Low(1), 4F+ → High(5)
              Rationale:
                1F = High: no vertical evacuation; full inundation exposure.
                2–3F = Low: upper floors enable vertical evacuation; added
                  structural mass improves resistance to flood lateral loads.
                4F+ = High: foundation overloading risk under flood-induced
                  soil saturation and scour.
              LIMITATION: SAL04 records only three binary floor columns
              (1st, 2nd, 3rd floor). The 4F+ High class cannot be scored
              from this dataset. nf_count >= 4 is unobservable. This
              limitation is documented in the script comments.
              Effective result: FVI10 is now BINARY in this dataset
              (1F → High=5, 2–3F → Low=1). Medium(3) never appears.

    ADDED     requirements.txt — full dependency list with version bounds
              and annotations for each package group.

    ADDED     SETUP.md — step-by-step environment setup instructions
              for conda (recommended) and pip/virtualenv, with
              troubleshooting section and FAIR reproducibility note.
