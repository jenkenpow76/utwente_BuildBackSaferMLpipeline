# RUN_ORDER — pipeline_v47

Step-by-step instructions for re-running the pipeline after the v46 → v47 update.

---

## Prerequisites

### Python environment

Ensure the following packages are available:

```
pandas
numpy
scipy
scikit-learn
statsmodels          ← required for OLS, mixed-effects, GAM
pygam                ← required for Stage 25, 26
geopandas            ← required for spatial stages
esda                 ← required for Moran's I
libpysal             ← required for spatial weights
matplotlib
seaborn
```

If any are missing, install them with `pip install -r requirements.txt`.

### Input data

The pipeline expects these files in the working directory:

- `20263003_Nepal_Lumbini_data.csv` — raw survey data
- `EVI_CIMDEN_scores.csv`           — EVI indicator scores (output of Stage 02)
- `FVI_CIMDEN_scores.csv`           — FVI indicator scores (output of Stage 01)
- `ward_information.csv`            — ward-level survey

---

## Recommended sequence

Stages can be run individually or via `run_pipeline.py`. The v47 changes affect the candidate pool, so **every stage from Stage 10 onwards must be re-run**.

### Stages that DO NOT need to be re-run

Stages 01-09 are unaffected by the v47 changes. Their outputs in `outputs/` can be reused. If you want to re-run them anyway for full reproducibility, you can — they will produce identical output.

| Stage | What it does | Re-run needed? |
|---|---|---|
| 01 | FVI scoring | No |
| 02 | EVI scoring | No |
| 03a, 03b | Indicator missingness diagnostic | No |
| 04 | Predictor missingness diagnostic | Optional (PP02 mislabel was fixed) |
| 05 | Typology clusters | No |
| 06 | MNAR hot-deck imputation | No |
| 07 | MICE imputation (indicators) | No |
| 08 | MICE imputation (MAO composites) | No |
| 08b | Ward covariates | No |
| 09 | KNN flood imputation | No |

### Stages that MUST be re-run

| # | Stage | Reason |
|---|---|---|
| 1 | `stage10_variable_screening.py` | Now filters CANDIDATES not DOMAINS |
| 2 | `stage11_fvi_descriptives.py` | (unchanged, but check completes cleanly) |
| 3 | `stage12_fvi_bivariate.py` | Will produce new `bivariate_screening_significant.csv` |
| 4 | `stage13_fvi_stepwise.py` | Reads new bivariate output; writes new `fvi_im_predictors.csv` |
| 5 | `stage14_fvi_morans.py` | Uses new `fvi_im_predictors.csv` |
| 6 | `stage15_fvi_mixed_effects.py` | Uses new `fvi_im_predictors.csv` |
| 7 | `stage16_evi_descriptives.py` | (unchanged, check completes cleanly) |
| 8 | `stage17_evi_bivariate.py` | Will produce new `evi_bivariate_screening_significant.csv` |
| 9 | `stage18_evi_stepwise.py` | Reads new bivariate output; writes new `evi_im_predictors.csv` |
| 10 | `stage19_evi_morans.py` | Uses new `evi_im_predictors.csv` |
| 11 | `stage20_lisa_cluster_maps.py` | Uses new IM predictors |
| 12 | `stage21_evi_mixed_effects.py` | Uses new `evi_im_predictors.csv` |
| 13 | `stage22_composite_cvi.py` | Uses outputs of stages 13 and 18 |
| 14 | `stage23_random_forest.py` | Uses new candidate pool |
| 15 | `stage24_rf_validation.py` | Uses stage 23 output |
| 16 | `stage25_gam.py` | Uses new IM predictors |
| 17 | `stage26_gam_validation.py` | Uses stage 25 output |
| 18 | `stage27_cross_method_comparison.py` | Aggregates all model outputs |

---

## How to run

### Option A — Run individually (recommended for first pass)

This lets you inspect output of each stage. Recommended for the first v47 run so you can catch any unexpected behaviour.

```bash
cd pipeline_v47

# FVI side
python3 stage10_variable_screening.py    | tee outputs/stage_logs/stage_10.txt
python3 stage11_fvi_descriptives.py      | tee outputs/stage_logs/stage_11.txt
python3 stage12_fvi_bivariate.py         | tee outputs/stage_logs/stage_12.txt
python3 stage13_fvi_stepwise.py          | tee outputs/stage_logs/stage_13.txt
python3 stage14_fvi_morans.py            | tee outputs/stage_logs/stage_14.txt
python3 stage15_fvi_mixed_effects.py     | tee outputs/stage_logs/stage_15.txt

# EVI side
python3 stage16_evi_descriptives.py      | tee outputs/stage_logs/stage_16.txt
python3 stage17_evi_bivariate.py         | tee outputs/stage_logs/stage_17.txt
python3 stage18_evi_stepwise.py          | tee outputs/stage_logs/stage_18.txt
python3 stage19_evi_morans.py            | tee outputs/stage_logs/stage_19.txt
python3 stage20_lisa_cluster_maps.py     | tee outputs/stage_logs/stage_20.txt
python3 stage21_evi_mixed_effects.py     | tee outputs/stage_logs/stage_21.txt

# Composite + supplementary
python3 stage22_composite_cvi.py         | tee outputs/stage_logs/stage_22.txt
python3 stage23_random_forest.py         | tee outputs/stage_logs/stage_23.txt
python3 stage24_rf_validation.py         | tee outputs/stage_logs/stage_24.txt
python3 stage25_gam.py                   | tee outputs/stage_logs/stage_25.txt
python3 stage26_gam_validation.py        | tee outputs/stage_logs/stage_26.txt
python3 stage27_cross_method_comparison.py | tee outputs/stage_logs/stage_27.txt
```

### Option B — Run everything via `run_pipeline.py`

If the individual run succeeds, future re-runs can use the orchestrator:

```bash
python3 run_pipeline.py
```

---

## What to look for in the output

### At the top of stages 13, 15, 18, and 21

You should see:

```
  assert_no_drift: PASSED — no drift detected.
```

If you see drift instead, the error message will tell you which list violates which invariant. Re-run the affected upstream stage.

### Stage 12 / Stage 17 (bivariate)

The new bivariate output will reflect the v47 candidate pool. The previously significant variables FA01 and RT02 will not appear in the output because they are no longer in the EVI candidate pool. The "Earthquake_Experience" section will only test DE01 and DE02.

### Stage 13 / Stage 18 (stepwise)

The Individual Model variable list will be smaller than v46, because the v46 model included variables that had bypassed bivariate screening. Expect 6-8 variables for FVI and 4-6 variables for EVI in the new IM.

### Stage 22 (Composite CVI)

The CVI ranking should be qualitatively similar to v46 (same buildings flagged as Low/Medium/High) but the underlying coefficients will differ.

---

## If something fails

Most likely failure modes:

1. **`FileNotFoundError: Cannot derive *_DOMAINS — bivariate output not found`**
   You skipped or did not complete the bivariate stage. Re-run Stage 12 (FVI) or Stage 17 (EVI) first.

2. **`AssertionError: DRIFT: ...`**
   The lazy loader detected drift between the bivariate output and the stepwise output. Re-run the offending stepwise stage.

3. **`ImportError: No module named statsmodels`**
   Install the missing package via `pip install statsmodels`.

If you hit any error not in the above list, paste the full traceback into the chat. I will diagnose and propose a fix.

---

## After all stages complete successfully

Two things to verify:

1. **The drift assertion passes everywhere.** Search the stage logs for `assert_no_drift: PASSED`.
2. **Re-package and upload back to me.** Zip the entire `pipeline_v47/` directory (including the regenerated `outputs/`) and upload it. I will then:
   - Verify the new outputs are consistent
   - Update Chapter 2 to reflect the new variable lists and any qualitative changes in the Results section
   - Build any new figures or tables that need updating
