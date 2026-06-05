# CHANGELOG — pipeline_v47

**Date:** 3 June 2026
**Built by:** Claude (Anthropic) on behalf of Jennie Powers (University of Twente, ITC)
**Companion to:** Chapter 2 (Data and Methods) of the master thesis
**Predecessor:** pipeline_v46 (May 2025)

---

## Summary

`pipeline_v47` fixes a pipeline integrity problem found in v46. Three hand-coded variable lists (`*_CANDIDATES`, `*_DOMAINS`, `*_IM_PREDICTORS`) were supposed to be related by simple filtering rules. Over time these lists drifted out of sync, so the documented relationship between them became false. v47 replaces the hand-coded `*_DOMAINS` and `*_IM_PREDICTORS` constants with **lazy-loading functions that derive them automatically from the upstream output CSVs at access time**. The hand-coded `*_CANDIDATES` lists remain as the single source of truth.

A runtime invariant function `assert_no_drift()` is added and called at the top of stages 13, 15, 18, and 21. Drift now fails loud rather than being silently absorbed.

Six additional variables are removed from the candidate pool because the SPSS syntax for the MAO composites shows they are already embedded inside MAO Utility, Applicability, or Acceptability scores.

---

## 1 — Pipeline integrity problem fixed

### What was wrong in v46

The header comment for `EVI_DOMAINS` claimed:

> *"variables that passed bivariate screening (p < 0.05)"*

The actual contents drifted from this claim over time:
- **22 of 40** variables in `EVI_DOMAINS` did **not** pass bivariate screening
- **6** of those 22 variables appeared in the final `EVI_IM_PREDICTORS` list
- **1 variable** (`DE12_shelter_score`) appeared in `EVI_IM_PREDICTORS` without ever appearing in `EVI_DOMAINS`
- **2 variables** (`PP01_rebuild_repair_house_001no`, `PP12_own_land`) appeared in `EVI_DOMAINS` without ever appearing in `EVI_CANDIDATES`

### Why this matters

A defender of v46 could argue these were intentional methodological choices (e.g., MAO pair-coherence override). But the code did not document them as choices; it claimed bivariate filtering and silently violated that claim. Any examiner who compared `EVI_DOMAINS` against the published Stage 17 output would see the mismatch.

### The architectural fix in v47

In `pipeline_config.py`:

- **`FVI_DOMAINS`, `EVI_DOMAINS`, `FVI_ALL_DOMAIN_VARS`, `EVI_ALL_DOMAIN_VARS`, `FVI_IM_PREDICTORS`, `EVI_IM_PREDICTORS`** are no longer module-level constants. They are accessed via PEP 562 `__getattr__`, which reads from the upstream CSV at access time:
  - `*_DOMAINS` reads from `outputs/bivariate_screening_significant.csv` (Stage 12) or `outputs/evi_bivariate_screening_significant.csv` (Stage 17)
  - `*_IM_PREDICTORS` reads from `outputs/fvi_im_predictors.csv` (Stage 13) or `outputs/evi_im_predictors.csv` (Stage 18)
- **`assert_no_drift()`** is a new function that verifies no drift between the three lists. It is called at the top of Stages 13, 15, 18, and 21.
- **Strict bivariate advancement** is now enforced. Variables that fail bivariate cannot enter the stepwise candidate pool.

If a downstream stage tries to access `FVI_IM_PREDICTORS` before Stage 13 has produced its output, the error message identifies the missing dependency explicitly:

> *Cannot derive `*_IM_PREDICTORS` — stepwise output not found at outputs/fvi_im_predictors.csv. Run stage13_fvi_stepwise.py (FVI) or stage18_evi_stepwise.py (EVI) first.*

---

## 2 — Variable changes in `*_CANDIDATES`

### Removed from `EVI_CANDIDATES`

These variables are removed because they are **already inputs to the MAO composite scores** (per `Syntax_Motivation_02_01_25__4_.sps`). Adding them as standalone predictors would double-count their information and would introduce by-construction collinearity with the MAO composites.

| Variable | Reason for removal |
|---|---|
| `PP01_rebuild_repair_house_001totally_damaged` | Embedded in MAO Utility composite (denominator) |
| `PP01_rebuild_repair_house_002partially_damaged` | Same |
| `PP02_rebuild_repair_house_001yes__minor_damage` | Same |
| `PP02_rebuild_repair_house_002no` | Same |
| `FA01_If_you_would_have_more_money_winvest_in_current_house` | No relationship with EVI in v46 bivariate (ρ=0.006, p=0.75); also indirectly captured in MAO Utility/Applicability |
| `RT02_household_reconstruction_orien` | Small effect (ρ=–0.106, p<0.0001) but dropped at stepwise; also embedded in MAO Opportunity |

### Not added to either index

These were considered for inclusion but rejected on the same MAO-collinearity grounds:

| Variable | Reason for non-inclusion |
|---|---|
| `PP12_own_land` | Embedded in MAO Utility denominator (all 11 PP12 dummies) |
| PP01 collapsed 3-level categorical | Same |
| PP02 collapsed 3-level categorical | Same |

### Unchanged

`FVI_CANDIDATES` has no variable changes. `DE12_shelter_score` was already in `FVI_CANDIDATES` (Flood_Experience domain) — no move required.

---

## 3 — Citation fix

| File | Line | v46 | v47 |
|---|---|---|---|
| `pipeline_config.py` | 594 | `Villagran De León 2004` | `Villagrán De León 2006` |

The correct publication is *Vulnerability: A Conceptual and Methodological Review*, SOURCE Publication Series of UNU-EHS No. 4/2006 (Bonn, ISBN 3-9810582-4-0). Verified against the UNU-EHS publication record.

---

## 4 — Stage-by-stage changes

### `pipeline_config.py`

- Line 594: Villagrán citation typo fixed.
- Lines 408-421: EVI_CANDIDATES `4_Earthquake_Experience` block: removed 4 PP01/PP02 dummies (mislabelled as "preparedness" in v46), kept only DE01_earthquake and DE02_earthquake. Added explanatory comment about MAO embedding.
- Lines 437-441 (approx): EVI_CANDIDATES `5_Socioeconomic` block: removed FA01 and RT02.
- Lines 280-510 (approx): Replaced hand-coded FVI_DOMAINS, EVI_DOMAINS, FVI_ALL_DOMAIN_VARS, EVI_ALL_DOMAIN_VARS, FVI_IM_PREDICTORS, EVI_IM_PREDICTORS with the lazy-loading architecture.
- Added `_load_domains_from_bivariate()`, `_load_im_predictors()`, `assert_no_drift()`, and module-level `__getattr__()`.

### `stage04_explanatory_vars_diagnostic.py`

- Lines 214-217: Fixed mislabel — PP02 dummies were labelled "Preparedness" but are actually "Flood rebuild". Updated all four PP01/PP02 label entries.
- Added clarifying comment to the local `FVI_IM_PREDICTORS` and `EVI_IM_PREDICTORS` snapshot lists noting they are used only for diagnostic-plot colour-coding and not as a source of truth.

### `stage10_variable_screening.py`

- Lines 83-89: Stopped importing `FVI_DOMAINS` and `EVI_DOMAINS` at module load time. Stage 10 now operates on `*_CANDIDATES` (which is the right object — DOMAINS does not exist until after bivariate runs).
- Lines 236-275 (approx): Updated filter logic to convert CANDIDATES → domain dict, then remove the excluded variables. New helper function `candidates_to_domain_dict()`.

### `stage13_fvi_stepwise.py`

- Lines 84-95: Added `assert_no_drift` to the import list. Added explicit `assert_no_drift()` call right after imports complete.

### `stage15_fvi_mixed_effects.py`

- Lines 105-115: Added `assert_no_drift` import and call.

### `stage18_evi_stepwise.py`

- Lines 88-99: Added `assert_no_drift` import and call.

### `stage21_evi_mixed_effects.py`

- Lines 98-110: Added `assert_no_drift` import and call.

### `stage08_mao_imputation.py`

- **No changes.** FA01 and RT02 are retained in the `COMPLETE_COLS` auxiliary list because they help impute missing MAO items. They are valid imputation auxiliaries even though they are no longer substantive predictors. This is a deliberate choice (per supervisor agreement) — see Section 2.5.10 of the chapter for the methodological rationale.

---

## 5 — Files preserved for reference

The `_v46_originals/` subdirectory contains unmodified copies of every v46 file that was changed in v47. Each file has a `.v46.py` suffix. Use these for diffing or rollback if needed.

```
_v46_originals/
├── pipeline_config.v46.py
├── stage04_explanatory_vars_diagnostic.v46.py
├── stage10_variable_screening.v46.py
├── stage13_fvi_stepwise.v46.py
├── stage15_fvi_mixed_effects.v46.py
├── stage18_evi_stepwise.v46.py
└── stage21_evi_mixed_effects.v46.py
```

---

## 6 — What you need to do after pulling v47

See `RUN_ORDER.md` for the exact sequence. The short version:

1. Verify your local Python environment has the required packages (statsmodels, pygam, geopandas, esda, libpysal — see `requirements.txt`).
2. Run the affected stages in order (everything from Stage 12 onwards needs to re-run because the candidate pool and `*_IM_PREDICTORS` lists have changed).
3. Verify `assert_no_drift()` passes at the top of each re-run stage.
4. Inspect the new outputs in `outputs/`. The new `FVI_IM_PREDICTORS` and `EVI_IM_PREDICTORS` will differ from v46 because the candidate pool changed.

---

## 7 — Honest caveats

- **Methodological impact on results.** Removing FA01, RT02, and the four PP01/PP02 dummies will change the EVI stepwise output. The most likely effect is fewer variables in the final EVI Individual Model. The MAO composites (which already carry the information from those PP variables) should pick up most of the lost signal, but the exact effect cannot be known until you re-run.
- **MAO syntax not fully verified.** Only the Motivation MAO syntax (Utility, Applicability, Acceptability) was reviewed. The Ability and Opportunity composite construction syntax files were not seen. The conservative position is that PP01/PP02/PP12 are *probably* inputs to those composites too. If you later upload those syntax files and PP variables are NOT in them, you could revisit and re-add them.
- **Stage 04 local snapshot lists are stale.** The diagnostic-plot colour coding will not reflect the new Individual Model until those snapshots are updated by hand. This is purely cosmetic — the actual analytical results in Chapter 3 will be correct.

---

# PIPELINE_V49 ADDENDUM — PP01 and PP02 added as 3-level categoricals

**Date:** 3 June 2026
**Predecessor:** pipeline_v48

## Summary

After review of Garbhit's `Final_POS_NEG_Coding_Variables_MAO_Constructs__1_.xlsx` (which turned out to be a blank coding template, not a marked coding key) and a closer re-read of the SPSS Motivation syntax (`Syntax_Motivation_02_01_25__4_.sps`), we determined that PP01 (earthquake rebuild history) and PP02 (flood rebuild history) are NOT inputs to the Motivation MAO composite scores. The earlier blanket exclusion was over-cautious. v49 re-adds these variables in Option A-pragmatic form (3-level collapsed categorical).

## Variable changes

### Added to FVI_CANDIDATES (Flood_Experience)
- `PP02_flood_rebuild_3lvl_damaged` (Spearman) — reference: undamaged
- `PP02_flood_rebuild_3lvl_unknown_other` (Spearman) — reference: undamaged

### Added to EVI_CANDIDATES (Earthquake_Experience)
- `PP01_eq_rebuild_3lvl_damaged` (Spearman) — reference: undamaged
- `PP01_eq_rebuild_3lvl_unknown_other` (Spearman) — reference: undamaged

## Architectural change

The new 3-level dummies are constructed in `stage08b_ward_covariates.py` from the parent categorical fields (`PP01_rebuild_repair_house_001` and `PP02_rebuild_repair_house_001_001`). The construction is deterministic with no missing values — every row maps to one of three levels. The reference category "undamaged" is encoded by both dummies = 0.

### Distribution audit (from v48 analysis_dataset.csv)
- **PP01 (earthquake)**: undamaged 2,505 (84%), damaged 416 (14%), unknown_other 72 (2%)
- **PP02 (flood)**: damaged 1,426 (48%), undamaged 809 (27%), unknown_other 758 (25%)

## Methodological caveat

Verified that PP01 and PP02 are not used in the Motivation MAO syntax (Utility, Applicability, Acceptability composites). The Ability and Opportunity MAO syntax files were not available for verification. This residual methodological uncertainty must be documented in Section 5.4 (Limitations) of the thesis.

## Pipeline counts

- FVI candidates: 44 → 46 (added 2)
- EVI candidates: 35 → 37 (added 2)

## What you need to do

Re-run the pipeline. The new candidate variables flow through stages 12 onwards. Stage 08b will construct the new dummies during the first re-run; the new analysis_dataset.csv will include them as columns.
