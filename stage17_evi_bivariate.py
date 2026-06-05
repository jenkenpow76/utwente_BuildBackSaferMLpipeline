"""
stage12_evi_bivariate.py
===================
Stage 3 | EVI — Bivariate Predictor Screening

Inputs  : imputed_EVI_scores.csv             (from Stage 0c)
          20263003_Nepal_Lumbini_data.csv    (raw survey — copied to OUTPUT_DIR)
Outputs : evi_bivariate_screening_results.csv
          evi_bivariate_screening_significant.csv  (p < 0.05 only)

Description
-----------
Mirrors stage07_fvi_bivariate.py exactly. Tests each candidate predictor
against EVI_norm_1_5 using the appropriate bivariate correlation:
  - Pearson r  : continuous MAO raw scores (0–100), age, household size
  - Spearman ρ : ordinal/binary predictors (CAT scores, education dummies,
                 occupation dummies, earthquake experience scores)

Predictors follow the same MAO + Socioeconomic domain structure as FVI:
  1. Motivation  (utility, applicability, acceptability perceptions)
  2. Ability     (self-efficacy, physical, financial, location, time)
  3. Opportunity (training, manpower, materials, location, funding)
  4. Earthquake Experience  ← domain specific to EVI
  5. Socioeconomic

MAO variables: same raw continuous scores (0–100) as FVI_3.
Education    : same binary dummies as FVI_3 (not the ordinal composite).
Occupation   : same set as FVI_3 (agriculture, day_labour,
               construction_worker, education, remittances, business,
               government_officer).

Threshold for advancement: p < 0.05 (two-tailed).

References
----------
Hendriks, E. & Stokmans, M. (2020). Drivers and barriers of adoption of
    hazard-resistant construction knowledge. IJDRR, 51, 101778.
    https://doi.org/10.1016/j.ijdrr.2020.101778

Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.). Lawrence Erlbaum.
    r ≥ 0.10 = small, r ≥ 0.30 = medium, r ≥ 0.50 = large effect.
"""

# ─── Pipeline note ───────────────────────────────────────────────────────────
# When run via run_evi_pipeline.py, the working directory is OUTPUT_DIR.
# Both SCORED_CSV and SURVEY_CSV resolve to files in OUTPUT_DIR.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from pipeline_config import (
    EVI_CANDIDATES,
CAP_COLS,
)

# =============================================================================
# FILE PATHS
# =============================================================================

SCORED_CSV = str(Path.cwd() / "analysis_dataset.csv")           # output of stage02_evi_scoring.py
SURVEY_CSV = str(Path.cwd() / "20263003_Nepal_Lumbini_data.csv")  # raw survey (same as FVI_3)
OUT_ALL    = str(Path.cwd() / "evi_bivariate_screening_results.csv")
OUT_SIG    = str(Path.cwd() / "evi_bivariate_screening_significant.csv")

P_THRESHOLD = 0.05   # significance threshold for advancement to stepwise

# =============================================================================
# CANDIDATE PREDICTORS BY DOMAIN
# =============================================================================
# Format: (column_name, human_label, test_type)
# test_type = 'pearson'  — continuous 0–100 raw scores
# test_type = 'spearman' — ordinal CAT scores, binary items, Likert scores
#
# MAO variables use the SAME raw score columns as FVI_3 (not CAT_33PROCENT).
# Education uses the SAME binary dummies as FVI_3.
# Occupation uses the SAME set as FVI_3.
# Earthquake experience variables are specific to EVI (confirmed correct).

# =============================================================================
# CANDIDATE PREDICTORS BY DOMAIN
# =============================================================================
# Stage 06a writes evi_domains_filtered.csv listing variables that survived
# the >50% missing threshold (van Buuren 2018, §9.1.4). If that file exists,
# we rebuild CANDIDATES from it so that removed variables are never screened.
# Falls back to the full EVI_CANDIDATES from pipeline_config.py if the file
# is not present (e.g. when running Stage 12 standalone).

_filtered_path = Path.cwd() / "evi_domains_filtered.csv"
if _filtered_path.exists():
    _filtered_df = pd.read_csv(_filtered_path)
    _filtered_vars = set(_filtered_df["variable"].tolist())
    CANDIDATES = {
        domain: [(col, label, test)
                 for col, label, test in var_list
                 if col in _filtered_vars]
        for domain, var_list in EVI_CANDIDATES.items()
    }
    CANDIDATES = {d: v for d, v in CANDIDATES.items() if v}
    print(f"Loaded filtered EVI candidate list from stage06a "
          f"({sum(len(v) for v in CANDIDATES.values())} variables).")
else:
    CANDIDATES = EVI_CANDIDATES
    print("evi_domains_filtered.csv not found — using full EVI_CANDIDATES from config.")

# =============================================================================
# LOAD DATA
# =============================================================================

scored = pd.read_csv(SCORED_CSV, encoding='utf-8-sig', low_memory=False)
survey = pd.read_csv(SURVEY_CSV, encoding='utf-8-sig', low_memory=False)

# Force EVI outcome to numeric
scored["EVI_norm_1_5"] = pd.to_numeric(scored["EVI_norm_1_5"], errors="coerce")
outcome = scored["EVI_norm_1_5"]

print(f"Outcome variable: EVI_norm_1_5  "
      f"(n valid = {outcome.notna().sum():,})\n")

# =============================================================================
# DATA QUALITY: cap known outlier columns before testing  (mirrors FVI_3)
# =============================================================================

for col in CAP_COLS:
    if col in survey.columns:
        s = pd.to_numeric(survey[col], errors="coerce")
        n_outliers = (s > 100).sum()
        if n_outliers > 0:
            print(f"  Data quality note: {col} — {n_outliers} values > 100 "
                  "capped to NaN before testing.")
            survey[col] = s.where(s <= 100)

# =============================================================================
# BIVARIATE SCREENING
# =============================================================================

results = []

for domain, predictors in CANDIDATES.items():
    domain_label = domain.split("_", 1)[1].replace("_", " ")

    for col, label, test in predictors:

        # Pull predictor from the correct dataframe
        src = scored if col in scored.columns else survey
        if col not in src.columns:
            print(f"  WARNING: '{col}' not found — skipped.")
            continue

        x = pd.to_numeric(src[col], errors="coerce")

        # Pairwise complete cases
        pair   = pd.DataFrame({"x": x.values, "y": outcome.values}).dropna()
        n_pair = len(pair)

        if n_pair < 30:
            print(f"  SKIPPED (n={n_pair} < 30): {label}")
            continue

        x_arr = pair["x"].values
        y_arr = pair["y"].values

        # Appropriate test: Pearson for continuous, Spearman for ordinal/binary
        if test == "pearson":
            r, p = stats.pearsonr(x_arr, y_arr)
            stat_name = "r"
        else:
            r, p = stats.spearmanr(x_arr, y_arr)
            stat_name = "rho"

        # Effect size label (Cohen, 1988)
        abs_r  = abs(r)
        effect = "small" if abs_r >= 0.10 else "negligible"
        if abs_r >= 0.30:
            effect = "medium"
        if abs_r >= 0.50:
            effect = "large"

        results.append({
            "Domain":      domain_label,
            "Column":      col,
            "Label":       label,
            "Test":        test.capitalize(),
            "Statistic":   stat_name,
            "r_or_rho":    round(r, 4),
            "p_value":     round(p, 4),
            "n":           n_pair,
            "Effect_size": effect,
            "Significant": "Yes" if p < P_THRESHOLD else "No",
        })

# =============================================================================
# ASSEMBLE AND SAVE
# =============================================================================

results_df = pd.DataFrame(results).sort_values(["Domain", "p_value"])
sig_df     = results_df[results_df["Significant"] == "Yes"].copy()

results_df.to_csv(OUT_ALL, index=False)
sig_df.to_csv(OUT_SIG,     index=False)
print(f"Saved: {OUT_ALL}  ({len(results_df)} predictors tested)")
print(f"Saved: {OUT_SIG}   ({len(sig_df)} significant at p < {P_THRESHOLD})")

# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print(f"EVI BIVARIATE SCREENING RESULTS  (threshold: p < {P_THRESHOLD})")
print("=" * 70)

for domain in results_df["Domain"].unique():
    sub = results_df[results_df["Domain"] == domain]
    sig = sub[sub["Significant"] == "Yes"]
    print(f"\n  {domain.upper()}  "
          f"({len(sig)}/{len(sub)} significant)")
    print(f"  {'Label':<42} {'Stat':>5}  {'r/rho':>7}  {'p':>7}  {'n':>5}  Effect")
    print(f"  {'-'*42} {'-'*5}  {'-'*7}  {'-'*7}  {'-'*5}  ------")
    for _, row in sub.iterrows():
        flag = "*" if row["Significant"] == "Yes" else " "
        print(f"  {flag}{row['Label']:<41} "
              f"{row['Statistic']:>5}  "
              f"{row['r_or_rho']:>7.4f}  "
              f"{row['p_value']:>7.4f}  "
              f"{row['n']:>5}  "
              f"{row['Effect_size']}")

print(f"\n{'='*70}")
print(f"Total tested         : {len(results_df)}")
print(f"Significant (p<{P_THRESHOLD}) : {len(sig_df)}")
print(f"{'='*70}")
print("\nVariables advancing to domain regression:")
for _, row in sig_df.iterrows():
    print(f"  [{row['Domain'][:3].upper()}] {row['Column']:<45}  "
          f"r={row['r_or_rho']:>7.4f}  p={row['p_value']:.4f}")
