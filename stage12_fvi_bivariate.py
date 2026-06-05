"""
stage07_fvi_bivariate.py
===================
Stage 3 | Bivariate Predictor Screening

Inputs  : imputed_FVI_scores.csv             (from Stage 0c)
          20263003_Nepal_Lumbini_data.csv    (raw survey — copied to OUTPUT_DIR)
Outputs : bivariate_screening_results.csv   — all tested predictors
          bivariate_screening_significant.csv — predictors at p < 0.05

Description
-----------
Tests each candidate predictor against FVI_norm_1_5 using the appropriate
bivariate correlation (Pearson r for continuous variables; Spearman ρ for
ordinal/binary variables). Predictors are grouped into five MAO domains:
Motivation, Ability, Opportunity, Flood Experience, and Socioeconomic.

Only predictors significant at p < 0.05 advance to Stage 4.

References
----------
Hendriks, E. & Stokmans, M. (2020). Drivers and barriers of adoption of
    hazard-resistant construction knowledge. IJDRR, 51, 101778.
    https://doi.org/10.1016/j.ijdrr.2020.101778

Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.). Lawrence Erlbaum.
"""

# ─── Pipeline note ────────────────────────────────────────────────────────────
# When run via run_fvi_pipeline.py, the working directory is OUTPUT_DIR.
# Both SCORED_CSV and SURVEY_CSV resolve to files in OUTPUT_DIR.
# ─────────────────────────────────────────────────────────────────────────────

"""
FVI — Bivariate Screening of Candidate Predictors
===================================================
Tests each candidate predictor against the outcome variable
(FVI_Norm_1_5) using the appropriate bivariate correlation:

  - Pearson r  : continuous predictors (MAO raw scores 0-100,
                 age, household size, flood scores)
  - Spearman ρ : ordinal predictors (CAT 1-5 variables,
                 Likert-type scores, binary SE items)

Predictors are grouped into five domains following Hendriks (2020):
  1. Motivation  (utility, applicability, acceptability perceptions)
  2. Ability     (self-efficacy, physical, financial, location, time)
  3. Opportunity (training, manpower, materials, location, funding)
  4. Flood Experience
  5. Socioeconomic

Threshold for advancement: p < 0.05 (two-tailed).

Inputs  : imputed_FVI_scores.csv   (output of fvi_cimden.py)
          20263003_Nepal_Lumbini_data.csv  (original survey data)
Outputs : bivariate_screening_results.csv
          bivariate_screening_significant.csv  (p < 0.05 only)

References:
  Hendriks, E. & Stokmans, M. (2020). Drivers and barriers of adoption
    of hazard-resistant construction knowledge. International Journal of
    Disaster Risk Reduction, 51, 101778.
    https://doi.org/10.1016/j.ijdrr.2020.101778

  Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.). Lawrence Erlbaum.
    r ≥ 0.10 = small, r ≥ 0.30 = medium, r ≥ 0.50 = large effect.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from pipeline_config import (
    FVI_CANDIDATES,
CAP_COLS,
)

# ── File paths ────────────────────────────────────────────────────────────────
# Both files must be in the same directory as this script.
SCORED_CSV  = str(Path.cwd() / "analysis_dataset.csv")          # output of fvi_cimden.py
SURVEY_CSV  = str(Path.cwd() / "20263003_Nepal_Lumbini_data.csv")
OUT_ALL     = str(Path.cwd() / "bivariate_screening_results.csv")
OUT_SIG     = str(Path.cwd() / "bivariate_screening_significant.csv")

P_THRESHOLD = 0.05   # significance threshold for advancement

# ── Candidate predictors by domain ───────────────────────────────────────────
# Stage 06a writes fvi_domains_filtered.csv listing variables that survived
# the >50% missing threshold (van Buuren 2018, §9.1.4). If that file exists,
# we rebuild CANDIDATES from it so that removed variables are never screened.
# Falls back to the full FVI_CANDIDATES from pipeline_config.py if the file
# is not present (e.g. when running Stage 07 standalone).

_filtered_path = Path.cwd() / "fvi_domains_filtered.csv"
if _filtered_path.exists():
    _filtered_df = pd.read_csv(_filtered_path)
    # Reconstruct the domain dict preserving original (col, label, test) tuples
    _filtered_vars = set(_filtered_df["variable"].tolist())
    CANDIDATES = {
        domain: [(col, label, test)
                 for col, label, test in var_list
                 if col in _filtered_vars]
        for domain, var_list in FVI_CANDIDATES.items()
    }
    # Drop empty domains
    CANDIDATES = {d: v for d, v in CANDIDATES.items() if v}
    print(f"Loaded filtered FVI candidate list from stage06a "
          f"({sum(len(v) for v in CANDIDATES.values())} variables).")
else:
    CANDIDATES = FVI_CANDIDATES
    print("fvi_domains_filtered.csv not found — using full FVI_CANDIDATES from config.")

# ── Load data ─────────────────────────────────────────────────────────────────
scored = pd.read_csv(SCORED_CSV, encoding='utf-8-sig', low_memory=False)
survey = pd.read_csv(SURVEY_CSV, encoding='utf-8-sig', low_memory=False)

# Force FVI outcome to numeric
scored["FVI_norm_1_5"] = pd.to_numeric(scored["FVI_norm_1_5"], errors="coerce")
outcome = scored["FVI_norm_1_5"]

print(f"Outcome variable: FVI_norm_1_5  "
      f"(n valid = {outcome.notna().sum():,})\n")

# ── Data quality: cap known outlier columns before testing ────────────────────
# AB_Physical_capacity_Pos and OP_Training_Neg contain impossible values
# (entry errors — values of 100000000000). Cap at 100 to preserve the 0-100 scale.
for col in CAP_COLS:
    if col in survey.columns:
        s = pd.to_numeric(survey[col], errors="coerce")
        n_outliers = (s > 100).sum()
        if n_outliers > 0:
            print(f"  Data quality note: {col} — {n_outliers} values > 100 "
                  f"capped to NaN before testing.")
            s = s.where(s <= 100)          # values > 100 become NaN
            survey[col] = s

# ── Bivariate screening ───────────────────────────────────────────────────────
results = []

for domain, predictors in CANDIDATES.items():
    domain_label = domain.split("_", 1)[1].replace("_", " ")  # e.g. "Motivation"

    for col, label, test in predictors:

        # Pull predictor from the correct dataframe
        src = scored if col in scored.columns else survey
        if col not in src.columns:
            print(f"  WARNING: '{col}' not found — skipped.")
            continue

        x = pd.to_numeric(src[col], errors="coerce")

        # Align on valid pairs (pairwise complete cases)
        pair = pd.DataFrame({"x": x.values, "y": outcome.values}).dropna()
        n_pair = len(pair)

        if n_pair < 30:
            # Too few observations to test reliably
            print(f"  SKIPPED (n={n_pair} < 30): {label}")
            continue

        x_arr = pair["x"].values
        y_arr = pair["y"].values

        # Run the appropriate test
        if test == "pearson":
            r, p = stats.pearsonr(x_arr, y_arr)
            stat_name = "r"
        else:
            r, p = stats.spearmanr(x_arr, y_arr)
            stat_name = "rho"

        # Effect size label (Cohen 1988)
        abs_r = abs(r)
        effect = "small" if abs_r >= 0.10 else "negligible"
        if abs_r >= 0.30:
            effect = "medium"
        if abs_r >= 0.50:
            effect = "large"

        results.append({
            "Domain":       domain_label,
            "Column":       col,
            "Label":        label,
            "Test":         test.capitalize(),
            "Statistic":    stat_name,
            "r_or_rho":     round(r, 4),
            "p_value":      round(p, 4),
            "n":            n_pair,
            "Effect_size":  effect,
            "Significant":  "Yes" if p < P_THRESHOLD else "No",
        })

# ── Assemble results table ────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["Domain", "p_value"])

sig_df = results_df[results_df["Significant"] == "Yes"].copy()

# ── Save outputs ──────────────────────────────────────────────────────────────
results_df.to_csv(OUT_ALL, index=False)
sig_df.to_csv(OUT_SIG,     index=False)
print(f"Saved: {OUT_ALL}  ({len(results_df)} predictors tested)")
print(f"Saved: {OUT_SIG}   ({len(sig_df)} significant at p < {P_THRESHOLD})")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"BIVARIATE SCREENING RESULTS  (threshold: p < {P_THRESHOLD})")
print("=" * 70)

for domain in results_df["Domain"].unique():
    sub = results_df[results_df["Domain"] == domain]
    sig = sub[sub["Significant"] == "Yes"]
    print(f"\n  {domain.upper()}  "
          f"({len(sig)}/{len(sub)} significant)")
    print(f"  {'Label':<42} {'Stat':>5}  {'r/rho':>7}  {'p':>7}  {'n':>5}  Effect")
    print(f"  {'-'*42} {'-'*5}  {'-'*7}  {'-'*7}  {'-'*5}  ------")
    for _, row in sub.iterrows():
        sig_flag = "*" if row["Significant"] == "Yes" else " "
        print(f"  {sig_flag}{row['Label']:<41} "
              f"{row['Statistic']:>5}  "
              f"{row['r_or_rho']:>7.4f}  "
              f"{row['p_value']:>7.4f}  "
              f"{row['n']:>5}  "
              f"{row['Effect_size']}")

print(f"\n{'='*70}")
print(f"Total tested       : {len(results_df)}")
print(f"Significant (p<{P_THRESHOLD}): {len(sig_df)}")
print(f"{'='*70}")
print("\nVariables advancing to domain regression:")
for _, row in sig_df.iterrows():
    print(f"  [{row['Domain'][:3].upper()}] {row['Column']:<45}  "
          f"r={row['r_or_rho']:>7.4f}  p={row['p_value']:.4f}")
