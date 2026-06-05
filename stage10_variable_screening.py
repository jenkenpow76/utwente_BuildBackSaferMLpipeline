"""
Stage 06a | PRE | Explanatory Variable Frequency & Missingness Screening
=========================================================================

Purpose
-------
Assess the frequency of valid observations for every candidate explanatory
variable in both the FVI and EVI candidate pools, then remove any variable
whose proportion of missing values exceeds the 50 % threshold recommended
by van Buuren (2018, §9.1.4).

Van Buuren (2018, §9.1.4) justification
-----------------------------------------
Variables with high proportions of missing data "generally create more
problems than they solve" in multiple imputation contexts. Van Buuren
recommends removing variables that require more than 50 % imputation unless
they are of direct scientific interest and their missingness mechanism is
well understood. Because the variables in the candidate pools are explanatory
predictors rather than outcomes, and because imputing more than half of their
values would introduce substantial model uncertainty, any variable exceeding
this threshold is excluded from the analysis before bivariate screening.

Reference:
    van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).
    CRC Press. https://stefvanbuuren.name/fimd/

Inputs
------
    analysis_dataset.csv        — produced by Stage 05b; contains all
                                  imputed MAO composite scores and the
                                  merged survey data used for regression.
    pipeline_config.py          — defines FVI_CANDIDATES, EVI_CANDIDATES,
                                  FVI_DOMAINS, EVI_DOMAINS.

Outputs (written to cwd = outputs/)
-------------------------------------
    variable_screening_report.csv  — one row per candidate variable:
                                     index, domain, n_valid, n_missing,
                                     pct_missing, retained (True/False),
                                     removal_reason.
    variable_screening_summary.txt — human-readable console-style report
                                     listing retained and removed variables.

Pipeline position
-----------------
    Runs AFTER Stage 05b (analysis_dataset.csv must exist).
    Runs BEFORE Stage 07 (FVI bivariate screening).
    Removes variables from FVI_DOMAINS / EVI_DOMAINS IN MEMORY so that
    Stages 07, 08, 12, 13 receive a clean candidate pool.

    Because pipeline stages are independent scripts executed sequentially,
    this stage writes the filtered domain lists to two CSVs:
        fvi_domains_filtered.csv
        evi_domains_filtered.csv
    Stages 07 and 12 (bivariate screening) check for these files at startup
    and use them if present; otherwise they fall back to the config lists.

Note on threshold
-----------------
    The 50 % threshold is conservative by design. If a variable falls just
    below 50 % (e.g. 48 %) but was already excluded from imputation for
    theoretical reasons, it will still appear in this report but will not
    be removed here — it will already be absent from the analysis dataset.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# ── Import analytical constants from pipeline_config ─────────────────────────
import importlib.util as _ilu
_cfg_path = Path(__file__).resolve().parent / "pipeline_config.py"
_cfg_spec  = _ilu.spec_from_file_location("pipeline_config", str(_cfg_path))
_cfg_mod   = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)

# Pull everything we need from the config module
FVI_CANDIDATES = _cfg_mod.FVI_CANDIDATES   # {domain: [(col, label, test), ...]}
EVI_CANDIDATES = _cfg_mod.EVI_CANDIDATES
# Note: pipeline_config v47 no longer exposes FVI_DOMAINS / EVI_DOMAINS as
# module-level constants — they are derived at runtime from the Stage 12/17
# bivariate output via __getattr__. Stage 10 runs BEFORE Stage 12, so we
# work with FVI_CANDIDATES / EVI_CANDIDATES here.

# =============================================================================
# CONSTANTS
# =============================================================================

# Van Buuren (2018, §9.1.4) recommends removing variables that require more
# than 50 % imputation. We use this as our exclusion threshold.
MISSING_THRESHOLD = 0.50   # proportion — variables above this are removed

# Input / output paths (stage runs in OUTPUT_DIR = outputs/)
ANALYSIS_DATASET   = Path.cwd() / "analysis_dataset.csv"
OUT_REPORT_CSV     = Path.cwd() / "variable_screening_report.csv"
OUT_SUMMARY_TXT    = Path.cwd() / "variable_screening_summary.txt"
OUT_FVI_FILTERED   = Path.cwd() / "fvi_domains_filtered.csv"
OUT_EVI_FILTERED   = Path.cwd() / "evi_domains_filtered.csv"
# Candidate-level filtered lists — consumed by Stage 18 (Random Forest).
# Written from FVI_CANDIDATES/EVI_CANDIDATES (the full pre-bivariate pool)
# rather than FVI_DOMAINS/EVI_DOMAINS (the smaller post-bivariate pool).
# This ensures the random forest evaluates all candidate variables that
# passed the >50% missing threshold, not just those that survived stepwise.
OUT_FVI_CAND_FILTERED = Path.cwd() / "fvi_candidates_filtered.csv"
OUT_EVI_CAND_FILTERED = Path.cwd() / "evi_candidates_filtered.csv"

# =============================================================================
# 1. LOAD ANALYSIS DATASET
# =============================================================================

print("=" * 70)
print("STAGE 06a — EXPLANATORY VARIABLE FREQUENCY & MISSINGNESS SCREENING")
print("=" * 70)
print()

if not ANALYSIS_DATASET.exists():
    sys.exit(
        f"ERROR: {ANALYSIS_DATASET} not found.\n"
        "Stage 05b must complete successfully before Stage 06a."
    )

df = pd.read_csv(ANALYSIS_DATASET, encoding="utf-8-sig", low_memory=False)

# Replace blank-string placeholders (survey export artefact) with NaN
df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)

n_total = len(df)
print(f"Loaded analysis_dataset.csv: {n_total:,} rows × {df.shape[1]:,} columns")
print()

# =============================================================================
# 2. HELPER — compute missingness statistics for a list of columns
# =============================================================================

def missingness_stats(cols, df, n_total):
    """
    For each column in `cols`, compute:
        n_valid, n_missing, pct_missing.

    Columns absent from the dataframe (already excluded upstream) are
    reported with 100 % missing.

    Parameters
    ----------
    cols    : list of str — column names to evaluate
    df      : pd.DataFrame — the analysis dataset
    n_total : int — total number of rows

    Returns
    -------
    pd.DataFrame with columns [variable, n_valid, n_missing, pct_missing]
    """
    rows = []
    for col in cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            n_valid  = int(series.notna().sum())
            n_miss   = n_total - n_valid
            pct_miss = n_miss / n_total
        else:
            # Column absent from dataset — treat as 100 % missing
            n_valid  = 0
            n_miss   = n_total
            pct_miss = 1.0
        rows.append({
            "variable":    col,
            "n_valid":     n_valid,
            "n_missing":   n_miss,
            "pct_missing": pct_miss,
        })
    return pd.DataFrame(rows)

# =============================================================================
# 3. EVALUATE CANDIDATE POOLS
# =============================================================================

def evaluate_candidates(candidates_dict, index_name):
    """
    Assess missingness for all candidate variables in one index's pool.

    Parameters
    ----------
    candidates_dict : dict — {domain_name: [(col, label, test), ...]}
    index_name      : str  — "FVI" or "EVI" (for display only)

    Returns
    -------
    pd.DataFrame — one row per variable with domain, stats, and retention flag.
    list         — variable names that should be REMOVED (pct_missing > threshold)
    """
    records = []
    for domain, var_list in candidates_dict.items():
        cols = [col for col, _label, _test in var_list]
        stats = missingness_stats(cols, df, n_total)
        for _, row in stats.iterrows():
            pct = row["pct_missing"]
            # Van Buuren (2018, §9.1.4): exclude variables requiring > 50 % imputation
            retained = pct <= MISSING_THRESHOLD
            removal_reason = (
                ""
                if retained
                else (
                    f"Excluded: {pct * 100:.1f}% missing "
                    f"(threshold {MISSING_THRESHOLD * 100:.0f}%). "
                    "Van Buuren (2018, §9.1.4): variables requiring more than "
                    "50% imputation 'generally create more problems than they solve'."
                )
            )
            records.append({
                "index":          index_name,
                "domain":         domain,
                "variable":       row["variable"],
                "n_valid":        row["n_valid"],
                "n_missing":      row["n_missing"],
                "pct_missing":    round(pct * 100, 1),
                "retained":       retained,
                "removal_reason": removal_reason,
            })

    result_df = pd.DataFrame(records)
    removed   = result_df.loc[~result_df["retained"], "variable"].tolist()
    return result_df, removed


print("Evaluating FVI candidate pool...")
fvi_report, fvi_removed = evaluate_candidates(FVI_CANDIDATES, "FVI")

print("Evaluating EVI candidate pool...")
evi_report, evi_removed = evaluate_candidates(EVI_CANDIDATES, "EVI")

all_report = pd.concat([fvi_report, evi_report], ignore_index=True)

# =============================================================================
# 4. FILTER CANDIDATE POOLS
# =============================================================================
# Remove excluded variables (those exceeding the 50% missing threshold) from
# the candidate pools that are passed to bivariate screening (Stages 12/17).
# In v47 architecture, FVI_DOMAINS / EVI_DOMAINS are derived from the
# Stage 12/17 bivariate output and do not exist at Stage 10 time. The
# pre-bivariate candidate pools (FVI_CANDIDATES, EVI_CANDIDATES) are the
# correct objects to filter here.

def candidates_to_domain_dict(candidates_dict):
    """
    Convert a CANDIDATES dict ({domain: [(col, label, test), ...]})
    into a plain {domain: [col, ...]} mapping for downstream use.
    Strips the leading "N_" prefix from domain names (e.g., "1_Motivation"
    -> "Motivation") for consistency with the post-bivariate naming.
    """
    out = {}
    for domain_key, predictor_list in candidates_dict.items():
        domain_name = domain_key.split("_", 1)[1] if "_" in domain_key else domain_key
        out[domain_name] = [col for col, _, _ in predictor_list]
    return out

def filter_domain_dict(domain_dict, removed_vars):
    """Remove excluded variables from a domain dictionary in place."""
    filtered = {}
    for domain, cols in domain_dict.items():
        kept = [c for c in cols if c not in removed_vars]
        if kept:
            filtered[domain] = kept
    return filtered

# Start from the full CANDIDATE pools, then drop the excluded variables.
fvi_domains_filtered = filter_domain_dict(
    candidates_to_domain_dict(FVI_CANDIDATES), fvi_removed)
evi_domains_filtered = filter_domain_dict(
    candidates_to_domain_dict(EVI_CANDIDATES), evi_removed)

# Write filtered domain lists so Stages 07/08 and 12/13 can read them
def write_domain_csv(domain_dict, path):
    rows = [{"domain": d, "variable": v}
            for d, cols in domain_dict.items() for v in cols]
    pd.DataFrame(rows).to_csv(path, index=False)

write_domain_csv(fvi_domains_filtered, OUT_FVI_FILTERED)
write_domain_csv(evi_domains_filtered, OUT_EVI_FILTERED)

# Write candidate-level filtered lists for Stage 18 (Random Forest).
# These preserve ALL candidates that passed the >50% missing threshold,
# not just those that survived bivariate screening and stepwise selection.
# The random forest should receive the full pre-bivariate candidate pool
# so that importance rankings are not pre-filtered by the linear pipeline.
def write_candidate_filtered_csv(candidates_dict, removed_vars, path):
    """
    Write a candidate-level filtered variable list to CSV.

    Parameters
    ----------
    candidates_dict : dict — {domain: [(col, label, test), ...]}
    removed_vars    : list of str — variable names to exclude (>50% missing)
    path            : Path — output CSV path
    """
    rows = [
        {"domain": domain, "variable": col}
        for domain, var_list in candidates_dict.items()
        for col, _label, _test in var_list
        if col not in removed_vars
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

write_candidate_filtered_csv(FVI_CANDIDATES, fvi_removed, OUT_FVI_CAND_FILTERED)
write_candidate_filtered_csv(EVI_CANDIDATES, evi_removed, OUT_EVI_CAND_FILTERED)

# =============================================================================
# 5. CONSOLE REPORT
# =============================================================================

def print_index_report(report_df, removed_vars, index_name):
    """Print a formatted missingness table for one index."""
    n_total_vars  = len(report_df)
    n_retained    = report_df["retained"].sum()
    n_removed_cnt = (~report_df["retained"]).sum()

    print()
    print("=" * 70)
    print(f"{index_name} — CANDIDATE VARIABLE MISSINGNESS REPORT")
    print(f"  Total candidates : {n_total_vars}")
    print(f"  Retained         : {n_retained}")
    print(f"  Removed (>50%)   : {n_removed_cnt}")
    print(f"  Threshold        : {MISSING_THRESHOLD * 100:.0f}%  "
          f"(van Buuren 2018, §9.1.4)")
    print("=" * 70)

    # Print by domain
    for domain in report_df["domain"].unique():
        sub = report_df[report_df["domain"] == domain]
        print(f"\n  {domain} ({len(sub)} variables)")
        print(f"  {'Variable':<52} {'n_valid':>8} {'n_miss':>8} {'%_miss':>8}  {'Status':>10}")
        print("  " + "-" * 92)
        for _, row in sub.iterrows():
            status = "RETAINED" if row["retained"] else "*** REMOVED ***"
            print(f"  {row['variable']:<52} "
                  f"{row['n_valid']:>8,} "
                  f"{row['n_missing']:>8,} "
                  f"{row['pct_missing']:>7.1f}%  "
                  f"{status:>15}")

    # Explicit list of removed variables with rationale
    if removed_vars:
        print()
        print(f"  VARIABLES REMOVED FROM {index_name} ANALYSIS")
        print("  " + "-" * 70)
        print(f"  Criterion: > {MISSING_THRESHOLD * 100:.0f}% missing values")
        print(f"  Basis    : van Buuren (2018, §9.1.4) — variables requiring")
        print(f"             more than 50% imputation 'generally create more")
        print(f"             problems than they solve'.")
        print()
        removed_sub = report_df[~report_df["retained"]]
        for _, row in removed_sub.iterrows():
            print(f"    - {row['variable']}")
            print(f"      n_valid={row['n_valid']:,}  "
                  f"n_missing={row['n_missing']:,}  "
                  f"missing={row['pct_missing']:.1f}%")
    else:
        print()
        print(f"  No {index_name} variables exceed the 50% missing threshold.")
        print("  All candidates are retained.")


print_index_report(fvi_report, fvi_removed, "FVI")
print_index_report(evi_report, evi_removed, "EVI")

# Summary footer
total_removed = len(set(fvi_removed + evi_removed))
print()
print("=" * 70)
print("SCREENING SUMMARY")
print("=" * 70)
print(f"  N total survey respondents : {n_total:,}")
print(f"  FVI candidates screened    : {len(fvi_report)}")
print(f"  EVI candidates screened    : {len(evi_report)}")
print(f"  FVI variables removed      : {len(fvi_removed)}")
print(f"  EVI variables removed      : {len(evi_removed)}")
print(f"  Unique variables removed   : {total_removed}")
print()
print("  Reference:")
print("  van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.).")
print("  CRC Press, §9.1.4. https://stefvanbuuren.name/fimd/")
print()
print(f"  Filtered domain lists written:")
print(f"    fvi_domains_filtered.csv  ({sum(len(v) for v in fvi_domains_filtered.values())} variables, for Stages 07/08)")
print(f"    evi_domains_filtered.csv  ({sum(len(v) for v in evi_domains_filtered.values())} variables, for Stages 12/13)")
n_fvi_cand = sum(1 for d in FVI_CANDIDATES.values() for c,_,_ in d if c not in fvi_removed)
n_evi_cand = sum(1 for d in EVI_CANDIDATES.values() for c,_,_ in d if c not in evi_removed)
print(f"    fvi_candidates_filtered.csv ({n_fvi_cand} variables, for Stage 18)")
print(f"    evi_candidates_filtered.csv ({n_evi_cand} variables, for Stage 18)")
print(f"  Full report    : variable_screening_report.csv")
print(f"  Summary report : variable_screening_summary.txt")

# =============================================================================
# 6. SAVE OUTPUTS
# =============================================================================

# Full CSV report
all_report.to_csv(OUT_REPORT_CSV, index=False)
print()
print(f"Saved: {OUT_REPORT_CSV.name}  ({len(all_report)} rows)")

# Plain-text summary (for thesis appendix / audit trail)
summary_lines = []
summary_lines.append("STAGE 06a — EXPLANATORY VARIABLE MISSINGNESS SCREENING")
summary_lines.append("=" * 70)
summary_lines.append(f"N total respondents: {n_total:,}")
summary_lines.append(f"Threshold: >{MISSING_THRESHOLD * 100:.0f}% missing → REMOVED")
summary_lines.append(
    "Reference: van Buuren, S. (2018). Flexible Imputation of Missing Data, "
    "§9.1.4. https://stefvanbuuren.name/fimd/"
)
summary_lines.append("")

for index_name, report_df, removed_vars in [
    ("FVI", fvi_report, fvi_removed),
    ("EVI", evi_report, evi_removed),
]:
    summary_lines.append(f"{index_name} CANDIDATES")
    summary_lines.append("-" * 50)
    summary_lines.append(
        f"{'Variable':<52} {'%_miss':>8}  {'Status'}"
    )
    for _, row in report_df.iterrows():
        status = "retained" if row["retained"] else "REMOVED (van Buuren 2018, §9.1.4)"
        summary_lines.append(
            f"{row['variable']:<52} {row['pct_missing']:>7.1f}%  {status}"
        )
    summary_lines.append("")
    if removed_vars:
        summary_lines.append(f"Removed from {index_name} ({len(removed_vars)} variables):")
        for v in removed_vars:
            row = report_df[report_df["variable"] == v].iloc[0]
            summary_lines.append(
                f"  {v}  [{row['pct_missing']:.1f}% missing]"
            )
    else:
        summary_lines.append(f"No {index_name} variables removed.")
    summary_lines.append("")

OUT_SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")
print(f"Saved: {OUT_SUMMARY_TXT.name}")

print()
print("Stage 06a complete.")
print(
    "  Stages 07 and 12 will read fvi_domains_filtered.csv / "
    "evi_domains_filtered.csv and use the filtered candidate lists."
)
