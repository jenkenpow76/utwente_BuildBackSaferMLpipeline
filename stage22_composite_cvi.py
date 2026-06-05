"""
stage17_composite_cvi.py
===============
Composite Vulnerability Index (CVI): FVI + EVI

Purpose
-------
Combines the Flood Vulnerability Index (FVI) and Earthquake Vulnerability
Index (EVI) into a single Composite Vulnerability Index (CVI) using equal
weights (0.5 FVI + 0.5 EVI). Both component scores must be on the same
normalised 1–5 scale before aggregation.

Equal weighting rationale
--------------------------
In the absence of empirical evidence justifying differential weights, equal
weighting is the methodologically sound default. This approach is consistent
with established multi-hazard vulnerability literature (Cutter et al., 2003;
Aksha et al., 2019; Bhochhibhoya & Maharjan, 2022). It ensures transparency
and reproducibility — a core FAIR research principle.

    CVI = 0.5 * FVI_norm_1_5 + 0.5 * EVI_norm_1_5

The CVI is then:
  - Re-normalised to 0–1  (CVI_norm_0_1)
  - Retained on 1–5 scale (CVI_norm_1_5)
  - Classified into Low / Medium / High using equal-interval bands

Input scenarios (auto-detected)
---------------------------------
This script handles three possible states of the EVI score:

  Scenario A — EVI column already exists in imputed_FVI_scores.csv
               (i.e. both scores are in one file, joined on a common key)

  Scenario B — EVI score exists in a separate CSV file
               (joined to FVI scores on a household ID column)

  Scenario C — EVI score does not yet exist
               (script exits with a clear, actionable message)

Configuration
-------------
Edit COMPOSITE_CONFIG below before running. At minimum set:
  FVI_CSV      : path to imputed_FVI_scores.csv (output of stage01_fvi_scoring.py)
  EVI_CSV      : path to EVI scored CSV, OR set to None if EVI is already
                 a column in FVI_CSV
  JOIN_KEY     : household/unit ID column used to merge FVI and EVI files
                 (only needed for Scenario B)
  OUTPUT_DIR   : directory where all composite outputs are written

Outputs
-------
  CVI_scores.csv                — full dataset with CVI columns appended
  cvi_distribution.png          — histogram of CVI_norm_1_5
  cvi_component_scatter.png     — FVI vs EVI scatter coloured by CVI class
  cvi_correlation_matrix.png    — FVI / EVI / CVI correlation heatmap
  cvi_class_by_community.png    — CVI class frequencies per VDC/ward
  cvi_descriptives.csv          — summary statistics for CVI, FVI, EVI

FAIR Research Principles
------------------------
  Findable      : All outputs use consistent 'cvi_' prefix
  Accessible    : Outputs saved to OUTPUT_DIR; paths logged to console
  Interoperable : CSV and PNG formats; column names documented here
  Reusable      : Script is self-contained, documented, and version-pinned

References
----------
  Cutter, S.L., Boruff, B.J., & Shirley, W.L. (2003). Social vulnerability
    to environmental hazards. Social Science Quarterly, 84(2), 242–261.
    https://doi.org/10.1111/1540-6237.8402002

  Aksha, S.K., Juran, L., Resler, L.M., & Zhang, Y. (2019). An analysis of
    social vulnerability to natural hazards in Nepal using a modified Social
    Vulnerability Index. International Journal of Disaster Risk Science,
    10(1), 103–116. https://doi.org/10.1007/s13753-018-0192-7

  Bhochhibhoya, S. & Maharjan, R. (2022). Integrated seismic risk assessment
    in Nepal. Natural Hazards and Earth System Sciences, 22, 3211–3230.
    https://doi.org/10.5194/nhess-22-3211-2022

  Villagran De León, J.C. (2004). Vulnerability: A Conceptual and
    Methodological Review. UNU-EHS Source No. 4. United Nations University.

  European Commission (2018). Step 6: Weighting — 10-Step Guide to
    Composite Indicators. Knowledge for Policy.
    https://knowledge4policy.ec.europa.eu/composite-indicators/
    10-step-guide/step-6-weighting_en
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── Import analytical constants from pipeline_config ─────────────────────────
import importlib.util as _ilu, pathlib as _pl
_cfg_path = _pl.Path(__file__).resolve().parent / "pipeline_config.py"
_cfg_spec  = _ilu.spec_from_file_location("pipeline_config", str(_cfg_path))
_cfg_mod   = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)
_PIPELINE_CFG = {k: getattr(_cfg_mod, k) for k in [
    "SURVEYOR_COL","COMMUNITY_COL","FVI_SCORE_COL","EVI_SCORE_COL",
    "BNDRY_LM","BNDRY_MH","STEPWISE_PIN","STEPWISE_POUT",
    "MORANS_THRESHOLD_KM","FVI_WEIGHT","EVI_WEIGHT","RANDOM_STATE",
]}


# =============================================================================
# COMPOSITE CONFIGURATION — edit these values before running
# =============================================================================

COMPOSITE_CONFIG = {
    # ── Input files ──────────────────────────────────────────────────────────

    # Path to imputed_FVI_scores.csv (output of stage01_fvi_scoring.py).
    # When run via run_fvi_pipeline.py Stage 7, this resolves to
    # OUTPUT_DIR/imputed_FVI_scores.csv automatically (working dir is set).
    # For standalone use, update this path to match your directory layout.
    # Input CSVs are resolved from the current working directory,
    # which the master runner (run_pipeline.py) sets to OUTPUT_DIR.
    "FVI_CSV": str(Path.cwd() / "imputed_FVI_scores.csv"),

    # Path to imputed_EVI_scores.csv (output of stage02_evi_scoring.py).
    # Set to the EVI pipeline output directory. When both pipelines write
    # to the same output folder this resolves automatically. If they write
    # to separate folders, provide the full path.
    "EVI_CSV": str(Path.cwd() / "imputed_EVI_scores.csv"),

    # ── Column names ─────────────────────────────────────────────────────────

    # Column name for the FVI score (1–5 scale) in FVI_CSV.
    "FVI_COL": "FVI_norm_1_5",

    # Column name for the EVI score (1–5 scale) in EVI_CSV.
    "EVI_COL": "EVI_norm_1_5",

    # Household / unit ID column used to join FVI and EVI files.
    # Both imputed_FVI_scores.csv and imputed_EVI_scores.csv are derived from
    # the same raw survey file, so they share the same row order. The join
    # uses the survey respondent ID column. Update if your dataset uses a
    # different ID column name.
    "JOIN_KEY": "ID",

    # Optional: community grouping column for the community-level bar chart.
    # GD005_VDC_name is the VDC/ward column used in Moran's I and mixed-effects
    # stages. Set to None to skip the community plot.
    "COMMUNITY_COL": "GD005_VDC_name",

    # ── Output ───────────────────────────────────────────────────────────────

    # Directory where all composite outputs are written.
    # When run as Stage 7 of run_fvi_pipeline.py, the working directory is
    # already set to OUTPUT_DIR — so "." resolves correctly. For standalone
    # use, set this to "fvi_outputs" or your preferred output directory.
    "OUTPUT_DIR": str(Path.cwd()),
}

# =============================================================================
# CONSTANTS
# =============================================================================

# Equal weighting: FVI and EVI contribute equally to the CVI.
# Justified by the absence of empirical evidence for differential weights
# (Cutter et al., 2003; Aksha et al., 2019; Bhochhibhoya & Maharjan, 2022).
FVI_WEIGHT = _PIPELINE_CFG["FVI_WEIGHT"]
EVI_WEIGHT = _PIPELINE_CFG["EVI_WEIGHT"]

# Classification thresholds on the 1–5 scale (three equal intervals):
#   Low    : 1.000 – 2.333
#   Medium : 2.333 – 3.667
#   High   : 3.667 – 5.000
BNDRY_LM = _PIPELINE_CFG["BNDRY_LM"]
BNDRY_MH = _PIPELINE_CFG["BNDRY_MH"]

# Colour palette (consistent with FVI pipeline)
C_LOW    = "#4DB6AC"   # teal       — Low vulnerability
C_MED    = "#FFB74D"   # amber      — Medium vulnerability
C_HIGH   = "#E57373"   # coral-red  — High vulnerability
C_FVI    = "#1565C0"   # dark blue  — FVI points
C_EVI    = "#6A1B9A"   # purple     — EVI points
C_MEAN   = "#E53935"   # red        — mean line
C_MEDIAN = "#7B1FA2"   # purple     — median line

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# =============================================================================
# STEP 1 — LOAD DATA AND DETECT SCENARIO
# =============================================================================

def load_data(cfg: dict) -> pd.DataFrame:
    """
    Load FVI and EVI scores and return a merged DataFrame.

    Auto-detects which scenario applies:
      A — EVI column already present in FVI_CSV
      B — EVI in a separate CSV; joined to FVI on JOIN_KEY
      C — EVI does not exist; exits with instructions

    Parameters
    ----------
    cfg : dict — COMPOSITE_CONFIG

    Returns
    -------
    pd.DataFrame with at least FVI_COL and EVI_COL as numeric columns.
    """
    fvi_path = Path(cfg["FVI_CSV"])
    evi_path = cfg["EVI_CSV"]
    fvi_col  = cfg["FVI_COL"]
    evi_col  = cfg["EVI_COL"]
    join_key = cfg["JOIN_KEY"]

    # Load FVI dataset
    if not fvi_path.exists():
        log.error(f"FVI_CSV not found: {fvi_path}")
        log.error("Run the FVI pipeline first: python run_fvi_pipeline.py")
        sys.exit(1)

    df = pd.read_csv(fvi_path, encoding='utf-8-sig', low_memory=False)
    df[fvi_col] = pd.to_numeric(df[fvi_col], errors="coerce")
    log.info(f"Loaded FVI data: {len(df):,} rows from '{fvi_path}'")
    log.info(f"  FVI column '{fvi_col}': "
             f"{df[fvi_col].notna().sum():,} valid values")

    # ── Scenario A: EVI column already in FVI file ────────────────────────────
    if evi_path is None and evi_col in df.columns:
        log.info(f"Scenario A: EVI column '{evi_col}' found in FVI file.")
        df[evi_col] = pd.to_numeric(df[evi_col], errors="coerce")
        return df

    # ── Scenario B: EVI in a separate CSV ────────────────────────────────────
    if evi_path is not None:
        evi_file = Path(evi_path)
        if not evi_file.exists():
            log.error(f"EVI_CSV not found: {evi_file}")
            sys.exit(1)

        evi_df = pd.read_csv(evi_file, encoding='utf-8-sig', low_memory=False)
        evi_df[evi_col] = pd.to_numeric(evi_df[evi_col], errors="coerce")
        log.info(f"Scenario B: Loaded EVI data: {len(evi_df):,} rows "
                 f"from '{evi_file}'")

        # Validate join key
        if join_key not in df.columns:
            log.error(f"JOIN_KEY '{join_key}' not found in FVI file. "
                      "Update COMPOSITE_CONFIG['JOIN_KEY'].")
            sys.exit(1)
        if join_key not in evi_df.columns:
            log.error(f"JOIN_KEY '{join_key}' not found in EVI file. "
                      "Update COMPOSITE_CONFIG['JOIN_KEY'].")
            sys.exit(1)

        # Left join: keep all FVI rows; attach EVI where matched
        n_before = len(df)
        df = df.merge(evi_df[[join_key, evi_col]],
                      on=join_key, how="left")
        n_matched = df[evi_col].notna().sum()
        log.info(f"  Joined on '{join_key}': "
                 f"{n_matched:,}/{n_before:,} rows matched.")
        return df

    # ── Scenario C: EVI does not exist ───────────────────────────────────────
    log.error("=" * 65)
    log.error("EVI score not found.")
    log.error("=" * 65)
    log.error(
        "The EVI (Earthquake Vulnerability Index) has not been computed yet.\n"
        "\n"
        "To proceed, choose one of the following:\n"
        "\n"
        "  Option 1 — Build the EVI pipeline (recommended):\n"
        "    Create EVI scoring notebooks mirroring the FVI pipeline.\n"
        "    The output must include a column named 'EVI_norm_1_5'\n"
        "    on the same 1–5 scale as FVI_norm_1_5.\n"
        "\n"
        "  Option 2 — Add EVI as a column to imputed_FVI_scores.csv:\n"
        "    If you have EVI scores in another format, add them as\n"
        "    'EVI_norm_1_5' to imputed_FVI_scores.csv, then re-run.\n"
        "\n"
        "  Option 3 — Provide a separate EVI CSV:\n"
        "    Set COMPOSITE_CONFIG['EVI_CSV'] to the path of the EVI file\n"
        "    and set COMPOSITE_CONFIG['JOIN_KEY'] to the shared ID column.\n"
        "\n"
        "Once EVI scores are available, re-run: python stage17_composite_cvi.py"
    )
    sys.exit(1)


# =============================================================================
# STEP 2 — COMPUTE CVI
# =============================================================================

def compute_cvi(df: pd.DataFrame, fvi_col: str, evi_col: str) -> pd.DataFrame:
    """
    Compute the Composite Vulnerability Index from FVI and EVI scores.

    Formula
    -------
    Both scores must be on a 1–5 scale (as produced by stage01_fvi_scoring.py
    and the equivalent EVI pipeline).

        CVI_norm_1_5 = 0.5 * FVI_norm_1_5 + 0.5 * EVI_norm_1_5

    CVI_norm_1_5 is then:
      - Re-normalised to 0–1: CVI_norm_0_1 = (CVI - 1) / 4
      - Classified into Low (1), Medium (2), High (3) using three
        equal-interval bands of the 1–5 range.

    Cases where either FVI or EVI is missing receive NaN for CVI.
    This is conservative: a household cannot be classified if either
    component index is unavailable.

    Parameters
    ----------
    df      : DataFrame containing fvi_col and evi_col
    fvi_col : column name for FVI (1–5 scale)
    evi_col : column name for EVI (1–5 scale)

    Returns
    -------
    df with four new columns:
      CVI_norm_1_5  — composite score (1–5)
      CVI_norm_0_1  — normalised composite score (0–1)
      CVI_CLASS     — 1=Low, 2=Medium, 3=High
      CVI_CLASS_LBL — 'Low', 'Medium', 'High'
    """
    fvi = pd.to_numeric(df[fvi_col], errors="coerce")
    evi = pd.to_numeric(df[evi_col], errors="coerce")

    # Verify scale: both inputs should be in [1, 5]
    for name, series in [(fvi_col, fvi), (evi_col, evi)]:
        valid = series.dropna()
        if len(valid) > 0:
            out_of_range = ((valid < 1.0) | (valid > 5.0)).sum()
            if out_of_range > 0:
                log.warning(
                    f"  {name}: {out_of_range} values outside [1, 5]. "
                    "Verify scale normalisation before aggregation."
                )

    # Equal-weight arithmetic mean — valid only when both scores present
    valid_pair = fvi.notna() & evi.notna()
    cvi_15 = pd.Series(np.nan, index=df.index, name="CVI_norm_1_5")
    cvi_15[valid_pair] = FVI_WEIGHT * fvi[valid_pair] + EVI_WEIGHT * evi[valid_pair]

    # Re-normalise to 0–1: linear mapping 1→0, 5→1
    cvi_01 = (cvi_15 - 1.0) / 4.0

    # Classify using equal-interval bands
    def _classify(v):
        """Return 1=Low, 2=Medium, 3=High, or NaN if missing."""
        if pd.isna(v):
            return np.nan
        return 1 if v <= BNDRY_LM else (2 if v <= BNDRY_MH else 3)

    cvi_class = cvi_15.apply(_classify)
    cvi_lbl   = cvi_class.map({1: "Low", 2: "Medium", 3: "High"})

    df["CVI_norm_1_5"]  = cvi_15
    df["CVI_norm_0_1"]  = cvi_01
    df["CVI_CLASS"]     = cvi_class
    df["CVI_CLASS_LBL"] = cvi_lbl

    # Console summary
    n_valid = valid_pair.sum()
    n_fvi   = fvi.notna().sum()
    n_evi   = evi.notna().sum()
    log.info(f"\n  FVI valid    : {n_fvi:,}")
    log.info(f"  EVI valid    : {n_evi:,}")
    log.info(f"  CVI computed : {n_valid:,} (both scores present)")
    log.info(f"  CVI missing  : {(~valid_pair).sum():,} "
             "(at least one component missing)\n")

    return df


# =============================================================================
# STEP 3 — DESCRIPTIVE STATISTICS
# =============================================================================

def descriptives(df: pd.DataFrame,
                 fvi_col: str,
                 evi_col: str,
                 output_dir: Path) -> None:
    """
    Print and save descriptive statistics for FVI, EVI, and CVI.

    Outputs
    -------
    cvi_descriptives.csv — mean, SD, min, max, quartiles for all three scores
    """
    rows = []
    for label, col in [("FVI", fvi_col), ("EVI", evi_col),
                        ("CVI", "CVI_norm_1_5")]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "Index":  label,
            "n":      len(s),
            "Mean":   round(s.mean(),   4),
            "Median": round(s.median(), 4),
            "SD":     round(s.std(),    4),
            "Min":    round(s.min(),    4),
            "Max":    round(s.max(),    4),
            "P25":    round(s.quantile(0.25), 4),
            "P75":    round(s.quantile(0.75), 4),
        })

    desc_df = pd.DataFrame(rows)
    out_path = output_dir / "cvi_descriptives.csv"
    desc_df.to_csv(out_path, index=False)

    log.info("=" * 58)
    log.info("DESCRIPTIVE STATISTICS  (1–5 scale)")
    log.info("=" * 58)
    log.info(f"\n{desc_df.to_string(index=False)}\n")

    log.info("CVI CLASS FREQUENCIES")
    log.info("=" * 58)
    class_counts = df["CVI_CLASS"].value_counts().sort_index()
    n_total = class_counts.sum()
    for k, cnt in class_counts.items():
        lbl = {1: "Low", 2: "Medium", 3: "High"}.get(int(k), "?")
        log.info(f"  {int(k)}  {lbl:<8}: {cnt:>5,}  ({cnt/n_total*100:.1f}%)")

    log.info(f"\nSaved: {out_path}")


# =============================================================================
# STEP 4 — FIGURE 1: CVI DISTRIBUTION HISTOGRAM
# =============================================================================

def plot_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot a histogram of CVI_norm_1_5 with zone backgrounds, mean/median lines,
    and a stats box. Mirrors the style of stage06_fvi_descriptives.py Figure 1.

    Output
    ------
    cvi_distribution.png
    """
    cvi = df["CVI_norm_1_5"].dropna()
    n   = len(cvi)

    mean_v   = cvi.mean()
    med_v    = cvi.median()
    sd_v     = cvi.std()
    min_v    = cvi.min()
    max_v    = cvi.max()
    p25_v    = cvi.quantile(0.25)
    p75_v    = cvi.quantile(0.75)

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Coloured zone backgrounds
    ax.axvspan(1.0,       BNDRY_LM, alpha=0.18, color=C_LOW,  zorder=0)
    ax.axvspan(BNDRY_LM, BNDRY_MH, alpha=0.18, color=C_MED,  zorder=0)
    ax.axvspan(BNDRY_MH,  5.0,     alpha=0.18, color=C_HIGH, zorder=0)

    # Histogram
    bins = np.arange(1.0, 5.26, 0.25)
    ax.hist(cvi, bins=bins, color="#37474F", edgecolor="white",
            linewidth=0.7, zorder=2)

    # Mean / median lines
    ax.axvline(mean_v, color=C_MEAN,   linestyle="--", linewidth=2.0,
               zorder=4, label=f"Mean = {mean_v:.2f}")
    ax.axvline(med_v,  color=C_MEDIAN, linestyle="-.", linewidth=2.0,
               zorder=4, label=f"Median = {med_v:.2f}")

    # Zone boundary lines
    ax.axvline(BNDRY_LM, color=C_LOW,  linestyle=":", linewidth=1.5,
               alpha=0.85, zorder=1)
    ax.axvline(BNDRY_MH, color=C_HIGH, linestyle=":", linewidth=1.5,
               alpha=0.85, zorder=1)

    # Zone labels
    fig.canvas.draw()
    _, yhi = ax.get_ylim()
    for lbl, xc, col in [
        ("Low",    (1.0       + BNDRY_LM) / 2, C_LOW),
        ("Medium", (BNDRY_LM + BNDRY_MH) / 2, "#8D6E00"),
        ("High",   (BNDRY_MH + 5.0)      / 2, C_HIGH),
    ]:
        ax.text(xc, yhi * 0.975, lbl, ha="center", va="top",
                fontstyle="italic", fontsize=12.5, color=col, zorder=5)

    # Stats box
    stats_txt = (
        f"Mean:    {mean_v:>5.2f}\n"
        f"Median: {med_v:>5.2f}\n"
        f"SD:        {sd_v:>5.2f}\n"
        f"Min:       {min_v:>5.2f}\n"
        f"Max:     {max_v:>5.2f}\n"
        f"P25:      {p25_v:>5.2f}\n"
        f"P75:      {p75_v:>5.2f}"
    )
    ax.text(0.978, 0.975, stats_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=10.0, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#BBBBBB", linewidth=0.8, alpha=0.93),
            zorder=6)

    # Legend
    handles = [
        mpatches.Patch(facecolor=C_LOW,  alpha=0.45, edgecolor="none",
                       label="Low zone  (1–2.33)"),
        mpatches.Patch(facecolor=C_MED,  alpha=0.45, edgecolor="none",
                       label="Medium zone (2.33–3.67)"),
        mpatches.Patch(facecolor=C_HIGH, alpha=0.45, edgecolor="none",
                       label="High zone  (3.67–5)"),
        Line2D([0], [0], color=C_MEAN,   linestyle="--", linewidth=2.0,
               label=f"Mean = {mean_v:.2f}"),
        Line2D([0], [0], color=C_MEDIAN, linestyle="-.", linewidth=2.0,
               label=f"Median = {med_v:.2f}"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9.5,
              framealpha=0.93, edgecolor="#BBBBBB", frameon=True)

    ax.set_xlim(1.0, 5.0)
    ax.set_xlabel("Composite Vulnerability Index — CVI (1–5 scale)",
                  fontsize=13, labelpad=8)
    ax.set_ylabel("Number of houses", fontsize=13, labelpad=8)
    ax.set_title(
        f"Distribution of Composite Vulnerability Index (FVI + EVI)\n"
        f"Equal weights — n = {n:,}",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout(pad=1.5)
    out = output_dir / "cvi_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {out}")


# =============================================================================
# STEP 5 — FIGURE 2: FVI vs EVI SCATTER COLOURED BY CVI CLASS
# =============================================================================

def plot_component_scatter(df: pd.DataFrame,
                           fvi_col: str,
                           evi_col: str,
                           output_dir: Path) -> None:
    """
    Scatter plot of FVI_norm_1_5 (x) vs EVI_norm_1_5 (y), with points
    coloured by CVI class. Includes the equal-weight iso-CVI line for
    each class boundary and a 45-degree reference line.

    This plot lets you see which component dominates in each household
    and where the two indices diverge — an important diagnostic for
    multi-hazard research.

    Output
    ------
    cvi_component_scatter.png
    """
    valid = df[[fvi_col, evi_col, "CVI_CLASS"]].dropna()
    fvi_v = valid[fvi_col].values
    evi_v = valid[evi_col].values
    cls_v = valid["CVI_CLASS"].values

    colour_map = {1.0: C_LOW, 2.0: C_MED, 3.0: C_HIGH}
    colours    = [colour_map.get(c, "#AAAAAA") for c in cls_v]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(fvi_v, evi_v, c=colours, alpha=0.35, s=14, zorder=3)

    # 45-degree reference line: FVI == EVI (perfect agreement)
    ax.plot([1, 5], [1, 5], color="#888888", linewidth=1.2,
            linestyle="--", zorder=2, label="FVI = EVI")

    # Iso-CVI boundary lines:
    # CVI = 0.5*FVI + 0.5*EVI → EVI = 2*boundary - FVI
    x_range = np.linspace(1.0, 5.0, 100)
    for bndry, lbl, col in [
        (BNDRY_LM, f"CVI = {BNDRY_LM:.2f} (Low/Med boundary)", C_MED),
        (BNDRY_MH, f"CVI = {BNDRY_MH:.2f} (Med/High boundary)", C_HIGH),
    ]:
        y_iso = 2.0 * bndry - x_range
        # Clip to [1, 5] range
        mask = (y_iso >= 1.0) & (y_iso <= 5.0) & (x_range >= 1.0) & (x_range <= 5.0)
        ax.plot(x_range[mask], y_iso[mask], color=col, linewidth=1.5,
                linestyle=":", zorder=4, label=lbl)

    # Legend
    handles = [
        mpatches.Patch(color=C_LOW,  label="Low CVI"),
        mpatches.Patch(color=C_MED,  label="Medium CVI"),
        mpatches.Patch(color=C_HIGH, label="High CVI"),
        Line2D([0], [0], color="#888888", linestyle="--",
               label="FVI = EVI"),
        Line2D([0], [0], color=C_MED,  linestyle=":",
               label=f"CVI = {BNDRY_LM:.2f} boundary"),
        Line2D([0], [0], color=C_HIGH, linestyle=":",
               label=f"CVI = {BNDRY_MH:.2f} boundary"),
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.93,
              edgecolor="#BBBBBB", loc="upper left")

    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(1.0, 5.0)
    ax.set_xlabel("FVI_norm_1_5 (Flood Vulnerability)", fontsize=12, labelpad=8)
    ax.set_ylabel("EVI_norm_1_5 (Earthquake Vulnerability)", fontsize=12, labelpad=8)
    ax.set_title(
        "Component Scores: FVI vs EVI\nColoured by CVI Class",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_aspect("equal")

    plt.tight_layout(pad=1.5)
    out = output_dir / "cvi_component_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {out}")


# =============================================================================
# STEP 6 — FIGURE 3: CORRELATION MATRIX
# =============================================================================

def plot_correlation_matrix(df: pd.DataFrame,
                             fvi_col: str,
                             evi_col: str,
                             output_dir: Path) -> None:
    """
    Pearson correlation heatmap for FVI, EVI, and CVI.

    This is a transparency diagnostic. Because CVI is an equal-weight
    average of FVI and EVI, the CVI–FVI and CVI–EVI correlations are
    mathematically constrained — but the FVI–EVI correlation tells you
    how independently the two hazards track across households.

    Output
    ------
    cvi_correlation_matrix.png
    """
    sub = df[[fvi_col, evi_col, "CVI_norm_1_5"]].dropna()
    sub.columns = ["FVI", "EVI", "CVI"]
    corr = sub.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    # Heatmap manually with imshow
    im = ax.imshow(corr.values, vmin=-1, vmax=1,
                   cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    labels = ["FVI", "EVI", "CVI"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Annotate cells with r values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=12,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")

    ax.set_title("Pearson Correlation: FVI, EVI, CVI",
                 fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    out = output_dir / "cvi_correlation_matrix.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {out}")


# =============================================================================
# STEP 7 — FIGURE 4: CVI CLASS BY COMMUNITY (optional)
# =============================================================================

def plot_community_bars(df: pd.DataFrame,
                        community_col: str,
                        output_dir: Path) -> None:
    """
    Stacked horizontal bar chart showing CVI class frequencies per
    VDC/ward community. Mirrors the style of stage06_fvi_descriptives.py
    Figure 2.

    Only rendered when COMPOSITE_CONFIG['COMMUNITY_COL'] is set and
    the column exists in the data.

    Output
    ------
    cvi_class_by_community.png
    """
    if community_col is None or community_col not in df.columns:
        log.info("COMMUNITY_COL not set or not found — skipping community plot.")
        return

    sub = df[[community_col, "CVI_CLASS"]].dropna()
    communities = sorted(sub[community_col].unique())
    n_comm = len(communities)

    rows = []
    for comm in communities:
        s   = sub[sub[community_col] == comm]["CVI_CLASS"]
        n_c = len(s)
        rows.append({
            "community": str(comm),
            "n":     n_c,
            "low":   (s == 1).sum() / n_c * 100 if n_c else 0.0,
            "med":   (s == 2).sum() / n_c * 100 if n_c else 0.0,
            "high":  (s == 3).sum() / n_c * 100 if n_c else 0.0,
        })

    labels   = [r["community"] for r in rows]
    ns       = [r["n"]         for r in rows]
    pct_low  = np.array([r["low"]  for r in rows])
    pct_med  = np.array([r["med"]  for r in rows])
    pct_high = np.array([r["high"] for r in rows])

    fig_h = max(6, n_comm * 0.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(n_comm)
    h = 0.55

    b_low  = ax.barh(y, pct_low,  height=h, color=C_LOW,  label="Low",    zorder=2)
    b_med  = ax.barh(y, pct_med,  left=pct_low, height=h,
                     color=C_MED,  label="Medium", zorder=2)
    b_high = ax.barh(y, pct_high, left=pct_low + pct_med, height=h,
                     color=C_HIGH, label="High",   zorder=2)

    # Percentage labels inside bars
    def _label_bars(bars, lefts, vals):
        for bar, lft, val in zip(bars, lefts, vals):
            if val >= 8.0:
                ax.text(lft + val / 2.0,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=9.0, fontweight="bold",
                        color="white", zorder=4)

    _label_bars(b_low,  np.zeros(n_comm), pct_low)
    _label_bars(b_med,  pct_low,          pct_med)
    _label_bars(b_high, pct_low + pct_med, pct_high)

    # n= labels
    for i, n_val in enumerate(ns):
        ax.text(101.8, i, f"n={n_val:,}", va="center", ha="left",
                fontsize=9.0, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Percentage of houses (%)", fontsize=11, labelpad=8)
    ax.set_xlim(0, 116)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax.set_title(
        "CVI Class Frequencies by Community (VDC/Ward)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.legend(title="CVI class", loc="lower right",
              fontsize=9.5, framealpha=0.93, edgecolor="#BBBBBB")

    plt.tight_layout(pad=1.5)
    out = output_dir / "cvi_class_by_community.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {out}")


# =============================================================================
# STEP 8 — SAVE FULL OUTPUT CSV
# =============================================================================

def save_output(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save the full dataset with CVI columns appended to CVI_scores.csv.

    New columns added
    -----------------
    CVI_norm_1_5  — composite score (1–5)
    CVI_norm_0_1  — normalised composite score (0–1)
    CVI_CLASS     — 1=Low, 2=Medium, 3=High
    CVI_CLASS_LBL — 'Low', 'Medium', 'High'
    """
    out = output_dir / "CVI_scores.csv"
    df.to_csv(out, index=False)
    log.info(f"\nFull dataset saved: {out}")
    log.info(f"  Rows    : {len(df):,}")
    log.info(f"  CVI col : CVI_norm_1_5  (ready for downstream modelling)")


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# RANDOM INTERCEPTS — CVI (composite of FVI and EVI community effects)
# =============================================================================

def compute_cvi_random_intercepts(output_dir: Path) -> None:
    """
    Derive CVI community random intercepts by combining the FVI and EVI
    random intercept CSVs produced by Stage 6 and Stage 12.

    Method
    ------
    Because the CVI is a straight equal-weight average of FVI and EVI:

        CVI_ij = 0.5 * FVI_ij + 0.5 * EVI_ij

    the community random intercept for CVI is the equal-weight average of
    the FVI and EVI random intercepts for that community:

        u_j(CVI) = 0.5 * u_j(FVI) + 0.5 * u_j(EVI)

    The combined SE is propagated assuming independence between models:

        se_u(CVI) = 0.5 * sqrt(se_u(FVI)^2 + se_u(EVI)^2)

    Communities present in only one index are retained with NaN for the
    missing index. Communities present in both are combined.

    Outputs
    -------
    cvi_random_intercepts.csv  — per-community CVI random intercepts
    cvi_random_intercepts.png  — caterpillar plot

    Reference
    ---------
    Raudenbush & Bryk (2002). Hierarchical Linear Models (2nd ed.). SAGE.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fvi_path = output_dir / "fvi_random_intercepts.csv"
    evi_path = output_dir / "evi_random_intercepts.csv"

    # Both files must exist — they are written by Stages 6 and 12
    if not fvi_path.exists():
        log.warning(f"FVI random intercepts not found: {fvi_path}  — skipping CVI RI")
        return
    if not evi_path.exists():
        log.warning(f"EVI random intercepts not found: {evi_path}  — skipping CVI RI")
        return

    fvi_ri = pd.read_csv(fvi_path)
    evi_ri = pd.read_csv(evi_path)

    # Rename columns before merge to avoid conflicts
    fvi_ri = fvi_ri.rename(columns={
        "u_j":      "fvi_u_j",
        "se_u":     "fvi_se_u",
        "ci_lo_95": "fvi_ci_lo",
        "ci_hi_95": "fvi_ci_hi",
        "sig":      "fvi_sig",
        "n":        "fvi_n",
        "direction":"fvi_direction",
    })
    evi_ri = evi_ri.rename(columns={
        "u_j":      "evi_u_j",
        "se_u":     "evi_se_u",
        "ci_lo_95": "evi_ci_lo",
        "ci_hi_95": "evi_ci_hi",
        "sig":      "evi_sig",
        "n":        "evi_n",
        "direction":"evi_direction",
    })

    # Outer join on community name — keeps all communities from both indices
    merged = pd.merge(fvi_ri[["community", "fvi_n", "fvi_u_j", "fvi_se_u"]],
                      evi_ri[["community", "evi_n", "evi_u_j", "evi_se_u"]],
                      on="community", how="outer")

    # Equal-weight combination of random intercepts
    merged["u_j"]   = 0.5 * merged["fvi_u_j"] + 0.5 * merged["evi_u_j"]
    merged["se_u"]  = 0.5 * np.sqrt(merged["fvi_se_u"]**2 + merged["evi_se_u"]**2)

    # Use FVI n as the community size (both come from same sample)
    merged["n"] = merged["fvi_n"].fillna(merged["evi_n"]).astype(int)

    # 95% CI  (approximate — uses z=1.96 for simplicity across models)
    merged["ci_lo_95"] = merged["u_j"] - 1.96 * merged["se_u"]
    merged["ci_hi_95"] = merged["u_j"] + 1.96 * merged["se_u"]
    merged["sig"]      = (merged["ci_lo_95"] > 0) | (merged["ci_hi_95"] < 0)
    merged["direction"] = merged["u_j"].apply(
        lambda v: "above average" if v > 0 else "below average"
    )

    # Sort low to high for caterpillar plot (consistent with FVI/EVI scripts)
    merged = merged.sort_values("u_j").reset_index(drop=True)

    # Save CSV
    out_cols = ["community", "n", "fvi_u_j", "evi_u_j", "u_j", "se_u",
                "ci_lo_95", "ci_hi_95", "sig", "direction"]
    out_csv = output_dir / "cvi_random_intercepts.csv"
    merged[out_cols].round(6).to_csv(str(out_csv), index=False)
    log.info(f"Saved: cvi_random_intercepts.csv  ({len(merged)} communities)")

    # Print table to console
    print("\n" + "=" * 65)
    print("CVI RANDOM INTERCEPTS (u_j = 0.5*FVI_u + 0.5*EVI_u)")
    print("=" * 65)
    print(f"{'Community':<32} {'n':>5} {'FVI_u':>8} {'EVI_u':>8} "
          f"{'CVI_u':>8} {'SE':>7} {'Sig':>4} {'Direction'}")
    print("-" * 85)
    for _, row in merged.iterrows():
        sig_str = "*" if row["sig"] else ""
        print(f"  {str(row['community']):<30} {int(row['n']):>5} "
              f"{row['fvi_u_j']:>8.4f} {row['evi_u_j']:>8.4f} "
              f"{row['u_j']:>8.4f} {row['se_u']:>7.4f} {sig_str:>4} "
              f"  {row['direction']}")

    # Caterpillar plot — mirrors FVI/EVI style
    J = len(merged)
    labels = [
        str(c).replace("_", " ").replace("ward", "Ward").title()
        + f" (n={int(n)})"
        for c, n in zip(merged["community"], merged["n"])
    ]

    fig, ax = plt.subplots(figsize=(9, max(6, J * 0.45)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, row in merged.reset_index(drop=True).iterrows():
        uj      = row["u_j"]
        ci_lo_v = row["ci_lo_95"]
        ci_hi_v = row["ci_hi_95"]
        sig_pt  = row["sig"]
        color   = "#D94F3D" if uj > 0 else "#2196A6"
        alpha   = 1.0 if sig_pt else 0.40

        ax.errorbar(
            x=uj, y=i,
            xerr=[[uj - ci_lo_v], [ci_hi_v - uj]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.5, capsize=4, markersize=6,
            alpha=alpha, zorder=3,
        )

    ax.axvline(0, color="#888888", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(range(J))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(
        "CVI Random Intercept (u_j = 0.5*FVI_u + 0.5*EVI_u)  +/-95% CI\n"
        "Red = above average vulnerability  |  Blue = below average  |  "
        "Faded = CI crosses zero",
        fontsize=9,
    )
    ax.set_title(
        "Caterpillar Plot: Community-level CVI Random Intercepts\n"
        "(Equal-weight composite of FVI and EVI community effects)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_png = output_dir / "cvi_random_intercepts.png"
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: cvi_random_intercepts.png")


def sensitivity_variance_normalisation(
        df: pd.DataFrame,
        fvi_col: str,
        evi_col: str,
        community_col: str,
        output_dir: Path,
) -> None:
    """
    Sensitivity analysis: variance-normalised CVI vs equal-weight CVI.

    Rationale
    ---------
    Although FVI and EVI are assigned equal numerical weights (0.5 each),
    the component with greater variance exerts proportionally more influence
    on the spread of the composite. In this dataset FVI SD = 0.50 and
    EVI SD = 0.61, so the EVI contributes slightly more variance to the CVI.
    This analysis tests whether SD-normalising before averaging materially
    changes community-level vulnerability rankings.

    Method
    ------
    1. Divide each component by its sample SD (zero-mean not required;
       we only need equal variance contributions).
    2. Average the SD-normalised components with equal weights.
    3. Rescale the result back to the 1–5 range for interpretability
       (linear min-max rescaling using the observed range).
    4. Rank communities under both formulations and compute Spearman rho.

    If rho > 0.95 and max rank change <= 2, the equal-weight CVI is
    considered robust to the variance difference (Nardo et al., 2008).

    Outputs
    -------
    cvi_sensitivity_variance_normalisation.csv — community-level comparison
        Columns: community, n, CVI_equal_weight, CVI_var_normalised,
                 rank_equal, rank_normalised, rank_change

    Reference
    ---------
    Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffman, A., &
      Giovannini, E. (2008). Handbook on Constructing Composite Indicators:
      Methodology and User Guide. OECD Publishing.
      https://doi.org/10.1787/9789264043466-en
    """
    from scipy.stats import spearmanr

    fvi = pd.to_numeric(df[fvi_col], errors="coerce")
    evi = pd.to_numeric(df[evi_col], errors="coerce")

    # Keep only rows where both components are present (mirrors compute_cvi)
    valid = fvi.notna() & evi.notna()

    fvi_sd = fvi[valid].std()
    evi_sd = evi[valid].std()

    # SD-normalised components (equal-variance scaling)
    fvi_scaled = fvi / fvi_sd
    evi_scaled = evi / evi_sd

    # Equal-weight average of SD-normalised components
    cvi_raw = 0.5 * fvi_scaled + 0.5 * evi_scaled

    # Rescale back to 1–5 using the observed min/max of valid rows
    raw_min = cvi_raw[valid].min()
    raw_max = cvi_raw[valid].max()
    cvi_normalised = 1.0 + ((cvi_raw - raw_min) / (raw_max - raw_min)) * 4.0

    df = df.copy()
    df["CVI_var_normalised"] = np.where(valid, cvi_normalised, np.nan)

    # Community-level comparison (requires community column)
    if community_col is None or community_col not in df.columns:
        log.info(
            "Sensitivity analysis: COMMUNITY_COL not set — "
            "Spearman rank comparison skipped. "
            "Household-level correlation only."
        )
        # Still report household-level correlation as a minimum check
        r_hw = df["CVI_norm_1_5"].corr(df["CVI_var_normalised"])
        log.info(
            f"  Household-level Pearson r (equal-weight vs "
            f"var-normalised CVI): {r_hw:.4f}"
        )
        return

    community_comparison = (
        df[[community_col, "CVI_norm_1_5", "CVI_var_normalised"]]
        .dropna()
        .groupby(community_col)
        .agg(
            CVI_equal_weight=("CVI_norm_1_5",     "mean"),
            CVI_var_normalised=("CVI_var_normalised", "mean"),
            n=("CVI_norm_1_5", "count"),
        )
        .reset_index()
        .rename(columns={community_col: "community"})
    )

    # Rank communities: rank 1 = most vulnerable (highest CVI)
    community_comparison["rank_equal"] = (
        community_comparison["CVI_equal_weight"]
        .rank(ascending=False)
        .astype(int)
    )
    community_comparison["rank_normalised"] = (
        community_comparison["CVI_var_normalised"]
        .rank(ascending=False)
        .astype(int)
    )
    community_comparison["rank_change"] = (
        community_comparison["rank_equal"]
        - community_comparison["rank_normalised"]
    ).abs()

    # Spearman rank correlation between the two formulations
    rho, p_val = spearmanr(
        community_comparison["rank_equal"],
        community_comparison["rank_normalised"],
    )

    max_change  = community_comparison["rank_change"].max()
    mean_change = community_comparison["rank_change"].mean()

    # Console report
    log.info("")
    log.info("=" * 65)
    log.info("SENSITIVITY ANALYSIS: Variance-normalised CVI")
    log.info("=" * 65)
    log.info(f"  FVI SD (sample)  : {fvi_sd:.4f}")
    log.info(f"  EVI SD (sample)  : {evi_sd:.4f}")
    log.info(f"  SD ratio EVI/FVI : {evi_sd / fvi_sd:.4f}")
    log.info("")
    log.info(f"  Spearman rho (community rankings)  : {rho:.4f}")
    log.info(f"  p-value                            : {p_val:.4f}")
    log.info(f"  Max rank change (positions)        : {max_change:.0f}")
    log.info(f"  Mean rank change (positions)       : {mean_change:.2f}")
    log.info("")

    # Check robustness for top-N and bottom-N communities specifically
    top_n = 5  # adjust as needed
    top_communities = community_comparison.nsmallest(top_n, "rank_equal")
    top_max_change = top_communities["rank_change"].max()

    robust = rho >= 0.95 and max_change <= 2
    if robust:
        log.info(
            "  VERDICT: Rankings are ROBUST to variance normalisation "
            "(rho >= 0.95 and max rank change <= 2 positions). "
            "Equal-weight CVI is the preferred formulation."
        )
    else:
        log.info(
            "  VERDICT: Rankings show SENSITIVITY to variance normalisation. "
            "Consider reporting both formulations or adopting SD-normalised "
            "weights as the primary CVI."
        )
    log.info("=" * 65)

    # Save CSV
    out_csv = output_dir / "cvi_sensitivity_variance_normalisation.csv"
    community_comparison.round(4).to_csv(str(out_csv), index=False)
    log.info(f"Saved: {out_csv.name}")


def run() -> None:
    """
    Execute the full CVI computation pipeline.

    Steps
    -----
    1.  Load FVI and EVI data (auto-detect scenario)
    2.  Compute CVI = 0.5*FVI + 0.5*EVI
    3.  Print and save descriptive statistics
    4.  Plot CVI distribution histogram
    5.  Plot FVI vs EVI component scatter
    6.  Plot Pearson correlation matrix
    7.  Plot CVI class by community (if community column configured)
    8.  Save full output CSV
    9.  Compute and save CVI random intercepts
    10. Sensitivity analysis: variance-normalised CVI
    """
    cfg        = COMPOSITE_CONFIG
    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("COMPOSITE VULNERABILITY INDEX (CVI)")
    log.info("=" * 65)
    log.info(f"FVI weight : {FVI_WEIGHT}")
    log.info(f"EVI weight : {EVI_WEIGHT}")
    log.info(f"Output dir : {output_dir}")
    log.info("=" * 65)

    # Step 1: load
    df = load_data(cfg)

    # Step 2: compute
    df = compute_cvi(df, cfg["FVI_COL"], cfg["EVI_COL"])

    # Step 3: descriptives
    descriptives(df, cfg["FVI_COL"], cfg["EVI_COL"], output_dir)

    # Steps 4–7: figures
    plot_distribution(df, output_dir)
    plot_component_scatter(df, cfg["FVI_COL"], cfg["EVI_COL"], output_dir)
    plot_correlation_matrix(df, cfg["FVI_COL"], cfg["EVI_COL"], output_dir)
    plot_community_bars(df, cfg["COMMUNITY_COL"], output_dir)

    # Step 8: save CSV
    save_output(df, output_dir)

    # Step 9: CVI random intercepts (requires FVI + EVI RI CSVs)
    compute_cvi_random_intercepts(output_dir)

    # Step 10: sensitivity analysis — variance-normalised CVI
    sensitivity_variance_normalisation(
        df,
        cfg["FVI_COL"],
        cfg["EVI_COL"],
        cfg["COMMUNITY_COL"],
        output_dir,
    )

    log.info("\n" + "=" * 65)
    log.info("CVI COMPUTATION COMPLETE")
    log.info(f"Final CVI column : CVI_norm_1_5  in  CVI_scores.csv")
    log.info("=" * 65)


# Run when loaded by run_pipeline.py via importlib (module name is
# the stage label, not "__main__"), and also when run directly.
run()

# (The if __name__ guard below is retained for documentation;
#  the unconditional call above handles both execution paths.)
if __name__ == "__main__":
    pass  # already called above