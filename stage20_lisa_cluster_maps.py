"""
stage15_lisa_cluster_maps.py
====================
Stage 5b | LISA Cluster Maps — FVI and EVI (with OpenStreetMap basemap)

Purpose
-------
Produces publication-quality Local Moran's I (LISA) cluster maps for both
the Flood Vulnerability Index (FVI) and Earthquake Vulnerability Index (EVI),
with an OpenStreetMap basemap rendered behind the cluster markers.

This script reads the per-household LISA results already computed in Stages 5
and 11 (stage09_fvi_morans.py and stage14_evi_morans.py) and renders:

    Figure 1 — FVI LISA cluster map (with OSM basemap)
    Figure 2 — FVI LISA cluster frequency bar chart
    Figure 3 — EVI LISA cluster map (with OSM basemap)
    Figure 4 — EVI LISA cluster frequency bar chart
    Figure 5 — Side-by-side comparison map (FVI vs EVI, with OSM basemap)
    lisa_summary.csv — cluster counts for both indices (thesis appendix)

Inputs (must exist in OUTPUT_DIR / working directory)
------------------------------------------------------
    fvi_local_morans_results.csv  — produced by stage09_fvi_morans.py (Stage 5)
    evi_local_morans_results.csv  — produced by stage14_evi_morans.py (Stage 11)

Pipeline placement
------------------
Register as Stage 11b in run_pipeline.py, immediately after Stage 11.

    (\"Stage 11b | LISA | Cluster Maps (FVI + EVI)\",
     HERE / \"stage15_lisa_cluster_maps.py\",    OUTPUT_DIR),

OSM basemap
-----------
Tiles are fetched via osm_tiles.py (must be in the pipeline root directory).
Tiles are cached in OUTPUT_DIR/osm_tile_cache/ — not re-downloaded on
subsequent runs. If the network is unavailable, basemap is skipped gracefully.

References
----------
Anselin, L. (1995). Local indicators of spatial association — LISA.
    Geographical Analysis, 27(2), 93-115.
    https://doi.org/10.1111/j.1538-4632.1995.tb00338.x

Aksha et al. (2019). Int J Disaster Risk Sci, 10, 103-116.
    https://doi.org/10.1007/s13753-018-0192-7
"""

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

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


warnings.filterwarnings("ignore")

# =============================================================================
# OSM BASEMAP IMPORT
# =============================================================================
# osm_tiles.py must sit in the same directory as this script (pipeline root).
# If not found, maps are produced without a basemap — pipeline does not crash.

try:
    from osm_tiles import add_basemap as _add_basemap_impl

    def add_basemap(ax, **kwargs) -> bool:
        """Wrapper around osm_tiles.add_basemap."""
        return _add_basemap_impl(ax, **kwargs)

    _BASEMAP_AVAILABLE = True

except ImportError:
    def add_basemap(ax, **kwargs) -> bool:  # noqa: F811
        """No-op fallback when osm_tiles.py is not found."""
        print(
            "  [lisa_cluster_maps] WARNING: osm_tiles.py not found.\n"
            "  Copy osm_tiles.py to the pipeline root directory.\n"
            "  Continuing without basemap."
        )
        return False

    _BASEMAP_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = Path.cwd()   # set by run_pipeline.py to OUTPUT_DIR

FVI_LISA_CSV = OUT_DIR / "fvi_local_morans_results.csv"
EVI_LISA_CSV = OUT_DIR / "evi_local_morans_results.csv"

# LISA parameters — must match stage09_fvi_morans.py / stage14_evi_morans.py
LISA_THRESHOLD_KM = _PIPELINE_CFG["MORANS_THRESHOLD_KM"]
N_PERMUTATIONS    = 999
LISA_ALPHA        = 0.05

# ── Basemap settings ──────────────────────────────────────────────────────────
# FVI maps use CartoDB Voyager: renders rivers in pale blue, which is directly
# relevant to flood hazard context. Roads and terrain are visible but subtle.
#
# EVI maps use CartoDB Positron: minimal pale grey, no rivers or terrain.
# Rivers are irrelevant to seismic vulnerability so the cleaner style is better.
#
# Available providers (defined in osm_tiles.py):
#   "carto_voyager"     — rivers + terrain, light roads  ← FVI default
#   "carto_positron"    — minimal grey, no rivers        ← EVI default
#   "osm"               — full OSM (colourful, busy)
#   "stamen_watercolor" — bold blue rivers, artistic
#   "stamen_toner"      — greyscale, high-contrast
#   "carto_darkmatter"  — dark background
#
# Set both to the same value if you want a consistent style throughout.
BASEMAP_PROVIDER_FVI = "carto_voyager"   # rivers visible — relevant for flood hazard
BASEMAP_PROVIDER_EVI = "carto_positron"  # minimal — rivers not relevant to seismic risk

BASEMAP_ALPHA = 0.55   # 0 = invisible, 1 = fully opaque
BASEMAP_ZOOM  = "auto" # or an integer (8–14)

# ── Cluster visual encoding (Anselin 1995 convention) ─────────────────────────
CLUSTER_COLOURS = {
    "High-High":       "#D7191C",   # red    — hotspot
    "Low-Low":         "#2C7BB6",   # blue   — coldspot
    "High-Low":        "#F4A58A",   # salmon — spatial outlier
    "Low-High":        "#74ADD1",   # mid-blue — spatial outlier
    "Not significant": "#D4D4D4",   # light grey
}

# Non-significant plotted first (background), significant on top
CLUSTER_PLOT_ORDER = [
    "Not significant", "Low-High", "High-Low", "Low-Low", "High-High"
]

CLUSTER_SIZES = {
    "High-High": 35, "Low-Low": 35,
    "High-Low":  25, "Low-High": 25,
    "Not significant": 9,
}
CLUSTER_ALPHA_VALS = {
    "High-High": 0.85, "Low-Low": 0.85,
    "High-Low":  0.75, "Low-High": 0.75,
    "Not significant": 0.40,
}
CLUSTER_EDGE = {
    "High-High": "#7A0B0B", "Low-Low": "#0E3660",
    "High-Low":  "#C0614A", "Low-High": "#3A7FA0",
    "Not significant": "none",
}
CLUSTER_EDGE_WIDTH = {
    "High-High": 0.5, "Low-Low": 0.5,
    "High-Low":  0.4, "Low-High": 0.4,
    "Not significant": 0.0,
}
CLUSTER_LABELS = {
    "High-High":       "High-High  (hotspot)",
    "Low-Low":         "Low-Low  (coldspot)",
    "High-Low":        "High-Low  (outlier)",
    "Low-High":        "Low-High  (outlier)",
    "Not significant": "Not significant",
}


# =============================================================================
# HELPERS
# =============================================================================

def load_lisa_csv(csv_path: Path, index_name: str):
    """
    Load a LISA results CSV. Returns DataFrame or None on failure.

    Validates required columns and drops GPS-less rows. Returns None if the
    file is missing or malformed so callers can skip figures gracefully.

    Parameters
    ----------
    csv_path   : Path — location of the CSV file
    index_name : str  — 'FVI' or 'EVI', used only in warning messages
    """
    if not csv_path.exists():
        print(
            f"  WARNING: {csv_path.name} not found. "
            f"{index_name} LISA figures skipped.\n"
            f"  Run Stage 5 (stage09_fvi_morans.py) or Stage 11 "
            f"(stage14_evi_morans.py) first."
        )
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    required = {"lat", "lon", "community", "cluster_type"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(
            f"  ERROR: {csv_path.name} missing columns: {missing_cols}. "
            f"{index_name} LISA figures skipped."
        )
        return None

    n_before = len(df)
    df = df.dropna(subset=["lat", "lon"])
    dropped = n_before - len(df)
    if dropped:
        print(f"  Note: {dropped} rows without GPS dropped from {index_name} LISA data.")

    df["cluster_type"] = df["cluster_type"].fillna("Not significant").astype(str)
    return df


def _legend_handles(df):
    """Build mpatches.Patch legend handles for cluster types present in df."""
    present = set(df["cluster_type"].unique())
    handles = []
    for ct in CLUSTER_PLOT_ORDER:
        if ct not in present:
            continue
        n   = (df["cluster_type"] == ct).sum()
        pct = n / len(df) * 100
        handles.append(
            mpatches.Patch(
                facecolor=CLUSTER_COLOURS[ct],
                edgecolor=CLUSTER_EDGE[ct] if ct != "Not significant" else "#999999",
                linewidth=0.8,
                label=f"{CLUSTER_LABELS[ct]}  (n={n:,}, {pct:.1f}%)",
            )
        )
    return handles


def _scatter_clusters(ax, df):
    """
    Plot LISA cluster points on ax.

    Non-significant points are drawn first (zorder=3) so significant clusters
    (zorder=4) appear on top and are never occluded.

    Parameters
    ----------
    ax : matplotlib Axes — lon/lat limits must already be set
    df : DataFrame with 'lon', 'lat', 'cluster_type' columns
    """
    for ct in CLUSTER_PLOT_ORDER:
        mask = df["cluster_type"] == ct
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "lon"],
            df.loc[mask, "lat"],
            c=CLUSTER_COLOURS[ct],
            s=CLUSTER_SIZES[ct],
            alpha=CLUSTER_ALPHA_VALS[ct],
            linewidths=CLUSTER_EDGE_WIDTH[ct],
            edgecolors=CLUSTER_EDGE[ct],
            zorder=4 if ct != "Not significant" else 3,
        )


def _add_community_labels(ax, df):
    """
    Annotate community centroids with VDC/ward name labels.

    Communities with n < 5 are suppressed to reduce clutter. Labels have
    a semi-transparent white background for readability over the OSM tiles.

    Parameters
    ----------
    ax : matplotlib Axes
    df : DataFrame with 'lat', 'lon', 'community' columns
    """
    MIN_N = 5
    for comm, grp in df.groupby("community"):
        if pd.isna(comm) or str(comm).strip() == "" or len(grp) < MIN_N:
            continue
        ax.text(
            grp["lon"].mean(), grp["lat"].mean(),
            str(comm).replace("_", " ").title(),
            fontsize=6.5, ha="center", va="bottom",
            color="#111111", alpha=0.9, zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor="white",
                edgecolor="#cccccc", linewidth=0.4, alpha=0.75,
            ),
        )


def _style_map(ax, title: str):
    """Apply consistent style to a map axes."""
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, zorder=1)


def _set_extent(ax, df, margin=0.04):
    """
    Set axes lon/lat limits from data with a fractional margin.

    Must be called BEFORE add_basemap() so the tile fetcher knows which
    tiles to download.

    Parameters
    ----------
    ax     : matplotlib Axes
    df     : DataFrame with 'lon' and 'lat' columns
    margin : float — fractional padding on each side (default 4%)
    """
    lon_pad = (df["lon"].max() - df["lon"].min()) * margin
    lat_pad = (df["lat"].max() - df["lat"].min()) * margin
    ax.set_xlim(df["lon"].min() - lon_pad, df["lon"].max() + lon_pad)
    ax.set_ylim(df["lat"].min() - lat_pad, df["lat"].max() + lat_pad)


# =============================================================================
# FIGURE 1 & 3 — LISA CLUSTER MAPS (single index)
# =============================================================================

def plot_lisa_map(df, index_name: str, output_path: Path, provider: str = "carto_positron"):
    """
    Produce a LISA cluster map for one vulnerability index.

    Rendering order:
        1. OSM basemap tiles     (zorder 0) — drawn first, behind everything
        2. Non-significant pts   (zorder 3) — grey background scatter layer
        3. Significant clusters  (zorder 4) — on top of non-significant
        4. Community name labels (zorder 6) — topmost data layer
        5. OSM attribution text  (zorder 10) — added automatically by add_basemap

    Parameters
    ----------
    df           : DataFrame from load_lisa_csv()
    index_name   : str  — 'FVI' or 'EVI', used in the title
    output_path  : Path — PNG save location
    """
    n_sig   = (df["cluster_type"] != "Not significant").sum()
    n_tot   = len(df)
    pct_sig = n_sig / n_tot * 100

    fig, ax = plt.subplots(figsize=(13, 8))

    # Step 1 — set axes extent BEFORE fetching tiles
    _set_extent(ax, df)

    # Step 2 — OSM basemap (zorder 0, behind all artists)
    basemap_ok = add_basemap(
        ax,
        zoom=BASEMAP_ZOOM,
        provider=provider,
        alpha=BASEMAP_ALPHA,
        cache_dir=OUT_DIR / "osm_tile_cache",
    )

    # Step 3 — scatter LISA clusters
    _scatter_clusters(ax, df)

    # Step 4 — community labels
    _add_community_labels(ax, df)

    # Step 5 — legend
    ax.legend(
        handles=_legend_handles(df),
        title="Cluster type", title_fontsize=9,
        fontsize=8.5, loc="lower left",
        framealpha=0.92, edgecolor="#bbbbbb",
    )

    # Step 6 — style and title
    # Build basemap credit text for the title
    _provider_labels = {
        "carto_voyager":    "CartoDB Voyager (rivers visible)",
        "carto_positron":   "CartoDB Positron",
        "carto_darkmatter": "CartoDB Dark Matter",
        "carto":            "CartoDB Positron",
        "osm":              "OpenStreetMap",
        "stamen_watercolor":"Stamen Watercolor",
        "stamen_toner":     "Stamen Toner-Lite",
        "stamen":           "Stamen Toner-Lite",
    }
    credit = (
        f" | Basemap: {_provider_labels.get(provider, provider)} © OSM contributors"
        if basemap_ok else ""
    )
    _style_map(
        ax,
        title=(
            f"LISA Cluster Map — Local Moran's I ({index_name})\n"
            f"Lumbini Province, Nepal  |  "
            f"Threshold = {LISA_THRESHOLD_KM} km, "
            f"p < {LISA_ALPHA}, {N_PERMUTATIONS} permutations  |  "
            f"{n_sig}/{n_tot} significant ({pct_sig:.1f}%){credit}"
        ),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# FIGURE 2 & 4 — CLUSTER FREQUENCY BAR CHARTS
# =============================================================================

def plot_lisa_frequency_chart(df, index_name: str, output_path: Path):
    """
    Produce a horizontal bar chart of LISA cluster type frequencies.

    Not a map — no basemap needed. Intended for the thesis appendix.

    Parameters
    ----------
    df           : DataFrame from load_lisa_csv()
    index_name   : str  — 'FVI' or 'EVI'
    output_path  : Path — PNG save location
    """
    counts = df["cluster_type"].value_counts()
    ordered_types = [
        ct for ct in
        ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]
        if ct in counts.index
    ]
    n_total = len(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [CLUSTER_LABELS[ct] for ct in ordered_types],
        [counts[ct] for ct in ordered_types],
        color=[CLUSTER_COLOURS[ct] for ct in ordered_types],
        edgecolor=[CLUSTER_EDGE[ct] if ct != "Not significant" else "#999999"
                   for ct in ordered_types],
        linewidth=0.6,
        height=0.55,
    )

    for bar, ct in zip(bars, ordered_types):
        n   = counts[ct]
        pct = n / n_total * 100
        ax.text(
            bar.get_width() + n_total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"  n={n:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_xlabel("Number of households", fontsize=10)
    ax.set_title(
        f"LISA Cluster Frequencies — {index_name}\n"
        f"Local Moran's I, Lumbini Province, Nepal  |  n = {n_total:,}",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.set_xlim(0, n_total * 1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# FIGURE 5 — SIDE-BY-SIDE COMPARISON MAP
# =============================================================================

def plot_comparison_map(fvi_df, evi_df, output_path: Path):
    """
    Side-by-side LISA cluster map comparing FVI and EVI.

    Both panels use OSM basemap tiles from the shared cache, so tiles
    downloaded for the FVI panel are reused for the EVI panel.

    This figure directly supports the thesis discussion of EVI ICC = 23.8%
    versus FVI ICC = 8.6% — the spatial cluster patterns differ noticeably
    between the two indices.

    Parameters
    ----------
    fvi_df      : DataFrame from load_lisa_csv() for FVI
    evi_df      : DataFrame from load_lisa_csv() for EVI
    output_path : Path — PNG save location
    """
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    basemap_notes = []

    # FVI panel uses Voyager (rivers visible); EVI uses Positron (minimal)
    _providers = [BASEMAP_PROVIDER_FVI, BASEMAP_PROVIDER_EVI]

    for ax, df, name, prov in zip(axes, [fvi_df, evi_df], ["FVI", "EVI"], _providers):
        n_sig   = (df["cluster_type"] != "Not significant").sum()
        n_tot   = len(df)
        pct_sig = n_sig / n_tot * 100

        _set_extent(ax, df)

        ok = add_basemap(
            ax, zoom=BASEMAP_ZOOM, provider=prov,
            alpha=BASEMAP_ALPHA, cache_dir=OUT_DIR / "osm_tile_cache",
        )
        basemap_notes.append(ok)

        _scatter_clusters(ax, df)
        _add_community_labels(ax, df)

        ax.legend(
            handles=_legend_handles(df),
            title="Cluster type", title_fontsize=8,
            fontsize=7.5, loc="lower left",
            framealpha=0.92, edgecolor="#bbbbbb",
        )
        _style_map(
            ax,
            title=(
                f"{name} — LISA Clusters\n"
                f"{n_sig}/{n_tot} significant ({pct_sig:.1f}%)"
            ),
        )

    credit = (
        "  |  Basemap: FVI = CartoDB Voyager (rivers), EVI = CartoDB Positron  © OpenStreetMap contributors"
        if any(basemap_notes) else ""
    )
    fig.suptitle(
        "LISA Cluster Map Comparison: FVI vs EVI\n"
        f"Lumbini Province, Nepal  |  "
        f"Local Moran's I, threshold = {LISA_THRESHOLD_KM} km, "
        f"p < {LISA_ALPHA}, {N_PERMUTATIONS} permutations{credit}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# SUMMARY CSV
# =============================================================================

def save_lisa_summary(fvi_df, evi_df, output_path: Path):
    """
    Save a tidy CSV with cluster frequencies for both indices.

    Columns: index, cluster_type, n, pct

    Parameters
    ----------
    fvi_df, evi_df : DataFrame or None
    output_path    : Path — CSV save location
    """
    rows = []
    for label, df in [("FVI", fvi_df), ("EVI", evi_df)]:
        if df is None:
            continue
        n_total = len(df)
        for ct in ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]:
            n = (df["cluster_type"] == ct).sum()
            rows.append({
                "index":        label,
                "cluster_type": ct,
                "n":            int(n),
                "pct":          round(n / n_total * 100, 1),
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")
    print()
    print(f"  {'Index':<6}  {'Cluster type':<20}  {'n':>5}  {'%':>6}")
    print("  " + "-" * 42)
    for row in rows:
        print(
            f"  {row['index']:<6}  {row['cluster_type']:<20}  "
            f"{row['n']:>5}  {row['pct']:>5.1f}%"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all LISA cluster mapping figures for FVI and EVI."""
    print("=" * 60)
    print("LISA CLUSTER MAPS")
    print("=" * 60)
    print(f"Working directory  : {OUT_DIR}")
    print(f"Threshold          : {LISA_THRESHOLD_KM} km")
    print(f"Significance level : p < {LISA_ALPHA}")
    print(f"Basemap provider   : FVI={BASEMAP_PROVIDER_FVI}, EVI={BASEMAP_PROVIDER_EVI}  (alpha={BASEMAP_ALPHA})")
    print(f"Tile zoom          : {BASEMAP_ZOOM}")
    print(f"Tile cache         : {OUT_DIR / 'osm_tile_cache'}")
    print(f"osm_tiles module   : {'available' if _BASEMAP_AVAILABLE else 'NOT FOUND — basemap disabled'}")
    print()

    fvi_df = load_lisa_csv(FVI_LISA_CSV, "FVI")
    evi_df = load_lisa_csv(EVI_LISA_CSV, "EVI")

    if fvi_df is None and evi_df is None:
        print("ERROR: No LISA CSVs found. Run Stages 5 and 11 first.")
        return

    if fvi_df is not None:
        print("FVI figures:")
        plot_lisa_map(fvi_df, "FVI", OUT_DIR / "fvi_lisa_cluster_map.png", provider=BASEMAP_PROVIDER_FVI)
        plot_lisa_frequency_chart(fvi_df, "FVI", OUT_DIR / "fvi_lisa_frequency_chart.png")
        print()

    if evi_df is not None:
        print("EVI figures:")
        plot_lisa_map(evi_df, "EVI", OUT_DIR / "evi_lisa_cluster_map.png", provider=BASEMAP_PROVIDER_EVI)
        plot_lisa_frequency_chart(evi_df, "EVI", OUT_DIR / "evi_lisa_frequency_chart.png")
        print()

    if fvi_df is not None and evi_df is not None:
        print("Comparison map:")
        plot_comparison_map(fvi_df, evi_df, OUT_DIR / "lisa_comparison_map.png")
        print()

    print("Cluster frequency summary:")
    save_lisa_summary(fvi_df, evi_df, OUT_DIR / "lisa_summary.csv")
    print()
    print("LISA cluster mapping complete.")
    print("=" * 60)


# Run when loaded by run_pipeline.py via importlib, and also directly.
main()

if __name__ == "__main__":
    pass  # already called above
