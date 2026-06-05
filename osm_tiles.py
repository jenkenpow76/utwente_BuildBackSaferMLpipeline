"""
osm_tiles.py
============
Lightweight OpenStreetMap basemap helper for the vulnerability index pipeline.

Usage
-----
    from osm_tiles import add_basemap

    fig, ax = plt.subplots()
    ax.scatter(lons, lats, ...)          # plot your data first
    add_basemap(ax, zoom="auto")         # then add the basemap behind it
    ax.set_xlim(lon_min, lon_max)        # tight axes already set by scatter
    ax.set_ylim(lat_min, lat_max)

How it works
------------
1.  Reads the current axes extent (lon/lat in WGS-84, EPSG:4326).
2.  Converts to Web Mercator (EPSG:3857) tile coordinates.
3.  Downloads the required OSM tile images from the Stamen Terrain or
    OpenStreetMap tile server and stitches them into a single image.
4.  Renders that image behind all existing artists using ax.imshow().
5.  Resets axes ticks to show plain lon/lat degree values (not metres).

The download is cached to a local directory (default: ./osm_tile_cache/)
so re-running the pipeline does not re-fetch tiles already on disk.

Zoom levels
-----------
zoom="auto"  selects a zoom level based on the spatial extent of the data:
    extent > 2°  → zoom 10
    extent > 0.5° → zoom 12
    else          → zoom 13

These levels are appropriate for the Banke / Dang study area (~1° × ~0.5°).
Higher zoom levels produce sharper tiles but download more files.

Tile providers
--------------
PROVIDER = "osm"      OpenStreetMap Standard (osm.org tiles)
PROVIDER = "stamen"   Stamen Toner-Lite (greyscale, data-friendly)
PROVIDER = "carto"    CartoDB Positron (light grey, minimal)

The default is "carto" (CartoDB Positron) because its pale grey palette
does not compete with the coloured LISA cluster markers.

Error handling
--------------
If any tile download fails (no internet, server timeout), the function
prints a warning and leaves the axes background as the default grey.
The pipeline continues; the map is produced without a basemap.

Dependencies
------------
urllib.request (stdlib), PIL/Pillow, numpy, matplotlib — no extra installs.

References
----------
OpenStreetMap tile usage policy: https://operations.osmfoundation.org/policies/tiles/
CartoDB basemap tiles: https://github.com/CartoDB/basemap-styles
Stamen tile service: http://maps.stamen.com/
"""

import io
import math
import os
import urllib.request
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# PIL import — required for tile image handling
# ---------------------------------------------------------------------------
try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# =============================================================================
# TILE SERVER URLS
# =============================================================================

# Each provider URL must contain {z}, {x}, {y} placeholders.
#
# Provider guide — choose based on which hazard is being mapped:
#
#   "carto_voyager"    CartoDB Voyager — RECOMMENDED FOR FLOOD MAPS.
#                      Shows rivers in pale blue, gentle terrain shading,
#                      light roads. Rivers are clearly visible without
#                      competing with LISA cluster colours.
#
#   "carto_positron"   CartoDB Positron — RECOMMENDED FOR EARTHQUAKE MAPS.
#                      Minimal pale grey style, no river/terrain features.
#                      Best when geographic context matters less than data.
#
#   "osm"              OpenStreetMap Standard — rivers clearly visible in
#                      blue, but the colourful land-use scheme competes with
#                      diverging LISA colour scales.
#
#   "stamen_watercolor" Stamen Watercolor — rivers rendered as bold blue
#                      washes. Visually striking for thesis figures; may be
#                      too stylised for journal submission.
#
#   "stamen_toner"     Stamen Toner-Lite — greyscale, high-contrast roads
#                      and water. Good for black-and-white printing.
#
#   "carto_darkmatter" CartoDB Dark Matter — dark background. Use only if
#                      your cluster colours are bright/saturated.
#
# Aliases for backward compatibility:
#   "carto"  → "carto_positron"
#   "stamen" → "stamen_toner"
TILE_PROVIDERS: dict[str, str] = {
    # CartoDB family — no API key required
    "carto_voyager":    "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    "carto_positron":   "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "carto_darkmatter": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    # OpenStreetMap Standard — rivers and terrain visible, colourful
    "osm":              "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    # Stamen family — artistic / high-contrast styles
    "stamen_watercolor": "https://stamen-tiles.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg",
    "stamen_toner":      "https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png",
    # Backward-compatible aliases (old names still work)
    "carto":  "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "stamen": "https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png",
}

# Browser-like User-Agent required by OSM tile policy.
# https://operations.osmfoundation.org/policies/tiles/
_USER_AGENT = (
    "Mozilla/5.0 (compatible; VulnerabilityIndexPipeline/1.0; "
    "+https://github.com/your-repo)"
)

# Default tile cache directory (relative to cwd, i.e. OUTPUT_DIR)
DEFAULT_CACHE_DIR = Path("osm_tile_cache")


# =============================================================================
# COORDINATE CONVERSION
# =============================================================================


def _lon_lat_to_tile(lon_deg: float, lat_deg: float, zoom: int) -> tuple[int, int]:
    """
    Convert WGS-84 longitude/latitude to OSM tile (x, y) at a given zoom.

    Uses the standard OSM slippy-map tile numbering scheme.
    Reference: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    Parameters
    ----------
    lon_deg : float — longitude in decimal degrees
    lat_deg : float — latitude in decimal degrees
    zoom    : int   — tile zoom level (0–19)

    Returns
    -------
    (tile_x, tile_y) : tuple[int, int]
    """
    lat_r = math.radians(lat_deg)
    n = 2**zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    # Clamp to valid range
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _tile_to_lon_lat(x: int, y: int, zoom: int) -> tuple[float, float]:
    """
    Convert OSM tile (x, y) at zoom to the NW corner WGS-84 lon/lat.

    Reference: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    Parameters
    ----------
    x, y : int — tile coordinates
    zoom : int — zoom level

    Returns
    -------
    (lon_deg, lat_deg) : tuple[float, float] — NW corner of the tile
    """
    n = 2**zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_r)
    return lon_deg, lat_deg


# =============================================================================
# TILE DOWNLOAD
# =============================================================================


def _download_tile(
    url: str,
    cache_path: Path,
) -> "Image.Image | None":
    """
    Download a single tile image, using a local cache.

    If the tile is already cached, load it from disk without a network
    request. On download failure (network error, HTTP error, timeout),
    returns None so the caller can skip this tile gracefully.

    Parameters
    ----------
    url        : str  — full tile URL (z/x/y already substituted)
    cache_path : Path — local path to store the downloaded tile

    Returns
    -------
    PIL.Image or None
    """
    # Return cached tile if available
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGBA")
        except Exception:
            pass  # corrupt cache — re-download below

    # Download
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()

        # Cache to disk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)

        return Image.open(io.BytesIO(data)).convert("RGBA")

    except Exception as exc:
        print(f"    [osm_tiles] WARNING: could not download tile {url}: {exc}")
        return None


# =============================================================================
# AUTO ZOOM SELECTION
# =============================================================================


def _auto_zoom(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> int:
    """
    Choose a tile zoom level appropriate for the spatial extent.

    The Banke/Dang study area spans roughly 1° longitude × 0.5° latitude.
    Zoom 11 gives ~4 km per tile at that latitude, which produces clear
    context without downloading too many tiles.

    Parameters
    ----------
    lon_min, lon_max, lat_min, lat_max : float — bounding box in degrees

    Returns
    -------
    zoom : int
    """
    max_extent = max(lon_max - lon_min, lat_max - lat_min)
    if max_extent > 2.0:
        return 10
    elif max_extent > 0.5:
        return 11
    elif max_extent > 0.2:
        return 12
    else:
        return 13


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================


def add_basemap(
    ax: plt.Axes,
    zoom: int | Literal["auto"] = "auto",
    provider: str = "carto",
    alpha: float = 0.55,
    cache_dir: Path | str | None = None,
    pad_fraction: float = 0.02,
) -> bool:
    """
    Add an OpenStreetMap basemap to an existing matplotlib axes.

    Call this AFTER plotting your data so the axes limits are already set.
    The basemap is rendered behind all existing artists (zorder=0).

    Parameters
    ----------
    ax           : plt.Axes — the target axes (data must already be plotted)
    zoom         : int or "auto" — tile zoom level; "auto" selects based on
                   the axes extent
    provider     : str — tile provider key: "osm", "stamen", or "carto"
                   Default is "carto" (pale grey, does not compete with data)
    alpha        : float — basemap transparency; 0 = invisible, 1 = opaque
                   Default 0.55 keeps coloured LISA markers clearly visible
    cache_dir    : Path or None — local tile cache directory;
                   defaults to ./osm_tile_cache/ (relative to cwd)
    pad_fraction : float — fractional padding added to axes extent before
                   fetching tiles; avoids white edge strips

    Returns
    -------
    bool — True if basemap was added successfully, False if skipped
           (missing PIL, no network, or unsupported provider)

    Notes
    -----
    - Axes tick labels are converted back to lon/lat degree format after the
      basemap is rendered (the imshow call does not change them).
    - Axes limits are preserved exactly as they were before the call.
    - The function does NOT call plt.tight_layout() or plt.show().

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(lons, lats, c=colours, s=20)
    >>> success = add_basemap(ax, zoom="auto", provider="carto", alpha=0.5)
    >>> plt.savefig("map.png", dpi=200, bbox_inches="tight")
    """
    if not _PIL_AVAILABLE:
        print(
            "  [osm_tiles] WARNING: Pillow (PIL) not installed. "
            "Install with: pip install Pillow --break-system-packages\n"
            "  Basemap skipped."
        )
        return False

    if provider not in TILE_PROVIDERS:
        print(
            f"  [osm_tiles] WARNING: Unknown provider '{provider}'. "
            f"Choose from: {list(TILE_PROVIDERS.keys())}.\n"
            "  Basemap skipped."
        )
        return False

    # ── Read axes extent ─────────────────────────────────────────────────────
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()

    # Add small padding so tiles cover the full extent including axis edges
    pad_lon = (lon_max - lon_min) * pad_fraction
    pad_lat = (lat_max - lat_min) * pad_fraction
    lon_min_p = lon_min - pad_lon
    lon_max_p = lon_max + pad_lon
    lat_min_p = lat_min - pad_lat
    lat_max_p = lat_max + pad_lat

    # ── Select zoom ──────────────────────────────────────────────────────────
    if zoom == "auto":
        zoom = _auto_zoom(lon_min_p, lon_max_p, lat_min_p, lat_max_p)

    # ── Compute tile range ───────────────────────────────────────────────────
    # NW corner of bounding box uses lat_max; SE corner uses lat_min
    x_min_t, y_min_t = _lon_lat_to_tile(lon_min_p, lat_max_p, zoom)
    x_max_t, y_max_t = _lon_lat_to_tile(lon_max_p, lat_min_p, zoom)

    n_x = x_max_t - x_min_t + 1
    n_y = y_max_t - y_min_t + 1
    n_tiles = n_x * n_y

    if n_tiles > 64:
        print(
            f"  [osm_tiles] WARNING: zoom={zoom} requires {n_tiles} tiles "
            f"({n_x} × {n_y}). Reducing zoom to avoid excessive downloads."
        )
        zoom = max(zoom - 2, 8)
        x_min_t, y_min_t = _lon_lat_to_tile(lon_min_p, lat_max_p, zoom)
        x_max_t, y_max_t = _lon_lat_to_tile(lon_max_p, lat_min_p, zoom)
        n_x = x_max_t - x_min_t + 1
        n_y = y_max_t - y_min_t + 1
        n_tiles = n_x * n_y
        print(f"  [osm_tiles] Using zoom={zoom}, {n_tiles} tiles.")

    # ── Cache directory ───────────────────────────────────────────────────────
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir) / provider / str(zoom)

    # ── Download and stitch tiles ────────────────────────────────────────────
    tile_size = 256  # OSM tiles are always 256×256 pixels
    canvas = Image.new("RGBA", (n_x * tile_size, n_y * tile_size), (200, 200, 200, 255))

    url_template = TILE_PROVIDERS[provider]
    n_failed = 0

    for row_idx, ty in enumerate(range(y_min_t, y_max_t + 1)):
        for col_idx, tx in enumerate(range(x_min_t, x_max_t + 1)):
            url = url_template.format(z=zoom, x=tx, y=ty)
            cache_path = cache_dir / f"{tx}_{ty}.png"
            tile_img = _download_tile(url, cache_path)
            if tile_img is None:
                n_failed += 1
                continue
            # Paste tile into the canvas at the correct pixel offset
            canvas.paste(
                tile_img,
                (col_idx * tile_size, row_idx * tile_size),
            )

    if n_failed == n_tiles:
        print(
            "  [osm_tiles] WARNING: All tile downloads failed. "
            "Check internet connection. Basemap skipped."
        )
        return False

    if n_failed > 0:
        print(
            f"  [osm_tiles] WARNING: {n_failed}/{n_tiles} tiles failed. "
            "Partial basemap rendered."
        )

    # ── Compute geographic extent of the stitched image ──────────────────────
    # The extent is the NW corner of tile (x_min_t, y_min_t) to the
    # SE corner of tile (x_max_t, y_max_t), i.e. NW corner of (x_max_t+1, y_max_t+1)
    img_lon_min, img_lat_max = _tile_to_lon_lat(x_min_t,     y_min_t,     zoom)
    img_lon_max, img_lat_min = _tile_to_lon_lat(x_max_t + 1, y_max_t + 1, zoom)

    # ── Render basemap behind existing artists ────────────────────────────────
    # imshow with zorder=0 places it below scatter/line artists.
    # extent=[left, right, bottom, top] in axes (lon/lat) coordinates.
    ax.imshow(
        np.array(canvas),
        extent=[img_lon_min, img_lon_max, img_lat_min, img_lat_max],
        origin="upper",        # image row 0 = north (lat_max)
        aspect="equal",
        alpha=alpha,
        zorder=0,              # behind all other artists
        interpolation="bilinear",
    )

    # Restore exact original axes limits (imshow can expand them)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # ── Attribution text (OSM tile usage policy requires attribution) ─────────
    # Required attribution per OSM tile usage policy and CARTO/Stamen terms
    provider_credits = {
        "osm":               "© OpenStreetMap contributors",
        "carto_voyager":     "© OpenStreetMap contributors © CARTO",
        "carto_positron":    "© OpenStreetMap contributors © CARTO",
        "carto_darkmatter":  "© OpenStreetMap contributors © CARTO",
        "carto":             "© OpenStreetMap contributors © CARTO",
        "stamen_watercolor": "Map tiles by Stamen Design (CC BY 3.0) · © OpenStreetMap contributors",
        "stamen_toner":      "Map tiles by Stamen Design (CC BY 3.0) · © OpenStreetMap contributors",
        "stamen":            "Map tiles by Stamen Design (CC BY 3.0) · © OpenStreetMap contributors",
    }
    ax.text(
        0.01, 0.01,
        provider_credits.get(provider, "© OpenStreetMap contributors"),
        transform=ax.transAxes,
        fontsize=6,
        color="#555555",
        va="bottom",
        ha="left",
        zorder=10,
        style="italic",
    )

    return True
