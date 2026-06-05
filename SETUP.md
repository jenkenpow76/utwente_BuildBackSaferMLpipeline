# Environment Setup — Vulnerability Index Pipeline (v27)

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.12 |
| Operating system | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| RAM | 8 GB minimum (16 GB recommended for stages 23–24) |
| Disk | ~2 GB for outputs |

Python 3.13 is not yet recommended — `pyampute` and `pygam` may not have
compatible wheels. Python 3.10–3.12 are confirmed working.

---

## Option A — conda (recommended)

Conda manages both Python and non-Python dependencies and avoids
conflicts between the spatial and ML packages.

```bash
# 1. Create a fresh environment named 'vuln_pipeline'
conda create -n vuln_pipeline python=3.11 -y

# 2. Activate it
conda activate vuln_pipeline

# 3. Install the majority of dependencies via conda-forge
conda install -c conda-forge \
    numpy pandas numexpr \
    matplotlib seaborn pillow \
    scipy scikit-learn \
    requests \
    -y

# 4. Install packages not on conda-forge via pip (inside the conda env)
pip install pyampute pygam
```

Verify the environment:

```bash
python - << 'EOF'
import numpy, pandas, matplotlib, seaborn, scipy, sklearn, pyampute, pygam, PIL, requests
print("numpy      ", numpy.__version__)
print("pandas     ", pandas.__version__)
print("matplotlib ", matplotlib.__version__)
print("seaborn    ", seaborn.__version__)
print("scipy      ", scipy.__version__)
print("sklearn    ", sklearn.__version__)
print("pygam      ", pygam.__version__)
print("PIL        ", PIL.__version__)
print("All imports OK.")
EOF
```

---

## Option B — pip with virtualenv

Use this if you do not have conda installed.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Upgrade pip to avoid resolver issues
python -m pip install --upgrade pip

# 4. Install all dependencies
pip install -r requirements.txt
```

---

## Running the pipeline

Place all pipeline files in a single folder alongside your data CSV:

```
pipeline_v27/
├── run_pipeline.py                         ← entry point
├── pipeline_config.py                      ← shared configuration
├── osm_tiles.py                            ← OSM tile helper
├── requirements.txt
├── stage01_fvi_scoring.py
├── stage02_evi_scoring.py
│   ... (all 28 stage scripts, including stage27_cross_method_comparison.py)
└── 20263003_Nepal_Lumbini_data.csv         ← survey data
```

From the `pipeline_v27/` directory, run:

```bash
python run_pipeline.py
```

Outputs are written to `pipeline_v27/outputs/`. A full execution log is
written to `pipeline_v27/pipeline.log`.

---

## Partial runs

To run from a specific stage (e.g. after fixing a mid-pipeline issue):

```bash
# Edit run_pipeline.py and set START_FROM at the top of the file:
python run_pipeline.py --start-stage 5
```

Or comment out completed stages in the `stages = [...]` list in
`run_pipeline.py` before re-running.

---

## Troubleshooting

### `ModuleNotFoundError: pyampute`
```bash
pip install pyampute
```
`pyampute` is not on conda-forge. Always install it via pip, even inside
a conda environment.

### `ModuleNotFoundError: pygam`
```bash
pip install pygam
```
Same situation as `pyampute` — pip only.

### `FutureWarning: Downcasting behavior in replace is deprecated`
This warning has been suppressed in v27+ by chaining
`.infer_objects(copy=False)` after all `.replace()` calls. If you see it,
confirm you are running v27 or later.

### `TypeError: Image data of dtype object cannot be converted to float`
Fixed in v27. The co-occurrence heatmap in the diagnostic stages crashed
when the missingness matrix retained `bool` dtype through the dot product.
Upgrade to v27 to resolve.

### `PerformanceWarning: DataFrame is highly fragmented`
This warning appears in pandas < 2.0 when columns are inserted one at a
time. Install pandas 2.0+ to eliminate it, or ignore it — it does not
affect results.

### OSM tiles not loading (stages 14, 19, 20)
The pipeline fetches OpenStreetMap tiles at runtime. Ensure your machine
has internet access when running stages 14, 19, and 20. Tile fetching
fails silently — the map is drawn without a basemap rather than crashing.

---

## Reproducibility note (FAIR principles)

To produce a fully reproducible environment, pin exact package versions:

```bash
# After installing, export the exact environment
pip freeze > requirements_pinned.txt
```

Include `requirements_pinned.txt` in your thesis appendix or repository.
This ensures any researcher can recreate the exact environment used to
produce your results, satisfying the **Reusable** component of FAIR.

For long-term archiving, also export the conda environment specification:

```bash
conda env export > environment.yml
```
