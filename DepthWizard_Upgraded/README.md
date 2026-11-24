# DepthWizard Upgraded

An expanded Streamlit application based on the existing DepthWizard MiDaS Small prototype.

## Features
- Pretrained MiDaS Small monocular relative-depth estimation.
- Upload and batch-process up to 12 JPG, PNG, or TIFF images.
- Multiple depth color maps (Viridis, grayscale, Turbo, Inferno, Plasma, Magma).
- Interactive Plotly 3D surface with rotation, pan, zoom, color selection, resolution, and vertical exaggeration controls.
- Region-of-interest statistics using pixel-coordinate sliders.
- PDF analysis report and PNG/NPY/ZIP downloads.
- Optional GeoTIFF RGB/DEM reading and calibrated surface GeoTIFF export through Rasterio.

## Windows setup
Use a fresh virtual environment. MiDaS/Torch installation can vary by Python version; Python 3.10–3.12 is generally a safer compatibility target than a newly released Python version.

```powershell
cd path\to\DepthWizard_Upgraded
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional DEM / georeferenced GeoTIFF support:
pip install -r requirements-geospatial.txt
python -m streamlit run app.py
```

First model load needs internet access to fetch the MiDaS repository and weights. If you already have a working PyTorch environment, use that environment rather than reinstalling Torch blindly.

## DEM workflow and scientific limitations
- MiDaS Small estimates relative monocular depth; its normalized output is not elevation in metres.
- Upload a single-band reference DEM GeoTIFF only when it covers the same scene and is already aligned to the image's pixel grid.
- The current calibration fits an affine mapping from relative depth to DEM values. Its MAE/RMSE are **in-sample fit residuals**, not independent accuracy validation.
- Matching array dimensions does not establish geographic co-registration. For robust work, reproject imagery and DEM to a common CRS, transform, extent, and resolution using GIS tooling, then evaluate on independent checkpoints.
- GeoTIFF export uses the uploaded DEM's grid metadata. Do not interpret output as a validated DSM without independent validation.

## Project structure
```text
DepthWizard_Upgraded/
├── app.py
├── depth_engine.py
├── visualization.py
├── analysis.py
├── geospatial.py
├── reporting.py
├── requirements.txt
├── README.md
├── .gitignore
├── outputs/
└── examples/
```

## Troubleshooting
- If `rasterio` fails to install on Windows, run the core app without DEM features; Rasterio is optional and installed separately from `requirements-geospatial.txt`.
- If MiDaS model download fails, check internet access and Torch Hub cache. Do not delete the working original project while troubleshooting.
- CPU inference can be slow for large images. Try a smaller image or reduce the number of batch uploads.
