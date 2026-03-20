# IBHS Granule Loss Analysis

IBHS desktop tool for analyzing granule loss from paired microscope shingle images.

The project provides:
- A Tkinter GUI (`app.py`) for selecting folders, running analysis, and viewing logs/plots
- A processing pipeline (`loss.py`) that detects image pairs, computes scale from a red reference bar, segments granule-loss regions, and computes IGL/PGL metrics

## What the analysis does

For each matched image pair:
1. Finds image pairs (`original` + `cropped`) from nested folders.
2. Detects red scale bar in the original image to estimate `mm_per_px`.
3. Segments granule-loss regions in the cropped image using local-density thresholding.
4. Measures connected-component areas in `mm²`.
5. Splits areas into:
- `IGL`: area `< threshold` (default `2.58 mm²`)
- `PGL`: area `>= threshold`
6. Computes percentile-based severity levels and final ratings (`0..3`) per impact.
7. Produces pooled distribution plots and a CSV summary.

## Requirements

- Python 3.11+
- Packages from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input data expectations

Set an **Input Folder** that contains first-level and optionally second-level subfolders.

Inside each processed folder, pairing is based on filename patterns:
- Original image: `N.png` (example: `1.png`, `12.png`)
- Associated cropped image can match one of:
- `..._N_gl.png`
- `..._glN.png`
- `..._glsN.png`

Where `N` is the same numeric index used for the original image.

Example valid pair:
- `12.png`
- `sample_gl12.png`

## Run the app (GUI)

```bash
python app.py
```

In the GUI:
- Select **Input Folder**
- Select **Output Folder**
- Optionally set **IGL vs PGL Threshold (mm²)** (default: `2.58`)
- Click **Run Analysis**

## Output files

In the selected output folder:
- `granule_loss_results.csv` 
- Plot is generated and shown in GUI; expected output filename is `granule_loss_plot.png`

## Code review notes
- If scale detection fails, code falls back to `20/300 mm/px`; results may be less accurate for those samples.

## Build Windows EXE with PyInstaller

From project root:

```bash
pyinstaller --clean --noconfirm --onefile --windowed --name GranuleLoss app.py
```

Generated executable:
- `dist/GranuleLoss.exe`

If your environment misses runtime modules, rebuild with hidden imports:

```bash
pyinstaller --clean --noconfirm --onefile --windowed --name GranuleLoss \
  --hidden-import matplotlib.backends.backend_tkagg \
  --hidden-import skimage.measure \
  app.py
```

Use the `.exe` by launching it directly, then select input/output folders in the GUI.
