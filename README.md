# IBHS Granule Loss Analysis

IBHS desktop tool for analyzing granule loss from microscope shingle images that include a red scale bar.

The project provides:
- A Tkinter GUI (`app.py`) for selecting folders, running analysis, and viewing logs/plots
- A processing pipeline (`loss.py`) that generates cropped/cleaned analysis images, computes scale from a red reference bar, segments granule-loss regions, and computes IGL/PGL metrics

## What the analysis does

For each source image:
1. Scans the input folder recursively for scale-bar images.
2. Generates a cropped/cleaned analysis image beside each source as `*_cropped.*`.
3. Detects red scale bar in the original image to estimate `mm_per_px`.
4. Segments granule-loss regions in the generated cropped image using local-density thresholding.
5. Measures connected-component areas in `mm²`.
6. Splits areas into:
   - `IGL`: area `< threshold` (default `2.58 mm²`)
   - `PGL`: area `>= threshold`
7. Computes percentile-based severity levels and final ratings (`0..3`) per impact.
8. Produces pooled distribution plots and a CSV summary.

## Requirements

- Python 3.11+
- Packages from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input data expectations

Set a **Scale-image folder** that contains the processed microscope images with scale bars. The scan is recursive, so nested folders are okay.

Supported image extensions:
- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.bmp`

The app no longer needs pre-existing associated `*_gl` images. It creates the cropped/cleaned analysis image automatically in the same folder as the source image:

Example generated pair:
- `12.png`
- `12_cropped.png`

## Run the app (GUI)

```bash
python app.py
```

In the GUI:
- Select **Scale-image folder**
- Select **Results folder** or leave it blank to use `granule_loss_results` under the selected input folder
- Optionally set **IGL vs PGL Threshold (mm²)** (default: `2.58`)
- Click **Generate Crops + Analyze**

## Output files

In the selected output folder:
- `granule_loss_results.csv` 
- `granule_loss_plot.png`

In the selected input folder:
- `*_cropped.*` generated beside each source image

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
