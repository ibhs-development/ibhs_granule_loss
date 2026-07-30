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
7. Writes an annotated (analysed) image beside each source as `*_annotated.*`, with every
   detected loss region tinted, outlined (blue = IGL, red = PGL) and labelled with its area
   in `mm²`, plus a header showing the per-image counts and area sums and a mm scale bar.
8. Computes percentile-based severity levels and final ratings (`0..3`) per impact.
9. Produces pooled distribution plots, a per-image CSV summary, and a per-region CSV.

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

Example generated set:
- `12.png`
- `12_cropped.png` (cleaned image the measurements are taken from)
- `12_annotated.png` (same image with each loss area labelled in `mm²`)

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
- `granule_loss_results.csv` — one row per image: `Count_IGL`, `Count_PGL`, `Count_All`,
  `AreaSum_IGL_mm2`, `AreaSum_PGL_mm2`, `AreaSum_All_mm2`, ratings, `mm_per_px_original`,
  and the paths of the cropped and annotated images
- `granule_loss_regions.csv` — one row per detected loss region: `Impact`, `Region_ID`,
  `Class` (`IGL`/`PGL`), `Area_px`, `Area_mm2`, centroid, and whether it got an on-image label
- `granule_loss_plot.png`

In the selected input folder:
- `*_cropped.*` generated beside each source image
- `*_annotated.*` generated beside each source image

## Code review notes
- If scale detection fails, code falls back to `20/300 mm/px`; results may be less accurate for those samples.
- Annotated images show exactly what the segmenter measured. Spots missed by the
  local-density segmentation are neither annotated nor counted — tune `LOCAL_DENSITY_THRESH`
  and `LOCAL_DENSITY_K` in `loss.py` if faint or very small losses are being skipped.
- Very small regions (`< ANNOTATION_MIN_LABEL_AREA_PX`) and regions whose label would
  overlap another label stay outlined but unlabelled; the header reports how many were
  labelled, and every area is in `granule_loss_regions.csv`.

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
