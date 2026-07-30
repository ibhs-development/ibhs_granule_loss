# IBHS Granule Loss Analysis

IBHS desktop tool for analyzing granule loss from microscope shingle images that include a red scale bar.

The project provides:
- A Tkinter GUI (`app.py`) for selecting folders, running analysis, and viewing logs/plots
- A processing pipeline (`loss.py`) that generates cropped/cleaned analysis images, computes scale from a red reference bar, segments granule-loss regions, and computes IGL/PGL metrics

## What the analysis does

For each source image:
1. Scans the input folder recursively for scale-bar images.
2. Generates a cropped/cleaned analysis image beside each source as `*_cropped.*`.
3. Detects the red scale bar in the original image, works out how long it is in mm
   (see **Scale bar length** below), and derives `mm_per_px = bar_mm / bar_px`.
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

## Scale bar length (10 mm vs 20 mm)

Areas scale with the **square** of `mm_per_px`, so mistaking a 10 mm bar for a 20 mm one
makes every reported area **4x too large** and pushes regions across the IGL/PGL cutoff.
The bar length is therefore resolved in this order, and how it was resolved is always
reported:

1. **Explicit override** — the GUI's *Scale bar length* setting (`10 mm` / `20 mm`), or
   `forced_scale_mm=` when calling `process_granule_loss` directly.
2. **File-name tag** — any image whose name contains `10mm` or `20mm` (e.g. `impact_7_10mm.png`).
3. **Reading the printed label** next to the bar. Since `10mm` and `20mm` differ only in
   the first digit, the reader measures that digit's ink width against the `0` beside it,
   which makes the test independent of font, size and stroke style. Measured over 33 system
   fonts, a `1` never exceeds `0.944` of the `0` and a `2` never drops below `0.883`; ratios
   inside that overlap are declared **unresolved** rather than guessed.
4. **Bar-width guess** (legacy `detect_scale_mm`) — only if the label cannot be read. This is
   unreliable: it compares the bar to the image width, which assumes a fixed framing. It is
   flagged as unverified everywhere.
5. **`DEFAULT_SCALE_BAR_MM`** (20 mm) if even that fails; also flagged.

Anything resolved by 1-3 is marked `Scale_Verified = True`. For 4-5 the run logs a warning,
the GUI shows the affected images, and the annotated image carries an amber
`UNVERIFIED` note — fix those by setting the override or tagging the file names.

Every annotated image redraws the bar at its **detected pixel length**, captioned
`10 mm = 211 px (label)`, and boxes where the bar was found, so the calibration behind the
areas can be checked against the source at a glance.

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
- Optionally set **Scale bar length** (default *Auto-detect*; pick `10 mm`/`20 mm` to force it)
- Click **Generate Crops + Analyze**

## Output files

In the selected output folder:
- `granule_loss_results.csv` — one row per image: `Count_IGL`, `Count_PGL`, `Count_All`,
  `AreaSum_IGL_mm2`, `AreaSum_PGL_mm2`, `AreaSum_All_mm2`, ratings, `mm_per_px_original`,
  `ScaleBar_mm` / `ScaleBar_px` / `Scale_Source` / `Scale_Verified`, and the paths of the
  cropped and annotated images
- `granule_loss_regions.csv` — one row per detected loss region: `Impact`, `Region_ID`,
  `Class` (`IGL`/`PGL`), `Area_px`, `Area_mm2`, centroid, and whether it got an on-image label
- `granule_loss_plot.png`

In the selected input folder:
- `*_cropped.*` generated beside each source image
- `*_annotated.*` generated beside each source image

## Code review notes
- If no red bar is found at all, the code falls back to `20/300 mm/px` and marks the image
  `Scale_Verified = False`; those areas are not trustworthy.
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
