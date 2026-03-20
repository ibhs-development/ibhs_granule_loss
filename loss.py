import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy import stats

# The script will scan this folder for image pairs.
# All steps we need

ROOT_DATA_FOLDER = "Completed_GL"

# Image processing for granule-loss segmentation
BLUR_K = 5                   # Gaussian blur kernel (odd); 5 is a good default
LOCAL_DENSITY_K = 15         # box-filter window for local granule density (odd)
LOCAL_DENSITY_THRESH = 0.45  # low-density cutoff -> loss (tune 0.35–0.55)
BINS = 30                    # histogram bins for the PDFs
FIGSIZE = (16, 4)            # size of the 3-panel PDF plot
OUTPUT_FIG = "Completed_GL"          # figure file name

# the threshold for IGL vs PGL classification (2.58 mm²)
IGL_CUTOFF_MM2 = 2.58

# -----------------------------
# Utilities
# -----------------------------
def _read_rgb(path: Path) -> np.ndarray:
    """Read image via cv2 and convert BGR->RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_long_side(img: np.ndarray, target_max_dim: int) -> np.ndarray:
    """Resize so that the longer dimension == target_max_dim."""
    h, w = img.shape[:2]
    scale = target_max_dim / max(h, w)
    if scale >= 1:
        return img.copy()
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def detect_scale_mm(image_path, target_max_dim=1200, bottom_frac=0.4, px_threshold=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # Normalize size
    h, w = img.shape[:2]
    scale = target_max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Detect red in HSV (two hue ranges for red)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0,120,120]), np.array([10,255,255])
    lower2, upper2 = np.array([170,120,120]), np.array([180,255,255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Focus on the bottom region where the bar lives
    H, W = mask.shape
    y0 = int(H * (1 - bottom_frac))
    roi = mask[y0:, :]

    # Clean up & connect the bar
    kernel = np.ones((5, 21), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the widest horizontal red blob
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bar_w = None
    if cnts:
        best_w = 0
        for c in cnts:
            x,y,wc,hc = cv2.boundingRect(c)
            if wc / max(hc,1) > 4 and wc*hc > 200:  # long & thin
                if wc > best_w:
                    best_w = wc
        if best_w > 0:
            bar_w = best_w

    if bar_w is None:
        return {"mm": None, "bar_width_px": None, "normalized_width": W}

    # Decide 10 vs 20
    if px_threshold is not None:
        mm = 20 if bar_w >= px_threshold else 10
    else:
        # Fraction-of-width threshold (works across images after normalization)
        frac = bar_w / W
        # Empirically: ~0.11 (10 mm) vs ~0.20 (20 mm). Midpoint ~0.16.
        mm = 20 if frac > 0.16 else 10

    return mm

def find_image_pairs(root_folder='results'):
    """
    Scans a directory for original and associated image pairs in all subfolders.

    Args:
        root_folder (str): The path to the main folder to start scanning from (e.g., 'results').

    Returns:
        pandas.DataFrame: A DataFrame containing the matched image pairs with columns:
                          'Impact', 'original', 'cropped'.
                          Returns an empty DataFrame if the root folder doesn't exist or no pairs are found.
    """
    if not os.path.isdir(root_folder):
        print(f"Error: The directory '{root_folder}' does not exist.")
        return pd.DataFrame()

    all_pairs = []

    # Regex to find original images like '1.png', '12.png'
    original_regex = re.compile(r'^(\d+)\.png$')

    # Regex patterns for associated images
    # Pattern 1 (number in the middle): e.g., ..._S1_19_gl.png
    assoc_regex_1 = re.compile(r'_(\d+)_gl\.png$')
    # Pattern 2 (number at the end): e.g., ..._gl15.png
    assoc_regex_2 = re.compile(r'_gl(\d+)\.png$')
    # Pattern 3
    assoc_regex_3 = re.compile(r'_gls(\d+)\.png$')

    print(f"🔍 Starting scan in '{root_folder}'...")

    # Get all first-level subdirectories
    first_level_dirs = [d for d in os.listdir(root_folder)
                        if os.path.isdir(os.path.join(root_folder, d))]

    for first_level_dir in first_level_dirs:
        first_level_path = os.path.join(root_folder, first_level_dir)

        # Get all second-level subdirectories (or empty if none exist)
        second_level_dirs = [d for d in os.listdir(first_level_path)
                             if os.path.isdir(os.path.join(first_level_path, d))]

        # If there are second-level subdirectories, process them
        if second_level_dirs:
            for second_level_dir in second_level_dirs:
                second_level_path = os.path.join(first_level_path, second_level_dir)
                process_directory(second_level_path, first_level_dir, second_level_dir,
                                  original_regex, assoc_regex_1, assoc_regex_2, assoc_regex_3, all_pairs)
        else:
            # If no second-level subdirectories, process first-level directory directly
            process_directory(first_level_path, first_level_dir, None,
                              original_regex, assoc_regex_1, assoc_regex_2, assoc_regex_3, all_pairs)

    print(f"✅ Scan complete. Found {len(all_pairs)} matched pairs.")
    df = pd.DataFrame(all_pairs)

    return df


def process_directory(dirpath, first_level_name, second_level_name,
                      original_regex, assoc_regex_1, assoc_regex_2, assoc_regex_3, all_pairs):
    """
    Helper function to process a directory and find matching image pairs.

    Args:
        dirpath: Path to the directory to process
        first_level_name: Name of the first-level folder
        second_level_name: Name of the second-level folder (or None if processing first-level)
        original_regex: Regex pattern for original images
        assoc_regex_1, assoc_regex_2, assoc_regex_3: Regex patterns for associated images
        all_pairs: List to append found pairs to
    """
    originals = {}
    associated = []

    # Get all files in the directory
    filenames = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]

    # First, categorize all files in the current directory
    for filename in filenames:
        # Check if it's an original image
        original_match = original_regex.match(filename)
        if original_match:
            number = int(original_match.group(1))
            originals[number] = filename
            continue

        # Check if it's an associated image using our patterns
        assoc_match = assoc_regex_1.search(filename)
        if not assoc_match:
            assoc_match = assoc_regex_2.search(filename)
        if not assoc_match:
            assoc_match = assoc_regex_3.search(filename)

        if assoc_match:
            number = int(assoc_match.group(1))
            associated.append({'number': number, 'filename': filename})

    # Second, match the categorized originals with their associated counterparts
    for assoc_img in associated:
        ref_num = assoc_img['number']
        if ref_num in originals:
            original_filename = originals[ref_num]
            assoc_filename = assoc_img['filename']

            # Construct the full path from the root
            original_full_path = os.path.join(dirpath, original_filename)
            assoc_full_path = os.path.join(dirpath, assoc_filename)

            # Create impact name based on whether we have second-level folder
            if second_level_name:
                impact_name = f"{first_level_name.replace(' ', '_')}_{second_level_name.replace(' ', '_')}_{ref_num}"
            else:
                impact_name = f"{first_level_name.replace(' ', '_')}_{ref_num}"

            all_pairs.append({
                'Impact': impact_name,
                'original': Path(original_full_path).as_posix(),
                'cropped': Path(assoc_full_path).as_posix()
            })
# ------------------------------------------------------------
# Robust mm/px from ORIGINAL image (uses your detect_scale_mm)
# ------------------------------------------------------------
def compute_scale_mm_per_px(img_path: Path) -> float:
    """
    Detect the red scale bar in an ORIGINAL image and return mm/px.
    Uses HSV + morphology + minAreaRect for a robust pixel length, then
    calls your detect_scale_mm() to decide 10 mm vs 20 mm.
    """
    rgb = _read_rgb(img_path)

    # --- robust red mask in HSV (two hue ranges because red wraps) ---
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # lower/upper bounds: adjust S,V from 80 if your bar is dimmer/brighter
    low1, high1 = (0, 80, 80), (10, 255, 255)
    # lower sensitivity
    low2, high2 = (170, 80, 80), (180, 255, 255)
    # low1, high1 = (0, 70, 50), (10, 255, 255)
    # low2, high2 = (170, 70, 50), (180, 255, 255)
    mask = cv2.inRange(hsv, np.array(low1), np.array(high1)) | \
           cv2.inRange(hsv, np.array(low2), np.array(high2))

    for iters in (2, 1):
        if iters == 1:
            print("Struggle in finding red scale bar, try with lower sensitivity:")
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=iters)
        n_labels, labels = cv2.connectedComponents(mask_clean)
        if n_labels > 1:
            break
    else:
        raise ValueError(f"No red scale bar detected in {img_path}")

    areas = [(labels == i).sum() for i in range(1, n_labels)]
    i_max = 1 + int(np.argmax(areas))
    bar = (labels == i_max)

    # Measure bar length along its major axis using minAreaRect
    ys, xs = np.where(bar)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    long_side_px = max(w, h)
    if long_side_px <= 0:
        raise ValueError(f"Detected scale bar has zero length in {img_path}")

    # Decide 10 vs 20 mm
    bar_mm = detect_scale_mm(str(img_path))
    if bar_mm is None:
        # If your detector fails, assume 20 mm
        warnings.warn(f"[{img_path.name}] detect_scale_mm() failed; assuming 20 mm bar.")
        bar_mm = 20

    return float(bar_mm) / float(long_side_px)

def _largest_component_bbox(mask: np.ndarray):
    """Return (bbox, area_px) for the largest CC. bbox=(min_row, min_col, max_row, max_col)."""
    labeled = label(mask)
    props = regionprops(labeled)
    if not props:
        return None, 0
    largest = max(props, key=lambda p: p.area)
    return largest.bbox, largest.area


# backup function for compute_scale_mm_per_px
def detect_scale_mm_per_px(img_path: Path) -> float:
    """
    Detect the red scale bar in an ORIGINAL image and return mm/px.
    Uses HSV + morphology + minAreaRect for a robust pixel length, then
    calls your detect_scale_mm() to decide 10 mm vs 20 mm.
    """
    rgb = _read_rgb(img_path)

    # --- robust red mask in HSV (two hue ranges because red wraps) ---
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # lower/upper bounds: adjust S,V from 80 if your bar is dimmer/brighter
    low1, high1 = (0, 80, 80), (10, 255, 255)
    low2, high2 = (170, 80, 80), (180, 255, 255)
    mask = cv2.inRange(hsv, np.array(low1), np.array(high1)) | \
           cv2.inRange(hsv, np.array(low2), np.array(high2))

    # Clean and select largest component
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels <= 1:
        raise ValueError(f"No red scale bar detected in {img_path}")

    areas = [(labels == i).sum() for i in range(1, n_labels)]
    max_idx = int(np.argmax(areas)) + 1
    mask_largest = (labels == max_idx).astype(np.uint8)

    y0, x0, y1, x1 = _largest_component_bbox(mask_largest)[0]
    if y0 is None:
        raise ValueError("Could not find a valid bounding box for the red bar.")
    w = x1 - x0

    # Decide 10 vs 20 mm using your original logic
    mm_mark = detect_scale_mm(rgb)
    mm_per_px = mm_mark / float(max(w, 1))
    return mm_per_px

def gl_mask_from_cropped(rgb_crop: np.ndarray) -> np.ndarray:
    """
    Segment granular loss mask (1=loss) from a cropped GL image using
    a simple local-density heuristic.
    """
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

    # Local "granule density": average over LOCAL_DENSITY_K box
    k = LOCAL_DENSITY_K
    kernel = np.ones((k, k), np.float32) / (k * k)
    local_mean = cv2.filter2D(gray, -1, kernel)

    # Loss where local density is below threshold
    loss_mask = (local_mean < (LOCAL_DENSITY_THRESH * np.max(local_mean))).astype(np.uint8)

    # Morphological cleanup
    loss_mask = cv2.morphologyEx(loss_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    loss_mask = cv2.morphologyEx(loss_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return loss_mask

def areas_mm2_from_mask(loss_mask: np.ndarray, mm_per_px: float) -> np.ndarray:
    """
    Convert connected-component areas (in pixels) from the loss mask to mm²,
    using the given mm_per_px scale factor.
    """
    labeled = label(loss_mask)
    props = regionprops(labeled)
    areas_px = np.array([p.area for p in props], dtype=float)
    return (areas_px * (mm_per_px ** 2)) if areas_px.size else np.array([], dtype=float)

def plot_pdf_panel(ax, data: np.ndarray, label_str: str, color: str):
    """
    Plot a histogram (PDF) and an exponential fit on the provided axis.
    """
    if data.size == 0:
        ax.set_title(f"{label_str}\n(no data)")
        ax.set_xlim(0, 1)
        return

    xmax = np.percentile(data, 99.5)
    x = np.linspace(0, xmax, 200)
    scale = np.mean(data) if np.mean(data) > 0 else np.nan

    ax.hist(
        data,
        bins=BINS,
        range=(0, xmax),
        density=True,
        alpha=0.35,
        color=color,
        edgecolor="none",
        label=f"{label_str} hist",
    )
    if np.isfinite(scale) and scale > 0:
        ax.plot(x, stats.expon.pdf(x, loc=0, scale=scale), color=color, lw=2, label=f"{label_str} fit")
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.1 / scale))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Area (mm²)")
    ax.set_ylabel("PDF")
    ax.legend(loc="upper right")

# ------------------------------------------------------------
# Severity mapping (integer 0..3 via percentiles)
# ------------------------------------------------------------
def compute_severity_from_percentiles(data: np.ndarray):
    """
    Map each area to severity level 0..3 using the 25/50/75th percentiles
    of the distribution (per the paper). Returns the integer severity array.
    """
    if data.size == 0:
        return np.array([], dtype=int)

    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    sev = np.zeros_like(data, dtype=int)
    sev[(data >= q25) & (data < q50)] = 1
    sev[(data >= q50) & (data < q75)] = 2
    sev[data >= q75] = 3
    return sev




def process_granule_loss(input_folder, output_folder, igl_cutoff_mm2=2.58, log_callback=None):
    """
    Process granule loss analysis on images in the input folder.

    Args:
        input_folder (str): Path to the folder containing image subdirectories
        output_folder (str): Path to save output files
        igl_cutoff_mm2 (float): Threshold for IGL vs PGL classification (default: 2.58)
        log_callback (callable): Optional callback function for logging messages

    Returns:
        tuple: (summary_df, fig) - DataFrame with results and matplotlib figure
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    image_pairs_df = find_image_pairs(root_folder=input_folder)

    if image_pairs_df.empty:
        raise ValueError(
            f"No matching image pairs were found in '{input_folder}'. "
            "Please check your folder structure and filenames."
        )

    rows = []
    pooled_igl, pooled_pgl, pooled_all = [], [], []

    for index, row_data in image_pairs_df.iterrows():
        impact_name = row_data['Impact']
        p_orig = Path(row_data['original'])
        p_crop = Path(row_data['cropped'])

        if not p_orig.exists() or not p_crop.exists():
            log(f"[{impact_name}] Skipping pair, ... original ('{p_orig}') or cropped ('{p_crop}') file not found.")
            continue

        # 1) compute mm/px from original
        try:
            mm_per_px_orig = compute_scale_mm_per_px(p_orig)
        except Exception as e:
            log(
                f"[{impact_name}] Could not compute mm/px from original: {e}. "
                "Falling back to a crude default assuming a ~20 mm bar spans ~300 px."
            )
            mm_per_px_orig = 20.0 / 300.0

        # 2) load cropped GL image and segment
        rgb_crop = _read_rgb(p_crop)
        loss_mask = gl_mask_from_cropped(rgb_crop)

        # 3) region areas in mm²
        areas = areas_mm2_from_mask(loss_mask, mm_per_px_orig)

        # 4) split by IGL vs PGL threshold
        igl = areas[areas < igl_cutoff_mm2]
        pgl = areas[areas >= igl_cutoff_mm2]

        # 5) per-region severity arrays (kept as-is; not used for totals-based scores)
        sev_igl = compute_severity_from_percentiles(igl)
        sev_pgl = compute_severity_from_percentiles(pgl)

        mean_igl_sev = np.mean(sev_igl) if sev_igl.size else np.nan
        mean_pgl_sev = np.mean(sev_pgl) if sev_pgl.size else np.nan

        # GL score (previously based on per-region means) will be overwritten later
        if np.isnan(mean_igl_sev) and np.isnan(mean_pgl_sev):
            gl_score = np.nan
        else:
            gl_score = (0 if np.isnan(mean_pgl_sev) else (2.0 / 3.0) * mean_pgl_sev) + \
                       (0 if np.isnan(mean_igl_sev) else (1.0 / 3.0) * mean_igl_sev)

        rows.append({
            "Impact": impact_name,
            "Count_IGL": int(igl.size),
            "Count_PGL": int(pgl.size),
            "AreaSum_IGL_mm2": float(np.sum(igl)) if igl.size else 0.0,
            "AreaSum_PGL_mm2": float(np.sum(pgl)) if pgl.size else 0.0,
            "MeanSev_IGL": float(mean_igl_sev) if np.isfinite(mean_igl_sev) else np.nan,
            "MeanSev_PGL": float(mean_pgl_sev) if np.isfinite(mean_pgl_sev) else np.nan,
            "GL_Score": float(gl_score) if np.isfinite(gl_score) else np.nan,
            "GL_Rating": (int(np.clip(np.rint(gl_score), 0, 3)) if np.isfinite(gl_score) else np.nan),
            "CombinedGL_Score": (float(np.mean(np.concatenate([sev_igl, sev_pgl]))) if (sev_igl.size or sev_pgl.size) else np.nan),
            "CombinedGL_Rating": (int(np.clip(np.rint(np.mean(np.concatenate([sev_igl, sev_pgl]))), 0, 3)) if (sev_igl.size or sev_pgl.size) else np.nan),
            "mm_per_px_original": float(mm_per_px_orig)
        })

        pooled_igl.append(igl)
        pooled_pgl.append(pgl)
        pooled_all.append(areas)

        # Quick log
        amin = float(np.min(areas)) if areas.size else float("nan")
        amax = float(np.max(areas)) if areas.size else float("nan")
        log(
            f"  - {impact_name}: regions={areas.size:4d}, min/max={amin:.3f}/{amax:.3f} mm², "
            f"IGL={igl.size}, PGL={pgl.size}, mm/px(orig)={mm_per_px_orig:.5f}"
        )

    if not rows:
        raise ValueError("No impacts processed successfully; nothing to plot.")

    summary_df = pd.DataFrame(rows).sort_values("Impact")
    # ------------------------------------------------------------
    # Percentile-based GL metrics computed from TOTAL areas per impact
    # (q25/q50/q75 computed across all impacts), per user instruction.
    # ------------------------------------------------------------
    pgl_totals = summary_df["AreaSum_PGL_mm2"].to_numpy(dtype=float)
    igl_totals = summary_df["AreaSum_IGL_mm2"].to_numpy(dtype=float)

    def _levels_from_totals(totals: np.ndarray):
        if totals.size == 0:
            return np.array([], dtype=float), (np.nan, np.nan, np.nan)
        q25, q50, q75 = np.percentile(totals, [25, 50, 75])
        if (q25 == q50) and (q50 == q75):
            # Degenerate case: all totals are identical. Assign zeros.
            levels = np.zeros_like(totals, dtype=float)
        else:
            levels = np.zeros_like(totals, dtype=float)
            levels[(totals >= q25) & (totals < q50)] = 1.0
            levels[(totals >= q50) & (totals < q75)] = 2.0
            levels[totals >= q75] = 3.0
        return levels, (q25, q50, q75)

    pgl_levels, _ = _levels_from_totals(pgl_totals)
    igl_levels, _ = _levels_from_totals(igl_totals)

    # GL_Score = 2/3 * PGL_level + 1/3 * IGL_level
    gl_scores = (2.0 / 3.0) * pgl_levels + (1.0 / 3.0) * igl_levels
    summary_df["GL_Score"] = gl_scores
    summary_df["GL_Rating"] = np.clip(np.rint(gl_scores), 0, 3).astype(int)

    # CombinedGL_Score = mean of PGL & IGL levels (both based on dataset-level totals percentiles)
    combined_scores = 0.5 * (pgl_levels + igl_levels)
    summary_df["CombinedGL_Score"] = combined_scores
    summary_df["CombinedGL_Rating"] = np.clip(np.rint(combined_scores), 0, 3).astype(int)

    # ------------------------------------------------------------
    # Plot PDFs (pooled across impacts) — independent axes per panel
    # ------------------------------------------------------------
    x_igl = np.concatenate([x for x in pooled_igl if x.size]) if any(x.size for x in pooled_igl) else np.array([])
    x_pgl = np.concatenate([x for x in pooled_pgl if x.size]) if any(x.size for x in pooled_pgl) else np.array([])
    x_all = np.concatenate([x for x in pooled_all if x.size]) if any(x.size for x in pooled_all) else np.array([])

    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)
    plot_pdf_panel(axs[0], x_igl, f"IGL (< {igl_cutoff_mm2} mm²)", "tab:blue")
    plot_pdf_panel(axs[1], x_pgl, f"PGL (≥ {igl_cutoff_mm2} mm²)", "tab:red")
    plot_pdf_panel(axs[2], x_all, "Combined (All Areas)", "tab:green")
    fig.suptitle("Granule Loss Area Distributions (pooled across impacts)", y=1.05)
    plt.tight_layout()

    # Save outputs
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / "granule_loss_plot.png"
    # fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
    log(f"\nSaved figure to: {fig_path}")

    # ------------------------------------------------------------
    # Output summary table and distributions
    # ------------------------------------------------------------
    if not summary_df.empty:
        log("\nSummary (first 20 rows):")
        log(summary_df.head(20).to_string(index=False))

    csv_path = output_path / "granule_loss_results.csv"
    summary_df.to_csv(str(csv_path), index=False)
    log(f"Saved CSV to: {csv_path}")

    return summary_df, fig


def main():
    """Command-line interface for granule loss analysis."""
    try:
        summary_df, fig = process_granule_loss(
            input_folder=ROOT_DATA_FOLDER,
            output_folder=OUTPUT_FIG,
            igl_cutoff_mm2=IGL_CUTOFF_MM2
        )
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()