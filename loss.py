from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy import stats
from PIL import Image

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CROPPED_SUFFIX = "_cropped"
ANNOTATED_SUFFIX = "_annotated"
GENERATED_IMAGE_SUFFIXES = (CROPPED_SUFFIX, ANNOTATED_SUFFIX, "_cleaned")
IGNORED_OUTPUT_FILENAMES = {"granule_loss_plot.png"}

# Scale-bar length detection
SCALE_BAR_CANDIDATES_MM = (10, 20)  # bar lengths the label reader resolves
DEFAULT_SCALE_BAR_MM = 20           # last-resort assumption; always logged as unverified
# w(first digit) / w("0"), measured on the label's own glyphs so the comparison is
# font-normalised. Measured over 33 system fonts: a "1" never exceeds 0.944 (fonts
# whose "1" carries a foot serif -- Times, Courier, Tahoma, DejaVu -- sit high), and
# a "2" never drops below 0.883. The band between the constants below therefore
# covers the whole overlap, and ratios landing in it are reported as unresolved
# rather than guessed: a silently wrong bar length would scale every area by 4x.
DIGIT_RATIO_ONE_MAX = 0.82
DIGIT_RATIO_TWO_MIN = 0.95

# Annotated (analysed) image rendering
IGL_COLOR = (0, 110, 230)          # RGB, blue: individual granule loss (< cutoff)
PGL_COLOR = (225, 35, 35)          # RGB, red: path granule loss (>= cutoff)
WARN_COLOR = (200, 90, 0)          # RGB, amber: unverified-scale warnings
ANNOTATION_FILL_ALPHA = 0.32       # tint strength over each detected loss region
ANNOTATION_MIN_LABEL_AREA_PX = 10  # regions smaller than this are outlined but not labelled

# -----------------------------
# Utilities
# -----------------------------
def _read_rgb(path: Path) -> np.ndarray:
    """Read image via cv2 and convert BGR->RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        return None

    # Decide 10 vs 20
    if px_threshold is not None:
        mm = 20 if bar_w >= px_threshold else 10
    else:
        # Fraction-of-width threshold (works across images after normalization)
        frac = bar_w / W
        # Empirically: ~0.11 (10 mm) vs ~0.20 (20 mm). Midpoint ~0.16.
        mm = 20 if frac > 0.16 else 10

    return mm

def luminance(rgb: np.ndarray) -> np.ndarray:
    """Return grayscale luminance for an RGB image array."""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def estimate_background(rgb: np.ndarray) -> np.ndarray:
    """Use the median very-bright pixel as fill color; fall back to white."""
    gray = luminance(rgb)
    bright_pixels = rgb[gray >= 245]
    if len(bright_pixels) < 100:
        return np.array([255, 255, 255], dtype=np.uint8)
    return np.median(bright_pixels, axis=0).astype(np.uint8)


def find_red_scale_line(
    rgb: np.ndarray,
    red_min: int = 140,
    red_gap: int = 50,
    search_x_fraction: float = 0.35,
    search_y_fraction: float = 0.50,
) -> tuple[int, int, int, int] | None:
    """
    Find the horizontal red reference line in the lower-right part of the image.

    Returns (x1, y1, x2, y2), or None if no red line is found.
    """
    h, w = rgb.shape[:2]
    x0 = int(w * search_x_fraction)
    y0 = int(h * search_y_fraction)
    crop = rgb[y0:, x0:]

    r = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    b = crop[:, :, 2].astype(np.int16)
    red = (r >= red_min) & ((r - g) >= red_gap) & ((r - b) >= red_gap)

    row_counts = red.sum(axis=1)
    if row_counts.size == 0 or row_counts.max() < 20:
        return None

    row_threshold = max(10, int(row_counts.max() * 0.35))
    candidate_rows = np.flatnonzero(row_counts >= row_threshold)
    if len(candidate_rows) == 0:
        return None

    groups = np.split(candidate_rows, np.where(np.diff(candidate_rows) > 1)[0] + 1)
    best_group = max(groups, key=lambda rows: row_counts[rows].sum())

    line_region = red[best_group.min(): best_group.max() + 1, :]
    xs = np.flatnonzero(line_region.any(axis=0))
    if len(xs) == 0:
        return None

    return (
        x0 + int(xs.min()),
        y0 + int(best_group.min()),
        x0 + int(xs.max()),
        y0 + int(best_group.max()),
    )


def annotation_mask(
    rgb: np.ndarray,
    line_box: tuple[int, int, int, int],
    annotation_dark_threshold: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Build a mask for the red line and dark/gray scale label near the line."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = line_box
    line_len = max(1, x2 - x1 + 1)

    pad_left = int(0.45 * line_len)
    pad_right = int(0.30 * line_len)
    pad_top = max(80, int(0.70 * line_len))
    pad_bottom = max(20, int(0.18 * line_len))

    bx1 = max(0, x1 - pad_left)
    by1 = max(0, y1 - pad_top)
    bx2 = min(w, x2 + pad_right + 1)
    by2 = min(h, y2 + pad_bottom + 1)

    box = rgb[by1:by2, bx1:bx2]
    r = box[:, :, 0].astype(np.int16)
    g = box[:, :, 1].astype(np.int16)
    b = box[:, :, 2].astype(np.int16)
    red = (r >= 120) & ((r - g) >= 40) & ((r - b) >= 40)
    dark_or_gray = luminance(box) <= annotation_dark_threshold

    mask = np.zeros((h, w), dtype=bool)
    mask[by1:by2, bx1:bx2] = red | dark_or_gray
    return mask, (bx1, by1, bx2, by2)


def edge_connected_mask(
    dark: np.ndarray,
    edge_margin: int,
    eight_connected: bool = True,
) -> np.ndarray:
    """Return dark pixels connected to any image edge or edge-margin band."""
    h, w = dark.shape
    edge_margin = max(1, min(edge_margin, h, w))

    seeds = np.zeros_like(dark, dtype=bool)
    seeds[:edge_margin, :] |= dark[:edge_margin, :]
    seeds[-edge_margin:, :] |= dark[-edge_margin:, :]
    seeds[:, :edge_margin] |= dark[:, :edge_margin]
    seeds[:, -edge_margin:] |= dark[:, -edge_margin:]

    remove = np.zeros_like(dark, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    seed_y, seed_x = np.nonzero(seeds)
    for y, x in zip(seed_y, seed_x):
        remove[y, x] = True
        queue.append((int(y), int(x)))

    if eight_connected:
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )
    else:
        neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))

    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dark[ny, nx] and not remove[ny, nx]:
                remove[ny, nx] = True
                queue.append((ny, nx))

    return remove


def clean_shingle_artifacts(
    image_path: Path,
    output_path: Path,
    edge_dark_threshold: int = 180,
    edge_margin: int = 8,
    annotation_dark_threshold: int = 235,
) -> dict[str, object]:
    """
    Generate the cropped/cleaned analysis image beside a scale-bar source image.

    The generated image keeps the original width and height, but removes dark
    border artifacts and the scale annotation so only loss regions remain.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        rgb = np.array(image)

    fill_color = estimate_background(rgb)
    gray = luminance(rgb)

    edge_dark = gray <= edge_dark_threshold
    remove = edge_connected_mask(edge_dark, edge_margin=edge_margin)

    line_box = find_red_scale_line(rgb)
    annotation_box = None
    if line_box is not None:
        annotation_remove, annotation_box = annotation_mask(
            rgb,
            line_box=line_box,
            annotation_dark_threshold=annotation_dark_threshold,
        )
        remove |= annotation_remove

    cleaned = rgb.copy()
    cleaned[remove] = fill_color

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cleaned).save(output_path)

    return {
        "input": str(image_path),
        "output": str(output_path),
        "size": f"{rgb.shape[1]}x{rgb.shape[0]}",
        "removed_pixels": int(remove.sum()),
        "scale_line_found": line_box is not None,
        "annotation_box": annotation_box,
    }


def cropped_output_path(input_path: Path, suffix: str = CROPPED_SUFFIX) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def annotated_output_path(input_path: Path, suffix: str = ANNOTATED_SUFFIX) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def is_generated_crop(path: Path, suffix: str = CROPPED_SUFFIX) -> bool:
    generated_suffixes = tuple(dict.fromkeys((suffix, *GENERATED_IMAGE_SUFFIXES)))
    return path.stem.endswith(generated_suffixes)


def impact_name_from_path(image_path: Path, root_folder: Path) -> str:
    rel = image_path.relative_to(root_folder).with_suffix("").as_posix()
    name = "".join(ch if ch.isalnum() else "_" for ch in rel)
    return "_".join(part for part in name.split("_") if part) or image_path.stem


def generate_image_pairs(
    root_folder: str | Path,
    cropped_suffix: str = CROPPED_SUFFIX,
    log_callback=None,
) -> pd.DataFrame:
    """
    Scan for scale-bar input images, generate their cropped/cleaned partners,
    and return the pair table used by the analysis pipeline.
    """
    root_path = Path(root_folder)
    if not root_path.is_dir():
        raise FileNotFoundError(f"The input folder does not exist: {root_path}")

    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    source_images = [
        path
        for path in sorted(root_path.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name not in IGNORED_OUTPUT_FILENAMES
        and not is_generated_crop(path, cropped_suffix)
    ]

    if not source_images:
        return pd.DataFrame(columns=["Impact", "original", "cropped"])

    log(f"Found {len(source_images)} scale-bar image(s). Generating cropped copies...")

    pairs = []
    for image_path in source_images:
        crop_path = cropped_output_path(image_path, cropped_suffix)
        info = clean_shingle_artifacts(image_path=image_path, output_path=crop_path)
        impact = impact_name_from_path(image_path, root_path)
        log(
            f"  - {image_path.relative_to(root_path)} -> {crop_path.name} | "
            f"removed {info['removed_pixels']} px | scale line found: {info['scale_line_found']}"
        )
        pairs.append({
            "Impact": impact,
            "original": image_path.as_posix(),
            "cropped": crop_path.as_posix(),
        })

    return pd.DataFrame(pairs)


# ------------------------------------------------------------
# Scale bar: find it, measure it, and read its mm label
# ------------------------------------------------------------
def find_scale_bar(rgb: np.ndarray) -> dict | None:
    """
    Locate the red scale bar and measure its length in pixels.

    Returns a dict with the bounding box and the minAreaRect long side, or None
    when no red bar can be found.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Two hue ranges because red wraps around the hue circle.
    mask = cv2.inRange(hsv, np.array((0, 80, 80)), np.array((10, 255, 255))) | \
           cv2.inRange(hsv, np.array((170, 80, 80)), np.array((180, 255, 255)))

    bar = None
    for iters in (2, 1):
        mask_clean = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=iters
        )
        n_labels, labels = cv2.connectedComponents(mask_clean)
        if n_labels > 1:
            areas = [(labels == i).sum() for i in range(1, n_labels)]
            bar = labels == (1 + int(np.argmax(areas)))
            break

    if bar is None or not bar.any():
        return None

    ys, xs = np.nonzero(bar)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    length_px = float(max(w, h))
    if length_px <= 0:
        return None

    return {
        "length_px": length_px,
        "x1": int(xs.min()), "x2": int(xs.max()),
        "y1": int(ys.min()), "y2": int(ys.max()),
    }


def _label_line_bbox(ink: np.ndarray, bar_len: float) -> tuple[int, int, int, int] | None:
    """
    Group ink components into text lines and return the line that most likely
    holds the scale label (text-shaped, close to the bar).
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if area < 12 or h < 0.04 * bar_len or h > 0.35 * bar_len or w > 1.6 * bar_len:
            continue  # noise, loss blobs, image borders
        comps.append((x, y, w, h))
    if not comps:
        return None

    # Greedily group components that share a horizontal band -> one text line.
    comps.sort(key=lambda c: c[1] + c[3] / 2.0)
    median_h = float(np.median([c[3] for c in comps]))
    lines: list[list[tuple[int, int, int, int]]] = []
    for comp in comps:
        center = comp[1] + comp[3] / 2.0
        for line in lines:
            line_center = float(np.mean([c[1] + c[3] / 2.0 for c in line]))
            if abs(center - line_center) <= max(4.0, 0.6 * median_h):
                line.append(comp)
                break
        else:
            lines.append([comp])

    best = None
    for line in lines:
        x1 = min(c[0] for c in line)
        y1 = min(c[1] for c in line)
        x2 = max(c[0] + c[2] for c in line)
        y2 = max(c[1] + c[3] for c in line)
        width, height = x2 - x1, y2 - y1
        if height <= 0 or not (1.2 <= width / height <= 8.0):
            continue  # a scale label is a short wide string
        # Prefer the text line closest to the bottom of the search box (the bar).
        score = y2
        if best is None or score > best[0]:
            best = (score, (x1, y1, x2, y2))
    return best[1] if best else None


def _glyph_box_candidates(ink: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    """
    Decompose a cropped label into per-glyph boxes, left to right.

    Two annotation styles occur in practice: plain dark text, where each glyph is
    its own ink component, and outlined/stroked text, where the glyph bodies touch
    and merge into one component but each body survives as an enclosed hole. Both
    decompositions are returned; the caller keeps whichever one validates.
    """
    height, width = ink.shape
    min_area = max(8, 0.004 * height * width)

    def _clean(boxes):
        if not boxes:
            return []
        tallest = max(b[3] for b in boxes)
        boxes = [b for b in boxes if b[3] >= 0.45 * tallest and b[2] >= 2]

        def nested(box):
            # A glyph counter (the hole in "0") shows up as its own component
            # inside the glyph it belongs to; it is not a glyph of its own.
            return any(
                other is not box
                and other[0] <= box[0] and other[1] <= box[1]
                and other[0] + other[2] >= box[0] + box[2]
                and other[1] + other[3] >= box[1] + box[3]
                for other in boxes
            )

        return sorted((b for b in boxes if not nested(b)), key=lambda b: b[0])

    # Strategy A: ink components (plain text)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    direct = _clean([
        tuple(int(v) for v in stats[i][:4])
        for i in range(1, n) if stats[i][4] >= min_area
    ])

    # Strategy B: enclosed holes (outlined text) -> the glyph bodies
    n, _, stats, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), 4)
    pockets = _clean([
        tuple(int(v) for v in stats[i][:4])
        for i in range(1, n)
        if stats[i][4] >= min_area
        and stats[i][0] > 0 and stats[i][1] > 0
        and stats[i][0] + stats[i][2] < width      # touching the crop border means
        and stats[i][1] + stats[i][3] < height     # background, not a glyph body
    ])

    return [boxes for boxes in (direct, pockets) if len(boxes) >= 4]


def _match_digit_mm_glyphs(glyphs: list[tuple[int, int, int, int]]):
    """
    Find a ``<digit><digit>mm`` run inside a list of glyph boxes.

    Validating the whole four-glyph signature — two cap-height digits followed by
    two matching, wider, x-height ``m`` glyphs — is what keeps the reader from
    accidentally measuring the two ``m`` glyphs against each other (they have equal
    widths, which would always look like a "2").

    Returns (first_digit_width, zero_width, cap_height) or None.
    """
    for i in range(len(glyphs) - 3):
        d1, d2, m1, m2 = glyphs[i:i + 4]
        cap = max(d1[3], d2[3])
        if cap <= 0 or min(d1[3], d2[3]) < 0.88 * cap:
            continue                                        # digits share cap height
        if not all(0.5 * cap <= m[3] <= 0.92 * cap for m in (m1, m2)):
            continue                                        # m sits at x-height
        if abs(m1[3] - m2[3]) > 0.18 * cap:
            continue                                        # the two m's match...
        if abs(m1[2] - m2[2]) > 0.22 * max(m1[2], m2[2]):
            continue
        if min(m1[2], m2[2]) <= 0.9 * d2[2]:
            continue                                        # ...and m is wider than a digit
        gaps = [b[0] - (a[0] + a[2]) for a, b in zip((d1, d2, m1), (d2, m1, m2))]
        if any(gap > 0.9 * cap or gap < -0.25 * cap for gap in gaps):
            continue                                        # one tight run, not stray blobs
        if d2[2] <= 0:
            continue
        return d1[2], d2[2], cap
    return None


def read_scale_label_mm(rgb: np.ndarray, bar: dict) -> tuple[int | None, dict]:
    """
    Read the "10mm"/"20mm" label printed next to the scale bar.

    The two candidates differ only in their first digit, so the first digit's ink
    width is compared against the "0" next to it. Using the label's own glyphs
    keeps the test independent of the font, size and stroke style. Returns
    (mm or None, info) where info explains the decision.
    """
    height, width = rgb.shape[:2]
    bar_len = bar["length_px"]
    info: dict[str, object] = {"reason": "label not found"}

    x1 = max(0, int(bar["x1"] - 0.6 * bar_len))
    x2 = min(width, int(bar["x2"] + 0.6 * bar_len) + 1)
    y1 = max(0, int(bar["y1"] - 1.4 * bar_len))
    y2 = min(height, int(bar["y2"] + 0.4 * bar_len) + 1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, info

    box = rgb[y1:y2, x1:x2]
    gray = luminance(box)
    # Blank the bar rows so the red bar itself is never mistaken for text.
    bar_rows = slice(max(0, bar["y1"] - y1 - 2), max(0, bar["y2"] - y1 + 3))

    for ink_threshold in (120, 160, 90):
        ink = gray <= ink_threshold
        ink[bar_rows, :] = False
        line = _label_line_bbox(ink, bar_len)
        if line is None:
            continue

        lx1, ly1, lx2, ly2 = line
        matched = None
        for glyphs in _glyph_box_candidates(ink[ly1:ly2, lx1:lx2]):
            matched = _match_digit_mm_glyphs(glyphs)
            if matched is not None:
                break
        if matched is None:
            info = {
                "reason": "could not resolve a '<digit><digit>mm' label next to the bar",
                "ink_threshold": ink_threshold,
            }
            continue

        first_w, zero_w, cap = matched
        ratio = first_w / zero_w
        info = {
            "ink_threshold": ink_threshold,
            "digit_width_ratio": round(float(ratio), 3),
            "label_box": (x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2),
            "text_height_px": int(cap),
        }
        if ratio <= DIGIT_RATIO_ONE_MAX:
            info["reason"] = f"first digit is narrow (w/w0={ratio:.2f}) -> '1'"
            return 10, info
        if ratio >= DIGIT_RATIO_TWO_MIN:
            info["reason"] = f"first digit is wide (w/w0={ratio:.2f}) -> '2'"
            return 20, info
        info["reason"] = (
            f"first-digit width ratio {ratio:.2f} falls in the ambiguous band "
            f"({DIGIT_RATIO_ONE_MAX}-{DIGIT_RATIO_TWO_MIN})"
        )
        return None, info

    return None, info


def scale_mm_from_filename(image_path: Path) -> int | None:
    """Read an explicit ``10mm`` / ``20mm`` tag out of the file name, if present."""
    stem = image_path.stem.lower().replace(" ", "")
    for candidate in SCALE_BAR_CANDIDATES_MM:
        if f"{candidate}mm" in stem:
            return int(candidate)
    return None


def measure_scale(
    img_path: Path,
    rgb: np.ndarray | None = None,
    forced_mm: float | None = None,
    log=None,
) -> dict:
    """
    Work out mm/px for an ORIGINAL image and report how the bar length was decided.

    Resolution order: explicit override -> ``10mm``/``20mm`` file-name tag ->
    the printed label next to the bar -> the legacy width-fraction guess ->
    ``DEFAULT_SCALE_BAR_MM``. Only the first three are treated as trustworthy;
    the rest are flagged so a wrong scale cannot pass unnoticed.
    """
    def emit(message):
        if log:
            log(message)

    if rgb is None:
        rgb = _read_rgb(img_path)

    bar = find_scale_bar(rgb)
    if bar is None:
        raise ValueError(f"No red scale bar detected in {img_path}")

    bar_px = bar["length_px"]
    label_info: dict[str, object] = {}

    if forced_mm:
        bar_mm, source, confident = float(forced_mm), "forced", True
    elif (tagged := scale_mm_from_filename(img_path)) is not None:
        bar_mm, source, confident = float(tagged), "filename", True
        emit(f"      scale from file name tag: {tagged} mm")
    else:
        read_mm, label_info = read_scale_label_mm(rgb, bar)
        if read_mm is not None:
            bar_mm, source, confident = float(read_mm), "label", True
            emit(f"      scale read from label: {read_mm} mm ({label_info.get('reason')})")
        else:
            guess = detect_scale_mm(str(img_path))
            if guess is not None:
                bar_mm, source, confident = float(guess), "width-heuristic", False
                emit(
                    f"      WARNING: could not read the scale label "
                    f"({label_info.get('reason')}); fell back to the bar-width guess "
                    f"-> {guess:g} mm. Verify this image, use the scale-bar override, "
                    f"or add a '10mm'/'20mm' tag to the file name."
                )
            else:
                bar_mm, source, confident = float(DEFAULT_SCALE_BAR_MM), "default", False
                emit(
                    f"      WARNING: no scale label and no bar-width guess; assuming "
                    f"{DEFAULT_SCALE_BAR_MM} mm. Areas for this image may be wrong."
                )

    return {
        "mm_per_px": bar_mm / bar_px,
        "bar_mm": bar_mm,
        "bar_px": bar_px,
        "bar_box": (bar["x1"], bar["y1"], bar["x2"], bar["y2"]),
        "source": source,
        "confident": confident,
        "label_info": label_info,
    }


def compute_scale_mm_per_px(img_path: Path, forced_mm: float | None = None) -> float:
    """Detect the red scale bar in an ORIGINAL image and return mm/px."""
    return measure_scale(img_path, forced_mm=forced_mm)["mm_per_px"]

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

def measure_loss_regions(loss_mask: np.ndarray, mm_per_px: float):
    """
    Label the loss mask and measure every connected component.

    Returns (labeled_image, regionprops_list, areas_mm2) so callers can both
    aggregate the areas and draw them back onto the image.
    """
    labeled = label(loss_mask)
    props = regionprops(labeled)
    areas_px = np.array([p.area for p in props], dtype=float)
    areas_mm2 = (areas_px * (mm_per_px ** 2)) if areas_px.size else np.array([], dtype=float)
    return labeled, props, areas_mm2


def areas_mm2_from_mask(loss_mask: np.ndarray, mm_per_px: float) -> np.ndarray:
    """
    Convert connected-component areas (in pixels) from the loss mask to mm²,
    using the given mm_per_px scale factor.
    """
    _, _, areas_mm2 = measure_loss_regions(loss_mask, mm_per_px)
    return areas_mm2


# -----------------------------
# Annotated ("analysed") image rendering
# -----------------------------
class _LabelPlacer:
    """Coarse occupancy grid so area labels do not print on top of each other."""

    def __init__(self, height: int, width: int, cell: int = 4):
        self.height = height
        self.width = width
        self.cell = max(1, cell)
        self.grid = np.zeros((height // self.cell + 1, width // self.cell + 1), dtype=bool)

    def _slices(self, x: int, y: int, w: int, h: int):
        c = self.cell
        return (
            slice(y // c, min(self.grid.shape[0], (y + h) // c + 1)),
            slice(x // c, min(self.grid.shape[1], (x + w) // c + 1)),
        )

    def reserve(self, x: int, y: int, w: int, h: int) -> bool:
        """Claim the box if it is inside the image and still free."""
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            return False
        rows, cols = self._slices(x, y, w, h)
        if self.grid[rows, cols].any():
            return False
        self.grid[rows, cols] = True
        return True


def _format_area(area_mm2: float) -> str:
    if area_mm2 >= 10:
        return f"{area_mm2:.1f}"
    if area_mm2 >= 0.1:
        return f"{area_mm2:.2f}"
    return f"{area_mm2:.3f}"


def _draw_text(img, text, org, font_scale, color, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw text with a white halo so it stays readable over granules."""
    cv2.putText(img, text, org, font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_label_chip(img, text, box, text_baseline, font_scale, color, thickness):
    """
    Draw an area label as a white chip with a coloured border and text.

    A filled chip keeps the number readable over both dark loss regions and
    speckled granule texture.
    """
    x, y, box_w, box_h = box
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), color, max(1, thickness - 1))
    cv2.putText(
        img, text, (x + 2, y + text_baseline), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA,
    )


def _draw_scale_reference(img, scale: dict, font_scale: float, thickness: int) -> None:
    """
    Redraw the scale bar exactly as it was measured, and mark where it was found.

    The bar is drawn at its detected pixel length and captioned with the mm value
    that was used, so the calibration behind every area can be checked against the
    source image at a glance instead of being taken on trust.
    """
    bar_px = float(scale.get("bar_px") or 0)
    bar_mm = float(scale.get("bar_mm") or 0)
    if bar_px <= 0 or bar_mm <= 0:
        return

    h, w = img.shape[:2]
    confident = bool(scale.get("confident", True))
    color = (20, 20, 20) if confident else WARN_COLOR

    # Show where the bar was detected in the source image (it was erased by cleaning).
    box = scale.get("bar_box")
    if box:
        bx1, by1, bx2, by2 = (int(v) for v in box)
        pad = max(3, int(round(w * 0.004)))
        cv2.rectangle(img, (bx1 - pad, by1 - pad), (bx2 + pad, by2 + pad), color, thickness)
        caption_y = by1 - pad - 4
        if caption_y > 12:
            _draw_text(img, "scale bar", (bx1 - pad, caption_y), font_scale * 0.8, color, thickness)

    drawn_px = int(round(min(bar_px, 0.9 * w)))
    bar_h = max(3, int(round(h * 0.006)))
    x1 = int(round(w * 0.02))
    y2 = h - int(round(h * 0.03))
    y1 = y2 - bar_h

    caption = f"{bar_mm:g} mm = {bar_px:.0f} px ({scale.get('source', 'unknown')})"
    if not confident:
        caption += " - UNVERIFIED"
    if drawn_px < bar_px:
        caption += " [bar shown truncated]"
    (caption_w, caption_h), _ = cv2.getTextSize(
        caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # One white plate behind both the bar and its caption, so neither is lost in the image.
    plate_x2 = min(w - 1, x1 + max(drawn_px, caption_w) + 4)
    cv2.rectangle(img, (x1 - 4, y1 - caption_h - 12), (plate_x2, y2 + 4), (255, 255, 255), -1)
    cv2.rectangle(img, (x1, y1), (x1 + drawn_px, y2), color, -1)
    _draw_text(img, caption, (x1, y1 - 8), font_scale, color, thickness)


def _draw_banner(img, lines, font_scale: float, thickness: int) -> np.ndarray:
    """
    Prepend a white header strip carrying the per-image totals.

    Each entry is either a string or a (text, colour) pair, so warnings can be
    called out in a different colour from the ordinary readings.
    """
    if not lines:
        return img

    line_h = int(round(28 * font_scale / 0.6))
    pad = max(6, line_h // 3)
    banner_h = pad * 2 + line_h * len(lines)
    banner = np.full((banner_h, img.shape[1], 3), 255, dtype=np.uint8)

    y = pad + int(line_h * 0.75)
    for index, entry in enumerate(lines):
        line, color = entry if isinstance(entry, tuple) else (
            entry, (20, 20, 20) if index == 0 else (60, 60, 60)
        )
        cv2.putText(
            banner, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, thickness, cv2.LINE_AA,
        )
        y += line_h

    cv2.line(banner, (0, banner_h - 1), (img.shape[1], banner_h - 1), (200, 200, 200), 1)
    return np.vstack([banner, img])


def render_loss_annotation(
    rgb: np.ndarray,
    labeled: np.ndarray,
    props,
    areas_mm2: np.ndarray,
    scale: dict,
    igl_cutoff_mm2: float,
    title: str = "",
) -> tuple[np.ndarray, list[dict], int]:
    """
    Draw every detected loss region on a copy of the analysed image.

    Each region is tinted and outlined (blue = IGL, red = PGL) and labelled with
    its area in mm² whenever a label fits without overlapping another one.

    Returns (annotated_rgb, region_records, labelled_count).
    """
    h, w = rgb.shape[:2]
    font_scale = max(0.34, min(1.1, w / 1700.0))
    thickness = max(1, int(round(w / 1400.0)))
    outline_thickness = max(1, int(round(w / 1200.0)))

    is_pgl = areas_mm2 >= igl_cutoff_mm2 if areas_mm2.size else np.array([], dtype=bool)

    # Tint the two classes, then outline them.
    canvas = rgb.astype(np.float32)
    for selector, color in ((~is_pgl, IGL_COLOR), (is_pgl, PGL_COLOR)):
        if not selector.any():
            continue
        ids = [prop.label for prop, keep in zip(props, selector) if keep]
        class_mask = np.isin(labeled, ids)
        tint = np.array(color, dtype=np.float32)
        canvas[class_mask] = (
            (1.0 - ANNOTATION_FILL_ALPHA) * canvas[class_mask] + ANNOTATION_FILL_ALPHA * tint
        )
    annotated = canvas.astype(np.uint8)

    for selector, color in ((~is_pgl, IGL_COLOR), (is_pgl, PGL_COLOR)):
        if not selector.any():
            continue
        ids = [prop.label for prop, keep in zip(props, selector) if keep]
        class_mask = np.isin(labeled, ids).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, outline_thickness, cv2.LINE_AA)

    # Label the largest regions first so the important numbers always land.
    placer = _LabelPlacer(h, w, cell=max(2, int(round(w / 400.0))))
    order = np.argsort(-areas_mm2) if areas_mm2.size else np.array([], dtype=int)

    records: list[dict] = []
    labelled = 0
    for position in order:
        prop = props[position]
        area_mm2 = float(areas_mm2[position])
        pgl = bool(is_pgl[position])
        color = PGL_COLOR if pgl else IGL_COLOR
        text = _format_area(area_mm2)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        box_w, box_h = text_w + 4, text_h + baseline + 4

        min_row, min_col, max_row, max_col = prop.bbox
        centroid_y, centroid_x = prop.centroid
        inside = (
            (int(centroid_x - box_w / 2), int(centroid_y - box_h / 2)),
        ) if (max_col - min_col >= box_w + 4 and max_row - min_row >= box_h + 4) else ()
        candidates = (
            *inside,                                                      # only when it fits
            (int(centroid_x - box_w / 2), int(min_row - box_h - 3)),       # above
            (int(centroid_x - box_w / 2), int(max_row + 3)),               # below
            (int(max_col + 4), int(centroid_y - box_h / 2)),               # right
            (int(min_col - box_w - 4), int(centroid_y - box_h / 2)),       # left
        )

        placed = False
        if prop.area >= ANNOTATION_MIN_LABEL_AREA_PX:
            for x, y in candidates:
                if placer.reserve(x, y, box_w, box_h):
                    _draw_label_chip(
                        annotated, text, (x, y, box_w, box_h), text_h + 2,
                        font_scale, color, thickness,
                    )
                    placed = True
                    labelled += 1
                    break

        records.append({
            "Region_ID": int(prop.label),
            "Class": "PGL" if pgl else "IGL",
            "Area_px": int(prop.area),
            "Area_mm2": area_mm2,
            "Centroid_X": float(centroid_x),
            "Centroid_Y": float(centroid_y),
            "Labelled_On_Image": placed,
        })

    _draw_scale_reference(annotated, scale, font_scale, thickness)

    igl_areas = areas_mm2[~is_pgl] if areas_mm2.size else np.array([])
    pgl_areas = areas_mm2[is_pgl] if areas_mm2.size else np.array([])
    mm_per_px = float(scale.get("mm_per_px") or 0.0)
    lines = [
        f"{title}  |  regions: {areas_mm2.size}  |  cutoff: {igl_cutoff_mm2:g} mm2",
        f"IGL (< {igl_cutoff_mm2:g} mm2, blue): n={igl_areas.size}, "
        f"sum={float(np.sum(igl_areas)) if igl_areas.size else 0.0:.3f} mm2   "
        f"PGL (>= {igl_cutoff_mm2:g} mm2, red): n={pgl_areas.size}, "
        f"sum={float(np.sum(pgl_areas)) if pgl_areas.size else 0.0:.3f} mm2",
        f"Total loss area: {float(np.sum(areas_mm2)) if areas_mm2.size else 0.0:.3f} mm2   "
        f"labelled {labelled}/{areas_mm2.size} regions "
        f"(unlabelled ones are outlined; all areas are in granule_loss_regions.csv)",
        f"Scale: {float(scale.get('bar_mm') or 0):g} mm bar = "
        f"{float(scale.get('bar_px') or 0):.0f} px -> {mm_per_px:.5f} mm/px "
        f"(from {scale.get('source', 'unknown')})",
    ]
    if not scale.get("confident", True):
        lines.append((
            "WARNING: the bar length could not be read from the image - "
            "areas are only correct if this bar length is right.",
            WARN_COLOR,
        ))
    annotated = _draw_banner(annotated, lines, font_scale, thickness)

    # Records are sorted largest-first above; keep them in label order on disk.
    records.sort(key=lambda record: record["Region_ID"])
    return annotated, records, labelled


def save_loss_annotation(
    rgb: np.ndarray,
    labeled: np.ndarray,
    props,
    areas_mm2: np.ndarray,
    scale: dict,
    igl_cutoff_mm2: float,
    output_path: Path,
    title: str = "",
) -> tuple[list[dict], int]:
    """Render and write the annotated analysis image beside the source image."""
    annotated, records, labelled = render_loss_annotation(
        rgb=rgb,
        labeled=labeled,
        props=props,
        areas_mm2=areas_mm2,
        scale=scale,
        igl_cutoff_mm2=igl_cutoff_mm2,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(output_path)
    return records, labelled

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




def process_granule_loss(
    input_folder,
    output_folder,
    igl_cutoff_mm2=2.58,
    log_callback=None,
    forced_scale_mm=None,
):
    """
    Process granule loss analysis on images in the input folder.

    Args:
        input_folder (str): Path to the folder containing image subdirectories
        output_folder (str): Path to save output files
        igl_cutoff_mm2 (float): Threshold for IGL vs PGL classification (default: 2.58)
        log_callback (callable): Optional callback function for logging messages
        forced_scale_mm (float): Scale-bar length in mm to use for every image.
            None (default) auto-detects per image: a ``10mm``/``20mm`` file-name tag
            wins, otherwise the label printed next to the bar is read.

    Returns:
        tuple: (summary_df, fig) - DataFrame with results and matplotlib figure
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    image_pairs_df = generate_image_pairs(
        root_folder=input_folder,
        cropped_suffix=CROPPED_SUFFIX,
        log_callback=log,
    )

    if image_pairs_df.empty:
        raise ValueError(
            f"No supported scale-bar images were found in '{input_folder}'. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}."
        )

    rows = []
    region_rows = []
    unverified_scales = []
    pooled_igl, pooled_pgl, pooled_all = [], [], []

    for index, row_data in image_pairs_df.iterrows():
        impact_name = row_data['Impact']
        p_orig = Path(row_data['original'])
        p_crop = Path(row_data['cropped'])

        if not p_orig.exists() or not p_crop.exists():
            log(f"[{impact_name}] Skipping pair, ... original ('{p_orig}') or cropped ('{p_crop}') file not found.")
            continue

        # 1) compute mm/px from original, and record how the bar length was decided
        try:
            scale = measure_scale(p_orig, forced_mm=forced_scale_mm, log=log)
        except Exception as e:
            log(
                f"[{impact_name}] WARNING: could not measure the scale bar: {e}. "
                "Falling back to a crude default assuming a ~20 mm bar spans ~300 px; "
                "areas for this image are not trustworthy."
            )
            scale = {
                "mm_per_px": 20.0 / 300.0, "bar_mm": 20.0, "bar_px": 300.0,
                "bar_box": None, "source": "no-bar-found", "confident": False,
                "label_info": {},
            }
        mm_per_px_orig = scale["mm_per_px"]
        if not scale["confident"]:
            unverified_scales.append(impact_name)

        # 2) load cropped GL image and segment
        rgb_crop = _read_rgb(p_crop)
        loss_mask = gl_mask_from_cropped(rgb_crop)

        # 3) region areas in mm²
        labeled, props, areas = measure_loss_regions(loss_mask, mm_per_px_orig)

        # 4) split by IGL vs PGL threshold
        igl = areas[areas < igl_cutoff_mm2]
        pgl = areas[areas >= igl_cutoff_mm2]

        # 4b) annotated ("analysed") image beside the source, one per input image
        annotated_path = annotated_output_path(p_orig)
        try:
            records, labelled_count = save_loss_annotation(
                rgb=rgb_crop,
                labeled=labeled,
                props=props,
                areas_mm2=areas,
                scale=scale,
                igl_cutoff_mm2=igl_cutoff_mm2,
                output_path=annotated_path,
                title=impact_name,
            )
            for record in records:
                region_rows.append({"Impact": impact_name, **record})
            log(
                f"  - {impact_name}: annotated -> {annotated_path.name} | "
                f"labelled {labelled_count}/{areas.size} region(s)"
            )
        except Exception as e:
            log(f"[{impact_name}] Could not write annotated image: {e}")
            annotated_path = None

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
            "Original_Image": str(p_orig),
            "Cropped_Image": str(p_crop),
            "Annotated_Image": str(annotated_path) if annotated_path is not None else "",
            "Count_IGL": int(igl.size),
            "Count_PGL": int(pgl.size),
            "Count_All": int(areas.size),
            "AreaSum_IGL_mm2": float(np.sum(igl)) if igl.size else 0.0,
            "AreaSum_PGL_mm2": float(np.sum(pgl)) if pgl.size else 0.0,
            "AreaSum_All_mm2": float(np.sum(areas)) if areas.size else 0.0,
            "MeanSev_IGL": float(mean_igl_sev) if np.isfinite(mean_igl_sev) else np.nan,
            "MeanSev_PGL": float(mean_pgl_sev) if np.isfinite(mean_pgl_sev) else np.nan,
            "GL_Score": float(gl_score) if np.isfinite(gl_score) else np.nan,
            "GL_Rating": (int(np.clip(np.rint(gl_score), 0, 3)) if np.isfinite(gl_score) else np.nan),
            "CombinedGL_Score": (float(np.mean(np.concatenate([sev_igl, sev_pgl]))) if (sev_igl.size or sev_pgl.size) else np.nan),
            "CombinedGL_Rating": (int(np.clip(np.rint(np.mean(np.concatenate([sev_igl, sev_pgl]))), 0, 3)) if (sev_igl.size or sev_pgl.size) else np.nan),
            "mm_per_px_original": float(mm_per_px_orig),
            "ScaleBar_mm": float(scale["bar_mm"]),
            "ScaleBar_px": float(scale["bar_px"]),
            "Scale_Source": scale["source"],
            "Scale_Verified": bool(scale["confident"]),
        })

        pooled_igl.append(igl)
        pooled_pgl.append(pgl)
        pooled_all.append(areas)

        # Quick log
        amin = float(np.min(areas)) if areas.size else float("nan")
        amax = float(np.max(areas)) if areas.size else float("nan")
        log(
            f"  - {impact_name}: regions={areas.size:4d}, min/max={amin:.3f}/{amax:.3f} mm², "
            f"IGL={igl.size}, PGL={pgl.size}, mm/px(orig)={mm_per_px_orig:.5f}, "
            f"scale={scale['bar_mm']:g} mm / {scale['bar_px']:.0f} px ({scale['source']})"
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

    # fig_path = output_path / "granule_loss_plot.png"
    # fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
    # log(f"\nSaved figure to: {fig_path}")

    # ------------------------------------------------------------
    # Output summary table and distributions
    # ------------------------------------------------------------
    if not summary_df.empty:
        log("\nSummary (first 20 rows):")
        log(summary_df.head(20).to_string(index=False))

    csv_path = output_path / "granule_loss_results.csv"
    summary_df.to_csv(str(csv_path), index=False)
    log(f"Saved CSV to: {csv_path}")

    # Every measured region (incl. the ones too small to label on the image)
    regions_df = pd.DataFrame(
        region_rows,
        columns=[
            "Impact", "Region_ID", "Class", "Area_px", "Area_mm2",
            "Centroid_X", "Centroid_Y", "Labelled_On_Image",
        ],
    )
    regions_csv_path = output_path / "granule_loss_regions.csv"
    regions_df.to_csv(str(regions_csv_path), index=False)
    log(f"Saved per-region CSV to: {regions_csv_path}")
    log(f"Annotated images saved beside each source image as *{ANNOTATED_SUFFIX}.*")

    if unverified_scales:
        log(
            f"\nWARNING: the scale bar length could not be verified for "
            f"{len(unverified_scales)} of {len(rows)} image(s): "
            f"{', '.join(unverified_scales[:12])}"
            f"{' ...' if len(unverified_scales) > 12 else ''}"
        )
        log(
            "  Areas scale with the square of mm/px, so a 10 mm bar read as 20 mm "
            "makes every area 4x too large. Fix these by setting the scale-bar length "
            "explicitly, or by adding a '10mm'/'20mm' tag to the file name. "
            "Check the 'Scale_Verified' column and the header of each annotated image."
        )

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
