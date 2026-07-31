"""
Gradio web GUI for IBHS granule loss analysis (single image).

The desktop Tkinter app (``app.py``) and the processing pipeline (``loss.py``) are
untouched: this module only stages one uploaded image into a temporary folder and
calls the same :func:`loss.process_granule_loss` used by the desktop version, then
renders the input, the annotated output and the reported metrics side by side.

Run locally:

    python gradio_app.py

Deploy: see SPACE_README.md (Hugging Face Space entry point / requirements).
"""

from __future__ import annotations

import queue
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless server: never touch a GUI backend
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

from loss import IMAGE_EXTENSIONS, process_granule_loss

SCALE_AUTO = "Auto-detect"
SCALE_CHOICES = (SCALE_AUTO, "10 mm", "20 mm")
DEFAULT_THRESHOLD = 2.58

# The web version processes one image per run, so nothing here should outlive a
# request by much. Uploads bigger than this are rejected before any work starts.
MAX_UPLOAD = "100mb"
LOG_FLUSH_SECONDS = 0.25

# ``analyze`` yields (log, <result components>, run_dir). Intermediate yields refresh
# only the log, so they skip this many components in between.
RESULT_COMPONENTS = 9

# Percentile-based ratings rank an impact against the other impacts in the same
# run. With a single image the 25/50/75th percentiles collapse onto one value, so
# these columns carry no information here and are hidden from the results rather
# than shown as a severity that could be misread. They stay in the downloaded
# granule_loss_results.csv, which is the pipeline's own output.
HIDDEN_COLUMNS = (
    "GL_Score",
    "GL_Rating",
    "CombinedGL_Score",
    "CombinedGL_Rating",
    "MeanSev_IGL",
    "MeanSev_PGL",
)

METRIC_ROWS = (
    ("Count_IGL", "IGL regions (count)"),
    ("Count_PGL", "PGL regions (count)"),
    ("Count_All", "All loss regions (count)"),
    ("AreaSum_IGL_mm2", "IGL area sum (mm2)"),
    ("AreaSum_PGL_mm2", "PGL area sum (mm2)"),
    ("AreaSum_All_mm2", "Total loss area (mm2)"),
    ("ScaleBar_mm", "Scale bar length (mm)"),
    ("ScaleBar_px", "Scale bar length (px)"),
    ("mm_per_px_original", "mm per pixel"),
    ("Scale_Source", "Scale resolved from"),
    ("Scale_Verified", "Scale verified"),
)


def _safe_stem(name: str) -> str:
    """
    Keep the parts of an uploaded file name the pipeline actually reads.

    ``loss.scale_mm_from_filename`` looks for a ``10mm``/``20mm`` tag in the stem,
    so the stem is preserved rather than replaced by a random name; everything that
    is not a plain name character is dropped so the staged path stays predictable.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._-")
    return stem or "uploaded_image"


def _scale_mm(choice: str) -> float | None:
    """Return the forced bar length in mm, or None when auto-detecting."""
    if not choice or choice == SCALE_AUTO:
        return None
    return float(choice.split()[0])


def _validate_threshold(value) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise gr.Error("Enter a valid number for the IGL/PGL threshold.")
    if not threshold > 0:
        raise gr.Error("The IGL/PGL threshold must be greater than 0 mm2.")
    return threshold


def _discard(run_dir) -> None:
    """Remove a previous run's temp folder; a failure here must not break the run."""
    if not run_dir:
        return
    shutil.rmtree(str(run_dir), ignore_errors=True)


def _metrics_table(row: pd.Series) -> pd.DataFrame:
    records = []
    for column, label in METRIC_ROWS:
        if column not in row:
            continue
        value = row[column]
        if isinstance(value, float):
            value = f"{value:.5f}" if column == "mm_per_px_original" else f"{value:.3f}"
        records.append({"Metric": label, "Value": str(value)})
    return pd.DataFrame(records, columns=["Metric", "Value"])


def _headline(row: pd.Series, threshold: float) -> str:
    verified = bool(row.get("Scale_Verified", False))
    bar_mm = float(row.get("ScaleBar_mm", 0.0))
    bar_px = float(row.get("ScaleBar_px", 0.0))
    source = row.get("Scale_Source", "unknown")

    scale_note = (
        f"Scale: **{bar_mm:g} mm = {bar_px:.0f} px** (from {source})"
        if verified
        else (
            f"### ⚠️ Unverified scale\n"
            f"The bar length could not be read from the image; **{bar_mm:g} mm** was assumed "
            f"(from {source}). Areas scale with the *square* of mm/px, so a 10 mm bar read as "
            f"20 mm makes every area 4x too large. Set **Scale bar length** explicitly and "
            f"re-run, or name the file with a `10mm` / `20mm` tag.\n\n"
            f"Scale used: **{bar_mm:g} mm = {bar_px:.0f} px**"
        )
    )

    return "\n\n".join([
        f"### Results — {row.get('Impact', 'image')}",
        (
            f"**{int(row.get('Count_All', 0))}** loss regions · "
            f"total **{float(row.get('AreaSum_All_mm2', 0.0)):.3f} mm²**  \n"
            f"IGL (< {threshold:g} mm², blue): **{int(row.get('Count_IGL', 0))}** regions, "
            f"**{float(row.get('AreaSum_IGL_mm2', 0.0)):.3f} mm²**  \n"
            f"PGL (≥ {threshold:g} mm², red): **{int(row.get('Count_PGL', 0))}** regions, "
            f"**{float(row.get('AreaSum_PGL_mm2', 0.0)):.3f} mm²**"
        ),
        scale_note,
    ])


def _regions_table(regions_csv: Path) -> pd.DataFrame:
    columns = ["Region_ID", "Class", "Area_px", "Area_mm2", "Centroid_X", "Centroid_Y"]
    if not regions_csv.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(regions_csv)
    keep = [column for column in columns if column in df.columns]
    df = df[keep].copy()
    if "Area_mm2" in df.columns:
        df = df.sort_values("Area_mm2", ascending=False)
        df["Area_mm2"] = df["Area_mm2"].round(4)
    for column in ("Centroid_X", "Centroid_Y"):
        if column in df.columns:
            df[column] = df[column].round(1)
    return df.reset_index(drop=True)


def _full_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary as Field/Value pairs, minus the columns single-image runs cannot fill."""
    skip = {"Original_Image", "Cropped_Image", "Annotated_Image", *HIDDEN_COLUMNS}
    row = summary_df.iloc[0]
    records = []
    for column in summary_df.columns:
        if column in skip:
            # image paths point at server-side temp files; HIDDEN_COLUMNS are the
            # percentile ratings, which are meaningless for one image.
            continue
        value = row[column]
        if isinstance(value, float):
            value = f"{value:.5f}" if column == "mm_per_px_original" else f"{value:.4f}"
        records.append({"Field": column, "Value": str(value)})
    return pd.DataFrame(records, columns=["Field", "Value"])


def analyze(image_path, threshold_value, scale_choice, previous_run_dir):
    """
    Run the desktop pipeline over a single uploaded image, streaming its log.

    Yields the full output tuple; intermediate yields only refresh the log and
    leave every other component untouched via ``gr.skip()``.
    """
    if not image_path:
        raise gr.Error("Upload a scale-bar image first.")

    threshold = _validate_threshold(threshold_value)
    forced_scale_mm = _scale_mm(scale_choice)

    source = Path(image_path)
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise gr.Error(
            f"Unsupported file type '{source.suffix}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}."
        )

    _discard(previous_run_dir)

    run_dir = Path(tempfile.mkdtemp(prefix="granule_web_"))
    input_dir = run_dir / "input"
    output_dir = run_dir / "results"
    input_dir.mkdir(parents=True, exist_ok=True)

    staged = input_dir / f"{_safe_stem(source.name)}{source.suffix.lower()}"
    shutil.copy2(source, staged)

    messages: "queue.Queue[str | None]" = queue.Queue()
    outcome: dict[str, object] = {}

    def log_callback(message):
        messages.put(str(message))

    def worker():
        try:
            summary_df, fig = process_granule_loss(
                input_folder=str(input_dir),
                output_folder=str(output_dir),
                igl_cutoff_mm2=threshold,
                log_callback=log_callback,
                forced_scale_mm=forced_scale_mm,
            )
            outcome["summary_df"] = summary_df
            outcome["fig"] = fig
        except Exception as error:  # surfaced to the user below
            outcome["error"] = error
        finally:
            messages.put(None)

    header = [
        "=" * 70,
        "Starting Granule Loss Analysis (single image)...",
        "=" * 70,
        f"Image: {source.name}",
        f"IGL/PGL threshold: {threshold} mm2",
        (
            f"Scale bar: forced to {forced_scale_mm:g} mm"
            if forced_scale_mm
            else "Scale bar: auto-detected (file-name tag, then printed label)"
        ),
        "",
    ]
    lines = list(header)
    skips = [gr.skip()] * RESULT_COMPONENTS
    yield ("\n".join(lines), *skips, str(run_dir))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_flush = time.monotonic()
    while True:
        message = messages.get()
        if message is None:
            break
        lines.append(message)
        now = time.monotonic()
        if now - last_flush >= LOG_FLUSH_SECONDS:
            last_flush = now
            yield ("\n".join(lines), *skips, gr.skip())
    thread.join()

    if "error" in outcome:
        lines.append(f"\nERROR: {outcome['error']}")
        yield ("\n".join(lines), *skips, str(run_dir))
        raise gr.Error(f"Analysis failed: {outcome['error']}")

    summary_df: pd.DataFrame = outcome["summary_df"]  # type: ignore[assignment]
    fig = outcome["fig"]

    # Render the pooled distribution panel to a file and close the figure, so a
    # long-lived server does not accumulate matplotlib state.
    plot_path = output_dir / "granule_loss_distributions.png"
    try:
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    except Exception as error:
        lines.append(f"Could not render the distribution plot: {error}")
        plot_path = None
    finally:
        plt.close(fig)

    row = summary_df.iloc[0]
    annotated = row.get("Annotated_Image") or ""
    cropped = row.get("Cropped_Image") or ""
    annotated_path = Path(annotated) if annotated else None
    cropped_path = Path(cropped) if cropped else None

    if annotated_path is None or not annotated_path.exists():
        lines.append("\nWARNING: the annotated image could not be written for this image.")

    downloads = [
        path
        for path in (
            annotated_path,
            cropped_path,
            output_dir / "granule_loss_results.csv",
            output_dir / "granule_loss_regions.csv",
            plot_path,
        )
        if path is not None and path.exists()
    ]

    lines.extend(["", "=" * 70, "Analysis completed.", "=" * 70])

    # The annotated image is shown twice: next to the input, and full width in its
    # own tab. Both components take the same file.
    annotated_value = (
        str(annotated_path) if annotated_path and annotated_path.exists() else None
    )

    yield (
        "\n".join(lines),
        _headline(row, threshold),
        annotated_value,
        annotated_value,
        str(cropped_path) if cropped_path and cropped_path.exists() else None,
        _metrics_table(row),
        _regions_table(output_dir / "granule_loss_regions.csv"),
        _full_summary_table(summary_df),
        str(plot_path) if plot_path is not None and plot_path.exists() else None,
        [str(path) for path in downloads],
        str(run_dir),
    )


def reset(previous_run_dir):
    """Clear the screen and drop the temp folder from the last run."""
    _discard(previous_run_dir)
    return (
        None,                                       # input image
        DEFAULT_THRESHOLD,
        SCALE_AUTO,
        "",                                         # log
        "",                                         # headline
        None, None,                                 # annotated (side by side / tab)
        None,                                       # cropped
        pd.DataFrame(columns=["Metric", "Value"]),
        pd.DataFrame(columns=["Region_ID", "Class", "Area_px", "Area_mm2"]),
        pd.DataFrame(columns=["Field", "Value"]),
        None,                                       # distribution plot
        None,                                       # downloads
        None,                                       # run dir state
    )


INTRO = """
# IBHS Granule Loss Analysis

Upload one microscope shingle image **that includes the red scale bar**. The image is
cleaned, the scale bar is measured, granule-loss regions are segmented and every region
is reported in mm² — the same pipeline as the IBHS desktop tool, run on a single image.

Blue = **IGL** (area below the threshold) · Red = **PGL** (area at or above it).
"""

SCALE_HELP = """
**Auto-detect** reads the `10mm`/`20mm` label printed next to the bar; a `10mm`/`20mm` tag
in the file name overrides it. Areas scale with the *square* of mm/px, so if the label
cannot be read the result is flagged **unverified** — set the length here and re-run.
"""

NOTES = """
**Reading the numbers**

* Areas come from the generated cropped image; regions the local-density segmenter misses
  are neither drawn nor counted.
* Every detected region is in the per-region table and CSV, including regions too small or
  too crowded to carry a label on the image.
* Severity ratings are not reported here: they rank an impact against the *other impacts in
  the same run*, which needs more than one image. Run the desktop app over a folder of
  impacts to get them.
"""


def build_demo() -> gr.Blocks:
    # Gradio 6 takes the theme in launch(), not in the Blocks constructor.
    with gr.Blocks(title="IBHS Granule Loss Analysis") as demo:
        run_dir_state = gr.State(None)

        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Input image (with red scale bar)",
                    type="filepath",
                    image_mode=None,       # pass the original file through untouched
                    sources=["upload", "clipboard"],
                    height=420,
                )
                threshold_input = gr.Number(
                    label="IGL / PGL threshold (mm²)",
                    value=DEFAULT_THRESHOLD,
                    minimum=0.0001,
                    step=0.01,
                )
                scale_input = gr.Radio(
                    label="Scale bar length",
                    choices=list(SCALE_CHOICES),
                    value=SCALE_AUTO,
                )
                gr.Markdown(SCALE_HELP)
                with gr.Row():
                    run_button = gr.Button("Analyze image", variant="primary")
                    clear_button = gr.Button("Clear")

            with gr.Column(scale=1):
                annotated_output = gr.Image(
                    label="Annotated result (blue = IGL, red = PGL)",
                    type="filepath",
                    format="png",
                    height=420,
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )
                headline_output = gr.Markdown()

        with gr.Row():
            metrics_output = gr.Dataframe(
                label="Key metrics",
                headers=["Metric", "Value"],
                wrap=True,
                interactive=False,
            )
            download_output = gr.File(
                label="Downloads (annotated image, cropped image, CSVs, plot)",
                interactive=False,
            )

        with gr.Tabs():
            with gr.Tab("Annotated image"):
                annotated_tab_output = gr.Image(
                    label="Annotated result at full width "
                          "(blue = IGL, red = PGL) — same image as above",
                    type="filepath",
                    format="png",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )
            with gr.Tab("Per-region areas"):
                regions_output = gr.Dataframe(
                    label="Every detected loss region, largest first",
                    interactive=False,
                    wrap=True,
                    max_height=420,
                )
            with gr.Tab("Cropped image"):
                cropped_output = gr.Image(
                    label="Cleaned image the measurements were taken from",
                    type="filepath",
                    format="png",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )
            with gr.Tab("Area distributions", visible=False):
                gr.Markdown(
                    "Histograms and exponential fits of the region areas "
                    "(IGL / PGL / combined) for this image."
                )
                plot_output = gr.Image(
                    label="Granule loss area distributions",
                    type="filepath",
                    format="png",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )
            with gr.Tab("Full summary row"):
                summary_output = gr.Dataframe(
                    label="granule_loss_results.csv for this image "
                          "(percentile ratings omitted — they need multiple impacts)",
                    interactive=False,
                    wrap=True,
                    max_height=420,
                )
            with gr.Tab("Log", visible=False):
                log_output = gr.Textbox(
                    label="Processing log",
                    lines=22,
                    max_lines=22,
                    buttons=["copy"],
                    autoscroll=True,
                )

        gr.Markdown(NOTES)

        outputs = [
            log_output,
            headline_output,
            annotated_output,
            annotated_tab_output,
            cropped_output,
            metrics_output,
            regions_output,
            summary_output,
            plot_output,
            download_output,
            run_dir_state,
        ]
        assert len(outputs) == RESULT_COMPONENTS + 2  # log + results + state

        run_event = run_button.click(
            fn=analyze,
            inputs=[image_input, threshold_input, scale_input, run_dir_state],
            outputs=outputs,
            concurrency_limit=1,   # the pipeline is CPU-bound; queue instead of thrash
            show_progress="full",
            api_name="analyze",
        )
        image_input.upload(lambda: "", outputs=headline_output)

        clear_button.click(
            fn=reset,
            inputs=[run_dir_state],
            outputs=[
                image_input,
                threshold_input,
                scale_input,
                log_output,
                headline_output,
                annotated_output,
                annotated_tab_output,
                cropped_output,
                metrics_output,
                regions_output,
                summary_output,
                plot_output,
                download_output,
                run_dir_state,
            ],
            cancels=[run_event],
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(max_size=16).launch(
        theme=gr.themes.Cyberpunk(),
        max_file_size=MAX_UPLOAD,
    )
