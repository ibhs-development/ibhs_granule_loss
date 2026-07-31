# Deploying the web GUI to a Hugging Face Space

The web GUI is `gradio_app.py`. It reuses `loss.py` unchanged and analyzes **one uploaded
image per run**; the Tkinter desktop app (`app.py`) is untouched and keeps its folder-batch
workflow.

## Run it locally first

```bash
pip install -r requirements-web.txt
python gradio_app.py
```

Opens on <http://127.0.0.1:7860>.

## Create the Space

1. On <https://huggingface.co/new-space> pick **Gradio** SDK, CPU basic is enough.
2. Clone the Space repo and copy these files into it:

   ```bash
   git clone https://huggingface.co/spaces/<user>/<space-name>
   cd <space-name>
   cp /path/to/ibhs_granule_loss/gradio_app.py .
   cp /path/to/ibhs_granule_loss/loss.py .
   cp /path/to/ibhs_granule_loss/requirements-web.txt requirements.txt
   ```

   `app.py` (Tkinter) is **not** needed on the Space — it cannot run there.
   `requirements-web.txt` must be renamed to `requirements.txt`; it pins
   `opencv-python-headless` because the Space container has no `libGL.so.1`.

3. Create the Space's `README.md` with this front matter (the `app_file` line is what
   points the Space at `gradio_app.py` instead of the default `app.py`):

   ```yaml
   ---
   title: IBHS Granule Loss Analysis
   emoji: 🔬
   colorFrom: blue
   colorTo: red
   sdk: gradio
   sdk_version: 6.22.0
   app_file: gradio_app.py
   pinned: false
   license: mit
   ---

   Upload one microscope shingle image that includes the red scale bar. The app cleans the
   image, measures the scale bar, segments granule-loss regions and reports every region's
   area in mm² (IGL vs PGL), the same pipeline as the IBHS desktop tool.
   ```

4. Push:

   ```bash
   git add gradio_app.py loss.py requirements.txt README.md
   git commit -m "Add IBHS granule loss Gradio app"
   git push
   ```

## Alternative: deploy this repo directly

If you push this repository to the Space instead of copying files, add the same YAML front
matter (with `app_file: gradio_app.py`) to the top of the existing `README.md`, and make
`requirements.txt` match `requirements-web.txt` — otherwise the Space installs
`opencv-python` and `pyinstaller` and starts `app.py`, which fails with no display.

## Notes for the hosted version

- One image per run, queued with `concurrency_limit=1`; the segmentation is CPU-bound and
  a large source image takes tens of seconds on Space CPU hardware.
- Uploads are capped at 100 MB (`MAX_UPLOAD` in `gradio_app.py`).
- Each run works in its own temp folder, which is deleted when that browser session starts
  the next run or presses **Clear**. Nothing is persisted, so results must be downloaded
  from the app.
- `GL_Rating` / `CombinedGL_Rating` rank an impact against the other impacts in the same
  run via percentiles. With a single image they are always `0` and carry no meaning — the
  UI says so. Use the desktop app over a folder of impacts to get real ratings.
