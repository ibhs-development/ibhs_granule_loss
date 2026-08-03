# Web GUI
The web GUI is `gradio_app.py`. It reuses `loss.py` unchanged and analyzes **one uploaded
image per run**;

## Run locally

```bash
pip install -r requirements-web.txt
python gradio_app.py
```

Opens on <http://127.0.0.1:7860>.

## Notes for the web version

- Uploads are capped at 100 MB (`MAX_UPLOAD` in `gradio_app.py`).
- Each run works in its own temp folder, which is deleted when that browser session starts
  the next run or presses **Clear**. Nothing is persisted, so results must be downloaded
  from the app.
- `GL_Rating` / `CombinedGL_Rating` / `GL_Score` / `CombinedGL_Score` / `MeanSev_*` rank an
  impact against the other impacts in the same run via percentiles. With a single image
  those percentiles collapse, so the web UI hides these columns (`HIDDEN_COLUMNS` in
  `gradio_app.py`) instead of showing a `0` that reads like a severity. They are still in
  the downloadable `granule_loss_results.csv`, which is the pipeline's own untouched output.
  Use the desktop app over a folder of impacts to get real ratings.
