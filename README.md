# Fish Tail Boundary Agent — Web App

This version gives the Fish Tail Boundary Agent a browser interface.

## 1. Install Python
Use Python 3.10 or newer.

## 2. Open a terminal in this folder

## 3. Install the packages

```bash
pip install -r requirements.txt
```

## 4. Start the website

```bash
streamlit run app.py
```

A browser window will open automatically, usually at:

`http://localhost:8501`

## Using the app

1. Upload one or more `.czi` files.
2. Leave **Auto-detect brightfield channel** checked unless you know the channel number.
3. Optionally enter the pixel size in µm/pixel.
4. Click **Run analysis**.
5. Review the table and boundary QA images.
6. Download the complete analysis ZIP.

The ZIP may contain:

- `*_metrics.json`
- `*_boundary.roi`
- `*_boundary_trace.png`
- `tail_metrics_summary.csv`
- `processing_log.txt`

## Important

Always inspect the generated boundary overlays or Fiji ROIs before using the measurements
for publication-quality statistics.
