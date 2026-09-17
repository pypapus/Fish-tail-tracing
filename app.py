from pathlib import Path
import tempfile
import zipfile
import struct
import re
import cv2
import numpy as np

import altair as alt
import pandas as pd
import streamlit as st

from fish_tail_agent import FishTailAgent, AgentConfig


REFERENCE_DIR = Path(__file__).parent / "reference_rois"
REFERENCE_GT_CSV = Path(__file__).parent / "reference_ground_truth.csv"

def _read_reference_roi(path: Path):
    b = path.read_bytes()
    if b[:4] != b"Iout":
        raise ValueError("Unsupported reference ROI")
    top, left, bottom, right, n = struct.unpack(">hhhhh", b[8:18])
    xs = np.frombuffer(b, dtype=">u2", count=n, offset=64).astype(int) + left
    ys = np.frombuffer(b, dtype=">u2", count=n, offset=64 + 2*n).astype(int) + top
    return np.column_stack([xs, ys]).astype(np.int32)

def _sample_number(filename: str):
    m = re.search(r"\bCtrl\s*[_ -]*(\d+)\b", filename, flags=re.I)
    return int(m.group(1)) if m else None

def _reference_landmark(coords: np.ndarray):
    xmin, xmax = int(coords[:,0].min()), int(coords[:,0].max())
    width = max(1, xmax - xmin)
    threshold = xmin + 0.08 * width
    moved = False
    end_idx = None
    for i in range(1, len(coords)):
        if coords[i,0] > xmin + 0.20 * width:
            moved = True
        if moved and coords[i,0] <= threshold:
            end_idx = i
            break
    if end_idx is None:
        end_idx = min(len(coords)-1, max(3, len(coords)//3))
    inner = coords[:end_idx+1]
    maxx = float(inner[:,0].max())
    tip_pts = inner[inner[:,0] >= maxx - 0.02 * width]
    return float(np.mean(tip_pts[:,0])), float(np.mean(tip_pts[:,1]))

def _mask_from_polygon(coords: np.ndarray, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
    return mask

def _roi_validation(pred_roi_path: Path, ref_roi_path: Path, record: dict, original_path: Path, out_path: Path):
    pred = _read_reference_roi(pred_roi_path)
    ref = _read_reference_roi(ref_roi_path)

    img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Could not load original preview for validation")
    shape = img.shape

    pm = _mask_from_polygon(pred, shape)
    rm = _mask_from_polygon(ref, shape)
    inter = int(np.logical_and(pm, rm).sum())
    union = int(np.logical_or(pm, rm).sum())
    parea = int(pm.sum())
    rarea = int(rm.sum())

    iou = inter / union if union else 0.0
    dice = (2 * inter) / (parea + rarea) if (parea + rarea) else 0.0
    area_err_pct = 100.0 * (parea - rarea) / rarea if rarea else np.nan

    tx, ty = _reference_landmark(ref)
    noto = record.get("processing_notes", {}).get("notochord_detection", {}) or {}
    px = noto.get("endpoint_x")
    py = noto.get("endpoint_y")

    xerr = float(px) - tx if px is not None else np.nan
    yerr = float(py) - ty if py is not None else np.nan
    euclid = float(np.hypot(xerr, yerr)) if np.isfinite(xerr) and np.isfinite(yerr) else np.nan

    pixel_size = record.get("file_metadata", {}).get("pixel_size_um")
    xerr_um = xerr * float(pixel_size) if pixel_size is not None and np.isfinite(xerr) else np.nan
    euclid_um = euclid * float(pixel_size) if pixel_size is not None and np.isfinite(euclid) else np.nan

    # Researcher vs prediction overlay
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.polylines(rgb, [ref.reshape(-1,1,2)], True, (255, 0, 255), 4)   # magenta reference
    cv2.polylines(rgb, [pred.reshape(-1,1,2)], True, (0, 255, 0), 3)    # green prediction
    cv2.circle(rgb, (int(round(tx)), int(round(ty))), 10, (255, 0, 0), -1)
    if px is not None and py is not None:
        cv2.circle(rgb, (int(round(px)), int(round(py))), 8, (255, 220, 0), -1)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return {
        "Reference_Tip_X_px": tx,
        "Reference_Tip_Y_px": ty,
        "Predicted_Tip_X_px": px,
        "Predicted_Tip_Y_px": py,
        "Tip_X_Error_px": xerr,
        "Tip_Abs_X_Error_px": abs(xerr) if np.isfinite(xerr) else np.nan,
        "Tip_Euclidean_Error_px": euclid,
        "Tip_X_Error_um": xerr_um,
        "Tip_Euclidean_Error_um": euclid_um,
        "ROI_IoU": iou,
        "ROI_Dice": dice,
        "ROI_Area_Error_pct": area_err_pct,
    }

st.set_page_config(
    page_title="Fish Tail Boundary Analysis",
    page_icon="🐟",
    layout="wide",
)

# ---------- small visual helpers ----------
def qa_badge(flag: str, recommendation: str = ""):
    flag = str(flag or "review").lower()
    if flag == "good":
        st.success(f"✅ GOOD — {recommendation or 'Boundary passed automated QA.'}")
    elif flag == "poor":
        st.error(f"❌ FAILED — {recommendation or 'Do not use this ROI or its measurements.'}")
    else:
        st.warning(f"⚠️ REVIEW — {recommendation or 'Inspect the boundary before using measurements.'}")


def fmt_num(value, decimals=1, suffix=""):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}{suffix}"
    except Exception:
        return "—"


st.title("🐟 Fish Tail Boundary Analysis")
st.caption(
    "Automated notochord-defined caudal-fin morphometry for Zeiss CZI microscopy images"
)

# ---------- workflow ----------
st.subheader("What the app does")
w1, w2, w3, w4, w5, w6, w7 = st.columns(7)
w1.markdown("### 📤\n**Upload**\n\nCZI images")
w2.markdown("### 🔬\n**Detect**\n\nBrightfield")
w3.markdown("### 🧠\n**Trace**\n\nOuter fin")
w4.markdown("### 📍\n**Find**\n\nNotochord tip")
w5.markdown("### 📐\n**Measure**\n\nDistal fin")
w6.markdown("### ✅\n**Quality check**\n\nGood / Review / Poor")
w7.markdown("### 📥\n**Export**\n\nROI + CSV + QA")

st.info(
    "The measured ROI is the caudal-fin tissue distal to a vertical line placed through "
    "the detected end of the notochord. The app first segments the outer fin, then traces "
    "the notochord signal and estimates where it terminates. If that landmark is uncertain, "
    "the image is automatically marked REVIEW."
)

# ---------- calibration ----------
with st.expander("📏 Measurement calibration", expanded=False):
    st.write(
        "By default, the app tries to read the pixel size directly from the CZI metadata. "
        "Enter a value below only if you want to override the embedded calibration or if the file lacks one."
    )
    manual_pixel_size = st.number_input(
        "Manual pixel size override (µm/pixel)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
        help="Leave at 0 to use CZI metadata automatically."
    )
    if manual_pixel_size > 0:
        st.success(f"Manual calibration will be used: {manual_pixel_size:g} µm/pixel")
    else:
        st.caption("Automatic CZI calibration will be used when available.")

# ---------- upload ----------
st.subheader("Upload images")
uploaded = st.file_uploader(
    "Drag and drop Zeiss CZI files here",
    accept_multiple_files=True,
    help="Select one or more .czi microscopy files."
)

valid_uploaded, invalid_uploaded = [], []
if uploaded:
    for f in uploaded:
        if Path(f.name).suffix.lower() == ".czi":
            valid_uploaded.append(f)
        else:
            invalid_uploaded.append(f)

    if valid_uploaded:
        st.success(f"📁 {len(valid_uploaded)} CZI file(s) ready for analysis.")
    if invalid_uploaded:
        st.warning("Skipped non-CZI file(s): " + ", ".join(f.name for f in invalid_uploaded))


reference_mode = st.checkbox(
    "🧪 Compare against bundled researcher reference ROIs when available",
    value=True,
    help="For Ctrl 1, 2, 4, 5, 6, 7, 8, 9 and 12, the app will compare its output with the researcher ROI you supplied."
)
if reference_mode:
    st.caption(
        "Reference validation is available for Ctrl 1, 2, 4, 5, 6, 7, 8, 9 and 12. "
        "This does not change the prediction; it measures how close the current algorithm is to the researcher annotation."
    )

run = st.button(
    "▶ Analyze images",
    type="primary",
    disabled=not valid_uploaded,
    use_container_width=True,
)

if run:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        raw_dir = work / "raw"
        analyzed_dir = work / "analyzed"
        raw_dir.mkdir()
        analyzed_dir.mkdir()

        for f in valid_uploaded:
            (raw_dir / f.name).write_bytes(f.getbuffer())

        config = AgentConfig(
            pixel_size_um=manual_pixel_size if manual_pixel_size > 0 else None,
            channel_index=None,
            threshold_value=None,
            min_area_px2=100.0,
            clahe_clip_limit=2.0,
            save_png=True,
            save_roi=True,
            verbose=False,
        )
        agent = FishTailAgent(config)

        st.subheader("Analysis progress")
        stage_text = st.empty()
        progress = st.progress(0.0)
        files = sorted(raw_dir.glob("*.czi"))
        results, errors = [], []
        validation_rows = []

        stages = [
            "Reading CZI",
            "Finding brightfield",
            "Segmenting outer fin",
            "Finding notochord endpoint",
            "Building distal-fin ROI",
            "Measuring",
            "Quality check",
            "Complete",
        ]

        for i, path in enumerate(files, 1):
            # The engine performs these steps internally; the sequence makes progress understandable.
            stage_text.markdown(
                f"**{path.name}**  \n"
                f"🔬 {stages[0]} → 🧠 {stages[1]} → ✨ {stages[2]} → "
                f"📍 {stages[3]} → 📐 {stages[4]} → ✅ {stages[6]}"
            )
            try:
                result = agent.process_file(path, analyzed_dir)
                results.append(result)

                if reference_mode:
                    sample_no = _sample_number(path.name)
                    ref_path = REFERENCE_DIR / f"{sample_no}.roi" if sample_no is not None else None
                    pred_roi = Path(result["roi"]) if result.get("roi") else None
                    original_preview = analyzed_dir / f"{path.stem}_original_brightfield.png"

                    if (
                        sample_no is not None
                        and ref_path is not None
                        and ref_path.exists()
                        and pred_roi is not None
                        and pred_roi.exists()
                        and original_preview.exists()
                    ):
                        val_overlay = analyzed_dir / f"{path.stem}_reference_validation.png"
                        vals = _roi_validation(
                            pred_roi,
                            ref_path,
                            result["record"],
                            original_preview,
                            val_overlay,
                        )
                        vals["Filename"] = path.name
                        vals["Sample"] = sample_no
                        validation_rows.append(vals)

            except Exception as exc:
                errors.append((path.name, str(exc)))
            progress.progress(i / max(len(files), 1))

        csv_path = agent.export_summary_csv(analyzed_dir)
        validation_csv_path = None
        if validation_rows:
            validation_df = pd.DataFrame(validation_rows)
            validation_csv_path = analyzed_dir / "reference_validation_summary.csv"
            validation_df.to_csv(validation_csv_path, index=False)
        if agent.logger is not None:
            agent.logger.close()

        stage_text.markdown("✅ **Analysis complete**")
        progress.progress(1.0)

        if errors:
            st.warning(f"{len(errors)} file(s) could not be processed.")
            for filename, error in errors:
                st.write(f"- **{filename}**: {error}")

        if results and csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)

            # ---------- batch overview ----------
            st.divider()
            st.header("Results overview")

            counts = (
                df["Quality_Flag"].astype(str).str.lower().value_counts()
                if "Quality_Flag" in df.columns else pd.Series(dtype=int)
            )
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Images analyzed", len(df))
            q2.metric("✅ Good", int(counts.get("good", 0)))
            q3.metric("⚠️ Review", int(counts.get("review", 0)))
            q4.metric("❌ Poor", int(counts.get("poor", 0)))

            if int(counts.get("poor", 0)) > 0:
                st.error(
                    "At least one boundary failed automated QA. Do not use those ROI files "
                    "or morphometric values without manual review."
                )

            if "Edge_Clipped" in df.columns:
                clipped_mask = df["Edge_Clipped"].astype(str).str.lower().isin(["true", "1"])
                if clipped_mask.any():
                    st.warning(
                        f"⚠️ {int(clipped_mask.sum())} image(s) intersect an image edge. "
                        "Those measurements describe only the visible portion of the fin."
                    )

            if "NotochordReviewRequired" in df.columns:
                noto_review = df["NotochordReviewRequired"].astype(str).str.lower().isin(["true", "1"])
                if noto_review.any():
                    st.warning(
                        f"📍 {int(noto_review.sum())} image(s) have an uncertain notochord endpoint. "
                        "Inspect the yellow vertical cutoff before using those measurements."
                    )

            # Calibration summary
            if "Pixel_Size_um" in df.columns and df["Pixel_Size_um"].notna().any():
                calibrated = int(df["Pixel_Size_um"].notna().sum())
                st.success(
                    f"📏 Micron calibration available for {calibrated}/{len(df)} image(s). "
                    "Area is reported in µm²; perimeter, width, and height are reported in µm."
                )
            else:
                st.warning(
                    "📏 No micron calibration was detected. Pixel measurements are still available. "
                    "Enter a manual µm/pixel value and rerun if calibrated units are required."
                )

            # ---------- batch chart ----------
            if len(df) > 1:
                st.subheader("Batch comparison")
                chart_measure = "Area_um2" if "Area_um2" in df.columns and df["Area_um2"].notna().any() else "Area_px2"
                ylabel = "Distal fin area (µm²)" if chart_measure == "Area_um2" else "Fin area (px²)"
                chart_df = df[["Filename", chart_measure, "Quality_Flag"]].copy()
                chart_df[chart_measure] = pd.to_numeric(chart_df[chart_measure], errors="coerce")

                chart = (
                    alt.Chart(chart_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Filename:N", sort=None, title=None),
                        y=alt.Y(f"{chart_measure}:Q", title=ylabel),
                        color=alt.Color(
                            "Quality_Flag:N",
                            title="QA status",
                            scale=alt.Scale(
                                domain=["good", "review", "poor"],
                                range=["#2ca02c", "#ffbf00", "#d62728"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Filename:N"),
                            alt.Tooltip(f"{chart_measure}:Q", format=",.2f", title=ylabel),
                            alt.Tooltip("Quality_Flag:N", title="QA"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)

            # ---------- per-image cards ----------
            st.header("Image results")

            for _, row in df.iterrows():
                filename = str(row["Filename"])
                stem = Path(filename).stem
                flag = str(row.get("Quality_Flag", "review"))
                recommendation = str(row.get("Recommendation", ""))

                with st.expander(f"{'✅' if flag.lower()=='good' else '⚠️' if flag.lower()=='review' else '❌'} {filename}", expanded=len(df) == 1):
                    qa_badge(flag, recommendation)

                    # Prominent clipping notice
                    edge_clipped = str(row.get("Edge_Clipped", "False")).lower() in ("true", "1")
                    if edge_clipped:
                        clipped_edges = row.get("Clipped_Edges", "")
                        st.warning(
                            f"Image-edge clipping detected{f' ({clipped_edges})' if clipped_edges else ''}. "
                            "Area/perimeter describe the visible portion only."
                        )

                    noto_conf = float(row.get("NotochordConfidence", 0) or 0)
                    noto_method = str(row.get("Notochord_Method", "unknown"))
                    cutoff_x = row.get("Notochord_Cutoff_X_px", None)
                    if noto_conf < 0.60:
                        st.warning(
                            f"📍 Notochord endpoint confidence: {100*noto_conf:.0f}%. "
                            "The app made a best-effort cutoff; visually inspect the yellow line."
                        )
                    else:
                        st.success(
                            f"📍 Notochord endpoint detected ({100*noto_conf:.0f}% confidence)."
                        )

                    # Calibration label
                    px_um = row.get("Pixel_Size_um", None)
                    if pd.notna(px_um):
                        source = row.get("Calibration_Source", "CZI metadata")
                        st.caption(
                            f"📏 Calibration: {float(px_um):g} µm/pixel · Source: {source}"
                        )
                    else:
                        st.caption("📏 Calibration unavailable — calibrated micron measurements not shown.")

                    # Main metric cards
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    if "Area_um2" in df.columns and pd.notna(row.get("Area_um2")):
                        m1.metric("Distal fin area", fmt_num(row.get("Area_um2"), 1, " µm²"))
                        m2.metric("Perimeter", fmt_num(row.get("Perimeter_um"), 1, " µm"))
                        m3.metric("Width", fmt_num(row.get("Width_um"), 1, " µm"))
                        m4.metric("Height", fmt_num(row.get("Height_um"), 1, " µm"))
                    else:
                        m1.metric("Distal fin area", fmt_num(row.get("Area_px2"), 1, " px²"))
                        m2.metric("Perimeter", fmt_num(row.get("Perimeter_px"), 1, " px"))
                        m3.metric("Aspect ratio", fmt_num(row.get("Aspect_Ratio"), 3))
                        m4.metric("Solidity", fmt_num(row.get("Solidity"), 3))
                    m5.metric(
                        "Outer-fin AI",
                        fmt_num(100 * float(row.get("SegmentationConfidence", 0)), 0, "%")
                    )
                    m6.metric(
                        "Notochord",
                        fmt_num(100 * float(row.get("NotochordConfidence", 0)), 0, "%")
                    )

                    # Before / after and landmark review
                    original_path = analyzed_dir / f"{stem}_original_brightfield.png"
                    overlay_path = analyzed_dir / f"{stem}_boundary_overlay.png"
                    noto_path = analyzed_dir / f"{stem}_notochord_overlay.png"
                    raw_mask_path = analyzed_dir / f"{stem}_raw_model_mask.png"
                    measurement_mask_path = analyzed_dir / f"{stem}_measurement_mask.png"

                    st.markdown("#### Boundary and notochord check")
                    st.caption(
                        "Green = final measured ROI · Yellow = vertical cutoff at the notochord endpoint · "
                        "Cyan = estimated notochord trace · Blue = outer AI fin envelope."
                    )
                    tabs = st.tabs([
                        "Before / After",
                        "Original",
                        "Final ROI",
                        "Notochord",
                        "Outer AI mask",
                        "Measurement mask",
                    ])

                    with tabs[0]:
                        a, b = st.columns(2)
                        if original_path.exists():
                            a.image(
                                str(original_path),
                                caption="Original brightfield",
                                use_container_width=True,
                            )
                        if overlay_path.exists():
                            b.image(
                                str(overlay_path),
                                caption="Final distal-fin ROI",
                                use_container_width=True,
                            )

                    with tabs[1]:
                        if original_path.exists():
                            st.image(
                                str(original_path),
                                caption="Original brightfield",
                                use_container_width=True,
                            )

                    with tabs[2]:
                        if overlay_path.exists():
                            st.image(
                                str(overlay_path),
                                caption="Green ROI closes vertically at the detected notochord endpoint",
                                use_container_width=True,
                            )

                    with tabs[3]:
                        if noto_path.exists():
                            st.image(
                                str(noto_path),
                                caption="Stain-guided notochord trace, endpoint, and vertical cutoff",
                                use_container_width=True,
                            )

                    with tabs[4]:
                        if raw_mask_path.exists():
                            st.image(
                                str(raw_mask_path),
                                caption="MobileSAM outer-fin mask before notochord cutoff",
                                use_container_width=True,
                            )

                    with tabs[5]:
                        if measurement_mask_path.exists():
                            st.image(
                                str(measurement_mask_path),
                                caption="Final distal-fin mask used for morphometrics",
                                use_container_width=True,
                            )

                    # Secondary morphometrics
                    st.markdown("#### Additional morphometrics")
                    extras = pd.DataFrame(
                        {
                            "Measurement": [
                                "Aspect ratio",
                                "Circularity",
                                "Solidity",
                                "Eccentricity",
                                "QA score",
                                "Boundary evidence",
                                "Notochord confidence",
                                "Notochord cutoff X",
                                "Proximal/body side",
                                "Fin retained after cutoff",
                            ],
                            "Value": [
                                fmt_num(row.get("Aspect_Ratio"), 3),
                                fmt_num(row.get("Circularity"), 3),
                                fmt_num(row.get("Solidity"), 3),
                                fmt_num(row.get("Eccentricity"), 3),
                                fmt_num(row.get("QualityScore"), 0, "/100"),
                                fmt_num(100 * float(row.get("BoundaryEvidenceScore", 0)), 0, "%"),
                                fmt_num(100 * float(row.get("NotochordConfidence", 0)), 0, "%"),
                                fmt_num(row.get("Notochord_Cutoff_X_px"), 0, " px"),
                                str(row.get("Proximal_Side", "—")),
                                fmt_num(100 * float(row.get("Distal_Fin_Retained_Fraction", 0)), 0, "%"),
                            ],
                        }
                    )
                    st.dataframe(extras, use_container_width=True, hide_index=True)

            # ---------- compact full table ----------
            with st.expander("View full batch measurement table"):
                preferred = [
                    "Filename", "Area_um2", "Perimeter_um", "Width_um", "Height_um",
                    "Area_px2", "Perimeter_px", "Aspect_Ratio", "Circularity", "Solidity",
                    "QualityScore", "SegmentationConfidence", "NotochordConfidence",
                    "Notochord_Cutoff_X_px", "Notochord_Method", "Proximal_Side",
                    "Distal_Fin_Retained_Fraction", "Edge_Clipped",
                    "Quality_Flag", "Recommendation",
                ]
                cols = [c for c in preferred if c in df.columns]
                st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)

            # ---------- researcher reference validation ----------
            if validation_rows:
                st.header("Researcher reference validation")
                vdf = pd.DataFrame(validation_rows)

                st.caption(
                    "Magenta = researcher ROI · Green = app ROI · Red dot = researcher notochord-tip landmark · "
                    "Yellow dot = app-predicted landmark."
                )

                a, b, c, d = st.columns(4)
                a.metric("Reference pairs", len(vdf))
                b.metric("Mean |X error|", f"{vdf['Tip_Abs_X_Error_px'].mean():.1f} px")
                c.metric("Mean ROI IoU", f"{vdf['ROI_IoU'].mean():.3f}")
                d.metric("Mean Dice", f"{vdf['ROI_Dice'].mean():.3f}")

                if "Tip_Euclidean_Error_um" in vdf and vdf["Tip_Euclidean_Error_um"].notna().any():
                    st.write(
                        f"Mean landmark error: **{vdf['Tip_Euclidean_Error_um'].mean():.1f} µm** "
                        f"across calibrated reference images."
                    )

                chart = (
                    alt.Chart(vdf)
                    .mark_bar()
                    .encode(
                        x=alt.X("Filename:N", sort=None, title=None),
                        y=alt.Y("Tip_Abs_X_Error_px:Q", title="Absolute notochord-tip X error (px)"),
                        tooltip=[
                            "Filename:N",
                            alt.Tooltip("Tip_Abs_X_Error_px:Q", format=".1f"),
                            alt.Tooltip("ROI_IoU:Q", format=".3f"),
                            alt.Tooltip("ROI_Dice:Q", format=".3f"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)

                st.dataframe(
                    vdf[
                        [
                            "Filename",
                            "Reference_Tip_X_px",
                            "Predicted_Tip_X_px",
                            "Tip_X_Error_px",
                            "Tip_Abs_X_Error_px",
                            "Tip_Euclidean_Error_px",
                            "ROI_IoU",
                            "ROI_Dice",
                            "ROI_Area_Error_pct",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                for _, vr in vdf.iterrows():
                    stem_v = Path(vr["Filename"]).stem
                    overlay_v = analyzed_dir / f"{stem_v}_reference_validation.png"
                    if overlay_v.exists():
                        with st.expander(f"Reference overlay — {vr['Filename']}"):
                            st.image(str(overlay_v), use_container_width=True)

            # ---------- download ----------
            st.header("Export results")
            st.write(
                "The download includes **Fiji ROI files, summary CSV, per-image JSON metrics, "
                "boundary QA images, notochord/cutoff overlays, raw outer-fin masks, final measurement masks, and technical diagnostics**."
            )

            zip_path = work / "fish_tail_analysis_results.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for f in analyzed_dir.rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(analyzed_dir))

            st.download_button(
                "⬇ Download all analysis results",
                data=zip_path.read_bytes(),
                file_name="fish_tail_analysis_results.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

st.divider()
st.caption(
    "Scientific QA: automated segmentation is a screening aid. "
    "Visually validate each ROI against the original microscopy image before final statistical analysis."
)
