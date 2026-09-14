from pathlib import Path
import tempfile
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

from fish_tail_agent import FishTailAgent, AgentConfig

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
    "Automated caudal-fin tracing and morphometric analysis for Zeiss CZI microscopy images"
)

# ---------- workflow ----------
st.subheader("What the app does")
w1, w2, w3, w4, w5, w6 = st.columns(6)
w1.markdown("### 📤\n**Upload**\n\nCZI images")
w2.markdown("### 🔬\n**Detect**\n\nBrightfield")
w3.markdown("### 🧠\n**Trace**\n\nFin boundary")
w4.markdown("### 📐\n**Measure**\n\nMorphometrics")
w5.markdown("### ✅\n**Quality check**\n\nGood / Review / Poor")
w6.markdown("### 📥\n**Export**\n\nROI + CSV + QA")

st.info(
    "The app automatically segments the visible caudal-fin region. "
    "Micron measurements are read from CZI calibration metadata when available. "
    "Always visually validate the traced boundary before final statistical analysis."
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

        stages = [
            "Reading CZI",
            "Finding brightfield",
            "Segmenting fin",
            "Measuring",
            "Quality check",
            "Complete",
        ]

        for i, path in enumerate(files, 1):
            # The engine performs these steps internally; the sequence makes progress understandable.
            stage_text.markdown(
                f"**{path.name}**  \n"
                f"🔬 {stages[0]} → 🧠 {stages[1]} → ✨ {stages[2]} → "
                f"📐 {stages[3]} → ✅ {stages[4]}"
            )
            try:
                results.append(agent.process_file(path, analyzed_dir))
            except Exception as exc:
                errors.append((path.name, str(exc)))
            progress.progress(i / max(len(files), 1))

        csv_path = agent.export_summary_csv(analyzed_dir)
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
                ylabel = "Fin area (µm²)" if chart_measure == "Area_um2" else "Fin area (px²)"
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
                    m1, m2, m3, m4, m5 = st.columns(5)
                    if "Area_um2" in df.columns and pd.notna(row.get("Area_um2")):
                        m1.metric("Area", fmt_num(row.get("Area_um2"), 1, " µm²"))
                        m2.metric("Perimeter", fmt_num(row.get("Perimeter_um"), 1, " µm"))
                        m3.metric("Width", fmt_num(row.get("Width_um"), 1, " µm"))
                        m4.metric("Height", fmt_num(row.get("Height_um"), 1, " µm"))
                    else:
                        m1.metric("Area", fmt_num(row.get("Area_px2"), 1, " px²"))
                        m2.metric("Perimeter", fmt_num(row.get("Perimeter_px"), 1, " px"))
                        m3.metric("Aspect ratio", fmt_num(row.get("Aspect_Ratio"), 3))
                        m4.metric("Solidity", fmt_num(row.get("Solidity"), 3))
                    m5.metric("Segmentation", fmt_num(100 * float(row.get("SegmentationConfidence", 0)), 0, "%"))

                    # Before / after
                    original_path = analyzed_dir / f"{stem}_original_brightfield.png"
                    overlay_path = analyzed_dir / f"{stem}_boundary_overlay.png"
                    mask_path = analyzed_dir / f"{stem}_raw_model_mask.png"

                    st.markdown("#### Boundary check")
                    tabs = st.tabs(["Before / After", "Original", "Boundary", "Mask"])
                    with tabs[0]:
                        a, b = st.columns(2)
                        if original_path.exists():
                            a.image(str(original_path), caption="Original brightfield", use_container_width=True)
                        if overlay_path.exists():
                            b.image(str(overlay_path), caption="Detected boundary", use_container_width=True)
                    with tabs[1]:
                        if original_path.exists():
                            st.image(str(original_path), caption="Original brightfield", use_container_width=True)
                    with tabs[2]:
                        if overlay_path.exists():
                            st.image(str(overlay_path), caption="Detected outer fin boundary", use_container_width=True)
                    with tabs[3]:
                        if mask_path.exists():
                            st.image(str(mask_path), caption="Model segmentation mask", use_container_width=True)
                        else:
                            st.caption("Raw model mask unavailable for this image.")

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
                            ],
                            "Value": [
                                fmt_num(row.get("Aspect_Ratio"), 3),
                                fmt_num(row.get("Circularity"), 3),
                                fmt_num(row.get("Solidity"), 3),
                                fmt_num(row.get("Eccentricity"), 3),
                                fmt_num(row.get("QualityScore"), 0, "/100"),
                                fmt_num(100 * float(row.get("BoundaryEvidenceScore", 0)), 0, "%"),
                            ],
                        }
                    )
                    st.dataframe(extras, use_container_width=True, hide_index=True)

            # ---------- compact full table ----------
            with st.expander("View full batch measurement table"):
                preferred = [
                    "Filename", "Area_um2", "Perimeter_um", "Width_um", "Height_um",
                    "Area_px2", "Perimeter_px", "Aspect_Ratio", "Circularity", "Solidity",
                    "QualityScore", "SegmentationConfidence", "Edge_Clipped",
                    "Quality_Flag", "Recommendation",
                ]
                cols = [c for c in preferred if c in df.columns]
                st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)

            # ---------- download ----------
            st.header("Export results")
            st.write(
                "The download includes **Fiji ROI files, summary CSV, per-image JSON metrics, "
                "boundary QA images, clean boundary overlays, raw model masks, and technical diagnostics**."
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
