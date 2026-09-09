
from pathlib import Path
import tempfile
import zipfile
import shutil
import pandas as pd
import streamlit as st

from fish_tail_agent import FishTailAgent, AgentConfig

st.set_page_config(page_title="Fish Tail Boundary Agent", page_icon="🐟", layout="wide")

st.title("🐟 Fish Tail Boundary Agent — Boundary Test 2")
st.write(
    "Upload Zeiss `.czi` microscopy files. This experimental build tests smooth outer-boundary continuation across faint translucent fin margins. The agent will detect the brightfield "
    "channel, trace the fish-tail boundary, calculate morphometric measurements, "
    "generate Fiji ROIs and QA images, and create a GraphPad Prism-ready CSV."
)

with st.sidebar:
    st.header("Analysis settings")
    pixel_size = st.number_input(
        "Pixel size (µm/pixel, optional)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Leave at 0 to report measurements only in pixels."
    )
    auto_channel = st.checkbox("Auto-detect brightfield channel", value=True)
    channel = None
    if not auto_channel:
        channel = st.number_input("Brightfield channel index", min_value=0, value=0, step=1)

    auto_threshold = st.checkbox("Automatic thresholding", value=True)
    threshold = None
    if not auto_threshold:
        threshold = st.slider("Manual threshold", 0, 255, 127)

    min_area = st.number_input("Minimum contour area (px²)", min_value=1.0, value=100.0, step=10.0)
    clahe_clip = st.slider("CLAHE contrast strength", 1.0, 4.0, 2.0, 0.1)
    save_roi = st.checkbox("Create Fiji .roi files", value=True)
    save_png = st.checkbox("Create QA PNG images", value=True)

uploaded = st.file_uploader(
    "Upload one or more Zeiss `.czi` files",
    accept_multiple_files=True,
    help="The browser filter is intentionally disabled because some systems do not recognize the .czi extension."
)

valid_uploaded = []
invalid_uploaded = []

if uploaded:
    for f in uploaded:
        if Path(f.name).suffix.lower() == ".czi":
            valid_uploaded.append(f)
        else:
            invalid_uploaded.append(f)

    if valid_uploaded:
        st.info(f"{len(valid_uploaded)} valid CZI file(s) selected.")
    if invalid_uploaded:
        st.warning(
            "These files were skipped because they are not `.czi`: "
            + ", ".join(f.name for f in invalid_uploaded)
        )

run = st.button("Run analysis", type="primary", disabled=not valid_uploaded)

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
            pixel_size_um=pixel_size if pixel_size > 0 else None,
            channel_index=None if auto_channel else int(channel),
            threshold_value=None if auto_threshold else int(threshold),
            min_area_px2=float(min_area),
            clahe_clip_limit=float(clahe_clip),
            save_png=save_png,
            save_roi=save_roi,
            verbose=False,
        )

        agent = FishTailAgent(config)

        progress = st.progress(0.0)
        status = st.empty()
        results = []
        errors = []

        files = sorted(raw_dir.glob("*.czi"))
        for i, path in enumerate(files, 1):
            status.write(f"Analyzing **{path.name}** ({i}/{len(files)})")
            try:
                result = agent.process_file(path, analyzed_dir)
                results.append(result)
            except Exception as exc:
                errors.append((path.name, str(exc)))
            progress.progress(i / len(files))

        csv_path = agent.export_summary_csv(analyzed_dir)
        if agent.logger is not None:
            agent.logger.close()

        status.empty()
        progress.empty()

        st.subheader("Results")
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True, hide_index=True)

            if "Quality_Flag" in df.columns:
                counts = df["Quality_Flag"].value_counts()
                cols = st.columns(3)
                for col, name in zip(cols, ["good", "review", "poor"]):
                    col.metric(name.title(), int(counts.get(name, 0)))

                if int(counts.get("poor", 0)) > 0:
                    st.error(
                        "One or more files failed segmentation QA. Do NOT use morphometric values or ROI files "
                        "from rows marked poor without manual review."
                    )

        if errors:
            st.warning(f"{len(errors)} file(s) could not be processed.")
            for filename, error in errors:
                st.write(f"- **{filename}**: {error}")
        else:
            st.success(f"Analysis complete: {len(results)} file(s) processed.")

        test2_images = sorted(analyzed_dir.glob("*_boundary_test2.png"))
        if test2_images:
            st.subheader("Boundary Continuation Test 2")
            st.caption(
                "Inspect whether the upper and lower paths remain on the OUTER fin margins, "
                "including where the translucent boundary becomes weak, instead of jumping inward to dark rays."
            )
            for img in test2_images[:12]:
                st.image(str(img), caption=img.name, use_container_width=True)

        prep_images = sorted(analyzed_dir.glob("*_preprocessing_test1.png"))
        if prep_images:
            st.subheader("Preprocessing Test 1")
            st.caption(
                "Compare Original, Previous CLAHE, the new background-corrected image, "
                "and the new edge response. The key question is whether the OUTER fin margin "
                "is clearer and more continuous than the internal rays/body texture."
            )
            for img in prep_images[:12]:
                st.image(str(img), caption=img.name, use_container_width=True)

        qa_images = sorted(analyzed_dir.glob("*_boundary_trace.png"))
        if qa_images:
            st.subheader("Boundary QA")
            for img in qa_images[:12]:
                st.image(str(img), caption=img.name, use_container_width=True)

        zip_path = work / "fish_tail_analysis_results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in analyzed_dir.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(analyzed_dir))

        st.download_button(
            "Download all analysis outputs (.zip)",
            data=zip_path.read_bytes(),
            file_name="fish_tail_analysis_results.zip",
            mime="application/zip",
            type="primary",
        )

st.divider()
st.caption(
    "Scientific QA recommendation: visually inspect generated boundary PNGs or Fiji ROIs "
    "before using measurements for final statistical analysis."
)
