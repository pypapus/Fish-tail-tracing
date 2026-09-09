# Fish Tail Agent v2 segmentation update

Changes made after QA of `Ctrl_72 hpbw 4.czi`:

- Adds an edge/texture-envelope segmentation mode for translucent caudal fins.
- Uses relative anatomical footprint checks instead of accepting tiny contours.
- Rejects candidates smaller than 0.8% of the image or with a bounding box smaller than 15% image width / 10% image height.
- Favors large candidates near the left side, matching the acquisition orientation in the calibration image.
- Keeps Otsu, Triangle, and adaptive thresholding as fallback candidates.
- Separates focus score from segmentation confidence.
- Prevents sharp images with bad segmentation from receiving misleadingly high overall QA.
- Poor segmentation now displays `SEGMENTATION FAILED - DO NOT USE ROI or morphometrics`.

Keep Streamlit Cloud on Python 3.12.
