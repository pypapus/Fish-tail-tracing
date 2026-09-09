# CZI upload compatibility fix

This version removes Streamlit's browser-side `type=["czi"]` filter.

Why:
Some browsers and operating systems do not recognize the uncommon `.czi`
extension and can refuse selection before the file reaches the Streamlit app.

The app now:
- lets the browser select files without an extension/MIME filter;
- validates `.czi` filenames inside Python;
- skips non-CZI files with a warning;
- raises Streamlit's upload limit to 500 MB per file.

Main file path remains:
`app.py`

Keep the Streamlit deployment on Python 3.12.
