# Streamlit Community Cloud deployment fix

The previous deployment failed because Streamlit used Python 3.14 and
`aicspylibczi==4.0.0` tried to compile from source. The build then failed
because CMake was unavailable.

## Required deployment settings

- Repository: your fish-tail-tracing repository
- Branch: main
- Main file path: app.py
- Python version: 3.12

## Important
Streamlit Community Cloud does not let you change the Python version of an
already deployed app in place. Delete the failed app and redeploy it.

During the new deployment:
1. Choose the same repository and branch.
2. Set main file path to `app.py`.
3. Open **Advanced settings**.
4. Select **Python 3.12**.
5. Deploy.

This package pins:
`aicspylibczi==3.2.0`

That release has a prebuilt Linux wheel for CPython 3.12, so it should not
need to compile C++ or invoke CMake.
