# AIRFECTION V3.6 — Update existing Streamlit website from a browser

## What changed
- Simple light/white public-information layout, conventional form controls, charts, plain typography. No government endorsement or branding.
- Map now uses a smooth transparent PNG bitmap overlay instead of individual square polygon cells.
- It fades near the boundary of available regional forecast samples; colouring follows **forecast concentrations**, not station density.
- Diwali 2026 forecasts, historical ML method, Health Shield, Sky Photo and other modules are retained from V3.4.

## Upload without VS Code
1. **Download and extract** `AIRFECTION_V3_6_GITHUB_PATCH.zip`.
2. Open the **existing** GitHub repo that Streamlit is deploying. Confirm you're on `main` (or your actual deployment branch).
3. Click `Add file` → `Upload files`. Upload the files from the ZIP **to the same directory as `app.py`**, not another nested folder. GitHub's upload page may not replace existing files; if it rejects them, press `.` in the repo to open `github.dev`, drag replacements into the file tree, and commit/push there.
4. Replace **app.py**, **heatmap.py**, **ui.css**, **diwali_page.py**, **README.md**. Keep every other Python file from your existing V3.4 project.
5. Important: put the replacement **config.toml** inside `.streamlit/config.toml` (same nested path). In the browser editor you can create/open the `.streamlit` directory and replace the file content.
6. Commit message: `Simplify AIRFECTION and improve smooth pollution heatmap`.
7. Open your existing Streamlit app link. Auto-deploy usually starts from the GitHub commit. If outdated, use the app's Manage menu → reboot and refresh.

## Important details
- No external API key is necessary for the *Diwali historical model* page, which uses a public historical archive. The other Data Sources page can still optionally accept an OpenAQ key for private station downloads.
- The map's colored regional surface is derived from **CAMS grid forecasts**, not your separate Random Forest point forecasts, and is not street-level.
- The map will show **points only** if fewer than three distinct locations are available; it never draws a fabricated continuous area from a single station.
- Streamlit Cloud must install `Pillow`, `numpy`, `pandas`, `pydeck` and the rest of `requirements.txt`.
- If the new map is blank, check deployment logs, confirm `heatmap.py` and `app.py` came from this ZIP, and confirm the GitHub commit deployed successfully. The map also needs the tile provider reachable by the browser.
