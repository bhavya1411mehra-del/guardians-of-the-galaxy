# AIRFECTION V3.2 — Update your existing Streamlit app without VS Code

## What changed
- **New prominently placed '🪔 Diwali Forecast' page**, immediately below Dashboard.
- Before (5–7 Nov 2026), on (8 Nov), after (9–11 Nov).
- A day slider and an hour slider with historical festival comparisons.
- Comparison of 2024/2025 historical CAMS modeled air pollution with **actual published 2026 CAMS forecast data when available**.
- Missing future dates explicitly say **Not available yet**; old-year readings are *not* presented as 2026 forecast measurements.
- Existing 1–6-hour Random Forest, regional future heatmap, Health Shield and smoke-aware CLIP Sky Photo AI are retained.
- Fixes 2024 Diwali calibration date to 31 Oct (India central government calendar).
- Future regional map fetch now supports up to 7 forecast days (where provider coverage permits).
- `requirements.txt` installs CLIP dependencies by referring to `requirements-vision.txt`; note free hosting RAM may be insufficient.

## The easiest update (browser only)

1. Download and **extract** `AIRFECTION_V3_2_GITHUB_PATCH.zip` on your computer. Do not upload the ZIP itself.
2. Open your **existing GitHub repository**, and make sure it is on the **same branch that Streamlit is using** (often `main`).
3. At the directory where `app.py` currently lives (usually repository root), select **Add file → Upload files**.
4. Drag the **files inside** the patch folder into the upload area. Make sure `app.py`, `festival.py`, `ui.css`, `data_sources.py`, `sample_data.py`, `vision_optional.py`, `train_sky.py`, `requirements.txt` and `requirements-vision.txt` appear at the **same level**, not inside a new `AIRFECTION_V3_2` folder. The other files in your existing repo should remain untouched.
5. Enter commit message `Bring back Diwali forecast + update Sky Photo AI`. Select **Commit changes** (or **Propose changes** and merge the pull request if your repository requires it).
6. Wait for GitHub to show the new commit, then open your existing `*.streamlit.app` website and refresh. **Do not create a new Streamlit app**. Your GitHub-connected Streamlit app redeploys as commits arrive.
7. Open **🪔 Diwali Forecast** in the sidebar. Start in **Demo** (clearly labeled synthetic) to verify the interface, then **Live** for provider data. Check the Sky Photo AI page separately.
8. For debugging, open Streamlit Community Cloud → your app → **Manage app → Logs**.

If GitHub says you cannot overwrite a file by uploading, open `app.py` → pencil icon and replace its text from the supplied `app.py` file; commit. Do the same for each changed file. For the **new** `festival.py`, select **Add file → Create new file**, name it `festival.py`, paste its text, commit. Or use GitHub's browser-based `github.dev` editor by pressing `.` on the repository page; no VS Code installation needed.

## Install dependencies on Streamlit Community Cloud

The updated `requirements.txt` ends with:

```text
-r requirements-vision.txt
```

That ensures Streamlit attempts to install `torch` and `transformers` for Sky Photo AI. **This can exceed Community Cloud free-tier resources**; if installation or memory fails, temporarily remove that one line to restore the rest of AIRFECTION, then move the sky model to a separate inference host later. Do not claim the image AI works until you actually verify an image run.

## Scientifically honest explanation for judges

- **Random Forest:** our model trains on pollution/weather history and forecasts only +1 to +6 hours for the selected point; it has Diwali-date features but is *not* proven to predict major festival spikes.
- **Historic Diwali reference:** Open-Meteo/CAMS modeled values for 2024/2025 (or synthetic in demo); the displayed daily mean is not a forecast for 2026.
- **2026 future festival timeline:** actual **provider CAMS regional forecasts** are shown only on dates with forecast data. Dates beyond its seven-day horizon show *Not available yet*.
- **The official Indian AQI is not calculated**; PM2.5 concentration is in µg/m³, with optional provider US AQI which is a different index.
- **Sky Photo:** CLIP matches visual descriptions including industrial smoke but cannot calculate AQI or prove pollution concentrations.

## Key files

- `app.py` — website UI and page routing
- `festival.py` — festival dates, historical loading, relative-day matching, and honest forecast labeling
- `data_sources.py` — CAMS spatial forecast requests
- `sample_data.py` — Diwali calendar features, synthetic demo
- `ui.css` — Diwali cards and styling
- `vision_optional.py` — smoke-specific CLIP text prompts and safe interpretation
- `train_sky.py` — optional image-prototype classifier training
- `requirements.txt`, `requirements-vision.txt` — dependencies

Sources: <https://open-meteo.com/en/docs/air-quality-api>, <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>, <https://www.cgca.gov.in/CCAPB/list-of-holiday>.
