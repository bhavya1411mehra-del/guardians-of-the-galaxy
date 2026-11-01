# AIRFECTION V3.4 — Diwali 2026 historical-ML update

## Browser-only GitHub update (no VS Code)
1. Extract `AIRFECTION_V3_4_GITHUB_PATCH.zip` on your computer.
2. Open your existing GitHub repository on its Streamlit-deployed branch (usually `main`).
3. Upload **app.py** (replace), **diwali_ml.py** (new), and **diwali_page.py** (new) into the **same folder** where your current app.py lives. Do NOT delete your other files. GitHub: `Add file` → `Upload files` → `Commit changes`.
4. If upload does not replace a file, press the `.` key on the GitHub repository page to open the browser-based github.dev editor. Edit/save those three files there, then **Commit & Push**.
5. Open your EXISTING Streamlit Cloud website. Wait for the connected GitHub commit to deploy, or choose `Manage app → Reboot app` and refresh. Select `Diwali Forecast`.

## What this actually does
- Automatically reads public archived real CPCB-derived Indian city-hour data (2015–2020), **without an API key**.
- Trains Random Forest on past Diwali event windows (2015–2019 where available), holds the newest event out for evaluation, then retrains on all available years.
- Estimates air quality at EVERY hour from 5–11 November 2026. Day and hour sliders change the displayed result. No CAMS in the Diwali page; no published-forecast requirement; no past-year graphs on the public page.
- Forecasts **AQI** where the historic AQI field has sufficient observations. Otherwise, shows PM2.5 concentration in µg/m³ and does not mislabel it as AQI.
- Historical patterns do NOT know actual 2026 winds, rainfall, firecracker activity, or new restrictions. These are exploratory forecasts, not official or guaranteed readings. A Diwali spike is NOT fabricated.

## Data source and caveats
Public multi-city hourly archive: https://github.com/edusatyaki/DataAnalyticsProjrectIdeas/tree/main/01_National_Air_Quality

Archive ends July 2020. Only 2015–2019 Diwali events are usable. First load needs an internet connection to GitHub; then Streamlit caches data and outputs for 24 hours. If the large archive is unavailable, the code tries a small Delhi-only 2015–2016 fallback, which by itself is NOT enough years to train; no fake forecast is generated.

## Keep API keys secret
The main Diwali page requires **no key at all**. The optional OpenAQ Data Lab integration reads `OPENAQ_API_KEY` from Streamlit Cloud → app Settings → Secrets, if configured by the owner. Visitors cannot see or enter that key. NEVER commit secrets to public GitHub.

## Error fixes
- `No module named diwali_ml` / `diwali_page`: Put both files beside `app.py`.
- Seeing the old `Not published` message: Streamlit is still on the older GitHub commit or wrong app main-file path.
- Dataset download failed: GitHub raw source might be temporarily unreachable. No invented results are substituted.
- `Too few independently observed Diwali events`: Choose Delhi or obtain more real archived events for that city.
