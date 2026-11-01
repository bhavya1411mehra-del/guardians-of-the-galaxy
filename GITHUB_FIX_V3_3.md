# AIRFECTION V3.3 — REAL DIWALI DATA FIX (browser only)

**What changed:** Previous version displayed historical regional-model phase means (such as 69/51/61) which can hide Diwali-evening pollution spikes and were not accurate 2026 predictions. V3.3 adds real measured CPCB PM2.5 history via XKDR (2024 works with its public demonstration key); separates Diwali evening from next-morning and 24-hour averages; and clearly separates historical readings from future provider CAMS forecasts. It does **not** force the Diwali value to be highest.

## Upload changes to GitHub (NO VS Code)

1. Extract `AIRFECTION_V3_3_GITHUB_PATCH.zip`.
2. Open **the existing GitHub repository** on the `main` branch. Do not create a new repo.
3. Press **.** (period) to open the GitHub web editor `github.dev`.
4. Replace **app.py**, **festival.py** with the extracted files; create **station_archive.py** in the SAME folder as app.py. Also upload this readme if wanted.
5. In Source Control (left fork icon), stage the changed files, enter `Fix Diwali with real CPCB station history` and **Commit & Push** to `main`. If GitHub allows direct replace using Add file → Upload files, that also works.
6. Open your same Streamlit Cloud app and refresh. If needed, open **Manage app → Reboot app**.
7. Choose **🪔 Diwali Forecast**, then **Measured CPCB stations**, **Delhi**, **2024**. The public XKDR key is embedded as a *published demo key* with coverage for 2024 only.
8. For 2025 or 2023, get your own key at https://airquality.xkdr.org/signup and paste it into the protected field. To keep it on deployment, use Streamlit Cloud app settings → Secrets: `XKDR_API_KEY = "YOUR_KEY"` (do NOT paste a private key in GitHub).

These are real PM2.5 concentrations in µg/m³, NOT Indian AQI. API network availability and 2024 city station coverage must be tested on Streamlit Cloud. The public demo key is an official published demo resource; it is not your secret. The site's data are preliminary/not fully validated.

**Data provider:** XKDR Forum (2026), India Air Quality Database, derived from CPCB CAAQM station measurements: https://airquality.xkdr.org (CC BY 4.0). **Forecast provider:** Open-Meteo CAMS: https://open-meteo.com/en/docs/air-quality-api.
