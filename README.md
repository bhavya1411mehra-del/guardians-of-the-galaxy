# AIRFECTION | Student air-quality information project

**Project type:** independent student work, not an official Government of India service.

This version uses a conventional, accessible white interface and clear navigation. It includes a 2026 Diwali historical-pattern outlook, a general 1–6 hour PM2.5 prediction, a regional pollution forecast map, risk alerts, evidence-linked health information and optional CLIP sky classification.

### Map design
The heatmap uses a georeferenced transparent raster layer (`BitmapLayer` in PyDeck). Nearby **forecast values** are interpolated using inverse-distance weighting; alpha fades outside sampled coverage instead of producing square polygons. The geographic source remains coarse regional modeled data; smoothing is presentation only and does not imply street-level accuracy. When spatial coverage is insufficient, only known points appear.

### Development
Install dependencies from `requirements.txt`, then run `streamlit run app.py`. The V3.4 Diwali forecasting logic and its caveats are unchanged.

### Deployment
See `GITHUB_UPDATE_V36.md` for browser-only GitHub and Streamlit instructions.
