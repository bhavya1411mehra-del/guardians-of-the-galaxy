"""Value-driven geographic forecast surface with explicit data coverage limits.

The surface is interpolated from regional CAMS model grid-cell values, or from
other independently supplied spatial readings. It is NOT measured street-level
pollution. With fewer than three distinct points it intentionally returns no
surface. Values, not station density, control colour.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Continuous ramps: greater estimated pollution -> darker red.
PM25_STOPS = [(0, (46, 173, 149)), (15, (112, 194, 130)),
              (35, (247, 204, 100)), (75, (240, 137, 67)),
              (125, (224, 67, 64)), (200, (157, 24, 52)), (300, (83, 7, 31))]
AQI_STOPS = [(0, (36, 155, 125)), (50, (54, 180, 138)),
             (100, (241, 206, 95)), (150, (240, 139, 72)),
             (200, (217, 55, 59)), (300, (127, 15, 46)), (500, (47, 4, 22))]


def color_for_value(value: float, metric: str = "pm2_5", alpha: int = 214) -> list[int]:
    ramp = AQI_STOPS if metric == "us_aqi" else PM25_STOPS
    val = max(float(value), 0.0)
    for i in range(1, len(ramp)):
        b, rgb1 = ramp[i]
        a, rgb0 = ramp[i-1]
        if val <= b:
            f = (val - a) / (b - a)
            return [int(round(c0 * (1-f) + c1 * f)) for c0, c1 in zip(rgb0, rgb1)] + [alpha]
    return [*ramp[-1][1], alpha]


def interpolate_surface(points: pd.DataFrame, metric: str, *, steps: int = 32,
                        max_nearest_km: float = 46.0) -> pd.DataFrame:
    """Generate polygons with inverse-distance-weighted predicted values.

    Only interpolate inside the data footprint and near sampled locations. This
    does not create new independent pollution measurements or resolution.
    """
    needed = ["lat", "lon", metric]
    if not set(needed).issubset(points):
        return pd.DataFrame(columns=["polygon", "value", "fill"])
    points = (points[needed].replace([np.inf, -np.inf], np.nan).dropna()
              .drop_duplicates(subset=["lat", "lon"]))
    if len(points) < 3 or steps < 2:
        return pd.DataFrame(columns=["polygon", "value", "fill"])
    lat = points.lat.to_numpy(dtype=float)
    lon = points.lon.to_numpy(dtype=float)
    values = points[metric].to_numpy(dtype=float)
    if (np.ptp(lat) < 0.08 or np.ptp(lon) < 0.08):
        return pd.DataFrame(columns=["polygon", "value", "fill"])
    # Bound surface to supplied forecast grid, with no extrapolation beyond it.
    edges_x = np.linspace(float(lon.min()), float(lon.max()), steps + 1)
    edges_y = np.linspace(float(lat.min()), float(lat.max()), steps + 1)
    xg, yg = np.meshgrid((edges_x[:-1] + edges_x[1:])/2,
                         (edges_y[:-1] + edges_y[1:])/2)
    # Local geographic approximation: longitude compressed by latitude cosine.
    lon_scale = 111.32 * np.cos(np.deg2rad(np.mean(lat)))
    distances = np.sqrt(((xg[..., None] - lon[None, None, :]) * lon_scale)**2 +
                        ((yg[..., None] - lat[None, None, :]) * 111.32)**2)
    nearest = distances.min(axis=-1)
    weights = 1.0 / np.maximum(distances, 1.0)**2
    modeled = (weights @ values) / weights.sum(axis=-1)
    result = []
    for iy, ix in np.argwhere(nearest <= max_nearest_km):
        v = float(modeled[iy, ix])
        west, east = float(edges_x[ix]), float(edges_x[ix + 1])
        south, north = float(edges_y[iy]), float(edges_y[iy + 1])
        result.append({"polygon": [[west, south], [east, south], [east, north], [west, north]],
                       "value": round(v, 1), "fill": color_for_value(v, metric)})
    return pd.DataFrame(result)


def demo_grid(lat: float, lon: float, current: float, hour: int) -> pd.DataFrame:
    """Fictional spatial fields for a clearly labeled demonstration only."""
    results = []
    for iy in range(-2, 3):
        for ix in range(-2, 3):
            shift = 38 * np.exp(-((ix-1)**2 + (iy+1)**2)/2.6) - 22 * np.exp(-((ix+1)**2 + (iy-1)**2)/2.5)
            value = max(1., current + shift + 3.5 * hour + 5 * np.sin(ix * 1.3 + iy))
            results.append({"lat": lat + iy * .24,
                            "lon": lon + ix * .24 / max(.35, np.cos(np.deg2rad(lat))),
                            "pm2_5": round(float(value), 1),
                            "us_aqi": np.nan,
                            "source": "SYNTHETIC / fictitious demonstration"})
    return pd.DataFrame(results)


# BitmapLayer is supported by deck.gl and pydeck and avoids the visibly
# square polygon cells of the earlier implementation. The overlay is only
# a rendering of existing forecast-grid values, NOT additional observations.
import base64
from io import BytesIO
from PIL import Image


def smooth_raster_overlay(points: pd.DataFrame, metric: str = "pm2_5", *,
                          resolution: int = 300) -> dict | None:
    """Return a feather-edged geographic PNG overlay for PyDeck BitmapLayer.

    The raster is an inverse-distance-weighted representation of regional
    forecast samples. Its transparency is driven by distance to the nearest
    source sample and fades outside coverage. It never generates a layer with
    fewer than three geographically distinct data points. 
    """
    needed = {"lat", "lon", metric}
    if not needed.issubset(points.columns):
        return None
    frame = points[list(needed)].replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.groupby(["lat", "lon"], as_index=False)[metric].mean()
    if len(frame) < 3:
        return None
    lat, lon = frame["lat"].to_numpy(float), frame["lon"].to_numpy(float)
    values = np.maximum(frame[metric].to_numpy(float), 0)
    if np.ptp(lat) < .08 or np.ptp(lon) < .08:
        return None
    # Estimate the normal spacing of actual forecast grid locations (km).
    km_x = 111.32 * max(.2, np.cos(np.deg2rad(float(np.mean(lat)))))
    pts_km = np.column_stack([lon * km_x, lat * 111.32])
    dd = np.sqrt(((pts_km[:,None,:] - pts_km[None,:,:])**2).sum(axis=2))
    np.fill_diagonal(dd, np.inf)
    spacing_km = float(np.median(dd.min(axis=1)))
    if not np.isfinite(spacing_km) or spacing_km <= 0:
        return None
    spacing_km = max(spacing_km, 5.0)
    # Extra margin is transparent, so there is no hard rectangular edge.
    pad_x, pad_y = 1.7 * spacing_km / km_x, 1.7 * spacing_km / 111.32
    west, east = float(lon.min()-pad_x), float(lon.max()+pad_x)
    south, north = float(lat.min()-pad_y), float(lat.max()+pad_y)
    n = int(max(120, min(resolution, 460)))
    xx = np.linspace(west, east, n, dtype=np.float32)
    yy = np.linspace(north, south, n, dtype=np.float32)  # PNG top = north
    gx, gy = np.meshgrid(xx, yy)
    distances = np.sqrt(((gx[...,None]-lon) * km_x)**2 +
                        ((gy[...,None]-lat) * 111.32)**2)
    # IDW controls the colour; coverage only controls alpha, not pollution.
    w = 1.0 / np.maximum(distances, 1.0)**2
    estimates = (w @ values) / w.sum(axis=-1)
    nearest = distances.min(axis=-1)
    # Full opacity close to source cells; natural, rounded fade to transparent.
    start, end = 0.75*spacing_km, 1.95*spacing_km
    t = np.clip((end-nearest)/(end-start), 0, 1)
    alpha = np.uint8(np.rint(222 * (t*t*(3-2*t))))
    # Map the interpolated concentration to a continuous, value-based colour.
    ramp = AQI_STOPS if metric == "us_aqi" else PM25_STOPS
    stops = np.array([s[0] for s in ramp], dtype=float)
    colors = np.array([s[1] for s in ramp], dtype=float)
    rgb = np.stack([np.interp(estimates, stops, colors[:,i]) for i in range(3)], axis=-1)
    rgba = np.concatenate([np.uint8(np.rint(rgb)), alpha[...,None]], axis=2)
    output = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", optimize=True)
    return {"image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
            "bounds": [west, south, east, north],
            "values_min": float(values.min()), "values_max": float(values.max()),
            "samples": len(frame), "png_bytes": len(output.getvalue())}
