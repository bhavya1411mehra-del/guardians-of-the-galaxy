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
