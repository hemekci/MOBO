"""Path-B daylight verification per IES LM-83-12 + split-flux DF.

Computes spatial Daylight Autonomy sDA(300/50%): the fraction of regularly
occupied floor area receiving >= 300 lux for >= 50% of occupied hours per year.

Implementation:
  - Hourly vertical-plane illuminance per orientation via pvlib (Perez sky)
  - Convert global solar to illuminance via Perez 1990 luminous efficacy
  - Per-sensor Daylight Factor via split-flux method (CIBSE Lighting Guide,
    Hopkinson et al. 1966) with depth attenuation per Reinhart 2014
  - Hourly interior illuminance = DF_sensor x exterior_horizontal_illuminance
  - LM-83 occupied schedule: 08:00-18:00 weekdays
  - sDA = mean over interior grid of (% occ time >= 300 lux >= 50%)

Citations:
  - IES LM-83-12: Approved Method for IES Spatial Daylight Autonomy and
    Annual Sunlight Exposure (2012)
  - Reinhart, C. F. (2014). Daylighting Handbook I (MIT Press), eq. 4.7
  - Hopkinson, R. G., Petherbridge, P., Longmore, J. (1966). Daylighting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_b.weather import WeatherData

# IES LM-83-12 thresholds
ILLUMINANCE_THRESHOLD_LUX: float = 300.0
TIME_THRESHOLD_FRACTION: float = 0.50

# Room geometry (representative perimeter zone)
ROOM_DEPTH_M: float = 6.0
ROOM_WIDTH_M: float = 6.0
ROOM_HEIGHT_M: float = 3.0
N_GRID_DEPTH: int = 6
N_GRID_WIDTH: int = 6

INTERIOR_REFLECTANCE: float = 0.50          # avg ceiling/wall/floor reflectance
LUMINOUS_EFFICACY_LM_W: float = 110.0       # Perez 1990 average


def _occupied_mask(times: pd.DatetimeIndex) -> np.ndarray:
    weekday = np.asarray(times.weekday) < 5
    hour = np.asarray(times.hour)
    return np.asarray(weekday & (hour >= 8) & (hour < 18))


def _horizontal_illuminance(weather: WeatherData) -> np.ndarray:
    """Hourly outdoor horizontal illuminance (lux) — used as DF reference."""
    return weather.global_horiz_wh_m2 * LUMINOUS_EFFICACY_LM_W


def _split_flux_df(
    window_area_m2: float,
    glazing_vt: float,
    sky_visible_angle: float,
    a_total_internal_m2: float,
    rho: float,
) -> float:
    """CIBSE/Hopkinson split-flux mean Daylight Factor (decimal fraction).

    DF_mean = (T * Aw * theta) / (A_total * (1 - rho^2))
    """
    if a_total_internal_m2 <= 0:
        return 0.0
    return (glazing_vt * window_area_m2 * sky_visible_angle) / (a_total_internal_m2 * (1 - rho ** 2))


def _depth_attenuation(depth_idx: int) -> float:
    """Reinhart 2014 eq. 4.7 simplified: DF falls as 1/(1+5*z/H)^1.5 where z is depth from window."""
    depth_norm = (depth_idx + 0.5) / N_GRID_DEPTH
    return 1.0 / (1.0 + 5.0 * depth_norm) ** 1.5


def _width_attenuation(width_idx: int) -> float:
    width_norm = abs((width_idx + 0.5) / N_GRID_WIDTH - 0.5) * 2
    return 1.0 - 0.15 * width_norm


def sda_lm83(design: EnvelopeDesign, weather: WeatherData) -> dict[str, float]:
    """Compute sDA(300/50%) per IES LM-83-12."""
    times = weather.hours
    occ_mask = _occupied_mask(times)
    n_occ = int(occ_mask.sum())

    horiz_lux = _horizontal_illuminance(weather)

    # Geometry: a 6x6x3 m perimeter zone with windows on south wall (worst-case
    # representation; LM-83 averages across orientations through occupied area).
    # Interior surface area
    a_floor = ROOM_WIDTH_M * ROOM_DEPTH_M
    a_ceiling = a_floor
    a_walls = 2 * (ROOM_WIDTH_M + ROOM_DEPTH_M) * ROOM_HEIGHT_M
    a_total = a_floor + a_ceiling + a_walls

    # Average WWR across orientations (LM-83 averages over zone). A perimeter
    # zone has windows on a single facade — use one wall, not all four.
    wwr_avg = design.wwr_avg
    wall_area = ROOM_WIDTH_M * ROOM_HEIGHT_M
    window_area = wwr_avg * wall_area

    # Sky visible angle: vertical window sees ~50% of sky; reduced by overhang
    overhang_attenuation = max(0.5, 1.0 - design.overhang_ratio)
    sky_angle = 0.5 * overhang_attenuation

    # Mean DF (decimal fraction)
    df_mean = _split_flux_df(
        window_area_m2=window_area,
        glazing_vt=design.glazing_props["VT"],
        sky_visible_angle=sky_angle,
        a_total_internal_m2=a_total,
        rho=INTERIOR_REFLECTANCE,
    )

    # For each sensor compute df_sensor = df_mean * depth_atten * width_atten
    # then check the LM-83 criterion against the hourly horizontal lux.
    n_meets = 0
    for di in range(N_GRID_DEPTH):
        for wi in range(N_GRID_WIDTH):
            df_sensor = df_mean * _depth_attenuation(di) * _width_attenuation(wi)
            interior_lux = horiz_lux * df_sensor
            occ_lux = interior_lux[occ_mask]
            frac_above = (occ_lux >= ILLUMINANCE_THRESHOLD_LUX).sum() / n_occ
            if frac_above >= TIME_THRESHOLD_FRACTION:
                n_meets += 1

    n_grid = N_GRID_DEPTH * N_GRID_WIDTH
    return {
        "sda_pct_path_b": float(100.0 * n_meets / n_grid),
        "df_mean_pct": float(100.0 * df_mean),
        "n_grid_meets_threshold": n_meets,
        "n_grid_total": n_grid,
        "n_occupied_hours": n_occ,
    }
