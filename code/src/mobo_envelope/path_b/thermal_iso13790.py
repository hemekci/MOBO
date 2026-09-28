"""Annual hourly thermal simulation per ISO 13790 / EN ISO 52016-1.

Implements a single-zone reduced-order model with:
  - Conductive heat transfer through opaque + glazed envelope
  - Solar gains on each orientation via pvlib (Perez sky model)
  - Infiltration with constant ACH
  - Constant internal gains (people, lights, equipment)
  - Ideal heating/cooling setpoint band (no HVAC system efficiency)

Output: annual heating + cooling energy use intensity (EUI), kWh per m^2 floor.
This is the high-fidelity comparison metric for Path A's instantaneous U_eff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from mobo_envelope.constants import GLAZING
from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_b.weather import WeatherData
from mobo_envelope.thermal import u_value_opaque

# Single-zone defaults (representative office/residential, ASHRAE 90.1 typical)
HEATING_SETPOINT_C: float = 20.0
COOLING_SETPOINT_C: float = 26.0
ACH_INFILTRATION: float = 0.5            # air changes per hour
INTERNAL_GAINS_W_M2: float = 8.0         # people + lights + equipment, conservative
ZONE_HEIGHT_M: float = 3.0               # floor-to-ceiling
ZONE_WIDTH_M: float = 6.0                # square zone footprint
RHO_AIR_KG_M3: float = 1.2
CP_AIR_J_KG_K: float = 1005.0


def _surface_irradiance(
    weather: WeatherData,
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
) -> np.ndarray:
    """Hourly plane-of-array irradiance on a tilted surface (W/m^2)."""
    times = weather.hours
    site = pvlib.location.Location(
        latitude=weather.latitude,
        longitude=weather.longitude,
        altitude=weather.elevation_m,
    )
    solpos = site.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt_deg,
        surface_azimuth=surface_azimuth_deg,
        solar_zenith=solpos["apparent_zenith"].to_numpy(),
        solar_azimuth=solpos["azimuth"].to_numpy(),
        dni=weather.direct_normal_wh_m2,
        ghi=weather.global_horiz_wh_m2,
        dhi=weather.diffuse_horiz_wh_m2,
        dni_extra=dni_extra,
        model="perez",
    )
    return np.nan_to_num(poa["poa_global"].to_numpy(), nan=0.0)


# Cache POA per (city, orientation) since it depends only on weather + geometry.
_POA_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _cached_poa(weather: WeatherData, orientation: str) -> np.ndarray:
    key = (weather.city, orientation)
    if key in _POA_CACHE:
        return _POA_CACHE[key]
    azimuth = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}[orientation]
    arr = _surface_irradiance(weather, surface_tilt_deg=90.0, surface_azimuth_deg=azimuth)
    _POA_CACHE[key] = arr
    return arr


def annual_eui(
    design: EnvelopeDesign,
    weather: WeatherData,
    floor_area_m2: float = ZONE_WIDTH_M * ZONE_WIDTH_M,
    facade_height_m: float = ZONE_HEIGHT_M,
) -> dict[str, float]:
    """Compute annual heating + cooling energy use intensity (kWh/m^2/yr).

    Reduced-order single-zone model: hourly conductive + solar + infiltration +
    internal balance, ideal HVAC with deadband.
    """
    Te = weather.dry_bulb_c
    n_hours = len(Te)

    # Envelope geometry: rectangular zone with one wall per orientation
    # Wall area per orientation = facade_height * ZONE_WIDTH_M
    a_wall_per_orient = facade_height_m * ZONE_WIDTH_M

    # Glazed and opaque areas per orientation
    wwr = {
        "north": design.wwr_north,
        "south": design.wwr_south,
        "east": design.wwr_east,
        "west": design.wwr_west,
    }
    a_glaz = {o: a_wall_per_orient * wwr[o] for o in wwr}
    a_opaque = {o: a_wall_per_orient * (1 - wwr[o]) for o in wwr}

    u_op = u_value_opaque(design.insulation_m)
    u_gl = design.glazing_props["U"]
    shgc = design.glazing_props["SHGC"]

    # Sum overall envelope conductance
    UA_total = sum(u_op * a_opaque[o] + u_gl * a_glaz[o] for o in wwr)

    # Hourly solar gain through windows (W) — accounts for SHGC and overhang shading
    overhang_factor = 1.0 - min(0.5, design.overhang_ratio * 0.7)  # vertical projection
    q_solar = np.zeros(n_hours)
    for o in wwr:
        poa = _cached_poa(weather, o)
        q_solar += a_glaz[o] * shgc * overhang_factor * poa

    # Infiltration heat-transfer coefficient
    zone_volume = floor_area_m2 * facade_height_m
    UA_infil = (ACH_INFILTRATION * zone_volume / 3600.0) * RHO_AIR_KG_M3 * CP_AIR_J_KG_K

    # Internal gains
    q_internal = INTERNAL_GAINS_W_M2 * floor_area_m2

    # Hourly energy balance with ideal deadband:
    # Q_loss = (UA_total + UA_infil) * (Tair - Te)
    # Q_gain = q_solar + q_internal
    # Setpoint: heat to T_h if free-floating < T_h, cool to T_c if free-floating > T_c.
    UA = UA_total + UA_infil
    free_air_temp = Te + (q_solar + q_internal) / UA
    heat_load_w = np.maximum(0.0, UA * (HEATING_SETPOINT_C - free_air_temp))
    cool_load_w = np.maximum(0.0, UA * (free_air_temp - COOLING_SETPOINT_C))

    # Energy in kWh (1 hour timestep)
    heat_kwh = heat_load_w.sum() / 1000.0
    cool_kwh = cool_load_w.sum() / 1000.0

    return {
        "heating_eui_kwh_m2_yr": heat_kwh / floor_area_m2,
        "cooling_eui_kwh_m2_yr": cool_kwh / floor_area_m2,
        "total_eui_kwh_m2_yr": (heat_kwh + cool_kwh) / floor_area_m2,
        "u_eff_w_m2k_path_b": float(UA_total / (4 * a_wall_per_orient)),  # area-weighted
        "annual_solar_gain_kwh_m2_yr": (q_solar.sum() / 1000.0) / floor_area_m2,
    }
