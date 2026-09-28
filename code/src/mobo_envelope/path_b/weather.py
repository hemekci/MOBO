"""TMY3 .epw reader producing hourly arrays for thermal/daylight simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "weather"


@dataclass(frozen=True)
class WeatherData:
    """Hourly weather data for one site (8760 entries)."""

    city: str
    latitude: float
    longitude: float
    elevation_m: float
    timezone_h: float
    dry_bulb_c: np.ndarray         # outdoor air temperature, deg C
    dew_point_c: np.ndarray        # deg C
    rel_humidity_pct: np.ndarray
    pressure_pa: np.ndarray
    extraterrestrial_horiz_wh_m2: np.ndarray
    direct_normal_wh_m2: np.ndarray
    diffuse_horiz_wh_m2: np.ndarray
    global_horiz_wh_m2: np.ndarray
    wind_speed_m_s: np.ndarray
    wind_direction_deg: np.ndarray

    def __len__(self) -> int:
        return len(self.dry_bulb_c)

    @property
    def hours(self) -> pd.DatetimeIndex:
        # TMY3: 8760 hourly values starting Jan 1 01:00 (end-of-hour stamps)
        return pd.date_range("2024-01-01 01:00", periods=8760, freq="h")


def read_epw(path: Path | str) -> WeatherData:
    """Parse an EPW file. Header is 8 lines; data is 8760 rows comma-separated."""
    path = Path(path)
    with open(path) as f:
        header = [next(f) for _ in range(8)]
    loc_line = header[0].strip().split(",")
    # LOCATION,city,state,country,source,wmo_id,latitude,longitude,timezone,elevation
    city = loc_line[1].strip()
    latitude = float(loc_line[6])
    longitude = float(loc_line[7])
    timezone_h = float(loc_line[8])
    elevation = float(loc_line[9])

    # Data columns (EPW spec, 0-indexed):
    # 0:Year 1:Month 2:Day 3:Hour 4:Minute 5:DataSourceFlags
    # 6:DryBulb 7:DewPoint 8:RelHum 9:AtmosPressure
    # 10:ExtraterrestrialHorizRadn 11:ExtraterrestrialDirectNormalRadn
    # 12:HorizIRSky 13:GlobalHorizontalRadn 14:DirectNormalRadn 15:DiffuseHorizontalRadn
    # 21:WindDir 22:WindSpeed
    df = pd.read_csv(path, header=None, skiprows=8)
    return WeatherData(
        city=city,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation,
        timezone_h=timezone_h,
        dry_bulb_c=df[6].to_numpy(),
        dew_point_c=df[7].to_numpy(),
        rel_humidity_pct=df[8].to_numpy(),
        pressure_pa=df[9].to_numpy(),
        extraterrestrial_horiz_wh_m2=df[10].to_numpy(),
        direct_normal_wh_m2=df[14].to_numpy(),
        diffuse_horiz_wh_m2=df[15].to_numpy(),
        global_horiz_wh_m2=df[13].to_numpy(),
        wind_speed_m_s=df[21].to_numpy(),
        wind_direction_deg=df[20].to_numpy(),
    )


_CITY_TO_FILE = {
    "Houston":     "Houston.epw",
    "NYC":         "NYC.epw",
    "Minneapolis": "Minneapolis.epw",
}


def load_weather(city: str) -> WeatherData:
    return read_epw(WEATHER_DIR / _CITY_TO_FILE[city])
