"""Daylight objective: spatial Daylight Autonomy (sDA) proxy.

Uses the analytical Daylight Factor approach (Reinhart, 2014) combined with
local diffuse availability to estimate sDA300/50 — the percentage of regularly
occupied floor area that meets >= 300 lux for >= 50 % of occupied hours.

Formula (simplified, conservative):
    DF = 0.1 * VT * WWR_average * (1 - shading_factor)
    sDA = clip(DF * 100 * availability_local + intercept, 0, 100)

This is a fast surrogate; Path B (Radiance) provides ground truth.
"""

from __future__ import annotations

from mobo_envelope.constants import DAYLIGHT_AVAILABILITY
from mobo_envelope.envelope import EnvelopeDesign


def shading_factor(overhang_ratio: float) -> float:
    """Linear penalty: 0% at no overhang, 35% reduction at full 0.6 ratio."""
    return min(0.35, overhang_ratio * 0.58)


def sda(design: EnvelopeDesign, city: str) -> float:
    """Estimate sDA(300lx, 50%) as a percentage in [0, 100]."""
    vt = design.glazing_props["VT"]
    # South WWR drives daylight more than other orientations; weight 1.5x.
    weighted_wwr = (
        design.wwr_north * 1.0
        + design.wwr_south * 1.5
        + design.wwr_east * 1.1
        + design.wwr_west * 1.1
    ) / 4.7

    sf = shading_factor(design.overhang_ratio)
    df_pct = 100.0 * 0.10 * vt * weighted_wwr * (1.0 - sf)
    availability = DAYLIGHT_AVAILABILITY[city]

    # Empirical scaling: DF of ~2 % typically maps to ~50 % sDA at temperate
    # latitudes (Reinhart & Walkenhorst 2001). We map 1 % DF -> ~25 % sDA at
    # availability 0.71 (NYC), with linear scaling by availability.
    sda_estimate = df_pct * 25.0 * availability
    return float(min(100.0, max(0.0, sda_estimate)))
