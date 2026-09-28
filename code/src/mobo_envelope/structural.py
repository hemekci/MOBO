"""Structural objective: mullion utilization under wind + gravity.

Closed-form beam analysis of a vertical curtain-wall mullion as a simply
supported beam over a typical 4 m floor-to-floor span. Wind load is the
governing case; gravity contributes the dead weight of glazing + framing.

Utilization ratio = applied_stress / yield_stress; reserve = 1 - utilization.
"""

from __future__ import annotations

from mobo_envelope.constants import WIND_PRESSURE_KPA
from mobo_envelope.envelope import EnvelopeDesign

# Geometry constants (representative curtain-wall mullion)
SPAN_M: float = 4.2           # floor-to-floor mullion span (typical commercial)
TRIB_WIDTH_M: float = 2.0     # tributary width per mullion (typical 6 ft module)
WALL_THICKNESS_M: float = 0.025  # box-section wall thickness, mullion


def _section_modulus(member_depth_m: float, wall_thickness_m: float = WALL_THICKNESS_M) -> float:
    """Approximate W = b*d^2/6 for a rectangular hollow box mullion (m^3)."""
    b = wall_thickness_m
    d = member_depth_m
    # For a thin-walled box: I ~ (b*d^3 - (b-2t)*(d-2t)^3) / 12.
    # Using a solid-rectangular surrogate b*d^2/6 as a slightly conservative
    # section modulus for ranking purposes.
    return b * d * d / 6.0


def utilization(design: EnvelopeDesign, city: str) -> float:
    """Compute mullion utilization ratio (applied/yield)."""
    wind_kpa = WIND_PRESSURE_KPA[city]
    w_wind_kn_per_m = wind_kpa * TRIB_WIDTH_M  # uniformly distributed line load
    # Bending moment from wind: M = w*L^2/8 (kN.m)
    m_wind = w_wind_kn_per_m * SPAN_M * SPAN_M / 8.0
    # Convert to N.m
    m_wind_nm = m_wind * 1000.0

    # Dead load contribution (glazing + framing self-weight) gives axial
    # compression that we approximate as additional bending via P-delta.
    # Simplified: increase bending moment by 5% to account for axial effect.
    m_design_nm = m_wind_nm * 1.05

    w_section = _section_modulus(design.member_depth_m)
    stress_pa = m_design_nm / w_section  # N/m^2
    stress_mpa = stress_pa / 1e6
    fy_mpa = design.framing_props["fy_MPa"]

    return stress_mpa / fy_mpa


def reserve(design: EnvelopeDesign, city: str) -> float:
    """Structural reserve = 1 - utilization. Higher is better."""
    return 1.0 - utilization(design, city)
