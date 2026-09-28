"""Thermal objective: effective envelope U-value.

Implements ISO 6946:2017 method for calculating the steady-state thermal
transmittance of a multi-layer wall and combining wall + glazing into an
area-weighted envelope U-value.
"""

from __future__ import annotations

from mobo_envelope.constants import (
    LAMBDA_INSULATION_W_MK,
    R_FIXED_LAYERS,
    R_SE,
    R_SI,
)
from mobo_envelope.envelope import EnvelopeDesign


def u_value_opaque(insulation_thickness_m: float) -> float:
    """ISO 6946:2017 §6.7. R_total = R_si + R_layers + R_se."""
    r_insulation = insulation_thickness_m / LAMBDA_INSULATION_W_MK
    r_total = R_SI + R_FIXED_LAYERS + r_insulation + R_SE
    return 1.0 / r_total


def u_value_glazing(glazing_type: str, props: dict[str, float]) -> float:
    return props["U"]


def u_eff(design: EnvelopeDesign) -> float:
    """Area-weighted envelope U-value (W/m^2.K).

    Uses the average WWR across orientations on a unit-area envelope.
    """
    wwr = design.wwr_avg
    u_op = u_value_opaque(design.insulation_m)
    u_gl = design.glazing_props["U"]
    return (1.0 - wwr) * u_op + wwr * u_gl
