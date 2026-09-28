"""Cost objective: envelope material + installation cost per floor area.

Linear sum of:
  - glazing area * unit cost (per glazing type)
  - opaque cladding area * cladding unit cost
  - insulation volume * insulation unit cost
  - mullion mass * framing material cost per kg
  - installation factor (GC overhead/profit + labor)
"""

from __future__ import annotations

from mobo_envelope.constants import (
    COST_GLAZING_USD_M2,
    COST_INSTALLATION_FACTOR,
    COST_INSULATION_USD_M3,
    COST_OPAQUE_CLADDING_USD_M2,
)
from mobo_envelope.envelope import EnvelopeDesign

# Reference geometry: 1 m^2 of envelope corresponds to a typical floor strip.
# Mullion length per envelope m^2 = 1 / tributary_width = 1 / 1.5 ~ 0.67 m
MULLION_LENGTH_PER_M2: float = 1.0 / 1.5


def cost_per_envelope_m2(design: EnvelopeDesign) -> float:
    """Total installed envelope cost per m^2 of envelope ($/m^2)."""
    wwr = design.wwr_avg

    glazing_cost = COST_GLAZING_USD_M2[design.glazing_type] * wwr
    cladding_cost = COST_OPAQUE_CLADDING_USD_M2 * (1.0 - wwr)
    insulation_cost = COST_INSULATION_USD_M3 * design.insulation_m * (1.0 - wwr)

    # Framing cost: mullion mass per envelope m^2
    section_area = WALL_THICKNESS_M * design.member_depth_m  # m^2
    mullion_mass = section_area * MULLION_LENGTH_PER_M2 * design.framing_props["rho"]  # kg
    framing_cost = mullion_mass * design.framing_props["cost_kg"]

    subtotal = glazing_cost + cladding_cost + insulation_cost + framing_cost
    return subtotal * COST_INSTALLATION_FACTOR


# import here to avoid circular import surface in module top
from mobo_envelope.structural import WALL_THICKNESS_M  # noqa: E402
