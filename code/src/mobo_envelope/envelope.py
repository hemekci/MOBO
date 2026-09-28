"""Parametric envelope decoding.

The optimizer works in a continuous design space. ``decode_design`` maps a
real-valued vector ``x`` to an :class:`EnvelopeDesign` with categorical and
continuous attributes resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mobo_envelope.constants import GLAZING, FRAMING

# Categorical mappings: integer index -> name (used by decode_design)
GLAZING_TYPES: list[str] = ["double_lowe", "triple_lowe", "vacuum"]
FRAMING_MATERIALS: list[str] = ["steel", "aluminum", "timber", "frp"]

# Variable order in x. Continuous variables in [0, 1]; categorical encoded as
# integer in [0, n_categories - 1].
VARIABLE_NAMES: list[str] = [
    "wwr_north",         # 0 [0.10, 0.80]
    "wwr_south",         # 1 [0.10, 0.80]
    "wwr_east",          # 2 [0.10, 0.80]
    "wwr_west",          # 3 [0.10, 0.80]
    "insulation_m",      # 4 [0.05, 0.30] m
    "glazing_idx",       # 5 [0, 2] integer
    "member_depth_m",    # 6 [0.05, 0.30] m
    "framing_idx",       # 7 [0, 3] integer
    "overhang_ratio",    # 8 [0.0, 0.6] depth/window-height
    "panel_thickness_m", # 9 [0.003, 0.012] m, opaque cladding panel
]
N_VARS: int = len(VARIABLE_NAMES)

# Bounds (inclusive). Categorical bounds are [0, n_cat - 1]; we use "real"
# coding and round inside ``decode_design`` so pymoo can treat the search
# space as a real-valued box.
LOWER: np.ndarray = np.array([0.10, 0.10, 0.10, 0.10, 0.05, 0.0, 0.05, 0.0, 0.0,  0.003])
UPPER: np.ndarray = np.array([0.80, 0.80, 0.80, 0.80, 0.30, 2.999, 0.30, 3.999, 0.6, 0.012])


@dataclass(frozen=True)
class EnvelopeDesign:
    """Decoded envelope design."""

    wwr_north: float
    wwr_south: float
    wwr_east: float
    wwr_west: float
    insulation_m: float
    glazing_type: str
    member_depth_m: float
    framing_material: str
    overhang_ratio: float
    panel_thickness_m: float

    @property
    def wwr_avg(self) -> float:
        return float(np.mean([self.wwr_north, self.wwr_south, self.wwr_east, self.wwr_west]))

    @property
    def glazing_props(self) -> dict[str, float]:
        return GLAZING[self.glazing_type]

    @property
    def framing_props(self) -> dict[str, float]:
        return FRAMING[self.framing_material]


def decode_design(x: np.ndarray) -> EnvelopeDesign:
    """Decode a real-valued design vector into an :class:`EnvelopeDesign`."""
    if x.shape[-1] != N_VARS:
        raise ValueError(f"x must have {N_VARS} variables, got {x.shape[-1]}")
    glazing_idx = int(np.clip(np.floor(x[5]), 0, len(GLAZING_TYPES) - 1))
    framing_idx = int(np.clip(np.floor(x[7]), 0, len(FRAMING_MATERIALS) - 1))
    return EnvelopeDesign(
        wwr_north=float(x[0]),
        wwr_south=float(x[1]),
        wwr_east=float(x[2]),
        wwr_west=float(x[3]),
        insulation_m=float(x[4]),
        glazing_type=GLAZING_TYPES[glazing_idx],
        member_depth_m=float(x[6]),
        framing_material=FRAMING_MATERIALS[framing_idx],
        overhang_ratio=float(x[8]),
        panel_thickness_m=float(x[9]),
    )
