"""Case study definitions: Houston, NYC, Minneapolis."""

from __future__ import annotations

from dataclasses import dataclass

from mobo_envelope.constants import ASHRAE_BASELINES


@dataclass(frozen=True)
class Case:
    name: str
    city: str
    climate_zone: str
    typology: str
    floor_area_m2: float
    n_design_vars: int
    description: str

    @property
    def baseline(self) -> dict[str, float]:
        return ASHRAE_BASELINES[self.climate_zone]


CASES: dict[str, Case] = {
    "A": Case(
        name="A_Houston",
        city="Houston",
        climate_zone="2A",
        typology="High-rise office tower",
        floor_area_m2=42000.0,
        n_design_vars=10,
        description="30-story commercial office tower, hot-humid climate (29.76N, 95.37W)",
    ),
    "B": Case(
        name="B_NYC",
        city="NYC",
        climate_zone="4A",
        typology="Mixed-use mid-rise",
        floor_area_m2=8500.0,
        n_design_vars=10,
        description="7-story mixed-use, temperate climate (40.71N, 74.01W)",
    ),
    "C": Case(
        name="C_Minneapolis",
        city="Minneapolis",
        climate_zone="6A",
        typology="Single-family dwelling",
        floor_area_m2=220.0,
        n_design_vars=10,
        description="2-story single-family residence, cold continental (44.98N, 93.27W)",
    ),
}
