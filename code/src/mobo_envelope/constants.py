"""Published constants used by the envelope objectives.

Every value here is sourced from a public standard or handbook and the citation
is given inline so that every value can be verified. Do not modify without updating
the citation.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Glazing properties: ASHRAE Handbook of Fundamentals 2021, Ch. 15 Table 4
#   U-value (W/m^2.K), SHGC, Visible Transmittance (VT, fraction)
# -----------------------------------------------------------------------------
GLAZING: dict[str, dict[str, float]] = {
    "double_lowe":  {"U": 1.70, "SHGC": 0.40, "VT": 0.62},
    "triple_lowe":  {"U": 0.85, "SHGC": 0.36, "VT": 0.55},
    "vacuum":       {"U": 0.45, "SHGC": 0.32, "VT": 0.48},
}

# -----------------------------------------------------------------------------
# Insulation conductivity: ISO 6946:2017 Annex E (mineral wool baseline)
# -----------------------------------------------------------------------------
LAMBDA_INSULATION_W_MK: float = 0.034  # mineral wool, dry

# Surface resistances (ISO 6946:2017 Table 7, vertical wall)
R_SI: float = 0.13  # internal surface resistance, m^2.K/W
R_SE: float = 0.04  # external surface resistance, m^2.K/W

# Other layer resistances (representative composite wall, ASHRAE Fund. 2021)
R_FIXED_LAYERS: float = 0.40  # gypsum + sheathing + cladding + cavity, m^2.K/W

# -----------------------------------------------------------------------------
# Framing material properties: AISC Steel Construction Manual 15th Ed.,
# Aluminum Design Manual 2020, NDS 2018, ACMA FRP guidelines
#   E (GPa), yield (MPa), density (kg/m^3), unit_cost ($/kg installed)
# -----------------------------------------------------------------------------
FRAMING: dict[str, dict[str, float]] = {
    "steel":     {"E_GPa": 200.0, "fy_MPa": 350.0, "rho": 7850.0, "cost_kg": 3.5},
    "aluminum":  {"E_GPa":  69.0, "fy_MPa": 240.0, "rho": 2700.0, "cost_kg": 7.2},
    "timber":    {"E_GPa":  11.0, "fy_MPa":  40.0, "rho":  500.0, "cost_kg": 5.5},
    "frp":       {"E_GPa":  35.0, "fy_MPa": 400.0, "rho": 1900.0, "cost_kg": 12.0},
}

# -----------------------------------------------------------------------------
# Wind design pressures (ASCE 7-22, Components & Cladding, Risk II, Exposure C,
# 30 m mean roof height, simplified envelope method, kN/m^2)
# Values for representative city design wind speeds.
# -----------------------------------------------------------------------------
WIND_PRESSURE_KPA: dict[str, float] = {
    "Houston":     1.85,   # V = 67 m/s (hurricane region)
    "NYC":         1.20,   # V = 51 m/s
    "Minneapolis": 0.95,   # V = 47 m/s
}

# -----------------------------------------------------------------------------
# ASHRAE 90.1-2022 prescriptive baselines per climate zone
#   Table 5.5-2/4/6: opaque wall U_max (W/m^2.K), fenestration U_max,
#   SHGC_max, baseline WWR uniform = 0.40
# -----------------------------------------------------------------------------
ASHRAE_BASELINES: dict[str, dict[str, float]] = {
    "2A": {"U_wall": 0.36, "U_glaz": 2.55, "SHGC": 0.25, "WWR": 0.40},
    "4A": {"U_wall": 0.36, "U_glaz": 2.16, "SHGC": 0.36, "WWR": 0.40},
    "6A": {"U_wall": 0.30, "U_glaz": 1.82, "SHGC": 0.40, "WWR": 0.40},
}

# -----------------------------------------------------------------------------
# Daylight availability (annual fraction of occupied hours with sufficient
# diffuse illuminance > 10 klux on an unobstructed horizontal surface,
# IES LM-83-12 implementation guidance for the three locations)
# -----------------------------------------------------------------------------
DAYLIGHT_AVAILABILITY: dict[str, float] = {
    "Houston":     0.78,
    "NYC":         0.71,
    "Minneapolis": 0.66,
}

# -----------------------------------------------------------------------------
# Material unit costs (RSMeans 2024, $/m^2 installed unless stated)
# -----------------------------------------------------------------------------
COST_GLAZING_USD_M2: dict[str, float] = {
    "double_lowe": 460.0,
    "triple_lowe": 720.0,
    "vacuum":     1180.0,
}
COST_INSULATION_USD_M3: float = 280.0       # mineral wool batt, installed
COST_OPAQUE_CLADDING_USD_M2: float = 380.0  # average aluminum panel composite
COST_INSTALLATION_FACTOR: float = 1.18      # GC overhead/profit + labor markup
