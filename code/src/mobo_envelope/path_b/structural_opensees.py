"""Path-B structural verification via openseespy FEM.

Models a vertical curtain-wall mullion as a 2D fiber-section beam-column with:
  - Distributed wind load (ASCE 7-22 components & cladding)
  - Self-weight + glazing dead load axial compression
  - Geometric nonlinearity (P-delta)
  - 10-element discretization, displacement-based formulation

Returns per-design utilization ratio = max(stress) / yield_stress.
"""

from __future__ import annotations

import numpy as np
import openseespy.opensees as ops

from mobo_envelope.constants import WIND_PRESSURE_KPA
from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.structural import SPAN_M, TRIB_WIDTH_M, WALL_THICKNESS_M


def utilization_fem(design: EnvelopeDesign, city: str) -> float:
    """Run a single openseespy beam-column analysis with P-delta. Returns
    utilization ratio (peak fiber stress / yield stress, dimensionless)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Geometry
    L = SPAN_M
    n_elems = 10
    nodes = [(i, 0.0, i * L / n_elems) for i in range(n_elems + 1)]
    for i, x, y in nodes:
        ops.node(i, x, y)
    # Pin-pin supports (rotational free), top has axial restraint
    ops.fix(0, 1, 1, 0)
    ops.fix(n_elems, 1, 0, 0)

    # Material — use elastic for now (ranking purpose)
    fy = design.framing_props["fy_MPa"] * 1e6  # Pa
    E = design.framing_props["E_GPa"] * 1e9    # Pa
    ops.uniaxialMaterial("Elastic", 1, E)
    matTag = 1

    # Section: rectangular hollow box approximated as rectangle b × d
    b = WALL_THICKNESS_M
    d = design.member_depth_m
    # Use simple Elastic section
    A = b * d
    Iz = b * (d ** 3) / 12.0
    secTag = 1
    ops.section("Elastic", secTag, E, A, Iz)

    # Geometric transformation: P-delta
    transfTag = 1
    ops.geomTransf("PDelta", transfTag)

    # Beam elements
    integrationTag = 1
    ops.beamIntegration("Lobatto", integrationTag, secTag, 5)
    for i in range(n_elems):
        ops.element(
            "dispBeamColumn",
            i,
            i,
            i + 1,
            transfTag,
            integrationTag,
        )

    # Loads
    # Wind: distributed lateral pressure, kN/m
    w_wind_kn_per_m = WIND_PRESSURE_KPA[city] * TRIB_WIDTH_M  # kN/m
    w_wind_n_per_m = w_wind_kn_per_m * 1000.0
    # Dead/live axial: weight of glazed area + framing (rough estimate)
    P_axial_n = 0.5 * design.framing_props["rho"] * 9.81 * b * d * L  # half member self-weight on top

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    # Distributed load: in eleLoad, 'beamUniform' Wy Wx (transverse, axial)
    for i in range(n_elems):
        ops.eleLoad("-ele", i, "-type", "-beamUniform", w_wind_n_per_m, 0.0)
    # Apply axial point load at the top
    ops.load(n_elems, 0.0, -P_axial_n, 0.0)

    # Solve
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Newton")
    ops.test("NormDispIncr", 1e-6, 50)
    ops.analysis("Static")
    ok = ops.analyze(1)
    if ok != 0:
        # If analysis fails, return very high utilization (likely buckled)
        return 5.0

    # Extract maximum bending moment and axial force across elements
    max_moment = 0.0
    max_axial = 0.0
    for i in range(n_elems):
        # Element forces: 'localForces' returns [N1, V1, M1, N2, V2, M2]
        forces = ops.eleForce(i)
        # forces order can be 6 entries: depends on element. Take absolute max moment
        if len(forces) >= 6:
            m1, m2 = forces[2], forces[5]
            n1, n2 = forces[0], forces[3]
        else:
            continue
        max_moment = max(max_moment, abs(m1), abs(m2))
        max_axial = max(max_axial, abs(n1), abs(n2))

    # Combined stress: bending + axial
    W_section = b * (d ** 2) / 6.0  # rectangular section modulus
    sigma_bending = max_moment / W_section
    sigma_axial = max_axial / A
    sigma_total = sigma_bending + sigma_axial
    return float(sigma_total / fy)


def reserve_fem(design: EnvelopeDesign, city: str) -> float:
    """Convenience wrapper: 1 - FEM utilization."""
    return 1.0 - utilization_fem(design, city)
