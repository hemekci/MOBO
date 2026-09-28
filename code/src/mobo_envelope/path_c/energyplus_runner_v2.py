"""Path-C EnergyPlus runner v2: DOE Reference Small Office (multi-zone).

Replaces the canonical 5ZoneAirCooled.idf with the DOE Commercial Reference
Building "Small Office, Post-2004" template (RefBldgSmallOfficeNew2004),
so that the verification model represents a realistic multi-zone case-study
building.

Building characteristics (from DOE Commercial Prototype Buildings, 2014):
  - 5 thermal zones (4 perimeter + core)
  - Conditioned floor area: 511.16 m^2
  - PSZ-AC HVAC system with realistic sizing
  - ASHRAE 90.1-2004 schedules (occupancy, lighting, equipment)
  - Detailed envelope: mass concrete + insulation + gypsum interior

Design overrides per call:
  1. "Mass NonRes Wall Insulation" thickness <- design.insulation_m
  2. Window construction "Window Non-res Fixed" -> SimpleGlazingSystem
     with design's U / SHGC / VT
  3. Per-orientation WWR via Z-scaling of FenestrationSurface:Detailed
     vertices, parsing orientation from surface names "wall_north" etc.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from mobo_envelope.constants import GLAZING, LAMBDA_INSULATION_W_MK
from mobo_envelope.envelope import EnvelopeDesign

ENERGYPLUS_HOME = Path(os.environ.get(
    "ENERGYPLUS_HOME",
    str(Path.home() / "local" / "EnergyPlus-26.1.0-6f2e40d102-Darwin-macOS13-arm64"),
))
ENERGYPLUS_BIN = ENERGYPLUS_HOME / "energyplus"
IDD_PATH = ENERGYPLUS_HOME / "Energy+.idd"
EXAMPLE_IDF = ENERGYPLUS_HOME / "ExampleFiles" / "RefBldgSmallOfficeNew2004_Chicago.idf"
WEATHER_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "weather"

CITY_TO_EPW = {
    "Houston":     WEATHER_DIR / "Houston.epw",
    "NYC":         WEATHER_DIR / "NYC.epw",
    "Minneapolis": WEATHER_DIR / "Minneapolis.epw",
}

# DOE Reference Small Office: 27.69 m W x 18.46 m D x 3.05 m H, single storey
_FLOOR_AREA_M2: float = 511.16

# The reference building has 4 perimeter zones (one per cardinal orientation)
# plus a core zone. Window names embed orientation.
_ORIENT_TOKENS = ("south", "north", "east", "west")


def _setup_eppy():
    from eppy.modeleditor import IDF
    if getattr(IDF, "_idd_set", False) is not True:
        IDF.setiddname(str(IDD_PATH))
        IDF._idd_set = True
    return IDF


def _orientation_from_surface(name: str) -> str:
    nm = name.lower()
    for tok in _ORIENT_TOKENS:
        if tok in nm:
            return tok
    return "south"


def _wall_area(idf, surface_name: str) -> float:
    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if s.Name == surface_name:
            xs = [float(getattr(s, f"Vertex_{i}_Xcoordinate")) for i in range(1, 5)]
            ys = [float(getattr(s, f"Vertex_{i}_Ycoordinate")) for i in range(1, 5)]
            zs = [float(getattr(s, f"Vertex_{i}_Zcoordinate")) for i in range(1, 5)]
            # Approximate as planar quad: width = max horizontal extent, height = z range
            w = max(max(xs) - min(xs), max(ys) - min(ys))
            h = max(zs) - min(zs)
            return float(w * h)
    return 0.0


def _window_area(fen) -> float:
    xs = [float(getattr(fen, f"Vertex_{i}_Xcoordinate")) for i in range(1, 5)]
    ys = [float(getattr(fen, f"Vertex_{i}_Ycoordinate")) for i in range(1, 5)]
    zs = [float(getattr(fen, f"Vertex_{i}_Zcoordinate")) for i in range(1, 5)]
    w = max(max(xs) - min(xs), max(ys) - min(ys))
    h = max(zs) - min(zs)
    return float(w * h)


def _scale_window_to_wwr(fen, target_wwr: float, original_wwr: float) -> None:
    """Scale a window's height symmetrically about its centre to hit target_wwr.

    Width is preserved to keep mullion patterns recognisable. We bound the
    final height to the parent wall (max ~2.7 m of the 3.05 m wall).
    """
    if original_wwr <= 0 or target_wwr <= 0:
        return
    z = [float(getattr(fen, f"Vertex_{i}_Zcoordinate")) for i in range(1, 5)]
    z_top = max(z)
    z_bot = min(z)
    z_mid = (z_top + z_bot) / 2.0
    h_old = z_top - z_bot
    scale = target_wwr / original_wwr
    h_new = max(0.3, min(2.7, h_old * scale))
    z_top_new = z_mid + h_new / 2.0
    z_bot_new = z_mid - h_new / 2.0
    for i, z_i in enumerate(z, start=1):
        new_z = z_top_new if z_i == z_top else z_bot_new
        setattr(fen, f"Vertex_{i}_Zcoordinate", new_z)


def _modify_idf_for_design(idf, design: EnvelopeDesign) -> None:
    glz = GLAZING[design.glazing_type]

    # 0. The DOE Reference IDF ships with Weather-File Run Periods disabled
    # (only sizing periods are simulated). Flip the flag so the annual run
    # actually executes against the EPW.
    for sc in idf.idfobjects["SIMULATIONCONTROL"]:
        sc.Run_Simulation_for_Weather_File_Run_Periods = "Yes"

    # 1. Insulation thickness on the rigid-insulation layer
    for mat in idf.idfobjects["MATERIAL"]:
        if mat.Name.strip().lower().startswith("mass nonres wall insulation"):
            mat.Thickness = float(design.insulation_m)
            mat.Conductivity = float(LAMBDA_INSULATION_W_MK)
            break

    # 2. Replace the window construction with SimpleGlazingSystem
    new_glz_name = "MOBO_Glazing"
    for sg in list(idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"]):
        if sg.Name == new_glz_name:
            idf.removeidfobject(sg)
    idf.newidfobject(
        "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
        Name=new_glz_name,
        UFactor=float(glz["U"]),
        Solar_Heat_Gain_Coefficient=float(glz["SHGC"]),
        Visible_Transmittance=float(glz["VT"]),
    )
    win_constructions: list[str] = []
    for con in list(idf.idfobjects["CONSTRUCTION"]):
        nm = con.Name.lower()
        if "window" in nm or "glaz" in nm:
            win_constructions.append(con.Name)
            idf.removeidfobject(con)
    for name in win_constructions:
        idf.newidfobject("CONSTRUCTION", Name=name, Outside_Layer=new_glz_name)

    # 3. Per-orientation WWR scaling
    orient_to_wwr = {
        "south": float(design.wwr_south),
        "north": float(design.wwr_north),
        "east":  float(design.wwr_east),
        "west":  float(design.wwr_west),
    }
    wall_areas: dict[str, float] = {}
    win_areas: dict[str, float] = {}
    for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        if fen.Surface_Type.lower() != "window":
            continue
        bs = fen.Building_Surface_Name
        wall_areas.setdefault(bs, _wall_area(idf, bs))
        win_areas[bs] = win_areas.get(bs, 0.0) + _window_area(fen)
    original_wwr = {
        bs: (win_areas.get(bs, 0.0) / wall_areas[bs] if wall_areas[bs] > 0 else 0.4)
        for bs in wall_areas
    }
    for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        if fen.Surface_Type.lower() != "window":
            continue
        bs = fen.Building_Surface_Name
        orient = _orientation_from_surface(bs)
        target_wwr = orient_to_wwr.get(orient, 0.4)
        _scale_window_to_wwr(fen, target_wwr, original_wwr.get(bs, 0.4))


@dataclass
class EplusResultV2:
    heating_eui_kwh_m2_yr: float
    cooling_eui_kwh_m2_yr: float
    total_eui_kwh_m2_yr: float
    runtime_s: float
    success: bool
    error: str = ""


def run_eplus_v2(design: EnvelopeDesign, city: str,
                 keep_dir: bool = False, debug: bool = False) -> EplusResultV2:
    """Run a single annual simulation on DOE RefBldg Small Office."""
    epw = CITY_TO_EPW[city]
    if not epw.exists():
        return EplusResultV2(0, 0, 0, 0.0, False, f"missing weather: {epw}")

    IDF = _setup_eppy()
    idf = IDF(str(EXAMPLE_IDF))
    _modify_idf_for_design(idf, design)

    workdir = Path(tempfile.mkdtemp(prefix="eplus_v2_"))
    idf_path = workdir / "in.idf"
    idf.saveas(str(idf_path))

    cmd = [
        str(ENERGYPLUS_BIN),
        "--idd", str(IDD_PATH),
        "--weather", str(epw),
        "--output-directory", str(workdir),
        "--readvars",
        str(idf_path),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, timeout=900, text=True)
    except subprocess.TimeoutExpired:
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)
        return EplusResultV2(0, 0, 0, time.time() - t0, False, "timeout")

    elapsed = time.time() - t0

    if proc.returncode != 0:
        err_path = workdir / "eplusout.err"
        err_text = err_path.read_text()[-3000:] if err_path.exists() else ""
        full_err = f"--- STDERR ---\n{(proc.stderr or '')[-1500:]}\n--- ERR FILE ---\n{err_text}"
        if debug:
            print(full_err)
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)
        return EplusResultV2(0, 0, 0, elapsed, False, full_err)

    # Use the meter output: aggregate the District Heating and District Cooling
    # equivalents via the Heating:Electricity, Cooling:Electricity, and
    # NaturalGas:Facility meters as exported in eplusout.csv.
    eso = workdir / "eplusout.csv"
    if not eso.exists():
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)
        return EplusResultV2(0, 0, 0, elapsed, False, "no eplusout.csv")

    import pandas as pd
    df = pd.read_csv(eso)
    # The DOE small office reports hourly meters in J. 1 J = 1/3.6e6 kWh.
    heat_e_cols = [c for c in df.columns if "Heating:Electricity" in c and "[J]" in c]
    cool_e_cols = [c for c in df.columns if "Cooling:Electricity" in c and "[J]" in c]
    heat_g_cols = [c for c in df.columns if "Heating:NaturalGas" in c and "[J]" in c]

    j_to_kwh = 1.0 / 3.6e6
    heat_kwh = (df[heat_e_cols].sum().sum() + df[heat_g_cols].sum().sum()) * j_to_kwh \
        if (heat_e_cols or heat_g_cols) else 0.0
    cool_kwh = df[cool_e_cols].sum().sum() * j_to_kwh if cool_e_cols else 0.0

    if not keep_dir:
        shutil.rmtree(workdir, ignore_errors=True)

    return EplusResultV2(
        heating_eui_kwh_m2_yr=float(heat_kwh / _FLOOR_AREA_M2),
        cooling_eui_kwh_m2_yr=float(cool_kwh / _FLOOR_AREA_M2),
        total_eui_kwh_m2_yr=float((heat_kwh + cool_kwh) / _FLOOR_AREA_M2),
        runtime_s=elapsed,
        success=True,
    )
