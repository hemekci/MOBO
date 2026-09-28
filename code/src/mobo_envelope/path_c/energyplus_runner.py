"""Path-C EnergyPlus runner: modify a known-working example IDF via eppy.

Base file: ExampleFiles/5ZoneAirCooled.idf — 5 thermal zones, ideal HVAC,
detailed surfaces, validated against EnergyPlus 26.1 schema.

Modifications per design:
  1. Insulation material IN46: thickness <- design.insulation_m,
     conductivity <- LAMBDA_INSULATION_W_MK.
  2. Replace window construction with WindowMaterial:SimpleGlazingSystem
     using design's U / SHGC / VT.
  3. Window-to-wall ratio is enforced by the example file's existing
     fenestration (~0.4 average); we report the TOTAL annual EUI.

Floor area is the example's ~232 m^2 conditioned floor.
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
EXAMPLE_IDF = ENERGYPLUS_HOME / "ExampleFiles" / "5ZoneAirCooled.idf"
WEATHER_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "weather"

CITY_TO_EPW = {
    "Houston":     WEATHER_DIR / "Houston.epw",
    "NYC":         WEATHER_DIR / "NYC.epw",
    "Minneapolis": WEATHER_DIR / "Minneapolis.epw",
}

_FLOOR_AREA_M2: float = 463.6  # 5ZoneAirCooled total conditioned floor area


def _setup_eppy():
    from eppy.modeleditor import IDF
    if getattr(IDF, "_idd_set", False) is not True:
        IDF.setiddname(str(IDD_PATH))
        IDF._idd_set = True
    return IDF


_ORIENTATION_TO_BUILDING_SURFACE = {
    # FRONT-1 faces south (positive y is north in 5ZoneAirCooled), confirm via Site axis = 30 deg
    "FRONT-1":  "south",
    "RIGHT-1":  "east",
    "BACK-1":   "north",
    "LEFT-1":   "west",
}


def _scale_window_to_wwr(fen, target_wwr: float, original_wwr: float) -> None:
    """Scale a FenestrationSurface:Detailed in-place to hit target_wwr.

    Window vertex order in this IDF is upper-left, lower-left, lower-right, upper-right
    (counter-clockwise in WORLD coords). We scale only the Z range (height) symmetrically
    around the existing center so wall position is preserved.
    """
    if original_wwr <= 0 or target_wwr <= 0:
        return
    z_coords = [
        float(fen.Vertex_1_Zcoordinate),
        float(fen.Vertex_2_Zcoordinate),
        float(fen.Vertex_3_Zcoordinate),
        float(fen.Vertex_4_Zcoordinate),
    ]
    z_top = max(z_coords)
    z_bot = min(z_coords)
    z_mid = (z_top + z_bot) / 2.0
    h_old = z_top - z_bot
    scale = target_wwr / original_wwr
    h_new = max(0.3, min(2.7, h_old * scale))
    z_top_new = z_mid + h_new / 2.0
    z_bot_new = z_mid - h_new / 2.0
    # Reassign Z keeping x/y intact: top vertices keep z_top_new, bottom keep z_bot_new
    for i, z in enumerate(z_coords, start=1):
        new_z = z_top_new if z == z_top else z_bot_new
        setattr(fen, f"Vertex_{i}_Zcoordinate", new_z)


def _wall_area(idf, surface_name: str) -> float:
    surf = next((s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"] if s.Name == surface_name), None)
    if surf is None:
        return 0.0
    # Use shoelace via x,y,z; for vertical wall, area = width * height computed from coords
    # For 5ZoneAirCooled walls are rectangles; take vertex 1 and 3 to get width and height
    x1, y1, z1 = float(surf.Vertex_1_Xcoordinate), float(surf.Vertex_1_Ycoordinate), float(surf.Vertex_1_Zcoordinate)
    x3, y3, z3 = float(surf.Vertex_3_Xcoordinate), float(surf.Vertex_3_Ycoordinate), float(surf.Vertex_3_Zcoordinate)
    width = ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5
    height = abs(z3 - z1) if z3 != z1 else 3.0  # default height
    return width * height


def _window_area(fen) -> float:
    xs = [float(getattr(fen, f"Vertex_{i}_Xcoordinate")) for i in (1, 2, 3, 4)]
    ys = [float(getattr(fen, f"Vertex_{i}_Ycoordinate")) for i in (1, 2, 3, 4)]
    zs = [float(getattr(fen, f"Vertex_{i}_Zcoordinate")) for i in (1, 2, 3, 4)]
    width = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    height = max(zs) - min(zs)
    return width * height


def _modify_idf_for_design(idf, design: EnvelopeDesign) -> None:
    """Inject envelope-specific properties into a loaded 5ZoneAirCooled IDF."""
    insul_t = max(0.005, design.insulation_m)
    glz = GLAZING[design.glazing_type]

    # 1. Modify insulation material IN46 (the wall insulation in this example)
    for mat in idf.idfobjects["MATERIAL"]:
        if mat.Name == "IN46":
            mat.Thickness = insul_t
            mat.Conductivity = LAMBDA_INSULATION_W_MK
            break

    # 2. Replace window glazing with SimpleGlazingSystem
    # First, drop existing WindowMaterial:Glazing and Gas to avoid name clashes
    # (we'll keep the layer names but redirect constructions to our new mat).
    new_name = "MOBO_Glazing"
    # Remove any prior occurrence of MOBO_Glazing
    for sg in list(idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"]):
        if sg.Name == new_name:
            idf.removeidfobject(sg)
    idf.newidfobject(
        "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
        Name=new_name,
        UFactor=float(glz["U"]),
        Solar_Heat_Gain_Coefficient=float(glz["SHGC"]),
        Visible_Transmittance=float(glz["VT"]),
    )

    # 3. Replace window constructions: remove the old multilayer ones and
    # create fresh single-layer constructions pointing at MOBO_Glazing.
    window_construction_names: list[str] = []
    for con in list(idf.idfobjects["CONSTRUCTION"]):
        nm = con.Name.upper()
        if "DBL" in nm or "GLZ" in nm or "GRY" in nm or "GREY" in nm or "WIN" in nm or "GLAZ" in nm or "SGL" in nm:
            window_construction_names.append(con.Name)
            idf.removeidfobject(con)

    # Recreate each window construction as single-layer
    for name in window_construction_names:
        idf.newidfobject(
            "CONSTRUCTION",
            Name=name,
            Outside_Layer=new_name,
        )

    # 4. Scale windows to match per-orientation WWR
    orient_to_wwr = {
        "south": design.wwr_south,
        "north": design.wwr_north,
        "east": design.wwr_east,
        "west": design.wwr_west,
    }
    # Compute per-wall original WWR
    wall_total: dict[str, float] = {}
    win_total: dict[str, float] = {}
    for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        bs = fen.Building_Surface_Name
        wall_total.setdefault(bs, _wall_area(idf, bs))
        win_total[bs] = win_total.get(bs, 0.0) + _window_area(fen)
    original_wwr = {
        bs: (win_total.get(bs, 0.0) / wall_total[bs] if wall_total[bs] > 0 else 0.4)
        for bs in wall_total
    }
    # Scale each window: only those marked as windows (skip doors)
    for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        if fen.Surface_Type.lower() != "window":
            continue
        bs = fen.Building_Surface_Name
        orient = _ORIENTATION_TO_BUILDING_SURFACE.get(bs, "south")
        target_wwr = orient_to_wwr[orient]
        _scale_window_to_wwr(fen, target_wwr, original_wwr.get(bs, 0.4))


@dataclass
class EplusResult:
    heating_eui_kwh_m2_yr: float
    cooling_eui_kwh_m2_yr: float
    total_eui_kwh_m2_yr: float
    runtime_s: float
    success: bool
    error: str = ""


def run_eplus(design: EnvelopeDesign, city: str, keep_dir: bool = False, debug: bool = False) -> EplusResult:
    """Run a single EnergyPlus annual simulation and return EUIs."""
    epw = CITY_TO_EPW[city]
    if not epw.exists():
        return EplusResult(0, 0, 0, 0.0, False, f"missing weather: {epw}")

    IDF = _setup_eppy()
    idf = IDF(str(EXAMPLE_IDF))
    _modify_idf_for_design(idf, design)

    workdir = Path(tempfile.mkdtemp(prefix="eplus_"))
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
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, timeout=600, text=True)
    except subprocess.TimeoutExpired:
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)
        return EplusResult(0, 0, 0, time.time() - t0, False, "timeout")

    elapsed = time.time() - t0

    if proc.returncode != 0:
        err_path = workdir / "eplusout.err"
        err_text = err_path.read_text()[-3000:] if err_path.exists() else ""
        full_err = f"--- STDERR ---\n{(proc.stderr or '')[-1500:]}\n--- ERR FILE ---\n{err_text}"
        if debug:
            print(full_err)
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)
        return EplusResult(0, 0, 0, elapsed, False, full_err)

    eso = workdir / "eplusout.csv"
    if not eso.exists():
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)
        return EplusResult(0, 0, 0, elapsed, False, "no eplusout.csv")

    import pandas as pd
    df = pd.read_csv(eso)
    # 5ZoneAirCooled outputs zone-level sensible heating/cooling rates in [W].
    # Each row is one hour, so summing W across rows gives Wh; divide by 1000.
    heat_cols = [c for c in df.columns
                 if "Zone Air System Sensible Heating Rate" in c
                 and "PLENUM" not in c.upper()]
    cool_cols = [c for c in df.columns
                 if "Zone Air System Sensible Cooling Rate" in c
                 and "PLENUM" not in c.upper()]

    heat_kwh = (df[heat_cols].sum().sum() / 1000.0) if heat_cols else 0.0
    cool_kwh = (df[cool_cols].sum().sum() / 1000.0) if cool_cols else 0.0

    if not keep_dir:
        shutil.rmtree(workdir, ignore_errors=True)

    return EplusResult(
        heating_eui_kwh_m2_yr=float(heat_kwh / _FLOOR_AREA_M2),
        cooling_eui_kwh_m2_yr=float(cool_kwh / _FLOOR_AREA_M2),
        total_eui_kwh_m2_yr=float((heat_kwh + cool_kwh) / _FLOOR_AREA_M2),
        runtime_s=elapsed,
        success=True,
    )
