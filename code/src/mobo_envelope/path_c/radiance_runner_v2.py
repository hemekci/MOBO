"""Path-C Radiance runner v2: four-orientation perimeter zone (multi-zone proxy).

Extends radiance_runner.py: instead of a single south-facing facade, the
Radiance scene now has glazing on ALL four orientations (N, S, E, W) with the
design's per-orientation WWR. Sensors cover the full floor on a 5x5 grid.
A four-orientation scene represents the case-study buildings more faithfully
than a single south-facing perimeter zone.

Pipeline per design (Tregenza 145-bin sky discretisation):
    1. Build single perimeter zone .rad with FOUR walls each carrying
       orientation-specific glazing.
    2. oconv -> octree of static scene.
    3. rfluxmtx -> daylight-coefficient (DC) matrix on a 5x5 sensor grid.
    4. dctimestep DC * sky.smx -> annual hourly illuminance per sensor.
    5. LM-83 sDA: % sensors with >= 300 lx for >= 50 % of occupied hours.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mobo_envelope.constants import GLAZING
from mobo_envelope.envelope import EnvelopeDesign

RADIANCE_HOME = Path(os.environ.get(
    "RADIANCE_HOME",
    str(Path.home() / "local" / "radiance"),
))
RAD_BIN = RADIANCE_HOME / "bin"
RAD_LIB = RADIANCE_HOME / "lib"

# rfluxmtx invokes rcontrib via PATH; ensure both are on PATH and RAYPATH.
_RAD_ENV = {
    **os.environ,
    "PATH": f"{RAD_BIN}:{os.environ.get('PATH', '')}",
    "RAYPATH": f"{RAD_LIB}:.:{os.environ.get('RAYPATH', '')}",
}
WEATHER_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "weather"
SKY_CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "sky_cache"
SKY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CITY_TO_EPW = {
    "Houston":     WEATHER_DIR / "Houston.epw",
    "NYC":         WEATHER_DIR / "NYC.epw",
    "Minneapolis": WEATHER_DIR / "Minneapolis.epw",
}

ILL_THRESHOLD_LX = 300.0
TIME_THRESHOLD_FRAC = 0.50
OCC_HOUR_START = 8
OCC_HOUR_END = 18

ROOM_W = 8.0   # x extent (E-W)
ROOM_D = 8.0   # y extent (S-N)
ROOM_H = 3.0
SENSOR_HEIGHT = 0.75
SENSOR_X = np.linspace(1.0, ROOM_W - 1.0, 5)
SENSOR_Y = np.linspace(1.0, ROOM_D - 1.0, 5)


@dataclass(frozen=True)
class RadianceResultV2:
    sda_pct: float
    n_sensors: int
    success: bool
    runtime_s: float
    error: str = ""


def _bin(name: str) -> str:
    return str(RAD_BIN / name)


def _ensure_sky_matrix(city: str) -> Path:
    smx_path = SKY_CACHE_DIR / f"{city}_tregenza145.smx"
    if smx_path.exists() and smx_path.stat().st_size > 1024:
        return smx_path
    epw = CITY_TO_EPW[city]
    if not epw.exists():
        raise FileNotFoundError(f"Missing weather file: {epw}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wea = tmp / "weather.wea"
        subprocess.run([_bin("epw2wea"), str(epw), str(wea)], check=True, capture_output=True, env=_RAD_ENV)
        with smx_path.open("wb") as f:
            subprocess.run(
                [_bin("gendaymtx"), "-m", "1", "-O1", str(wea)],
                check=True, stdout=f, stderr=subprocess.PIPE, env=_RAD_ENV,
            )
    return smx_path


def _write_sensors(path: Path) -> int:
    rows = []
    for x in SENSOR_X:
        for y in SENSOR_Y:
            rows.append(f"{x:.3f} {y:.3f} {SENSOR_HEIGHT:.3f}  0 0 1")
    path.write_text("\n".join(rows) + "\n")
    return len(rows)


def _wall_with_window(name: str, vt: float, wwr: float,
                      v0: tuple, v1: tuple, v2: tuple, v3: tuple) -> str:
    """Build a wall polygon (v0,v1,v2,v3 counter-clockwise) with a centred
    rectangular window cutout sized to wwr * wall_area.

    Wall vertex convention: v0=lower-left, v1=lower-right, v2=upper-right,
    v3=upper-left (looking from outside).
    """
    v0, v1, v2, v3 = map(np.array, (v0, v1, v2, v3))
    wall_w = float(np.linalg.norm(v1 - v0))
    wall_h = float(np.linalg.norm(v3 - v0))
    win_area = wwr * wall_w * wall_h
    win_w = min(wall_w - 0.4, np.sqrt(win_area * (wall_w / wall_h)))
    win_h = win_area / max(win_w, 1e-6)
    win_h = min(win_h, wall_h - 0.4)
    win_w = win_area / max(win_h, 1e-6)
    # window centre on wall in local coords
    u_c = wall_w / 2.0
    v_c = 1.0 + win_h / 2.0
    u0 = u_c - win_w / 2.0
    u1 = u_c + win_w / 2.0
    vv0 = v_c - win_h / 2.0
    vv1 = v_c + win_h / 2.0
    # local frame: u along (v1-v0), v along (v3-v0)
    u_hat = (v1 - v0) / wall_w
    v_hat = (v3 - v0) / wall_h
    p = lambda u, vv: v0 + u * u_hat + vv * v_hat

    parts = []
    # Bottom strip (full wall width, y=0..vv0)
    parts.append(f"wall_mat polygon {name}_bot\n0\n0\n12\n"
                 f"{_pt(p(0,0))}\n{_pt(p(wall_w,0))}\n"
                 f"{_pt(p(wall_w,vv0))}\n{_pt(p(0,vv0))}\n")
    # Top strip
    parts.append(f"wall_mat polygon {name}_top\n0\n0\n12\n"
                 f"{_pt(p(0,vv1))}\n{_pt(p(wall_w,vv1))}\n"
                 f"{_pt(p(wall_w,wall_h))}\n{_pt(p(0,wall_h))}\n")
    # Left strip
    parts.append(f"wall_mat polygon {name}_left\n0\n0\n12\n"
                 f"{_pt(p(0,vv0))}\n{_pt(p(u0,vv0))}\n"
                 f"{_pt(p(u0,vv1))}\n{_pt(p(0,vv1))}\n")
    # Right strip
    parts.append(f"wall_mat polygon {name}_right\n0\n0\n12\n"
                 f"{_pt(p(u1,vv0))}\n{_pt(p(wall_w,vv0))}\n"
                 f"{_pt(p(wall_w,vv1))}\n{_pt(p(u1,vv1))}\n")
    # Glazing
    parts.append(f"glaz_mat polygon {name}_glazing\n0\n0\n12\n"
                 f"{_pt(p(u0,vv0))}\n{_pt(p(u1,vv0))}\n"
                 f"{_pt(p(u1,vv1))}\n{_pt(p(u0,vv1))}\n")
    return "\n".join(parts)


def _pt(p) -> str:
    return f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"


def _write_room_rad(path: Path, design: EnvelopeDesign) -> None:
    """Single zone with glazing on all four walls per design WWR_{N,S,E,W}.

    Coordinate system: +y north, +x east, +z up. Room footprint
    (0,0) to (ROOM_W, ROOM_D), height 0 to ROOM_H.
    """
    glazing = GLAZING[design.glazing_type]
    vt = float(glazing["VT"])

    parts = []
    parts.append("# Materials")
    parts.append("void plastic wall_mat\n0\n0\n5  0.5 0.5 0.5  0 0\n")
    parts.append("void plastic floor_mat\n0\n0\n5  0.2 0.2 0.2  0 0\n")
    parts.append("void plastic ceiling_mat\n0\n0\n5  0.8 0.8 0.8  0 0\n")
    parts.append("void plastic ground_mat\n0\n0\n5  0.2 0.2 0.2  0 0\n")
    parts.append(f"void glass glaz_mat\n0\n0\n3  {vt:.3f} {vt:.3f} {vt:.3f}\n")

    # Floor (z=0)
    parts.append("floor_mat polygon floor\n0\n0\n12\n"
                 f"0 0 0\n{ROOM_W} 0 0\n{ROOM_W} {ROOM_D} 0\n0 {ROOM_D} 0\n")
    # Ceiling (z=H)
    parts.append("ceiling_mat polygon ceiling\n0\n0\n12\n"
                 f"0 0 {ROOM_H}\n0 {ROOM_D} {ROOM_H}\n"
                 f"{ROOM_W} {ROOM_D} {ROOM_H}\n{ROOM_W} 0 {ROOM_H}\n")

    # South wall (y=0): viewed from outside, lower-left=(W,0,0), lower-right=(0,0,0)
    parts.append(_wall_with_window(
        "south", vt, float(design.wwr_south),
        (ROOM_W, 0, 0), (0, 0, 0), (0, 0, ROOM_H), (ROOM_W, 0, ROOM_H)))
    # North wall (y=D): lower-left=(0,D,0), lower-right=(W,D,0)
    parts.append(_wall_with_window(
        "north", vt, float(design.wwr_north),
        (0, ROOM_D, 0), (ROOM_W, ROOM_D, 0), (ROOM_W, ROOM_D, ROOM_H), (0, ROOM_D, ROOM_H)))
    # East wall (x=W): lower-left=(W,0,0), lower-right=(W,D,0)
    parts.append(_wall_with_window(
        "east", vt, float(design.wwr_east),
        (ROOM_W, 0, 0), (ROOM_W, ROOM_D, 0), (ROOM_W, ROOM_D, ROOM_H), (ROOM_W, 0, ROOM_H)))
    # West wall (x=0): lower-left=(0,D,0), lower-right=(0,0,0)
    parts.append(_wall_with_window(
        "west", vt, float(design.wwr_west),
        (0, ROOM_D, 0), (0, 0, 0), (0, 0, ROOM_H), (0, ROOM_D, ROOM_H)))

    # Ground plane below + around
    parts.append("ground_mat polygon ground\n0\n0\n12\n"
                 f"-30 -30 0\n30 -30 0\n30 30 0\n-30 30 0\n")

    path.write_text("\n".join(parts))


def _write_sky_glow(path: Path) -> None:
    path.write_text(
        "#@rfluxmtx h=u u=Y\n"
        "void glow groundglow\n0\n0\n4 1 1 1 0\n"
        "groundglow source ground\n0\n0\n4 0 0 -1 180\n"
        "#@rfluxmtx h=r1 u=Y\n"
        "void glow skyglow\n0\n0\n4 1 1 1 0\n"
        "skyglow source sky\n0\n0\n4 0 0 1 180\n"
    )


def run_radiance_v2(design: EnvelopeDesign, city: str, n_amb: int = 1) -> RadianceResultV2:
    """Compute LM-83 sDA via DC method on a 4-orientation perimeter zone."""
    t0 = time.time()
    try:
        smx = _ensure_sky_matrix(city)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            room_rad = tmpdir / "room.rad"
            sky_rad = tmpdir / "sky.rad"
            sensors_pts = tmpdir / "sensors.pts"
            scene_oct = tmpdir / "scene.oct"
            dc_mtx = tmpdir / "dc.mtx"

            _write_room_rad(room_rad, design)
            _write_sky_glow(sky_rad)
            n_sensors = _write_sensors(sensors_pts)

            subprocess.run(
                [_bin("oconv"), str(room_rad), str(sky_rad)],
                check=True, stdout=open(scene_oct, "wb"), stderr=subprocess.PIPE,
                env=_RAD_ENV,
            )
            with open(sensors_pts, "rb") as fpts, open(dc_mtx, "wb") as fdc:
                subprocess.run(
                    [_bin("rfluxmtx"),
                     "-y", str(n_sensors),
                     "-I+",
                     "-faa",
                     "-ab", str(n_amb), "-ad", "1024", "-lw", "1e-4",
                     "-",
                     str(sky_rad),
                     "-i", str(scene_oct)],
                    check=True, stdin=fpts, stdout=fdc, stderr=subprocess.PIPE,
                    env=_RAD_ENV,
                )
            ill_proc = subprocess.run(
                [_bin("dctimestep"), str(dc_mtx), str(smx)],
                check=True, capture_output=True, env=_RAD_ENV,
            )
            # dctimestep ASCII output: header lines, blank line, then NROWS data
            # rows of NCOLS RGB triples each. Layout is (n_sensors, 8760, 3).
            text = ill_proc.stdout.decode(errors="ignore")
            split = text.split("\n\n", 1)
            data_text = split[1] if len(split) == 2 else text
            arr = np.fromstring(data_text, sep=" ")
            arr = arr.reshape(n_sensors, 8760, 3)  # (sensor, hour, RGB)
            ill = arr[..., 0] * 47.4 + arr[..., 1] * 119.9 + arr[..., 2] * 11.6
            # ill shape: (n_sensors, 8760)

            occ_mask = np.zeros(8760, dtype=bool)
            for d in range(365):
                for h in range(OCC_HOUR_START, OCC_HOUR_END):
                    occ_mask[d * 24 + h] = True
            occ_ill = ill[:, occ_mask]  # (n_sensors, n_occ_hours)
            frac = (occ_ill >= ILL_THRESHOLD_LX).mean(axis=1)  # per sensor
            sda_pct = float((frac >= TIME_THRESHOLD_FRAC).mean() * 100.0)

            return RadianceResultV2(
                sda_pct=sda_pct, n_sensors=n_sensors, success=True,
                runtime_s=time.time() - t0,
            )
    except subprocess.CalledProcessError as e:
        return RadianceResultV2(
            sda_pct=float("nan"), n_sensors=0, success=False,
            runtime_s=time.time() - t0,
            error=f"{e}: {e.stderr.decode(errors='ignore')[:300] if e.stderr else ''}",
        )
    except Exception as e:
        return RadianceResultV2(
            sda_pct=float("nan"), n_sensors=0, success=False,
            runtime_s=time.time() - t0, error=f"{type(e).__name__}: {e}",
        )
