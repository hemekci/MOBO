"""Path-C Radiance runner: real LM-83 sDA via daylight-coefficient method.

Pipeline per design (Tregenza 145-bin sky discretisation):

    1. Build parametric perimeter zone .rad (single south-facing facade).
    2. Build glazing surface scaled to design.wwr_south.
    3. oconv → octree of static scene.
    4. rfluxmtx → daylight-coefficient (DC) matrix on a 9-point sensor grid.
    5. dctimestep DC * sky.smx → annual hourly illuminance per sensor.
    6. LM-83 sDA: % sensors with ≥ 300 lx for ≥ 50 % of occupied hours
       (08:00–18:00 local, weekdays not distinguished — LM-83 default).

Sky matrix is pre-cached per city (design-invariant).
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
WEATHER_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "weather"
SKY_CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "sky_cache"
SKY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CITY_TO_EPW = {
    "Houston":     WEATHER_DIR / "Houston.epw",
    "NYC":         WEATHER_DIR / "NYC.epw",
    "Minneapolis": WEATHER_DIR / "Minneapolis.epw",
}

# LM-83 defaults
ILL_THRESHOLD_LX = 300.0
TIME_THRESHOLD_FRAC = 0.50
OCC_HOUR_START = 8
OCC_HOUR_END = 18  # exclusive

# Room geometry: 6 m wide × 8 m deep × 3 m tall, south-facing facade on +Y wall
ROOM_W = 6.0
ROOM_D = 8.0
ROOM_H = 3.0
SENSOR_HEIGHT = 0.75
# 5×5 sensor grid (LM-83 typical: 0.6 m spacing on 6×8 m room)
SENSOR_X = np.linspace(1.0, ROOM_W - 1.0, 5)
SENSOR_Y = np.linspace(1.0, ROOM_D - 1.0, 5)


@dataclass(frozen=True)
class RadianceResult:
    sda_pct: float
    n_sensors: int
    success: bool
    runtime_s: float
    error: str = ""


def _bin(name: str) -> str:
    return str(RAD_BIN / name)


def _ensure_sky_matrix(city: str) -> Path:
    """Pre-compute Tregenza-145 sky matrix once per city."""
    smx_path = SKY_CACHE_DIR / f"{city}_tregenza145.smx"
    if smx_path.exists() and smx_path.stat().st_size > 1024:
        return smx_path
    epw = CITY_TO_EPW[city]
    if not epw.exists():
        raise FileNotFoundError(f"Missing weather file: {epw}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wea = tmp / "weather.wea"
        # epw2wea writes a .wea file
        subprocess.run([_bin("epw2wea"), str(epw), str(wea)], check=True, capture_output=True)
        # gendaymtx -m 1 → 145 patches (Tregenza), -O 1 = solar radiance
        with smx_path.open("wb") as f:
            subprocess.run(
                [_bin("gendaymtx"), "-m", "1", "-O1", str(wea)],
                check=True, stdout=f, stderr=subprocess.PIPE,
            )
    return smx_path


def _write_sensors(path: Path) -> int:
    """Sensor points: position (x,y,z) + upward normal (0,0,1)."""
    rows = []
    for x in SENSOR_X:
        for y in SENSOR_Y:
            rows.append(f"{x:.3f} {y:.3f} {SENSOR_HEIGHT:.3f}  0 0 1")
    path.write_text("\n".join(rows) + "\n")
    return len(rows)


def _write_room_rad(path: Path, design: EnvelopeDesign) -> None:
    """Single south-facing perimeter zone with WWR-scaled glazing."""
    glazing = GLAZING[design.glazing_type]
    vt = float(glazing["VT"])

    wwr = float(design.wwr_south)
    # Glazing centred on south wall (y=0), 1 m sill height, scaled to WWR
    win_area = wwr * (ROOM_W * ROOM_H)
    win_w = min(ROOM_W - 0.4, np.sqrt(win_area * (ROOM_W / ROOM_H)))
    win_h = win_area / max(win_w, 1e-6)
    win_h = min(win_h, ROOM_H - 0.4)
    win_w = win_area / max(win_h, 1e-6)

    cx = ROOM_W / 2.0
    cz = 1.0 + win_h / 2.0
    x0, x1 = cx - win_w / 2.0, cx + win_w / 2.0
    z0, z1 = cz - win_h / 2.0, cz + win_h / 2.0

    parts = []
    parts.append("# Materials")
    parts.append("void plastic wall_mat\n0\n0\n5  0.5 0.5 0.5  0 0\n")
    parts.append("void plastic floor_mat\n0\n0\n5  0.2 0.2 0.2  0 0\n")
    parts.append("void plastic ceiling_mat\n0\n0\n5  0.8 0.8 0.8  0 0\n")
    parts.append("void plastic ground_mat\n0\n0\n5  0.2 0.2 0.2  0 0\n")
    parts.append(f"void glass glaz_mat\n0\n0\n3  {vt:.3f} {vt:.3f} {vt:.3f}\n")

    # Floor
    parts.append("floor_mat polygon floor\n0\n0\n12\n"
                 f"0 0 0\n{ROOM_W} 0 0\n{ROOM_W} {ROOM_D} 0\n0 {ROOM_D} 0\n")
    # Ceiling
    parts.append("ceiling_mat polygon ceiling\n0\n0\n12\n"
                 f"0 0 {ROOM_H}\n0 {ROOM_D} {ROOM_H}\n{ROOM_W} {ROOM_D} {ROOM_H}\n{ROOM_W} 0 {ROOM_H}\n")
    # North wall (y = D)
    parts.append("wall_mat polygon north\n0\n0\n12\n"
                 f"0 {ROOM_D} 0\n{ROOM_W} {ROOM_D} 0\n{ROOM_W} {ROOM_D} {ROOM_H}\n0 {ROOM_D} {ROOM_H}\n")
    # East wall (x = W)
    parts.append("wall_mat polygon east\n0\n0\n12\n"
                 f"{ROOM_W} 0 0\n{ROOM_W} {ROOM_D} 0\n{ROOM_W} {ROOM_D} {ROOM_H}\n{ROOM_W} 0 {ROOM_H}\n")
    # West wall (x = 0)
    parts.append("wall_mat polygon west\n0\n0\n12\n"
                 f"0 0 0\n0 0 {ROOM_H}\n0 {ROOM_D} {ROOM_H}\n0 {ROOM_D} 0\n")
    # South wall with window cutout — use 4 polygons surrounding window
    # bottom strip
    parts.append("wall_mat polygon south_bottom\n0\n0\n12\n"
                 f"0 0 0\n{ROOM_W} 0 0\n{ROOM_W} 0 {z0}\n0 0 {z0}\n")
    # top strip
    parts.append("wall_mat polygon south_top\n0\n0\n12\n"
                 f"0 0 {z1}\n{ROOM_W} 0 {z1}\n{ROOM_W} 0 {ROOM_H}\n0 0 {ROOM_H}\n")
    # left strip
    parts.append("wall_mat polygon south_left\n0\n0\n12\n"
                 f"0 0 {z0}\n{x0} 0 {z0}\n{x0} 0 {z1}\n0 0 {z1}\n")
    # right strip
    parts.append("wall_mat polygon south_right\n0\n0\n12\n"
                 f"{x1} 0 {z0}\n{ROOM_W} 0 {z0}\n{ROOM_W} 0 {z1}\n{x1} 0 {z1}\n")
    # Glazing
    parts.append("glaz_mat polygon glazing\n0\n0\n12\n"
                 f"{x0} 0 {z0}\n{x1} 0 {z0}\n{x1} 0 {z1}\n{x0} 0 {z1}\n")
    # Ground plane (large, in front of facade for sky reflection)
    parts.append("ground_mat polygon ground\n0\n0\n12\n"
                 f"-30 -30 0\n30 -30 0\n30 0 0\n-30 0 0\n")

    path.write_text("\n".join(parts))


def _write_sky_glow(path: Path) -> None:
    """Receiver file with rfluxmtx hint. Tregenza basis = Reinhart MF=1."""
    path.write_text(
        "#@rfluxmtx h=u u=Y\n"
        "void glow groundglow\n0\n0\n4 1 1 1 0\n"
        "groundglow source ground\n0\n0\n4 0 0 -1 180\n"
        "#@rfluxmtx h=r1 u=Y\n"
        "void glow skyglow\n0\n0\n4 1 1 1 0\n"
        "skyglow source sky\n0\n0\n4 0 0 1 180\n"
    )


def run_radiance(design: EnvelopeDesign, city: str, n_amb: int = 1) -> RadianceResult:
    """Compute LM-83 sDA for `design` via Radiance daylight-coefficient method."""
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

            # 1. oconv: build scene octree (no sky in static scene; rfluxmtx adds sky source)
            with scene_oct.open("wb") as f:
                subprocess.run(
                    [_bin("oconv"), str(room_rad)],
                    check=True, stdout=f, stderr=subprocess.PIPE,
                )

            # 2. rfluxmtx: 145-patch DC matrix, sender=sensors, receiver=sky
            #    -ab = ambient bounces; -ad = ambient divisions; -lw = limit weight
            env = os.environ.copy()
            env["PATH"] = f"{RAD_BIN}:{env.get('PATH', '')}"
            with sensors_pts.open("rb") as fin, dc_mtx.open("wb") as fout:
                subprocess.run(
                    [
                        _bin("rfluxmtx"),
                        "-fa",
                        "-I+",  # irradiance
                        "-y", str(n_sensors),
                        "-ab", str(n_amb),
                        "-ad", "1024",
                        "-lw", "0.001",
                        "-",
                        str(sky_rad),
                        "-i", str(scene_oct),
                    ],
                    stdin=fin, stdout=fout, stderr=subprocess.PIPE,
                    check=True, env=env,
                )

            # 3. dctimestep: combine DC matrix with annual sky matrix
            #    Strip Radiance header with getinfo -
            dct = subprocess.Popen(
                [_bin("dctimestep"), str(dc_mtx), str(smx)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            )
            strip = subprocess.Popen(
                [_bin("getinfo"), "-"],
                stdin=dct.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            )
            dct.stdout.close()
            ascii_out, strip_err = strip.communicate()
            _, dct_err = dct.communicate()
            if strip.returncode != 0 or dct.returncode != 0:
                raise RuntimeError(
                    f"dctimestep/getinfo failed: dct={dct_err.decode()[:200]} strip={strip_err.decode()[:200]}"
                )
            class _DctOut:
                stdout = ascii_out
            dct_proc = _DctOut()
            # 4. rmtxop with luminous efficacy (179 lm/W per Radiance convention),
            #    plus photopic luminance = 0.265R + 0.670G + 0.065B
            #    We pipe text-format dctimestep output through python.
            data = np.loadtxt(dct_proc.stdout.decode().splitlines(), dtype=float)
            # Reshape: dctimestep yields 1 line per (sensor, timestep), 3 columns RGB
            # Actually, default ascii output: each line = one timestep for a sensor,
            # written sensor-major. But formats vary across Radiance versions.
            # The robust approach: shape == (n_sensors * 8760, 3) sensor-major.
            if data.size == n_sensors * 8760 * 3:
                data = data.reshape(n_sensors, 8760, 3)
            elif data.size == 8760 * n_sensors * 3:
                data = data.reshape(8760, n_sensors, 3).transpose(1, 0, 2)
            else:
                return RadianceResult(
                    sda_pct=float("nan"), n_sensors=n_sensors,
                    success=False, runtime_s=time.time() - t0,
                    error=f"unexpected dctimestep size: {data.size} (expected {n_sensors*8760*3})",
                )
            illum_lx = 179.0 * (
                0.265 * data[:, :, 0] + 0.670 * data[:, :, 1] + 0.065 * data[:, :, 2]
            )
            # 5. LM-83 sDA: count occupied hours where illum >= 300 lx
            occ_mask = np.zeros(8760, dtype=bool)
            # Each hour h corresponds to (day_of_year, hour_of_day):
            for h in range(8760):
                hod = h % 24
                if OCC_HOUR_START <= hod < OCC_HOUR_END:
                    occ_mask[h] = True
            n_occ = int(occ_mask.sum())
            sensor_pass = (illum_lx[:, occ_mask] >= ILL_THRESHOLD_LX).sum(axis=1)
            sensor_frac = sensor_pass / max(n_occ, 1)
            sda = float((sensor_frac >= TIME_THRESHOLD_FRAC).mean()) * 100.0

            return RadianceResult(
                sda_pct=sda, n_sensors=n_sensors,
                success=True, runtime_s=time.time() - t0,
            )
    except subprocess.CalledProcessError as e:
        return RadianceResult(
            sda_pct=float("nan"), n_sensors=0,
            success=False, runtime_s=time.time() - t0,
            error=(e.stderr.decode() if e.stderr else str(e))[:500],
        )
    except Exception as e:
        return RadianceResult(
            sda_pct=float("nan"), n_sensors=0,
            success=False, runtime_s=time.time() - t0,
            error=str(e)[:500],
        )


__all__ = ["run_radiance", "RadianceResult"]
