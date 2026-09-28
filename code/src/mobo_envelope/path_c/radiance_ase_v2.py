"""Path-C Radiance runner: IES LM-83 Annual Sunlight Exposure (ASE_1000,250).

This module supplies ASE, the LM-83 companion metric to sDA, on the same
four-orientation perimeter zone and the same sensor grid as
``radiance_runner_v2.run_radiance_v2`` so the two metrics are directly
comparable per design.

ASE is the complement of sDA: sDA rewards daylight availability and therefore
saturates as glazing grows, whereas ASE penalises *excess direct sun* and
discriminates precisely among the high-WWR designs that sDA cannot separate.

Definition (IES LM-83-12):
    ASE_1000,250 = percentage of sensor points receiving >= 1000 lx of
    DIRECT sunlight for more than 250 occupied hours per year.

Pipeline differences from the sDA runner:
    1. Sky matrix uses ``gendaymtx -d`` -> direct solar component only, so the
       annual matrix carries the sun and nothing else.
    2. Threshold 1000 lx, counted as ABSOLUTE occupied hours (> 250), not a
       fraction of occupied hours.
    3. Finer sky subdivision (Reinhart MF:4 by default) so the solar disc is
       localised better than the 145-patch Tregenza sky used for sDA. The
       rfluxmtx receiver subdivision (``h=r{mf}``) is kept in sync with the
       ``gendaymtx -m {mf}`` setting.

Why ``-ab 1`` and not ``-ab 0``:
    The sky is modelled as ``glow``-based rfluxmtx receiver patches, which are
    reached only by ambient rays. Running ``rfluxmtx -ab 0`` therefore casts no
    rays that arrive at the sky and yields an all-zero daylight-coefficient
    matrix (verified empirically: 172,950 coefficients, none non-zero, hence
    ASE = 0% for every design). ``-ab 0`` is valid only for the alternative
    formulation in which suns are explicit light sources in the octree, since
    light sources are sampled by direct shadow rays rather than ambient rays.
    One ambient bounce is used here so the sensors can see the sun through the
    apertures.

Method note for the manuscript: two approximations follow from the above and
should be stated rather than glossed. (i) At ``-ab 1`` a path in which sunlight
reflects once off an interior surface before reaching the sensor is included,
so the metric slightly over-estimates strict direct-beam ASE; the direct path
dominates, since a diffusely reflected bounce returns far less irradiance than
the beam. (ii) The solar disc is discretised onto Reinhart MF:``mf`` patches
rather than represented as a discrete sun ("5-phase") matrix, which smears the
beam over a finite solid angle and smooths per-hour peaks. MF:4 (2305 patches)
keeps this tight enough for a comparative screening metric, and it is reported
as such rather than as a certification-grade glare calculation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_c.radiance_runner_v2 import (
    CITY_TO_EPW,
    OCC_HOUR_END,
    OCC_HOUR_START,
    SKY_CACHE_DIR,
    _bin,
    _RAD_ENV,
    _write_room_rad,
    _write_sensors,
)

# LM-83-12 ASE criterion
ASE_ILL_THRESHOLD_LX = 1000.0
ASE_HOUR_THRESHOLD = 250.0  # occupied hours per year, strictly exceeded

# Default sky subdivision for the direct component (Reinhart MF:4 = 2305 patches)
DEFAULT_MF = 4


@dataclass(frozen=True)
class AseResultV2:
    ase_pct: float
    mean_exceed_hours: float
    max_exceed_hours: float
    n_sensors: int
    mf: int
    success: bool
    runtime_s: float
    error: str = ""


def _ensure_sun_matrix(city: str, mf: int) -> Path:
    """Direct-only annual sky matrix (``gendaymtx -d``), cached per city and mf."""
    smx_path = SKY_CACHE_DIR / f"{city}_direct_mf{mf}.smx"
    if smx_path.exists() and smx_path.stat().st_size > 1024:
        return smx_path
    epw = CITY_TO_EPW[city]
    if not epw.exists():
        raise FileNotFoundError(f"Missing weather file: {epw}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wea = tmp / "weather.wea"
        subprocess.run(
            [_bin("epw2wea"), str(epw), str(wea)],
            check=True, capture_output=True, env=_RAD_ENV,
        )
        tmp_out = smx_path.with_suffix(".smx.partial")
        with tmp_out.open("wb") as f:
            subprocess.run(
                [_bin("gendaymtx"), "-d", "-m", str(mf), "-O1", str(wea)],
                check=True, stdout=f, stderr=subprocess.PIPE, env=_RAD_ENV,
            )
        # Only publish the cache entry once the write completed, so a crashed
        # or concurrent run never leaves a truncated matrix behind.
        os.replace(tmp_out, smx_path)
    return smx_path


def _write_sky_glow_mf(path: Path, mf: int) -> None:
    """Sky/ground receiver whose subdivision matches ``gendaymtx -m {mf}``."""
    path.write_text(
        "#@rfluxmtx h=u u=Y\n"
        "void glow groundglow\n0\n0\n4 1 1 1 0\n"
        "groundglow source ground\n0\n0\n4 0 0 -1 180\n"
        f"#@rfluxmtx h=r{mf} u=Y\n"
        "void glow skyglow\n0\n0\n4 1 1 1 0\n"
        "skyglow source sky\n0\n0\n4 0 0 1 180\n"
    )


def _occupied_mask() -> np.ndarray:
    mask = np.zeros(8760, dtype=bool)
    for d in range(365):
        for h in range(OCC_HOUR_START, OCC_HOUR_END):
            mask[d * 24 + h] = True
    return mask


def run_ase_v2(design: EnvelopeDesign, city: str, mf: int = DEFAULT_MF,
               n_amb: int = 1) -> AseResultV2:
    """Compute LM-83 ASE_1000,250 on the 4-orientation perimeter zone."""
    t0 = time.time()
    try:
        smx = _ensure_sun_matrix(city, mf)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            room_rad = tmpdir / "room.rad"
            sky_rad = tmpdir / "sky.rad"
            sensors_pts = tmpdir / "sensors.pts"
            scene_oct = tmpdir / "scene.oct"
            dc_mtx = tmpdir / "dc.mtx"

            _write_room_rad(room_rad, design)
            _write_sky_glow_mf(sky_rad, mf)
            n_sensors = _write_sensors(sensors_pts)

            with open(scene_oct, "wb") as foct:
                subprocess.run(
                    [_bin("oconv"), str(room_rad), str(sky_rad)],
                    check=True, stdout=foct, stderr=subprocess.PIPE, env=_RAD_ENV,
                )

            # Directness comes from the sun-only sky matrix, not from -ab.
            # See module docstring: -ab 0 yields an all-zero DC matrix here.
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
            text = ill_proc.stdout.decode(errors="ignore")
            split = text.split("\n\n", 1)
            data_text = split[1] if len(split) == 2 else text
            arr = np.fromstring(data_text, sep=" ")
            expected = n_sensors * 8760 * 3
            if arr.size != expected:
                raise ValueError(
                    f"dctimestep returned {arr.size} values, expected {expected} "
                    f"({n_sensors} sensors x 8760 h x 3)"
                )
            arr = arr.reshape(n_sensors, 8760, 3)
            ill = arr[..., 0] * 47.4 + arr[..., 1] * 119.9 + arr[..., 2] * 11.6

            occ_ill = ill[:, _occupied_mask()]                       # (sensors, occ_h)
            exceed_hours = (occ_ill >= ASE_ILL_THRESHOLD_LX).sum(axis=1).astype(float)
            ase_pct = float((exceed_hours > ASE_HOUR_THRESHOLD).mean() * 100.0)

            return AseResultV2(
                ase_pct=ase_pct,
                mean_exceed_hours=float(exceed_hours.mean()),
                max_exceed_hours=float(exceed_hours.max()),
                n_sensors=n_sensors,
                mf=mf,
                success=True,
                runtime_s=time.time() - t0,
            )
    except subprocess.CalledProcessError as e:
        return AseResultV2(
            ase_pct=float("nan"), mean_exceed_hours=float("nan"),
            max_exceed_hours=float("nan"), n_sensors=0, mf=mf, success=False,
            runtime_s=time.time() - t0,
            error=f"{e}: {e.stderr.decode(errors='ignore')[:300] if e.stderr else ''}",
        )
    except Exception as e:  # noqa: BLE001 - surfaced in the result record
        return AseResultV2(
            ase_pct=float("nan"), mean_exceed_hours=float("nan"),
            max_exceed_hours=float("nan"), n_sensors=0, mf=mf, success=False,
            runtime_s=time.time() - t0, error=f"{type(e).__name__}: {e}",
        )
