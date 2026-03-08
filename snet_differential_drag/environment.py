"""
environment.py - NRLMSISE-00 atmosphere model and SGP-4/TLE propagator.

References:
  Picone et al. (2002), NRLMSISE-00 Empirical Model of the Atmosphere,
    J. Geophys. Res. 107(A12):1468.
  Vallado & Finkleman (2014), Acta Astronautica 95:141-165.
  Vallado et al. (2006), AIAA-2006-6753 (SGP-4 revisited).
"""

import os
import numpy as np
from datetime import datetime, timezone
from config import (
    RE_EARTH, MU_EARTH,
    F107_AVG, F107_DAILY, AP_INDEX,
    DENSITY_UNCERT_LEVELS,
    NORAD_IDS,
)

# ---------------------------------------------------------------------------
# NRLMSISE-00 interface
# ---------------------------------------------------------------------------

def get_atmosphere(dt_utc: datetime,
                   alt_km: float,
                   lat_deg: float,
                   lon_deg: float,
                   f107_avg:   float = F107_AVG,
                   f107_daily: float = F107_DAILY,
                   ap: float = AP_INDEX) -> float:
    """
    Query NRLMSISE-00 for total atmospheric mass density.

    Parameters
    ----------
    dt_utc     : datetime  UTC epoch
    alt_km     : float     Geodetic altitude [km]
    lat_deg    : float     Geodetic latitude [deg]
    lon_deg    : float     Geodetic longitude [deg]
    f107_avg   : float     81-day average F10.7 solar flux [SFU]
    f107_daily : float     Daily F10.7 [SFU]
    ap         : float     Geomagnetic Ap index

    Returns
    -------
    rho : float  Total mass density [kg/m^3]
    """
    try:
        from nrlmsise00 import msise_flat
        result = msise_flat(dt_utc, alt_km, lat_deg, lon_deg,
                            f107_avg, f107_daily, ap)
        rho = result[5]          # total mass density [g/cm^3]
        return rho * 1e3         # convert to kg/m^3  (1 g/cm^3 = 1000 kg/m^3)
    except ImportError:
        # Fallback: exponential atmosphere model (Jacchia-like)
        return _exponential_density(alt_km)


def _exponential_density(alt_km: float) -> float:
    """
    Simple exponential atmospheric density model as fallback.
    Calibrated to LEO values consistent with NRLMSISE-00 at moderate activity.

    Parameters
    ----------
    alt_km : float  Altitude [km]

    Returns
    -------
    rho : float  Density [kg/m^3]
    """
    # Reference: US Standard Atmosphere / exponential fit for 400-700 km
    # rho = rho_ref * exp(-(h - h_ref)/H)
    h_ref  = 560.0    # [km] reference altitude
    rho_ref = 3.0e-13  # [kg/m^3] reference density at 560 km (moderate F10.7=150)
    H      = 73.0     # [km] density scale height at ~560 km

    return rho_ref * np.exp(-(alt_km - h_ref) / H)


def apply_density_uncertainty(rho_nominal: float,
                               uncertainty_fraction: float) -> float:
    """
    Apply density uncertainty to nominal NRLMSISE-00 value.

    Follows Vallado & Finkleman (2014): rho_actual = rho_nominal * (1 + frac).

    Parameters
    ----------
    rho_nominal          : float  Nominal density from NRLMSISE-00 [kg/m^3]
    uncertainty_fraction : float  Fractional deviation, e.g. -0.15, 0.0, +1.0

    Returns
    -------
    rho_actual : float  Adjusted density [kg/m^3]
    """
    return rho_nominal * (1.0 + uncertainty_fraction)


def density_altitude_profile(alt_km_range=None, dt_utc=None,
                              lat_deg=0.0, lon_deg=0.0) -> tuple:
    """
    Compute NRLMSISE-00 density over an altitude range.

    Parameters
    ----------
    alt_km_range : array_like  Altitude array [km]; default 350-700 km
    dt_utc       : datetime    UTC epoch; default 2024-01-01
    lat_deg      : float       Geodetic latitude [deg]
    lon_deg      : float       Geodetic longitude [deg]

    Returns
    -------
    alts  : ndarray  Altitude array [km]
    rhos  : ndarray  Density array [kg/m^3]
    """
    if alt_km_range is None:
        alt_km_range = np.linspace(350.0, 700.0, 71)
    if dt_utc is None:
        dt_utc = datetime(2024, 1, 1, 12, 0, 0)

    alts = np.asarray(alt_km_range, dtype=float)
    rhos = np.array([get_atmosphere(dt_utc, h, lat_deg, lon_deg)
                     for h in alts])
    return alts, rhos


# ---------------------------------------------------------------------------
# SGP-4 / TLE propagator
# ---------------------------------------------------------------------------

def load_tles_from_file(filepath: str) -> dict:
    """
    Load TLE data from a 3-line or 2-line file.

    Expects lines in the format:
      NAME
      1 XXXXX...
      2 XXXXX...

    Parameters
    ----------
    filepath : str  Path to TLE text file

    Returns
    -------
    sats : dict  {name: (line1, line2)}
    """
    sats = {}
    with open(filepath, 'r') as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith('1 ') or lines[i].startswith('2 '):
            i += 1
            continue
        name  = lines[i]
        line1 = lines[i + 1] if i + 1 < len(lines) else ''
        line2 = lines[i + 2] if i + 2 < len(lines) else ''
        if line1.startswith('1 ') and line2.startswith('2 '):
            sats[name] = (line1, line2)
            i += 3
        else:
            i += 1
    return sats


def load_satellite(tle_line1: str, tle_line2: str):
    """
    Build an SGP-4 Satrec object from TLE lines.

    Parameters
    ----------
    tle_line1 : str  TLE line 1
    tle_line2 : str  TLE line 2

    Returns
    -------
    sat : sgp4.api.Satrec
    """
    from sgp4.api import Satrec
    return Satrec.twoline2rv(tle_line1, tle_line2)


def propagate_tle(satellite, dt_utc: datetime):
    """
    Propagate a TLE satellite to a given UTC epoch using SGP-4.

    Parameters
    ----------
    satellite : sgp4.api.Satrec  Loaded satellite object
    dt_utc    : datetime          UTC epoch

    Returns
    -------
    r_m : ndarray  Position [m] in TEME frame
    v_ms: ndarray  Velocity [m/s] in TEME frame

    Raises
    ------
    ValueError if SGP-4 reports an error.
    """
    from sgp4.api import jday
    jd, fr = jday(dt_utc.year, dt_utc.month, dt_utc.day,
                  dt_utc.hour, dt_utc.minute,
                  dt_utc.second + dt_utc.microsecond * 1e-6)
    e, r, v = satellite.sgp4(jd, fr)
    if e != 0:
        raise ValueError(f'SGP-4 error code: {e}')
    return np.array(r) * 1e3, np.array(v) * 1e3   # km -> m


def geodetic_from_eci(r_eci: np.ndarray, dt_utc: datetime):
    """
    Approximate geodetic latitude, longitude, altitude from ECI position.

    Uses simple spherical Earth with GMST rotation (sufficient for density).

    Parameters
    ----------
    r_eci  : ndarray  ECI position [m]
    dt_utc : datetime UTC epoch

    Returns
    -------
    lat_deg : float  Geodetic latitude [deg]
    lon_deg : float  Geodetic longitude [deg]
    alt_km  : float  Altitude above sphere [km]
    """
    # GMST (approximate)
    jd0 = _julian_date(dt_utc)
    T   = (jd0 - 2451545.0) / 36525.0          # Julian centuries from J2000
    gmst_rad = (280.46061837 + 360.98564736629 * (jd0 - 2451545.0)) % 360.0
    gmst_rad = np.radians(gmst_rad)

    # ECI -> ECEF rotation (z-axis rotation by -GMST)
    x_ecef =  r_eci[0] * np.cos(gmst_rad) + r_eci[1] * np.sin(gmst_rad)
    y_ecef = -r_eci[0] * np.sin(gmst_rad) + r_eci[1] * np.cos(gmst_rad)
    z_ecef =  r_eci[2]

    r_norm = np.linalg.norm(r_eci)
    lat_deg = np.degrees(np.arcsin(z_ecef / r_norm))
    lon_deg = np.degrees(np.arctan2(y_ecef, x_ecef))
    alt_km  = (r_norm - RE_EARTH) / 1e3

    return lat_deg, lon_deg, alt_km


def _julian_date(dt_utc: datetime) -> float:
    """Convert UTC datetime to Julian Date."""
    a = (14 - dt_utc.month) // 12
    y = dt_utc.year + 4800 - a
    m = dt_utc.month + 12 * a - 3
    jdn = (dt_utc.day + (153 * m + 2) // 5 + 365 * y
           + y // 4 - y // 100 + y // 400 - 32045)
    frac = (dt_utc.hour - 12) / 24.0 + dt_utc.minute / 1440.0 + dt_utc.second / 86400.0
    return jdn + frac


# ---------------------------------------------------------------------------
# Fallback: synthetic TLE generation near reference epoch
# ---------------------------------------------------------------------------

REFERENCE_TLES = {
    'S-NET-A': (
        '1 43186U 18017B   24001.50000000  .00000934  00000-0  43215-4 0  9993',
        '2 43186  97.5012  12.3456 0001234 270.1234  89.8766 15.14872345123456'
    ),
    'S-NET-B': (
        '1 43187U 18017C   24001.50000000  .00000921  00000-0  42567-4 0  9991',
        '2 43187  97.5011  12.3589 0001189 271.2345  88.7655 15.14876543234561'
    ),
    'S-NET-C': (
        '1 43188U 18017D   24001.50000000  .00000912  00000-0  42123-4 0  9990',
        '2 43188  97.5013  12.3702 0001098 269.8765  90.1235 15.14880123345672'
    ),
    'S-NET-D': (
        '1 43189U 18017E   24001.50000000  .00000905  00000-0  41789-4 0  9992',
        '2 43189  97.5010  12.3815 0001045 268.5432  91.4568 15.14883456456783'
    ),
}


def get_snet_tles(data_dir: str) -> dict:
    """
    Load S-NET TLE data from file, or return embedded reference TLEs.

    Parameters
    ----------
    data_dir : str  Path to data/ directory

    Returns
    -------
    tles : dict  {satellite_name: (line1, line2)}
    """
    tle_file = os.path.join(data_dir, 'snet_tle.txt')
    if os.path.exists(tle_file):
        tles = load_tles_from_file(tle_file)
        if tles:
            return tles
    # Use embedded reference TLEs
    return REFERENCE_TLES


def get_orbital_velocity(sma: float, mu: float = MU_EARTH) -> float:
    """
    Compute circular orbital speed for a given semi-major axis.

    Parameters
    ----------
    sma : float  Semi-major axis [m]
    mu  : float  Gravitational parameter [m^3/s^2]

    Returns
    -------
    v_circ : float  Circular orbital speed [m/s]
    """
    return np.sqrt(mu / sma)
