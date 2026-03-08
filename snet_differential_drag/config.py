"""
config.py - Physical constants and S-NET mission parameters.

All values cited from Ingrillini et al. (2025), CEAS Space Journal,
DOI: 10.1007/s12567-025-00630-x, and referenced foundational works.
SI units throughout unless noted.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_EARTH = 3.986004418e14      # [m^3/s^2] Earth's gravitational parameter
RE_EARTH = 6_378_137.0         # [m]        Earth's equatorial radius (WGS-84)
J2       = 1.08263e-3          # [-]        Earth's second zonal harmonic
OMEGA_EARTH = 7.2921150e-5     # [rad/s]    Earth's rotation rate

# ---------------------------------------------------------------------------
# S-NET formation parameters  (paper Section 1.2 and Table 2)
# ---------------------------------------------------------------------------
SNET_MASS     = 9.0            # [kg]   mass of each S-NET satellite
SNET_ALT_REF  = 560e3         # [m]    reference altitude (Jan 2024 epoch)
SNET_INCL_DEG = 97.5          # [deg]  orbit inclination (Sun-synchronous)
SNET_SMA      = RE_EARTH + SNET_ALT_REF   # ~6 938 km semi-major axis

# ---------------------------------------------------------------------------
# ADBSat aerodynamic characterisation (paper Table 2)
# Evaluated: 1 January 2024, h=560 km, i=97.5 deg, DRIA GSI model
# ---------------------------------------------------------------------------
# Minimum-drag attitude
CD_MIN_DRAG   = 3.303          # [-]      drag coefficient
AREA_MIN_DRAG = 0.0583         # [m^2]   reference (projected) area
BC_MIN_DRAG   = 45.70          # [kg/m^2] ballistic coefficient  m/(CD*A)

# Maximum-drag attitude
CD_MAX_DRAG   = 2.573          # [-]
AREA_MAX_DRAG = 0.1006         # [m^2]
BC_MAX_DRAG   = 34.00          # [kg/m^2]

# NOTE (paper footnote 3): Under the DRIA/hypothermal flow assumption
# CD_min > CD_max because momentum transfer depends on incidence angle.
# The physically meaningful quantity is the ballistic coefficient:
#   BC_min = 45.70  (LESS drag)   BC_max = 34.00  (MORE drag)

# Pitch angles from Table 2 [deg] (angle in orbit plane wrt velocity vector)
PITCH_MIN_DRAG_DEG = 45.70
PITCH_MAX_DRAG_DEG = 34.00

# ---------------------------------------------------------------------------
# Density uncertainty model  (Vallado & Finkleman 2014, Acta Astro. 95:141)
# ---------------------------------------------------------------------------
DENSITY_UNCERT_LONGTERM  = 0.15   # 15%  long-term mean bias
DENSITY_UNCERT_SHORTTERM = 1.00   # 100% short-term worst-case
DENSITY_UNCERT_LEVELS    = [-1.00, -0.15, 0.00, +0.15, +1.00]  # fractions

# ---------------------------------------------------------------------------
# Attitude accuracy levels (paper Section 4)
# ---------------------------------------------------------------------------
ATT_ACC_BASELINE = 0.0    # [deg] ideal
ATT_ACC_LEVEL1   = 5.0    # [deg]
ATT_ACC_LEVEL2   = 10.0   # [deg]
ATT_ACC_LEVEL3   = 15.0   # [deg]
ATT_ACC_LEVELS   = [0.0, 5.0, 10.0, 15.0]

# ---------------------------------------------------------------------------
# Satellite pair naming (6 pairs from 4 satellites)
# ---------------------------------------------------------------------------
SNET_SATELLITES = ['A', 'B', 'C', 'D']
SNET_PAIRS      = ['A-B', 'A-C', 'A-D', 'B-C', 'B-D', 'C-D']

# NORAD catalog IDs for Space-Track / Celestrak
NORAD_IDS = {
    'A': 43186,
    'B': 43187,
    'C': 43188,
    'D': 43189,
}

# ---------------------------------------------------------------------------
# Manoeuvre type flags
# ---------------------------------------------------------------------------
MANOEUVRE_APPROACH = 'approach'   # reduce relative separation
MANOEUVRE_RETREAT  = 'retreat'    # increase relative separation

# ---------------------------------------------------------------------------
# Solar / geomagnetic indices for Jan 2024 (moderate solar cycle 25)
# ---------------------------------------------------------------------------
F107_AVG   = 150.0   # 81-day average F10.7 solar flux index [SFU]
F107_DAILY = 150.0   # daily F10.7 [SFU]
AP_INDEX   = 4.0     # geomagnetic Ap index (quiet conditions)

# ---------------------------------------------------------------------------
# Derived constants (for quick access)
# ---------------------------------------------------------------------------
SNET_N_KEPLERIAN = np.sqrt(MU_EARTH / SNET_SMA**3)   # [rad/s] mean motion
SNET_T_ORBITAL   = 2 * np.pi / SNET_N_KEPLERIAN       # [s]    orbital period
