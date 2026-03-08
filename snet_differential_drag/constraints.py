"""
constraints.py - Operational constraint models for differential drag manoeuvres.

Models the effect of:
  1. Satellite availability (eclipse, ground contact, charging)
  2. Attitude accuracy degradation of differential drag
  3. Density uncertainty on manoeuvre duration

Reference:
  Traub, Ben-Larbi, Turco et al. (2024), BEESAT-4 in-flight results,
  12th ISCFF Workshop. [~90% manoeuvre effectiveness loss from constraints]
"""

import numpy as np
from config import ATT_ACC_LEVELS, DENSITY_UNCERT_LEVELS


# ---------------------------------------------------------------------------
# Availability constraint
# ---------------------------------------------------------------------------

def apply_availability(manoeuvre_duration_ideal: float,
                        availability_fraction: float) -> float:
    """
    Scale ideal manoeuvre duration by satellite availability.

    If the satellite can only perform drag manoeuvres a fraction f of
    the time (due to eclipse, ground contact, battery charging, etc.),
    the actual manoeuvre takes 1/f times as long.

    Parameters
    ----------
    manoeuvre_duration_ideal : float  Ideal duration [days] (availability=1)
    availability_fraction    : float  Usable fraction of time in (0, 1]

    Returns
    -------
    duration_actual : float  Effective manoeuvre duration [days]
    """
    if not (0.0 < availability_fraction <= 1.0):
        raise ValueError('availability_fraction must be in (0, 1]')
    return manoeuvre_duration_ideal / availability_fraction


def availability_sweep(duration_ideal: float,
                        fractions=None) -> tuple:
    """
    Compute manoeuvre duration over a range of availability fractions.

    Parameters
    ----------
    duration_ideal : float      Ideal manoeuvre duration [days]
    fractions      : array_like Availability fractions to sweep; default 0.1-1.0

    Returns
    -------
    fractions : ndarray  Availability fractions
    durations : ndarray  Corresponding manoeuvre durations [days]
    """
    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 19)
    fractions = np.asarray(fractions, dtype=float)
    durations = duration_ideal / fractions
    return fractions, durations


# ---------------------------------------------------------------------------
# Attitude accuracy constraint
# ---------------------------------------------------------------------------

def effective_delta_fy_with_attitude_error(delta_fy_ideal: float,
                                            sigma_deg: float) -> float:
    """
    Reduce the differential drag force magnitude due to attitude pointing error.

    With a Gaussian pointing error epsilon ~ N(0, sigma^2 deg^2), the
    expected effective differential drag is reduced.  The reduction factor
    is derived from the cosine-squared projection model in aerodynamics.py.

    This function provides a scalar approximation for use in the planner.

    Parameters
    ----------
    delta_fy_ideal : float  Ideal differential specific force [m/s^2]
    sigma_deg      : float  1-sigma pointing error [deg]

    Returns
    -------
    delta_fy_eff : float  Effective differential force [m/s^2]
    """
    from aerodynamics import get_drag_params
    from config import (CD_MIN_DRAG, AREA_MIN_DRAG, PITCH_MIN_DRAG_DEG,
                        CD_MAX_DRAG, AREA_MAX_DRAG, PITCH_MAX_DRAG_DEG,
                        SNET_MASS)

    CD_min, A_min, m = get_drag_params('min', sigma_deg)
    CD_max, A_max, _ = get_drag_params('max', sigma_deg)

    # Compute BC ratio to scale the ideal force
    BC_min = m / (CD_min * A_min)
    BC_max = m / (CD_max * A_max)
    delta_BC_eff = BC_min - BC_max

    # Ideal (sigma=0)
    from config import BC_MIN_DRAG, BC_MAX_DRAG
    delta_BC_ideal = BC_MIN_DRAG - BC_MAX_DRAG

    if delta_BC_ideal == 0:
        return 0.0

    scale = delta_BC_eff / delta_BC_ideal
    return delta_fy_ideal * scale


def attitude_duration_factor(sigma_deg: float) -> float:
    """
    Return the factor by which manoeuvre duration increases due to attitude error.

    Factor = delta_BC_ideal / delta_BC_effective.

    Parameters
    ----------
    sigma_deg : float  1-sigma pointing error [deg]

    Returns
    -------
    factor : float  Duration scaling factor (>= 1)
    """
    from aerodynamics import get_drag_params
    from config import BC_MIN_DRAG, BC_MAX_DRAG, SNET_MASS

    CD_min, A_min, m = get_drag_params('min', sigma_deg)
    CD_max, A_max, _ = get_drag_params('max', sigma_deg)

    BC_min = m / (CD_min * A_min)
    BC_max = m / (CD_max * A_max)
    delta_BC_eff = BC_min - BC_max

    delta_BC_ideal = BC_MIN_DRAG - BC_MAX_DRAG

    if delta_BC_eff <= 0:
        return np.inf   # manoeuvre not feasible
    return delta_BC_ideal / delta_BC_eff


# ---------------------------------------------------------------------------
# Density uncertainty on manoeuvre duration
# ---------------------------------------------------------------------------

def duration_uncertainty_envelope(duration_nominal: float,
                                   rho_nominal: float) -> dict:
    """
    Compute manoeuvre duration bounds for each density uncertainty level.

    Since manoeuvre duration scales as 1/rho (drag force is proportional
    to density), a density of rho*(1+f) gives duration * 1/(1+f).

    Parameters
    ----------
    duration_nominal : float  Nominal manoeuvre duration [days]
    rho_nominal      : float  Nominal atmospheric density [kg/m^3]

    Returns
    -------
    envelope : dict  {uncertainty_fraction: duration_days}
    """
    envelope = {}
    for frac in DENSITY_UNCERT_LEVELS:
        factor = 1.0 + frac
        if factor <= 0:
            factor = 1e-6    # avoid division by zero for -100% case
        envelope[frac] = duration_nominal / factor
    return envelope
