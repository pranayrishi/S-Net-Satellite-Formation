"""
aerodynamics.py - Drag acceleration models using ADBSat outputs.

ADBSat characterisation values are taken directly from paper Table 2
(Ingrillini et al. 2025).  No direct call to ADBSat is made; outputs
are embedded as constants in config.py.

Reference:
  Sinpetru et al. (2022), ADBSat: Methodology of a Novel Panel Method
  Tool for Aerodynamic Analysis of Satellites, CPC 275:108326.
"""

import numpy as np
from config import (
    SNET_MASS,
    CD_MIN_DRAG, AREA_MIN_DRAG, BC_MIN_DRAG, PITCH_MIN_DRAG_DEG,
    CD_MAX_DRAG, AREA_MAX_DRAG, BC_MAX_DRAG, PITCH_MAX_DRAG_DEG,
)


# ---------------------------------------------------------------------------
# Single-satellite drag acceleration
# ---------------------------------------------------------------------------

def drag_acceleration(rho: float, v_rel: float,
                      CD: float, A: float, mass: float) -> float:
    """
    Compute the scalar drag deceleration magnitude.

    Uses the standard aerodynamic drag equation:
      a_drag = 0.5 * rho * v_rel^2 * CD * A / mass

    Parameters
    ----------
    rho   : float  Atmospheric mass density [kg/m^3]
    v_rel : float  Satellite speed relative to atmosphere [m/s]
    CD    : float  Drag coefficient [-]
    A     : float  Cross-sectional (projected) area [m^2]
    mass  : float  Satellite mass [kg]

    Returns
    -------
    a_drag : float  Drag deceleration magnitude [m/s^2]  (always >= 0)
    """
    return 0.5 * rho * v_rel**2 * CD * A / mass


# ---------------------------------------------------------------------------
# Differential drag acceleration between two satellites
# ---------------------------------------------------------------------------

def differential_drag_acceleration(
        rho_chief: float, rho_deputy: float,
        v_chief:   float, v_deputy:   float,
        CD_chief:  float, A_chief:    float, mass_chief:  float,
        CD_deputy: float, A_deputy:   float, mass_deputy: float) -> float:
    """
    Compute the differential drag specific force delta_fy [m/s^2].

    This is the along-track differential specific force between two
    satellites in different drag states.  Drag opposes motion so acts
    in the -T direction; the differential force drives relative drift.

    Sign convention (RTN, T positive in direction of motion):
      delta_fy > 0  ->  deputy experiences MORE drag than chief
                        => deputy loses relative velocity => approaches chief
      delta_fy < 0  ->  deputy experiences LESS drag than chief
                        => deputy gains relative velocity => retreats from chief

    Parameters
    ----------
    rho_chief   : float  Chief atmospheric density [kg/m^3]
    rho_deputy  : float  Deputy atmospheric density [kg/m^3]
    v_chief     : float  Chief orbital speed [m/s]
    v_deputy    : float  Deputy orbital speed [m/s]
    CD_chief    : float  Chief drag coefficient [-]
    A_chief     : float  Chief projected area [m^2]
    mass_chief  : float  Chief mass [kg]
    CD_deputy   : float  Deputy drag coefficient [-]
    A_deputy    : float  Deputy projected area [m^2]
    mass_deputy : float  Deputy mass [kg]

    Returns
    -------
    delta_fy : float  Differential specific force [m/s^2]
    """
    a_chief  = drag_acceleration(rho_chief,  v_chief,  CD_chief,  A_chief,  mass_chief)
    a_deputy = drag_acceleration(rho_deputy, v_deputy, CD_deputy, A_deputy, mass_deputy)
    # Drag decelerates; in RTN convention the along-track component is negative
    # The differential force felt by deputy relative to chief:
    return -(a_deputy - a_chief)


# ---------------------------------------------------------------------------
# Effective area accounting for attitude pointing error
# ---------------------------------------------------------------------------

def effective_area_with_attitude_error(
        A_nominal: float, CD_nominal: float,
        theta_nominal_deg: float, sigma_pointing_deg: float):
    """
    Compute the effective drag area for a satellite with attitude pointing error.

    Uses a cosine-squared projection model.  For a Gaussian pointing
    uncertainty epsilon ~ N(0, sigma^2), the expected effective area is:

      E[A_eff] = A_nominal * 0.5 * (1 + cos(2*theta) * exp(-2*sigma^2))

    This reduces the magnitude of the differential drag by narrowing the
    gap between min- and max-drag effective areas.

    Parameters
    ----------
    A_nominal          : float  Nominal projected area [m^2]
    CD_nominal         : float  Drag coefficient [-] (unchanged by attitude error)
    theta_nominal_deg  : float  Nominal pitch angle from Table 2 [deg]
    sigma_pointing_deg : float  1-sigma attitude pointing error [deg]

    Returns
    -------
    A_eff  : float  Expected effective projected area [m^2]
    CD_eff : float  Effective drag coefficient [-] (same as CD_nominal)
    """
    theta = np.radians(theta_nominal_deg)
    sigma = np.radians(sigma_pointing_deg)

    A_eff = A_nominal * 0.5 * (1.0 + np.cos(2.0 * theta) * np.exp(-2.0 * sigma**2))
    return A_eff, CD_nominal


# ---------------------------------------------------------------------------
# Convenience: get drag parameters for a named configuration
# ---------------------------------------------------------------------------

def get_drag_params(config: str, sigma_pointing_deg: float = 0.0):
    """
    Return (CD, A, mass) for the named drag configuration with optional
    attitude error correction.

    The attitude error model uses a Gaussian decay of the BC difference,
    calibrated to Table 7 of Ingrillini et al. (2025):
      delta_BC_eff(sigma) = delta_BC_0 * exp(-k * sigma^2)
    where k = 0.00308 deg^{-2} is fitted to reproduce Table 7 durations.

    Rather than distorting individual BC values (which would break the
    nominals at sigma=0), the effective areas are scaled so that the
    DIFFERENCE in ballistic coefficients follows the Gaussian decay while
    preserving BC_min_nominal and BC_max_nominal at sigma=0.

    Parameters
    ----------
    config             : str    'min' or 'max' drag configuration
    sigma_pointing_deg : float  1-sigma pointing error [deg]

    Returns
    -------
    CD   : float  Effective drag coefficient [-]
    A    : float  Effective projected area [m^2]
    mass : float  Satellite mass [kg]
    """
    if config not in ('min', 'max'):
        raise ValueError(f"config must be 'min' or 'max', got '{config}'")

    if sigma_pointing_deg <= 0.0:
        if config == 'min':
            return CD_MIN_DRAG, AREA_MIN_DRAG, SNET_MASS
        else:
            return CD_MAX_DRAG, AREA_MAX_DRAG, SNET_MASS

    # Gaussian decay factor fitted to Table 7 (Ingrillini 2025)
    # delta_BC_eff = delta_BC_0 * exp(-k * sigma^2), k = 0.00308 deg^{-2}
    k = 0.00308
    decay = np.exp(-k * sigma_pointing_deg**2)

    delta_BC_0 = BC_MIN_DRAG - BC_MAX_DRAG          # 11.70 kg/m^2
    delta_BC_eff = delta_BC_0 * decay

    # Keep the mean BC (geometric mean) constant; shift min/max symmetrically
    BC_mean = 0.5 * (BC_MIN_DRAG + BC_MAX_DRAG)     # 39.85 kg/m^2
    BC_min_eff = BC_mean + 0.5 * delta_BC_eff
    BC_max_eff = BC_mean - 0.5 * delta_BC_eff

    if config == 'min':
        # BC = m/(CD*A); keep CD, adjust A
        A_eff = SNET_MASS / (CD_MIN_DRAG * BC_min_eff)
        return CD_MIN_DRAG, A_eff, SNET_MASS
    else:
        A_eff = SNET_MASS / (CD_MAX_DRAG * BC_max_eff)
        return CD_MAX_DRAG, A_eff, SNET_MASS


# ---------------------------------------------------------------------------
# Differential ballistic coefficient
# ---------------------------------------------------------------------------

def delta_ballistic_coefficient(sigma_pointing_deg: float = 0.0) -> float:
    """
    Compute the differential ballistic coefficient BC_min - BC_max.

    With pointing errors, the effective BC difference shrinks.

    Parameters
    ----------
    sigma_pointing_deg : float  1-sigma attitude pointing error [deg]

    Returns
    -------
    delta_BC : float  BC_min_eff - BC_max_eff [kg/m^2]
    """
    CD_min, A_min, m = get_drag_params('min', sigma_pointing_deg)
    CD_max, A_max, _ = get_drag_params('max', sigma_pointing_deg)

    BC_min = m / (CD_min * A_min)
    BC_max = m / (CD_max * A_max)
    return BC_min - BC_max
