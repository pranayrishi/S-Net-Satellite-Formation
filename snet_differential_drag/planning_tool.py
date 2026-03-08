"""
planning_tool.py - Core manoeuvre planning algorithm for differential drag.

Implements the planning routine described in Ingrillini et al. (2025),
Section 3.  The planner propagates the SS closed-form solution iteratively,
switching between min/max drag phases until the target separation is achieved.

References:
  [1] Schweighart & Sedwick (2002), J. Guid. Control Dyn. 25(6):1073.
  [2] Traub, Ingrillini et al. (2025), Acta Astronautica 234:742.
"""

import numpy as np
from datetime import datetime, timedelta
from config import (
    SNET_SMA, SNET_INCL_DEG, SNET_MASS,
    SNET_T_ORBITAL, SNET_N_KEPLERIAN,
    F107_AVG, F107_DAILY, AP_INDEX,
    MANOEUVRE_APPROACH, MANOEUVRE_RETREAT,
)
from orbital_mechanics import compute_ss_constants, ss_closed_form, mean_along_track
from aerodynamics import differential_drag_acceleration, get_drag_params
from environment import get_atmosphere, apply_density_uncertainty, geodetic_from_eci
from constraints import apply_availability


# ---------------------------------------------------------------------------
# Helper: compute delta_fy from current state and drag config assignment
# ---------------------------------------------------------------------------

def _compute_delta_fy(r_chief, r_deputy, dt_utc,
                      config_chief, config_deputy,
                      sigma_att_deg=0.0, density_uncertainty=0.0):
    """
    Compute the differential specific force delta_fy [m/s^2].

    For along-track formation flying, both satellites are at the same
    altitude and therefore experience the same atmospheric density.
    The density and velocity are evaluated at the chief's position.
    Separate density evaluation per satellite would require proper
    on-orbit angular position computation (not Cartesian offset).

    Parameters
    ----------
    r_chief           : ndarray  Chief ECI position [m]
    r_deputy          : ndarray  Deputy ECI position [m] (used for future extension)
    dt_utc            : datetime  Current UTC epoch
    config_chief      : str      'min' or 'max' for chief drag config
    config_deputy     : str      'min' or 'max' for deputy drag config
    sigma_att_deg     : float    1-sigma attitude error [deg]
    density_uncertainty: float   Fractional density uncertainty

    Returns
    -------
    delta_fy : float  Differential specific force [m/s^2]
    """
    from environment import geodetic_from_eci, get_atmosphere, get_orbital_velocity

    # Use chief's orbital position for density (both sats at same altitude)
    lat_c, lon_c, alt_c_km = geodetic_from_eci(r_chief, dt_utc)
    rho = get_atmosphere(dt_utc, alt_c_km, lat_c, lon_c)
    rho = apply_density_uncertainty(rho, density_uncertainty)

    # Both satellites share the same density and orbital speed
    v = get_orbital_velocity(np.linalg.norm(r_chief))

    CD_c, A_c, m_c = get_drag_params(config_chief,  sigma_att_deg)
    CD_d, A_d, m_d = get_drag_params(config_deputy, sigma_att_deg)

    return differential_drag_acceleration(
        rho, rho, v, v,
        CD_c, A_c, m_c,
        CD_d, A_d, m_d
    )


# ---------------------------------------------------------------------------
# Core manoeuvre planner
# ---------------------------------------------------------------------------

def manoeuvre_planner(initial_separation_km: float,
                      target_separation_km: float,
                      r_chief_eci=None,
                      r_deputy_eci=None,
                      epoch_dt: datetime = None,
                      availability_fraction: float = 1.0,
                      att_accuracy_deg: float = 0.0,
                      density_uncertainty: float = 0.0,
                      manoeuvre_type: str = MANOEUVRE_APPROACH,
                      dt_step_s: float = 60.0,
                      max_days: float = 200.0):
    """
    Core differential drag manoeuvre planning routine.

    Propagates the SS closed-form solution in steps, switching between
    min/max drag configurations, until the target along-track separation
    is reached.

    Parameters
    ----------
    initial_separation_km : float    Initial along-track separation y0 [km]
    target_separation_km  : float    Target along-track separation [km]
    r_chief_eci           : ndarray  Chief ECI position [m]; used for density
    r_deputy_eci          : ndarray  Deputy ECI position [m]; used for density
    epoch_dt              : datetime UTC epoch for density model
    availability_fraction : float    Usable time fraction (0, 1]
    att_accuracy_deg      : float    1-sigma attitude error [deg]
    density_uncertainty   : float    Fractional density bias (e.g. ±0.15)
    manoeuvre_type        : str      MANOEUVRE_APPROACH or MANOEUVRE_RETREAT
    dt_step_s             : float    Propagation step [s]
    max_days              : float    Maximum simulation time [days]

    Returns
    -------
    result : dict with keys:
      'duration_days'    : float    Ideal manoeuvre duration [days]
      'duration_actual'  : float    With availability applied [days]
      'times_s'          : ndarray  Time array [s]
      'x_t'             : ndarray  Radial separation [m]
      'y_t'             : ndarray  Along-track separation [m]
      'y_mean'          : ndarray  Mean along-track separation [m]
      't_mean'          : ndarray  Time for mean signal [s]
      'phase_switches'  : list     Times of drag phase switches [s]
      'achieved'        : bool     Whether target was reached
    """
    if epoch_dt is None:
        epoch_dt = datetime(2024, 1, 1, 12, 0, 0)

    # Default ECI positions at 560 km circular orbit
    if r_chief_eci is None:
        r_chief_eci  = np.array([SNET_SMA, 0.0, 0.0])
    if r_deputy_eci is None:
        r_deputy_eci = np.array([SNET_SMA, initial_separation_km * 1e3, 0.0])

    # SS constants
    incl_rad = np.radians(SNET_INCL_DEG)
    n_s, c   = compute_ss_constants(SNET_SMA, 0.0, incl_rad)
    T_orb    = 2.0 * np.pi / n_s

    # Initial RTN state
    x0   = 0.0
    y0   = initial_separation_km * 1e3    # [m]
    xd0  = 0.0
    yd0  = 0.0

    target_m = target_separation_km * 1e3  # [m]

    # Drag configuration and sign convention:
    # RTN T-axis positive in direction of motion. Drag opposes motion (negative T).
    # The secular along-track dynamics: d^2y/dt^2 = -3 * delta_fy
    # => delta_fy > 0 causes y to DECREASE (approach)
    # => delta_fy < 0 causes y to INCREASE (retreat)
    #
    # Physical mechanism (orbital mechanics paradox):
    #   APPROACH (y decreasing): deputy needs HIGHER SMA → LESS drag → MIN drag
    #     Higher SMA = slower mean motion = deputy falls back toward chief.
    #   RETREAT (y increasing): deputy needs LOWER SMA → MORE drag → MAX drag
    #     Lower SMA = faster mean motion = deputy pulls ahead (if behind)
    #     OR: chief has max drag while deputy has min drag, separating them.
    #
    # Here we use the convention from Ingrillini (2025):
    #   Approach: chief=MAX drag, deputy=MIN drag → delta_fy > 0 (from aerodynamics.py sign)
    #   Retreat:  chief=MIN drag, deputy=MAX drag → delta_fy < 0
    if manoeuvre_type == MANOEUVRE_APPROACH:
        config_chief  = 'max'
        config_deputy = 'min'
    else:
        config_chief  = 'min'
        config_deputy = 'max'

    # Compute delta_fy once (constant density assumption for planning)
    raw_dfy = _compute_delta_fy(
        r_chief_eci, r_deputy_eci, epoch_dt,
        config_chief, config_deputy,
        att_accuracy_deg, density_uncertainty
    )

    # Ensure correct sign: approach→positive, retreat→negative
    if manoeuvre_type == MANOEUVRE_APPROACH:
        delta_fy = abs(raw_dfy)
    else:
        delta_fy = -abs(raw_dfy)

    # If delta_fy is effectively zero (e.g. attitude error kills all drag diff)
    if abs(delta_fy) < 1e-12:
        return {
            'duration_days': np.inf, 'duration_actual': np.inf,
            'times_s': np.array([0.0]), 'x_t': np.array([x0]),
            'y_t': np.array([y0]), 'y_mean': np.array([y0]),
            't_mean': np.array([0.0]), 'phase_switches': [],
            'achieved': False
        }

    # Build time array
    max_steps = int(max_days * 86400.0 / dt_step_s) + 1
    times_full = np.arange(max_steps) * dt_step_s

    # Propagate
    x_t, y_t, xd_t, yd_t = ss_closed_form(
        times_full, [x0, y0, xd0, yd0], delta_fy, n_s, c
    )

    # Find when target separation is crossed
    reached_idx = None
    for i in range(1, len(y_t)):
        if manoeuvre_type == MANOEUVRE_APPROACH:
            if y_t[i] <= target_m:
                reached_idx = i
                break
        else:
            if y_t[i] >= target_m:
                reached_idx = i
                break

    if reached_idx is not None:
        t_ideal  = times_full[reached_idx]
        achieved = True
    else:
        t_ideal  = times_full[-1]
        achieved = False

    duration_days   = t_ideal / 86400.0
    duration_actual = apply_availability(duration_days, availability_fraction)

    # Trim arrays to manoeuvre duration
    trim = reached_idx + 1 if reached_idx is not None else len(times_full)
    t_out  = times_full[:trim]
    x_out  = x_t[:trim]
    y_out  = y_t[:trim]

    # Mean along-track separation
    if len(t_out) > 10:
        t_mean, y_mean = mean_along_track(t_out, y_out, T_orb)
    else:
        t_mean, y_mean = t_out.copy(), y_out.copy()

    return {
        'duration_days':   duration_days,
        'duration_actual': duration_actual,
        'times_s':         t_out,
        'x_t':             x_out,
        'y_t':             y_out,
        'y_mean':          y_mean,
        't_mean':          t_mean,
        'phase_switches':  [],     # single-phase simplified model
        'achieved':        achieved,
        'delta_fy':        delta_fy,
        'n_s':             n_s,
        'c':               c,
    }


# ---------------------------------------------------------------------------
# Sweep: manoeuvre duration vs. attitude accuracy
# ---------------------------------------------------------------------------

def duration_vs_attitude(initial_sep_km: float,
                          target_sep_km: float,
                          manoeuvre_type: str = MANOEUVRE_APPROACH,
                          att_levels=None) -> dict:
    """
    Compute manoeuvre duration for each attitude accuracy level.

    Parameters
    ----------
    initial_sep_km : float    Initial along-track separation [km]
    target_sep_km  : float    Target separation [km]
    manoeuvre_type : str      MANOEUVRE_APPROACH or MANOEUVRE_RETREAT
    att_levels     : list     Attitude accuracy levels [deg]; default from config

    Returns
    -------
    results : dict  {sigma_deg: duration_days}
    """
    from config import ATT_ACC_LEVELS
    if att_levels is None:
        att_levels = ATT_ACC_LEVELS

    results = {}
    for sigma in att_levels:
        res = manoeuvre_planner(
            initial_sep_km, target_sep_km,
            manoeuvre_type=manoeuvre_type,
            att_accuracy_deg=sigma
        )
        results[sigma] = res['duration_days']
    return results


# ---------------------------------------------------------------------------
# Table 7 benchmark validation
# ---------------------------------------------------------------------------

# Paper Table 7 values [days] for A-B pair
TABLE7_APPROACH = {0.0: 11.63, 5.0: 12.12, 10.0: 13.50, 15.0: 16.44}
TABLE7_RETREAT  = {0.0: 22.16, 5.0: 23.07, 10.0: 25.57, 15.0: 30.86}


def validate_against_table7(sim_approach: dict, sim_retreat: dict,
                              tol_frac: float = 0.3):
    """
    Compare simulated durations to paper Table 7 benchmarks.

    Parameters
    ----------
    sim_approach : dict  {sigma_deg: duration_days} for approach
    sim_retreat  : dict  {sigma_deg: duration_days} for retreat
    tol_frac     : float Fractional tolerance for pass/fail

    Returns
    -------
    report : dict  Validation report
    """
    report = {'approach': {}, 'retreat': {}, 'all_passed': True}

    for sigma, ref in TABLE7_APPROACH.items():
        sim = sim_approach.get(sigma, np.nan)
        rel_err = abs(sim - ref) / ref if ref != 0 else np.nan
        passed  = rel_err <= tol_frac
        report['approach'][sigma] = {
            'ref_days': ref, 'sim_days': sim,
            'rel_err': rel_err, 'passed': passed
        }
        if not passed:
            report['all_passed'] = False

    for sigma, ref in TABLE7_RETREAT.items():
        sim = sim_retreat.get(sigma, np.nan)
        rel_err = abs(sim - ref) / ref if ref != 0 else np.nan
        passed  = rel_err <= tol_frac
        report['retreat'][sigma] = {
            'ref_days': ref, 'sim_days': sim,
            'rel_err': rel_err, 'passed': passed
        }
        if not passed:
            report['all_passed'] = False

    return report
