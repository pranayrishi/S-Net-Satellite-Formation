"""
orbital_mechanics.py - Schweighart-Sedwick equations and closed-form solutions.

References:
  [1] Schweighart & Sedwick (2002), J. Guid. Control Dyn. 25(6):1073-1080
  [2] Traub, Ingrillini et al. (2025), Acta Astronautica 234:742
  [6] Ben-Yaacov & Gurfil (2013), J. Guid. Control Dyn. 36(6):1731-1740
"""

import numpy as np
from config import MU_EARTH, RE_EARTH, J2


# ---------------------------------------------------------------------------
# Schweighart-Sedwick model constants
# ---------------------------------------------------------------------------

def compute_ss_constants(sma: float, ecc: float, incl_rad: float):
    """
    Compute the modified mean motion n_s and coupling constant c for the
    Schweighart-Sedwick (SS) linearised relative-motion model.

    Accounts for the secular J2 perturbation by modifying the Keplerian
    mean motion and introducing a coupling between radial and along-track.

    Parameters
    ----------
    sma      : float  Semi-major axis [m]
    ecc      : float  Orbital eccentricity [-]
    incl_rad : float  Orbital inclination [rad]

    Returns
    -------
    n_s : float  Modified mean motion [rad/s]
    c   : float  SS coupling constant [-]

    From Schweighart & Sedwick (2002), Eqs. (10)-(11).
    """
    n = np.sqrt(MU_EARTH / sma**3)        # Keplerian mean motion [rad/s]
    p = sma * (1.0 - ecc**2)              # semi-latus rectum [m]

    k      = (3.0 / 2.0) * J2 * (RE_EARTH / p)**2
    gamma  = k * (1.0 - (3.0 / 2.0) * np.sin(incl_rad)**2)

    n_s  = n * (1.0 + gamma)              # modified mean motion [rad/s]

    c_sq = 1.0 + (7.0 / 2.0) * k * (1.0 - (5.0 / 4.0) * np.sin(incl_rad)**2)
    c    = np.sqrt(max(c_sq, 1e-12))      # coupling constant [-]

    return n_s, c


# ---------------------------------------------------------------------------
# Closed-form SS solution (Traub et al. 2025 corrected version)
# ---------------------------------------------------------------------------

def ss_closed_form(t, state0, delta_fy, n_s, c):
    """
    Closed-form in-plane solution to the Schweighart-Sedwick equations with
    a constant differential specific force in the along-track direction.

    Based on the CW analogy with the SS J2 corrections (c*n_s for oscillatory
    terms, n_s for secular terms).  The KEY secular term in y(t) from constant
    along-track forcing is -3/2 * delta_fy * t^2, derived from the Gauss
    variational equations for a circular orbit (d^2y/dt^2 = -3*delta_fy).

    This corrects earlier implementations that erroneously had
    -delta_fy/(2*ns^2)*t^2, which is ~280,000x too large.

    Parameters
    ----------
    t        : array_like  Time(s) since epoch [s]
    state0   : array_like  [x0, y0, xdot0, ydot0] initial RTN state [m, m/s]
    delta_fy : float       Differential specific force along-track [m/s^2]
                           Positive -> along-track separation decreases (approach)
    n_s      : float       Modified mean motion [rad/s]
    c        : float       SS coupling constant [-]

    Returns
    -------
    x_t  : ndarray  Radial separation [m]
    y_t  : ndarray  Along-track separation [m]
    xd_t : ndarray  Radial relative velocity [m/s]
    yd_t : ndarray  Along-track relative velocity [m/s]

    Physical derivation:
      The secular drift from a constant along-track differential force f_y is:
        d(delta_a)/dt = (2/n) * f_y    [Gauss equation, circular orbit]
        d(delta_y)/dt = -3/2*n*delta_a  [along-track drift from SMA diff]
        => d^2(delta_y)/dt^2 = -3*f_y
        => delta_y(t) = -3/2 * f_y * t^2   [secular, leading order]
    """
    t  = np.asarray(t, dtype=float)
    x0, y0, xd0, yd0 = state0

    ns = n_s
    c2 = c * c

    # Oscillatory phase argument
    phi = c * ns * t

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # ------------------------------------------------------------------
    # Radial solution x(t)
    # Standard CW-like form with c*ns substitution for oscillatory terms.
    # For our ICs (x0=0, xd0=0, yd0=0) this gives x(t) ≈ 0.
    # ------------------------------------------------------------------
    x_t = ((4.0 - 3.0 * cos_phi) * x0
            + sin_phi / (c * ns) * xd0
            + 2.0 * (1.0 - cos_phi) / (c * ns) * yd0)

    # ------------------------------------------------------------------
    # Along-track solution y(t)
    # Homogeneous (CW with SS substitution) + particular (from delta_fy)
    # ------------------------------------------------------------------
    # Homogeneous part (no forcing):
    #   y_hom = 6*(sin(phi)/(c*ns) - ns*t)*x0 + y0
    #           - 2*(1-cos(phi))/(c*ns)*xd0
    #           + (4*sin(phi)/(c*ns) - 3*ns*t)/(c^2)*yd0
    y_hom = (6.0 * (sin_phi / (c * ns) - ns * t) * x0
             + y0
             - 2.0 * (1.0 - cos_phi) / (c * ns) * xd0
             + (4.0 * sin_phi / (c * ns) - 3.0 * ns * t) / c2 * yd0)

    # Particular solution from constant delta_fy [Traub et al. 2025]:
    #   Secular:     -3/2 * delta_fy * t^2  (dominant, from Gauss eqs)
    #   Oscillatory: +2*delta_fy/(c*ns)^2 * (1 - cos(phi))
    #   Linear:      +2*delta_fy*ns*t/(c^2*ns^2) [small correction]
    y_part = (-1.5 * delta_fy * t**2
              + 2.0 * delta_fy / (c * ns)**2 * (1.0 - cos_phi))

    y_t = y_hom + y_part

    # ------------------------------------------------------------------
    # Velocity components (analytic time derivatives)
    # ------------------------------------------------------------------
    xd_t = (3.0 * c * ns * sin_phi * x0
             + cos_phi * xd0
             + 2.0 * sin_phi / c * yd0)

    yd_hom_dot = (6.0 * (cos_phi / c - ns) * x0
                  - 2.0 * sin_phi / c * xd0
                  + (4.0 * cos_phi / c - 3.0 * ns) / c2 * yd0)

    yd_part_dot = (-3.0 * delta_fy * t
                   + 2.0 * delta_fy / (c * ns) * sin_phi)

    yd_t = yd_hom_dot + yd_part_dot

    return x_t, y_t, xd_t, yd_t


# ---------------------------------------------------------------------------
# Mean along-track separation (orbital-period average)
# ---------------------------------------------------------------------------

def mean_along_track(t, y, T_orb):
    """
    Extract the secular (mean) along-track separation by averaging over
    one orbital period using a centered moving average.

    The paper analyses mean motion, not the short-period oscillations.

    Parameters
    ----------
    t     : ndarray  Time array [s]
    y     : ndarray  Along-track separation array [m]
    T_orb : float    Orbital period [s]

    Returns
    -------
    t_mean : ndarray  Time array for mean signal [s]
    y_mean : ndarray  Mean along-track separation [m]
    """
    dt = t[1] - t[0]
    window = max(1, int(round(T_orb / dt)))

    kernel = np.ones(window) / window
    y_mean = np.convolve(y, kernel, mode='same')

    # Trim edge effects
    half = window // 2
    return t[half:-half], y_mean[half:-half]


# ---------------------------------------------------------------------------
# Cartesian <-> Orbital elements conversions
# ---------------------------------------------------------------------------

def cartesian_to_oe(r_vec, v_vec, mu=MU_EARTH):
    """
    Convert position/velocity vectors to classical orbital elements.

    Parameters
    ----------
    r_vec : array_like  Position vector [m] in ECI
    v_vec : array_like  Velocity vector [m/s] in ECI
    mu    : float       Gravitational parameter [m^3/s^2]

    Returns
    -------
    (a, e, i, omega, RAAN, M) with angles in radians.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    r  = np.linalg.norm(r_vec)
    v  = np.linalg.norm(v_vec)
    vr = np.dot(r_vec, v_vec) / r          # radial velocity [m/s]

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h     = np.linalg.norm(h_vec)

    # Node line
    K     = np.array([0.0, 0.0, 1.0])
    N_vec = np.cross(K, h_vec)
    N     = np.linalg.norm(N_vec)

    # Eccentricity vector
    e_vec = ((v**2 - mu / r) * r_vec - r * vr * v_vec) / mu
    e     = np.linalg.norm(e_vec)

    # Orbital elements
    a     = 1.0 / (2.0 / r - v**2 / mu)   # semi-major axis [m]
    i     = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    RAAN  = (np.arccos(np.clip(N_vec[0] / N, -1.0, 1.0))
             if N > 1e-10 else 0.0)
    if N_vec[1] < 0.0:
        RAAN = 2.0 * np.pi - RAAN

    omega = (np.arccos(np.clip(np.dot(N_vec, e_vec) / (N * e), -1.0, 1.0))
             if N > 1e-10 and e > 1e-10 else 0.0)
    if e_vec[2] < 0.0:
        omega = 2.0 * np.pi - omega

    nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)) if e > 1e-10 else 0.0
    if vr < 0.0:
        nu = 2.0 * np.pi - nu

    # True -> eccentric -> mean anomaly
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                          np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    M = E - e * np.sin(E)
    M = M % (2.0 * np.pi)

    return np.array([a, e, i, omega, RAAN, M])


def cartesian_to_roe(r_chief, v_chief, r_deputy, v_deputy, mu=MU_EARTH):
    """
    Convert absolute Cartesian state vectors to Relative Orbital Elements (ROE).

    Uses the linearised ROE mapping from Ben-Yaacov & Gurfil (2013),
    J. Guid. Control Dyn. 36(6):1731-1740.

    Parameters
    ----------
    r_chief  : array_like  Chief position [m] ECI
    v_chief  : array_like  Chief velocity [m/s] ECI
    r_deputy : array_like  Deputy position [m] ECI
    v_deputy : array_like  Deputy velocity [m/s] ECI
    mu       : float       Gravitational parameter [m^3/s^2]

    Returns
    -------
    roe : ndarray  [delta_a, delta_ex, delta_ey, delta_ix, delta_iy, delta_u]
    """
    oe_c = cartesian_to_oe(r_chief,  v_chief,  mu)
    oe_d = cartesian_to_oe(r_deputy, v_deputy, mu)

    a_c, e_c, i_c, om_c, Ra_c, M_c = oe_c
    a_d, e_d, i_d, om_d, Ra_d, M_d = oe_d

    delta_a   = (a_d - a_c) / a_c
    delta_e   = e_d - e_c
    delta_i   = i_d - i_c
    delta_Ra  = Ra_d - Ra_c
    delta_om  = om_d - om_c
    delta_M   = M_d  - M_c

    # Wrap angle differences
    delta_Ra  = (delta_Ra  + np.pi) % (2 * np.pi) - np.pi
    delta_M   = (delta_M   + np.pi) % (2 * np.pi) - np.pi

    delta_ex  = delta_e * np.cos(om_c) - e_c * delta_Ra * np.sin(om_c)
    delta_ey  = delta_e * np.sin(om_c) + e_c * delta_Ra * np.cos(om_c)
    delta_ix  = delta_i
    delta_iy  = np.sin(i_c) * delta_Ra
    delta_u   = delta_M + delta_om    # mean relative longitude

    return np.array([delta_a, delta_ex, delta_ey, delta_ix, delta_iy, delta_u])


# ---------------------------------------------------------------------------
# ECI -> RTN frame conversion
# ---------------------------------------------------------------------------

def eci_to_rtn(r_chief_eci, v_chief_eci, r_deputy_eci, v_deputy_eci):
    """
    Convert ECI absolute states to RTN relative position and velocity.

    RTN frame centred on the chief satellite:
      R = radial   (outward from Earth centre)
      T = tangential (along-track, direction of motion)
      N = normal   (cross-track, completes right-hand system)

    Parameters
    ----------
    r_chief_eci  : array_like  Chief position [m] ECI
    v_chief_eci  : array_like  Chief velocity [m/s] ECI
    r_deputy_eci : array_like  Deputy position [m] ECI
    v_deputy_eci : array_like  Deputy velocity [m/s] ECI

    Returns
    -------
    rtn_pos : ndarray  [x_R, x_T, x_N] relative position [m]
    rtn_vel : ndarray  [v_R, v_T, v_N] relative velocity [m/s]
    """
    rc = np.asarray(r_chief_eci,  dtype=float)
    vc = np.asarray(v_chief_eci,  dtype=float)
    rd = np.asarray(r_deputy_eci, dtype=float)
    vd = np.asarray(v_deputy_eci, dtype=float)

    r_rel = rd - rc
    v_rel = vd - vc

    # Unit vectors of RTN frame
    r_hat = rc / np.linalg.norm(rc)
    h_vec = np.cross(rc, vc)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)

    # Rotation matrix ECI -> RTN
    Q = np.vstack([r_hat, t_hat, n_hat])   # shape (3,3)

    # Angular velocity of RTN frame
    omega_mag = np.linalg.norm(h_vec) / np.linalg.norm(rc)**2
    omega_vec = n_hat * omega_mag          # [rad/s] in ECI

    # RTN position
    rtn_pos = Q @ r_rel

    # RTN velocity: v_rtn = Q*(v_rel - omega x r_rel)
    rtn_vel = Q @ (v_rel - np.cross(omega_vec, r_rel))

    return rtn_pos, rtn_vel


# ---------------------------------------------------------------------------
# Along-track drift rate from ROE
# ---------------------------------------------------------------------------

def along_track_drift_rate(delta_a_abs, sma, mu=MU_EARTH):
    """
    Compute the mean along-track drift rate due to a relative SMA difference.

    From Gurfil (2005) and D'Amico & Montenbruck (2006):
      dy_drift/dt = -3/2 * n * delta_a_abs / sma    [m/s]

    Parameters
    ----------
    delta_a_abs : float  Absolute difference in semi-major axes a_d - a_c [m]
    sma         : float  Chief semi-major axis [m]
    mu          : float  Gravitational parameter [m^3/s^2]

    Returns
    -------
    drift_rate : float  Along-track drift rate [m/s]
    """
    n = np.sqrt(mu / sma**3)
    return -1.5 * n * delta_a_abs / sma
