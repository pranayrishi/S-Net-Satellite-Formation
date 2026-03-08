"""
visualization.py - All figure generation for the S-NET reproduction.

Produces Figures 1-12 as described in Ingrillini et al. (2025).
All figures saved to figures/ at 300 DPI.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

from config import (
    SNET_SMA, SNET_ALT_REF, SNET_INCL_DEG, RE_EARTH,
    CD_MIN_DRAG, AREA_MIN_DRAG, BC_MIN_DRAG,
    CD_MAX_DRAG, AREA_MAX_DRAG, BC_MAX_DRAG,
    ATT_ACC_LEVELS, DENSITY_UNCERT_LEVELS,
    SNET_PAIRS, SNET_T_ORBITAL,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

_PAIR_COLORS = plt.cm.tab10(np.linspace(0, 0.9, 6))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi']  = 100


def _save(fig, name, dpi=300):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Figure 1 — S-NET orbit diagram
# ---------------------------------------------------------------------------

def fig1_snet_orbit():
    """Fig 1: Schematic of S-NET formation in LEO."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Earth
    earth = plt.Circle((0, 0), RE_EARTH / 1e6, color='steelblue',
                        zorder=2, label='Earth')
    ax.add_patch(earth)

    # Orbit
    theta = np.linspace(0, 2 * np.pi, 360)
    r_orb = SNET_SMA / 1e6
    ax.plot(r_orb * np.cos(theta), r_orb * np.sin(theta),
            'k--', linewidth=1, label=f'Orbit (~560 km alt)')

    # Four satellites at representative separations
    sep_angles = np.radians([0, 8, 18, 30])   # representative separations
    labels = ['S-NET-A', 'S-NET-B', 'S-NET-C', 'S-NET-D']
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple']
    for ang, lab, col in zip(sep_angles, labels, colors):
        x = r_orb * np.cos(ang + np.pi / 6)
        y = r_orb * np.sin(ang + np.pi / 6)
        ax.plot(x, y, 'o', markersize=10, color=col, zorder=5)
        ax.annotate(lab, (x, y), textcoords='offset points',
                    xytext=(8, 4), fontsize=9, color=col)

    ax.set_xlim(-1.15 * r_orb, 1.35 * r_orb)
    ax.set_ylim(-1.15 * r_orb, 1.15 * r_orb)
    ax.set_aspect('equal')
    ax.set_xlabel('x [10$^3$ km]')
    ax.set_ylabel('y [10$^3$ km]')
    ax.set_title(f'Fig. 1 — S-NET Formation in LEO\n'
                 f'(h ≈ 560 km, i = 97.5°, Sun-synchronous)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('', xy=(r_orb * np.cos(0.6), r_orb * np.sin(0.6)),
                xytext=(r_orb * np.cos(0.5), r_orb * np.sin(0.5)),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    return _save(fig, 'fig1_snet_orbit.png')


# ---------------------------------------------------------------------------
# Figure 2 — Phase-plane portrait
# ---------------------------------------------------------------------------

def fig2_phase_plane(trajectories: list):
    """
    Fig 2: Relative motion phase plane (x radial vs y along-track).

    Parameters
    ----------
    trajectories : list of dicts, each with 'x_t', 'y_t', 'label', 'color'
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for tr in trajectories:
        ax.plot(tr['y_t'] / 1e3, tr['x_t'] / 1e3,
                color=tr.get('color', 'tab:blue'),
                label=tr.get('label', ''),
                linewidth=1.2, alpha=0.85)
        if len(tr['x_t']) > 0:
            ax.plot(tr['y_t'][0] / 1e3, tr['x_t'][0] / 1e3,
                    'o', color=tr.get('color', 'tab:blue'), markersize=6)
            ax.plot(tr['y_t'][-1] / 1e3, tr['x_t'][-1] / 1e3,
                    's', color=tr.get('color', 'tab:blue'), markersize=6)

    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Along-track separation y [km]')
    ax.set_ylabel('Radial separation x [km]')
    ax.set_title('Fig. 2 — Relative Motion Phase Plane, S-NET')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return _save(fig, 'fig2_phase_plane.png')


# ---------------------------------------------------------------------------
# Figure 3 — Relative separation from TLE data
# ---------------------------------------------------------------------------

def fig3_relative_separation(pair_data: dict):
    """
    Fig 3: Along-track separation of all S-NET pairs over time.

    Parameters
    ----------
    pair_data : dict  {pair_name: (dates_list, separations_km_list)}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (pair, (dates, seps)) in enumerate(pair_data.items()):
        ax.plot(dates, seps, color=_PAIR_COLORS[idx],
                label=pair, linewidth=1.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Along-track separation [km]')
    ax.set_title('Fig. 3 — S-NET Relative Separation from TLE Data (SGP-4 Propagation)')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    return _save(fig, 'fig3_relative_separation.png')


# ---------------------------------------------------------------------------
# Figure 4 — ADBSat schematic
# ---------------------------------------------------------------------------

def fig4_adbsat():
    """Fig 4: ADBSat S-NET mesh schematic for min/max drag configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cmap = plt.cm.RdYlBu_r

    configs = [
        dict(ax=axes[0], label='Minimum Drag Configuration\nBC = 45.70 kg/m²',
             angle=45.0, CD=CD_MIN_DRAG, A=AREA_MIN_DRAG),
        dict(ax=axes[1], label='Maximum Drag Configuration\nBC = 34.00 kg/m²',
             angle=0.0,  CD=CD_MAX_DRAG, A=AREA_MAX_DRAG),
    ]

    for cfg in configs:
        ax    = cfg['ax']
        theta = np.radians(cfg['angle'])

        # Draw simplified satellite body as a rotated rectangle
        W, H = 0.3, 0.2        # [m] approximate S-NET body dims
        corners = np.array([[-W/2, -H/2], [W/2, -H/2],
                             [W/2,  H/2], [-W/2,  H/2]])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        corners_rot = (rot @ corners.T).T

        # Colour panels by drag contribution (normalised)
        n_panels = 20
        panel_contributions = np.linspace(0, 1, n_panels)
        panel_h = H / n_panels
        for j, contrib in enumerate(panel_contributions):
            y_bot = -H / 2 + j * panel_h
            panel_corners = np.array([
                [-W/2, y_bot], [W/2, y_bot],
                [W/2, y_bot + panel_h], [-W/2, y_bot + panel_h]
            ])
            pc_rot = (rot @ panel_corners.T).T
            poly = plt.Polygon(pc_rot, closed=True,
                               facecolor=cmap(contrib), edgecolor='k',
                               linewidth=0.3, alpha=0.85)
            ax.add_patch(poly)

        # Velocity arrow
        ax.annotate('', xy=(0.5, 0.0), xytext=(-0.5, 0.0),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(arrowstyle='->', color='navy', lw=2))
        ax.text(0.0, -0.32, 'v (velocity)', ha='center', fontsize=9, color='navy')

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.35, 0.35)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_title(cfg['label'], fontsize=10)
        ax.grid(True, alpha=0.2)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.04)
    cbar.set_label('Normalised Panel Drag Contribution [–]')

    fig.suptitle('Fig. 4 — ADBSat Characterisation of S-NET\n'
                 '1 January 2024, h = 560 km, i = 97.5°, DRIA model',
                 fontsize=11)

    return _save(fig, 'fig4_adbsat.png')


# ---------------------------------------------------------------------------
# Figure 5 — NRLMSISE-00 density altitude profile
# ---------------------------------------------------------------------------

def fig5_density_profile(alts_km: np.ndarray, rhos_kgm3: np.ndarray):
    """Fig 5: Atmospheric density vs altitude from NRLMSISE-00."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.semilogy(alts_km, rhos_kgm3, 'tab:blue', linewidth=2)
    ax.axvline(560, color='tab:red', linestyle='--',
               label='S-NET reference altitude (560 km)')

    rho_560 = np.interp(560.0, alts_km, rhos_kgm3)
    ax.plot(560, rho_560, 'ro', markersize=8)
    ax.annotate(f'ρ(560 km) = {rho_560:.2e} kg/m³',
                xy=(560, rho_560), xytext=(570, rho_560 * 3),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='k'))

    ax.set_xlabel('Altitude [km]')
    ax.set_ylabel('Total mass density [kg/m³]')
    ax.set_title('Fig. 5 — NRLMSISE-00 Atmospheric Density Profile\n'
                 '1 January 2024, lat=0°, lon=0°, F10.7=150, Ap=4')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    return _save(fig, 'fig5_density_profile.png')


# ---------------------------------------------------------------------------
# Figure 6 — Density time series with uncertainty bands
# ---------------------------------------------------------------------------

def fig6_density_uncertainty(times_days: np.ndarray, rho_nominal: np.ndarray):
    """Fig 6: Density time series showing ±15% and ±100% uncertainty bands."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(times_days, rho_nominal, 'k', linewidth=2, label='Nominal NRLMSISE-00')

    rho_p15 = rho_nominal * 1.15
    rho_m15 = rho_nominal * 0.85
    rho_p100 = rho_nominal * 2.0
    rho_m100 = rho_nominal * 0.0 + 1e-20  # avoid log(0)

    ax.fill_between(times_days, rho_m15, rho_p15,
                    alpha=0.4, color='tab:blue', label='±15% (long-term)')
    ax.fill_between(times_days, rho_m100, rho_p100,
                    alpha=0.2, color='tab:red', label='±100% (short-term worst-case)')

    ax.set_xlabel('Time [days from 1 Jan 2024]')
    ax.set_ylabel('Density [kg/m³]')
    ax.set_title('Fig. 6 — NRLMSISE-00 Density Uncertainty Bands\n'
                 'Vallado & Finkleman (2014) bounds at h = 560 km')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    return _save(fig, 'fig6_density_uncertainty.png')


# ---------------------------------------------------------------------------
# Figure 7 — Baseline manoeuvre (no constraints)
# ---------------------------------------------------------------------------

def fig7_baseline_manoeuvre(result: dict,
                              y0_km: float, target_km: float,
                              pair_label: str = 'A-B'):
    """Fig 7: Baseline manoeuvre trajectory, no operational constraints."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    t_days = result['times_s'] / 86400.0
    y_km   = result['y_t'] / 1e3
    x_km   = result['x_t'] / 1e3

    # --- Along-track ---
    ax1.plot(t_days, y_km, color='tab:blue', linewidth=1, alpha=0.6,
             label='Osculating')
    if len(result['t_mean']) > 0:
        t_mean_d = result['t_mean'] / 86400.0
        y_mean_km = result['y_mean'] / 1e3
        ax1.plot(t_mean_d, y_mean_km, color='tab:orange', linewidth=2,
                 label='Mean (orbital-period average)')

    ax1.axhline(y0_km, color='k', linestyle='--', linewidth=1,
                label=f'Initial sep. {y0_km:.0f} km')
    ax1.axhline(target_km, color='green', linestyle=':', linewidth=1.5,
                label=f'Target sep. {target_km:.0f} km')

    dur = result['duration_days']
    ax1.annotate(f'Duration: {dur:.2f} days',
                 xy=(0.02, 0.05), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.set_ylabel('Along-track separation y [km]')
    ax1.set_title(f'Fig. 7 — Baseline Manoeuvre, S-NET {pair_label}, No Constraints')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Radial ---
    ax2.plot(t_days, x_km, color='tab:red', linewidth=1, alpha=0.8)
    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Radial separation x [km]')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, 'fig7_baseline_manoeuvre.png')


# ---------------------------------------------------------------------------
# Figure 8 — Manoeuvre with ±15% density uncertainty
# ---------------------------------------------------------------------------

def fig8_density_uncert(result_nom: dict, result_p15: dict, result_m15: dict,
                         y0_km: float, target_km: float):
    """Fig 8: Manoeuvre envelope with ±15% long-term density uncertainty."""
    fig, ax = plt.subplots(figsize=(10, 6))

    t_nom = result_nom['t_mean'] / 86400.0
    y_nom = result_nom['y_mean'] / 1e3

    t_p = result_p15['t_mean'] / 86400.0
    y_p = result_p15['y_mean'] / 1e3

    t_m = result_m15['t_mean'] / 86400.0
    y_m = result_m15['y_mean'] / 1e3

    # Envelope: interpolate to common time base
    t_max = min(t_nom[-1] if len(t_nom) else 0,
                t_p[-1]  if len(t_p)  else 0,
                t_m[-1]  if len(t_m)  else 0)
    t_common = np.linspace(0, t_max, 500)

    if len(t_nom) > 1:
        y_nom_i = np.interp(t_common, t_nom, y_nom)
    else:
        y_nom_i = np.full_like(t_common, y0_km)

    if len(t_p) > 1:
        y_p_i = np.interp(t_common, t_p, y_p)
    else:
        y_p_i = y_nom_i.copy()

    if len(t_m) > 1:
        y_m_i = np.interp(t_common, t_m, y_m)
    else:
        y_m_i = y_nom_i.copy()

    ax.fill_between(t_common, y_m_i, y_p_i,
                    alpha=0.35, color='tab:blue', label='±15% density (long-term)')
    ax.plot(t_common, y_nom_i, 'k', linewidth=2, label='Nominal NRLMSISE-00')

    ax.axhline(y0_km,   color='k',     linestyle='--', linewidth=1)
    ax.axhline(target_km, color='green', linestyle=':', linewidth=1.5,
               label=f'Target {target_km:.0f} km')

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Mean along-track separation [km]')
    ax.set_title('Fig. 8 — Manoeuvre with ±15% Density Uncertainty, S-NET A-B')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return _save(fig, 'fig8_density_uncert_manoeuvre.png')


# ---------------------------------------------------------------------------
# Figure 9 — Manoeuvre with ±100% density uncertainty (worst case)
# ---------------------------------------------------------------------------

def fig9_density_worst_case(result_nom: dict, result_p100: dict, result_m100: dict,
                              y0_km: float, target_km: float):
    """Fig 9: Manoeuvre envelope with worst-case ±100% density uncertainty."""
    fig, ax = plt.subplots(figsize=(10, 6))

    def _mean_km(res):
        if len(res['t_mean']) > 1:
            return res['t_mean'] / 86400.0, res['y_mean'] / 1e3
        return np.array([0.0]), np.array([y0_km])

    t_nom, y_nom = _mean_km(result_nom)
    t_p,   y_p   = _mean_km(result_p100)
    t_m,   y_m   = _mean_km(result_m100)

    t_max    = min(t_nom[-1], t_p[-1], t_m[-1]) if (len(t_p) > 1 and len(t_m) > 1) else t_nom[-1]
    t_common = np.linspace(0, t_max, 500)

    y_nom_i = np.interp(t_common, t_nom, y_nom) if len(t_nom) > 1 else np.full(500, y0_km)
    y_p_i   = np.interp(t_common, t_p, y_p)   if len(t_p)  > 1 else y_nom_i.copy()
    y_m_i   = np.interp(t_common, t_m, y_m)   if len(t_m)  > 1 else y_nom_i.copy()

    ax.fill_between(t_common, y_m_i, y_p_i,
                    alpha=0.25, color='tab:red', label='±100% density (worst-case)')
    ax.plot(t_common, y_nom_i, 'k', linewidth=2, label='Nominal')

    ax.axhline(target_km, color='green', linestyle=':', linewidth=1.5,
               label=f'Target {target_km:.0f} km')

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Mean along-track separation [km]')
    ax.set_title('Fig. 9 — Manoeuvre with ±100% Density Uncertainty, S-NET A-B')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return _save(fig, 'fig9_density_worst_case.png')


# ---------------------------------------------------------------------------
# Figure 10 — Manoeuvre duration vs. availability fraction
# ---------------------------------------------------------------------------

def fig10_availability(duration_ideal_days: float, fractions=None):
    """Fig 10: Manoeuvre duration vs. satellite availability fraction."""
    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 50)

    durations = duration_ideal_days / np.asarray(fractions)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, durations, 'tab:blue', linewidth=2)
    ax.axvline(1.0, color='k', linestyle='--', linewidth=1, label='Full availability')
    ax.axvline(0.5, color='tab:orange', linestyle=':', linewidth=1.5,
               label='Representative (50%)')
    ax.set_xlabel('Availability fraction')
    ax.set_ylabel('Manoeuvre duration [days]')
    ax.set_title('Fig. 10 — Manoeuvre Duration vs. Availability Fraction, S-NET A-B')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 1.05)

    return _save(fig, 'fig10_availability.png')


# ---------------------------------------------------------------------------
# Figure 11 — Manoeuvre duration vs. attitude accuracy
# ---------------------------------------------------------------------------

def fig11_attitude_accuracy(approach_results: dict, retreat_results: dict):
    """
    Fig 11: Manoeuvre duration vs. attitude accuracy for all S-NET pairs.

    Parameters
    ----------
    approach_results : dict  {pair: {sigma_deg: duration_days}}
    retreat_results  : dict  {pair: {sigma_deg: duration_days}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sigma_levels = ATT_ACC_LEVELS

    for idx, pair in enumerate(SNET_PAIRS):
        col = _PAIR_COLORS[idx]

        if pair in approach_results:
            d_app = [approach_results[pair].get(s, np.nan) for s in sigma_levels]
            ax.plot(sigma_levels, d_app, 'o-', color=col, linewidth=2,
                    label=f'{pair} approach', markersize=6)

        if pair in retreat_results:
            d_ret = [retreat_results[pair].get(s, np.nan) for s in sigma_levels]
            ax.plot(sigma_levels, d_ret, 's--', color=col, linewidth=1.5,
                    label=f'{pair} retreat', markersize=6, alpha=0.75)

    # Table 7 benchmark overlay for A-B
    from planning_tool import TABLE7_APPROACH, TABLE7_RETREAT
    ref_app = [TABLE7_APPROACH[s] for s in sigma_levels]
    ref_ret = [TABLE7_RETREAT[s]  for s in sigma_levels]
    ax.plot(sigma_levels, ref_app, 'k^-', linewidth=2.5, markersize=8,
            label='A-B approach (Table 7 ref)', zorder=10)
    ax.plot(sigma_levels, ref_ret, 'kv--', linewidth=2.5, markersize=8,
            label='A-B retreat (Table 7 ref)', zorder=10)

    ax.axhline(TABLE7_APPROACH[0.0], color='gray', linestyle=':',
               linewidth=1, label='Baseline (ideal) approach')

    ax.set_xlabel('Attitude accuracy (1σ pointing error) [deg]')
    ax.set_ylabel('Manoeuvre duration [days]')
    ax.set_title('Fig. 11 — Manoeuvre Duration vs. Attitude Accuracy, S-NET Formation')
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sigma_levels)

    return _save(fig, 'fig11_attitude_accuracy.png')


# ---------------------------------------------------------------------------
# Figure 12 — Full mission scenario (2x3 grid, one subplot per pair)
# ---------------------------------------------------------------------------

def fig12_mission_scenario(pair_results: dict, targets_km: dict):
    """
    Fig 12: Full S-NET formation reconfiguration scenario.

    Parameters
    ----------
    pair_results : dict  {pair: result_dict from manoeuvre_planner}
    targets_km   : dict  {pair: target_separation_km}
    """
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    for idx, pair in enumerate(SNET_PAIRS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        if pair not in pair_results:
            ax.set_title(f'{pair} — no data')
            continue

        res    = pair_results[pair]
        t_d    = res['times_s'] / 86400.0
        y_km   = res['y_t'] / 1e3
        color  = _PAIR_COLORS[idx]
        target = targets_km.get(pair, 0.0)
        y0_km  = y_km[0] if len(y_km) else 0.0

        # Nominal trajectory
        ax.plot(t_d, y_km, color=color, linewidth=1, alpha=0.5)

        # Mean trajectory
        if len(res['t_mean']) > 1:
            ax.plot(res['t_mean'] / 86400.0, res['y_mean'] / 1e3,
                    color=color, linewidth=2.5)

        # Uncertainty band (±15%)
        y_lo = y_km * 0.85
        y_hi = y_km * 1.15
        ax.fill_between(t_d, y_lo, y_hi, alpha=0.25, color=color)

        ax.axhline(y0_km,  color='k',     linestyle='--', linewidth=1)
        ax.axhline(target, color='green', linestyle=':',  linewidth=1.5)

        dur = res['duration_days']
        ax.set_title(f'Pair {pair}  |  {dur:.1f} d', fontsize=10)
        ax.set_xlabel('Time [days]', fontsize=8)
        ax.set_ylabel('Δy [km]', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.suptitle('Fig. 12 — Full S-NET Formation Reconfiguration Scenario\n'
                 'Availability=50%, Attitude σ=10°, Nominal Density ±15%',
                 fontsize=13)

    return _save(fig, 'fig12_mission_scenario.png')
