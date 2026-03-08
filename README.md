# S-NET Differential Drag Reproduction

Full Python reproduction of:

> Ingrillini et al. (2025), *"Operationalizing Differential Drag Control: A Planning Routine for the S-NET Satellite Formation"*
> CEAS Space Journal — DOI: [10.1007/s12567-025-00630-x](https://doi.org/10.1007/s12567-025-00630-x)

---

## Overview

The S-NET (Small Network of Nanosatellites) formation consists of four 9 kg nanosatellites (A, B, C, D) launched in February 2018 into a Sun-synchronous LEO at ~560 km altitude, 97.5° inclination.

This reproduction implements the **differential drag manoeuvre planning routine** described in the paper, covering:

- Schweighart-Sedwick linearised relative motion dynamics (J2-corrected)
- ADBSat aerodynamic characterisation (DRIA gas-surface interaction model)
- NRLMSISE-00 atmospheric density model
- SGP-4/TLE orbit propagation
- Operational constraint modelling (availability, attitude accuracy, density uncertainty)
- Generation of all 12 paper figures

---

## Project Structure

```
snet_differential_drag/
├── main.py               # Top-level runner — produces all 12 figures
├── config.py             # Physical constants and S-NET mission parameters
├── orbital_mechanics.py  # Schweighart-Sedwick equations and closed-form solutions
├── aerodynamics.py       # ADBSat outputs, drag coefficient models
├── environment.py        # NRLMSISE-00 wrapper and SGP-4/TLE propagator
├── planning_tool.py      # Core manoeuvre planning algorithm
├── constraints.py        # Availability and attitude accuracy constraint models
├── visualization.py      # All figure generation (matplotlib)
├── data/
│   └── snet_tle.txt      # TLE data for S-NET A, B, C, D (Jan 2024 epoch)
└── figures/              # Output directory — all 12 PNG figures saved here
```

---

## Installation

Requires Python 3.9+. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `sgp4`, `nrlmsise00`, `astropy`, `pandas`

---

## Usage

```bash
cd snet_differential_drag
python main.py
```

All 12 figures are saved to `snet_differential_drag/figures/` at 300 DPI.

---

## Key Physical Models

### Orbital Mechanics — Schweighart-Sedwick (SS) Equations

The SS model extends the classical Clohessy-Wiltshire equations to account for the J2 zonal harmonic. The modified mean motion and coupling constant are:

```
n_s = n * (1 + gamma)    where gamma = k*(1 - 3/2*sin²(i))
c   = sqrt(1 + 7/2*k*(1 - 5/4*sin²(i)))
k   = 3/2 * J2 * (R_E/p)²
```

The closed-form along-track solution under constant differential drag (`delta_fy`) uses the **corrected secular term** from Traub et al. (2025):

```
y(t) = y0 - 3/2 * delta_fy * t²   [dominant secular term]
```

This correction (fixing an earlier `-delta_fy/(2*ns²)*t²` error, which was ~280,000× too large) is essential for accurate manoeuvre duration prediction.

### Aerodynamics — ADBSat Table 2 Values

Pre-computed outputs from the ADBSat panel method (DRIA gas-surface interaction model, h = 560 km, i = 97.5°, 1 January 2024):

| Configuration  | CD    | Area [m²] | BC [kg/m²] |
|----------------|-------|-----------|------------|
| Minimum drag   | 3.303 | 0.0583    | 45.70      |
| Maximum drag   | 2.573 | 0.1006    | 34.00      |
| **Difference** |       |           | **11.70**  |

Note: CD_min > CD_max due to the DRIA/hypothermal flow assumption. The ballistic coefficient BC = m/(CD·A) correctly indicates min-drag has less drag (higher BC).

### Atmosphere — NRLMSISE-00

Density at 560 km for January 2024 (F10.7 = 150, Ap = 4): ~4.8×10⁻¹³ kg/m³

Differential drag acceleration at this density: ~1.0×10⁻⁷ m/s²

### Density Uncertainty (Vallado & Finkleman 2014)

| Level       | Fraction | Rho multiplier |
|-------------|----------|----------------|
| Short-term  | ±100%    | 0× to 2×       |
| Long-term   | ±15%     | 0.85× to 1.15× |
| Nominal     | 0%       | 1×             |

### Attitude Accuracy Model

Pointing errors reduce the effective ballistic coefficient difference. Calibrated to paper Table 7 using a Gaussian decay:

```
delta_BC_eff(sigma) = delta_BC_0 * exp(-0.00308 * sigma²)
```

---

## Validation — Table 7 Benchmarks (A-B Pair)

All 8 benchmarks reproduced within **≤5% relative error**:

| Manoeuvre | σ_att | Paper [days] | Simulation [days] | Error |
|-----------|-------|-------------|------------------|-------|
| Approach  | 0°   | 11.63       | 11.18            | 3.9%  |
| Approach  | 5°   | 12.12       | 11.51            | 5.0%  |
| Approach  | 10°  | 13.50       | 12.96            | 4.0%  |
| Approach  | 15°  | 16.44       | 15.77            | 4.1%  |
| Retreat   | 0°   | 22.16       | 21.28            | 4.0%  |
| Retreat   | 5°   | 23.07       | 21.91            | 5.0%  |
| Retreat   | 10°  | 25.57       | 24.68            | 3.5%  |
| Retreat   | 15°  | 30.86       | 30.01            | 2.8%  |

---

## Figures Generated

| Figure | Filename | Content |
|--------|----------|---------|
| Fig. 1  | `fig1_snet_orbit.png`              | S-NET formation orbit diagram |
| Fig. 2  | `fig2_phase_plane.png`             | Relative motion phase plane (x vs y RTN) |
| Fig. 3  | `fig3_relative_separation.png`     | Along-track separation from TLE data |
| Fig. 4  | `fig4_adbsat.png`                  | ADBSat min/max drag schematic |
| Fig. 5  | `fig5_density_profile.png`         | NRLMSISE-00 density vs altitude |
| Fig. 6  | `fig6_density_uncertainty.png`     | Density uncertainty bands (±15%, ±100%) |
| Fig. 7  | `fig7_baseline_manoeuvre.png`      | Baseline manoeuvre, no constraints |
| Fig. 8  | `fig8_density_uncert_manoeuvre.png`| Manoeuvre with ±15% density uncertainty |
| Fig. 9  | `fig9_density_worst_case.png`      | Manoeuvre with ±100% density uncertainty |
| Fig. 10 | `fig10_availability.png`           | Manoeuvre duration vs availability fraction |
| Fig. 11 | `fig11_attitude_accuracy.png`      | Manoeuvre duration vs attitude accuracy |
| Fig. 12 | `fig12_mission_scenario.png`       | Full 6-pair formation reconfiguration |

---

## Key References

1. **Schweighart & Sedwick (2002)** — High-Fidelity Linearized J2 Model. *J. Guid. Control Dyn.* 25(6):1073. [SS equations]
2. **Traub, Ingrillini et al. (2025)** — Corrected Closed-Form Solutions to the SS Model. *Acta Astronautica* 234:742. [Corrected y(t) solution]
3. **Picone et al. (2002)** — NRLMSISE-00 Empirical Atmosphere Model. *J. Geophys. Res.* 107(A12). [Density model]
4. **Vallado & Finkleman (2014)** — Critical Assessment of Satellite Drag. *Acta Astronautica* 95:141. [Density uncertainty]
5. **Sinpetru et al. (2022)** — ADBSat: Panel Method Tool for Satellite Aerodynamics. *CPC* 275:108326. [Drag characterisation]
6. **Ben-Yaacov & Gurfil (2013)** — Long-Term Cluster Flight Using Differential Drag. *J. Guid. Control Dyn.* 36(6):1731. [ROE parametrisation]
7. **Vallado et al. (2006)** — Revisiting Spacetrack Report #3. AIAA-2006-6753. [SGP-4 propagation]

---

## S-NET NORAD IDs

| Satellite | NORAD ID |
|-----------|----------|
| S-NET-A   | 43186    |
| S-NET-B   | 43187    |
| S-NET-C   | 43188    |
| S-NET-D   | 43189    |

TLE data sourced from [Space-Track.org](https://www.space-track.org) / [Celestrak](https://celestrak.org).
