# Warp Drive QI Audit & Toroidal PIC Simulation

This repository houses two distinct, high-performance physics simulations related to advanced propulsion concepts:
1. **Toroidal PIC Simulation**: A Particle-in-Cell simulation of Xe⁺ ions in a toroidal magnetic field.
2. **Warp Drive QI Audit**: A numerical evaluation of the physical feasibility of Alcubierre, White-Natário, **Rodal (2025)**, and **Fuchs (2024)** warp metrics against the Ford–Roman quantum inequality.

> [!WARNING]
> **Lentz (2021) is NOT included in this audit.** The Lentz metric requires a full Einstein–Maxwell–plasma coupling to achieve T₀₀ ≥ 0. The simple sign-flip of the Alcubierre ADM Hamiltonian formula is physically invalid and has been removed. See Lentz (2021) §IV and Bobrick & Martire (2021) for context.

---

## 1. Toroidal PIC Simulation

> [!WARNING]
> **EXPERIMENTAL — not yet peer-validated, use results with caution.** This simulation code is currently unvalidated and should not be treated with the same confidence as the Zenodo-cited QI Audit below.

`toroidal_pic.py` is a self-contained, experimental Particle-in-Cell (PIC) simulation of **100,000 Xe⁺ ions** confined in a **toroidal magnetic field** (2.6 T peak, major radius 3 m, minor radius 0.5 m).

### Physics modelled

| Component | Method |
|---|---|
| Particle integrator | Boris pusher + leapfrog (3-D Cartesian) |
| Magnetic field | Toroidal B = B₀(R₀/R) φ̂ + Bz ẑ (1/R decay + vertical) |
| Electric field | Self-consistent via 2-D (R,Z) Poisson solve |
| Poisson solver | DST (Type-I) in Z + tridiagonal (Thomas) in R |
| Charge deposition | Area-weighted (CIC) on cylindrical grid |
| Boundaries | Conducting walls; particles absorbed on contact |

### Diagnostics

- Kinetic energy vs. time
- Cyclotron resonance peak (FFT of total axial current)
- Density maps (R, Z) saved as PNG every 1 µs

### Execution

```bash
python toroidal_pic.py
```

All configuration lives in the `CFG` dictionary at the top of the script. All output goes to the `pic_output/` directory.

---

## 2. Warp Drive QI Audit

An open-source quantum-inequality audit of Alcubierre, White-Natário, Rodal (2025), and Fuchs (2024) warp metrics evaluating their physical feasibility using the **Ford-Roman Quantum Inequality (QI)** bound and the **ADM Hamiltonian constraint** for extrinsic curvature.

### Key Results

| Metric | Energy type | Key result |
|---|---|---|
| **Alcubierre (1994)** | Negative (exotic) | Exceeds QI by ~68–69 orders of magnitude |
| **White-Natário** | Negative (exotic) | Peak reduced 70%, total comparable to Alcubierre |
| Rodal (2025)† | Predominantly positive, net ≈ 0 | Peak deficit reduced ~38× vs. Alcubierre; global Type I; QI exceeded by ~10⁶³ |
| Fuchs (2024)* | Positive only (shell) | Zero negative energy – passes QI audit trivially. Requires ~1.8 M☉ of positive-energy shell. Subluminal only. |

\* Fuchs is subluminal; QI gap is trivial because no negative energy is present.  
† Rodal QI gap computed using the actual diffuse negative-energy volume V₋ (numerically integrated).

> [!IMPORTANT]
> **Precise conclusion:** None of the audited superluminal warp metrics satisfy the Ford–Roman quantum inequality bound. This tool has not yet evaluated all known solutions; notably, the Bobrick–Martire class of "positive-energy warp drives" has not been audited.

**Bottom line:** No known warp metric is physically feasible within current physics. This repository makes the math explicit and reproducible.

The Rodal (2025) irrotational warp drive [arXiv:2512.18008] is the first fully explicit, continuous, analytically derived solution with vanishing spatial vorticity. Our implementation reproduces its published canonical results. Despite classical improvements, the QI bound is still exceeded by a factor ~10⁶³.

The Fuchs (2024) constant-velocity subluminal warp drive [Class. Quantum Grav. 41 (2024) 095013] is the only known metric that completely avoids negative energy, satisfying all classical energy conditions. Our independent audit (powered by warpax) confirms this across the full subluminal parameter space.

### Usage

Run the full QI audit with the command-line interface:

```bash
python testfisico.py --delta 0.2614 --radius 3.0 --velocity_c 1.1 --mass_ship 100000
```

This produces:
- Console output: QI bound, gap analysis, hypothetical Δ, subluminal comparison
- `qi_vs_Delta.png` / `.pdf` — QI energy vs wall thickness
- `qi_vs_Radius.png` / `.pdf` — QI energy vs bubble radius
- `qi_vs_Radius_with_SHIP.png` / `.pdf` — QI limit vs ship requirement
- `energy_vs_velocity.png` / `.pdf` — Required energy vs ship speed
- `metric_comparison.png` / `.pdf` — Four-panel metric comparison
- `rodal_energy_map.png` / `.pdf` — 2-D polar map of Rodal proper energy density

### Run the metric explorer standalone

```python
from metric_explorer import run_metric_comparison

run_metric_comparison(
    v_s   = 1.1 * 2.99792458e8,   # 1.1c in m/s
    R     = 3.0,                   # bubble radius [m]
    Delta = 0.2614,                # wall thickness [m]
    include_rodal=True
)
```

### Fuchs/WarpShell audit (requires warpax)

```bash
python convergence_fuchs.py     # prove E_minus = 0 is converged
python sweep_fuchs.py           # generate fuchs_phase_space.png
python warpax_qi_audit.py       # single-point audit
```

### Run tests

```bash
pytest tests/
```

### Caveats and Scope

- **Lentz Metric:** **Not evaluated.** Lentz (2021) requires a full Einstein–Maxwell–plasma coupling to achieve T₀₀ ≥ 0 everywhere. The sign-flip of the Alcubierre ADM formula is physically invalid and has been removed from this codebase. Future work would need to implement the full Lentz stress-energy tensor.
- **Bobrick–Martire class:** Not yet audited. These solutions may have different energy properties and are a target for future work.
- **Rodal Metric:** The proper energy density is computed from the spatial Hessian of the scalar potential Φ(r,θ). The QI gap for Rodal uses the actual diffuse negative-energy volume V₋ rather than a thin-shell model.
- **Alcubierre & White-Natário:** The QI volume now uses the numerically integrated negative-energy volume V_minus = ∫_{ρ<0} 4π r² dr, replacing the thin-shell approximation 4πR²Δ previously used.
- **Fuchs Metric:** Results obtained via warpax (MIT-licensed). The audit confirms zero negative energy for the canonical subluminal parameters.
- **QI Bound:** The QI bound applied here is the Ford-Roman (1995) flat-spacetime result; curved-spacetime corrections exist but do not change the order-of-magnitude gap.
- **E_req_total:** The total energy requirement is now the numerically integrated ADM result for the Alcubierre metric, not a dimensional estimate.

---

## Installation & Dependencies

A `requirements.txt` is provided with pinned versions for scientific reproducibility:

```bash
git clone https://github.com/mabgresearch/warp-qi-audit.git
cd warp-qi-audit
pip install -r requirements.txt
```

*(Note: Numba is optional but strongly recommended for the Toroidal PIC Simulation performance.)*

## License

MIT — see [LICENSE](LICENSE).

Contributions, issues, and pull requests welcome.
