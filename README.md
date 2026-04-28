# Warp Drive QI Audit & Toroidal PIC Simulation

This repository houses two distinct, high-performance physics simulations related to advanced propulsion concepts:
1. **Toroidal PIC Simulation**: A Particle-in-Cell simulation of Xe⁺ ions in a toroidal magnetic field.
2. **Warp Drive QI Audit**: A numerical evaluation of the physical feasibility of Alcubierre, White-Natário, and Lentz warp metrics.

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

All configuration lives in the `CFG` dictionary at the top of the script. Key parameters include `N_particles`, `B0`, `t_end`, `dt`, `NR x NZ`, and `macro_weight`. All output goes to the `pic_output/` directory (e.g. `particles_final.npz`, `energy_vs_time.png`, `cyclotron_fft.png`, `density_NNNNNN.png`).

---

## 2. Warp Drive QI Audit

An open-source quantum-inequality audit of Alcubierre, White-Natário, and Lentz warp metrics evaluating their physical feasibility using the **Ford-Roman Quantum Inequality (QI)** bound and the **ADM Hamiltonian constraint** for extrinsic curvature.

### Key Results

| Metric | Energy type | Key result |
|---|---|---|
| **Alcubierre (1994)** | Negative (exotic) | Exceeds QI by ~68–69 orders of magnitude |
| **White-Natário** | Negative (exotic) | Peak reduced 70%, total comparable to Alcubierre |
| **Lentz (2021)** | Positive | ~420 Earth masses at 1.1c; collapses to noise at useful speeds |

**Bottom line:** No known warp metric is physically feasible within current physics. This repository makes the math explicit and reproducible.

### Usage

Run the full QI audit with the new command-line interface:

```bash
python testfisico.py --delta 0.2614 --radius 3.0 --velocity_c 1.1 --mass_ship 100000
```

This produces:
- Console output: QI bound, gap analysis, hypothetical Δ, subluminal comparison
- `qi_vs_Delta.png` — QI energy vs wall thickness
- `qi_vs_Radius.png` — QI energy vs bubble radius
- `qi_vs_Radius_with_SHIP.png` — QI limit vs ship requirement
- `metric_comparison.png` — Four-panel metric comparison

### Run the metric explorer standalone

```python
from metric_explorer import run_metric_comparison

run_metric_comparison(
    v_s   = 1.1 * 2.99792458e8,   # 1.1c in m/s
    R     = 3.0,                   # bubble radius [m]
    Delta = 0.2614,                # wall thickness [m]
)
```

### Run tests

```bash
pytest test_physics.py
```

### Caveats and Scope
- **Lentz Metric:** The Lentz energy density uses an algebraic ADM estimate. Lentz 2021 required a full Einstein-Maxwell-plasma coupling to get T₀₀ ≥ 0 — you can't reproduce that with the same scalar shift-vector formula just flipping a sign. The ~420 Earth masses figure is indicative at best.
- **QI Bound:** The QI bound applied here is the Ford-Roman (1995) flat-spacetime result; curved-spacetime corrections exist but do not change the order-of-magnitude gap.
- **E_req_total:** The total energy requirement is a dimensional order-of-magnitude estimate from Alcubierre's original paper, not a precision numerical ADM result.

---

## Installation & Dependencies

A `requirements.txt` is provided with pinned versions for scientific reproducibility:

```bash
git clone https://github.com/mabgresearch/warp-qi-audit.git
cd warp-qi-audit
pip install -r requirements.txt
```

*(Note: Numba is optional but strongly recommended for the Toroidal PIC Simulation performance. The code falls back to pure NumPy if Numba is not found).*

## License

MIT — see [LICENSE](LICENSE).

Contributions, issues, and pull requests welcome.
