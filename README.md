# Warp Drive QI Audit

**An open-source quantum-inequality audit of Alcubierre, White-Natário, and Lentz warp metrics**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Overview

This repository contains two Python modules that numerically evaluate the physical feasibility of three proposed warp-drive spacetime geometries using the **Ford-Roman Quantum Inequality (QI)** bound and the **ADM Hamiltonian constraint** for extrinsic curvature.

The analysis covers:

| Metric | Energy type | Key result |
|---|---|---|
| **Alcubierre (1994)** | Negative (exotic) | Exceeds QI by ~68–69 orders of magnitude |
| **White-Natário** | Negative (exotic) | Peak reduced 70%, total comparable to Alcubierre |
| **Lentz (2021)** | Positive | ~420 Earth masses at 1.1c; collapses to noise at useful speeds |

**Bottom line:** No known warp metric is physically feasible within current physics. This repository makes the math explicit and reproducible.

---

## Physics Background

### Quantum Inequality (Ford-Roman bound)

The QI constrains how negative the energy density of any quantum field can be over a sampling timescale τ₀:

```
ρ_QI ≥ −(3ℏ) / (32π²c³ τ₀⁴)
```

For a bubble wall of thickness Δ, the natural sampling time is τ₀ = Δ/c. The total negative energy allowed in the shell volume 4πR²Δ is then:

```
|E_QI| = |ρ_QI| · 4πR²Δ
```

For the parameters used here (Δ = 0.2614 m, R = 3.0 m), this gives **|E_QI| ≈ 1.9 × 10⁻²⁴ J** — roughly one photon's worth of energy.

### ADM Energy Density

For shift-vector metrics of the form `ds² = −c²dt² + (dx − β^x dt)² + dy² + dz²` with `β^x = v_s · f(r_s)`, the angle-averaged energy density from the Hamiltonian constraint is:

```
ρ(r_s) = ∓ (v_s² c²) / (96π G) · [f′(r_s)]²
```

The sign is negative for Alcubierre/White-Natário (exotic matter required) and positive for the Lentz soliton.

### Shaping Functions

- **Alcubierre:** Classic top-hat via `tanh`, transition width ~ 1/σ
- **White-Natário:** Thickened wall (σ_eff = σ/2) with oscillatory micro-structure; reduces peak ρ but not total energy
- **Lentz:** Dipole-like `sech²` difference yielding T₀₀ ≥ 0 everywhere (at enormous energy cost)

---

## Repository Structure

```
warp-qi-audit/
├── testfisico.py        # QI bound analysis, phase-space, hypothetical gap closure
├── metric_explorer.py   # ADM energy density for all three metrics + plots
├── results/
│   └── sample_output.txt  # Full console output from a reference run
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/mabgresearch/warp-qi-audit.git
cd warp-qi-audit
pip install -r requirements.txt
```

Requires Python 3.8+. Compatible with NumPy 1.x and 2.x (trapz/trapezoid handled automatically).

---

## Usage

### Run the full QI audit (Part 1 + metric comparison)

```bash
python testfisico.py
```

This produces:
- Console output: QI bound, gap analysis, hypothetical Δ, subluminal comparison
- `qi_vs_Delta.png` — QI energy vs wall thickness
- `qi_vs_Radius.png` — QI energy vs bubble radius
- `qi_vs_Radius_with_SHIP.png` — QI limit vs ship requirement
- `phase_space_ship_vs_qi.png` — Phase-space footprint comparison
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

### Default parameters

| Parameter | Value | Description |
|---|---|---|
| `v_s` | 1.1c | Bubble coordinate speed |
| `R` | 3.0 m | Bubble radius |
| `Δ` | 0.2614 m | Wall thickness (σ ≈ 3.8 m⁻¹) |
| `m_ship` | 100,000 kg | Reference ship mass (100 tonnes) |

---

## Key Results

From `results/sample_output.txt` (v_s = 1.1c, R = 3.0 m, Δ = 0.2614 m):

```
Metric               Peak ρ [J/m³]     E_total [J]    Sign
Alcubierre 1994        -1.776e+42       -7.019e+43     NEG
White-Natário          -5.258e+41       -3.924e+43     NEG
Lentz 2021             +4.210e+42       +2.254e+44     POS

QI Cap (Ford-Roman)  :  1.902e-24 J
Alcubierre negative  :  7.019e+43 J
QI gap factor        :  3.7e+67
```

The Lentz soliton requires **~420 Earth masses** of positive energy at 1.1c. The speed at which its total energy falls below a "small asteroid" threshold (10²⁰ J) is:

```
v_crossover = 1.00e-10 c  =  3 cm/s
```

At that speed the metric perturbation is negligible — no propulsion advantage.

---

## Caveats and Scope

- The Lentz energy density uses an **algebraic ADM estimate**, not a full numerical integration of the Lentz stress-energy tensor. Results are indicative.
- The QI bound applied here is the Ford-Roman (1995) flat-spacetime result; curved-spacetime corrections exist but do not change the order-of-magnitude gap.
- This code is an **educational tool** for understanding why warp drives are physically prohibited, not a design tool.

---

## References

1. Alcubierre, M. (1994). *The warp drive: hyper-fast travel within general relativity.* Classical and Quantum Gravity, 11(5), L73.
2. Ford, L. H., & Roman, T. A. (1995). *Averaged energy conditions and quantum inequalities.* Physical Review D, 51(8), 4277.
3. White, H. (2011). *A discussion of space-time metric engineering.* Acta Astronautica, 68(7–8), 1081–1094.
4. Natário, J. (2002). *Warp drive with zero expansion.* Classical and Quantum Gravity, 19(6), 1157.
5. Lentz, E. W. (2021). *Breaking the warp barrier: hyper-fast solitons in Einstein-Maxwell-plasma theory.* Classical and Quantum Gravity, 38(7), 075015.

---

## License

MIT — see [LICENSE](LICENSE).

Contributions, issues, and pull requests welcome.
