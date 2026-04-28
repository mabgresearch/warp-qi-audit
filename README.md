# Toroidal PIC Simulation — README

## Overview

`toroidal_pic.py` is a self-contained, production-ready Particle-in-Cell (PIC)
simulation of **100,000 Xe⁺ ions** confined in a **toroidal magnetic field**
(2.6 T peak, major radius 3 m, minor radius 0.5 m).

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

## Dependencies

A `requirements.txt` is provided with pinned versions for scientific reproducibility:

```bash
pip install -r requirements.txt
```

If Numba is not installed the code falls back to pure NumPy (much slower).

## Execution

```bash
python toroidal_pic.py
```

All configuration lives in the `CFG` dictionary at the top of the script.
Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `N_particles` | 100 000 | Number of Xe⁺ macro-particles |
| `B0` | 2.6 T | Peak toroidal field |
| `t_end` | 0.1 ms | Total simulation time |
| `dt` | 0.2 µs | Timestep (ωc·dt ≈ 0.38) |
| `NR × NZ` | 128 × 64 | Poisson grid resolution |
| `macro_weight` | 1.0 | Scale factor for particle charge |

## Output

All output goes to the `pic_output/` directory:

| File | Contents |
|---|---|
| `particles_final.npz` | Final (x,y,z,vx,vy,vz,alive) arrays |
| `energy_vs_time.png` | Total kinetic energy evolution |
| `cyclotron_fft.png` | FFT of total Z-current (resonance peak) |
| `density_NNNNNN.png` | R-Z density snapshot at each diagnostic step |

## Approximate runtime

| Hardware | Numba | Estimated wall time |
|---|---|---|
| Modern laptop (8-core) | Yes | 2–5 minutes |
| Modern laptop | No (pure NumPy) | 30–60 minutes |
| GPU (CuPy path) | N/A | Not yet implemented |

The Boris pusher loop (the bottleneck) is parallelised via `numba.prange`.

## Key physics notes

- **Timestep**: dt = 0.2 µs gives ωc·dt ≈ 0.38 (Xe⁺ cyclotron period ≈ 3.3 µs at 2.6 T). The Boris method is unconditionally stable for uniform B but ωc·dt < 0.5 ensures good orbit accuracy.
- **Self-field**: At 100k physical particles the Debye length vastly exceeds the device, so the self-consistent E-field is negligible. Increase `macro_weight` (e.g. 10¹⁰) for meaningful self-field effects.
- **Confinement**: The vertical field Bz prevents vertical particle drift out of the midplane.

## License

MIT
