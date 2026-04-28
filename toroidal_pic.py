#!/usr/bin/env python3
"""
toroidal_pic.py – Axisymmetric PIC simulation of Xe⁺ ions in a toroidal B field.

Physics: Boris pusher + leapfrog in 3-D Cartesian coordinates.
         Self-consistent E via 2-D (R,Z) Poisson solve (DST in Z, tridiagonal in R).
         Toroidal B = B0*(R0/R) phi-hat + Bz_vert z-hat.

Author : Mauricio Bravo (2026)
License: MIT
"""

import os, time, warnings
import numpy as np
from scipy.fft import dst, idst
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── try Numba; fall back to pure NumPy stubs ──
try:
    from numba import njit, prange, config as nb_config
    nb_config.THREADING_LAYER = "omp"
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*a, **kw):
        def wrap(f): return f
        return wrap
    prange = range

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # Geometry
    R0        = 3.0,        # major radius [m]
    a         = 0.5,        # minor radius [m]
    R_min     = 0.3,        # inner wall [m]
    R_max     = 3.5,        # outer wall [m]
    Z_min     = -1.0,       # bottom wall [m]
    Z_max     = 1.0,        # top wall [m]
    # Magnetic field
    B0        = 2.6,        # peak toroidal field [T]
    Bz_vert   = 0.05,       # vertical confinement [T]
    # Particles
    N_particles = 100_000,
    T_eV        = 10.0,     # temperature [eV]
    macro_weight= 1.0,      # macro-particle weight (scale for stronger self-field)
    # Time
    t_end     = 1.0e-4,     # 0.1 ms
    dt        = 2.0e-7,     # 0.2 µs  (ωc·dt ≈ 0.38)
    diag_dt   = 1.0e-6,     # output every 1 µs
    # Grid
    NR        = 128,
    NZ        = 64,
    # Output
    out_dir   = "pic_output",
)

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
QE   = 1.602176634e-19       # elementary charge [C]
ME   = 9.1093837015e-31      # electron mass [kg]
AMU  = 1.66053906660e-27     # atomic mass unit [kg]
EPS0 = 8.8541878128e-12      # vacuum permittivity [F/m]
M_XE = 131.293 * AMU         # Xe mass [kg]
Q_XE = QE                    # Xe⁺ charge [C]

# ═══════════════════════════════════════════════════════════════════════════════
# GRID
# ═══════════════════════════════════════════════════════════════════════════════

def build_grid(cfg):
    """Return node-centred R, Z arrays and spacings."""
    NR, NZ = cfg["NR"], cfg["NZ"]
    R = np.linspace(cfg["R_min"], cfg["R_max"], NR + 1)
    Z = np.linspace(cfg["Z_min"], cfg["Z_max"], NZ + 1)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    return R, Z, dR, dZ

# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_particles(cfg, rng=None):
    """
    Place particles uniformly inside the torus cross-section
    (R-R0)²+Z² < a² with random toroidal angle, Maxwellian velocities.
    Returns x, y, z, vx, vy, vz  (all shape (N,)).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    R0, a, N = cfg["R0"], cfg["a"], cfg["N_particles"]
    v_th = np.sqrt(QE * cfg["T_eV"] / M_XE)

    # Rejection-sample uniform in the circular cross-section
    Rp = np.empty(N)
    Zp = np.empty(N)
    filled = 0
    while filled < N:
        batch = max(N - filled, 1024)
        r_try = R0 + a * (2 * rng.random(batch) - 1)
        z_try = a * (2 * rng.random(batch) - 1)
        mask  = (r_try - R0)**2 + z_try**2 < a**2
        n_ok  = mask.sum()
        end   = min(filled + n_ok, N)
        Rp[filled:end] = r_try[mask][:end - filled]
        Zp[filled:end] = z_try[mask][:end - filled]
        filled = end

    phi_tor = 2.0 * np.pi * rng.random(N)
    x = Rp * np.cos(phi_tor)
    y = Rp * np.sin(phi_tor)
    z = Zp.copy()

    # Maxwellian velocities (3-D isotropic)
    vx = rng.normal(0.0, v_th, N)
    vy = rng.normal(0.0, v_th, N)
    vz = rng.normal(0.0, v_th, N)
    return x, y, z, vx, vy, vz

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGE DEPOSITION  (area-weighted, cylindrical)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _deposit_serial(x, y, z, q_macro, R_arr, Z_arr, dR, dZ, NR, NZ, rho):
    """Deposit charge density onto (NR+1)x(NZ+1) node grid.  Serial inner."""
    Rmin = R_arr[0]
    Zmin = Z_arr[0]
    N = x.shape[0]
    for p in range(N):
        Rp = np.sqrt(x[p]*x[p] + y[p]*y[p])
        Zp = z[p]
        fi = (Rp - Rmin) / dR
        fj = (Zp - Zmin) / dZ
        i0 = int(fi)
        j0 = int(fj)
        if i0 < 0 or i0 >= NR or j0 < 0 or j0 >= NZ:
            continue
        wr = fi - i0
        wz = fj - j0
        w00 = (1-wr)*(1-wz)
        w10 = wr*(1-wz)
        w01 = (1-wr)*wz
        w11 = wr*wz
        rho[i0,   j0]   += q_macro * w00
        rho[i0+1, j0]   += q_macro * w10
        rho[i0,   j0+1] += q_macro * w01
        rho[i0+1, j0+1] += q_macro * w11


def deposit_charge(x, y, z, cfg, R_arr, Z_arr, dR, dZ):
    NR, NZ = cfg["NR"], cfg["NZ"]
    q_macro = Q_XE * cfg["macro_weight"]
    Q_grid = np.zeros((NR+1, NZ+1), dtype=np.float64)
    _deposit_serial(x, y, z, q_macro, R_arr, Z_arr, dR, dZ, NR, NZ, Q_grid)
    # Convert accumulated charge to density: ρ = Q / V_cell
    rho = np.zeros_like(Q_grid)
    for i in range(NR+1):
        Ri = R_arr[i]
        if Ri > 0:
            vol = 2.0 * np.pi * Ri * dR * dZ
            rho[i, :] = Q_grid[i, :] / vol
    return rho

# ═══════════════════════════════════════════════════════════════════════════════
# POISSON SOLVER  (DST in Z + tridiagonal in R)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_tridiag_coeffs(R_arr, dR, NR):
    """Pre-compute tridiagonal coefficients for the R operator."""
    # Interior nodes i = 1 .. NR-1
    n = NR - 1
    a_coeff = np.zeros(n)
    c_coeff = np.zeros(n)
    diag0   = np.zeros(n)   # without the Z eigenvalue part
    for idx in range(n):
        i = idx + 1  # actual node index
        Ri   = R_arr[i]
        Rim  = 0.5*(R_arr[i-1] + R_arr[i])   # R_{i-1/2}
        Rip  = 0.5*(R_arr[i]   + R_arr[i+1]) # R_{i+1/2}
        a_coeff[idx] = Rim / (Ri * dR * dR)
        c_coeff[idx] = Rip / (Ri * dR * dR)
        diag0[idx]   = -(a_coeff[idx] + c_coeff[idx])
    return a_coeff, c_coeff, diag0


@njit(cache=True)
def _tridiag_solve(a, b, c, d, n):
    """Thomas algorithm for a tridiagonal system of size n."""
    cp = np.empty(n)
    dp = np.empty(n)
    x  = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = 1.0 / (b[i] - a[i] * cp[i-1])
        cp[i] = c[i] * m
        dp[i] = (d[i] - a[i] * dp[i-1]) * m
    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


def solve_poisson(rho, R_arr, Z_arr, dR, dZ, NR, NZ,
                   a_coeff, c_coeff, diag0):
    """
    Solve  (1/R)∂_R(R ∂_R φ) + ∂²_Z φ = -ρ/ε0
    with φ=0 on all boundaries.
    """
    phi = np.zeros((NR+1, NZ+1))
    n_interior_z = NZ - 1
    n_interior_r = NR - 1
    if n_interior_z < 1 or n_interior_r < 1:
        return phi

    # Extract interior RHS in Z (j=1..NZ-1) for each R-node (i=1..NR-1)
    rhs_interior = -rho[1:NR, 1:NZ] / EPS0   # shape (NR-1, NZ-1)

    # DST-I in Z for each row
    rhs_hat = dst(rhs_interior, type=1, axis=1)

    # Z-eigenvalues:  λ_k = -(4/dZ²) sin²(kπ/(2*NZ))  for k=1..NZ-1
    k_modes = np.arange(1, NZ)
    lam_z   = -(4.0 / (dZ*dZ)) * np.sin(k_modes * np.pi / (2.0 * NZ))**2

    # Solve tridiagonal in R for each Z-mode
    phi_hat = np.zeros_like(rhs_hat)  # (NR-1, NZ-1)
    for km in range(n_interior_z):
        b_full = diag0 + lam_z[km]
        phi_hat[:, km] = _tridiag_solve(
            a_coeff, b_full, c_coeff, rhs_hat[:, km], n_interior_r)

    # Inverse DST-I in Z
    phi_interior = idst(phi_hat, type=1, axis=1) / (2.0 * NZ)
    phi[1:NR, 1:NZ] = phi_interior
    return phi


def compute_E(phi, R_arr, Z_arr, dR, dZ, NR, NZ):
    """Central-difference E = -grad(phi) on the 2-D grid."""
    ER = np.zeros((NR+1, NZ+1))
    EZ = np.zeros((NR+1, NZ+1))
    for i in range(1, NR):
        for j in range(1, NZ):
            ER[i, j] = -(phi[i+1, j] - phi[i-1, j]) / (2.0 * dR)
            EZ[i, j] = -(phi[i, j+1] - phi[i, j-1]) / (2.0 * dZ)
    return ER, EZ

# ═══════════════════════════════════════════════════════════════════════════════
# FIELD INTERPOLATION + BORIS PUSH  (Numba-accelerated)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def boris_push(x, y, z, vx, vy, vz,
               ER_grid, EZ_grid,
               R_arr, Z_arr, dR, dZ, NR, NZ,
               B0, R0, Bz_vert,
               qm_dt2, dt, alive):
    """
    Boris push for all particles.
    qm_dt2 = q*dt/(2*m).
    alive[p] = 1 if particle is active, 0 if absorbed.
    """
    Rmin = R_arr[0]
    Zmin = Z_arr[0]
    Rmax = R_arr[NR]
    Zmax = Z_arr[NZ]
    N = x.shape[0]
    for p in prange(N):
        if alive[p] == 0:
            continue
        # ── Particle R, Z ──
        Rp = np.sqrt(x[p]*x[p] + y[p]*y[p])
        Zp = z[p]
        if Rp < 1e-12:
            Rp = 1e-12

        # ── Interpolate E from grid ──
        fi = (Rp - Rmin) / dR
        fj = (Zp - Zmin) / dZ
        i0 = int(fi);  j0 = int(fj)
        if i0 < 0: i0 = 0
        if i0 >= NR: i0 = NR - 1
        if j0 < 0: j0 = 0
        if j0 >= NZ: j0 = NZ - 1
        wr = fi - i0;  wz = fj - j0
        if wr < 0: wr = 0.0
        if wr > 1: wr = 1.0
        if wz < 0: wz = 0.0
        if wz > 1: wz = 1.0
        w00=(1-wr)*(1-wz); w10=wr*(1-wz); w01=(1-wr)*wz; w11=wr*wz

        E_R = (ER_grid[i0,j0]*w00 + ER_grid[i0+1,j0]*w10
             + ER_grid[i0,j0+1]*w01 + ER_grid[i0+1,j0+1]*w11)
        E_Z = (EZ_grid[i0,j0]*w00 + EZ_grid[i0+1,j0]*w10
             + EZ_grid[i0,j0+1]*w01 + EZ_grid[i0+1,j0+1]*w11)

        # Convert E_R to Cartesian
        cos_p = x[p] / Rp
        sin_p = y[p] / Rp
        Ex = E_R * cos_p
        Ey = E_R * sin_p
        Ez = E_Z

        # ── Magnetic field at particle position ──
        B_tor = B0 * R0 / Rp
        Bx = -B_tor * sin_p
        By =  B_tor * cos_p
        Bz =  Bz_vert

        # ── Boris algorithm ──
        # Half-kick by E
        vmx = vx[p] + qm_dt2 * Ex
        vmy = vy[p] + qm_dt2 * Ey
        vmz = vz[p] + qm_dt2 * Ez

        # Rotation by B
        tx = qm_dt2 * Bx
        ty = qm_dt2 * By
        tz = qm_dt2 * Bz
        t_mag2 = tx*tx + ty*ty + tz*tz
        sx = 2.0*tx / (1.0 + t_mag2)
        sy = 2.0*ty / (1.0 + t_mag2)
        sz = 2.0*tz / (1.0 + t_mag2)

        # v' = v- + v- x t
        vpx = vmx + (vmy*tz - vmz*ty)
        vpy = vmy + (vmz*tx - vmx*tz)
        vpz = vmz + (vmx*ty - vmy*tx)

        # v+ = v- + v' x s
        vpx2 = vmx + (vpy*sz - vpz*sy)
        vpy2 = vmy + (vpz*sx - vpx*sz)
        vpz2 = vmz + (vpx*sy - vpy*sx)

        # Second half-kick by E
        vx[p] = vpx2 + qm_dt2 * Ex
        vy[p] = vpy2 + qm_dt2 * Ey
        vz[p] = vpz2 + qm_dt2 * Ez

        # ── Position advance ──
        x[p] += dt * vx[p]
        y[p] += dt * vy[p]
        z[p] += dt * vz[p]

        # ── Wall check ──
        Rp_new = np.sqrt(x[p]*x[p] + y[p]*y[p])
        if Rp_new < Rmin or Rp_new > Rmax or z[p] < Zmin or z[p] > Zmax:
            alive[p] = 0

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def kinetic_energy(vx, vy, vz, alive):
    mask = alive.astype(bool)
    return 0.5 * M_XE * np.sum(vx[mask]**2 + vy[mask]**2 + vz[mask]**2)


def total_current_z(vz, alive):
    return Q_XE * np.sum(vz[alive.astype(bool)])


def density_map(x, y, z, alive, R_arr, Z_arr, dR, dZ, NR, NZ):
    """Return particle-count density on the (R,Z) grid."""
    mask = alive.astype(bool)
    Rp = np.sqrt(x[mask]**2 + y[mask]**2)
    Zp = z[mask]
    H, _, _ = np.histogram2d(
        Rp, Zp, bins=[NR, NZ],
        range=[[R_arr[0], R_arr[-1]], [Z_arr[0], Z_arr[-1]]])
    return H

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_density(H, R_arr, Z_arr, step, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ext = [R_arr[0], R_arr[-1], Z_arr[0], Z_arr[-1]]
    im = ax.imshow(H.T, origin="lower", aspect="auto", extent=ext,
                   cmap="inferno", interpolation="bilinear")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"Xe⁺ density  (step {step})")
    plt.colorbar(im, ax=ax, label="counts / cell")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"density_{step:06d}.png"), dpi=150)
    plt.close(fig)


def plot_energy(times, energies, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.array(times)*1e6, np.array(energies)*1e3, "b-")
    ax.set_xlabel("Time [µs]"); ax.set_ylabel("Kinetic energy [mJ]")
    ax.set_title("Total kinetic energy vs time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "energy_vs_time.png"), dpi=150)
    plt.close(fig)


def plot_fft_current(times, currents, out_dir):
    dt_diag = times[1] - times[0] if len(times) > 1 else 1e-6
    I = np.array(currents)
    I -= I.mean()
    spec = np.abs(np.fft.rfft(I))
    freq = np.fft.rfftfreq(len(I), d=dt_diag)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(freq * 1e-3, spec, "r-")
    ax.set_xlabel("Frequency [kHz]"); ax.set_ylabel("|FFT(I_z)|")
    ax.set_title("Cyclotron resonance (FFT of total Z-current)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cyclotron_fft.png"), dpi=150)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg=None):
    if cfg is None:
        cfg = CFG
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # Grid
    R_arr, Z_arr, dR, dZ = build_grid(cfg)
    NR, NZ = cfg["NR"], cfg["NZ"]
    a_coeff, c_coeff, diag0 = _build_tridiag_coeffs(R_arr, dR, NR)

    # Particles
    print(f"Initialising {cfg['N_particles']:,} Xe⁺ ions …")
    x, y, z, vx, vy, vz = init_particles(cfg)
    alive = np.ones(cfg["N_particles"], dtype=np.int32)

    # Precompute Boris constant
    qm_dt2 = Q_XE * cfg["dt"] / (2.0 * M_XE)

    # Time-stepping bookkeeping
    dt = cfg["dt"]
    n_steps = int(np.ceil(cfg["t_end"] / dt))
    diag_interval = max(1, int(round(cfg["diag_dt"] / dt)))

    times_diag, energies, currents = [], [], []
    n_alive_hist = []

    # ── Warm up Numba (first call compiles) ──
    print("Compiling Numba kernels (first step) …")

    wall0 = time.perf_counter()

    for step in range(n_steps + 1):
        t_now = step * dt

        # ── Diagnostics ──
        if step % diag_interval == 0:
            KE = kinetic_energy(vx, vy, vz, alive)
            Iz = total_current_z(vz, alive)
            na = int(alive.sum())
            times_diag.append(t_now)
            energies.append(KE)
            currents.append(Iz)
            n_alive_hist.append(na)
            H = density_map(x, y, z, alive, R_arr, Z_arr, dR, dZ, NR, NZ)
            plot_density(H, R_arr, Z_arr, step, cfg["out_dir"])
            elapsed = time.perf_counter() - wall0
            print(f"  step {step:6d}/{n_steps}  t={t_now*1e6:7.2f} µs  "
                  f"alive={na:7,}  KE={KE:.4e} J  wall={elapsed:.1f}s")

        if step == n_steps:
            break

        # ── Charge deposition ──
        rho = deposit_charge(x, y, z, cfg, R_arr, Z_arr, dR, dZ)

        # ── Poisson solve ──
        phi = solve_poisson(rho, R_arr, Z_arr, dR, dZ, NR, NZ,
                            a_coeff, c_coeff, diag0)

        # ── Electric field ──
        ER, EZ = compute_E(phi, R_arr, Z_arr, dR, dZ, NR, NZ)

        # ── Boris push ──
        boris_push(x, y, z, vx, vy, vz,
                   ER, EZ, R_arr, Z_arr, dR, dZ, NR, NZ,
                   cfg["B0"], cfg["R0"], cfg["Bz_vert"],
                   qm_dt2, dt, alive)

    wall_total = time.perf_counter() - wall0
    print(f"\nSimulation complete in {wall_total:.1f} s  "
          f"({n_steps} steps, {alive.sum()} particles surviving).\n")

    # ── Save particle data ──
    npz_path = os.path.join(cfg["out_dir"], "particles_final.npz")
    np.savez_compressed(npz_path,
                        x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, alive=alive)
    print(f"Saved particle data → {npz_path}")

    # ── Summary plots ──
    plot_energy(times_diag, energies, cfg["out_dir"])
    plot_fft_current(times_diag, currents, cfg["out_dir"])
    # Final density map (already saved above, but save a labelled copy)
    H = density_map(x, y, z, alive, R_arr, Z_arr, dR, dZ, NR, NZ)
    plot_density(H, R_arr, Z_arr, n_steps, cfg["out_dir"])
    print(f"Plots saved to {cfg['out_dir']}/")

    return dict(times=times_diag, energies=energies, currents=currents,
                n_alive=n_alive_hist)


if __name__ == "__main__":
    run()
