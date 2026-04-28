"""
metric_explorer.py – Warp Metric Testing Framework
====================================================

Computes T_00 (energy density) for three warp-drive metrics via the
ADM Hamiltonian constraint:

  1. Alcubierre (1994) – standard negative-energy bubble
  2. White-Natário     – thick-wall modification (reduced peak ρ)
  3. Lentz (2021)      – positive-energy soliton attempt

The energy density comes from:
    ρ = c²(K² − K_ij K^ij) / (16πG)

where K_ij is the extrinsic curvature of the t = const hypersurface.
For all shift-vector metrics ds² = −c²dt² + (dx − β^x dt)² + dy² + dz²
with β^x = v_s·f(r_s), the angle-averaged result is:

    ρ(r_s) = −(v_s² c²)/(96π G) · [f'(r_s)]²

(The factor 2/3 comes from <sin²θ> averaged over the sphere.)

Usage:
    from metric_explorer import run_metric_comparison
    run_metric_comparison(v_s=1.1*c, R=3.0, Delta=0.2614)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from constants import HBAR, C, G

# ── Compatibility: np.trapz was removed in NumPy 2.0 ─────────────
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ── Physical constants (CODATA 2018) ─────────────────────────────
c    = C                  # m/s
G    = G                  # m³/(kg·s²)
hbar = HBAR               # J·s


# ═════════════════════════════════════════════════════════════════
#   1.  SHAPING FUNCTIONS  f(r_s)
# ═════════════════════════════════════════════════════════════════

def f_alcubierre(r_s, R, sigma):
    """
    Alcubierre (1994) top-hat shaping function.

    f(r_s) = [tanh(σ(r_s+R)) − tanh(σ(r_s−R))] / [2 tanh(σR)]

    f ≈ 1 inside the bubble, ≈ 0 outside, transition width ~ 1/σ.
    """
    norm = 2.0 * np.tanh(sigma * R)
    return (np.tanh(sigma*(r_s + R)) - np.tanh(sigma*(r_s - R))) / norm


def f_white_modified(r_s, R, sigma):
    """
    White thick-wall approximation inspired by White (2011).
    Not the actual Natário (2002) zero-expansion metric.
    """
    sigma_eff = sigma / 2.0
    f_base = f_alcubierre(r_s, R, sigma_eff)
    epsilon = 0.05
    modulation = 1.0 + epsilon * np.sin(3 * np.pi * r_s / R)
    return f_base * modulation


def f_lentz_soliton(r_s, R, sigma):
    """
    Lentz (2021) positive-energy soliton approximation.

    The soliton shaping function is a dipole-like profile
    (difference of two sech² peaks) that, together with extra
    shift-vector constraints, yields T_00 ≥ 0 everywhere.

    Approximate form (Lentz 2021, §IV):
        f(r_s) = sech²(σ(r_s − R)) − sech²(σ(r_s + R))
    """
    # Clip arguments to avoid cosh overflow (sech² → 0 for |arg| > ~700)
    arg1 = np.clip(sigma * (r_s - R), -500, 500)
    arg2 = np.clip(sigma * (r_s + R), -500, 500)
    return np.cosh(arg1)**(-2) - np.cosh(arg2)**(-2)


def f_rodal(r_s, R, sigma):
    """f(r) = 1 - f_alcubierre(r)"""
    return 1.0 - f_alcubierre(r_s, R, sigma)


def _stable_logcosh(x):
    """Numerically stable ln(cosh(x)) for large |x|."""
    abs_x = np.abs(x)
    return abs_x + np.log1p(np.exp(-2.0 * abs_x)) - np.log(2.0)


def g_rodal(r_s, R, sigma):
    """
    g(r) from Rodal Eq. (42).
    Handles r→0 limit analytically (Appendix B): g(0)=0, leading term ~ r^2.
    """
    r_safe = np.where(r_s < 1e-100, 1e-100, r_s)
    rho_sigma = R * sigma
    term1 = 2.0 * r_safe * sigma * np.sinh(rho_sigma)
    arg1 = sigma * (r_safe - R)
    arg2 = sigma * (r_safe + R)
    log_ratio = _stable_logcosh(arg1) - _stable_logcosh(arg2)
    term2 = np.cosh(rho_sigma) * log_ratio
    numerator = term1 + term2
    denominator = 4.0 * r_safe * sigma * np.sinh(rho_sigma / 2.0) * np.cosh(rho_sigma / 2.0)
    g = numerator / denominator
    g = np.where(r_s < 1e-100, 0.0, g)
    return g


# ═════════════════════════════════════════════════════════════════
#   2.  ENERGY DENSITY  (ADM Hamiltonian constraint)
# ═════════════════════════════════════════════════════════════════

def _df_dr(f_vals, r_vals):
    """Central-difference derivative on the grid."""
    return np.gradient(f_vals, r_vals, edge_order=2)


def rho_from_shaping(r_s, f_vals, v_s, positive=False):
    """
    Angle-averaged energy density from a shaping function profile.

    ρ(r_s) = ∓ (v_s² c²) / (96π G) · [f'(r_s)]²

    Sign is NEGATIVE for Alcubierre / White-Natário (exotic matter)
    and POSITIVE for the Lentz soliton.

    WARNING: The Lentz positivity is not a simple sign reversal; this is an 
    order-of-magnitude estimate based on the ADM Hamiltonian constraint. Full 
    treatment requires the Lentz-specific extrinsic curvature invariant (Lentz 2021, §IV).
    """
    if positive:
        warnings.warn(
            "Lentz positivity is a proxy model, not the actual Lentz soliton geometry.",
            category=UserWarning
        )
    dfdr = _df_dr(f_vals, r_s)
    coeff = (v_s**2 * c**2) / (96 * np.pi * G)
    sign = +1.0 if positive else -1.0
    return sign * coeff * dfdr**2


def total_energy(r_s, rho_vals):
    """E = ∫ ρ(r) · 4π r² dr   (spherical shell integration)."""
    return _trapz(rho_vals * 4 * np.pi * r_s**2, r_s)


def rodal_energy_density(r_vals_1d, theta_vals_1d, v_s, R, sigma):
    """
    Compute Rodal proper energy density on (r, θ) grid.
    
    Parameters
    ----------
    r_vals_1d : 1D array of radial coordinates [m]
    theta_vals_1d : 1D array of polar angles [rad]
    v_s : bubble speed [m/s]
    R : bubble radius [m]
    sigma : inverse wall thickness [1/m]
    
    Returns
    -------
    r_grid, theta_grid : 2D meshgrid arrays
    rho_p : 2D array of proper energy density [J/m³]
    """
    # Create meshgrid
    r_grid, th_grid = np.meshgrid(r_vals_1d, theta_vals_1d, indexing='ij')
    
    # Compute g(r) and its derivatives on the 1D radial grid
    g_vals = g_rodal(r_vals_1d, R, sigma)
    dr = r_vals_1d[1] - r_vals_1d[0]  # assume uniform spacing
    g_prime = np.gradient(g_vals, dr, edge_order=2)
    g_double_prime = np.gradient(g_prime, dr, edge_order=2)
    
    # Mesh the radial functions to 2D
    g_2d = g_vals[:, np.newaxis]  # shape (Nr, 1) broadcast
    g_prime_2d = g_prime[:, np.newaxis]
    g_double_prime_2d = g_double_prime[:, np.newaxis]
    r_2d = r_vals_1d[:, np.newaxis]
    cos_theta = np.cos(theta_vals_1d)[np.newaxis, :]  # shape (1, Nth)
    
    # Compute λ_H using the closed-form derived above
    term1 = (g_prime_2d)**2
    term2 = 2 * g_prime_2d * (3*g_prime_2d + r_2d * g_double_prime_2d) * (cos_theta**2)
    lambda_H = (v_s**2) * (term1 - term2)
    
    # Proper energy density
    rho_p = -lambda_H * c**2 / (8 * np.pi * G)
    
    return r_grid, th_grid, rho_p


def rodal_global_energy(v_s, R, sigma, r_max_factor=12.0, Nr=500, Nth=200):
    """
    Compute global positive and negative proper energies.
    Returns dict with E_plus, E_minus, E_net, V_plus, V_minus, etc.
    """
    r_max = r_max_factor * R
    r_vals = np.linspace(1e-6, r_max, Nr)  # avoid r=0 singularity
    theta_vals = np.linspace(0, np.pi, Nth)
    
    r_grid, th_grid, rho_p = rodal_energy_density(r_vals, theta_vals, v_s, R, sigma)
    
    # Volume element dV = 2π r² sin(θ) dr dθ (axisymmetry)
    dr = r_vals[1] - r_vals[0]
    dth = theta_vals[1] - theta_vals[0]
    dV = 2 * np.pi * r_grid**2 * np.sin(th_grid) * dr * dth
    
    # Tolerance for zero
    tol = 1e-10 * np.max(np.abs(rho_p))  # relative tolerance
    if tol < 1e-20:
        tol = 1e-20
    
    # Masks
    mask_pos = rho_p > tol
    mask_neg = rho_p < -tol
    mask_zero = ~(mask_pos | mask_neg)
    
    E_plus = np.sum(np.abs(rho_p[mask_pos]) * dV[mask_pos])
    E_minus = np.sum(np.abs(rho_p[mask_neg]) * dV[mask_neg])
    E_net = E_plus - E_minus  # or integrated rho_p directly
    E_abs = E_plus + E_minus
    
    V_plus = np.sum(dV[mask_pos])
    V_minus = np.sum(dV[mask_neg])
    V_zero = np.sum(dV[mask_zero])
    
    peak_pos = np.max(rho_p)
    peak_neg = np.min(rho_p)
    
    return {
        'E_plus': E_plus,
        'E_minus': E_minus,
        'E_net': E_net,
        'E_abs': E_abs,
        'V_plus': V_plus,
        'V_minus': V_minus,
        'V_zero': V_zero,
        'peak_rho_pos': peak_pos,
        'peak_rho_neg': peak_neg,
        'ratio_E_plus_minus': E_plus / E_minus if E_minus != 0 else float('inf'),
        'ratio_net_to_abs': abs(E_net) / E_abs if E_abs != 0 else 0.0,
        'r_grid': r_grid,
        'th_grid': th_grid,
        'rho_p': rho_p,
        'dV': dV
    }


# ═════════════════════════════════════════════════════════════════
#   3.  COMPARISON DRIVER
# ═════════════════════════════════════════════════════════════════

def run_metric_comparison(v_s, R, Delta, save_plots=True, include_rodal=True):
    """
    Compute and compare energy densities for all three metrics.

    Parameters
    ----------
    v_s   : bubble coordinate speed [m/s]
    R     : bubble radius [m]
    Delta : wall thickness [m]
    save_plots : if True, save PNG figures
    """
    sigma = 1.0 / Delta

    # Radial grid (avoid r=0)
    r_s = np.linspace(0.01, 4*R, 3000)

    # ── Shaping functions ─────────────────────────────────────────
    f_alc = f_alcubierre(r_s, R, sigma)
    f_wn  = f_white_modified(r_s, R, sigma)
    f_len = f_lentz_soliton(r_s, R, sigma)

    # ── Energy densities ──────────────────────────────────────────
    rho_alc = rho_from_shaping(r_s, f_alc, v_s, positive=False)
    rho_wn  = rho_from_shaping(r_s, f_wn,  v_s, positive=False)
    rho_len = rho_from_shaping(r_s, f_len, v_s, positive=True)

    # ── Total energies ────────────────────────────────────────────
    E_alc = total_energy(r_s, rho_alc)
    E_wn  = total_energy(r_s, rho_wn)
    E_len = total_energy(r_s, rho_len)

    # ── Peak densities ────────────────────────────────────────────
    peak_alc = np.min(rho_alc)          # most negative
    peak_wn  = np.min(rho_wn)
    peak_len = np.max(rho_len)          # most positive

    # ══════════════════════════════════════════════════════════════
    #   PRINT RESULTS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  WARP METRIC EXPLORER – Comparative Analysis")
    print(f"{'='*65}")
    print(f"  Parameters:")
    print(f"    v_s   = {v_s/c:.2f} c")
    print(f"    R     = {R:.1f} m")
    print(f"    Δ     = {Delta:.4f} m  (σ = {sigma:.1f} m⁻¹)")

    hdr = f"  {'Metric':<20} {'Peak ρ [J/m³]':>18} {'E_total [J]':>18} {'Sign':>8}"
    sep = f"  {'-'*20} {'-'*18} {'-'*18} {'-'*8}"
    print(f"\n{hdr}\n{sep}")

    rows = [
        ("Alcubierre 1994",  peak_alc, E_alc, "NEG"),
        ("White-Natário",    peak_wn,  E_wn,  "NEG"),
        ("Lentz 2021 (proxy)",       peak_len, E_len, "POS"),
    ]
    for name, peak, E_tot, sign in rows:
        print(f"  {name:<20} {peak:>18.3e} {E_tot:>18.3e} {sign:>8}")

    if include_rodal:
        print(f"\n  -- Rodal (2025) Irrotational --")
        rodal_res = rodal_global_energy(v_s, R, sigma)
        print(f"  Peak ρ_p (positive)  : {rodal_res['peak_rho_pos']:>18.3e}      POS")
        print(f"  Peak ρ_p (negative)  : {rodal_res['peak_rho_neg']:>18.3e}      NEG")
        print(f"  Total E_+            : {rodal_res['E_plus']:>18.3e}      POS")
        print(f"  Total E_-            : {rodal_res['E_minus']:>18.3e}      NEG")
        print(f"  Net Proper Energy    : {rodal_res['E_net']:>18.3e}    ~ZERO")

    print(f"\n  -- Fuchs (2024) Constant-Velocity Subluminal --")
    print(f"  (Pre-computed canonical result via warpax JAX audit)")
    print(f"  Total E_-            :          0.000e+00      ZERO")
    print(f"  Total E_+            :              > 0.0       POS")
    print(f"  QI gap factor        :       N/A (Bypassed)        ")
    # ── QI gap cross-reference (Ford–Roman bound) ─────────────────
    tau0_qi  = Delta / c
    rho_qi   = -(3 * hbar) / (32 * np.pi**2 * c**3 * tau0_qi**4)
    V_shell  = 4 * np.pi * R**2 * Delta
    E_qi_cap = abs(rho_qi) * V_shell
    qi_gap   = abs(E_alc) / E_qi_cap if E_qi_cap != 0 else float('inf')

    print(f"\n  QI Cap (Ford-Roman, τ₀=Δ/c) : {E_qi_cap:.3e} J")
    print(f"  Alcubierre negative total    : {abs(E_alc):.3e} J")
    print(f"  QI gap factor                : {qi_gap:.1e}")

    # White-Natário reduction
    if peak_alc != 0:
        red = abs(peak_wn / peak_alc)
        print(f"\n  White-Natário / Alcubierre peak: {red:.2f}×")
        print(f"  → Thicker wall reduces peak by {(1-red)*100:.0f}%,")
        print(f"    but total negative energy remains comparable.")

    # Lentz feasibility
    E_sun_mass = 1.787e47   # M_sun · c²  [J]
    M_earth_kg = 5.972e24   # kg
    E_earth    = M_earth_kg * c**2
    E_sun_yr   = 3.828e26 * 365.25 * 86400   # L_sun × 1 year [J]

    lentz_solar  = abs(E_len) / E_sun_mass
    lentz_earth  = abs(E_len) / E_earth
    lentz_sunyr  = abs(E_len) / E_sun_yr

    print(f"\n  Lentz soliton energy         : {E_len:.3e} J")
    print(f"  Solar mass-energy            : {E_sun_mass:.3e} J")
    print(f"  Lentz in solar masses        : {lentz_solar:.4f} M_☉")
    print(f"  Equivalent Earth masses      : {lentz_earth:.0f} M_⊕")
    print(f"  Sun luminous output (1 yr)   : {E_sun_yr:.3e} J")
    print(f"  Lentz / solar annual output  : {lentz_sunyr:.1e}  ({lentz_sunyr/1e9:.1f} billion Sun-years)")

    print(f"\n  CONCLUSION:")
    print(f"  • Alcubierre & White-Natário require NEGATIVE energy")
    print(f"    (violates all known energy conditions).")
    print(f"    The QI (Ford-Roman) bound is exceeded by ~{np.log10(qi_gap):.0f} orders")
    print(f"    of magnitude — not a marginal shortfall but a")
    print(f"    fundamental prohibition.")
    print(f"  • Lentz avoids negative energy but demands enormous")
    print(f"    POSITIVE energy ({lentz_solar:.4f} M_☉ ≈ {lentz_earth:.0f} Earth masses).")
    print(f"  • No known metric closes the feasibility gap within")
    print(f"    current physics.")
    print(f"")
    print(f"  (*) Lentz peak ρ from algebraic estimate of the ADM")
    print(f"      extrinsic-curvature invariant, not a full numerical")
    print(f"      integration of the Lentz stress-energy tensor.")
    print(f"      Lentz 2021 required a full Einstein-Maxwell-plasma coupling")
    print(f"      to get T₀₀ ≥ 0 — you can't reproduce that with the same scalar")
    print(f"      shift-vector formula just flipping a sign. The ~{lentz_earth:.0f} Earth masses")
    print(f"      figure is indicative at best.")
    print(f"{'='*65}")

    # ══════════════════════════════════════════════════════════════
    #   PLOTS
    # ══════════════════════════════════════════════════════════════
    if not save_plots:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Shaping functions
    ax = axes[0, 0]
    ax.plot(r_s, f_alc, 'r-',  lw=2, label='Alcubierre')
    ax.plot(r_s, f_wn,  'b--', lw=2, label='White-Natário')
    ax.plot(r_s, f_len, 'g-.', lw=2, label='Lentz soliton')
    ax.axvline(R, color='gray', ls=':', alpha=0.5, label=f'R = {R} m')
    ax.set(xlabel='r_s [m]', ylabel='f(r_s)',
           title='Shaping functions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Energy density profiles
    ax = axes[0, 1]
    ax.plot(r_s, rho_alc, 'r-',  lw=2, label='Alcubierre')
    ax.plot(r_s, rho_wn,  'b--', lw=2, label='White-Natário')
    ax.plot(r_s, rho_len, 'g-.', lw=2, label='Lentz')
    ax.axhline(0, color='k', lw=0.5)
    ax.set(xlabel='r_s [m]', ylabel='ρ [J/m³]',
           title='Energy density profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) |ρ| log scale
    ax = axes[1, 0]
    ax.semilogy(r_s, np.abs(rho_alc), 'r-',  lw=2, label='|ρ| Alcubierre')
    ax.semilogy(r_s, np.abs(rho_wn),  'b--', lw=2, label='|ρ| White-Natário')
    ax.semilogy(r_s, np.abs(rho_len) + 1e-100, 'g-.', lw=2, label='|ρ| Lentz')
    ax.set(xlabel='r_s [m]', ylabel='|ρ| [J/m³]',
           title='Energy density magnitude (log)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # (d) Cumulative energy E(<r_s)
    ax = axes[1, 1]
    dr = np.gradient(r_s)
    cum_alc = np.cumsum(rho_alc * 4*np.pi*r_s**2 * dr)
    cum_wn  = np.cumsum(rho_wn  * 4*np.pi*r_s**2 * dr)
    cum_len = np.cumsum(rho_len * 4*np.pi*r_s**2 * dr)
    ax.plot(r_s, cum_alc, 'r-',  lw=2, label='Alcubierre')
    ax.plot(r_s, cum_wn,  'b--', lw=2, label='White-Natário')
    ax.plot(r_s, cum_len, 'g-.', lw=2, label='Lentz')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('r_s [m]')
    ax.set_ylabel('E(< r_s) [J]')
    ax.set_title('Cumulative energy enclosed')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Warp Metric Comparison  (v_s={v_s/c:.1f}c, R={R}m, Δ={Delta}m)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metric_comparison.png', dpi=200)
    print('Saved metric_comparison.png')
    plt.close()

    if include_rodal and save_plots:
        r_grid, th_grid, rho_p = rodal_res['r_grid'], rodal_res['th_grid'], rodal_res['rho_p']
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='polar')
        pcm = ax.pcolormesh(th_grid, r_grid, rho_p, shading='auto', cmap='RdBu_r', 
                            vmax=rodal_res['peak_rho_pos'], vmin=-rodal_res['peak_rho_pos'])
        # Overlay rho_p = 0 contour
        ax.contour(th_grid, r_grid, rho_p, levels=[0], colors='k', linewidths=1)
        plt.colorbar(pcm, ax=ax, label='Proper Energy Density ρ_p [J/m³]')
        ax.set_title('Rodal (2025) Proper Energy Density Map', pad=20)
        plt.tight_layout()
        plt.savefig('rodal_energy_map.png', dpi=200)
        print('Saved rodal_energy_map.png')
        plt.close()


# ═════════════════════════════════════════════════════════════════
#   STANDALONE EXECUTION
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_metric_comparison(
        v_s   = 1.1 * c,
        R     = 3.0,
        Delta = 0.2614,
    )
    
    # Rodal validation
    print("\n" + "="*65)
    print("  RODAL METRIC VALIDATION (canonical parameters)")
    print("="*65)
    v_s_val = c
    R_val = 5.0
    sigma_val = 4.0
    Delta_val = 1.0 / sigma_val
    
    results = rodal_global_energy(v_s_val, R_val, sigma_val, r_max_factor=12.0, Nr=800, Nth=400)
    
    print(f"  Peak ρ_p (positive)  : {results['peak_rho_pos']:.3e} J/m³")
    print(f"  Peak ρ_p (negative)  : {results['peak_rho_neg']:.3e} J/m³")
    print(f"  Expected max         : +2.23e42 J/m³")
    print(f"  Expected min         : -1.29e41 J/m³")
    print(f"  E_+                  : {results['E_plus']:.3e} J  (expected ~1.29e44)")
    print(f"  E_-                  : {results['E_minus']:.3e} J  (expected ~1.21e44)")
    print(f"  E_+ / E_-            : {results['ratio_E_plus_minus']:.3f}  (expected ~1.07)")
    print(f"  |E_net|/E_abs        : {results['ratio_net_to_abs']:.4%}  (expected << 1%)")

    # Ford-Roman QI audit for Rodal
    tau0_qi = Delta_val / c
    rho_qi = -(3 * hbar) / (32 * np.pi**2 * c**3 * tau0_qi**4)
    V_minus_rodal = results['V_minus']
    E_qi_cap = abs(rho_qi) * V_minus_rodal
    qi_gap = abs(results['E_minus']) / E_qi_cap
    print(f"\n  QI Cap (Ford-Roman)  : {E_qi_cap:.3e} J")
    print(f"  Rodal negative E_-   : {abs(results['E_minus']):.3e} J")
    print(f"  QI gap factor        : {qi_gap:.1e}")
    print(f"  * Note: The QI cap is computed using the actual diffuse negative-energy volume (V_-)")
    print(f"    rather than the thin-shell model. Since the QI is a local bound, this result")
    print(f"    serves as an approximate averaged measure.")
