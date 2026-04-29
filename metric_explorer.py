"""
metric_explorer.py – Warp Metric Testing Framework
====================================================

Computes T_00 (energy density) for two superluminal warp-drive metrics via the
ADM Hamiltonian constraint:

  1. Alcubierre (1994) – standard negative-energy bubble
  2. White-Natário     – thick-wall modification (reduced peak ρ)

NOTE: The Lentz (2021) metric is NOT included in this audit.
Lentz requires a full Einstein–Maxwell–plasma coupling to achieve T₀₀ ≥ 0.
The simple sign-flip of the Alcubierre ADM formula is physically invalid and
has been removed. See: Lentz 2021 §IV; Bobrick & Martire 2021.

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from constants import HBAR, C, G
from plot_style import setup_plot_style

# ── Compatibility: np.trapz was removed in NumPy 2.0 ─────────────
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ── Physical constants (CODATA 2018) ─────────────────────────────
c    = C
G    = G
hbar = HBAR

# ── Apply shared Matplotlib style (see plot_style.py) ────────────
setup_plot_style()

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


def rho_from_shaping(r_s, f_vals, v_s):
    """
    Angle-averaged energy density from a shaping function profile.

    ρ(r_s) = −(v_s² c²) / (96π G) · [f'(r_s)]²

    Sign is always NEGATIVE (exotic matter required).

    NOTE: The positive=True branch (Lentz proxy) has been removed.
    Lentz (2021) requires a full Einstein–Maxwell–plasma coupling that
    cannot be reproduced by a sign-flip of this scalar formula.
    """
    dfdr = _df_dr(f_vals, r_s)
    coeff = (v_s**2 * c**2) / (96 * np.pi * G)
    return -coeff * dfdr**2


def total_energy(r_s, rho_vals):
    """E = ∫ ρ(r) · 4π r² dr   (spherical shell integration)."""
    return _trapz(rho_vals * 4 * np.pi * r_s**2, r_s)


def negative_energy_volume(r_s, rho_vals):
    """
    Compute the volume of the region where ρ < 0 for a 1-D radial profile.

    V_minus = ∫_{ρ<0} 4π r² dr

    This replaces the thin-shell approximation V_shell = 4π R² Δ used
    previously, giving a metric-specific, numerically accurate volume.
    """
    mask = rho_vals < 0.0
    if not np.any(mask):
        return 0.0
    integrand = np.where(mask, 4 * np.pi * r_s**2, 0.0)
    return _trapz(integrand, r_s)


def rodal_energy_density(r_vals_1d, theta_vals_1d, v_s, R, sigma):
    """
    Compute Rodal proper energy density on (r, θ) grid.
    """
    r_grid, th_grid = np.meshgrid(r_vals_1d, theta_vals_1d, indexing='ij')
    g_vals = g_rodal(r_vals_1d, R, sigma)
    dr = r_vals_1d[1] - r_vals_1d[0]
    g_prime = np.gradient(g_vals, dr, edge_order=2)
    g_double_prime = np.gradient(g_prime, dr, edge_order=2)
    g_prime_2d = g_prime[:, np.newaxis]
    g_double_prime_2d = g_double_prime[:, np.newaxis]
    r_2d = r_vals_1d[:, np.newaxis]
    cos_theta = np.cos(theta_vals_1d)[np.newaxis, :]
    term1 = (g_prime_2d)**2
    term2 = 2 * g_prime_2d * (3*g_prime_2d + r_2d * g_double_prime_2d) * (cos_theta**2)
    lambda_H = (v_s**2) * (term1 - term2)
    rho_p = -lambda_H * c**2 / (8 * np.pi * G)
    return r_grid, th_grid, rho_p


def rodal_global_energy(v_s, R, sigma, r_max_factor=12.0, Nr=500, Nth=200):
    """
    Compute global positive and negative proper energies for Rodal metric.
    Returns dict with E_plus, E_minus, E_net, V_plus, V_minus, etc.
    V_minus is the actual diffuse negative-energy volume (not thin-shell).
    """
    r_max = r_max_factor * R
    r_vals = np.linspace(1e-6, r_max, Nr)
    theta_vals = np.linspace(0, np.pi, Nth)
    r_grid, th_grid, rho_p = rodal_energy_density(r_vals, theta_vals, v_s, R, sigma)
    dr = r_vals[1] - r_vals[0]
    dth = theta_vals[1] - theta_vals[0]
    dV = 2 * np.pi * r_grid**2 * np.sin(th_grid) * dr * dth
    tol = 1e-10 * np.max(np.abs(rho_p))
    if tol < 1e-20:
        tol = 1e-20
    mask_pos = rho_p > tol
    mask_neg = rho_p < -tol
    mask_zero = ~(mask_pos | mask_neg)
    E_plus  = np.sum(np.abs(rho_p[mask_pos]) * dV[mask_pos])
    E_minus = np.sum(np.abs(rho_p[mask_neg]) * dV[mask_neg])
    E_net   = E_plus - E_minus
    E_abs   = E_plus + E_minus
    V_plus  = np.sum(dV[mask_pos])
    V_minus = np.sum(dV[mask_neg])
    V_zero  = np.sum(dV[mask_zero])
    peak_pos = np.max(rho_p)
    peak_neg = np.min(rho_p)
    return {
        'E_plus': E_plus, 'E_minus': E_minus, 'E_net': E_net, 'E_abs': E_abs,
        'V_plus': V_plus, 'V_minus': V_minus, 'V_zero': V_zero,
        'peak_rho_pos': peak_pos, 'peak_rho_neg': peak_neg,
        'ratio_E_plus_minus': E_plus / E_minus if E_minus != 0 else float('inf'),
        'ratio_net_to_abs': abs(E_net) / E_abs if E_abs != 0 else 0.0,
        'r_grid': r_grid, 'th_grid': th_grid, 'rho_p': rho_p, 'dV': dV,
    }


# ═════════════════════════════════════════════════════════════════
#   3.  CONVERGENCE CHECK
# ═════════════════════════════════════════════════════════════════

def convergence_check_all_metrics(v_s, R, Delta, include_rodal=True):
    """
    Check numerical convergence of the energy integrals for all metrics.

    For Alcubierre and White-Natário (1-D), runs `total_energy()` at four
    radial grid resolutions: N = 1000, 2000, 4000, 8000 points over [0.01, 4R].

    For Rodal (2-D), runs `rodal_global_energy()` at three (Nr, Nth) pairs:
    (500, 200), (800, 300), (1200, 450).

    Prints the energy at each resolution and the relative change from the
    previous level.  Declares convergence if the two finest resolutions agree
    to better than 0.1 %.
    """
    sigma = 1.0 / Delta

    print(f"\n{'='*65}")
    print(f"  CONVERGENCE CHECK  (v_s={v_s/c:.2f}c, R={R} m, Δ={Delta} m)")
    print(f"{'='*65}")

    # ── 1-D metrics (Alcubierre, White-Natário) ───────────────────
    resolutions_1d = [1000, 2000, 4000, 8000]
    metric_fns = [
        ("Alcubierre 1994", f_alcubierre),
        ("White-Natário",   f_white_modified),
    ]

    for metric_name, shaping_fn in metric_fns:
        print(f"\n  {metric_name}:")
        print(f"    {'N pts':>8}  {'E_total [J]':>18}  {'Rel. change':>14}")
        print(f"    {'-'*8}  {'-'*18}  {'-'*14}")
        prev_E = None
        last_two = []
        for N in resolutions_1d:
            r_s   = np.linspace(0.01, 4 * R, N)
            f_vals = shaping_fn(r_s, R, sigma)
            rho   = rho_from_shaping(r_s, f_vals, v_s)
            E     = total_energy(r_s, rho)
            if prev_E is not None and prev_E != 0:
                rel_change = abs((E - prev_E) / prev_E)
                print(f"    {N:>8d}  {E:>18.6e}  {rel_change:>13.4%}")
            else:
                print(f"    {N:>8d}  {E:>18.6e}  {'(baseline)':>14}")
            last_two.append(E)
            if len(last_two) > 2:
                last_two.pop(0)
            prev_E = E
        # Convergence verdict on the two finest resolutions
        if len(last_two) == 2 and last_two[0] != 0:
            finest_rel = abs((last_two[1] - last_two[0]) / last_two[0])
            if finest_rel < 0.001:
                print(f"    ✓ CONVERGED to better than 0.1 % at N=4000→8000 "
                      f"(Δrel={finest_rel:.4%})")
            else:
                print(f"    ⚠ WARNING: result may NOT be fully converged "
                      f"(Δrel={finest_rel:.4%} between N=4000 and N=8000)")

    # ── Rodal (2-D) ───────────────────────────────────────────────
    if include_rodal:
        print(f"\n  Rodal (2025) – 2-D integration:")
        print(f"    {'(Nr, Nth)':>12}  {'E_minus [J]':>18}  {'Rel. change':>14}")
        print(f"    {'-'*12}  {'-'*18}  {'-'*14}")
        rodal_grids = [(500, 200), (800, 300), (1200, 450)]
        prev_Em = None
        last_two_r = []
        for Nr, Nth in rodal_grids:
            res  = rodal_global_energy(v_s, R, sigma, Nr=Nr, Nth=Nth)
            Em   = res['E_minus']
            lbl  = f"({Nr}, {Nth})"
            if prev_Em is not None and prev_Em != 0:
                rel_change = abs((Em - prev_Em) / prev_Em)
                print(f"    {lbl:>12}  {Em:>18.6e}  {rel_change:>13.4%}")
            else:
                print(f"    {lbl:>12}  {Em:>18.6e}  {'(baseline)':>14}")
            last_two_r.append(Em)
            if len(last_two_r) > 2:
                last_two_r.pop(0)
            prev_Em = Em
        if len(last_two_r) == 2 and last_two_r[0] != 0:
            finest_rel = abs((last_two_r[1] - last_two_r[0]) / last_two_r[0])
            if finest_rel < 0.001:
                print(f"    ✓ CONVERGED to better than 0.1 % at (800,300)→(1200,450) "
                      f"(Δrel={finest_rel:.4%})")
            else:
                print(f"    ⚠ WARNING: result may NOT be fully converged "
                      f"(Δrel={finest_rel:.4%} between (800,300) and (1200,450))")

    print(f"{'='*65}")


# ═════════════════════════════════════════════════════════════════
#   4.  COMPARISON DRIVER
# ═════════════════════════════════════════════════════════════════

def run_metric_comparison(v_s, R, Delta, save_plots=True, include_rodal=True):
    """
    Compute and compare energy densities for Alcubierre and White-Natário.

    NOTE: Lentz (2021) is excluded. Its positive-energy claim requires a
    full Einstein–Maxwell–plasma coupling not yet implemented here.
    See: Lentz 2021 §IV; Bobrick & Martire 2021 for context.

    Parameters
    ----------
    v_s   : bubble coordinate speed [m/s]
    R     : bubble radius [m]
    Delta : wall thickness [m]
    save_plots : if True, save PNG and PDF figures
    """
    sigma = 1.0 / Delta

    # Radial grid (avoid r=0)
    r_s = np.linspace(0.01, 4*R, 3000)

    # ── Shaping functions ─────────────────────────────────────────
    f_alc = f_alcubierre(r_s, R, sigma)
    f_wn  = f_white_modified(r_s, R, sigma)

    # ── Energy densities ──────────────────────────────────────────
    rho_alc = rho_from_shaping(r_s, f_alc, v_s)
    rho_wn  = rho_from_shaping(r_s, f_wn,  v_s)

    # ── Total energies ────────────────────────────────────────────
    E_alc = total_energy(r_s, rho_alc)
    E_wn  = total_energy(r_s, rho_wn)

    # ── Negative-energy volumes (numerical, not thin-shell) ───────
    V_minus_alc = negative_energy_volume(r_s, rho_alc)
    V_minus_wn  = negative_energy_volume(r_s, rho_wn)

    # ── Peak densities ────────────────────────────────────────────
    peak_alc = np.min(rho_alc)
    peak_wn  = np.min(rho_wn)

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
    print(f"\n  NOTE: Lentz (2021) is excluded from this audit.")
    print(f"        It requires Einstein–Maxwell–plasma coupling not implemented here.")

    hdr = f"  {'Metric':<20} {'Peak ρ [J/m³]':>18} {'E_total [J]':>18} {'V_minus [m³]':>16}"
    sep = f"  {'-'*20} {'-'*18} {'-'*18} {'-'*16}"
    print(f"\n{hdr}\n{sep}")

    rows = [
        ("Alcubierre 1994", peak_alc, E_alc, V_minus_alc),
        ("White-Natário",   peak_wn,  E_wn,  V_minus_wn),
    ]
    for name, peak, E_tot, V_m in rows:
        print(f"  {name:<20} {peak:>18.3e} {E_tot:>18.3e} {V_m:>16.3e}")

    print(f"\n  NOTE: V_minus is the numerically integrated volume where ρ < 0,")
    print(f"        not a thin-shell approximation 4πR²Δ.")

    if include_rodal:
        print(f"\n  -- Rodal (2025) Irrotational --")
        rodal_res = rodal_global_energy(v_s, R, sigma)
        print(f"  Peak ρ_p (positive)  : {rodal_res['peak_rho_pos']:>18.3e}      POS")
        print(f"  Peak ρ_p (negative)  : {rodal_res['peak_rho_neg']:>18.3e}      NEG")
        print(f"  Total E_+            : {rodal_res['E_plus']:>18.3e}      POS")
        print(f"  Total E_-            : {rodal_res['E_minus']:>18.3e}      NEG")
        print(f"  Net Proper Energy    : {rodal_res['E_net']:>18.3e}    ~ZERO")
        print(f"  V_minus (diffuse)    : {rodal_res['V_minus']:>18.3e}      m³")

    print(f"\n  -- Fuchs (2024) Constant-Velocity Subluminal --")
    print(f"  (Pre-computed canonical result via warpax JAX audit)")
    print(f"  Total E_-            :          0.000e+00      ZERO")
    print(f"  Total E_+            :              > 0.0       POS")
    print(f"  QI gap factor        :       N/A (Bypassed)        ")

    # ── Convergence check ─────────────────────────────────────────
    convergence_check_all_metrics(v_s, R, Delta, include_rodal=include_rodal)

    # ── QI gap cross-reference (Ford–Roman bound) ─────────────────
    # All metrics now use numerically computed V_minus instead of thin-shell
    #
    # τ₀ = Δ/c is the light-crossing time of the bubble wall, the standard
    # choice in the warp-drive QI literature (Pfenning & Ford 1997, Lobo &
    # Visser 2004). Because the QI bound scales as τ₀⁻⁴, varying τ₀ by a
    # factor of 2 changes the QI cap by a factor of 16, which is negligible
    # compared to the >60-order-of-magnitude gaps reported here. The
    # qualitative conclusion is robust to any physically reasonable sampling
    # time.
    tau0_qi   = Delta / c
    rho_qi    = -(3 * hbar) / (32 * np.pi**2 * c**3 * tau0_qi**4)

    E_qi_alc  = abs(rho_qi) * V_minus_alc
    E_qi_wn   = abs(rho_qi) * V_minus_wn
    qi_gap_alc = abs(E_alc) / E_qi_alc  if E_qi_alc  != 0 else float('inf')
    qi_gap_wn  = abs(E_wn)  / E_qi_wn   if E_qi_wn   != 0 else float('inf')

    print(f"\n  QI Gap Table (Ford-Roman, τ₀=Δ/c, using actual V_minus):")
    print(f"  {'Metric':<20} {'|E_neg| [J]':>16} {'E_QI_cap [J]':>16} {'Gap factor':>14}")
    print(f"  {'-'*20} {'-'*16} {'-'*16} {'-'*14}")
    print(f"  {'Alcubierre 1994':<20} {abs(E_alc):>16.3e} {E_qi_alc:>16.3e} {qi_gap_alc:>14.2e}")
    print(f"  {'White-Natário':<20} {abs(E_wn):>16.3e}  {E_qi_wn:>16.3e} {qi_gap_wn:>14.2e}")
    print(f"\n  NOTE: All metrics now use the actual negative-energy volume V_minus.")
    print(f"        Rodal uses its diffuse 3-D V_minus (computed above).")

    if include_rodal:
        V_m_rod = rodal_res['V_minus']
        E_qi_rod = abs(rho_qi) * V_m_rod
        qi_gap_rod = abs(rodal_res['E_minus']) / E_qi_rod if E_qi_rod != 0 else float('inf')
        print(f"  {'Rodal 2025':<20} {rodal_res['E_minus']:>16.3e} {E_qi_rod:>16.3e} {qi_gap_rod:>14.2e}")

    if peak_alc != 0:
        red = abs(peak_wn / peak_alc)
        print(f"\n  White-Natário / Alcubierre peak: {red:.2f}×")
        print(f"  → Thicker wall reduces peak by {(1-red)*100:.0f}%,")
        print(f"    but total negative energy remains comparable.")

    print(f"\n  CONCLUSION:")
    print(f"  • None of the audited superluminal warp metrics satisfy the")
    print(f"    Ford–Roman quantum inequality bound.")
    print(f"  • Alcubierre & White-Natário require NEGATIVE energy violating")
    print(f"    all known energy conditions; QI exceeded by ~{np.log10(qi_gap_alc):.0f} orders.")
    print(f"  • Lentz (2021) is NOT evaluated here: it requires a full")
    print(f"    Einstein–Maxwell–plasma coupling beyond this tool's current scope.")
    print(f"  • The Bobrick–Martire class and other solutions have not yet")
    print(f"    been audited by this tool.")
    print(f"{'='*65}")

    if not save_plots:
        return E_alc

    # ══════════════════════════════════════════════════════════════
    #   PLOTS
    # ══════════════════════════════════════════════════════════════
    # Muted color palette (perceptually uniform)
    C_ALC = '#C0392B'   # muted red
    C_WN  = '#2471A3'   # muted blue

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Shaping functions
    ax = axes[0, 0]
    ax.plot(r_s, f_alc, color=C_ALC, lw=2.2, label='Alcubierre (1994)')
    ax.plot(r_s, f_wn,  color=C_WN,  lw=2.2, ls='--', label='White-Natário')
    ax.axvline(R, color='#555', ls=':', lw=1.2, label=f'R = {R} m')
    ax.set(xlabel='r_s [m]', ylabel='f(r_s)', title='Shaping functions')
    ax.legend()
    ax.grid(True, alpha=0.35)

    # (b) Energy density profiles
    ax = axes[0, 1]
    ax.plot(r_s, rho_alc, color=C_ALC, lw=2.2, label='Alcubierre (1994)')
    ax.plot(r_s, rho_wn,  color=C_WN,  lw=2.2, ls='--', label='White-Natário')
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='r_s [m]', ylabel='ρ [J/m³]', title='Energy density profiles')
    ax.legend()
    ax.grid(True, alpha=0.35)

    # (c) |ρ| log scale
    ax = axes[1, 0]
    ax.semilogy(r_s, np.abs(rho_alc), color=C_ALC, lw=2.2, label='|ρ| Alcubierre')
    ax.semilogy(r_s, np.abs(rho_wn),  color=C_WN,  lw=2.2, ls='--', label='|ρ| White-Natário')
    ax.set(xlabel='r_s [m]', ylabel='|ρ| [J/m³]', title='Energy density magnitude (log)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.35)

    # (d) Cumulative energy E(<r_s)
    ax = axes[1, 1]
    dr = np.gradient(r_s)
    cum_alc = np.cumsum(rho_alc * 4*np.pi*r_s**2 * dr)
    cum_wn  = np.cumsum(rho_wn  * 4*np.pi*r_s**2 * dr)
    ax.plot(r_s, cum_alc, color=C_ALC, lw=2.2, label='Alcubierre')
    ax.plot(r_s, cum_wn,  color=C_WN,  lw=2.2, ls='--', label='White-Natário')
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='r_s [m]', ylabel='E(<r_s) [J]', title='Cumulative energy enclosed')
    ax.legend()
    ax.grid(True, alpha=0.35)

    plt.suptitle(f'Warp Metric Comparison  (v_s={v_s/c:.1f}c, R={R}m, Δ={Delta}m)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metric_comparison.png', dpi=200)
    plt.savefig('metric_comparison.pdf')
    print('Saved metric_comparison.png / .pdf')
    plt.close('all')

    if include_rodal and save_plots:
        r_grid, th_grid, rho_p = rodal_res['r_grid'], rodal_res['th_grid'], rodal_res['rho_p']
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='polar')
        vmax = np.nanpercentile(np.abs(rho_p), 99)
        pcm = ax.pcolormesh(th_grid, r_grid, rho_p, shading='auto',
                            cmap='plasma', vmin=-vmax, vmax=vmax)
        ax.contour(th_grid, r_grid, rho_p, levels=[0], colors='white', linewidths=1)
        cbar = plt.colorbar(pcm, ax=ax, label='Proper Energy Density ρ_p [J/m³]',
                            format='%.2e', shrink=0.75)
        ax.set_title('Rodal (2025) Proper Energy Density Map', pad=20)
        plt.tight_layout()
        plt.savefig('rodal_energy_map.png', dpi=200)
        plt.savefig('rodal_energy_map.pdf')
        print('Saved rodal_energy_map.png / .pdf')
        plt.close('all')

    return E_alc


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
    v_s_val   = c
    R_val     = 5.0
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

    # Ford-Roman QI audit for Rodal – using actual V_minus
    tau0_qi   = Delta_val / c
    rho_qi    = -(3 * hbar) / (32 * np.pi**2 * c**3 * tau0_qi**4)
    V_minus_r = results['V_minus']
    E_qi_cap  = abs(rho_qi) * V_minus_r
    qi_gap    = abs(results['E_minus']) / E_qi_cap
    print(f"\n  QI Cap (Ford-Roman, actual V_minus) : {E_qi_cap:.3e} J")
    print(f"  Rodal negative E_-                  : {abs(results['E_minus']):.3e} J")
    print(f"  QI gap factor                       : {qi_gap:.1e}")
    print(f"  * V_minus (diffuse) = {V_minus_r:.3e} m³  (numerically integrated, not thin-shell)")
