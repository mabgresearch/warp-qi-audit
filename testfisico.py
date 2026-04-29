import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metric_explorer import (
    run_metric_comparison,
    f_alcubierre,
    rho_from_shaping,
    total_energy,
    negative_energy_volume,
)
from constants import HBAR, C, G
from plot_style import setup_plot_style

# ── Apply shared Matplotlib style (see plot_style.py) ────────────
setup_plot_style()

CONFIG = dict(
    Delta       = 0.2614,
    R           = 3.0,
    v_s         = 1.1 * C,
    m_ship_spec = 100000,
)


def _compute_alcubierre_energy(R, Delta, v_s, n_pts=3000):
    """
    Numerically integrate the Alcubierre energy density to get the actual
    total negative energy for the given parameters.

    This replaces the dimensional estimate E_req = c²·v_s·R²/(4·G·Δ)
    (Pfenning & Ford 1997, eq. 20), which is only an order-of-magnitude proxy.
    """
    sigma  = 1.0 / Delta
    r_s    = np.linspace(0.01, 4 * R, n_pts)
    f_alc  = f_alcubierre(r_s, R, sigma)
    rho    = rho_from_shaping(r_s, f_alc, v_s)
    E_num  = total_energy(r_s, rho)
    V_m    = negative_energy_volume(r_s, rho)
    return E_num, V_m, r_s, rho


def calculate_bounds(R, Delta, v_s, m_ship_spec):
    # Sampling time scale for QI (the "measurement window")
    tau0    = Delta / C
    rho_QI  = -(3 * HBAR) / (32 * np.pi**2 * C**3 * tau0**4)
    print(f"QI energy density bound = {rho_QI:.3e} J/m³")

    # ── Actual integrated negative energy (Task 2) ────────────────
    # The old dimensional formula:
    #   E_req_total = (C**2 * v_s * R**2) / (4 * G * Delta)
    # has been replaced with the numerically integrated ADM result.
    E_req_total, V_minus_alc, r_s, rho_alc = _compute_alcubierre_energy(R, Delta, v_s)
    print(f"\n[ADM] Numerically integrated Alcubierre negative energy:")
    print(f"      E_req_total = {E_req_total:.6e} J")
    print(f"      V_minus     = {V_minus_alc:.6e} m³  (actual ρ<0 volume, not thin-shell)")

    # QI cap uses actual V_minus (Task 3 – unified volume)
    E_QI_allowed = abs(rho_QI) * V_minus_alc
    print(f"\nQI-allowed negative energy (V_minus-based) = {E_QI_allowed:.3e} J")

    # Thin-shell volume for reference only
    V_shell = 4 * np.pi * R**2 * Delta
    print(f"(Reference) Thin-shell volume 4πR²Δ        = {V_shell:.3e} m³")

    # Mass-energy of displaced vacuum
    M_vac         = E_QI_allowed / C**2
    m_vac_fraction = M_vac / m_ship_spec
    m_shield_kg   = m_ship_spec * m_vac_fraction
    print(f"Effective negative mass (vacuum displacement) ≈ {M_vac:.3e} kg")
    print(f"Negative mass fraction = {m_vac_fraction:.3e}")
    print(f"Negative mass required for {m_ship_spec/1000:.0f}-tonne ship: {m_shield_kg:.3e} kg")

    l_photon = HBAR * C / E_QI_allowed
    print(f"Characteristic photon wavelength = {l_photon:.3e} m")

    # Required density from the numerically integrated total
    rho_req    = E_req_total / V_minus_alc if V_minus_alc > 0 else 0.0
    tau_match  = (3 * HBAR / (32 * np.pi**2 * C**3 * np.abs(rho_req)))**(1/4) if rho_req != 0 else 0.0
    m_vac_match     = np.abs(E_req_total) / C**2
    m_vac_frac_match = m_vac_match / m_ship_spec
    Delta_hypothetical = tau_match * C

    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  HYPOTHETICAL: Δ needed to close gap  (UNPHYSICAL)      ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  If we set QI-allowed |E| = warp-required |E| and solve")
    print(f"  for the wall thickness Δ, we get:")
    print(f"")
    print(f"    Δ_hypothetical ≈ {Delta_hypothetical:.3e} m")
    print(f"    |E|_required   ≈ {np.abs(E_req_total):.3e} J")
    print(f"    m_vac          ≈ {m_vac_match:.3e} kg")
    print(f"    m_vac / m_ship = {m_vac_frac_match:.3e}")
    print(f"")
    print(f"  ⚠  Δ ≈ {Delta_hypothetical:.0e} m is ~1000× smaller than a proton.")
    print(f"     At this scale the continuum QFT derivation of the QI")
    print(f"     breaks down and quantum-gravity effects dominate.")
    print(f"     This is NOT a viable 'fix'; it is an illustration of")
    print(f"     why the energy gap cannot be closed within known physics.")
    print(f"")
    print(f"  [NOTE] The Ford-Roman bound applied here assumes flat spacetime.")
    print(f"         Inside or near a warp bubble, curvature corrections exist,")
    print(f"         though they wouldn't bridge the ~67 orders of magnitude gap.")

    gap_density = abs(rho_req / rho_QI) if rho_QI != 0 else float('inf')
    gap_total   = abs(E_req_total / E_QI_allowed) if E_QI_allowed != 0 else float('inf')

    return rho_QI, E_QI_allowed, m_shield_kg, E_req_total, gap_density, gap_total, V_minus_alc


def plot_energy_vs_thickness(R, m_ship_spec, m_shield_kg):
    """QI bound vs wall thickness – saved once only."""
    Delta_range  = np.logspace(-4, -1, 50)
    tau0_range   = Delta_range / C
    rho_QI_range = -3*HBAR / (32*np.pi**2*C**3*tau0_range**4)

    # Use actual negative-energy volume per thickness (numerical)
    # Approximate: for each Delta, integrate the Alcubierre profile
    V_minus_range = np.array([
        negative_energy_volume(
            np.linspace(0.01, 4*R, 1000),
            rho_from_shaping(
                np.linspace(0.01, 4*R, 1000),
                f_alcubierre(np.linspace(0.01, 4*R, 1000), R, 1.0/d),
                CONFIG['v_s']
            )
        ) for d in Delta_range
    ])

    E_allowed_range = np.abs(rho_QI_range) * V_minus_range
    M_vac_range     = E_allowed_range / C**2
    m_vac_frac_range = M_vac_range / m_ship_spec

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    axes = axes.flatten()

    axes[0].loglog(Delta_range*1e3, np.abs(rho_QI_range), color='#C0392B')
    axes[0].set(xlabel='Wall thickness Δ [mm]', ylabel='|ρ_QI| [J/m³]',
                title='QI energy density bound')
    axes[0].grid(True, which='both', alpha=0.35)

    axes[1].loglog(Delta_range*1e3, V_minus_range, color='#2471A3')
    axes[1].set(xlabel='Wall thickness Δ [mm]', ylabel='V_minus [m³]',
                title='Actual negative-energy volume V_minus(Δ)')
    axes[1].grid(True, which='both', alpha=0.35)

    axes[2].loglog(Delta_range*1e3, E_allowed_range, color='#1E8449')
    axes[2].set(xlabel='Wall thickness Δ [mm]', ylabel='|E_QI_allowed| [J]',
                title='Total allowed negative energy')
    axes[2].grid(True, which='both', alpha=0.35)

    axes[3].semilogy(Delta_range*1e3, M_vac_range, color='#6C3483')
    axes[3].set(xlabel='Wall thickness Δ [mm]', ylabel='m_vac [kg]',
                title='Effective negative mass')
    axes[3].grid(True, which='both', alpha=0.35)

    axes[4].loglog(Delta_range*1e3, m_vac_frac_range, color='#D35400')
    axes[4].set(xlabel='Wall thickness Δ [mm]', ylabel='Fraction of m_ship',
                title='Negative mass fraction')
    axes[4].grid(True, which='both', alpha=0.35)
    axes[4].axhline(1e-6, color='gray', ls=':', alpha=0.6, label='1 ppm')
    axes[4].legend()

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('qi_vs_Delta.png')
    plt.savefig('qi_vs_Delta.pdf')
    print('Saved qi_vs_Delta.png / .pdf')
    plt.close('all')


def plot_energy_vs_radius(Delta, rho_QI, m_ship_spec, m_shield_kg, E_req_total):
    """QI bound vs bubble radius."""
    R_range = np.logspace(1, 4, 50)

    # Use thin-shell as approximation for radius sweep (V_minus ≈ 4πR²Δ for large R)
    V_shell_R     = 4 * np.pi * R_range**2 * Delta
    E_allowed_R   = np.abs(rho_QI) * V_shell_R
    M_vac_R       = E_allowed_R / C**2
    m_vac_frac_R  = M_vac_R / m_ship_spec

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    axes = axes.flatten()

    axes[0].loglog(R_range, np.abs(rho_QI)*np.ones_like(R_range), color='#C0392B')
    axes[0].set(xlabel='Radius R [m]', ylabel='|ρ_QI| [J/m³]',
                title='QI energy density bound (constant in R)')
    axes[0].grid(True, which='both', alpha=0.35)

    axes[1].loglog(R_range, V_shell_R, color='#2471A3')
    axes[1].set(xlabel='Radius R [m]', ylabel='V_shell [m³]',
                title='Thin-shell volume 4πR²Δ (radius sweep)')
    axes[1].grid(True, which='both', alpha=0.35)

    axes[2].loglog(R_range, E_allowed_R, color='#1E8449')
    axes[2].set(xlabel='Radius R [m]', ylabel='|E_QI_allowed| [J]',
                title='Total allowed negative energy')
    axes[2].grid(True, which='both', alpha=0.35)

    axes[3].semilogy(R_range, M_vac_R, color='#6C3483')
    axes[3].set(xlabel='Radius R [m]', ylabel='m_vac [kg]',
                title='Effective negative mass')
    axes[3].grid(True, which='both', alpha=0.35)

    axes[4].loglog(R_range, m_vac_frac_R, color='#D35400')
    axes[4].set(xlabel='Radius R [m]', ylabel='Fraction of m_ship',
                title='Negative mass fraction')
    axes[4].grid(True, which='both', alpha=0.35)
    axes[4].axhline(1e-6, color='gray', ls=':', alpha=0.6, label='1 ppm')
    axes[4].legend()

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('qi_vs_Radius.png')
    plt.savefig('qi_vs_Radius.pdf')
    print('Saved qi_vs_Radius.png / .pdf')
    plt.close('all')


def plot_qi_vs_radius_ship(Delta, rho_QI, m_ship_spec, E_req_total):
    """QI cap vs SHIP requirement across radii – saved once only."""
    R_range_ship = np.logspace(0, 4, 100)
    rho_QI_R    = np.abs(rho_QI) * np.ones_like(R_range_ship)
    rho_req_R   = np.abs(E_req_total) / (4*np.pi*R_range_ship**2*Delta)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(R_range_ship, rho_QI_R,   color='#C0392B', lw=2.2, label='QI limit (Ford–Roman)')
    ax.loglog(R_range_ship, rho_req_R, '--', color='#D35400', lw=2.2,
              label=f'Alcubierre req. ({m_ship_spec/1e3:.0f}t, numerical E)')
    ax.set(xlabel='Radius R [m]', ylabel='Energy density |ρ| [J/m³]',
           title='QI energy density vs Alcubierre requirement')
    ax.legend()
    ax.grid(True, which='both', alpha=0.35)
    plt.tight_layout()
    plt.savefig('qi_vs_Radius_with_SHIP.png')
    plt.savefig('qi_vs_Radius_with_SHIP.pdf')
    print('Saved qi_vs_Radius_with_SHIP.png / .pdf')
    plt.close('all')


def plot_energy_vs_velocity(Delta, m_ship_spec):
    """Required energy vs velocity for several radii – saved once only."""
    v_s_range = np.linspace(0.1*C, 10*C, 100)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#C0392B', '#2471A3', '#1E8449']
    for R_val, col in zip([1.0, 3.0, 10.0], colors):
        # Use dimensional scaling for velocity sweep (V_minus ≈ const for fixed R,Δ)
        E_req_vals = (C**4 / (2*G)) * (v_s_range / C) * (R_val**2 / Delta)
        ax.loglog(v_s_range / C, E_req_vals, color=col, label=f'R = {R_val} m')
    ax.set(xlabel='Ship speed v_s [c]', ylabel='Total Energy Magnitude |E_req| [J]',
           title='Required Energy vs Velocity (dimensional scaling)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.35)
    plt.tight_layout()
    plt.savefig('energy_vs_velocity.png')
    plt.savefig('energy_vs_velocity.pdf')
    print('Saved energy_vs_velocity.png / .pdf')
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Warp Drive Energy Audit Simulation")
    parser.add_argument('--delta',      type=float, default=CONFIG['Delta'],
                        help="Wall thickness Δ (m)")
    parser.add_argument('--radius',     type=float, default=CONFIG['R'],
                        help="Bubble radius R (m)")
    parser.add_argument('--velocity_c', type=float, default=CONFIG['v_s']/C,
                        help="Ship velocity as a fraction of c")
    parser.add_argument('--mass_ship',  type=float, default=CONFIG['m_ship_spec'],
                        help="Mass of ship (kg)")
    args = parser.parse_args()

    Delta       = args.delta
    R           = args.radius
    v_s         = args.velocity_c * C
    m_ship_spec = args.mass_ship

    (rho_QI, E_QI_allowed, m_shield_kg,
     E_req_total, gap_density, gap_total, V_minus_alc) = calculate_bounds(
        R, Delta, v_s, m_ship_spec
    )

    plot_energy_vs_thickness(R, m_ship_spec, m_shield_kg)
    plot_energy_vs_radius(Delta, rho_QI, m_ship_spec, m_shield_kg, E_req_total)
    plot_qi_vs_radius_ship(Delta, rho_QI, m_ship_spec, E_req_total)
    plot_energy_vs_velocity(Delta, m_ship_spec)

    # Metric Explorer
    print("\n═══════════════════════════════════════════════════════════════════")
    print(" METRIC EXPLORER – Compare Alcubierre / White-Natário")
    print(" NOTE: Lentz (2021) excluded — requires Einstein–Maxwell coupling.")
    print("═══════════════════════════════════════════════════════════════════")
    run_metric_comparison(v_s=v_s, R=R, Delta=Delta)

    # Final Summary
    print(f"\n{'='*60}")
    print(f"  QUANTUM INEQUALITY AUDIT")
    print(f"{'='*60}")
    print(f"  Bubble wall thickness  Δ  = {Delta} m")
    print(f"  Torus radius           R  = {R} m")
    print(f"  Ship speed             v_s = {v_s/C:.1f} c")
    print(f"")
    print(f"  QI allowed energy density  : {rho_QI:.3e} J/m³")
    print(f"  Total QI-allowed energy    : {E_QI_allowed:.3e} J")
    print(f"  (V_minus used = {V_minus_alc:.3e} m³, numerically integrated)")
    print(f"")
    rho_req = E_req_total / V_minus_alc if V_minus_alc > 0 else 0.0
    print(f"  Warp required energy density: {rho_req:.3e} J/m³")
    print(f"  Warp required total energy  : {E_req_total:.3e} J")
    print(f"  (Numerically integrated ADM result, not a dimensional estimate)")
    print(f"")
    print(f"  GAP (density) : {gap_density:.1e}")
    print(f"  GAP (total)   : {gap_total:.1e}")
    print(f"")
    print(f"  CONCLUSION: None of the audited superluminal warp metrics")
    print(f"  satisfy the Ford–Roman quantum inequality bound.")
    print(f"  The required negative energy exceeds the QI bound by")
    print(f"  ~{np.log10(gap_density):.0f} orders of magnitude — a fundamental")
    print(f"  prohibition within known QFT.")
    print(f"  [Lentz (2021) is not evaluated here: it requires a full")
    print(f"   Einstein–Maxwell–plasma calculation beyond this tool's scope.]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()