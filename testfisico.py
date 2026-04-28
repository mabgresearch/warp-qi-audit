import argparse
import numpy as np
import matplotlib.pyplot as plt
from metric_explorer import run_metric_comparison
from constants import HBAR, C, G

CONFIG = dict(
    Delta = 0.2614,
    R = 3.0,
    v_s = 1.1 * C,
    m_ship_spec = 100000
)

# Physical constants (CODATA 2018 or similar)
ME     = 9.1093837015e-31  # kg  (electron mass)

def calculate_bounds(R, Delta, v_s, m_ship_spec):
    # Sampling time scale for QI (the "measurement window")
    tau0  = Delta / C          # seconds
    rho_QI = - (3 * HBAR) / (32 * np.pi**2 * C**3 * tau0**4)
    print(f"QI energy density bound = {rho_QI:.3e} J/m³")
    
    # Volume of thin shell
    V_shell = 4 * np.pi * R**2 * Delta   # m³
    print(f"Bubble wall volume ≈ {V_shell:.2f} m³")
    
    E_QI_allowed = abs(rho_QI) * V_shell   # J
    print(f"Total allowed negative energy ≈ {E_QI_allowed:.3e} J")
    
    # Mass-energy of displaced vacuum
    M_vac = E_QI_allowed / C**2   # kg
    print(f"Effective negative mass (vacuum displacement) ≈ {M_vac:.3e} kg")
    
    # Compare to SHIP budget
    m_vac_fraction = M_vac / m_ship_spec
    print(f"Negative mass fraction = {m_vac_fraction:.3e} (\"part per million\")")
    m_shield_kg = m_ship_spec * m_vac_fraction
    print(f"Negative mass required for {m_ship_spec/1000:.0f}-tonne ship: {m_shield_kg:.3e} kg")
    
    l_photon = HBAR * C / E_QI_allowed   # m
    print(f"Characteristic photon wavelength = {l_photon:.3e} m")
    
    l_fermi = (HBAR**3 / (ME**3 * C))**(1/4)  # Fermi momentum wavelength
    print(f"Atomic Fermi wavelength ≈ {l_fermi:.3e} m")
    
    m_vac_particles = int(np.ceil(M_vac / ME))
    print(f"Negative mass particles needed: {m_vac_particles} (approx)")

    # Dimensional estimate of energy
    E_req_total = -(C**4 / (2*G)) * (v_s / C) * (R**2 / Delta)   # J
    print("\n[NOTE] E_req_total is a dimensional order-of-magnitude estimate from Alcubierre's original paper,")
    print("       not a precision numerical ADM result.")
    print(f"Required total negative energy ≈ {E_req_total:.3e} J")
    
    rho_req = E_req_total / V_shell   # J/m³
    print(f"Required negative energy density ≈ {rho_req:.3e} J/m³")
    
    tau_match = (3 * HBAR / (32 * np.pi**2 * C**3 * np.abs(rho_req)))**(1/4)
    m_vac_match = np.abs(E_req_total) / C**2
    m_vac_frac_match = m_vac_match / m_ship_spec
    m_vac_particles_match = int(np.ceil(m_vac_match / ME))
    Delta_hypothetical = tau_match * C   # m

    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  HYPOTHETICAL: Δ needed to close gap  (UNPHYSICAL)      ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  If we set QI-allowed |E| = warp-required |E| and solve")
    print(f"  for the wall thickness Δ, we get:")
    print(f"")
    print(f"    Δ_hypothetical ≈ {Delta_hypothetical:.3e} m")
    print(f"    |E|_required   ≈ {np.abs(E_req_total):.3e} J")
    print(f"    m_vac          ≈ {m_vac_match:.3e} kg")
    print(f"    N_particles    ≈ {m_vac_particles_match:,}")
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

    gap_density = abs(rho_req / rho_QI)
    gap_total   = abs(E_req_total / E_QI_allowed)

    return rho_QI, E_QI_allowed, m_shield_kg, E_req_total, gap_density, gap_total

def plot_energy_vs_thickness(R, m_ship_spec, m_shield_kg):
    Delta_range = np.logspace(-4, -1, 50)   # 0.1 mm to 10 cm
    tau0_range  = Delta_range / C
    rho_QI_range = -3*HBAR / (32*np.pi**2*C**3*tau0_range**4)
    V_shell_range = 4*np.pi*R**2*Delta_range
    E_allowed_range = np.abs(rho_QI_range)*V_shell_range
    M_vac_range = E_allowed_range/C**2
    m_vac_frac_range = M_vac_range/m_ship_spec
    m_vac_particles_range = (m_shield_kg/ME) * np.ones_like(m_vac_frac_range)

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    axes = axes.flatten()

    axes[0].loglog(Delta_range*1e3, np.abs(rho_QI_range), color='red')
    axes[0].set(xlabel='Wall thickness Δ [mm]', ylabel='|ρ_QI| [J/m³]', title='QI energy density bound')
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].loglog(Delta_range*1e3, V_shell_range, color='blue')
    axes[1].set(xlabel='Wall thickness Δ [mm]', ylabel='Bubble volume [m³]', title='Effective volume of QI region')
    axes[1].grid(True, which='both', alpha=0.3)

    axes[2].loglog(Delta_range*1e3, E_allowed_range, color='green')
    axes[2].set(xlabel='Wall thickness Δ [mm]', ylabel='|E_QI_allowed| [J]', title='Total allowed negative energy')
    axes[2].grid(True, which='both', alpha=0.3)

    axes[3].semilogy(Delta_range*1e3, M_vac_range, color='purple')
    axes[3].set(xlabel='Wall thickness Δ [mm]', ylabel='m_vac [kg]', title='Effective negative mass')
    axes[3].grid(True, which='both', alpha=0.3)

    axes[4].loglog(Delta_range*1e3, m_vac_frac_range, color='orange')
    axes[4].set(xlabel='Wall thickness Δ [mm]', ylabel='Fraction of m_ship', title='Negative mass fraction')
    axes[4].grid(True, which='both', alpha=0.3)
    axes[4].axhline(1e-6, color='gray', ls=':', alpha=0.5, label='1 ppm')
    axes[4].legend(loc='best')

    axes[5].semilogy(Delta_range*1e3, m_vac_particles_range, color='brown')
    axes[5].set(xlabel='Wall thickness Δ [mm]', ylabel='Particles |N_vac|', title='Negative particles needed')
    axes[5].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('qi_vs_Delta.png', dpi=200)
    print('Saved qi_vs_Delta.png')
    plt.close()

def plot_energy_vs_radius(Delta, rho_QI, m_ship_spec, m_shield_kg, E_req_total):
    R_range = np.logspace(1, 4, 50)     # 10 m to 10 km
    V_shell_R = 4*np.pi*R_range**2*Delta
    E_allowed_R = np.abs(rho_QI)*V_shell_R
    M_vac_R = E_allowed_R/C**2
    m_vac_frac_R = M_vac_R/m_ship_spec
    m_vac_particles_R = (m_shield_kg/ME) * np.ones_like(m_vac_frac_R)

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    axes = axes.flatten()

    axes[0].loglog(R_range, np.abs(rho_QI)*np.ones_like(R_range), color='red')
    axes[0].set(xlabel='Radius R [m]', ylabel='|ρ_QI| [J/m³]', title='QI energy density bound (constant)')
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].loglog(R_range, V_shell_R, color='blue')
    axes[1].set(xlabel='Radius R [m]', ylabel='Bubble volume [m³]', title='Effective volume of QI region')
    axes[1].grid(True, which='both', alpha=0.3)

    axes[2].loglog(R_range, E_allowed_R, color='green')
    axes[2].set(xlabel='Radius R [m]', ylabel='|E_QI_allowed| [J]', title='Total allowed negative energy')
    axes[2].grid(True, which='both', alpha=0.3)

    axes[3].semilogy(R_range, M_vac_R, color='purple')
    axes[3].set(xlabel='Radius R [m]', ylabel='m_vac [kg]', title='Effective negative mass')
    axes[3].grid(True, which='both', alpha=0.3)

    axes[4].loglog(R_range, m_vac_frac_R, color='orange')
    axes[4].set(xlabel='Radius R [m]', ylabel='Fraction of m_ship', title='Negative mass fraction')
    axes[4].grid(True, which='both', alpha=0.3)
    axes[4].axhline(1e-6, color='gray', ls=':', alpha=0.5, label='1 ppm')
    axes[4].legend(loc='best')

    axes[5].semilogy(R_range, m_vac_particles_R, color='brown')
    axes[5].set(xlabel='Radius R [m]', ylabel='Particles |N_vac|', title='Negative particles needed')
    axes[5].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('qi_vs_Radius.png', dpi=200)
    print('Saved qi_vs_Radius.png')
    plt.close()


def plot_qi_vs_radius_ship(Delta, rho_QI, m_ship_spec, E_req_total):
    R_range_ship = np.logspace(0, 4, 100)      # 1 m to 10 km
    rho_QI_R = rho_QI * np.ones_like(R_range_ship)
    rho_req_R = np.abs(E_req_total) / (4*np.pi*R_range_ship**2*Delta)

    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(R_range_ship, np.abs(rho_QI_R), color='red', label='QI limit')
    ax.loglog(R_range_ship, rho_req_R, '--', color='orange', label=f'SHIP ({m_ship_spec/1e3:.0f}t)')
    ax.set(xlabel='Radius R [m]', ylabel='Energy density |ρ| [J/m³]', title='QI energy density vs SHIP requirement')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('qi_vs_Radius_with_SHIP.png', dpi=200)
    print('Saved qi_vs_Radius_with_SHIP.png')
    plt.close()

def plot_phase_space_ship_vs_qi(Delta, R, rho_QI, m_ship_spec):
    v_s_range = np.linspace(0.1*C, 10*C, 100)
    E_QI_allowed = np.abs(rho_QI) * 4 * np.pi * R**2 * Delta
    E_req_vals = (C**4 / (2*G)) * (v_s_range / C) * (R**2 / Delta)

    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(v_s_range / C, np.ones_like(v_s_range) * E_QI_allowed, color='red', label='QI-allowed total |E| (constant)')
    ax.loglog(v_s_range / C, E_req_vals, '--', color='blue', label='Warp required total |E|')
    ax.set(xlabel='Ship speed v_s [c]', ylabel='Total Energy Magnitude |E| [J]', title='QI vs Warp Requirement across Speeds')
    ax.text(0.5, 0.05, "Note: Proper propulsion comparison requires specific ship model.", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase_space_ship_vs_qi.png', dpi=200)
    print('Saved phase_space_ship_vs_qi.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Warp Drive Energy Audit Simulation")
    parser.add_argument('--delta', type=float, default=CONFIG['Delta'], help="Wall thickness Δ (m)")
    parser.add_argument('--radius', type=float, default=CONFIG['R'], help="Bubble radius R (m)")
    parser.add_argument('--velocity_c', type=float, default=CONFIG['v_s']/C, help="Ship velocity as a fraction of c")
    parser.add_argument('--mass_ship', type=float, default=CONFIG['m_ship_spec'], help="Mass of ship (kg)")
    args = parser.parse_args()

    Delta = args.delta
    R = args.radius
    v_s = args.velocity_c * C
    m_ship_spec = args.mass_ship

    rho_QI, E_QI_allowed, m_shield_kg, E_req_total, gap_density, gap_total = calculate_bounds(R, Delta, v_s, m_ship_spec)
    
    plot_energy_vs_thickness(R, m_ship_spec, m_shield_kg)
    plot_energy_vs_radius(Delta, rho_QI, m_ship_spec, m_shield_kg, E_req_total)
    plot_qi_vs_radius_ship(Delta, rho_QI, m_ship_spec, E_req_total)
    plot_phase_space_ship_vs_qi(Delta, R, rho_QI, m_ship_spec)

    # Metric Explorer
    print("\n═══════════════════════════════════════════════════════════════════")
    print(" METRIC EXPLORER – Compare Alcubierre / White-Natário / Lentz")
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
    print(f"")
    # rho_req calculated within bounds
    rho_req = E_req_total / (4 * np.pi * R**2 * Delta)
    print(f"  Warp required energy density: {rho_req:.3e} J/m³")
    print(f"  Warp required total energy  : {E_req_total:.3e} J")
    print(f"")
    print(f"  GAP (density) : {gap_density:.1e}")
    print(f"  GAP (total)   : {gap_total:.1e}")
    print(f"")
    print(f"  CONCLUSION: The required negative energy exceeds the")
    print(f"  quantum inequality bound by ~{np.log10(gap_density):.0f} orders of magnitude.")
    print(f"  Within known QFT, such a warp bubble is physically")
    print(f"  impossible.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()