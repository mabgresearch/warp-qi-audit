import numpy as np

# Physical constants (CODATA 2018 or similar)
hbar   = 1.054571817e-34   # J·s
c      = 2.99792458e8      # m/s
G      = 6.67430e-11       # m³/(kg·s²)
ME     = 9.1093837015e-31  # kg  (electron mass)
Delta = 0.2614             # wall thickness (m)
R     = 3.0                # approximate bubble radius (m)
v_s   = 1.1 * c            # ship velocity (m/s) – superluminal, but we keep it symbolic
# Sampling time scale for QI (the “measurement window”)
tau0  = Delta / c          # seconds
rho_QI = - (3 * hbar) / (32 * np.pi**2 * c**3 * tau0**4)
print(f"QI energy density bound = {rho_QI:.3e} J/m³")
# Volume of thin shell
V_shell = 4 * np.pi * R**2 * Delta   # m³
print(f"Bubble wall volume ≈ {V_shell:.2f} m³")
E_QI_allowed = abs(rho_QI) * V_shell   # J
print(f"Total allowed negative energy ≈ {E_QI_allowed:.3e} J")
# Mass-energy of displaced vacuum
M_vac = E_QI_allowed / c**2   # kg
print(f"Effective negative mass (vacuum displacement) ≈ {M_vac:.3e} kg")
# Compare to SHIP budget
m_ship_spec = 100e3   # kg (100 tonnes)
m_vac_fraction = M_vac / m_ship_spec
print(f"Negative mass fraction = {m_vac_fraction:.3e} (\"part per million\")")
m_shield_kg = m_ship_spec * m_vac_fraction
print(f"Negative mass required for 100-tonne ship: {m_shield_kg:.3e} kg")
l_photon = hbar * c / E_QI_allowed   # m
print(f"Characteristic photon wavelength = {l_photon:.3e} m")
l_fermi = (hbar**3 / (ME**3 * c))**(1/4)  # Fermi momentum wavelength
print(f"Atomic Fermi wavelength ≈ {l_fermi:.3e} m")
m_ship_particles = 100   # number of electrons
m_per_ship = m_ship_particles * ME
m_vac_particles = int(np.ceil(M_vac / ME))
print(f"Negative mass particles needed: {m_vac_particles} (approx)")
import matplotlib.pyplot as plt

# Same constants as before
# ... (hbar, c, G, Delta, R, v_s, tau0, rho_QI, V_shell, E_QI_allowed, M_vac, m_shield_kg, ...)

# ═══════════════════════════════════════════════════════════════════
# 1) Energy vs wall thickness (Delta)
# ═══════════════════════════════════════════════════════════════════
Delta_range = np.logspace(-4, -1, 50)   # 0.1 mm to 10 cm
tau0_range  = Delta_range / c
rho_QI_range = -3*hbar / (32*np.pi**2*c**3*tau0_range**4)
V_shell_range = 4*np.pi*R**2*Delta_range
E_allowed_range = np.abs(rho_QI_range)*V_shell_range
M_vac_range = E_allowed_range/c**2
m_vac_frac_range = M_vac_range/m_ship_spec
m_vac_particles_range = (m_shield_kg/ME) * np.ones_like(m_vac_frac_range)

fig, axes = plt.subplots(3, 2, figsize=(10, 14))
axes = axes.flatten()

axes[0].loglog(Delta_range*1e3, np.abs(rho_QI_range), color='red')
axes[0].set(xlabel='Wall thickness Δ [mm]',
            ylabel='|ρ_QI| [J/m³]',
            title='QI energy density bound')
axes[0].grid(True, which='both', alpha=0.3)

axes[1].loglog(Delta_range*1e3, V_shell_range, color='blue')
axes[1].set(xlabel='Wall thickness Δ [mm]',
            ylabel='Bubble volume [m³]',
            title='Effective volume of QI region')
axes[1].grid(True, which='both', alpha=0.3)

axes[2].loglog(Delta_range*1e3, E_allowed_range, color='green')
axes[2].set(xlabel='Wall thickness Δ [mm]',
            ylabel='|E_QI_allowed| [J]',
            title='Total allowed negative energy')
axes[2].grid(True, which='both', alpha=0.3)

axes[3].semilogy(Delta_range*1e3, M_vac_range, color='purple')
axes[3].set(xlabel='Wall thickness Δ [mm]',
            ylabel='m_vac [kg]',
            title='Effective negative mass')
axes[3].grid(True, which='both', alpha=0.3)

axes[4].loglog(Delta_range*1e3, m_vac_frac_range, color='orange')
axes[4].set(xlabel='Wall thickness Δ [mm]',
            ylabel='Fraction of m_ship', title='Negative mass fraction')
axes[4].grid(True, which='both', alpha=0.3)
axes[4].axhline(1e-6, color='gray', ls=':', alpha=0.5, label='1 ppm')
axes[4].legend(loc='best')

axes[5].semilogy(Delta_range*1e3, m_vac_particles_range, color='brown')
axes[5].set(xlabel='Wall thickness Δ [mm]',
            ylabel='Particles |N_vac|', title='Negative particles needed')
axes[5].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('qi_vs_Delta.png', dpi=200)
print('Saved qi_vs_Delta.png')

# ═══════════════════════════════════════════════════════════════════
# 2) Energy vs radius (R)
# ═══════════════════════════════════════════════════════════════════
R_range = np.logspace(1, 4, 50)     # 10 m to 10 km
# tau0 and rho_QI stay fixed: we only scale the volume
V_shell_R = 4*np.pi*R_range**2*Delta
E_allowed_R = np.abs(rho_QI)*V_shell_R
M_vac_R = E_allowed_R/c**2
m_vac_frac_R = M_vac_R/m_ship_spec
m_vac_particles_R = (m_shield_kg/ME) * np.ones_like(m_vac_frac_R)

fig, axes = plt.subplots(3, 2, figsize=(10, 14))
axes = axes.flatten()

axes[0].loglog(R_range, np.abs(rho_QI)*np.ones_like(R_range), color='red')
axes[0].set(xlabel='Radius R [m]',
            ylabel='|ρ_QI| [J/m³]',
            title='QI energy density bound (constant)')
axes[0].grid(True, which='both', alpha=0.3)

axes[1].loglog(R_range, V_shell_R, color='blue')
axes[1].set(xlabel='Radius R [m]',
            ylabel='Bubble volume [m³]',
            title='Effective volume of QI region')
axes[1].grid(True, which='both', alpha=0.3)

axes[2].loglog(R_range, E_allowed_R, color='green')
axes[2].set(xlabel='Radius R [m]',
            ylabel='|E_QI_allowed| [J]',
            title='Total allowed negative energy')
axes[2].grid(True, which='both', alpha=0.3)

axes[3].semilogy(R_range, M_vac_R, color='purple')
axes[3].set(xlabel='Radius R [m]',
            ylabel='m_vac [kg]',
            title='Effective negative mass')
axes[3].grid(True, which='both', alpha=0.3)

axes[4].loglog(R_range, m_vac_frac_R, color='orange')
axes[4].set(xlabel='Radius R [m]',
            ylabel='Fraction of m_ship', title='Negative mass fraction')
axes[4].grid(True, which='both', alpha=0.3)
axes[4].axhline(1e-6, color='gray', ls=':', alpha=0.5, label='1 ppm')
axes[4].legend(loc='best')

axes[5].semilogy(R_range, m_vac_particles_R, color='brown')
axes[5].set(xlabel='Radius R [m]',
            ylabel='Particles |N_vac|', title='Negative particles needed')
axes[5].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('qi_vs_Radius.png', dpi=200)
print('Saved qi_vs_Radius.png')
E_req_total = -(c**4 / (2*G)) * (v_s / c) * (R**2 / Delta)   # J
print(f"Required total negative energy ≈ {E_req_total:.3e} J")
rho_req = E_req_total / V_shell   # J/m³
print(f"Required negative energy density ≈ {rho_req:.3e} J/m³")
tau_match = (3 * hbar / (32 * np.pi**2 * c**3 * np.abs(rho_req)))**(1/4)
m_vac_match = np.abs(E_req_total) / c**2
m_vac_frac_match = m_vac_match / m_ship_spec
m_vac_particles_match = int(np.ceil(m_vac_match / ME))
Delta_hypothetical = tau_match * c   # m

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
# ── Phase-space volume (Appendix – illustrative) ──────────────────
# Definition: Φ = (Δx · p)³  where Δx = bubble wall thickness and
#             p = coordinate momentum m·v_s (warp-metric, no γ).
# This gives a rough measure of the phase-space "footprint" of
# the ship mass vs. the negative-energy requirement.  It does NOT
# enter the QI bound directly; it is included for intuition only.
p_ship = m_ship_spec * v_s               # coordinate momentum
Delta_phi = (Delta * p_ship)**3           # phase-space volume element
p_qi   = m_vac_match * v_s
Delta_phi_qi = (Delta * p_qi)**3
print(f"\n[Phase-space volumes – illustrative]")
print(f"  Φ_ship = (Δ·m·v_s)³ = {Delta_phi:.3e} m³·(kg·m/s)³")
print(f"  Φ_QI   = (Δ·m_vac·v_s)³ = {Delta_phi_qi:.3e} m³·(kg·m/s)³")
print(f"  Ratio Φ_QI/Φ_ship ≈ {Delta_phi_qi/Delta_phi:.1e}")

# ── Subluminal comparison (separate exercise) ─────────────────────
# For reference: Lorentz factor at a subluminal cruise speed.
# (γ is well-defined here because v_sub < c.)
v_sub = 0.99 * c
Gamma_sub = 1 / np.sqrt(1 - v_sub**2 / c**2)
p_ship_sub = m_ship_spec * Gamma_sub * v_sub
print(f"\n[Subluminal comparison at v = 0.99c]")
print(f"  Gamma        = {Gamma_sub:.2f}")
print(f"  p_ship (SR)  = {p_ship_sub:.3e} kg·m/s")

# ── Compute gap ────────────────────────────────────────────────────
gap_density = abs(rho_req / rho_QI)
gap_total   = abs(E_req_total / E_QI_allowed)
# Energy density vs radius (R), with SHIP requirement overlaid
R_range = np.logspace(0, 4, 100)      # 1 m to 10 km
rho_QI_R = rho_QI * np.ones_like(R_range)   # constant
rho_req_R = np.abs(E_req_total) / (4*np.pi*R_range**2*Delta)  # varies with R

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(R_range, np.abs(rho_QI_R), color='red', label='QI limit')
ax.loglog(R_range, rho_req_R, '--', color='orange', label=f'SHIP ({m_ship_spec/1e3:.0f}t)')
ax.set(xlabel='Radius R [m]',
        ylabel='Energy density |ρ| [J/m³]',
        title='QI energy density vs SHIP requirement')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('qi_vs_Radius_with_SHIP.png', dpi=200)
print('Saved qi_vs_Radius_with_SHIP.png')
# Phase-space diagram: log energy vs log momentum for SHIP vs QI
p_min = 1e-30
p_max = 1e10
p_vals = np.logspace(np.log10(p_min), np.log10(p_max), 200)
# For SHIP: momentum is sharply peaked around p_ship
p_vals_ship = p_vals * (p_ship / p_vals.mean())  # keep mean same
E_ship_vals = np.sqrt(p_vals_ship**2*c**2 + m_vac_match**2*c**4)
# For QI: energy ~ momentum (power-law, not sharply peaked)
E_qi_vals = c*p_vals

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(p_vals_ship, E_ship_vals, color='black', label=f'SHIP (100t at 1.1c)')
ax.loglog(p_vals, E_qi_vals, color='orange', linestyle='--', label=f'QI (negative mass)')
ax.set(xlabel='Momentum p [kg·m/s]',
       ylabel='Energy E [J]',
       title='Phase-space: SHIP vs QI')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('phase_space_ship_vs_qi.png', dpi=200)
print('Saved phase_space_ship_vs_qi.png')
# Energy density vs radius (R), with SHIP requirement overlaid
R_range = np.logspace(0, 4, 100)      # 1 m to 10 km
rho_QI_R = rho_QI * np.ones_like(R_range)   # constant
rho_req_R = np.abs(E_req_total) / (4*np.pi*R_range**2*Delta)  # varies with R

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(R_range, np.abs(rho_QI_R), color='red', label='QI limit')
ax.loglog(R_range, rho_req_R, '--', color='orange', label=f'SHIP ({m_ship_spec/1e3:.0f}t)')
ax.set(xlabel='Radius R [m]',
        ylabel='Energy density |ρ| [J/m³]',
        title='QI energy density vs SHIP requirement')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('qi_vs_Radius_with_SHIP.png', dpi=200)
print('Saved qi_vs_Radius_with_SHIP.png')
# Phase-space diagram: log energy vs log momentum for SHIP vs QI
p_min = 1e-30
p_max = 1e10
p_vals = np.logspace(np.log10(p_min), np.log10(p_max), 200)
# For SHIP: momentum is sharply peaked around p_ship
p_vals_ship = p_vals * (p_ship / p_vals.mean())  # keep mean same
E_ship_vals = np.sqrt(p_vals_ship**2*c**2 + m_vac_match**2*c**4)
# For QI: energy ~ momentum (power-law, not sharply peaked)
E_qi_vals = c*p_vals

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(p_vals_ship, E_ship_vals, color='black', label=f'SHIP (100t at 1.1c)')
ax.loglog(p_vals, E_qi_vals, color='orange', linestyle='--', label=f'QI (negative mass)')
ax.set(xlabel='Momentum p [kg·m/s]',
       ylabel='Energy E [J]',
       title='Phase-space: SHIP vs QI')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('phase_space_ship_vs_qi.png', dpi=200)
print('Saved phase_space_ship_vs_qi.png')

# ═══════════════════════════════════════════════════════════════════
# METRIC EXPLORER – Compare Alcubierre / White-Natário / Lentz
# ═══════════════════════════════════════════════════════════════════
from metric_explorer import run_metric_comparison
run_metric_comparison(v_s=v_s, R=R, Delta=Delta)


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY – QUANTUM INEQUALITY AUDIT
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  QUANTUM INEQUALITY AUDIT")
print(f"{'='*60}")
print(f"  Bubble wall thickness  Δ  = {Delta} m")
print(f"  Torus radius           R  = {R} m")
print(f"  Ship speed             v_s = {v_s/c:.1f} c")
print(f"")
print(f"  QI allowed energy density  : {rho_QI:.3e} J/m³")
print(f"  Total QI-allowed energy    : {E_QI_allowed:.3e} J")
print(f"")
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