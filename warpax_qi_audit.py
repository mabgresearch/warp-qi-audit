import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from warpax.metrics.warpshell import WarpShellStressTest
from warpax.geometry.geometry import compute_curvature_chain
from constants import C, G, HBAR

c = C
G_val = G
hbar = HBAR

def make_grid(R1, R2, nr=100, nth=50):
    r_max = 2.5 * R2
    r = np.linspace(1e-6, r_max, nr)
    theta = np.linspace(0, np.pi, nth)
    R, TH = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    Z = np.zeros_like(X)
    T = np.zeros_like(X)
    coords = np.stack([T, X, Y, Z], axis=-1)
    return r, theta, coords

def make_metric_fn(adm_object):
    """Return a function g(x) that gives the full 4x4 metric tensor."""
    def g_func(x):
        alpha = adm_object.lapse(x)
        beta = adm_object.shift(x)        # vector (3,)
        gamma = adm_object.spatial_metric(x)  # (3,3) matrix
        g = jnp.zeros((4, 4))
        g = g.at[0, 0].set(-alpha**2 + (gamma @ beta) @ beta)
        g = g.at[0, 1:].set(gamma @ beta)
        g = g.at[1:, 0].set(gamma @ beta)
        g = g.at[1:, 1:].set(gamma)
        return g
    return g_func

def proper_energy_density(metric_fn, point):
    """Compute proper energy density in geometric units."""
    res = compute_curvature_chain(metric_fn, point)
    G_mixed = res.einstein  # mixed-index G^mu_nu
    # eigenvalues of mixed tensor
    evals = jnp.linalg.eigvals(G_mixed)
    # The timelike eigenvalue is the most negative real part
    lambda_0 = jnp.min(jnp.real(evals))
    # proper energy density = -lambda_0 / (8π) in geometric units
    return -lambda_0 / (8 * jnp.pi)

def compute_rho_grid(metric_fn, coords):
    shape = coords.shape
    N = shape[0] * shape[1]
    flat_coords = jnp.array(coords.reshape(N, 4))
    # vmap over all points
    rho_flat = vmap(lambda pt: proper_energy_density(metric_fn, pt))(flat_coords)
    return np.array(rho_flat).reshape(shape[0], shape[1])

def fuchs_qi_audit(R_1, R_2, r_s_param, v_s, nr=50, nth=25):
    """
    Audit Fuchs/WarpShell metric against Ford-Roman QI bound.
    Start with low grid resolution for a fast first run.
    """
    print(f"Parameters: R_1={R_1}, R_2={R_2}, r_s_param={r_s_param}, v_s={v_s}")
    print(f"Grid: {nr} x {nth}")

    r_vals, th_vals, coords = make_grid(R_1, R_2, nr, nth)

    # Build ADM object and metric function
    adm = WarpShellStressTest(
        v_s=v_s, R_1=R_1, R_2=R_2,
        r_s_param=r_s_param,
        smooth_width=0.12 * (R_2 - R_1)
    )
    metric_fn = make_metric_fn(adm)

    print("Computing proper energy density on grid...")
    rho_geom = compute_rho_grid(metric_fn, coords)

    # Convert geometric ρ to SI (J/m³)
    rho_SI = rho_geom * c**4 / (8 * np.pi * G_val)

    # Volume element dV = 2π r² sinθ dr dθ
    dr = r_vals[1] - r_vals[0]
    dth = th_vals[1] - th_vals[0]
    R2D, TH2D = np.meshgrid(r_vals, th_vals, indexing='ij')
    dV = 2 * np.pi * R2D**2 * np.sin(TH2D) * dr * dth

    tol = 1e-10 * np.max(np.abs(rho_SI)) if np.max(np.abs(rho_SI)) > 0 else 1e-10
    mask_neg = rho_SI < -tol
    mask_pos = rho_SI > tol

    E_minus = np.sum(np.abs(rho_SI[mask_neg]) * dV[mask_neg])
    E_plus  = np.sum(np.abs(rho_SI[mask_pos]) * dV[mask_pos])
    V_minus = np.sum(dV[mask_neg])

    # QI bound
    Delta = R_2 - R_1
    tau0 = Delta / c
    rho_qi = -(3 * hbar) / (32 * np.pi**2 * c**3 * tau0**4)
    E_qi_cap = np.abs(rho_qi) * V_minus
    qi_gap = E_minus / E_qi_cap if E_qi_cap > 1e-300 else np.inf

    print(f"\n--- Fuchs/WarpShell QI Audit ---")
    print(f"  E_minus     = {E_minus:.3e} J")
    print(f"  E_plus      = {E_plus:.3e} J")
    print(f"  V_minus     = {V_minus:.3e} m³")
    print(f"  QI cap      = {E_qi_cap:.3e} J")
    print(f"  QI gap      = {qi_gap:.1e}")
    print(f"\nComparison:")
    print(f"  Alcubierre QI gap  ~ 1e68")
    print(f"  Rodal QI gap       ~ 1e63")
    print(f"  Fuchs QI gap       ~ {qi_gap:.1e}")

    return E_minus, E_plus, qi_gap

if __name__ == "__main__":
    fuchs_qi_audit(R_1=10.0, R_2=20.0, r_s_param=5.0, v_s=0.02, nr=50, nth=25)