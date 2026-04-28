import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from warpax.metrics.warpshell import WarpShellStressTest
from warpax.geometry.geometry import compute_curvature_chain
from constants import C, G, HBAR

c = C
G_val = G
hbar = HBAR

def make_grid(R1, R2, nr, nth):
    r_max = 2.5 * R2
    r = np.linspace(1e-6, r_max, nr)
    theta = np.linspace(0, np.pi, nth)
    R2D, TH2D = np.meshgrid(r, theta, indexing='ij')
    X = R2D * np.cos(TH2D)
    Y = R2D * np.sin(TH2D)
    Z = np.zeros_like(X)
    T = np.zeros_like(X)
    coords = np.stack([T, X, Y, Z], axis=-1)
    return r, theta, coords

def make_metric_fn(adm_object):
    def g_func(x):
        alpha = adm_object.lapse(x)
        beta = adm_object.shift(x)
        gamma = adm_object.spatial_metric(x)
        g = jnp.zeros((4, 4))
        g = g.at[0, 0].set(-alpha**2 + (gamma @ beta) @ beta)
        g = g.at[0, 1:].set(gamma @ beta)
        g = g.at[1:, 0].set(gamma @ beta)
        g = g.at[1:, 1:].set(gamma)
        return g
    return g_func

def proper_energy_density(metric_fn, point):
    res = compute_curvature_chain(metric_fn, point)
    G_mixed = res.einstein
    evals = jnp.linalg.eigvals(G_mixed)
    lambda_0 = jnp.min(jnp.real(evals))
    return -lambda_0 / (8 * jnp.pi)

def compute_rho_grid(metric_fn, coords):
    shape = coords.shape
    N = shape[0] * shape[1]
    flat_coords = jnp.array(coords.reshape(N, 4))
    rho_flat = vmap(lambda pt: proper_energy_density(metric_fn, pt))(flat_coords)
    return np.array(rho_flat).reshape(shape[0], shape[1])

def audit_fuchs_point(R_1, R_2, r_s_param, v_s, nr=15, nth=15):
    r_vals, th_vals, coords = make_grid(R_1, R_2, nr, nth)
    adm = WarpShellStressTest(v_s=v_s, R_1=R_1, R_2=R_2, r_s_param=r_s_param,
                              smooth_width=0.12 * (R_2 - R_1))
    metric_fn = make_metric_fn(adm)
    rho_geom = compute_rho_grid(metric_fn, coords)
    rho_SI = rho_geom * c**4 / (8 * np.pi * G_val)
    dr = r_vals[1] - r_vals[0]
    dth = th_vals[1] - th_vals[0]
    R2D, TH2D = np.meshgrid(r_vals, th_vals, indexing='ij')
    dV = 2 * np.pi * R2D**2 * np.sin(TH2D) * dr * dth
    tol = 1e-10 * np.max(np.abs(rho_SI)) if np.max(np.abs(rho_SI)) > 0 else 1e-10
    mask_neg = rho_SI < -tol
    E_minus = np.sum(np.abs(rho_SI[mask_neg]) * dV[mask_neg])
    return E_minus

if __name__ == "__main__":
    R_1, R_2 = 10.0, 20.0
    
    # Ranges requested by user
    v_over_c = np.logspace(-2, -0.01, 15)   # 0.01c -> ~0.98c
    r_s_vals = np.linspace(0.8, 9.99, 15)   # 0.8 -> 9.99
    
    E_minus_grid = np.zeros((len(r_s_vals), len(v_over_c)))
    
    print("Starting Fuchs parameter sweep...")
    for i, rs in enumerate(r_s_vals):
        for j, vc in enumerate(v_over_c):
            v_s = vc * c
            Em = audit_fuchs_point(R_1, R_2, rs, v_s, nr=15, nth=15)
            E_minus_grid[i, j] = Em
            print(f"r_s_param={rs:.2f}, v_s={vc:.3f}c => E_minus={Em:.3e}")

    # Plot
    V_MESH, R_MESH = np.meshgrid(v_over_c, r_s_vals)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.pcolormesh(V_MESH, R_MESH, E_minus_grid, cmap='Reds', shading='auto')
    ax.set_xlabel('Ship velocity $v_s / c$')
    ax.set_ylabel('Shell parameter $r_{s,\\text{param}}$ [m]')
    ax.set_title('Fuchs Phase Space: Total Negative Energy $|E_-|$')
    fig.colorbar(cax, label='$|E_-|$ [Joules]')
    
    plt.tight_layout()
    plt.savefig('fuchs_phase_space.png', dpi=200)
    print("Saved fuchs_phase_space.png")
