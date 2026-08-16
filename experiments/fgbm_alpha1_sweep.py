import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from solver.core_mldnn import brownian_paths
from solver.parallel import solve_affine_fubini_batch, build_fubini_tensor

def run_sweep(n_paths=5000, n_steps=4096 * 4, nq=128):
    alpha = 1.0
    y0 = 1.0
    mu = 0.15
    sigma = 0.25
    m_list = list(range(4, 104, 4))  # [4, 8, 12, ..., 100]
    t_eval = np.linspace(0, 1, 100)
    t_mesh = np.linspace(0, 1, n_steps + 1)
    
    print(f"=====================================================================================================", flush=True)
    print(f" COMPARISON OF ALL 3 MALLIAVIN TRACE METHODS (m = 4, 8, ..., 100 | N_q = {nq})                      ", flush=True)
    print(f" Parameters: alpha={alpha:.1f}, mu={mu}, sigma={sigma} | N_paths={n_paths} | N_steps={n_steps} | N_q={nq} ", flush=True)
    print(f"=====================================================================================================", flush=True)
    
    print(f"\n[1/3] Generating {n_paths} Brownian paths with {n_steps} steps...", flush=True)
    dB = brownian_paths(n_steps, n_paths)
    B = np.zeros((n_paths, n_steps + 1))
    B[:, 1:] = np.cumsum(dB, axis=1)
    B_eval = np.array([np.interp(t_eval, t_mesh, B[i]) for i in range(n_paths)])
    
    # Exact Analytical Itô Solution
    y_analytic_ito = y0 * np.exp((mu - 0.5 * sigma**2) * t_eval[None, :] + sigma * B_eval)
    
    print(f"\n{'m':>4} | {'Method 0 (0-th Order)':>23} | {'Method 1 (Neumann)':>23} | {'Method 2 (Mittag-Leffler)':>26} | {'Time (s)':>10}", flush=True)
    print("-" * 96, flush=True)
    
    results = []
    for m in m_list:
        t0 = time.time()
        
        # Precompute Stochastic Fubini tensor for order m
        M_tens, _ = build_fubini_tensor(alpha, m, n_steps)
        
        # 1. Method 0 (0-th Order Local Impulse)
        sols_m0 = solve_affine_fubini_batch(
            alpha=alpha,
            mhat=m,
            dB=dB,
            y0=y0,
            b0=0.0,
            b1=mu,
            s0=0.0,
            s1=sigma,
            Nq=nq,
            t_eval=t_eval,
            M_tens=M_tens,
            trace_order=0
        )
        err_m0 = float(np.max(np.mean((sols_m0 - y_analytic_ito)**2, axis=0)))
        
        # 2. Method 1 (1st-Order Neumann Series Expansion)
        sols_m1 = solve_affine_fubini_batch(
            alpha=alpha,
            mhat=m,
            dB=dB,
            y0=y0,
            b0=0.0,
            b1=mu,
            s0=0.0,
            s1=sigma,
            Nq=nq,
            t_eval=t_eval,
            M_tens=M_tens,
            trace_order=1
        )
        err_m1 = float(np.max(np.mean((sols_m1 - y_analytic_ito)**2, axis=0)))
        
        # 3. Method 2 (Mittag-Leffler Resolvent)
        sols_m2 = solve_affine_fubini_batch(
            alpha=alpha,
            mhat=m,
            dB=dB,
            y0=y0,
            b0=0.0,
            b1=mu,
            s0=0.0,
            s1=sigma,
            Nq=nq,
            t_eval=t_eval,
            M_tens=M_tens,
            trace_order=2
        )
        err_m2 = float(np.max(np.mean((sols_m2 - y_analytic_ito)**2, axis=0)))
        
        elapsed = time.time() - t0
        print(f"{m:4d} | {err_m0:23.6e} | {err_m1:23.6e} | {err_m2:26.6e} | {elapsed:10.3f}", flush=True)
        results.append({
            "m": m,
            "err_m0": err_m0,
            "err_m1": err_m1,
            "err_m2": err_m2,
            "time": elapsed
        })
        
    print("-" * 96, flush=True)
    return results

if __name__ == "__main__":
    np.random.seed(42)
    run_sweep(n_paths=5000, n_steps=4096 * 4, nq=128)
