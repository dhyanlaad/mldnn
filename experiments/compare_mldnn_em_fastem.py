import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from solver.core_mldnn import brownian_paths
from solver.parallel import solve_affine_fubini_batch, build_fubini_tensor

def standard_em(dB, y0, mu, sigma):
    """Classical sequential Euler-Maruyama solver."""
    n_paths, n = dB.shape
    dt = 1.0 / n
    y = np.empty((n_paths, n + 1), dtype=np.float64)
    y[:, 0] = y0
    for k in range(n):
        y[:, k + 1] = y[:, k] * (1.0 + mu * dt + sigma * dB[:, k])
    return y

def fast_em(dB, y0, mu, sigma):
    """Vectorized cumulative-product Euler-Maruyama solver."""
    n_paths, n = dB.shape
    dt = 1.0 / n
    factors = 1.0 + mu * dt + sigma * dB
    y = np.empty((n_paths, n + 1), dtype=np.float64)
    y[:, 0] = y0
    y[:, 1:] = y0 * np.cumprod(factors, axis=1)
    return y

def run_comparison(n_paths=5000, n_fine=16384, nq=128):
    alpha = 1.0
    y0 = 1.0
    mu = 0.15
    sigma = 0.25
    t_eval = np.linspace(0, 1, 100)
    t_fine = np.linspace(0, 1, n_fine + 1)
    
    print("=" * 96)
    print(f" BENCHMARK: MLDNN vs. CLASSICAL EM vs. FAST EM (alpha = {alpha:.1f})")
    print(f" Parameters: mu = {mu}, sigma = {sigma}, y0 = {y0} | N_paths = {n_paths} | Fine Mesh = {n_fine}")
    print("=" * 96)
    
    # 1. Generate fine Brownian paths and true analytical solution
    np.random.seed(42)
    dB_fine = brownian_paths(n_fine, n_paths)
    B_fine = np.zeros((n_paths, n_fine + 1))
    B_fine[:, 1:] = np.cumsum(dB_fine, axis=1)
    B_eval = np.array([np.interp(t_eval, t_fine, B_fine[i]) for i in range(n_paths)])
    
    # Exact analytical Itô solution
    y_true = y0 * np.exp((mu - 0.5 * sigma**2) * t_eval[None, :] + sigma * B_eval)
    
    # =========================================================================
    # 2. Benchmark Classical EM & Fast EM across step resolutions N
    # =========================================================================
    N_steps_list = [16, 64, 256, 1024, 4096, 16384]
    
    em_results = []
    print(f"\n--- [1/2] Classical EM & Fast EM Performance ---")
    print(f"{'N steps':>8} | {'Classical EM Error':>20} | {'EM Time (s)':>12} | {'Fast EM Error':>20} | {'Fast EM Time (s)':>17}")
    print("-" * 88)
    
    for N in N_steps_list:
        stride = n_fine // N
        # Subsampled Brownian increments matching exact same realization
        dB_N = np.sum(dB_fine.reshape(n_paths, N, stride), axis=2)
        t_mesh_N = np.linspace(0, 1, N + 1)
        
        # Classical EM
        t0 = time.time()
        y_em_raw = standard_em(dB_N, y0, mu, sigma)
        t_em = time.time() - t0
        y_em_eval = np.array([np.interp(t_eval, t_mesh_N, y_em_raw[i]) for i in range(n_paths)])
        err_em = float(np.max(np.mean((y_em_eval - y_true)**2, axis=0)))
        
        # Fast EM
        t0 = time.time()
        y_fast_raw = fast_em(dB_N, y0, mu, sigma)
        t_fast = time.time() - t0
        y_fast_eval = np.array([np.interp(t_eval, t_mesh_N, y_fast_raw[i]) for i in range(n_paths)])
        err_fast = float(np.max(np.mean((y_fast_eval - y_true)**2, axis=0)))
        
        print(f"{N:8d} | {err_em:20.6e} | {t_em:12.4f} | {err_fast:20.6e} | {t_fast:17.4f}")
        em_results.append({
            "N": N,
            "err_em": err_em,
            "time_em": t_em,
            "err_fast": err_fast,
            "time_fast": t_fast
        })
        
    # =========================================================================
    # 3. Benchmark MLDNN Solver (Method 2) across Müntz orders m
    # =========================================================================
    m_list = [4, 8, 12, 16, 20, 24, 28, 32, 40, 60, 80, 100]
    
    print(f"\n--- [2/2] MLDNN (Stochastic Fubini Tensor Solver, N_q = {nq}) Performance ---")
    print(f"{'Order m':>8} | {'MLDNN L2 Error vs True':>24} | {'Total Time (s)':>16} | {'Speed (paths/sec)':>20}")
    print("-" * 75)
    
    mldnn_results = []
    for m in m_list:
        t0 = time.time()
        M_tens, _ = build_fubini_tensor(alpha, m, n_fine)
        sols_mldnn = solve_affine_fubini_batch(
            alpha=alpha,
            mhat=m,
            dB=dB_fine,
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
        t_mldnn = time.time() - t0
        err_mldnn = float(np.max(np.mean((sols_mldnn - y_true)**2, axis=0)))
        paths_per_sec = n_paths / t_mldnn
        
        print(f"{m:8d} | {err_mldnn:24.6e} | {t_mldnn:16.4f} | {paths_per_sec:20.1f}")
        mldnn_results.append({
            "m": m,
            "err": err_mldnn,
            "time": t_mldnn,
            "speed": paths_per_sec
        })
        
    print("-" * 75)

if __name__ == "__main__":
    run_comparison(n_paths=5000, n_fine=16384, nq=128)
