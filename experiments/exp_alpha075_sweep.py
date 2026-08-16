import os, sys, time, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.special import gamma as sgamma
from solver.core_mldnn import brownian_paths
from solver.parallel import solve_affine_fubini_batch, build_fubini_tensor

def run_fractional_sweep(n_paths=1000, n_steps=2**18, nq=128):
    alpha = 0.75
    y0 = 1.0
    mu = 0.15
    sigma = 0.25
    m_list = list(range(4, 44, 4))  # [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    t_eval = np.linspace(0, 1, 100)
    
    print("=" * 95, flush=True)
    print(f" FRACTIONAL SDE SWEEP (alpha = {alpha}) WITH 2^18 ({n_steps}) FEM BENCHMARK", flush=True)
    print(f" Parameters: mu = {mu}, sigma = {sigma}, y0 = {y0} | N_paths = {n_paths} | N_q = {nq}", flush=True)
    print("=" * 95, flush=True)
    
    # 1. Generate Brownian paths with 2^18 steps
    print(f"\n[1/3] Generating {n_paths} Brownian paths with {n_steps} increments (2^18)...", flush=True)
    t0 = time.time()
    np.random.seed(42)
    dB = brownian_paths(n_steps, n_paths)
    print(f"Brownian paths generated in {time.time() - t0:.2f} s", flush=True)
    
    # 2. Compute 2^18 Fractional Euler-Maruyama Benchmark
    print(f"\n[2/3] Computing ultra-high-resolution Fractional EM benchmark (2^18 steps)...", flush=True)
    t1 = time.time()
    D = 1.0 / n_steps
    ga = sgamma(alpha)
    d = np.arange(1, n_steps + 1, dtype=np.float64)
    with np.errstate(divide='ignore'):
        pdiff_a = -np.expm1(alpha * np.log1p(-1.0 / d)) * np.power(d, alpha)
    wdet = (D**alpha * pdiff_a / alpha) / ga
    ksto = (np.power(d * D, alpha - 1.0)) / ga
    
    lib = ctypes.CDLL(os.path.abspath('scratch/libfast_fem.dylib'))
    lib.solve_fem_c.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    
    y_fem = np.empty((n_paths, len(t_eval)), dtype=np.float64)
    lib.solve_fem_c(
        n_paths, n_steps, alpha, mu, sigma, y0,
        dB.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wdet.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ksto.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_fem.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(t_eval),
        t_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        os.cpu_count() or 8
    )
    print(f"2^18 FEM benchmark computed in {time.time() - t1:.2f} s", flush=True)
    
    # 3. Sweep MLDNN for m = 4, 8, ..., 40 comparing Method 0 and Method 2
    print(f"\n[3/3] Sweeping Müntz order m = 4, 8, ..., 40 for Method 0 vs Method 2...", flush=True)
    print(f"\n{'m':>4} | {'Method 0 (0-th Order)':>24} | {'Method 2 (Mittag-Leffler)':>27} | {'Time (s)':>10}", flush=True)
    print("-" * 75, flush=True)
    
    for m in m_list:
        t_m = time.time()
        
        # Precompute Stochastic Fubini tensor for order m
        M_tens, _ = build_fubini_tensor(alpha, m, n_steps)
        
        # Method 0: 0-th order local impulse
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
        err_m0 = float(np.max(np.mean((sols_m0 - y_fem)**2, axis=0)))
        
        # Method 2: Mittag-Leffler Resolvent
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
        err_m2 = float(np.max(np.mean((sols_m2 - y_fem)**2, axis=0)))
        
        elapsed = time.time() - t_m
        print(f"{m:4d} | {err_m0:24.6e} | {err_m2:27.6e} | {elapsed:10.3f}", flush=True)
        
    print("-" * 75, flush=True)

if __name__ == "__main__":
    run_fractional_sweep(n_paths=1000, n_steps=2**18, nq=128)
