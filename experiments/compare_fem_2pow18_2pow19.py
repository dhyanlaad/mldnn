import os, sys, time, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.special import gamma as sgamma
from solver.core_mldnn import brownian_paths
from solver.parallel import solve_affine_fubini_batch, build_fubini_tensor

def compute_fem_paths(n_paths, n_steps, alpha, mu, sigma, y0, dB, t_eval, lib):
    D = 1.0 / n_steps
    ga = sgamma(alpha)
    d = np.arange(1, n_steps + 1, dtype=np.float64)
    with np.errstate(divide='ignore'):
        pdiff_a = -np.expm1(alpha * np.log1p(-1.0 / d)) * np.power(d, alpha)
    wdet = (D**alpha * pdiff_a / alpha) / ga
    ksto = (np.power(d * D, alpha - 1.0)) / ga
    
    y_out = np.empty((n_paths, len(t_eval)), dtype=np.float64)
    lib.solve_fem_c(
        n_paths, n_steps, alpha, mu, sigma, y0,
        dB.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wdet.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ksto.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(t_eval),
        t_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        os.cpu_count() or 8
    )
    return y_out

def run_fem_convergence_study(n_paths=500, alpha=0.75):
    y0 = 1.0
    mu = 0.15
    sigma = 0.25
    n_18 = 2**18  # 262,144
    n_19 = 2**19  # 524,288
    t_eval = np.linspace(0, 1, 100)
    
    print("=" * 95)
    print(f" FRACTIONAL SDE (alpha = {alpha:.2f}) CONVERGENCE: FEM(2^18) vs. FEM(2^19) vs. MLDNN")
    print(f" Parameters: mu = {mu}, sigma = {sigma}, y0 = {y0} | Sample Paths = {n_paths}")
    print("=" * 95)
    
    lib = ctypes.CDLL(os.path.abspath('scratch/libfast_fem.dylib'))
    lib.solve_fem_c.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    
    # 1. Generate Brownian paths at finest resolution 2^19
    print(f"\n[1/4] Generating {n_paths} Brownian paths at 2^19 ({n_19}) resolution...", flush=True)
    t0 = time.time()
    np.random.seed(42)
    dB_19 = brownian_paths(n_19, n_paths)
    # Subsampled increments for 2^18 (sum of pairs)
    dB_18 = (dB_19[:, 0::2] + dB_19[:, 1::2])
    print(f"Brownian paths generated in {time.time() - t0:.2f} s", flush=True)
    
    # 2. Compute FEM at 2^18
    print(f"\n[2/4] Running Fractional EM at N = 2^18 ({n_18} steps)...", flush=True)
    t0 = time.time()
    y_fem_18 = compute_fem_paths(n_paths, n_18, alpha, mu, sigma, y0, dB_18, t_eval, lib)
    t_fem_18 = time.time() - t0
    print(f"FEM(2^18) finished in {t_fem_18:.2f} s", flush=True)
    
    # 3. Compute FEM at 2^19
    print(f"\n[3/4] Running Fractional EM at N = 2^19 ({n_19} steps)...", flush=True)
    t0 = time.time()
    y_fem_19 = compute_fem_paths(n_paths, n_19, alpha, mu, sigma, y0, dB_19, t_eval, lib)
    t_fem_19 = time.time() - t0
    print(f"FEM(2^19) finished in {t_fem_19:.2f} s", flush=True)
    
    # Compare FEM(2^18) vs FEM(2^19)
    diff_t = np.mean((y_fem_18 - y_fem_19)**2, axis=0)
    sup_fem_diff = float(np.max(diff_t))
    
    print("\n" + "=" * 80)
    print(f" FRACTIONAL EM DISCRETIZATION ERROR (2^18 vs. 2^19):")
    print(f"   sup_t E[( Y_FEM(2^18) - Y_FEM(2^19) )^2] = {sup_fem_diff:.6e}")
    print(f"   FEM(2^18) Compute Time: {t_fem_18:.2f} s")
    print(f"   FEM(2^19) Compute Time: {t_fem_19:.2f} s (Ratio: {t_fem_19/t_fem_18:.2f}x, Theoretical: 4.0x)")
    print("=" * 80)
    
    # 4. Sweep MLDNN (Method 2) against 2^19 Reference
    print(f"\n[4/4] Evaluating MLDNN (Method 2) against the 2^19 Reference...", flush=True)
    print(f"\n{'m':>4} | {'Error vs. FEM(2^18)':>24} | {'Error vs. FEM(2^19)':>24} | {'MLDNN Time (s)':>15}")
    print("-" * 72)
    
    for m in range(4, 44, 4):
        t0 = time.time()
        M_tens, _ = build_fubini_tensor(alpha, m, n_19)
        sols_mldnn = solve_affine_fubini_batch(
            alpha=alpha,
            mhat=m,
            dB=dB_19,
            y0=y0,
            b0=0.0,
            b1=mu,
            s0=0.0,
            s1=sigma,
            Nq=128,
            t_eval=t_eval,
            M_tens=M_tens,
            trace_order=2
        )
        t_mldnn = time.time() - t0
        err_vs_18 = float(np.max(np.mean((sols_mldnn - y_fem_18)**2, axis=0)))
        err_vs_19 = float(np.max(np.mean((sols_mldnn - y_fem_19)**2, axis=0)))
        print(f"{m:4d} | {err_vs_18:24.6e} | {err_vs_19:24.6e} | {t_mldnn:15.3f}")
        
    print("-" * 72)

if __name__ == "__main__":
    run_fem_convergence_study(n_paths=500, alpha=0.75)
