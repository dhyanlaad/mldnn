import time
import numpy as np
from solver.core_mldnn import em_caputo, build_S, brownian_increments, coarsen, solve_gauss_newton, evaluate_solution
from solver.plotter import plot_expectation, plot_pareto

def run():
    print("Running Exp 5: Sine-Cosine")
    alpha = 0.75
    y0, N_paths = 1.0, 500
    t_eval = np.linspace(0, 1, 100)
    
    n_gt = 4096
    print("Computing Ground Truth...")
    dB_gt = brownian_increments(n_gt * N_paths).reshape(N_paths, n_gt)
    gt_sols = em_caputo(alpha, lambda t,y: np.sin(y), lambda t,y: np.cos(y), y0, dB_gt)
    y_true = np.interp(t_eval, np.linspace(0, 1, n_gt+1), np.mean(gt_sols, axis=0))
    
    m_list = [4, 8, 12, 16]
    times_mldnn, err_mldnn, labels_mldnn = [], [], []
    mldnn_mean_best = None
    
    print("Computing MLELM (Gauss-Newton) Pareto...")
    for m in m_list:
        t0 = time.time()
        sols = []
        for i in range(N_paths):
            dB_mldnn_i = coarsen(dB_gt[i], 64)
            S = build_S(alpha, m, dB_mldnn_i)
            c, _, _, _, _ = solve_gauss_newton(
                alpha, m, S, y0, 
                lambda t, y: np.sin(y), lambda t, y: np.cos(y), 
                lambda t, y: np.cos(y), lambda t, y: -np.sin(y),
                Nq=64
            )
            sols.append(evaluate_solution(alpha, m, c, t_eval))
        mldnn_mean = np.mean(sols, axis=0)
        t1 = time.time()
        
        times_mldnn.append(t1 - t0)
        err_mldnn.append(np.max(np.abs(y_true - mldnn_mean)))
        labels_mldnn.append(f"m={m}")
        if m == 16:
            mldnn_mean_best = mldnn_mean
            
    n_list = [256, 512, 1024, 2048]
    times_fem, err_fem, labels_fem = [], [], []
    fem_mean_best = None
    
    print("Computing fEM Pareto...")
    for n in n_list:
        t0 = time.time()
        dB_fem = np.array([coarsen(dB_gt[i], n) for i in range(N_paths)])
        fem_sols = em_caputo(alpha, lambda t,y: np.sin(y), lambda t,y: np.cos(y), y0, dB_fem)
        fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
        t1 = time.time()
        
        times_fem.append(t1 - t0)
        err_fem.append(np.max(np.abs(y_true - fem_mean)))
        labels_fem.append(f"N={n}")
        if n == 2048:
            fem_mean_best = fem_mean
    
    plot_expectation(t_eval, y_true, mldnn_mean_best, fem_mean_best, f"MLELM ($\\hat{{m}}$ = {m_list[-1]})", f"fEM (N={n_list[-1]})", "Ground truth", "Example 4.5", "exp5_sine_cosine.pdf")
    plot_pareto(times_mldnn, err_mldnn, times_fem, err_fem, "exp5_pareto.pdf", labels_mldnn, labels_fem)
    print(f"Saved exp5!\n")

if __name__ == '__main__': np.random.seed(42); run()
