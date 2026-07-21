import time
import numpy as np
from solver.core_mldnn import em_caputo, solve_affine, evaluate_solution, brownian_paths, build_S, coarsen, ml_vec
from solver.plotter import plot_expectation, save_pareto_table, plot_qq, plot_variance

def run():
    print("Running Exp 2: Linear Drift & Diffusion Pareto")
    alpha = 0.75
    y0, N_paths = 1.0, 5000
    t_eval = np.linspace(0, 1, 100)
    
    idx_30 = np.argmin(np.abs(t_eval - 0.3))
    idx_60 = np.argmin(np.abs(t_eval - 0.6))
    
    # Fine EM baseline (super refined mesh n=4096)
    n_fine = 8192
    print("Computing fine EM baseline (n=4096)...")
    dB_fine = brownian_paths(n_fine, N_paths)
    fem_fine_sols = em_caputo(alpha, lambda t, y: -y, lambda t, y: 1.0, y0, dB_fine)
    fem_fine_paths = np.array([np.interp(t_eval, np.linspace(0, 1, n_fine+1), sol) for sol in fem_fine_sols])
    fem_fine_mean = np.mean(fem_fine_paths, axis=0)
    fem_fine_var = np.var(fem_fine_paths, axis=0)

    # Ground Truth proxy using high-fidelity MLDNN (m=24)
    n_gt = 4096
    print("Computing Ground Truth (MLELM m=24 proxy)...")
    dB_gt = brownian_paths(n_gt, N_paths)

    gt_sols = []
    for i in range(N_paths):
        S = build_S(alpha, 24, dB_gt[i])
        c, _, _, _ = solve_affine(alpha, 24, S, y0, b0=0.0, b1=-1.0, s0=1.0, s1=0.0, Nq=256)
        gt_sols.append(evaluate_solution(alpha, 24, c, t_eval))
    
    y_true_paths = np.array(gt_sols)
    y_true_weak = np.mean(y_true_paths, axis=0)
    
    m_list = [4, 8, 12, 16]
    times_mldnn, err_mldnn, labels_mldnn = [], [], []
    mldnn_mean_best = None
    mldnn_paths_best = None
    
    print("Computing MLELM Pareto...")
    for m in m_list:
        t0 = time.time()
        sols = []
        for i in range(N_paths):
            dB_mldnn_i = coarsen(dB_gt[i], 64)
            S = build_S(alpha, m, dB_mldnn_i)
            c, _, _, _ = solve_affine(alpha, m, S, y0, b0=0.0, b1=-1.0, s0=1.0, s1=0.0, Nq=128)
            sols.append(evaluate_solution(alpha, m, c, t_eval))
            
        mldnn_paths = np.array(sols)
        mldnn_mean = np.mean(mldnn_paths, axis=0)
        t1 = time.time()
        
        times_mldnn.append(t1 - t0)
        weak_error = np.max(np.abs(y_true_weak - mldnn_mean))
        err_mldnn.append(weak_error)
        labels_mldnn.append(f"m={m}")
        
        if m == 16:
            mldnn_mean_best = mldnn_mean
            mldnn_paths_best = mldnn_paths
            
    n_list = [256, 512, 1024, 2048]
    times_fem, err_fem, labels_fem = [], [], []
    fem_mean_best = None
    
    print("Computing fEM Pareto...")
    for n in n_list:
        t0 = time.time()
        dB_fem = np.array([coarsen(dB_gt[i], n) for i in range(N_paths)])
        
        fem_sols = em_caputo(alpha, lambda t,y: -y, lambda t,y: 1.0, y0, dB_fem)
        fem_paths = np.array([np.interp(t_eval, np.linspace(0, 1, n+1), sol) for sol in fem_sols])
        fem_mean = np.mean(fem_paths, axis=0)
        t1 = time.time()
        
        times_fem.append(t1 - t0)
        weak_error = np.max(np.abs(y_true_weak - fem_mean))
        err_fem.append(weak_error)
        labels_fem.append(f"N={n}")
        if n == 2048:
            fem_mean_best = fem_mean
    
    y_true_expectation = ml_vec(alpha, 1.0, -(t_eval ** alpha)) 
    plot_expectation(t_eval, y_true_expectation, mldnn_mean_best, fem_mean_best, f"MLELM ($\\hat{{m}}$={m_list[-1]})", f"fEM (N={n_list[-1]})", "Exact Expectation", "Example 4.2", "exports/exp2_linear_diffusion.pdf")
    save_pareto_table(times_mldnn, err_mldnn, times_fem, err_fem, "exports/exp2_pareto.md", labels_mldnn, labels_fem)
    
    plot_qq(y_true_paths[:, idx_30], mldnn_paths_best[:, idx_30], f"Q-Q Plot at t={t_eval[idx_30]:.2f}", "exports/exp2_qq_t0.3.pdf")
    plot_qq(y_true_paths[:, idx_60], mldnn_paths_best[:, idx_60], f"Q-Q Plot at t={t_eval[idx_60]:.2f}", "exports/exp2_qq_t0.6.pdf")
    # Plot variance comparison between fine EM baseline and MLDNN
    plot_variance(t_eval, fem_fine_var, np.var(mldnn_paths_best, axis=0), "Variance Comparison (Exp2)", "exports/exp2_variance.pdf")
    
    print(f"Saved exp2!\n")

if __name__ == '__main__':
    np.random.seed(42)
    run()
