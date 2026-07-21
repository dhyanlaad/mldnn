import time
import numpy as np
from solver.core_mldnn import em_caputo, build_S, brownian_paths, coarsen, solve_gauss_newton, solve_gauss_newton_deep, evaluate_solution
from solver.plotter import plot_expectation, save_pareto_table, plot_qq, plot_variance

def run():
    print("Running Exp 3: Nonlinear Drift")
    alpha = 0.75
    y0, N_paths = 1.0, 5000
    t_eval = np.linspace(0, 1, 100)
    
    idx_30 = np.argmin(np.abs(t_eval - 0.3))
    idx_60 = np.argmin(np.abs(t_eval - 0.6))
    
    n_gt = 8192
    print("Computing Ground Truth...")
    dB_gt = brownian_paths(n_gt, N_paths)
    gt_sols = em_caputo(alpha, lambda t,y: -y**3, lambda t,y: 1.0, y0, dB_gt)
    y_true_paths = np.array([np.interp(t_eval, np.linspace(0, 1, n_gt+1), sol) for sol in gt_sols])
    y_true_weak = np.mean(y_true_paths, axis=0)
    fem_fine_var = np.var(y_true_paths, axis=0)
    
    m_list = [4, 8, 12, 16, 24]
    times_mldnn, err_mldnn, labels_mldnn = [], [], []
    mldnn_mean_best = None
    mldnn_paths_best = None
    
    print("Computing MLELM (Gauss-Newton) Pareto...")
    for m in m_list:
        t0 = time.time()
        sols = []
        for i in range(N_paths):
            dB_mldnn_i = coarsen(dB_gt[i], 64)
            S = build_S(alpha, m, dB_mldnn_i)
            c, _, _, _, _ = solve_gauss_newton_deep(
                alpha, m, S, y0,
                lambda t, y: -y**3, lambda t, y: -3*y**2,
                lambda t, y: np.ones_like(y), lambda t, y: np.zeros_like(y),
                Nq=128, hidden=16, maxit=30
            )
            sols.append(evaluate_solution(alpha, m, c, t_eval))
        mldnn_paths = np.array(sols)
        mldnn_mean = np.mean(mldnn_paths, axis=0)
        t1 = time.time()
        
        times_mldnn.append(t1 - t0)
        err_mldnn.append(np.max(np.abs(y_true_weak - mldnn_mean)))
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
        fem_sols = em_caputo(alpha, lambda t,y: -y**3, lambda t,y: 1.0, y0, dB_fem)
        fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
        t1 = time.time()
        
        times_fem.append(t1 - t0)
        err_fem.append(np.max(np.abs(y_true_weak - fem_mean)))
        labels_fem.append(f"N={n}")
        if n == 2048:
            fem_mean_best = fem_mean
    
    plot_expectation(t_eval, y_true_weak, mldnn_mean_best, fem_mean_best, f"MLELM ($\\hat{{m}}$ = {m_list[-1]})", f"fEM (N={n_list[-1]})", "Ground truth", "Example 4.3", "exports/exp3_nonlinear_drift.pdf")
    save_pareto_table(times_mldnn, err_mldnn, times_fem, err_fem, "exports/exp3_pareto.md", labels_mldnn, labels_fem)
    
    plot_qq(y_true_paths[:, idx_30], mldnn_paths_best[:, idx_30], f"Q-Q Plot at t={t_eval[idx_30]:.2f}", "exports/exp3_qq_t0.3.pdf")
    plot_qq(y_true_paths[:, idx_60], mldnn_paths_best[:, idx_60], f"Q-Q Plot at t={t_eval[idx_60]:.2f}", "exports/exp3_qq_t0.6.pdf")
    # Plot variance comparison between fine EM baseline and MLDNN for Exp3
    plot_variance(t_eval, fem_fine_var, np.var(mldnn_paths_best, axis=0), "Variance Comparison (Exp3)", "exports/exp3_variance.pdf")
    
    print(f"Saved exp3!\n")

if __name__ == '__main__':
    np.random.seed(42)
    run()
