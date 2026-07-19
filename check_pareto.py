import time
import numpy as np
from solver.core_mldnn import em_caputo, build_S, brownian_increments, coarsen, solve_gauss_newton, evaluate_solution

def check_pareto():
    alpha = 0.75
    y0, N_paths = 1.0, 500
    t_eval = np.linspace(0, 1, 100)
    
    n_gt = 4096
    dB_gt = brownian_increments(n_gt * N_paths).reshape(N_paths, n_gt)
    gt_sols = em_caputo(alpha, lambda t,y: -y**3, lambda t,y: 1.0, y0, dB_gt)
    y_true = np.interp(t_eval, np.linspace(0, 1, n_gt+1), np.mean(gt_sols, axis=0))
    
    m_list = [4, 8, 12, 16]
    for m in m_list:
        t0 = time.time()
        sols = []
        for i in range(N_paths):
            # Pass the full high-resolution Brownian path to accurately compute the stochastic operational matrix
            S = build_S(alpha, m, dB_gt[i])
            c, _, _, _, _ = solve_gauss_newton(
                alpha, m, S, y0, 
                lambda t, y: -y**3, lambda t, y: -3*y**2, 
                lambda t, y: np.ones_like(y), lambda t, y: np.zeros_like(y),
                Nq=64
            )
            sols.append(evaluate_solution(alpha, m, c, t_eval))
        mldnn_mean = np.mean(sols, axis=0)
        t1 = time.time()
        print(f"MLDNN m={m:2d}: time={t1-t0:.3f}s, error={np.max(np.abs(y_true - mldnn_mean)):.3e}")

if __name__ == '__main__':
    np.random.seed(42)
    check_pareto()
