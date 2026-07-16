import numpy as np
from solver.core_mldnn import em_caputo, solve_affine, evaluate_solution, ml_vec, build_S, brownian_increments
from solver.plotter import plot_expectation

def run():
    print("Running Exp 2: Linear Drift & Diffusion")
    alpha = 0.75
    y0, N_paths = 1.0, 500
    t_eval = np.linspace(0, 1, 100)
    y_true = ml_vec(alpha, 1.0, -(t_eval ** alpha))
    
    dB_mldnn = brownian_increments(64 * N_paths).reshape(N_paths, 64)
    best_err, best_m, best_mldnn_mean = float('inf'), None, None
    for m in [4, 8, 12, 16, 20, 24]:
        sols = []
        for i in range(N_paths):
            S = build_S(alpha, m, dB_mldnn[i])
            c, _, _, _ = solve_affine(alpha, m, S, y0, b0=0.0, b1=-1.0, s0=1.0, s1=0.0, Nq=64)
            sols.append(evaluate_solution(alpha, m, c, t_eval))
        m_mean = np.mean(sols, axis=0)
        err = np.max(np.abs(y_true - m_mean))
        if err < best_err: best_err, best_m, best_mldnn_mean = err, m, m_mean
        
    n = 256
    dB_fem = brownian_increments(n * N_paths).reshape(N_paths, n)
    fem_sols = [em_caputo(alpha, lambda t,y: -y, lambda t,y: 1.0, y0, dB_fem[i]) for i in range(N_paths)]
    fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), np.mean(fem_sols, axis=0))
    
    plot_expectation(t_eval, y_true, best_mldnn_mean, fem_mean, f"MLELM ($\\hat{{m}}$ = {best_m})", f"fEM (N={n})", "Ground truth", "Example 4.2", "exp2_linear_diffusion.pdf")
    print(f"Saved exp2! Best m={best_m}\n")

if __name__ == '__main__': np.random.seed(42); run()
