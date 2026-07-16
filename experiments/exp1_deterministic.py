import numpy as np
from solver.core_mldnn import em_caputo, solve_affine, evaluate_solution, ml_vec, build_S
from solver.plotter import plot_expectation

def run():
    print("Running Exp 1: Linear Deterministic")
    alpha = 0.75
    y0 = 1.0
    def exact_mean(t): return ml_vec(alpha, 1.0, -(t ** alpha))
    t_eval = np.linspace(0, 1, 100)
    y_true = exact_mean(t_eval)
    
    # MLDNN Sweep
    best_err, best_m, best_mldnn_mean = float('inf'), None, None
    for m in [4, 8, 12, 16, 20, 24]:
        dB = np.zeros(64)
        S = build_S(alpha, m, dB)
        c, _, _, _ = solve_affine(alpha, m, S, y0, b0=0.0, b1=-1.0, s0=0.0, s1=0.0, Nq=64)
        m_mean = evaluate_solution(alpha, m, c, t_eval)
        err = np.max(np.abs(y_true - m_mean))
        if err < best_err: best_err, best_m, best_mldnn_mean = err, m, m_mean
    
    # fEM
    n = 1024
    def bfun_np(t, y): return -y
    def sfun_np(t, y): return 0.0
    fem_sol = em_caputo(alpha, bfun_np, sfun_np, y0, np.zeros(n))
    fem_mean = np.interp(t_eval, np.linspace(0, 1, n+1), fem_sol)
    
    plot_expectation(t_eval, y_true, best_mldnn_mean, fem_mean, f"MLDNN (ELM, m={best_m})", f"fEM (N={n})", "Exact Analytic", "Exp 1: Deterministic Physics", "exp1_deterministic.pdf")
    print(f"Saved exp1! Best m={best_m}\n")

if __name__ == '__main__': run()
